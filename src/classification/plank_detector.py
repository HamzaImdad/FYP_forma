"""
Dedicated plank detector — static hold, no rep counting.

Angles tracked:
  - Body line (shoulder-hip-ankle): should be ~180 degrees (straight line)
  - Hip sag/pike (hip position relative to shoulder-ankle line)
  - Shoulder position (should be over wrists/elbows)

Tracks hold duration instead of reps.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .base_detector import RobustExerciseDetector
from .posture_classifier import PostureLabel
from ..pose_estimation.base import PoseResult
from ..utils.constants import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
)


# ── Thresholds ──
# Research (Strand et al. 2014, ACFT): straight line shoulders→hips→knees→ankles
# is 175-185°. Acceptable 170-190°. Mild sag/pike 165-170 / 190-195.
# Excessive: <160° / >200° — should stop the exercise.
BODY_LINE_GOOD = 175    # was 170 — research perfect
BODY_LINE_WARNING = 165 # Mild sag/pike
BODY_LINE_BAD = 155     # Significant form breakdown

# Hip position relative to shoulder-ankle line (world coords, meters)
HIP_SAG_THRESHOLD = 0.04    # meters — hips sagging below line
HIP_PIKE_THRESHOLD = 0.04   # meters — hips piking above line

# Shoulder over wrist alignment (world coords, meters)
SHOULDER_WRIST_GOOD = 0.08   # meters x-offset tolerance

# Knee angle — legs should be straight (~180°). Flag bent knees.
KNEE_STRAIGHT = 170

# Fatigue drift tracking (T35/T36)
BASELINE_FRAMES = 10     # average this many frames after form_validated to set baseline
DRIFT_WARN_DEG = 5       # warn at this much drift from baseline
DRIFT_STOP_DEG = 10      # recommend stopping at this much drift


class PlankDetector(RobustExerciseDetector):
    EXERCISE_NAME = "plank"
    IS_STATIC = True
    PRIMARY_ANGLE_TOP = BODY_LINE_GOOD
    PRIMARY_ANGLE_BOTTOM = BODY_LINE_BAD

    # ── Robust FSM config (static hold) ─────────────────────────────────
    # Plank uses the PostureClassifier's PLANK label (empirically works
    # well — real pushup sessions produce 200-500 PLANK frames). The
    # robust FSM's static-hold path runs when IS_STATIC = True and skips
    # the rep FSM entirely, while still using the outer session FSM
    # (IDLE/SETUP/ACTIVE/RESTING) to pause/resume the hold timer on
    # walk-aways.
    SESSION_POSTURE_GATE = PostureLabel.PLANK
    MIN_REAL_PRIMARY_DEG = 0.0  # body-line angle can legitimately be small
    MAX_PRIMARY_JUMP_DEG = 90.0  # generous for a static hold

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Baseline body-line angle captured after form validation (T35)
        self._baseline_body_line: Optional[float] = None
        self._baseline_sample_buf: List[float] = []
        self._drift_stop_warned: bool = False

    def reset(self):
        super().reset()
        self._baseline_body_line = None
        self._baseline_sample_buf = []
        self._drift_stop_warned = False

    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        body_line = self._avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_HIP, LEFT_ANKLE),
            (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_ANKLE),
            name="hip",
        )

        hip_deviation = self._compute_hip_deviation(pose)
        shoulder_alignment = self._compute_shoulder_alignment(pose)

        # T39: knee angle — straight legs expected
        knee = self._avg_angle(
            pose,
            (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
            (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
            name="knee",
        )

        # T35: build baseline from first BASELINE_FRAMES valid frames after
        # form validation. Once baseline is set, drift can be computed.
        if (self._form_validated and body_line is not None
                and self._baseline_body_line is None):
            self._baseline_sample_buf.append(body_line)
            if len(self._baseline_sample_buf) >= BASELINE_FRAMES:
                self._baseline_body_line = sum(self._baseline_sample_buf) / len(self._baseline_sample_buf)
                self._baseline_sample_buf = []

        drift = None
        if self._baseline_body_line is not None and body_line is not None:
            drift = abs(body_line - self._baseline_body_line)

        return {
            "primary": body_line,
            "body_line": body_line,
            "hip_deviation": hip_deviation,
            "shoulder_alignment": shoulder_alignment,
            "knee": knee,
            "drift": drift,
        }

    def _compute_hip_deviation(self, pose: PoseResult) -> Optional[float]:
        """Compute how much hips deviate from straight shoulder-ankle line.
        Positive = sag (hips below line), Negative = pike (hips above line).
        Uses world coordinates for camera-angle independence.
        """
        deviations = []
        for s_idx, h_idx, a_idx in [(LEFT_SHOULDER, LEFT_HIP, LEFT_ANKLE),
                                     (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_ANKLE)]:
            if all(pose.is_visible(i) for i in (s_idx, h_idx, a_idx)):
                s = np.array(pose.get_world_landmark(s_idx))
                h = np.array(pose.get_world_landmark(h_idx))
                a = np.array(pose.get_world_landmark(a_idx))
                # Project hip onto shoulder-ankle line, measure perpendicular distance
                sa = a - s
                sh = h - s
                sa_len = np.linalg.norm(sa)
                if sa_len < 1e-6:
                    continue
                # Parameter t: where hip projects onto the shoulder-ankle line
                t = np.dot(sh, sa) / (sa_len ** 2)
                t = np.clip(t, 0, 1)
                projected = s + t * sa
                # Perpendicular vector from line to hip
                perp = h - projected
                # Sign: positive Y in world = down, so positive perp.y = hip sagging
                deviations.append(float(perp[1]))

        return sum(deviations) / len(deviations) if deviations else None

    def _compute_shoulder_alignment(self, pose: PoseResult) -> Optional[float]:
        """Check if shoulders are over wrists/elbows using world coordinates."""
        offsets = []
        for s_idx, w_idx in [(LEFT_SHOULDER, LEFT_WRIST), (RIGHT_SHOULDER, RIGHT_WRIST)]:
            if pose.is_visible(s_idx) and pose.is_visible(w_idx):
                s = np.array(pose.get_world_landmark(s_idx))
                w = np.array(pose.get_world_landmark(w_idx))
                # Horizontal offset (x-axis in world coords)
                offsets.append(float(s[0] - w[0]))

        return sum(offsets) / len(offsets) if offsets else None

    def _check_start_position(self, angles: Dict[str, Optional[float]]) -> Tuple[bool, List[str]]:
        issues = []
        body_line = angles.get("body_line")
        if body_line is None:
            return False, ["Get into plank position — full body must be visible from the side"]
        if body_line < BODY_LINE_BAD:
            issues.append("Straighten your body — align shoulders, hips, and ankles")
        if not issues:
            return True, []
        return False, issues

    def _get_missing_parts(self, angles: Dict[str, Optional[float]]) -> List[str]:
        if angles.get("body_line") is None:
            return ["shoulders, hips, or ankles"]
        return ["body"]

    def _assess_form(self, angles: Dict[str, Optional[float]]) -> Tuple[float, Dict[str, str], List[str]]:
        joint_feedback = {}
        issues = []
        scores = {}

        body_line = angles.get("body_line")
        hip_dev = angles.get("hip_deviation")
        shoulder_align = angles.get("shoulder_alignment")

        # Pre-compute suppression: drift is the better fatigue signal than hip sag
        drift = angles.get("drift")
        _drift_active = drift is not None and drift >= DRIFT_WARN_DEG

        # ── Body line straightness (most important — weighted 1.5x) ──
        if body_line is not None:
            if body_line >= BODY_LINE_GOOD:
                joint_feedback["left_hip"] = "correct"
                joint_feedback["right_hip"] = "correct"
                joint_feedback["left_shoulder"] = "correct"
                joint_feedback["right_shoulder"] = "correct"
                scores["body_line"] = 1.0
            elif body_line >= BODY_LINE_WARNING:
                joint_feedback["left_hip"] = "warning"
                joint_feedback["right_hip"] = "warning"
                scores["body_line"] = 0.6
            elif body_line >= BODY_LINE_BAD:
                joint_feedback["left_hip"] = "incorrect"
                joint_feedback["right_hip"] = "incorrect"
                scores["body_line"] = 0.3
            else:
                joint_feedback["left_hip"] = "incorrect"
                joint_feedback["right_hip"] = "incorrect"
                scores["body_line"] = 0.1

        # ── Hip sag/pike detection ──
        if hip_dev is not None:
            if hip_dev > HIP_SAG_THRESHOLD:
                if not _drift_active:
                    issues.append("Hips sagging — tighten core and glutes, lift hips")
                joint_feedback["left_hip"] = "incorrect"
                joint_feedback["right_hip"] = "incorrect"
                scores["hip_dev"] = 0.3
            elif hip_dev < -HIP_PIKE_THRESHOLD:
                issues.append("Hips too high — lower them in line with shoulders and ankles")
                joint_feedback["left_hip"] = "warning"
                joint_feedback["right_hip"] = "warning"
                scores["hip_dev"] = 0.5
            else:
                scores["hip_dev"] = 1.0

        # ── Shoulder alignment ──
        if shoulder_align is not None:
            if abs(shoulder_align) <= SHOULDER_WRIST_GOOD:
                scores["shoulder"] = 1.0
            elif abs(shoulder_align) <= SHOULDER_WRIST_GOOD * 2:
                issues.append("Shift shoulders over wrists")
                scores["shoulder"] = 0.6
            else:
                issues.append("Shoulders too far from wrists — reposition")
                scores["shoulder"] = 0.3

        # ── Knee straightness (T39) ──
        knee = angles.get("knee")
        if knee is not None and knee < KNEE_STRAIGHT:
            issues.append("Straighten legs — keep knees locked")
            scores["knee"] = 0.5
            joint_feedback["left_knee"] = "warning"
            joint_feedback["right_knee"] = "warning"

        # ── Fatigue drift tracking (T36) ──
        drift = angles.get("drift")
        if drift is not None:
            if drift >= DRIFT_STOP_DEG:
                if not self._drift_stop_warned:
                    self._drift_stop_warned = True
                issues.append("Form breaking down — consider resting")
                scores["drift"] = 0.3
            elif drift >= DRIFT_WARN_DEG:
                issues.append("Form starting to drift — stay tight")
                scores["drift"] = 0.7

        if not scores:
            return 0.0, joint_feedback, issues

        WEIGHTS = {
            "body_line": 1.5, "hip_dev": 1.2, "shoulder": 1.0,
            "knee": 0.8, "drift": 1.1,
        }
        weighted_sum = sum(scores[k] * WEIGHTS.get(k, 1.0) for k in scores)
        weight_total = sum(WEIGHTS.get(k, 1.0) for k in scores)
        total = weighted_sum / weight_total

        return max(0.0, min(1.0, total)), joint_feedback, issues
