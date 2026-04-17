"""
Dedicated plank detector — static hold, forearm plank only.

Angles tracked:
  - Body line (shoulder-hip-ankle): should be ~180 degrees (straight line)
  - Hip sag/pike (hip position relative to shoulder-ankle line)
  - Elbow angle: must be ~90° (forearm plank); high plank (>150°) is flagged
  - Head/neck alignment (ear-shoulder-hip): head on floor = cheat
  - Shoulder position (should be over elbows, not wrists)

Tracks hold duration instead of reps.
Camera: side view.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .base_detector import RobustExerciseDetector, SessionState
from ..pose_estimation.base import PoseResult
from ..utils.constants import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_EAR, RIGHT_EAR,
)
from ..utils.geometry import calculate_angle


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

# Shoulder over ELBOW alignment (forearm plank — shoulders stack over elbows)
SHOULDER_ELBOW_GOOD = 0.08   # meters x-offset tolerance

# Knee angle — legs should be straight (~180°). Flag bent knees.
KNEE_STRAIGHT = 170

# Fatigue drift tracking (T35/T36)
BASELINE_FRAMES = 10     # average this many frames after form_validated to set baseline
DRIFT_WARN_DEG = 5       # warn at this much drift from baseline
DRIFT_STOP_DEG = 10      # recommend stopping at this much drift

# Side-view visibility threshold (same as squat/bicep_curl)
PLANK_VIS_THRESHOLD = 0.3

# Elbow angle — forearm plank vs high plank distinction
ELBOW_FOREARM_MAX = 120    # <120° = on forearms (correct forearm plank)
ELBOW_HIGH_PLANK = 150     # >150° = high plank (arms extended, WRONG)

# Head/neck alignment — ear-shoulder-hip angle
# Neutral neck: head in line with body (~170-180°)
HEAD_NEUTRAL_MIN = 165
HEAD_DROPPING = 155        # <155° = head dropping significantly
HEAD_ON_FLOOR = 140        # <140° = head resting on floor (CHEAT — harsh penalty)
HEAD_CRANING = 190         # >190° = craning up (neck strain risk)

# Movement-gated FSM for plank (static hold — no motion range needed)
PLANK_BODY_LINE_MIN = 155.0   # body must be roughly horizontal
PLANK_BODY_LINE_MAX = 195.0   # not standing upright
PLANK_POSITION_LATCH_S = 1.5  # hold position for 1.5s to latch ACTIVE
PLANK_REST_NOT_PLANK_S = 5.0  # out of plank position for 5s → RESTING


class PlankDetector(RobustExerciseDetector):
    EXERCISE_NAME = "plank"
    IS_STATIC = True
    PRIMARY_ANGLE_TOP = BODY_LINE_GOOD
    PRIMARY_ANGLE_BOTTOM = BODY_LINE_BAD

    # ── Robust FSM config (static hold, side view) ─────────────────────
    # Switched from PostureLabel.PLANK to visibility-only gate for
    # side-view reliability. The custom _update_session_state gates on
    # body-line angle + elbow angle (forearm position) instead.
    SESSION_POSTURE_GATE = None
    VISIBILITY_GATE_LANDMARKS = (
        LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
        LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE,
    )
    VISIBILITY_GATE_THRESHOLD = PLANK_VIS_THRESHOLD
    MIN_REAL_PRIMARY_DEG = 0.0  # body-line angle can legitimately be small
    MAX_PRIMARY_JUMP_DEG = 90.0  # generous for a static hold

    def __init__(self, **kwargs):
        # Instance state BEFORE super().__init__() — reset() touches these
        self._baseline_body_line: Optional[float] = None
        self._baseline_sample_buf: List[float] = []
        self._drift_stop_warned: bool = False
        # Movement-gated session FSM state
        self._plank_position_since: Optional[float] = None
        self._not_plank_since: Optional[float] = None
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self._baseline_body_line = None
        self._baseline_sample_buf = []
        self._drift_stop_warned = False
        self._plank_position_since = None
        self._not_plank_since = None

    # ── Side-view helpers ──────────────────────────────────────────────

    def _lenient_side_angle(
        self,
        pose: PoseResult,
        a_idx: int,
        b_idx: int,
        c_idx: int,
        threshold: float = PLANK_VIS_THRESHOLD,
    ) -> Optional[float]:
        if not all(pose.is_visible(i, threshold) for i in (a_idx, b_idx, c_idx)):
            return None
        return calculate_angle(
            pose.get_world_landmark(a_idx),
            pose.get_world_landmark(b_idx),
            pose.get_world_landmark(c_idx),
        )

    def _lenient_avg_angle(
        self,
        pose: PoseResult,
        left_triple: Tuple[int, int, int],
        right_triple: Tuple[int, int, int],
        name: str,
    ) -> Optional[float]:
        left = self._lenient_side_angle(pose, *left_triple)
        right = self._lenient_side_angle(pose, *right_triple)
        self._lr_angles[name] = (left, right)
        if left is not None and right is not None:
            return (left + right) / 2
        return left if left is not None else right

    def _check_visibility_gate(self, pose) -> bool:
        """Accept if EITHER side's shoulder-elbow-hip-ankle chain is visible."""
        left_ok = all(
            pose.is_visible(idx, PLANK_VIS_THRESHOLD)
            for idx in (LEFT_SHOULDER, LEFT_ELBOW, LEFT_HIP, LEFT_ANKLE)
        )
        right_ok = all(
            pose.is_visible(idx, PLANK_VIS_THRESHOLD)
            for idx in (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_HIP, RIGHT_ANKLE)
        )
        return left_ok or right_ok

    # ── Movement-gated session FSM (static hold) ──────────────────────

    def _update_session_state(self, now: float, visibility_ok: bool) -> None:
        """Plank is IS_STATIC — gate on being in plank position (body
        roughly horizontal + on forearms), not on angle motion range."""
        body_line = self._get_smooth("body_line")
        elbow = self._get_smooth("elbow")

        # Determine if user is in plank position
        in_plank_position = (
            body_line is not None
            and elbow is not None
            and PLANK_BODY_LINE_MIN <= body_line <= PLANK_BODY_LINE_MAX
            and elbow < ELBOW_FOREARM_MAX
        )

        # Update position timers
        if in_plank_position:
            if self._plank_position_since is None:
                self._plank_position_since = now
            self._not_plank_since = None
        else:
            if self._not_plank_since is None:
                self._not_plank_since = now
            self._plank_position_since = None

        s = self._session_state

        # Visibility loss handling
        if not visibility_ok:
            if s == SessionState.ACTIVE:
                self._unknown_streak += 1
                if self._unknown_streak >= self.UNKNOWN_GRACE_FRAMES:
                    self._close_active_set_or_rollback(now)
            return
        self._unknown_streak = 0

        # IDLE → SETUP: body visible
        if s == SessionState.IDLE:
            self._session_state = SessionState.SETUP
            return

        # SETUP → ACTIVE: in plank position sustained for latch duration
        if s == SessionState.SETUP:
            if (self._plank_position_since is not None
                    and now - self._plank_position_since >= PLANK_POSITION_LATCH_S):
                self._session_state = SessionState.ACTIVE
            return

        # RESTING → ACTIVE: same as SETUP
        if s == SessionState.RESTING:
            if (self._plank_position_since is not None
                    and now - self._plank_position_since >= PLANK_POSITION_LATCH_S):
                self._session_state = SessionState.ACTIVE
            return

        # ACTIVE → RESTING: out of plank position for rest duration
        if s == SessionState.ACTIVE:
            if (self._not_plank_since is not None
                    and now - self._not_plank_since >= PLANK_REST_NOT_PLANK_S):
                self._close_active_set_or_rollback(now)
                return

    # ── Angle computation ─────────────────────────────────────────────

    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        body_line = self._lenient_avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_HIP, LEFT_ANKLE),
            (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_ANKLE),
            name="hip",
        )

        # Elbow angle — forearm plank should be ~90°
        elbow = self._lenient_avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
            (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
            name="elbow",
        )

        # Head/neck alignment — ear-shoulder-hip angle
        head_neck = self._compute_head_neck_angle(pose)

        hip_deviation = self._compute_hip_deviation(pose)
        shoulder_alignment = self._compute_shoulder_alignment(pose)

        # Knee angle — straight legs expected
        knee = self._lenient_avg_angle(
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
            "elbow": elbow,
            "head_neck": head_neck,
            "hip_deviation": hip_deviation,
            "shoulder_alignment": shoulder_alignment,
            "knee": knee,
            "drift": drift,
        }

    def _compute_head_neck_angle(self, pose: PoseResult) -> Optional[float]:
        """Head/neck alignment: angle at shoulder between ear-shoulder-hip.
        ~170-180° = neutral head position (in line with body).
        <155° = head dropping. <140° = head on floor (cheating).
        >190° = craning up (neck strain)."""
        angles = []
        for ear_idx, s_idx, h_idx in [
            (LEFT_EAR, LEFT_SHOULDER, LEFT_HIP),
            (RIGHT_EAR, RIGHT_SHOULDER, RIGHT_HIP),
        ]:
            angle = self._lenient_side_angle(pose, ear_idx, s_idx, h_idx)
            if angle is not None:
                angles.append(angle)
        return sum(angles) / len(angles) if angles else None

    def _compute_hip_deviation(self, pose: PoseResult) -> Optional[float]:
        """Compute how much hips deviate from straight shoulder-ankle line.
        Positive = sag (hips below line), Negative = pike (hips above line).
        Uses world coordinates for camera-angle independence.
        """
        deviations = []
        for s_idx, h_idx, a_idx in [(LEFT_SHOULDER, LEFT_HIP, LEFT_ANKLE),
                                     (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_ANKLE)]:
            if all(pose.is_visible(i, PLANK_VIS_THRESHOLD) for i in (s_idx, h_idx, a_idx)):
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
        """Check if shoulders are over elbows using world coordinates.
        For forearm plank: elbows on ground, shoulders directly above."""
        offsets = []
        for s_idx, e_idx in [(LEFT_SHOULDER, LEFT_ELBOW), (RIGHT_SHOULDER, RIGHT_ELBOW)]:
            if pose.is_visible(s_idx, PLANK_VIS_THRESHOLD) and pose.is_visible(e_idx, PLANK_VIS_THRESHOLD):
                s = np.array(pose.get_world_landmark(s_idx))
                e = np.array(pose.get_world_landmark(e_idx))
                # Horizontal offset (x-axis in world coords)
                offsets.append(float(s[0] - e[0]))

        return sum(offsets) / len(offsets) if offsets else None

    # ── Start position + missing parts ────────────────────────────────

    def _check_start_position(self, angles: Dict[str, Optional[float]]) -> Tuple[bool, List[str]]:
        issues = []
        body_line = angles.get("body_line")
        elbow = angles.get("elbow")
        if body_line is None:
            return False, ["Get into plank position — full body must be visible from the side"]
        if body_line < BODY_LINE_BAD:
            issues.append("Straighten your body — align shoulders, hips, and ankles")
        if elbow is not None and elbow > ELBOW_HIGH_PLANK:
            issues.append("Get onto your forearms — this is a forearm plank")
        if not issues:
            return True, []
        return False, issues

    def _get_missing_parts(self, angles: Dict[str, Optional[float]]) -> List[str]:
        if angles.get("body_line") is None:
            return ["shoulders, hips, or ankles"]
        return ["body"]

    # ── Form assessment ───────────────────────────────────────────────

    def _assess_form(self, angles: Dict[str, Optional[float]]) -> Tuple[float, Dict[str, str], List[str]]:
        joint_feedback = {}
        issues = []
        scores = {}

        body_line = angles.get("body_line")
        hip_dev = angles.get("hip_deviation")
        shoulder_align = angles.get("shoulder_alignment")
        elbow = angles.get("elbow")
        head_neck = angles.get("head_neck")

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

        # ── Elbow angle — forearm plank check ──
        if elbow is not None:
            if elbow <= ELBOW_FOREARM_MAX:           # <=120° — on forearms (correct)
                scores["elbow"] = 1.0
            elif elbow <= ELBOW_HIGH_PLANK:          # 120-150° — transitional
                issues.append("Get down onto your forearms — elbows should be bent at 90°")
                joint_feedback["left_elbow"] = "warning"
                joint_feedback["right_elbow"] = "warning"
                scores["elbow"] = 0.4
            else:                                     # >150° — high plank (wrong)
                issues.append("Wrong position — get onto forearms, not hands")
                joint_feedback["left_elbow"] = "incorrect"
                joint_feedback["right_elbow"] = "incorrect"
                scores["elbow"] = 0.1

        # ── Head/neck alignment ──
        if head_neck is not None:
            if HEAD_NEUTRAL_MIN <= head_neck <= 185:
                scores["head_neck"] = 1.0
            elif head_neck < HEAD_ON_FLOOR:           # <140° — head on floor = CHEAT
                issues.append("Head resting on floor — lift your head in line with spine")
                joint_feedback["left_shoulder"] = "incorrect"
                scores["head_neck"] = 0.1
            elif head_neck < HEAD_DROPPING:           # <155° — head dropping
                issues.append("Head dropping — look at the floor just ahead of hands")
                joint_feedback["left_shoulder"] = "warning"
                scores["head_neck"] = 0.5
            elif head_neck > HEAD_CRANING:            # >190° — craning up
                issues.append("Neck craning up — keep head neutral, look at floor")
                joint_feedback["left_shoulder"] = "warning"
                scores["head_neck"] = 0.5

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

        # ── Shoulder alignment (over elbows for forearm plank) ──
        if shoulder_align is not None:
            if abs(shoulder_align) <= SHOULDER_ELBOW_GOOD:
                scores["shoulder"] = 1.0
            elif abs(shoulder_align) <= SHOULDER_ELBOW_GOOD * 2:
                issues.append("Shift shoulders over elbows")
                scores["shoulder"] = 0.6
            else:
                issues.append("Shoulders too far from elbows — reposition")
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
            "elbow": 1.3, "head_neck": 1.0,
        }
        weighted_sum = sum(scores[k] * WEIGHTS.get(k, 1.0) for k in scores)
        weight_total = sum(WEIGHTS.get(k, 1.0) for k in scores)
        total = weighted_sum / weight_total

        return max(0.0, min(1.0, total)), joint_feedback, issues
