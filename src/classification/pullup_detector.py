"""
Dedicated pull-up detector with state machine and form assessment.

Angles tracked:
  - Elbow (shoulder-elbow-wrist): primary rep counting
  - Chin above bar: hands (wrists) should be below chin at top
  - Body swing: hip stability during pull

State machine: BOTTOM (hanging) -> GOING_UP -> TOP (chin above bar) -> GOING_DOWN -> BOTTOM (1 rep)
Note: inverted — starts at bottom, angle DECREASES going up.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .base_detector import (
    RobustExerciseDetector,
    RepPhase,
    REP_DIRECTION_DECREASING,
)
from ..pose_estimation.base import PoseResult
from ..utils.constants import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_EAR, RIGHT_EAR,
    NOSE,
)


# ── Thresholds ──
# Youdas et al. 2010 (Vicon motion capture) measured elbow angle of ~70-90°
# at the chin-clears-bar position. Previous 60° demanded hyperflexion and
# broke rep counting for correct form (same class of bug as old deadlift 90°).
# 85° is the upper bound of research + 5° margin for the state-machine gate.
ELBOW_HANGING = 170     # Full dead hang (research: 170-180°) — was 160
ELBOW_TOP = 85          # Chin-above-bar research position (was 60, too strict)
ELBOW_TOP_MAX_VALID = 100  # >100° at top = incomplete rep (for form scoring, not state gate)
ELBOW_HALF = 120        # Partial pull-up

# Body swing
SWING_GOOD = 10         # degrees of hip deviation
SWING_WARNING = 20
SWING_BAD = 30


class PullUpDetector(RobustExerciseDetector):
    EXERCISE_NAME = "pullup"
    PRIMARY_ANGLE_TOP = ELBOW_HANGING
    PRIMARY_ANGLE_BOTTOM = ELBOW_TOP
    DESCENT_THRESHOLD = 15.0
    ASCENT_THRESHOLD = 10.0
    MIN_FORM_GATE = 0.4  # higher — kipping produces full ROM but bad form

    # ── Robust FSM config ───────────────────────────────────────────────
    # Visibility-only gate — user confirmed pullup legs vary (strict
    # extended vs curled/tucked), so posture labels can't be relied on.
    # Gate on arm chain + hip visibility, which stays reliable whether
    # legs are bent or straight. Face occlusion from hands above head
    # is absorbed by UNKNOWN_GRACE_FRAMES.
    SESSION_POSTURE_GATE = None
    VISIBILITY_GATE_LANDMARKS = (
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_ELBOW, RIGHT_ELBOW,
        LEFT_WRIST, RIGHT_WRIST,
        LEFT_HIP, RIGHT_HIP,
    )
    VISIBILITY_GATE_THRESHOLD = 0.6

    REP_DIRECTION = REP_DIRECTION_DECREASING
    # Elbow: 170° hanging → 85° chin over bar → 170°. Strict thresholds
    # because pullups are hard and MIN_FORM_GATE = 0.4 already enforces
    # form quality — the rep FSM just needs to count the ROM.
    REP_COMPLETE_TOP = 160.0      # back at dead hang
    REP_START_THRESHOLD = 155.0   # committed to pull
    REP_DEPTH_THRESHOLD = 100.0   # must reach chin-near-bar
    REP_BOTTOM_CLEAR = 110.0
    MIN_REAL_PRIMARY_DEG = 30.0
    MAX_PRIMARY_JUMP_DEG = 80.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Baseline shoulder-to-ear distance captured during dead hang (T33)
        self._baseline_shoulder_ear: Optional[float] = None

    def reset(self):
        super().reset()
        self._baseline_shoulder_ear = None

    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        elbow = self._avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
            (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
            name="elbow",
        )

        chin_above = self._check_chin_above_bar(pose)
        body_swing = self._compute_body_swing(pose)
        scap_depression = self._compute_scapular_depression(pose)

        return {
            "primary": elbow, "elbow": elbow,
            "chin_above": chin_above, "body_swing": body_swing,
            "scap_depression": scap_depression,
        }

    def _check_chin_above_bar(self, pose: PoseResult) -> Optional[float]:
        """Check if chin/nose is above wrist level, normalized to torso length.
        T32: was raw pixel threshold which broke at different camera distances.
        Returns (nose - wrist) / torso_image_length → dimensionless ratio.
        Positive = nose above wrists (chin cleared bar)."""
        if not pose.is_visible(NOSE):
            return None

        wrist_ys = []
        for w_idx in [LEFT_WRIST, RIGHT_WRIST]:
            if pose.is_visible(w_idx):
                wrist_ys.append(pose.get_landmark(w_idx)[1])
        if not wrist_ys:
            return None

        # Torso length in image space for normalization
        shoulder_ys = []
        hip_ys = []
        for s_idx in [LEFT_SHOULDER, RIGHT_SHOULDER]:
            if pose.is_visible(s_idx):
                shoulder_ys.append(pose.get_landmark(s_idx)[1])
        for h_idx in [LEFT_HIP, RIGHT_HIP]:
            if pose.is_visible(h_idx):
                hip_ys.append(pose.get_landmark(h_idx)[1])
        if not shoulder_ys or not hip_ys:
            return None

        torso_len = abs((sum(hip_ys) / len(hip_ys)) - (sum(shoulder_ys) / len(shoulder_ys)))
        if torso_len < 0.05:  # degenerate — body too small in frame
            return None

        nose_y = pose.get_landmark(NOSE)[1]
        avg_wrist_y = sum(wrist_ys) / len(wrist_ys)
        # In image coords, Y increases downward → nose above wrists = nose_y < wrist_y
        return (avg_wrist_y - nose_y) / torso_len

    def _compute_body_swing(self, pose: PoseResult) -> Optional[float]:
        """Body swing angle: deviation of shoulder-hip line from pure vertical.
        T31: previous metric measured sagittal hip flexion (misleading).
        Strict pullup hangs the body straight down, so shoulder-hip line is
        vertical. Kipping swings the body → angle from vertical grows."""
        pts_s = []
        pts_h = []
        for s_idx in [LEFT_SHOULDER, RIGHT_SHOULDER]:
            if pose.is_visible(s_idx):
                pts_s.append(np.array(pose.get_world_landmark(s_idx)))
        for h_idx in [LEFT_HIP, RIGHT_HIP]:
            if pose.is_visible(h_idx):
                pts_h.append(np.array(pose.get_world_landmark(h_idx)))
        if not pts_s or not pts_h:
            return None

        mid_shoulder = np.mean(pts_s, axis=0)
        mid_hip = np.mean(pts_h, axis=0)
        diff = mid_hip - mid_shoulder  # pointing down in MediaPipe world (Y increases downward)
        norm = float(np.linalg.norm(diff))
        if norm < 0.1:
            return None
        vertical = np.array([0.0, 1.0, 0.0])
        cos = float(np.dot(diff, vertical) / norm)
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    def _compute_scapular_depression(self, pose: PoseResult) -> Optional[float]:
        """Shoulder-to-ear distance. Decreasing value = shoulder shrugging up
        toward the ears = loss of scapular depression (a common form fault).
        Returns (baseline - current) in meters. Positive value = shoulder rising toward ear."""
        dists = []
        for s_idx, ear_idx in [(LEFT_SHOULDER, LEFT_EAR), (RIGHT_SHOULDER, RIGHT_EAR)]:
            if pose.is_visible(s_idx) and pose.is_visible(ear_idx):
                s = np.array(pose.get_world_landmark(s_idx))
                ear = np.array(pose.get_world_landmark(ear_idx))
                dists.append(float(np.linalg.norm(s - ear)))
        if not dists:
            return None
        current = sum(dists) / len(dists)
        # Snapshot baseline at first hang
        if self._baseline_shoulder_ear is None and self._state == RepPhase.TOP:
            self._baseline_shoulder_ear = current
            return 0.0
        if self._baseline_shoulder_ear is None:
            return None
        return self._baseline_shoulder_ear - current  # positive = shrug toward ears

    def _check_start_position(self, angles: Dict[str, Optional[float]]) -> Tuple[bool, List[str]]:
        issues = []
        elbow = angles.get("elbow")
        if elbow is None:
            return False, ["Hang from bar — arms must be visible"]
        if elbow < ELBOW_HANGING - 30:
            issues.append("Extend arms fully — dead hang to start")
        return (len(issues) == 0, issues)

    def _get_missing_parts(self, angles: Dict[str, Optional[float]]) -> List[str]:
        if angles.get("elbow") is None:
            return ["arms"]
        return ["body"]

    def _assess_form(self, angles: Dict[str, Optional[float]]) -> Tuple[float, Dict[str, str], List[str]]:
        joint_feedback = {}
        issues = []
        scores = {}

        elbow = angles.get("elbow")
        chin_above = angles.get("chin_above")
        body_swing = angles.get("body_swing")
        scap_depression = angles.get("scap_depression")

        # Pullup state machine is inverted: TOP = dead hang, BOTTOM = chin over bar.
        in_pull = self._state in (RepPhase.BOTTOM, RepPhase.GOING_UP)

        # Pre-compute suppression: if swinging badly, don't nag about height
        _swing_active = body_swing is not None and body_swing > SWING_WARNING

        # ── Elbow / height ──
        if elbow is not None:
            if in_pull:
                if elbow <= ELBOW_TOP + 5:             # <= 90 — good chin-over-bar
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 1.0
                elif elbow <= ELBOW_TOP_MAX_VALID:     # 90-100 — valid but shallow
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 0.85
                elif elbow <= ELBOW_HALF:              # 100-120 — incomplete
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    if not _swing_active:
                        issues.append("Pull higher — chin toward the bar")
                    scores["elbow"] = 0.5
                else:
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    scores["elbow"] = 0.3
            else:
                # At top (dead hang) — should be fully extended
                if elbow >= ELBOW_HANGING - 10:        # >= 160
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 1.0
                else:
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    issues.append("Extend arms fully at the bottom — full dead hang")
                    scores["elbow"] = 0.6

        # ── Chin above bar (normalized to torso length, T32) ──
        # chin_above is now (wrist_y - nose_y) / torso_image_length, dimensionless.
        # Threshold: 0.05 = nose 5% of torso length above wrist (chin cleared).
        if chin_above is not None and in_pull:
            if chin_above > 0.05:
                scores["chin"] = 1.0
            elif chin_above > -0.05:
                if not _swing_active:
                    issues.append("Pull a bit higher — chin just below bar level")
                scores["chin"] = 0.6
            else:
                if not _swing_active:
                    issues.append("Not quite high enough — aim to get chin above the bar")
                scores["chin"] = 0.3

        # ── Body swing (kipping) — now in frontal-plane vertical deviation ──
        if body_swing is not None:
            if body_swing <= SWING_GOOD:
                joint_feedback["left_hip"] = "correct"
                joint_feedback["right_hip"] = "correct"
                scores["swing"] = 1.0
            elif body_swing <= SWING_WARNING:
                joint_feedback["left_hip"] = "warning"
                joint_feedback["right_hip"] = "warning"
                issues.append("Body swinging — keep core tight and controlled")
                scores["swing"] = 0.5
            else:
                joint_feedback["left_hip"] = "incorrect"
                joint_feedback["right_hip"] = "incorrect"
                issues.append("Too much swing — slow down and use strict form")
                scores["swing"] = 0.2

        # ── Scapular depression (T33) — shoulder shrugging up during pull ──
        # Positive scap_depression = shoulder rose toward ear = shrug = fault.
        if scap_depression is not None and scap_depression > 0.03:
            issues.append("Engage lats — pull shoulders down before pulling up")
            scores["scap"] = 0.5

        if not scores:
            return 0.0, joint_feedback, issues

        WEIGHTS = {"elbow": 1.0, "chin": 1.0, "swing": 1.3, "scap": 0.8}
        weighted_sum = sum(scores[k] * WEIGHTS.get(k, 1.0) for k in scores)
        weight_total = sum(WEIGHTS.get(k, 1.0) for k in scores)
        total = weighted_sum / weight_total

        return max(0.0, min(1.0, total)), joint_feedback, issues
