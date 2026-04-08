"""
Dedicated tricep dip detector with state machine and form assessment.

Angles tracked:
  - Elbow (shoulder-elbow-wrist): primary rep counting + depth
  - Shoulder (elbow-shoulder-hip): forward lean / shoulder depression
  - Hip position: should stay close to bench

State machine: TOP (arms extended) -> GOING_DOWN -> BOTTOM (dipped) -> GOING_UP -> TOP (1 rep)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .base_detector import BaseExerciseDetector
from ..pose_estimation.base import PoseResult
from ..utils.constants import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
)


# ── Thresholds ──
ELBOW_EXTENDED = 155
ELBOW_DIPPED = 90       # Target depth
ELBOW_HALF = 110        # Partial dip

# Shoulder angle: forward lean control
SHOULDER_GOOD_MIN = 20
SHOULDER_GOOD_MAX = 50
SHOULDER_WARNING_MAX = 65
SHOULDER_BAD_MAX = 80   # Excessive forward lean — impingement risk


class TricepDipDetector(BaseExerciseDetector):
    EXERCISE_NAME = "tricep_dip"
    PRIMARY_ANGLE_TOP = ELBOW_EXTENDED
    PRIMARY_ANGLE_BOTTOM = ELBOW_DIPPED
    DESCENT_THRESHOLD = 10.0
    ASCENT_THRESHOLD = 10.0

    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        elbow = self._avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
            (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
            name="elbow",
        )
        shoulder = self._avg_angle(
            pose,
            (LEFT_ELBOW, LEFT_SHOULDER, LEFT_HIP),
            (RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP),
            name="shoulder",
        )
        hip_closeness = self._compute_hip_closeness(pose)
        return {"primary": elbow, "elbow": elbow, "shoulder": shoulder, "hip": hip_closeness}

    def _compute_hip_closeness(self, pose: PoseResult) -> Optional[float]:
        """Compute how far hips drift from shoulder vertical line (should stay close)."""
        offsets = []
        for s_idx, h_idx in [(LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP)]:
            if pose.is_visible(s_idx) and pose.is_visible(h_idx):
                s = np.array(pose.get_world_landmark(s_idx))
                h = np.array(pose.get_world_landmark(h_idx))
                # Horizontal distance (x-axis) — hips should be close to under shoulders
                offsets.append(abs(float(h[0] - s[0])))
        return sum(offsets) / len(offsets) if offsets else None

    def _check_start_position(self, angles: Dict[str, Optional[float]]) -> Tuple[bool, List[str]]:
        issues = []
        elbow = angles.get("elbow")
        if elbow is None:
            return False, ["Position camera to see arms — elbows must be visible"]
        if elbow < ELBOW_EXTENDED - 20:
            issues.append("Extend arms fully to start — press up on the bench")
        return (len(issues) == 0, issues)

    def _get_missing_parts(self, angles: Dict[str, Optional[float]]) -> List[str]:
        if angles.get("elbow") is None:
            return ["arms and elbows"]
        return ["body"]

    def _assess_form(self, angles: Dict[str, Optional[float]]) -> Tuple[float, Dict[str, str], List[str]]:
        joint_feedback = {}
        issues = []
        scores = {}

        elbow = angles.get("elbow")
        shoulder = angles.get("shoulder")
        hip = angles.get("hip")

        # ── Elbow depth ──
        if elbow is not None:
            if self._state in ("bottom", "going_up", "going_down"):
                if elbow <= ELBOW_DIPPED + 5:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 1.0
                elif elbow <= ELBOW_HALF:
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    issues.append("Go deeper — bend elbows to 90°")
                    scores["elbow"] = 0.5
                else:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 0.8
            else:
                if elbow >= ELBOW_EXTENDED - 10:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 1.0
                else:
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    issues.append("Extend arms fully at top")
                    scores["elbow"] = 0.6

        # ── Shoulder / forward lean (weighted 1.3x) ──
        if shoulder is not None:
            if SHOULDER_GOOD_MIN <= shoulder <= SHOULDER_GOOD_MAX:
                joint_feedback["left_shoulder"] = "correct"
                joint_feedback["right_shoulder"] = "correct"
                scores["shoulder"] = 1.0
            elif shoulder <= SHOULDER_WARNING_MAX:
                joint_feedback["left_shoulder"] = "warning"
                joint_feedback["right_shoulder"] = "warning"
                issues.append("Leaning forward — keep body close to bench")
                scores["shoulder"] = 0.5
            elif shoulder <= SHOULDER_BAD_MAX:
                joint_feedback["left_shoulder"] = "incorrect"
                joint_feedback["right_shoulder"] = "incorrect"
                issues.append("Too much forward lean — shoulder impingement risk")
                scores["shoulder"] = 0.2
            else:
                joint_feedback["left_shoulder"] = "incorrect"
                joint_feedback["right_shoulder"] = "incorrect"
                issues.append("Excessive lean — stay upright, push straight down")
                scores["shoulder"] = 0.1

        # ── Hip closeness (should stay near bench/shoulders) ──
        if hip is not None:
            if hip <= 0.05:  # within 5cm
                joint_feedback["left_hip"] = "correct"
                joint_feedback["right_hip"] = "correct"
                scores["hip"] = 1.0
            elif hip <= 0.12:
                joint_feedback["left_hip"] = "warning"
                joint_feedback["right_hip"] = "warning"
                issues.append("Keep hips close to the bench")
                scores["hip"] = 0.5
            else:
                joint_feedback["left_hip"] = "incorrect"
                joint_feedback["right_hip"] = "incorrect"
                issues.append("Hips too far from bench — stay close")
                scores["hip"] = 0.2

        if not scores:
            return 0.0, joint_feedback, issues

        WEIGHTS = {"elbow": 1.0, "shoulder": 1.3, "hip": 1.0}
        weighted_sum = sum(scores[k] * WEIGHTS.get(k, 1.0) for k in scores)
        weight_total = sum(WEIGHTS.get(k, 1.0) for k in scores)
        total = weighted_sum / weight_total

        return max(0.0, min(1.0, total)), joint_feedback, issues
