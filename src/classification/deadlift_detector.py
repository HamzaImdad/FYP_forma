"""
Dedicated deadlift detector with state machine and form assessment.

Angles tracked:
  - Hip (shoulder-hip-knee): primary angle for hip hinge pattern
  - Knee (hip-knee-ankle): slight bend, not excessive
  - Back straightness (shoulder-hip alignment)

State machine: TOP (standing) -> GOING_DOWN (hinge) -> BOTTOM -> GOING_UP -> TOP (1 rep)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .base_detector import BaseExerciseDetector
from ..pose_estimation.base import PoseResult
from ..utils.constants import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
)


# ── Thresholds ──
HIP_STANDING = 165
HIP_BOTTOM = 90          # Full hip hinge
HIP_HALF = 120           # Partial hinge

KNEE_MAX_BEND = 140      # Knees should stay relatively straight
KNEE_MIN = 160           # Ideal slight bend
KNEE_SQUAT = 120         # Too much knee bend = squatting not hinging

BACK_GOOD = 175          # Nearly straight back (shoulder-hip-knee angle close)
BACK_WARNING = 160       # Slight rounding
BACK_BAD = 145           # Significant rounding


class DeadliftDetector(BaseExerciseDetector):
    EXERCISE_NAME = "deadlift"
    PRIMARY_ANGLE_TOP = HIP_STANDING
    PRIMARY_ANGLE_BOTTOM = HIP_BOTTOM
    DESCENT_THRESHOLD = 15.0
    ASCENT_THRESHOLD = 10.0

    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        hip = self._avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
            (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
        )
        knee = self._avg_angle(
            pose,
            (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
            (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
        )

        # Back straightness: shoulder-hip-knee angle
        back = self._compute_back_angle(pose)

        return {"primary": hip, "hip": hip, "knee": knee, "back": back}

    def _compute_back_angle(self, pose: PoseResult) -> Optional[float]:
        """Measure back straightness via shoulder-hip alignment relative to hip-knee."""
        # Use world coords for 3D accuracy
        left = self._angle_at(pose, LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE)
        right = self._angle_at(pose, RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE)
        if left is not None and right is not None:
            return (left + right) / 2
        return left if left is not None else right

    def _check_start_position(self, angles: Dict[str, Optional[float]]) -> Tuple[bool, List[str]]:
        issues = []
        hip = angles.get("hip")
        if hip is None:
            return False, ["Stand facing the camera — full body must be visible"]
        if hip < HIP_STANDING - 20:
            issues.append("Stand up straight to begin")
        return (len(issues) == 0, issues)

    def _get_missing_parts(self, angles: Dict[str, Optional[float]]) -> List[str]:
        if angles.get("hip") is None:
            return ["hips and shoulders"]
        return ["body"]

    def _assess_form(self, angles: Dict[str, Optional[float]]) -> Tuple[float, Dict[str, str], List[str]]:
        joint_feedback = {}
        issues = []
        scores = {}

        hip = angles.get("hip")
        knee = angles.get("knee")
        back = angles.get("back")

        # ── Hip hinge depth ──
        if hip is not None:
            if self._state in ("bottom", "going_up", "going_down"):
                if hip <= HIP_BOTTOM + 10:
                    joint_feedback["left_hip"] = "correct"
                    joint_feedback["right_hip"] = "correct"
                    scores["hip"] = 1.0
                elif hip <= HIP_HALF:
                    joint_feedback["left_hip"] = "warning"
                    joint_feedback["right_hip"] = "warning"
                    issues.append("Hinge deeper — push hips back further")
                    scores["hip"] = 0.5
                else:
                    joint_feedback["left_hip"] = "correct"
                    joint_feedback["right_hip"] = "correct"
                    scores["hip"] = 0.8
            else:
                # At top — should be fully extended
                if hip >= HIP_STANDING - 10:
                    joint_feedback["left_hip"] = "correct"
                    joint_feedback["right_hip"] = "correct"
                    scores["hip"] = 1.0
                else:
                    joint_feedback["left_hip"] = "warning"
                    joint_feedback["right_hip"] = "warning"
                    issues.append("Stand up fully at the top — lock out hips")
                    scores["hip"] = 0.6

        # ── Knee bend (should NOT squat) — weighted 1.3x ──
        if knee is not None:
            if knee >= KNEE_MIN:
                joint_feedback["left_knee"] = "correct"
                joint_feedback["right_knee"] = "correct"
                scores["knee"] = 1.0
            elif knee >= KNEE_MAX_BEND:
                joint_feedback["left_knee"] = "warning"
                joint_feedback["right_knee"] = "warning"
                issues.append("Keep legs straighter — this is a hip hinge, not a squat")
                scores["knee"] = 0.5
            elif knee >= KNEE_SQUAT:
                joint_feedback["left_knee"] = "incorrect"
                joint_feedback["right_knee"] = "incorrect"
                issues.append("Too much knee bend — push hips back instead of bending knees")
                scores["knee"] = 0.2
            else:
                joint_feedback["left_knee"] = "incorrect"
                joint_feedback["right_knee"] = "incorrect"
                issues.append("You're squatting, not deadlifting — keep knees soft but mostly straight")
                scores["knee"] = 0.1

        # ── Back straightness (most important — weighted 1.5x) ──
        if back is not None:
            # During hinge, shoulder-hip-knee angle tells us about back rounding
            # A straight back means this angle stays close to hip angle
            if back >= BACK_WARNING:
                joint_feedback["left_shoulder"] = "correct"
                joint_feedback["right_shoulder"] = "correct"
                scores["back"] = 1.0
            elif back >= BACK_BAD:
                joint_feedback["left_shoulder"] = "warning"
                joint_feedback["right_shoulder"] = "warning"
                issues.append("Keep back straight — chest up, shoulders back")
                scores["back"] = 0.4
            else:
                joint_feedback["left_shoulder"] = "incorrect"
                joint_feedback["right_shoulder"] = "incorrect"
                issues.append("Back rounding! Straighten spine — risk of injury")
                scores["back"] = 0.1

        if not scores:
            return 0.0, joint_feedback, issues

        WEIGHTS = {"hip": 1.0, "knee": 1.3, "back": 1.5}
        weighted_sum = sum(scores[k] * WEIGHTS.get(k, 1.0) for k in scores)
        weight_total = sum(WEIGHTS.get(k, 1.0) for k in scores)
        total = weighted_sum / weight_total

        return max(0.0, min(1.0, total)), joint_feedback, issues
