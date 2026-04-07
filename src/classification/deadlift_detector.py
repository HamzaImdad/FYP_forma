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
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
)


# ── Thresholds ──
HIP_STANDING = 165
HIP_BOTTOM = 90          # Full hip hinge
HIP_HALF = 120           # Partial hinge

KNEE_MAX_BEND = 140      # Knees should stay relatively straight
KNEE_MIN = 160           # Ideal slight bend
KNEE_SQUAT = 120         # Too much knee bend = squatting not hinging

BACK_GOOD = 168          # Nearly straight back (spine curvature proxy)
BACK_WARNING = 160       # Slight rounding
BACK_BAD = 145           # Significant rounding


class DeadliftDetector(BaseExerciseDetector):
    EXERCISE_NAME = "deadlift"
    PRIMARY_ANGLE_TOP = HIP_STANDING
    PRIMARY_ANGLE_BOTTOM = HIP_BOTTOM
    DESCENT_THRESHOLD = 15.0
    ASCENT_THRESHOLD = 10.0

    def __init__(self):
        super().__init__()
        self._prev_shoulder_y = None
        self._prev_hip_y = None

    def reset(self):
        super().reset()
        self._prev_shoulder_y = None
        self._prev_hip_y = None

    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        hip = self._avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
            (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
            name="hip",
        )
        knee = self._avg_angle(
            pose,
            (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
            (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
            name="knee",
        )

        # Back straightness: spine curvature proxy (NOSE -> mid-shoulder -> mid-hip)
        back = self._compute_back_angle(pose)

        # Hip shoot detection (hips rising faster than shoulders)
        hip_shoot = self._check_hip_shoot(pose)

        return {"primary": hip, "hip": hip, "knee": knee, "back": back, "hip_shoot": hip_shoot}

    def _compute_back_angle(self, pose: PoseResult) -> Optional[float]:
        """Measure spine curvature via angle at mid-shoulder (NOSE -> mid-shoulder -> mid-hip).
        Detects upper back rounding that shoulder-hip-knee misses."""
        if not pose.is_visible(NOSE):
            # Fall back to original method
            left = self._angle_at(pose, LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE)
            right = self._angle_at(pose, RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE)
            if left is not None and right is not None:
                return (left + right) / 2
            return left if left is not None else right

        # Get mid-shoulder and mid-hip
        ls = pose.get_world_landmark(LEFT_SHOULDER) if pose.is_visible(LEFT_SHOULDER) else None
        rs = pose.get_world_landmark(RIGHT_SHOULDER) if pose.is_visible(RIGHT_SHOULDER) else None
        lh = pose.get_world_landmark(LEFT_HIP) if pose.is_visible(LEFT_HIP) else None
        rh = pose.get_world_landmark(RIGHT_HIP) if pose.is_visible(RIGHT_HIP) else None

        if (ls is None and rs is None) or (lh is None and rh is None):
            return None

        nose = np.array(pose.get_world_landmark(NOSE))
        mid_shoulder = np.array(ls if rs is None else (rs if ls is None else (np.array(ls) + np.array(rs)) / 2))
        mid_hip = np.array(lh if rh is None else (rh if lh is None else (np.array(lh) + np.array(rh)) / 2))

        from ..utils.geometry import calculate_angle
        return calculate_angle(nose, mid_shoulder, mid_hip)

    def _check_hip_shoot(self, pose: PoseResult) -> Optional[float]:
        """Detect if hips rise faster than shoulders during ascent (stripper pull).
        Returns ratio of hip_rise/shoulder_rise. >1.5 = hip shoot."""
        ls = pose.get_world_landmark(LEFT_SHOULDER) if pose.is_visible(LEFT_SHOULDER) else None
        rs = pose.get_world_landmark(RIGHT_SHOULDER) if pose.is_visible(RIGHT_SHOULDER) else None
        lh = pose.get_world_landmark(LEFT_HIP) if pose.is_visible(LEFT_HIP) else None
        rh = pose.get_world_landmark(RIGHT_HIP) if pose.is_visible(RIGHT_HIP) else None

        if (ls is None and rs is None) or (lh is None and rh is None):
            self._prev_shoulder_y = None
            self._prev_hip_y = None
            return None

        shoulder_y = float(np.mean([l[1] for l in [ls, rs] if l is not None]))
        hip_y = float(np.mean([l[1] for l in [lh, rh] if l is not None]))

        ratio = None
        if self._prev_shoulder_y is not None and self._prev_hip_y is not None:
            shoulder_rise = self._prev_shoulder_y - shoulder_y  # positive = rising (Y decreases going up in MediaPipe)
            hip_rise = self._prev_hip_y - hip_y
            if shoulder_rise > 0.001:  # shoulders actually moving up
                ratio = hip_rise / shoulder_rise

        self._prev_shoulder_y = shoulder_y
        self._prev_hip_y = hip_y
        return ratio

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
                # Check knee lockout at top
                if knee is not None and knee < 170:
                    issues.append("Extend knees fully at lockout")

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
            # Spine curvature proxy: NOSE->mid-shoulder->mid-hip angle
            # Higher = straighter back
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

        # ── Hip shoot (during ascent) ──
        hip_shoot = angles.get("hip_shoot")
        if hip_shoot is not None and self._state == "going_up":
            if hip_shoot > 2.0:
                issues.append("Hips shooting up! Lead with your chest")
                scores["hip_shoot"] = 0.2
            elif hip_shoot > 1.5:
                issues.append("Hips rising too fast -- drive chest up together")
                scores["hip_shoot"] = 0.5

        if not scores:
            return 0.0, joint_feedback, issues

        WEIGHTS = {"hip": 1.0, "knee": 1.3, "back": 1.5, "hip_shoot": 1.2}
        weighted_sum = sum(scores[k] * WEIGHTS.get(k, 1.0) for k in scores)
        weight_total = sum(WEIGHTS.get(k, 1.0) for k in scores)
        total = weighted_sum / weight_total

        return max(0.0, min(1.0, total)), joint_feedback, issues
