"""
Dedicated lunge detector with state machine and form assessment.

Angles tracked:
  - Front knee (hip-knee-ankle): primary rep counting angle
  - Back knee: should approach floor
  - Torso: should stay upright
  - Front shin: should be roughly vertical

State machine: TOP (standing) -> GOING_DOWN -> BOTTOM (lunged) -> GOING_UP -> TOP (1 rep)
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
KNEE_STANDING = 160
KNEE_LUNGED = 90        # Front knee at 90 degrees
KNEE_HALF = 120         # Partial lunge

BACK_KNEE_GOOD = 100    # Back knee approaching floor
BACK_KNEE_WARNING = 130 # Not deep enough

TORSO_GOOD = 15         # degrees from vertical
TORSO_WARNING = 30
TORSO_BAD = 45


class LungeDetector(BaseExerciseDetector):
    EXERCISE_NAME = "lunge"
    PRIMARY_ANGLE_TOP = KNEE_STANDING
    PRIMARY_ANGLE_BOTTOM = KNEE_LUNGED
    DESCENT_THRESHOLD = 15.0
    ASCENT_THRESHOLD = 10.0

    FRONT_LEG_LOCK_FRAMES = 15  # keep front leg decision stable for this many frames

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._front_leg = None  # "left" or "right"
        self._front_leg_lock = 0  # frames remaining in current lock

    def reset(self):
        super().reset()
        self._front_leg = None
        self._front_leg_lock = 0

    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        left_knee = self._angle_at(pose, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
        right_knee = self._angle_at(pose, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)

        # Determine which leg is in front (lower knee angle = front leg)
        # Use hysteresis: once decided, lock for FRONT_LEG_LOCK_FRAMES frames
        if left_knee is not None and right_knee is not None:
            if self._front_leg_lock > 0:
                self._front_leg_lock -= 1
            else:
                # Re-evaluate: only switch if difference is significant (>15°)
                diff = left_knee - right_knee
                if diff < -15:
                    self._front_leg = "left"
                    self._front_leg_lock = self.FRONT_LEG_LOCK_FRAMES
                elif diff > 15:
                    self._front_leg = "right"
                    self._front_leg_lock = self.FRONT_LEG_LOCK_FRAMES
                elif self._front_leg is None:
                    # First detection — use smaller threshold
                    self._front_leg = "left" if left_knee < right_knee else "right"
                    self._front_leg_lock = self.FRONT_LEG_LOCK_FRAMES

            if self._front_leg == "left":
                front_knee = left_knee
                back_knee = right_knee
            else:
                front_knee = right_knee
                back_knee = left_knee
        elif left_knee is not None:
            self._front_leg = "left"
            front_knee = left_knee
            back_knee = None
        elif right_knee is not None:
            self._front_leg = "right"
            front_knee = right_knee
            back_knee = None
        else:
            front_knee = None
            back_knee = None

        torso = self._compute_torso_lean(pose)

        return {
            "primary": front_knee,
            "front_knee": front_knee,
            "back_knee": back_knee,
            "torso": torso,
        }

    def _compute_torso_lean(self, pose: PoseResult) -> Optional[float]:
        """Compute torso lean from vertical."""
        angles = []
        for s_idx, h_idx in [(LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP)]:
            if pose.is_visible(s_idx) and pose.is_visible(h_idx):
                s = np.array(pose.get_world_landmark(s_idx))
                h = np.array(pose.get_world_landmark(h_idx))
                diff = s - h
                vertical = np.array([0, 1, 0])
                cos = np.dot(diff, vertical) / (np.linalg.norm(diff) + 1e-8)
                angles.append(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
        return sum(angles) / len(angles) if angles else None

    def _check_start_position(self, angles: Dict[str, Optional[float]]) -> Tuple[bool, List[str]]:
        issues = []
        fk = angles.get("front_knee")
        if fk is None:
            return False, ["Stand facing camera — legs must be visible"]
        if fk < KNEE_STANDING - 20:
            issues.append("Stand up straight to begin")
        return (len(issues) == 0, issues)

    def _get_missing_parts(self, angles: Dict[str, Optional[float]]) -> List[str]:
        if angles.get("front_knee") is None:
            return ["legs"]
        return ["body"]

    def _assess_form(self, angles: Dict[str, Optional[float]]) -> Tuple[float, Dict[str, str], List[str]]:
        joint_feedback = {}
        issues = []
        scores = {}

        front_knee = angles.get("front_knee")
        back_knee = angles.get("back_knee")
        torso = angles.get("torso")

        fl = self._front_leg or "left"
        bl = "right" if fl == "left" else "left"

        # ── Front knee depth ──
        if front_knee is not None:
            if self._state in ("bottom", "going_up", "going_down"):
                if front_knee <= KNEE_LUNGED + 10:
                    joint_feedback[f"{fl}_knee"] = "correct"
                    scores["front_knee"] = 1.0
                elif front_knee <= KNEE_HALF:
                    joint_feedback[f"{fl}_knee"] = "warning"
                    issues.append("Go deeper — front thigh should be parallel to floor")
                    scores["front_knee"] = 0.5
                else:
                    joint_feedback[f"{fl}_knee"] = "correct"
                    scores["front_knee"] = 0.8
            else:
                joint_feedback[f"{fl}_knee"] = "correct"
                scores["front_knee"] = 1.0

        # ── Back knee depth ──
        if back_knee is not None and self._state in ("bottom", "going_up", "going_down"):
            if back_knee <= BACK_KNEE_GOOD:
                joint_feedback[f"{bl}_knee"] = "correct"
                scores["back_knee"] = 1.0
            elif back_knee <= BACK_KNEE_WARNING:
                joint_feedback[f"{bl}_knee"] = "warning"
                issues.append("Drop back knee closer to floor")
                scores["back_knee"] = 0.5
            else:
                joint_feedback[f"{bl}_knee"] = "warning"
                scores["back_knee"] = 0.7

        # ── Torso upright (weighted 1.3x) ──
        if torso is not None:
            if torso <= TORSO_GOOD:
                joint_feedback["left_shoulder"] = "correct"
                joint_feedback["right_shoulder"] = "correct"
                scores["torso"] = 1.0
            elif torso <= TORSO_WARNING:
                joint_feedback["left_shoulder"] = "warning"
                joint_feedback["right_shoulder"] = "warning"
                issues.append("Keep torso upright — don't lean forward")
                scores["torso"] = 0.5
            else:
                joint_feedback["left_shoulder"] = "incorrect"
                joint_feedback["right_shoulder"] = "incorrect"
                issues.append("Excessive forward lean — chest up!")
                scores["torso"] = 0.2

        if not scores:
            return 0.0, joint_feedback, issues

        WEIGHTS = {"front_knee": 1.0, "back_knee": 1.0, "torso": 1.3}
        weighted_sum = sum(scores[k] * WEIGHTS.get(k, 1.0) for k in scores)
        weight_total = sum(WEIGHTS.get(k, 1.0) for k in scores)
        total = weighted_sum / weight_total

        return max(0.0, min(1.0, total)), joint_feedback, issues
