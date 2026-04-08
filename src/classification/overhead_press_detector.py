"""
Dedicated overhead press detector with state machine and form assessment.

Angles tracked:
  - Elbow (shoulder-elbow-wrist): primary rep counting
  - Lockout angle (full extension at top)
  - Torso lean (should stay upright, not arch back)

State machine: BOTTOM (bar at shoulders) -> GOING_UP -> TOP (locked out) -> GOING_DOWN -> BOTTOM (1 rep)
Note: inverted from most exercises — starts at bottom, counts when returning to bottom.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .base_detector import BaseExerciseDetector, RepPhase
from ..pose_estimation.base import PoseResult
from ..utils.constants import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
)


# ── Thresholds ──
ELBOW_LOCKED = 160      # Arms fully extended overhead
ELBOW_SHOULDER = 80     # Bar at shoulder level
ELBOW_HALF = 120        # Partial press

TORSO_GOOD = 10         # degrees from vertical
TORSO_WARNING = 20      # slight back arch
TORSO_BAD = 30          # excessive arch


class OverheadPressDetector(BaseExerciseDetector):
    EXERCISE_NAME = "overhead_press"
    # Inverted: "top" is bottom (bar at shoulders), primary angle INCREASES as you press
    PRIMARY_ANGLE_TOP = ELBOW_LOCKED
    PRIMARY_ANGLE_BOTTOM = ELBOW_SHOULDER
    DESCENT_THRESHOLD = 10.0
    ASCENT_THRESHOLD = 10.0

    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        elbow = self._avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
            (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
            name="elbow",
        )
        torso = self._compute_torso_lean(pose)

        return {"primary": elbow, "elbow": elbow, "torso": torso}

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
        elbow = angles.get("elbow")
        if elbow is None:
            return False, ["Stand facing camera — arms must be visible"]
        # Accept bar at shoulders (65-95°) OR arms locked out (150-175°)
        at_shoulders = 65 <= elbow <= 95
        locked_out = 150 <= elbow <= 175
        if at_shoulders or locked_out:
            return True, []
        if elbow < 65:
            issues.append("Raise bar to shoulder height to begin")
        elif elbow > 175:
            issues.append("Arms too far back — bring bar to shoulders")
        else:
            issues.append("Hold bar at shoulder height or lock out overhead to begin")
        return False, issues

    def _get_missing_parts(self, angles: Dict[str, Optional[float]]) -> List[str]:
        if angles.get("elbow") is None:
            return ["arms"]
        return ["body"]

    def _assess_form(self, angles: Dict[str, Optional[float]]) -> Tuple[float, Dict[str, str], List[str]]:
        joint_feedback = {}
        issues = []
        scores = {}

        elbow = angles.get("elbow")
        torso = angles.get("torso")

        # ── Elbow / lockout ──
        if elbow is not None:
            if self._state in ("top", "going_down"):
                # At top or descending — should be locked out
                if elbow >= ELBOW_LOCKED:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 1.0
                elif elbow >= ELBOW_HALF:
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    issues.append("Lock out arms fully overhead")
                    scores["elbow"] = 0.6
                else:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 0.8
            else:
                # At bottom — bar should be at shoulders
                if elbow <= ELBOW_SHOULDER + 10:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 1.0
                else:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 0.8

        # ── Torso lean (weighted 1.5x — back arch is dangerous) ──
        if torso is not None:
            if torso <= TORSO_GOOD:
                joint_feedback["left_hip"] = "correct"
                joint_feedback["right_hip"] = "correct"
                scores["torso"] = 1.0
            elif torso <= TORSO_WARNING:
                joint_feedback["left_hip"] = "warning"
                joint_feedback["right_hip"] = "warning"
                issues.append("Don't arch back — brace core, stay upright")
                scores["torso"] = 0.5
            else:
                joint_feedback["left_hip"] = "incorrect"
                joint_feedback["right_hip"] = "incorrect"
                issues.append("Excessive back arch — reduce weight, brace core!")
                scores["torso"] = 0.2

        if not scores:
            return 0.0, joint_feedback, issues

        WEIGHTS = {"elbow": 1.0, "torso": 1.5}
        weighted_sum = sum(scores[k] * WEIGHTS.get(k, 1.0) for k in scores)
        weight_total = sum(WEIGHTS.get(k, 1.0) for k in scores)
        total = weighted_sum / weight_total

        return max(0.0, min(1.0, total)), joint_feedback, issues
