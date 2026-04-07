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

from .base_detector import BaseExerciseDetector
from ..pose_estimation.base import PoseResult
from ..utils.constants import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    NOSE,
)


# ── Thresholds ──
ELBOW_HANGING = 160     # Arms extended (dead hang)
ELBOW_TOP = 60          # Chin above bar
ELBOW_HALF = 110        # Partial pull-up

# Body swing
SWING_GOOD = 10         # degrees of hip deviation
SWING_WARNING = 20
SWING_BAD = 30


class PullUpDetector(BaseExerciseDetector):
    EXERCISE_NAME = "pullup"
    PRIMARY_ANGLE_TOP = ELBOW_HANGING
    PRIMARY_ANGLE_BOTTOM = ELBOW_TOP
    DESCENT_THRESHOLD = 15.0
    ASCENT_THRESHOLD = 10.0

    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        elbow = self._avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
            (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
        )

        chin_above = self._check_chin_above_bar(pose)
        body_swing = self._compute_body_swing(pose)

        return {
            "primary": elbow, "elbow": elbow,
            "chin_above": chin_above, "body_swing": body_swing,
        }

    def _check_chin_above_bar(self, pose: PoseResult) -> Optional[float]:
        """Check if chin/nose is above wrist level. Returns signed distance (positive = above)."""
        if not pose.is_visible(NOSE):
            return None

        wrist_ys = []
        for w_idx in [LEFT_WRIST, RIGHT_WRIST]:
            if pose.is_visible(w_idx):
                wrist_ys.append(pose.get_landmark(w_idx)[1])

        if not wrist_ys:
            return None

        nose_y = pose.get_landmark(NOSE)[1]
        avg_wrist_y = sum(wrist_ys) / len(wrist_ys)
        # In image coords, y increases downward
        return avg_wrist_y - nose_y  # positive = nose above wrists

    def _compute_body_swing(self, pose: PoseResult) -> Optional[float]:
        """Compute body swing as hip deviation from shoulder-ankle line."""
        angles = []
        for s_idx, h_idx, k_idx in [(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
                                     (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE)]:
            if all(pose.is_visible(i) for i in (s_idx, h_idx, k_idx)):
                s = np.array(pose.get_world_landmark(s_idx))
                h = np.array(pose.get_world_landmark(h_idx))
                k = np.array(pose.get_world_landmark(k_idx))
                # Angle at hip — should be close to 180 (straight body)
                v1 = s - h
                v2 = k - h
                cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                angle = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
                angles.append(abs(180 - angle))  # deviation from straight
        return sum(angles) / len(angles) if angles else None

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
        scores = []

        elbow = angles.get("elbow")
        chin_above = angles.get("chin_above")
        body_swing = angles.get("body_swing")

        # ── Elbow / height ──
        if elbow is not None:
            if self._state in ("bottom", "going_up"):
                if elbow <= ELBOW_TOP + 15:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores.append(1.0)
                elif elbow <= ELBOW_HALF:
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    issues.append("Pull higher — chin should clear the bar")
                    scores.append(0.5)
                else:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores.append(0.8)
            else:
                # At bottom — should be fully extended
                if elbow >= ELBOW_HANGING - 15:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores.append(1.0)
                else:
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    issues.append("Extend arms fully at the bottom — full dead hang")
                    scores.append(0.6)

        # ── Chin above bar ──
        if chin_above is not None and self._state in ("bottom",):
            if chin_above > 0.02:
                scores.append(1.0)
            elif chin_above > -0.02:
                issues.append("Pull a bit higher — chin just below bar level")
                scores.append(0.6)
            else:
                issues.append("Not high enough — chin must clear the bar")
                scores.append(0.3)

        # ── Body swing (kipping) — weighted 1.3x ──
        if body_swing is not None:
            if body_swing <= SWING_GOOD:
                joint_feedback["left_hip"] = "correct"
                joint_feedback["right_hip"] = "correct"
                scores.append(1.0)
            elif body_swing <= SWING_WARNING:
                joint_feedback["left_hip"] = "warning"
                joint_feedback["right_hip"] = "warning"
                issues.append("Body swinging — keep core tight, no kipping")
                scores.append(0.5)
            else:
                joint_feedback["left_hip"] = "incorrect"
                joint_feedback["right_hip"] = "incorrect"
                issues.append("Excessive kipping — use strict form, no swinging")
                scores.append(0.2)

        if not scores:
            return 0.0, joint_feedback, issues

        if len(scores) >= 3:
            weights = [1.0, 1.0, 1.3][:len(scores)]
            weighted = [s * w for s, w in zip(scores, weights)]
            total = sum(weighted) / sum(weights)
        else:
            total = sum(scores) / len(scores)

        return max(0.0, min(1.0, total)), joint_feedback, issues
