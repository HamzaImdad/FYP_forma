"""
Dedicated bicep curl detector with state machine and form assessment.

Angles tracked:
  - Elbow (shoulder-elbow-wrist): primary rep counting angle
  - Upper arm stability (shoulder-elbow vertical angle)
  - Torso swing (shoulder-hip vertical angle)

State machine: TOP (extended) -> GOING_DOWN (curling) -> BOTTOM (curled) -> GOING_UP -> TOP (1 rep)
Note: "down" here means curling UP the weight (elbow angle decreases).
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
ELBOW_EXTENDED = 155    # Arms fully extended
ELBOW_CURLED = 45       # Fully curled
ELBOW_HALF = 90         # Partial curl

# Upper arm should stay vertical (pinned to side)
UPPER_ARM_GOOD = 20     # degrees from vertical
UPPER_ARM_WARNING = 35  # swinging elbow forward/back
UPPER_ARM_BAD = 50      # major upper arm movement

# Torso swing (using momentum instead of bicep)
TORSO_GOOD = 10         # degrees of lean from vertical
TORSO_WARNING = 20
TORSO_BAD = 30


class BicepCurlDetector(BaseExerciseDetector):
    EXERCISE_NAME = "bicep_curl"
    PRIMARY_ANGLE_TOP = ELBOW_EXTENDED
    PRIMARY_ANGLE_BOTTOM = ELBOW_CURLED
    DESCENT_THRESHOLD = 15.0
    ASCENT_THRESHOLD = 10.0

    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        elbow = self._avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
            (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
        )

        upper_arm = self._compute_upper_arm_stability(pose)
        torso = self._compute_torso_swing(pose)

        return {"primary": elbow, "elbow": elbow, "upper_arm": upper_arm, "torso": torso}

    def _compute_upper_arm_stability(self, pose: PoseResult) -> Optional[float]:
        """Compute how much upper arm deviates from vertical (shoulder-elbow line)."""
        angles = []
        for s_idx, e_idx in [(LEFT_SHOULDER, LEFT_ELBOW), (RIGHT_SHOULDER, RIGHT_ELBOW)]:
            if pose.is_visible(s_idx) and pose.is_visible(e_idx):
                s = np.array(pose.get_world_landmark(s_idx))
                e = np.array(pose.get_world_landmark(e_idx))
                diff = e - s
                vertical = np.array([0, -1, 0])  # down is negative y in world coords
                cos = np.dot(diff, vertical) / (np.linalg.norm(diff) + 1e-8)
                angles.append(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
        return sum(angles) / len(angles) if angles else None

    def _compute_torso_swing(self, pose: PoseResult) -> Optional[float]:
        """Compute torso lean from vertical (shoulder-hip line)."""
        angles = []
        for s_idx, h_idx in [(LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP)]:
            if pose.is_visible(s_idx) and pose.is_visible(h_idx):
                s = np.array(pose.get_world_landmark(s_idx))
                h = np.array(pose.get_world_landmark(h_idx))
                diff = s - h  # shoulder above hip
                vertical = np.array([0, 1, 0])
                cos = np.dot(diff, vertical) / (np.linalg.norm(diff) + 1e-8)
                angles.append(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
        return sum(angles) / len(angles) if angles else None

    def _check_start_position(self, angles: Dict[str, Optional[float]]) -> Tuple[bool, List[str]]:
        issues = []
        elbow = angles.get("elbow")
        if elbow is None:
            return False, ["Stand facing camera — arms must be visible"]
        if elbow < ELBOW_EXTENDED - 30:
            issues.append("Extend arms fully to begin — let weights hang at sides")
        return (len(issues) == 0, issues)

    def _get_missing_parts(self, angles: Dict[str, Optional[float]]) -> List[str]:
        if angles.get("elbow") is None:
            return ["arms and elbows"]
        return ["body"]

    def _assess_form(self, angles: Dict[str, Optional[float]]) -> Tuple[float, Dict[str, str], List[str]]:
        joint_feedback = {}
        issues = []
        scores = []

        elbow = angles.get("elbow")
        upper_arm = angles.get("upper_arm")
        torso = angles.get("torso")

        # ── Elbow ROM ──
        if elbow is not None:
            if self._state in ("bottom", "going_up", "going_down"):
                if elbow <= ELBOW_CURLED + 15:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores.append(1.0)
                elif elbow <= ELBOW_HALF:
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    issues.append("Curl higher — squeeze at the top")
                    scores.append(0.5)
                else:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores.append(0.8)
            else:
                if elbow >= ELBOW_EXTENDED - 10:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores.append(1.0)
                else:
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    issues.append("Extend arms fully at the bottom")
                    scores.append(0.6)

        # ── Upper arm stability (most important — weighted 1.5x) ──
        if upper_arm is not None:
            if upper_arm <= UPPER_ARM_GOOD:
                joint_feedback["left_shoulder"] = "correct"
                joint_feedback["right_shoulder"] = "correct"
                scores.append(1.0)
            elif upper_arm <= UPPER_ARM_WARNING:
                joint_feedback["left_shoulder"] = "warning"
                joint_feedback["right_shoulder"] = "warning"
                issues.append("Pin elbows to your sides — upper arm is moving")
                scores.append(0.5)
            else:
                joint_feedback["left_shoulder"] = "incorrect"
                joint_feedback["right_shoulder"] = "incorrect"
                issues.append("Elbows swinging! Keep upper arms stationary")
                scores.append(0.2)

        # ── Torso swing (cheating with momentum) ──
        if torso is not None:
            if torso <= TORSO_GOOD:
                joint_feedback["left_hip"] = "correct"
                joint_feedback["right_hip"] = "correct"
                scores.append(1.0)
            elif torso <= TORSO_WARNING:
                joint_feedback["left_hip"] = "warning"
                joint_feedback["right_hip"] = "warning"
                issues.append("Slight body swing — stay more upright")
                scores.append(0.5)
            else:
                joint_feedback["left_hip"] = "incorrect"
                joint_feedback["right_hip"] = "incorrect"
                issues.append("Too much body swing — reduce weight or slow down")
                scores.append(0.2)

        if not scores:
            return 0.0, joint_feedback, issues

        # Weight upper arm stability highest
        if len(scores) >= 2:
            weights = [1.0, 1.5, 1.2][:len(scores)]
            weighted = [s * w for s, w in zip(scores, weights)]
            total = sum(weighted) / sum(weights)
        else:
            total = sum(scores) / len(scores)

        return max(0.0, min(1.0, total)), joint_feedback, issues
