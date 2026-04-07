"""
Dedicated squat detector with state machine rep counting and form assessment.

Angles tracked:
  - Knee (hip-knee-ankle): primary rep counting angle + depth check
  - Hip (shoulder-hip-knee): hip hinge assessment
  - Torso lean (vertical angle of shoulder-hip line)
  - Knee valgus (knee x vs ankle x offset)

State machine: TOP (standing) -> GOING_DOWN -> BOTTOM (squat) -> GOING_UP -> TOP (1 rep)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .base_detector import BaseExerciseDetector
from ..pose_estimation.base import PoseResult
from ..utils.constants import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
)


# ── Thresholds ──────────────────────────────────────────────────────────

# Knee angle (hip->knee->ankle)
KNEE_STANDING = 160     # Standing upright
KNEE_PARALLEL = 95      # Thighs parallel to ground (good depth)
KNEE_HALF = 120         # Half squat (warning)

# Hip angle (shoulder->hip->knee)
HIP_STANDING = 160
HIP_GOOD_DEPTH = 90     # Hips at knee level
HIP_WARNING = 110       # Not deep enough

# Torso lean — vertical angle of shoulder-hip vector
TORSO_GOOD = 30         # degrees from vertical (slight forward lean OK)
TORSO_WARNING = 45      # too much forward lean
TORSO_BAD = 60          # excessive lean

# Knee valgus — knee should track over toes, not cave inward
VALGUS_THRESHOLD = 0.03  # normalized x-offset threshold


class SquatDetector(BaseExerciseDetector):
    EXERCISE_NAME = "squat"
    PRIMARY_ANGLE_TOP = KNEE_STANDING
    PRIMARY_ANGLE_BOTTOM = KNEE_PARALLEL
    DESCENT_THRESHOLD = 15.0
    ASCENT_THRESHOLD = 10.0

    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        knee = self._avg_angle(
            pose,
            (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
            (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
        )
        hip = self._avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
            (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
        )

        # Torso lean: vertical angle of shoulder-hip line
        torso = self._compute_torso_lean(pose)

        # Knee valgus: x-offset of knees relative to ankles
        valgus = self._compute_knee_valgus(pose)

        return {"primary": knee, "knee": knee, "hip": hip, "torso": torso, "valgus": valgus}

    def _compute_torso_lean(self, pose: PoseResult) -> Optional[float]:
        """Compute torso forward lean angle from vertical."""
        left_s = pose.get_world_landmark(LEFT_SHOULDER) if pose.is_visible(LEFT_SHOULDER) else None
        right_s = pose.get_world_landmark(RIGHT_SHOULDER) if pose.is_visible(RIGHT_SHOULDER) else None
        left_h = pose.get_world_landmark(LEFT_HIP) if pose.is_visible(LEFT_HIP) else None
        right_h = pose.get_world_landmark(RIGHT_HIP) if pose.is_visible(RIGHT_HIP) else None

        if left_s is None and right_s is None:
            return None
        if left_h is None and right_h is None:
            return None

        shoulder = left_s if right_s is None else (right_s if left_s is None else (np.array(left_s) + np.array(right_s)) / 2)
        hip = left_h if right_h is None else (right_h if left_h is None else (np.array(left_h) + np.array(right_h)) / 2)

        shoulder = np.array(shoulder)
        hip = np.array(hip)
        diff = shoulder - hip
        # Angle from vertical (y-axis)
        vertical = np.array([0, 1, 0])
        cos_angle = np.dot(diff, vertical) / (np.linalg.norm(diff) * np.linalg.norm(vertical) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))

    def _compute_knee_valgus(self, pose: PoseResult) -> Optional[float]:
        """Compute average knee valgus (inward collapse) as normalized x-offset."""
        offsets = []
        for knee_idx, ankle_idx in [(LEFT_KNEE, LEFT_ANKLE), (RIGHT_KNEE, RIGHT_ANKLE)]:
            if pose.is_visible(knee_idx) and pose.is_visible(ankle_idx):
                knee = pose.get_landmark(knee_idx)
                ankle = pose.get_landmark(ankle_idx)
                offsets.append(abs(knee[0] - ankle[0]))

        if not offsets:
            return None
        return sum(offsets) / len(offsets)

    def _check_start_position(self, angles: Dict[str, Optional[float]]) -> Tuple[bool, List[str]]:
        issues = []
        knee = angles.get("knee")
        if knee is None:
            issues.append("Stand facing the camera — knees must be visible")
            return False, issues
        if knee < KNEE_STANDING - 20:
            issues.append("Stand up straight to begin")
        if not issues:
            return True, []
        return False, issues

    def _get_missing_parts(self, angles: Dict[str, Optional[float]]) -> List[str]:
        missing = []
        if angles.get("knee") is None:
            missing.append("knees and hips")
        return missing if missing else ["body"]

    def _assess_form(self, angles: Dict[str, Optional[float]]) -> Tuple[float, Dict[str, str], List[str]]:
        joint_feedback = {}
        issues = []
        scores = []

        knee = angles.get("knee")
        hip = angles.get("hip")
        torso = angles.get("torso")
        valgus = angles.get("valgus")

        # ── Knee depth assessment ──
        if knee is not None:
            if self._state in ("bottom", "going_up", "going_down"):
                if knee <= KNEE_PARALLEL:
                    joint_feedback["left_knee"] = "correct"
                    joint_feedback["right_knee"] = "correct"
                    scores.append(1.0)
                elif knee <= KNEE_HALF:
                    joint_feedback["left_knee"] = "warning"
                    joint_feedback["right_knee"] = "warning"
                    issues.append("Go deeper — thighs should reach parallel")
                    scores.append(0.5)
                else:
                    joint_feedback["left_knee"] = "correct"
                    joint_feedback["right_knee"] = "correct"
                    scores.append(0.8)
            else:
                if knee >= KNEE_STANDING - 10:
                    joint_feedback["left_knee"] = "correct"
                    joint_feedback["right_knee"] = "correct"
                    scores.append(1.0)
                else:
                    joint_feedback["left_knee"] = "warning"
                    joint_feedback["right_knee"] = "warning"
                    scores.append(0.7)

        # ── Hip depth assessment (weighted 1.5x) ──
        if hip is not None:
            if self._state in ("bottom", "going_up"):
                if hip <= HIP_GOOD_DEPTH + 5:
                    joint_feedback["left_hip"] = "correct"
                    joint_feedback["right_hip"] = "correct"
                    scores.append(1.0)
                elif hip <= HIP_WARNING:
                    joint_feedback["left_hip"] = "warning"
                    joint_feedback["right_hip"] = "warning"
                    issues.append("Drop hips lower — aim for hip crease below knee")
                    scores.append(0.5)
                else:
                    joint_feedback["left_hip"] = "warning"
                    joint_feedback["right_hip"] = "warning"
                    scores.append(0.7)
            else:
                joint_feedback["left_hip"] = "correct"
                joint_feedback["right_hip"] = "correct"
                scores.append(1.0)

        # ── Torso lean ──
        if torso is not None:
            if torso <= TORSO_GOOD:
                joint_feedback["left_shoulder"] = "correct"
                joint_feedback["right_shoulder"] = "correct"
                scores.append(1.0)
            elif torso <= TORSO_WARNING:
                joint_feedback["left_shoulder"] = "warning"
                joint_feedback["right_shoulder"] = "warning"
                issues.append("Keep chest up — reduce forward lean")
                scores.append(0.5)
            else:
                joint_feedback["left_shoulder"] = "incorrect"
                joint_feedback["right_shoulder"] = "incorrect"
                issues.append("Excessive forward lean — keep torso upright")
                scores.append(0.2)

        # ── Knee valgus ──
        if valgus is not None and self._state in ("bottom", "going_down", "going_up"):
            if valgus > VALGUS_THRESHOLD * 2:
                issues.append("Knees caving in — push knees out over toes")
                scores.append(0.3)
            elif valgus > VALGUS_THRESHOLD:
                scores.append(0.7)

        if not scores:
            return 0.0, joint_feedback, issues

        # Weight hip depth higher
        if hip is not None and len(scores) >= 2:
            weighted = []
            for i, s in enumerate(scores):
                w = 1.5 if i == 1 else 1.0  # hip is index 1
                weighted.append(s * w)
            total = sum(weighted) / sum(1.5 if i == 1 else 1.0 for i in range(len(scores)))
        else:
            total = sum(scores) / len(scores)

        return max(0.0, min(1.0, total)), joint_feedback, issues
