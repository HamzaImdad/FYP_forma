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
    LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
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

# Knee valgus — normalized ratio of x-offset to femur length
VALGUS_WARN_RATIO = 0.06   # 6% of leg length = warning
VALGUS_BAD_RATIO = 0.10    # 10% = significant valgus

# Heel rise detection
HEEL_RISE_THRESHOLD = 0.02  # meters


class SquatDetector(BaseExerciseDetector):
    EXERCISE_NAME = "squat"
    PRIMARY_ANGLE_TOP = KNEE_STANDING
    PRIMARY_ANGLE_BOTTOM = KNEE_PARALLEL
    DESCENT_THRESHOLD = 15.0
    ASCENT_THRESHOLD = 10.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mid_descent_hip_angle = None

    def reset(self):
        super().reset()
        self._mid_descent_hip_angle = None

    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        knee = self._avg_angle(
            pose,
            (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
            (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
            name="knee",
        )
        hip = self._avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
            (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
            name="hip",
        )

        # Torso lean: vertical angle of shoulder-hip line
        torso = self._compute_torso_lean(pose)

        # Knee valgus: normalized ratio of x-offset to femur length
        valgus = self._compute_knee_valgus(pose)

        # Heel rise detection
        heel_rise = self._compute_heel_rise(pose)

        # World-coord depth check
        depth_world = self._compute_depth_world(pose)

        # Track hip angle at mid-descent for butt wink detection
        if knee is not None and self._state == "going_down" and 115 <= knee <= 125:
            hip_val = self._avg_angle(
                pose,
                (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
                (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
                name="hip",
            )
            if hip_val is not None:
                self._mid_descent_hip_angle = hip_val

        return {
            "primary": knee, "knee": knee, "hip": hip, "torso": torso,
            "valgus": valgus, "heel_rise": heel_rise, "depth_world": depth_world,
        }

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
        """Compute knee valgus as ratio of x-offset to femur length (normalized)."""
        ratios = []
        for hip_idx, knee_idx, ankle_idx in [(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
                                              (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)]:
            if all(pose.is_visible(i) for i in (hip_idx, knee_idx, ankle_idx)):
                hip = np.array(pose.get_world_landmark(hip_idx))
                knee = np.array(pose.get_world_landmark(knee_idx))
                ankle = np.array(pose.get_world_landmark(ankle_idx))
                x_offset = abs(knee[0] - ankle[0])
                leg_length = np.linalg.norm(hip - knee)
                if leg_length > 0.01:
                    ratios.append(x_offset / leg_length)
        return sum(ratios) / len(ratios) if ratios else None

    def _compute_heel_rise(self, pose: PoseResult) -> Optional[float]:
        """Detect if heels are lifting off ground. Returns avg rise in meters."""
        rises = []
        for heel_idx, foot_idx in [(LEFT_HEEL, LEFT_FOOT_INDEX), (RIGHT_HEEL, RIGHT_FOOT_INDEX)]:
            if pose.is_visible(heel_idx) and pose.is_visible(foot_idx):
                heel_y = pose.get_world_landmark(heel_idx)[1]
                foot_y = pose.get_world_landmark(foot_idx)[1]
                rises.append(heel_y - foot_y)  # positive = heel higher than toes
        return sum(rises) / len(rises) if rises else None

    def _compute_depth_world(self, pose: PoseResult) -> Optional[float]:
        """Check if hip is below knee using world Y coordinates.
        Returns difference: positive = hip below knee (good depth)."""
        hip_ys = []
        knee_ys = []
        for h_idx, k_idx in [(LEFT_HIP, LEFT_KNEE), (RIGHT_HIP, RIGHT_KNEE)]:
            if pose.is_visible(h_idx) and pose.is_visible(k_idx):
                hip_ys.append(pose.get_world_landmark(h_idx)[1])
                knee_ys.append(pose.get_world_landmark(k_idx)[1])
        if not hip_ys:
            return None
        # In MediaPipe world coords, Y increases downward, so hip_y > knee_y means hip is lower
        return (sum(hip_ys) / len(hip_ys)) - (sum(knee_ys) / len(knee_ys))

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
        scores = {}

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
                    scores["knee"] = 1.0
                elif knee <= KNEE_HALF:
                    joint_feedback["left_knee"] = "warning"
                    joint_feedback["right_knee"] = "warning"
                    issues.append("Go deeper — thighs should reach parallel")
                    scores["knee"] = 0.5
                else:
                    joint_feedback["left_knee"] = "correct"
                    joint_feedback["right_knee"] = "correct"
                    scores["knee"] = 0.8
            else:
                if knee >= KNEE_STANDING - 10:
                    joint_feedback["left_knee"] = "correct"
                    joint_feedback["right_knee"] = "correct"
                    scores["knee"] = 1.0
                else:
                    joint_feedback["left_knee"] = "warning"
                    joint_feedback["right_knee"] = "warning"
                    scores["knee"] = 0.7

        # ── Hip depth assessment (weighted 1.5x) ──
        if hip is not None:
            if self._state in ("bottom", "going_up"):
                if hip <= HIP_GOOD_DEPTH + 5:
                    joint_feedback["left_hip"] = "correct"
                    joint_feedback["right_hip"] = "correct"
                    scores["hip"] = 1.0
                elif hip <= HIP_WARNING:
                    joint_feedback["left_hip"] = "warning"
                    joint_feedback["right_hip"] = "warning"
                    issues.append("Drop hips lower — aim for hip crease below knee")
                    scores["hip"] = 0.5
                else:
                    joint_feedback["left_hip"] = "warning"
                    joint_feedback["right_hip"] = "warning"
                    scores["hip"] = 0.7
            else:
                joint_feedback["left_hip"] = "correct"
                joint_feedback["right_hip"] = "correct"
                scores["hip"] = 1.0

        # ── Torso lean ──
        if torso is not None:
            if torso <= TORSO_GOOD:
                joint_feedback["left_shoulder"] = "correct"
                joint_feedback["right_shoulder"] = "correct"
                scores["torso"] = 1.0
            elif torso <= TORSO_WARNING:
                joint_feedback["left_shoulder"] = "warning"
                joint_feedback["right_shoulder"] = "warning"
                issues.append("Keep chest up — reduce forward lean")
                scores["torso"] = 0.5
            else:
                joint_feedback["left_shoulder"] = "incorrect"
                joint_feedback["right_shoulder"] = "incorrect"
                issues.append("Excessive forward lean — keep torso upright")
                scores["torso"] = 0.2

        # ── Knee valgus (normalized ratio) ──
        if valgus is not None and self._state in ("bottom", "going_down", "going_up"):
            if valgus > VALGUS_BAD_RATIO:
                issues.append("Knees caving in -- push knees out over toes")
                scores["valgus"] = 0.2
            elif valgus > VALGUS_WARN_RATIO:
                issues.append("Slight knee cave -- focus on pushing knees outward")
                scores["valgus"] = 0.6

        # ── Heel rise ──
        heel_rise = angles.get("heel_rise")
        if heel_rise is not None and heel_rise > HEEL_RISE_THRESHOLD:
            issues.append("Heels lifting -- keep weight on heels")
            scores["heel_rise"] = 0.4
            joint_feedback["left_ankle"] = "warning"
            joint_feedback["right_ankle"] = "warning"

        # ── Butt wink (pelvis tuck at bottom) ──
        if (self._state == "bottom" and self._mid_descent_hip_angle is not None
                and hip is not None):
            hip_drop = self._mid_descent_hip_angle - hip
            if hip_drop > 20:
                issues.append("Butt wink -- pelvis tucking under at bottom")
                scores["butt_wink"] = 0.3
            elif hip_drop > 15:
                issues.append("Slight butt wink -- try stopping just above this depth")
                scores["butt_wink"] = 0.6

        if not scores:
            return 0.0, joint_feedback, issues

        WEIGHTS = {"knee": 1.0, "hip": 1.5, "torso": 1.0, "valgus": 1.0, "heel_rise": 0.8, "butt_wink": 0.8}
        weighted_sum = sum(scores[k] * WEIGHTS.get(k, 1.0) for k in scores)
        weight_total = sum(WEIGHTS.get(k, 1.0) for k in scores)
        total = weighted_sum / weight_total

        return max(0.0, min(1.0, total)), joint_feedback, issues
