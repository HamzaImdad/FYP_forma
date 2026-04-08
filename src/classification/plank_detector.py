"""
Dedicated plank detector — static hold, no rep counting.

Angles tracked:
  - Body line (shoulder-hip-ankle): should be ~180 degrees (straight line)
  - Hip sag/pike (hip position relative to shoulder-ankle line)
  - Shoulder position (should be over wrists/elbows)

Tracks hold duration instead of reps.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .base_detector import BaseExerciseDetector
from ..pose_estimation.base import PoseResult
from ..utils.constants import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
)


# ── Thresholds ──
BODY_LINE_GOOD = 170    # Nearly straight
BODY_LINE_WARNING = 160 # Slight sag/pike
BODY_LINE_BAD = 150     # Significant form breakdown

# Hip position relative to shoulder-ankle line (world coords, meters)
HIP_SAG_THRESHOLD = 0.04    # meters — hips sagging below line
HIP_PIKE_THRESHOLD = 0.04   # meters — hips piking above line

# Shoulder over wrist alignment (world coords, meters)
SHOULDER_WRIST_GOOD = 0.08   # meters x-offset tolerance


class PlankDetector(BaseExerciseDetector):
    EXERCISE_NAME = "plank"
    IS_STATIC = True
    PRIMARY_ANGLE_TOP = BODY_LINE_GOOD
    PRIMARY_ANGLE_BOTTOM = BODY_LINE_BAD

    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        body_line = self._avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_HIP, LEFT_ANKLE),
            (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_ANKLE),
            name="hip",
        )

        hip_deviation = self._compute_hip_deviation(pose)
        shoulder_alignment = self._compute_shoulder_alignment(pose)

        return {
            "primary": body_line,
            "body_line": body_line,
            "hip_deviation": hip_deviation,
            "shoulder_alignment": shoulder_alignment,
        }

    def _compute_hip_deviation(self, pose: PoseResult) -> Optional[float]:
        """Compute how much hips deviate from straight shoulder-ankle line.
        Positive = sag (hips below line), Negative = pike (hips above line).
        Uses world coordinates for camera-angle independence.
        """
        deviations = []
        for s_idx, h_idx, a_idx in [(LEFT_SHOULDER, LEFT_HIP, LEFT_ANKLE),
                                     (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_ANKLE)]:
            if all(pose.is_visible(i) for i in (s_idx, h_idx, a_idx)):
                s = np.array(pose.get_world_landmark(s_idx))
                h = np.array(pose.get_world_landmark(h_idx))
                a = np.array(pose.get_world_landmark(a_idx))
                # Project hip onto shoulder-ankle line, measure perpendicular distance
                sa = a - s
                sh = h - s
                sa_len = np.linalg.norm(sa)
                if sa_len < 1e-6:
                    continue
                # Parameter t: where hip projects onto the shoulder-ankle line
                t = np.dot(sh, sa) / (sa_len ** 2)
                t = np.clip(t, 0, 1)
                projected = s + t * sa
                # Perpendicular vector from line to hip
                perp = h - projected
                # Sign: positive Y in world = down, so positive perp.y = hip sagging
                deviations.append(float(perp[1]))

        return sum(deviations) / len(deviations) if deviations else None

    def _compute_shoulder_alignment(self, pose: PoseResult) -> Optional[float]:
        """Check if shoulders are over wrists/elbows using world coordinates."""
        offsets = []
        for s_idx, w_idx in [(LEFT_SHOULDER, LEFT_WRIST), (RIGHT_SHOULDER, RIGHT_WRIST)]:
            if pose.is_visible(s_idx) and pose.is_visible(w_idx):
                s = np.array(pose.get_world_landmark(s_idx))
                w = np.array(pose.get_world_landmark(w_idx))
                # Horizontal offset (x-axis in world coords)
                offsets.append(float(s[0] - w[0]))

        return sum(offsets) / len(offsets) if offsets else None

    def _check_start_position(self, angles: Dict[str, Optional[float]]) -> Tuple[bool, List[str]]:
        issues = []
        body_line = angles.get("body_line")
        if body_line is None:
            return False, ["Get into plank position — full body must be visible from the side"]
        if body_line < BODY_LINE_BAD:
            issues.append("Straighten your body — align shoulders, hips, and ankles")
        if not issues:
            return True, []
        return False, issues

    def _get_missing_parts(self, angles: Dict[str, Optional[float]]) -> List[str]:
        if angles.get("body_line") is None:
            return ["shoulders, hips, or ankles"]
        return ["body"]

    def _assess_form(self, angles: Dict[str, Optional[float]]) -> Tuple[float, Dict[str, str], List[str]]:
        joint_feedback = {}
        issues = []
        scores = {}

        body_line = angles.get("body_line")
        hip_dev = angles.get("hip_deviation")
        shoulder_align = angles.get("shoulder_alignment")

        # ── Body line straightness (most important — weighted 1.5x) ──
        if body_line is not None:
            if body_line >= BODY_LINE_GOOD:
                joint_feedback["left_hip"] = "correct"
                joint_feedback["right_hip"] = "correct"
                joint_feedback["left_shoulder"] = "correct"
                joint_feedback["right_shoulder"] = "correct"
                scores["body_line"] = 1.0
            elif body_line >= BODY_LINE_WARNING:
                joint_feedback["left_hip"] = "warning"
                joint_feedback["right_hip"] = "warning"
                scores["body_line"] = 0.6
            elif body_line >= BODY_LINE_BAD:
                joint_feedback["left_hip"] = "incorrect"
                joint_feedback["right_hip"] = "incorrect"
                scores["body_line"] = 0.3
            else:
                joint_feedback["left_hip"] = "incorrect"
                joint_feedback["right_hip"] = "incorrect"
                scores["body_line"] = 0.1

        # ── Hip sag/pike detection ──
        if hip_dev is not None:
            if hip_dev > HIP_SAG_THRESHOLD:
                issues.append("Hips sagging — tighten core and glutes, lift hips")
                joint_feedback["left_hip"] = "incorrect"
                joint_feedback["right_hip"] = "incorrect"
                scores["hip_dev"] = 0.3
            elif hip_dev < -HIP_PIKE_THRESHOLD:
                issues.append("Hips too high — lower them in line with shoulders and ankles")
                joint_feedback["left_hip"] = "warning"
                joint_feedback["right_hip"] = "warning"
                scores["hip_dev"] = 0.5
            else:
                scores["hip_dev"] = 1.0

        # ── Shoulder alignment ──
        if shoulder_align is not None:
            if abs(shoulder_align) <= SHOULDER_WRIST_GOOD:
                scores["shoulder"] = 1.0
            elif abs(shoulder_align) <= SHOULDER_WRIST_GOOD * 2:
                issues.append("Shift shoulders over wrists")
                scores["shoulder"] = 0.6
            else:
                issues.append("Shoulders too far from wrists — reposition")
                scores["shoulder"] = 0.3

        if not scores:
            return 0.0, joint_feedback, issues

        WEIGHTS = {"body_line": 1.5, "hip_dev": 1.2, "shoulder": 1.0}
        weighted_sum = sum(scores[k] * WEIGHTS.get(k, 1.0) for k in scores)
        weight_total = sum(WEIGHTS.get(k, 1.0) for k in scores)
        total = weighted_sum / weight_total

        return max(0.0, min(1.0, total)), joint_feedback, issues
