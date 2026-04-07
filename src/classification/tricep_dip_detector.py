"""
Dedicated tricep dip detector with state machine and form assessment.

Angles tracked:
  - Elbow (shoulder-elbow-wrist): primary rep counting + depth
  - Shoulder (elbow-shoulder-hip): forward lean / shoulder depression
  - Hip position: should stay close to bench

State machine: TOP (arms extended) -> GOING_DOWN -> BOTTOM (dipped) -> GOING_UP -> TOP (1 rep)
"""

from typing import Dict, List, Optional, Tuple

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
        )
        shoulder = self._avg_angle(
            pose,
            (LEFT_ELBOW, LEFT_SHOULDER, LEFT_HIP),
            (RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP),
        )
        return {"primary": elbow, "elbow": elbow, "shoulder": shoulder}

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
        scores = []

        elbow = angles.get("elbow")
        shoulder = angles.get("shoulder")

        # ── Elbow depth ──
        if elbow is not None:
            if self._state in ("bottom", "going_up", "going_down"):
                if elbow <= ELBOW_DIPPED + 5:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores.append(1.0)
                elif elbow <= ELBOW_HALF:
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    issues.append("Go deeper — bend elbows to 90°")
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
                    issues.append("Extend arms fully at top")
                    scores.append(0.6)

        # ── Shoulder / forward lean (weighted 1.3x) ──
        if shoulder is not None:
            if SHOULDER_GOOD_MIN <= shoulder <= SHOULDER_GOOD_MAX:
                joint_feedback["left_shoulder"] = "correct"
                joint_feedback["right_shoulder"] = "correct"
                scores.append(1.0)
            elif shoulder <= SHOULDER_WARNING_MAX:
                joint_feedback["left_shoulder"] = "warning"
                joint_feedback["right_shoulder"] = "warning"
                issues.append("Leaning forward — keep body close to bench")
                scores.append(0.5)
            elif shoulder <= SHOULDER_BAD_MAX:
                joint_feedback["left_shoulder"] = "incorrect"
                joint_feedback["right_shoulder"] = "incorrect"
                issues.append("Too much forward lean — shoulder impingement risk")
                scores.append(0.2)
            else:
                joint_feedback["left_shoulder"] = "incorrect"
                joint_feedback["right_shoulder"] = "incorrect"
                issues.append("Excessive lean — stay upright, push straight down")
                scores.append(0.1)

        if not scores:
            return 0.0, joint_feedback, issues

        if len(scores) >= 2:
            weights = [1.0, 1.3]
            weighted = [s * w for s, w in zip(scores, weights)]
            total = sum(weighted) / sum(weights)
        else:
            total = sum(scores) / len(scores)

        return max(0.0, min(1.0, total)), joint_feedback, issues
