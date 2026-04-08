"""
Dedicated bench press detector with state machine and form assessment.

Angles tracked:
  - Elbow (shoulder-elbow-wrist): primary rep counting + depth
  - Elbow symmetry: left vs right should match
  - Shoulder angle (elbow-shoulder-hip): arm flare control

State machine: TOP (arms extended) -> GOING_DOWN -> BOTTOM (bar at chest) -> GOING_UP -> TOP (1 rep)
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
ELBOW_BOTTOM = 80       # Bar at chest level
ELBOW_HALF = 110        # Partial rep

# Shoulder angle for arm flare
SHOULDER_GOOD_MIN = 40
SHOULDER_GOOD_MAX = 75
SHOULDER_WARNING_MAX = 85  # Too much flare
SHOULDER_BAD_MAX = 95

# Symmetry
SYMMETRY_GOOD = 15      # degrees difference allowed
SYMMETRY_WARNING = 25


class BenchPressDetector(BaseExerciseDetector):
    EXERCISE_NAME = "bench_press"
    PRIMARY_ANGLE_TOP = ELBOW_EXTENDED
    PRIMARY_ANGLE_BOTTOM = ELBOW_BOTTOM
    DESCENT_THRESHOLD = 10.0
    ASCENT_THRESHOLD = 10.0

    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        elbow = self._avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
            (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
            name="elbow",
        )

        # Individual elbows for symmetry
        left_elbow = self._angle_at(pose, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
        right_elbow = self._angle_at(pose, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        symmetry = abs(left_elbow - right_elbow) if (left_elbow and right_elbow) else None

        shoulder = self._avg_angle(
            pose,
            (LEFT_ELBOW, LEFT_SHOULDER, LEFT_HIP),
            (RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP),
            name="shoulder",
        )

        return {
            "primary": elbow, "elbow": elbow,
            "symmetry": symmetry, "shoulder": shoulder,
        }

    def _check_start_position(self, angles: Dict[str, Optional[float]]) -> Tuple[bool, List[str]]:
        issues = []
        elbow = angles.get("elbow")
        if elbow is None:
            return False, ["Lie on bench — arms must be visible above"]
        if elbow < ELBOW_EXTENDED - 25:
            issues.append("Extend arms fully — press bar to lockout")
        return (len(issues) == 0, issues)

    def _get_missing_parts(self, angles: Dict[str, Optional[float]]) -> List[str]:
        if angles.get("elbow") is None:
            return ["arms"]
        return ["body"]

    def _assess_form(self, angles: Dict[str, Optional[float]]) -> Tuple[float, Dict[str, str], List[str]]:
        joint_feedback = {}
        issues = []
        scores = {}

        elbow = angles.get("elbow")
        symmetry = angles.get("symmetry")
        shoulder = angles.get("shoulder")

        # ── Elbow depth ──
        if elbow is not None:
            if self._state in ("bottom", "going_up", "going_down"):
                if elbow <= ELBOW_BOTTOM + 10:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 1.0
                elif elbow <= ELBOW_HALF:
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    issues.append("Lower the bar more — touch or near chest")
                    scores["elbow"] = 0.5
                else:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 0.8
            else:
                if elbow >= ELBOW_EXTENDED - 10:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 1.0
                else:
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    issues.append("Lock out arms at the top")
                    scores["elbow"] = 0.6

        # ── Symmetry ──
        if symmetry is not None:
            if symmetry <= SYMMETRY_GOOD:
                scores["symmetry"] = 1.0
            elif symmetry <= SYMMETRY_WARNING:
                issues.append("Uneven press — one arm ahead of the other")
                scores["symmetry"] = 0.5
            else:
                issues.append("Arms very uneven — press both sides equally")
                scores["symmetry"] = 0.2

        # ── Shoulder flare (weighted 1.3x) ──
        if shoulder is not None:
            if SHOULDER_GOOD_MIN <= shoulder <= SHOULDER_GOOD_MAX:
                joint_feedback["left_shoulder"] = "correct"
                joint_feedback["right_shoulder"] = "correct"
                scores["shoulder"] = 1.0
            elif shoulder <= SHOULDER_WARNING_MAX:
                joint_feedback["left_shoulder"] = "warning"
                joint_feedback["right_shoulder"] = "warning"
                issues.append("Elbows flaring — tuck elbows ~45° to body")
                scores["shoulder"] = 0.5
            elif shoulder <= SHOULDER_BAD_MAX:
                joint_feedback["left_shoulder"] = "incorrect"
                joint_feedback["right_shoulder"] = "incorrect"
                issues.append("Elbows too wide — shoulder impingement risk!")
                scores["shoulder"] = 0.2
            else:
                joint_feedback["left_shoulder"] = "incorrect"
                joint_feedback["right_shoulder"] = "incorrect"
                scores["shoulder"] = 0.1

        if not scores:
            return 0.0, joint_feedback, issues

        WEIGHTS = {"elbow": 1.0, "symmetry": 1.0, "shoulder": 1.3}
        weighted_sum = sum(scores[k] * WEIGHTS.get(k, 1.0) for k in scores)
        weight_total = sum(WEIGHTS.get(k, 1.0) for k in scores)
        total = weighted_sum / weight_total

        return max(0.0, min(1.0, total)), joint_feedback, issues
