"""
Rule-based form classifier using biomechanical threshold ranges.

Uses graduated scoring: each feature gets a 0-1 quality score based on
distance from the ideal range, with configurable soft margins and weights.

Includes an activity gate with relaxed thresholds: tracks multiple joint
angles over a 1-second window and requires modest movement to activate.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

from .base import FormClassifier, ClassificationResult


# Rule format per feature: (min_good, max_good, margin, weight)
#   min_good / max_good: ideal range (score = 1.0 inside)
#   margin: distance outside range where score decays to 0.0
#   weight: importance of this feature in final score (default 1.0)
#
# Ranges cover the FULL expected ROM for correct form during any rep phase.
# The activity gate (not the thresholds) rejects static standing.

EXERCISE_RULES = {
    "squat": {
        "angle_left_knee":   (50, 175, 20, 1.0),
        "angle_right_knee":  (50, 175, 20, 1.0),
        "angle_left_hip":    (35, 175, 20, 1.0),
        "angle_right_hip":   (35, 175, 20, 1.0),
        "knee_symmetry":     (0.75, 1.0, 0.20, 0.7),
        "torso_lean":        (0, 55, 20, 0.8),
    },
    "lunge": {
        "angle_left_knee":   (50, 175, 20, 1.0),
        "angle_right_knee":  (50, 175, 20, 1.0),
        "angle_left_hip":    (35, 175, 20, 0.8),
        "angle_right_hip":   (35, 175, 20, 0.8),
        "torso_lean":        (0, 35, 20, 0.8),
    },
    "deadlift": {
        "angle_left_hip":    (35, 175, 20, 1.0),
        "angle_right_hip":   (35, 175, 20, 1.0),
        "angle_left_knee":   (50, 175, 20, 0.6),
        "angle_right_knee":  (50, 175, 20, 0.6),
        "spine_alignment":   (0, 40, 20, 1.0),
        "hip_symmetry":      (0.75, 1.0, 0.20, 0.7),
    },
    "bench_press": {
        "angle_left_elbow":  (30, 175, 20, 1.0),
        "angle_right_elbow": (30, 175, 20, 1.0),
        "elbow_symmetry":    (0.80, 1.0, 0.20, 0.7),
    },
    "overhead_press": {
        "angle_left_elbow":  (30, 180, 20, 1.0),
        "angle_right_elbow": (30, 180, 20, 1.0),
        "elbow_symmetry":    (0.75, 1.0, 0.20, 0.7),
        "torso_lean":        (0, 20, 15, 0.8),
        "lockout_angle":     (145, 180, 20, 0.6),
    },
    "pullup": {
        "angle_left_elbow":  (20, 175, 20, 1.0),
        "angle_right_elbow": (20, 175, 20, 1.0),
        "elbow_symmetry":    (0.80, 1.0, 0.20, 0.7),
        "body_swing":        (0, 0.10, 0.10, 0.8),
        "chin_above_bar":    (0.0, 1.0, 0.5, 0.5),
    },
    "pushup": {
        "angle_left_elbow":  (35, 170, 20, 1.0),
        "angle_right_elbow": (35, 170, 20, 1.0),
        "body_line":         (0, 20, 15, 0.9),
        "elbow_symmetry":    (0.75, 1.0, 0.20, 0.6),
        "hip_sag":           (-0.10, 0.10, 0.10, 0.8),
    },
    "plank": {
        "body_line":         (0, 15, 15, 1.0),
        "hip_sag":           (-0.08, 0.08, 0.10, 1.0),
    },
    "bicep_curl": {
        "angle_left_elbow":  (15, 170, 20, 1.0),
        "angle_right_elbow": (15, 170, 20, 1.0),
        "upper_arm_movement": (0, 40, 20, 0.7),
        "elbow_symmetry":    (0.65, 1.0, 0.25, 0.4),
        "torso_swing":       (0, 20, 15, 0.6),
    },
    "tricep_dip": {
        "angle_left_elbow":  (40, 170, 20, 1.0),
        "angle_right_elbow": (40, 170, 20, 1.0),
        "elbow_symmetry":    (0.80, 1.0, 0.20, 0.7),
    },
}

# Primary angles per exercise -- used for activity detection.
# Both left AND right are checked; max range is used.
PRIMARY_ANGLES = {
    "squat":          ("angle_left_knee", "angle_right_knee"),
    "lunge":          ("angle_left_knee", "angle_right_knee"),
    "deadlift":       ("angle_left_hip", "angle_right_hip"),
    "bench_press":    ("angle_left_elbow", "angle_right_elbow"),
    "overhead_press": ("angle_left_elbow", "angle_right_elbow"),
    "pullup":         ("angle_left_elbow", "angle_right_elbow"),
    "pushup":         ("angle_left_elbow", "angle_right_elbow"),
    "plank":          None,  # Static hold -- always active
    "bicep_curl":     ("angle_left_elbow", "angle_right_elbow"),
    "tricep_dip":     ("angle_left_elbow", "angle_right_elbow"),
}

# Map feature names to related joints (for color-coded feedback)
FEATURE_TO_JOINTS = {
    "angle_left_knee": ["left_knee"],
    "angle_right_knee": ["right_knee"],
    "angle_left_hip": ["left_hip"],
    "angle_right_hip": ["right_hip"],
    "angle_left_elbow": ["left_elbow"],
    "angle_right_elbow": ["right_elbow"],
    "angle_left_shoulder": ["left_shoulder"],
    "angle_right_shoulder": ["right_shoulder"],
    "angle_left_ankle": ["left_ankle"],
    "angle_right_ankle": ["right_ankle"],
    "knee_symmetry": ["left_knee", "right_knee"],
    "hip_symmetry": ["left_hip", "right_hip"],
    "elbow_symmetry": ["left_elbow", "right_elbow"],
    "shoulder_symmetry": ["left_shoulder", "right_shoulder"],
    "torso_lean": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "torso_swing": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "spine_alignment": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "body_line": ["left_hip", "right_hip"],
    "hip_sag": ["left_hip", "right_hip"],
    "hip_below_knee": ["left_hip", "right_hip"],
    "upper_arm_movement": ["left_shoulder", "right_shoulder"],
    "body_swing": ["left_hip", "right_hip"],
    "lockout_angle": ["left_elbow", "right_elbow"],
    "chin_above_bar": ["left_elbow", "right_elbow"],
}


def _feature_score(value: float, min_good: float, max_good: float, margin: float) -> float:
    """Score a feature value with soft margins.

    Returns 1.0 inside [min_good, max_good], decaying linearly to 0.0
    at distance `margin` outside the range.
    """
    if min_good <= value <= max_good:
        return 1.0
    if value < min_good:
        dist = min_good - value
    else:
        dist = value - max_good
    return max(0.0, 1.0 - dist / max(margin, 1e-6))


class RuleBasedClassifier(FormClassifier):
    """Classifies exercise form using biomechanical thresholds with graduated scoring.

    Activity gate: tracks primary joint angles over a 1-second window and
    requires modest range of motion. Uses both left and right sides.
    """

    MOTION_WINDOW = 30          # frames (~1s at 30fps)
    MOTION_MIN_RANGE = 15.0     # degrees — raised to reduce false active states

    def __init__(self):
        # Separate histories for left and right primary angles
        self._angle_history_a: deque = deque(maxlen=self.MOTION_WINDOW)
        self._angle_history_b: deque = deque(maxlen=self.MOTION_WINDOW)
        self._last_exercise: Optional[str] = None

    def _is_active(self, exercise: str, features: Dict[str, Optional[float]]) -> bool:
        """Check whether the user is actively exercising using multi-frame motion."""
        primary = PRIMARY_ANGLES.get(exercise)

        # Plank is a static hold — always active if pose is detected
        if primary is None:
            return True

        angle_key_a, angle_key_b = primary
        val_a = features.get(angle_key_a)
        val_b = features.get(angle_key_b)

        # Reset history when exercise changes
        if exercise != self._last_exercise:
            self._angle_history_a.clear()
            self._angle_history_b.clear()
            self._last_exercise = exercise

        if val_a is not None:
            self._angle_history_a.append(val_a)
        if val_b is not None:
            self._angle_history_b.append(val_b)

        # Need a few frames before judging
        if len(self._angle_history_a) < 3 and len(self._angle_history_b) < 3:
            return False

        # Use the MAXIMUM range across both sides
        range_a = (max(self._angle_history_a) - min(self._angle_history_a)
                   if self._angle_history_a else 0.0)
        range_b = (max(self._angle_history_b) - min(self._angle_history_b)
                   if self._angle_history_b else 0.0)

        return max(range_a, range_b) >= self.MOTION_MIN_RANGE

    def classify(self, features: Dict[str, Optional[float]], exercise: str) -> ClassificationResult:
        rules = EXERCISE_RULES.get(exercise, {})
        joint_feedback = {}
        details = {}
        weighted_scores = []

        # Motion-based activity gate
        if not self._is_active(exercise, features):
            return ClassificationResult(
                exercise=exercise,
                is_correct=False,
                confidence=0.0,
                joint_feedback={},
                details={"activity": "Not exercising \u2014 begin movement to get feedback"},
                is_active=False,
                form_score=0.0,
            )

        for feature_name, (min_val, max_val, margin, weight) in rules.items():
            value = features.get(feature_name)
            if value is None:
                continue

            score = _feature_score(value, min_val, max_val, margin)
            weighted_scores.append((score, weight))
            related_joints = FEATURE_TO_JOINTS.get(feature_name, [])

            if score >= 0.8:
                status = "correct"
            elif score >= 0.4:
                status = "warning"
                details[feature_name] = (
                    f"{feature_name}: {value:.1f} (ideal {min_val}-{max_val})"
                )
            else:
                status = "incorrect"
                details[feature_name] = (
                    f"{feature_name}: {value:.1f} (ideal {min_val}-{max_val})"
                )

            for j in related_joints:
                existing = joint_feedback.get(j)
                # Only overwrite if new status is worse
                if existing is None or _status_priority(status) > _status_priority(existing):
                    joint_feedback[j] = status

        # Weighted average of per-feature scores
        if weighted_scores:
            total_weight = sum(w for _, w in weighted_scores)
            form_score = sum(s * w for s, w in weighted_scores) / max(total_weight, 1e-6)
        else:
            form_score = 0.0

        form_score = max(0.0, min(1.0, form_score))
        is_correct = form_score >= 0.7 and len(weighted_scores) > 0

        return ClassificationResult(
            exercise=exercise,
            is_correct=is_correct,
            confidence=form_score,
            joint_feedback=joint_feedback,
            details=details,
            is_active=True,
            form_score=form_score,
        )

    def get_supported_exercises(self) -> List[str]:
        return list(EXERCISE_RULES.keys())


def _status_priority(status: str) -> int:
    """Higher number = worse status (for overwrite logic)."""
    return {"correct": 0, "warning": 1, "incorrect": 2}.get(status, 0)
