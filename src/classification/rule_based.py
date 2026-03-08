"""
Rule-based form classifier using biomechanical threshold ranges.

Includes an activity gate: the user must be actively performing the exercise
(sufficient range of motion in the primary joint) for the form to be
evaluated. Standing still will not pass as "correct".
"""

from typing import Dict, List, Optional

from .base import FormClassifier, ClassificationResult


# Threshold ranges: (min_good, max_good) for each feature
# Values outside this range indicate incorrect form.
# Ranges are tightened to reject standing/resting poses.
EXERCISE_RULES = {
    "squat": {
        "angle_left_knee":   (60, 150),    # Must bend knees (standing ~170)
        "angle_right_knee":  (60, 150),
        "angle_left_hip":    (40, 130),    # Must hinge hips
        "angle_right_hip":   (40, 130),
        "knee_symmetry":     (0.80, 1.0),
        "torso_lean":        (0, 50),
        "hip_below_knee":    (0.1, 1.0),   # Hip should drop near/below knee level
    },
    "lunge": {
        "angle_left_knee":   (60, 140),
        "angle_right_knee":  (60, 140),
        "torso_lean":        (0, 30),
    },
    "deadlift": {
        "angle_left_hip":    (40, 160),    # Must hinge (standing ~170)
        "angle_right_hip":   (40, 160),
        "spine_alignment":   (0, 30),      # Allow moderate torso angle
        "hip_symmetry":      (0.80, 1.0),
    },
    "bench_press": {
        "angle_left_elbow":  (40, 160),    # Must bend elbows
        "angle_right_elbow": (40, 160),
        "elbow_symmetry":    (0.85, 1.0),
    },
    "overhead_press": {
        "angle_left_elbow":  (40, 175),
        "angle_right_elbow": (40, 175),
        "elbow_symmetry":    (0.80, 1.0),
        "torso_lean":        (0, 20),      # Allow slight lean
        "lockout_angle":     (150, 180),   # Extension at top
    },
    "pullup": {
        "angle_left_elbow":  (30, 170),
        "angle_right_elbow": (30, 170),
        "elbow_symmetry":    (0.85, 1.0),
        "body_swing":        (0, 0.08),    # Minimal kipping
        "chin_above_bar":    (0.5, 1.0),   # Nose should be near/above wrist level
    },
    "pushup": {
        "angle_left_elbow":  (40, 160),
        "angle_right_elbow": (40, 160),
        "body_line":         (0, 15),      # Moderate body alignment
        "elbow_symmetry":    (0.80, 1.0),
        "hip_sag":           (-0.08, 0.08),
    },
    "plank": {
        "body_line":         (0, 15),      # Allow moderate deviation
        "hip_sag":           (-0.06, 0.06),
    },
    "bicep_curl": {
        "angle_left_elbow":  (20, 160),
        "angle_right_elbow": (20, 160),
        "upper_arm_movement": (0, 30),     # Stricter: less shoulder swing
        "elbow_symmetry":    (0.80, 1.0),
        "torso_swing":       (0, 15),      # Minimal body swing
    },
    "tricep_dip": {
        "angle_left_elbow":  (50, 160),
        "angle_right_elbow": (50, 160),
        "elbow_symmetry":    (0.85, 1.0),
    },
}

# Primary angles per exercise -- used for activity detection
# If the primary angle hasn't moved enough, the user isn't exercising
PRIMARY_ANGLES = {
    "squat":          "angle_left_knee",
    "lunge":          "angle_left_knee",
    "deadlift":       "angle_left_hip",
    "bench_press":    "angle_left_elbow",
    "overhead_press": "angle_left_elbow",
    "pullup":         "angle_left_elbow",
    "pushup":         "angle_left_elbow",
    "plank":          None,  # Static hold -- no movement required
    "bicep_curl":     "angle_left_elbow",
    "tricep_dip":     "angle_left_elbow",
}

# Minimum angle deviation from standing/resting to count as "active"
# Standing knee ~170, standing elbow ~170, standing hip ~170
RESTING_ANGLES = {
    "angle_left_knee":  160,
    "angle_right_knee": 160,
    "angle_left_hip":   160,
    "angle_right_hip":  160,
    "angle_left_elbow": 160,
    "angle_right_elbow": 160,
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


class RuleBasedClassifier(FormClassifier):
    """Classifies exercise form using hand-tuned threshold ranges.

    Includes an activity gate: if the primary joint angle is near its
    resting position (i.e., the user is standing still), the classifier
    returns "not exercising" instead of "correct".
    """

    def classify(self, features: Dict[str, Optional[float]], exercise: str) -> ClassificationResult:
        rules = EXERCISE_RULES.get(exercise, {})
        joint_feedback = {}
        details = {}
        violations = 0
        total_checked = 0

        # Activity gate: check if user is actually performing the exercise
        primary_angle = PRIMARY_ANGLES.get(exercise)
        if primary_angle is not None:
            angle_val = features.get(primary_angle)
            resting_threshold = RESTING_ANGLES.get(primary_angle, 160)
            if angle_val is not None and angle_val > resting_threshold:
                # User is standing/resting -- not actually exercising
                return ClassificationResult(
                    exercise=exercise,
                    is_correct=False,
                    confidence=0.0,
                    joint_feedback={},
                    details={"activity": "Begin exercise movement to receive feedback"},
                )

        for feature_name, (min_val, max_val) in rules.items():
            value = features.get(feature_name)
            if value is None:
                continue

            total_checked += 1
            related_joints = FEATURE_TO_JOINTS.get(feature_name, [])

            if min_val <= value <= max_val:
                for j in related_joints:
                    joint_feedback.setdefault(j, "correct")
            else:
                violations += 1
                for j in related_joints:
                    joint_feedback[j] = "incorrect"
                details[feature_name] = (
                    f"{feature_name}: {value:.1f} (expected {min_val}-{max_val})"
                )

        is_correct = violations == 0 and total_checked > 0
        confidence = 1.0 - (violations / max(total_checked, 1)) if total_checked > 0 else 0.0

        return ClassificationResult(
            exercise=exercise,
            is_correct=is_correct,
            confidence=max(0.0, confidence),
            joint_feedback=joint_feedback,
            details=details,
        )

    def get_supported_exercises(self) -> List[str]:
        return list(EXERCISE_RULES.keys())
