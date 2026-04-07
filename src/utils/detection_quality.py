"""
Detection quality assessment for pose landmarks.

Implements ghost detection, body completeness validation, and a state machine
(GOOD/DEGRADED/POOR/LOST) that prevents classification on garbage landmarks.

Per-exercise landmark requirements ensure the exercise-relevant joints are
visible before running the BiLSTM classifier.
"""

from enum import Enum, auto
from typing import Optional, Tuple

from ..pose_estimation.base import PoseResult


class DetectionQuality(Enum):
    GOOD = auto()      # All checks pass — normal operation
    DEGRADED = auto()  # Some landmarks missing — use rule-based fallback
    POOR = auto()      # Too many missing — pause classification
    LOST = auto()      # No pose detected


# Per-exercise required landmarks (must have visibility >= threshold)
EXERCISE_REQUIRED_LANDMARKS = {
    "squat":          [23, 24, 25, 26, 27, 28],                      # hips, knees, ankles
    "lunge":          [23, 24, 25, 26, 27, 28],
    "deadlift":       [23, 24, 25, 26, 11, 12],                      # hips, knees, shoulders
    "bench_press":    [11, 12, 13, 14, 15, 16],                      # shoulders, elbows, wrists
    "overhead_press": [11, 12, 13, 14, 15, 16],
    "pullup":         [11, 12, 13, 14, 15, 16],
    "pushup":         [11, 12, 13, 14, 15, 16, 23, 24, 27, 28],     # upper + hips + ankles
    "plank":          [11, 12, 13, 14, 15, 16, 23, 24, 27, 28],
    "bicep_curl":     [11, 12, 13, 14, 15, 16],
    "tricep_dip":     [11, 12, 13, 14, 15, 16],
}

# Minimum for any exercise: both shoulders + both hips
UNIVERSAL_MIN_LANDMARKS = [11, 12, 23, 24]


class DetectionQualityChecker:
    """Assesses pose detection quality per frame with hysteresis.

    Requires 5 consecutive bad frames before downgrading state to prevent
    flickering between states during momentary occlusions.
    """

    def __init__(self, visibility_threshold: float = 0.65, completeness_ratio: float = 0.8):
        self.vis_threshold = visibility_threshold
        self.completeness_ratio = completeness_ratio
        self._bad_frame_count = 0
        self._state = DetectionQuality.LOST

    def check(self, pose: Optional[PoseResult], exercise: str) -> Tuple[DetectionQuality, str]:
        """Assess detection quality and return (quality_level, user_message).

        Args:
            pose: PoseResult or None if no detection.
            exercise: Current exercise name for landmark requirements.

        Returns:
            (DetectionQuality enum, guidance message for the user).
        """
        if pose is None:
            self._bad_frame_count += 1
            if self._bad_frame_count >= 5:
                self._state = DetectionQuality.LOST
            return self._state, "Step into camera view"

        vis = pose.landmarks[:, 3]  # (33,) visibility scores

        # Check universal minimum (shoulders + hips)
        universal_visible = sum(1 for i in UNIVERSAL_MIN_LANDMARKS if vis[i] >= self.vis_threshold)
        if universal_visible < len(UNIVERSAL_MIN_LANDMARKS):
            self._bad_frame_count += 1
            if self._bad_frame_count >= 5:
                self._state = DetectionQuality.POOR
            return self._state, "Face the camera \u2014 shoulders and hips must be visible"

        # Check exercise-specific completeness
        required = EXERCISE_REQUIRED_LANDMARKS.get(exercise, UNIVERSAL_MIN_LANDMARKS)
        visible_count = sum(1 for i in required if vis[i] >= self.vis_threshold)
        ratio = visible_count / len(required)

        if ratio >= self.completeness_ratio:
            self._bad_frame_count = 0
            self._state = DetectionQuality.GOOD
            return DetectionQuality.GOOD, ""
        elif ratio >= 0.5:
            self._bad_frame_count = 0
            self._state = DetectionQuality.DEGRADED
            return DetectionQuality.DEGRADED, "Some joints not visible \u2014 move closer"
        else:
            self._bad_frame_count += 1
            if self._bad_frame_count >= 5:
                self._state = DetectionQuality.POOR
            return self._state, "Too many joints hidden \u2014 adjust your position"

    def reset(self) -> None:
        """Reset state machine."""
        self._bad_frame_count = 0
        self._state = DetectionQuality.LOST
