"""
Repetition counter using peak detection on joint angle time series.
"""

from typing import Dict, List, Optional

from ..utils.constants import JOINT_ANGLES, MIN_VISIBILITY
from ..utils.geometry import calculate_angle
from ..pose_estimation.base import PoseResult

# Primary angle to track for rep counting per exercise
PRIMARY_ANGLE_FOR_REP = {
    "squat": "left_knee",
    "lunge": "left_knee",
    "deadlift": "left_hip",
    "bench_press": "left_elbow",
    "overhead_press": "left_elbow",
    "pullup": "left_elbow",
    "pushup": "left_elbow",
    "plank": None,  # static hold, no reps
    "bicep_curl": "left_elbow",
    "tricep_dip": "left_elbow",
}


class RepCounter:
    """Detects exercise repetitions from streaming angle data."""

    def __init__(self, exercise: str, min_angle_change: float = 30.0):
        """
        Args:
            exercise: Exercise name.
            min_angle_change: Minimum angle change to count as a rep (degrees).
        """
        self.exercise = exercise
        self.min_angle_change = min_angle_change
        self._angle_name = PRIMARY_ANGLE_FOR_REP.get(exercise)
        self._angle_history = []
        self._rep_count = 0
        self._direction = None  # "down" or "up"
        self._min_angle = float("inf")
        self._max_angle = float("-inf")
        self._rep_boundaries = []  # list of (start_idx, end_idx) per rep
        self._current_rep_start = 0

    def update(self, pose: PoseResult) -> None:
        """Feed a new frame. Updates rep count internally."""
        if self._angle_name is None:
            return

        angle = self._compute_angle(pose)
        if angle is None:
            return

        self._angle_history.append(angle)
        idx = len(self._angle_history) - 1

        if self._direction is None:
            # Initializing - need at least 2 points to determine direction
            if len(self._angle_history) >= 2:
                if angle < self._angle_history[-2]:
                    self._direction = "down"
                else:
                    self._direction = "up"
            self._min_angle = min(self._min_angle, angle)
            self._max_angle = max(self._max_angle, angle)
            return

        if self._direction == "down":
            if angle < self._min_angle:
                self._min_angle = angle
            # Detect reversal (started going up)
            if angle > self._min_angle + 10:
                self._direction = "up"
                self._max_angle = angle

        elif self._direction == "up":
            if angle > self._max_angle:
                self._max_angle = angle
            # Detect reversal (started going down) = one rep complete
            if angle < self._max_angle - 10:
                rom = self._max_angle - self._min_angle
                if rom >= self.min_angle_change:
                    self._rep_count += 1
                    self._rep_boundaries.append((self._current_rep_start, idx))
                    self._current_rep_start = idx
                self._direction = "down"
                self._min_angle = angle

    def _compute_angle(self, pose: PoseResult) -> Optional[float]:
        """Compute the primary tracking angle from a pose."""
        if self._angle_name not in JOINT_ANGLES:
            return None

        idx_a, idx_b, idx_c = JOINT_ANGLES[self._angle_name]
        if not all(pose.is_visible(i, MIN_VISIBILITY) for i in [idx_a, idx_b, idx_c]):
            return None

        a = pose.get_world_landmark(idx_a)
        b = pose.get_world_landmark(idx_b)
        c = pose.get_world_landmark(idx_c)
        return calculate_angle(a, b, c)

    def get_rep_count(self) -> int:
        """Return current number of completed reps."""
        return self._rep_count

    def get_current_rom(self) -> float:
        """Return range of motion of the current (possibly incomplete) rep."""
        return self._max_angle - self._min_angle

    def get_rep_boundaries(self) -> List[tuple]:
        """Return list of (start_idx, end_idx) for each completed rep."""
        return list(self._rep_boundaries)

    def reset(self) -> None:
        """Reset the counter."""
        self._angle_history.clear()
        self._rep_count = 0
        self._direction = None
        self._min_angle = float("inf")
        self._max_angle = float("-inf")
        self._rep_boundaries.clear()
        self._current_rep_start = 0
