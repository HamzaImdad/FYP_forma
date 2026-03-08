"""
Temporal smoothing for pose landmarks.
Reduces jitter by averaging over a sliding window of frames.
"""

from collections import deque
from typing import Optional

import numpy as np

from ..pose_estimation.base import PoseResult
from .constants import NUM_LANDMARKS


class TemporalSmoother:
    """Sliding window smoother for PoseResult sequences."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self._buffer = deque(maxlen=window_size)

    def add(self, pose: PoseResult) -> None:
        """Add a new PoseResult to the buffer."""
        self._buffer.append(pose)

    def get_smoothed(self) -> Optional[PoseResult]:
        """
        Return a smoothed PoseResult by averaging the buffer.
        Landmarks are weighted by their visibility scores.
        Returns None if buffer is empty.
        """
        if not self._buffer:
            return None

        n = len(self._buffer)
        if n == 1:
            return self._buffer[0]

        # Stack all landmarks and world landmarks
        all_lm = np.stack([p.landmarks for p in self._buffer])        # (n, 33, 4)
        all_wlm = np.stack([p.world_landmarks for p in self._buffer]) # (n, 33, 3)

        # Use visibility as weights for averaging coordinates
        vis = all_lm[:, :, 3]                  # (n, 33)
        weights = vis / (vis.sum(axis=0, keepdims=True) + 1e-8)  # (n, 33)

        # Weighted average of normalized landmarks (x, y, z)
        smoothed_xyz = np.zeros((NUM_LANDMARKS, 3))
        for i in range(3):
            smoothed_xyz[:, i] = (all_lm[:, :, i] * weights).sum(axis=0)

        # Average visibility
        smoothed_vis = vis.mean(axis=0)

        # Combine into (33, 4)
        smoothed_landmarks = np.column_stack([smoothed_xyz, smoothed_vis])

        # Weighted average of world landmarks
        smoothed_world = np.zeros((NUM_LANDMARKS, 3))
        for i in range(3):
            smoothed_world[:, i] = (all_wlm[:, :, i] * weights).sum(axis=0)

        return PoseResult(
            landmarks=smoothed_landmarks,
            world_landmarks=smoothed_world,
            detection_confidence=float(smoothed_vis.mean()),
            timestamp_ms=self._buffer[-1].timestamp_ms,
        )

    def reset(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()

    @property
    def is_ready(self) -> bool:
        """True when buffer is full."""
        return len(self._buffer) >= self.window_size
