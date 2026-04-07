"""
Temporal smoothing for pose landmarks.

Two implementations:
- TemporalSmoother: Simple sliding window with visibility-weighted averaging (legacy).
- OneEuroSmoother: Adaptive smoothing using One-Euro Filters with MediaPipe's exact
  parameters (min_cutoff=0.05, beta=80.0, d_cutoff=1.0). Near-zero lag during fast
  movement, heavy smoothing when still.
"""

import math
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


class OneEuroFilter:
    """Single-channel adaptive low-pass filter.

    Reproduces MediaPipe's internal landmark smoothing from
    pose_landmark_filtering.pbtxt: min_cutoff=0.05, beta=80.0, d_cutoff=1.0.

    Uses raw derivative estimation (x - x_prev) matching MediaPipe's C++ impl,
    not the filtered derivative from the original paper.
    """

    def __init__(self, t0: float, x0: float,
                 min_cutoff: float = 0.05, beta: float = 80.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        self.dx_prev = 0.0
        self.t_prev = t0

    def __call__(self, t: float, x: float) -> float:
        dt = max(t - self.t_prev, 1e-6)

        # Raw derivative (MediaPipe style)
        dx = (x - self.x_prev) / dt

        # Smooth the derivative
        a_d = self._alpha(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # Adaptive cutoff based on speed
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Smooth the signal
        a = self._alpha(dt, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)


class OneEuroSmoother:
    """Drop-in replacement for TemporalSmoother using per-landmark One-Euro Filters.

    Same interface: add(pose), get_smoothed(), reset(), is_ready.
    Uses MediaPipe's exact parameters for landmark smoothing.

    Visibility-aware: skips filter update for invisible landmarks (visibility < 0.5),
    resets filters after 10 consecutive invisible frames per landmark.
    """

    def __init__(self, min_cutoff: float = 0.05, beta: float = 80.0, d_cutoff: float = 1.0):
        self._params = (min_cutoff, beta, d_cutoff)
        self._filters = {}           # (landmark_idx, axis, 'lm'|'wlm') -> OneEuroFilter
        self._invisible_count = {}   # landmark_idx -> consecutive invisible frame count
        self._last_pose: Optional[PoseResult] = None

    def add(self, pose: PoseResult) -> None:
        """Add a new PoseResult to be smoothed."""
        self._last_pose = pose

    def get_smoothed(self) -> Optional[PoseResult]:
        """Return a smoothed PoseResult using adaptive One-Euro filtering."""
        if self._last_pose is None:
            return None

        pose = self._last_pose
        t = pose.timestamp_ms / 1000.0  # convert to seconds

        smoothed_lm = np.copy(pose.landmarks)         # (33, 4)
        smoothed_wlm = np.copy(pose.world_landmarks)  # (33, 3)

        for i in range(NUM_LANDMARKS):
            vis = pose.landmarks[i, 3]

            if vis < 0.5:
                # Don't update filter for invisible landmarks
                count = self._invisible_count.get(i, 0) + 1
                self._invisible_count[i] = count
                if count > 10:
                    # Reset after prolonged invisibility
                    for ax in range(3):
                        self._filters.pop((i, ax, 'lm'), None)
                        self._filters.pop((i, ax, 'wlm'), None)
                continue

            self._invisible_count[i] = 0

            for ax in range(3):
                key_lm = (i, ax, 'lm')
                key_wlm = (i, ax, 'wlm')
                val_lm = float(pose.landmarks[i, ax])
                val_wlm = float(pose.world_landmarks[i, ax])

                if key_lm not in self._filters:
                    # First observation — initialise filters
                    self._filters[key_lm] = OneEuroFilter(t, val_lm, *self._params)
                    self._filters[key_wlm] = OneEuroFilter(t, val_wlm, *self._params)
                else:
                    smoothed_lm[i, ax] = self._filters[key_lm](t, val_lm)
                    smoothed_wlm[i, ax] = self._filters[key_wlm](t, val_wlm)

        return PoseResult(
            landmarks=smoothed_lm,
            world_landmarks=smoothed_wlm,
            detection_confidence=pose.detection_confidence,
            timestamp_ms=pose.timestamp_ms,
        )

    def reset(self) -> None:
        """Clear all filter state."""
        self._filters.clear()
        self._invisible_count.clear()
        self._last_pose = None

    @property
    def is_ready(self) -> bool:
        """Always ready after first frame (no window to fill)."""
        return self._last_pose is not None
