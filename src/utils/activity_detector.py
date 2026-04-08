"""
Velocity-based activity detection and rest state classification.

Tracks landmark positions between frames to compute motion velocity,
classifies static poses (standing, sitting), and manages multi-tier
rest timeouts for accurate set detection.
"""

import time
from collections import deque
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from ..pose_estimation.base import PoseResult
from ..utils.constants import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_WRIST, RIGHT_WRIST, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_KNEE, RIGHT_KNEE,
)


# ── Velocity thresholds (m/s, world coordinates) ──────────────────────
VELOCITY_ACTIVE = 0.15      # average joint velocity for "exercising"
VELOCITY_TRANSITION = 0.05  # below this = idle/resting
VELOCITY_BUFFER_SIZE = 10   # frames to smooth velocity

# ── Key landmarks to track for velocity ───────────────────────────────
TRACKED_LANDMARKS = [
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_ANKLE, RIGHT_ANKLE,
]

# ── Rest tier timeouts (seconds) ──────────────────────────────────────
REST_BETWEEN_REPS = 5.0     # normal inter-rep pause
REST_SHORT = 15.0           # short rest
REST_EXTENDED = 60.0        # extended rest — triggers new set
REST_LONG = 120.0           # long rest — prompt to resume


class ActivityState(Enum):
    """Current activity classification."""
    EXERCISING = "exercising"
    TRANSITION = "transition"
    STANDING_IDLE = "standing_idle"
    SITTING = "sitting"
    NO_POSE = "no_pose"


class RestTier(Enum):
    """Rest duration tier for set management."""
    ACTIVE = "active"                   # currently moving
    BETWEEN_REPS = "between_reps"       # 0-5s pause
    SHORT_REST = "short_rest"           # 5-15s
    EXTENDED_REST = "extended_rest"     # 15-60s (new set)
    LONG_REST = "long_rest"             # >60s


class ActivityDetector:
    """Detects whether user is actively exercising, resting, or idle.

    Uses landmark velocity and static pose classification to determine
    activity state, with multi-tier rest timeout for set management.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracking state."""
        self._prev_landmarks: Optional[np.ndarray] = None
        self._prev_time: Optional[float] = None
        self._velocity_buffer: deque = deque(maxlen=VELOCITY_BUFFER_SIZE)
        self._idle_start_time: Optional[float] = None
        self._last_movement_time: float = time.time()
        self._current_state = ActivityState.NO_POSE
        self._current_tier = RestTier.ACTIVE

    def update(self, pose: PoseResult, now: float) -> Tuple[ActivityState, RestTier]:
        """Process a new pose frame and return activity state + rest tier.

        Args:
            pose: Current pose result with world landmarks
            now: Current timestamp (time.time())

        Returns:
            Tuple of (ActivityState, RestTier)
        """
        # Extract tracked landmark positions
        positions = self._extract_positions(pose)

        if positions is None:
            self._prev_landmarks = None
            self._prev_time = None
            self._current_state = ActivityState.NO_POSE
            self._current_tier = self._compute_rest_tier(now)
            return self._current_state, self._current_tier

        # Compute velocity
        velocity = self._compute_velocity(positions, now)
        if velocity is not None:
            self._velocity_buffer.append(velocity)

        self._prev_landmarks = positions
        self._prev_time = now

        # Get smoothed velocity
        avg_velocity = self._get_smoothed_velocity()

        # Classify activity state
        if avg_velocity >= VELOCITY_ACTIVE:
            state = ActivityState.EXERCISING
            self._last_movement_time = now
            self._idle_start_time = None
        elif avg_velocity >= VELOCITY_TRANSITION:
            state = ActivityState.TRANSITION
            self._last_movement_time = now
            self._idle_start_time = None
        else:
            # Low velocity — classify static pose
            state = self._classify_static_pose(pose)
            if self._idle_start_time is None:
                self._idle_start_time = now

        self._current_state = state
        self._current_tier = self._compute_rest_tier(now)
        return self._current_state, self._current_tier

    def _extract_positions(self, pose: PoseResult) -> Optional[np.ndarray]:
        """Extract world positions of tracked landmarks.
        Returns array of shape (N, 3) or None if insufficient landmarks visible.
        """
        positions = []
        visible_count = 0
        for idx in TRACKED_LANDMARKS:
            if pose.is_visible(idx):
                positions.append(pose.get_world_landmark(idx))
                visible_count += 1
            else:
                positions.append(np.zeros(3))

        # Need at least half of tracked landmarks visible
        if visible_count < len(TRACKED_LANDMARKS) // 2:
            return None

        return np.array(positions)

    def _compute_velocity(self, positions: np.ndarray, now: float) -> Optional[float]:
        """Compute average velocity of tracked landmarks between frames."""
        if self._prev_landmarks is None or self._prev_time is None:
            return None

        dt = now - self._prev_time
        if dt <= 0 or dt > 1.0:  # skip if too long between frames
            return None

        # Compute per-landmark displacement
        displacements = np.linalg.norm(positions - self._prev_landmarks, axis=1)
        # Only average landmarks that were visible in both frames (non-zero)
        mask = (np.any(positions != 0, axis=1)) & (np.any(self._prev_landmarks != 0, axis=1))
        if not np.any(mask):
            return None

        avg_displacement = np.mean(displacements[mask])
        return avg_displacement / dt  # velocity in m/s

    def _get_smoothed_velocity(self) -> float:
        """Get smoothed velocity from buffer."""
        if not self._velocity_buffer:
            return 0.0
        return sum(self._velocity_buffer) / len(self._velocity_buffer)

    def _classify_static_pose(self, pose: PoseResult) -> ActivityState:
        """Classify static pose as standing idle or sitting."""
        # Compute hip and knee angles to distinguish standing vs sitting
        hip_angles = []
        knee_angles = []

        for s_idx, h_idx, k_idx, a_idx in [
            (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
            (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
        ]:
            if all(pose.is_visible(i) for i in (s_idx, h_idx, k_idx)):
                s = np.array(pose.get_world_landmark(s_idx))
                h = np.array(pose.get_world_landmark(h_idx))
                k = np.array(pose.get_world_landmark(k_idx))
                # Hip angle (shoulder-hip-knee)
                v1 = s - h
                v2 = k - h
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                hip_angles.append(float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))))

            if all(pose.is_visible(i) for i in (h_idx, k_idx, a_idx)):
                h = np.array(pose.get_world_landmark(h_idx))
                k = np.array(pose.get_world_landmark(k_idx))
                a = np.array(pose.get_world_landmark(a_idx))
                v1 = h - k
                v2 = a - k
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                knee_angles.append(float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))))

        if not hip_angles:
            return ActivityState.STANDING_IDLE

        avg_hip = sum(hip_angles) / len(hip_angles)
        avg_knee = sum(knee_angles) / len(knee_angles) if knee_angles else 170.0

        # Sitting: hip ~90, knee ~90
        if 70 <= avg_hip <= 120 and 70 <= avg_knee <= 120:
            return ActivityState.SITTING

        # Standing idle: hip >150, knee >150
        return ActivityState.STANDING_IDLE

    def _compute_rest_tier(self, now: float) -> RestTier:
        """Compute rest tier based on time since last movement."""
        if self._current_state in (ActivityState.EXERCISING, ActivityState.TRANSITION):
            return RestTier.ACTIVE

        idle_duration = now - self._last_movement_time

        if idle_duration < REST_BETWEEN_REPS:
            return RestTier.BETWEEN_REPS
        elif idle_duration < REST_SHORT:
            return RestTier.SHORT_REST
        elif idle_duration < REST_EXTENDED:
            return RestTier.EXTENDED_REST
        else:
            return RestTier.LONG_REST

    @property
    def idle_duration(self) -> float:
        """Seconds since last detected movement."""
        if self._idle_start_time is None:
            return 0.0
        return time.time() - self._idle_start_time

    @property
    def velocity(self) -> float:
        """Current smoothed velocity in m/s."""
        return self._get_smoothed_velocity()
