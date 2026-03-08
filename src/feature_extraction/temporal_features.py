"""
Temporal feature extraction from pose sequences.
Computes velocity, acceleration, and range of motion over frame windows.
"""

from typing import Dict, List, Optional

import numpy as np

from ..pose_estimation.base import PoseResult
from ..utils.constants import JOINT_ANGLES, MIN_VISIBILITY
from ..utils.geometry import calculate_angle, distance


def _get_world_landmark(pose: PoseResult, idx: int) -> np.ndarray:
    return pose.get_world_landmark(idx)


def compute_velocity(
    frames: List[PoseResult],
    landmark_idx: int,
) -> Optional[float]:
    """
    Average velocity of a landmark across frames (meters/second).
    Requires at least 2 frames with valid timestamps.
    """
    if len(frames) < 2:
        return None

    velocities = []
    for i in range(1, len(frames)):
        dt_ms = frames[i].timestamp_ms - frames[i - 1].timestamp_ms
        if dt_ms <= 0:
            continue

        if not (frames[i].is_visible(landmark_idx, MIN_VISIBILITY) and
                frames[i - 1].is_visible(landmark_idx, MIN_VISIBILITY)):
            continue

        p1 = _get_world_landmark(frames[i - 1], landmark_idx)
        p2 = _get_world_landmark(frames[i], landmark_idx)
        dist = distance(p1, p2)
        vel = dist / (dt_ms / 1000.0)  # meters per second
        velocities.append(vel)

    return float(np.mean(velocities)) if velocities else None


def compute_acceleration(
    frames: List[PoseResult],
    landmark_idx: int,
) -> Optional[float]:
    """
    Average acceleration of a landmark (m/s^2).
    Requires at least 3 frames.
    """
    if len(frames) < 3:
        return None

    velocities = []
    timestamps = []

    for i in range(1, len(frames)):
        dt_ms = frames[i].timestamp_ms - frames[i - 1].timestamp_ms
        if dt_ms <= 0:
            continue
        if not (frames[i].is_visible(landmark_idx, MIN_VISIBILITY) and
                frames[i - 1].is_visible(landmark_idx, MIN_VISIBILITY)):
            continue

        p1 = _get_world_landmark(frames[i - 1], landmark_idx)
        p2 = _get_world_landmark(frames[i], landmark_idx)
        vel = distance(p1, p2) / (dt_ms / 1000.0)
        velocities.append(vel)
        timestamps.append((frames[i].timestamp_ms + frames[i - 1].timestamp_ms) / 2.0)

    if len(velocities) < 2:
        return None

    accels = []
    for i in range(1, len(velocities)):
        dt = (timestamps[i] - timestamps[i - 1]) / 1000.0
        if dt <= 0:
            continue
        accels.append(abs(velocities[i] - velocities[i - 1]) / dt)

    return float(np.mean(accels)) if accels else None


def compute_angle_rom(
    frames: List[PoseResult],
    angle_name: str,
    use_world: bool = True,
) -> Optional[float]:
    """
    Range of motion for a joint angle across frames.
    Returns max_angle - min_angle in degrees.
    """
    if angle_name not in JOINT_ANGLES:
        return None

    idx_a, idx_b, idx_c = JOINT_ANGLES[angle_name]
    angles = []

    for pose in frames:
        if not all(pose.is_visible(idx, MIN_VISIBILITY) for idx in [idx_a, idx_b, idx_c]):
            continue

        if use_world:
            a = pose.get_world_landmark(idx_a)
            b = pose.get_world_landmark(idx_b)
            c = pose.get_world_landmark(idx_c)
        else:
            a = pose.get_landmark(idx_a)
            b = pose.get_landmark(idx_b)
            c = pose.get_landmark(idx_c)

        angle = calculate_angle(a, b, c)
        if angle is not None:
            angles.append(angle)

    if len(angles) < 2:
        return None

    return float(max(angles) - min(angles))


class TemporalFeatureExtractor:
    """Extracts temporal features from a sequence of PoseResults."""

    # Only use the most recent N frames for temporal features (performance)
    _TEMPORAL_WINDOW = 10

    def __init__(self, use_world: bool = True):
        self.use_world = use_world

    def extract_temporal(
        self, frames, exercise: str
    ) -> Dict[str, Optional[float]]:
        """
        Extract all temporal features for an exercise from a frame sequence.
        Accepts a list or deque; only uses the last _TEMPORAL_WINDOW frames.
        Returns dict of feature_name -> value.
        """
        from .exercise_features import EXERCISE_FEATURES

        features = {}
        exercise_def = EXERCISE_FEATURES.get(exercise, {})

        # Slice to last N frames to limit computation
        if len(frames) > self._TEMPORAL_WINDOW:
            recent = list(frames)[-self._TEMPORAL_WINDOW:]
        else:
            recent = list(frames)

        for tf_name in exercise_def.get("temporal_features", []):
            features[tf_name] = self._compute(tf_name, recent, exercise_def)

        return features

    def _compute(
        self, name: str, frames: List[PoseResult], exercise_def: dict
    ) -> Optional[float]:
        """Dispatch temporal feature computation by name."""
        # ROM features: "{angle}_rom"
        if name.endswith("_rom"):
            angle_name = name[:-4]  # strip "_rom"
            return compute_angle_rom(frames, angle_name, self.use_world)

        # Velocity features: "{landmark}_velocity"
        if name.endswith("_velocity"):
            landmark_name = name[:-9]  # strip "_velocity"
            from ..utils.constants import LANDMARK_INDICES
            idx = LANDMARK_INDICES.get(landmark_name)
            if idx is not None:
                return compute_velocity(frames, idx)
            return None

        # Acceleration features: "{landmark}_accel"
        if name.endswith("_accel"):
            landmark_name = name[:-6]  # strip "_accel"
            from ..utils.constants import LANDMARK_INDICES
            idx = LANDMARK_INDICES.get(landmark_name)
            if idx is not None:
                return compute_acceleration(frames, idx)
            return None

        return None

    def get_temporal_feature_names(self, exercise: str) -> List[str]:
        """Return ordered list of temporal feature names for an exercise."""
        from .exercise_features import EXERCISE_FEATURES
        exercise_def = EXERCISE_FEATURES.get(exercise, {})
        return list(exercise_def.get("temporal_features", []))
