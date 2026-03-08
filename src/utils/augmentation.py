"""
Data augmentation for pose landmarks.
Operates on PoseResult objects (not images), applying geometric transforms.
"""

from typing import List

import numpy as np

from ..pose_estimation.base import PoseResult
from ..utils.constants import NUM_LANDMARKS

# Left/right landmark index pairs for horizontal flip
_LR_PAIRS = [
    (1, 4), (2, 5), (3, 6),     # eyes
    (7, 8),                       # ears
    (9, 10),                      # mouth
    (11, 12),                     # shoulders
    (13, 14),                     # elbows
    (15, 16),                     # wrists
    (17, 18),                     # pinkies
    (19, 20),                     # indices
    (21, 22),                     # thumbs
    (23, 24),                     # hips
    (25, 26),                     # knees
    (27, 28),                     # ankles
    (29, 30),                     # heels
    (31, 32),                     # foot indices
]


def add_gaussian_noise(
    pose: PoseResult, std: float = 0.005
) -> PoseResult:
    """Add small Gaussian noise to landmark coordinates."""
    noise_lm = np.random.normal(0, std, pose.landmarks[:, :3].shape)
    noise_wlm = np.random.normal(0, std * 10, pose.world_landmarks.shape)  # world is in meters

    new_lm = pose.landmarks.copy()
    new_lm[:, :3] += noise_lm

    new_wlm = pose.world_landmarks.copy() + noise_wlm

    return PoseResult(
        landmarks=new_lm,
        world_landmarks=new_wlm,
        detection_confidence=pose.detection_confidence,
        timestamp_ms=pose.timestamp_ms,
    )


def scale_landmarks(
    pose: PoseResult, factor: float = None
) -> PoseResult:
    """Scale all coordinates by a random factor (default 0.9-1.1)."""
    if factor is None:
        factor = np.random.uniform(0.9, 1.1)

    new_lm = pose.landmarks.copy()
    new_lm[:, :3] *= factor

    new_wlm = pose.world_landmarks.copy() * factor

    return PoseResult(
        landmarks=new_lm,
        world_landmarks=new_wlm,
        detection_confidence=pose.detection_confidence,
        timestamp_ms=pose.timestamp_ms,
    )


def horizontal_flip(pose: PoseResult) -> PoseResult:
    """Swap left and right landmarks (mirror the pose)."""
    new_lm = pose.landmarks.copy()
    new_wlm = pose.world_landmarks.copy()

    # Flip x-coordinates
    new_lm[:, 0] = 1.0 - new_lm[:, 0]  # normalized coords: flip around 0.5
    new_wlm[:, 0] = -new_wlm[:, 0]      # world coords: negate x

    # Swap left/right landmark pairs
    for l_idx, r_idx in _LR_PAIRS:
        new_lm[[l_idx, r_idx]] = new_lm[[r_idx, l_idx]]
        new_wlm[[l_idx, r_idx]] = new_wlm[[r_idx, l_idx]]

    return PoseResult(
        landmarks=new_lm,
        world_landmarks=new_wlm,
        detection_confidence=pose.detection_confidence,
        timestamp_ms=pose.timestamp_ms,
    )


def rotate_y_axis(
    pose: PoseResult, angle_deg: float = None
) -> PoseResult:
    """Rotate landmarks around the vertical (y) axis. World coords only."""
    if angle_deg is None:
        angle_deg = np.random.uniform(-15, 15)

    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Rotation matrix around y-axis
    rot = np.array([
        [cos_t,  0, sin_t],
        [0,      1, 0],
        [-sin_t, 0, cos_t],
    ])

    new_wlm = (rot @ pose.world_landmarks.T).T

    return PoseResult(
        landmarks=pose.landmarks.copy(),  # normalized coords unchanged
        world_landmarks=new_wlm,
        detection_confidence=pose.detection_confidence,
        timestamp_ms=pose.timestamp_ms,
    )


def augment_pose(pose: PoseResult, n_augmented: int = 4) -> List[PoseResult]:
    """
    Generate augmented copies of a PoseResult.
    Returns list of n_augmented new PoseResult objects.
    """
    augmented = []

    # 1) Gaussian noise (2 variants)
    augmented.append(add_gaussian_noise(pose, std=0.003))
    augmented.append(add_gaussian_noise(pose, std=0.007))

    # 2) Scale
    if n_augmented >= 3:
        augmented.append(scale_landmarks(pose))

    # 3) Horizontal flip
    if n_augmented >= 4:
        augmented.append(horizontal_flip(pose))

    return augmented[:n_augmented]
