"""
Data augmentation for pose landmarks and extracted features.

Provides two levels of augmentation:
  1. PoseResult-level: geometric transforms on raw landmarks (augment_pose)
  2. Feature-level: noise/mirror on extracted angle features (augment_angle_features)

The feature-level functions operate on dicts of {column_name: value} and are
used by scripts/balance_data.py to augment the minority class directly in
feature space without needing to go back to raw landmarks.
"""

from typing import Dict, List, Optional

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


# ---------------------------------------------------------------------------
# Feature-level augmentation (operates on extracted angle/feature dicts)
# ---------------------------------------------------------------------------

# Left/right feature name pairs for mirroring
_LR_FEATURE_PAIRS = [
    ("left_elbow", "right_elbow"),
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
    ("left_wrist", "right_wrist"),
]


def mirror_angle_feature_names(name: str) -> str:
    """Swap 'left' <-> 'right' in a feature column name."""
    for left, right in _LR_FEATURE_PAIRS:
        if left in name:
            return name.replace(left, right)
        if right in name:
            return name.replace(right, left)
    return name


def _add_angle_noise(features: Dict[str, float], std: float = 0.02) -> Dict[str, float]:
    """Add Gaussian noise to all numeric feature values."""
    result = {}
    for k, v in features.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            result[k] = v
        elif isinstance(v, (int, float)):
            result[k] = float(v) + np.random.normal(0, std)
        else:
            result[k] = v
    return result


def _mirror_angle_features(features: Dict[str, float]) -> Dict[str, float]:
    """Swap left/right feature columns (mirror augmentation)."""
    result = {}
    swapped = set()

    for k, v in features.items():
        if k in swapped:
            continue
        mirrored_k = mirror_angle_feature_names(k)
        if mirrored_k != k and mirrored_k in features:
            result[k] = features[mirrored_k]
            result[mirrored_k] = features[k]
            swapped.add(k)
            swapped.add(mirrored_k)
        else:
            result[k] = v

    return result


def _rotation_perturbation(features: Dict[str, float], max_deg: float = 3.0) -> Dict[str, float]:
    """Add uniform ±max_deg noise to all angle columns."""
    result = {}
    for k, v in features.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            result[k] = v
        elif isinstance(v, (int, float)) and k.startswith("angle_"):
            result[k] = float(v) + np.random.uniform(-max_deg, max_deg)
        else:
            result[k] = v
    return result


def augment_angle_features(
    features: Dict[str, float],
    n_augmented: int = 4,
) -> List[Dict[str, float]]:
    """
    Generate augmented copies of an angle-feature dict.

    Augmentation strategies:
      1. Gaussian noise (std=0.02) on all numeric features
      2. Mirror left/right
      3. Rotation perturbation (±3 degrees on angle columns)
      4. Gaussian noise (std=0.04) — stronger variant

    Returns list of up to n_augmented new feature dicts.
    """
    augmented = []

    # 1) Light Gaussian noise
    augmented.append(_add_angle_noise(features, std=0.02))

    # 2) Mirror left/right
    if n_augmented >= 2:
        augmented.append(_mirror_angle_features(features))

    # 3) Rotation perturbation
    if n_augmented >= 3:
        augmented.append(_rotation_perturbation(features, max_deg=3.0))

    # 4) Stronger Gaussian noise
    if n_augmented >= 4:
        augmented.append(_add_angle_noise(features, std=0.04))

    return augmented[:n_augmented]


def augment_landmark_features(row: np.ndarray) -> List[np.ndarray]:
    """
    Augment a raw landmark feature vector (99 values = 33 landmarks x 3 coords).

    Returns list of augmented copies:
      1. Gaussian noise N(0, 0.01)
      2. Mirror left/right (swap landmark pairs + negate x)
      3. Scale perturbation (uniform 0.95-1.05)
      4. Combined noise + scale

    Parameters
    ----------
    row : np.ndarray
        Shape (99,) — flattened [x0, y0, z0, x1, y1, z1, ...] for 33 landmarks.

    Returns
    -------
    List of np.ndarray, each shape (99,).
    """
    augmented = []

    # 1) Gaussian noise
    noisy = row.copy()
    noisy += np.random.normal(0, 0.01, noisy.shape)
    augmented.append(noisy)

    # 2) Mirror left/right
    mirrored = row.copy().reshape(33, 3)
    mirrored[:, 0] = -mirrored[:, 0]  # negate x (world coords are hip-centered)
    for l_idx, r_idx in _LR_PAIRS:
        mirrored[[l_idx, r_idx]] = mirrored[[r_idx, l_idx]]
    augmented.append(mirrored.flatten())

    # 3) Scale perturbation
    scale = np.random.uniform(0.95, 1.05)
    scaled = row.copy() * scale
    augmented.append(scaled)

    # 4) Combined noise + scale
    combined = row.copy()
    combined *= np.random.uniform(0.95, 1.05)
    combined += np.random.normal(0, 0.01, combined.shape)
    augmented.append(combined)

    return augmented
