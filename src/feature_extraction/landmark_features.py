"""
Raw normalized landmark feature extraction.

Extracts all 33 MediaPipe landmarks as body-centered, scale-normalized
(x, y, z) coordinates = 99 features per frame. This replaces the sparse
hand-crafted features for ML/BiLSTM training.

Wave 4 additions:
  - Temporal features (velocity + acceleration): 99 + 99 = 198 extra dims
  - Symmetry features (left-right diffs for 11 pairs): 33 extra dims
  - Bone-length normalization: alternative 99-dim extraction
  - Full temporal extraction: 330 features total
"""

from typing import Dict, List, Tuple

import numpy as np

from ..pose_estimation.base import PoseResult
from ..utils.constants import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    NUM_LANDMARKS, LANDMARK_NAMES,
)
from ..utils.geometry import distance, midpoint
from .angles import compute_joint_angles
from .exercise_features import EXERCISE_FEATURES


# Column names for the 99 landmark features
LANDMARK_FEATURE_NAMES = []
for _name in LANDMARK_NAMES:
    LANDMARK_FEATURE_NAMES.extend([f"lm_{_name}_x", f"lm_{_name}_y", f"lm_{_name}_z"])

# Column names for the 33 visibility mask values
VISIBILITY_MASK_NAMES = [f"vis_{_name}" for _name in LANDMARK_NAMES]

# Wave 4A: Velocity and acceleration feature names
VELOCITY_FEATURE_NAMES = [f"vel_{name}" for name in LANDMARK_FEATURE_NAMES]
ACCELERATION_FEATURE_NAMES = [f"acc_{name}" for name in LANDMARK_FEATURE_NAMES]

# Wave 4B: Symmetry pair indices (left, right) and feature names
SYMMETRY_PAIR_INDICES = [
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]
SYMMETRY_FEATURE_NAMES = []
for _left, _right in SYMMETRY_PAIR_INDICES:
    _lname = LANDMARK_NAMES[_left]
    _rname = LANDMARK_NAMES[_right]
    SYMMETRY_FEATURE_NAMES.extend([
        f"sym_{_lname}_{_rname}_x",
        f"sym_{_lname}_{_rname}_y",
        f"sym_{_lname}_{_rname}_z",
    ])

# All temporal feature names: position(99) + velocity(99) + acceleration(99) + symmetry(33) = 330
ALL_TEMPORAL_FEATURE_NAMES = (
    LANDMARK_FEATURE_NAMES
    + VELOCITY_FEATURE_NAMES
    + ACCELERATION_FEATURE_NAMES
    + SYMMETRY_FEATURE_NAMES
)


class LandmarkFeatureExtractor:
    """Extracts scale-normalized raw landmark coordinates from PoseResult."""

    def __init__(self, visibility_threshold: float = 0.5):
        self.visibility_threshold = visibility_threshold

    def extract_landmarks(self, pose: PoseResult) -> np.ndarray:
        """
        Extract normalized landmark vector from world coordinates.

        Normalization:
          1. Translate: subtract hip midpoint (already hip-centered in world coords,
             but we re-center explicitly for robustness)
          2. Scale: divide by torso length (hip midpoint to shoulder midpoint)

        Returns:
            np.ndarray of shape (99,) — 33 landmarks x 3 coords each
        """
        wl = pose.world_landmarks.copy()  # (33, 3)

        # Hip midpoint (translation origin)
        hip_mid = midpoint(wl[LEFT_HIP], wl[RIGHT_HIP])

        # Shoulder midpoint
        shoulder_mid = midpoint(wl[LEFT_SHOULDER], wl[RIGHT_SHOULDER])

        # Torso length for scale normalization
        torso_len = distance(hip_mid, shoulder_mid)
        if torso_len < 1e-6:
            torso_len = 1.0  # fallback to avoid division by zero

        # Translate and scale
        normalized = (wl - hip_mid) / torso_len

        # Zero out low-visibility landmarks
        for i in range(NUM_LANDMARKS):
            if not pose.is_visible(i, self.visibility_threshold):
                normalized[i] = 0.0

        return normalized.flatten().astype(np.float32)  # (99,)

    def extract_landmarks_with_mask(self, pose: PoseResult) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract normalized landmarks plus a visibility mask.

        Returns:
            (landmarks shape (99,), visibility_mask shape (33,))
            mask[i] = 1.0 if landmark i is visible, 0.0 otherwise
        """
        landmarks = self.extract_landmarks(pose)
        mask = np.array([
            1.0 if pose.is_visible(i, self.visibility_threshold) else 0.0
            for i in range(NUM_LANDMARKS)
        ], dtype=np.float32)
        return landmarks, mask

    def extract_full(self, pose: PoseResult, exercise: str) -> Dict[str, float]:
        """
        Extract both raw normalized landmarks AND the original hand-crafted
        angle/custom features needed by the rule-based classifier.

        Returns:
            dict with keys: lm_<name>_x/y/z (99 keys) + angle_* + custom features
        """
        # Raw landmark features
        lm_vec = self.extract_landmarks(pose)
        result = {}
        for i, col_name in enumerate(LANDMARK_FEATURE_NAMES):
            result[col_name] = float(lm_vec[i])

        # Hand-crafted angle features (same as FeatureExtractor.extract)
        angles = compute_joint_angles(pose, use_world=True)
        exercise_def = EXERCISE_FEATURES.get(exercise, {})

        for angle_name in exercise_def.get("primary_angles", []):
            val = angles.get(angle_name)
            result[f"angle_{angle_name}"] = float(val) if val is not None else 0.0

        for angle_name in exercise_def.get("secondary_angles", []):
            val = angles.get(angle_name)
            result[f"angle_{angle_name}"] = float(val) if val is not None else 0.0

        return result

    # ------------------------------------------------------------------
    # Wave 4A: Temporal features (velocity + acceleration)
    # ------------------------------------------------------------------

    def extract_temporal_landmarks(self, poses: List[PoseResult]) -> np.ndarray:
        """
        Compute position + velocity + acceleration from a frame buffer.

        Takes a list of PoseResult objects (the frame buffer, typically last
        3+ frames). Uses the last frame's position as the base, and computes
        finite-difference velocity and acceleration.

        Args:
            poses: List of PoseResult objects (oldest first, newest last).
                   Must contain at least 1 frame.

        Returns:
            np.ndarray of shape (297,): position(99) + velocity(99) + acceleration(99)
        """
        # Extract normalized landmarks for each frame
        all_lm = [self.extract_landmarks(p) for p in poses]

        # Current position is always the last frame
        position = all_lm[-1]  # (99,)

        # Velocity: difference between last two frames
        if len(all_lm) >= 2:
            velocity = all_lm[-1] - all_lm[-2]  # (99,)
        else:
            velocity = np.zeros(99, dtype=np.float32)

        # Acceleration: difference between last two velocities
        if len(all_lm) >= 3:
            vel_prev = all_lm[-2] - all_lm[-3]
            acceleration = velocity - vel_prev  # (99,)
        else:
            acceleration = np.zeros(99, dtype=np.float32)

        return np.concatenate([position, velocity, acceleration]).astype(np.float32)  # (297,)

    # ------------------------------------------------------------------
    # Wave 4B: Symmetry features (left-right differences)
    # ------------------------------------------------------------------

    def extract_symmetry_features(self, pose: PoseResult) -> np.ndarray:
        """
        Compute left-right differences for 11 paired body landmarks.

        Uses the same hip-centered, torso-scaled normalization as
        extract_landmarks. For each pair, computes:
            diff = left_xyz - right_xyz  (3 values)

        Pairs (11 total):
            shoulders (11,12), elbows (13,14), wrists (15,16),
            pinkies (17,18), indices (19,20), thumbs (21,22),
            hips (23,24), knees (25,26), ankles (27,28),
            heels (29,30), foot_indices (31,32)

        Returns:
            np.ndarray of shape (33,): 11 pairs x 3 coords
        """
        wl = pose.world_landmarks.copy()  # (33, 3)

        # Apply same normalization as extract_landmarks
        hip_mid = midpoint(wl[LEFT_HIP], wl[RIGHT_HIP])
        shoulder_mid = midpoint(wl[LEFT_SHOULDER], wl[RIGHT_SHOULDER])
        torso_len = distance(hip_mid, shoulder_mid)
        if torso_len < 1e-6:
            torso_len = 1.0

        normalized = (wl - hip_mid) / torso_len

        # Zero out low-visibility landmarks
        for i in range(NUM_LANDMARKS):
            if not pose.is_visible(i, self.visibility_threshold):
                normalized[i] = 0.0

        # Compute left - right differences for each pair
        diffs = []
        for left_idx, right_idx in SYMMETRY_PAIR_INDICES:
            diff = normalized[left_idx] - normalized[right_idx]
            diffs.append(diff)

        return np.concatenate(diffs).astype(np.float32)  # (33,)

    # ------------------------------------------------------------------
    # Wave 4C: Bone-length normalization
    # ------------------------------------------------------------------

    def extract_landmarks_bone_normalized(self, pose: PoseResult) -> np.ndarray:
        """
        Extract landmarks with additional bone-segment normalization for limbs.

        Same as extract_landmarks (hip-centered, torso-scaled) but limb
        landmarks are further divided by their segment bone length:
          - Upper arm landmarks (elbows, wrists, hands) by shoulder-to-elbow dist
          - Forearm landmarks (wrists, hands) by elbow-to-wrist dist
          - Thigh landmarks (knees) by hip-to-knee dist
          - Shin landmarks (ankles, heels, feet) by knee-to-ankle dist

        Only limb landmarks are additionally normalized; torso and head
        landmarks keep the standard torso normalization.

        Returns:
            np.ndarray of shape (99,) — same format as extract_landmarks
        """
        wl = pose.world_landmarks.copy()  # (33, 3)

        # Standard hip-centered, torso-scaled normalization
        hip_mid = midpoint(wl[LEFT_HIP], wl[RIGHT_HIP])
        shoulder_mid = midpoint(wl[LEFT_SHOULDER], wl[RIGHT_SHOULDER])
        torso_len = distance(hip_mid, shoulder_mid)
        if torso_len < 1e-6:
            torso_len = 1.0

        normalized = (wl - hip_mid) / torso_len

        # Zero out low-visibility landmarks
        for i in range(NUM_LANDMARKS):
            if not pose.is_visible(i, self.visibility_threshold):
                normalized[i] = 0.0

        # Bone segment lengths (computed on already-normalized coords)
        # Left arm
        left_upper_arm = distance(normalized[11], normalized[13])   # L shoulder -> L elbow
        left_forearm = distance(normalized[13], normalized[15])     # L elbow -> L wrist
        # Right arm
        right_upper_arm = distance(normalized[12], normalized[14])  # R shoulder -> R elbow
        right_forearm = distance(normalized[14], normalized[16])    # R elbow -> R wrist
        # Left leg
        left_thigh = distance(normalized[23], normalized[25])       # L hip -> L knee
        left_shin = distance(normalized[25], normalized[27])        # L knee -> L ankle
        # Right leg
        right_thigh = distance(normalized[24], normalized[26])      # R hip -> R knee
        right_shin = distance(normalized[26], normalized[28])       # R knee -> R ankle

        def _safe(length: float) -> float:
            return length if length > 1e-6 else 1.0

        # Normalize limb landmarks by their segment bone length
        # Left upper arm: elbow(13)
        normalized[13] /= _safe(left_upper_arm)
        # Left forearm: wrist(15), pinky(17), index(19), thumb(21)
        for idx in [15, 17, 19, 21]:
            normalized[idx] /= _safe(left_forearm)
        # Right upper arm: elbow(14)
        normalized[14] /= _safe(right_upper_arm)
        # Right forearm: wrist(16), pinky(18), index(20), thumb(22)
        for idx in [16, 18, 20, 22]:
            normalized[idx] /= _safe(right_forearm)
        # Left thigh: knee(25)
        normalized[25] /= _safe(left_thigh)
        # Left shin: ankle(27), heel(29), foot_index(31)
        for idx in [27, 29, 31]:
            normalized[idx] /= _safe(left_shin)
        # Right thigh: knee(26)
        normalized[26] /= _safe(right_thigh)
        # Right shin: ankle(28), heel(30), foot_index(32)
        for idx in [28, 30, 32]:
            normalized[idx] /= _safe(right_shin)

        return normalized.flatten().astype(np.float32)  # (99,)

    # ------------------------------------------------------------------
    # Wave 4D: Full temporal extraction (330 features + angle fallback)
    # ------------------------------------------------------------------

    def extract_full_temporal(
        self,
        pose: PoseResult,
        exercise: str,
        frame_history: List[PoseResult],
    ) -> Dict[str, float]:
        """
        Full feature extraction combining landmarks, temporal, and symmetry.

        Combines:
          - Position landmarks (99)
          - Velocity features (99)
          - Acceleration features (99)
          - Symmetry features (33)
          Total: 330 features

        Also includes angle features for rule-based fallback (same as
        extract_full).

        Args:
            pose: Current frame's PoseResult.
            exercise: Exercise name (for angle feature lookup).
            frame_history: List of recent PoseResult objects (oldest first,
                           newest last). Should include the current pose as
                           the last element.

        Returns:
            dict with keys for all 330 temporal features + angle_* features
        """
        result = {}

        # Temporal landmarks: position + velocity + acceleration (297 features)
        # Use frame_history which should include current pose
        history = frame_history if frame_history else [pose]
        temporal_vec = self.extract_temporal_landmarks(history)  # (297,)

        # Position features (99)
        for i, name in enumerate(LANDMARK_FEATURE_NAMES):
            result[name] = float(temporal_vec[i])
        # Velocity features (99)
        for i, name in enumerate(VELOCITY_FEATURE_NAMES):
            result[name] = float(temporal_vec[99 + i])
        # Acceleration features (99)
        for i, name in enumerate(ACCELERATION_FEATURE_NAMES):
            result[name] = float(temporal_vec[198 + i])

        # Symmetry features (33)
        sym_vec = self.extract_symmetry_features(pose)  # (33,)
        for i, name in enumerate(SYMMETRY_FEATURE_NAMES):
            result[name] = float(sym_vec[i])

        # Hand-crafted angle features for rule-based fallback
        angles = compute_joint_angles(pose, use_world=True)
        exercise_def = EXERCISE_FEATURES.get(exercise, {})

        for angle_name in exercise_def.get("primary_angles", []):
            val = angles.get(angle_name)
            result[f"angle_{angle_name}"] = float(val) if val is not None else 0.0

        for angle_name in exercise_def.get("secondary_angles", []):
            val = angles.get(angle_name)
            result[f"angle_{angle_name}"] = float(val) if val is not None else 0.0

        return result

    @staticmethod
    def get_landmark_feature_names():
        """Return the 99 landmark feature column names."""
        return list(LANDMARK_FEATURE_NAMES)

    @staticmethod
    def get_landmark_with_mask_feature_names():
        """Return 99 landmark + 33 visibility = 132 feature column names."""
        return list(LANDMARK_FEATURE_NAMES) + list(VISIBILITY_MASK_NAMES)

    @staticmethod
    def get_temporal_feature_names():
        """Return all 330 temporal feature column names."""
        return list(ALL_TEMPORAL_FEATURE_NAMES)
