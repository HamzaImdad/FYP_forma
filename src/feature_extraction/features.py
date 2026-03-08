"""
Main feature extraction engine.
Computes exercise-specific biomechanical features from pose landmarks.
"""

from typing import Dict, List, Optional

import numpy as np

from ..pose_estimation.base import PoseResult
from ..utils.constants import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE, NOSE,
)
from ..utils.geometry import (
    calculate_angle, distance, midpoint, vertical_angle, lateral_symmetry,
)
from .angles import compute_joint_angles
from .exercise_features import EXERCISE_FEATURES
from .temporal_features import TemporalFeatureExtractor


class FeatureExtractor:
    """Computes exercise-specific feature vectors from pose landmarks."""

    def __init__(self, use_world: bool = True):
        self.use_world = use_world
        self._temporal_extractor = TemporalFeatureExtractor(use_world=use_world)

    def extract(self, pose: PoseResult, exercise: str) -> Dict[str, Optional[float]]:
        """
        Extract all features for a given exercise.
        Returns dict of feature_name -> value (or None if not computable).
        """
        features = {}
        angles = compute_joint_angles(pose, use_world=self.use_world)

        exercise_def = EXERCISE_FEATURES.get(exercise, {})

        for angle_name in exercise_def.get("primary_angles", []):
            features[f"angle_{angle_name}"] = angles.get(angle_name)
        for angle_name in exercise_def.get("secondary_angles", []):
            features[f"angle_{angle_name}"] = angles.get(angle_name)

        for feat_name in exercise_def.get("custom_features", []):
            features[feat_name] = self._compute_custom(feat_name, pose, angles)

        return features

    def extract_with_temporal(
        self,
        pose: PoseResult,
        exercise: str,
        frame_history=None,
    ) -> Dict[str, Optional[float]]:
        """
        Extract single-frame features + temporal features from frame history.
        Accepts a list or deque. If empty, temporal features are set to None.
        """
        features = self.extract(pose, exercise)

        # Add temporal features if history is available
        if frame_history and len(frame_history) >= 2:
            temporal = self._temporal_extractor.extract_temporal(frame_history, exercise)
            features.update(temporal)
        else:
            # Fill temporal features with None
            exercise_def = EXERCISE_FEATURES.get(exercise, {})
            for tf_name in exercise_def.get("temporal_features", []):
                features[tf_name] = None

        return features

    def extract_vector(self, pose: PoseResult, exercise: str) -> np.ndarray:
        """Extract features as a flat numpy array (for ML input). None replaced with 0."""
        features = self.extract(pose, exercise)
        names = self.get_feature_names(exercise)
        return np.array([features.get(n, 0.0) or 0.0 for n in names])

    def extract_full_vector(
        self,
        pose: PoseResult,
        exercise: str,
        frame_history: List[PoseResult] = None,
    ) -> np.ndarray:
        """Extract single-frame + temporal features as a flat numpy array."""
        features = self.extract_with_temporal(pose, exercise, frame_history)
        names = self.get_all_feature_names(exercise)
        return np.array([features.get(n, 0.0) or 0.0 for n in names])

    def get_feature_names(self, exercise: str) -> List[str]:
        """Return ordered list of single-frame feature names for an exercise."""
        exercise_def = EXERCISE_FEATURES.get(exercise, {})
        names = []
        for angle_name in exercise_def.get("primary_angles", []):
            names.append(f"angle_{angle_name}")
        for angle_name in exercise_def.get("secondary_angles", []):
            names.append(f"angle_{angle_name}")
        for feat_name in exercise_def.get("custom_features", []):
            names.append(feat_name)
        return names

    def get_all_feature_names(self, exercise: str) -> List[str]:
        """Return ordered list of all feature names (single-frame + temporal)."""
        names = self.get_feature_names(exercise)
        exercise_def = EXERCISE_FEATURES.get(exercise, {})
        names.extend(exercise_def.get("temporal_features", []))
        return names

    # --- Internal dispatch ---

    def _compute_custom(self, name: str, pose: PoseResult, angles: Dict) -> Optional[float]:
        method = getattr(self, f"_feat_{name}", None)
        if method:
            try:
                return method(pose, angles)
            except Exception:
                return None
        return None

    def _pt(self, pose: PoseResult, idx: int) -> np.ndarray:
        return pose.get_world_landmark(idx) if self.use_world else pose.get_landmark(idx)

    # --- Symmetry features ---

    def _feat_knee_symmetry(self, pose, angles):
        l, r = angles.get("left_knee"), angles.get("right_knee")
        if l is None or r is None:
            return None
        return lateral_symmetry(l, r)

    def _feat_hip_symmetry(self, pose, angles):
        l, r = angles.get("left_hip"), angles.get("right_hip")
        if l is None or r is None:
            return None
        return lateral_symmetry(l, r)

    def _feat_elbow_symmetry(self, pose, angles):
        l, r = angles.get("left_elbow"), angles.get("right_elbow")
        if l is None or r is None:
            return None
        return lateral_symmetry(l, r)

    def _feat_shoulder_symmetry(self, pose, angles):
        l, r = angles.get("left_shoulder"), angles.get("right_shoulder")
        if l is None or r is None:
            return None
        return lateral_symmetry(l, r)

    # --- Torso / alignment features ---

    def _feat_torso_lean(self, pose, angles):
        """Angle of torso (mid-hip to mid-shoulder) from vertical."""
        mid_hip = midpoint(self._pt(pose, LEFT_HIP), self._pt(pose, RIGHT_HIP))
        mid_shoulder = midpoint(self._pt(pose, LEFT_SHOULDER), self._pt(pose, RIGHT_SHOULDER))
        return vertical_angle(mid_hip, mid_shoulder)

    def _feat_spine_alignment(self, pose, angles):
        return self._feat_torso_lean(pose, angles)

    def _feat_torso_swing(self, pose, angles):
        return self._feat_torso_lean(pose, angles)

    def _feat_forward_lean(self, pose, angles):
        return self._feat_torso_lean(pose, angles)

    def _feat_body_line(self, pose, angles):
        """Deviation from a straight shoulder-hip-ankle line (0 = perfect)."""
        mid_shoulder = midpoint(self._pt(pose, LEFT_SHOULDER), self._pt(pose, RIGHT_SHOULDER))
        mid_hip = midpoint(self._pt(pose, LEFT_HIP), self._pt(pose, RIGHT_HIP))
        mid_ankle = midpoint(self._pt(pose, LEFT_ANKLE), self._pt(pose, RIGHT_ANKLE))
        angle = calculate_angle(mid_shoulder, mid_hip, mid_ankle)
        if angle is None:
            return None
        return abs(180.0 - angle)

    def _feat_back_straightness(self, pose, angles):
        return self._feat_body_line(pose, angles)

    # --- Position-based features ---

    def _feat_hip_sag(self, pose, angles):
        """Hip sag below shoulder-ankle line. Positive = sagging."""
        mid_shoulder = midpoint(self._pt(pose, LEFT_SHOULDER), self._pt(pose, RIGHT_SHOULDER))
        mid_hip = midpoint(self._pt(pose, LEFT_HIP), self._pt(pose, RIGHT_HIP))
        mid_ankle = midpoint(self._pt(pose, LEFT_ANKLE), self._pt(pose, RIGHT_ANKLE))
        expected_y = (mid_shoulder[1] + mid_ankle[1]) / 2.0
        return mid_hip[1] - expected_y

    def _feat_hip_pike(self, pose, angles):
        """Hip pike above shoulder-ankle line. Positive = piking."""
        sag = self._feat_hip_sag(pose, angles)
        return -sag if sag is not None else None

    def _feat_hip_below_knee(self, pose, angles):
        """Squat depth ratio: 0 = hip at knee level, 1 = well below knee.
        In world coords, more positive y = lower position (negative = up).
        Returns 0 when hip is at or above knee level."""
        mid_hip = midpoint(self._pt(pose, LEFT_HIP), self._pt(pose, RIGHT_HIP))
        mid_knee = midpoint(self._pt(pose, LEFT_KNEE), self._pt(pose, RIGHT_KNEE))
        # Thigh length as normalizer (hip-to-knee distance)
        thigh_len = abs(mid_hip[1] - mid_knee[1]) + 1e-6
        # How far below knee is the hip (positive = below)
        depth = (mid_hip[1] - mid_knee[1]) / thigh_len
        return float(np.clip(depth, 0.0, 1.0))

    def _feat_knee_over_toe(self, pose, angles):
        """How far knee extends past ankle on the z-axis."""
        mid_knee = midpoint(self._pt(pose, LEFT_KNEE), self._pt(pose, RIGHT_KNEE))
        mid_ankle = midpoint(self._pt(pose, LEFT_ANKLE), self._pt(pose, RIGHT_ANKLE))
        return mid_knee[2] - mid_ankle[2]

    def _feat_hip_alignment(self, pose, angles):
        """Distance between left and right hip (lateral spread)."""
        return distance(self._pt(pose, LEFT_HIP), self._pt(pose, RIGHT_HIP))

    def _feat_wrist_over_elbow(self, pose, angles):
        """How aligned wrists are above elbows (x-axis difference). 0 = perfect."""
        mid_wrist = midpoint(self._pt(pose, LEFT_WRIST), self._pt(pose, RIGHT_WRIST))
        mid_elbow = midpoint(self._pt(pose, LEFT_ELBOW), self._pt(pose, RIGHT_ELBOW))
        return abs(mid_wrist[0] - mid_elbow[0])

    def _feat_lockout_angle(self, pose, angles):
        """Average elbow angle (for overhead press lockout check)."""
        l, r = angles.get("left_elbow"), angles.get("right_elbow")
        if l is None or r is None:
            return None
        return (l + r) / 2.0

    def _feat_chin_above_bar(self, pose, angles):
        """Whether nose is above wrist level. 1=above, 0=below."""
        nose = self._pt(pose, NOSE)
        mid_wrist = midpoint(self._pt(pose, LEFT_WRIST), self._pt(pose, RIGHT_WRIST))
        # In world coords, more negative y = higher
        return 1.0 if nose[1] < mid_wrist[1] else 0.0

    def _feat_body_swing(self, pose, angles):
        """Hip x deviation from shoulder x (kipping check)."""
        mid_shoulder = midpoint(self._pt(pose, LEFT_SHOULDER), self._pt(pose, RIGHT_SHOULDER))
        mid_hip = midpoint(self._pt(pose, LEFT_HIP), self._pt(pose, RIGHT_HIP))
        return abs(mid_hip[0] - mid_shoulder[0])

    def _feat_upper_arm_movement(self, pose, angles):
        """Average shoulder angle - should be minimal during curls."""
        l, r = angles.get("left_shoulder"), angles.get("right_shoulder")
        if l is None or r is None:
            return None
        return (l + r) / 2.0
