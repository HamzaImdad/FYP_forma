"""
Base classes for pose estimation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PoseResult:
    """Container for a single pose detection result."""
    landmarks: np.ndarray           # shape (33, 4): x, y, z, visibility (normalized 0-1)
    world_landmarks: np.ndarray     # shape (33, 3): x, y, z (meters, hip-centered)
    detection_confidence: float
    timestamp_ms: int = 0

    def get_landmark(self, idx: int) -> np.ndarray:
        """Get normalized (x, y, z) for a landmark by index."""
        return self.landmarks[idx, :3]

    def get_world_landmark(self, idx: int) -> np.ndarray:
        """Get world (x, y, z) for a landmark by index."""
        return self.world_landmarks[idx]

    def is_visible(self, idx: int, threshold: float = 0.5) -> bool:
        """Check if a landmark meets the visibility threshold."""
        return self.landmarks[idx, 3] >= threshold


class PoseEstimator(ABC):
    """Abstract base class for pose estimation backends."""

    @abstractmethod
    def process_image(self, image: np.ndarray) -> Optional[PoseResult]:
        """Process a single BGR image. Returns PoseResult or None."""
        pass

    @abstractmethod
    def process_video_frame(self, frame: np.ndarray, timestamp_ms: int) -> Optional[PoseResult]:
        """Process a video frame with timestamp. Returns PoseResult or None."""
        pass

    @abstractmethod
    def close(self):
        """Release resources."""
        pass
