"""
Project configuration with sensible defaults.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    project_root: Path = None
    model_path: Path = None
    data_dir: Path = None
    landmarks_dir: Path = None
    models_dir: Path = None

    # MediaPipe settings
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    num_poses: int = 1

    # Feature extraction
    use_world_coordinates: bool = True

    def __post_init__(self):
        if self.project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        if self.model_path is None:
            self.model_path = self.project_root / "models" / "mediapipe" / "pose_landmarker_heavy.task"
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.landmarks_dir is None:
            self.landmarks_dir = self.data_dir / "processed" / "landmarks"
        if self.models_dir is None:
            self.models_dir = self.project_root / "models" / "trained"
