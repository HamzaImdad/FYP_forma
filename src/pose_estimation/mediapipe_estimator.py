"""
MediaPipe BlazePose wrapper supporting IMAGE and VIDEO modes.
"""

import logging
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .base import PoseEstimator, PoseResult
from ..utils.config import Config
from ..utils.constants import NUM_LANDMARKS


logger = logging.getLogger(__name__)


class MediaPipePoseEstimator(PoseEstimator):
    """MediaPipe BlazePose pose estimator."""

    def __init__(self, config: Config = None, mode: str = "video"):
        """
        Args:
            config: Config with model path and confidence settings.
            mode: "image" for single images, "video" for video/streaming.
        """
        self.config = config or Config()
        self.mode = mode

        running_mode = (
            vision.RunningMode.IMAGE if mode == "image"
            else vision.RunningMode.VIDEO
        )

        base_options = mp_python.BaseOptions(
            model_asset_path=str(self.config.model_path)
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_poses=self.config.num_poses,
            min_pose_detection_confidence=self.config.min_detection_confidence,
            min_pose_presence_confidence=self.config.min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        logger.info("MediaPipe pose estimator initialised (mode=%s, model=%s)", mode, self.config.model_path.name)

    def _to_pose_result(self, result, timestamp_ms: int = 0) -> Optional[PoseResult]:
        """Convert MediaPipe result to PoseResult."""
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        landmarks = np.array([
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in result.pose_landmarks[0]
        ])

        if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
            world_landmarks = np.array([
                [lm.x, lm.y, lm.z]
                for lm in result.pose_world_landmarks[0]
            ])
        else:
            world_landmarks = np.zeros((NUM_LANDMARKS, 3))

        return PoseResult(
            landmarks=landmarks,
            world_landmarks=world_landmarks,
            detection_confidence=float(landmarks[:, 3].mean()),
            timestamp_ms=timestamp_ms,
        )

    def process_image(self, image: np.ndarray) -> Optional[PoseResult]:
        """Process a single BGR image."""
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect(mp_image)
            return self._to_pose_result(result)
        except Exception as e:
            logger.warning("MediaPipe image processing error: %s", e)
            return None

    def process_video_frame(self, frame: np.ndarray, timestamp_ms: int) -> Optional[PoseResult]:
        """Process a video frame with monotonically increasing timestamp."""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
            return self._to_pose_result(result, timestamp_ms)
        except Exception as e:
            logger.warning("MediaPipe processing error at ts=%d: %s", timestamp_ms, e)
            return None

    def close(self):
        """Release the MediaPipe landmarker."""
        self._landmarker.close()
