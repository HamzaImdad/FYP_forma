"""
Real-time exercise form evaluation pipeline.
Orchestrates pose estimation, feature extraction, classification, and visualization.
Supports session tracking with per-rep form scores and session reports.
"""

import logging
import time
from collections import deque
from typing import Optional, Tuple, List, Dict

import numpy as np

logger = logging.getLogger(__name__)

from ..pose_estimation.base import PoseResult
from ..pose_estimation.mediapipe_estimator import MediaPipePoseEstimator
from ..feature_extraction.features import FeatureExtractor
from ..feature_extraction.rep_counter import RepCounter
from ..classification.base import ClassificationResult
from ..classification.rule_based import RuleBasedClassifier
from ..classification.ml_classifier import MLClassifier
from ..feedback.feedback_engine import FeedbackEngine
from ..visualization.overlay import draw_skeleton, draw_angle_zones, draw_feedback_panel, draw_exercise_label
from ..utils.temporal import TemporalSmoother
from ..utils.config import Config
from ..utils.constants import EXERCISES


class ExerVisionPipeline:
    """Full frame-by-frame exercise form evaluation pipeline.

    Supports three classifier modes: rule_based, ml, bilstm.
    Tracks per-rep classification results for session reports.
    """

    def __init__(
        self,
        exercise: str = "squat",
        classifier_type: str = "rule_based",
        smooth: bool = True,
        config: Config = None,
        preloaded_classifier=None,
    ):
        self.config = config or Config()
        self._exercise = exercise
        self._classifier_type = classifier_type

        # Pose estimation
        self._estimator = MediaPipePoseEstimator(self.config, mode="video")

        # Feature extraction
        self._feature_extractor = FeatureExtractor(use_world=True)

        # Classification (use preloaded if provided and type matches)
        if preloaded_classifier is not None:
            self._classifier = preloaded_classifier
        else:
            self._classifier = self._create_classifier(classifier_type)

        # Feedback
        self._feedback = FeedbackEngine()

        # Temporal smoothing
        self._smoother = TemporalSmoother(window_size=5) if smooth else None

        # Rep counting
        self._rep_counter = RepCounter(exercise)
        self._prev_rep_count = 0

        # Frame history for temporal features
        self._frame_history = deque(maxlen=30)

        # FPS tracking
        self._frame_times = deque(maxlen=30)
        self._frame_count = 0

        # Last result
        self._last_result = None

        # Session tracking
        self._session_active = False
        self._session_start_time = None
        self._rep_results: List[Dict] = []
        self._current_rep_scores: List[float] = []
        self._current_rep_issues: List[str] = []
        self._last_rep_completed = None  # set when a new rep is detected

    def _create_classifier(self, classifier_type: str):
        """Create classifier based on type string."""
        if classifier_type == "ml":
            clf = MLClassifier(self.config)
            clf.load_all_models()
            return clf
        elif classifier_type == "bilstm":
            try:
                from ..classification.bilstm_classifier import BiLSTMClassifier
                clf = BiLSTMClassifier(
                    models_dir=str(self.config.models_dir),
                )
                clf.load_all_models()
                return clf
            except ImportError:
                logger.warning("PyTorch not available, falling back to rule-based")
                return RuleBasedClassifier()
        else:
            return RuleBasedClassifier()

    @property
    def exercise(self) -> str:
        return self._exercise

    @exercise.setter
    def exercise(self, value: str):
        if value not in EXERCISES:
            raise ValueError(f"Unknown exercise: {value}")
        self._exercise = value
        self._rep_counter = RepCounter(value)
        self._prev_rep_count = 0
        self._frame_history.clear()
        if self._smoother:
            self._smoother.reset()
        self._last_result = None
        # Reset BiLSTM buffer if applicable
        if hasattr(self._classifier, "reset"):
            self._classifier.reset(value)

    @property
    def classifier_type(self) -> str:
        return self._classifier_type

    @property
    def rep_count(self) -> int:
        return self._rep_counter.get_rep_count()

    @property
    def fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        elapsed = self._frame_times[-1] - self._frame_times[0]
        if elapsed <= 0:
            return 0.0
        return len(self._frame_times) / elapsed

    def set_classifier(self, classifier_type: str):
        """Switch classifier type at runtime."""
        self._classifier_type = classifier_type
        self._classifier = self._create_classifier(classifier_type)

    def start_session(self):
        """Start a new exercise session for tracking."""
        self._session_active = True
        self._session_start_time = time.time()
        self._rep_results = []
        self._current_rep_scores = []
        self._current_rep_issues = []
        self._last_rep_completed = None
        self.reset()

    def end_session(self) -> Dict:
        """End session and return summary report."""
        self._session_active = False
        return self.get_session_summary()

    def get_session_summary(self) -> Dict:
        """Compile a session report with per-rep scores."""
        duration = 0.0
        if self._session_start_time:
            duration = time.time() - self._session_start_time

        total_reps = len(self._rep_results)
        good_reps = sum(1 for r in self._rep_results if r.get("form_score", 0) >= 0.7)

        # Count common issues across all reps
        issue_counts: Dict[str, int] = {}
        for rep in self._rep_results:
            for issue in rep.get("issues", []):
                key = issue.split(":")[0].strip()
                issue_counts[key] = issue_counts.get(key, 0) + 1

        common_issues = sorted(issue_counts.items(), key=lambda x: -x[1])[:5]

        return {
            "exercise": self._exercise,
            "classifier": self._classifier_type,
            "duration_sec": round(duration, 1),
            "total_reps": total_reps,
            "good_reps": good_reps,
            "avg_form_score": round(
                sum(r.get("form_score", 0) for r in self._rep_results) / max(total_reps, 1),
                2
            ),
            "reps": self._rep_results,
            "common_issues": [
                {"issue": name, "count": count}
                for name, count in common_issues
            ],
        }

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: int = None,
    ) -> Tuple[np.ndarray, Optional[ClassificationResult]]:
        """Process a single BGR video frame through the full pipeline.

        Returns:
            (annotated_frame, classification_result) -- result is None if no pose detected.
        """
        now = time.time()
        self._frame_times.append(now)
        self._frame_count += 1
        self._last_rep_completed = None

        if timestamp_ms is None:
            timestamp_ms = int(self._frame_count * (1000 / 30))

        # 1. Pose estimation
        pose = self._estimator.process_video_frame(frame, timestamp_ms)

        if pose is None:
            draw_exercise_label(frame, self._exercise)
            if self._last_result:
                draw_feedback_panel(frame, "No pose detected", is_correct=True, fps=self.fps)
            return frame, None

        # 2. Temporal smoothing
        if self._smoother:
            self._smoother.add(pose)
            smoothed = self._smoother.get_smoothed()
            if smoothed:
                pose = smoothed

        # 3. Update frame history and rep counter
        self._frame_history.append(pose)
        self._rep_counter.update(pose)

        # 4. Feature extraction (with temporal features for ML/BiLSTM)
        if self._classifier_type in ("ml", "bilstm"):
            features = self._feature_extractor.extract_with_temporal(
                pose, self._exercise, self._frame_history
            )
        else:
            features = self._feature_extractor.extract(pose, self._exercise)

        # 5. Classification
        result = self._classifier.classify(features, self._exercise)
        self._last_result = result

        # 6. Track per-rep results during session
        if self._session_active:
            score = (result.confidence or 0.0) if result.is_correct else 0.0
            self._current_rep_scores.append(score)
            if result.details:
                for issue in result.details.values():
                    if issue not in self._current_rep_issues:
                        self._current_rep_issues.append(issue)

            # Check if a new rep was completed
            current_count = self._rep_counter.get_rep_count()
            if current_count > self._prev_rep_count:
                # Rep completed -- save results
                form_score = (
                    sum(self._current_rep_scores) / max(len(self._current_rep_scores), 1)
                )
                self._last_rep_completed = {
                    "rep_num": current_count,
                    "form_score": round(form_score, 2),
                    "issues": self._current_rep_issues[:3],
                    "timestamp": round(time.time() - (self._session_start_time or now), 1),
                }
                self._rep_results.append(self._last_rep_completed)
                self._current_rep_scores = []
                self._current_rep_issues = []
                self._prev_rep_count = current_count

        # 7. Get visual feedback
        joint_colors = self._feedback.get_joint_colors(result)
        skeleton_colors = self._feedback.get_skeleton_colors(result)
        feedback_text = self._feedback.get_text_feedback(result)

        # 8. Draw overlay — mesh zones first (behind skeleton)
        draw_angle_zones(frame, pose, result.joint_feedback)
        draw_skeleton(frame, pose, joint_colors, skeleton_colors)
        draw_feedback_panel(
            frame, feedback_text,
            is_correct=result.is_correct,
            rep_count=self.rep_count,
            fps=self.fps,
        )
        draw_exercise_label(frame, self._exercise)

        return frame, result

    @property
    def last_rep_completed(self) -> Optional[Dict]:
        """Returns info about the last completed rep (None if no new rep this frame)."""
        return self._last_rep_completed

    def reset(self):
        """Reset all stateful components."""
        self._frame_history.clear()
        self._rep_counter.reset()
        self._prev_rep_count = 0
        if self._smoother:
            self._smoother.reset()
        self._last_result = None
        self._frame_times.clear()
        self._frame_count = 0
        if hasattr(self._classifier, "reset"):
            self._classifier.reset()

    def close(self):
        """Release resources."""
        self._estimator.close()
