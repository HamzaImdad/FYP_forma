"""
ML-based form classifier using pre-trained sklearn/XGBoost models.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .base import FormClassifier, ClassificationResult
from .rule_based import RuleBasedClassifier
from ..feature_extraction.features import FeatureExtractor
from ..feature_extraction.exercise_features import EXERCISE_FEATURES
from ..utils.config import Config


logger = logging.getLogger(__name__)


class MLClassifier(FormClassifier):
    """Loads and runs pre-trained ML models for each exercise."""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self._models = {}  # exercise -> trained model
        self._feature_extractor = FeatureExtractor()
        self._rule_classifier = RuleBasedClassifier()

    def load_model(self, exercise: str, model_path: str = None):
        """Load a trained model for a specific exercise."""
        if model_path is None:
            model_path = self.config.models_dir / f"{exercise}_classifier.pkl"
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            if not hasattr(model, "predict"):
                raise ValueError(f"Loaded object for '{exercise}' is not a valid ML model")
            self._models[exercise] = model
            logger.info("Loaded ML model for %s", exercise)
        except Exception as e:
            logger.warning("Failed to load ML model for %s: %s", exercise, e)

    def load_all_models(self):
        """Load all available trained models from models/trained/."""
        if not self.config.models_dir.exists():
            return
        for exercise in EXERCISE_FEATURES:
            model_path = self.config.models_dir / f"{exercise}_classifier.pkl"
            if model_path.exists():
                self.load_model(exercise, str(model_path))

    def classify(self, features: Dict[str, Optional[float]], exercise: str) -> ClassificationResult:
        if exercise not in self._models:
            logger.warning("No trained ML model for '%s', returning neutral result", exercise)
            return ClassificationResult(
                exercise=exercise,
                is_correct=False,
                confidence=0.0,
                joint_feedback={},
                details={"model": "none", "error": f"No trained model for {exercise}"},
            )

        model = self._models[exercise]
        feature_names = self._feature_extractor.get_feature_names(exercise)
        X = np.array([[features.get(n, 0.0) or 0.0 for n in feature_names]])

        prediction = model.predict(X)[0]
        is_correct = bool(prediction)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            confidence = float(proba[1]) if len(proba) > 1 else float(proba[0])
        else:
            confidence = 1.0 if is_correct else 0.0

        # Use rule-based classifier for joint-level feedback
        rule_result = self._rule_classifier.classify(features, exercise)
        joint_feedback = rule_result.joint_feedback if rule_result else {}

        # Blend ML confidence with rule-based form score
        rule_score = rule_result.form_score if rule_result else 0.0
        form_score = 0.7 * confidence + 0.3 * rule_score

        # Filter out internal details from user-facing feedback
        user_details = {k: v for k, v in (rule_result.details or {}).items()
                        if k not in ("model", "confidence", "threshold")}

        return ClassificationResult(
            exercise=exercise,
            is_correct=is_correct,
            confidence=confidence,
            joint_feedback=joint_feedback,
            details=user_details,
            form_score=form_score,
            is_active=rule_result.is_active if rule_result else True,
        )

    def get_supported_exercises(self) -> List[str]:
        return list(self._models.keys())
