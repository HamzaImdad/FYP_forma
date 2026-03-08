"""
BiLSTM-based form classifier using temporal pose sequences.

Architecture:
  - 2-layer Bidirectional LSTM (64 hidden units)
  - Self-attention pooling over timesteps
  - Dropout regularization (0.3)
  - Binary classification head

The BiLSTM provides the overall correct/incorrect judgment using temporal
context (full sequence of frames), while per-joint color feedback is
delegated to the rule-based classifier for interpretability.
"""

import logging
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

from .base import FormClassifier, ClassificationResult
from .rule_based import RuleBasedClassifier
from ..feature_extraction.features import FeatureExtractor
from ..utils.constants import EXERCISES


class FormBiLSTM(nn.Module):
    """Bidirectional LSTM with attention for exercise form classification.

    Input: (batch, seq_len, input_dim)
    Output: (batch,) -- logits for binary classification
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Self-attention over timesteps: learns which frames matter most
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            logits: (batch,) -- raw logits (use sigmoid for probabilities)
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

        # Attention-weighted pooling
        attn_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden*2)

        # Classify
        logits = self.classifier(context).squeeze(-1)  # (batch,)
        return logits

    def get_attention_weights(self, x: "torch.Tensor") -> "torch.Tensor":
        """Get attention weights for visualization (which frames mattered)."""
        with torch.no_grad():
            lstm_out, _ = self.lstm(x)
            attn_scores = self.attention(lstm_out)
            attn_weights = torch.softmax(attn_scores, dim=1)
        return attn_weights.squeeze(-1)  # (batch, seq_len)


class BiLSTMClassifier(FormClassifier):
    """Pipeline wrapper for BiLSTM classifier.

    Maintains an internal frame buffer. Once enough frames accumulate,
    uses the BiLSTM for classification. Falls back to rule-based
    classifier until the buffer is full.

    The BiLSTM decides overall correct/incorrect with temporal context.
    The rule-based classifier provides per-joint color feedback.
    """

    def __init__(
        self,
        models_dir: str = None,
        seq_len: int = 30,
        device: str = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for BiLSTM classifier. "
                "Install with: pip install torch"
            )

        self.seq_len = seq_len
        self._models_dir = Path(models_dir) if models_dir else Path("models/trained")
        self._feature_extractor = FeatureExtractor(use_world=True)
        self._fallback = RuleBasedClassifier()
        self._models: Dict[str, FormBiLSTM] = {}
        self._thresholds: Dict[str, float] = {}
        self._buffers: Dict[str, deque] = {}

        # Inference throttling: only run BiLSTM every N frames
        self._infer_every = 5  # run inference every 5th frame
        self._frame_counter: Dict[str, int] = {}
        self._last_bilstm_result: Dict[str, ClassificationResult] = {}

        # Device selection
        if device:
            self._device = torch.device(device)
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

    def load_model(self, exercise: str) -> bool:
        """Load a trained BiLSTM model for a specific exercise."""
        model_path = self._models_dir / f"{exercise}_bilstm.pt"
        if not model_path.exists():
            return False

        try:
            checkpoint = torch.load(model_path, map_location=self._device, weights_only=False)
        except Exception as e:
            logger.warning("Failed to load BiLSTM model for %s: %s", exercise, e)
            return False

        model = FormBiLSTM(
            input_dim=checkpoint.get("input_dim", 15),
            hidden_dim=checkpoint.get("hidden_dim", 64),
            num_layers=checkpoint.get("num_layers", 2),
            dropout=0.0,  # No dropout at inference
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self._device)
        model.eval()

        self._models[exercise] = model
        self._thresholds[exercise] = checkpoint.get("best_threshold", 0.5)
        self._buffers[exercise] = deque(maxlen=self.seq_len)
        logger.info("Loaded BiLSTM model for %s (threshold=%.2f)", exercise, self._thresholds[exercise])
        return True

    def load_all_models(self):
        """Load BiLSTM models for all available exercises."""
        for exercise in EXERCISES:
            self.load_model(exercise)

    def classify(
        self,
        features: Dict[str, Optional[float]],
        exercise: str,
    ) -> ClassificationResult:
        """Classify form using BiLSTM on accumulated frame buffer.

        If the buffer isn't full yet, falls back to rule-based classifier.
        Always uses rule-based for per-joint color feedback.
        """
        # Always get rule-based result for per-joint feedback
        rb_result = self._fallback.classify(features, exercise)

        # Check if we have a trained model for this exercise
        if exercise not in self._models:
            if not self.load_model(exercise):
                return rb_result  # No model available, use rule-based

        # Get feature names and build vector
        feature_names = self._feature_extractor.get_all_feature_names(exercise)
        vec = [features.get(fn, 0.0) or 0.0 for fn in feature_names]

        # Add to buffer
        if exercise not in self._buffers:
            self._buffers[exercise] = deque(maxlen=self.seq_len)
        self._buffers[exercise].append(vec)

        # If buffer not full, use rule-based
        if len(self._buffers[exercise]) < self.seq_len:
            return rb_result

        # Throttle: only run BiLSTM every N frames, reuse last result otherwise
        self._frame_counter[exercise] = self._frame_counter.get(exercise, 0) + 1
        if self._frame_counter[exercise] % self._infer_every != 1:
            cached = self._last_bilstm_result.get(exercise)
            if cached is not None:
                # Update joint feedback from fresh rule-based result
                return ClassificationResult(
                    exercise=exercise,
                    is_correct=cached.is_correct,
                    confidence=cached.confidence,
                    joint_feedback=rb_result.joint_feedback,
                    details=cached.details,
                )
            return rb_result

        # Run BiLSTM inference
        model = self._models[exercise]
        seq = np.array(list(self._buffers[exercise]), dtype=np.float32)
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logit = model(x)
            prob = torch.sigmoid(logit).item()

        threshold = self._thresholds.get(exercise, 0.5)
        is_correct = prob >= threshold
        confidence = prob if is_correct else (1.0 - prob)

        result = ClassificationResult(
            exercise=exercise,
            is_correct=is_correct,
            confidence=confidence,
            joint_feedback=rb_result.joint_feedback,
            details={
                **rb_result.details,
                "model": "BiLSTM",
                "bilstm_confidence": f"{prob:.2f}",
            },
        )
        self._last_bilstm_result[exercise] = result
        return result

    def reset(self, exercise: str = None):
        """Clear frame buffer(s) and cached results."""
        if exercise:
            if exercise in self._buffers:
                self._buffers[exercise].clear()
            self._frame_counter.pop(exercise, None)
            self._last_bilstm_result.pop(exercise, None)
        else:
            for buf in self._buffers.values():
                buf.clear()
            self._frame_counter.clear()
            self._last_bilstm_result.clear()

    def get_supported_exercises(self) -> List[str]:
        """Return exercises with loaded BiLSTM models."""
        return list(self._models.keys())
