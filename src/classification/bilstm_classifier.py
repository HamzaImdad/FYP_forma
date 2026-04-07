"""
BiLSTM-based form classifier using temporal pose sequences.

Architecture (original proven design):
  - Conv1D front-end (2 layers) for local temporal pattern extraction
  - 2-layer Bidirectional LSTM (128 hidden units)
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
    """CNN-BiLSTM with attention for exercise form classification.

    Architecture:
      1. Conv1D front-end extracts local temporal patterns
      2. BiLSTM captures long-range temporal dependencies
      3. Self-attention pools over timesteps
      4. Classification head outputs binary logit

    Input: (batch, seq_len, input_dim)
    Output: (batch,) -- logits for binary classification
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_conv: bool = True,
    ):
        super().__init__()

        self.use_conv = use_conv

        # Conv1D front-end for local temporal pattern extraction
        if use_conv:
            self.conv = nn.Sequential(
                nn.Conv1d(input_dim, input_dim * 2, kernel_size=3, padding=1),
                nn.BatchNorm1d(input_dim * 2),
                nn.ReLU(),
                nn.Conv1d(input_dim * 2, input_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(input_dim),
                nn.ReLU(),
            )

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
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # Conv1D front-end: expects (batch, channels, seq_len)
        if self.use_conv:
            x_conv = x.permute(0, 2, 1)
            x_conv = self.conv(x_conv)
            x = x_conv.permute(0, 2, 1)

        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

        # Attention-weighted pooling
        attn_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden*2)

        # Classify
        logits = self.classifier(context).squeeze(-1)
        return logits

    def get_attention_weights(self, x: "torch.Tensor") -> "torch.Tensor":
        with torch.no_grad():
            if self.use_conv:
                x_conv = x.permute(0, 2, 1)
                x_conv = self.conv(x_conv)
                x = x_conv.permute(0, 2, 1)
            lstm_out, _ = self.lstm(x)
            attn_scores = self.attention(lstm_out)
            attn_weights = torch.softmax(attn_scores, dim=1)
        return attn_weights.squeeze(-1)


class BiLSTMClassifier(FormClassifier):
    """Pipeline wrapper for BiLSTM classifier."""

    def __init__(
        self,
        models_dir: str = None,
        seq_len: int = 30,
        device: str = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for BiLSTM classifier.")

        self.seq_len = seq_len
        self._models_dir = Path(models_dir) if models_dir else Path("models/trained")
        self._feature_extractor = FeatureExtractor(use_world=True)
        self._fallback = RuleBasedClassifier()
        self._models: Dict[str, FormBiLSTM] = {}
        self._onnx_sessions: Dict[str, object] = {}  # exercise -> ort.InferenceSession
        self._thresholds: Dict[str, float] = {}
        self._buffers: Dict[str, deque] = {}
        self._feature_names: Dict[str, List[str]] = {}
        self._seq_lens: Dict[str, int] = {}

        # Inference throttling — run BiLSTM every Nth frame (lower = more responsive)
        self._infer_every = 2
        self._frame_counter: Dict[str, int] = {}
        self._last_bilstm_result: Dict[str, ClassificationResult] = {}

        if device:
            self._device = torch.device(device)
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

    def load_model(self, exercise: str) -> bool:
        model_path = self._models_dir / f"{exercise}_bilstm_v2.pt"
        if not model_path.exists():
            model_path = self._models_dir / f"{exercise}_bilstm.pt"
        if not model_path.exists():
            return False

        try:
            checkpoint = torch.load(model_path, map_location=self._device, weights_only=False)
        except Exception as e:
            logger.warning("Failed to load BiLSTM model for %s: %s", exercise, e)
            return False

        use_conv = checkpoint.get("use_conv", False)
        hidden_dim = checkpoint.get("hidden_dim", 128)
        num_layers = checkpoint.get("num_layers", 2)
        input_dim = checkpoint.get("input_dim", 15)

        model = FormBiLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.0,
            use_conv=use_conv,
        )

        state_dict = checkpoint["model_state_dict"]
        try:
            model.load_state_dict(state_dict, strict=False)
        except RuntimeError:
            cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(cleaned, strict=False)

        model.to(self._device)
        model.eval()

        self._models[exercise] = model

        # Load per-exercise threshold from training (not hardcoded 0.5)
        self._thresholds[exercise] = checkpoint.get("best_threshold",
                                                     checkpoint.get("val_best_thresh_info", 0.5))

        # Per-exercise seq_len from checkpoint
        actual_seq_len = checkpoint.get("seq_len", self.seq_len)
        self._seq_lens[exercise] = actual_seq_len
        self._buffers[exercise] = deque(maxlen=actual_seq_len)
        self._feature_names[exercise] = list(checkpoint.get("feature_names", []))

        # Try loading ONNX model for faster inference
        onnx_path = model_path.with_name(f"{exercise}_bilstm_v2.onnx")
        if onnx_path.exists():
            try:
                import onnxruntime as ort
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 20
                sess_options.inter_op_num_threads = 4
                providers = []
                if 'CUDAExecutionProvider' in ort.get_available_providers():
                    providers.append('CUDAExecutionProvider')
                providers.append('CPUExecutionProvider')
                self._onnx_sessions[exercise] = ort.InferenceSession(
                    str(onnx_path), sess_options, providers=providers,
                )
                logger.info("Loaded ONNX model for %s (%s)", exercise, providers[0])
            except Exception as e:
                logger.debug("ONNX load failed for %s, using PyTorch: %s", exercise, e)

        logger.info(
            "Loaded BiLSTM model for %s (threshold=%.2f, input_dim=%d, "
            "hidden=%d, layers=%d, seq_len=%d, features=%d%s)",
            exercise, self._thresholds[exercise],
            input_dim, hidden_dim, num_layers, actual_seq_len,
            len(self._feature_names[exercise]),
            ", ONNX" if exercise in self._onnx_sessions else "",
        )
        return True

    def load_all_models(self):
        for exercise in EXERCISES:
            self.load_model(exercise)

    def classify(
        self,
        features: Dict[str, Optional[float]],
        exercise: str,
    ) -> ClassificationResult:
        rb_result = self._fallback.classify(features, exercise)

        if exercise not in self._models:
            if not self.load_model(exercise):
                return rb_result

        model_feature_names = self._feature_names.get(exercise, [])
        if model_feature_names:
            vec = [features.get(fn, 0.0) or 0.0 for fn in model_feature_names]
        else:
            old_names = self._feature_extractor.get_all_feature_names(exercise)
            vec = [features.get(fn, 0.0) or 0.0 for fn in old_names]

        actual_seq_len = self._seq_lens.get(exercise, self.seq_len)
        if exercise not in self._buffers or self._buffers[exercise].maxlen != actual_seq_len:
            self._buffers[exercise] = deque(maxlen=actual_seq_len)
        self._buffers[exercise].append(vec)

        if len(self._buffers[exercise]) < actual_seq_len:
            return rb_result

        self._frame_counter[exercise] = self._frame_counter.get(exercise, 0) + 1
        if self._frame_counter[exercise] % self._infer_every != 1:
            cached = self._last_bilstm_result.get(exercise)
            if cached is not None:
                return ClassificationResult(
                    exercise=exercise,
                    is_correct=cached.is_correct,
                    confidence=cached.confidence,
                    joint_feedback=rb_result.joint_feedback,
                    details=cached.details,
                    is_active=rb_result.is_active,
                    form_score=cached.form_score,
                )
            return rb_result

        seq = np.array(list(self._buffers[exercise]), dtype=np.float32)

        # Use ONNX Runtime if available (20-50% faster), else fall back to PyTorch
        if exercise in self._onnx_sessions:
            onnx_out = self._onnx_sessions[exercise].run(
                None, {"input": seq[np.newaxis]}
            )
            logit_val = float(onnx_out[0].item())
            prob = 1.0 / (1.0 + np.exp(-logit_val))
        else:
            model = self._models[exercise]
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self._device)
            with torch.no_grad():
                logit = model(x)
                prob = torch.sigmoid(logit).item()

        threshold = self._thresholds.get(exercise, 0.5)
        is_correct = prob >= threshold

        bilstm_score = prob
        if rb_result.is_active and rb_result.form_score > 0:
            form_score = 0.7 * bilstm_score + 0.3 * rb_result.form_score
        else:
            form_score = bilstm_score

        result = ClassificationResult(
            exercise=exercise,
            is_correct=is_correct,
            confidence=max(prob, 1.0 - prob),
            joint_feedback=rb_result.joint_feedback,
            details={
                **rb_result.details,
                "model": "BiLSTM",
                "bilstm_confidence": f"{prob:.2f}",
            },
            is_active=rb_result.is_active,
            form_score=max(0.0, min(1.0, form_score)),
        )
        self._last_bilstm_result[exercise] = result
        return result

    def reset(self, exercise: str = None):
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
        return list(self._models.keys())
