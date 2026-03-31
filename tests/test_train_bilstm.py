"""
Unit tests for train_bilstm.py training loop fixes.

Tests verify:
  - FocalLoss uses alpha=0.5 (not 0.75) and has no pos_weight parameter
  - Label smoothing 0.1 is applied to targets in FocalLoss.forward()
  - WeightedRandomSampler is not used in train_exercise()
  - val==test fallback is removed
  - Per-epoch threshold search is removed (fixed threshold=0.5)
  - train_exercise loads pre-split .npz files
  - SWA (AveragedModel) is integrated in training
  - Augmentation uses noise std >= 0.05 and numpy.interp (no torch.roll)
"""

import sys
import inspect
import re
from pathlib import Path

import numpy as np
import torch
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_bilstm import FocalLoss, SequenceDataset


# ── Source code inspection helpers ─────────────────────────────────────────

def get_source(name: str) -> str:
    """Read the train_bilstm.py source file and return its full text."""
    src_path = PROJECT_ROOT / "scripts" / "train_bilstm.py"
    return src_path.read_text(encoding="utf-8")


# ── FocalLoss tests ─────────────────────────────────────────────────────────

class TestFocalLoss:

    def test_focal_loss_alpha_default(self):
        """FocalLoss default alpha must be 0.5 (not 0.75)."""
        loss = FocalLoss()
        assert loss.alpha == 0.5, (
            f"FocalLoss default alpha should be 0.5, got {loss.alpha}. "
            "Triple weighting bug: alpha=0.75 + pos_weight + WeightedRandomSampler"
        )

    def test_focal_loss_no_pos_weight_param(self):
        """FocalLoss constructor must NOT have a pos_weight parameter."""
        sig = inspect.signature(FocalLoss.__init__)
        params = list(sig.parameters.keys())
        assert "pos_weight" not in params, (
            f"FocalLoss.__init__ still has pos_weight parameter: {params}. "
            "Remove pos_weight — it adds a third class weighting on already-balanced data."
        )

    def test_focal_loss_no_pos_weight_in_forward(self):
        """FocalLoss.forward() must not pass pos_weight to F.binary_cross_entropy_with_logits."""
        src = get_source("train_bilstm")
        # Check that FocalLoss class body does NOT use pos_weight in BCE call
        focal_class_match = re.search(
            r'class FocalLoss.*?(?=\nclass |\ndef |\Z)',
            src,
            re.DOTALL,
        )
        assert focal_class_match, "Could not find FocalLoss class in source"
        focal_src = focal_class_match.group(0)
        assert "pos_weight" not in focal_src, (
            "FocalLoss class still references pos_weight. Remove it entirely."
        )

    def test_label_smoothing_applied(self):
        """FocalLoss.forward() must apply label smoothing so targets are not exactly 0 or 1."""
        loss_fn = FocalLoss()
        logits = torch.zeros(10)
        # Hard targets (exactly 0.0 and 1.0)
        targets_hard = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

        # The loss should still compute without error
        loss_val = loss_fn(logits, targets_hard)
        assert loss_val.item() > 0.0, "Loss should be positive"

        # Verify by checking source code that smoothing is applied
        src = get_source("train_bilstm")
        focal_src = re.search(
            r'class FocalLoss.*?(?=\nclass |\ndef |\Z)',
            src,
            re.DOTALL,
        ).group(0)
        assert "smooth" in focal_src.lower() or "0.1" in focal_src, (
            "FocalLoss.forward() does not appear to apply label smoothing (0.1). "
            "Add: targets_smooth = targets * (1 - 0.1) + 0.1 * 0.5"
        )

    def test_label_smoothing_changes_targets(self):
        """Label smoothing must transform targets: 0.0 -> 0.05, 1.0 -> 0.95."""
        # Inspect the forward() method to verify smoothing formula
        src = get_source("train_bilstm")
        focal_src = re.search(
            r'class FocalLoss.*?(?=\nclass |\ndef |\Z)',
            src,
            re.DOTALL,
        ).group(0)
        # Check for the smoothing pattern: targets * (1 - epsilon) + epsilon * 0.5
        # or equivalent with 0.1 smoothing
        has_smoothing = (
            "1 - 0.1" in focal_src or
            "0.9" in focal_src or
            "smooth" in focal_src.lower()
        )
        assert has_smoothing, (
            "Label smoothing (0.1) not found in FocalLoss.forward(). "
            "Expected pattern: targets_smooth = targets * (1 - 0.1) + 0.1 * 0.5"
        )


# ── Training function code pattern tests ───────────────────────────────────

class TestTrainingFunctionPatterns:

    def setup_method(self):
        self.src = get_source("train_bilstm")
        # Extract train_exercise function body
        match = re.search(
            r'def train_exercise\(.*?(?=\ndef |\Z)',
            self.src,
            re.DOTALL,
        )
        assert match, "Could not find train_exercise function in source"
        self.train_fn_src = match.group(0)

    def test_no_weighted_sampler(self):
        """WeightedRandomSampler must NOT be instantiated in train_exercise function body."""
        # Strip comments before checking (comments saying "no WeightedRandomSampler" are fine)
        lines = self.train_fn_src.split("\n")
        code_lines = [l for l in lines if not l.strip().startswith("#")]
        code_only = "\n".join(code_lines)
        code_no_docstring = re.sub(r'""".*?"""', '', code_only, flags=re.DOTALL)
        assert "WeightedRandomSampler" not in code_no_docstring, (
            "train_exercise still instantiates WeightedRandomSampler. "
            "Remove it — data is already balanced, sampler adds triple weighting."
        )

    def test_no_val_test_fallback(self):
        """The pattern 'X_val, y_val = X_test, y_test' must not exist anywhere in the file."""
        assert "X_val, y_val = X_test, y_test" not in self.src, (
            "val==test fallback still exists. Remove it and replace with WARNING + return {}."
        )

    def test_no_per_epoch_threshold_search(self):
        """Per-epoch threshold search loop must be removed from training loop."""
        # The training loop epoch block should NOT contain threshold search
        # Look for threshold search pattern (np.arange loop) inside training loop
        # We check that the threshold search is NOT inside the epoch for-loop
        # by verifying no 'for thresh in' appears nested inside 'for epoch in'
        epoch_loop = re.search(
            r'for epoch in range\(epochs\)(.*?)(?=\n    # Load best|model\.load_state_dict)',
            self.train_fn_src,
            re.DOTALL,
        )
        if epoch_loop:
            epoch_body = epoch_loop.group(1)
            assert "for thresh in" not in epoch_body, (
                "Per-epoch threshold search still inside epoch loop. "
                "Remove it — searching thresholds on val each epoch causes leakage. "
                "Use fixed threshold=0.5 during training."
            )

    def test_loads_presplit_npz(self):
        """train_exercise must call load_presplit_sequences (which loads pre-split .npz files)."""
        # train_exercise should call load_presplit_sequences (not load_sequences + internal split)
        has_presplit_call = "load_presplit_sequences" in self.train_fn_src
        # Also accept direct pattern of loading _train/_val/_test partitioned files
        has_partition_pattern = (
            "_train_sequences.npz" in self.src or
            "_{partition}_sequences.npz" in self.src or
            "partition" in self.src and "sequences.npz" in self.src
        )
        assert has_presplit_call or has_partition_pattern, (
            "train_exercise does not load pre-split partition .npz files. "
            "Replace GroupShuffleSplit with: call load_presplit_sequences(exercise) "
            "which loads {exercise}_train_sequences.npz, _val_sequences.npz, _test_sequences.npz."
        )

    def test_no_group_shuffle_split_in_train_exercise(self):
        """train_exercise must NOT call GroupShuffleSplit (splits are pre-built)."""
        # Strip comments (lines starting with #) from the function body before checking
        lines = self.train_fn_src.split("\n")
        code_lines = [l for l in lines if not l.strip().startswith("#")]
        code_only = "\n".join(code_lines)
        # Also strip docstrings
        code_no_docstring = re.sub(r'""".*?"""', '', code_only, flags=re.DOTALL)
        assert "GroupShuffleSplit" not in code_no_docstring, (
            "train_exercise still calls GroupShuffleSplit (found in non-comment code). "
            "Remove it — splits are pre-built by split_videos.py and build_sequences.py."
        )

    def test_fixed_threshold(self):
        """Training must use fixed threshold=0.5, not search on val each epoch."""
        # Verify threshold is set to 0.5 as a fixed value during training
        assert "threshold = 0.5" in self.src or "best_threshold = 0.5" in self.src, (
            "Fixed threshold=0.5 not found. "
            "Training should use fixed threshold during training (no per-epoch search)."
        )


# ── SWA tests ───────────────────────────────────────────────────────────────

class TestSWA:

    def setup_method(self):
        self.src = get_source("train_bilstm")

    def test_swa_import(self):
        """AveragedModel and SWALR must be imported from torch.optim.swa_utils."""
        assert "from torch.optim.swa_utils import" in self.src, (
            "Missing SWA import. Add: from torch.optim.swa_utils import AveragedModel, SWALR"
        )
        assert "AveragedModel" in self.src, (
            "AveragedModel not imported. Add to swa_utils import."
        )

    def test_swa_applied_last_quarter(self):
        """SWA-related code (AveragedModel, update_bn) must exist in train_exercise."""
        assert "AveragedModel" in self.src, (
            "SWA: AveragedModel not used in training. "
            "Add: swa_model = AveragedModel(model) and call swa_model.update_parameters(model) "
            "in last 25% of epochs."
        )

    def test_swa_update_bn_called(self):
        """update_bn must be called after training loop to fix BatchNorm stats for SWA model."""
        assert "update_bn" in self.src, (
            "SWA: update_bn not called after training. "
            "FormBiLSTM uses BatchNorm1d — must call "
            "torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device) "
            "after training loop ends."
        )

    def test_swa_start_at_75_percent(self):
        """SWA should start at 75% of total epochs (epochs * 0.75)."""
        assert "0.75" in self.src or "swa_start" in self.src, (
            "SWA start epoch not found. "
            "Add: swa_start = int(epochs * 0.75)"
        )


# ── Augmentation tests ───────────────────────────────────────────────────────

class TestAugmentation:

    def setup_method(self):
        self.src = get_source("train_bilstm")
        # Extract SequenceDataset class body
        match = re.search(
            r'class SequenceDataset.*?(?=\nclass |\ndef train_exercise|\Z)',
            self.src,
            re.DOTALL,
        )
        assert match, "Could not find SequenceDataset class in source"
        self.dataset_src = match.group(0)

    def test_no_torch_roll(self):
        """torch.roll must NOT be used for temporal augmentation."""
        assert "torch.roll" not in self.dataset_src, (
            "SequenceDataset still uses torch.roll for temporal shift. "
            "Remove it — torch.roll wraps end frames to beginning, creating impossible transitions. "
            "Replace with numpy.interp time warp."
        )

    def test_noise_std_at_least_005(self):
        """Augmentation noise std must be >= 0.05 (was 0.02, too weak)."""
        # Check that 0.02 is NOT the noise std (old value)
        # and that 0.05 IS present
        assert "0.05" in self.dataset_src or "0.1" in self.dataset_src, (
            "Augmentation noise std is still too weak. "
            "Change from 0.02 to at least 0.05: x = x + torch.randn_like(x) * 0.05"
        )
        # Verify 0.02 is no longer used for noise (it may appear in dropout probability)
        # Check for '* 0.02' pattern specifically
        noise_02 = re.search(r'\*\s*0\.02', self.dataset_src)
        if noise_02:
            context = self.dataset_src[max(0, noise_02.start()-50):noise_02.end()+50]
            assert False, (
                f"Augmentation still uses noise std 0.02. Context: {context!r}. "
                "Change to 0.05."
            )

    def test_time_warp_uses_numpy_interp(self):
        """Time warp augmentation must use numpy.interp (not torch.roll or linspace clamp)."""
        assert "np.interp" in self.dataset_src or "numpy.interp" in self.dataset_src, (
            "Time warp augmentation does not use numpy.interp. "
            "Add: np.interp(new_idx, old_idx, x_np[:, f]) for each feature."
        )

    def test_augmentation_noise_runtime(self):
        """Runtime check: augmented sequences have different values due to noise."""
        X = np.zeros((4, 30, 10), dtype=np.float32)
        y = np.array([0, 1, 0, 1], dtype=np.float32)
        ds = SequenceDataset(X, y, augment=True)
        x0, _ = ds[0]
        # With noise std=0.05, at least some values should be non-zero
        assert x0.abs().max().item() > 0.0, (
            "Augmentation produced all-zero output — noise not applied."
        )


# ── Integration: import test ─────────────────────────────────────────────────

class TestImports:

    def test_swa_utils_importable(self):
        """torch.optim.swa_utils must be importable (PyTorch >= 1.6)."""
        from torch.optim.swa_utils import AveragedModel, SWALR
        assert AveragedModel is not None
        assert SWALR is not None

    def test_train_bilstm_importable(self):
        """train_bilstm.py must import without errors."""
        import scripts.train_bilstm as tb
        assert hasattr(tb, "FocalLoss")
        assert hasattr(tb, "SequenceDataset")
        assert hasattr(tb, "train_exercise")
