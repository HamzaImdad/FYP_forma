"""
Train BiLSTM models for exercise form classification.

Uses PyTorch with GPU acceleration. Trains one model per exercise on
sequence data (sliding windows of biomechanical features).

v2 improvements:
  - CNN-BiLSTM architecture (Conv1D front-end + BiLSTM)
  - Learning rate warmup + cosine annealing with warm restarts
  - Threshold search on validation set (no data leakage)
  - Early stopping on val_f1 (not val_loss)
  - Support for landmark-based features (99-dim input)
  - Training diagnostics (confusion matrix, degenerate model warnings)
  - Per-exercise architecture configs (hidden_dim, num_layers, seq_len)
  - Mixed precision training (AMP) on CUDA
  - Mixup augmentation
  - Class-weighted focal loss

Usage:
    python scripts/train_bilstm.py
    python scripts/train_bilstm.py --exercise squat
    python scripts/train_bilstm.py --exercise bench_press --use-landmarks --epochs 500
    python scripts/train_bilstm.py --data-source balanced
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix,
)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.bilstm_classifier import FormBiLSTM
from src.utils.constants import EXERCISES

SEQUENCES_DIR = PROJECT_ROOT / "data" / "processed" / "sequences"
MODELS_DIR = PROJECT_ROOT / "models" / "trained"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Per-exercise architecture configs — CLI args override these
# Reverted to simple architecture (hidden=128, layers=2) — complex models
# (256/3-layer) failed in Attempt 1 (0.463 avg F1, regressed from 0.621 baseline).
# Simple BiLSTM is optimal for this data scale (~2K-7K sequences per exercise).
EXERCISE_CONFIGS = {
    "deadlift":       {"hidden_dim": 128, "num_layers": 2, "seq_len": 45},
    "squat":          {"hidden_dim": 128, "num_layers": 2, "seq_len": 45},
    "bench_press":    {"hidden_dim": 128, "num_layers": 2, "seq_len": 30},
    "pushup":         {"hidden_dim": 128, "num_layers": 2, "seq_len": 30},
    "pullup":         {"hidden_dim": 128, "num_layers": 2, "seq_len": 30},
    "plank":          {"hidden_dim": 128, "num_layers": 2, "seq_len": 30},
    "lunge":          {"hidden_dim": 128, "num_layers": 2, "seq_len": 30},
    "bicep_curl":     {"hidden_dim": 128, "num_layers": 2, "seq_len": 30},
    "overhead_press": {"hidden_dim": 128, "num_layers": 2, "seq_len": 30},
    "tricep_dip":     {"hidden_dim": 128, "num_layers": 2, "seq_len": 30},
}


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced binary classification.

    Down-weights easy examples and focuses training on hard ones.
    Supports dynamic class-weighted alpha based on class frequencies.
    Label smoothing 0.1 prevents overconfident memorisation.
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def set_class_weights(self, n_positive: int, n_negative: int):
        """Set alpha based on class frequencies (higher weight for minority class).

        alpha = n_negative / (n_positive + n_negative) gives higher weight
        to the positive class when it is the minority.
        """
        total = n_positive + n_negative
        if total > 0:
            self.alpha = n_negative / total
        print(f"    FocalLoss alpha set to {self.alpha:.3f} "
              f"(pos={n_positive}, neg={n_negative})")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Label smoothing: 0 -> 0.05, 1 -> 0.95 (epsilon=0.1, num_classes=2)
        targets_smooth = targets * (1 - 0.1) + 0.1 * 0.5
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets_smooth,
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_weight * bce_loss
        return loss.mean()


class SequenceDataset(Dataset):
    """PyTorch Dataset for exercise form sequences with optional augmentation."""

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()  # shape: (seq_len, 99) — 33 landmarks x 3 coords (x,y,z)
        if self.augment:
            # 1. Gaussian noise — std=0.05
            x = x + torch.randn_like(x) * 0.05

            # 2. Random scaling — simulate different body sizes / camera distances
            #    Scale all coordinates uniformly (preserves form, changes apparent size)
            if torch.rand(1).item() < 0.5:
                scale = 0.8 + torch.rand(1).item() * 0.4  # 0.8-1.2
                x = x * scale

            # 3. Joint dropout — zero out entire landmarks (3 coords each)
            #    Simulates occlusion (arm behind body, leg cut off by camera)
            if torch.rand(1).item() < 0.2:
                n_landmarks = 33
                n_drop = int(torch.randint(1, 3, (1,)).item())  # drop 1-2 landmarks
                drop_ids = torch.randperm(n_landmarks)[:n_drop]
                for lid in drop_ids:
                    x[:, lid*3:(lid+1)*3] = 0.0

            # 4. Random translation — shift all x,y coords (simulate camera position)
            if torch.rand(1).item() < 0.4:
                # Shift x coords (indices 0,3,6,...,96)
                dx = (torch.rand(1).item() - 0.5) * 0.2  # -0.1 to +0.1
                dy = (torch.rand(1).item() - 0.5) * 0.2
                for i in range(33):
                    x[:, i*3] += dx      # x coord
                    x[:, i*3 + 1] += dy  # y coord

            # 5. Left-right mirror — negate x coordinates (hip-centered coords)
            #    Simulates facing opposite direction
            if torch.rand(1).item() < 0.3:
                for i in range(33):
                    x[:, i*3] = -x[:, i*3]  # negate x (coords are hip-centered, not 0-1)
                # Swap left/right landmark pairs (indices in groups of 3)
                # MediaPipe pairs: (11,12), (13,14), (15,16), (17,18), (19,20),
                # (21,22), (23,24), (25,26), (27,28), (29,30), (31,32)
                # plus eyes/ears: (1,4), (2,5), (3,6), (7,8), (9,10)
                swap_pairs = [(1,4),(2,5),(3,6),(7,8),(9,10),
                              (11,12),(13,14),(15,16),(17,18),(19,20),
                              (21,22),(23,24),(25,26),(27,28),(29,30),(31,32)]
                for a, b in swap_pairs:
                    tmp = x[:, a*3:(a+1)*3].clone()
                    x[:, a*3:(a+1)*3] = x[:, b*3:(b+1)*3]
                    x[:, b*3:(b+1)*3] = tmp

            # 6. Time warp — stretch/compress temporal axis
            if torch.rand(1).item() < 0.4:
                scale = 0.8 + torch.rand(1).item() * 0.4  # 0.8-1.2
                seq_len = x.shape[0]
                x_np = x.numpy()
                orig_idx = np.arange(seq_len)
                warped_idx = np.linspace(0, seq_len - 1, seq_len) / scale
                warped_idx = np.clip(warped_idx, 0, seq_len - 1)
                x_warped = np.stack([
                    np.interp(warped_idx, orig_idx, x_np[:, f])
                    for f in range(x_np.shape[1])
                ], axis=1)
                x = torch.tensor(x_warped, dtype=torch.float32)

            # 7. Temporal feature dropout — zero random timesteps entirely
            if torch.rand(1).item() < 0.2:
                n_drop = int(torch.randint(1, 4, (1,)).item())  # drop 1-3 frames
                drop_frames = torch.randperm(x.shape[0])[:n_drop]
                x[drop_frames] = 0.0

        return x, self.y[idx]


def load_sequences(exercise: str, use_landmarks: bool = False, data_source: str = "default") -> tuple:
    """Load sequence data for an exercise (legacy single-file loader).

    Args:
        exercise: Exercise name
        use_landmarks: If True, load landmark-based sequences (99-dim)
        data_source: "default" or "balanced"

    Returns: X, y, video_ids, feature_names
    """
    if use_landmarks and data_source == "balanced":
        filename = f"{exercise}_balanced_landmark_sequences.npz"
    elif use_landmarks:
        filename = f"{exercise}_landmark_sequences.npz"
    elif data_source == "balanced":
        filename = f"{exercise}_balanced_sequences.npz"
    else:
        filename = f"{exercise}_sequences.npz"

    path = SEQUENCES_DIR / filename
    if not path.exists():
        return None, None, None, None

    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    video_ids = data["video_ids"]
    feature_names = data["feature_names"]

    return X, y, video_ids, feature_names


def load_presplit_sequences(exercise: str, hybrid: bool = False) -> tuple:
    """Load pre-split train/val/test .npz files for an exercise.

    Files created by build_sequences.py using partition manifests from split_videos.py.
    Expected keys: X (n, 30, 99), y (n,), video_ids (n,), feature_names (99,)

    Returns: (X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
             All None if any file is missing.
    """
    suffix = "_hybrid" if hybrid else ""
    partitions = {}
    for partition in ("train", "val", "test"):
        path = SEQUENCES_DIR / f"{exercise}_{partition}{suffix}_sequences.npz"
        if not path.exists():
            print(f"  [WARNING] Missing pre-split file: {path.name}")
            return (None,) * 7
        data = np.load(path, allow_pickle=True)
        partitions[partition] = {
            "X": data["X"],
            "y": data["y"],
            "feature_names": data["feature_names"],
        }

    return (
        partitions["train"]["X"], partitions["train"]["y"],
        partitions["val"]["X"],   partitions["val"]["y"],
        partitions["test"]["X"],  partitions["test"]["y"],
        partitions["train"]["feature_names"],
    )


def find_best_threshold(model, dataloader, criterion, device):
    """Find the threshold that maximizes F1 on the given dataset.

    Returns: (best_threshold, best_f1, all_probs, all_labels)
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device, non_blocking=True)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())

    y_prob = np.array(all_probs)
    y_true = np.array(all_labels)

    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (y_prob >= thresh).astype(int)
        f1_val = f1_score(y_true, preds, zero_division=0)
        if f1_val > best_f1:
            best_f1 = f1_val
            best_thresh = thresh

    return best_thresh, best_f1, y_prob, y_true


def print_diagnostics(y_true, y_pred, y_prob, exercise, threshold):
    """Print confusion matrix and per-class diagnostics."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    print(f"\n  Confusion Matrix (threshold={threshold:.2f}):")
    print(f"                Predicted 0   Predicted 1")
    print(f"    Actual 0      {tn:>7}       {fp:>7}")
    print(f"    Actual 1      {fn:>7}       {tp:>7}")
    print(f"    TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    # Per-class accuracy
    if (tn + fp) > 0:
        print(f"    Class 0 (incorrect form) accuracy: {tn / (tn + fp):.3f}")
    if (tp + fn) > 0:
        print(f"    Class 1 (correct form) accuracy:   {tp / (tp + fn):.3f}")

    # Degenerate model warning
    total = len(y_pred)
    pred_0_pct = (y_pred == 0).sum() / total * 100
    pred_1_pct = (y_pred == 1).sum() / total * 100
    if pred_0_pct > 90 or pred_1_pct > 90:
        print(f"  WARNING: Degenerate model! Predicts class 0: {pred_0_pct:.1f}%, class 1: {pred_1_pct:.1f}%")

    return {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}


def train_exercise(
    exercise: str,
    epochs: int = 500,
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden_dim: Optional[int] = None,
    num_layers: Optional[int] = None,
    dropout: float = 0.3,
    patience: int = 50,
    warmup_epochs: int = 5,
    use_landmarks: bool = False,
    data_source: str = "default",
    device: torch.device = None,
    no_mixup: bool = False,
    no_swa: bool = False,
    use_bce: bool = False,
    hybrid: bool = False,
) -> dict:
    """Train BiLSTM for one exercise. Returns metrics dict.

    Loads pre-split {exercise}_{train|val|test}_sequences.npz files created by
    build_sequences.py using partition manifests from split_videos.py.
    No internal GroupShuffleSplit — splits are done once, before this script runs.
    """

    # Merge per-exercise config with CLI args (CLI overrides config)
    ex_config = EXERCISE_CONFIGS.get(exercise, {})
    if hidden_dim is None:
        hidden_dim = ex_config.get("hidden_dim", 128)
    if num_layers is None:
        num_layers = ex_config.get("num_layers", 2)
    seq_len = ex_config.get("seq_len", 30)

    # Load pre-split partitions (no internal splitting)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_presplit_sequences(exercise, hybrid=hybrid)
    if X_train is None:
        print(f"  [WARNING] {exercise}: pre-split .npz files not found — skipping. "
              "Run split_videos.py then build_sequences.py first.")
        return {}

    n_correct_train = int((y_train == 1).sum())
    n_incorrect_train = int((y_train == 0).sum())

    print(f"\n{'='*60}")
    print(f"Exercise: {exercise}")
    print(f"  Train: {len(X_train)} (correct: {n_correct_train}, incorrect: {n_incorrect_train})")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")
    print(f"  Shape: {X_train.shape}")
    print(f"  Device: {device}")
    print(f"  Config: hidden_dim={hidden_dim}, num_layers={num_layers}, seq_len={seq_len}")
    if use_landmarks or X_train.shape[2] == 99:
        print(f"  Mode: landmark features ({X_train.shape[2]}-dim)")

    if n_correct_train == 0 or n_incorrect_train == 0:
        print(f"  [SKIP] Only one class present in training set")
        return {}

    # Check class ratio — should be ~1.0 on balanced data
    ratio = n_correct_train / max(n_incorrect_train, 1)
    print(f"  Class ratio (correct/incorrect): {ratio:.3f}  — expected ~1.0 on balanced data")

    # Determine if we can use AMP (mixed precision)
    use_amp = (device is not None and device.type == "cuda")
    if use_amp:
        print(f"  Mixed precision: enabled (AMP)")

    # DataLoaders — no WeightedRandomSampler (data is pre-balanced)
    # pin_memory + num_workers for GPU transfer optimization
    # Scale workers based on input dim to avoid OOM on 330-dim temporal features
    n_workers = 4 if X_train.shape[2] > 200 else 8
    dl_kwargs = {
        "batch_size": batch_size,
        "pin_memory": True,
        "num_workers": n_workers,
        "persistent_workers": True,
        "prefetch_factor": 4,
    }

    train_ds = SequenceDataset(X_train, y_train, augment=True)
    val_ds = SequenceDataset(X_val, y_val, augment=False)
    test_ds = SequenceDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **dl_kwargs)

    # Model (CNN-BiLSTM with multi-head attention)
    input_dim = X_train.shape[2]
    model = FormBiLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_conv=True,
    ).to(device)

    # torch.compile disabled — causes runtime errors on PyTorch 2.5.1+cu121
    # with FormBiLSTM's dynamic shapes. AMP + TF32 + cuDNN benchmark provide
    # comparable speedups without the compilation overhead.
    compiled = False

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")
    print(f"  Architecture: CNN-BiLSTM (Conv1D + {num_layers}-layer BiLSTM + Attention)")

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    if use_bce:
        # Standard BCE with pos_weight for class imbalance
        pos_weight = torch.tensor([n_incorrect_train / max(n_correct_train, 1)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"  Loss: BCEWithLogitsLoss (pos_weight={pos_weight.item():.3f})")
    else:
        criterion = FocalLoss(alpha=0.5, gamma=2.0)
        criterion.set_class_weights(n_positive=n_correct_train, n_negative=n_incorrect_train)

    # LR schedule: linear warmup -> cosine annealing with warm restarts
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # SWA setup: activates in last 25% of training epochs (disabled with --no-swa)
    if not no_swa:
        swa_model = AveragedModel(model)
        swa_start = int(epochs * 0.75)
        swa_scheduler = SWALR(optimizer, swa_lr=1e-5)
        print(f"  SWA: activates at epoch {swa_start} (last 25% of {epochs} epochs)")
    else:
        swa_model = None
        swa_start = epochs + 1  # never activate
        swa_scheduler = None
        print(f"  SWA: disabled")

    # Mixed precision scaler
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Training loop with early stopping on val_f1
    # Fixed threshold=0.5 during training — no per-epoch search on val (leakage)
    threshold = 0.5
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    train_losses = []
    val_losses = []
    val_f1s = []

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            # Mixup augmentation (50% of the time, disabled with --no-mixup)
            if not no_mixup and torch.rand(1).item() < 0.5:
                lam = np.random.beta(0.2, 0.2)
                perm = torch.randperm(X_batch.size(0), device=X_batch.device)
                X_batch = lam * X_batch + (1 - lam) * X_batch[perm]
                y_batch = lam * y_batch + (1 - lam) * y_batch[perm]

            optimizer.zero_grad(set_to_none=True)

            # Mixed precision forward pass
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(X_batch)
                loss = criterion(logits, y_batch)

            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # Validate: compute loss and F1 at fixed threshold=0.5
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_probs = []
        val_labels = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)

                val_loss += loss.item()
                val_batches += 1
                val_probs.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())

        avg_val_loss = val_loss / max(val_batches, 1)
        val_losses.append(avg_val_loss)

        # F1 at fixed threshold — no per-epoch search
        val_probs_arr = np.array(val_probs)
        val_labels_arr = np.array(val_labels)
        val_preds = (val_probs_arr >= threshold).astype(int)
        epoch_val_f1 = f1_score(val_labels_arr, val_preds, zero_division=0)
        val_f1s.append(epoch_val_f1)

        # Step LR scheduler (or SWA scheduler in last 25%)
        if swa_model is not None and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # Early stopping: track best val_f1 (primary) and val_loss (secondary)
        improved = False
        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            improved = True
        elif epoch_val_f1 == best_val_f1 and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            improved = True

        if epoch % 10 == 0 or epoch == epochs - 1 or improved:
            lr_now = optimizer.param_groups[0]["lr"]
            marker = " *" if improved else ""
            print(f"  Epoch {epoch:3d}/{epochs}: "
                  f"train_loss={avg_train_loss:.4f}, "
                  f"val_loss={avg_val_loss:.4f}, "
                  f"val_f1={epoch_val_f1:.3f}, "
                  f"lr={lr_now:.6f}{marker}")

        if epoch - best_epoch >= patience:
            print(f"  Early stopping at epoch {epoch} (best={best_epoch})")
            break

    # Update BatchNorm stats for SWA model (critical: FormBiLSTM uses BatchNorm1d)
    if swa_model is not None:
        print("  Updating BatchNorm stats for SWA model...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

    # Use SWA model if available, otherwise use best checkpoint directly
    eval_model = swa_model if swa_model is not None else model
    if swa_model is None and best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        eval_model = model

    # One-time threshold search on val set — save best threshold into checkpoint
    print(f"  [Threshold search] Finding best threshold on val set...")
    eval_model.eval()
    val_probs_final = []
    val_labels_final = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            logits = eval_model(X_batch)
            val_probs_final.extend(torch.sigmoid(logits).cpu().numpy())
            val_labels_final.extend(y_batch.numpy())
    val_prob_arr = np.array(val_probs_final)
    val_true_arr = np.array(val_labels_final)
    best_thresh_val = 0.5
    best_f1_val = 0.0
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (val_prob_arr >= thresh).astype(int)
        f1_val = f1_score(val_true_arr, preds, zero_division=0)
        if f1_val > best_f1_val:
            best_f1_val = f1_val
            best_thresh_val = thresh
    print(f"  [Threshold search] Best val threshold: {best_thresh_val:.2f}, F1={best_f1_val:.3f}")

    # Evaluate model on test set with fixed threshold=0.5
    eval_model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            logits = eval_model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())

    y_prob = np.array(all_probs)
    y_true = np.array(all_labels)
    y_pred = (y_prob >= threshold).astype(int)

    # Compute train F1 (no augmentation, same fixed threshold=0.5 as test eval)
    train_ds_eval = SequenceDataset(X_train, y_train, augment=False)
    train_loader_eval = DataLoader(train_ds_eval, shuffle=False, **dl_kwargs)
    train_probs_list, train_labels_list = [], []
    eval_model.eval()
    with torch.no_grad():
        for X_batch, y_batch in train_loader_eval:
            X_batch = X_batch.to(device, non_blocking=True)
            logits = eval_model(X_batch)
            train_probs_list.extend(torch.sigmoid(logits).cpu().numpy())
            train_labels_list.extend(y_batch.numpy())
    train_preds = (np.array(train_probs_list) >= threshold).astype(int)
    train_f1_val = float(f1_score(np.array(train_labels_list), train_preds, zero_division=0))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "train_f1": train_f1_val,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_val_f1": float(best_val_f1),
        "threshold_used_for_eval": float(threshold),
        "val_best_thresh": float(best_thresh_val),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_params": n_params,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
                "seq_len": seq_len,
        "use_landmarks": use_landmarks or (input_dim == 99),
        "use_conv": True,
        "swa_used": swa_model is not None,
        "amp_used": use_amp,
        "compiled": compiled,
    }

    model_label = "SWA model" if swa_model is not None else "best checkpoint"
    print(f"\n  Test Results ({model_label}, epoch {best_epoch}, threshold={threshold:.2f}):")
    print(f"    Accuracy:  {metrics['accuracy']:.3f}")
    print(f"    Precision: {metrics['precision']:.3f}")
    print(f"    Recall:    {metrics['recall']:.3f}")
    print(f"    F1:        {metrics['f1']:.3f}")
    print(f"    Train F1:  {train_f1_val:.3f}")

    # Print diagnostics
    cm_stats = print_diagnostics(y_true, y_pred, y_prob, exercise, threshold)
    metrics["confusion_matrix"] = cm_stats

    # If eval model is degenerate (F1 near 0), fall back to best checkpoint
    is_degenerate = metrics["f1"] < 0.1 and best_val_f1 > 0.2
    if is_degenerate and best_state is not None:
        print(f"\n  WARNING: Eval model degenerate (F1={metrics['f1']:.3f}), "
              f"falling back to best checkpoint (epoch {best_epoch}, val_f1={best_val_f1:.3f})")
        model.load_state_dict(best_state)
        model.to(device)
        model.eval()
        y_prob_best, y_true_best = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                logits = model(X_batch)
                y_prob_best.extend(torch.sigmoid(logits).cpu().numpy())
                y_true_best.extend(y_batch.numpy())
        y_prob = np.array(y_prob_best)
        y_true = np.array(y_true_best)
        y_pred = (y_prob >= threshold).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        train_prob_best, train_true_best = [], []
        train_eval_ds = SequenceDataset(X_train, y_train, augment=False)
        train_eval_loader = DataLoader(train_eval_ds, shuffle=False, **dl_kwargs)
        with torch.no_grad():
            for X_batch, y_batch in train_eval_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                logits = model(X_batch)
                train_prob_best.extend(torch.sigmoid(logits).cpu().numpy())
                train_true_best.extend(y_batch.numpy())
        train_pred_best = (np.array(train_prob_best) >= threshold).astype(int)
        train_f1_val = float(f1_score(np.array(train_true_best), train_pred_best, zero_division=0))
        metrics["train_f1"] = train_f1_val
        cm_stats = print_diagnostics(y_true, y_pred, y_prob, exercise, threshold)
        metrics["confusion_matrix"] = cm_stats
        print(f"  Best checkpoint test F1: {metrics['f1']:.3f}, train F1: {train_f1_val:.3f}")

        val_probs_ck = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                logits = model(X_batch)
                val_probs_ck.extend(torch.sigmoid(logits).cpu().numpy())
        val_prob_ck = np.array(val_probs_ck)
        best_thresh_val = 0.5
        best_f1_ck = 0.0
        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (val_prob_ck >= thresh).astype(int)
            f1_ck = f1_score(val_true_arr, preds, zero_division=0)
            if f1_ck > best_f1_ck:
                best_f1_ck = f1_ck
                best_thresh_val = thresh
        print(f"  [Threshold search] Best val threshold (checkpoint): {best_thresh_val:.2f}, F1={best_f1_ck:.3f}")

        save_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        used_swa = False
    elif swa_model is not None:
        save_state = swa_model.cpu().state_dict()
        used_swa = True
    else:
        # No SWA — save from eval_model (which is the best checkpoint model)
        save_state = {k: v.cpu().clone() for k, v in eval_model.state_dict().items()}
        used_swa = False

    # Save model checkpoint as v2
    checkpoint = {
        "model_state_dict": save_state,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "use_conv": True,
                "seq_len": seq_len,
        "best_threshold": float(best_thresh_val),
        "temperature": 1.0,
        "best_epoch": best_epoch,
        "threshold_used_for_eval": threshold,
        "metrics": metrics,
        "feature_names": list(feature_names),
        "swa_used": used_swa,
    }
    model_path = MODELS_DIR / f"{exercise}_bilstm_v2.pt"
    torch.save(checkpoint, model_path)
    print(f"  Saved: {model_path}")

    # Save evaluation JSON
    eval_path = MODELS_DIR / f"{exercise}_bilstm_v2_eval.json"
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save training curves
    curves_path = REPORTS_DIR / f"{exercise}_bilstm_v2_curves.json"
    with open(curves_path, "w") as f:
        json.dump({
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_f1s": val_f1s,
            "best_epoch": best_epoch,
        }, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train BiLSTM form classifiers (v2)")
    parser.add_argument("--exercise", type=str, default=None,
                        help="Train for a specific exercise only")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=None,
                        help="Override hidden_dim (default: from EXERCISE_CONFIGS)")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Override num_layers (default: from EXERCISE_CONFIGS)")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--use-landmarks", action="store_true",
                        help="Use landmark-based features (99-dim) instead of hand-crafted")
    parser.add_argument("--data-source", type=str, default="default",
                        choices=["default", "balanced"],
                        help="Data source: default or balanced")
    parser.add_argument("--balanced", action="store_true",
                        help="Shortcut for --data-source balanced")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda, cpu, or auto (default)")
    parser.add_argument("--no-mixup", action="store_true",
                        help="Disable mixup augmentation")
    parser.add_argument("--no-swa", action="store_true",
                        help="Disable Stochastic Weight Averaging")
    parser.add_argument("--use-bce", action="store_true",
                        help="Use standard BCE loss instead of FocalLoss")
    parser.add_argument("--hybrid", action="store_true",
                        help="Load hybrid sequences (99 landmarks + 10 angles = 109 features)")
    args = parser.parse_args()

    # --balanced is a shortcut for --data-source balanced
    if args.balanced:
        args.data_source = "balanced"

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        print(f"GPU Memory: {gpu_mem / 1e9:.1f} GB")
        # Allow GPU to use shared system memory (overflow VRAM to RAM)
        try:
            torch.cuda.set_per_process_memory_fraction(0.95)
        except Exception:
            pass  # Not supported on all setups
        # Enable TF32 for faster matrix ops on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"  TF32: enabled, cuDNN benchmark: enabled")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU detected, training on CPU (will be slower)")

    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if args.use_landmarks:
        print("Mode: Landmark features (99-dim)")
    if args.data_source != "default":
        print(f"Data source: {args.data_source}")

    exercises = [args.exercise] if args.exercise else EXERCISES
    all_results = {}

    for exercise in exercises:
        metrics = train_exercise(
            exercise,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            patience=args.patience,
            warmup_epochs=args.warmup_epochs,
            use_landmarks=args.use_landmarks,
            data_source=args.data_source,
            device=device,
            no_mixup=args.no_mixup,
            no_swa=args.no_swa,
            use_bce=args.use_bce,
            hybrid=args.hybrid,
        )
        if metrics:
            all_results[exercise] = metrics

    # Summary
    if all_results:
        print(f"\n{'='*60}")
        print("BiLSTM v2 TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"{'Exercise':<18} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Train F1':>9} {'Epoch':>6} {'Thresh':>7}")
        print("-" * 72)
        for ex in sorted(all_results.keys()):
            m = all_results[ex]
            print(f"{ex:<18} {m['accuracy']:>6.3f} {m['precision']:>6.3f} "
                  f"{m['recall']:>6.3f} {m['f1']:>6.3f} {m.get('train_f1', 0.0):>9.3f} "
                  f"{m.get('best_epoch', 0):>6} {m.get('threshold_used_for_eval', 0.5):>7.2f}")

        avg_f1 = np.mean([m["f1"] for m in all_results.values()])
        print("-" * 65)
        print(f"{'AVERAGE F1':<18} {'':>6} {'':>6} {'':>6} {avg_f1:>6.3f}")

        # Warn about degenerate models
        for ex, m in all_results.items():
            if m["f1"] < 0.2:
                print(f"  WARNING: {ex} has F1 < 0.2 — model may be degenerate")
            if m.get("best_epoch", 0) == 0:
                print(f"  WARNING: {ex} best_epoch=0 — model never improved beyond init")

        # Save summary — accumulate across sequential single-exercise runs (load-merge-save)
        summary_path = REPORTS_DIR / "bilstm_v2_training_summary.json"
        existing = {}
        if summary_path.exists():
            with open(summary_path, encoding="utf-8") as f:
                existing = json.load(f)
        existing.update(all_results)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
