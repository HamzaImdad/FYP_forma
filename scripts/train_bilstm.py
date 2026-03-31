"""
Train BiLSTM models for exercise form classification.

Uses PyTorch with GPU acceleration. Trains one model per exercise on
sequence data (sliding windows of biomechanical features).

v2 improvements:
  - CNN-BiLSTM architecture (Conv1D front-end + BiLSTM)
  - Learning rate warmup + cosine annealing
  - Threshold search on validation set (no data leakage)
  - Early stopping on val_f1 (not val_loss)
  - Support for landmark-based features (99-dim input)
  - Training diagnostics (confusion matrix, degenerate model warnings)

Usage:
    python scripts/train_bilstm.py
    python scripts/train_bilstm.py --exercise squat
    python scripts/train_bilstm.py --exercise bench_press --use-landmarks --epochs 100
    python scripts/train_bilstm.py --data-source balanced
"""

import sys
import json
import argparse
from pathlib import Path

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


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced binary classification.

    Down-weights easy examples and focuses training on hard ones.
    alpha=0.5 is symmetric (no class bias) — data is pre-balanced.
    Label smoothing 0.1 prevents overconfident memorisation.
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

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
        # alpha_t: symmetric (0.5) — balanced data needs no class upweighting
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
        x = self.X[idx].clone()  # shape: (seq_len, features)
        if self.augment:
            # Gaussian noise — std=0.05 (strengthened from 0.02)
            x = x + torch.randn_like(x) * 0.05
            # Random feature dropout (zero out some features for some timesteps)
            if torch.rand(1).item() < 0.15:
                mask = torch.rand_like(x) > 0.05  # 5% dropout
                x = x * mask
            # Time warp via numpy.interp — proper temporal stretching without frame wrapping
            if torch.rand(1).item() < 0.3:
                scale = 0.85 + torch.rand(1).item() * 0.30  # 0.85-1.15
                seq_len = x.shape[0]
                x_np = x.numpy()
                old_idx = np.linspace(0, seq_len - 1, int(seq_len * scale))
                new_idx = np.linspace(0, seq_len - 1, seq_len)
                x_warped = np.stack([
                    np.interp(new_idx, old_idx, x_np[:, f])
                    for f in range(x_np.shape[1])
                ], axis=1)
                x = torch.tensor(x_warped, dtype=torch.float32)
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


def load_presplit_sequences(exercise: str) -> tuple:
    """Load pre-split train/val/test .npz files for an exercise.

    Files created by build_sequences.py using partition manifests from split_videos.py.
    Expected keys: X (n, 30, 99), y (n,), video_ids (n,), feature_names (99,)

    Returns: (X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
             All None if any file is missing.
    """
    partitions = {}
    for partition in ("train", "val", "test"):
        path = SEQUENCES_DIR / f"{exercise}_{partition}_sequences.npz"
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
            X_batch = X_batch.to(device)
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
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    patience: int = 30,
    warmup_epochs: int = 5,
    use_landmarks: bool = False,
    data_source: str = "default",
    device: torch.device = None,
) -> dict:
    """Train BiLSTM for one exercise. Returns metrics dict.

    Loads pre-split {exercise}_{train|val|test}_sequences.npz files created by
    build_sequences.py using partition manifests from split_videos.py.
    No internal GroupShuffleSplit — splits are done once, before this script runs.
    """

    # Load pre-split partitions (no internal splitting)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_presplit_sequences(exercise)
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
    if use_landmarks or X_train.shape[2] == 99:
        print(f"  Mode: landmark features ({X_train.shape[2]}-dim)")

    if n_correct_train == 0 or n_incorrect_train == 0:
        print(f"  [SKIP] Only one class present in training set")
        return {}

    # Check class ratio — should be ~1.0 on balanced data
    ratio = n_correct_train / max(n_incorrect_train, 1)
    print(f"  Class ratio (correct/incorrect): {ratio:.3f}  — expected ~1.0 on balanced data")

    # DataLoaders — no WeightedRandomSampler (data is pre-balanced)
    train_ds = SequenceDataset(X_train, y_train, augment=True)
    val_ds = SequenceDataset(X_val, y_val, augment=False)
    test_ds = SequenceDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model (CNN-BiLSTM)
    input_dim = X_train.shape[2]
    model = FormBiLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_conv=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")
    print(f"  Architecture: CNN-BiLSTM (Conv1D + {num_layers}-layer BiLSTM + Attention)")

    # Training setup — FocalLoss(alpha=0.5) only (single weighting mechanism on balanced data)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = FocalLoss(alpha=0.5, gamma=2.0)

    # LR schedule: linear warmup -> cosine annealing
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # SWA setup: activates in last 25% of training epochs
    swa_model = AveragedModel(model)
    swa_start = int(epochs * 0.75)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-5)
    print(f"  SWA: activates at epoch {swa_start} (last 25% of {epochs} epochs)")

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
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

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
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
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
        if epoch >= swa_start:
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
    print("  Updating BatchNorm stats for SWA model...")
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

    # One-time threshold report on val set (informational only — not used for model selection)
    print(f"  [Threshold report] Searching best threshold on val set (informational)...")
    swa_model.eval()
    val_probs_final = []
    val_labels_final = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            logits = swa_model(X_batch)
            val_probs_final.extend(torch.sigmoid(logits).cpu().numpy())
            val_labels_final.extend(y_batch.numpy())
    val_prob_arr = np.array(val_probs_final)
    val_true_arr = np.array(val_labels_final)
    best_thresh_info = 0.5
    best_f1_info = 0.0
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (val_prob_arr >= thresh).astype(int)
        f1_val = f1_score(val_true_arr, preds, zero_division=0)
        if f1_val > best_f1_info:
            best_f1_info = f1_val
            best_thresh_info = thresh
    print(f"  [Threshold report] Best val threshold (informational): {best_thresh_info:.2f}, F1={best_f1_info:.3f}")

    # Evaluate SWA model on test set with fixed threshold=0.5
    swa_model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = swa_model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())

    y_prob = np.array(all_probs)
    y_true = np.array(all_labels)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_val_f1": float(best_val_f1),
        "best_threshold": float(threshold),
        "val_best_thresh_info": float(best_thresh_info),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_params": n_params,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "use_landmarks": use_landmarks or (input_dim == 99),
        "use_conv": True,
        "swa_used": True,
    }

    print(f"\n  Test Results (SWA model, epoch {best_epoch}, threshold={threshold:.2f}):")
    print(f"    Accuracy:  {metrics['accuracy']:.3f}")
    print(f"    Precision: {metrics['precision']:.3f}")
    print(f"    Recall:    {metrics['recall']:.3f}")
    print(f"    F1:        {metrics['f1']:.3f}")

    # Print diagnostics
    cm_stats = print_diagnostics(y_true, y_pred, y_prob, exercise, threshold)
    metrics["confusion_matrix"] = cm_stats

    # Save SWA model checkpoint as v2
    checkpoint = {
        "model_state_dict": swa_model.cpu().state_dict(),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "use_conv": True,
        "best_epoch": best_epoch,
        "best_threshold": threshold,
        "metrics": metrics,
        "feature_names": list(feature_names),
        "swa_used": True,
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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=30)
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
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
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
        )
        if metrics:
            all_results[exercise] = metrics

    # Summary
    if all_results:
        print(f"\n{'='*60}")
        print("BiLSTM v2 TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"{'Exercise':<18} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Epoch':>6} {'Thresh':>7}")
        print("-" * 65)
        for ex in sorted(all_results.keys()):
            m = all_results[ex]
            print(f"{ex:<18} {m['accuracy']:>6.3f} {m['precision']:>6.3f} "
                  f"{m['recall']:>6.3f} {m['f1']:>6.3f} {m['best_epoch']:>6} {m['best_threshold']:>7.2f}")

        avg_f1 = np.mean([m["f1"] for m in all_results.values()])
        print("-" * 65)
        print(f"{'AVERAGE F1':<18} {'':>6} {'':>6} {'':>6} {avg_f1:>6.3f}")

        # Warn about degenerate models
        for ex, m in all_results.items():
            if m["f1"] < 0.2:
                print(f"  WARNING: {ex} has F1 < 0.2 — model may be degenerate")
            if m["best_epoch"] == 0:
                print(f"  WARNING: {ex} best_epoch=0 — model never improved beyond init")

        # Save summary
        summary_path = REPORTS_DIR / "bilstm_v2_training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
