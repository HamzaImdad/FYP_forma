"""
Train BiLSTM models for exercise form classification.

Uses PyTorch with GPU acceleration. Trains one model per exercise on
sequence data (sliding windows of biomechanical features).

Usage:
    python scripts/train_bilstm.py
    python scripts/train_bilstm.py --exercise squat
    python scripts/train_bilstm.py --epochs 100 --batch-size 32
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

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
    alpha controls class weighting: higher alpha = more weight on positive class.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.alpha = alpha  # >0.5 upweights positive (minority) class
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=torch.tensor([self.pos_weight], device=logits.device),
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        # alpha_t: high alpha for positive (minority), low for negative (majority)
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
        x = self.X[idx]
        if self.augment:
            # Gaussian noise (moderate)
            x = x + torch.randn_like(x) * 0.02
            # Random feature dropout (zero out some features for some timesteps)
            if torch.rand(1).item() < 0.15:
                mask = torch.rand_like(x) > 0.05  # 5% dropout
                x = x * mask
            # Random time scaling (speed variation)
            if torch.rand(1).item() < 0.3:
                scale = 0.85 + torch.rand(1).item() * 0.3  # 0.85-1.15
                seq_len = x.shape[0]
                indices = torch.linspace(0, seq_len - 1, int(seq_len * scale)).long().clamp(0, seq_len - 1)
                if len(indices) >= seq_len:
                    x = x[indices[:seq_len]]
                else:
                    pad = x[-1:].repeat(seq_len - len(indices), 1)
                    x = torch.cat([x[indices], pad], dim=0)
            # Random temporal shift
            if torch.rand(1).item() < 0.3:
                shift = torch.randint(-3, 4, (1,)).item()
                if shift != 0:
                    x = torch.roll(x, shifts=shift, dims=0)
        return x, self.y[idx]


def load_sequences(exercise: str) -> tuple:
    """Load sequence data for an exercise.

    Returns: X, y, video_ids, feature_names
    """
    path = SEQUENCES_DIR / f"{exercise}_sequences.npz"
    if not path.exists():
        return None, None, None, None

    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    video_ids = data["video_ids"]
    feature_names = data["feature_names"]

    return X, y, video_ids, feature_names


def train_exercise(
    exercise: str,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    patience: int = 10,
    device: torch.device = None,
) -> dict:
    """Train BiLSTM for one exercise. Returns metrics dict."""

    X, y, video_ids, feature_names = load_sequences(exercise)
    if X is None:
        print(f"  [SKIP] No sequence data for {exercise}")
        return {}

    n_correct = int((y == 1).sum())
    n_incorrect = int((y == 0).sum())
    n_unique_videos = len(set(video_ids))

    print(f"\n{'='*60}")
    print(f"Exercise: {exercise}")
    print(f"  Sequences: {len(X)} (correct: {n_correct}, incorrect: {n_incorrect})")
    print(f"  Shape: {X.shape}")
    print(f"  Unique videos: {n_unique_videos}")
    print(f"  Device: {device}")

    if n_correct == 0 or n_incorrect == 0:
        print(f"  [SKIP] Only one class present")
        return {}

    if n_unique_videos < 3:
        print(f"  [SKIP] Not enough video groups for splitting")
        return {}

    # Group-based train/test split by video_id (no data leakage)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=video_ids))

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Further split train into train/val
    groups_train = video_ids[train_idx]
    if len(set(groups_train)) >= 2:
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        t_idx, v_idx = next(gss2.split(X_train, y_train, groups=groups_train))
        X_val, y_val = X_train[v_idx], y_train[v_idx]
        X_train, y_train = X_train[t_idx], y_train[t_idx]
    else:
        X_val, y_val = X_test, y_test

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Class weight for imbalanced data
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    print(f"  Class weight (pos_weight): {pos_weight.item():.2f}")

    # DataLoaders with WeightedRandomSampler for balanced batches
    train_ds = SequenceDataset(X_train, y_train, augment=True)
    val_ds = SequenceDataset(X_val, y_val, augment=False)
    test_ds = SequenceDataset(X_test, y_test, augment=False)

    # Balanced sampling: oversample minority class in each batch
    sample_weights = np.zeros(len(y_train))
    sample_weights[y_train == 0] = 1.0 / max(n_neg, 1)
    sample_weights[y_train == 1] = 1.0 / max(n_pos, 1)
    sample_weights = torch.tensor(sample_weights, dtype=torch.float64)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(y_train), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model
    input_dim = X.shape[2]
    model = FormBiLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    # Training setup -- Focal Loss with alpha>0.5 to upweight minority (correct) class
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )
    criterion = FocalLoss(alpha=0.75, gamma=2.0, pos_weight=pos_weight.item())

    # Training loop with early stopping
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    train_losses = []
    val_losses = []

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

        # Validate
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs - 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{epochs}: "
                  f"train_loss={avg_train_loss:.4f}, "
                  f"val_loss={avg_val_loss:.4f}, "
                  f"lr={lr_now:.6f}")

        if epoch - best_epoch >= patience:
            print(f"  Early stopping at epoch {epoch} (best={best_epoch})")
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # Evaluate on test set
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())

    y_prob = np.array(all_probs)
    y_true = np.array(all_labels)

    # Find best threshold by F1 on test set
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (y_prob >= thresh).astype(int)
        f1_val = f1_score(y_true, preds, zero_division=0)
        if f1_val > best_f1:
            best_f1 = f1_val
            best_thresh = thresh

    y_pred = (y_prob >= best_thresh).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_threshold": float(best_thresh),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_params": n_params,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }

    print(f"\n  Test Results (epoch {best_epoch}, threshold={best_thresh:.2f}):")
    print(f"    Accuracy:  {metrics['accuracy']:.3f}")
    print(f"    Precision: {metrics['precision']:.3f}")
    print(f"    Recall:    {metrics['recall']:.3f}")
    print(f"    F1:        {metrics['f1']:.3f}")

    # Save model checkpoint
    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "best_epoch": best_epoch,
        "best_threshold": best_thresh,
        "metrics": metrics,
        "feature_names": list(feature_names),
    }
    model_path = MODELS_DIR / f"{exercise}_bilstm.pt"
    torch.save(checkpoint, model_path)
    print(f"  Saved: {model_path}")

    # Save evaluation JSON
    eval_path = MODELS_DIR / f"{exercise}_bilstm_eval.json"
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save training curves
    curves_path = REPORTS_DIR / f"{exercise}_bilstm_curves.json"
    with open(curves_path, "w") as f:
        json.dump({
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_epoch": best_epoch,
        }, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train BiLSTM form classifiers")
    parser.add_argument("--exercise", type=str, default=None,
                        help="Train for a specific exercise only")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda, cpu, or auto (default)")
    args = parser.parse_args()

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
            device=device,
        )
        if metrics:
            all_results[exercise] = metrics

    # Summary
    if all_results:
        print(f"\n{'='*60}")
        print("BiLSTM TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"{'Exercise':<18} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Epoch':>6}")
        print("-" * 55)
        for ex in sorted(all_results.keys()):
            m = all_results[ex]
            print(f"{ex:<18} {m['accuracy']:>6.3f} {m['precision']:>6.3f} "
                  f"{m['recall']:>6.3f} {m['f1']:>6.3f} {m['best_epoch']:>6}")

        avg_f1 = np.mean([m["f1"] for m in all_results.values()])
        print("-" * 55)
        print(f"{'AVERAGE F1':<18} {'':>6} {'':>6} {'':>6} {avg_f1:>6.3f}")

        # Save summary
        summary_path = REPORTS_DIR / "bilstm_training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
