"""
Cross-exercise pretraining: train a shared BiLSTM encoder on ALL exercises combined,
then save the encoder weights for per-exercise fine-tuning.

The key insight: body mechanics features (joint angles, movement patterns) transfer
across exercises. A model trained on 45K+ combined sequences learns much better
representations than one trained on 2K-7K per exercise.

Architecture:
  - Same FormBiLSTM but with exercise_id as additional input (one-hot, 10-dim)
  - Binary classification: correct vs incorrect form
  - Trained on concatenated data from all 10 exercises

Usage:
    python scripts/pretrain_cross_exercise.py
    python scripts/pretrain_cross_exercise.py --epochs 300 --batch-size 128
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
from sklearn.metrics import f1_score, accuracy_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.bilstm_classifier import FormBiLSTM
from src.utils.constants import EXERCISES

SEQUENCES_DIR = PROJECT_ROOT / "data" / "processed" / "sequences"
MODELS_DIR = PROJECT_ROOT / "models" / "trained"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = PROJECT_ROOT / "reports"

# Sequence lengths per exercise (must match build_sequences.py)
SEQ_LENS = {
    "deadlift": 45, "squat": 45,
    "bench_press": 30, "pushup": 30, "pullup": 30, "plank": 30,
    "lunge": 30, "bicep_curl": 30, "overhead_press": 30, "tricep_dip": 30,
}


class CrossExerciseDataset(Dataset):
    """Combined dataset from all exercises with exercise_id encoding."""

    def __init__(self, X_list, y_list, ex_ids_list, augment=False):
        self.X = torch.tensor(np.concatenate(X_list, axis=0), dtype=torch.float32)
        self.y = torch.tensor(np.concatenate(y_list, axis=0), dtype=torch.float32)
        self.ex_ids = torch.tensor(np.concatenate(ex_ids_list, axis=0), dtype=torch.long)
        self.augment = augment
        self.n_exercises = 10

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        ex_id = self.ex_ids[idx]

        if self.augment:
            # Gaussian noise
            x = x + torch.randn_like(x) * 0.03
            # Random scaling
            if torch.rand(1).item() < 0.3:
                scale = 0.9 + torch.rand(1).item() * 0.2
                x = x * scale
            # Left-right mirror
            if torch.rand(1).item() < 0.3:
                for i in range(33):
                    x[:, i*3] = -x[:, i*3]
                swap_pairs = [(1,4),(2,5),(3,6),(7,8),(9,10),
                              (11,12),(13,14),(15,16),(17,18),(19,20),
                              (21,22),(23,24),(25,26),(27,28),(29,30),(31,32)]
                for a, b in swap_pairs:
                    tmp = x[:, a*3:(a+1)*3].clone()
                    x[:, a*3:(a+1)*3] = x[:, b*3:(b+1)*3]
                    x[:, b*3:(b+1)*3] = tmp

        # Add exercise one-hot encoding to each frame
        ex_onehot = torch.zeros(self.n_exercises)
        ex_onehot[ex_id] = 1.0
        # Expand to all frames: (seq_len, n_exercises)
        ex_expanded = ex_onehot.unsqueeze(0).expand(x.shape[0], -1)
        # Concatenate: (seq_len, 99 + 10) = (seq_len, 109)
        x_with_ex = torch.cat([x, ex_expanded], dim=1)

        return x_with_ex, self.y[idx]


def load_all_exercises(target_seq_len=30):
    """Load and pad/truncate all exercise sequences to a common length."""
    X_trains, y_trains, ex_trains = [], [], []
    X_vals, y_vals, ex_vals = [], [], []
    X_tests, y_tests, ex_tests = [], [], []

    for i, ex in enumerate(EXERCISES):
        train_path = SEQUENCES_DIR / f"{ex}_train_sequences.npz"
        val_path = SEQUENCES_DIR / f"{ex}_val_sequences.npz"
        test_path = SEQUENCES_DIR / f"{ex}_test_sequences.npz"

        if not all(p.exists() for p in [train_path, val_path, test_path]):
            print(f"  [SKIP] {ex}: missing split files")
            continue

        for path, X_list, y_list, ex_list in [
            (train_path, X_trains, y_trains, ex_trains),
            (val_path, X_vals, y_vals, ex_vals),
            (test_path, X_tests, y_tests, ex_tests),
        ]:
            data = np.load(path, allow_pickle=True)
            X = data["X"]  # (n, seq_len, features)
            y = data["y"]

            # Only use 99-dim features (skip 330-dim temporal)
            if X.shape[2] > 99:
                X = X[:, :, :99]

            # Pad or truncate to target_seq_len
            if X.shape[1] < target_seq_len:
                pad_width = target_seq_len - X.shape[1]
                X = np.pad(X, ((0,0), (0,pad_width), (0,0)), mode='edge')
            elif X.shape[1] > target_seq_len:
                X = X[:, :target_seq_len, :]

            X_list.append(X)
            y_list.append(y)
            ex_list.append(np.full(len(y), i, dtype=np.int64))

        n_train = len(y_trains[-1]) if y_trains else 0
        print(f"  {ex}: train={n_train}, val={len(y_vals[-1])}, test={len(y_tests[-1])}")

    return (X_trains, y_trains, ex_trains,
            X_vals, y_vals, ex_vals,
            X_tests, y_tests, ex_tests)


def main():
    parser = argparse.ArgumentParser(description="Cross-exercise pretraining")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Load all exercises
    print("\nLoading all exercises...")
    target_seq_len = 30  # Common seq_len (truncate 45-frame exercises)
    (X_trains, y_trains, ex_trains,
     X_vals, y_vals, ex_vals,
     X_tests, y_tests, ex_tests) = load_all_exercises(target_seq_len)

    # Create datasets
    input_dim = 99 + 10  # landmarks + exercise one-hot
    train_ds = CrossExerciseDataset(X_trains, y_trains, ex_trains, augment=True)
    val_ds = CrossExerciseDataset(X_vals, y_vals, ex_vals, augment=False)
    test_ds = CrossExerciseDataset(X_tests, y_tests, ex_tests, augment=False)

    print(f"\nCombined dataset:")
    print(f"  Train: {len(train_ds)} sequences")
    print(f"  Val:   {len(val_ds)} sequences")
    print(f"  Test:  {len(test_ds)} sequences")
    print(f"  Input dim: {input_dim} (99 landmarks + 10 exercise one-hot)")

    n_correct = int((train_ds.y == 1).sum())
    n_incorrect = int((train_ds.y == 0).sum())
    print(f"  Train balance: {n_correct} correct, {n_incorrect} incorrect ({n_correct/(n_incorrect+1):.2f}:1)")

    dl_kwargs = {
        "batch_size": args.batch_size,
        "pin_memory": True,
        "num_workers": 8,
        "persistent_workers": True,
        "prefetch_factor": 4,
    }

    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **dl_kwargs)

    # Model — same architecture but with 109-dim input
    model = FormBiLSTM(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        dropout=0.3,
        use_conv=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # BCE with pos_weight for class imbalance
    pos_weight = torch.tensor([n_incorrect / max(n_correct, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"  Loss: BCE (pos_weight={pos_weight.item():.3f})")

    # LR schedule: warmup + cosine
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs-5, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[5])

    # AMP
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Training loop
    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(X_batch)
                loss = criterion(logits, y_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(X_batch)
                val_probs.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels.extend(y_batch.numpy())

        val_preds = (np.array(val_probs) >= 0.5).astype(int)
        val_f1 = f1_score(np.array(val_labels), val_preds, zero_division=0)

        improved = val_f1 > best_val_f1
        if improved:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == args.epochs - 1 or improved:
            lr = optimizer.param_groups[0]["lr"]
            m = " *" if improved else ""
            print(f"  Epoch {epoch:3d}/{args.epochs}: loss={avg_loss:.4f}, val_f1={val_f1:.3f}, lr={lr:.6f}{m}")

        if epoch - best_epoch >= args.patience:
            print(f"  Early stopping at epoch {epoch} (best={best_epoch})")
            break

    # Load best checkpoint
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # Evaluate on test set
    test_probs, test_labels, test_ex_ids = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            logits = model(X_batch)
            test_probs.extend(torch.sigmoid(logits).cpu().numpy())
            test_labels.extend(y_batch.numpy())

    y_prob = np.array(test_probs)
    y_true = np.array(test_labels)
    y_pred = (y_prob >= 0.5).astype(int)

    overall_f1 = f1_score(y_true, y_pred, zero_division=0)
    overall_acc = accuracy_score(y_true, y_pred)
    print(f"\n  Cross-exercise test results (epoch {best_epoch}):")
    print(f"    Overall F1:  {overall_f1:.3f}")
    print(f"    Overall Acc: {overall_acc:.3f}")

    # Per-exercise test F1
    test_ex_all = np.concatenate([np.full(len(np.load(SEQUENCES_DIR / f"{ex}_test_sequences.npz")["y"]), i)
                                   for i, ex in enumerate(EXERCISES)
                                   if (SEQUENCES_DIR / f"{ex}_test_sequences.npz").exists()])
    print(f"\n  Per-exercise test F1:")
    for i, ex in enumerate(EXERCISES):
        mask = test_ex_all == i
        if mask.sum() == 0:
            continue
        ex_f1 = f1_score(y_true[mask], y_pred[mask], zero_division=0)
        print(f"    {ex:20s}: {ex_f1:.3f}")

    # Save pretrained encoder
    # The encoder is everything EXCEPT the classifier head
    # We save the full model state — fine-tuning scripts will load encoder weights
    checkpoint = {
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": 2,
        "dropout": 0.3,
        "use_conv": True,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "overall_test_f1": overall_f1,
        "training_config": vars(args),
    }

    save_path = MODELS_DIR / "cross_exercise_pretrained.pt"
    torch.save(checkpoint, save_path)
    print(f"\n  Saved pretrained model: {save_path}")

    # Save report
    report = {
        "overall_f1": overall_f1,
        "overall_acc": overall_acc,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
    }
    report_path = REPORTS_DIR / "cross_exercise_pretrain_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved report: {report_path}")


if __name__ == "__main__":
    main()
