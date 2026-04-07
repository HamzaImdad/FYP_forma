"""
Fine-tune per-exercise models from cross-exercise pretrained encoder.

Loads the shared encoder from pretrain_cross_exercise.py, replaces the
classifier head, freezes encoder for first N epochs, then unfreezes all.

Usage:
    python scripts/finetune_from_pretrained.py
    python scripts/finetune_from_pretrained.py --exercise squat --epochs 200
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.bilstm_classifier import FormBiLSTM
from src.utils.constants import EXERCISES

SEQUENCES_DIR = PROJECT_ROOT / "data" / "processed" / "sequences"
MODELS_DIR = PROJECT_ROOT / "models" / "trained"
REPORTS_DIR = PROJECT_ROOT / "reports"


class FineTuneDataset(Dataset):
    def __init__(self, X, y, exercise_idx, n_exercises=10, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.exercise_idx = exercise_idx
        self.n_exercises = n_exercises
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            x = x + torch.randn_like(x) * 0.03
            if torch.rand(1).item() < 0.3:
                x = x * (0.9 + torch.rand(1).item() * 0.2)

        # Add exercise one-hot
        ex_onehot = torch.zeros(self.n_exercises)
        ex_onehot[self.exercise_idx] = 1.0
        ex_expanded = ex_onehot.unsqueeze(0).expand(x.shape[0], -1)
        x_with_ex = torch.cat([x, ex_expanded], dim=1)

        return x_with_ex, self.y[idx]


def finetune_exercise(exercise, pretrained_path, device, epochs=200, batch_size=64,
                      lr=1e-4, freeze_epochs=20, patience=40):
    """Fine-tune pretrained model for one exercise."""
    ex_idx = EXERCISES.index(exercise)

    # Load pretrained checkpoint
    ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
    input_dim = ckpt["input_dim"]
    hidden_dim = ckpt["hidden_dim"]

    # Build model with same architecture
    model = FormBiLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        dropout=0.3,
        use_conv=True,
    )

    # Load pretrained weights (encoder + old classifier)
    model.load_state_dict(ckpt["model_state_dict"])

    # Replace classifier head (fresh weights for per-exercise fine-tuning)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(hidden_dim * 2, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 1),
    )
    model.to(device)

    # Load exercise data
    target_seq_len = 30
    data = {}
    for part in ["train", "val", "test"]:
        path = SEQUENCES_DIR / f"{exercise}_{part}_sequences.npz"
        if not path.exists():
            print(f"  [SKIP] Missing {path.name}")
            return None
        d = np.load(path, allow_pickle=True)
        X = d["X"]
        if X.shape[2] > 99:
            X = X[:, :, :99]
        if X.shape[1] < target_seq_len:
            X = np.pad(X, ((0,0), (0, target_seq_len - X.shape[1]), (0,0)), mode='edge')
        elif X.shape[1] > target_seq_len:
            X = X[:, :target_seq_len, :]
        data[part] = (X, d["y"])

    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    n_correct = int((y_train == 1).sum())
    n_incorrect = int((y_train == 0).sum())

    print(f"\n{'='*60}")
    print(f"Fine-tuning: {exercise} (from pretrained)")
    print(f"  Train: {len(X_train)} ({n_correct} correct, {n_incorrect} incorrect)")
    print(f"  Val: {len(X_val)}, Test: {len(X_test)}")

    dl_kwargs = {"batch_size": batch_size, "pin_memory": True, "num_workers": 8,
                 "persistent_workers": True, "prefetch_factor": 4}

    train_ds = FineTuneDataset(X_train, y_train, ex_idx, augment=True)
    val_ds = FineTuneDataset(X_val, y_val, ex_idx, augment=False)
    test_ds = FineTuneDataset(X_test, y_test, ex_idx, augment=False)

    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **dl_kwargs)

    # Loss
    pos_weight = torch.tensor([n_incorrect / max(n_correct, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Phase 1: freeze encoder, train only classifier head
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=lr * 5)  # Higher LR for fresh head
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None

    for epoch in range(epochs):
        # Unfreeze encoder after freeze_epochs
        if epoch == freeze_epochs:
            print(f"  [Epoch {epoch}] Unfreezing encoder")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            scaler = torch.amp.GradScaler(enabled=use_amp)

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

        # Validate
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
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

        if epoch % 20 == 0 or improved or epoch == epochs - 1:
            m = " *" if improved else ""
            print(f"  Epoch {epoch:3d}: loss={epoch_loss/max(n_batches,1):.4f}, val_f1={val_f1:.3f}{m}")

        if epoch - best_epoch >= patience:
            print(f"  Early stopping at epoch {epoch} (best={best_epoch})")
            break

    # Load best and evaluate
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    test_probs, test_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            logits = model(X_batch)
            test_probs.extend(torch.sigmoid(logits).cpu().numpy())
            test_labels.extend(y_batch.numpy())

    y_prob = np.array(test_probs)
    y_true = np.array(test_labels)
    y_pred = (y_prob >= 0.5).astype(int)

    # Threshold search on val
    val_prob_arr = np.array(val_probs)
    val_true_arr = np.array(val_labels)
    best_thresh = 0.5
    best_f1_thresh = 0.0
    for t in np.arange(0.1, 0.9, 0.05):
        f1_t = f1_score(val_true_arr, (val_prob_arr >= t).astype(int), zero_division=0)
        if f1_t > best_f1_thresh:
            best_f1_thresh = f1_t
            best_thresh = t

    metrics = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "val_best_thresh": best_thresh,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "pretrained": True,
    }

    print(f"\n  Test Results (epoch {best_epoch}):")
    print(f"    F1:        {metrics['f1']:.3f}")
    print(f"    Precision: {metrics['precision']:.3f}")
    print(f"    Recall:    {metrics['recall']:.3f}")
    print(f"    Accuracy:  {metrics['accuracy']:.3f}")

    # Save model — compatible with existing BiLSTMClassifier loader
    # Note: input_dim is 109 (99 landmarks + 10 exercise one-hot)
    # The BiLSTMClassifier will need to add exercise one-hot at inference
    save_ckpt = {
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": 2,
        "dropout": 0.3,
        "use_conv": True,
        "best_threshold": best_thresh,
        "best_epoch": best_epoch,
        "metrics": metrics,
        "feature_names": [f"lm_{i}" for i in range(99)] + [f"ex_{ex}" for ex in EXERCISES],
        "pretrained": True,
        "exercise_idx": ex_idx,
    }

    save_path = MODELS_DIR / f"{exercise}_bilstm_v2_pretrained.pt"
    torch.save(save_ckpt, save_path)
    print(f"  Saved: {save_path}")

    eval_path = MODELS_DIR / f"{exercise}_bilstm_v2_pretrained_eval.json"
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune from pretrained cross-exercise model")
    parser.add_argument("--exercise", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=40)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    pretrained_path = MODELS_DIR / "cross_exercise_pretrained.pt"
    if not pretrained_path.exists():
        print(f"ERROR: Pretrained model not found at {pretrained_path}")
        print("Run pretrain_cross_exercise.py first.")
        return

    exercises = [args.exercise] if args.exercise else EXERCISES
    all_results = {}

    for exercise in exercises:
        metrics = finetune_exercise(
            exercise, pretrained_path, device,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, freeze_epochs=args.freeze_epochs,
            patience=args.patience,
        )
        if metrics:
            all_results[exercise] = metrics

    if all_results:
        print(f"\n{'='*60}")
        print("FINE-TUNING SUMMARY (from pretrained)")
        print(f"{'='*60}")
        f1_sum = 0
        for ex in sorted(all_results.keys()):
            m = all_results[ex]
            print(f"  {ex:20s} F1={m['f1']:.3f}  Prec={m['precision']:.3f}  Rec={m['recall']:.3f}")
            f1_sum += m["f1"]
        avg = f1_sum / len(all_results)
        print(f"  {'AVERAGE':20s} F1={avg:.3f}")

        summary_path = REPORTS_DIR / "finetune_pretrained_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
