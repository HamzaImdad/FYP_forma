"""
Optuna hyperparameter search for BiLSTM v2 models.

Runs Bayesian optimization to find optimal architecture and training
hyperparameters per exercise. Uses MedianPruner for early stopping of
bad trials.

Usage:
    python scripts/optuna_search.py --exercise squat --n-trials 50
    python scripts/optuna_search.py --n-trials 30  (all exercises)
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import optuna
    from optuna.exceptions import TrialPruned
except ImportError:
    print("ERROR: optuna is required. Install with: pip install optuna")
    sys.exit(1)

from src.classification.bilstm_classifier import FormBiLSTM
from src.utils.constants import EXERCISES

SEQUENCES_DIR = PROJECT_ROOT / "data" / "processed" / "sequences"
MODELS_DIR = PROJECT_ROOT / "models" / "trained"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Reuse FocalLoss and SequenceDataset from train_bilstm.py
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced binary classification."""

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_smooth = targets * (1 - 0.1) + 0.1 * 0.5
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets_smooth, reduction="none",
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_weight * bce_loss
        return loss.mean()


class SequenceDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for exercise form sequences with optional augmentation."""

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            # Gaussian noise
            x = x + torch.randn_like(x) * 0.05
            # Random scaling
            if torch.rand(1).item() < 0.5:
                scale = 0.8 + torch.rand(1).item() * 0.4
                x = x * scale
            # Joint dropout
            if torch.rand(1).item() < 0.2:
                n_drop = int(torch.randint(1, 3, (1,)).item())
                drop_ids = torch.randperm(33)[:n_drop]
                for lid in drop_ids:
                    x[:, lid*3:(lid+1)*3] = 0.0
            # Random translation
            if torch.rand(1).item() < 0.4:
                dx = (torch.rand(1).item() - 0.5) * 0.2
                dy = (torch.rand(1).item() - 0.5) * 0.2
                for i in range(33):
                    x[:, i*3] += dx
                    x[:, i*3 + 1] += dy
            # Left-right mirror
            if torch.rand(1).item() < 0.3:
                for i in range(33):
                    x[:, i*3] = -x[:, i*3]
                swap_pairs = [
                    (1,4),(2,5),(3,6),(7,8),(9,10),
                    (11,12),(13,14),(15,16),(17,18),(19,20),
                    (21,22),(23,24),(25,26),(27,28),(29,30),(31,32),
                ]
                for a, b in swap_pairs:
                    tmp = x[:, a*3:(a+1)*3].clone()
                    x[:, a*3:(a+1)*3] = x[:, b*3:(b+1)*3]
                    x[:, b*3:(b+1)*3] = tmp
        return x, self.y[idx]


# ---------------------------------------------------------------------------
# Data loading (same as train_bilstm.py)
# ---------------------------------------------------------------------------

def load_presplit_sequences(exercise: str) -> tuple:
    """Load pre-split train/val/test .npz files for an exercise."""
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


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def create_objective(exercise: str, device: torch.device, max_epochs: int = 100):
    """Create an Optuna objective function for the given exercise."""

    # Load data once (shared across all trials)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = \
        load_presplit_sequences(exercise)

    if X_train is None:
        raise FileNotFoundError(
            f"Pre-split sequences not found for {exercise}. "
            "Run build_sequences.py first."
        )

    input_dim = X_train.shape[2]
    print(f"\n{'='*60}")
    print(f"Exercise: {exercise}")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Input dim: {input_dim}, Seq len: {X_train.shape[1]}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    def objective(trial: optuna.Trial) -> float:
        # --- Suggest hyperparameters ---
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 192, 256])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.2, 0.5, step=0.05)
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        n_heads = trial.suggest_categorical("n_heads", [1, 2])

        # --- Build model ---
        model = FormBiLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_conv=True,
            n_heads=n_heads,
        ).to(device)

        # --- DataLoaders ---
        train_ds = SequenceDataset(X_train, y_train, augment=True)
        val_ds = SequenceDataset(X_val, y_val, augment=False)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True,
                                  persistent_workers=True, prefetch_factor=2)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=True,
                                persistent_workers=True, prefetch_factor=2)

        # --- Optimizer and scheduler ---
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = FocalLoss(alpha=0.5, gamma=2.0)

        warmup_epochs = 5
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(max_epochs - warmup_epochs, 1), eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        # --- AMP scaler ---
        use_amp = device.type == "cuda"
        scaler = GradScaler(enabled=use_amp)

        # --- Training loop ---
        best_val_f1 = 0.0
        patience = 20
        no_improve = 0

        for epoch in range(max_epochs):
            # Train
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=use_amp):
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            scheduler.step()

            # Validate
            model.eval()
            val_probs = []
            val_labels = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device, non_blocking=True)
                    with autocast(enabled=use_amp):
                        logits = model(X_batch)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_probs.extend(probs)
                    val_labels.extend(y_batch.numpy())

            val_preds = (np.array(val_probs) >= 0.5).astype(int)
            val_f1 = f1_score(np.array(val_labels), val_preds, zero_division=0)

            # Report to Optuna for pruning
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise TrialPruned()

            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        return best_val_f1

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for BiLSTM v2")
    parser.add_argument("--exercise", type=str, default=None,
                        help="Exercise name (default: all exercises)")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of Optuna trials per exercise (default: 50)")
    parser.add_argument("--max-epochs", type=int, default=100,
                        help="Max epochs per trial (default: 100)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu, default: auto)")
    args = parser.parse_args()

    # Device — maximize all hardware
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.cuda.set_per_process_memory_fraction(0.95)
        except Exception:
            pass
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"TF32: enabled, cuDNN benchmark: enabled")
    else:
        device = torch.device("cpu")

    exercises = [args.exercise] if args.exercise else list(EXERCISES)

    all_results = {}

    for exercise in exercises:
        print(f"\n{'#'*60}")
        print(f"# Optuna search: {exercise} ({args.n_trials} trials)")
        print(f"{'#'*60}")

        try:
            objective = create_objective(exercise, device, max_epochs=args.max_epochs)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        # Create study with MedianPruner
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
            ),
            study_name=f"{exercise}_bilstm_v2",
        )

        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

        # Best trial results
        best = study.best_trial
        print(f"\n  Best trial #{best.number}:")
        print(f"    Val F1:     {best.value:.4f}")
        print(f"    Params:")
        for k, v in best.params.items():
            print(f"      {k}: {v}")

        # Save best params
        result = {
            "exercise": exercise,
            "best_val_f1": best.value,
            "best_params": best.params,
            "n_trials": len(study.trials),
            "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        }
        all_results[exercise] = result

        out_path = REPORTS_DIR / f"{exercise}_optuna_best.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_path}")

    # Summary table
    if all_results:
        print(f"\n{'='*60}")
        print(f"{'Exercise':<20} {'Best F1':>8} {'Trials':>8} {'Pruned':>8}")
        print(f"{'-'*60}")
        for ex, res in all_results.items():
            print(f"{ex:<20} {res['best_val_f1']:>8.4f} {res['n_complete']:>8} {res['n_pruned']:>8}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
