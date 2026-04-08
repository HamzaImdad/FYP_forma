"""
Re-evaluate existing BiLSTM v2 models at optimal thresholds.

The current v2 models were all evaluated at threshold=0.5, but the optimal
thresholds found on validation range from 0.10 to 0.55. This script sweeps
thresholds on the test set to find the true F1 at each operating point.

This is a READ-ONLY evaluation — no models are modified or retrained.

Output: reports/{exercise}_threshold_analysis.json
        Console comparison table

Usage:
    python scripts/evaluate_thresholds.py
    python scripts/evaluate_thresholds.py --exercise squat bicep_curl
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

# ── Project setup ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.bilstm_classifier import FormBiLSTM

# ── Paths ──
MODELS_DIR = PROJECT_ROOT / "models" / "trained"
SEQUENCES_DIR = PROJECT_ROOT / "data" / "processed" / "sequences"
REPORTS_DIR = PROJECT_ROOT / "reports"

EXERCISES = [
    "squat", "deadlift", "lunge", "bench_press", "overhead_press",
    "pullup", "pushup", "plank", "bicep_curl", "tricep_dip",
]


def load_model(exercise: str, device: torch.device):
    """Load a v2 BiLSTM model from checkpoint."""
    model_path = MODELS_DIR / f"{exercise}_bilstm_v2.pt"
    if not model_path.exists():
        return None, None

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = FormBiLSTM(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint.get("hidden_dim", 128),
        num_layers=checkpoint.get("num_layers", 2),
        dropout=checkpoint.get("dropout", 0.3),
        use_conv=checkpoint.get("use_conv", True),
    )
    # Handle SWA-wrapped checkpoints (module. prefix)
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()
                      if not k == "n_averaged"}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, checkpoint


def evaluate_at_threshold(y_true, y_prob, threshold):
    """Compute metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": round(threshold, 3),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
    }


def evaluate_exercise(exercise: str, device: torch.device) -> dict:
    """Evaluate a single exercise model at multiple thresholds."""
    model, checkpoint = load_model(exercise, device)
    if model is None:
        return {"exercise": exercise, "error": "model not found"}

    # Load test sequences — match model's input_dim
    model_dim = checkpoint["input_dim"]
    test_candidates = [
        SEQUENCES_DIR / f"{exercise}_test_sequences.npz",
        SEQUENCES_DIR / f"{exercise}_test_hybrid_sequences.npz",
    ]
    test_path = None
    for candidate in test_candidates:
        if candidate.exists():
            data = np.load(candidate)
            if data["X"].shape[2] == model_dim:
                test_path = candidate
                break

    if test_path is None:
        return {"exercise": exercise, "error": f"no test sequences with dim={model_dim}"}

    X_test = torch.tensor(data["X"], dtype=torch.float32)
    y_test = data["y"]

    # Run inference
    all_probs = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i + batch_size].to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)

    y_prob = np.array(all_probs)
    y_true = y_test.astype(int)

    # Sweep thresholds
    thresholds = np.arange(0.05, 0.96, 0.05)
    sweep_results = []
    best_f1 = 0
    best_thresh = 0.5

    for thresh in thresholds:
        result = evaluate_at_threshold(y_true, y_prob, thresh)
        sweep_results.append(result)
        if result["f1"] > best_f1:
            best_f1 = result["f1"]
            best_thresh = thresh

    # Get specific results
    val_best_thresh = checkpoint.get("best_threshold", 0.5)
    result_at_05 = evaluate_at_threshold(y_true, y_prob, 0.5)
    result_at_val_best = evaluate_at_threshold(y_true, y_prob, val_best_thresh)
    result_at_test_optimal = evaluate_at_threshold(y_true, y_prob, best_thresh)

    return {
        "exercise": exercise,
        "n_test_sequences": len(y_true),
        "n_correct": int((y_true == 1).sum()),
        "n_incorrect": int((y_true == 0).sum()),
        "val_best_threshold": round(val_best_thresh, 3),
        "test_optimal_threshold": round(best_thresh, 3),
        "f1_at_0.5": result_at_05,
        "f1_at_val_best": result_at_val_best,
        "f1_at_test_optimal": result_at_test_optimal,
        "threshold_sweep": sweep_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate v2 models at optimal thresholds")
    parser.add_argument("--exercise", nargs="+", default=None,
                        help="Specific exercises (default: all)")
    args = parser.parse_args()

    exercises = args.exercise or EXERCISES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for exercise in exercises:
        result = evaluate_exercise(exercise, device)
        all_results[exercise] = result

        # Save individual report
        report_path = REPORTS_DIR / f"{exercise}_threshold_analysis.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

    # Print comparison table
    print()
    print("=" * 110)
    print(f"{'Exercise':<18} {'F1@0.5':>8} {'ValBestTh':>10} {'F1@ValBest':>11} {'TestOptTh':>10} {'F1@TestOpt':>11} {'Gain':>7}")
    print("-" * 110)

    gains = []
    for ex in exercises:
        r = all_results.get(ex, {})
        if "error" in r:
            print(f"{ex:<18} ERROR: {r['error']}")
            continue

        f1_05 = r["f1_at_0.5"]["f1"]
        f1_val = r["f1_at_val_best"]["f1"]
        f1_opt = r["f1_at_test_optimal"]["f1"]
        val_th = r["val_best_threshold"]
        opt_th = r["test_optimal_threshold"]
        gain = f1_opt - f1_05

        print(f"{ex:<18} {f1_05:>8.4f} {val_th:>10.3f} {f1_val:>11.4f} {opt_th:>10.3f} {f1_opt:>11.4f} {gain:>+7.4f}")
        gains.append(gain)

    print("-" * 110)
    if gains:
        print(f"{'Average':>18} {' ':>8} {' ':>10} {' ':>11} {' ':>10} {' ':>11} {np.mean(gains):>+7.4f}")
    print("=" * 110)

    # Save combined report
    combined_path = REPORTS_DIR / "threshold_analysis_all.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined report saved to {combined_path}")


if __name__ == "__main__":
    main()
