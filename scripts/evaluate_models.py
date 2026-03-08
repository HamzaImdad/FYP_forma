"""
Comprehensive model evaluation with plots and reports.

Loads trained models, evaluates on test data, generates:
- Per-exercise confusion matrix heatmaps
- F1 score bar chart across exercises
- ROC curves per exercise
- Full classification report

Usage:
    python scripts/evaluate_models.py
    python scripts/evaluate_models.py --data augmented_features.csv
"""

import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import EXERCISES

FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
MODELS_DIR = PROJECT_ROOT / "models" / "trained"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_test_data(features_path: Path, exercise: str) -> tuple:
    """Load and split data, return test set matching train_classifier.py group splits."""
    df = pd.read_csv(features_path)
    df = df[df["exercise"] == exercise].copy()

    if len(df) == 0:
        return None, None, []

    df["label_encoded"] = (df["label"] == "correct").astype(int)

    meta_cols = {"exercise", "label", "source", "label_encoded", "video_id"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].fillna(0).values
    y = df["label_encoded"].values

    if len(np.unique(y)) < 2:
        return None, None, feature_cols

    # Group-based split matching train_classifier.py (by video_id)
    if "video_id" in df.columns:
        groups = df["video_id"].values
        n_unique = len(np.unique(groups))

        if n_unique < 3:
            return None, None, feature_cols

        gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
        train_idx, temp_idx = next(gss.split(X, y, groups=groups))

        X_temp, y_temp = X[temp_idx], y[temp_idx]
        groups_temp = groups[temp_idx]

        if len(np.unique(groups_temp)) >= 2:
            gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
            val_idx, test_idx = next(gss2.split(X_temp, y_temp, groups=groups_temp))
            X_test, y_test = X_temp[test_idx], y_temp[test_idx]
        else:
            X_test, y_test = X_temp, y_temp
    else:
        # Fallback for old CSVs without video_id
        from sklearn.model_selection import train_test_split
        _, X_temp, _, y_temp = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=42
        )
        _, X_test, _, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
        )

    return X_test, y_test, feature_cols


def plot_confusion_matrix(y_true, y_pred, exercise, model_name, save_path):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Incorrect", "Correct"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"{exercise.replace('_', ' ').title()} - {model_name}")
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    # Add text annotations
    for i in range(2):
        for j in range(2):
            val = cm[i, j] if i < len(cm) and j < len(cm[i]) else 0
            ax.text(j, i, str(val), ha="center", va="center",
                    color="white" if val > cm.max() / 2 else "black", fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def plot_roc_curve(y_true, y_proba, exercise, model_name, save_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC - {exercise.replace('_', ' ').title()}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

    return roc_auc


def plot_f1_comparison(results, save_path):
    """Bar chart of F1 scores across exercises."""
    exercises = sorted(results.keys())
    f1_scores = [results[ex]["f1"] for ex in exercises]
    labels = [ex.replace("_", " ").title() for ex in exercises]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2ecc71" if f >= 0.7 else "#e74c3c" for f in f1_scores]
    bars = ax.bar(range(len(exercises)), f1_scores, color=colors)

    ax.set_xticks(range(len(exercises)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Exercise F1 Scores (Best Model)")
    ax.set_ylim([0, 1.05])
    ax.axhline(y=0.7, color="orange", linestyle="--", alpha=0.5, label="Target (0.7)")
    ax.legend()

    # Add value labels
    for bar, val in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--data", type=str, default=None,
                        help="Feature CSV filename (in features dir)")
    args = parser.parse_args()

    if args.data:
        features_path = FEATURES_DIR / args.data
    else:
        # Try augmented first, then all, then image
        for name in ["augmented_features.csv", "all_features.csv", "image_features.csv"]:
            features_path = FEATURES_DIR / name
            if features_path.exists():
                break

    if not features_path.exists():
        print(f"No feature CSV found. Run build_features.py first.")
        sys.exit(1)

    print(f"Evaluating models using: {features_path.name}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Reports directory: {REPORTS_DIR}\n")

    all_results = {}
    full_report = []

    for exercise in EXERCISES:
        # Find best model
        model_path = MODELS_DIR / f"{exercise}_classifier.pkl"
        if not model_path.exists():
            print(f"[SKIP] No model for {exercise}")
            continue

        X_test, y_test, feature_cols = load_test_data(features_path, exercise)
        if X_test is None or len(X_test) == 0:
            print(f"[SKIP] No test data for {exercise}")
            continue

        # Load model
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Predict
        y_pred = model.predict(X_test)

        # Get probabilities if available
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "test_samples": len(y_test),
            "correct_in_test": int(y_test.sum()),
            "incorrect_in_test": int(len(y_test) - y_test.sum()),
        }

        # Find which model type it is
        model_name = type(model).__name__
        if hasattr(model, "named_steps"):
            clf_step = model.named_steps.get("clf", model)
            model_name = type(clf_step).__name__

        metrics["model_type"] = model_name
        all_results[exercise] = metrics

        print(f"\n{'='*50}")
        print(f"Exercise: {exercise} ({model_name})")
        print(f"  Test samples: {metrics['test_samples']} "
              f"(correct: {metrics['correct_in_test']}, incorrect: {metrics['incorrect_in_test']})")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1:        {metrics['f1']:.3f}")

        # Classification report
        report = classification_report(y_test, y_pred,
                                       target_names=["Incorrect", "Correct"],
                                       zero_division=0)
        print(f"\n{report}")
        full_report.append(f"=== {exercise} ({model_name}) ===\n{report}\n")

        # Confusion matrix plot
        cm_path = REPORTS_DIR / f"{exercise}_confusion_matrix.png"
        plot_confusion_matrix(y_test, y_pred, exercise, model_name, cm_path)

        # ROC curve
        if y_proba is not None:
            roc_path = REPORTS_DIR / f"{exercise}_roc_curve.png"
            roc_auc = plot_roc_curve(y_test, y_proba, exercise, model_name, roc_path)
            metrics["auc"] = roc_auc
            print(f"  AUC:       {roc_auc:.3f}")

        # Save per-exercise eval JSON
        eval_path = REPORTS_DIR / f"{exercise}_evaluation.json"
        with open(eval_path, "w") as f:
            json.dump(metrics, f, indent=2)

    if not all_results:
        print("\nNo models found to evaluate. Run train_classifier.py first.")
        sys.exit(1)

    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Exercise':<18} {'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("-" * 70)
    for ex in sorted(all_results.keys()):
        m = all_results[ex]
        print(f"{ex:<18} {m['model_type']:<22} {m['accuracy']:>6.3f} "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}")

    # Averages
    avg = {
        k: np.mean([m[k] for m in all_results.values()])
        for k in ["accuracy", "precision", "recall", "f1"]
    }
    print("-" * 70)
    print(f"{'AVERAGE':<18} {'':22} {avg['accuracy']:>6.3f} "
          f"{avg['precision']:>6.3f} {avg['recall']:>6.3f} {avg['f1']:>6.3f}")

    # F1 comparison plot
    f1_path = REPORTS_DIR / "f1_comparison.png"
    plot_f1_comparison(all_results, f1_path)

    # Save full report
    report_path = REPORTS_DIR / "full_evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("ExerVision Model Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        for r in full_report:
            f.write(r)
        f.write("\n\nOverall Summary\n")
        f.write("-" * 60 + "\n")
        for ex in sorted(all_results.keys()):
            m = all_results[ex]
            f.write(f"{ex}: F1={m['f1']:.3f}, Acc={m['accuracy']:.3f}, "
                    f"Model={m['model_type']}\n")

    # Save summary JSON
    summary_path = REPORTS_DIR / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nReports saved to: {REPORTS_DIR}")
    print(f"  - Confusion matrices: {exercise}_confusion_matrix.png")
    print(f"  - ROC curves: {exercise}_roc_curve.png")
    print(f"  - F1 comparison: f1_comparison.png")
    print(f"  - Full report: full_evaluation_report.txt")
    print(f"  - Summary JSON: evaluation_summary.json")

    # Flag weak exercises
    weak = [ex for ex, m in all_results.items() if m["f1"] < 0.7]
    if weak:
        print(f"\n[WARNING] Exercises with F1 < 0.7: {', '.join(weak)}")
        print("Consider: more data, hyperparameter tuning, or checking data quality")


if __name__ == "__main__":
    main()
