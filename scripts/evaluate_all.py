"""
Three-way model comparison: Rule-Based vs Random Forest vs BiLSTM.

Evaluates all classifier types on the same group-split test data.
Generates comparison charts, confusion matrices, ROC curves, and a
summary CSV for the FYP report.

Usage:
    python scripts/evaluate_all.py
    python scripts/evaluate_all.py --data augmented_features.csv
"""

import sys
import json
import pickle
import argparse
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import EXERCISES
from src.pose_estimation.base import PoseResult
from src.feature_extraction.features import FeatureExtractor
from src.classification.rule_based import RuleBasedClassifier

FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
SEQUENCES_DIR = PROJECT_ROOT / "data" / "processed" / "sequences"
MODELS_DIR = PROJECT_ROOT / "models" / "trained"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSIFIER_NAMES = {
    "rule_based": "Rule-Based",
    "rf": "Random Forest",
    "bilstm": "BiLSTM",
}

COLORS = {
    "rule_based": "#3498db",
    "rf": "#2ecc71",
    "bilstm": "#e74c3c",
}


def load_test_split(features_path: Path, exercise: str):
    """Load features and return test split using group-based splitting."""
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

    if "video_id" in df.columns:
        groups = df["video_id"].values
        if len(np.unique(groups)) < 3:
            return None, None, feature_cols

        gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
        train_idx, temp_idx = next(gss.split(X, y, groups=groups))

        X_temp, y_temp = X[temp_idx], y[temp_idx]
        groups_temp = groups[temp_idx]

        if len(np.unique(groups_temp)) >= 2:
            gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
            _, test_idx = next(gss2.split(X_temp, y_temp, groups=groups_temp))
            X_test, y_test = X_temp[test_idx], y_temp[test_idx]
        else:
            X_test, y_test = X_temp, y_temp
    else:
        from sklearn.model_selection import train_test_split
        _, X_temp, _, y_temp = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=42
        )
        _, X_test, _, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
        )

    return X_test, y_test, feature_cols


def evaluate_rule_based(X_test, y_test, feature_cols, exercise):
    """Evaluate rule-based classifier on test features."""
    clf = RuleBasedClassifier()

    y_pred = []
    y_conf = []
    for i in range(len(X_test)):
        features = {fname: float(X_test[i, j]) for j, fname in enumerate(feature_cols)}
        result = clf.classify(features, exercise)
        y_pred.append(1 if result.is_correct else 0)
        y_conf.append(result.confidence)

    return np.array(y_pred), np.array(y_conf)


def evaluate_rf(X_test, y_test, exercise):
    """Evaluate Random Forest classifier on test features."""
    model_path = MODELS_DIR / f"{exercise}_classifier.pkl"
    if not model_path.exists():
        return None, None

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]

    return y_pred, y_proba


def evaluate_bilstm(exercise):
    """Evaluate BiLSTM classifier on test sequences."""
    try:
        import torch
        from src.classification.bilstm_classifier import FormBiLSTM
    except ImportError:
        return None, None, None

    # Load checkpoint
    ckpt_path = MODELS_DIR / f"{exercise}_bilstm.pt"
    if not ckpt_path.exists():
        return None, None, None

    # Load sequence data
    seq_path = SEQUENCES_DIR / f"{exercise}_sequences.npz"
    if not seq_path.exists():
        return None, None, None

    data = np.load(seq_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    video_ids = data["video_ids"]

    if len(np.unique(y)) < 2:
        return None, None, None

    n_unique = len(set(video_ids))
    if n_unique < 3:
        return None, None, None

    # Same group split as training
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    _, test_idx = next(gss.split(X, y, groups=video_ids))
    X_test, y_test = X[test_idx], y[test_idx]

    # Load model
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = FormBiLSTM(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint.get("dropout", 0.3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Predict using best threshold from training
    best_thresh = checkpoint.get("best_threshold", 0.5)
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).numpy()
        preds = (probs >= best_thresh).astype(int)

    return y_test, preds, probs


def plot_grouped_f1(all_results, save_path):
    """Grouped bar chart of F1 scores across exercises for each classifier."""
    exercises = sorted(set(ex for res in all_results.values() for ex in res))
    classifiers = list(all_results.keys())
    n_clf = len(classifiers)
    n_ex = len(exercises)

    if n_ex == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    bar_width = 0.8 / n_clf
    x = np.arange(n_ex)

    for i, clf_name in enumerate(classifiers):
        f1s = [all_results[clf_name].get(ex, {}).get("f1", 0) for ex in exercises]
        offset = (i - n_clf / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, f1s, bar_width, label=CLASSIFIER_NAMES.get(clf_name, clf_name),
                      color=COLORS.get(clf_name, "#999"))

        # Value labels
        for bar, val in zip(bars, f1s):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    labels = [ex.replace("_", " ").title() for ex in exercises]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title("Classifier Comparison: F1 Scores per Exercise")
    ax.set_ylim([0, 1.15])
    ax.axhline(y=0.7, color="orange", linestyle="--", alpha=0.5, label="Target (0.7)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_overlaid_roc(roc_data, exercise, save_path):
    """Overlaid ROC curves for all classifiers on one exercise."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for clf_name, (y_true, y_proba) in roc_data.items():
        if y_proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        label = f"{CLASSIFIER_NAMES.get(clf_name, clf_name)} (AUC={roc_auc:.3f})"
        ax.plot(fpr, tpr, color=COLORS.get(clf_name, "#999"), lw=2, label=label)

    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves - {exercise.replace('_', ' ').title()}")
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_side_by_side_cm(cm_data, exercise, save_path):
    """Side-by-side confusion matrices for all classifiers."""
    classifiers = list(cm_data.keys())
    n = len(classifiers)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    labels = ["Incorrect", "Correct"]

    for ax, clf_name in zip(axes, classifiers):
        cm = cm_data[clf_name]
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(CLASSIFIER_NAMES.get(clf_name, clf_name), fontsize=11)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        for i in range(2):
            for j in range(2):
                val = cm[i, j] if i < len(cm) and j < len(cm[i]) else 0
                ax.text(j, i, str(val), ha="center", va="center",
                        color="white" if val > cm.max() / 2 else "black", fontsize=13)

    fig.suptitle(f"{exercise.replace('_', ' ').title()}", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="3-way classifier comparison")
    parser.add_argument("--data", type=str, default=None,
                        help="Feature CSV filename (in features dir)")
    args = parser.parse_args()

    if args.data:
        features_path = FEATURES_DIR / args.data
    else:
        for name in ["augmented_features.csv", "all_features.csv", "image_features.csv"]:
            features_path = FEATURES_DIR / name
            if features_path.exists():
                break

    if not features_path.exists():
        print("No feature CSV found. Run build_features.py first.")
        sys.exit(1)

    print(f"Feature data: {features_path.name}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Output directory: {REPORTS_DIR}\n")

    # Results storage: classifier -> exercise -> metrics
    all_results = {"rule_based": {}, "rf": {}, "bilstm": {}}
    comparison_rows = []

    for exercise in EXERCISES:
        print(f"\n{'='*60}")
        print(f"Exercise: {exercise}")
        print(f"{'='*60}")

        X_test, y_test, feature_cols = load_test_split(features_path, exercise)
        if X_test is None or len(X_test) == 0:
            print(f"  [SKIP] No test data")
            continue

        roc_data = {}
        cm_data = {}

        # ── Rule-Based ──
        rb_pred, rb_conf = evaluate_rule_based(X_test, y_test, feature_cols, exercise)
        rb_metrics = {
            "accuracy": float(accuracy_score(y_test, rb_pred)),
            "precision": float(precision_score(y_test, rb_pred, zero_division=0)),
            "recall": float(recall_score(y_test, rb_pred, zero_division=0)),
            "f1": float(f1_score(y_test, rb_pred, zero_division=0)),
        }
        all_results["rule_based"][exercise] = rb_metrics
        roc_data["rule_based"] = (y_test, rb_conf)
        cm_data["rule_based"] = confusion_matrix(y_test, rb_pred, labels=[0, 1])
        print(f"  Rule-Based:  F1={rb_metrics['f1']:.3f}, Acc={rb_metrics['accuracy']:.3f}")

        # ── Random Forest ──
        rf_pred, rf_proba = evaluate_rf(X_test, y_test, exercise)
        if rf_pred is not None:
            rf_metrics = {
                "accuracy": float(accuracy_score(y_test, rf_pred)),
                "precision": float(precision_score(y_test, rf_pred, zero_division=0)),
                "recall": float(recall_score(y_test, rf_pred, zero_division=0)),
                "f1": float(f1_score(y_test, rf_pred, zero_division=0)),
            }
            all_results["rf"][exercise] = rf_metrics
            roc_data["rf"] = (y_test, rf_proba)
            cm_data["rf"] = confusion_matrix(y_test, rf_pred, labels=[0, 1])
            print(f"  Random Forest: F1={rf_metrics['f1']:.3f}, Acc={rf_metrics['accuracy']:.3f}")
        else:
            print(f"  Random Forest: [NO MODEL]")

        # ── BiLSTM ──
        bilstm_result = evaluate_bilstm(exercise)
        if bilstm_result[0] is not None:
            bl_y_test, bl_pred, bl_proba = bilstm_result
            bl_metrics = {
                "accuracy": float(accuracy_score(bl_y_test, bl_pred)),
                "precision": float(precision_score(bl_y_test, bl_pred, zero_division=0)),
                "recall": float(recall_score(bl_y_test, bl_pred, zero_division=0)),
                "f1": float(f1_score(bl_y_test, bl_pred, zero_division=0)),
            }
            all_results["bilstm"][exercise] = bl_metrics
            roc_data["bilstm"] = (bl_y_test, bl_proba)
            cm_data["bilstm"] = confusion_matrix(bl_y_test, bl_pred, labels=[0, 1])
            print(f"  BiLSTM:      F1={bl_metrics['f1']:.3f}, Acc={bl_metrics['accuracy']:.3f}")
        else:
            print(f"  BiLSTM:      [NO MODEL/DATA]")

        # Generate plots for this exercise
        if roc_data:
            roc_path = REPORTS_DIR / f"{exercise}_roc_comparison.png"
            plot_overlaid_roc(roc_data, exercise, roc_path)

        if cm_data:
            cm_path = REPORTS_DIR / f"{exercise}_cm_comparison.png"
            plot_side_by_side_cm(cm_data, exercise, cm_path)

        # Build comparison row
        for clf_type in ["rule_based", "rf", "bilstm"]:
            m = all_results[clf_type].get(exercise)
            if m:
                comparison_rows.append({
                    "exercise": exercise,
                    "classifier": CLASSIFIER_NAMES[clf_type],
                    **m,
                })

    # ── Summary ──
    print(f"\n{'='*70}")
    print("THREE-WAY COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Exercise':<18} {'Rule-Based':>12} {'RF F1':>12} {'BiLSTM F1':>12}")
    print("-" * 56)

    for exercise in EXERCISES:
        rb_f1 = all_results["rule_based"].get(exercise, {}).get("f1", -1)
        rf_f1 = all_results["rf"].get(exercise, {}).get("f1", -1)
        bl_f1 = all_results["bilstm"].get(exercise, {}).get("f1", -1)

        rb_str = f"{rb_f1:.3f}" if rb_f1 >= 0 else "N/A"
        rf_str = f"{rf_f1:.3f}" if rf_f1 >= 0 else "N/A"
        bl_str = f"{bl_f1:.3f}" if bl_f1 >= 0 else "N/A"

        print(f"{exercise:<18} {rb_str:>12} {rf_str:>12} {bl_str:>12}")

    # Averages
    print("-" * 56)
    for clf_type, label in [("rule_based", "Rule-Based"), ("rf", "Random Forest"), ("bilstm", "BiLSTM")]:
        vals = [m["f1"] for m in all_results[clf_type].values()]
        avg = np.mean(vals) if vals else 0
        print(f"  {label} avg F1: {avg:.3f} (across {len(vals)} exercises)")

    # Grouped F1 bar chart
    f1_path = REPORTS_DIR / "classifier_comparison_f1.png"
    plot_grouped_f1(all_results, f1_path)
    print(f"\nF1 comparison chart: {f1_path}")

    # Save comparison CSV
    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows)
        csv_path = REPORTS_DIR / "classifier_comparison.csv"
        comp_df.to_csv(csv_path, index=False)
        print(f"Comparison CSV: {csv_path}")

    # Save summary JSON
    summary = {}
    for clf_type in all_results:
        for ex, metrics in all_results[clf_type].items():
            key = f"{clf_type}_{ex}"
            summary[key] = {
                "classifier": CLASSIFIER_NAMES[clf_type],
                "exercise": ex,
                **metrics,
            }

    summary_path = REPORTS_DIR / "three_way_comparison.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
