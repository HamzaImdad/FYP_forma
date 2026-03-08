"""
Evaluation metrics for form classification.
Accuracy, precision, recall, F1, confusion matrices, per-exercise breakdown.
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List = None,
) -> Dict[str, float]:
    """
    Compute standard classification metrics.

    Returns dict with: accuracy, precision, recall, f1
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
    }


def confusion_matrix_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = None,
) -> str:
    """
    Generate a formatted confusion matrix string.
    """
    if labels is None:
        labels = ["incorrect", "correct"]
    cm = confusion_matrix(y_true, y_pred)
    lines = [f"{'':>12} {'Pred Incorrect':>16} {'Pred Correct':>14}"]
    for i, label in enumerate(labels):
        row = cm[i] if i < len(cm) else [0, 0]
        lines.append(f"{label:>12} {row[0]:>16} {row[1] if len(row) > 1 else 0:>14}")
    return "\n".join(lines)


def per_exercise_report(
    results: Dict[str, Dict[str, float]],
) -> str:
    """
    Format per-exercise metrics into a readable table.

    Args:
        results: Dict of exercise_name -> {accuracy, precision, recall, f1}
    """
    header = f"{'Exercise':<18} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
    lines = [header, "-" * len(header)]

    for exercise, metrics in sorted(results.items()):
        lines.append(
            f"{exercise:<18} {metrics['accuracy']:>10.3f} {metrics['precision']:>10.3f} "
            f"{metrics['recall']:>10.3f} {metrics['f1']:>10.3f}"
        )

    # Average
    if results:
        avg = {
            k: np.mean([m[k] for m in results.values()])
            for k in ["accuracy", "precision", "recall", "f1"]
        }
        lines.append("-" * len(header))
        lines.append(
            f"{'AVERAGE':<18} {avg['accuracy']:>10.3f} {avg['precision']:>10.3f} "
            f"{avg['recall']:>10.3f} {avg['f1']:>10.3f}"
        )

    return "\n".join(lines)


def compare_models(
    model_results: Dict[str, Dict[str, float]],
) -> str:
    """
    Compare multiple models side-by-side.

    Args:
        model_results: Dict of model_name -> {accuracy, precision, recall, f1}
    """
    header = f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
    lines = [header, "-" * len(header)]

    for model_name, metrics in model_results.items():
        lines.append(
            f"{model_name:<25} {metrics['accuracy']:>10.3f} {metrics['precision']:>10.3f} "
            f"{metrics['recall']:>10.3f} {metrics['f1']:>10.3f}"
        )

    return "\n".join(lines)
