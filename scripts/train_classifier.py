"""
Train per-exercise classifiers on extracted features.

Trains Random Forest, SVM, and Logistic Regression models for each exercise.
Uses GROUP-BASED train/val/test splits by video_id to prevent data leakage.
Performs GroupKFold cross-validation.

Usage:
    python scripts/train_classifier.py
    python scripts/train_classifier.py --exercise squat
    python scripts/train_classifier.py --model random_forest
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    cross_val_score, GroupKFold, GroupShuffleSplit,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import EXERCISES
from src.evaluation.metrics import evaluate_classifier, per_exercise_report, compare_models

FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
MODELS_DIR = PROJECT_ROOT / "models" / "trained"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Available model types
MODEL_CONFIGS = {
    "random_forest": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ]),
    "svm": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(LinearSVC(max_iter=5000, random_state=42), cv=3)),
    ]),
    "logistic_regression": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=42)),
    ]),
}


def load_features(exercise: str = None, data_path: str = None) -> pd.DataFrame:
    """Load feature CSV(s). Returns combined DataFrame."""
    if data_path:
        path = Path(data_path) if Path(data_path).is_absolute() else FEATURES_DIR / data_path
        df = pd.read_csv(path)
    else:
        # Prefer augmented, then all, then individual
        for name in ["augmented_features.csv", "all_features.csv"]:
            path = FEATURES_DIR / name
            if path.exists():
                df = pd.read_csv(path)
                print(f"Loaded features from: {path.name}")
                break
        else:
            dfs = []
            for f in FEATURES_DIR.glob("*_features.csv"):
                dfs.append(pd.read_csv(f))
            if not dfs:
                print("No feature CSVs found. Run build_features.py first.")
                sys.exit(1)
            df = pd.concat(dfs, ignore_index=True)

    if exercise:
        df = df[df["exercise"] == exercise]

    return df


def train_exercise(
    df: pd.DataFrame,
    exercise: str,
    model_types: list = None,
    max_samples: int = 50000,
) -> dict:
    """
    Train and evaluate models for a single exercise.
    Uses group-based splitting by video_id to prevent data leakage.
    Returns dict of model_name -> metrics.
    """
    if model_types is None:
        model_types = list(MODEL_CONFIGS.keys())

    # Encode labels: correct=1, incorrect=0
    df = df.copy()
    df["label_encoded"] = (df["label"] == "correct").astype(int)

    # Subsample if too large, but keep all frames within each video together
    if max_samples > 0 and len(df) > max_samples:
        # Sample at video level to maintain group integrity
        video_labels = df.groupby("video_id")["label_encoded"].first()
        correct_vids = video_labels[video_labels == 1].index.tolist()
        incorrect_vids = video_labels[video_labels == 0].index.tolist()

        np.random.seed(42)
        n_vids = max(10, max_samples // df.groupby("video_id").size().median().astype(int))
        n_per_class = n_vids // 2
        sampled_vids = (
            list(np.random.choice(correct_vids, min(len(correct_vids), n_per_class), replace=False))
            + list(np.random.choice(incorrect_vids, min(len(incorrect_vids), n_per_class), replace=False))
        )
        df = df[df["video_id"].isin(sampled_vids)].reset_index(drop=True)

    # Get feature columns (exclude metadata)
    meta_cols = {"exercise", "label", "source", "label_encoded", "video_id"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].fillna(0).values
    y = df["label_encoded"].values
    groups = df["video_id"].values

    if len(np.unique(y)) < 2:
        print(f"  [SKIP] {exercise}: only one class present ({np.unique(y)})")
        return {}

    n_unique_groups = len(np.unique(groups))
    if n_unique_groups < 3:
        print(f"  [SKIP] {exercise}: only {n_unique_groups} video groups (need >=3)")
        return {}

    # Group-based train/test split (70/30) -- no video in both train and test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss.split(X, y, groups=groups))

    X_train, y_train = X[train_idx], y[train_idx]
    X_temp, y_temp = X[temp_idx], y[temp_idx]
    groups_train = groups[train_idx]
    groups_temp = groups[temp_idx]

    # Split temp into val/test (each 15%)
    if len(np.unique(groups_temp)) >= 2:
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
        val_idx, test_idx = next(gss2.split(X_temp, y_temp, groups=groups_temp))
        X_val, y_val = X_temp[val_idx], y_temp[val_idx]
        X_test, y_test = X_temp[test_idx], y_temp[test_idx]
    else:
        # Not enough groups for val/test split -- use all temp as test
        X_val, y_val = X_temp, y_temp
        X_test, y_test = X_temp, y_temp

    print(f"\n{'='*60}")
    print(f"Exercise: {exercise}")
    print(f"  Total samples: {len(y)} (correct: {y.sum()}, incorrect: {len(y)-y.sum()})")
    print(f"  Unique videos: {n_unique_groups}")
    print(f"  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Split: GROUP-BASED by video_id (no data leakage)")

    model_results = {}

    for model_name in model_types:
        if model_name not in MODEL_CONFIGS:
            continue

        model = MODEL_CONFIGS[model_name]()

        # GroupKFold cross-validation on training data
        n_splits = min(3, len(np.unique(groups_train)))
        if n_splits >= 2:
            cv = GroupKFold(n_splits=n_splits)
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=cv,
                scoring="f1", groups=groups_train,
            )
        else:
            cv_scores = np.array([0.0])

        # Train on full training set
        model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test)
        metrics = evaluate_classifier(y_test, y_pred)
        metrics["cv_f1_mean"] = float(cv_scores.mean())
        metrics["cv_f1_std"] = float(cv_scores.std())
        metrics["n_train_videos"] = int(len(np.unique(groups_train)))
        metrics["n_test_videos"] = int(len(np.unique(groups[temp_idx] if len(np.unique(groups_temp)) >= 2 else groups_temp)))

        print(f"\n  {model_name}:")
        print(f"    CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        print(f"    Test - Acc: {metrics['accuracy']:.3f}, P: {metrics['precision']:.3f}, "
              f"R: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}")

        # Save model
        model_path = MODELS_DIR / f"{exercise}_{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save evaluation
        eval_path = MODELS_DIR / f"{exercise}_{model_name}_eval.json"
        with open(eval_path, "w") as f:
            json.dump(metrics, f, indent=2)

        model_results[model_name] = metrics

    # Save the best model as the default classifier
    if model_results:
        best_model_name = max(model_results, key=lambda k: model_results[k]["f1"])
        best_src = MODELS_DIR / f"{exercise}_{best_model_name}.pkl"
        best_dst = MODELS_DIR / f"{exercise}_classifier.pkl"

        # Copy best model as default
        with open(best_src, "rb") as f:
            best_model = pickle.load(f)
        with open(best_dst, "wb") as f:
            pickle.dump(best_model, f)

        print(f"\n  Best model: {best_model_name} (F1={model_results[best_model_name]['f1']:.3f})")
        print(f"  Saved as: {best_dst.name}")

    return model_results


def main():
    parser = argparse.ArgumentParser(description="Train form classifiers")
    parser.add_argument("--exercise", type=str, help="Train for a specific exercise only")
    parser.add_argument("--model", type=str, choices=list(MODEL_CONFIGS.keys()),
                        help="Train only a specific model type")
    parser.add_argument("--data", type=str, default=None,
                        help="Path or filename for feature CSV (default: augmented_features.csv)")
    parser.add_argument("--max-samples", type=int, default=50000,
                        help="Max samples per exercise (0=all, default 50000)")
    args = parser.parse_args()

    model_types = [args.model] if args.model else None

    if args.exercise:
        exercises = [args.exercise]
    else:
        exercises = EXERCISES

    all_results = {}

    for exercise in exercises:
        df = load_features(exercise, args.data)
        if len(df) == 0:
            print(f"[SKIP] No data for {exercise}")
            continue

        if "video_id" not in df.columns:
            print(f"[ERROR] Feature CSV missing 'video_id' column. "
                  f"Re-run build_features.py to generate updated features.")
            sys.exit(1)

        results = train_exercise(df, exercise, model_types, max_samples=args.max_samples)
        if results:
            # Use best model's metrics for the summary
            best = max(results.values(), key=lambda m: m["f1"])
            all_results[exercise] = best

    if all_results:
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY (best model per exercise)")
        print(per_exercise_report(all_results))


if __name__ == "__main__":
    main()
