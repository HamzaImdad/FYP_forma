"""
Train per-exercise classifiers on extracted features.

Trains Random Forest, SVM, and Logistic Regression models for each exercise.
Performs stratified train/val/test splits and 5-fold cross-validation.
Saves trained models and evaluation reports.

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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]),
    "svm": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42), cv=3)),
    ]),
    "logistic_regression": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
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
    Returns dict of model_name -> metrics.
    """
    if model_types is None:
        model_types = list(MODEL_CONFIGS.keys())

    # Encode labels: correct=1, incorrect=0
    df = df.copy()
    df["label_encoded"] = (df["label"] == "correct").astype(int)

    # Subsample if too large (video frames are highly correlated)
    if max_samples > 0 and len(df) > max_samples:
        df = df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), max_samples // 2), random_state=42)
        ).reset_index(drop=True)

    # Get feature columns (exclude metadata)
    meta_cols = {"exercise", "label", "source", "label_encoded"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].fillna(0).values
    y = df["label_encoded"].values

    if len(np.unique(y)) < 2:
        print(f"  [SKIP] {exercise}: only one class present ({np.unique(y)})")
        return {}

    # Stratified train/test split (70/15/15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print(f"\n{'='*60}")
    print(f"Exercise: {exercise}")
    print(f"  Total samples: {len(y)} (correct: {y.sum()}, incorrect: {len(y)-y.sum()})")
    print(f"  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    print(f"  Features: {len(feature_cols)}")

    model_results = {}

    for model_name in model_types:
        if model_name not in MODEL_CONFIGS:
            continue

        model = MODEL_CONFIGS[model_name]()

        # 3-fold cross-validation on training data
        cv = StratifiedKFold(n_splits=min(3, min(np.bincount(y_train))), shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")

        # Train on full training set
        model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test)
        metrics = evaluate_classifier(y_test, y_pred)
        metrics["cv_f1_mean"] = float(cv_scores.mean())
        metrics["cv_f1_std"] = float(cv_scores.std())

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
