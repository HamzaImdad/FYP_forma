"""
Data augmentation script to balance correct/incorrect classes.

Supports two modes:
  1. Landmark-based: Loads landmark CSVs, applies pose augmentation
     (noise, scaling, flipping), re-extracts features, and appends to
     the feature CSV to balance class distribution.
  2. Feature-based: Directly augments extracted angle features using
     noise, mirroring, and rotation perturbation (faster, no re-extraction).

Usage:
    python scripts/augment_data.py
    python scripts/augment_data.py --mode features --target-ratio 1.5
    python scripts/augment_data.py --mode landmarks --target-ratio 2.0
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pose_estimation.base import PoseResult
from src.feature_extraction.features import FeatureExtractor
from src.utils.augmentation import (
    augment_pose,
    augment_angle_features,
    augment_landmark_features,
)
from src.utils.constants import LANDMARK_NAMES, NUM_LANDMARKS, EXERCISES

LANDMARKS_DIR = PROJECT_ROOT / "data" / "processed" / "landmarks"
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"

META_COLS = ["exercise", "label", "source", "video_id"]


def row_to_pose_result(row: pd.Series) -> PoseResult:
    """Convert a landmark CSV row to PoseResult."""
    landmarks = np.zeros((NUM_LANDMARKS, 4))
    world_landmarks = np.zeros((NUM_LANDMARKS, 3))

    def safe_float(val, default=0.0):
        try:
            v = float(val)
            return default if np.isnan(v) else v
        except (ValueError, TypeError):
            return default

    for i, name in enumerate(LANDMARK_NAMES):
        landmarks[i, 0] = safe_float(row.get(f"{name}_x", 0.0))
        landmarks[i, 1] = safe_float(row.get(f"{name}_y", 0.0))
        landmarks[i, 2] = safe_float(row.get(f"{name}_z", 0.0))
        landmarks[i, 3] = safe_float(row.get(f"{name}_vis", 0.0))
        world_landmarks[i, 0] = safe_float(row.get(f"{name}_wx", 0.0))
        world_landmarks[i, 1] = safe_float(row.get(f"{name}_wy", 0.0))
        world_landmarks[i, 2] = safe_float(row.get(f"{name}_wz", 0.0))

    ts_val = row.get("timestamp_ms", 0)
    timestamp = int(ts_val) if not (isinstance(ts_val, float) and np.isnan(ts_val)) else 0

    return PoseResult(
        landmarks=landmarks,
        world_landmarks=world_landmarks,
        detection_confidence=float(landmarks[:, 3].mean()),
        timestamp_ms=timestamp,
    )


def load_minority_landmarks(exercise: str, minority_label: str) -> pd.DataFrame:
    """Load all landmark CSVs for a specific exercise and label."""
    dfs = []

    for csv_path in LANDMARKS_DIR.glob("*.csv"):
        name = csv_path.stem
        # Image CSVs: kaggle_images_{exercise}_{label}
        if name.startswith("kaggle_images_"):
            parts = name.split("_")
            ex = "_".join(parts[2:-1])
            lbl = parts[-1]
            if ex == exercise and lbl == minority_label:
                dfs.append(pd.read_csv(csv_path))
            continue

        # Video CSVs: {source}_{exercise}_{label}_{videoname}
        parts = name.split("_")
        if len(parts) < 3:
            continue

        ex = parts[1]
        lbl = parts[2]

        if ex not in EXERCISES and len(parts) >= 4:
            ex = f"{parts[1]}_{parts[2]}"
            lbl = parts[3]

        if ex == exercise and lbl == minority_label:
            dfs.append(pd.read_csv(csv_path))

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def augment_from_landmarks(
    exercise: str,
    minority_label: str,
    n_needed: int,
    fe: FeatureExtractor,
) -> list:
    """Generate augmented feature rows from raw landmarks (landmark mode)."""
    landmark_df = load_minority_landmarks(exercise, minority_label)

    if len(landmark_df) == 0:
        print(f"  [WARN] No landmark data found for {exercise}/{minority_label}")
        return []

    feature_names = fe.get_feature_names(exercise)
    augmented_rows = []
    n_per_sample = max(1, n_needed // len(landmark_df) + 1)

    for row_idx, (_, row) in enumerate(landmark_df.iterrows()):
        if len(augmented_rows) >= n_needed:
            break

        if row.get("nose_vis", 0) == 0:
            continue

        pose = row_to_pose_result(row)
        if pose.detection_confidence < 0.1:
            continue

        aug_poses = augment_pose(pose, n_augmented=min(n_per_sample, 4))

        for aug_idx, aug_pose in enumerate(aug_poses):
            if len(augmented_rows) >= n_needed:
                break

            features = fe.extract(aug_pose, exercise)
            feat_row = {
                "exercise": exercise,
                "label": minority_label,
                "source": "augmented",
                "video_id": f"aug_{exercise}_{minority_label}_{row_idx}_{aug_idx}",
            }
            for fname in feature_names:
                feat_row[fname] = features.get(fname)

            augmented_rows.append(feat_row)

    return augmented_rows


def augment_from_features(
    df_minority: pd.DataFrame,
    exercise: str,
    minority_label: str,
    n_needed: int,
) -> list:
    """Generate augmented feature rows directly from feature space (feature mode)."""
    feature_cols = [c for c in df_minority.columns if c not in META_COLS]
    augmented_rows = []
    n_per_sample = min(4, max(1, n_needed // len(df_minority) + 1))

    for row_idx, (_, row) in enumerate(df_minority.iterrows()):
        if len(augmented_rows) >= n_needed:
            break

        row_features = row[feature_cols].to_dict()
        aug_list = augment_angle_features(row_features, n_augmented=n_per_sample)

        for aug_idx, aug_features in enumerate(aug_list):
            if len(augmented_rows) >= n_needed:
                break

            aug_row = {
                "exercise": exercise,
                "label": minority_label,
                "source": "augmented",
                "video_id": f"aug_{row['video_id']}_{row_idx}_{aug_idx}",
            }
            aug_row.update(aug_features)
            augmented_rows.append(aug_row)

    return augmented_rows


def main():
    parser = argparse.ArgumentParser(description="Augment data to balance classes")
    parser.add_argument("--target-ratio", type=float, default=1.5,
                        help="Max allowed majority:minority ratio (default 1.5)")
    parser.add_argument("--input", type=str, default=None,
                        help="Input feature CSV (default: all_features.csv)")
    parser.add_argument("--mode", choices=["landmarks", "features"], default="features",
                        help="Augmentation mode: 'landmarks' re-extracts from raw data, "
                             "'features' augments in feature space (default: features)")
    args = parser.parse_args()

    # Load features
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = FEATURES_DIR / "all_features.csv"
        if not input_path.exists():
            input_path = FEATURES_DIR / "image_features.csv"

    if not input_path.exists():
        print(f"No feature CSV found at {input_path}. Run build_features.py first.")
        sys.exit(1)

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} feature rows from {input_path.name}")
    print(f"Augmentation mode: {args.mode}")

    fe = FeatureExtractor(use_world=True) if args.mode == "landmarks" else None
    all_augmented = []

    print(f"\nClass distribution before augmentation:")
    print(f"{'Exercise':<18} {'Correct':>10} {'Incorrect':>10} {'Ratio':>8}")
    print("-" * 50)

    for exercise in EXERCISES:
        ex_df = df[df["exercise"] == exercise]
        if len(ex_df) == 0:
            print(f"{exercise:<18} {'--':>10} {'--':>10} {'--':>8}")
            continue

        n_correct = len(ex_df[ex_df["label"] == "correct"])
        n_incorrect = len(ex_df[ex_df["label"] == "incorrect"])

        if n_correct == 0 and n_incorrect == 0:
            print(f"{exercise:<18} {0:>10} {0:>10} {'--':>8}")
            continue

        majority = max(n_correct, n_incorrect)
        minority = min(n_correct, n_incorrect)

        if minority == 0:
            ratio_str = "inf"
        else:
            ratio_str = f"{majority/minority:.1f}:1"

        print(f"{exercise:<18} {n_correct:>10,} {n_incorrect:>10,} {ratio_str:>8}")

        if minority == 0:
            print(f"  [SKIP] Only one class present for {exercise}")
            continue

        if majority / minority <= args.target_ratio:
            continue

        minority_label = "correct" if n_correct < n_incorrect else "incorrect"
        target_minority = int(majority / args.target_ratio)
        n_needed = target_minority - minority

        print(f"  Augmenting {minority_label}: {minority:,} -> {target_minority:,} (+{n_needed:,})")

        if args.mode == "landmarks":
            aug_rows = augment_from_landmarks(exercise, minority_label, n_needed, fe)
        else:
            minority_df = ex_df[ex_df["label"] == minority_label]
            aug_rows = augment_from_features(minority_df, exercise, minority_label, n_needed)

        if aug_rows:
            all_augmented.extend(aug_rows)
            print(f"  Generated {len(aug_rows):,} augmented samples")

    # Combine and save
    if all_augmented:
        aug_df = pd.DataFrame(all_augmented)
        combined = pd.concat([df, aug_df], ignore_index=True)

        out_path = FEATURES_DIR / "augmented_features.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nSaved {len(combined):,} total rows to {out_path.name}")
        print(f"  Original: {len(df):,}, Augmented: {len(aug_df):,}")

        print(f"\nClass distribution after augmentation:")
        print(f"{'Exercise':<18} {'Correct':>10} {'Incorrect':>10}")
        print("-" * 40)
        for exercise in EXERCISES:
            ex_df = combined[combined["exercise"] == exercise]
            n_c = len(ex_df[ex_df["label"] == "correct"])
            n_i = len(ex_df[ex_df["label"] == "incorrect"])
            if n_c > 0 or n_i > 0:
                print(f"{exercise:<18} {n_c:>10,} {n_i:>10,}")
    else:
        print("\nNo augmentation needed, classes are balanced enough.")
        out_path = FEATURES_DIR / "augmented_features.csv"
        df.to_csv(out_path, index=False)
        print(f"Copied original to {out_path.name}")


if __name__ == "__main__":
    main()
