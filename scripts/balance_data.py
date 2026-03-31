"""
Balance the feature dataset by cleaning bad rows, undersampling majority,
and augmenting minority class per exercise.

Produces data/processed/features/balanced_features.csv with same column format.

Usage:
    python scripts/balance_data.py
    python scripts/balance_data.py --max-ratio 3.0
    python scripts/balance_data.py --input data/processed/features/all_features.csv
    python scripts/balance_data.py --landmarks
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction.exercise_features import EXERCISE_FEATURES
from src.feature_extraction.landmark_features import LANDMARK_FEATURE_NAMES
from src.utils.augmentation import augment_angle_features, augment_landmark_features, mirror_angle_feature_names
from src.utils.constants import EXERCISES

FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"

# All angle columns in the CSV (from JOINT_ANGLES)
ANGLE_COLUMNS = [
    "angle_left_elbow", "angle_right_elbow",
    "angle_left_shoulder", "angle_right_shoulder",
    "angle_left_hip", "angle_right_hip",
    "angle_left_knee", "angle_right_knee",
    "angle_left_ankle", "angle_right_ankle",
]

# Metadata columns (not features)
META_COLS = ["exercise", "label", "source", "video_id"]


def get_primary_angle_columns(exercise: str) -> list:
    """Get the angle column names that are primary for this exercise."""
    feat_def = EXERCISE_FEATURES.get(exercise, {})
    primary = feat_def.get("primary_angles", [])
    return [f"angle_{a}" for a in primary]


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return all non-metadata columns."""
    return [c for c in df.columns if c not in META_COLS]


def get_relevant_columns(exercise: str, df_columns: list) -> list:
    """Get the feature columns that are actually relevant for this exercise."""
    feat_def = EXERCISE_FEATURES.get(exercise, {})
    cols = []
    for a in feat_def.get("primary_angles", []) + feat_def.get("secondary_angles", []):
        cols.append(f"angle_{a}")
    cols += feat_def.get("custom_features", [])
    cols += feat_def.get("temporal_features", [])
    return [c for c in cols if c in df_columns]


def clean_landmark_data(ex_df: pd.DataFrame, exercise: str) -> tuple:
    """Clean landmark rows for a single exercise.
    Removes rows where >70% of landmark columns are zero (failed pose detection).
    Returns (cleaned_df, stats_dict).
    """
    lm_cols = [c for c in LANDMARK_FEATURE_NAMES if c in ex_df.columns]
    n_before = len(ex_df)

    if not lm_cols:
        return ex_df, {"before": n_before, "zero_removed": 0, "after": n_before}

    # Count fraction of zero values per row across landmark columns
    zero_frac = (ex_df[lm_cols] == 0).sum(axis=1) / len(lm_cols)
    mask = zero_frac <= 0.7
    cleaned = ex_df[mask].copy()

    stats = {
        "before": n_before,
        "zero_removed": int((~mask).sum()),
        "after": len(cleaned),
    }
    return cleaned, stats


def augment_landmark_minority(df: pd.DataFrame, target_n: int, exercise: str) -> pd.DataFrame:
    """[DEPRECATED] Offline frame-level augmentation for landmark minority class.

    GATED OFF: Offline frame augmentation has been removed for the BiLSTM pipeline.
    Frame-level augmentation (aug_ prefixed rows) corrupts temporal sequences and
    causes data leakage when sequences are built from pre-augmented CSVs.

    Use SequenceDataset online augmentation in train_bilstm.py instead (see DATA-05).
    # Offline frame augmentation removed; use SequenceDataset online augmentation instead (see DATA-05)

    This function returns the input DataFrame unchanged to preserve the call
    signature for any legacy ML classifier code that may invoke it.
    """
    import warnings
    warnings.warn(
        "augment_landmark_minority is gated off for the BiLSTM pipeline. "
        "Returning input DataFrame unchanged. "
        "Use SequenceDataset online augmentation instead (DATA-05).",
        DeprecationWarning,
        stacklevel=2,
    )
    print(
        "[balance_data] WARNING: augment_landmark_minority is disabled. "
        "Offline frame augmentation has been removed for the BiLSTM pipeline. "
        "Use SequenceDataset online augmentation instead (see DATA-05)."
    )
    # Early return: no aug_ rows created
    return df


def clean_exercise_data(ex_df: pd.DataFrame, exercise: str) -> tuple:
    """Clean rows for a single exercise. Returns (cleaned_df, stats_dict)."""
    # Use only exercise-relevant columns for NaN checking
    # (each exercise uses ~10-16 of 41 total feature columns; the rest are naturally NaN)
    relevant_cols = get_relevant_columns(exercise, ex_df.columns)
    primary_cols = get_primary_angle_columns(exercise)
    primary_cols = [c for c in primary_cols if c in ex_df.columns]
    n_before = len(ex_df)

    if not relevant_cols:
        # Fallback: use all feature columns
        relevant_cols = get_feature_columns(ex_df)

    # Count NaN fraction per row across RELEVANT feature columns only
    nan_frac = ex_df[relevant_cols].isna().sum(axis=1) / len(relevant_cols)

    # Remove rows where >50% of relevant feature columns are NaN
    mask_sparse = nan_frac <= 0.5
    n_sparse_removed = (~mask_sparse).sum()

    # Remove rows where ALL primary angles are NaN
    if primary_cols:
        mask_primary = ex_df[primary_cols].notna().any(axis=1)
    else:
        mask_primary = pd.Series(True, index=ex_df.index)
    n_primary_removed = (mask_sparse & ~mask_primary).sum()

    combined_mask = mask_sparse & mask_primary
    cleaned = ex_df[combined_mask].copy()

    stats = {
        "before": n_before,
        "sparse_removed": int(n_sparse_removed),
        "primary_removed": int(n_primary_removed),
        "after": len(cleaned),
    }
    return cleaned, stats


def undersample_majority(df: pd.DataFrame, target_n: int) -> pd.DataFrame:
    """Undersample using stratified sampling by video_id to keep diversity."""
    if len(df) <= target_n:
        return df

    # Group by video_id, sample proportionally
    video_ids = df["video_id"].unique()
    n_videos = len(video_ids)

    if n_videos == 0:
        return df.sample(n=target_n, random_state=42)

    # Allocate samples per video proportionally
    video_counts = df.groupby("video_id").size()
    total = video_counts.sum()
    allocations = (video_counts / total * target_n).clip(lower=1).astype(int)

    # Adjust to hit target exactly
    while allocations.sum() > target_n:
        # Remove from the largest allocation
        largest = allocations.idxmax()
        if allocations[largest] > 1:
            allocations[largest] -= 1
        else:
            break

    while allocations.sum() < target_n:
        # Add to the smallest allocation (that hasn't hit its max)
        for vid in allocations.index:
            if allocations[vid] < video_counts[vid]:
                allocations[vid] += 1
                if allocations.sum() >= target_n:
                    break

    sampled = []
    for vid, n_alloc in allocations.items():
        vid_df = df[df["video_id"] == vid]
        n_take = min(n_alloc, len(vid_df))
        sampled.append(vid_df.sample(n=n_take, random_state=42))

    result = pd.concat(sampled, ignore_index=True)
    # If still over target (due to rounding), trim
    if len(result) > target_n:
        result = result.sample(n=target_n, random_state=42)
    return result


def augment_minority(df: pd.DataFrame, target_n: int, exercise: str) -> pd.DataFrame:
    """Augment minority class rows to reach target count."""
    n_existing = len(df)
    n_needed = target_n - n_existing
    if n_needed <= 0:
        return df

    feature_cols = get_feature_columns(df)

    # How many augmentations per original sample
    n_per_sample = max(1, n_needed // n_existing + 1)
    # Cap at 4 augmentations per sample
    n_per_sample = min(n_per_sample, 4)

    augmented_rows = []
    sample_idx = 0

    for _, row in df.iterrows():
        if len(augmented_rows) >= n_needed:
            break

        row_dict = row[feature_cols].to_dict()
        aug_list = augment_angle_features(row_dict, n_augmented=n_per_sample)

        for aug_idx, aug_features in enumerate(aug_list):
            if len(augmented_rows) >= n_needed:
                break

            aug_row = {
                "exercise": exercise,
                "label": row["label"],
                "source": "augmented",
                "video_id": f"aug_{row['video_id']}_{sample_idx}_{aug_idx}",
            }
            aug_row.update(aug_features)
            augmented_rows.append(aug_row)

        sample_idx += 1

    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
        return pd.concat([df, aug_df], ignore_index=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Balance feature dataset")
    parser.add_argument("--input", type=str, default=None,
                        help="Input feature CSV (default: all_features.csv)")
    parser.add_argument("--max-ratio", type=float, default=1.5,
                        help="Max majority:minority ratio after balancing (default 1.5)")
    parser.add_argument("--no-augment", action="store_true",
                        help="Only clean + undersample, no augmentation")
    parser.add_argument("--landmarks", action="store_true",
                        help="Balance landmark features (99-dim) instead of angle features")
    args = parser.parse_args()

    use_landmarks = args.landmarks

    # Load data
    if args.input:
        input_path = Path(args.input)
    elif use_landmarks:
        input_path = FEATURES_DIR / "landmark_features.csv"
    else:
        input_path = FEATURES_DIR / "all_features.csv"

    if not input_path.exists():
        print(f"ERROR: Feature CSV not found at {input_path}")
        sys.exit(1)

    print(f"Loading {input_path.name}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows\n")

    # Print before distribution
    print(f"{'Exercise':<18} {'Correct':>10} {'Incorrect':>10} {'Ratio':>10}")
    print("=" * 52)
    for ex in EXERCISES:
        ex_df = df[df["exercise"] == ex]
        nc = (ex_df["label"] == "correct").sum()
        ni = (ex_df["label"] == "incorrect").sum()
        if nc + ni == 0:
            print(f"{ex:<18} {'--':>10} {'--':>10} {'--':>10}")
        elif min(nc, ni) == 0:
            print(f"{ex:<18} {nc:>10,} {ni:>10,} {'inf':>10}")
        else:
            ratio = max(nc, ni) / min(nc, ni)
            print(f"{ex:<18} {nc:>10,} {ni:>10,} {f'1:{ratio:.0f}':>10}")

    # Process each exercise
    balanced_dfs = []
    total_cleaned = 0

    print(f"\n{'='*60}")
    print("CLEANING & BALANCING")
    print(f"{'='*60}\n")

    for ex in EXERCISES:
        ex_df = df[df["exercise"] == ex].copy()
        if len(ex_df) == 0:
            print(f"[{ex}] No data, skipping")
            continue

        # Step 1: Clean
        if use_landmarks:
            cleaned, stats = clean_landmark_data(ex_df, ex)
            removed = stats["before"] - stats["after"]
            total_cleaned += removed
            if removed > 0:
                print(f"[{ex}] Cleaned: removed {removed:,} rows "
                      f"({stats['zero_removed']:,} failed pose detection)")
        else:
            cleaned, stats = clean_exercise_data(ex_df, ex)
            removed = stats["before"] - stats["after"]
            total_cleaned += removed
            if removed > 0:
                print(f"[{ex}] Cleaned: removed {removed:,} rows "
                      f"({stats['sparse_removed']:,} sparse, {stats['primary_removed']:,} missing primary)")

        nc = (cleaned["label"] == "correct").sum()
        ni = (cleaned["label"] == "incorrect").sum()

        if nc + ni == 0:
            print(f"[{ex}] No valid rows after cleaning, skipping")
            continue

        minority_count = min(nc, ni)
        majority_count = max(nc, ni)
        minority_label = "correct" if nc < ni else "incorrect"
        majority_label = "incorrect" if minority_label == "correct" else "correct"

        # Step 2: Undersample majority to at most max_ratio * minority_count
        # But first figure out target sizes
        if minority_count == 0:
            print(f"[{ex}] Only {majority_label} samples ({majority_count:,}), skipping balance")
            balanced_dfs.append(cleaned)
            continue

        current_ratio = majority_count / minority_count

        if current_ratio <= args.max_ratio:
            print(f"[{ex}] Already balanced ({nc:,}/{ni:,}, ratio {current_ratio:.1f}:1)")
            balanced_dfs.append(cleaned)
            continue

        # Undersample majority: keep at most 3x minority (pre-augmentation)
        # We'll augment minority up, so undersample to 3x pre-augmented minority
        undersample_target = int(minority_count * 3)
        # But don't undersample below what we'd need for final ratio
        undersample_target = max(undersample_target, int(minority_count * args.max_ratio))

        majority_df = cleaned[cleaned["label"] == majority_label]
        minority_df = cleaned[cleaned["label"] == minority_label]

        majority_sampled = undersample_majority(majority_df, undersample_target)

        # Step 3: Augment minority
        if not args.no_augment:
            # Target: match majority after undersampling
            augment_target = len(majority_sampled)
            if use_landmarks:
                minority_augmented = augment_landmark_minority(minority_df, augment_target, ex)
            else:
                minority_augmented = augment_minority(minority_df, augment_target, ex)
        else:
            minority_augmented = minority_df

        combined = pd.concat([majority_sampled, minority_augmented], ignore_index=True)
        balanced_dfs.append(combined)

        nc_after = (combined["label"] == "correct").sum()
        ni_after = (combined["label"] == "incorrect").sum()
        n_aug = (combined["source"] == "augmented").sum()
        print(f"[{ex}] {nc:,}/{ni:,} -> {nc_after:,}/{ni_after:,} "
              f"(undersampled {len(majority_df):,}->{len(majority_sampled):,}, "
              f"augmented +{n_aug:,})")

    # Combine all exercises
    if not balanced_dfs:
        print("\nERROR: No data after processing")
        sys.exit(1)

    result = pd.concat(balanced_dfs, ignore_index=True)

    # Ensure same column order as original
    result = result[df.columns]

    # Save
    if use_landmarks:
        out_path = FEATURES_DIR / "balanced_landmark_features.csv"
    else:
        out_path = FEATURES_DIR / "balanced_features.csv"
    result.to_csv(out_path, index=False)

    # Print final distribution
    print(f"\n{'='*60}")
    print("FINAL DISTRIBUTION")
    print(f"{'='*60}\n")
    print(f"{'Exercise':<18} {'Correct':>10} {'Incorrect':>10} {'Ratio':>10} {'Augmented':>10}")
    print("-" * 62)

    total_rows = 0
    total_aug = 0
    for ex in EXERCISES:
        ex_df = result[result["exercise"] == ex]
        nc = (ex_df["label"] == "correct").sum()
        ni = (ex_df["label"] == "incorrect").sum()
        na = (ex_df["source"] == "augmented").sum()
        total_rows += nc + ni
        total_aug += na
        if nc + ni == 0:
            continue
        if min(nc, ni) == 0:
            ratio_str = "inf"
        else:
            ratio = max(nc, ni) / min(nc, ni)
            ratio_str = f"1:{ratio:.1f}"
        print(f"{ex:<18} {nc:>10,} {ni:>10,} {ratio_str:>10} {na:>10,}")

    print(f"\nTotal: {total_rows:,} rows ({total_aug:,} augmented)")
    print(f"Original: {len(df):,} rows")
    print(f"Cleaned: {total_cleaned:,} rows removed")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
