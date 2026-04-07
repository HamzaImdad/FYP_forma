"""
Build feature CSVs from extracted landmark CSVs.

Reads all landmark CSVs (image and video formats), parses each row into a
PoseResult, runs FeatureExtractor, and saves combined feature CSVs.

Usage:
    python scripts/build_features.py
    python scripts/build_features.py --source images
    python scripts/build_features.py --source videos
    python scripts/build_features.py --landmarks
"""

import os
import sys
import argparse
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pose_estimation.base import PoseResult
from src.feature_extraction.features import FeatureExtractor
from src.feature_extraction.landmark_features import (
    LandmarkFeatureExtractor, LANDMARK_FEATURE_NAMES,
    VELOCITY_FEATURE_NAMES, ACCELERATION_FEATURE_NAMES,
    SYMMETRY_FEATURE_NAMES, ALL_TEMPORAL_FEATURE_NAMES,
)
from src.utils.constants import LANDMARK_NAMES, NUM_LANDMARKS, EXERCISES

LANDMARKS_DIR = PROJECT_ROOT / "data" / "processed" / "landmarks"
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def row_to_pose_result(row: pd.Series, fmt: str = "image") -> PoseResult:
    """
    Convert a CSV row into a PoseResult.

    Args:
        row: pandas Series with landmark columns.
        fmt: "image" (has image_id,source,exercise,label prefix)
             or "video" (has frame_id,timestamp_ms prefix)
    """
    landmarks = np.zeros((NUM_LANDMARKS, 4))
    world_landmarks = np.zeros((NUM_LANDMARKS, 3))

    for i, name in enumerate(LANDMARK_NAMES):
        landmarks[i, 0] = row.get(f"{name}_x", 0.0)
        landmarks[i, 1] = row.get(f"{name}_y", 0.0)
        landmarks[i, 2] = row.get(f"{name}_z", 0.0)
        landmarks[i, 3] = row.get(f"{name}_vis", 0.0)
        world_landmarks[i, 0] = row.get(f"{name}_wx", 0.0)
        world_landmarks[i, 1] = row.get(f"{name}_wy", 0.0)
        world_landmarks[i, 2] = row.get(f"{name}_wz", 0.0)

    timestamp = int(row.get("timestamp_ms", 0))

    return PoseResult(
        landmarks=landmarks,
        world_landmarks=world_landmarks,
        detection_confidence=float(landmarks[:, 3].mean()),
        timestamp_ms=timestamp,
    )


def process_image_landmarks():
    """Process all kaggle_images_*_correct.csv files."""
    fe = FeatureExtractor(use_world=True)
    csv_files = sorted(LANDMARKS_DIR.glob("kaggle_images_*_correct.csv"))

    if not csv_files:
        print("No image landmark CSVs found.")
        return

    all_rows = []

    for csv_path in tqdm(csv_files, desc="Image CSVs"):
        df = pd.read_csv(csv_path)
        exercise = df["exercise"].iloc[0] if "exercise" in df.columns else csv_path.stem.split("_")[2]
        label = df["label"].iloc[0] if "label" in df.columns else "correct"

        if exercise not in EXERCISES:
            continue

        feature_names = fe.get_feature_names(exercise)

        for idx, (_, row) in enumerate(df.iterrows()):
            # Skip rows where all landmarks are zero (failed detection)
            if row.get("nose_vis", 0) == 0:
                continue

            pose = row_to_pose_result(row, fmt="image")
            features = fe.extract(pose, exercise)

            feat_row = {
                "exercise": exercise,
                "label": label,
                "source": row.get("source", "kaggle_images"),
                "video_id": f"img_{csv_path.stem}_{idx}",
            }
            for fname in feature_names:
                feat_row[fname] = features.get(fname)

            all_rows.append(feat_row)

    if all_rows:
        out_df = pd.DataFrame(all_rows)
        out_path = FEATURES_DIR / "image_features.csv"
        out_df.to_csv(out_path, index=False)
        print(f"Saved {len(out_df)} image feature rows to {out_path}")
        print(f"  Exercises: {out_df['exercise'].value_counts().to_dict()}")
    else:
        print("No valid image features extracted.")


def process_video_landmarks():
    """Process all video landmark CSVs (youtube_*, kaggle_*)."""
    fe = FeatureExtractor(use_world=True)

    # Video CSVs have names like: youtube_squat_correct_videoname.csv
    video_csvs = []
    for f in sorted(LANDMARKS_DIR.glob("*.csv")):
        if f.name.startswith("kaggle_images_"):
            continue  # skip image CSVs
        video_csvs.append(f)

    if not video_csvs:
        print("No video landmark CSVs found.")
        return

    all_rows = []

    for csv_path in tqdm(video_csvs, desc="Video CSVs"):
        parts = csv_path.stem.split("_")
        # Format: source_exercise_label_videoname
        if len(parts) < 3:
            continue

        source = parts[0]
        exercise = parts[1]
        label = parts[2]

        if exercise not in EXERCISES:
            # Try combining parts for two-word exercises
            if len(parts) >= 4:
                exercise = f"{parts[1]}_{parts[2]}"
                label = parts[3]
            if exercise not in EXERCISES:
                continue

        df = pd.read_csv(csv_path)
        if len(df) == 0:
            continue

        # Use all feature names including temporal
        feature_names = fe.get_all_feature_names(exercise)

        # Video-level ID for group splitting (prevents data leakage)
        vid_id = csv_path.stem

        # Sliding window for temporal feature extraction
        frame_buffer = deque(maxlen=30)

        for _, row in df.iterrows():
            pose = row_to_pose_result(row, fmt="video")

            # Skip frames with no detection
            if pose.detection_confidence < 0.1:
                continue

            frame_buffer.append(pose)

            # Extract features with temporal context
            features = fe.extract_with_temporal(pose, exercise, list(frame_buffer))

            feat_row = {
                "exercise": exercise,
                "label": label,
                "source": source,
                "video_id": vid_id,
            }
            for fname in feature_names:
                feat_row[fname] = features.get(fname)

            all_rows.append(feat_row)

    if all_rows:
        out_df = pd.DataFrame(all_rows)
        out_path = FEATURES_DIR / "video_features.csv"
        out_df.to_csv(out_path, index=False)
        print(f"Saved {len(out_df)} video feature rows to {out_path}")
        print(f"  Exercises: {out_df['exercise'].value_counts().to_dict()}")
        print(f"  Labels: {out_df['label'].value_counts().to_dict()}")
    else:
        print("No valid video features extracted.")


def combine_all():
    """Combine image and video features into a single CSV."""
    dfs = []
    for name in ["image_features.csv", "video_features.csv"]:
        f = FEATURES_DIR / name
        if f.exists():
            dfs.append(pd.read_csv(f))

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        out_path = FEATURES_DIR / "all_features.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nCombined: {len(combined)} total rows saved to {out_path}")
        print(f"  Exercises: {combined['exercise'].value_counts().to_dict()}")
        print(f"  Labels: {combined['label'].value_counts().to_dict()}")


def process_landmark_features():
    """Extract raw normalized landmark features from ALL landmark CSVs (images + videos)."""
    lfe = LandmarkFeatureExtractor()
    col_names = LANDMARK_FEATURE_NAMES
    all_rows = []

    # --- Process image CSVs ---
    image_csvs = sorted(LANDMARKS_DIR.glob("kaggle_images_*_correct.csv"))
    for csv_path in tqdm(image_csvs, desc="Image CSVs (landmarks)"):
        df = pd.read_csv(csv_path)
        exercise = df["exercise"].iloc[0] if "exercise" in df.columns else csv_path.stem.split("_")[2]
        label = df["label"].iloc[0] if "label" in df.columns else "correct"

        if exercise not in EXERCISES:
            continue

        for idx, (_, row) in enumerate(df.iterrows()):
            if row.get("nose_vis", 0) == 0:
                continue

            pose = row_to_pose_result(row, fmt="image")
            lm_vec = lfe.extract_landmarks(pose)

            feat_row = {
                "exercise": exercise,
                "label": label,
                "source": row.get("source", "kaggle_images"),
                "video_id": f"img_{csv_path.stem}_{idx}",
            }
            for i, cname in enumerate(col_names):
                feat_row[cname] = float(lm_vec[i])

            all_rows.append(feat_row)

    # --- Process video CSVs ---
    video_csvs = []
    for f in sorted(LANDMARKS_DIR.glob("*.csv")):
        if f.name.startswith("kaggle_images_"):
            continue
        video_csvs.append(f)

    for csv_path in tqdm(video_csvs, desc="Video CSVs (landmarks)"):
        parts = csv_path.stem.split("_")
        if len(parts) < 3:
            continue

        source = parts[0]
        exercise = parts[1]
        label = parts[2]

        if exercise not in EXERCISES:
            if len(parts) >= 4:
                exercise = f"{parts[1]}_{parts[2]}"
                label = parts[3]
            if exercise not in EXERCISES:
                continue

        df = pd.read_csv(csv_path)
        if len(df) == 0:
            continue

        vid_id = csv_path.stem

        for _, row in df.iterrows():
            pose = row_to_pose_result(row, fmt="video")
            if pose.detection_confidence < 0.1:
                continue

            lm_vec = lfe.extract_landmarks(pose)

            feat_row = {
                "exercise": exercise,
                "label": label,
                "source": source,
                "video_id": vid_id,
            }
            for i, cname in enumerate(col_names):
                feat_row[cname] = float(lm_vec[i])

            all_rows.append(feat_row)

    if all_rows:
        out_df = pd.DataFrame(all_rows)
        out_path = FEATURES_DIR / "landmark_features.csv"
        out_df.to_csv(out_path, index=False)
        print(f"\nSaved {len(out_df)} landmark feature rows to {out_path}")
        print(f"  Columns: {len(out_df.columns)} (4 meta + {len(col_names)} features)")
        print(f"  Exercises: {out_df['exercise'].value_counts().to_dict()}")
        print(f"  Labels: {out_df['label'].value_counts().to_dict()}")

        # Print sample stats to verify normalization
        feat_cols = out_df[col_names]
        print(f"\n  Feature stats (should be centered ~0, range roughly -2 to 2):")
        print(f"    Mean: {feat_cols.mean().mean():.4f}")
        print(f"    Std:  {feat_cols.std().mean():.4f}")
        print(f"    Min:  {feat_cols.min().min():.4f}")
        print(f"    Max:  {feat_cols.max().max():.4f}")
    else:
        print("No valid landmark features extracted.")


def process_temporal_landmark_features():
    """Extract landmark features with velocity, acceleration, and symmetry (330-dim).

    Processes video CSVs only (images don't have temporal context).
    Uses a sliding window of 3 frames for velocity/acceleration computation.
    """
    lfe = LandmarkFeatureExtractor()
    col_names = ALL_TEMPORAL_FEATURE_NAMES  # 330 features
    all_rows = []

    # --- Process video CSVs ---
    video_csvs = []
    for f in sorted(LANDMARKS_DIR.glob("*.csv")):
        if f.name.startswith("kaggle_images_"):
            continue
        video_csvs.append(f)

    for csv_path in tqdm(video_csvs, desc="Video CSVs (temporal landmarks)"):
        parts = csv_path.stem.split("_")
        if len(parts) < 3:
            continue

        source = parts[0]
        exercise = parts[1]
        label = parts[2]

        if exercise not in EXERCISES:
            if len(parts) >= 4:
                exercise = f"{parts[1]}_{parts[2]}"
                label = parts[3]
            if exercise not in EXERCISES:
                continue

        df = pd.read_csv(csv_path)
        if len(df) == 0:
            continue

        vid_id = csv_path.stem
        frame_history = []

        for _, row in df.iterrows():
            pose = row_to_pose_result(row, fmt="video")
            if pose.detection_confidence < 0.1:
                continue

            frame_history.append(pose)

            # Extract temporal features (needs 3+ frames for velocity/acceleration)
            features = lfe.extract_full_temporal(pose, exercise, frame_history)

            feat_row = {
                "exercise": exercise,
                "label": label,
                "source": source,
                "video_id": vid_id,
            }
            for cname in col_names:
                feat_row[cname] = features.get(cname, 0.0)

            all_rows.append(feat_row)

    if all_rows:
        out_df = pd.DataFrame(all_rows)
        out_path = FEATURES_DIR / "temporal_landmark_features.csv"
        out_df.to_csv(out_path, index=False)
        print(f"\nSaved {len(out_df)} temporal landmark feature rows to {out_path}")
        print(f"  Columns: {len(out_df.columns)} (4 meta + {len(col_names)} features)")
        print(f"  Exercises: {out_df['exercise'].value_counts().to_dict()}")
        print(f"  Labels: {out_df['label'].value_counts().to_dict()}")

        feat_cols = out_df[col_names]
        print(f"\n  Feature stats:")
        print(f"    Mean: {feat_cols.mean().mean():.4f}")
        print(f"    Std:  {feat_cols.std().mean():.4f}")
    else:
        print("No valid temporal landmark features extracted.")


def main():
    parser = argparse.ArgumentParser(description="Build feature CSVs from landmarks")
    parser.add_argument("--source", choices=["images", "videos", "all"], default="all")
    parser.add_argument("--landmarks", action="store_true",
                        help="Extract raw normalized landmarks (99 features) instead of hand-crafted features")
    parser.add_argument("--temporal-landmarks", action="store_true",
                        help="Extract temporal landmarks (330 features: position + velocity + acceleration + symmetry)")
    args = parser.parse_args()

    if args.temporal_landmarks:
        process_temporal_landmark_features()
        return

    if args.landmarks:
        process_landmark_features()
        return

    if args.source in ("images", "all"):
        process_image_landmarks()
    if args.source in ("videos", "all"):
        process_video_landmarks()
    if args.source == "all":
        combine_all()


if __name__ == "__main__":
    main()
