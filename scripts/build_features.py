"""
Build feature CSVs from extracted landmark CSVs.

Reads all landmark CSVs (image and video formats), parses each row into a
PoseResult, runs FeatureExtractor, and saves combined feature CSVs.

Usage:
    python scripts/build_features.py
    python scripts/build_features.py --source images
    python scripts/build_features.py --source videos
"""

import os
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

        for _, row in df.iterrows():
            # Skip rows where all landmarks are zero (failed detection)
            if row.get("nose_vis", 0) == 0:
                continue

            pose = row_to_pose_result(row, fmt="image")
            features = fe.extract(pose, exercise)

            feat_row = {
                "exercise": exercise,
                "label": label,
                "source": row.get("source", "kaggle_images"),
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

        feature_names = fe.get_feature_names(exercise)

        for _, row in df.iterrows():
            pose = row_to_pose_result(row, fmt="video")

            # Skip frames with no detection
            if pose.detection_confidence < 0.1:
                continue

            features = fe.extract(pose, exercise)

            feat_row = {
                "exercise": exercise,
                "label": label,
                "source": source,
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
    for f in FEATURES_DIR.glob("*_features.csv"):
        dfs.append(pd.read_csv(f))

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        out_path = FEATURES_DIR / "all_features.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nCombined: {len(combined)} total rows saved to {out_path}")
        print(f"  Exercises: {combined['exercise'].value_counts().to_dict()}")
        print(f"  Labels: {combined['label'].value_counts().to_dict()}")


def main():
    parser = argparse.ArgumentParser(description="Build feature CSVs from landmarks")
    parser.add_argument("--source", choices=["images", "videos", "all"], default="all")
    args = parser.parse_args()

    if args.source in ("images", "all"):
        process_image_landmarks()
    if args.source in ("videos", "all"):
        process_video_landmarks()
    if args.source == "all":
        combine_all()


if __name__ == "__main__":
    main()
