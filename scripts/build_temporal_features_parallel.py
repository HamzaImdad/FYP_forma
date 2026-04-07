"""
Parallel temporal landmark feature extraction (330-dim).

Uses all CPU cores to extract landmark + velocity + acceleration + symmetry
features from landmark CSVs. Much faster than single-threaded build_features.py.

Usage:
    python scripts/build_temporal_features_parallel.py
    python scripts/build_temporal_features_parallel.py --workers 20
"""

import sys
import os
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pose_estimation.base import PoseResult
from src.feature_extraction.landmark_features import (
    LandmarkFeatureExtractor,
    ALL_TEMPORAL_FEATURE_NAMES,
)
from src.utils.constants import LANDMARK_NAMES, NUM_LANDMARKS, EXERCISES

LANDMARKS_DIR = PROJECT_ROOT / "data" / "processed" / "landmarks"
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def row_to_pose_result(row):
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


def process_single_csv(args):
    """Process a single CSV file. Called from multiprocessing pool."""
    csv_path, source, exercise, label = args

    if exercise not in EXERCISES:
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if len(df) == 0:
        return None

    lfe = LandmarkFeatureExtractor()
    col_names = ALL_TEMPORAL_FEATURE_NAMES
    vid_id = Path(csv_path).stem
    frame_history = []
    rows = []

    for _, row in df.iterrows():
        pose = row_to_pose_result(row)
        if pose.detection_confidence < 0.1:
            continue

        frame_history.append(pose)
        features = lfe.extract_full_temporal(pose, exercise, frame_history)

        feat_row = {
            "exercise": exercise,
            "label": label,
            "source": source,
            "video_id": vid_id,
        }
        for cname in col_names:
            feat_row[cname] = features.get(cname, 0.0)

        rows.append(feat_row)

    return rows


def collect_csvs():
    """Collect all video CSVs with their metadata."""
    tasks = []

    for csv_path in sorted(LANDMARKS_DIR.glob("*.csv")):
        if csv_path.name.startswith("kaggle_images_"):
            continue

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

        tasks.append((str(csv_path), source, exercise, label))

    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0 = auto)")
    args = parser.parse_args()

    tasks = collect_csvs()
    print(f"CSVs to process: {len(tasks)}")

    n_workers = args.workers if args.workers > 0 else max(1, cpu_count() - 2)
    print(f"Workers: {n_workers} (CPU cores: {cpu_count()})")

    all_rows = []
    completed = 0

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(process_single_csv, tasks):
            completed += 1
            if result:
                all_rows.extend(result)
            if completed % 50 == 0:
                print(f"  [{completed}/{len(tasks)}] {len(all_rows)} rows so far")

    print(f"\nProcessed all {completed} CSVs -> {len(all_rows)} total rows")

    if all_rows:
        col_names = ALL_TEMPORAL_FEATURE_NAMES
        out_df = pd.DataFrame(all_rows)
        out_path = FEATURES_DIR / "temporal_landmark_features.csv"
        out_df.to_csv(out_path, index=False)
        print(f"Saved {len(out_df)} rows to {out_path}")
        print(f"  Columns: {len(out_df.columns)} (4 meta + {len(col_names)} features)")
        print(f"  Exercises: {out_df['exercise'].value_counts().to_dict()}")
        print(f"  Labels: {out_df['label'].value_counts().to_dict()}")

        feat_cols = out_df[col_names]
        print(f"\n  Feature stats:")
        print(f"    Mean: {feat_cols.mean().mean():.4f}")
        print(f"    Std:  {feat_cols.std().mean():.4f}")


if __name__ == "__main__":
    main()
