"""
Build sequence datasets for BiLSTM training.

Reads landmark CSVs, extracts fixed-length feature sequences (windows),
and saves as .npz files per exercise for PyTorch training.

Each sequence = one sliding window of N frames, labeled by video label.

Usage:
    python scripts/build_sequences.py
    python scripts/build_sequences.py --exercise squat --seq-len 60
"""

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
from src.utils.constants import LANDMARK_NAMES, NUM_LANDMARKS, EXERCISES

LANDMARKS_DIR = PROJECT_ROOT / "data" / "processed" / "landmarks"
SEQUENCES_DIR = PROJECT_ROOT / "data" / "processed" / "sequences"
SEQUENCES_DIR.mkdir(parents=True, exist_ok=True)


def row_to_pose_result(row: pd.Series) -> PoseResult:
    """Convert a landmark CSV row to PoseResult."""
    landmarks = np.zeros((NUM_LANDMARKS, 4))
    world_landmarks = np.zeros((NUM_LANDMARKS, 3))

    for i, name in enumerate(LANDMARK_NAMES):
        landmarks[i, 0] = float(row.get(f"{name}_x", 0.0) or 0.0)
        landmarks[i, 1] = float(row.get(f"{name}_y", 0.0) or 0.0)
        landmarks[i, 2] = float(row.get(f"{name}_z", 0.0) or 0.0)
        landmarks[i, 3] = float(row.get(f"{name}_vis", 0.0) or 0.0)
        world_landmarks[i, 0] = float(row.get(f"{name}_wx", 0.0) or 0.0)
        world_landmarks[i, 1] = float(row.get(f"{name}_wy", 0.0) or 0.0)
        world_landmarks[i, 2] = float(row.get(f"{name}_wz", 0.0) or 0.0)

    ts_val = row.get("timestamp_ms", 0)
    try:
        timestamp = int(ts_val) if not (isinstance(ts_val, float) and np.isnan(ts_val)) else 0
    except (ValueError, TypeError):
        timestamp = 0

    return PoseResult(
        landmarks=landmarks,
        world_landmarks=world_landmarks,
        detection_confidence=float(landmarks[:, 3].mean()),
        timestamp_ms=timestamp,
    )


def parse_video_csv_name(csv_path: Path) -> tuple:
    """Parse video CSV filename to extract (source, exercise, label, video_id).
    Returns (None, None, None, None) if unparseable."""
    name = csv_path.stem
    parts = name.split("_")

    if name.startswith("kaggle_images_"):
        return None, None, None, None  # Skip image CSVs

    if len(parts) < 3:
        return None, None, None, None

    source = parts[0]
    exercise = parts[1]
    label = parts[2]
    video_id = name

    if exercise not in EXERCISES and len(parts) >= 4:
        exercise = f"{parts[1]}_{parts[2]}"
        label = parts[3]

    if exercise not in EXERCISES:
        return None, None, None, None

    return source, exercise, label, video_id


def build_sequences_for_exercise(
    exercise: str,
    fe: FeatureExtractor,
    seq_len: int = 60,
    stride: int = 15,
) -> tuple:
    """Build all sequences for a single exercise.

    Returns:
        X: np.ndarray of shape (n_sequences, seq_len, n_features)
        y: np.ndarray of shape (n_sequences,)  -- 0=incorrect, 1=correct
        video_ids: list of str -- video_id per sequence (for group splitting)
    """
    feature_names = fe.get_all_feature_names(exercise)
    n_features = len(feature_names)

    all_sequences = []
    all_labels = []
    all_video_ids = []

    # Find all video CSVs for this exercise
    video_csvs = []
    for csv_path in sorted(LANDMARKS_DIR.glob("*.csv")):
        source, ex, label, vid_id = parse_video_csv_name(csv_path)
        if ex == exercise:
            video_csvs.append((csv_path, source, label, vid_id))

    if not video_csvs:
        return np.array([]), np.array([]), []

    for csv_path, source, label, vid_id in tqdm(
        video_csvs, desc=f"  {exercise}", leave=False
    ):
        df = pd.read_csv(csv_path)
        if len(df) < seq_len // 2:
            continue  # Video too short

        # Convert all frames to PoseResults and extract features
        frame_buffer = deque(maxlen=30)
        frame_features = []

        for _, row in df.iterrows():
            pose = row_to_pose_result(row)
            if pose.detection_confidence < 0.1:
                continue

            frame_buffer.append(pose)
            features = fe.extract_with_temporal(pose, exercise, list(frame_buffer))

            # Convert to vector in consistent order
            vec = [features.get(fn, 0.0) or 0.0 for fn in feature_names]
            frame_features.append(vec)

        if len(frame_features) < seq_len:
            # Pad short videos by repeating last frame
            while len(frame_features) < seq_len:
                frame_features.append(frame_features[-1] if frame_features else [0.0] * n_features)

        frame_features = np.array(frame_features, dtype=np.float32)

        # Create sliding windows
        label_encoded = 1 if label == "correct" else 0

        for start in range(0, len(frame_features) - seq_len + 1, stride):
            window = frame_features[start : start + seq_len]
            all_sequences.append(window)
            all_labels.append(label_encoded)
            all_video_ids.append(vid_id)

    if not all_sequences:
        return np.array([]), np.array([]), []

    X = np.stack(all_sequences, axis=0)  # (n_seq, seq_len, n_features)
    y = np.array(all_labels, dtype=np.int64)

    return X, y, all_video_ids


def main():
    parser = argparse.ArgumentParser(description="Build sequence datasets for BiLSTM")
    parser.add_argument("--exercise", type=str, default=None,
                        help="Build for a specific exercise only")
    parser.add_argument("--seq-len", type=int, default=60,
                        help="Sequence length in frames (default: 60)")
    parser.add_argument("--stride", type=int, default=15,
                        help="Stride between windows (default: 15)")
    args = parser.parse_args()

    exercises = [args.exercise] if args.exercise else EXERCISES
    fe = FeatureExtractor(use_world=True)

    print(f"Building sequence datasets:")
    print(f"  Sequence length: {args.seq_len} frames")
    print(f"  Stride: {args.stride} frames")
    print(f"  Landmarks dir: {LANDMARKS_DIR}")
    print(f"  Output dir: {SEQUENCES_DIR}\n")

    summary = {}

    for exercise in exercises:
        print(f"Processing {exercise}...")
        X, y, video_ids = build_sequences_for_exercise(
            exercise, fe, seq_len=args.seq_len, stride=args.stride
        )

        if len(X) == 0:
            print(f"  [SKIP] No sequences generated for {exercise}")
            continue

        n_correct = int((y == 1).sum())
        n_incorrect = int((y == 0).sum())
        n_videos = len(set(video_ids))

        # Save
        out_path = SEQUENCES_DIR / f"{exercise}_sequences.npz"
        np.savez_compressed(
            out_path,
            X=X,
            y=y,
            video_ids=np.array(video_ids),
            feature_names=np.array(fe.get_all_feature_names(exercise)),
        )

        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  Saved: {out_path.name} ({size_mb:.1f} MB)")
        print(f"  Sequences: {len(X)} (correct: {n_correct}, incorrect: {n_incorrect})")
        print(f"  Shape: {X.shape}")
        print(f"  Videos: {n_videos}")

        summary[exercise] = {
            "sequences": len(X),
            "correct": n_correct,
            "incorrect": n_incorrect,
            "videos": n_videos,
            "shape": list(X.shape),
        }

    # Print summary
    if summary:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Exercise':<18} {'Seqs':>8} {'Correct':>8} {'Incorrect':>8} {'Videos':>8}")
        print("-" * 55)
        total_seqs = 0
        for ex in sorted(summary.keys()):
            s = summary[ex]
            print(f"{ex:<18} {s['sequences']:>8} {s['correct']:>8} {s['incorrect']:>8} {s['videos']:>8}")
            total_seqs += s['sequences']
        print("-" * 55)
        print(f"{'TOTAL':<18} {total_seqs:>8}")


if __name__ == "__main__":
    main()
