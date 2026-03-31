"""
Build sequence datasets for BiLSTM training.

Reads landmark CSVs, extracts fixed-length feature sequences (windows),
and saves as .npz files per exercise for PyTorch training.

Each sequence = one sliding window of N frames, labeled by video label.

NEW (manifest-based) usage:
    python scripts/build_sequences.py --use-landmarks --exercise squat
    python scripts/build_sequences.py --use-landmarks --exercise squat --partition train
    python scripts/build_sequences.py --use-landmarks  (all exercises, all partitions)

Legacy usage (backward compat, not recommended):
    python scripts/build_sequences.py
    python scripts/build_sequences.py --exercise squat --seq-len 60
"""

import sys
import argparse
import warnings
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
from src.feature_extraction.landmark_features import LandmarkFeatureExtractor, LANDMARK_FEATURE_NAMES
from src.utils.constants import LANDMARK_NAMES, NUM_LANDMARKS, EXERCISES
from src.utils.video_parsing import parse_video_csv_name, get_base_video_id

LANDMARKS_DIR = PROJECT_ROOT / "data" / "processed" / "landmarks"
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
SEQUENCES_DIR = PROJECT_ROOT / "data" / "processed" / "sequences"
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
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


def row_to_landmark_vector(row: pd.Series) -> np.ndarray:
    """Extract 99-dim landmark vector directly from CSV row (no PoseResult conversion).

    Reads lm_{name}_{x|y|z} columns that were written by build_features.py --landmarks.
    Falls back to reading from world-coordinate columns if lm_ columns are absent.

    Returns:
        np.ndarray of shape (99,), dtype float32
    """
    vec = np.zeros(99, dtype=np.float32)
    idx = 0
    for name in LANDMARK_NAMES:
        # Prefer pre-normalized lm_ columns (written by LandmarkFeatureExtractor)
        lm_x_key = f"lm_{name}_x"
        if lm_x_key in row.index:
            vec[idx] = float(row.get(f"lm_{name}_x", 0.0) or 0.0)
            vec[idx + 1] = float(row.get(f"lm_{name}_y", 0.0) or 0.0)
            vec[idx + 2] = float(row.get(f"lm_{name}_z", 0.0) or 0.0)
        else:
            # Fall back to raw world coordinates
            vec[idx] = float(row.get(f"{name}_wx", 0.0) or 0.0)
            vec[idx + 1] = float(row.get(f"{name}_wy", 0.0) or 0.0)
            vec[idx + 2] = float(row.get(f"{name}_wz", 0.0) or 0.0)
        idx += 3
    return vec


def _check_manifest_staleness(
    exercise: str,
    landmarks_dir: Path,
    splits_dir: Path,
) -> None:
    """Warn if there are landmark CSVs for this exercise not listed in any manifest.

    This catches the case where new videos were added after split_videos.py was run.
    Per Research Open Question 1 — manifest staleness check.
    """
    all_in_manifests = set()
    for partition in ("train", "val", "test"):
        manifest = splits_dir / f"{exercise}_{partition}.txt"
        if manifest.exists():
            for line in manifest.read_text().splitlines():
                line = line.strip()
                if line:
                    all_in_manifests.add(line)

    if not all_in_manifests:
        return  # No manifests yet — skip check

    unlisted = []
    for csv_path in sorted(landmarks_dir.glob("*.csv")):
        name = csv_path.name
        if name.startswith("aug_"):
            continue
        source, ex, label, vid_id = parse_video_csv_name(csv_path)
        if ex != exercise:
            continue
        if name not in all_in_manifests:
            unlisted.append(name)

    if unlisted:
        warnings.warn(
            f"[{exercise}] {len(unlisted)} CSV file(s) in landmarks dir are NOT listed in any "
            f"manifest (possible new additions after split_videos.py was run). "
            f"Re-run split_videos.py --force to refresh manifests.\n"
            f"  Unlisted files: {unlisted[:5]}{'...' if len(unlisted) > 5 else ''}",
            UserWarning,
            stacklevel=3,
        )


def build_partition_sequences(
    exercise: str,
    partition: str,
    splits_dir: Path,
    landmarks_dir: Path,
    sequences_dir: Path,
    seq_len: int = 30,
    lfe: "LandmarkFeatureExtractor | None" = None,
) -> tuple:
    """Build landmark sequences for one exercise partition from its manifest file.

    Reads the manifest at splits_dir/{exercise}_{partition}.txt and loads only
    the CSVs listed there. Temporal row order is preserved (no sort_index).

    Stride policy:
        - train:     stride = seq_len       (non-overlapping windows)
        - val/test:  stride = seq_len // 2  (50% overlap for denser coverage)

    Args:
        exercise:      Exercise name string.
        partition:     "train", "val", or "test".
        splits_dir:    Directory containing manifest .txt files.
        landmarks_dir: Directory containing landmark CSV files.
        sequences_dir: Directory to write output .npz file.
        seq_len:       Sequence window length in frames (default 30).
        lfe:           LandmarkFeatureExtractor instance (created if None).

    Returns:
        (X, y, video_ids) tuple where X has shape (n, seq_len, 99).
        Returns (empty_array, empty_array, []) if the manifest is missing or empty.
    """
    manifest_path = splits_dir / f"{exercise}_{partition}.txt"
    if not manifest_path.exists():
        print(f"  [WARN] Manifest not found: {manifest_path}. Skipping partition.")
        return np.array([]), np.array([]), []

    csv_names = [
        line.strip()
        for line in manifest_path.read_text().splitlines()
        if line.strip()
    ]

    if not csv_names:
        print(f"  [WARN] Manifest is empty: {manifest_path}. Skipping partition.")
        return np.array([]), np.array([]), []

    # Manifest staleness check (warns only, does not abort)
    _check_manifest_staleness(exercise, landmarks_dir, splits_dir)

    if lfe is None:
        lfe = LandmarkFeatureExtractor()

    stride = seq_len if partition == "train" else seq_len // 2
    n_features = 99

    all_sequences = []
    all_labels = []
    all_video_ids = []

    for csv_name in tqdm(csv_names, desc=f"  {exercise}/{partition}", leave=False):
        csv_path = landmarks_dir / csv_name
        if not csv_path.exists():
            print(f"  [WARN] CSV not found: {csv_path}. Skipping.")
            continue

        source, ex, label, video_id = parse_video_csv_name(csv_path)
        if source is None:
            print(f"  [WARN] Could not parse CSV name: {csv_name}. Skipping.")
            continue

        base_video_id = get_base_video_id(csv_path) or video_id

        df = pd.read_csv(csv_path)

        # --- Temporal order must be preserved: no sort_index() ---
        # The CSV is assumed to be in recording order (frame 0 first).
        # Do NOT re-sort rows — doing so would destroy temporal sequence integrity.

        frame_features = []
        for _, row in df.iterrows():
            pose = row_to_pose_result(row)
            if pose.detection_confidence < 0.1:
                continue

            lm_vec = lfe.extract_landmarks(pose)
            frame_features.append(lm_vec)

        if len(frame_features) < seq_len:
            if not frame_features:
                continue  # Skip completely empty videos
            # Pad short videos by repeating last frame
            while len(frame_features) < seq_len:
                frame_features.append(frame_features[-1].copy())

        frame_features_arr = np.array(frame_features, dtype=np.float32)
        label_encoded = 1 if label == "correct" else 0

        for start in range(0, len(frame_features_arr) - seq_len + 1, stride):
            window = frame_features_arr[start: start + seq_len]
            all_sequences.append(window)
            all_labels.append(label_encoded)
            all_video_ids.append(base_video_id)

    if not all_sequences:
        return np.array([]), np.array([]), []

    X = np.stack(all_sequences, axis=0).astype(np.float32)  # (n, seq_len, 99)
    y = np.array(all_labels, dtype=np.int64)

    out_path = sequences_dir / f"{exercise}_{partition}_sequences.npz"
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        video_ids=np.array(all_video_ids),
        feature_names=np.array(LANDMARK_FEATURE_NAMES),
    )

    return X, y, all_video_ids


def build_sequences_for_exercise(
    exercise: str,
    fe: FeatureExtractor,
    seq_len: int = 60,
    stride: int = 15,
) -> tuple:
    """Build all sequences for a single exercise using hand-crafted features.

    Legacy function — kept for backward compatibility.
    Not used in the new BiLSTM pipeline (see build_partition_sequences instead).

    Returns:
        X: np.ndarray of shape (n_sequences, seq_len, n_features)
        y: np.ndarray of shape (n_sequences,)  -- 0=incorrect, 1=correct
        video_ids: list of str
    """
    feature_names = fe.get_all_feature_names(exercise)
    n_features = len(feature_names)

    all_sequences = []
    all_labels = []
    all_video_ids = []

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
            continue

        frame_buffer = deque(maxlen=30)
        frame_features = []

        for _, row in df.iterrows():
            pose = row_to_pose_result(row)
            if pose.detection_confidence < 0.1:
                continue

            frame_buffer.append(pose)
            features = fe.extract_with_temporal(pose, exercise, list(frame_buffer))

            vec = [features.get(fn, 0.0) or 0.0 for fn in feature_names]
            frame_features.append(vec)

        if len(frame_features) < seq_len:
            while len(frame_features) < seq_len:
                frame_features.append(frame_features[-1] if frame_features else [0.0] * n_features)

        frame_features = np.array(frame_features, dtype=np.float32)

        label_encoded = 1 if label == "correct" else 0

        for start in range(0, len(frame_features) - seq_len + 1, stride):
            window = frame_features[start: start + seq_len]
            all_sequences.append(window)
            all_labels.append(label_encoded)
            all_video_ids.append(vid_id)

    if not all_sequences:
        return np.array([]), np.array([]), []

    X = np.stack(all_sequences, axis=0)
    y = np.array(all_labels, dtype=np.int64)

    return X, y, all_video_ids


def build_landmark_sequences_for_exercise(
    exercise: str,
    lfe: LandmarkFeatureExtractor,
    seq_len: int = 30,
    stride: int = 15,
) -> tuple:
    """Build landmark-based sequences for a single exercise.

    Legacy function — kept for backward compatibility.
    In the new pipeline, use build_partition_sequences instead, which reads
    from manifests and uses the correct per-partition stride.

    Returns:
        X: np.ndarray of shape (n_sequences, seq_len, 99)
        y: np.ndarray of shape (n_sequences,)
        video_ids: list of str
    """
    n_features = 99

    all_sequences = []
    all_labels = []
    all_video_ids = []

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
            continue

        frame_features = []

        for _, row in df.iterrows():
            pose = row_to_pose_result(row)
            if pose.detection_confidence < 0.1:
                continue

            lm_vec = lfe.extract_landmarks(pose)
            frame_features.append(lm_vec)

        if len(frame_features) < seq_len:
            while len(frame_features) < seq_len:
                frame_features.append(frame_features[-1] if frame_features else np.zeros(n_features, dtype=np.float32))

        frame_features = np.array(frame_features, dtype=np.float32)

        label_encoded = 1 if label == "correct" else 0

        for start in range(0, len(frame_features) - seq_len + 1, stride):
            window = frame_features[start: start + seq_len]
            all_sequences.append(window)
            all_labels.append(label_encoded)
            all_video_ids.append(vid_id)

    if not all_sequences:
        return np.array([]), np.array([]), []

    X = np.stack(all_sequences, axis=0)
    y = np.array(all_labels, dtype=np.int64)

    return X, y, all_video_ids


# Removed: row-level augmented balanced CSV path replaced by manifest-based splitting.
# The build_balanced_landmark_sequences_for_exercise function previously loaded from
# balanced_landmark_features.csv (which contained aug_ prefixed rows), called
# sort_index() destroying temporal order, and used stride=15 for all partitions.
# All three issues are fixed in build_partition_sequences above.
#
# def build_balanced_landmark_sequences_for_exercise(...): REMOVED


def main():
    parser = argparse.ArgumentParser(description="Build sequence datasets for BiLSTM")
    parser.add_argument("--exercise", type=str, default=None,
                        help="Build for a specific exercise only")
    parser.add_argument("--seq-len", type=int, default=60,
                        help="Sequence length in frames (default: 60)")
    parser.add_argument("--stride", type=int, default=15,
                        help="Stride between windows (default: 15, only for legacy mode)")
    parser.add_argument("--use-landmarks", action="store_true",
                        help="Use raw normalized landmarks (99 features) instead of hand-crafted features")
    parser.add_argument("--partition", type=str, default="all",
                        choices=["train", "val", "test", "all"],
                        help="Which partition(s) to build (default: all)")
    parser.add_argument("--use-manifests", action="store_true", default=True,
                        help="Use manifest-based splitting (default: True). "
                             "Set --no-use-manifests for legacy all-at-once mode.")
    parser.add_argument("--no-use-manifests", dest="use_manifests", action="store_false",
                        help="Disable manifest-based splitting (legacy mode)")
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR,
                        help=f"Directory with manifest .txt files (default: {SPLITS_DIR})")
    parser.add_argument("--landmarks-dir", type=Path, default=LANDMARKS_DIR,
                        help=f"Directory with landmark CSVs (default: {LANDMARKS_DIR})")
    parser.add_argument("--sequences-dir", type=Path, default=SEQUENCES_DIR,
                        help=f"Output directory for .npz files (default: {SEQUENCES_DIR})")
    args = parser.parse_args()

    exercises = [args.exercise] if args.exercise else EXERCISES

    # Default seq_len for landmarks is 30 (unless user overrides)
    if args.use_landmarks and args.seq_len == 60:
        args.seq_len = 30

    args.sequences_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building sequence datasets:")
    print(f"  Mode: {'raw landmarks (99 features)' if args.use_landmarks else 'hand-crafted features'}")
    if args.use_landmarks and args.use_manifests:
        print(f"  Source: manifest-based (data/processed/splits/)")
        print(f"  Partition(s): {args.partition}")
    print(f"  Sequence length: {args.seq_len} frames")
    print(f"  Landmarks dir: {args.landmarks_dir}")
    print(f"  Output dir: {args.sequences_dir}\n")

    summary = {}

    if args.use_landmarks and args.use_manifests:
        # --- New manifest-based path (correct) ---
        partitions = ["train", "val", "test"] if args.partition == "all" else [args.partition]
        lfe = LandmarkFeatureExtractor()

        for exercise in exercises:
            # Check that manifests exist for this exercise
            missing_manifests = []
            for part in partitions:
                manifest = args.splits_dir / f"{exercise}_{part}.txt"
                if not manifest.exists():
                    missing_manifests.append(str(manifest))

            if missing_manifests:
                print(f"[{exercise}] ERROR: Missing manifests:")
                for m in missing_manifests:
                    print(f"  {m}")
                print(f"  Run: python scripts/split_videos.py --exercise {exercise}")
                continue

            print(f"Processing {exercise}...")
            ex_summary = {}

            for part in partitions:
                X, y, video_ids = build_partition_sequences(
                    exercise=exercise,
                    partition=part,
                    splits_dir=args.splits_dir,
                    landmarks_dir=args.landmarks_dir,
                    sequences_dir=args.sequences_dir,
                    seq_len=args.seq_len,
                    lfe=lfe,
                )

                if len(X) == 0:
                    print(f"  [{part}] No sequences generated.")
                    continue

                n_correct = int((y == 1).sum())
                n_incorrect = int((y == 0).sum())
                n_videos = len(set(video_ids))
                out_path = args.sequences_dir / f"{exercise}_{part}_sequences.npz"
                size_mb = out_path.stat().st_size / (1024 * 1024)
                print(f"  [{part}] {len(X)} seqs ({n_correct} correct, {n_incorrect} incorrect), "
                      f"{n_videos} videos, shape={X.shape} ({size_mb:.1f} MB)")
                ex_summary[part] = {"sequences": len(X), "correct": n_correct,
                                    "incorrect": n_incorrect, "videos": n_videos}

            summary[exercise] = ex_summary

    elif args.use_landmarks:
        # Legacy non-manifest landmark mode
        lfe = LandmarkFeatureExtractor()
        for exercise in exercises:
            print(f"Processing {exercise}...")
            X, y, video_ids = build_landmark_sequences_for_exercise(
                exercise, lfe, seq_len=args.seq_len, stride=args.stride
            )

            if len(X) == 0:
                print(f"  [SKIP] No sequences generated for {exercise}")
                continue

            n_correct = int((y == 1).sum())
            n_incorrect = int((y == 0).sum())
            n_videos = len(set(video_ids))

            out_path = args.sequences_dir / f"{exercise}_landmark_sequences.npz"
            np.savez_compressed(
                out_path,
                X=X,
                y=y,
                video_ids=np.array(video_ids),
                feature_names=np.array(LANDMARK_FEATURE_NAMES),
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
    else:
        fe = FeatureExtractor(use_world=True)
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

            out_path = args.sequences_dir / f"{exercise}_sequences.npz"
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
        if args.use_landmarks and args.use_manifests:
            for ex in sorted(summary.keys()):
                print(f"\n{ex}:")
                for part, s in summary[ex].items():
                    print(f"  {part}: {s['sequences']} seqs ({s['correct']} correct, "
                          f"{s['incorrect']} incorrect, {s['videos']} videos)")
        else:
            print(f"{'Exercise':<18} {'Seqs':>8} {'Correct':>8} {'Incorrect':>8} {'Videos':>8}")
            print("-" * 55)
            total_seqs = 0
            for ex in sorted(summary.keys()):
                s = summary[ex]
                seqs = s.get("sequences", 0)
                print(f"{ex:<18} {seqs:>8} {s.get('correct', 0):>8} {s.get('incorrect', 0):>8} {s.get('videos', 0):>8}")
                total_seqs += seqs
            print("-" * 55)
            print(f"{'TOTAL':<18} {total_seqs:>8}")


if __name__ == "__main__":
    main()
