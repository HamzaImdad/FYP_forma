"""
Split source videos into train/val/test partitions BEFORE any augmentation.

This is Stage 1 of the corrected data pipeline. It must be run once before
build_sequences.py or any augmentation. The manifests it produces are the
single source of truth for which CSV files belong to each partition.

Output: data/processed/splits/{exercise}_{train|val|test}.txt
  One CSV filename (basename) per line, e.g.:
    youtube_squat_correct_ABC123.csv
    youtube_squat_correct_ABC123.f398.csv

Usage:
    python scripts/split_videos.py                     # all exercises
    python scripts/split_videos.py --exercise squat    # single exercise
    python scripts/split_videos.py --force             # overwrite existing manifests
    python scripts/split_videos.py --dry-run           # print stats, no files written
"""

import sys
import argparse
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# Add project root so src/ imports work
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import EXERCISES
from src.utils.video_parsing import parse_video_csv_name, get_base_video_id


LANDMARKS_DIR = PROJECT_ROOT / "data" / "processed" / "landmarks"
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"


def _collect_exercise_csvs(
    exercise: str, landmarks_dir: Path
) -> Tuple[List[Path], List[str], List[str]]:
    """Collect all non-augmented, non-image CSV files for an exercise.

    Returns:
        csv_files: list of Path objects
        base_ids:  list of base video IDs (stripped of .f398) for grouping
        labels:    list of label strings ("correct" or "incorrect")
    """
    csv_files = []
    base_ids = []
    labels = []

    for csv_path in sorted(landmarks_dir.glob("*.csv")):
        name = csv_path.name

        # Exclude augmented files — they must NEVER appear in manifests
        if name.startswith("aug_"):
            continue

        source, exercise_found, label, video_id = parse_video_csv_name(csv_path)

        # Skip kaggle_images_ and unparseable files (parse_video_csv_name returns Nones)
        if source is None:
            continue

        if exercise_found != exercise:
            continue

        base_id = get_base_video_id(csv_path)
        if base_id is None:
            continue

        csv_files.append(csv_path)
        base_ids.append(base_id)
        labels.append(label)

    return csv_files, base_ids, labels


def split_exercise_videos(
    exercise: str,
    landmarks_dir: Path,
    splits_dir: Path,
    force: bool = False,
    dry_run: bool = False,
) -> str:
    """Generate train/val/test manifests for a single exercise.

    Splitting strategy:
    - Groups all .f398 variants by base video ID to prevent leakage
    - test_size=0.20 (or 0.10 for exercises with <=3 unique videos per class)
    - val_size=0.15 of train+val remainder
    - Uses GroupShuffleSplit with random_state=42 for reproducibility

    Args:
        exercise:      Exercise name (from EXERCISES list)
        landmarks_dir: Directory containing landmark CSV files
        splits_dir:    Directory to write manifest .txt files
        force:         Overwrite existing manifests if True
        dry_run:       Print stats but do not write files

    Returns:
        "skipped"   — manifests already existed and force=False
        "dry-run"   — dry_run=True, nothing written
        "ok"        — manifests written successfully
    """
    train_path = splits_dir / f"{exercise}_train.txt"
    val_path = splits_dir / f"{exercise}_val.txt"
    test_path = splits_dir / f"{exercise}_test.txt"

    # Idempotency guard
    if not force and not dry_run:
        if train_path.exists() and val_path.exists() and test_path.exists():
            print(f"[{exercise}] Manifests already exist. Use --force to overwrite.")
            return "skipped"

    csv_files, base_ids, labels = _collect_exercise_csvs(exercise, landmarks_dir)

    if not csv_files:
        print(f"[{exercise}] No CSV files found in {landmarks_dir}. Skipping.")
        return "skipped"

    # Count unique base video IDs per class
    unique_base_ids = list(set(base_ids))
    correct_ids = {bid for bid, lbl in zip(base_ids, labels) if lbl == "correct"}
    incorrect_ids = {bid for bid, lbl in zip(base_ids, labels) if lbl != "correct"}

    n_correct = len(correct_ids)
    n_incorrect = len(incorrect_ids)
    n_unique = len(unique_base_ids)

    # Use smaller test split for exercises with few source videos
    min_class_size = min(n_correct, n_incorrect) if (n_correct > 0 and n_incorrect > 0) else n_unique
    test_size = 0.10 if min_class_size <= 3 else 0.20

    print(f"\n[{exercise}] {len(csv_files)} CSVs | {n_unique} unique base IDs "
          f"({n_correct} correct, {n_incorrect} incorrect)")

    # Convert to arrays for sklearn
    indices = np.arange(len(csv_files))
    groups = np.array(base_ids)

    # If we only have 1 unique video, we can't split — put everything in train
    if n_unique == 1:
        print(f"[{exercise}] WARNING: Only 1 unique video. All data goes to train.")
        train_files = [f.name for f in csv_files]
        val_files = []
        test_files = []
    elif n_unique == 2:
        # Special case: 2 unique videos, one to test, one to train (no val)
        print(f"[{exercise}] WARNING: Only 2 unique videos. No val split possible.")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        train_idx, test_idx = next(gss.split(indices, groups=groups))
        train_files = [csv_files[i].name for i in train_idx]
        val_files = []
        test_files = [csv_files[i].name for i in test_idx]
    else:
        # Standard 3-way split: train+val vs test
        try:
            gss_outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
            trainval_idx, test_idx = next(gss_outer.split(indices, groups=groups))

            # Check if we have at least 2 unique groups in trainval for the inner split
            trainval_groups = groups[trainval_idx]
            unique_trainval = len(set(trainval_groups))

            if unique_trainval < 2:
                print(f"[{exercise}] WARNING: Too few groups for val split. Skipping val.")
                train_files = [csv_files[i].name for i in trainval_idx]
                val_files = []
                test_files = [csv_files[i].name for i in test_idx]
            else:
                # Inner split: train vs val (15% of train+val)
                trainval_indices = np.arange(len(trainval_idx))
                trainval_groups_arr = groups[trainval_idx]

                gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
                train_local_idx, val_local_idx = next(
                    gss_inner.split(trainval_indices, groups=trainval_groups_arr)
                )

                train_global_idx = trainval_idx[train_local_idx]
                val_global_idx = trainval_idx[val_local_idx]

                train_files = [csv_files[i].name for i in train_global_idx]
                val_files = [csv_files[i].name for i in val_global_idx]
                test_files = [csv_files[i].name for i in test_idx]

        except ValueError as e:
            print(f"[{exercise}] WARNING: Split failed ({e}). Assigning all to train.")
            train_files = [f.name for f in csv_files]
            val_files = []
            test_files = []

    # Print partition stats
    def count_labels(files):
        c = sum(1 for f in files if "_correct_" in f)
        i = sum(1 for f in files if "_incorrect_" in f)
        return c, i

    tc, ti = count_labels(train_files)
    vc, vi = count_labels(val_files)
    ec, ei = count_labels(test_files)

    print(f"  train: {len(train_files):3d} files ({tc} correct, {ti} incorrect)")
    print(f"  val:   {len(val_files):3d} files ({vc} correct, {vi} incorrect)")
    print(f"  test:  {len(test_files):3d} files ({ec} correct, {ei} incorrect)")

    if dry_run:
        return "dry-run"

    # Write manifests
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_path.write_text("\n".join(train_files) + ("\n" if train_files else ""))
    val_path.write_text("\n".join(val_files) + ("\n" if val_files else ""))
    test_path.write_text("\n".join(test_files) + ("\n" if test_files else ""))

    print(f"  -> Written to {splits_dir}/")
    return "ok"


def main():
    parser = argparse.ArgumentParser(
        description="Split source videos into train/val/test manifests before augmentation."
    )
    parser.add_argument(
        "--exercise",
        type=str,
        default=None,
        help="Single exercise to process (default: all 10 exercises)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing manifests",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print partition stats without writing files",
    )
    parser.add_argument(
        "--landmarks-dir",
        type=Path,
        default=LANDMARKS_DIR,
        help=f"Directory with landmark CSV files (default: {LANDMARKS_DIR})",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=SPLITS_DIR,
        help=f"Output directory for manifest files (default: {SPLITS_DIR})",
    )
    args = parser.parse_args()

    exercises = [args.exercise] if args.exercise else EXERCISES

    print(f"Landmarks dir: {args.landmarks_dir}")
    print(f"Splits dir:    {args.splits_dir}")
    print(f"Force:         {args.force}")
    print(f"Dry run:       {args.dry_run}")
    print(f"Exercises:     {exercises}")

    for exercise in exercises:
        split_exercise_videos(
            exercise,
            landmarks_dir=args.landmarks_dir,
            splits_dir=args.splits_dir,
            force=args.force,
            dry_run=args.dry_run,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
