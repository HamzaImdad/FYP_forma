"""
Shared utility for parsing video landmark CSV filenames.

Provides a canonical parser for the naming convention used across the pipeline:
  {source}_{exercise}_{label}_{video_id}.csv
  e.g. youtube_squat_correct_ABC123.csv
       youtube_overhead_press_incorrect_XYZ.csv
       youtube_squat_correct_ABC123.f398.csv  (yt-dlp format variant)
       kaggle_images_squat_correct_01234.csv  (image dataset, no video sequence)

Used by: scripts/split_videos.py, scripts/build_sequences.py
"""

from pathlib import Path
from typing import Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.constants import EXERCISES


def parse_video_csv_name(csv_path) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Parse a video landmark CSV filename into (source, exercise, label, video_id).

    Handles:
    - Single-word exercises: squat, lunge, deadlift, pushup, pullup, plank
    - Two-word exercises: bench_press, overhead_press, bicep_curl, tricep_dip
    - .f398 format variants (yt-dlp alternative stream downloads): stem ends in .f398
    - kaggle_images_ prefix: returns (None, None, None, None) — not a video sequence

    Args:
        csv_path: Path object or string of the CSV filename or full path.

    Returns:
        Tuple of (source, exercise, label, video_id) where video_id is the
        original stem (with .f398 intact). Returns (None, None, None, None)
        for kaggle_images_ files or unparseable filenames.

    Raises:
        ValueError: If the filename structure is recognisable but inconsistent
                    (e.g. starts with youtube_ but contains an unknown exercise).
    """
    path = Path(csv_path)

    # Work on the stem only (strip .csv extension)
    # Handle double-extension like "name.f398.csv" -> stem is "name.f398"
    stem = path.stem  # e.g. "youtube_squat_correct_ABC123" or "youtube_squat_correct_ABC123.f398"

    # Strip .f398 suffix for parsing (keep original stem for video_id)
    parse_stem = stem
    if parse_stem.endswith(".f398"):
        parse_stem = parse_stem[:-5]  # remove ".f398"

    # Skip kaggle image datasets — they are not video sequences
    if parse_stem.startswith("kaggle_images_"):
        return None, None, None, None

    parts = parse_stem.split("_")

    if len(parts) < 3:
        return None, None, None, None

    source = parts[0]

    # Try to find exercise by checking against EXERCISES list
    # Try two-word exercise first (parts[1]_parts[2]), then one-word (parts[1])
    exercise = None
    label_idx = None

    if len(parts) >= 4:
        candidate_two = f"{parts[1]}_{parts[2]}"
        if candidate_two in EXERCISES:
            exercise = candidate_two
            label_idx = 3

    if exercise is None:
        candidate_one = parts[1]
        if candidate_one in EXERCISES:
            exercise = candidate_one
            label_idx = 2

    if exercise is None or label_idx is None or label_idx >= len(parts):
        return None, None, None, None

    label = parts[label_idx]
    # video_id is the original stem (preserving .f398 for traceability)
    video_id = stem

    return source, exercise, label, video_id


def get_base_video_id(csv_path) -> Optional[str]:
    """Return the base video ID with .f398 variant suffix stripped.

    This is used for grouping format variants of the same source video into
    the same train/val/test partition to prevent data leakage.

    For example:
        youtube_squat_correct_ABC123.csv      -> "youtube_squat_correct_ABC123"
        youtube_squat_correct_ABC123.f398.csv -> "youtube_squat_correct_ABC123"

    Returns None for kaggle_images_ files or unparseable filenames.
    """
    path = Path(csv_path)
    stem = path.stem  # strips .csv

    # Strip .f398 suffix
    if stem.endswith(".f398"):
        return stem[:-5]

    # Quick check: if it starts with kaggle_images_, skip it
    if stem.startswith("kaggle_images_"):
        return None

    # Verify it's a parseable video CSV
    source, exercise, label, video_id = parse_video_csv_name(csv_path)
    if source is None:
        return None

    return stem
