"""
Build DTW reference templates from correct-form landmark CSVs.

Walks data/processed/landmarks/, finds CSVs matching "*{exercise}_correct*",
replays each CSV through the exercise's dedicated detector to find rep
boundaries, extracts the torso-normalized 99-dim sequence for each rep,
ranks reps by the detector's own form score, and saves the top-N as an
.npz template file used by DTWMatcher at runtime.

Usage:
  "C:/Users/Hamza/AppData/Local/Programs/Python/Python310/python.exe" \\
      scripts/build_dtw_templates.py --exercise squat --num-templates 5
  "C:/Users/Hamza/AppData/Local/Programs/Python/Python310/python.exe" \\
      scripts/build_dtw_templates.py --exercise all
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.base_detector import BaseExerciseDetector, RepPhase  # noqa: E402
from src.classification.bench_press_detector import BenchPressDetector  # noqa: E402
from src.classification.bicep_curl_detector import BicepCurlDetector  # noqa: E402
from src.classification.deadlift_detector import DeadliftDetector  # noqa: E402
from src.classification.lunge_detector import LungeDetector  # noqa: E402
from src.classification.overhead_press_detector import OverheadPressDetector  # noqa: E402
from src.classification.plank_detector import PlankDetector  # noqa: E402
from src.classification.pullup_detector import PullUpDetector  # noqa: E402
from src.classification.squat_detector import SquatDetector  # noqa: E402
from src.classification.tricep_dip_detector import TricepDipDetector  # noqa: E402
from src.feature_extraction.landmark_features import LandmarkFeatureExtractor  # noqa: E402
from src.pose_estimation.base import PoseResult  # noqa: E402
from src.utils.constants import LANDMARK_NAMES, NUM_LANDMARKS  # noqa: E402


LANDMARKS_DIR = PROJECT_ROOT / "data" / "processed" / "landmarks"
TEMPLATES_DIR = PROJECT_ROOT / "models" / "templates"

# Plank and pushup are excluded: plank is static (no reps), pushup uses a
# standalone detector. Both can be added later if desired.
DETECTOR_FACTORIES: Dict[str, type] = {
    "squat": SquatDetector,
    "deadlift": DeadliftDetector,
    "lunge": LungeDetector,
    "bench_press": BenchPressDetector,
    "overhead_press": OverheadPressDetector,
    "bicep_curl": BicepCurlDetector,
    "tricep_dip": TricepDipDetector,
    "pullup": PullUpDetector,
    "plank": PlankDetector,  # static — script skips gracefully
}

SUPPORTED_EXERCISES = list(DETECTOR_FACTORIES.keys())


def row_to_pose_result(row: pd.Series) -> PoseResult:
    """Convert a landmark CSV row to PoseResult. Mirrors build_sequences.py.

    Handles both column naming conventions found in data/processed/landmarks/:
      - New: world_{name}_{x,y,z}, {name}_visibility
      - Legacy: {name}_w{x,y,z}, {name}_vis
    """
    landmarks = np.zeros((NUM_LANDMARKS, 4))
    world_landmarks = np.zeros((NUM_LANDMARKS, 3))
    for i, name in enumerate(LANDMARK_NAMES):
        landmarks[i, 0] = float(row.get(f"{name}_x", 0.0) or 0.0)
        landmarks[i, 1] = float(row.get(f"{name}_y", 0.0) or 0.0)
        landmarks[i, 2] = float(row.get(f"{name}_z", 0.0) or 0.0)
        vis_val = row.get(f"{name}_visibility", row.get(f"{name}_vis", 0.0))
        landmarks[i, 3] = float(vis_val or 0.0)
        world_landmarks[i, 0] = float(row.get(f"world_{name}_x", row.get(f"{name}_wx", 0.0)) or 0.0)
        world_landmarks[i, 1] = float(row.get(f"world_{name}_y", row.get(f"{name}_wy", 0.0)) or 0.0)
        world_landmarks[i, 2] = float(row.get(f"world_{name}_z", row.get(f"{name}_wz", 0.0)) or 0.0)

    try:
        timestamp = int(row.get("timestamp_ms", 0) or 0)
    except (ValueError, TypeError):
        timestamp = 0

    return PoseResult(
        landmarks=landmarks,
        world_landmarks=world_landmarks,
        detection_confidence=float(landmarks[:, 3].mean()),
        timestamp_ms=timestamp,
    )


def _find_csvs(exercise: str) -> List[Path]:
    """Find correct-form landmark CSVs for an exercise.

    Current layout has flat files like `kaggle_squat_correct_squat_1.csv` in
    `data/processed/landmarks/`. We match on `*{exercise}_correct*`.
    """
    patterns = [
        str(LANDMARKS_DIR / f"*{exercise}_correct*.csv"),
        str(LANDMARKS_DIR / exercise / "correct" / "*.csv"),
    ]
    found: List[Path] = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            found.append(Path(path))
    # Deduplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for p in found:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def extract_reps_from_csv(
    csv_path: Path,
    exercise: str,
    landmark_extractor: LandmarkFeatureExtractor,
) -> List[Tuple[np.ndarray, float]]:
    """Replay a CSV through the detector and return completed reps.

    Returns a list of (sequence, avg_score) tuples. `sequence` is a
    torso-normalized (T, 99) float32 array for one rep; `avg_score` is the
    detector's own per-rep form score in [0, 1].
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []

    if df.empty:
        return []

    detector_cls = DETECTOR_FACTORIES[exercise]
    detector: BaseExerciseDetector = detector_cls()

    reps: List[Tuple[np.ndarray, float]] = []
    current_seq: List[np.ndarray] = []
    prev_rep_count = 0

    for row_idx, (_, row) in enumerate(df.iterrows()):
        pose = row_to_pose_result(row)
        # Use a synthetic per-frame timestamp so the detector doesn't bail
        # on MIN_REP_DURATION. Real CSVs are at ~30 FPS.
        ts = row_idx / 30.0

        try:
            result = detector.classify(pose, timestamp=ts)
        except Exception:
            continue

        # Capture landmarks while the detector is in a mid-rep phase
        if detector._state != RepPhase.TOP:
            try:
                lm = landmark_extractor.extract_landmarks(pose)
                current_seq.append(lm)
            except Exception:
                pass

        # On rep completion, the detector appends to its rep_history
        if detector.rep_count > prev_rep_count:
            prev_rep_count = detector.rep_count
            if detector._rep_history:
                last_rep = detector._rep_history[-1]
                score = float(last_rep.get("score", 0.0))
            else:
                score = 0.0
            if len(current_seq) >= 3:
                seq_arr = np.stack(current_seq).astype(np.float32)
                reps.append((seq_arr, score))
            current_seq = []

    return reps


def _process_csv_worker(args: Tuple[str, str]) -> List[Tuple[np.ndarray, float]]:
    """Process pool worker — needs module-level callable + picklable args."""
    csv_path_str, exercise = args
    extractor = LandmarkFeatureExtractor()
    return extract_reps_from_csv(Path(csv_path_str), exercise, extractor)


def build_templates_for_exercise(
    exercise: str,
    num_templates: int,
    min_rep_frames: int = 6,
    max_rep_frames: int = 240,
    workers: int = max(1, (os.cpu_count() or 4) - 2),
) -> Optional[Path]:
    """Build and save template file for one exercise.

    Uses a process pool to replay CSVs in parallel — matches the user's
    "always maximize hardware utilization" preference.
    """
    if exercise not in DETECTOR_FACTORIES:
        print(f"[{exercise}] no detector registered, skipping")
        return None

    csvs = _find_csvs(exercise)
    if not csvs:
        print(f"[{exercise}] no correct-form CSVs found under {LANDMARKS_DIR}")
        return None

    print(f"[{exercise}] found {len(csvs)} correct CSVs, extracting reps with {workers} workers")

    all_reps: List[Tuple[np.ndarray, float]] = []
    tasks = [(str(p), exercise) for p in csvs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_csv_worker, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=exercise):
            try:
                reps = fut.result()
            except Exception as e:
                print(f"  worker error: {e}")
                continue
            all_reps.extend(reps)

    # Filter by length; static exercises (plank) won't produce any reps
    filtered = [
        (seq, score) for seq, score in all_reps
        if min_rep_frames <= seq.shape[0] <= max_rep_frames
    ]

    if not filtered:
        print(f"[{exercise}] no valid reps after filtering (total extracted: {len(all_reps)})")
        return None

    # Rank by detector's own form score — pick the best
    filtered.sort(key=lambda t: t[1], reverse=True)
    chosen = filtered[:num_templates]

    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TEMPLATES_DIR / f"{exercise}.npz"

    templates_obj = np.empty(len(chosen), dtype=object)
    for i, (seq, _score) in enumerate(chosen):
        templates_obj[i] = seq
    lengths = np.array([seq.shape[0] for seq, _ in chosen], dtype=np.int32)
    scores = np.array([s for _, s in chosen], dtype=np.float32)

    np.savez(out_path, templates=templates_obj, lengths=lengths, scores=scores)

    avg_len = np.mean(lengths)
    print(
        f"[{exercise}] saved {len(chosen)} templates to {out_path} "
        f"(avg length {avg_len:.0f} frames, avg score {scores.mean():.2f})"
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build DTW reference templates")
    parser.add_argument(
        "--exercise",
        nargs="+",
        default=["squat", "bicep_curl", "overhead_press"],
        help="Exercise name(s) or 'all'",
    )
    parser.add_argument("--num-templates", type=int, default=5,
                        help="Top-N reps to keep as templates")
    parser.add_argument("--min-rep-frames", type=int, default=6)
    parser.add_argument("--max-rep-frames", type=int, default=240)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = parser.parse_args()

    requested = args.exercise
    if len(requested) == 1 and requested[0].lower() == "all":
        requested = SUPPORTED_EXERCISES

    for ex in requested:
        if ex not in DETECTOR_FACTORIES:
            print(f"[{ex}] unsupported (supported: {SUPPORTED_EXERCISES})")
            continue
        build_templates_for_exercise(
            ex,
            num_templates=args.num_templates,
            min_rep_frames=args.min_rep_frames,
            max_rep_frames=args.max_rep_frames,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
