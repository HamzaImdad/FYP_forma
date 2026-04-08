"""
Generate per-frame soft labels for all landmark CSVs using dedicated exercise detectors.

Runs each exercise's detector offline on landmark data, producing continuous form scores (0.0-1.0)
that can replace noisy video-level binary labels for BiLSTM training.

Output: data/processed/soft_labels/{csv_stem}_scores.csv
        reports/soft_label_summary.json

Usage:
    python scripts/generate_soft_labels.py
    python scripts/generate_soft_labels.py --exercise squat bicep_curl
    python scripts/generate_soft_labels.py --workers 10
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project setup ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.video_parsing import parse_video_csv_name
from src.utils.constants import EXERCISES, LANDMARK_NAMES, NUM_LANDMARKS
from src.pose_estimation.base import PoseResult

# ── Detector imports ──
from src.classification.squat_detector import SquatDetector
from src.classification.deadlift_detector import DeadliftDetector
from src.classification.lunge_detector import LungeDetector
from src.classification.bench_press_detector import BenchPressDetector
from src.classification.overhead_press_detector import OverheadPressDetector
from src.classification.pullup_detector import PullUpDetector
from src.classification.pushup_detector import PushUpDetector
from src.classification.plank_detector import PlankDetector
from src.classification.bicep_curl_detector import BicepCurlDetector
from src.classification.tricep_dip_detector import TricepDipDetector

# ── Paths ──
LANDMARKS_DIR = PROJECT_ROOT / "data" / "processed" / "landmarks"
SOFT_LABELS_DIR = PROJECT_ROOT / "data" / "processed" / "soft_labels"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Skip first N frames (angle buffer warmup) and last N frames (score buffer incomplete)
WARMUP_FRAMES = 3
COOLDOWN_FRAMES = 8

# Map exercise name to detector class
DETECTOR_MAP = {
    "squat": SquatDetector,
    "deadlift": DeadliftDetector,
    "lunge": LungeDetector,
    "bench_press": BenchPressDetector,
    "overhead_press": OverheadPressDetector,
    "pullup": PullUpDetector,
    "pushup": PushUpDetector,
    "plank": PlankDetector,
    "bicep_curl": BicepCurlDetector,
    "tricep_dip": TricepDipDetector,
}


def row_to_pose_result(row: pd.Series) -> PoseResult:
    """Convert a landmark CSV row to PoseResult. Reuses logic from build_sequences.py."""
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


def process_csv(csv_path: Path, exercise: str, label: str, detector_cls) -> dict:
    """Process a single landmark CSV through its detector, return per-frame scores."""
    df = pd.read_csv(csv_path)
    n_frames = len(df)

    if n_frames == 0:
        return {"csv": csv_path.name, "exercise": exercise, "label": label,
                "n_frames": 0, "scores": [], "error": "empty CSV"}

    # Create detector with coverage penalty for honest scoring
    # PushUpDetector is standalone — no coverage_penalty param
    is_pushup = (exercise == "pushup" and detector_cls == PushUpDetector)
    if is_pushup:
        detector = detector_cls()
    else:
        detector = detector_cls(coverage_penalty=True)

    scores = []
    for idx, (_, row) in enumerate(df.iterrows()):
        pose = row_to_pose_result(row)
        ts = row.get("timestamp_ms", idx * 33)  # ~30fps fallback
        timestamp = float(ts) / 1000.0 if ts else idx * 0.033

        if is_pushup:
            result = detector.classify(pose)
        else:
            result = detector.classify(pose, timestamp=timestamp)

        coverage = getattr(detector, '_last_coverage', 1.0)

        scores.append({
            "frame_idx": idx,
            "detector_score": round(result.form_score, 4),
            "is_active": result.is_active,
            "coverage": round(coverage, 3),
            "warmup": idx < WARMUP_FRAMES,
            "cooldown": idx >= n_frames - COOLDOWN_FRAMES,
        })

    return {
        "csv": csv_path.name,
        "exercise": exercise,
        "label": label,
        "n_frames": n_frames,
        "scores": scores,
    }


def process_exercise(exercise: str, csv_files: list) -> dict:
    """Process all CSVs for a single exercise. Runs in one process."""
    detector_cls = DETECTOR_MAP.get(exercise)
    if detector_cls is None:
        return {"exercise": exercise, "error": f"No detector for {exercise}", "files": 0}

    SOFT_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    total_frames = 0
    t0 = time.time()

    for csv_path, label in csv_files:
        result = process_csv(csv_path, exercise, label, detector_cls)

        if result.get("scores"):
            # Save per-frame scores
            scores_df = pd.DataFrame(result["scores"])
            scores_df["original_label"] = label
            out_path = SOFT_LABELS_DIR / f"{csv_path.stem}_scores.csv"
            scores_df.to_csv(out_path, index=False)
            total_frames += result["n_frames"]

        results.append({
            "csv": result["csv"],
            "label": label,
            "n_frames": result["n_frames"],
            "avg_score": np.mean([s["detector_score"] for s in result.get("scores", [])
                                  if not s.get("warmup") and not s.get("cooldown")]) if result.get("scores") else 0.0,
            "active_pct": np.mean([s["is_active"] for s in result.get("scores", [])]) if result.get("scores") else 0.0,
            "avg_coverage": np.mean([s["coverage"] for s in result.get("scores", [])]) if result.get("scores") else 0.0,
        })

    elapsed = time.time() - t0

    # Compute exercise-level stats
    correct_scores = [r["avg_score"] for r in results if r["label"] == "correct"]
    incorrect_scores = [r["avg_score"] for r in results if r["label"] == "incorrect"]

    summary = {
        "exercise": exercise,
        "n_files": len(csv_files),
        "total_frames": total_frames,
        "elapsed_sec": round(elapsed, 1),
        "n_correct": len(correct_scores),
        "n_incorrect": len(incorrect_scores),
        "correct_avg_score": round(float(np.mean(correct_scores)), 4) if correct_scores else None,
        "incorrect_avg_score": round(float(np.mean(incorrect_scores)), 4) if incorrect_scores else None,
        "correct_std": round(float(np.std(correct_scores)), 4) if correct_scores else None,
        "incorrect_std": round(float(np.std(incorrect_scores)), 4) if incorrect_scores else None,
        "per_video": results,
    }

    print(f"  {exercise}: {len(csv_files)} files, {total_frames} frames, "
          f"correct_avg={summary['correct_avg_score']}, incorrect_avg={summary['incorrect_avg_score']}, "
          f"{elapsed:.1f}s")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate soft labels using exercise detectors")
    parser.add_argument("--exercise", nargs="+", default=None,
                        help="Specific exercises to process (default: all)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of parallel processes (default: 10, one per exercise)")
    args = parser.parse_args()

    exercises = args.exercise or list(DETECTOR_MAP.keys())

    # Gather CSVs per exercise
    exercise_csvs = {ex: [] for ex in exercises}
    for csv_path in sorted(LANDMARKS_DIR.glob("*.csv")):
        if csv_path.name.startswith("aug_"):
            continue
        source, ex, label, vid_id = parse_video_csv_name(csv_path)
        if ex in exercise_csvs and label in ("correct", "incorrect"):
            exercise_csvs[ex].append((csv_path, label))

    total_files = sum(len(v) for v in exercise_csvs.values())
    print(f"Processing {total_files} landmark CSVs across {len(exercises)} exercises")
    print(f"Output: {SOFT_LABELS_DIR}")
    print()

    SOFT_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # Process exercises in parallel
    all_summaries = {}
    t_start = time.time()

    n_workers = min(args.workers, len(exercises))
    if n_workers <= 1:
        # Sequential
        for ex in exercises:
            if exercise_csvs[ex]:
                all_summaries[ex] = process_exercise(ex, exercise_csvs[ex])
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for ex in exercises:
                if exercise_csvs[ex]:
                    futures[pool.submit(process_exercise, ex, exercise_csvs[ex])] = ex

            for future in as_completed(futures):
                ex = futures[future]
                try:
                    all_summaries[ex] = future.result()
                except Exception as e:
                    print(f"  ERROR {ex}: {e}")
                    all_summaries[ex] = {"exercise": ex, "error": str(e)}

    total_time = time.time() - t_start

    # Save summary report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_files": total_files,
        "total_time_sec": round(total_time, 1),
        "exercises": {},
    }

    print()
    print("=" * 80)
    print(f"{'Exercise':<18} {'Files':>6} {'Correct Avg':>12} {'Incorrect Avg':>14} {'Gap':>8}")
    print("-" * 80)

    for ex in exercises:
        s = all_summaries.get(ex, {})
        if "error" in s:
            print(f"  {ex}: ERROR — {s['error']}")
            continue

        c_avg = s.get("correct_avg_score")
        i_avg = s.get("incorrect_avg_score")
        gap = round(c_avg - i_avg, 4) if (c_avg is not None and i_avg is not None) else None

        print(f"{ex:<18} {s.get('n_files', 0):>6} {c_avg or 'N/A':>12} {i_avg or 'N/A':>14} {gap or 'N/A':>8}")

        # Store summary without per-video detail (too large for report)
        report["exercises"][ex] = {k: v for k, v in s.items() if k != "per_video"}

    print("=" * 80)
    print(f"Total time: {total_time:.1f}s")

    report_path = REPORTS_DIR / "soft_label_summary.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Summary saved to {report_path}")

    # Also save detailed per-video report
    detailed_report = {ex: all_summaries.get(ex, {}) for ex in exercises}
    detailed_path = REPORTS_DIR / "soft_label_detailed.json"
    with open(detailed_path, "w") as f:
        json.dump(detailed_report, f, indent=2, default=str)
    print(f"Detailed report saved to {detailed_path}")


if __name__ == "__main__":
    main()
