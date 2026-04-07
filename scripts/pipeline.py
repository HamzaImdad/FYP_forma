"""
ExerVision Unified Data Pipeline — Add Data & Retrain in One Command.

Chains all stages: extract landmarks -> split -> build sequences -> train -> report.
Maximizes hardware: all CPU cores for extraction, GPU for training, full RAM.

Usage:
    # Full pipeline for all exercises (detects new unprocessed videos automatically)
    python scripts/pipeline.py

    # Full pipeline for specific exercise(s)
    python scripts/pipeline.py --exercise squat deadlift

    # Add new videos and retrain only affected exercises
    python scripts/pipeline.py --videos path/to/video1.mp4 path/to/video2.mp4 --exercise squat --label correct

    # Skip extraction (landmarks already exist, just rebuild sequences + retrain)
    python scripts/pipeline.py --skip-extract

    # Skip training (just rebuild data pipeline, don't retrain)
    python scripts/pipeline.py --skip-train

    # Force rebuild everything from scratch
    python scripts/pipeline.py --force

    # Dry run — show what would be done
    python scripts/pipeline.py --dry-run
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Set, Tuple

import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PYTHON = "C:/Users/Hamza/AppData/Local/Programs/Python/Python310/python.exe"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"
LANDMARKS_DIR = DATA_DIR / "processed" / "landmarks"
SPLITS_DIR = DATA_DIR / "processed" / "splits"
SEQUENCES_DIR = DATA_DIR / "processed" / "sequences"
MODELS_DIR = PROJECT_ROOT / "models" / "trained"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Video sources to scan for new data
VIDEO_SOURCES = {
    "youtube": DATA_DIR / "datasets" / "youtube",
    "youtube_incorrect": DATA_DIR / "datasets" / "youtube_incorrect",
    "kaggle": DATA_DIR / "datasets" / "kaggle_workout",
    "raw": DATA_DIR / "raw",
}

VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}

# Per-exercise feature dimension config (matches current best cherry-picked models)
# This is the SOURCE OF TRUTH for what dim each exercise trains with.
EXERCISE_FEATURE_CONFIG = {
    "squat":          {"dim": 99,  "mode": "landmarks"},
    "deadlift":       {"dim": 99,  "mode": "landmarks"},
    "bench_press":    {"dim": 99,  "mode": "landmarks"},
    "bicep_curl":     {"dim": 99,  "mode": "landmarks"},
    "pullup":         {"dim": 109, "mode": "hybrid"},
    "pushup":         {"dim": 109, "mode": "hybrid"},
    "plank":          {"dim": 109, "mode": "hybrid"},
    "lunge":          {"dim": 330, "mode": "temporal"},
    "overhead_press": {"dim": 330, "mode": "temporal"},
    "tricep_dip":     {"dim": 330, "mode": "temporal"},
}

ALL_EXERCISES = list(EXERCISE_FEATURE_CONFIG.keys())


# ─── Utility ────────────────────────────────────────────────────────────────

def _ts():
    return datetime.now().strftime("%H:%M:%S")


def _banner(msg: str):
    w = max(len(msg) + 4, 60)
    print(f"\n{'=' * w}")
    print(f"  {msg}")
    print(f"{'=' * w}\n")


def _run(cmd: List[str], desc: str, dry_run: bool = False) -> subprocess.CompletedProcess:
    """Run a subprocess, streaming output live."""
    print(f"[{_ts()}] {desc}")
    print(f"  $ {' '.join(cmd[:5])}{'...' if len(cmd) > 5 else ''}")
    if dry_run:
        print("  [DRY RUN — skipped]")
        return subprocess.CompletedProcess(cmd, 0)

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=None,  # Stream to terminal
        stderr=None,
        text=True,
    )
    if result.returncode != 0:
        print(f"  WARNING: Command exited with code {result.returncode}")
    return result


def get_cpu_count() -> int:
    """Get available CPU cores for parallel work."""
    try:
        count = os.cpu_count() or 4
        # Use all cores — user explicitly wants max utilization
        return count
    except Exception:
        return 4


def get_existing_landmarks() -> Set[str]:
    """Get set of already-extracted landmark CSV basenames."""
    if not LANDMARKS_DIR.exists():
        return set()
    return {f.stem for f in LANDMARKS_DIR.glob("*.csv")}


def detect_new_videos() -> Dict[str, List[Tuple[Path, str, str]]]:
    """Scan all video sources for unprocessed videos.

    Returns: {exercise: [(video_path, label, source), ...]}
    """
    existing = get_existing_landmarks()
    new_videos: Dict[str, List[Tuple[Path, str, str]]] = {}

    for source_name, source_dir in VIDEO_SOURCES.items():
        if not source_dir.exists():
            continue

        if source_name == "kaggle":
            # Kaggle: flat structure, all "correct"
            for video in source_dir.iterdir():
                if video.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue
                exercise = video.stem.split("_")[0] if "_" in video.stem else video.stem
                csv_stem = f"kaggle_{exercise}_correct_{video.stem}"
                if csv_stem not in existing:
                    new_videos.setdefault(exercise, []).append(
                        (video, "correct", "kaggle")
                    )
        elif source_name == "youtube_incorrect":
            # youtube_incorrect/{exercise}/video.mp4 — all incorrect
            for ex_dir in source_dir.iterdir():
                if not ex_dir.is_dir():
                    continue
                exercise = ex_dir.name
                for video in ex_dir.iterdir():
                    if video.suffix.lower() not in VIDEO_EXTENSIONS:
                        continue
                    csv_stem = f"ytincorrect_{exercise}_incorrect_{video.stem}"
                    if csv_stem not in existing:
                        new_videos.setdefault(exercise, []).append(
                            (video, "incorrect", "ytincorrect")
                        )
        else:
            # youtube/{exercise}/{correct|incorrect}/video.mp4
            # raw/{exercise}/{correct|incorrect}/video.mp4
            for ex_dir in source_dir.iterdir():
                if not ex_dir.is_dir():
                    continue
                exercise = ex_dir.name
                for label_dir in ex_dir.iterdir():
                    if not label_dir.is_dir():
                        continue
                    label = label_dir.name
                    if label not in ("correct", "incorrect"):
                        continue
                    for video in label_dir.iterdir():
                        if video.suffix.lower() not in VIDEO_EXTENSIONS:
                            continue
                        csv_stem = f"{source_name}_{exercise}_{label}_{video.stem}"
                        if csv_stem not in existing:
                            new_videos.setdefault(exercise, []).append(
                                (video, label, source_name)
                            )

    return new_videos


def detect_affected_exercises(new_videos: Dict[str, list]) -> List[str]:
    """Determine which exercises need reprocessing."""
    return sorted(new_videos.keys())


# ─── Stage 1: Landmark Extraction ───────────────────────────────────────────

def stage_extract_landmarks(
    exercises: Optional[List[str]] = None,
    video_paths: Optional[List[str]] = None,
    exercise_hint: Optional[str] = None,
    label_hint: Optional[str] = None,
    dry_run: bool = False,
    force: bool = False,
) -> Set[str]:
    """Extract landmarks from new videos using all CPU cores.

    Returns: set of exercises that had new data extracted.
    """
    _banner("STAGE 1: LANDMARK EXTRACTION")
    workers = get_cpu_count()
    print(f"Using {workers} CPU workers for parallel extraction")

    # If specific video paths provided, use single-video extraction
    if video_paths:
        affected = set()
        for vp in video_paths:
            ex = exercise_hint or "unknown"
            lb = label_hint or "correct"
            cmd = [
                PYTHON, str(SCRIPTS_DIR / "extract_landmarks.py"),
                "--video", vp,
                "--exercise", ex,
                "--label", lb,
            ]
            _run(cmd, f"Extracting landmarks: {Path(vp).name}", dry_run)
            affected.add(ex)
        return affected

    # Otherwise use parallel extraction for all sources
    # The parallel extractor auto-skips already-processed CSVs, so it's safe
    # to run on all sources — only new/unprocessed videos get extracted.
    cmd = [
        PYTHON, str(SCRIPTS_DIR / "extract_landmarks_parallel.py"),
        "--workers", str(workers),
        "--source", "all",
    ]

    result = _run(cmd, f"Parallel landmark extraction ({workers} workers, all sources)", dry_run)

    # Detect which exercises had new data
    if exercises:
        return set(exercises)

    new_videos = detect_new_videos()
    return set(new_videos.keys()) if new_videos else set(ALL_EXERCISES)


# ─── Stage 2: Train/Val/Test Splitting ──────────────────────────────────────

def stage_split_videos(
    exercises: Optional[List[str]] = None,
    force: bool = False,
    dry_run: bool = False,
):
    """Create/update train/val/test manifests."""
    _banner("STAGE 2: TRAIN/VAL/TEST SPLIT")

    cmd = [PYTHON, str(SCRIPTS_DIR / "split_videos.py")]
    if exercises and len(exercises) < len(ALL_EXERCISES):
        # Split only affected exercises
        for ex in exercises:
            ex_cmd = cmd + ["--exercise", ex, "--force"]
            _run(ex_cmd, f"Splitting {ex}", dry_run)
    else:
        cmd.append("--force")
        _run(cmd, "Splitting all exercises", dry_run)


# ─── Stage 3: Build Sequences ───────────────────────────────────────────────

def _build_sequences_for_exercise(exercise: str, dry_run: bool = False):
    """Build sequences for one exercise with its configured feature mode."""
    config = EXERCISE_FEATURE_CONFIG[exercise]
    mode = config["mode"]

    cmd = [
        PYTHON, str(SCRIPTS_DIR / "build_sequences.py"),
        "--use-landmarks",
        "--exercise", exercise,
    ]

    if mode == "temporal":
        cmd.append("--temporal")
    elif mode == "hybrid":
        cmd.append("--hybrid")
    # mode == "landmarks" is the default (99-dim)

    _run(cmd, f"Building {config['dim']}-dim sequences for {exercise}", dry_run)


def stage_build_sequences(
    exercises: Optional[List[str]] = None,
    dry_run: bool = False,
):
    """Build sequence datasets for all target exercises.

    Uses multiprocessing to build multiple exercises in parallel (CPU-bound).
    """
    _banner("STAGE 3: BUILD SEQUENCES")
    targets = exercises or ALL_EXERCISES
    workers = min(get_cpu_count(), len(targets))
    print(f"Building sequences for {len(targets)} exercises ({workers} parallel workers)")

    if dry_run:
        for ex in targets:
            config = EXERCISE_FEATURE_CONFIG[ex]
            print(f"  [DRY RUN] {ex}: {config['dim']}-dim ({config['mode']})")
        return

    # Run in parallel — each exercise is independent
    # Use ThreadPoolExecutor since each spawns a subprocess anyway
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_build_sequences_for_exercise, ex): ex
            for ex in targets
        }
        for future in as_completed(futures):
            ex = futures[future]
            try:
                future.result()
                print(f"  [{_ts()}] {ex} sequences DONE")
            except Exception as e:
                print(f"  [{_ts()}] {ex} FAILED: {e}")


# ─── Stage 4: Training ──────────────────────────────────────────────────────

def _get_current_f1(exercise: str) -> Optional[float]:
    """Get the current best test F1 for an exercise from reports."""
    # Try curves file first
    report_path = REPORTS_DIR / f"{exercise}_bilstm_v2_curves.json"
    if report_path.exists():
        try:
            with open(report_path, encoding="utf-8") as f:
                data = json.load(f)
            f1 = data.get("test_f1") or data.get("f1")
            if f1 is not None:
                return f1
        except Exception:
            pass

    # Fall back to training summary
    summary_path = REPORTS_DIR / "bilstm_v2_training_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, encoding="utf-8") as f:
                data = json.load(f)
            if exercise in data:
                return data[exercise].get("f1")
        except Exception:
            pass

    return None


def stage_train(
    exercises: Optional[List[str]] = None,
    epochs: int = 500,
    dry_run: bool = False,
    backup: bool = True,
):
    """Train BiLSTM models for target exercises.

    Trains sequentially (GPU is the bottleneck, one model at a time maximizes GPU util).
    Backs up existing models before overwriting.
    """
    _banner("STAGE 4: TRAINING")
    targets = exercises or ALL_EXERCISES

    # Snapshot current F1 scores for comparison
    old_f1 = {}
    for ex in targets:
        f1 = _get_current_f1(ex)
        if f1 is not None:
            old_f1[ex] = f1

    if old_f1:
        print("Current best F1 scores:")
        for ex, f1 in sorted(old_f1.items()):
            print(f"  {ex:<18} {f1:.3f}")
        print()

    for ex in targets:
        config = EXERCISE_FEATURE_CONFIG[ex]

        # Backup existing model before overwriting
        if backup:
            model_path = MODELS_DIR / f"{ex}_bilstm_v2.pt"
            if model_path.exists():
                backup_path = MODELS_DIR / f"{ex}_bilstm_v2_pre_pipeline.pt"
                if not backup_path.exists():
                    if not dry_run:
                        shutil.copy2(model_path, backup_path)
                    print(f"  Backed up {model_path.name} -> {backup_path.name}")

        cmd = [
            PYTHON, str(SCRIPTS_DIR / "train_bilstm.py"),
            "--exercise", ex,
            "--use-landmarks",
            "--epochs", str(epochs),
            "--batch-size", "128",
            "--use-bce",          # BCE is proven better than FocalLoss
            "--no-swa",           # SWA hurt performance in prior experiments
            "--no-mixup",         # Mixup hurt performance in prior experiments
            "--patience", "50",
        ]

        # Add hybrid flag for 109-dim exercises
        if config["mode"] == "hybrid":
            cmd.append("--hybrid")

        _run(cmd, f"Training {ex} ({config['dim']}-dim, {epochs} epochs)", dry_run)

    # Compare results
    if not dry_run:
        _print_comparison(targets, old_f1)


def _print_comparison(exercises: List[str], old_f1: Dict[str, float]):
    """Print before/after F1 comparison."""
    _banner("TRAINING RESULTS")
    print(f"{'Exercise':<18} {'Old F1':>8} {'New F1':>8} {'Change':>8}")
    print("-" * 44)

    new_f1s = []
    for ex in sorted(exercises):
        new_f1 = _get_current_f1(ex)
        old = old_f1.get(ex)
        old_str = f"{old:.3f}" if old else "  N/A"
        new_str = f"{new_f1:.3f}" if new_f1 else "  N/A"

        if old and new_f1:
            diff = new_f1 - old
            sign = "+" if diff >= 0 else ""
            diff_str = f"{sign}{diff:.3f}"
            if diff < -0.02:
                diff_str += " !!REGRESSED"
        else:
            diff_str = ""

        print(f"{ex:<18} {old_str:>8} {new_str:>8} {diff_str:>8}")
        if new_f1:
            new_f1s.append(new_f1)

    if new_f1s:
        avg = np.mean(new_f1s)
        print("-" * 44)
        print(f"{'AVERAGE':<18} {'':>8} {avg:>8.3f}")

    # Rollback advice
    for ex in exercises:
        new_f1 = _get_current_f1(ex)
        old = old_f1.get(ex)
        if old and new_f1 and new_f1 < old - 0.02:
            backup_path = MODELS_DIR / f"{ex}_bilstm_v2_pre_pipeline.pt"
            if backup_path.exists():
                print(f"\n  TIP: {ex} regressed. Rollback with:")
                print(f"    cp models/trained/{ex}_bilstm_v2_pre_pipeline.pt models/trained/{ex}_bilstm_v2.pt")


# ─── Stage 5: Validation Report ─────────────────────────────────────────────

def stage_report(exercises: Optional[List[str]] = None):
    """Generate a summary report of all models."""
    _banner("PIPELINE COMPLETE — MODEL SUMMARY")
    targets = exercises or ALL_EXERCISES

    print(f"{'Exercise':<18} {'Dim':>5} {'Mode':<10} {'F1':>6} {'Model Exists':>13}")
    print("-" * 56)

    f1s = []
    for ex in sorted(targets):
        config = EXERCISE_FEATURE_CONFIG[ex]
        model_path = MODELS_DIR / f"{ex}_bilstm_v2.pt"
        exists = "YES" if model_path.exists() else "NO"
        f1 = _get_current_f1(ex)
        f1_str = f"{f1:.3f}" if f1 else "  N/A"
        if f1:
            f1s.append(f1)
        print(f"{ex:<18} {config['dim']:>5} {config['mode']:<10} {f1_str:>6} {exists:>13}")

    if f1s:
        print("-" * 56)
        print(f"{'AVERAGE':<18} {'':>5} {'':>10} {np.mean(f1s):>6.3f}")

    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ExerVision Unified Data Pipeline — Add Data & Retrain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/pipeline.py                          # Full pipeline, all exercises
  python scripts/pipeline.py --exercise squat lunge   # Only squat and lunge
  python scripts/pipeline.py --skip-extract           # Landmarks exist, just rebuild+train
  python scripts/pipeline.py --skip-train             # Rebuild data only, no training
  python scripts/pipeline.py --dry-run                # Show plan, don't execute
  python scripts/pipeline.py --videos v1.mp4 v2.mp4 --exercise squat --label correct
        """,
    )
    parser.add_argument("--exercise", nargs="+", default=None,
                        help="Exercise(s) to process (default: all)")
    parser.add_argument("--videos", nargs="+", default=None,
                        help="Specific video file(s) to add")
    parser.add_argument("--label", type=str, default="correct",
                        choices=["correct", "incorrect"],
                        help="Label for --videos (default: correct)")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip landmark extraction (use existing)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (just rebuild data)")
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild everything from scratch")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done, don't execute")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Training epochs (default: 500)")
    parser.add_argument("--no-backup", action="store_true",
                        help="Don't backup models before overwriting")
    args = parser.parse_args()

    start_time = time.time()
    exercises = args.exercise

    _banner("EXERVISION DATA PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CPU cores: {get_cpu_count()}")
    print(f"Target exercises: {exercises or 'ALL'}")
    print(f"Epochs: {args.epochs}")

    # Auto-detect new videos if not manually specified
    if not args.skip_extract and not args.videos:
        new_videos = detect_new_videos()
        if new_videos:
            total = sum(len(v) for v in new_videos.values())
            print(f"\nDetected {total} new unprocessed videos:")
            for ex, vids in sorted(new_videos.items()):
                correct = sum(1 for _, l, _ in vids if l == "correct")
                incorrect = len(vids) - correct
                print(f"  {ex}: {correct} correct, {incorrect} incorrect")

            # If no exercises specified, auto-target affected ones
            if not exercises:
                affected = detect_affected_exercises(new_videos)
                if affected:
                    print(f"\nWill process affected exercises: {', '.join(affected)}")
        else:
            print("\nNo new unprocessed videos detected.")
            if not args.force:
                print("Use --force to rebuild everything, or add new videos first.")

    # ─── Stage 1: Extract ───
    affected_exercises = set()
    if not args.skip_extract:
        if args.videos:
            ex_hint = exercises[0] if exercises else None
            affected_exercises = stage_extract_landmarks(
                exercises=exercises,
                video_paths=args.videos,
                exercise_hint=ex_hint,
                label_hint=args.label,
                dry_run=args.dry_run,
            )
        else:
            affected_exercises = stage_extract_landmarks(
                exercises=exercises,
                dry_run=args.dry_run,
                force=args.force,
            )
    else:
        print("\n[Skipping Stage 1: Landmark Extraction]")
        affected_exercises = set(exercises) if exercises else set(ALL_EXERCISES)

    # Use affected exercises if none specified
    if not exercises:
        exercises = sorted(affected_exercises) if affected_exercises else ALL_EXERCISES

    # ─── Stage 2: Split ───
    stage_split_videos(exercises, force=True, dry_run=args.dry_run)

    # ─── Stage 3: Build Sequences ───
    stage_build_sequences(exercises, dry_run=args.dry_run)

    # ─── Stage 4: Train ───
    if not args.skip_train:
        stage_train(
            exercises,
            epochs=args.epochs,
            dry_run=args.dry_run,
            backup=not args.no_backup,
        )
    else:
        print("\n[Skipping Stage 4: Training]")

    # ─── Stage 5: Report ───
    stage_report(exercises)

    elapsed = time.time() - start_time
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    print(f"\nTotal pipeline time: {mins}m {secs}s")


if __name__ == "__main__":
    main()
