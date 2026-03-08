"""
Download additional training data from Kaggle and YouTube.

Downloads:
1. Kaggle datasets with exercise videos/images
2. YouTube videos showing CORRECT and INCORRECT form for each exercise

Usage:
    python scripts/download_more_data.py --source all
    python scripts/download_more_data.py --source kaggle
    python scripts/download_more_data.py --source youtube
    python scripts/download_more_data.py --source youtube --exercise squat
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "datasets"

KAGGLE_CLI = "C:/Users/Hamza/AppData/Local/Programs/Python/Python310/Scripts/kaggle.exe"

# Kaggle datasets to download
KAGGLE_DATASETS = [
    {
        "slug": "riccardoriccio/real-time-exercise-recognition-dataset",
        "dest": "kaggle_exercise_recognition",
        "description": "Real-Time Exercise Recognition Dataset (~3GB, videos with labels)",
    },
    {
        "slug": "philosopher0808/gym-workoutexercises-video",
        "dest": "kaggle_gym_videos",
        "description": "Gym Workout/Exercises Video (~10GB, extended video dataset)",
    },
    {
        "slug": "mrigaankjaswal/exercise-detection-dataset",
        "dest": "kaggle_exercise_detection",
        "description": "Exercise Detection Dataset (~1.5MB, labeled exercise data)",
    },
    {
        "slug": "thashmiladewmini/squat-exercise-pose-dataset",
        "dest": "kaggle_squat_pose",
        "description": "Squat Exercise Pose Dataset (~2.3MB, squat-specific)",
    },
    {
        "slug": "muhannadtuameh/exercise-recognition",
        "dest": "kaggle_exercise_recognition_pose",
        "description": "Physical Exercise Recognition Dataset (~1.1MB, pose landmarks)",
    },
]

# YouTube search queries for INCORRECT form (the biggest gap in our data)
YOUTUBE_QUERIES = {
    "squat": {
        "correct": [
            "perfect squat form tutorial",
            "how to squat correctly step by step",
            "squat technique proper form",
        ],
        "incorrect": [
            "common squat mistakes demonstration",
            "squat form mistakes to avoid",
            "bad squat form examples",
            "correct vs incorrect squat form",
            "squat knee cave valgus mistake",
            "butt wink squat problem",
        ],
    },
    "lunge": {
        "correct": [
            "perfect lunge form tutorial",
            "how to lunge correctly",
        ],
        "incorrect": [
            "common lunge mistakes",
            "lunge form errors to avoid",
            "correct vs incorrect lunge form",
        ],
    },
    "deadlift": {
        "correct": [
            "perfect deadlift form tutorial",
            "how to deadlift correctly",
        ],
        "incorrect": [
            "deadlift form mistakes to avoid",
            "common deadlift mistakes",
            "deadlift round back mistake",
            "correct vs incorrect deadlift form",
        ],
    },
    "bench_press": {
        "correct": [
            "perfect bench press form tutorial",
            "how to bench press correctly",
        ],
        "incorrect": [
            "bench press form mistakes",
            "common bench press errors",
            "bench press elbow flare mistake",
            "correct vs incorrect bench press form",
        ],
    },
    "overhead_press": {
        "correct": [
            "perfect overhead press form tutorial",
            "how to overhead press correctly",
        ],
        "incorrect": [
            "overhead press form mistakes",
            "shoulder press common mistakes",
            "overhead press lower back arch mistake",
            "correct vs incorrect overhead press",
        ],
    },
    "pullup": {
        "correct": [
            "perfect pull up form tutorial",
            "how to do pull ups correctly",
        ],
        "incorrect": [
            "pull up form mistakes",
            "common pull up mistakes",
            "pull up kipping vs strict bad form",
            "correct vs incorrect pull up form",
        ],
    },
    "pushup": {
        "correct": [
            "perfect push up form tutorial",
            "how to do push ups correctly",
        ],
        "incorrect": [
            "push up form mistakes",
            "common push up mistakes",
            "push up hip sag mistake",
            "correct vs incorrect push up form",
        ],
    },
    "plank": {
        "correct": [
            "perfect plank form tutorial",
            "how to plank correctly",
        ],
        "incorrect": [
            "plank form mistakes",
            "common plank mistakes",
            "plank hip sag mistake",
            "correct vs incorrect plank form",
        ],
    },
    "bicep_curl": {
        "correct": [
            "perfect bicep curl form tutorial",
            "how to bicep curl correctly",
        ],
        "incorrect": [
            "bicep curl form mistakes",
            "common bicep curl mistakes",
            "bicep curl swinging mistake",
            "correct vs incorrect bicep curl form",
        ],
    },
    "tricep_dip": {
        "correct": [
            "perfect tricep dip form tutorial",
            "how to do dips correctly",
        ],
        "incorrect": [
            "tricep dip form mistakes",
            "common dip mistakes",
            "dip shoulder injury form mistake",
            "correct vs incorrect dip form",
        ],
    },
}


def download_kaggle_datasets():
    """Download all Kaggle datasets."""
    print("=" * 60)
    print("DOWNLOADING KAGGLE DATASETS")
    print("=" * 60)

    for ds in KAGGLE_DATASETS:
        dest = DATA_DIR / ds["dest"]
        dest.mkdir(parents=True, exist_ok=True)

        print(f"\n--- {ds['description']} ---")
        print(f"  Slug: {ds['slug']}")
        print(f"  Dest: {dest}")

        if any(dest.iterdir()):
            print(f"  [SKIP] Already downloaded (directory not empty)")
            continue

        try:
            subprocess.run(
                [KAGGLE_CLI, "datasets", "download", "-d", ds["slug"],
                 "-p", str(dest), "--unzip"],
                check=True,
                timeout=3600,  # 1 hour timeout per dataset
            )
            print(f"  [OK] Downloaded successfully")
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Download failed: {e}")
        except subprocess.TimeoutExpired:
            print(f"  [ERROR] Download timed out (1 hour)")


def download_youtube_videos(exercise_filter: str = None, max_per_query: int = 5):
    """Download YouTube videos for correct and incorrect form."""
    print("\n" + "=" * 60)
    print("DOWNLOADING YOUTUBE VIDEOS")
    print("=" * 60)

    yt_dir = DATA_DIR / "youtube"

    exercises = [exercise_filter] if exercise_filter else list(YOUTUBE_QUERIES.keys())

    for exercise in exercises:
        queries = YOUTUBE_QUERIES.get(exercise, {})

        for label in ["correct", "incorrect"]:
            label_queries = queries.get(label, [])
            out_dir = yt_dir / exercise / label
            out_dir.mkdir(parents=True, exist_ok=True)

            for query in label_queries:
                print(f"\n  Searching: '{query}'")
                print(f"  Dest: {out_dir}")

                try:
                    subprocess.run(
                        [
                            sys.executable, "-m", "yt_dlp",
                            f"ytsearch{max_per_query}:{query}",
                            "-o", str(out_dir / "%(title).60s_%(id)s.%(ext)s"),
                            "-f", "best[height<=720]",
                            "--max-downloads", str(max_per_query),
                            "--no-playlist",
                            "--max-filesize", "100M",
                            "--no-overwrites",
                            "--quiet",
                            "--no-warnings",
                        ],
                        timeout=300,  # 5 min per query
                    )
                    print(f"  [OK] Downloaded")
                except subprocess.TimeoutExpired:
                    print(f"  [WARN] Timed out, moving on")
                except Exception as e:
                    print(f"  [ERROR] {e}")

    # Count results
    print(f"\n{'='*60}")
    print("YOUTUBE DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    for exercise in exercises:
        for label in ["correct", "incorrect"]:
            out_dir = yt_dir / exercise / label
            if out_dir.exists():
                n = len(list(out_dir.glob("*.mp4")) + list(out_dir.glob("*.webm")))
                print(f"  {exercise}/{label}: {n} videos")


def main():
    parser = argparse.ArgumentParser(description="Download additional training data")
    parser.add_argument("--source", choices=["all", "kaggle", "youtube"],
                        default="all", help="Data source to download")
    parser.add_argument("--exercise", type=str, default=None,
                        help="Download YouTube videos for a specific exercise only")
    parser.add_argument("--max-per-query", type=int, default=5,
                        help="Max YouTube videos per search query (default: 5)")
    args = parser.parse_args()

    if args.source in ("all", "kaggle"):
        download_kaggle_datasets()

    if args.source in ("all", "youtube"):
        download_youtube_videos(
            exercise_filter=args.exercise,
            max_per_query=args.max_per_query,
        )

    print("\nDone! Run extract_landmarks.py next to process new data.")


if __name__ == "__main__":
    main()
