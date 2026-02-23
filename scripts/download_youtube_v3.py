"""
YouTube Exercise Video Downloader v3 - Working search-based approach.

Usage:
    python scripts/download_youtube_v3.py --all
    python scripts/download_youtube_v3.py --exercise squat --label incorrect
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
YOUTUBE_DIR = PROJECT_ROOT / "data" / "datasets" / "youtube"
PYTHON = "C:/Users/Hamza/AppData/Local/Programs/Python/Python310/python.exe"

SEARCH_QUERIES = {
    "squat": {
        "correct": [
            "how to squat properly form tutorial",
            "correct squat form demonstration gym",
            "barbell squat perfect technique",
        ],
        "incorrect": [
            "common squat mistakes wrong form gym",
            "squat form errors what not to do",
            "worst squat mistakes beginners",
        ],
    },
    "deadlift": {
        "correct": [
            "how to deadlift properly form tutorial",
            "correct deadlift technique demonstration",
            "conventional deadlift form guide",
        ],
        "incorrect": [
            "common deadlift mistakes wrong form",
            "deadlift form errors round back",
            "worst deadlift mistakes beginners",
        ],
    },
    "bench_press": {
        "correct": [
            "how to bench press properly form tutorial",
            "correct bench press technique guide",
            "bench press form demonstration gym",
        ],
        "incorrect": [
            "common bench press mistakes wrong form",
            "bench press form errors what not to do",
            "worst bench press mistakes beginners",
        ],
    },
    "overhead_press": {
        "correct": [
            "how to overhead press properly form",
            "correct shoulder press technique guide",
            "standing barbell overhead press tutorial",
        ],
        "incorrect": [
            "overhead press mistakes wrong form",
            "shoulder press common errors technique",
            "worst overhead press mistakes",
        ],
    },
    "lunge": {
        "correct": [
            "how to lunge properly form tutorial",
            "correct lunge technique demonstration",
            "walking lunge perfect form guide",
        ],
        "incorrect": [
            "common lunge mistakes wrong form",
            "lunge form errors what not to do",
            "worst lunge mistakes beginners",
        ],
    },
    "pushup": {
        "correct": [
            "how to do push ups properly form",
            "correct pushup technique tutorial",
            "perfect push up form demonstration",
        ],
        "incorrect": [
            "common push up mistakes wrong form",
            "pushup form errors what not to do",
            "worst push up mistakes beginners",
        ],
    },
    "pullup": {
        "correct": [
            "how to do pull ups properly form",
            "correct pull up technique tutorial",
            "perfect pull up form demonstration",
        ],
        "incorrect": [
            "common pull up mistakes wrong form",
            "pullup form errors what not to do",
            "worst pull up mistakes beginners",
        ],
    },
    "plank": {
        "correct": [
            "how to plank properly form tutorial",
            "correct plank technique demonstration",
            "perfect plank form guide",
        ],
        "incorrect": [
            "common plank mistakes wrong form",
            "plank form errors what not to do",
            "worst plank mistakes beginners",
        ],
    },
    "bicep_curl": {
        "correct": [
            "how to bicep curl properly form tutorial",
            "correct dumbbell curl technique",
            "perfect bicep curl form demonstration",
        ],
        "incorrect": [
            "common bicep curl mistakes swinging",
            "bicep curl form errors wrong technique",
            "worst bicep curl mistakes beginners",
        ],
    },
    "tricep_dip": {
        "correct": [
            "how to do dips properly form tutorial",
            "correct tricep dip technique guide",
            "parallel bar dip form demonstration",
        ],
        "incorrect": [
            "common dip mistakes wrong form shoulder",
            "tricep dip form errors what not to do",
            "worst dip mistakes beginners",
        ],
    },
}


def search_download(query: str, output_dir: Path, max_results: int = 2) -> int:
    """Search YouTube and download. Returns count of new downloads."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count existing files before
    video_exts = {".mp4", ".webm", ".mkv"}
    before = set(f.name for f in output_dir.iterdir() if f.suffix.lower() in video_exts) if output_dir.exists() else set()

    output_template = str(output_dir / "%(id)s.%(ext)s")

    cmd = [
        PYTHON, "-m", "yt_dlp",
        f"ytsearch{max_results}:{query}",
        "--format", "best[height<=720]",
        "--output", output_template,
        "--no-playlist",
        "--match-filter", "duration<600",
        "--no-overwrites",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT]")
        return 0
    except Exception as e:
        print(f"    [ERROR] {e}")
        return 0

    # Count new files
    after = set(f.name for f in output_dir.iterdir() if f.suffix.lower() in video_exts) if output_dir.exists() else set()
    new_files = after - before
    return len(new_files)


def download_exercise(exercise: str, label: str = None, per_query: int = 2):
    labels = [label] if label else ["correct", "incorrect"]

    for lbl in labels:
        queries = SEARCH_QUERIES.get(exercise, {}).get(lbl, [])
        output_dir = YOUTUBE_DIR / exercise / lbl

        video_exts = {".mp4", ".webm", ".mkv"}
        existing = len([f for f in output_dir.iterdir() if f.suffix.lower() in video_exts]) if output_dir.exists() else 0

        print(f"\n--- {exercise} / {lbl} ({existing} existing) ---")

        total_new = 0
        for query in queries:
            print(f"  Search: \"{query}\"", end="", flush=True)
            count = search_download(query, output_dir, per_query)
            print(f" -> {count} downloaded")
            total_new += count

        final = len([f for f in output_dir.iterdir() if f.suffix.lower() in video_exts]) if output_dir.exists() else 0
        print(f"  Total for {exercise}/{lbl}: {final} videos")


def download_all(per_query: int = 2):
    print("="*60)
    print("Downloading exercise videos from YouTube (search-based)")
    print(f"Videos per query: {per_query}")
    print("="*60)

    for exercise in SEARCH_QUERIES:
        download_exercise(exercise, per_query=per_query)

    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    video_exts = {".mp4", ".webm", ".mkv"}
    total = 0
    for exercise in SEARCH_QUERIES:
        for label in ["correct", "incorrect"]:
            d = YOUTUBE_DIR / exercise / label
            count = len([f for f in d.iterdir() if f.suffix.lower() in video_exts]) if d.exists() else 0
            total += count
            print(f"  {exercise:20s} {label:12s}: {count} videos")
    print(f"\n  TOTAL: {total} videos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exercise", type=str)
    parser.add_argument("--label", type=str, choices=["correct", "incorrect"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--count", type=int, default=2, help="Videos per search query")
    args = parser.parse_args()

    if args.all:
        download_all(args.count)
    elif args.exercise:
        download_exercise(args.exercise, args.label, args.count)
    else:
        parser.print_help()
