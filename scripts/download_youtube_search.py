"""
YouTube Exercise Video Downloader v2 - Search-based.

Instead of hardcoded URLs, searches YouTube for exercise form videos
and downloads the top results. Much more reliable than hardcoded URLs.

Usage:
    python scripts/download_youtube_search.py --exercise squat --label correct --count 5
    python scripts/download_youtube_search.py --all --count 3
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
YOUTUBE_DIR = PROJECT_ROOT / "data" / "datasets" / "youtube"

# Search queries for each exercise - correct and incorrect form
SEARCH_QUERIES = {
    "squat": {
        "correct": [
            "how to squat properly form tutorial",
            "correct squat form demonstration",
            "barbell squat technique guide",
        ],
        "incorrect": [
            "common squat mistakes wrong form",
            "squat form errors bad technique",
            "squat mistakes to avoid",
        ],
    },
    "deadlift": {
        "correct": [
            "how to deadlift properly form tutorial",
            "correct deadlift technique guide",
            "conventional deadlift form demonstration",
        ],
        "incorrect": [
            "common deadlift mistakes wrong form",
            "deadlift form errors bad technique",
            "deadlift mistakes to avoid round back",
        ],
    },
    "bench_press": {
        "correct": [
            "how to bench press properly form",
            "correct bench press technique tutorial",
            "bench press form guide demonstration",
        ],
        "incorrect": [
            "common bench press mistakes wrong form",
            "bench press errors bad technique",
            "bench press mistakes to avoid",
        ],
    },
    "overhead_press": {
        "correct": [
            "how to overhead press properly form",
            "correct shoulder press technique",
            "standing overhead press tutorial",
        ],
        "incorrect": [
            "overhead press mistakes wrong form",
            "shoulder press common errors",
            "overhead press mistakes to avoid",
        ],
    },
    "lunge": {
        "correct": [
            "how to lunge properly form tutorial",
            "correct lunge technique demonstration",
            "walking lunge form guide",
        ],
        "incorrect": [
            "common lunge mistakes wrong form",
            "lunge form errors bad technique",
            "lunge mistakes to avoid",
        ],
    },
    "pushup": {
        "correct": [
            "how to do push ups properly form",
            "correct pushup technique tutorial",
            "perfect push up form guide",
        ],
        "incorrect": [
            "common push up mistakes wrong form",
            "pushup form errors bad technique",
            "push up mistakes to avoid",
        ],
    },
    "pullup": {
        "correct": [
            "how to do pull ups properly form",
            "correct pull up technique tutorial",
            "pull up form guide demonstration",
        ],
        "incorrect": [
            "common pull up mistakes wrong form",
            "pullup form errors bad technique",
            "pull up mistakes to avoid",
        ],
    },
    "plank": {
        "correct": [
            "how to plank properly form tutorial",
            "correct plank technique guide",
            "perfect plank form demonstration",
        ],
        "incorrect": [
            "common plank mistakes wrong form",
            "plank form errors bad technique",
            "plank mistakes to avoid",
        ],
    },
    "bicep_curl": {
        "correct": [
            "how to bicep curl properly form",
            "correct bicep curl technique tutorial",
            "dumbbell curl form guide",
        ],
        "incorrect": [
            "common bicep curl mistakes wrong form",
            "bicep curl form errors swinging",
            "bicep curl mistakes to avoid",
        ],
    },
    "tricep_dip": {
        "correct": [
            "how to do dips properly form tutorial",
            "correct tricep dip technique",
            "parallel bar dip form guide",
        ],
        "incorrect": [
            "common dip mistakes wrong form",
            "tricep dip form errors bad technique",
            "dip mistakes to avoid shoulder",
        ],
    },
}


def search_and_download(query: str, output_dir: Path, max_results: int = 2,
                        max_duration: int = 600) -> list:
    """
    Search YouTube and download top results.
    Returns list of downloaded file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(id)s.%(ext)s")

    cmd = [
        sys.executable, "-m", "yt_dlp",
        f"ytsearch{max_results}:{query}",
        "--format", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
        "--output", output_template,
        "--no-playlist",
        "--match-filter", f"duration<{max_duration}",
        "--quiet",
        "--no-warnings",
        "--print", "after_move:filepath",
    ]

    downloaded = []
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                if line.strip() and Path(line.strip()).exists():
                    downloaded.append(line.strip())
        if result.returncode != 0 and not downloaded:
            # Try with simpler format
            cmd_simple = [
                sys.executable, "-m", "yt_dlp",
                f"ytsearch{max_results}:{query}",
                "--format", "best[height<=720]",
                "--output", output_template,
                "--no-playlist",
                "--match-filter", f"duration<{max_duration}",
                "--quiet",
                "--no-warnings",
                "--print", "after_move:filepath",
            ]
            result = subprocess.run(cmd_simple, capture_output=True, text=True, timeout=180)
            if result.stdout.strip():
                for line in result.stdout.strip().split("\n"):
                    if line.strip() and Path(line.strip()).exists():
                        downloaded.append(line.strip())
    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT] Search timed out for: {query}")
    except Exception as e:
        print(f"    [ERROR] {e}")

    return downloaded


def download_exercise(exercise: str, label: str = None, videos_per_query: int = 2):
    """Download videos for an exercise using search queries."""
    if exercise not in SEARCH_QUERIES:
        print(f"Unknown exercise: {exercise}")
        return

    labels = [label] if label else ["correct", "incorrect"]

    for lbl in labels:
        queries = SEARCH_QUERIES[exercise].get(lbl, [])
        output_dir = YOUTUBE_DIR / exercise / lbl

        # Check how many videos already exist
        existing = len(list(output_dir.glob("*.*"))) if output_dir.exists() else 0

        print(f"\n{'='*60}")
        print(f"{exercise} / {lbl} ({existing} existing, searching {len(queries)} queries)")
        print(f"{'='*60}")

        total_downloaded = 0
        for query in queries:
            print(f"  Searching: \"{query}\"")
            files = search_and_download(query, output_dir, max_results=videos_per_query)
            for f in files:
                print(f"    -> Downloaded: {Path(f).name}")
                total_downloaded += 1

        print(f"  Downloaded {total_downloaded} new videos for {exercise}/{lbl}")


def download_all(videos_per_query: int = 2):
    """Download videos for all exercises."""
    total_queries = sum(
        len(queries)
        for exercise in SEARCH_QUERIES.values()
        for queries in exercise.values()
    )
    print(f"Total search queries: {total_queries}")
    print(f"Videos per query: {videos_per_query}")
    print(f"Max potential downloads: {total_queries * videos_per_query}")
    print(f"Output: {YOUTUBE_DIR}")

    for exercise in SEARCH_QUERIES:
        download_exercise(exercise, videos_per_query=videos_per_query)

    # Print summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    for exercise in SEARCH_QUERIES:
        for label in ["correct", "incorrect"]:
            d = YOUTUBE_DIR / exercise / label
            if d.exists():
                video_exts = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
                count = len([f for f in d.iterdir() if f.suffix.lower() in video_exts])
                print(f"  {exercise}/{label}: {count} videos")
            else:
                print(f"  {exercise}/{label}: 0 videos")

    # Save metadata
    metadata = {}
    for exercise in SEARCH_QUERIES:
        metadata[exercise] = {}
        for label in ["correct", "incorrect"]:
            d = YOUTUBE_DIR / exercise / label
            if d.exists():
                video_exts = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
                files = [f.name for f in d.iterdir() if f.suffix.lower() in video_exts]
                metadata[exercise][label] = files
            else:
                metadata[exercise][label] = []

    meta_path = YOUTUBE_DIR / "download_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search and download YouTube exercise videos")
    parser.add_argument("--exercise", type=str, help="Exercise name")
    parser.add_argument("--label", type=str, choices=["correct", "incorrect"])
    parser.add_argument("--all", action="store_true", help="Download all exercises")
    parser.add_argument("--count", type=int, default=2,
                        help="Videos to download per search query (default: 2)")

    args = parser.parse_args()

    if args.all:
        download_all(videos_per_query=args.count)
    elif args.exercise:
        download_exercise(args.exercise, args.label, args.count)
    else:
        parser.print_help()
