"""
Download additional training videos from YouTube for weak exercises.

Targets SHORT clips (< 90 seconds) showing clear correct/incorrect form.
Avoids long tutorials which cause label noise.

Usage:
    python scripts/download_training_data.py --exercise squat
    python scripts/download_training_data.py --all
    python scripts/download_training_data.py --all --dry-run
"""

import os
import sys
import json
import argparse
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "datasets"

# Exercises that need more data (test F1 < 0.65)
WEAK_EXERCISES = [
    "squat",        # F1=0.527, 91% noisy incorrect data
    "deadlift",     # F1=0.515, 93% noisy incorrect data
    "bench_press",  # F1=0.514
    "pushup",       # F1=0.750 but DEGENERATE (0 TN)
    "overhead_press",  # F1=0.577
    "plank",        # F1=0.580
    "tricep_dip",   # F1=0.595
]

# Search queries designed to find SHORT clips with CLEAR form (not tutorials)
# Each query targets a specific type of content
SEARCH_QUERIES = {
    "squat": {
        "incorrect": [
            "squat form mistakes gym",
            "squat bad form compilation",
            "squat common errors demonstration",
            "squat wrong technique example",
            "squat gym fail form",
            "squat depth mistake",
            "squat knee cave demonstration",
            "squat butt wink example",
            "squat rounded back mistake",
            "bad squat form gym clip",
        ],
        "correct": [
            "perfect squat form demonstration",
            "squat proper technique short",
            "squat correct form side view",
            "squat good form gym",
            "deep squat perfect form",
            "barbell squat correct technique",
        ],
    },
    "deadlift": {
        "incorrect": [
            "deadlift form mistakes gym",
            "deadlift bad form compilation",
            "deadlift rounded back example",
            "deadlift common errors demonstration",
            "deadlift wrong technique",
            "deadlift gym fail form",
            "deadlift hitching mistake",
            "deadlift lockout mistake example",
            "bad deadlift form clip",
            "deadlift lower back rounding",
        ],
        "correct": [
            "perfect deadlift form demonstration",
            "deadlift proper technique short",
            "deadlift correct form side view",
            "conventional deadlift good form",
            "deadlift perfect rep",
            "deadlift correct technique gym",
        ],
    },
    "bench_press": {
        "incorrect": [
            "bench press form mistakes",
            "bench press bad form compilation",
            "bench press wrong technique gym",
            "bench press common errors demonstration",
            "bench press elbow flare mistake",
            "bench press no arch mistake",
            "bench press bouncing mistake",
            "bad bench press form clip",
            "bench press gym fail form",
            "bench press shoulder injury form",
        ],
        "correct": [
            "perfect bench press form",
            "bench press proper technique short",
            "bench press correct form side view",
            "bench press good form gym",
            "bench press correct technique demonstration",
            "competition bench press form",
        ],
    },
    "pushup": {
        "incorrect": [
            "pushup form mistakes",
            "pushup bad form compilation",
            "pushup wrong technique example",
            "pushup common errors demonstration",
            "pushup sagging hips mistake",
            "pushup elbow flare mistake",
            "pushup half rep mistake",
            "bad pushup form clip",
            "pushup gym fail form",
            "wrong pushup technique demonstration",
        ],
        "correct": [
            "perfect pushup form demonstration",
            "pushup proper technique short",
            "pushup correct form side view",
            "pushup good form example",
            "military pushup correct technique",
            "strict pushup form",
        ],
    },
    "overhead_press": {
        "incorrect": [
            "overhead press form mistakes",
            "overhead press bad form",
            "shoulder press wrong technique gym",
            "overhead press common errors demonstration",
            "overhead press lean back mistake",
            "military press bad form example",
            "overhead press gym fail",
            "shoulder press mistake demonstration",
            "bad overhead press form clip",
            "OHP form mistakes",
        ],
        "correct": [
            "perfect overhead press form",
            "overhead press proper technique short",
            "military press correct form",
            "shoulder press good form gym",
            "overhead press correct technique",
            "strict press form demonstration",
        ],
    },
    "plank": {
        "incorrect": [
            "plank form mistakes",
            "plank bad form compilation",
            "plank wrong technique example",
            "plank common errors demonstration",
            "plank sagging hips mistake",
            "plank butt too high mistake",
            "plank incorrect form example",
            "bad plank form clip",
            "plank form errors gym",
            "wrong plank technique demonstration",
        ],
        "correct": [
            "perfect plank form demonstration",
            "plank proper technique short",
            "plank correct form side view",
            "plank good form example",
            "plank correct technique 30 seconds",
            "strict plank form",
        ],
    },
    "tricep_dip": {
        "incorrect": [
            "tricep dip form mistakes",
            "dip bad form compilation",
            "tricep dip wrong technique",
            "dip common errors demonstration",
            "tricep dip shoulder impingement",
            "dip too deep mistake",
            "tricep dip elbow flare mistake",
            "bad dip form clip",
            "dip gym fail form",
            "wrong dip technique demonstration",
        ],
        "correct": [
            "perfect tricep dip form",
            "tricep dip proper technique short",
            "dip correct form side view",
            "tricep dip good form gym",
            "parallel bar dip correct technique",
            "bench dip proper form",
        ],
    },
}

# yt-dlp max duration filter (seconds) — avoid long tutorials
MAX_DURATION = 120
# Videos per search query
VIDEOS_PER_QUERY = 3


def download_videos(exercise, label, queries, output_dir, dry_run=False, max_per_query=VIDEOS_PER_QUERY):
    """Download videos for a specific exercise and label."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    failed = []
    skipped = []

    for i, query in enumerate(queries):
        print(f"\n  [{i+1}/{len(queries)}] Searching: '{query}'")

        # Use yt-dlp to search YouTube and download
        # --match-filter to only get short videos
        # --max-downloads to limit per query
        output_template = str(output_dir / f"%(id)s.%(ext)s")

        cmd = [
            "yt-dlp",
            f"ytsearch{max_per_query}:{query}",
            "--match-filter", f"duration<={MAX_DURATION}",
            "--max-downloads", str(max_per_query),
            "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
            "--merge-output-format", "mp4",
            "-o", output_template,
            "--no-playlist",
            "--no-overwrites",
            "--quiet",
            "--progress",
            "--print", "after_filter:%(id)s|%(title)s|%(duration)s",
            "--no-write-info-json",
            "--no-write-thumbnail",
            "--no-write-description",
            "--no-write-comments",
            "--socket-timeout", "30",
            "--retries", "3",
        ]

        if dry_run:
            cmd.append("--simulate")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 min timeout per query
                cwd=str(PROJECT_ROOT),
            )

            # Parse output for downloaded video info
            for line in result.stdout.strip().split("\n"):
                if "|" in line:
                    parts = line.split("|", 2)
                    if len(parts) >= 3:
                        vid_id, title, duration = parts[0], parts[1], parts[2]
                        downloaded.append({
                            "id": vid_id,
                            "title": title,
                            "duration": duration,
                            "query": query,
                            "exercise": exercise,
                            "label": label,
                        })
                        safe_title = title[:50].encode("ascii", errors="replace").decode("ascii")
                        print(f"    ✓ {vid_id} ({duration}s) - {safe_title}")

            if result.returncode != 0 and result.stderr:
                # Filter out common non-error messages
                errors = [l for l in result.stderr.split("\n")
                         if l.strip() and "WARNING" not in l and "has already been downloaded" not in l]
                if errors:
                    for e in errors[:3]:
                        safe_err = e[:100].encode("ascii", errors="replace").decode("ascii")
                        print(f"    ! {safe_err}")

        except subprocess.TimeoutExpired:
            print(f"    ! Timeout on query: {query}")
            failed.append(query)
        except Exception as e:
            print(f"    ! Error: {e}")
            failed.append(query)

    return downloaded, failed


def count_existing_videos(directory):
    """Count existing video files in a directory."""
    if not Path(directory).exists():
        return 0
    extensions = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
    return sum(1 for f in Path(directory).iterdir() if f.suffix.lower() in extensions)


def main():
    parser = argparse.ArgumentParser(description="Download training videos for weak exercises")
    parser.add_argument("--exercise", type=str, choices=WEAK_EXERCISES,
                        help="Download for a specific exercise")
    parser.add_argument("--all", action="store_true", help="Download for all weak exercises")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without downloading")
    parser.add_argument("--label", type=str, choices=["correct", "incorrect", "both"],
                        default="both", help="Which label to download (default: both)")
    parser.add_argument("--max-per-query", type=int, default=VIDEOS_PER_QUERY,
                        help=f"Max videos per search query (default: {VIDEOS_PER_QUERY})")

    args = parser.parse_args()

    if not args.exercise and not args.all:
        parser.print_help()
        sys.exit(1)

    exercises = WEAK_EXERCISES if args.all else [args.exercise]

    all_downloaded = []
    all_failed = []

    for exercise in exercises:
        print(f"\n{'='*60}")
        print(f"  EXERCISE: {exercise}")
        print(f"{'='*60}")

        queries = SEARCH_QUERIES.get(exercise, {})

        for label in ["incorrect", "correct"]:
            if args.label != "both" and args.label != label:
                continue

            label_queries = queries.get(label, [])
            if not label_queries:
                continue

            # Determine output directory
            if label == "incorrect":
                out_dir = DATA_DIR / "youtube_incorrect" / exercise
            else:
                out_dir = DATA_DIR / "youtube" / exercise / "correct"

            existing = count_existing_videos(out_dir)
            print(f"\n  [{label.upper()}] {existing} existing videos in {out_dir.name}/")
            print(f"  Running {len(label_queries)} search queries ({args.max_per_query} videos each)...")

            downloaded, failed = download_videos(
                exercise, label, label_queries, out_dir,
                dry_run=args.dry_run, max_per_query=args.max_per_query
            )

            new_count = count_existing_videos(out_dir) - existing
            print(f"\n  [{label.upper()}] Downloaded {new_count} new videos (total now: {existing + new_count})")

            all_downloaded.extend(downloaded)
            all_failed.extend(failed)

    # Summary
    print(f"\n{'='*60}")
    print(f"  DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"  Total new videos: {len(all_downloaded)}")
    print(f"  Failed queries: {len(all_failed)}")

    # Save download log
    log_path = DATA_DIR / "download_log_new.json"
    with open(log_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "downloaded": all_downloaded,
            "failed": all_failed,
        }, f, indent=2)
    print(f"  Log saved: {log_path}")


if __name__ == "__main__":
    main()
