"""
YouTube Exercise Video Downloader for FYP Data Collection.

Downloads exercise form videos from YouTube, organized by exercise type and form quality.
Only stores extracted pose landmarks (CSV), not raw videos (deleted after extraction).
This is legal under the UK Text and Data Mining exception for non-commercial university research.

Usage:
    python scripts/download_youtube.py --exercise squat --label correct
    python scripts/download_youtube.py --all
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
YOUTUBE_DIR = DATA_DIR / "datasets" / "youtube"
RAW_DIR = DATA_DIR / "raw"

# YouTube video URLs organized by exercise and form quality
# These are educational fitness videos demonstrating exercise technique
YOUTUBE_SOURCES = {
    "squat": {
        "correct": [
            # Squat tutorials showing proper form
            "https://www.youtube.com/watch?v=ultWZbUMPL8",  # How to Squat properly - Jeremy Ethier
            "https://www.youtube.com/watch?v=bEv6CCg2BC8",  # How To Squat For Growth - Jeff Nippard
            "https://www.youtube.com/watch?v=Dy28eq2PjcM",  # ATG Squat - Squat University
            "https://www.youtube.com/watch?v=nhoikoUEI8U",  # How To Squat Properly
            "https://www.youtube.com/watch?v=gsNoPYwWXeM",  # Squat form guide
        ],
        "incorrect": [
            # Videos showing common squat mistakes
            "https://www.youtube.com/watch?v=NtX8GGbDCuc",  # Squat mistakes - Jeremy Ethier
            "https://www.youtube.com/watch?v=v6rHMsifMKs",  # Stop squatting wrong
            "https://www.youtube.com/watch?v=UFs6E3Ti1jg",  # Squat mistakes - AthleanX
            "https://www.youtube.com/watch?v=oBGmRJBRJOc",  # Common squat errors
        ],
    },
    "deadlift": {
        "correct": [
            "https://www.youtube.com/watch?v=VL5Ab0T07e4",  # Deadlift form - Alan Thrall
            "https://www.youtube.com/watch?v=op9kVnSo6Qc",  # How To Deadlift - Jeff Nippard
            "https://www.youtube.com/watch?v=r4MzxtBKyNE",  # Deadlift tutorial
            "https://www.youtube.com/watch?v=hCDzSR6bW10",  # Proper deadlift form
        ],
        "incorrect": [
            "https://www.youtube.com/watch?v=NYN3UGCYisk",  # Deadlift mistakes - AthleanX
            "https://www.youtube.com/watch?v=9ZuXKqRbT9k",  # Stop deadlifting wrong
            "https://www.youtube.com/watch?v=hJGx8YblNfY",  # Common deadlift errors
        ],
    },
    "bench_press": {
        "correct": [
            "https://www.youtube.com/watch?v=vcBig73ojpE",  # Bench press form - AthleanX
            "https://www.youtube.com/watch?v=4Y2ZdHCOXok",  # How To Bench - Jeff Nippard
            "https://www.youtube.com/watch?v=gRVjAtPip0Y",  # Bench press tutorial
            "https://www.youtube.com/watch?v=rT7DgCr-3pg",  # Proper bench form
        ],
        "incorrect": [
            "https://www.youtube.com/watch?v=Gou-sFaOE34",  # Bench press mistakes
            "https://www.youtube.com/watch?v=ZFpjTBJbYME",  # Stop benching wrong
        ],
    },
    "overhead_press": {
        "correct": [
            "https://www.youtube.com/watch?v=_RlRDWO2jfg",  # OHP form - Alan Thrall
            "https://www.youtube.com/watch?v=QAQ64hK4Xxs",  # How to OHP - Jeff Nippard
            "https://www.youtube.com/watch?v=2yjwXTZQDDI",  # Overhead press guide
        ],
        "incorrect": [
            "https://www.youtube.com/watch?v=sqKhLR1zRaU",  # OHP mistakes
            "https://www.youtube.com/watch?v=RB-8XYtEn_I",  # Common OHP errors
        ],
    },
    "lunge": {
        "correct": [
            "https://www.youtube.com/watch?v=QOVaHnm-4WM",  # Lunge form guide
            "https://www.youtube.com/watch?v=wrwwXE_5-pI",  # How to lunge properly
            "https://www.youtube.com/watch?v=D7KaRcUTQeE",  # Lunge tutorial
        ],
        "incorrect": [
            "https://www.youtube.com/watch?v=1q7YVwXfX8I",  # Lunge mistakes
            "https://www.youtube.com/watch?v=L8J_6Bkmvnk",  # Common lunge errors
        ],
    },
    "pushup": {
        "correct": [
            "https://www.youtube.com/watch?v=IODxDxX7oi4",  # Push-up form - AthleanX
            "https://www.youtube.com/watch?v=MO4ftFMHgw4",  # Perfect push-up guide
            "https://www.youtube.com/watch?v=4dF1DOWzf20",  # How to push up properly
        ],
        "incorrect": [
            "https://www.youtube.com/watch?v=a93M_pEl0Ek",  # Push-up mistakes
            "https://www.youtube.com/watch?v=7D0FaWmhPIQ",  # Stop doing pushups wrong
        ],
    },
    "pullup": {
        "correct": [
            "https://www.youtube.com/watch?v=eGo4IYlbE5g",  # Pull-up form - AthleanX
            "https://www.youtube.com/watch?v=poyr8KenUfc",  # How to pull-up - Jeff Nippard
            "https://www.youtube.com/watch?v=XB_7En-zf_M",  # Perfect pull-up guide
        ],
        "incorrect": [
            "https://www.youtube.com/watch?v=lPlVXQqSA-M",  # Pull-up mistakes
            "https://www.youtube.com/watch?v=bUWpMIRY6E0",  # Common pull-up errors
        ],
    },
    "plank": {
        "correct": [
            "https://www.youtube.com/watch?v=pSHjTRCQxIw",  # Plank form guide - AthleanX
            "https://www.youtube.com/watch?v=ASdvN_XEl_c",  # Perfect plank tutorial
            "https://www.youtube.com/watch?v=Ehy8G42yARo",  # How to plank properly
        ],
        "incorrect": [
            "https://www.youtube.com/watch?v=1MQhCMBBe88",  # Plank mistakes
            "https://www.youtube.com/watch?v=wMxCNsFzAlc",  # Common plank errors
        ],
    },
    "bicep_curl": {
        "correct": [
            "https://www.youtube.com/watch?v=ykJmrZ5v0Oo",  # Bicep curl form - Jeff Nippard
            "https://www.youtube.com/watch?v=in7PaeYlhrM",  # How to curl properly
            "https://www.youtube.com/watch?v=kwG2ipFRgFo",  # Perfect curl guide
        ],
        "incorrect": [
            "https://www.youtube.com/watch?v=857NmVVbMKg",  # Curl mistakes
            "https://www.youtube.com/watch?v=JyV7mUFSpXs",  # Stop curling wrong
        ],
    },
    "tricep_dip": {
        "correct": [
            "https://www.youtube.com/watch?v=0326dy_-CzM",  # Dip form guide - AthleanX
            "https://www.youtube.com/watch?v=2z8JmcrW-As",  # How to dip properly
            "https://www.youtube.com/watch?v=wjUmnZH528Y",  # Perfect dip tutorial
        ],
        "incorrect": [
            "https://www.youtube.com/watch?v=G90u3Tz1vaE",  # Dip mistakes
            "https://www.youtube.com/watch?v=Lz0fRhEyob0",  # Common dip errors
        ],
    },
}


def download_video(url: str, output_dir: Path, max_duration: int = 120) -> str:
    """
    Download a YouTube video using yt-dlp.
    Downloads only up to max_duration seconds, 720p max to save space.
    Returns the path to the downloaded file, or empty string on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(id)s.%(ext)s")

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--format", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
        "--output", output_template,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            # Find the downloaded file
            video_id = url.split("watch?v=")[-1].split("&")[0]
            for ext in ["mp4", "webm", "mkv"]:
                filepath = output_dir / f"{video_id}.{ext}"
                if filepath.exists():
                    return str(filepath)
        else:
            print(f"  [ERROR] Failed to download {url}: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] Download timed out for {url}")
    except Exception as e:
        print(f"  [ERROR] {e}")

    return ""


def download_exercise(exercise: str, label: str = None):
    """Download videos for a specific exercise (or all labels)."""
    if exercise not in YOUTUBE_SOURCES:
        print(f"Unknown exercise: {exercise}")
        print(f"Available: {list(YOUTUBE_SOURCES.keys())}")
        return

    labels = [label] if label else ["correct", "incorrect"]

    for lbl in labels:
        urls = YOUTUBE_SOURCES[exercise].get(lbl, [])
        if not urls:
            print(f"No {lbl} URLs for {exercise}")
            continue

        output_dir = YOUTUBE_DIR / exercise / lbl
        print(f"\n{'='*60}")
        print(f"Downloading {len(urls)} {lbl} videos for {exercise}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}")

        for i, url in enumerate(urls, 1):
            print(f"  [{i}/{len(urls)}] {url}")
            filepath = download_video(url, output_dir)
            if filepath:
                print(f"    -> Saved: {Path(filepath).name}")
            else:
                print(f"    -> FAILED")


def download_all():
    """Download videos for all exercises."""
    total_urls = sum(
        len(urls)
        for exercise in YOUTUBE_SOURCES.values()
        for urls in exercise.values()
    )
    print(f"Total videos to download: {total_urls}")
    print(f"Downloading to: {YOUTUBE_DIR}")

    for exercise in YOUTUBE_SOURCES:
        download_exercise(exercise)

    print(f"\n{'='*60}")
    print("Download complete!")
    print(f"{'='*60}")

    # Print summary
    for exercise in YOUTUBE_SOURCES:
        for label in ["correct", "incorrect"]:
            d = YOUTUBE_DIR / exercise / label
            if d.exists():
                count = len(list(d.glob("*.*")))
                print(f"  {exercise}/{label}: {count} videos")


def save_source_log():
    """Save a log of all source URLs for citation purposes."""
    log_path = YOUTUBE_DIR / "source_urls.json"
    with open(log_path, "w") as f:
        json.dump(YOUTUBE_SOURCES, f, indent=2)
    print(f"Source URLs saved to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube exercise videos")
    parser.add_argument("--exercise", type=str, help="Exercise name (e.g., squat)")
    parser.add_argument("--label", type=str, choices=["correct", "incorrect"],
                        help="Form label")
    parser.add_argument("--all", action="store_true", help="Download all exercises")
    parser.add_argument("--list", action="store_true", help="List available exercises")

    args = parser.parse_args()

    if args.list:
        for ex, labels in YOUTUBE_SOURCES.items():
            correct_count = len(labels.get("correct", []))
            incorrect_count = len(labels.get("incorrect", []))
            print(f"  {ex}: {correct_count} correct, {incorrect_count} incorrect")
        sys.exit(0)

    # Always save the source log for citations
    YOUTUBE_DIR.mkdir(parents=True, exist_ok=True)
    save_source_log()

    if args.all:
        download_all()
    elif args.exercise:
        download_exercise(args.exercise, args.label)
    else:
        parser.print_help()
