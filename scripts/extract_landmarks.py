"""
Batch Landmark Extraction Script for ExerVision.

Processes all video files in the data directories using MediaPipe BlazePose,
extracts 33 body landmarks per frame, and saves them as CSV files.

This script processes videos from:
  - data/datasets/youtube/{exercise}/{correct|incorrect}/
  - data/datasets/kaggle_workout/
  - data/raw/{exercise}/{correct|incorrect}/

Output:
  - data/processed/landmarks/{exercise}_{label}_{video_id}.csv
    Columns: frame_id, timestamp_ms, lm0_x, lm0_y, lm0_z, lm0_vis, ..., lm32_x, lm32_y, lm32_z, lm32_vis

Usage:
    python scripts/extract_landmarks.py --source youtube
    python scripts/extract_landmarks.py --source kaggle
    python scripts/extract_landmarks.py --source all
    python scripts/extract_landmarks.py --video path/to/video.mp4 --exercise squat --label correct
"""

import os
import sys
import csv
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tqdm import tqdm

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LANDMARKS_DIR = DATA_DIR / "processed" / "landmarks"
LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)

# MediaPipe model path - will download if not present
MODEL_DIR = PROJECT_ROOT / "models" / "mediapipe"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "pose_landmarker_heavy.task"

# Number of MediaPipe BlazePose landmarks
NUM_LANDMARKS = 33

# Landmark names for CSV headers
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def download_model():
    """Download the MediaPipe pose landmarker model if not present."""
    if MODEL_PATH.exists():
        print(f"Model already exists: {MODEL_PATH}")
        return

    print("Downloading MediaPipe Pose Landmarker Heavy model...")
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    urllib.request.urlretrieve(url, str(MODEL_PATH))
    print(f"Model downloaded to: {MODEL_PATH}")


def get_csv_headers():
    """Generate CSV column headers."""
    headers = ["frame_id", "timestamp_ms"]
    for i, name in enumerate(LANDMARK_NAMES):
        headers.extend([f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_vis"])
    # Also add world landmarks
    for i, name in enumerate(LANDMARK_NAMES):
        headers.extend([f"{name}_wx", f"{name}_wy", f"{name}_wz"])
    return headers


def extract_from_video(video_path: str, output_csv: str, max_frames: int = 0, frame_skip: int = 1) -> dict:
    """
    Extract landmarks from a single video file.

    Args:
        video_path: Path to the video file
        output_csv: Path to save the CSV output
        max_frames: Maximum frames to process (0 = all frames)
        frame_skip: Process every Nth frame (1 = all, 3 = every 3rd frame)

    Returns:
        dict with stats: total_frames, valid_frames, fps, duration
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open video: {video_path}")
        return {"total_frames": 0, "valid_frames": 0, "fps": 0, "duration": 0}

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create the pose landmarker
    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )

    landmarker = vision.PoseLandmarker.create_from_options(options)
    headers = get_csv_headers()

    frame_count = 0
    valid_count = 0
    rows = []

    if max_frames > 0:
        total_to_process = min(max_frames, total_frames_in_video)
    else:
        total_to_process = total_frames_in_video

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames > 0 and frame_count >= max_frames:
            break

        # Skip frames if frame_skip > 1
        if frame_skip > 1 and frame_count % frame_skip != 0:
            frame_count += 1
            continue

        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Calculate timestamp in ms
        timestamp_ms = int(frame_count * (1000.0 / video_fps))

        try:
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            frame_count += 1
            continue

        row = [frame_count, timestamp_ms]

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            # Normalized landmarks (0-1 range)
            for lm in result.pose_landmarks[0]:
                row.extend([lm.x, lm.y, lm.z, lm.visibility])

            # World landmarks (meters, centered at hip)
            if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
                for lm in result.pose_world_landmarks[0]:
                    row.extend([lm.x, lm.y, lm.z])
            else:
                row.extend([0.0] * (NUM_LANDMARKS * 3))

            valid_count += 1
        else:
            # No pose detected - fill with zeros
            row.extend([0.0] * (NUM_LANDMARKS * 4))  # normalized
            row.extend([0.0] * (NUM_LANDMARKS * 3))  # world

        rows.append(row)
        frame_count += 1

    cap.release()
    landmarker.close()

    # Write CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    duration = frame_count / video_fps if video_fps > 0 else 0
    stats = {
        "total_frames": frame_count,
        "valid_frames": valid_count,
        "fps": video_fps,
        "duration": round(duration, 2),
        "detection_rate": round(valid_count / max(frame_count, 1) * 100, 1),
    }
    return stats


def process_youtube_videos(frame_skip: int = 1):
    """Process all downloaded YouTube videos."""
    youtube_dir = DATA_DIR / "datasets" / "youtube"
    if not youtube_dir.exists():
        print("No YouTube videos found. Run download_youtube.py first.")
        return

    video_extensions = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
    videos = []

    for exercise_dir in sorted(youtube_dir.iterdir()):
        if not exercise_dir.is_dir():
            continue
        exercise = exercise_dir.name
        for label_dir in sorted(exercise_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            for video_file in sorted(label_dir.iterdir()):
                if video_file.suffix.lower() in video_extensions:
                    videos.append((str(video_file), exercise, label, video_file.stem))

    print(f"\nFound {len(videos)} YouTube videos to process")
    process_video_list(videos, "youtube", frame_skip=frame_skip)


def process_kaggle_videos(frame_skip: int = 1):
    """Process Kaggle workout videos."""
    kaggle_dir = DATA_DIR / "datasets" / "kaggle_workout"
    if not kaggle_dir.exists():
        print("No Kaggle videos found. Download from Kaggle first.")
        return

    # Map Kaggle folder names to our exercise names
    # Exact folder name → exercise mapping
    exercise_mapping = {
        "squat": "squat",
        "squats": "squat",
        "lunge": "lunge",
        "lunges": "lunge",
        "deadlift": "deadlift",
        "romanian deadlift": "deadlift",
        "bench press": "bench_press",
        "bench_press": "bench_press",
        "overhead press": "overhead_press",
        "shoulder press": "overhead_press",
        "pull up": "pullup",
        "pull-up": "pullup",
        "pullup": "pullup",
        "pull ups": "pullup",
        "push up": "pushup",
        "push-up": "pushup",
        "pushup": "pushup",
        "push ups": "pushup",
        "plank": "plank",
        "bicep curl": "bicep_curl",
        "bicep_curl": "bicep_curl",
        "biceps curl": "bicep_curl",
        "barbell biceps curl": "bicep_curl",
        "tricep dip": "tricep_dip",
        "tricep_dip": "tricep_dip",
        "tricep dips": "tricep_dip",
        "dips": "tricep_dip",
    }
    # Folders that should NOT match via partial matching
    exclude_folders = {
        "decline bench press", "incline bench press",
        "chest fly machine", "hammer curl", "hip thrust",
        "lat pulldown", "lateral raise", "leg extension",
        "leg raises", "russian twist", "t bar row",
        "tricep pushdown",
    }

    video_extensions = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
    videos = []

    # Walk through all subdirectories
    for root, dirs, files in os.walk(kaggle_dir):
        for f in files:
            if Path(f).suffix.lower() not in video_extensions:
                continue

            filepath = Path(root) / f
            # Try to determine exercise from folder name
            rel_path = filepath.relative_to(kaggle_dir)
            folder_name = rel_path.parts[0].lower() if len(rel_path.parts) > 1 else ""

            exercise = exercise_mapping.get(folder_name, None)
            if exercise is None and folder_name not in exclude_folders:
                # Try matching partial names
                for key, val in exercise_mapping.items():
                    if key in folder_name:
                        exercise = val
                        break

            if exercise:
                # Kaggle videos are generally correct form demonstrations
                videos.append((str(filepath), exercise, "correct", filepath.stem))

    print(f"\nFound {len(videos)} Kaggle videos matching our exercises")
    process_video_list(videos, "kaggle", frame_skip=frame_skip)


def process_youtube_incorrect_videos(frame_skip: int = 1):
    """Process newly downloaded incorrect-form YouTube videos."""
    yt_dir = DATA_DIR / "datasets" / "youtube_incorrect"
    if not yt_dir.exists():
        print("No youtube_incorrect directory found.")
        return

    video_extensions = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
    videos = []

    for exercise_dir in sorted(yt_dir.iterdir()):
        if not exercise_dir.is_dir():
            continue
        exercise = exercise_dir.name
        for video_file in sorted(exercise_dir.iterdir()):
            if video_file.suffix.lower() in video_extensions:
                videos.append((str(video_file), exercise, "incorrect", video_file.stem))

    print(f"\nFound {len(videos)} YouTube incorrect-form videos to process")
    process_video_list(videos, "ytincorrect", frame_skip=frame_skip)


def process_raw_videos(frame_skip: int = 1):
    """Process self-recorded videos in data/raw/."""
    raw_dir = DATA_DIR / "raw"
    if not raw_dir.exists():
        print("No raw videos found in data/raw/")
        return

    video_extensions = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
    videos = []

    for exercise_dir in sorted(raw_dir.iterdir()):
        if not exercise_dir.is_dir():
            continue
        exercise = exercise_dir.name
        for label_dir in sorted(exercise_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            for video_file in sorted(label_dir.iterdir()):
                if video_file.suffix.lower() in video_extensions:
                    videos.append((str(video_file), exercise, label, video_file.stem))

    print(f"\nFound {len(videos)} raw videos to process")
    process_video_list(videos, "raw", frame_skip=frame_skip)


def process_video_list(videos: list, source: str, frame_skip: int = 1):
    """
    Process a list of (video_path, exercise, label, video_id) tuples.
    """
    if not videos:
        return

    download_model()  # Ensure model is available

    if frame_skip > 1:
        print(f"  Frame skip: processing every {frame_skip}th frame")

    total_stats = {"processed": 0, "failed": 0, "total_frames": 0, "valid_frames": 0}

    for video_path, exercise, label, video_id in tqdm(videos, desc=f"Processing {source}"):
        output_csv = LANDMARKS_DIR / f"{source}_{exercise}_{label}_{video_id}.csv"

        # Skip if already processed
        if output_csv.exists():
            safe_name = output_csv.name.encode("ascii", errors="replace").decode("ascii")
            print(f"  [SKIP] Already processed: {safe_name}")
            continue

        stats = extract_from_video(str(video_path), str(output_csv), frame_skip=frame_skip)

        if stats["total_frames"] > 0:
            total_stats["processed"] += 1
            total_stats["total_frames"] += stats["total_frames"]
            total_stats["valid_frames"] += stats["valid_frames"]
            # Sanitize video_id for safe printing (avoid Unicode encode errors)
            safe_id = video_id.encode("ascii", errors="replace").decode("ascii")
            tqdm.write(
                f"  {exercise}/{label}/{safe_id}: "
                f"{stats['total_frames']} frames, "
                f"{stats['detection_rate']}% detected, "
                f"{stats['duration']}s"
            )
        else:
            total_stats["failed"] += 1
            safe_path = str(video_path).encode("ascii", errors="replace").decode("ascii")
            tqdm.write(f"  [FAIL] {safe_path}")

    print(f"\n{'='*60}")
    print(f"Source: {source}")
    print(f"Processed: {total_stats['processed']}, Failed: {total_stats['failed']}")
    print(f"Total frames: {total_stats['total_frames']}")
    print(f"Valid detections: {total_stats['valid_frames']}")
    if total_stats["total_frames"] > 0:
        rate = total_stats["valid_frames"] / total_stats["total_frames"] * 100
        print(f"Overall detection rate: {rate:.1f}%")
    print(f"Output directory: {LANDMARKS_DIR}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Extract landmarks from exercise videos")
    parser.add_argument("--source", type=str,
                        choices=["youtube", "youtube_incorrect", "kaggle", "raw", "all"],
                        help="Which video source to process")
    parser.add_argument("--video", type=str, help="Process a single video file")
    parser.add_argument("--exercise", type=str, help="Exercise name (for single video)")
    parser.add_argument("--label", type=str, choices=["correct", "incorrect"],
                        help="Form label (for single video)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames per video (0=all)")
    parser.add_argument("--frame-skip", type=int, default=1,
                        help="Process every Nth frame (1=all, 3=every 3rd frame)")

    args = parser.parse_args()

    if args.video:
        if not args.exercise:
            print("--exercise is required when using --video")
            sys.exit(1)
        label = args.label or "unknown"
        video_id = Path(args.video).stem
        output_csv = LANDMARKS_DIR / f"single_{args.exercise}_{label}_{video_id}.csv"

        download_model()
        print(f"Processing: {args.video}")
        stats = extract_from_video(args.video, str(output_csv), args.max_frames, args.frame_skip)
        print(f"Done: {stats}")
        print(f"Output: {output_csv}")

    elif args.source:
        if args.source in ("youtube", "all"):
            process_youtube_videos(frame_skip=args.frame_skip)
        if args.source in ("youtube_incorrect", "all"):
            process_youtube_incorrect_videos(frame_skip=args.frame_skip)
        if args.source in ("kaggle", "all"):
            process_kaggle_videos(frame_skip=args.frame_skip)
        if args.source in ("raw", "all"):
            process_raw_videos(frame_skip=args.frame_skip)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
