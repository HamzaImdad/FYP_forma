"""
Parallel landmark extraction using multiprocessing.

Splits videos across N worker processes, each with its own MediaPipe instance.
Skips already-processed videos automatically.

Usage:
    python scripts/extract_parallel.py --workers 4
    python scripts/extract_parallel.py --workers 4 --source youtube_incorrect
"""

import os
import sys
import csv
import argparse
import time
from pathlib import Path
from multiprocessing import Pool, current_process

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LANDMARKS_DIR = DATA_DIR / "processed" / "landmarks"
LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = PROJECT_ROOT / "models" / "mediapipe"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "pose_landmarker_heavy.task"

NUM_LANDMARKS = 33

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
    if MODEL_PATH.exists():
        return
    print("Downloading MediaPipe Pose Landmarker Heavy model...")
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    urllib.request.urlretrieve(url, str(MODEL_PATH))
    print(f"Model downloaded to: {MODEL_PATH}")


def get_csv_headers():
    headers = ["frame_id", "timestamp_ms"]
    for name in LANDMARK_NAMES:
        headers.extend([f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_vis"])
    for name in LANDMARK_NAMES:
        headers.extend([f"{name}_wx", f"{name}_wy", f"{name}_wz"])
    return headers


def process_single_video(args):
    """Worker function: extract landmarks from one video.

    Args is a tuple: (video_path, output_csv, frame_skip)
    """
    video_path, output_csv, frame_skip = args
    worker = current_process().name

    # Skip if already done
    if os.path.exists(output_csv):
        return {"status": "skipped", "video": video_path}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"status": "failed", "video": video_path, "error": "cannot open"}

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
        landmarker = vision.PoseLandmarker.create_from_options(options)
    except Exception as e:
        cap.release()
        return {"status": "failed", "video": video_path, "error": str(e)}

    headers = get_csv_headers()
    rows = []
    frame_count = 0
    valid_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_skip > 1 and frame_count % frame_skip != 0:
            frame_count += 1
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(frame_count * (1000.0 / video_fps))

        try:
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception:
            frame_count += 1
            continue

        row = [frame_count, timestamp_ms]

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            for lm in result.pose_landmarks[0]:
                row.extend([lm.x, lm.y, lm.z, lm.visibility])
            if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
                for lm in result.pose_world_landmarks[0]:
                    row.extend([lm.x, lm.y, lm.z])
            else:
                row.extend([0.0] * (NUM_LANDMARKS * 3))
            valid_count += 1
        else:
            row.extend([0.0] * (NUM_LANDMARKS * 4))
            row.extend([0.0] * (NUM_LANDMARKS * 3))

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
    det_rate = round(valid_count / max(frame_count, 1) * 100, 1)
    safe_name = Path(video_path).stem.encode("ascii", errors="replace").decode("ascii")

    return {
        "status": "done",
        "video": safe_name,
        "frames": frame_count,
        "valid": valid_count,
        "det_rate": det_rate,
        "duration": round(duration, 1),
        "worker": worker,
    }


def collect_videos(source):
    """Collect all videos to process for a given source."""
    video_extensions = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
    tasks = []

    if source in ("youtube_incorrect", "all"):
        yt_dir = DATA_DIR / "datasets" / "youtube_incorrect"
        if yt_dir.exists():
            for exercise_dir in sorted(yt_dir.iterdir()):
                if not exercise_dir.is_dir():
                    continue
                exercise = exercise_dir.name
                for vf in sorted(exercise_dir.iterdir()):
                    if vf.suffix.lower() in video_extensions:
                        out = LANDMARKS_DIR / f"ytincorrect_{exercise}_incorrect_{vf.stem}.csv"
                        tasks.append((str(vf), str(out)))

    if source in ("youtube", "all"):
        yt_dir = DATA_DIR / "datasets" / "youtube"
        if yt_dir.exists():
            for exercise_dir in sorted(yt_dir.iterdir()):
                if not exercise_dir.is_dir():
                    continue
                exercise = exercise_dir.name
                for label_dir in sorted(exercise_dir.iterdir()):
                    if not label_dir.is_dir():
                        continue
                    label = label_dir.name
                    for vf in sorted(label_dir.iterdir()):
                        if vf.suffix.lower() in video_extensions:
                            out = LANDMARKS_DIR / f"youtube_{exercise}_{label}_{vf.stem}.csv"
                            tasks.append((str(vf), str(out)))

    if source in ("kaggle", "all"):
        kaggle_dir = DATA_DIR / "datasets" / "kaggle_workout"
        if kaggle_dir.exists():
            exercise_mapping = {
                "squat": "squat", "squats": "squat",
                "lunge": "lunge", "lunges": "lunge",
                "deadlift": "deadlift",
                "bench press": "bench_press", "bench_press": "bench_press",
                "overhead press": "overhead_press", "shoulder press": "overhead_press",
                "pull up": "pullup", "pull-up": "pullup", "pullup": "pullup",
                "push up": "pushup", "push-up": "pushup", "pushup": "pushup",
                "plank": "plank",
                "bicep curl": "bicep_curl", "bicep_curl": "bicep_curl",
                "tricep dip": "tricep_dip", "tricep_dip": "tricep_dip",
            }
            for root, dirs, files in os.walk(kaggle_dir):
                for f in files:
                    if Path(f).suffix.lower() not in video_extensions:
                        continue
                    filepath = Path(root) / f
                    rel_path = filepath.relative_to(kaggle_dir)
                    folder_name = rel_path.parts[0].lower() if len(rel_path.parts) > 1 else ""
                    exercise = exercise_mapping.get(folder_name)
                    if exercise is None:
                        for key, val in exercise_mapping.items():
                            if key in folder_name:
                                exercise = val
                                break
                    if exercise:
                        out = LANDMARKS_DIR / f"kaggle_{exercise}_correct_{filepath.stem}.csv"
                        tasks.append((str(filepath), str(out)))

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Parallel landmark extraction")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--source", type=str, default="youtube_incorrect",
                        choices=["youtube", "youtube_incorrect", "kaggle", "all"],
                        help="Which video source to process")
    parser.add_argument("--frame-skip", type=int, default=2,
                        help="Process every Nth frame (default: 2)")
    args = parser.parse_args()

    download_model()

    all_tasks = collect_videos(args.source)

    # Filter out already-processed
    pending = [(vp, out) for vp, out in all_tasks if not os.path.exists(out)]
    skipped = len(all_tasks) - len(pending)

    print(f"Source: {args.source}")
    print(f"Total videos: {len(all_tasks)}")
    print(f"Already done: {skipped}")
    print(f"To process: {len(pending)}")
    print(f"Workers: {args.workers}")
    print(f"Frame skip: {args.frame_skip}")
    print(f"{'='*60}")

    if not pending:
        print("Nothing to do!")
        return

    # Build args for pool
    pool_args = [(vp, out, args.frame_skip) for vp, out in pending]

    done = 0
    failed = 0
    start_time = time.time()

    with Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(process_single_video, pool_args):
            if result["status"] == "done":
                done += 1
                elapsed = time.time() - start_time
                rate = done / elapsed * 60  # videos per minute
                remaining = (len(pending) - done - failed) / max(rate, 0.01)
                print(
                    f"[{done+failed}/{len(pending)}] "
                    f"{result['video'][:50]}: "
                    f"{result['frames']} frames, "
                    f"{result['det_rate']}% detected, "
                    f"{result['duration']}s "
                    f"(~{remaining:.0f} min left)"
                )
            elif result["status"] == "failed":
                failed += 1
                safe = str(result.get("video", "?")).encode("ascii", errors="replace").decode("ascii")
                print(f"[FAIL] {safe}: {result.get('error', '?')}")
            # skipped silently

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Done: {done}, Failed: {failed}, Skipped: {skipped}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Rate: {done/max(elapsed,1)*60:.1f} videos/minute")
    print(f"Output: {LANDMARKS_DIR}")


if __name__ == "__main__":
    main()
