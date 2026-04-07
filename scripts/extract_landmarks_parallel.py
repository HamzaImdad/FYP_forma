"""
Parallel landmark extraction — runs multiple exercises concurrently.
Uses multiprocessing to extract landmarks from videos across exercises.
Skips already-processed CSVs.

Usage:
    python scripts/extract_landmarks_parallel.py --workers 4
"""
import os
import sys
import csv
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LANDMARKS_DIR = DATA_DIR / "processed" / "landmarks"
LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = PROJECT_ROOT / "models" / "mediapipe"
MODEL_PATH = MODEL_DIR / "pose_landmarker_heavy.task"

NUM_LANDMARKS = 33
FRAME_SKIP = 1  # Full frame density for maximum data

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


def get_csv_headers():
    """Match exact header format from extract_landmarks.py."""
    headers = ["frame_id", "timestamp_ms"]
    for name in LANDMARK_NAMES:
        headers.extend([f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_vis"])
    for name in LANDMARK_NAMES:
        headers.extend([f"{name}_wx", f"{name}_wy", f"{name}_wz"])
    return headers


def extract_single_video(args):
    """Extract landmarks from one video. Designed for multiprocessing."""
    video_path, output_csv, source, exercise, label, video_id = args

    if os.path.exists(output_csv):
        return {"status": "skipped", "video": video_id, "exercise": exercise}

    # Import mediapipe inside worker to avoid pickling issues
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"status": "error", "video": video_id, "exercise": exercise, "error": "Cannot open"}

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Create landmarker
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
        landmarker = vision.PoseLandmarker.create_from_options(options)

        headers = get_csv_headers()
        rows = []
        frame_count = 0
        valid_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if FRAME_SKIP > 1 and frame_count % FRAME_SKIP != 0:
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

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

        return {
            "status": "done",
            "video": video_id,
            "exercise": exercise,
            "frames": len(rows),
            "valid": valid_count,
        }
    except Exception as e:
        return {"status": "error", "video": video_id, "exercise": exercise, "error": str(e)}


def collect_youtube_videos():
    """Collect all youtube videos that need processing."""
    youtube_dir = DATA_DIR / "datasets" / "youtube"
    video_extensions = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
    tasks = []

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
                    video_id = video_file.stem
                    output_csv = str(LANDMARKS_DIR / f"youtube_{exercise}_{label}_{video_id}.csv")
                    tasks.append((str(video_file), output_csv, "youtube", exercise, label, video_id))

    return tasks


def collect_kaggle_videos():
    """Collect all kaggle workout videos that map to our exercises."""
    kaggle_dir = DATA_DIR / "datasets" / "kaggle_workout"
    if not kaggle_dir.exists():
        return []

    exercise_mapping = {
        "squat": "squat", "squats": "squat",
        "lunge": "lunge", "lunges": "lunge",
        "deadlift": "deadlift", "romanian deadlift": "deadlift",
        "bench press": "bench_press",
        "shoulder press": "overhead_press",
        "pull up": "pullup", "push-up": "pushup",
        "plank": "plank",
        "barbell biceps curl": "bicep_curl",
        "tricep dips": "tricep_dip",
    }
    exclude_folders = {
        "decline bench press", "incline bench press",
        "chest fly machine", "hammer curl", "hip thrust",
        "lat pulldown", "lateral raise", "leg extension",
        "leg raises", "russian twist", "t bar row",
        "tricep pushdown",
    }

    video_extensions = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
    tasks = []

    for root, dirs, files in os.walk(kaggle_dir):
        for f in files:
            from pathlib import Path as P
            if P(f).suffix.lower() not in video_extensions:
                continue
            filepath = P(root) / f
            rel_path = filepath.relative_to(kaggle_dir)
            folder_name = rel_path.parts[0].lower() if len(rel_path.parts) > 1 else ""

            if folder_name in exclude_folders:
                continue

            exercise = exercise_mapping.get(folder_name, None)
            if exercise is None:
                for key, val in exercise_mapping.items():
                    if key in folder_name:
                        exercise = val
                        break

            if exercise:
                video_id = filepath.stem
                output_csv = str(LANDMARKS_DIR / f"kaggle_{exercise}_correct_{video_id}.csv")
                tasks.append((str(filepath), output_csv, "kaggle", exercise, "correct", video_id))

    return tasks


def collect_youtube_incorrect_videos():
    """Collect all youtube incorrect-form videos."""
    yt_dir = DATA_DIR / "datasets" / "youtube_incorrect"
    if not yt_dir.exists():
        return []

    video_extensions = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
    tasks = []

    for exercise_dir in sorted(yt_dir.iterdir()):
        if not exercise_dir.is_dir():
            continue
        exercise = exercise_dir.name
        for video_file in sorted(exercise_dir.iterdir()):
            if video_file.suffix.lower() in video_extensions:
                video_id = video_file.stem
                output_csv = str(LANDMARKS_DIR / f"ytincorrect_{exercise}_incorrect_{video_id}.csv")
                tasks.append((str(video_file), output_csv, "ytincorrect", exercise, "incorrect", video_id))

    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=18, help="Number of parallel workers (default: 18 for i7-14700HX)")
    parser.add_argument("--source", choices=["youtube", "kaggle", "youtube_incorrect", "all"],
                        default="all", help="Which source to process")
    args = parser.parse_args()

    tasks = []
    if args.source in ("youtube", "all"):
        tasks.extend(collect_youtube_videos())
    if args.source in ("kaggle", "all"):
        tasks.extend(collect_kaggle_videos())
    if args.source in ("youtube_incorrect", "all"):
        tasks.extend(collect_youtube_incorrect_videos())

    to_process = [t for t in tasks if not os.path.exists(t[1])]
    already_done = len(tasks) - len(to_process)

    print(f"Total videos: {len(tasks)}")
    print(f"Already processed: {already_done}")
    print(f"To extract: {len(to_process)}")
    print(f"Workers: {args.workers}")
    print(f"Frame skip: {FRAME_SKIP}")
    print()

    if not to_process:
        print("Nothing to do!")
        return

    done = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(extract_single_video, task): task for task in to_process}
        for future in as_completed(futures):
            result = future.result()
            done += 1
            if result["status"] == "done":
                print(f"[{done}/{len(to_process)}] {result['exercise']}/{result['video']}: "
                      f"{result['frames']} frames ({result['valid']} valid)")
            elif result["status"] == "error":
                errors += 1
                print(f"[{done}/{len(to_process)}] ERROR {result['exercise']}/{result['video']}: "
                      f"{result.get('error', 'unknown')}")
            else:
                print(f"[{done}/{len(to_process)}] SKIP {result['exercise']}/{result['video']}")

    print(f"\nDone! Processed: {done}, Errors: {errors}")


if __name__ == "__main__":
    main()
