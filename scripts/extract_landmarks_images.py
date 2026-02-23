"""
Extract MediaPipe landmarks from exercise images (Kaggle dataset).

Processes images and saves one row per image with 33 landmark coordinates.

Usage:
    python scripts/extract_landmarks_images.py --source kaggle_images
    python scripts/extract_landmarks_images.py --source kaggle_images --exercise squat
"""

import os
import sys
import csv
import argparse
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tqdm import tqdm

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

# Map Kaggle folder names to our exercise names
KAGGLE_IMAGE_MAPPING = {
    "squat": "squat",
    "deadlift": "deadlift",
    "bench press": "bench_press",
    "shoulder press": "overhead_press",
    "pull up": "pullup",
    "push up": "pushup",
    "plank": "plank",
    "barbell biceps curl": "bicep_curl",
    "tricep dips": "tricep_dip",
}


def download_model():
    """Download MediaPipe model if needed."""
    if MODEL_PATH.exists():
        return
    print("Downloading MediaPipe Pose Landmarker Heavy model...")
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    urllib.request.urlretrieve(url, str(MODEL_PATH))
    print(f"Model downloaded: {MODEL_PATH}")


def get_csv_headers():
    """CSV column headers."""
    headers = ["image_id", "source", "exercise", "label"]
    for name in LANDMARK_NAMES:
        headers.extend([f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_vis"])
    for name in LANDMARK_NAMES:
        headers.extend([f"{name}_wx", f"{name}_wy", f"{name}_wz"])
    return headers


def extract_from_image(landmarker, image_path: str) -> list:
    """
    Extract landmarks from a single image.
    Returns list of values (without the metadata columns) or None if no pose detected.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)
    except Exception:
        return None

    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return None

    row = []
    for lm in result.pose_landmarks[0]:
        row.extend([lm.x, lm.y, lm.z, lm.visibility])

    if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
        for lm in result.pose_world_landmarks[0]:
            row.extend([lm.x, lm.y, lm.z])
    else:
        row.extend([0.0] * (NUM_LANDMARKS * 3))

    return row


def process_kaggle_images(exercise_filter: str = None):
    """Process Kaggle exercise images."""
    kaggle_img_dir = DATA_DIR / "datasets" / "kaggle_images"
    if not kaggle_img_dir.exists():
        print("Kaggle images not found. Download first.")
        return

    download_model()

    # Create landmarker for IMAGE mode
    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    headers = get_csv_headers()
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for kaggle_name, our_name in KAGGLE_IMAGE_MAPPING.items():
        if exercise_filter and our_name != exercise_filter:
            continue

        img_dir = kaggle_img_dir / kaggle_name
        if not img_dir.exists():
            print(f"Skipping {kaggle_name} (not found)")
            continue

        output_csv = LANDMARKS_DIR / f"kaggle_images_{our_name}_correct.csv"

        if output_csv.exists():
            print(f"[SKIP] Already processed: {output_csv.name}")
            continue

        images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in image_exts])
        print(f"\nProcessing {our_name}: {len(images)} images from '{kaggle_name}'")

        rows = []
        valid = 0
        for img_path in tqdm(images, desc=f"  {our_name}"):
            landmarks = extract_from_image(landmarker, str(img_path))
            if landmarks:
                row = [img_path.stem, "kaggle_images", our_name, "correct"] + landmarks
                rows.append(row)
                valid += 1

        # Write CSV
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

        print(f"  -> {valid}/{len(images)} poses detected ({valid/len(images)*100:.1f}%)")
        print(f"  -> Saved: {output_csv.name}")

    landmarker.close()
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="kaggle_images",
                        choices=["kaggle_images"])
    parser.add_argument("--exercise", type=str, help="Filter to one exercise")
    args = parser.parse_args()

    if args.source == "kaggle_images":
        process_kaggle_images(args.exercise)
