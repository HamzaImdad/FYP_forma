"""
Local webcam demo for ExerVision.

Opens webcam, runs the full pipeline, shows annotated video in an OpenCV window.
Press 'q' to quit, number keys 0-9 to switch exercises.

Usage:
    python scripts/demo_webcam.py
    python scripts/demo_webcam.py --exercise squat
    python scripts/demo_webcam.py --exercise bicep_curl --classifier bilstm
"""

import sys
import argparse
from pathlib import Path

import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.realtime import ExerVisionPipeline
from src.utils.constants import EXERCISES


def main():
    parser = argparse.ArgumentParser(description="ExerVision webcam demo")
    parser.add_argument("--exercise", type=str, default="squat", choices=EXERCISES)
    parser.add_argument("--classifier", type=str, default="rule_based",
                        choices=["rule_based", "ml", "bilstm"],
                        help="Classifier type")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    args = parser.parse_args()

    # Initialize pipeline
    print(f"Starting ExerVision - Exercise: {args.exercise}")
    print(f"Classifier: {args.classifier}")
    pipeline = ExerVisionPipeline(exercise=args.exercise, classifier_type=args.classifier)

    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    print("\nControls:")
    print("  q - Quit")
    print("  r - Reset rep counter")
    for i, ex in enumerate(EXERCISES):
        print(f"  {i} - Switch to {ex}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC)) or frame_idx * 33

        # Process frame
        annotated, result = pipeline.process_frame(frame, timestamp_ms)

        # Display
        cv2.imshow("ExerVision", annotated)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            pipeline.reset()
            print("Reset.")
        elif ord("0") <= key <= ord("9"):
            idx = key - ord("0")
            if idx < len(EXERCISES):
                pipeline.exercise = EXERCISES[idx]
                print(f"Switched to: {EXERCISES[idx]}")

    cap.release()
    cv2.destroyAllWindows()
    pipeline.close()
    print("Done.")


if __name__ == "__main__":
    main()
