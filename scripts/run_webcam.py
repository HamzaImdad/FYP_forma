"""
Direct webcam exercise form evaluator — no web server, no latency.
Opens webcam with OpenCV, processes frames locally, displays annotated output.

Usage:
    python scripts/run_webcam.py --exercise squat
    python scripts/run_webcam.py --exercise bicep_curl --classifier bilstm

Controls:
    Q / ESC  — quit
    SPACE    — pause/resume
    M        — toggle mirror
    1-0      — switch exercise (1=squat, 2=lunge, ..., 0=tricep_dip)
    R        — switch to rule_based
    F        — switch to ml (Random Forest)
    B        — switch to bilstm
"""

import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.realtime import ExerVisionPipeline
from src.utils.constants import EXERCISES
from src.utils.config import Config


def main():
    parser = argparse.ArgumentParser(description="ExerVision direct webcam evaluator")
    parser.add_argument("--exercise", default="squat", choices=EXERCISES,
                        help="Exercise to evaluate")
    parser.add_argument("--classifier", default="rule_based",
                        choices=["rule_based", "ml", "bilstm"],
                        help="Classifier type")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default 0)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--mirror", action="store_true", default=True,
                        help="Mirror the webcam (default: on)")
    args = parser.parse_args()

    config = Config()
    config.model_path = config.project_root / "models" / "mediapipe" / "pose_landmarker_lite.task"

    pipeline = ExerVisionPipeline(
        exercise=args.exercise,
        classifier_type=args.classifier,
        config=config,
    )
    pipeline.start_session()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("ERROR: Could not open camera", args.camera)
        return

    print(f"Exercise: {args.exercise}  |  Classifier: {args.classifier}")
    print("Controls: Q=quit  SPACE=pause  M=mirror  1-0=exercise  R/F/B=classifier")

    mirror = args.mirror
    paused = False
    exercise = args.exercise
    classifier = args.classifier
    frame_count = 0

    # Exercise key mapping: 1-9,0 maps to EXERCISES[0]-EXERCISES[9]
    exercise_keys = {
        ord('1'): 0, ord('2'): 1, ord('3'): 2, ord('4'): 3, ord('5'): 4,
        ord('6'): 5, ord('7'): 6, ord('8'): 7, ord('9'): 8, ord('0'): 9,
    }

    window_name = "ExerVision"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed")
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        if not paused:
            frame_count += 1
            timestamp_ms = int(frame_count * (1000 / 30))
            annotated, result = pipeline.process_frame(frame, timestamp_ms)
        else:
            annotated = frame.copy()
            cv2.putText(annotated, "PAUSED", (annotated.shape[1] // 2 - 80, annotated.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Draw HUD overlay
        fps = pipeline.fps
        reps = pipeline.rep_count
        score_text = ""
        if not paused and result:
            score_pct = int(result.form_score * 100)
            active = "ACTIVE" if result.is_active else "INACTIVE"
            color = (0, 200, 0) if result.is_correct else (0, 0, 255)
            score_text = f"Score: {score_pct}%  |  {active}"
            cv2.putText(annotated, score_text, (10, annotated.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        hud = f"{exercise.upper()} | {classifier} | FPS: {fps:.0f} | Reps: {reps}"
        cv2.putText(annotated, hud, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(window_name, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # Q or ESC
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('m'):
            mirror = not mirror
        elif key == ord('r'):
            classifier = "rule_based"
            pipeline.set_classifier(classifier)
            print(f"Switched to: {classifier}")
        elif key == ord('f'):
            classifier = "ml"
            pipeline.set_classifier(classifier)
            print(f"Switched to: {classifier}")
        elif key == ord('b'):
            classifier = "bilstm"
            pipeline.set_classifier(classifier)
            print(f"Switched to: {classifier}")
        elif key in exercise_keys:
            idx = exercise_keys[key]
            if idx < len(EXERCISES):
                exercise = EXERCISES[idx]
                pipeline.exercise = exercise
                print(f"Switched exercise: {exercise}")

    # End session and print summary
    summary = pipeline.end_session()
    cap.release()
    cv2.destroyAllWindows()

    print("\n--- Session Summary ---")
    print(f"Exercise: {summary['exercise']}")
    print(f"Duration: {summary['duration_sec']:.0f}s")
    print(f"Total reps: {summary['total_reps']}")
    print(f"Good reps: {summary['good_reps']}")
    print(f"Avg form score: {int(summary['avg_form_score'] * 100)}%")
    if summary['common_issues']:
        print("Common issues:")
        for issue in summary['common_issues']:
            print(f"  - {issue['issue']} ({issue['count']}x)")


if __name__ == "__main__":
    main()
