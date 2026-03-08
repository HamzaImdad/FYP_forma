"""
Flask + Socket.IO server for ExerVision web app.

Receives video frames from the browser via WebSocket,
processes them through the pipeline, and returns annotated frames + feedback.
Supports session lifecycle, classifier switching, and exercise guidance.

Usage:
    python app/server.py
    python app/server.py --port 5000 --host 0.0.0.0
"""

import os
import sys
import json
import base64
import logging
import argparse
from pathlib import Path

import cv2
import eventlet
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.realtime import ExerVisionPipeline
from src.utils.constants import EXERCISES
from src.utils.config import Config

# Real-time config: use lite model for speed
REALTIME_CONFIG = Config()
REALTIME_CONFIG.model_path = REALTIME_CONFIG.project_root / "models" / "mediapipe" / "pose_landmarker_lite.task"

# Max frame dimension for processing (resize large frames)
MAX_FRAME_DIM = 360

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("EXERVISION_SECRET_KEY", "exervision-dev-fallback")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# One pipeline per connected session
pipelines = {}
# Per-session processing lock to drop frames when busy
processing = {}

# Load exercise metadata
EXERCISE_DATA_PATH = Path(__file__).parent / "exercise_data.json"
EXERCISE_DATA = {}
if EXERCISE_DATA_PATH.exists():
    with open(EXERCISE_DATA_PATH, "r") as f:
        EXERCISE_DATA = json.load(f)

# Pre-load ML models at import time so first connection is fast
from src.classification.ml_classifier import MLClassifier
_preloaded_ml = MLClassifier(REALTIME_CONFIG)
_preloaded_ml.load_all_models()


# ── Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", exercises=EXERCISES)


@app.route("/api/exercises")
def get_exercises():
    """Return list of supported exercises with display names."""
    result = []
    for ex in EXERCISES:
        info = EXERCISE_DATA.get(ex, {})
        result.append({
            "id": ex,
            "display_name": info.get("display_name", ex.replace("_", " ").title()),
            "muscles": info.get("muscles", []),
        })
    return jsonify({"exercises": result})


@app.route("/api/exercise_guide/<exercise>")
def get_exercise_guide(exercise):
    """Return detailed guide for an exercise."""
    if exercise not in EXERCISES:
        return jsonify({"error": f"Unknown exercise: {exercise}"}), 404

    info = EXERCISE_DATA.get(exercise, {})
    return jsonify({
        "exercise": exercise,
        "display_name": info.get("display_name", exercise.replace("_", " ").title()),
        "muscles": info.get("muscles", []),
        "instructions": info.get("instructions", []),
        "common_mistakes": info.get("common_mistakes", []),
    })


# ── Socket.IO Events ────────────────────────────────────────────────────

@socketio.on("connect")
def handle_connect():
    sid = request.sid
    pipelines[sid] = ExerVisionPipeline(
        exercise="squat", classifier_type="ml", config=REALTIME_CONFIG,
        preloaded_classifier=_preloaded_ml,
    )
    processing[sid] = False
    logger.info("Client connected: %s", sid)
    emit("connected", {"status": "ok", "exercises": EXERCISES})


@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    if sid in pipelines:
        try:
            pipelines[sid].close()
        except Exception as e:
            logger.error("Failed to close pipeline for %s: %s", sid, e)
        del pipelines[sid]
    processing.pop(sid, None)
    logger.info("Client disconnected: %s", sid)


@socketio.on("set_exercise")
def handle_set_exercise(data):
    sid = request.sid
    exercise = data.get("exercise", "squat")
    if exercise in EXERCISES and sid in pipelines:
        pipelines[sid].exercise = exercise
        emit("exercise_changed", {"exercise": exercise})


@socketio.on("set_classifier")
def handle_set_classifier(data):
    """Switch classifier type at runtime."""
    sid = request.sid
    classifier_type = data.get("classifier", "rule_based")
    if classifier_type not in ("rule_based", "ml", "bilstm"):
        classifier_type = "rule_based"

    if sid in pipelines:
        pipelines[sid].set_classifier(classifier_type)
        emit("classifier_changed", {"classifier": classifier_type})


@socketio.on("start_session")
def handle_start_session(data=None):
    """Start a tracked exercise session."""
    sid = request.sid
    if sid not in pipelines:
        return

    pipeline = pipelines[sid]

    # Optionally set exercise and classifier from data
    if data:
        exercise = data.get("exercise")
        if exercise and exercise in EXERCISES:
            pipeline.exercise = exercise

        classifier = data.get("classifier")
        if classifier in ("rule_based", "ml", "bilstm"):
            pipeline.set_classifier(classifier)

    pipeline.start_session()
    emit("session_started", {
        "exercise": pipeline.exercise,
        "classifier": pipeline.classifier_type,
    })


@socketio.on("end_session")
def handle_end_session():
    """End session and return the summary report."""
    sid = request.sid
    if sid not in pipelines:
        return

    summary = pipelines[sid].end_session()
    emit("session_report", summary)


@socketio.on("frame")
def handle_frame(data):
    sid = request.sid
    if sid not in pipelines:
        return

    # Drop frame if previous one is still being processed
    if processing.get(sid, False):
        return
    processing[sid] = True

    pipeline = pipelines[sid]

    # Decode base64 JPEG from browser
    frame_data = data.get("image", "")
    if "," in frame_data:
        frame_data = frame_data.split(",")[1]

    try:
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        processing[sid] = False
        return

    if frame is None:
        processing[sid] = False
        return

    # Resize large frames for faster processing
    h, w = frame.shape[:2]
    if max(h, w) > MAX_FRAME_DIM:
        scale = MAX_FRAME_DIM / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    timestamp_ms = data.get("timestamp", 0)

    # Process through pipeline (offload to real thread to avoid blocking eventlet)
    try:
        annotated, result = eventlet.tpool.execute(pipeline.process_frame, frame, int(timestamp_ms))
    except Exception as e:
        logger.error("Frame processing error for %s: %s", sid, e)
        processing[sid] = False
        emit("processed", {"image": "", "error": "Processing error"})
        return

    # Encode annotated frame back to JPEG
    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 65])
    annotated_b64 = base64.b64encode(buffer).decode("utf-8")

    # Build feedback response
    feedback = {
        "image": f"data:image/jpeg;base64,{annotated_b64}",
        "exercise": pipeline.exercise,
        "rep_count": pipeline.rep_count,
        "fps": round(pipeline.fps, 1),
        "classifier": pipeline.classifier_type,
    }

    if result:
        feedback["is_correct"] = result.is_correct
        feedback["confidence"] = round(result.confidence, 2) if result.confidence is not None else 0
        feedback["details"] = list(result.details.values()) if result.details else []
    else:
        feedback["is_correct"] = None
        feedback["confidence"] = 0
        feedback["details"] = []

    processing[sid] = False
    emit("processed", feedback)

    # Emit rep_completed event if a new rep was detected
    rep_info = pipeline.last_rep_completed
    if rep_info:
        emit("rep_completed", rep_info)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ExerVision web server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger.info("Starting ExerVision server at http://%s:%s", args.host, args.port)
    logger.info("Open in browser: http://localhost:%s", args.port)
    socketio.run(app, host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
