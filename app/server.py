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
import time
import base64
import logging
import argparse
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.realtime import ExerVisionPipeline
from src.pose_estimation.base import PoseResult
from src.utils.constants import EXERCISES
from src.utils.config import Config

# Real-time config: use lite model for speed
REALTIME_CONFIG = Config()
REALTIME_CONFIG.model_path = REALTIME_CONFIG.project_root / "models" / "mediapipe" / "pose_landmarker_lite.task"
REALTIME_CONFIG.min_detection_confidence = 0.5
REALTIME_CONFIG.min_tracking_confidence = 0.5

# Max frame dimension for processing (resize large frames)
MAX_FRAME_DIM = 720

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("EXERVISION_SECRET_KEY", "exervision-dev-fallback")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# One pipeline per connected session
pipelines = {}
# Per-session processing lock to drop frames when busy
processing = {}
# Per-session last frame timestamp for server-side rate limiting (max 15 fps)
last_frame_time = {}
MIN_FRAME_INTERVAL = 1.0 / 20  # ~50ms

# Initialize session history database
from app.database import ExerVisionDB
db = ExerVisionDB(str(PROJECT_ROOT / "exervision.db"))

# Load exercise metadata
EXERCISE_DATA_PATH = Path(__file__).parent / "exercise_data.json"
EXERCISE_DATA = {}
if EXERCISE_DATA_PATH.exists():
    with open(EXERCISE_DATA_PATH, "r", encoding="utf-8") as f:
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
    """Return list of supported exercises with display names and model info."""
    models_dir = PROJECT_ROOT / "models" / "trained"
    result = []
    for ex in EXERCISES:
        info = EXERCISE_DATA.get(ex, {})
        # Check which classifier models are available and their F1 scores
        models_available = {}
        for model_type, suffix in [("ml", "_classifier.pkl"), ("bilstm", "_bilstm_v2.pt")]:
            model_path = models_dir / f"{ex}{suffix}"
            if model_path.exists():
                f1 = None
                # Try exercise-specific eval file first
                eval_path = models_dir / f"{ex}_eval.json"
                if eval_path.exists():
                    try:
                        with open(eval_path, "r") as ef:
                            eval_data = json.load(ef)
                            f1 = eval_data.get("f1_score") or eval_data.get("f1")
                    except Exception:
                        pass
                # Fall back to BiLSTM training summary for v2 models
                if f1 is None and model_type == "bilstm":
                    summary_path = PROJECT_ROOT / "reports" / "bilstm_v2_training_summary.json"
                    if summary_path.exists():
                        try:
                            with open(summary_path, "r") as sf:
                                summary = json.load(sf)
                            if ex in summary:
                                f1 = summary[ex].get("f1")
                        except Exception:
                            pass
                models_available[model_type] = {"available": True, "f1": round(f1, 3) if f1 is not None else None}
        # Rule-based is always available
        models_available["rule_based"] = {"available": True, "f1": None}
        # Dedicated detector always available for all exercises
        models_available["detector"] = {"available": True, "f1": None}

        result.append({
            "id": ex,
            "display_name": info.get("display_name", ex.replace("_", " ").title()),
            "muscles": info.get("muscles", []),
            "models": models_available,
            "has_detector": True,
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
        "camera_setup": info.get("camera_setup", ""),
        "camera_angle": info.get("camera_angle", "front"),
        "camera_distance": info.get("camera_distance", "2-3m"),
        "camera_height": info.get("camera_height", "waist"),
        "instructions": info.get("instructions", []),
        "common_mistakes": info.get("common_mistakes", []),
    })


def _add_detector_fields(feedback, pipeline, result):
    """Add detector-specific fields (set/phase/progress) to the response."""
    detector = pipeline._detectors.get(pipeline.exercise)
    if detector:
        feedback["set_count"] = detector.set_count
        feedback["reps_in_set"] = detector.reps_in_current_set
        if result and isinstance(result.details, dict):
            feedback["phase"] = result.details.get("phase", "")
            feedback["progress"] = result.details.get("progress", "")
            feedback["rest_tier"] = result.details.get("rest_tier", "")
            feedback["activity_state"] = result.details.get("activity", "")
            # Plank: include hold duration
            if hasattr(detector, '_hold_duration'):
                feedback["hold_duration"] = round(detector._hold_duration, 1)
        else:
            feedback["phase"] = ""
            feedback["progress"] = ""


# ── Dashboard API ──────────────────────────────────────────────────────

@app.route("/api/dashboard/stats")
def dashboard_stats():
    """Aggregate stats for dashboard summary cards."""
    period = request.args.get("period")
    exercise = request.args.get("exercise")
    return jsonify(db.get_stats(period, exercise))


@app.route("/api/dashboard/sessions")
def dashboard_sessions():
    """List sessions with optional filters."""
    exercise = request.args.get("exercise")
    period = request.args.get("period")
    limit = int(request.args.get("limit", 20))
    offset = int(request.args.get("offset", 0))
    return jsonify(db.get_sessions(exercise, period, limit, offset))


@app.route("/api/dashboard/session/<int:session_id>")
def dashboard_session_detail(session_id):
    """Full session detail with per-rep breakdown."""
    detail = db.get_session_detail(session_id)
    if not detail:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(detail)


@app.route("/api/dashboard/scores")
def dashboard_scores():
    """Daily average form scores for line chart."""
    days = int(request.args.get("days", 30))
    exercise = request.args.get("exercise")
    return jsonify(db.get_daily_scores(days, exercise))


@app.route("/api/dashboard/distribution")
def dashboard_distribution():
    """Exercise breakdown for pie/doughnut chart."""
    period = request.args.get("period")
    return jsonify(db.get_exercise_distribution(period))


# ── Socket.IO Events ────────────────────────────────────────────────────

@socketio.on("connect")
def handle_connect():
    sid = request.sid
    pipelines[sid] = ExerVisionPipeline(
        exercise="squat", classifier_type="rule_based", config=REALTIME_CONFIG,
        preloaded_classifier=_preloaded_ml,
    )
    processing[sid] = False
    last_frame_time[sid] = 0
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
    last_frame_time.pop(sid, None)
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
    """End session, save to database, and return the summary report."""
    sid = request.sid
    if sid not in pipelines:
        return

    summary = pipelines[sid].end_session()

    # Auto-save session to database
    try:
        session_id = db.save_session(summary)
        summary["session_id"] = session_id
    except Exception as e:
        logger.warning(f"Failed to save session: {e}")

    emit("session_report", summary)


@socketio.on("frame")
def handle_frame(data):
    sid = request.sid
    if sid not in pipelines:
        return

    # Drop frame if previous one is still being processed
    if processing.get(sid, False):
        return

    # Server-side rate limiting: max 20 fps
    now = time.monotonic()
    if now - last_frame_time.get(sid, 0) < MIN_FRAME_INTERVAL:
        return
    last_frame_time[sid] = now

    processing[sid] = True

    pipeline = pipelines[sid]
    t_total_start = time.perf_counter()

    # Decode base64 JPEG from browser
    t0 = time.perf_counter()
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
    t_decode = time.perf_counter() - t0

    timestamp_ms = data.get("timestamp", 0)

    # Process through pipeline
    t0 = time.perf_counter()
    try:
        annotated, result = pipeline.process_frame(frame, int(timestamp_ms))
    except Exception as e:
        logger.error("Frame processing error for %s: %s", sid, e)
        processing[sid] = False
        emit("processed", {"image": "", "error": "Processing error"})
        return
    t_pipeline = time.perf_counter() - t0

    # Encode annotated frame back to JPEG
    t0 = time.perf_counter()
    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
    annotated_b64 = base64.b64encode(buffer).decode("utf-8")
    t_encode = time.perf_counter() - t0

    t_total = time.perf_counter() - t_total_start

    # Build feedback response
    feedback = {
        "image": f"data:image/jpeg;base64,{annotated_b64}",
        "exercise": pipeline.exercise,
        "rep_count": pipeline.rep_count,
        "fps": round(pipeline.fps, 1),
        "classifier": pipeline.classifier_type,
        "timing": {
            "decode_ms": round(t_decode * 1000, 1),
            "pipeline_ms": round(t_pipeline * 1000, 1),
            "encode_ms": round(t_encode * 1000, 1),
            "total_ms": round(t_total * 1000, 1),
        },
        "client_timestamp": timestamp_ms,
    }

    if result:
        feedback["is_correct"] = result.is_correct
        feedback["confidence"] = round(result.confidence, 2) if result.confidence is not None else 0
        feedback["details"] = list(result.details.values()) if result.details else []
        feedback["form_score"] = round(result.form_score, 2)
        feedback["is_active"] = result.is_active
        feedback["joint_feedback"] = result.joint_feedback
    else:
        feedback["is_correct"] = None
        feedback["confidence"] = 0
        feedback["details"] = []
        feedback["form_score"] = 0
        feedback["is_active"] = False
        feedback["joint_feedback"] = {}

    # Push-up specific: send set/phase info for on-video HUD
    _add_detector_fields(feedback, pipeline, result)

    processing[sid] = False
    emit("processed", feedback)

    # Emit rep_completed event if a new rep was detected
    rep_info = pipeline.last_rep_completed
    if rep_info:
        emit("rep_completed", rep_info)


@socketio.on("landmarks")
def handle_landmarks(data):
    """Handle pre-computed landmarks from browser-side MediaPipe.

    Receives flat landmark arrays (~1KB) instead of JPEG frames (~50KB).
    Skips pose estimation and visualization server-side.
    """
    sid = request.sid
    if sid not in pipelines:
        return

    if processing.get(sid, False):
        return

    now = time.monotonic()
    if now - last_frame_time.get(sid, 0) < MIN_FRAME_INTERVAL:
        return
    last_frame_time[sid] = now
    processing[sid] = True

    pipeline = pipelines[sid]
    t_start = time.perf_counter()

    try:
        # Reconstruct PoseResult from flat arrays
        lm_flat = np.array(data["landmarks"], dtype=np.float32)
        wlm_flat = np.array(data["world_landmarks"], dtype=np.float32)

        landmarks = lm_flat.reshape(33, 4)          # x, y, z, visibility
        world_landmarks = wlm_flat.reshape(33, 3)   # x, y, z

        pose = PoseResult(
            landmarks=landmarks,
            world_landmarks=world_landmarks,
            detection_confidence=float(landmarks[:, 3].mean()),
            timestamp_ms=int(data.get("timestamp", 0)),
        )

        # Process through pipeline (skip pose estimation + visualization)
        result = pipeline.process_landmarks(pose, int(data.get("timestamp", 0)))

        t_total = time.perf_counter() - t_start

        if result:
            feedback = {
                "is_correct": result.is_correct,
                "confidence": round(result.confidence, 2) if result.confidence is not None else 0,
                "details": list(result.details.values()) if isinstance(result.details, dict) else result.details,
                "form_score": round(result.form_score, 2),
                "is_active": result.is_active,
                "joint_feedback": result.joint_feedback,
                "rep_count": pipeline.rep_count,
                "fps": round(pipeline.fps, 1),
                "classifier": pipeline.classifier_type,
                "timing": {"total_ms": round(t_total * 1000, 1)},
                "client_timestamp": data.get("timestamp", 0),
            }
        else:
            feedback = {
                "form_score": 0, "is_active": False, "rep_count": 0,
                "joint_feedback": {}, "details": [],
                "timing": {"total_ms": round(t_total * 1000, 1)},
                "client_timestamp": data.get("timestamp", 0),
            }

        # Push-up specific: send set/phase info for on-video HUD
        _add_detector_fields(feedback, pipeline, result)

        emit("result", feedback)

        rep_info = pipeline.last_rep_completed
        if rep_info:
            emit("rep_completed", rep_info)

    except Exception as e:
        logger.error("Landmarks processing error for %s: %s", sid, e)
    finally:
        processing[sid] = False


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ExerVision web server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)), help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger.info("Starting ExerVision server at http://%s:%s", args.host, args.port)
    logger.info("Open in browser: http://localhost:%s", args.port)
    socketio.run(app, host=args.host, port=args.port, debug=args.debug, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
