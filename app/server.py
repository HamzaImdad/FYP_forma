"""
Flask + Socket.IO server for FORMA web app.

Receives video frames from the browser via WebSocket,
processes them through the pipeline, and returns annotated frames + feedback.
Supports session lifecycle, classifier switching, and exercise guidance.

Usage:
    python app/server.py
    python app/server.py --port 5000 --host 0.0.0.0
"""

import os
import re
import sys
import json
import time
import base64
import logging
import argparse
from pathlib import Path


import cv2
import numpy as np
from dotenv import load_dotenv

# Load .env BEFORE importing anything that reads FORMA_JWT_SECRET.
load_dotenv(Path(__file__).parent.parent / ".env")

from flask import Flask, render_template, request, jsonify, send_from_directory, abort, g, make_response
from flask_socketio import SocketIO, emit, disconnect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

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

# React build output (Vite → app/static/dist/)
REACT_DIST_DIR = Path(__file__).parent / "static" / "dist"
REACT_INDEX = REACT_DIST_DIR / "index.html"

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("EXERVISION_SECRET_KEY", "exervision-dev-fallback")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Rate limiter — in-memory storage is fine for single-instance local dev.
limiter = Limiter(get_remote_address, app=app, default_limits=[])

# FORMA auth primitives (depends on FORMA_JWT_SECRET already being loaded)
from app.auth import (
    JWT_COOKIE,
    clear_auth_cookie,
    create_jwt,
    current_user_id,
    decode_jwt,
    hash_password,
    require_auth,
    set_auth_cookie,
    verify_password,
)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# One pipeline per connected session
pipelines = {}
# Parallel per-sid → user_id map (set on connect from JWT cookie)
user_ids: dict = {}
# Per-session processing lock to drop frames when busy
processing = {}
# Per-session last frame timestamp for server-side rate limiting (max 15 fps)
last_frame_time = {}
MIN_FRAME_INTERVAL = 1.0 / 20  # ~50ms
# Per-session last voice text — used to emit voice_text only when it changes
last_voice_text: dict = {}

# Initialize session history database
from app.database import ExerVisionDB
db = ExerVisionDB(str(PROJECT_ROOT / "exervision.db"))

# Session capture for offline analysis (always-on during tuning)
from app.session_capture import SessionCapture, build_frame_trace
captures: dict = {}  # sid -> SessionCapture instance

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

# Client-side routes the React router owns — Flask serves the SPA shell for each.
# Keep in sync with app/web/src/App.tsx routes.
SPA_ROUTES = {
    "",
    "exercises",
    "guide",
    "session",
    "report",
    "dashboard",
    "about",
    "voice-coaching",
    "chatbot",
    "plans",
    "milestones",
    "workout",
    "_dev",
}


def _serve_react_shell():
    """Serve the built React SPA shell, falling back to the legacy template if
    the Vite build output is missing (first run, clean checkouts)."""
    if REACT_INDEX.exists():
        return send_from_directory(REACT_DIST_DIR, "index.html")
    return render_template("index.html", exercises=EXERCISES)


@app.route("/")
def index():
    return _serve_react_shell()


@app.route("/legacy")
def legacy_index():
    """Old vanilla-JS UI, kept during the React migration (phases 1–5)."""
    return render_template("index.html", exercises=EXERCISES)


@app.route("/assets/<path:filename>")
def react_assets(filename):
    """Serve hashed JS/CSS/font assets emitted by `vite build`."""
    assets_dir = REACT_DIST_DIR / "assets"
    if not assets_dir.exists():
        abort(404)
    return send_from_directory(assets_dir, filename)


@app.route("/<path:spa_path>")
def spa_catchall(spa_path):
    """Client-side routes owned by React Router — serve the SPA shell.
    API and socket.io paths are handled by their own routes above."""
    first = spa_path.split("/", 1)[0]
    if first in SPA_ROUTES:
        return _serve_react_shell()
    abort(404)


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
    """Add detector-specific fields to the response.

    Unified path: PushUpDetector and RobustExerciseDetector now expose the
    same @property interface (set_count, reps_in_current_set, session_state,
    current_posture, last_set_reps), so every exercise emits the same rich
    session fields. Per-exercise branching is limited to the phase/progress
    label and plank-specific hold duration.
    """
    detector = pipeline._detectors.get(pipeline.exercise)
    if not detector:
        return

    # ── Shared rep/set/session fields (all exercises) ───────────────────
    feedback["set_count"] = getattr(detector, "set_count", 0)
    feedback["reps_in_set"] = getattr(detector, "reps_in_current_set", 0)

    # session_state / posture_label / last_set_reps — present on PushUp and
    # RobustExerciseDetector; fall back gracefully if a future detector
    # doesn't expose them.
    session_state_val = None
    ss = getattr(detector, "session_state", None)
    if ss is not None:
        session_state_val = ss.value if hasattr(ss, "value") else str(ss)
        feedback["session_state"] = session_state_val

    posture = getattr(detector, "current_posture", None)
    if posture is not None:
        feedback["posture_label"] = posture.value if hasattr(posture, "value") else str(posture)

    last_set_reps = getattr(detector, "last_set_reps", 0)
    feedback["last_set_reps"] = last_set_reps

    # Display fields — HUD reads these directly so it never reconciles
    # session state client-side.
    if session_state_val == "active":
        feedback["current_set_number"] = detector.set_count + 1
        feedback["current_set_reps"] = detector.reps_in_current_set
    elif session_state_val == "resting":
        feedback["current_set_number"] = max(detector.set_count, 1)
        feedback["current_set_reps"] = last_set_reps
    else:  # setup, idle, or unknown
        feedback["current_set_number"] = max(detector.set_count + 1, 1)
        feedback["current_set_reps"] = 0

    feedback["between_sets"] = (
        session_state_val == "resting"
        if session_state_val is not None
        else getattr(detector, "_between_sets", False)
    )

    # ── Phase / progress labels ─────────────────────────────────────────
    if pipeline.exercise == "pushup":
        phase_names = {
            "up": "Top position",
            "going_down": "Lowering",
            "down": "Bottom position",
            "going_up": "Pushing up",
        }
        feedback["phase"] = phase_names.get(detector._state, "")
        elbow = detector.last_elbow_angle
        if elbow is not None:
            progress_pct = max(0, min(100, int((150 - elbow) / (150 - 105) * 100)))
            feedback["progress"] = f"{progress_pct}%"
        else:
            feedback["progress"] = ""
    else:
        # RobustExerciseDetector-based detectors stash phase/progress in
        # result.details — surface them verbatim.
        if result and isinstance(result.details, dict):
            feedback["phase"] = result.details.get("phase", "")
            feedback["progress"] = result.details.get("progress", "")
            feedback["rest_tier"] = result.details.get("rest_tier", "")
            feedback["activity_state"] = result.details.get("activity", "")
        else:
            feedback["phase"] = ""
            feedback["progress"] = ""

    # ── Plank-specific: time-held fields ────────────────────────────────
    if hasattr(detector, "_hold_duration"):
        # Total hold across the entire session (accumulates across rests).
        feedback["total_hold_duration"] = round(detector._hold_duration, 1)
        # Current attempt's hold (resets to 0 after each set close).
        feedback["current_set_hold"] = round(
            getattr(detector, "current_set_hold", detector._hold_duration), 1
        )
        # Just-closed attempt's hold (0 before any set closes).
        feedback["last_set_hold"] = round(getattr(detector, "last_set_hold", 0.0), 1)
        # Legacy name kept for existing consumers.
        feedback["hold_duration"] = feedback["current_set_hold"]


# ── Auth API ───────────────────────────────────────────────────────────

def _public_user(row: dict) -> dict:
    """Strip sensitive columns before returning a user row to the client."""
    return {k: v for k, v in row.items() if k != "password_hash"}


@app.route("/api/auth/signup", methods=["POST"])
@limiter.limit("5/minute")
def auth_signup():
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""
    display_name = (body.get("display_name") or "").strip()

    if not EMAIL_RE.match(email):
        return jsonify({"error": "invalid_email"}), 400
    if len(password) < 8:
        return jsonify({"error": "password_too_short"}), 400
    if not display_name:
        return jsonify({"error": "display_name_required"}), 400

    if db.get_user_by_email(email) is not None:
        return jsonify({"error": "email_exists"}), 409

    user_id = db.create_user(email, hash_password(password), display_name)

    # Claim every orphan session so the user's existing training history
    # shows up in the dashboard from day one.
    try:
        claimed = db.reassign_orphan_sessions(user_id)
        if claimed:
            logger.info("Signup reassigned %d orphan sessions → user %d", claimed, user_id)
    except Exception as e:
        logger.warning("Failed to reassign orphan sessions: %s", e)

    user = db.get_user_by_id(user_id)
    resp = make_response(jsonify({"user": _public_user(user)}))
    set_auth_cookie(resp, create_jwt(user_id))
    return resp


@app.route("/api/auth/login", methods=["POST"])
@limiter.limit("10/minute")
def auth_login():
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""

    user = db.get_user_by_email(email)
    if not user or not verify_password(password, user["password_hash"]):
        return jsonify({"error": "invalid_credentials"}), 401

    db.update_last_login(user["id"])
    user = db.get_user_by_id(user["id"])
    resp = make_response(jsonify({"user": _public_user(user)}))
    set_auth_cookie(resp, create_jwt(user["id"]))
    return resp


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    resp = make_response(jsonify({"ok": True}))
    clear_auth_cookie(resp)
    return resp


@app.route("/api/auth/me")
@require_auth
def auth_me():
    user = db.get_user_by_id(g.user_id)
    if not user:
        return jsonify({"error": "unauthorized"}), 401
    return jsonify({"user": _public_user(user)})


@app.route("/api/auth/profile", methods=["PATCH"])
@require_auth
def auth_profile():
    body = request.get_json(silent=True) or {}
    db.update_user_profile(g.user_id, body)
    user = db.get_user_by_id(g.user_id)
    return jsonify({"user": _public_user(user)})


# ── Dashboard API ──────────────────────────────────────────────────────

@app.route("/api/dashboard/stats")
@require_auth
def dashboard_stats():
    """Aggregate stats for dashboard summary cards."""
    period = request.args.get("period")
    exercise = request.args.get("exercise")
    return jsonify(db.get_stats(g.user_id, period, exercise))


@app.route("/api/dashboard/sessions")
@require_auth
def dashboard_sessions():
    """List sessions with optional filters."""
    exercise = request.args.get("exercise")
    period = request.args.get("period")
    limit = int(request.args.get("limit", 20))
    offset = int(request.args.get("offset", 0))
    return jsonify(db.get_sessions(g.user_id, exercise, period, limit, offset))


@app.route("/api/dashboard/session/<int:session_id>")
@require_auth
def dashboard_session_detail(session_id):
    """Full session detail with per-rep breakdown."""
    detail = db.get_session_detail(g.user_id, session_id)
    if not detail:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(detail)


@app.route("/api/dashboard/scores")
@require_auth
def dashboard_scores():
    """Daily average form scores for line chart."""
    days = int(request.args.get("days", 30))
    exercise = request.args.get("exercise")
    return jsonify(db.get_daily_scores(g.user_id, days, exercise))


@app.route("/api/dashboard/distribution")
@require_auth
def dashboard_distribution():
    """Exercise breakdown for pie/doughnut chart."""
    period = request.args.get("period")
    return jsonify(db.get_exercise_distribution(g.user_id, period))


# ── Socket.IO Events ────────────────────────────────────────────────────

@socketio.on("connect")
def handle_connect():
    sid = request.sid
    # Enforce JWT cookie auth on every Socket.IO session.
    token = request.cookies.get(JWT_COOKIE)
    uid = decode_jwt(token)
    if uid is None:
        logger.info("Rejected unauthenticated Socket.IO connect: %s", sid)
        return False  # reject the handshake
    user_ids[sid] = uid
    pipelines[sid] = ExerVisionPipeline(
        exercise="squat", classifier_type="rule_based", config=REALTIME_CONFIG,
        preloaded_classifier=_preloaded_ml,
    )
    processing[sid] = False
    last_frame_time[sid] = 0
    logger.info("Client connected: %s (user %d)", sid, uid)
    emit("connected", {"status": "ok", "exercises": EXERCISES})


@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    # Finalize any in-flight capture so we don't leak the trace file
    capture = captures.pop(sid, None)
    if capture is not None:
        try:
            capture.end({"note": "session_aborted_on_disconnect"})
        except Exception as e:
            logger.debug("Capture end on disconnect failed: %s", e)
    if sid in pipelines:
        try:
            pipelines[sid].close()
        except Exception as e:
            logger.error("Failed to close pipeline for %s: %s", sid, e)
        del pipelines[sid]
    processing.pop(sid, None)
    last_frame_time.pop(sid, None)
    last_voice_text.pop(sid, None)
    user_ids.pop(sid, None)
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
    weight_kg = None
    if data:
        exercise = data.get("exercise")
        if exercise and exercise in EXERCISES:
            pipeline.exercise = exercise

        classifier = data.get("classifier")
        if classifier in ("rule_based", "ml", "bilstm"):
            pipeline.set_classifier(classifier)

        # Weight logging for weight-based exercises (deadlift, squat, bench, etc.)
        raw_weight = data.get("weight_kg")
        if raw_weight is not None:
            try:
                weight_kg = float(raw_weight)
                if weight_kg <= 0:
                    weight_kg = None
            except (TypeError, ValueError):
                weight_kg = None

    pipeline.start_session(weight_kg=weight_kg)

    # Start offline session capture (trace.jsonl + thresholds snapshot)
    try:
        detector = pipeline._detectors.get(pipeline.exercise)
        capture = SessionCapture(
            project_root=PROJECT_ROOT,
            exercise=pipeline.exercise,
            classifier=pipeline.classifier_type,
            detector=detector,
        )
        capture.start()
        captures[sid] = capture
    except Exception as e:
        logger.warning("Failed to start session capture for %s: %s", sid, e)

    emit("session_started", {
        "exercise": pipeline.exercise,
        "classifier": pipeline.classifier_type,
        "capture_id": captures[sid].session_id if sid in captures else None,
        "weight_kg": weight_kg,
    })

    # Push today's running totals for this exercise so the UI can show
    # "Today · N sets · M reps" before the user does anything new.
    try:
        totals = db.get_today_totals(user_ids[sid], pipeline.exercise)
        emit("today_totals", totals)
    except Exception as e:
        logger.warning("Failed to load today's totals: %s", e)


@socketio.on("end_session")
def handle_end_session():
    """End session, save to database, and return the summary report."""
    sid = request.sid
    if sid not in pipelines:
        return

    summary = pipelines[sid].end_session()

    # Auto-save session to database (scoped to the authed user)
    uid = user_ids.get(sid)
    try:
        if uid is None:
            raise RuntimeError("socket session has no user_id")
        session_id = db.save_session(summary, uid)
        summary["session_id"] = session_id
    except Exception as e:
        logger.warning(f"Failed to save session: {e}")

    # Finalize offline capture (write summary.json + metadata.json + close trace)
    capture = captures.pop(sid, None)
    if capture is not None:
        try:
            capture.end(summary)
            summary["capture_id"] = capture.session_id
            summary["capture_dir"] = str(capture.session_dir.relative_to(PROJECT_ROOT))
        except Exception as e:
            logger.warning("Failed to finalize session capture: %s", e)

    emit("session_report", summary)

    # Re-query today's totals so the UI can show the updated "today so far"
    # immediately after the session ends.
    try:
        if uid is not None:
            totals = db.get_today_totals(uid, pipelines[sid].exercise)
            emit("today_totals", totals)
    except Exception as e:
        logger.warning("Failed to refresh today's totals: %s", e)


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
        feedback["details"] = [v for k, v in result.details.items() if k != "voice"] if result.details else []
        feedback["form_score"] = round(result.form_score, 2)
        feedback["is_active"] = result.is_active
        feedback["joint_feedback"] = result.joint_feedback
        feedback["dtw_similarity"] = round(result.dtw_similarity, 2)
        feedback["dtw_worst_joint"] = result.dtw_worst_joint
    else:
        feedback["is_correct"] = None
        feedback["confidence"] = 0
        feedback["details"] = []
        feedback["form_score"] = 0
        feedback["is_active"] = False
        feedback["joint_feedback"] = {}
        feedback["dtw_similarity"] = 1.0
        feedback["dtw_worst_joint"] = None

    # Push-up specific: send set/phase info for on-video HUD
    _add_detector_fields(feedback, pipeline, result)

    # Voice coaching: emit only when text changes (severe-form triggers)
    if pipeline.exercise == "pushup" and result is not None:
        details_dict = result.details if isinstance(result.details, dict) else {}
        raw_voice = details_dict.get("voice", "") or ""
        prev = last_voice_text.get(sid, "")
        feedback["voice_text"] = raw_voice if raw_voice and raw_voice != prev else ""
        if raw_voice:
            last_voice_text[sid] = raw_voice
    else:
        feedback["voice_text"] = ""

    # Write per-frame trace to disk if session capture is active
    capture = captures.get(sid)
    if capture is not None:
        try:
            detector = pipeline._detectors.get(pipeline.exercise)
            trace = build_frame_trace(pipeline, result, detector)
            capture.write_frame(trace)
        except Exception as e:
            logger.debug("Trace write failed: %s", e)

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
            details_dict = result.details if isinstance(result.details, dict) else {}
            feedback = {
                "is_correct": result.is_correct,
                "confidence": round(result.confidence, 2) if result.confidence is not None else 0,
                "details": [v for k, v in details_dict.items() if k != "voice"],
                "form_score": round(result.form_score, 2),
                "is_active": result.is_active,
                "joint_feedback": result.joint_feedback,
                "dtw_similarity": round(result.dtw_similarity, 2),
                "dtw_worst_joint": result.dtw_worst_joint,
                "rep_count": pipeline.rep_count,
                "fps": round(pipeline.fps, 1),
                "classifier": pipeline.classifier_type,
                "timing": {"total_ms": round(t_total * 1000, 1)},
                "client_timestamp": data.get("timestamp", 0),
            }
        else:
            details_dict = {}
            feedback = {
                "is_correct": None,
                "confidence": 0,
                "details": [],
                "form_score": 0,
                "is_active": False,
                "joint_feedback": {},
                "dtw_similarity": 1.0,
                "dtw_worst_joint": None,
                "rep_count": 0,
                "fps": round(pipeline.fps, 1),
                "classifier": pipeline.classifier_type,
                "timing": {"total_ms": round(t_total * 1000, 1)},
                "client_timestamp": data.get("timestamp", 0),
            }

        # Push-up specific: send set/phase info for on-video HUD
        _add_detector_fields(feedback, pipeline, result)

        # Voice coaching: emit only when text changes (severe-form triggers)
        if pipeline.exercise == "pushup" and result is not None:
            raw_voice = details_dict.get("voice", "") or ""
            prev = last_voice_text.get(sid, "")
            feedback["voice_text"] = raw_voice if raw_voice and raw_voice != prev else ""
            if raw_voice:
                last_voice_text[sid] = raw_voice
        else:
            feedback["voice_text"] = ""

        # Write per-frame trace to disk if session capture is active
        capture = captures.get(sid)
        if capture is not None:
            try:
                detector = pipeline._detectors.get(pipeline.exercise)
                trace = build_frame_trace(pipeline, result, detector)
                capture.write_frame(trace)
            except Exception as te:
                logger.debug("Trace write failed: %s", te)

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
    parser = argparse.ArgumentParser(description="FORMA web server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)), help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger.info("Starting FORMA server at http://%s:%s", args.host, args.port)
    logger.info("Open in browser: http://localhost:%s", args.port)
    socketio.run(app, host=args.host, port=args.port, debug=args.debug, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
