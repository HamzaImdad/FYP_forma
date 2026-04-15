"""
Dedicated push-up detector.

Two-layer state model:
  - Outer session FSM (IDLE / SETUP / ACTIVE / RESTING) driven PURELY by the
    smoothed PostureClassifier label. No timers — the posture classifier's
    5-frame hysteresis is the single source of noise tolerance.
  - Inner rep FSM (UP / GOING_DOWN / DOWN / GOING_UP) driven by elbow angle.

Transition rules (posture-driven, zero timers):
  * IDLE     → SETUP on first non-UNKNOWN posture (or straight to ACTIVE
               if user is already in PLANK on the first stable frame)
  * SETUP    → ACTIVE on PLANK
  * ACTIVE   → RESTING on any non-PLANK labeled posture (STANDING, SITTING,
               KNEELING, LYING). UNKNOWN holds the state — we never commit
               a transition on UNKNOWN.
  * RESTING  → ACTIVE on PLANK
  * (false start: ACTIVE → SETUP if 0 reps were counted in the current set)

Rep counting only runs when session_state == ACTIVE. The session never
auto-ends — it stays in RESTING indefinitely until the user explicitly
restarts or pose is lost for 30+ frames.

Angles tracked:
  - Elbow (shoulder-elbow-wrist): rep counting + depth check
  - Hip (shoulder-hip-ankle): body alignment / hip sag
  - Shoulder (elbow-shoulder-hip): arm position relative to torso
"""

import json
import os
import time
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..classification.base import ClassificationResult
from ..classification.posture_classifier import PostureClassifier, PostureLabel
from ..coaching.engine import CoachingEngine
from ..pose_estimation.base import PoseResult
from ..utils.constants import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    MIN_VISIBILITY,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from ..utils.geometry import calculate_angle
from .base_detector import _load_muscles_for  # Session-1 muscle group lookup


# ── Rep FSM thresholds ───────────────────────────────────────────────────

ELBOW_UP = 150          # arms extended (top)
ELBOW_DOWN = 105        # full depth (bottom)
ELBOW_HALF_DOWN = 120   # partial depth

HIP_GOOD = 170
HIP_NUDGE = 160
HIP_WARNING = 150
HIP_MODERATE = 140
HIP_BAD = 125

SHOULDER_IDEAL_MIN = 20
SHOULDER_IDEAL_MAX = 45
SHOULDER_OK_MAX = 60
SHOULDER_WARN_MAX = 75

ASYMMETRY_WARN_DEG = 15.0
PIKE_THRESHOLD_RATIO = 0.05

MIN_BOTTOM_FRAMES = 1
TEMPO_WARN_DURATION = 1.5

ANGLE_SMOOTH_WINDOW = 3
NO_POSE_RESET_FRAMES = 30
MAX_REP_HISTORY = 200

# Rep-counting thresholds — operate on RAW elbow, not smoothed. Distinct
# from ELBOW_UP/ELBOW_DOWN (which remain the "ideal ROM" form-scoring
# reference). Looser bounds tolerate raw-signal noise and the 10-12 fps
# effective frame rate without requiring exact threshold-crossing frames.
REP_COMPLETE_UP = 140    # raw elbow at/above this → considered "up"
REP_START_DOWN = 135     # raw elbow below this → committed descent (UP → GOING_DOWN)
REP_DEPTH_MAX = 115      # min raw elbow during rep must reach this → valid depth
REP_BOTTOM_CLEAR = 120   # raw elbow above this while at bottom → ascending started

# Single-frame None tolerance for _avg_elbow_angle. During fast movement
# a single frame of low-visibility is common; freezing the rep FSM on
# every such frame loses reps. Reuse the last raw elbow for up to
# MAX_NONE_TOLERANCE consecutive frames before we actually give up.
MAX_NONE_TOLERANCE = 5

# Sanity floor on the raw elbow reading. A real human elbow can't hyperflex
# much below ~40° — if MediaPipe reports 0–35° it's almost always because
# landmark visibility passed but the shoulder/elbow/wrist points collapsed
# onto each other (hallucinated landmarks). Treat these as None so the
# single-frame tolerance path handles them instead of the rep FSM accepting
# them as "deep bend" and committing phantom reps.
MIN_REAL_ELBOW_DEG = 40.0
# Max biomechanically plausible inter-frame jump on the raw elbow signal.
# A real push-up moves the elbow at most ~90°/0.1s ≈ 900°/s; at 12–15 fps
# that's ~60–75° per frame. Anything above this is signal noise from
# landmark flicker and should not be fed to the rep FSM as a state-changing
# reading. Reuse the previous value instead.
MAX_ELBOW_JUMP_DEG = 80.0

# Trailing buffer size for pre-ACTIVE elbow history — lets us retroactively
# seed the rep FSM if the user started descending during the PostureClassifier
# bootstrap (5 UNKNOWN frames) and the first descent would otherwise be
# invisible to rep counting.
PRE_ACTIVE_TRACE_SIZE = 15

# Rep-FSM per-frame trace (debug). Writes one JSON line per frame to
# data/sessions/_rep_trace_debug.jsonl. Overwritten on each detector
# reset() so the file always reflects the most recent session.
# Turn off by setting env PUSHUP_REP_TRACE=0.
_TRACE_ENABLED = os.environ.get("PUSHUP_REP_TRACE", "1") != "0"
_TRACE_PATH = (
    Path(__file__).resolve().parents[2]
    / "data" / "sessions" / "_rep_trace_debug.jsonl"
)

# Session FSM grace: how many consecutive UNKNOWN frames in ACTIVE before
# we commit the set-close. Explicit non-plank labels (STANDING/SITTING/
# KNEELING/LYING) still close instantly — this grace ONLY applies to
# UNKNOWN, which is the classifier's "I don't know" signal and is usually
# a transient glitch (camera jitter, partial occlusion during a rep).
UNKNOWN_GRACE_FRAMES = 20

# ── Session FSM ──────────────────────────────────────────────────────────
# No timers. The PostureClassifier has built-in 5-frame hysteresis
# (~0.33s at 15fps) so single-frame noise can't flip states. Anything
# slower than that would be a UX delay, not safety.

class SessionState(str, Enum):
    IDLE = "idle"
    SETUP = "setup"
    ACTIVE = "active"
    RESTING = "resting"


class PushUpState:
    """Inner rep phase."""
    UP = "up"
    GOING_DOWN = "going_down"
    DOWN = "down"
    GOING_UP = "going_up"


class RepQuality:
    GOOD = "good"
    MODERATE = "moderate"
    BAD = "bad"


class PushUpDetector:
    """Self-contained push-up detection with posture-driven session FSM."""

    EXERCISE_NAME = "pushup"
    IS_STATIC = False

    def __init__(self, dtw_matcher=None, **kwargs):
        # dtw_matcher kept for API parity with BaseExerciseDetector subclasses
        self._dtw_matcher = dtw_matcher
        self._posture_classifier = PostureClassifier()
        self._coach = CoachingEngine("pushup")
        self._muscle_groups: List[str] = _load_muscles_for(self.EXERCISE_NAME)
        self.reset()

    def reset(self):
        # Inner rep FSM
        self._state = PushUpState.UP
        self._rep_count = 0
        self._half_rep = 0.0
        self._last_rep_time = 0.0
        self._rep_start_time = 0.0
        self._bottom_frame_count = 0
        self._first_rep_started = False

        # Extremum tracking — min raw elbow seen during the current in-progress
        # rep. Reset on UP → GOING_DOWN transition and after rep commit.
        self._rep_min_elbow: float = 180.0

        # Last valid raw elbow + consecutive-None counter for brief
        # visibility dips (motion blur during fast movement).
        self._last_raw_elbow: Optional[float] = None
        self._consec_none_frames: int = 0

        # Pre-ACTIVE elbow trace — records raw elbow during IDLE/SETUP so the
        # first rep can be recovered if the user started descending during
        # the posture classifier's bootstrap window.
        self._pre_active_elbow_trace: deque = deque(maxlen=PRE_ACTIVE_TRACE_SIZE)

        # Set tracking
        self._set_count = 0
        self._reps_in_current_set = 0
        self._set_reps: List[int] = []
        # Session-1: parallel per-set metadata
        self._set_records: List[Dict] = []
        self._last_set_end_time: float = 0.0
        self._recent_rep_durations: deque = deque(maxlen=3)

        # Session-1: rep tempo timestamps. descent = eccentric start;
        # concentric = start of the up phase (stamped at DOWN/GOING_UP).
        self._descent_start_time: float = 0.0
        self._concentric_start_time: float = 0.0

        # Per-rep form tracking
        self._current_rep_issues: List[str] = []
        self._current_rep_scores: List[float] = []
        self._rep_history: List[Dict] = []

        # Angle smoothing buffers
        self._elbow_buf: deque = deque(maxlen=ANGLE_SMOOTH_WINDOW)
        self._hip_buf: deque = deque(maxlen=ANGLE_SMOOTH_WINDOW)
        self._shoulder_buf: deque = deque(maxlen=ANGLE_SMOOTH_WINDOW)

        # Form score smoothing
        self._score_buf: deque = deque(maxlen=8)

        # Pose presence tracking
        self._pose_detected = False
        self._frames_since_pose = 0

        # Session FSM state — posture-driven, no timers
        self._session_state: SessionState = SessionState.IDLE
        self._current_posture: PostureLabel = PostureLabel.UNKNOWN
        self._unknown_streak: int = 0   # consecutive UNKNOWN frames in ACTIVE
        self._posture_classifier.reset()
        if hasattr(self, "_coach"):
            self._coach.reset()

        # Last computed angles (for UI display)
        self.last_elbow_angle = None
        self.last_hip_angle = None
        self.last_shoulder_angle = None

        # Last frame check tags (used internally by _assess_form)
        self._frame_checks: List[str] = []

        # Rep-FSM debug trace — truncate on every reset so a session
        # always starts with a clean file.
        self._trace_fh = None
        self._trace_start = time.time()
        if _TRACE_ENABLED:
            try:
                _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
                self._trace_fh = _TRACE_PATH.open("w", buffering=1)
            except Exception:
                self._trace_fh = None

    # ── Public properties ────────────────────────────────────────────────

    @property
    def rep_count(self) -> int:
        return self._rep_count

    @property
    def set_count(self) -> int:
        return self._set_count

    @property
    def reps_in_current_set(self) -> int:
        return self._reps_in_current_set

    @property
    def session_state(self) -> SessionState:
        return self._session_state

    @property
    def current_posture(self) -> PostureLabel:
        return self._current_posture

    @property
    def last_set_reps(self) -> int:
        return self._set_reps[-1] if self._set_reps else 0

    # ── Main entry point ─────────────────────────────────────────────────

    def classify(self, pose: PoseResult) -> ClassificationResult:
        """Process a pose and return ClassificationResult."""
        now = time.time()

        # Compute angles
        elbow_angle = self._avg_elbow_angle(pose)
        hip_angle = self._compute_hip_angle(pose)
        shoulder_angle = self._compute_shoulder_angle(pose)

        # Sanity guards on the raw elbow reading — MediaPipe occasionally
        # passes visibility checks while the shoulder/elbow/wrist landmarks
        # collapse onto each other (angle ≈ 0°) or flicker by huge amounts
        # between frames. Treat both as "no signal" so the None-tolerance
        # path handles them instead of the rep FSM accepting a 0° reading
        # as a deep-bend commit.
        if elbow_angle is not None and elbow_angle < MIN_REAL_ELBOW_DEG:
            elbow_angle = None
        if (elbow_angle is not None
                and self._last_raw_elbow is not None
                and abs(elbow_angle - self._last_raw_elbow) > MAX_ELBOW_JUMP_DEG):
            elbow_angle = None

        self.last_elbow_angle = elbow_angle
        self.last_hip_angle = hip_angle
        self.last_shoulder_angle = shoulder_angle

        # Single-frame None tolerance: if elbow visibility dipped for just
        # this frame, reuse the last known raw elbow so the rep FSM can
        # keep running. Only give up once _consec_none_frames exceeds the
        # tolerance — then fall back to the pose-loss path below.
        fsm_raw_elbow: Optional[float] = elbow_angle
        if elbow_angle is None:
            self._consec_none_frames += 1
            if (self._consec_none_frames <= MAX_NONE_TOLERANCE
                    and self._last_raw_elbow is not None):
                fsm_raw_elbow = self._last_raw_elbow
        else:
            self._consec_none_frames = 0
            self._last_raw_elbow = elbow_angle

        if elbow_angle is not None:
            self._elbow_buf.append(elbow_angle)
        if hip_angle is not None:
            self._hip_buf.append(hip_angle)
        if shoulder_angle is not None:
            self._shoulder_buf.append(shoulder_angle)

        smooth_elbow = self._smooth(self._elbow_buf)
        smooth_hip = self._smooth(self._hip_buf)
        smooth_shoulder = self._smooth(self._shoulder_buf)

        # Posture classification (smoothed)
        posture_result = self._posture_classifier.update(pose)
        self._current_posture = posture_result.label

        # Pre-ACTIVE elbow trace: record (timestamp, raw_elbow, raw_is_plank)
        # while in IDLE/SETUP so a first rep started during the posture-
        # classifier bootstrap window can still be recovered when ACTIVE
        # fires — but ONLY if the user was actually in plank geometry on
        # those frames (raw posture label, bypassing the 5-frame hysteresis).
        # Without the raw_is_plank flag, the seed logic would happily read
        # elbow movements from standing/gesturing users and emit false reps.
        if (self._session_state in (SessionState.IDLE, SessionState.SETUP)
                and fsm_raw_elbow is not None):
            raw_is_plank = (
                getattr(self._posture_classifier, "_last_raw_label", None)
                == PostureLabel.PLANK
            )
            self._pre_active_elbow_trace.append((now, fsm_raw_elbow, raw_is_plank))

        # Pose-presence tracking — extended pose loss counts as REST when
        # the user was actively working, not a full reset. Preserves set
        # history so the user can walk off-camera briefly without losing
        # their session.
        #
        # Trigger the pose-loss path only when elbow+hip have been absent
        # for longer than the single-frame tolerance handled above.
        if smooth_elbow is None or smooth_hip is None:
            self._frames_since_pose += 1
            if self._frames_since_pose >= NO_POSE_RESET_FRAMES:
                self._handle_pose_loss()
                self._elbow_buf.clear()
                self._hip_buf.clear()
                self._shoulder_buf.clear()
                self._last_raw_elbow = None
                self._consec_none_frames = 0
                self._pre_active_elbow_trace.clear()
            return self._build_result(
                form_score=0.0,
                joint_feedback={},
                details=self._session_details(),
            )

        self._pose_detected = True
        self._frames_since_pose = 0

        # ── Session FSM ───────────────────────────────────────────────
        self._update_session_state(now)

        # ── Inner rep FSM (only when ACTIVE) ──────────────────────────
        # Feeds RAW elbow (not smoothed) — min/max tracking + lenient
        # thresholds handle noise without the smoothing-induced lag.
        rep_completed = False
        if self._session_state == SessionState.ACTIVE:
            rep_completed = self._update_rep_fsm(fsm_raw_elbow, now)

        # ── Rep-FSM debug trace (every frame) ──────────────────────────
        if self._trace_fh is not None:
            try:
                self._trace_fh.write(json.dumps({
                    "t": round(now - self._trace_start, 3),
                    "raw_elbow": None if elbow_angle is None else round(elbow_angle, 1),
                    "fsm_elbow": None if fsm_raw_elbow is None else round(fsm_raw_elbow, 1),
                    "smooth_elbow": None if smooth_elbow is None else round(smooth_elbow, 1),
                    "rep_min": round(self._rep_min_elbow, 1),
                    "consec_none": self._consec_none_frames,
                    "session_state": self._session_state.value,
                    "posture": self._current_posture.value if hasattr(self._current_posture, "value") else str(self._current_posture),
                    "rep_state": self._state,
                    "bottom_frames": self._bottom_frame_count,
                    "reps_in_set": self._reps_in_current_set,
                    "rep_count": self._rep_count,
                    "rep_completed": rep_completed,
                }) + "\n")
            except Exception:
                pass

        # ── Per-frame form assessment ────────────────────────────────
        frame_score, joint_feedback, issues = self._assess_form(
            smooth_elbow, smooth_hip, smooth_shoulder, pose
        )

        # Suppress issues outside ACTIVE — no point coaching a resting user
        if self._session_state != SessionState.ACTIVE:
            issues = []
            for k in joint_feedback:
                joint_feedback[k] = "correct"
            frame_score = max(frame_score, 0.8)
        else:
            self._current_rep_scores.append(frame_score)
            for issue in issues:
                if issue not in self._current_rep_issues:
                    self._current_rep_issues.append(issue)

        # ── Record rep on completion ─────────────────────────────────
        if rep_completed:
            self._record_rep(now)

        # ── Smooth form score ────────────────────────────────────────
        self._score_buf.append(frame_score)
        smooth_score = sum(self._score_buf) / len(self._score_buf)

        # ── Voice coaching (session-state edges only) ──
        voice_text = self._coach.frame_feedback(
            session_state=self._session_state.value,
        )

        details = self._session_details()
        details["voice"] = voice_text

        return self._build_result(
            form_score=smooth_score,
            joint_feedback=joint_feedback,
            details=details,
        )

    # ── Session FSM ──────────────────────────────────────────────────────

    def _update_session_state(self, now: float) -> None:
        """Drive IDLE / SETUP / ACTIVE / RESTING transitions from the
        smoothed posture label.

        Rule in ACTIVE:
          * PLANK                           → stay ACTIVE, reset grace
          * STANDING/SITTING/KNEELING/LYING → close set instantly
            (deliberate position change — user is clearly resting)
          * UNKNOWN                         → grace period; only after
            UNKNOWN_GRACE_FRAMES consecutive UNKNOWNs do we commit. This
            absorbs camera glitches and transient occlusion mid-rep so
            a one-second hiccup doesn't split the set.

        Rule elsewhere (IDLE/SETUP/RESTING): UNKNOWN holds the current
        state; any other label drives the usual transitions.
        """
        posture = self._current_posture
        in_plank = posture == PostureLabel.PLANK
        s = self._session_state

        if s == SessionState.ACTIVE:
            if in_plank:
                self._unknown_streak = 0
                return

            if posture == PostureLabel.UNKNOWN:
                # Transient glitch tolerance — don't split the set unless
                # UNKNOWN persists past the grace window.
                self._unknown_streak += 1
                if self._unknown_streak < UNKNOWN_GRACE_FRAMES:
                    return
                # Grace exhausted — fall through to set-close
            else:
                # Explicit non-plank label (STANDING / SITTING / KNEELING
                # / LYING) — close instantly, don't wait.
                self._unknown_streak = 0

            if self._reps_in_current_set > 0:
                self._set_reps.append(self._reps_in_current_set)
                self._close_current_set(now, failure_type="clean_stop")
                self._set_count += 1
                self._reps_in_current_set = 0
                self._reset_rep_fsm()
                self._session_state = SessionState.RESTING
            else:
                # False start — entered plank then left without a rep.
                # Roll back to SETUP so the next plank entry still
                # announces "form locked".
                self._reset_rep_fsm()
                self._session_state = SessionState.SETUP
            self._unknown_streak = 0
            return

        # Non-ACTIVE states: UNKNOWN is held, other labels are honoured.
        if posture == PostureLabel.UNKNOWN:
            return

        if s == SessionState.IDLE:
            # First stable posture — either jump straight to ACTIVE (user
            # is already in position) or park in SETUP waiting for them.
            if in_plank:
                self._session_state = SessionState.ACTIVE
                self._reset_rep_fsm()
                self._rep_start_time = now
                self._unknown_streak = 0
                self._seed_rep_fsm_from_pre_active(now)
            else:
                self._session_state = SessionState.SETUP

        elif s == SessionState.SETUP:
            if in_plank:
                self._session_state = SessionState.ACTIVE
                self._reset_rep_fsm()
                self._rep_start_time = now
                self._unknown_streak = 0
                self._seed_rep_fsm_from_pre_active(now)

        elif s == SessionState.RESTING:
            if in_plank:
                self._session_state = SessionState.ACTIVE
                self._reset_rep_fsm()
                self._rep_start_time = now
                self._unknown_streak = 0
                # Don't seed from the trailing buffer here — entering
                # RESTING cleared the posture semantics, and the buffer
                # may contain frames from the user walking back into
                # position. Safer to start fresh.
            # else: stay in RESTING — no auto-end, no timer

    def _reset_rep_fsm(self) -> None:
        """Reset the inner rep state machine. Drops any in-progress rep."""
        self._state = PushUpState.UP
        self._bottom_frame_count = 0
        self._rep_min_elbow = 180.0
        self._current_rep_issues = []
        self._current_rep_scores = []

    def _seed_rep_fsm_from_pre_active(self, now: float) -> None:
        """Retroactively initialise the rep FSM from the pre-ACTIVE elbow
        trace so a first rep that started during the posture-classifier
        bootstrap (5 UNKNOWN frames) is not lost.

        Safety rules — these exist to prevent false reps from leaking in
        when the user was standing/moving before entering plank:

          1. Only the trailing suffix of frames whose RAW posture was
             PLANK is considered. Any STANDING/UNKNOWN/etc. frame is a
             hard reset of the candidate window.
          2. At least 3 consecutive PLANK frames are required. A 1-2
             frame flicker (e.g. brief bend during gesturing) is too
             little signal to justify retroactive rep counting.
          3. Only frames from the trailing PLANK run contribute to the
             descent-start timestamp and minimum. Anything before the
             user was in plank is discarded.
        """
        if not self._pre_active_elbow_trace:
            return

        # Pull out the trailing run of frames with raw_is_plank == True.
        # Walk backwards so we build the contiguous suffix that ended on
        # the ACTIVE transition frame.
        plank_suffix: List[Tuple[float, float]] = []
        for t, elbow, is_plank in reversed(self._pre_active_elbow_trace):
            if not is_plank:
                break
            plank_suffix.append((t, elbow))
        plank_suffix.reverse()

        if len(plank_suffix) < 3:
            # Not enough plank geometry in the trailing window to trust
            # a seed. Better to lose the first rep than to false-count one.
            return

        descent_start_t: Optional[float] = None
        descent_min: float = 180.0
        last_elbow: Optional[float] = None
        for t, elbow in plank_suffix:
            last_elbow = elbow
            if elbow < REP_START_DOWN:
                if descent_start_t is None:
                    descent_start_t = t
                if elbow < descent_min:
                    descent_min = elbow

        # No descent visible inside the trailing plank run → stay in UP.
        if descent_start_t is None or last_elbow is None:
            return

        self._first_rep_started = True
        self._rep_start_time = descent_start_t
        self._rep_min_elbow = descent_min
        self._current_rep_issues = []
        self._current_rep_scores = []

        # Decide which state to land in based on where the last observed
        # raw elbow sits relative to the rep thresholds.
        depth_reached = descent_min <= REP_DEPTH_MAX
        if last_elbow <= REP_DEPTH_MAX:
            # Still at or below depth → DOWN.
            self._state = PushUpState.DOWN
            self._bottom_frame_count = 1
            self._half_rep = 0.5
        elif depth_reached:
            # Depth was already reached and user is above bottom → ascent.
            # Land in GOING_UP so rep commits cleanly at the top.
            self._state = PushUpState.GOING_UP
            self._half_rep = 0.5
        else:
            # Still descending, depth not yet reached.
            self._state = PushUpState.GOING_DOWN
            self._bottom_frame_count = 0

        self._pre_active_elbow_trace.clear()

    def _handle_pose_loss(self) -> None:
        """Called when the pose has been lost for NO_POSE_RESET_FRAMES
        consecutive frames. If the user was mid-set, close the set and
        move to RESTING so their rep count is preserved. Otherwise fall
        back to IDLE."""
        s = self._session_state
        if s == SessionState.ACTIVE and self._reps_in_current_set > 0:
            self._set_reps.append(self._reps_in_current_set)
            self._close_current_set(time.time(), failure_type="timeout")
            self._set_count += 1
            self._reps_in_current_set = 0
            self._reset_rep_fsm()
            self._session_state = SessionState.RESTING
        elif s == SessionState.RESTING:
            # already resting — nothing to close, stay put
            return
        else:
            # ACTIVE with 0 reps (false start), SETUP, or IDLE → full reset
            self._reset_rep_fsm()
            self._session_state = SessionState.IDLE

    # ── Inner rep FSM ────────────────────────────────────────────────────

    def _update_rep_fsm(self, raw_elbow: Optional[float], now: float) -> bool:
        """Run the elbow-angle rep state machine on RAW elbow (no rolling
        mean). Returns True if a rep completed this frame.

        Design: this FSM does not rely on the elbow crossing a specific
        threshold on a specific frame. Instead it tracks `_rep_min_elbow`
        (the minimum raw elbow seen since the descent began). A rep is
        committed when the user is back up (raw ≥ REP_COMPLETE_UP) AND
        the tracked minimum actually reached depth (≤ REP_DEPTH_MAX).

        This makes rep counting robust to:
          * dropped frames at the peak/trough (min latches the low water mark)
          * raw-signal noise (jitter of ±5° doesn't flip the decision)
          * MediaPipe VIDEO-mode's own temporal smoothing (no second
            rolling-mean lag)
          * the "didn't go deep enough" false-bail that occurred when the
            smoothed signal skipped the bottom frame entirely.
        """
        if raw_elbow is None:
            return False

        rep_completed = False

        # Track the minimum elbow angle seen since the descent started.
        # Only meaningful once we've left UP. This is the load-bearing
        # state: rep_min is the ONLY thing that proves depth was reached,
        # so we keep it up-to-date every frame regardless of which branch
        # of the state machine fires below.
        if self._state != PushUpState.UP:
            if raw_elbow < self._rep_min_elbow:
                self._rep_min_elbow = raw_elbow

        # ──────────────────────────────────────────────────────────────
        # PRIMARY COMMIT CHECK — state-independent, signal-driven.
        # This runs BEFORE the state-machine branches so that frame
        # drops, posture hiccups, or elbow-angle jitter that kept the
        # state stuck in GOING_DOWN can't swallow a rep. The ONLY
        # criteria for a rep are:
        #   (1) a descent was committed (_first_rep_started)
        #   (2) we're not currently back at the top (state != UP)
        #   (3) raw elbow has returned to the top (>= REP_COMPLETE_UP)
        #   (4) the tracked minimum elbow reached depth (<= REP_DEPTH_MAX)
        # Any path through the state machine that satisfies (2)–(4)
        # counts as a rep, full stop. This matches how a human counts
        # reps from a video: bottom touched → top reached → "one".
        # ──────────────────────────────────────────────────────────────
        if (self._first_rep_started
                and self._state != PushUpState.UP
                and raw_elbow >= REP_COMPLETE_UP
                and self._rep_min_elbow <= REP_DEPTH_MAX):
            elapsed = now - self._rep_start_time
            self._half_rep = 0.0
            rep_completed = True
            self._rep_count += 1
            self._reps_in_current_set += 1
            self._recent_rep_durations.append(elapsed)
            self._state = PushUpState.UP
            self._rep_min_elbow = 180.0
            return rep_completed

        # ──────────────────────────────────────────────────────────────
        # State machine (secondary — drives phase labels for UI, and
        # the "didn't go deep enough" bailout when the user aborts a
        # descent without reaching depth).
        # ──────────────────────────────────────────────────────────────
        if self._state == PushUpState.UP:
            # Committed descent once raw elbow crosses the start threshold.
            # REP_START_DOWN (135°) is below REP_COMPLETE_UP (140°) so a
            # brief bounce near the top doesn't ping-pong the state.
            if raw_elbow < REP_START_DOWN:
                self._state = PushUpState.GOING_DOWN
                self._rep_start_time = now
                self._descent_start_time = now           # eccentric begins
                self._concentric_start_time = 0.0        # reset — stamped at DOWN
                self._bottom_frame_count = 0
                self._rep_min_elbow = raw_elbow
                self._first_rep_started = True
                self._current_rep_issues = []
                self._current_rep_scores = []

        elif self._state == PushUpState.GOING_DOWN:
            if raw_elbow <= REP_DEPTH_MAX:
                self._state = PushUpState.DOWN
                self._concentric_start_time = now        # end of eccentric
                self._half_rep = 0.5
                self._bottom_frame_count = 0
            elif raw_elbow >= REP_COMPLETE_UP:
                # Elbow is back up without ever crossing REP_DEPTH_MAX
                # (and the primary commit check above didn't fire, so
                # rep_min never reached depth). Genuine bailout.
                if "Didn't go deep enough" not in self._current_rep_issues:
                    self._current_rep_issues.append("Didn't go deep enough")
                self._state = PushUpState.UP
                self._rep_min_elbow = 180.0

        elif self._state == PushUpState.DOWN:
            self._bottom_frame_count += 1
            if (raw_elbow > REP_BOTTOM_CLEAR
                    and self._bottom_frame_count >= MIN_BOTTOM_FRAMES):
                self._state = PushUpState.GOING_UP

        elif self._state == PushUpState.GOING_UP:
            if raw_elbow <= REP_DEPTH_MAX:
                # User dropped back into the bottom — second dip.
                self._state = PushUpState.DOWN
                self._concentric_start_time = now        # re-stamp on dip
                self._bottom_frame_count = 0

        return rep_completed

    # ── Session-1 capture helpers ────────────────────────────────────────

    def _augment_rep_entry(self, rep_entry: Dict, now: float) -> None:
        """Mutate rep_entry in place with Session-1 capture fields.
        Mirrors BaseExerciseDetector._augment_rep_entry so the push-up
        summary schema stays consistent with all other exercises."""
        scores = self._current_rep_scores or [0.0]
        # For push-up, depth is tracked as _rep_min_elbow (lower = deeper)
        peak = getattr(self, "_rep_min_elbow", None)
        rep_entry["peak_angle"] = round(float(peak), 1) if peak is not None else None
        rep_entry["score_min"] = round(float(min(scores)), 2)
        rep_entry["score_max"] = round(float(max(scores)), 2)
        ecc = 0.0
        con = 0.0
        if self._descent_start_time and self._concentric_start_time:
            ecc = max(0.0, self._concentric_start_time - self._descent_start_time)
        if self._concentric_start_time:
            con = max(0.0, now - self._concentric_start_time)
        rep_entry["ecc_sec"] = round(ecc, 2)
        rep_entry["con_sec"] = round(con, 2)
        rep_entry["set_num"] = self._set_count + 1

    def _close_current_set(self, now: float, failure_type: str = "clean_stop") -> None:
        """Append a per-set record based on the reps in the just-finished set."""
        reps_count = int(self._reps_in_current_set)
        if reps_count <= 0 and not self._rep_history:
            return

        tail = self._rep_history[-reps_count:] if reps_count > 0 else []
        scores = [float(r.get("score", 0.0)) for r in tail]

        if scores:
            avg = sum(scores) / len(scores)
            dropoff = scores[0] - scores[-1]
        else:
            avg = 0.0
            dropoff = 0.0

        ft = failure_type
        if ft == "clean_stop" and len(scores) >= 3:
            if scores[-3] > scores[-2] > scores[-1]:
                ft = "form_breakdown"

        rest_before = 0.0
        if self._last_set_end_time > 0.0:
            rest_before = min(600.0, max(0.0, now - self._last_set_end_time))

        record = {
            "set_num": int(self._set_count + 1),
            "reps_count": reps_count,
            "rest_before_sec": round(rest_before, 2),
            "avg_form_score": round(avg, 2),
            "score_dropoff": round(dropoff, 2),
            "failure_type": ft,
        }
        self._set_records.append(record)
        self._last_set_end_time = now

    def _record_rep(self, now: float) -> None:
        avg_score = sum(self._current_rep_scores) / max(
            len(self._current_rep_scores), 1
        )
        duration = now - self._rep_start_time

        if duration < TEMPO_WARN_DURATION:
            tempo_msg = "Slow down — full range of motion beats speed"
            if tempo_msg not in self._current_rep_issues:
                self._current_rep_issues.append(tempo_msg)

        # Lenient thresholds — the form-detection layer is noisy, so a
        # completed rep (which already cleared the bottom-dwell guard and
        # full ROM) gets credit unless the avg score is genuinely poor.
        quality = (
            RepQuality.GOOD if avg_score >= 0.5
            else RepQuality.MODERATE if avg_score >= 0.3
            else RepQuality.BAD
        )
        if len(self._rep_history) < MAX_REP_HISTORY:
            rep_entry = {
                "rep": self._rep_count,
                "score": round(avg_score, 2),
                "quality": quality,
                "issues": self._current_rep_issues[:3],
                "duration": round(duration, 1),
            }
            self._augment_rep_entry(rep_entry, now)
            self._rep_history.append(rep_entry)

    # ── Form assessment (unchanged scoring logic) ────────────────────────

    def _check_pike(self, pose: PoseResult) -> Optional[float]:
        """Detect hip pike normalized to torso length."""
        shoulder_pts = []
        hip_pts = []
        ankle_ys = []
        for s_idx, h_idx, a_idx in [
            (LEFT_SHOULDER, LEFT_HIP, LEFT_ANKLE),
            (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_ANKLE),
        ]:
            if pose.is_visible(s_idx) and pose.is_visible(h_idx) and pose.is_visible(a_idx):
                shoulder_pts.append(np.array(pose.get_world_landmark(s_idx)))
                hip_pts.append(np.array(pose.get_world_landmark(h_idx)))
                ankle_ys.append(pose.get_world_landmark(a_idx)[1])
        if not shoulder_pts:
            return None

        mid_shoulder = np.mean(shoulder_pts, axis=0)
        mid_hip = np.mean(hip_pts, axis=0)
        torso_length = float(np.linalg.norm(mid_shoulder - mid_hip))
        if torso_length < 0.1:
            return None

        shoulder_y = float(mid_shoulder[1])
        hip_y = float(mid_hip[1])
        mid_ankle_y = sum(ankle_ys) / len(ankle_ys)
        midline_y = (shoulder_y + mid_ankle_y) / 2

        return (hip_y - midline_y) / torso_length

    def _check_head_alignment(self, pose: PoseResult) -> Optional[str]:
        if not pose.is_visible(NOSE):
            return None

        nose_y = pose.get_world_landmark(NOSE)[1]
        shoulder_ys = []
        for s_idx in [LEFT_SHOULDER, RIGHT_SHOULDER]:
            if pose.is_visible(s_idx):
                shoulder_ys.append(pose.get_world_landmark(s_idx)[1])
        if not shoulder_ys:
            return None

        avg_shoulder_y = sum(shoulder_ys) / len(shoulder_ys)
        diff = nose_y - avg_shoulder_y

        if diff > 0.15:
            return "Head dropping -- keep neck neutral"
        if diff < -0.15:
            return "Don't crane neck up -- look at floor ahead of hands"
        return None

    def _assess_form(
        self,
        elbow: Optional[float],
        hip: Optional[float],
        shoulder: Optional[float],
        pose: Optional[PoseResult] = None,
    ) -> Tuple[float, Dict[str, str], List[str]]:
        joint_feedback = {}
        issues = []
        scores = {}
        self._frame_checks = []

        # Elbow
        if elbow is not None:
            if self._state in (PushUpState.DOWN, PushUpState.GOING_UP, PushUpState.GOING_DOWN):
                joint_feedback["left_elbow"] = "correct"
                joint_feedback["right_elbow"] = "correct"
                scores["elbow"] = 1.0 if elbow <= ELBOW_DOWN else 0.8
            else:
                if elbow >= ELBOW_UP:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 1.0
                else:
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    scores["elbow"] = 0.6

        # Hip / body alignment
        if hip is not None:
            if hip >= HIP_GOOD:
                joint_feedback["left_hip"] = "correct"
                joint_feedback["right_hip"] = "correct"
                scores["hip"] = 1.0
            elif hip >= HIP_NUDGE:
                joint_feedback["left_hip"] = "correct"
                joint_feedback["right_hip"] = "correct"
                scores["hip"] = 0.9
            elif hip >= HIP_WARNING:
                joint_feedback["left_hip"] = "warning"
                joint_feedback["right_hip"] = "warning"
                issues.append("Slight hip sag — squeeze glutes and brace core")
                self._frame_checks.append("hip_sag_mild")
                scores["hip"] = 0.6
            elif hip >= HIP_MODERATE:
                joint_feedback["left_hip"] = "warning"
                joint_feedback["right_hip"] = "warning"
                issues.append("Tighten your core — keep hips level")
                self._frame_checks.append("hip_sag_moderate")
                scores["hip"] = 0.4
            elif hip >= HIP_BAD:
                joint_feedback["left_hip"] = "incorrect"
                joint_feedback["right_hip"] = "incorrect"
                issues.append("Hips sagging — squeeze glutes, brace core")
                self._frame_checks.append("hip_sag_severe")
                scores["hip"] = 0.2
            else:
                joint_feedback["left_hip"] = "incorrect"
                joint_feedback["right_hip"] = "incorrect"
                issues.append("Core check — keep your body in a straight line")
                self._frame_checks.append("hip_sag_severe")
                scores["hip"] = 0.0

        # Shoulder / elbow flare
        if shoulder is not None:
            if SHOULDER_IDEAL_MIN <= shoulder <= SHOULDER_IDEAL_MAX:
                joint_feedback["left_shoulder"] = "correct"
                joint_feedback["right_shoulder"] = "correct"
                scores["shoulder"] = 1.0
            elif shoulder < SHOULDER_IDEAL_MIN:
                joint_feedback["left_shoulder"] = "warning"
                joint_feedback["right_shoulder"] = "warning"
                issues.append("Arms too tight to body — widen grip slightly")
                self._frame_checks.append("elbow_too_tucked")
                scores["shoulder"] = 0.5
            elif shoulder <= SHOULDER_OK_MAX:
                joint_feedback["left_shoulder"] = "correct"
                joint_feedback["right_shoulder"] = "correct"
                scores["shoulder"] = 0.75
            elif shoulder <= SHOULDER_WARN_MAX:
                joint_feedback["left_shoulder"] = "correct"
                joint_feedback["right_shoulder"] = "correct"
                scores["shoulder"] = 0.7
            else:
                joint_feedback["left_shoulder"] = "warning"
                joint_feedback["right_shoulder"] = "warning"
                issues.append("Elbows flaring out — tuck elbows closer to body")
                self._frame_checks.append("elbow_flare_severe")
                scores["shoulder"] = 0.3

        # Pike
        if pose is not None:
            pike = self._check_pike(pose)
            if pike is not None and pike < -PIKE_THRESHOLD_RATIO:
                issues.append("Hips piking up — lower hips into straight line")
                self._frame_checks.append("hip_pike")
                scores["pike"] = 0.3
                joint_feedback["left_hip"] = "incorrect"
                joint_feedback["right_hip"] = "incorrect"

        # Head/neck
        if pose is not None:
            head_issue = self._check_head_alignment(pose)
            if head_issue:
                issues.append(head_issue)
                if "drop" in head_issue.lower():
                    self._frame_checks.append("head_drop")
                else:
                    self._frame_checks.append("head_crane")
                scores["head"] = 0.5

        # L/R asymmetry
        if pose is not None:
            asym_penalty = self._check_symmetry(pose, joint_feedback, issues)
        else:
            asym_penalty = 0.0

        if not scores:
            return 0.0, joint_feedback, issues

        WEIGHTS = {"elbow": 1.0, "hip": 1.5, "shoulder": 1.0, "pike": 1.2, "head": 0.3}
        weighted_sum = sum(scores[k] * WEIGHTS.get(k, 1.0) for k in scores)
        weight_total = sum(WEIGHTS.get(k, 1.0) for k in scores)
        total = (weighted_sum / weight_total) - asym_penalty

        return max(0.0, min(1.0, total)), joint_feedback, issues

    def _check_symmetry(
        self,
        pose: PoseResult,
        joint_feedback: Dict[str, str],
        issues: List[str],
    ) -> float:
        penalty = 0.0

        l_elbow = self._angle_at(pose, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
        r_elbow = self._angle_at(pose, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        if l_elbow is not None and r_elbow is not None:
            diff = abs(l_elbow - r_elbow)
            if diff >= ASYMMETRY_WARN_DEG:
                weaker = "left" if l_elbow > r_elbow else "right"
                msg = f"Uneven arms — {weaker} side not matching"
                if msg not in issues:
                    issues.append(msg)
                if joint_feedback.get(f"{weaker}_elbow") != "incorrect":
                    joint_feedback[f"{weaker}_elbow"] = "warning"
                penalty = max(penalty, 0.1)

        l_sh = self._angle_at(pose, LEFT_ELBOW, LEFT_SHOULDER, LEFT_HIP)
        r_sh = self._angle_at(pose, RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP)
        if l_sh is not None and r_sh is not None:
            diff = abs(l_sh - r_sh)
            if diff >= ASYMMETRY_WARN_DEG:
                flared = "left" if l_sh > r_sh else "right"
                msg = f"{flared.capitalize()} elbow flaring more than the other side"
                if msg not in issues:
                    issues.append(msg)
                if joint_feedback.get(f"{flared}_shoulder") != "incorrect":
                    joint_feedback[f"{flared}_shoulder"] = "warning"
                penalty = max(penalty, 0.1)

        return penalty

    # ── Result building ──────────────────────────────────────────────────

    def _session_details(self) -> Dict[str, str]:
        return {
            "session_state": self._session_state.value,
            "posture": self._current_posture.value,
            "voice": "",
        }

    def _build_result(
        self,
        form_score: float,
        joint_feedback: Dict[str, str],
        details: Dict[str, str],
    ) -> ClassificationResult:
        is_active = self._session_state == SessionState.ACTIVE
        return ClassificationResult(
            exercise="pushup",
            is_correct=form_score >= 0.7 and is_active,
            confidence=form_score,
            joint_feedback=joint_feedback,
            details=details,
            is_active=is_active,
            form_score=max(0.0, min(1.0, form_score)),
        )

    # ── Angle helpers ────────────────────────────────────────────────────

    def _avg_elbow_angle(self, pose: PoseResult) -> Optional[float]:
        left = self._angle_at(pose, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
        right = self._angle_at(pose, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        if left is not None and right is not None:
            return (left + right) / 2
        return left or right

    def _compute_hip_angle(self, pose: PoseResult) -> Optional[float]:
        left = self._angle_at(pose, LEFT_SHOULDER, LEFT_HIP, LEFT_ANKLE)
        right = self._angle_at(pose, RIGHT_SHOULDER, RIGHT_HIP, RIGHT_ANKLE)
        if left is not None and right is not None:
            return (left + right) / 2
        return left or right

    def _compute_shoulder_angle(self, pose: PoseResult) -> Optional[float]:
        left = self._angle_at(pose, LEFT_ELBOW, LEFT_SHOULDER, LEFT_HIP)
        right = self._angle_at(pose, RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP)
        if left is not None and right is not None:
            return (left + right) / 2
        return left or right

    def _angle_at(
        self, pose: PoseResult, idx_a: int, idx_b: int, idx_c: int
    ) -> Optional[float]:
        if not all(pose.is_visible(i, MIN_VISIBILITY) for i in (idx_a, idx_b, idx_c)):
            return None
        a = pose.get_world_landmark(idx_a)
        b = pose.get_world_landmark(idx_b)
        c = pose.get_world_landmark(idx_c)
        return calculate_angle(a, b, c)

    def _smooth(self, buf: deque) -> Optional[float]:
        if not buf:
            return None
        return sum(buf) / len(buf)

    # ── Session summary ──────────────────────────────────────────────────

    def get_session_summary(self) -> Dict:
        all_set_reps = list(self._set_reps)
        if self._reps_in_current_set > 0:
            all_set_reps.append(self._reps_in_current_set)

        # Flush any open final set so the summary is complete.
        if self._reps_in_current_set > 0 and len(self._set_records) < len(all_set_reps):
            self._close_current_set(time.time(), failure_type="clean_stop")

        good_reps = sum(1 for r in self._rep_history if r["quality"] == RepQuality.GOOD)
        moderate_reps = sum(1 for r in self._rep_history if r["quality"] == RepQuality.MODERATE)
        bad_reps = sum(1 for r in self._rep_history if r["quality"] == RepQuality.BAD)

        issue_counts: Dict[str, int] = {}
        for rep in self._rep_history:
            for issue in rep.get("issues", []):
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        common_issues = sorted(issue_counts.items(), key=lambda x: -x[1])[:5]

        rep_scores = [float(r["score"]) for r in self._rep_history]
        avg_score = (sum(rep_scores) / len(rep_scores)) if rep_scores else 0.0

        total_rest_sec = sum(
            float(s.get("rest_before_sec", 0.0)) for s in self._set_records
        )
        duration_sec = float(getattr(self, "_session_duration_sec", 0.0) or 0.0)
        work_to_rest_ratio = round(
            duration_sec / max(total_rest_sec, 1.0), 2
        ) if duration_sec > 0 else 0.0

        if rep_scores:
            consistency_score = round(max(0.0, 1.0 - float(np.std(rep_scores))), 2)
        else:
            consistency_score = 0.0

        if len(rep_scores) >= 3:
            try:
                slope = float(np.polyfit(range(len(rep_scores)), rep_scores, 1)[0])
            except Exception:
                slope = 0.0
            fatigue_index = round(slope, 4)
        else:
            fatigue_index = 0.0

        return {
            "exercise": "pushup",
            "total_reps": self._rep_count,
            "num_sets": len(all_set_reps),
            "reps_per_set": all_set_reps,
            "good_reps": good_reps,
            "moderate_reps": moderate_reps,
            "bad_reps": bad_reps,
            "avg_form_score": round(avg_score, 2),
            # Session-1: per-set + session aggregates
            "sets": list(self._set_records),
            "total_rest_sec": round(total_rest_sec, 2),
            "work_to_rest_ratio": work_to_rest_ratio,
            "consistency_score": consistency_score,
            "fatigue_index": fatigue_index,
            "muscle_groups": list(self._muscle_groups),
            # Session-1: per-rep details in the same shape as other exercises
            "reps": [
                {
                    "rep_num": r["rep"],
                    "form_score": r["score"],
                    "quality": r["quality"],
                    "issues": r.get("issues", []),
                    "duration": r.get("duration", 0),
                    "peak_angle": r.get("peak_angle"),
                    "ecc_sec": r.get("ecc_sec", 0),
                    "con_sec": r.get("con_sec", 0),
                    "score_min": r.get("score_min"),
                    "score_max": r.get("score_max"),
                    "set_num": r.get("set_num"),
                }
                for r in self._rep_history
            ],
            # Legacy field — kept for the vanilla UI at app/static/app.js.
            "rep_details": self._rep_history,
            "common_issues": [
                {"issue": name, "count": count}
                for name, count in common_issues
            ],
        }
