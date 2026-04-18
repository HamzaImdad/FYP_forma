"""
Base exercise detector with shared state machine, rep counting, set tracking,
form scoring, and session summary logic.

Each exercise detector overrides:
  - THRESHOLDS: angle thresholds for state transitions and form checks
  - _compute_angles(): extract relevant angles from pose
  - _assess_form(): per-frame form quality scoring
  - _build_feedback_text(): actionable feedback string
"""

import json
import time
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..pose_estimation.base import PoseResult
from ..classification.base import ClassificationResult
from ..utils.constants import MIN_VISIBILITY
from ..utils.geometry import calculate_angle
from ..utils.activity_detector import ActivityDetector, ActivityState, RestTier
from ..feature_extraction.landmark_features import LandmarkFeatureExtractor
from .dtw_matcher import DTWMatcher, DTWScore
from .posture_classifier import PostureClassifier, PostureLabel


# ── Muscle-group lookup (shared across all detector instances) ──────────
_EXERCISE_DATA_PATH = Path(__file__).resolve().parents[2] / "app" / "exercise_data.json"


@lru_cache(maxsize=1)
def _load_exercise_data() -> Dict[str, Dict]:
    try:
        return json.loads(_EXERCISE_DATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_muscles_for(exercise: str) -> List[str]:
    data = _load_exercise_data()
    entry = data.get(exercise) or {}
    muscles = entry.get("muscles")
    return list(muscles) if isinstance(muscles, list) else []


# ── Shared constants ────────────────────────────────────────────────────
MIN_REP_DURATION = 1.0      # seconds — real reps take >1s (was 0.4, counted noise)

# Issue priority: lower number = higher priority (shown first to user).
# When multiple issues fire, only the highest-priority one is shown.
ISSUE_PRIORITY_KEYWORDS = {
    "hip": 1, "core": 1, "sag": 1, "pike": 1, "hips": 1,
    "back": 2, "spine": 2, "round": 2, "chest": 2,
    "knee": 3, "squat": 3, "leg": 3,
    "elbow": 4, "shoulder": 4, "arm": 4, "flare": 4,
    "head": 5, "neck": 5,
}
MIN_DESCENT_DURATION = 0.5  # seconds — descent must be controlled, not an angle spike
MIN_BOTTOM_FRAMES = 3       # frames in BOTTOM before allowing GOING_UP transition
SET_REST_TIMEOUT = 8.0      # seconds of inactivity to end a set
ANGLE_SMOOTH_WINDOW = 3     # frames for moving average
SCORE_SMOOTH_WINDOW = 8     # frames for form score smoothing
NO_POSE_RESET_FRAMES = 30   # re-validate if pose lost this many frames
MAX_REP_HISTORY = 200       # cap rep history
ASYMMETRY_WARN_DEG = 12.0   # degrees L/R diff to flag asymmetry
ASYMMETRY_BAD_DEG = 20.0    # degrees L/R diff — significant imbalance
LOW_VIS_THRESHOLD = 0.65    # below this, discount score confidence


class RepPhase:
    """Universal rep phase states."""
    TOP = "top"
    GOING_DOWN = "going_down"
    BOTTOM = "bottom"
    GOING_UP = "going_up"


class RepQuality:
    GOOD = "good"
    MODERATE = "moderate"
    BAD = "bad"


class BaseExerciseDetector(ABC):
    """Base class for exercise-specific detectors.

    Provides: state machine, rep counting, set tracking, form scoring,
    session summary. Subclasses implement exercise-specific angles and rules.
    """

    # Override in subclass
    EXERCISE_NAME: str = ""
    PRIMARY_ANGLE_TOP: float = 170.0    # angle at top of movement
    PRIMARY_ANGLE_BOTTOM: float = 90.0  # angle at bottom of movement
    DESCENT_THRESHOLD: float = 10.0     # degrees drop to start descent
    ASCENT_THRESHOLD: float = 10.0      # degrees rise to start ascent
    IS_STATIC: bool = False             # True for plank (no rep counting)
    MIN_FORM_GATE: float = 0.0          # avg score to count rep (0 = always count)

    # Shared landmark extractor for DTW rep capture (torso-normalized 99-dim)
    _landmark_extractor = LandmarkFeatureExtractor()

    def __init__(self, coverage_penalty: bool = False,
                 dtw_matcher: Optional["DTWMatcher"] = None):
        self._coverage_penalty = coverage_penalty
        self._dtw_matcher: Optional["DTWMatcher"] = dtw_matcher
        self._muscle_groups: List[str] = _load_muscles_for(self.EXERCISE_NAME)
        self.reset()

    def reset(self):
        """Reset all state."""
        self._state = RepPhase.TOP
        self._form_validated = False

        # Rep counting
        self._rep_count = 0
        self._last_rep_time = 0.0
        self._rep_start_time = 0.0
        self._descent_start_time = 0.0    # when GOING_DOWN started (= ecc start)
        self._concentric_start_time = 0.0 # when GOING_UP started (= con start)
        self._bottom_frame_count = 0     # frames spent in BOTTOM state
        self._first_rep_started = False  # suppress issues before first descent

        # Set tracking
        self._set_count = 0
        self._reps_in_current_set = 0
        self._last_active_time = 0.0
        self._set_reps: List[int] = []
        # Per-set records with tempo/fatigue metadata — parallel to _set_reps
        # for back-compat. Closed sets are appended by _close_current_set().
        self._set_records: List[Dict] = []
        self._last_set_end_time: float = 0.0
        # Between-sets gate: True while user is resting *after* finishing a set
        # and before starting the next one. Cleared only when a new rep's
        # descent actually begins (not on idle/movement alone).
        self._between_sets = False
        # Latched flag consumed by the coaching layer to speak the next-set
        # announce exactly once per set boundary.
        self._pending_set_announce = False
        # Rolling window of recent rep durations (seconds) for pace detection
        self._recent_rep_durations: deque = deque(maxlen=3)

        # Per-rep form tracking
        self._current_rep_issues: List[str] = []
        self._current_rep_scores: List[float] = []
        self._current_rep_landmarks: List[np.ndarray] = []
        self._rep_history: List[Dict] = []

        # Last DTW score (set when a rep completes with a matcher attached)
        self._last_dtw_similarity: float = 1.0
        self._last_dtw_worst_joint: Optional[str] = None

        # Angle smoothing buffers (subclass adds more as needed)
        self._angle_buffers: Dict[str, deque] = {}

        # Form score smoothing
        self._score_buf: deque = deque(maxlen=SCORE_SMOOTH_WINDOW)

        # Pose tracking
        self._frames_since_pose = 0

        # Last computed angles (for UI)
        self.last_angles: Dict[str, Optional[float]] = {}

        # Static hold tracking (plank)
        self._hold_start_time: Optional[float] = None
        self._hold_duration: float = 0.0

        # Asymmetry tracking: stores L/R angles per named angle
        self._lr_angles: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        # Visibility confidence per frame
        self._frame_visibility: float = 1.0
        # Coverage tracking (for offline scoring)
        self._last_coverage: float = 1.0

        # Activity & rest detection
        self._activity_detector = ActivityDetector()
        self._current_activity = ActivityState.NO_POSE
        self._current_rest_tier = RestTier.ACTIVE

    @property
    def rep_count(self) -> int:
        return self._rep_count

    @property
    def set_count(self) -> int:
        return self._set_count

    @property
    def reps_in_current_set(self) -> int:
        return self._reps_in_current_set

    def _get_smooth(self, name: str) -> Optional[float]:
        """Get smoothed value from named angle buffer."""
        buf = self._angle_buffers.get(name)
        if not buf:
            return None
        return sum(buf) / len(buf)

    def _push_angle(self, name: str, value: Optional[float]):
        """Push angle value into named smoothing buffer."""
        if name not in self._angle_buffers:
            self._angle_buffers[name] = deque(maxlen=ANGLE_SMOOTH_WINDOW)
        if value is not None:
            self._angle_buffers[name].append(value)

    def _angle_at(self, pose: PoseResult, idx_a: int, idx_b: int, idx_c: int) -> Optional[float]:
        """Compute angle at vertex idx_b using world coordinates."""
        if not all(pose.is_visible(i, MIN_VISIBILITY) for i in (idx_a, idx_b, idx_c)):
            return None
        a = pose.get_world_landmark(idx_a)
        b = pose.get_world_landmark(idx_b)
        c = pose.get_world_landmark(idx_c)
        return calculate_angle(a, b, c)

    def _avg_angle(self, pose: PoseResult,
                   left_triple: Tuple[int, int, int],
                   right_triple: Tuple[int, int, int],
                   name: Optional[str] = None) -> Optional[float]:
        """Average angle from left and right sides (or whichever visible).
        Stores L/R values for asymmetry detection if name is provided.
        """
        left = self._angle_at(pose, *left_triple)
        right = self._angle_at(pose, *right_triple)
        if name:
            self._lr_angles[name] = (left, right)
        if left is not None and right is not None:
            return (left + right) / 2
        return left if left is not None else right

    def _compute_visibility(self, pose: PoseResult, indices: List[int]) -> float:
        """Compute average visibility confidence for a set of landmark indices."""
        vis_sum = 0.0
        count = 0
        for idx in indices:
            if idx < len(pose.landmarks):
                vis_sum += pose.landmarks[idx, 3]  # visibility is 4th column
                count += 1
        return vis_sum / max(count, 1)

    def _check_asymmetry(self, joint_feedback: Dict[str, str],
                         issues: List[str]) -> float:
        """Check L/R asymmetry across all tracked angles.
        Returns asymmetry penalty (0.0 = none, up to 0.15 deduction).
        Updates joint_feedback to mark the weaker side.
        """
        penalty = 0.0
        side_map = {
            "knee": ("left_knee", "right_knee"),
            "hip": ("left_hip", "right_hip"),
            "elbow": ("left_elbow", "right_elbow"),
            "shoulder": ("left_shoulder", "right_shoulder"),
        }
        for angle_name, (left, right) in self._lr_angles.items():
            if left is None or right is None:
                continue
            diff = abs(left - right)
            if diff >= ASYMMETRY_BAD_DEG:
                # Significant imbalance
                penalty = max(penalty, 0.15)
                weaker_side = "left" if left < right else "right"
                fb_key = side_map.get(angle_name)
                if fb_key:
                    weak_key = fb_key[0] if weaker_side == "left" else fb_key[1]
                    if joint_feedback.get(weak_key) != "incorrect":
                        joint_feedback[weak_key] = "warning"
                issues.append(f"Uneven {angle_name.replace('_', ' ')} — {weaker_side} side weaker")
            elif diff >= ASYMMETRY_WARN_DEG:
                penalty = max(penalty, 0.07)
        return penalty

    # ── Abstract methods (subclass must implement) ──────────────────────

    @abstractmethod
    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        """Compute all relevant angles for this exercise.
        Returns dict of angle_name -> value_or_None.
        Must include 'primary' key for state machine angle.
        """
        ...

    @abstractmethod
    def _assess_form(self, angles: Dict[str, Optional[float]]) -> Tuple[float, Dict[str, str], List[str]]:
        """Assess form quality for a single frame.
        Returns: (score 0-1, joint_feedback dict, list of issue strings)
        """
        ...

    @abstractmethod
    def _check_start_position(self, angles: Dict[str, Optional[float]]) -> Tuple[bool, List[str]]:
        """Check if user is in correct starting position.
        Returns: (is_valid, list of setup instructions)
        """
        ...

    def _should_count_rep(self, elapsed: float, angles: Dict[str, Optional[float]]) -> bool:
        """Decide whether a completed state-machine cycle should count as a rep.

        Override in subclass to add exercise-specific validation (e.g., form
        score gating, movement pattern checks).  Called when the state machine
        reaches TOP from GOING_UP.
        """
        if elapsed < MIN_REP_DURATION:
            return False
        # Require controlled descent (not an instantaneous angle spike)
        descent_dur = (self._rep_start_time + elapsed) - self._descent_start_time
        if descent_dur < MIN_DESCENT_DURATION:
            return False
        # Form gate: reject reps that averaged below threshold (random movements)
        if self.MIN_FORM_GATE > 0 and self._current_rep_scores:
            avg = sum(self._current_rep_scores) / len(self._current_rep_scores)
            if avg < self.MIN_FORM_GATE:
                return False
        return True

    def _get_missing_parts(self, angles: Dict[str, Optional[float]]) -> List[str]:
        """Return list of body parts that aren't visible. Override for custom messages."""
        missing = []
        if angles.get("primary") is None:
            missing.append("primary joints")
        return missing

    def _build_feedback_text(self, score: float, issues: List[str],
                             angles: Dict[str, Optional[float]],
                             rep_completed: bool) -> str:
        """Build actionable feedback text. Can override for custom messages."""
        parts = []
        if rep_completed and self._rep_history:
            last_rep = self._rep_history[-1]
            q = last_rep["quality"]
            if q == RepQuality.GOOD:
                parts.append(f"Rep {self._rep_count}: Great form!")
            elif q == RepQuality.MODERATE:
                parts.append(f"Rep {self._rep_count}: Decent")
                if last_rep["issues"]:
                    parts.append(last_rep["issues"][0])
            else:
                parts.append(f"Rep {self._rep_count}: Needs work")
                if last_rep["issues"]:
                    parts.append(last_rep["issues"][0])
        elif issues:
            parts.append(issues[0])
        elif score >= 0.7:
            phase_names = {
                RepPhase.TOP: "Good — begin rep",
                RepPhase.GOING_DOWN: "Good — keep going",
                RepPhase.BOTTOM: "Good depth! Come back up",
                RepPhase.GOING_UP: "Push! Extend fully",
            }
            parts.append(phase_names.get(self._state, "Good form"))

        if self._set_count > 0:
            parts.append(f"Set {self._set_count + 1}")

        return " | ".join(parts) if parts else "Good form"

    # ── Main classify method ────────────────────────────────────────────

    def classify(self, pose: PoseResult, timestamp: float = None) -> ClassificationResult:
        """Process a pose and return classification result.

        Args:
            pose: PoseResult from frame
            timestamp: Optional timestamp in seconds (for offline CSV processing).
                       Defaults to time.time() for real-time use.
        """
        now = timestamp if timestamp is not None else time.time()

        # Compute and smooth angles
        raw_angles = self._compute_angles(pose)
        for name, val in raw_angles.items():
            self._push_angle(name, val)
            self.last_angles[name] = val

        # Track overall landmark visibility for confidence weighting
        self._frame_visibility = float(np.mean(pose.landmarks[:, 3]))

        angles = {name: self._get_smooth(name) for name in raw_angles}

        primary = angles.get("primary")

        # Check if essential angles are available
        if primary is None:
            self._frames_since_pose += 1
            if self._frames_since_pose >= NO_POSE_RESET_FRAMES:
                # User left frame — reset state machine, don't stay stuck mid-rep.
                # The session itself is NOT ended; any reps in the current set
                # are flushed so returning to frame starts a fresh set.
                if self._reps_in_current_set > 0 and not self._between_sets:
                    self._set_reps.append(self._reps_in_current_set)
                    self._close_current_set(now, failure_type="timeout")
                    self._set_count += 1
                    self._reps_in_current_set = 0
                    self._between_sets = True
                    self._pending_set_announce = True
                if self._form_validated:
                    self._form_validated = False
                self._state = RepPhase.TOP
                self._bottom_frame_count = 0
                for buf in self._angle_buffers.values():
                    buf.clear()

            missing = self._get_missing_parts(angles)
            return self._inactive_result(
                f"Can't detect {', '.join(missing) if missing else 'body'}. "
                "Position camera so full body is visible."
            )

        self._frames_since_pose = 0

        # ── Form validation gate ──
        if not self._form_validated:
            valid, setup_issues = self._check_start_position(angles)
            if valid:
                self._form_validated = True
                self._rep_start_time = now
                if self.IS_STATIC:
                    self._hold_start_time = now
            else:
                return self._build_result(
                    form_score=0.0, is_active=False,
                    joint_feedback={},
                    details={"setup": " | ".join(setup_issues) if setup_issues else f"Get into {self.EXERCISE_NAME.replace('_', ' ')} position"},
                    feedback_text=setup_issues[0] if setup_issues else f"Get into position",
                )

        # ── Static hold (plank) ──
        if self.IS_STATIC:
            self._last_active_time = now
            if self._hold_start_time:
                self._hold_duration = now - self._hold_start_time

            frame_score, joint_feedback, issues = self._assess_form(angles)

            # Coverage penalty for static holds too
            if self._coverage_penalty:
                n_angles = sum(1 for k, v in angles.items() if k != "primary" and v is not None)
                n_expected = max(1, len(angles) - 1)
                coverage = n_angles / n_expected
                if coverage < 0.6:
                    frame_score *= coverage
                self._last_coverage = coverage

            # T38: asymmetry check also for static holds — previously the
            # early-return on line ~368 skipped this entirely for plank.
            asym_penalty = self._check_asymmetry(joint_feedback, issues)
            frame_score = max(0.0, frame_score - asym_penalty)

            self._score_buf.append(frame_score)
            smooth_score = sum(self._score_buf) / len(self._score_buf)

            details = {}
            for i, issue in enumerate(issues[:3]):
                details[f"issue_{i}"] = issue
            details["hold_duration"] = f"{self._hold_duration:.0f}s"

            feedback_text = issues[0] if issues else (
                f"Great hold! {self._hold_duration:.0f}s" if smooth_score >= 0.7
                else f"Hold position — {self._hold_duration:.0f}s"
            )

            return self._build_result(
                form_score=smooth_score, is_active=True,
                joint_feedback=joint_feedback, details=details,
                feedback_text=feedback_text,
            )

        # ── Activity detection ──
        activity, rest_tier = self._activity_detector.update(pose, now)
        self._current_activity = activity
        self._current_rest_tier = rest_tier
        is_actively_exercising = activity in (ActivityState.EXERCISING, ActivityState.TRANSITION)

        # Only update last_active_time when actually moving
        if is_actively_exercising:
            self._last_active_time = now

        # ── Set detection (tiered) ──
        # Flush the pending set as soon as the user has been idle long enough
        # to clearly be resting (≥15s of stillness). LONG_REST is also included
        # so multi-minute rests don't leave a pending set dangling.
        long_idle = rest_tier in (RestTier.EXTENDED_REST, RestTier.LONG_REST)
        if long_idle and self._reps_in_current_set > 0 and not self._between_sets:
            self._set_reps.append(self._reps_in_current_set)
            self._close_current_set(now, failure_type="clean_stop")
            self._set_count += 1
            self._reps_in_current_set = 0
            self._between_sets = True
            self._pending_set_announce = True

        # ── State machine ──
        rep_completed = False

        if self._state == RepPhase.TOP:
            if primary < self.PRIMARY_ANGLE_TOP - self.DESCENT_THRESHOLD:
                self._state = RepPhase.GOING_DOWN
                self._rep_start_time = now
                self._descent_start_time = now
                self._concentric_start_time = 0.0  # reset — stamped at BOTTOM
                self._bottom_frame_count = 0
                self._first_rep_started = True
                self._current_rep_issues = []
                self._current_rep_scores = []
                self._current_rep_landmarks = []
                # A new rep's descent starting is the only signal that set N+1
                # has truly begun. Clear the between-sets gate here.
                if self._between_sets:
                    self._between_sets = False

        elif self._state == RepPhase.GOING_DOWN:
            if primary <= self.PRIMARY_ANGLE_BOTTOM:
                self._state = RepPhase.BOTTOM
                self._concentric_start_time = now  # end of eccentric phase
                self._bottom_frame_count = 0
            elif primary > self.PRIMARY_ANGLE_TOP:
                self._state = RepPhase.TOP
                self._current_rep_issues.append("Didn't reach full depth")

        elif self._state == RepPhase.BOTTOM:
            self._bottom_frame_count += 1
            if (primary > self.PRIMARY_ANGLE_BOTTOM + self.ASCENT_THRESHOLD
                    and self._bottom_frame_count >= MIN_BOTTOM_FRAMES):
                self._state = RepPhase.GOING_UP

        elif self._state == RepPhase.GOING_UP:
            if primary >= self.PRIMARY_ANGLE_TOP:
                elapsed = now - self._rep_start_time
                if self._should_count_rep(elapsed, angles):
                    rep_completed = True
                    self._rep_count += 1
                    self._reps_in_current_set += 1
                    self._recent_rep_durations.append(elapsed)
                self._state = RepPhase.TOP
            elif primary < self.PRIMARY_ANGLE_BOTTOM:
                self._state = RepPhase.BOTTOM
                self._bottom_frame_count = 0

        # ── Per-frame form assessment ──
        self._lr_angles.clear()
        frame_score, joint_feedback, issues = self._assess_form(angles)

        # Sort issues by priority (hips/core first, head/neck last)
        issues.sort(key=lambda msg: min(
            (pri for kw, pri in ISSUE_PRIORITY_KEYWORDS.items() if kw in msg.lower()),
            default=9
        ))

        # Suppress issues while user is standing idle before first rep OR
        # while they're resting between sets (green skeleton, no nagging).
        idle_before_first = not self._first_rep_started and self._state == RepPhase.TOP
        resting_between = self._between_sets and self._state == RepPhase.TOP
        if idle_before_first or resting_between:
            issues = []
            for k in joint_feedback:
                joint_feedback[k] = "correct"
            frame_score = max(frame_score, 0.8)  # don't show bad score while idle

        # ── Coverage penalty (for offline scoring — penalise missing joints) ──
        if self._coverage_penalty:
            n_angles = sum(1 for k, v in angles.items() if k != "primary" and v is not None)
            n_expected = max(1, len(angles) - 1)  # exclude "primary" (duplicate of another)
            coverage = n_angles / n_expected
            if coverage < 0.6:
                frame_score *= coverage
            self._last_coverage = coverage
        else:
            self._last_coverage = 1.0

        # ── Asymmetry detection (auto for all exercises) ──
        asym_penalty = self._check_asymmetry(joint_feedback, issues)
        frame_score = max(0.0, frame_score - asym_penalty)

        # ── Visibility-weighted confidence ──
        if self._frame_visibility < LOW_VIS_THRESHOLD:
            # Discount score when landmarks are poorly visible
            vis_factor = 0.5 + 0.5 * (self._frame_visibility / LOW_VIS_THRESHOLD)
            frame_score *= vis_factor

        self._current_rep_scores.append(frame_score)
        for issue in issues:
            if issue not in self._current_rep_issues:
                self._current_rep_issues.append(issue)

        # ── Capture landmark frame for DTW (only while mid-rep) ──
        if self._dtw_matcher is not None and self._state != RepPhase.TOP:
            try:
                self._current_rep_landmarks.append(
                    self._landmark_extractor.extract_landmarks(pose)
                )
            except Exception:
                pass  # DTW capture is best-effort, never breaks classification

        # ── Record completed rep ──
        if rep_completed:
            avg_score = sum(self._current_rep_scores) / max(len(self._current_rep_scores), 1)

            # Run DTW template matching against ideal templates for this rep
            dtw_sim = 1.0
            dtw_worst: Optional[str] = None
            if self._dtw_matcher is not None and len(self._current_rep_landmarks) >= 3:
                try:
                    rep_seq = np.stack(self._current_rep_landmarks).astype(np.float32)
                    dtw_score = self._dtw_matcher.compare(self.EXERCISE_NAME, rep_seq)
                    if dtw_score is not None:
                        dtw_sim = dtw_score.similarity
                        dtw_worst = dtw_score.worst_joint
                        # Pattern-match penalty: a rep that completes the state machine
                        # but diverges significantly from the ideal template gets marked down.
                        if dtw_sim < 0.5:
                            avg_score *= (0.6 + 0.4 * dtw_sim)
                            if dtw_worst and f"Pattern deviates at {dtw_worst}" not in self._current_rep_issues:
                                self._current_rep_issues.insert(0, f"Pattern deviates at {dtw_worst}")
                except Exception as e:
                    pass  # never let DTW break a rep

            self._last_dtw_similarity = dtw_sim
            self._last_dtw_worst_joint = dtw_worst

            quality = (RepQuality.GOOD if avg_score >= 0.7
                       else RepQuality.MODERATE if avg_score >= 0.4
                       else RepQuality.BAD)
            if len(self._rep_history) < MAX_REP_HISTORY:
                rep_entry = {
                    "rep": self._rep_count,
                    "score": round(avg_score, 2),
                    "quality": quality,
                    "issues": self._current_rep_issues[:3],
                    "duration": round(now - self._rep_start_time, 1),
                }
                if self._dtw_matcher is not None:
                    rep_entry["dtw_similarity"] = round(dtw_sim, 2)
                    if dtw_worst:
                        rep_entry["dtw_worst_joint"] = dtw_worst
                self._augment_rep_entry(rep_entry, now)
                self._rep_history.append(rep_entry)

        # ── Smooth form score ──
        self._score_buf.append(frame_score)
        smooth_score = sum(self._score_buf) / len(self._score_buf)

        # ── Feedback text ──
        feedback_text = self._build_feedback_text(smooth_score, issues, angles, rep_completed)

        # ── Build details dict ──
        details = {}
        for i, issue in enumerate(issues[:3]):
            details[f"issue_{i}"] = issue

        phase_names = {
            RepPhase.TOP: "Top position",
            RepPhase.GOING_DOWN: "Lowering",
            RepPhase.BOTTOM: "Bottom position",
            RepPhase.GOING_UP: "Coming up",
        }
        details["phase"] = phase_names.get(self._state, "")

        if primary is not None:
            progress = np.clip(
                (self.PRIMARY_ANGLE_TOP - primary) /
                max(self.PRIMARY_ANGLE_TOP - self.PRIMARY_ANGLE_BOTTOM, 1), 0, 1
            )
            details["progress"] = f"{int(progress * 100)}%"

        # Activity & rest state for frontend
        details["activity"] = self._current_activity.value
        details["rest_tier"] = self._current_rest_tier.value
        details["between_sets"] = "1" if self._between_sets else "0"
        details["set_number"] = str(self._set_count + 1)
        details["reps_in_current_set"] = str(self._reps_in_current_set)
        if self._recent_rep_durations:
            avg_dur = sum(self._recent_rep_durations) / len(self._recent_rep_durations)
            details["avg_rep_duration"] = f"{avg_dur:.2f}"

        return self._build_result(
            form_score=smooth_score, is_active=is_actively_exercising,
            joint_feedback=joint_feedback, details=details,
            feedback_text=feedback_text,
        )

    # ── Result builders ─────────────────────────────────────────────────

    def _build_result(self, form_score, is_active, joint_feedback, details,
                      feedback_text="") -> ClassificationResult:
        return ClassificationResult(
            exercise=self.EXERCISE_NAME,
            is_correct=form_score >= 0.7 and is_active,
            confidence=form_score,
            joint_feedback=joint_feedback,
            details=details,
            is_active=is_active,
            form_score=max(0.0, min(1.0, form_score)),
            dtw_similarity=float(self._last_dtw_similarity),
            dtw_worst_joint=self._last_dtw_worst_joint,
        )

    def _inactive_result(self, message: str) -> ClassificationResult:
        return self._build_result(0.0, False, {}, {"setup": message})

    # ── Session-1 capture helpers ───────────────────────────────────────

    def _augment_rep_entry(self, rep_entry: Dict, now: float) -> None:
        """Mutate rep_entry in place with the Session-1 capture fields
        (tempo, peak angle, score min/max, set_num). Called from both the
        legacy and robust rep-commit paths."""
        scores = self._current_rep_scores or [0.0]
        peak = getattr(self, "_rep_peak", None)
        rep_entry["peak_angle"] = round(float(peak), 1) if peak is not None else None
        rep_entry["score_min"] = round(float(min(scores)), 2)
        rep_entry["score_max"] = round(float(max(scores)), 2)
        # ecc_sec = descent duration; con_sec = ascent duration. Zero when a
        # phase timestamp wasn't stamped (defensive for the legacy path).
        ecc = 0.0
        con = 0.0
        if self._descent_start_time and self._concentric_start_time:
            ecc = max(0.0, self._concentric_start_time - self._descent_start_time)
        if self._concentric_start_time:
            con = max(0.0, now - self._concentric_start_time)
        rep_entry["ecc_sec"] = round(ecc, 2)
        rep_entry["con_sec"] = round(con, 2)
        # set_num is 1-indexed: sets 1..N while the current set is open
        # and _set_count has not yet been incremented.
        rep_entry["set_num"] = self._set_count + 1

    def _close_current_set(self, now: float, failure_type: str = "clean_stop") -> None:
        """Build and append a per-set record to `_set_records` based on the
        reps in the just-finished set. Call this at every site that also
        appends to `_set_reps` + increments `_set_count`.

        `failure_type` is the caller's best guess:
          - 'timeout' when the caller fired due to SET_REST_TIMEOUT / idle
          - 'clean_stop' when the user voluntarily stopped (the default)
          - 'form_breakdown' is auto-inferred here if the final 3 reps of
             the set strictly trended downward (overrides 'clean_stop').
        """
        reps_count = int(self._reps_in_current_set)
        if reps_count <= 0 and not self._rep_history:
            return

        # Pull the scores from the trailing `reps_count` rep_history entries
        # (this set's reps). Guard against short histories.
        tail = self._rep_history[-reps_count:] if reps_count > 0 else []
        scores = [float(r.get("score", 0.0)) for r in tail]

        if scores:
            avg = sum(scores) / len(scores)
            dropoff = scores[0] - scores[-1]
        else:
            avg = 0.0
            dropoff = 0.0

        # Auto-detect form breakdown (only if caller didn't already say timeout)
        ft = failure_type
        if ft == "clean_stop" and len(scores) >= 3:
            if scores[-3] > scores[-2] > scores[-1]:
                ft = "form_breakdown"

        rest_before = 0.0
        if self._last_set_end_time > 0.0:
            rest_before = min(600.0, max(0.0, now - self._last_set_end_time))

        record = {
            # _set_count is incremented at the same caller site — use the
            # post-increment value so set_num matches the outward-facing
            # "set 1, 2, 3" numbering.
            "set_num": int(self._set_count + 1),
            "reps_count": reps_count,
            "rest_before_sec": round(rest_before, 2),
            "avg_form_score": round(avg, 2),
            "score_dropoff": round(dropoff, 2),
            "failure_type": ft,
        }
        self._set_records.append(record)
        self._last_set_end_time = now

    # ── Session summary ─────────────────────────────────────────────────

    def get_session_summary(self) -> Dict:
        all_set_reps = list(self._set_reps)
        if self._reps_in_current_set > 0:
            all_set_reps.append(self._reps_in_current_set)

        # Flush any open final set into _set_records so the summary includes it.
        # Only applies if reps were committed but the set hasn't been closed
        # (e.g. user hit End Session mid-set).
        if self._reps_in_current_set > 0 and len(self._set_records) < len(all_set_reps):
            self._close_current_set(time.time(), failure_type="clean_stop")

        good = sum(1 for r in self._rep_history if r["quality"] == RepQuality.GOOD)

        issue_counts: Dict[str, int] = {}
        for rep in self._rep_history:
            for issue in rep.get("issues", []):
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        common_issues = sorted(issue_counts.items(), key=lambda x: -x[1])[:5]

        rep_scores = [float(r["score"]) for r in self._rep_history]
        avg_score = (sum(rep_scores) / len(rep_scores)) if rep_scores else 0.0

        # Session-1 aggregates
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

        summary = {
            "exercise": self.EXERCISE_NAME,
            "total_reps": self._rep_count,
            "good_reps": good,
            "avg_form_score": round(avg_score, 2),
            "reps_per_set": all_set_reps,
            "common_issues": [{"issue": n, "count": c} for n, c in common_issues],
            # Session-1 session-level fields
            "total_rest_sec": round(total_rest_sec, 2),
            "work_to_rest_ratio": work_to_rest_ratio,
            "consistency_score": consistency_score,
            "fatigue_index": fatigue_index,
            "muscle_groups": list(self._muscle_groups),
            "sets": list(self._set_records),
            "reps": [
                {
                    "rep_num": r["rep"],
                    "form_score": r["score"],
                    "quality": r["quality"],
                    "issues": r.get("issues", []),
                    "duration": r.get("duration", 0),
                    # Session-1 per-rep fields
                    "peak_angle": r.get("peak_angle"),
                    "ecc_sec": r.get("ecc_sec", 0),
                    "con_sec": r.get("con_sec", 0),
                    "score_min": r.get("score_min"),
                    "score_max": r.get("score_max"),
                    "set_num": r.get("set_num"),
                }
                for r in self._rep_history
            ],
        }

        if self.IS_STATIC:
            summary["hold_duration"] = round(self._hold_duration, 1)
            summary["total_reps"] = 0

        return summary


# ══════════════════════════════════════════════════════════════════════════
# Robust rep-counting layer — ports the push-up detector's winning recipe
# (state-independent commit, posture-driven session FSM, raw-angle sanity
# guards, pre-active trace seeding) to any `BaseExerciseDetector` subclass.
#
# Subclasses set ~8 configuration constants + keep their existing
# `_compute_angles`, `_assess_form`, `_check_start_position` implementations.
# All form-check logic, DTW, and session summary code stays untouched.
# ══════════════════════════════════════════════════════════════════════════


# Rep direction — whether the primary angle DECREASES during the working
# phase (pushup/squat/pullup/bicep curl/deadlift/bench/tricep/lunge) or
# INCREASES (overhead press).
REP_DIRECTION_DECREASING = "DECREASING"
REP_DIRECTION_INCREASING = "INCREASING"


class SessionState(str, Enum):
    """Outer session FSM — when is the user actually exercising?"""
    IDLE = "idle"
    SETUP = "setup"
    ACTIVE = "active"
    RESTING = "resting"


class RobustExerciseDetector(BaseExerciseDetector):
    """Adds push-up-style robust rep counting on top of BaseExerciseDetector.

    All five mechanisms ported from pushup_detector.py:
      1. State-independent rep commit (tracks rep_peak, fires when signal
         returns to top AND peak crossed depth threshold).
      2. Raw-angle sanity guards on the primary angle
         (MIN_REAL_PRIMARY_DEG floor, MAX_PRIMARY_JUMP_DEG outlier guard).
      3. Single-frame None tolerance (reuses last value up to
         MAX_NONE_TOLERANCE frames before giving up).
      4. Outer session FSM (IDLE/SETUP/ACTIVE/RESTING) driven by
         PostureClassifier — replaces timer-based set detection.
      5. Pre-active primary-angle trace seeding — recovers the first rep
         even if descent started during posture-classifier bootstrap.

    The subclass contract (`_compute_angles`, `_assess_form`,
    `_check_start_position`) is unchanged.
    """

    # ── Robust-FSM configuration (subclass overrides as needed) ─────────

    # Which PostureClassifier label represents "ready to exercise".
    # `None` = visibility-only gate (IDLE→ACTIVE when key landmarks visible).
    SESSION_POSTURE_GATE: Optional[PostureLabel] = PostureLabel.STANDING

    # Rep direction — DECREASING for most exercises, INCREASING for OHP.
    REP_DIRECTION: str = REP_DIRECTION_DECREASING

    # Rep FSM thresholds on the raw primary angle (distinct from the
    # form-scoring PRIMARY_ANGLE_TOP / PRIMARY_ANGLE_BOTTOM).
    #
    # For DECREASING:
    #   angle ≥ REP_COMPLETE_TOP    → "at top"       (commit zone, no descent yet)
    #   angle <  REP_START_THRESHOLD → descent committed (UP → GOING_DOWN)
    #   angle ≤ REP_DEPTH_THRESHOLD → valid depth reached (rep_peak ≤ this)
    #   angle >  REP_BOTTOM_CLEAR   → ascent started (DOWN → GOING_UP)
    #
    # For INCREASING (overhead press), semantics flip:
    #   angle ≤ REP_COMPLETE_TOP    → "at bottom" (at shoulders, commit zone)
    #   angle >  REP_START_THRESHOLD → press committed (TOP → GOING_UP_TO_DEPTH)
    #   angle ≥ REP_DEPTH_THRESHOLD → lockout reached
    #   angle <  REP_BOTTOM_CLEAR   → descent from lockout started
    REP_COMPLETE_TOP: float = 140.0
    REP_START_THRESHOLD: float = 135.0
    REP_DEPTH_THRESHOLD: float = 115.0
    REP_BOTTOM_CLEAR: float = 120.0

    # Sanity guards on the raw primary angle before it reaches the rep FSM.
    MIN_REAL_PRIMARY_DEG: float = 20.0
    MAX_PRIMARY_JUMP_DEG: float = 80.0
    MAX_NONE_TOLERANCE: int = 5

    # Session FSM tolerances.
    UNKNOWN_GRACE_FRAMES: int = 20
    PRE_ACTIVE_TRACE_SIZE: int = 15

    # Stance-based set-boundary detection.
    # A set closes only when the user has been OUT of the exercise stance
    # for this long. While in stance, the set stays open regardless of how
    # long the user pauses between reps — mid-set rest is same set.
    OUT_OF_STANCE_DEBOUNCE_S: float = 3.0

    # Visibility-gate config (used when SESSION_POSTURE_GATE is None).
    # Subclass can override to require different landmarks.
    VISIBILITY_GATE_LANDMARKS: Tuple[int, ...] = ()  # set in subclass if needed
    VISIBILITY_GATE_THRESHOLD: float = 0.6

    def reset(self):
        super().reset()
        # ── Robust additions ─────────────────────────────────────────────
        self._posture_classifier = PostureClassifier()
        self._session_state: SessionState = SessionState.IDLE
        self._unknown_streak: int = 0

        # Raw angle tracking for sanity guards + None tolerance
        self._last_raw_primary: Optional[float] = None
        self._consec_none_frames: int = 0

        # Robust rep FSM state
        # _rep_peak tracks the extremum of primary angle since the descent
        # (or press) began. For DECREASING direction: min. For INCREASING: max.
        self._rep_peak: float = self._init_rep_peak()
        self._robust_first_rep_started: bool = False

        # Pending rep commit — set by _update_rep_fsm_robust when commit
        # conditions are met, consumed by _finalize_pending_rep after
        # form assessment has scored the current frame.
        self._pending_rep_elapsed: Optional[float] = None

        # Pre-active primary-angle trace (for first-rep seeding)
        # Elements are (timestamp, primary_angle, is_gate_posture_raw)
        self._pre_active_trace: deque = deque(maxlen=self.PRE_ACTIVE_TRACE_SIZE)

        # Current posture (exposed for trace)
        self._current_posture: PostureLabel = PostureLabel.UNKNOWN

        # Static-hold (plank) per-set breakdown.
        # _hold_duration accumulates total time across rests; these track
        # per-attempt chunks so the HUD can show "Set 1: 0:45, Set 2: 0:32".
        self._set_hold_durations: List[float] = []
        self._current_set_hold_start: float = 0.0

        # Stance-based set boundary. Tracks when the user first left stance
        # so we can debounce a 3-second absence before closing the set.
        self._out_of_stance_since: Optional[float] = None
        self._last_pose: Optional[PoseResult] = None

    # ── Public accessors (parity with PushUpDetector) ───────────────────
    @property
    def session_state(self) -> SessionState:
        return self._session_state

    @property
    def current_posture(self) -> PostureLabel:
        return self._current_posture

    @property
    def last_set_reps(self) -> int:
        return self._set_reps[-1] if self._set_reps else 0

    @property
    def last_set_hold(self) -> float:
        """Static-hold seconds held in the most recently closed attempt."""
        return self._set_hold_durations[-1] if self._set_hold_durations else 0.0

    @property
    def current_set_hold(self) -> float:
        """Static-hold seconds held in the currently active attempt
        (resets to 0 when a set closes, ticks upward while ACTIVE)."""
        return max(0.0, self._hold_duration - self._current_set_hold_start)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _init_rep_peak(self) -> float:
        """Initial value for _rep_peak — opposite end of the ROM.
        Decreasing → peak starts high (180°); increasing → peak starts low (0°).
        """
        return 180.0 if self.REP_DIRECTION == REP_DIRECTION_DECREASING else 0.0

    def _reset_robust_rep_fsm(self) -> None:
        """Clear in-progress rep state. Called on session-state changes
        and after every committed rep."""
        self._state = RepPhase.TOP
        self._bottom_frame_count = 0
        self._rep_peak = self._init_rep_peak()
        self._robust_first_rep_started = False
        self._pending_rep_elapsed = None
        self._current_rep_issues = []
        self._current_rep_scores = []
        self._current_rep_landmarks = []

    def _is_deeper(self, new_val: float, current_peak: float) -> bool:
        """Is `new_val` more extreme than `current_peak` in the rep direction?"""
        if self.REP_DIRECTION == REP_DIRECTION_DECREASING:
            return new_val < current_peak
        return new_val > current_peak

    def _passed_start_threshold(self, primary: float) -> bool:
        """Has the user committed to a descent/press past REP_START_THRESHOLD?"""
        if self.REP_DIRECTION == REP_DIRECTION_DECREASING:
            return primary < self.REP_START_THRESHOLD
        return primary > self.REP_START_THRESHOLD

    def _reached_depth(self, peak: float) -> bool:
        """Has _rep_peak crossed REP_DEPTH_THRESHOLD in the rep direction?"""
        if self.REP_DIRECTION == REP_DIRECTION_DECREASING:
            return peak <= self.REP_DEPTH_THRESHOLD
        return peak >= self.REP_DEPTH_THRESHOLD

    def _back_at_top(self, primary: float) -> bool:
        """Is the raw primary angle back at the top (commit zone)?"""
        if self.REP_DIRECTION == REP_DIRECTION_DECREASING:
            return primary >= self.REP_COMPLETE_TOP
        return primary <= self.REP_COMPLETE_TOP

    def _passed_bottom_clear(self, primary: float) -> bool:
        """Has primary angle moved past REP_BOTTOM_CLEAR away from depth?"""
        if self.REP_DIRECTION == REP_DIRECTION_DECREASING:
            return primary > self.REP_BOTTOM_CLEAR
        return primary < self.REP_BOTTOM_CLEAR

    # ── Sanity guards on raw primary angle ──────────────────────────────

    def _apply_raw_angle_guards(self, angle: Optional[float]) -> Optional[float]:
        """Floor + jump guards. Returns None if angle looks hallucinated."""
        if angle is None:
            return None
        if angle < self.MIN_REAL_PRIMARY_DEG:
            return None
        if (self._last_raw_primary is not None
                and abs(angle - self._last_raw_primary) > self.MAX_PRIMARY_JUMP_DEG):
            return None
        return angle

    def _update_none_tolerance(self, angle: Optional[float]) -> Optional[float]:
        """Reuse last raw primary for up to MAX_NONE_TOLERANCE frames when
        the current frame's angle is None. Returns the value to feed into
        the rep FSM."""
        if angle is None:
            self._consec_none_frames += 1
            if (self._consec_none_frames <= self.MAX_NONE_TOLERANCE
                    and self._last_raw_primary is not None):
                return self._last_raw_primary
            return None
        self._consec_none_frames = 0
        self._last_raw_primary = angle
        return angle

    # ── Session FSM (posture-driven) ─────────────────────────────────────

    def _is_gate_posture(self, posture: PostureLabel) -> bool:
        """Does the current posture satisfy the session gate?
        Handles the `SESSION_POSTURE_GATE = None` case (visibility-only)
        by returning True when the visibility gate is active — this method
        is only called for the posture-gated path.
        """
        if self.SESSION_POSTURE_GATE is None:
            return True  # caller uses visibility gate instead
        return posture == self.SESSION_POSTURE_GATE

    def _check_visibility_gate(self, pose: PoseResult) -> bool:
        """Fallback gate when SESSION_POSTURE_GATE is None.
        Returns True if all VISIBILITY_GATE_LANDMARKS are visible at
        VISIBILITY_GATE_THRESHOLD confidence or better."""
        if not self.VISIBILITY_GATE_LANDMARKS:
            return True  # no landmarks specified → always pass
        return all(
            pose.is_visible(idx, self.VISIBILITY_GATE_THRESHOLD)
            for idx in self.VISIBILITY_GATE_LANDMARKS
        )

    # ── Stance-based set boundary (push-up style, unified across exercises) ─

    def _is_in_stance(self, pose: PoseResult, visibility_ok: bool) -> bool:
        """Is the user currently in the exercise stance?

        Default: posture gate if set, else the visibility gate. Subclasses
        can override to add exercise-specific witnesses (e.g. "torso must
        be horizontal for crunch/side_plank", "must be hanging for pullup").

        The point of this method is to drive the set-boundary decision:
        mid-rep pauses IN stance do not end a set; leaving stance for
        OUT_OF_STANCE_DEBOUNCE_S seconds does.
        """
        if self.SESSION_POSTURE_GATE is not None:
            return self._current_posture == self.SESSION_POSTURE_GATE
        return visibility_ok

    def _should_close_for_out_of_stance(self, now: float, in_stance: bool) -> bool:
        """Debounced out-of-stance detector.

        While in stance, resets the timer and returns False so the set
        stays open indefinitely. When out of stance, starts a timer and
        only returns True once OUT_OF_STANCE_DEBOUNCE_S has elapsed —
        forgiving brief stumbles / momentary landmark loss.
        """
        if in_stance:
            self._out_of_stance_since = None
            return False
        if self._out_of_stance_since is None:
            self._out_of_stance_since = now
            return False
        return (now - self._out_of_stance_since) >= self.OUT_OF_STANCE_DEBOUNCE_S

    def _update_session_state(self, now: float, visibility_ok: bool) -> None:
        """Drive IDLE/SETUP/ACTIVE/RESTING transitions using stance-based
        set boundary detection.

        Set stays open as long as user is IN stance — pauses of any length
        between reps do not close the set. Set only closes after the user
        has been OUT of stance for OUT_OF_STANCE_DEBOUNCE_S seconds.

        Subclasses may still override this for exercise-specific SETUP→
        ACTIVE entry witnesses (motion/stance guards against false starts).
        """
        s = self._session_state
        pose = self._last_pose
        in_stance = bool(pose is not None and self._is_in_stance(pose, visibility_ok))

        # Visibility-only gate path.
        if self.SESSION_POSTURE_GATE is None:
            if s == SessionState.ACTIVE:
                if self._should_close_for_out_of_stance(now, in_stance):
                    self._close_active_set_or_rollback(now)
                else:
                    self._unknown_streak = 0
                return
            if s in (SessionState.IDLE, SessionState.SETUP, SessionState.RESTING):
                if in_stance:
                    self._session_state = SessionState.ACTIVE
                    self._reset_robust_rep_fsm()
                    self._rep_start_time = now
                    self._unknown_streak = 0
                    self._out_of_stance_since = None
                    if s in (SessionState.IDLE, SessionState.SETUP):
                        self._seed_rep_fsm_from_pre_active(now)
                elif s == SessionState.IDLE:
                    return
            return

        # Posture-gated path (push-up and anything else with a hard label).
        posture = self._current_posture
        in_gate = self._is_gate_posture(posture)

        if s == SessionState.ACTIVE:
            # Stance = in gate posture. UNKNOWN and explicit non-gate both
            # count as "out of stance"; 3-second debounce forgives brief
            # landmark dropouts mid-rep.
            if self._should_close_for_out_of_stance(now, in_gate):
                self._close_active_set_or_rollback(now)
            else:
                self._unknown_streak = 0
            return

        # Non-ACTIVE states: UNKNOWN is held, other labels drive transitions.
        if posture == PostureLabel.UNKNOWN:
            return

        if s == SessionState.IDLE:
            if in_gate:
                self._session_state = SessionState.ACTIVE
                self._reset_robust_rep_fsm()
                self._rep_start_time = now
                self._unknown_streak = 0
                self._seed_rep_fsm_from_pre_active(now)
            else:
                self._session_state = SessionState.SETUP
        elif s == SessionState.SETUP:
            if in_gate:
                self._session_state = SessionState.ACTIVE
                self._reset_robust_rep_fsm()
                self._rep_start_time = now
                self._unknown_streak = 0
                self._seed_rep_fsm_from_pre_active(now)
        elif s == SessionState.RESTING:
            if in_gate:
                self._session_state = SessionState.ACTIVE
                self._reset_robust_rep_fsm()
                self._rep_start_time = now
                self._unknown_streak = 0
                # No seed from pre-active trace on RESTING→ACTIVE — safer
                # to start fresh after a rest.

    def _close_active_set_or_rollback(self, now: float) -> None:
        """Shared logic for ACTIVE→RESTING (set close) or ACTIVE→SETUP (false
        start rollback). Called from both posture and visibility paths."""
        # Static-hold branch (plank): snapshot the current-attempt hold
        # duration, advance the per-set cursor, and transition to RESTING.
        # Without this, plank would always fall to the "false start" path
        # because _reps_in_current_set is never incremented for holds.
        if self.IS_STATIC:
            attempt_secs = self._hold_duration - self._current_set_hold_start
            if attempt_secs >= 1.0:  # ignore <1s accidental re-entries
                self._set_hold_durations.append(round(attempt_secs, 1))
                # Static holds record a minimal set entry so downstream
                # consumers can still iterate `_set_records` uniformly.
                hold_record = {
                    "set_num": int(self._set_count + 1),
                    "reps_count": 0,
                    "rest_before_sec": round(
                        min(600.0, max(0.0, now - self._last_set_end_time))
                        if self._last_set_end_time > 0.0 else 0.0, 2),
                    "avg_form_score": 0.0,
                    "score_dropoff": 0.0,
                    "failure_type": "clean_stop",
                    "hold_sec": round(attempt_secs, 1),
                }
                self._set_records.append(hold_record)
                self._last_set_end_time = now
                self._set_count += 1
                self._current_set_hold_start = self._hold_duration
                self._between_sets = True
                self._pending_set_announce = True
                self._session_state = SessionState.RESTING
            else:
                # Too-brief entry — treat as a false start, stay in SETUP.
                self._session_state = SessionState.SETUP
            self._unknown_streak = 0
            self._out_of_stance_since = None
            return

        if self._reps_in_current_set > 0:
            self._set_reps.append(self._reps_in_current_set)
            self._close_current_set(now, failure_type="clean_stop")
            self._set_count += 1
            self._reps_in_current_set = 0
            self._between_sets = True
            self._pending_set_announce = True
            self._reset_robust_rep_fsm()
            self._session_state = SessionState.RESTING
        else:
            # False start — entered gate, never committed a rep
            self._reset_robust_rep_fsm()
            self._session_state = SessionState.SETUP
        self._unknown_streak = 0
        # Fresh stance-debounce window for the next ACTIVE transition.
        self._out_of_stance_since = None

    # ── Pre-active trace seeding ─────────────────────────────────────────

    def _seed_rep_fsm_from_pre_active(self, now: float) -> None:
        """Retroactively initialise the rep FSM from the pre-ACTIVE primary-
        angle trace so a first rep that started during the posture-classifier
        bootstrap (≤5 UNKNOWN frames) isn't lost.

        Safety rules:
          1. Only the trailing suffix of frames whose RAW posture was the
             gate label is considered.
          2. At least 3 consecutive gate-posture frames required.
          3. Only frames from that trailing run contribute to the descent-
             start timestamp and peak value.
        """
        if not self._pre_active_trace:
            return

        # Walk the buffer backwards and collect the trailing run of
        # "in gate" frames. For posture-gated exercises, "in gate" means
        # the raw posture label matched SESSION_POSTURE_GATE. For
        # visibility-only exercises, "in gate" means the key landmarks
        # were visible that frame. Both flags are stored in
        # `_pre_active_trace` as the third tuple element at append time.
        plank_suffix: List[Tuple[float, float]] = []
        for t, primary, is_gate in reversed(self._pre_active_trace):
            if not is_gate:
                break
            plank_suffix.append((t, primary))
        plank_suffix.reverse()

        if len(plank_suffix) < 3:
            return

        descent_start_t: Optional[float] = None
        descent_peak: float = self._init_rep_peak()
        last_primary: Optional[float] = None
        for entry in plank_suffix:
            t, primary = entry[0], entry[1]
            last_primary = primary
            if self._passed_start_threshold(primary):
                if descent_start_t is None:
                    descent_start_t = t
                if self._is_deeper(primary, descent_peak):
                    descent_peak = primary

        if descent_start_t is None or last_primary is None:
            return

        self._robust_first_rep_started = True
        self._first_rep_started = True  # base-class flag for issue suppression
        self._rep_start_time = descent_start_t
        self._descent_start_time = descent_start_t
        self._rep_peak = descent_peak
        self._current_rep_issues = []
        self._current_rep_scores = []

        # Decide landing state based on where last primary sits.
        depth_reached = self._reached_depth(descent_peak)
        if self._reached_depth(last_primary):
            self._state = RepPhase.BOTTOM
            self._bottom_frame_count = 1
            self._concentric_start_time = descent_start_t
        elif depth_reached:
            self._state = RepPhase.GOING_UP
            self._concentric_start_time = descent_start_t
        else:
            self._state = RepPhase.GOING_DOWN
            self._concentric_start_time = 0.0
            self._bottom_frame_count = 0

        self._pre_active_trace.clear()

    # ── Robust rep FSM ───────────────────────────────────────────────────

    def _update_rep_fsm_robust(self, primary: Optional[float], now: float) -> bool:
        """Direction-aware, state-independent rep commit.

        Returns True if a rep is PENDING COMMIT this frame (state already
        reset to TOP, but _rep_count NOT yet incremented). The caller is
        expected to run form assessment next, then call
        `_finalize_pending_rep(angles, now)` to actually commit the rep
        through the base-class `_should_count_rep` gate (MIN_REP_DURATION,
        MIN_DESCENT_DURATION, MIN_FORM_GATE, and any subclass override
        like Deadlift's "random movement" rejection).

        Deferring the commit to after form assessment is what lets
        `_should_count_rep` see the full `_current_rep_scores` list
        including this frame's score, and matches the base class's flow
        where the state-machine commit fires then the check runs.

        Mirrors pushup_detector.py:643–750 generalized to any direction.
        """
        if primary is None:
            return False

        # Update rep peak (min or max depending on direction) whenever we've
        # left the TOP state. This is the single source of truth for "did
        # the user actually reach depth in this rep cycle?"
        if self._state != RepPhase.TOP:
            if self._is_deeper(primary, self._rep_peak):
                self._rep_peak = primary

        # ──────────────────────────────────────────────────────────────
        # PRIMARY COMMIT CHECK — state-independent, signal-driven.
        # Runs BEFORE state-machine branches so frame drops, posture
        # hiccups, or angle jitter that left the state stuck can't drop
        # a rep. Criteria:
        #   (1) a descent was committed (_robust_first_rep_started)
        #   (2) we're not already back at top (state != TOP)
        #   (3) raw primary has returned to the top zone
        #   (4) tracked peak reached depth
        # On hit: store elapsed, reset state immediately (so the form
        # assessment that runs right after sees the correct phase), but
        # DO NOT increment _rep_count here. Let classify() call
        # _finalize_pending_rep() after scoring the current frame.
        # ──────────────────────────────────────────────────────────────
        if (self._robust_first_rep_started
                and self._state != RepPhase.TOP
                and self._back_at_top(primary)
                and self._reached_depth(self._rep_peak)):
            self._pending_rep_elapsed = now - self._rep_start_time
            self._state = RepPhase.TOP
            self._rep_peak = self._init_rep_peak()
            return True

        # ──────────────────────────────────────────────────────────────
        # State-machine fallback (drives phase labels for UI and catches
        # "didn't reach depth" bailouts).
        # ──────────────────────────────────────────────────────────────
        if self._state == RepPhase.TOP:
            if self._passed_start_threshold(primary):
                self._state = RepPhase.GOING_DOWN
                self._rep_start_time = now
                self._descent_start_time = now  # for MIN_DESCENT_DURATION gate
                self._concentric_start_time = 0.0  # reset — stamped at BOTTOM
                self._bottom_frame_count = 0
                self._rep_peak = primary
                self._robust_first_rep_started = True
                self._first_rep_started = True  # base-class flag
                self._current_rep_issues = []
                self._current_rep_scores = []
                self._current_rep_landmarks = []
                if self._between_sets:
                    self._between_sets = False
        elif self._state == RepPhase.GOING_DOWN:
            if self._reached_depth(primary):
                self._state = RepPhase.BOTTOM
                self._concentric_start_time = now  # end of eccentric phase
                self._bottom_frame_count = 0
            elif self._back_at_top(primary):
                # Elbow back to top without reaching depth AND primary
                # commit check above didn't fire — genuine bailout.
                if "Didn't reach full depth" not in self._current_rep_issues:
                    self._current_rep_issues.append("Didn't reach full depth")
                self._state = RepPhase.TOP
                self._rep_peak = self._init_rep_peak()
        elif self._state == RepPhase.BOTTOM:
            self._bottom_frame_count += 1
            if (self._passed_bottom_clear(primary)
                    and self._bottom_frame_count >= 1):
                self._state = RepPhase.GOING_UP
        elif self._state == RepPhase.GOING_UP:
            if self._reached_depth(primary):
                # Dropped back into bottom — second dip
                self._state = RepPhase.BOTTOM
                self._concentric_start_time = now  # re-stamp on double-dip
                self._bottom_frame_count = 0

        return False

    def _finalize_pending_rep(self, angles: Dict[str, Optional[float]], now: float) -> bool:
        """Called from classify() AFTER form assessment has appended the
        current frame's score. Runs the base-class `_should_count_rep`
        gate (MIN_REP_DURATION, MIN_DESCENT_DURATION, MIN_FORM_GATE, any
        subclass override) and either commits the rep or silently drops
        it. Either way clears `_pending_rep_elapsed`.

        Returns True if the rep was committed, False if it was rejected
        or there was nothing pending.
        """
        if self._pending_rep_elapsed is None:
            return False
        elapsed = self._pending_rep_elapsed
        self._pending_rep_elapsed = None

        if self._should_count_rep(elapsed, angles):
            self._rep_count += 1
            self._reps_in_current_set += 1
            self._recent_rep_durations.append(elapsed)
            return True

        # Rep rejected by gate — drop it silently. State is already reset
        # to TOP so the next descent will start cleanly.
        return False

    # ── classify() override ──────────────────────────────────────────────

    def classify(self, pose: PoseResult, timestamp: float = None) -> ClassificationResult:
        """Robust rep-counting classify() — replaces base class's
        threshold-crossing FSM and timer-based set detection with the
        push-up methodology."""
        now = timestamp if timestamp is not None else time.time()
        # Stash the current pose so _update_session_state can consult it
        # for stance witnesses (e.g. torso orientation, wrist positions).
        self._last_pose = pose

        # 1. Compute and smooth angles (same as base)
        raw_angles = self._compute_angles(pose)
        for name, val in raw_angles.items():
            self._push_angle(name, val)
            self.last_angles[name] = val

        self._frame_visibility = float(np.mean(pose.landmarks[:, 3]))

        angles = {name: self._get_smooth(name) for name in raw_angles}
        primary_smooth = angles.get("primary")
        raw_primary = raw_angles.get("primary")

        # 2. Apply raw-angle sanity guards to primary
        guarded_raw = self._apply_raw_angle_guards(raw_primary)

        # 3. None tolerance — reuse last value if guard tripped or angle missing
        fsm_primary = self._update_none_tolerance(guarded_raw)

        # 4. Posture classifier update
        posture_result = self._posture_classifier.update(pose)
        self._current_posture = posture_result.label

        # 5. Visibility gate (only relevant if SESSION_POSTURE_GATE is None)
        visibility_ok = self._check_visibility_gate(pose)

        # 6. Record pre-active trace while waiting for ACTIVE
        if (self._session_state in (SessionState.IDLE, SessionState.SETUP)
                and fsm_primary is not None):
            if self.SESSION_POSTURE_GATE is None:
                is_gate_raw = visibility_ok
            else:
                raw_label = getattr(
                    self._posture_classifier, "_last_raw_label", PostureLabel.UNKNOWN
                )
                is_gate_raw = (raw_label == self.SESSION_POSTURE_GATE)
            self._pre_active_trace.append((now, fsm_primary, is_gate_raw))

        # 7. Pose-presence tracking — if primary missing for too long, reset
        if primary_smooth is None:
            self._frames_since_pose += 1
            if self._frames_since_pose >= NO_POSE_RESET_FRAMES:
                if self._reps_in_current_set > 0 and not self._between_sets:
                    self._set_reps.append(self._reps_in_current_set)
                    self._close_current_set(now, failure_type="timeout")
                    self._set_count += 1
                    self._reps_in_current_set = 0
                    self._between_sets = True
                    self._pending_set_announce = True
                if self._form_validated:
                    self._form_validated = False
                self._reset_robust_rep_fsm()
                self._session_state = SessionState.IDLE
                self._last_raw_primary = None
                self._consec_none_frames = 0
                self._pre_active_trace.clear()
                for buf in self._angle_buffers.values():
                    buf.clear()
            missing = self._get_missing_parts(angles)
            return self._inactive_result(
                f"Can't detect {', '.join(missing) if missing else 'body'}. "
                "Position camera so full body is visible."
            )
        self._frames_since_pose = 0

        # 8. Form validation gate (base-class hook)
        if not self._form_validated:
            valid, setup_issues = self._check_start_position(angles)
            if valid:
                self._form_validated = True
                self._rep_start_time = now
                if self.IS_STATIC:
                    self._hold_start_time = now
            else:
                return self._build_result(
                    form_score=0.0, is_active=False,
                    joint_feedback={},
                    details={
                        "setup": " | ".join(setup_issues) if setup_issues
                        else f"Get into {self.EXERCISE_NAME.replace('_', ' ')} position",
                        "session_state": self._session_state.value,
                        "posture": self._current_posture.value,
                    },
                    feedback_text=setup_issues[0] if setup_issues else "Get into position",
                )

        # 9. Session FSM — drives when rep counting runs
        self._update_session_state(now, visibility_ok)

        # 10. Static hold path (plank) — no rep counting
        if self.IS_STATIC:
            return self._classify_static_hold(angles, now)

        # 11. Rep FSM — only runs when session_state == ACTIVE.
        # Sets _pending_rep_elapsed if a rep is ready to commit; actual
        # commit happens in step 13 after form assessment has scored the
        # current frame (so _should_count_rep sees the full avg).
        rep_pending = False
        if self._session_state == SessionState.ACTIVE:
            rep_pending = self._update_rep_fsm_robust(fsm_primary, now)
            # Activity detection still runs (for compatibility with trace
            # consumers that read activity state), but does NOT drive set
            # detection anymore — posture does.
            self._last_active_time = now

        # Activity for trace compatibility
        activity, rest_tier = self._activity_detector.update(pose, now)
        self._current_activity = activity
        self._current_rest_tier = rest_tier
        is_actively_exercising = (self._session_state == SessionState.ACTIVE)

        # 12. Per-frame form assessment (same as base)
        self._lr_angles.clear()
        frame_score, joint_feedback, issues = self._assess_form(angles)

        # Sort issues by priority
        issues.sort(key=lambda msg: min(
            (pri for kw, pri in ISSUE_PRIORITY_KEYWORDS.items() if kw in msg.lower()),
            default=9
        ))

        # Suppress issues outside ACTIVE (mirrors pushup_detector.py:415–420)
        if self._session_state != SessionState.ACTIVE:
            issues = []
            for k in joint_feedback:
                joint_feedback[k] = "correct"
            frame_score = max(frame_score, 0.8)
        else:
            # Coverage penalty (same as base)
            if self._coverage_penalty:
                n_angles = sum(1 for k, v in angles.items() if k != "primary" and v is not None)
                n_expected = max(1, len(angles) - 1)
                coverage = n_angles / n_expected
                if coverage < 0.6:
                    frame_score *= coverage
                self._last_coverage = coverage
            else:
                self._last_coverage = 1.0

            # Asymmetry detection (same as base)
            asym_penalty = self._check_asymmetry(joint_feedback, issues)
            frame_score = max(0.0, frame_score - asym_penalty)

            # Visibility discount
            if self._frame_visibility < LOW_VIS_THRESHOLD:
                vis_factor = 0.5 + 0.5 * (self._frame_visibility / LOW_VIS_THRESHOLD)
                frame_score *= vis_factor

            self._current_rep_scores.append(frame_score)
            for issue in issues:
                if issue not in self._current_rep_issues:
                    self._current_rep_issues.append(issue)

            # DTW landmark capture
            if self._dtw_matcher is not None and self._state != RepPhase.TOP:
                try:
                    self._current_rep_landmarks.append(
                        self._landmark_extractor.extract_landmarks(pose)
                    )
                except Exception:
                    pass

        # 13. Finalize pending rep — runs _should_count_rep (MIN_FORM_GATE,
        # MIN_REP_DURATION, MIN_DESCENT_DURATION, subclass overrides) on
        # the fully-scored current frame, and commits or silently drops
        # the rep. State has already been reset inside _update_rep_fsm_robust.
        rep_completed = False
        if rep_pending:
            rep_completed = self._finalize_pending_rep(angles, now)
            if rep_completed:
                self._record_robust_rep(now)

        # 14. Smooth form score
        self._score_buf.append(frame_score)
        smooth_score = sum(self._score_buf) / len(self._score_buf)

        # 15. Feedback + details
        feedback_text = self._build_feedback_text(smooth_score, issues, angles, rep_completed)

        details: Dict[str, str] = {}
        for i, issue in enumerate(issues[:3]):
            details[f"issue_{i}"] = issue

        phase_names = {
            RepPhase.TOP: "Top position",
            RepPhase.GOING_DOWN: "Lowering",
            RepPhase.BOTTOM: "Bottom position",
            RepPhase.GOING_UP: "Coming up",
        }
        details["phase"] = phase_names.get(self._state, "")

        if primary_smooth is not None:
            progress = np.clip(
                (self.PRIMARY_ANGLE_TOP - primary_smooth) /
                max(self.PRIMARY_ANGLE_TOP - self.PRIMARY_ANGLE_BOTTOM, 1), 0, 1
            )
            details["progress"] = f"{int(progress * 100)}%"

        details["session_state"] = self._session_state.value
        details["posture"] = self._current_posture.value
        details["activity"] = self._current_activity.value
        details["rest_tier"] = self._current_rest_tier.value
        details["between_sets"] = "1" if self._between_sets else "0"
        details["set_number"] = str(self._set_count + 1)
        details["reps_in_current_set"] = str(self._reps_in_current_set)
        if self._recent_rep_durations:
            avg_dur = sum(self._recent_rep_durations) / len(self._recent_rep_durations)
            details["avg_rep_duration"] = f"{avg_dur:.2f}"

        return self._build_result(
            form_score=smooth_score,
            is_active=is_actively_exercising,
            joint_feedback=joint_feedback,
            details=details,
            feedback_text=feedback_text,
        )

    def _record_robust_rep(self, now: float) -> None:
        """Record a committed rep into _rep_history. Mirrors the base-class
        logic but uses _rep_count which was incremented in the FSM."""
        avg_score = sum(self._current_rep_scores) / max(len(self._current_rep_scores), 1)

        # DTW template matching on captured landmarks
        dtw_sim = 1.0
        dtw_worst: Optional[str] = None
        if self._dtw_matcher is not None and len(self._current_rep_landmarks) >= 3:
            try:
                rep_seq = np.stack(self._current_rep_landmarks).astype(np.float32)
                dtw_score = self._dtw_matcher.compare(self.EXERCISE_NAME, rep_seq)
                if dtw_score is not None:
                    dtw_sim = dtw_score.similarity
                    dtw_worst = dtw_score.worst_joint
                    if dtw_sim < 0.5:
                        avg_score *= (0.6 + 0.4 * dtw_sim)
                        if dtw_worst and f"Pattern deviates at {dtw_worst}" not in self._current_rep_issues:
                            self._current_rep_issues.insert(0, f"Pattern deviates at {dtw_worst}")
            except Exception:
                pass

        self._last_dtw_similarity = dtw_sim
        self._last_dtw_worst_joint = dtw_worst

        quality = (RepQuality.GOOD if avg_score >= 0.7
                   else RepQuality.MODERATE if avg_score >= 0.4
                   else RepQuality.BAD)
        if len(self._rep_history) < MAX_REP_HISTORY:
            rep_entry = {
                "rep": self._rep_count,
                "score": round(avg_score, 2),
                "quality": quality,
                "issues": self._current_rep_issues[:3],
                "duration": round(now - self._rep_start_time, 1),
            }
            if self._dtw_matcher is not None:
                rep_entry["dtw_similarity"] = round(dtw_sim, 2)
                if dtw_worst:
                    rep_entry["dtw_worst_joint"] = dtw_worst
            self._augment_rep_entry(rep_entry, now)
            self._rep_history.append(rep_entry)

    def _classify_static_hold(self, angles: Dict[str, Optional[float]], now: float) -> ClassificationResult:
        """Static hold path (plank) — no rep counting. Same logic as base-class
        static path but wrapped in the posture-gated session FSM so walk-aways
        are handled by the outer state machine."""
        # Hold duration only runs during ACTIVE
        if self._session_state == SessionState.ACTIVE:
            if self._hold_start_time is None:
                self._hold_start_time = now
            self._hold_duration = now - self._hold_start_time
            self._last_active_time = now
        else:
            # Paused/resting — freeze the hold_start_time so next ACTIVE
            # frame continues from where we left off
            if self._hold_start_time is not None:
                # Shift start so duration stays the same
                self._hold_start_time = now - self._hold_duration

        frame_score, joint_feedback, issues = self._assess_form(angles)

        if self._coverage_penalty:
            n_angles = sum(1 for k, v in angles.items() if k != "primary" and v is not None)
            n_expected = max(1, len(angles) - 1)
            coverage = n_angles / n_expected
            if coverage < 0.6:
                frame_score *= coverage
            self._last_coverage = coverage

        asym_penalty = self._check_asymmetry(joint_feedback, issues)
        frame_score = max(0.0, frame_score - asym_penalty)

        # Suppress issues outside ACTIVE
        if self._session_state != SessionState.ACTIVE:
            issues = []
            for k in joint_feedback:
                joint_feedback[k] = "correct"
            frame_score = max(frame_score, 0.8)

        self._score_buf.append(frame_score)
        smooth_score = sum(self._score_buf) / len(self._score_buf)

        details: Dict[str, str] = {}
        for i, issue in enumerate(issues[:3]):
            details[f"issue_{i}"] = issue
        details["hold_duration"] = f"{self._hold_duration:.0f}s"
        details["session_state"] = self._session_state.value
        details["posture"] = self._current_posture.value

        feedback_text = issues[0] if issues else (
            f"Great hold! {self._hold_duration:.0f}s" if smooth_score >= 0.7
            else f"Hold position — {self._hold_duration:.0f}s"
        )

        return self._build_result(
            form_score=smooth_score,
            is_active=(self._session_state == SessionState.ACTIVE),
            joint_feedback=joint_feedback,
            details=details,
            feedback_text=feedback_text,
        )
