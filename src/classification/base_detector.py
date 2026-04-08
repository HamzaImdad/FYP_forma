"""
Base exercise detector with shared state machine, rep counting, set tracking,
form scoring, and session summary logic.

Each exercise detector overrides:
  - THRESHOLDS: angle thresholds for state transitions and form checks
  - _compute_angles(): extract relevant angles from pose
  - _assess_form(): per-frame form quality scoring
  - _build_feedback_text(): actionable feedback string
"""

import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..pose_estimation.base import PoseResult
from ..classification.base import ClassificationResult
from ..utils.constants import MIN_VISIBILITY
from ..utils.geometry import calculate_angle
from ..utils.activity_detector import ActivityDetector, ActivityState, RestTier


# ── Shared constants ────────────────────────────────────────────────────
MIN_REP_DURATION = 0.4      # seconds — prevents counting jitter
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

    def __init__(self, coverage_penalty: bool = False):
        self._coverage_penalty = coverage_penalty
        self.reset()

    def reset(self):
        """Reset all state."""
        self._state = RepPhase.TOP
        self._form_validated = False

        # Rep counting
        self._rep_count = 0
        self._last_rep_time = 0.0
        self._rep_start_time = 0.0

        # Set tracking
        self._set_count = 0
        self._reps_in_current_set = 0
        self._last_active_time = 0.0
        self._set_reps: List[int] = []

        # Per-rep form tracking
        self._current_rep_issues: List[str] = []
        self._current_rep_scores: List[float] = []
        self._rep_history: List[Dict] = []

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
                issues.append(f"Uneven {angle_name.replace('_', ' ')} — {weaker_side} side weaker ({diff:.0f}° diff)")
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
            if self._frames_since_pose >= NO_POSE_RESET_FRAMES and self._form_validated:
                self._form_validated = False
                self._state = RepPhase.TOP
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
        if rest_tier == RestTier.EXTENDED_REST and self._reps_in_current_set > 0:
            self._set_reps.append(self._reps_in_current_set)
            self._set_count += 1
            self._reps_in_current_set = 0

        # ── State machine ──
        rep_completed = False

        if self._state == RepPhase.TOP:
            if primary < self.PRIMARY_ANGLE_TOP - self.DESCENT_THRESHOLD:
                self._state = RepPhase.GOING_DOWN
                self._rep_start_time = now
                self._current_rep_issues = []
                self._current_rep_scores = []

        elif self._state == RepPhase.GOING_DOWN:
            if primary <= self.PRIMARY_ANGLE_BOTTOM:
                self._state = RepPhase.BOTTOM
            elif primary > self.PRIMARY_ANGLE_TOP:
                self._state = RepPhase.TOP
                self._current_rep_issues.append("Didn't reach full depth")

        elif self._state == RepPhase.BOTTOM:
            if primary > self.PRIMARY_ANGLE_BOTTOM + self.ASCENT_THRESHOLD:
                self._state = RepPhase.GOING_UP

        elif self._state == RepPhase.GOING_UP:
            if primary >= self.PRIMARY_ANGLE_TOP:
                elapsed = now - self._rep_start_time
                if elapsed >= MIN_REP_DURATION:
                    rep_completed = True
                    self._rep_count += 1
                    self._reps_in_current_set += 1
                self._state = RepPhase.TOP
            elif primary < self.PRIMARY_ANGLE_BOTTOM:
                self._state = RepPhase.BOTTOM

        # ── Per-frame form assessment ──
        self._lr_angles.clear()
        frame_score, joint_feedback, issues = self._assess_form(angles)

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

        # ── Record completed rep ──
        if rep_completed:
            avg_score = sum(self._current_rep_scores) / max(len(self._current_rep_scores), 1)
            quality = (RepQuality.GOOD if avg_score >= 0.7
                       else RepQuality.MODERATE if avg_score >= 0.4
                       else RepQuality.BAD)
            if len(self._rep_history) < MAX_REP_HISTORY:
                self._rep_history.append({
                    "rep": self._rep_count,
                    "score": round(avg_score, 2),
                    "quality": quality,
                    "issues": self._current_rep_issues[:3],
                    "duration": round(now - self._rep_start_time, 1),
                })

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
        )

    def _inactive_result(self, message: str) -> ClassificationResult:
        return self._build_result(0.0, False, {}, {"setup": message})

    # ── Session summary ─────────────────────────────────────────────────

    def get_session_summary(self) -> Dict:
        all_set_reps = list(self._set_reps)
        if self._reps_in_current_set > 0:
            all_set_reps.append(self._reps_in_current_set)

        good = sum(1 for r in self._rep_history if r["quality"] == RepQuality.GOOD)

        issue_counts: Dict[str, int] = {}
        for rep in self._rep_history:
            for issue in rep.get("issues", []):
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        common_issues = sorted(issue_counts.items(), key=lambda x: -x[1])[:5]

        avg_score = (
            sum(r["score"] for r in self._rep_history) / max(len(self._rep_history), 1)
        ) if self._rep_history else 0.0

        summary = {
            "exercise": self.EXERCISE_NAME,
            "total_reps": self._rep_count,
            "good_reps": good,
            "avg_form_score": round(avg_score, 2),
            "reps_per_set": all_set_reps,
            "common_issues": [{"issue": n, "count": c} for n, c in common_issues],
            "reps": [
                {
                    "rep_num": r["rep"],
                    "form_score": r["score"],
                    "quality": r["quality"],
                    "issues": r.get("issues", []),
                    "duration": r.get("duration", 0),
                }
                for r in self._rep_history
            ],
        }

        if self.IS_STATIC:
            summary["hold_duration"] = round(self._hold_duration, 1)
            summary["total_reps"] = 0

        return summary
