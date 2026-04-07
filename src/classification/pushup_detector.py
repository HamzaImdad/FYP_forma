"""
Dedicated push-up detector with state machine rep counting, form quality
assessment, set tracking, and actionable per-joint feedback.

Based on research from aryanvij02/PushUpCounter, BodyBuddy, yakupzengin,
and biomechanics literature (Built With Science).

Angles tracked:
  - Elbow (shoulder-elbow-wrist): rep counting + depth check
  - Hip (shoulder-hip-ankle): body alignment / hip sag
  - Shoulder (elbow-shoulder-hip): arm position relative to torso

State machine: UP -> GOING_DOWN -> DOWN -> GOING_UP -> UP (1 rep)
"""

import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..pose_estimation.base import PoseResult
from ..classification.base import ClassificationResult
from ..utils.constants import (
    NOSE,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    MIN_VISIBILITY,
)
from ..utils.geometry import calculate_angle


# ── Thresholds (from research consensus) ─────────────────────────────────

# Elbow angle (shoulder->elbow->wrist)
ELBOW_UP = 150          # Arms extended (top of push-up) — accounts for smoothing lag
ELBOW_DOWN = 90         # Full depth (bottom of push-up)
ELBOW_HALF_DOWN = 110   # Partial depth (warning)

# Hip angle (shoulder->hip->ankle)
HIP_GOOD = 170          # Straight body line — research says within 10 deg of straight
HIP_WARNING = 155       # Slight sag/pike
HIP_BAD = 140           # Major form breakdown

# Shoulder angle (elbow->shoulder->hip) — arm position relative to torso
SHOULDER_MIN = 30       # Arms too tight to body
SHOULDER_MAX = 70       # Arms too flared — >60 deg starts compromising shoulder
SHOULDER_IDEAL_MIN = 35
SHOULDER_IDEAL_MAX = 60

# Timing
MIN_REP_DURATION = 0.4  # seconds — prevents counting jitter
SET_REST_TIMEOUT = 8.0  # seconds of inactivity to end a set

# Smoothing
ANGLE_SMOOTH_WINDOW = 3  # frames for moving average (low to avoid lag in state transitions)
NO_POSE_RESET_FRAMES = 30  # re-validate form if no pose detected for this many frames
MAX_REP_HISTORY = 200  # cap rep history to prevent unbounded growth


class PushUpState:
    """Push-up rep phase."""
    UP = "up"
    GOING_DOWN = "going_down"
    DOWN = "down"
    GOING_UP = "going_up"


class RepQuality:
    """Per-rep quality assessment."""
    GOOD = "good"
    MODERATE = "moderate"
    BAD = "bad"


class PushUpDetector:
    """Self-contained push-up detection, counting, and form evaluation.

    Produces ClassificationResult compatible with the existing pipeline,
    but with meaningful per-joint feedback and actionable text.
    """

    EXERCISE_NAME = "pushup"
    IS_STATIC = False

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all state."""
        # State machine
        self._state = PushUpState.UP
        self._form_validated = False

        # Rep counting
        self._rep_count = 0
        self._half_rep = 0.0  # 0.5 increments
        self._last_rep_time = 0.0
        self._rep_start_time = 0.0

        # Set tracking
        self._set_count = 0
        self._reps_in_current_set = 0
        self._last_active_time = 0.0
        self._set_reps: List[int] = []  # reps per completed set

        # Per-rep form tracking
        self._current_rep_issues: List[str] = []
        self._current_rep_scores: List[float] = []
        self._rep_history: List[Dict] = []  # completed rep details

        # Angle smoothing buffers
        self._elbow_buf: deque = deque(maxlen=ANGLE_SMOOTH_WINDOW)
        self._hip_buf: deque = deque(maxlen=ANGLE_SMOOTH_WINDOW)
        self._shoulder_buf: deque = deque(maxlen=ANGLE_SMOOTH_WINDOW)

        # Form score smoothing
        self._score_buf: deque = deque(maxlen=8)

        # Activity tracking
        self._pose_detected = False
        self._frames_since_pose = 0

        # Last computed angles (for UI display)
        self.last_elbow_angle = None
        self.last_hip_angle = None
        self.last_shoulder_angle = None

    @property
    def rep_count(self) -> int:
        return self._rep_count

    @property
    def set_count(self) -> int:
        return self._set_count

    @property
    def reps_in_current_set(self) -> int:
        return self._reps_in_current_set

    def classify(self, pose: PoseResult) -> ClassificationResult:
        """Process a pose and return full classification result.

        This is the main entry point — call once per frame.
        """
        now = time.time()

        # Compute angles
        elbow_angle = self._avg_elbow_angle(pose)
        hip_angle = self._compute_hip_angle(pose)
        shoulder_angle = self._compute_shoulder_angle(pose)

        # Store for UI
        self.last_elbow_angle = elbow_angle
        self.last_hip_angle = hip_angle
        self.last_shoulder_angle = shoulder_angle

        # Smooth angles
        if elbow_angle is not None:
            self._elbow_buf.append(elbow_angle)
        if hip_angle is not None:
            self._hip_buf.append(hip_angle)
        if shoulder_angle is not None:
            self._shoulder_buf.append(shoulder_angle)

        smooth_elbow = self._smooth(self._elbow_buf)
        smooth_hip = self._smooth(self._hip_buf)
        smooth_shoulder = self._smooth(self._shoulder_buf)

        # Check if we have enough landmarks for push-up detection
        # Need elbows AND hips visible — partial body won't work
        if smooth_elbow is None or smooth_hip is None:
            self._frames_since_pose += 1
            # Reset form validation if pose lost for too long
            if self._frames_since_pose >= NO_POSE_RESET_FRAMES and self._form_validated:
                self._form_validated = False
                self._state = PushUpState.UP
                self._elbow_buf.clear()
                self._hip_buf.clear()
                self._shoulder_buf.clear()
            missing = []
            if smooth_elbow is None:
                missing.append("arms")
            if smooth_hip is None:
                missing.append("hips and legs")
            return self._inactive_result(
                f"Full body must be visible — can't see {' and '.join(missing)}. "
                "Position camera so head to feet are in frame."
            )

        self._pose_detected = True
        self._frames_since_pose = 0

        # ── Form validation gate ──
        # Must achieve correct starting position before counting
        if not self._form_validated:
            if (smooth_elbow > ELBOW_UP and
                    smooth_hip > HIP_WARNING and
                    smooth_shoulder is not None and SHOULDER_MIN < smooth_shoulder < SHOULDER_MAX):
                self._form_validated = True
                self._rep_start_time = now
            else:
                issues = []
                if smooth_elbow <= ELBOW_UP:
                    issues.append("Extend your arms fully to start")
                if smooth_hip is not None and smooth_hip <= HIP_WARNING:
                    issues.append("Straighten your body — keep hips level")
                return self._build_result(
                    form_score=0.0,
                    is_active=False,
                    joint_feedback={},
                    details={"setup": " | ".join(issues) if issues else "Get into push-up position — full body must be visible"},
                    feedback_text="Get into push-up position with arms extended",
                )

        # ── Set detection (rest period check) ──
        if now - self._last_active_time > SET_REST_TIMEOUT and self._reps_in_current_set > 0:
            self._set_reps.append(self._reps_in_current_set)
            self._set_count += 1
            self._reps_in_current_set = 0

        self._last_active_time = now

        # ── State machine ──
        rep_completed = False
        state_before = self._state

        if self._state == PushUpState.UP:
            if smooth_elbow < ELBOW_UP - 10:  # Started descending
                self._state = PushUpState.GOING_DOWN
                self._rep_start_time = now
                self._current_rep_issues = []
                self._current_rep_scores = []

        elif self._state == PushUpState.GOING_DOWN:
            if smooth_elbow <= ELBOW_DOWN:
                self._state = PushUpState.DOWN
                self._half_rep = 0.5
            elif smooth_elbow > ELBOW_UP:
                # Went back up without reaching bottom — partial rep
                self._state = PushUpState.UP
                self._current_rep_issues.append("Didn't go deep enough")

        elif self._state == PushUpState.DOWN:
            if smooth_elbow > ELBOW_DOWN + 10:  # Started ascending
                self._state = PushUpState.GOING_UP

        elif self._state == PushUpState.GOING_UP:
            if smooth_elbow >= ELBOW_UP:
                # Full rep completed
                elapsed = now - self._rep_start_time
                if elapsed >= MIN_REP_DURATION:
                    self._half_rep = 0.0
                    rep_completed = True
                    self._rep_count += 1
                    self._reps_in_current_set += 1
                self._state = PushUpState.UP
            elif smooth_elbow < ELBOW_DOWN:
                # Went back down — reset
                self._state = PushUpState.DOWN

        # ── Per-frame form assessment ──
        frame_score, joint_feedback, issues = self._assess_form(
            smooth_elbow, smooth_hip, smooth_shoulder, pose
        )
        self._current_rep_scores.append(frame_score)
        for issue in issues:
            if issue not in self._current_rep_issues:
                self._current_rep_issues.append(issue)

        # ── If rep just completed, record it ──
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

        # ── Build feedback text ──
        feedback_text = self._build_feedback_text(
            smooth_score, issues, smooth_elbow, rep_completed
        )

        # ── Build details dict with actionable info ──
        details = {}
        for i, issue in enumerate(issues[:3]):
            details[f"issue_{i}"] = issue

        # Add rep phase info
        phase_names = {
            PushUpState.UP: "Top position",
            PushUpState.GOING_DOWN: "Lowering",
            PushUpState.DOWN: "Bottom position",
            PushUpState.GOING_UP: "Pushing up",
        }
        details["phase"] = phase_names.get(self._state, "Unknown")

        # Progress percentage (0% at top, 100% at bottom)
        if smooth_elbow is not None:
            progress = np.clip((ELBOW_UP - smooth_elbow) / max(ELBOW_UP - ELBOW_DOWN, 1), 0, 1)
            details["progress"] = f"{int(progress * 100)}%"

        return self._build_result(
            form_score=smooth_score,
            is_active=True,
            joint_feedback=joint_feedback,
            details=details,
            feedback_text=feedback_text,
        )

    def _check_pike(self, pose: PoseResult) -> Optional[float]:
        """Detect hip pike (hips above shoulder-ankle line).
        Returns vertical offset: negative = pike (hips above midline)."""
        shoulder_ys = []
        ankle_ys = []
        hip_ys = []
        for s_idx, h_idx, a_idx in [(LEFT_SHOULDER, LEFT_HIP, LEFT_ANKLE),
                                     (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_ANKLE)]:
            if pose.is_visible(s_idx) and pose.is_visible(h_idx) and pose.is_visible(a_idx):
                shoulder_ys.append(pose.get_world_landmark(s_idx)[1])
                hip_ys.append(pose.get_world_landmark(h_idx)[1])
                ankle_ys.append(pose.get_world_landmark(a_idx)[1])
        if not shoulder_ys:
            return None
        midline_y = (sum(shoulder_ys) / len(shoulder_ys) + sum(ankle_ys) / len(ankle_ys)) / 2
        avg_hip_y = sum(hip_ys) / len(hip_ys)
        return avg_hip_y - midline_y  # negative = pike (hip above midline in MediaPipe where Y increases downward)

    def _check_head_alignment(self, pose: PoseResult) -> Optional[str]:
        """Check if head is drooping or craning during pushup.
        Returns issue string or None."""
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

        # In MediaPipe world coords, Y increases downward
        # Head drooping: nose is significantly BELOW shoulders (nose_y much > shoulder_y)
        if diff > 0.08:
            return "Head dropping -- keep neck neutral"
        # Head craning up: nose is significantly ABOVE shoulders
        if diff < -0.12:
            return "Don't crane neck up -- look at floor ahead of hands"
        return None

    def _assess_form(
        self,
        elbow: Optional[float],
        hip: Optional[float],
        shoulder: Optional[float],
        pose: Optional[PoseResult] = None,
    ) -> Tuple[float, Dict[str, str], List[str]]:
        """Assess push-up form quality for a single frame.

        Returns: (score 0-1, joint_feedback dict, list of issue strings)
        """
        joint_feedback = {}
        issues = []
        scores = {}

        # ── Elbow assessment ──
        if elbow is not None:
            # During descent/ascent, elbows should be controlled
            if self._state in (PushUpState.DOWN, PushUpState.GOING_UP, PushUpState.GOING_DOWN):
                if elbow > ELBOW_HALF_DOWN and self._state == PushUpState.DOWN:
                    issues.append("Go lower — bend elbows to 90\u00b0")
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    scores["elbow"] = 0.4
                elif elbow <= ELBOW_DOWN:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 1.0
                else:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 0.8
            else:
                # At top — arms should be extended
                if elbow >= ELBOW_UP:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 1.0
                else:
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    scores["elbow"] = 0.6

        # ── Hip/body alignment assessment (most important form check) ──
        if hip is not None:
            if hip >= HIP_GOOD:
                joint_feedback["left_hip"] = "correct"
                joint_feedback["right_hip"] = "correct"
                scores["hip"] = 1.0
            elif hip >= HIP_WARNING:
                joint_feedback["left_hip"] = "warning"
                joint_feedback["right_hip"] = "warning"
                issues.append("Keep your body straight — slight hip sag")
                scores["hip"] = 0.5
            elif hip >= HIP_BAD:
                joint_feedback["left_hip"] = "incorrect"
                joint_feedback["right_hip"] = "incorrect"
                issues.append("Hips sagging! Tighten your core")
                scores["hip"] = 0.2
            else:
                joint_feedback["left_hip"] = "incorrect"
                joint_feedback["right_hip"] = "incorrect"
                issues.append("Major hip sag — engage core and glutes")
                scores["hip"] = 0.0

        # ── Shoulder/elbow flare assessment ──
        if shoulder is not None:
            if SHOULDER_IDEAL_MIN <= shoulder <= SHOULDER_IDEAL_MAX:
                joint_feedback["left_shoulder"] = "correct"
                joint_feedback["right_shoulder"] = "correct"
                scores["shoulder"] = 1.0
            elif SHOULDER_MIN <= shoulder <= SHOULDER_MAX:
                joint_feedback["left_shoulder"] = "correct"
                joint_feedback["right_shoulder"] = "correct"
                scores["shoulder"] = 0.7
            elif shoulder > SHOULDER_MAX:
                joint_feedback["left_shoulder"] = "warning"
                joint_feedback["right_shoulder"] = "warning"
                issues.append("Elbows flaring out — tuck elbows closer to body")
                scores["shoulder"] = 0.3
            else:
                joint_feedback["left_shoulder"] = "warning"
                joint_feedback["right_shoulder"] = "warning"
                issues.append("Arms too tight to body — widen grip slightly")
                scores["shoulder"] = 0.4

        # ── Pike detection ──
        if pose is not None:
            pike = self._check_pike(pose)
            if pike is not None and pike < -0.03:  # hip significantly above midline
                issues.append("Hips piking up -- lower hips into straight line")
                scores["pike"] = 0.3
                joint_feedback["left_hip"] = "incorrect"
                joint_feedback["right_hip"] = "incorrect"

        # ── Head/neck alignment (low weight) ──
        if pose is not None:
            head_issue = self._check_head_alignment(pose)
            if head_issue:
                issues.append(head_issue)
                scores["head"] = 0.5

        if not scores:
            return 0.0, joint_feedback, issues

        WEIGHTS = {"elbow": 1.0, "hip": 1.5, "shoulder": 1.0, "pike": 1.2, "head": 0.3}
        weighted_sum = sum(scores[k] * WEIGHTS.get(k, 1.0) for k in scores)
        weight_total = sum(WEIGHTS.get(k, 1.0) for k in scores)
        total = weighted_sum / weight_total

        return max(0.0, min(1.0, total)), joint_feedback, issues

    def _build_feedback_text(
        self,
        score: float,
        issues: List[str],
        elbow: Optional[float],
        rep_completed: bool,
    ) -> str:
        """Build actionable feedback text."""
        parts = []

        if rep_completed:
            last_rep = self._rep_history[-1] if self._rep_history else None
            if last_rep:
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
        else:
            # Ongoing feedback
            if issues:
                parts.append(issues[0])  # Most important issue
            elif score >= 0.7:
                phase = self._state
                if phase == PushUpState.GOING_DOWN:
                    parts.append("Good — keep lowering")
                elif phase == PushUpState.DOWN:
                    parts.append("Good depth! Push up")
                elif phase == PushUpState.GOING_UP:
                    parts.append("Push! Extend fully")
                else:
                    parts.append("Good form — go down")

        if self._set_count > 0:
            parts.append(f"Set {self._set_count + 1}")

        return " | ".join(parts) if parts else "Good form"

    def _build_result(
        self,
        form_score: float,
        is_active: bool,
        joint_feedback: Dict[str, str],
        details: Dict[str, str],
        feedback_text: str = "",
    ) -> ClassificationResult:
        """Build a ClassificationResult compatible with the pipeline."""
        return ClassificationResult(
            exercise="pushup",
            is_correct=form_score >= 0.7 and is_active,
            confidence=form_score,
            joint_feedback=joint_feedback,
            details=details,
            is_active=is_active,
            form_score=max(0.0, min(1.0, form_score)),
        )

    def _inactive_result(self, message: str) -> ClassificationResult:
        return self._build_result(
            form_score=0.0,
            is_active=False,
            joint_feedback={},
            details={"setup": message},
        )

    # ── Angle computation ────────────────────────────────────────────────

    def _avg_elbow_angle(self, pose: PoseResult) -> Optional[float]:
        """Average of left and right elbow angles (or whichever is visible)."""
        left = self._angle_at(pose, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
        right = self._angle_at(pose, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        if left is not None and right is not None:
            return (left + right) / 2
        return left or right

    def _compute_hip_angle(self, pose: PoseResult) -> Optional[float]:
        """Average hip angle (shoulder->hip->ankle) for body alignment."""
        left = self._angle_at(pose, LEFT_SHOULDER, LEFT_HIP, LEFT_ANKLE)
        right = self._angle_at(pose, RIGHT_SHOULDER, RIGHT_HIP, RIGHT_ANKLE)
        if left is not None and right is not None:
            return (left + right) / 2
        return left or right

    def _compute_shoulder_angle(self, pose: PoseResult) -> Optional[float]:
        """Average shoulder angle (elbow->shoulder->hip) for arm flare."""
        left = self._angle_at(pose, LEFT_ELBOW, LEFT_SHOULDER, LEFT_HIP)
        right = self._angle_at(pose, RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP)
        if left is not None and right is not None:
            return (left + right) / 2
        return left or right

    def _angle_at(
        self, pose: PoseResult, idx_a: int, idx_b: int, idx_c: int
    ) -> Optional[float]:
        """Compute angle at vertex idx_b using world coordinates."""
        if not all(pose.is_visible(i, MIN_VISIBILITY) for i in (idx_a, idx_b, idx_c)):
            return None
        a = pose.get_world_landmark(idx_a)
        b = pose.get_world_landmark(idx_b)
        c = pose.get_world_landmark(idx_c)
        return calculate_angle(a, b, c)

    def _smooth(self, buf: deque) -> Optional[float]:
        """Moving average of angle buffer."""
        if not buf:
            return None
        return sum(buf) / len(buf)

    # ── Session summary ──────────────────────────────────────────────────

    def get_session_summary(self) -> Dict:
        """Return summary of all reps and sets."""
        # Include current set if it has reps
        all_set_reps = list(self._set_reps)
        if self._reps_in_current_set > 0:
            all_set_reps.append(self._reps_in_current_set)

        good_reps = sum(1 for r in self._rep_history if r["quality"] == RepQuality.GOOD)
        moderate_reps = sum(1 for r in self._rep_history if r["quality"] == RepQuality.MODERATE)
        bad_reps = sum(1 for r in self._rep_history if r["quality"] == RepQuality.BAD)

        # Most common issues
        issue_counts: Dict[str, int] = {}
        for rep in self._rep_history:
            for issue in rep.get("issues", []):
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        common_issues = sorted(issue_counts.items(), key=lambda x: -x[1])[:5]

        avg_score = (
            sum(r["score"] for r in self._rep_history) / max(len(self._rep_history), 1)
        )

        return {
            "exercise": "pushup",
            "total_reps": self._rep_count,
            "sets": len(all_set_reps),
            "reps_per_set": all_set_reps,
            "good_reps": good_reps,
            "moderate_reps": moderate_reps,
            "bad_reps": bad_reps,
            "avg_form_score": round(avg_score, 2),
            "rep_details": self._rep_history,
            "common_issues": [
                {"issue": name, "count": count}
                for name, count in common_issues
            ],
        }
