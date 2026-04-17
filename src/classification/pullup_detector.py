"""
Dedicated pull-up detector with state machine and form assessment.

Angles tracked:
  - Elbow (shoulder-elbow-wrist): primary rep counting
  - Chin above bar: nose Y rises to meet fixed wrist Y
  - Body swing: hip stability during pull (sagittal + lateral)
  - Hip angle: kipping detection (shoulder-hip-knee straightness)
  - Wrist Y variance: anti-cheat (hands must stay fixed on bar)

Camera: FRONT VIEW — full body from hands on bar to feet.
State machine: TOP (hanging) -> GOING_DOWN (pulling up) -> BOTTOM (chin above bar) -> GOING_UP (lowering) -> TOP (1 rep)
Note: inverted — starts at top (dead hang), angle DECREASES going up.
"""

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import time

import numpy as np

from .base_detector import (
    RobustExerciseDetector,
    RepPhase,
    REP_DIRECTION_DECREASING,
    SessionState,
)
from ..pose_estimation.base import PoseResult
from ..utils.constants import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_EAR, RIGHT_EAR,
    NOSE,
)
from ..utils.geometry import calculate_angle


# ── Elbow thresholds ──
ELBOW_HANGING = 170
ELBOW_TOP = 85
ELBOW_TOP_MAX_VALID = 100
ELBOW_HALF = 120

# ── Body swing (sagittal plane — shoulder-hip deviation from vertical) ──
SWING_GOOD = 10
SWING_WARNING = 20
SWING_BAD = 30

# ── Anti-cheat: wrist Y variance ──
# Fixed bar: wrists don't move, variance ≈ 0.
# Standing arm pump: wrists move a lot, variance >> threshold.
WRIST_Y_VAR_THRESHOLD = 0.015

# ── Kipping: hip angle (shoulder-hip-knee) ──
HIP_ANGLE_STRICT = 160.0
HIP_ANGLE_WARNING = 150.0
HIP_ANGLE_BAD = 140.0

# ── Lateral swing: hip X frame-to-frame variance ──
HIP_X_VAR_THRESHOLD = 0.02

# ── Controlled descent ──
DESCENT_ASCENT_SPEED_RATIO_MAX = 2.0

# ── Movement-gated session FSM ──
PULLUP_ELBOW_AT_TOP = 160.0
PULLUP_MOTION_WINDOW_S = 2.0
PULLUP_MOTION_MIN_RANGE = 25.0
PULLUP_REST_AT_TOP_S = 8.0
PULLUP_REST_BELOW_TOP_S = 60.0


class PullUpDetector(RobustExerciseDetector):
    EXERCISE_NAME = "pullup"
    PRIMARY_ANGLE_TOP = ELBOW_HANGING
    PRIMARY_ANGLE_BOTTOM = ELBOW_TOP
    DESCENT_THRESHOLD = 15.0
    ASCENT_THRESHOLD = 10.0
    MIN_FORM_GATE = 0.4

    SESSION_POSTURE_GATE = None
    VISIBILITY_GATE_LANDMARKS = (
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_ELBOW, RIGHT_ELBOW,
        LEFT_WRIST, RIGHT_WRIST,
        LEFT_HIP, RIGHT_HIP,
    )
    VISIBILITY_GATE_THRESHOLD = 0.6

    REP_DIRECTION = REP_DIRECTION_DECREASING
    REP_COMPLETE_TOP = 160.0
    REP_START_THRESHOLD = 155.0
    REP_DEPTH_THRESHOLD = 100.0
    REP_BOTTOM_CLEAR = 110.0
    MIN_REAL_PRIMARY_DEG = 30.0
    MAX_PRIMARY_JUMP_DEG = 80.0

    def __init__(self, **kwargs):
        # Instance state BEFORE super().__init__() — reset() calls .clear()
        self._baseline_shoulder_ear: Optional[float] = None

        # Anti-cheat: wrist Y tracking
        self._wrist_y_history: Deque[float] = deque(maxlen=90)
        self._wrists_above_head: bool = False
        self._wrist_y_var_current: float = 1.0

        # Kipping: hip angle tracking
        self._hip_angle_history: Deque[float] = deque(maxlen=90)

        # Lateral swing: hip X tracking
        self._hip_x_history: Deque[float] = deque(maxlen=30)

        # Controlled descent: phase timing
        self._phase_start_time: Optional[float] = None
        self._last_phase: Optional[RepPhase] = None
        self._last_ascent_duration: Optional[float] = None
        self._last_descent_duration: Optional[float] = None
        self._descent_speed_ratio: Optional[float] = None

        # Movement-gated session FSM
        self._at_top_since: Optional[float] = None
        self._below_top_since: Optional[float] = None
        self._elbow_history: Deque[Tuple[float, float]] = deque()

        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self._baseline_shoulder_ear = None
        self._wrist_y_history.clear()
        self._wrists_above_head = False
        self._wrist_y_var_current = 1.0
        self._hip_angle_history.clear()
        self._hip_x_history.clear()
        self._phase_start_time = None
        self._last_phase = None
        self._last_ascent_duration = None
        self._last_descent_duration = None
        self._descent_speed_ratio = None
        self._at_top_since = None
        self._below_top_since = None
        self._elbow_history.clear()

    # ── Movement-gated session FSM ─────────────────────────────────────
    def _update_session_state(self, now: float, visibility_ok: bool) -> None:
        elbow = self._get_smooth("elbow")
        if elbow is None:
            super()._update_session_state(now, visibility_ok)
            return

        is_at_top = elbow >= PULLUP_ELBOW_AT_TOP

        if is_at_top:
            if self._at_top_since is None:
                self._at_top_since = now
            self._below_top_since = None
        else:
            if self._below_top_since is None:
                self._below_top_since = now
            self._at_top_since = None

        self._elbow_history.append((now, elbow))
        window_start = now - PULLUP_MOTION_WINDOW_S
        while self._elbow_history and self._elbow_history[0][0] < window_start:
            self._elbow_history.popleft()

        if len(self._elbow_history) >= 2:
            values = [v for _, v in self._elbow_history]
            motion_range = max(values) - min(values)
        else:
            motion_range = 0.0
        has_active_motion = motion_range >= PULLUP_MOTION_MIN_RANGE

        s = self._session_state

        if not visibility_ok:
            if s == SessionState.ACTIVE:
                self._unknown_streak += 1
                if self._unknown_streak >= self.UNKNOWN_GRACE_FRAMES:
                    self._close_active_set_or_rollback(now)
                    self._elbow_history.clear()
            return
        self._unknown_streak = 0

        if s == SessionState.IDLE:
            self._session_state = SessionState.SETUP
            return

        # SETUP → ACTIVE: wrists on bar + pulling motion
        if s == SessionState.SETUP:
            if (
                self._wrists_above_head
                and self._wrist_y_var_current < WRIST_Y_VAR_THRESHOLD
                and has_active_motion
            ):
                self._session_state = SessionState.ACTIVE
                self._reset_robust_rep_fsm()
                self._rep_start_time = now
                self._seed_rep_fsm_from_pre_active(now)
            return

        if s == SessionState.RESTING:
            if (
                self._wrists_above_head
                and self._wrist_y_var_current < WRIST_Y_VAR_THRESHOLD
                and has_active_motion
            ):
                self._session_state = SessionState.ACTIVE
                self._reset_robust_rep_fsm()
                self._rep_start_time = now
            return

        if s == SessionState.ACTIVE:
            if (
                self._at_top_since is not None
                and now - self._at_top_since >= PULLUP_REST_AT_TOP_S
            ):
                self._close_active_set_or_rollback(now)
                self._elbow_history.clear()
                return
            if (
                self._below_top_since is not None
                and now - self._below_top_since >= PULLUP_REST_BELOW_TOP_S
            ):
                self._close_active_set_or_rollback(now)
                self._elbow_history.clear()
                return

    # ── Angle computation ──────────────────────────────────────────────
    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        now = time.monotonic()

        elbow = self._avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
            (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
            name="elbow",
        )

        chin_above = self._check_chin_above_bar(pose)
        body_swing = self._compute_body_swing(pose)
        scap_depression = self._compute_scapular_depression(pose)

        # Anti-cheat: wrist Y tracking
        self._update_wrist_tracking(pose)

        # Kipping: hip angle (shoulder-hip-knee)
        hip_angle = self._compute_hip_angle(pose)

        # Lateral swing: hip X variance
        hip_x_var = self._update_hip_x_tracking(pose)

        # Controlled descent: track phase durations
        self._track_phase_timing(now)

        return {
            "primary": elbow, "elbow": elbow,
            "chin_above": chin_above, "body_swing": body_swing,
            "scap_depression": scap_depression,
            "hip_angle": hip_angle,
            "wrist_stability": self._wrist_y_var_current,
            "hip_x_var": hip_x_var,
        }

    def _update_wrist_tracking(self, pose: PoseResult) -> None:
        """Track wrist Y positions and compute variance + above-head check."""
        wrist_ys = []
        for w_idx in [LEFT_WRIST, RIGHT_WRIST]:
            if pose.is_visible(w_idx):
                wrist_ys.append(pose.get_landmark(w_idx)[1])
        if not wrist_ys:
            return

        avg_wrist_y = sum(wrist_ys) / len(wrist_ys)
        self._wrist_y_history.append(avg_wrist_y)

        # Variance of wrist Y over the rolling window
        if len(self._wrist_y_history) >= 5:
            arr = np.array(self._wrist_y_history)
            self._wrist_y_var_current = float(np.var(arr))
        else:
            self._wrist_y_var_current = 1.0

        # Wrists above head: in image coords Y increases downward
        shoulder_ys = []
        for s_idx in [LEFT_SHOULDER, RIGHT_SHOULDER]:
            if pose.is_visible(s_idx):
                shoulder_ys.append(pose.get_landmark(s_idx)[1])
        if shoulder_ys:
            avg_shoulder_y = sum(shoulder_ys) / len(shoulder_ys)
            self._wrists_above_head = avg_wrist_y < avg_shoulder_y
        else:
            self._wrists_above_head = False

    def _compute_hip_angle(self, pose: PoseResult) -> Optional[float]:
        """Shoulder-hip-knee angle for kipping detection. Straight body ≈ 170-180°."""
        angles = []
        for s_idx, h_idx, k_idx in [
            (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
            (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
        ]:
            if all(pose.is_visible(idx) for idx in (s_idx, h_idx, k_idx)):
                angle = calculate_angle(
                    pose.get_world_landmark(s_idx),
                    pose.get_world_landmark(h_idx),
                    pose.get_world_landmark(k_idx),
                )
                if angle is not None:
                    angles.append(angle)
        if not angles:
            return None
        result = sum(angles) / len(angles)
        self._hip_angle_history.append(result)
        return result

    def _update_hip_x_tracking(self, pose: PoseResult) -> Optional[float]:
        """Track lateral hip X position and compute variance."""
        hip_xs = []
        for h_idx in [LEFT_HIP, RIGHT_HIP]:
            if pose.is_visible(h_idx):
                hip_xs.append(pose.get_landmark(h_idx)[0])
        if not hip_xs:
            return None
        avg_hip_x = sum(hip_xs) / len(hip_xs)
        self._hip_x_history.append(avg_hip_x)
        if len(self._hip_x_history) >= 5:
            arr = np.array(self._hip_x_history)
            return float(np.var(arr))
        return None

    def _track_phase_timing(self, now: float) -> None:
        """Track ascent/descent durations for controlled-descent scoring."""
        current_phase = self._state
        if current_phase != self._last_phase:
            if self._phase_start_time is not None:
                duration = now - self._phase_start_time
                # GOING_DOWN in pullup FSM = pulling UP (ascending body)
                if self._last_phase == RepPhase.GOING_DOWN:
                    self._last_ascent_duration = duration
                # GOING_UP in pullup FSM = lowering body (descending)
                elif self._last_phase == RepPhase.GOING_UP:
                    self._last_descent_duration = duration

                if self._last_ascent_duration and self._last_descent_duration:
                    if self._last_ascent_duration > 0.1:
                        self._descent_speed_ratio = (
                            self._last_ascent_duration / self._last_descent_duration
                        )
            self._phase_start_time = now
            self._last_phase = current_phase

    def _check_chin_above_bar(self, pose: PoseResult) -> Optional[float]:
        """Normalized (wrist_y - nose_y) / torso_length. Positive = nose above wrists."""
        if not pose.is_visible(NOSE):
            return None

        wrist_ys = []
        for w_idx in [LEFT_WRIST, RIGHT_WRIST]:
            if pose.is_visible(w_idx):
                wrist_ys.append(pose.get_landmark(w_idx)[1])
        if not wrist_ys:
            return None

        shoulder_ys = []
        hip_ys = []
        for s_idx in [LEFT_SHOULDER, RIGHT_SHOULDER]:
            if pose.is_visible(s_idx):
                shoulder_ys.append(pose.get_landmark(s_idx)[1])
        for h_idx in [LEFT_HIP, RIGHT_HIP]:
            if pose.is_visible(h_idx):
                hip_ys.append(pose.get_landmark(h_idx)[1])
        if not shoulder_ys or not hip_ys:
            return None

        torso_len = abs((sum(hip_ys) / len(hip_ys)) - (sum(shoulder_ys) / len(shoulder_ys)))
        if torso_len < 0.05:
            return None

        nose_y = pose.get_landmark(NOSE)[1]
        avg_wrist_y = sum(wrist_ys) / len(wrist_ys)
        return (avg_wrist_y - nose_y) / torso_len

    def _compute_body_swing(self, pose: PoseResult) -> Optional[float]:
        """Deviation of shoulder-hip line from pure vertical (world coords)."""
        pts_s = []
        pts_h = []
        for s_idx in [LEFT_SHOULDER, RIGHT_SHOULDER]:
            if pose.is_visible(s_idx):
                pts_s.append(np.array(pose.get_world_landmark(s_idx)))
        for h_idx in [LEFT_HIP, RIGHT_HIP]:
            if pose.is_visible(h_idx):
                pts_h.append(np.array(pose.get_world_landmark(h_idx)))
        if not pts_s or not pts_h:
            return None

        mid_shoulder = np.mean(pts_s, axis=0)
        mid_hip = np.mean(pts_h, axis=0)
        diff = mid_hip - mid_shoulder
        norm = float(np.linalg.norm(diff))
        if norm < 0.1:
            return None
        vertical = np.array([0.0, 1.0, 0.0])
        cos = float(np.dot(diff, vertical) / norm)
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    def _compute_scapular_depression(self, pose: PoseResult) -> Optional[float]:
        """Shoulder-to-ear distance. Positive return = shoulder rising toward ear."""
        dists = []
        for s_idx, ear_idx in [(LEFT_SHOULDER, LEFT_EAR), (RIGHT_SHOULDER, RIGHT_EAR)]:
            if pose.is_visible(s_idx) and pose.is_visible(ear_idx):
                s = np.array(pose.get_world_landmark(s_idx))
                ear = np.array(pose.get_world_landmark(ear_idx))
                dists.append(float(np.linalg.norm(s - ear)))
        if not dists:
            return None
        current = sum(dists) / len(dists)
        if self._baseline_shoulder_ear is None and self._state == RepPhase.TOP:
            self._baseline_shoulder_ear = current
            return 0.0
        if self._baseline_shoulder_ear is None:
            return None
        return self._baseline_shoulder_ear - current

    # ── Start position / missing parts ─────────────────────────────────
    def _check_start_position(self, angles: Dict[str, Optional[float]]) -> Tuple[bool, List[str]]:
        issues = []
        elbow = angles.get("elbow")
        if elbow is None:
            return False, ["Hang from bar \u2014 arms must be visible"]
        if elbow < ELBOW_HANGING - 30:
            issues.append("Extend arms fully \u2014 dead hang to start")
        return (len(issues) == 0, issues)

    def _get_missing_parts(self, angles: Dict[str, Optional[float]]) -> List[str]:
        if angles.get("elbow") is None:
            return ["arms"]
        return ["body"]

    # ── Rep counting gate ──────────────────────────────────────────────
    def _should_count_rep(self, elapsed: float, angles: Dict[str, Optional[float]]) -> bool:
        if not super()._should_count_rep(elapsed, angles):
            return False
        # Hard reject if wrists are moving (not on a bar)
        if self._wrist_y_var_current > WRIST_Y_VAR_THRESHOLD:
            return False
        return True

    # ── Form assessment ────────────────────────────────────────────────
    def _assess_form(self, angles: Dict[str, Optional[float]]) -> Tuple[float, Dict[str, str], List[str]]:
        joint_feedback = {}
        issues = []
        scores = {}

        elbow = angles.get("elbow")
        chin_above = angles.get("chin_above")
        body_swing = angles.get("body_swing")
        scap_depression = angles.get("scap_depression")
        hip_angle = angles.get("hip_angle")
        wrist_stability = angles.get("wrist_stability")
        hip_x_var = angles.get("hip_x_var")

        in_pull = self._state in (RepPhase.BOTTOM, RepPhase.GOING_UP)
        _swing_active = body_swing is not None and body_swing > SWING_WARNING

        # ── Anti-cheat: wrist stability ──
        if wrist_stability is not None and wrist_stability > WRIST_Y_VAR_THRESHOLD:
            issues.append("Hands are moving \u2014 grip a fixed bar")
            scores["wrist_cheat"] = 0.0

        # ── Elbow / height ──
        if elbow is not None:
            if in_pull:
                if elbow <= ELBOW_TOP + 5:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 1.0
                elif elbow <= ELBOW_TOP_MAX_VALID:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 0.85
                elif elbow <= ELBOW_HALF:
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    if not _swing_active:
                        issues.append("Pull higher \u2014 chin toward the bar")
                    scores["elbow"] = 0.5
                else:
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    scores["elbow"] = 0.3
            else:
                if elbow >= ELBOW_HANGING - 10:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 1.0
                else:
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    issues.append("Extend arms fully at the bottom \u2014 full dead hang")
                    scores["elbow"] = 0.6

        # ── Chin above bar ──
        if chin_above is not None and in_pull:
            if chin_above > 0.05:
                scores["chin"] = 1.0
            elif chin_above > -0.05:
                if not _swing_active:
                    issues.append("Pull a bit higher \u2014 chin just below bar level")
                scores["chin"] = 0.6
            else:
                if not _swing_active:
                    issues.append("Not quite high enough \u2014 aim to get chin above the bar")
                scores["chin"] = 0.3

        # ── Body swing (sagittal plane) ──
        if body_swing is not None:
            if body_swing <= SWING_GOOD:
                joint_feedback["left_hip"] = "correct"
                joint_feedback["right_hip"] = "correct"
                scores["swing"] = 1.0
            elif body_swing <= SWING_WARNING:
                joint_feedback["left_hip"] = "warning"
                joint_feedback["right_hip"] = "warning"
                issues.append("Body swinging \u2014 keep core tight and controlled")
                scores["swing"] = 0.5
            else:
                joint_feedback["left_hip"] = "incorrect"
                joint_feedback["right_hip"] = "incorrect"
                issues.append("Too much swing \u2014 slow down and use strict form")
                scores["swing"] = 0.2

        # ── Scapular depression ──
        if scap_depression is not None and scap_depression > 0.03:
            issues.append("Engage lats \u2014 pull shoulders down before pulling up")
            scores["scap"] = 0.5

        # ── No kipping (hip angle — shoulder-hip-knee straightness) ──
        if hip_angle is not None:
            if hip_angle >= HIP_ANGLE_STRICT:
                scores["kip"] = 1.0
            elif hip_angle >= HIP_ANGLE_WARNING:
                joint_feedback["left_knee"] = "warning"
                joint_feedback["right_knee"] = "warning"
                issues.append("Slight kip \u2014 keep legs straight")
                scores["kip"] = 0.5
            elif hip_angle >= HIP_ANGLE_BAD:
                joint_feedback["left_knee"] = "incorrect"
                joint_feedback["right_knee"] = "incorrect"
                issues.append("Knees coming up \u2014 strict form means straight body")
                scores["kip"] = 0.2
            else:
                joint_feedback["left_knee"] = "incorrect"
                joint_feedback["right_knee"] = "incorrect"
                issues.append("Full kip \u2014 straighten legs and control the pull")
                scores["kip"] = 0.1

        # ── Lateral swing (hip X variance) ──
        if hip_x_var is not None:
            if hip_x_var <= HIP_X_VAR_THRESHOLD:
                scores["lateral_swing"] = 1.0
            else:
                issues.append("Lateral swinging \u2014 tighten core")
                scores["lateral_swing"] = 0.3

        # ── Controlled descent ──
        if self._descent_speed_ratio is not None:
            if self._descent_speed_ratio <= 1.5:
                scores["descent_control"] = 1.0
            elif self._descent_speed_ratio <= DESCENT_ASCENT_SPEED_RATIO_MAX:
                scores["descent_control"] = 0.7
            else:
                issues.append("Dropping too fast \u2014 control the descent")
                scores["descent_control"] = 0.3

        if not scores:
            return 0.0, joint_feedback, issues

        WEIGHTS = {
            "elbow": 1.0, "chin": 1.0, "swing": 1.3, "scap": 0.8,
            "kip": 1.5, "wrist_cheat": 2.0, "lateral_swing": 1.0,
            "descent_control": 0.8,
        }
        weighted_sum = sum(scores[k] * WEIGHTS.get(k, 1.0) for k in scores)
        weight_total = sum(WEIGHTS.get(k, 1.0) for k in scores)
        total = weighted_sum / weight_total

        return max(0.0, min(1.0, total)), joint_feedback, issues
