"""
Dedicated bicep curl detector with state machine and form assessment.

Angles tracked:
  - Elbow (shoulder-elbow-wrist): primary rep counting angle
  - Upper arm stability (shoulder-elbow vertical angle)
  - Torso swing (shoulder-hip vertical angle)

Camera: SIDE VIEW — working arm faces camera.
State machine: TOP (extended) -> GOING_DOWN (curling) -> BOTTOM (curled) -> GOING_UP -> TOP (1 rep)
Note: "down" here means curling UP the weight (elbow angle decreases).
"""

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

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
    LEFT_INDEX, RIGHT_INDEX,
)
from ..utils.geometry import calculate_angle


# ── Thresholds ──
ELBOW_EXTENDED = 165
ELBOW_CURLED = 45
ELBOW_TOP_INCOMPLETE = 60
ELBOW_HALF = 90

UPPER_ARM_GOOD = 10
UPPER_ARM_WARNING = 20
UPPER_ARM_BAD = 30

TORSO_GOOD = 5
TORSO_WARNING = 10
TORSO_BAD = 15

# ── Side-view visibility ──
CURL_VIS_THRESHOLD = 0.3

# ── Movement-gated session FSM ──
CURL_ELBOW_AT_TOP = 150.0
CURL_ELBOW_DESCEND = 130.0
CURL_MOTION_WINDOW_S = 2.0
CURL_MOTION_MIN_RANGE = 20.0
CURL_REST_AT_TOP_S = 8.0
CURL_REST_BELOW_TOP_S = 60.0


class BicepCurlDetector(RobustExerciseDetector):
    EXERCISE_NAME = "bicep_curl"
    PRIMARY_ANGLE_TOP = ELBOW_EXTENDED
    PRIMARY_ANGLE_BOTTOM = ELBOW_CURLED
    DESCENT_THRESHOLD = 15.0
    ASCENT_THRESHOLD = 10.0
    MIN_FORM_GATE = 0.3

    SESSION_POSTURE_GATE = None
    VISIBILITY_GATE_LANDMARKS = (
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_ELBOW, RIGHT_ELBOW,
        LEFT_WRIST, RIGHT_WRIST,
        LEFT_HIP, RIGHT_HIP,
    )
    VISIBILITY_GATE_THRESHOLD = CURL_VIS_THRESHOLD

    REP_DIRECTION = REP_DIRECTION_DECREASING
    REP_COMPLETE_TOP = 150.0
    REP_START_THRESHOLD = 145.0
    REP_DEPTH_THRESHOLD = 55.0
    REP_BOTTOM_CLEAR = 65.0
    MIN_REAL_PRIMARY_DEG = 20.0
    MAX_PRIMARY_JUMP_DEG = 80.0

    def __init__(self, **kwargs):
        # Instance state BEFORE super().__init__() — reset() calls .clear()
        self._prev_torso: Optional[float] = None
        self._at_top_since: Optional[float] = None
        self._below_top_since: Optional[float] = None
        self._elbow_history: Deque[Tuple[float, float]] = deque()
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self._prev_torso = None
        self._at_top_since = None
        self._below_top_since = None
        self._elbow_history.clear()

    # ── Side-view visibility gate ──────────────────────────────────────
    def _check_visibility_gate(self, pose) -> bool:
        """Accept if EITHER side's shoulder-elbow-wrist chain is visible."""
        left_ok = all(
            pose.is_visible(idx, CURL_VIS_THRESHOLD)
            for idx in (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
        )
        right_ok = all(
            pose.is_visible(idx, CURL_VIS_THRESHOLD)
            for idx in (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        )
        return left_ok or right_ok

    # ── Lenient angle helpers (side-view) ──────────────────────────────
    def _lenient_side_angle(
        self,
        pose: PoseResult,
        a_idx: int,
        b_idx: int,
        c_idx: int,
        threshold: float = CURL_VIS_THRESHOLD,
    ) -> Optional[float]:
        if not all(pose.is_visible(i, threshold) for i in (a_idx, b_idx, c_idx)):
            return None
        return calculate_angle(
            pose.get_world_landmark(a_idx),
            pose.get_world_landmark(b_idx),
            pose.get_world_landmark(c_idx),
        )

    def _lenient_avg_angle(
        self,
        pose: PoseResult,
        left_triple: Tuple[int, int, int],
        right_triple: Tuple[int, int, int],
        name: str,
    ) -> Optional[float]:
        left = self._lenient_side_angle(pose, *left_triple)
        right = self._lenient_side_angle(pose, *right_triple)
        self._lr_angles[name] = (left, right)
        if left is not None and right is not None:
            return (left + right) / 2
        return left if left is not None else right

    # ── Movement-gated session FSM ─────────────────────────────────────
    def _update_session_state(self, now: float, visibility_ok: bool) -> None:
        elbow = self._get_smooth("elbow")
        if elbow is None:
            super()._update_session_state(now, visibility_ok)
            return

        is_at_top = elbow >= CURL_ELBOW_AT_TOP
        is_curling = elbow < CURL_ELBOW_DESCEND

        if is_at_top:
            if self._at_top_since is None:
                self._at_top_since = now
            self._below_top_since = None
        else:
            if self._below_top_since is None:
                self._below_top_since = now
            self._at_top_since = None

        self._elbow_history.append((now, elbow))
        window_start = now - CURL_MOTION_WINDOW_S
        while self._elbow_history and self._elbow_history[0][0] < window_start:
            self._elbow_history.popleft()

        if len(self._elbow_history) >= 2:
            values = [v for _, v in self._elbow_history]
            motion_range = max(values) - min(values)
        else:
            motion_range = 0.0
        has_active_motion = motion_range >= CURL_MOTION_MIN_RANGE

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

        if s == SessionState.SETUP:
            if is_curling and has_active_motion:
                self._session_state = SessionState.ACTIVE
                self._reset_robust_rep_fsm()
                self._rep_start_time = now
                self._seed_rep_fsm_from_pre_active(now)
            return

        if s == SessionState.RESTING:
            if is_curling and has_active_motion:
                self._session_state = SessionState.ACTIVE
                self._reset_robust_rep_fsm()
                self._rep_start_time = now
            return

        if s == SessionState.ACTIVE:
            if (
                self._at_top_since is not None
                and now - self._at_top_since >= CURL_REST_AT_TOP_S
            ):
                self._close_active_set_or_rollback(now)
                self._elbow_history.clear()
                return
            if (
                self._below_top_since is not None
                and now - self._below_top_since >= CURL_REST_BELOW_TOP_S
            ):
                self._close_active_set_or_rollback(now)
                self._elbow_history.clear()
                return

    # ── Angle computation ──────────────────────────────────────────────
    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        elbow = self._lenient_avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
            (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
            name="elbow",
        )

        upper_arm = self._compute_upper_arm_stability(pose)
        torso = self._compute_torso_swing(pose)
        wrist_neutral = self._compute_wrist_neutral(pose)

        torso_velocity = None
        if torso is not None and self._prev_torso is not None:
            torso_velocity = abs(torso - self._prev_torso)
        self._prev_torso = torso

        return {
            "primary": elbow, "elbow": elbow,
            "upper_arm": upper_arm, "torso": torso,
            "wrist_neutral": wrist_neutral,
            "torso_velocity": torso_velocity,
        }

    def _compute_wrist_neutral(self, pose: PoseResult) -> Optional[float]:
        min_angle = None
        for e_idx, w_idx, i_idx in [
            (LEFT_ELBOW, LEFT_WRIST, LEFT_INDEX),
            (RIGHT_ELBOW, RIGHT_WRIST, RIGHT_INDEX),
        ]:
            if all(pose.is_visible(idx, CURL_VIS_THRESHOLD) for idx in (e_idx, w_idx, i_idx)):
                angle = self._angle_at(pose, e_idx, w_idx, i_idx)
                if angle is not None:
                    if min_angle is None or angle < min_angle:
                        min_angle = angle
        return min_angle

    def _compute_upper_arm_stability(self, pose: PoseResult) -> Optional[float]:
        """Upper arm deviation from vertical — worst (max) of visible sides."""
        angles = []
        for s_idx, e_idx in [(LEFT_SHOULDER, LEFT_ELBOW), (RIGHT_SHOULDER, RIGHT_ELBOW)]:
            if pose.is_visible(s_idx, CURL_VIS_THRESHOLD) and pose.is_visible(e_idx, CURL_VIS_THRESHOLD):
                s = np.array(pose.get_world_landmark(s_idx))
                e = np.array(pose.get_world_landmark(e_idx))
                diff = e - s
                vertical = np.array([0, 1, 0])
                cos = np.dot(diff, vertical) / (np.linalg.norm(diff) + 1e-8)
                angles.append(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
        return max(angles) if angles else None

    def _compute_torso_swing(self, pose: PoseResult) -> Optional[float]:
        """Torso lean from vertical — worst (max) of visible sides."""
        angles = []
        for s_idx, h_idx in [(LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP)]:
            if pose.is_visible(s_idx, CURL_VIS_THRESHOLD) and pose.is_visible(h_idx, CURL_VIS_THRESHOLD):
                s = np.array(pose.get_world_landmark(s_idx))
                h = np.array(pose.get_world_landmark(h_idx))
                diff = s - h
                vertical = np.array([0, 1, 0])
                cos = np.dot(diff, vertical) / (np.linalg.norm(diff) + 1e-8)
                angles.append(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
        return max(angles) if angles else None

    # ── Start position / missing parts ─────────────────────────────────
    def _check_start_position(self, angles: Dict[str, Optional[float]]) -> Tuple[bool, List[str]]:
        issues = []
        elbow = angles.get("elbow")
        if elbow is None:
            return False, ["Stand sideways to camera \u2014 working arm must be visible"]
        if elbow < ELBOW_EXTENDED - 30:
            issues.append("Extend arms fully to begin \u2014 let weights hang at sides")
        return (len(issues) == 0, issues)

    def _get_missing_parts(self, angles: Dict[str, Optional[float]]) -> List[str]:
        if angles.get("elbow") is None:
            return ["arms and elbows"]
        return ["body"]

    # ── Form assessment ────────────────────────────────────────────────
    def _assess_form(self, angles: Dict[str, Optional[float]]) -> Tuple[float, Dict[str, str], List[str]]:
        joint_feedback = {}
        issues = []
        scores = {}

        elbow = angles.get("elbow")
        upper_arm = angles.get("upper_arm")
        torso = angles.get("torso")
        wrist_neutral = angles.get("wrist_neutral")
        torso_velocity = angles.get("torso_velocity")

        in_curl = self._state in (RepPhase.BOTTOM, RepPhase.GOING_UP, RepPhase.GOING_DOWN)
        _momentum_active = torso_velocity is not None and torso_velocity > 5.0

        # ── Elbow ROM ──
        if elbow is not None:
            if self._state == RepPhase.BOTTOM:
                if elbow <= ELBOW_CURLED + 5:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 1.0
                elif elbow <= ELBOW_TOP_INCOMPLETE:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 0.85
                else:
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    issues.append("Curl higher \u2014 full contraction at the top")
                    scores["elbow"] = 0.5
            elif in_curl:
                joint_feedback["left_elbow"] = "correct"
                joint_feedback["right_elbow"] = "correct"
                scores["elbow"] = 0.9
            else:
                if elbow >= ELBOW_EXTENDED - 10:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 1.0
                else:
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    issues.append("Extend arms fully at the bottom")
                    scores["elbow"] = 0.6

        # ── Upper arm stability (most critical from side view) ──
        if upper_arm is not None:
            if upper_arm <= UPPER_ARM_GOOD:
                joint_feedback["left_shoulder"] = "correct"
                joint_feedback["right_shoulder"] = "correct"
                scores["upper_arm"] = 1.0
            elif upper_arm <= UPPER_ARM_WARNING:
                joint_feedback["left_shoulder"] = "warning"
                joint_feedback["right_shoulder"] = "warning"
                issues.append("Pin elbows to your sides \u2014 upper arm is moving")
                scores["upper_arm"] = 0.5
            else:
                joint_feedback["left_shoulder"] = "incorrect"
                joint_feedback["right_shoulder"] = "incorrect"
                issues.append("Elbows swinging \u2014 keep upper arms still")
                scores["upper_arm"] = 0.2

        # ── Torso swing ──
        if torso is not None:
            if torso <= TORSO_GOOD:
                joint_feedback["left_hip"] = "correct"
                joint_feedback["right_hip"] = "correct"
                scores["torso"] = 1.0
            elif torso <= TORSO_WARNING:
                joint_feedback["left_hip"] = "warning"
                joint_feedback["right_hip"] = "warning"
                if not _momentum_active:
                    issues.append("Slight body swing \u2014 stay more upright")
                scores["torso"] = 0.5
            else:
                joint_feedback["left_hip"] = "incorrect"
                joint_feedback["right_hip"] = "incorrect"
                if not _momentum_active:
                    issues.append("Too much body swing \u2014 reduce weight or slow down")
                scores["torso"] = 0.2

        # ── Torso angular velocity ──
        if torso_velocity is not None and torso_velocity > 5.0:
            issues.append("Using momentum \u2014 try slowing down and controlling the weight")
            scores["torso_velocity"] = 0.4

        # ── Wrist neutral ──
        if wrist_neutral is not None and wrist_neutral < 160:
            issues.append("Keep wrists straight \u2014 avoid curling them")
            scores["wrist"] = 0.5

        if not scores:
            return 0.0, joint_feedback, issues

        WEIGHTS = {
            "elbow": 1.0, "upper_arm": 2.0, "torso": 1.2,
            "torso_velocity": 1.0, "wrist": 0.6,
        }
        weighted_sum = sum(scores[k] * WEIGHTS.get(k, 1.0) for k in scores)
        weight_total = sum(WEIGHTS.get(k, 1.0) for k in scores)
        total = weighted_sum / weight_total

        return max(0.0, min(1.0, total)), joint_feedback, issues
