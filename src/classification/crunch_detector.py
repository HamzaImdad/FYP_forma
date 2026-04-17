"""
Dedicated V-Up crunch detector with state machine and form assessment.

Angles tracked:
  - Hip (shoulder-hip-knee): primary rep counting angle
  - Elbow (shoulder-elbow-wrist): arms should stay straight
  - Legs-elevated check: ankles must stay above floor reference

Camera: SIDE VIEW — full body from head to feet at floor level.
State machine: TOP (extended) -> GOING_DOWN (closing V) -> BOTTOM (V-shape) -> GOING_UP -> TOP (1 rep)
Note: hip angle DECREASES as torso + legs come together (V closes).
"""

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

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
)
from ..utils.geometry import calculate_angle


# ── Thresholds ──
# Hip angle: V-up ROM
HIP_FULLY_CLOSED = 75     # tight V-shape
HIP_GOOD_CLOSE = 90       # acceptable crunch depth
HIP_HALF = 100             # partial — not enough ROM

# Elbow: arms straight
ELBOW_STRAIGHT = 150
ELBOW_BENT_WARNING = 130

# Legs-elevated: world-coord Y offset (meters)
# MediaPipe world Y increases downward. Ankle should NOT be much
# below hip (i.e., ankle_y should not exceed hip_y + threshold).
LEGS_ELEVATED_GOOD = 0.02    # meters — ankle close to hip level
LEGS_ELEVATED_WARNING = 0.05 # meters — feet drifting toward floor
LEGS_ELEVATED_BAD = 0.10     # meters — feet on/near floor

# Hip velocity (degrees per frame at ~30fps)
VELOCITY_GOOD = 8.0
VELOCITY_WARNING = 12.0

# Side-view visibility
CRUNCH_VIS_THRESHOLD = 0.3

# ── Movement-gated session FSM ──
CRUNCH_HIP_AT_TOP = 140.0
CRUNCH_HIP_DESCEND = 120.0
CRUNCH_MOTION_WINDOW_S = 2.0
CRUNCH_MOTION_MIN_RANGE = 20.0
CRUNCH_REST_AT_TOP_S = 8.0
CRUNCH_REST_BELOW_TOP_S = 60.0


class CrunchDetector(RobustExerciseDetector):
    EXERCISE_NAME = "crunch"
    PRIMARY_ANGLE_TOP = 155.0     # extended hollow body (hip open)
    PRIMARY_ANGLE_BOTTOM = 75.0   # V-shape (hip closed)
    DESCENT_THRESHOLD = 15.0
    ASCENT_THRESHOLD = 10.0
    MIN_FORM_GATE = 0.3

    SESSION_POSTURE_GATE = None
    VISIBILITY_GATE_LANDMARKS = (
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_HIP, RIGHT_HIP,
        LEFT_KNEE, RIGHT_KNEE,
        LEFT_ANKLE, RIGHT_ANKLE,
    )
    VISIBILITY_GATE_THRESHOLD = CRUNCH_VIS_THRESHOLD

    REP_DIRECTION = REP_DIRECTION_DECREASING  # hip angle DECREASES as V closes
    REP_COMPLETE_TOP = 140.0      # hip extended = top
    REP_START_THRESHOLD = 135.0   # below this = descent committed
    REP_DEPTH_THRESHOLD = 100.0   # must reach V-shape depth
    REP_BOTTOM_CLEAR = 110.0      # past this on return = leaving bottom
    MIN_REAL_PRIMARY_DEG = 30.0   # hip can't close below ~30°
    MAX_PRIMARY_JUMP_DEG = 60.0

    def __init__(self, **kwargs):
        # Instance state BEFORE super().__init__() — reset() calls .clear()
        self._prev_hip: Optional[float] = None
        self._at_top_since: Optional[float] = None
        self._below_top_since: Optional[float] = None
        self._hip_history: Deque[Tuple[float, float]] = deque()
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self._prev_hip = None
        self._at_top_since = None
        self._below_top_since = None
        self._hip_history.clear()

    # ── Side-view visibility gate ──────────────────────────────────────
    def _check_visibility_gate(self, pose) -> bool:
        """Accept if EITHER side's shoulder-hip-knee chain is visible."""
        left_ok = all(
            pose.is_visible(idx, CRUNCH_VIS_THRESHOLD)
            for idx in (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE)
        )
        right_ok = all(
            pose.is_visible(idx, CRUNCH_VIS_THRESHOLD)
            for idx in (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE)
        )
        return left_ok or right_ok

    # ── Lenient angle helpers (side-view) ──────────────────────────────
    def _lenient_side_angle(
        self,
        pose: PoseResult,
        a_idx: int,
        b_idx: int,
        c_idx: int,
        threshold: float = CRUNCH_VIS_THRESHOLD,
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

    # ── Angle computation ──────────────────────────────────────────────
    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        # Primary: hip angle (shoulder-hip-knee)
        hip = self._lenient_avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
            (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
            name="hip",
        )

        # Elbow angle: arms should stay straight
        elbow = self._lenient_avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
            (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
            name="elbow",
        )

        # Legs-elevated check: ankle Y vs hip Y in world coords
        legs_offset = self._compute_legs_elevation(pose)

        # Hip velocity (momentum check)
        hip_velocity = None
        if hip is not None and self._prev_hip is not None:
            hip_velocity = abs(hip - self._prev_hip)
        self._prev_hip = hip

        return {
            "primary": hip, "hip": hip,
            "elbow": elbow,
            "legs_offset": legs_offset,
            "hip_velocity": hip_velocity,
        }

    def _compute_legs_elevation(self, pose: PoseResult) -> Optional[float]:
        """Compute how far ankles are below hip level in world coords.
        Positive = ankles below hips (feet toward floor, bad).
        Negative = ankles above hips (good).
        Returns the WORST (highest) of visible sides."""
        offsets = []
        for a_idx, h_idx in [(LEFT_ANKLE, LEFT_HIP), (RIGHT_ANKLE, RIGHT_HIP)]:
            if (pose.is_visible(a_idx, CRUNCH_VIS_THRESHOLD)
                    and pose.is_visible(h_idx, CRUNCH_VIS_THRESHOLD)):
                ankle = pose.get_world_landmark(a_idx)
                hip = pose.get_world_landmark(h_idx)
                # World Y: positive = downward. ankle_y > hip_y = ankle below hip
                offsets.append(float(ankle[1] - hip[1]))
        return max(offsets) if offsets else None

    # ── Movement-gated session FSM ─────────────────────────────────────
    def _update_session_state(self, now: float, visibility_ok: bool) -> None:
        hip = self._get_smooth("hip")
        if hip is None:
            super()._update_session_state(now, visibility_ok)
            return

        is_at_top = hip >= CRUNCH_HIP_AT_TOP
        is_crunching = hip < CRUNCH_HIP_DESCEND

        if is_at_top:
            if self._at_top_since is None:
                self._at_top_since = now
            self._below_top_since = None
        else:
            if self._below_top_since is None:
                self._below_top_since = now
            self._at_top_since = None

        self._hip_history.append((now, hip))
        window_start = now - CRUNCH_MOTION_WINDOW_S
        while self._hip_history and self._hip_history[0][0] < window_start:
            self._hip_history.popleft()

        if len(self._hip_history) >= 2:
            values = [v for _, v in self._hip_history]
            motion_range = max(values) - min(values)
        else:
            motion_range = 0.0
        has_active_motion = motion_range >= CRUNCH_MOTION_MIN_RANGE

        s = self._session_state

        if not visibility_ok:
            if s == SessionState.ACTIVE:
                self._unknown_streak += 1
                if self._unknown_streak >= self.UNKNOWN_GRACE_FRAMES:
                    self._close_active_set_or_rollback(now)
                    self._hip_history.clear()
            return
        self._unknown_streak = 0

        if s == SessionState.IDLE:
            self._session_state = SessionState.SETUP
            return

        if s == SessionState.SETUP:
            if is_crunching and has_active_motion:
                self._session_state = SessionState.ACTIVE
                self._reset_robust_rep_fsm()
                self._rep_start_time = now
                self._seed_rep_fsm_from_pre_active(now)
            return

        if s == SessionState.RESTING:
            if is_crunching and has_active_motion:
                self._session_state = SessionState.ACTIVE
                self._reset_robust_rep_fsm()
                self._rep_start_time = now
            return

        if s == SessionState.ACTIVE:
            if (
                self._at_top_since is not None
                and now - self._at_top_since >= CRUNCH_REST_AT_TOP_S
            ):
                self._close_active_set_or_rollback(now)
                self._hip_history.clear()
                return
            if (
                self._below_top_since is not None
                and now - self._below_top_since >= CRUNCH_REST_BELOW_TOP_S
            ):
                self._close_active_set_or_rollback(now)
                self._hip_history.clear()
                return

    # ── Start position / missing parts ─────────────────────────────────
    def _check_start_position(
        self, angles: Dict[str, Optional[float]]
    ) -> Tuple[bool, List[str]]:
        issues = []
        hip = angles.get("hip")
        if hip is None:
            return False, [
                "Lie on your back with legs elevated \u2014 full body must be visible from the side"
            ]
        if hip < 130:
            issues.append("Extend your body to start \u2014 legs out, arms forward")
        return (len(issues) == 0, issues)

    def _get_missing_parts(self, angles: Dict[str, Optional[float]]) -> List[str]:
        if angles.get("hip") is None:
            return ["hips, knees, or shoulders"]
        return ["body"]

    # ── Form assessment ────────────────────────────────────────────────
    def _assess_form(
        self, angles: Dict[str, Optional[float]]
    ) -> Tuple[float, Dict[str, str], List[str]]:
        joint_feedback: Dict[str, str] = {}
        issues: List[str] = []
        scores: Dict[str, float] = {}

        hip = angles.get("hip")
        elbow = angles.get("elbow")
        legs_offset = angles.get("legs_offset")
        hip_velocity = angles.get("hip_velocity")

        in_crunch = self._state in (RepPhase.BOTTOM, RepPhase.GOING_UP, RepPhase.GOING_DOWN)

        # ── Legs elevated (most critical — weight 2.0) ──
        if legs_offset is not None:
            if legs_offset <= LEGS_ELEVATED_GOOD:
                scores["legs_elevated"] = 1.0
            elif legs_offset <= LEGS_ELEVATED_WARNING:
                issues.append("Keep feet off the floor \u2014 legs drifting down")
                joint_feedback["left_ankle"] = "warning"
                joint_feedback["right_ankle"] = "warning"
                scores["legs_elevated"] = 0.5
            else:
                issues.append("Feet touching the floor \u2014 lift legs back up")
                joint_feedback["left_ankle"] = "incorrect"
                joint_feedback["right_ankle"] = "incorrect"
                scores["legs_elevated"] = 0.2

        # ── Hip ROM at bottom (weight 1.5) ──
        if hip is not None and self._state == RepPhase.BOTTOM:
            if hip <= HIP_FULLY_CLOSED + 5:
                joint_feedback["left_hip"] = "correct"
                joint_feedback["right_hip"] = "correct"
                scores["hip_rom"] = 1.0
            elif hip <= HIP_GOOD_CLOSE:
                joint_feedback["left_hip"] = "correct"
                joint_feedback["right_hip"] = "correct"
                scores["hip_rom"] = 0.85
            elif hip <= HIP_HALF:
                joint_feedback["left_hip"] = "warning"
                joint_feedback["right_hip"] = "warning"
                issues.append("Crunch higher \u2014 bring torso and legs closer together")
                scores["hip_rom"] = 0.5
            else:
                joint_feedback["left_hip"] = "incorrect"
                joint_feedback["right_hip"] = "incorrect"
                issues.append("Not enough range \u2014 curl up more into the V")
                scores["hip_rom"] = 0.3
        elif hip is not None and in_crunch:
            joint_feedback["left_hip"] = "correct"
            joint_feedback["right_hip"] = "correct"
            scores["hip_rom"] = 0.9

        # ── Arms straight (weight 0.8) ──
        if elbow is not None:
            if elbow >= ELBOW_STRAIGHT:
                joint_feedback["left_elbow"] = "correct"
                joint_feedback["right_elbow"] = "correct"
                scores["arms_straight"] = 1.0
            elif elbow >= ELBOW_BENT_WARNING:
                joint_feedback["left_elbow"] = "warning"
                joint_feedback["right_elbow"] = "warning"
                issues.append("Keep arms extended \u2014 reach toward your shins")
                scores["arms_straight"] = 0.6
            else:
                joint_feedback["left_elbow"] = "incorrect"
                joint_feedback["right_elbow"] = "incorrect"
                issues.append("Arms too bent \u2014 straighten and reach forward")
                scores["arms_straight"] = 0.3

        # ── Controlled movement (weight 0.6) ──
        if hip_velocity is not None:
            if hip_velocity <= VELOCITY_GOOD:
                scores["velocity"] = 1.0
            elif hip_velocity <= VELOCITY_WARNING:
                scores["velocity"] = 0.6
            else:
                issues.append("Slow down \u2014 control the movement")
                scores["velocity"] = 0.3

        if not scores:
            return 0.0, joint_feedback, issues

        WEIGHTS = {
            "legs_elevated": 2.0, "hip_rom": 1.5,
            "arms_straight": 0.8, "velocity": 0.6,
        }
        weighted_sum = sum(scores[k] * WEIGHTS.get(k, 1.0) for k in scores)
        weight_total = sum(WEIGHTS.get(k, 1.0) for k in scores)
        total = weighted_sum / weight_total

        return max(0.0, min(1.0, total)), joint_feedback, issues
