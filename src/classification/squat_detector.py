"""
Dedicated squat detector with state machine rep counting and form assessment.

Angles tracked:
  - Knee (hip-knee-ankle): primary rep counting angle + depth check
  - Hip (shoulder-hip-knee): hip hinge assessment
  - Torso lean (vertical angle of shoulder-hip line)
  - Knee valgus (knee x vs ankle x offset)

State machine: TOP (standing) -> GOING_DOWN -> BOTTOM (squat) -> GOING_UP -> TOP (1 rep)
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
from .posture_classifier import PostureLabel
from ..pose_estimation.base import PoseResult
from ..utils.constants import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
)
from ..utils.geometry import calculate_angle

# Lowered visibility floor so side-view near-side landmarks still pass.
# Session trace analysis (data/sessions/2026-04-14_*_squat_*/trace.jsonl)
# shows MediaPipe's near-side landmarks from a pure side view genuinely
# live in 0.35–0.55 with frequent dips. 0.3 is the minimum floor where
# MediaPipe still returns *located* (not rig-inferred) landmarks.
SQUAT_VIS_THRESHOLD = 0.3

# ── Movement-gated session FSM thresholds ──────────────────────────────
# FORM LOCKED fires only when the knee has both (a) moved through at
# least SQUAT_MOTION_MIN_RANGE degrees in the recent rolling window AND
# (b) currently sits below SQUAT_KNEE_DESCEND. The movement range cleanly
# separates an active squat (lots of angle change) from a static sit
# (zero angle change), and catches slow descents that a pure time-window
# gate would miss.
SQUAT_KNEE_DESCEND = 140.0        # knee below this → descent committed
SQUAT_KNEE_AT_TOP = 165.0         # knee >= this → clearly standing at lockout
SQUAT_MOTION_WINDOW_S = 2.0       # rolling window for movement detection
SQUAT_MOTION_MIN_RANGE = 25.0     # degrees of knee range required for ACTIVE
SQUAT_REST_AT_TOP_S = 8.0         # standing at top this long → RESTING
SQUAT_REST_BELOW_TOP_S = 60.0     # below top this long without rep → RESTING

# Minimum forward torso lean (degrees from vertical) the user must exhibit
# **during the rep** for it to count as a squat. A standing knee-lift or
# leg-raise keeps the torso near-vertical (0–5°), whereas every legitimate
# squat — front, back, high-bar, low-bar, box — produces at least 8–10° of
# forward lean at the bottom. 7° is the conservative floor that rejects
# leg-lifts without rejecting stiff-upright front squats.
SQUAT_MIN_TORSO_LEAN_FOR_REP = 7.0
# Minimum hip-angle drop (degrees) during the rep. A squat hinges the hips
# from ~175° standing → ~95° parallel, so Δhip ≈ 80°. A standing leg-lift
# keeps the stance-leg hip at ~175° and only shrinks the *average* hip
# because one leg flexes. 35° is the conservative floor.
SQUAT_MIN_HIP_DROP_FOR_REP = 35.0


# ── Thresholds ──────────────────────────────────────────────────────────

# Knee angle (hip->knee->ankle).
# Escamilla (2001) defined parallel depth as ~100° anatomical hip flexion, with
# knee included angles of 70-90° at parallel (Caterisano 2002; Kotiuk 2023).
# Standing lockout is 170-180°. KNEE_PARALLEL is the state-machine bottom gate,
# so we pick 85° — mid-range of research — so that correct 80° parallel squats
# trigger BOTTOM (previous 95° demanded over-squatting and broke rep counting).
KNEE_STANDING = 175     # Research: 170-180° standing lockout
KNEE_PARALLEL = 85      # State-machine bottom gate (was 95, too strict)
KNEE_HALF = 115         # Half squat (warning zone — tightened from 120)

# Hip angle (shoulder->hip->knee). Research: 170-180° standing, 70-90° parallel.
HIP_STANDING = 175      # Was 160 — matches research lockout range
HIP_GOOD_DEPTH = 85     # Hips at knee level (was 90)
HIP_WARNING = 105       # Not deep enough (was 110)

# Torso lean — vertical angle of shoulder-hip vector.
# Hales et al. 2009: high-bar 20-35° at bottom, low-bar 30-45°. Default to high-bar.
# TODO: future per-user/bar-position calibration.
TORSO_GOOD = 30         # degrees from vertical (slight forward lean OK)
TORSO_WARNING = 45      # too much forward lean (high-bar limit)
TORSO_BAD = 55          # excessive lean (low-bar limit — beyond this is bad for any style)

# Knee valgus — frontal-plane projection angle (FPPA) in degrees.
# Herrington et al. 2014: healthy = 8.4° ± 5.1° FPPA, patellofemoral patients = 16.8°.
# Hewett et al. 2005: >8° valgus predicted ACL injury with 73% sensitivity.
# Tier-1 injury-critical.
VALGUS_WARN_DEG = 8.0    # research warning threshold
VALGUS_BAD_DEG = 15.0    # research injury-risk threshold

# Heel rise detection
HEEL_RISE_THRESHOLD = 0.02  # meters


class SquatDetector(RobustExerciseDetector):
    EXERCISE_NAME = "squat"
    PRIMARY_ANGLE_TOP = KNEE_STANDING
    PRIMARY_ANGLE_BOTTOM = KNEE_PARALLEL
    MIN_FORM_GATE = 0.3
    DESCENT_THRESHOLD = 15.0
    ASCENT_THRESHOLD = 10.0

    # ── Robust FSM config ───────────────────────────────────────────────
    # Visibility-only gate (the posture classifier's STANDING rule
    # empirically never fires on real MediaPipe poses — see trace replay
    # of 2026-04-13 sessions, all poses labeled plank/unknown). For
    # squat we only need the full lower body visible.
    SESSION_POSTURE_GATE = None
    VISIBILITY_GATE_LANDMARKS = (
        LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    )
    # Lowered from 0.6 so side-view landmarks (which hover ~0.4–0.6 on
    # the near side) can still latch into ACTIVE. Paired with the custom
    # _check_visibility_gate override which accepts one full side.
    VISIBILITY_GATE_THRESHOLD = SQUAT_VIS_THRESHOLD

    REP_DIRECTION = REP_DIRECTION_DECREASING
    # Rep-FSM thresholds on knee angle. Kept slightly lenient so that a
    # legitimate parallel squat (bottom ~85°) commits, while phantom reps
    # from idle micro-bends are rejected via REP_START_THRESHOLD gate.
    REP_COMPLETE_TOP = 160.0      # back at top zone (knee extended)
    REP_START_THRESHOLD = 155.0   # must dip below this to commit descent
    REP_DEPTH_THRESHOLD = 110.0   # knee must reach this to count as valid
    REP_BOTTOM_CLEAR = 120.0      # past this on ascent = leaving bottom
    # Knee angle can't plausibly go below ~30° (calf against hamstring).
    MIN_REAL_PRIMARY_DEG = 30.0
    # Knee moves slower than elbow — 60°/frame jump is the outlier floor.
    MAX_PRIMARY_JUMP_DEG = 60.0

    def __init__(self, **kwargs):
        # Initialize instance attributes BEFORE super().__init__ because the
        # base class __init__ calls self.reset(), which touches these fields.
        self._mid_descent_hip_angle = None
        self._bottom_trunk_angle = None   # torso lean snapshot at BOTTOM (for good-morning detection)
        # Movement-gated session FSM state
        self._at_top_since: Optional[float] = None
        self._below_top_since: Optional[float] = None
        # Rolling knee-angle history for movement-range check.
        self._knee_history: Deque[Tuple[float, float]] = deque()
        # Per-rep squat-shape witnesses — reset on TOP, used by
        # _should_count_rep to reject leg-lifts / knee-bends that are not
        # actually squats.
        self._rep_max_torso_lean: Optional[float] = None
        self._rep_hip_at_start: Optional[float] = None
        self._rep_min_hip: Optional[float] = None
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self._mid_descent_hip_angle = None
        self._bottom_trunk_angle = None
        self._at_top_since = None
        self._below_top_since = None
        self._knee_history.clear()
        self._rep_max_torso_lean = None
        self._rep_hip_at_start = None
        self._rep_min_hip = None

    # ── Movement-gated session FSM ──────────────────────────────────────
    # FORM LOCKED only fires when the knee shows active movement — the
    # rolling-window range check rules out static positions (sitting,
    # standing still) and catches slow descents that a fixed time-window
    # gate would miss. Sitting in a chair has zero knee motion after
    # settling, so it never latches into ACTIVE.
    #
    # Rest conditions: 8s standing idle at lockout, or 60s below top
    # without returning (gave up / sat down permanently).
    def _update_session_state(self, now: float, visibility_ok: bool) -> None:
        knee = self._get_smooth("knee")

        if knee is None:
            super()._update_session_state(now, visibility_ok)
            return

        is_at_top = knee >= SQUAT_KNEE_AT_TOP
        is_descending = knee < SQUAT_KNEE_DESCEND

        # Update top/below-top continuous-window timers
        if is_at_top:
            if self._at_top_since is None:
                self._at_top_since = now
            self._below_top_since = None
        else:
            if self._below_top_since is None:
                self._below_top_since = now
            self._at_top_since = None

        # Rolling knee-angle history (drop entries older than the window)
        self._knee_history.append((now, knee))
        window_start = now - SQUAT_MOTION_WINDOW_S
        while self._knee_history and self._knee_history[0][0] < window_start:
            self._knee_history.popleft()

        # Movement range over the window
        if len(self._knee_history) >= 2:
            values = [v for _, v in self._knee_history]
            motion_range = max(values) - min(values)
        else:
            motion_range = 0.0
        has_active_motion = motion_range >= SQUAT_MOTION_MIN_RANGE

        s = self._session_state

        # Visibility loss → delegate to base close logic
        if not visibility_ok:
            if s == SessionState.ACTIVE:
                self._unknown_streak += 1
                if self._unknown_streak >= self.UNKNOWN_GRACE_FRAMES:
                    self._close_active_set_or_rollback(now)
                    self._knee_history.clear()
            return
        self._unknown_streak = 0

        # IDLE → SETUP: body visible
        if s == SessionState.IDLE:
            self._session_state = SessionState.SETUP
            return

        # SETUP → ACTIVE: active knee motion + currently in descent zone
        if s == SessionState.SETUP:
            if is_descending and has_active_motion:
                self._session_state = SessionState.ACTIVE
                self._reset_robust_rep_fsm()
                self._rep_start_time = now
                self._seed_rep_fsm_from_pre_active(now)
            return

        # RESTING → ACTIVE: same as SETUP but no trace seeding
        if s == SessionState.RESTING:
            if is_descending and has_active_motion:
                self._session_state = SessionState.ACTIVE
                self._reset_robust_rep_fsm()
                self._rep_start_time = now
            return

        # ACTIVE → RESTING
        if s == SessionState.ACTIVE:
            if (
                self._at_top_since is not None
                and now - self._at_top_since >= SQUAT_REST_AT_TOP_S
            ):
                self._close_active_set_or_rollback(now)
                # Next set requires fresh motion to re-latch.
                self._knee_history.clear()
                return
            if (
                self._below_top_since is not None
                and now - self._below_top_since >= SQUAT_REST_BELOW_TOP_S
            ):
                self._close_active_set_or_rollback(now)
                self._knee_history.clear()
                return

    # ── Side-view-friendly visibility gate ──────────────────────────────
    # Default base class gate requires ALL of (L+R hip, knee, ankle) at
    # 0.6 — excludes pure side cameras because MediaPipe infers the far
    # side and reports it at ~0.3–0.5 confidence. Accept the gate if
    # EITHER side's hip-knee-ankle chain is fully visible at the squat
    # threshold. `_lenient_avg_angle` below falls back to whichever side
    # is actually available.
    def _check_visibility_gate(self, pose) -> bool:
        left_ok = all(
            pose.is_visible(idx, self.VISIBILITY_GATE_THRESHOLD)
            for idx in (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
        )
        right_ok = all(
            pose.is_visible(idx, self.VISIBILITY_GATE_THRESHOLD)
            for idx in (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
        )
        return left_ok or right_ok

    def _lenient_side_angle(
        self,
        pose: PoseResult,
        a_idx: int,
        b_idx: int,
        c_idx: int,
        threshold: float = SQUAT_VIS_THRESHOLD,
    ) -> Optional[float]:
        """World-coordinate angle at vertex b, accepting landmarks whose
        visibility is at least `threshold` (default 0.4 for squat, vs the
        global MIN_VISIBILITY=0.5 used by the base class's `_angle_at`).
        Near-side landmarks from a pure side view often sit in that
        0.4–0.5 range; world coordinates themselves are camera-angle
        independent once MediaPipe has located the joint."""
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
        """Average left/right angles using the lenient visibility floor,
        falling back to whichever side is available. Stores L/R values
        for asymmetry detection (same contract as base `_avg_angle`)."""
        left = self._lenient_side_angle(pose, *left_triple)
        right = self._lenient_side_angle(pose, *right_triple)
        self._lr_angles[name] = (left, right)
        if left is not None and right is not None:
            return (left + right) / 2
        return left if left is not None else right

    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        # Use lenient angle computation so side-view (and 3/4 view) work:
        # near-side landmarks from MediaPipe frequently sit in the 0.4–0.6
        # visibility range which the default 0.5 floor would reject.
        knee = self._lenient_avg_angle(
            pose,
            (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
            (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
            name="knee",
        )
        hip = self._lenient_avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
            (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
            name="hip",
        )

        # Torso lean: vertical angle of shoulder-hip line
        torso = self._compute_torso_lean(pose)

        # Knee valgus: normalized ratio of x-offset to femur length
        # NOTE: this is a proxy for frontal-plane projection angle (FPPA) rather
        # than true FPPA (which requires projection onto the frontal plane).
        # Thresholds VALGUS_WARN_RATIO/BAD_RATIO are tuned to catch visually
        # noticeable valgus corresponding approximately to Herrington 2014's 8°/15°.
        # TODO: proper FPPA via frontal-plane projection.
        valgus = self._compute_knee_valgus(pose)

        # Ankle dorsiflexion — knee->ankle->foot_index.
        # Research (Hemmerich 2006): parallel squat needs 25-35° dorsiflexion,
        # which corresponds to included angles of 55-65°. Limited dorsiflexion
        # is the strongest predictor of squat depth (Kim 2015).
        ankle_dorsi = self._lenient_avg_angle(
            pose,
            (LEFT_KNEE, LEFT_ANKLE, LEFT_FOOT_INDEX),
            (RIGHT_KNEE, RIGHT_ANKLE, RIGHT_FOOT_INDEX),
            name="ankle_dorsi",
        )

        # Heel rise detection
        heel_rise = self._compute_heel_rise(pose)

        # World-coord depth check
        depth_world = self._compute_depth_world(pose)

        # Track hip angle at mid-descent for butt wink detection
        if knee is not None and self._state == RepPhase.GOING_DOWN and 115 <= knee <= 125:
            hip_val = self._lenient_avg_angle(
                pose,
                (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
                (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
                name="hip",
            )
            if hip_val is not None:
                self._mid_descent_hip_angle = hip_val

        # Snapshot trunk angle at BOTTOM for good-morning detection during ascent
        if self._state == RepPhase.BOTTOM and torso is not None and self._bottom_trunk_angle is None:
            self._bottom_trunk_angle = torso
        elif self._state == RepPhase.TOP:
            self._bottom_trunk_angle = None  # clear for next rep

        # Per-rep squat-shape witnesses. The rep FSM enters GOING_DOWN the
        # moment primary (knee) dips below PRIMARY_ANGLE_TOP-DESCENT, which
        # is the cleanest "rep started" signal available here.
        if self._state == RepPhase.TOP:
            self._rep_max_torso_lean = None
            self._rep_hip_at_start = hip  # freshest standing-hip baseline
            self._rep_min_hip = None
        else:
            if torso is not None:
                if self._rep_max_torso_lean is None or torso > self._rep_max_torso_lean:
                    self._rep_max_torso_lean = torso
            if hip is not None:
                if self._rep_min_hip is None or hip < self._rep_min_hip:
                    self._rep_min_hip = hip

        return {
            "primary": knee, "knee": knee, "hip": hip, "torso": torso,
            "valgus": valgus, "ankle_dorsi": ankle_dorsi,
            "heel_rise": heel_rise, "depth_world": depth_world,
        }

    def _compute_torso_lean(self, pose: PoseResult) -> Optional[float]:
        """Compute torso forward lean angle from vertical."""
        left_s = pose.get_world_landmark(LEFT_SHOULDER) if pose.is_visible(LEFT_SHOULDER, SQUAT_VIS_THRESHOLD) else None
        right_s = pose.get_world_landmark(RIGHT_SHOULDER) if pose.is_visible(RIGHT_SHOULDER, SQUAT_VIS_THRESHOLD) else None
        left_h = pose.get_world_landmark(LEFT_HIP) if pose.is_visible(LEFT_HIP, SQUAT_VIS_THRESHOLD) else None
        right_h = pose.get_world_landmark(RIGHT_HIP) if pose.is_visible(RIGHT_HIP, SQUAT_VIS_THRESHOLD) else None

        if left_s is None and right_s is None:
            return None
        if left_h is None and right_h is None:
            return None

        shoulder = left_s if right_s is None else (right_s if left_s is None else (np.array(left_s) + np.array(right_s)) / 2)
        hip = left_h if right_h is None else (right_h if left_h is None else (np.array(left_h) + np.array(right_h)) / 2)

        shoulder = np.array(shoulder)
        hip = np.array(hip)
        diff = shoulder - hip
        # Angle from vertical (y-axis)
        vertical = np.array([0, 1, 0])
        cos_angle = np.dot(diff, vertical) / (np.linalg.norm(diff) * np.linalg.norm(vertical) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))

    def _compute_knee_valgus(self, pose: PoseResult) -> Optional[float]:
        """Compute knee valgus as frontal-plane projection angle (FPPA) in degrees.
        Proper FPPA: angle at knee between (hip→knee) and (ankle→knee) vectors,
        projected onto the frontal plane (XY — drop Z/depth). Straight leg = 0°
        deviation from collinear, valgus = positive deviation. Returns max(L, R)."""
        max_dev = None
        for hip_idx, knee_idx, ankle_idx in [(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
                                              (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)]:
            if not all(pose.is_visible(i, SQUAT_VIS_THRESHOLD) for i in (hip_idx, knee_idx, ankle_idx)):
                continue
            hip = np.array(pose.get_world_landmark(hip_idx))
            knee = np.array(pose.get_world_landmark(knee_idx))
            ankle = np.array(pose.get_world_landmark(ankle_idx))

            # Project onto frontal plane: keep X (lateral) and Y (vertical), drop Z
            v1 = np.array([hip[0] - knee[0], hip[1] - knee[1]])
            v2 = np.array([ankle[0] - knee[0], ankle[1] - knee[1]])
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 < 0.05 or n2 < 0.05:
                continue

            # Included angle at knee (180° = straight leg, collinear)
            cos = float(np.dot(v1, v2) / (n1 * n2))
            angle = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
            deviation = abs(180.0 - angle)  # degrees from straight

            if max_dev is None or deviation > max_dev:
                max_dev = deviation
        return max_dev

    def _compute_heel_rise(self, pose: PoseResult) -> Optional[float]:
        """Detect if heels are lifting off ground. Returns avg rise in meters."""
        rises = []
        for heel_idx, foot_idx in [(LEFT_HEEL, LEFT_FOOT_INDEX), (RIGHT_HEEL, RIGHT_FOOT_INDEX)]:
            if pose.is_visible(heel_idx, SQUAT_VIS_THRESHOLD) and pose.is_visible(foot_idx, SQUAT_VIS_THRESHOLD):
                heel_y = pose.get_world_landmark(heel_idx)[1]
                foot_y = pose.get_world_landmark(foot_idx)[1]
                rises.append(heel_y - foot_y)  # positive = heel higher than toes
        return sum(rises) / len(rises) if rises else None

    def _compute_depth_world(self, pose: PoseResult) -> Optional[float]:
        """Check if hip is below knee using world Y coordinates.
        Returns difference: positive = hip below knee (good depth)."""
        hip_ys = []
        knee_ys = []
        for h_idx, k_idx in [(LEFT_HIP, LEFT_KNEE), (RIGHT_HIP, RIGHT_KNEE)]:
            if pose.is_visible(h_idx, SQUAT_VIS_THRESHOLD) and pose.is_visible(k_idx, SQUAT_VIS_THRESHOLD):
                hip_ys.append(pose.get_world_landmark(h_idx)[1])
                knee_ys.append(pose.get_world_landmark(k_idx)[1])
        if not hip_ys:
            return None
        # In MediaPipe world coords, Y increases downward, so hip_y > knee_y means hip is lower
        return (sum(hip_ys) / len(hip_ys)) - (sum(knee_ys) / len(knee_ys))

    def _should_count_rep(self, elapsed: float, angles: Dict[str, Optional[float]]) -> bool:
        # Base gate: duration, descent time, form score.
        if not super()._should_count_rep(elapsed, angles):
            return False
        # Shape gate: a real squat produces BOTH (a) visible forward torso
        # lean (≥ SQUAT_MIN_TORSO_LEAN_FOR_REP) AND (b) a meaningful hip
        # hinge (Δhip ≥ SQUAT_MIN_HIP_DROP_FOR_REP). Standing knee-lifts,
        # leg raises, and seated knee extensions fail one or both.
        #
        # If neither torso nor hip can be measured for the whole rep (very
        # bad visibility on both shoulders AND one full hip side), fall
        # through — we don't want to reject legitimate reps when the
        # measurement itself failed.
        torso_ok = (
            self._rep_max_torso_lean is None
            or self._rep_max_torso_lean >= SQUAT_MIN_TORSO_LEAN_FOR_REP
        )
        hip_drop = None
        if self._rep_hip_at_start is not None and self._rep_min_hip is not None:
            hip_drop = self._rep_hip_at_start - self._rep_min_hip
        hip_ok = (hip_drop is None) or (hip_drop >= SQUAT_MIN_HIP_DROP_FOR_REP)

        # Both must pass. If one measurement is unavailable we accept on
        # the other alone (leniency for occlusion).
        return torso_ok and hip_ok

    def _check_start_position(self, angles: Dict[str, Optional[float]]) -> Tuple[bool, List[str]]:
        issues = []
        knee = angles.get("knee")
        if knee is None:
            issues.append("Stand facing the camera — knees must be visible")
            return False, issues
        if knee < KNEE_STANDING - 20:
            issues.append("Stand up straight to begin")
        if not issues:
            return True, []
        return False, issues

    def _get_missing_parts(self, angles: Dict[str, Optional[float]]) -> List[str]:
        missing = []
        if angles.get("knee") is None:
            missing.append("knees and hips")
        return missing if missing else ["body"]

    def _assess_form(self, angles: Dict[str, Optional[float]]) -> Tuple[float, Dict[str, str], List[str]]:
        joint_feedback = {}
        issues = []
        scores = {}

        knee = angles.get("knee")
        hip = angles.get("hip")
        torso = angles.get("torso")
        valgus = angles.get("valgus")
        ankle_dorsi = angles.get("ankle_dorsi")

        in_transit = self._state in (RepPhase.BOTTOM, RepPhase.GOING_UP, RepPhase.GOING_DOWN)

        # Pre-compute suppression flags
        heel_rise_val = angles.get("heel_rise")
        _valgus_active = valgus is not None and in_transit and valgus > VALGUS_WARN_DEG
        _butt_wink_active = (self._state == RepPhase.BOTTOM
                             and self._mid_descent_hip_angle is not None
                             and hip is not None
                             and (self._mid_descent_hip_angle - hip) > 15)
        _good_morning_active = (self._state == RepPhase.GOING_UP
                                and torso is not None
                                and self._bottom_trunk_angle is not None
                                and torso - self._bottom_trunk_angle > 10)
        _heel_rise_active = heel_rise_val is not None and heel_rise_val > HEEL_RISE_THRESHOLD

        # ── Knee depth assessment (monotonic curve) ──
        if knee is not None:
            if in_transit:
                if knee <= KNEE_PARALLEL:          # <= 85 — parallel depth (research)
                    joint_feedback["left_knee"] = "correct"
                    joint_feedback["right_knee"] = "correct"
                    scores["knee"] = 1.0
                elif knee <= KNEE_HALF:            # 85-115 — acceptable partial
                    joint_feedback["left_knee"] = "correct"
                    joint_feedback["right_knee"] = "correct"
                    scores["knee"] = 0.8
                elif knee <= 140:                  # 115-140 — too shallow, active coaching
                    joint_feedback["left_knee"] = "warning"
                    joint_feedback["right_knee"] = "warning"
                    if not _valgus_active and not _butt_wink_active:
                        issues.append("Go deeper — thighs toward parallel")
                    scores["knee"] = 0.5
                else:                              # > 140 — barely squatting
                    joint_feedback["left_knee"] = "warning"
                    joint_feedback["right_knee"] = "warning"
                    if not _valgus_active and not _butt_wink_active:
                        issues.append("Squat deeper — aim for thighs parallel")
                    scores["knee"] = 0.3
            else:
                # TOP state — lockout check
                if knee >= KNEE_STANDING - 10:
                    joint_feedback["left_knee"] = "correct"
                    joint_feedback["right_knee"] = "correct"
                    scores["knee"] = 1.0
                else:
                    joint_feedback["left_knee"] = "warning"
                    joint_feedback["right_knee"] = "warning"
                    scores["knee"] = 0.7

        # ── Hip depth assessment (weighted 1.5x) ──
        if hip is not None:
            if self._state in (RepPhase.BOTTOM, RepPhase.GOING_UP):
                if hip <= HIP_GOOD_DEPTH + 5:      # <= 90 — hip crease at/below knee
                    joint_feedback["left_hip"] = "correct"
                    joint_feedback["right_hip"] = "correct"
                    scores["hip"] = 1.0
                elif hip <= HIP_WARNING:           # 90-105 — close but not quite
                    joint_feedback["left_hip"] = "warning"
                    joint_feedback["right_hip"] = "warning"
                    issues.append("Drop hips lower — aim for hip crease below knee")
                    scores["hip"] = 0.5
                else:
                    joint_feedback["left_hip"] = "warning"
                    joint_feedback["right_hip"] = "warning"
                    scores["hip"] = 0.7
            else:
                joint_feedback["left_hip"] = "correct"
                joint_feedback["right_hip"] = "correct"
                scores["hip"] = 1.0

        # ── Torso lean ──
        if torso is not None:
            if torso <= TORSO_GOOD:
                joint_feedback["left_shoulder"] = "correct"
                joint_feedback["right_shoulder"] = "correct"
                scores["torso"] = 1.0
            elif torso <= TORSO_WARNING:
                joint_feedback["left_shoulder"] = "warning"
                joint_feedback["right_shoulder"] = "warning"
                if not _good_morning_active:
                    issues.append("Keep chest up — reduce forward lean")
                scores["torso"] = 0.5
            else:
                joint_feedback["left_shoulder"] = "incorrect"
                joint_feedback["right_shoulder"] = "incorrect"
                if not _good_morning_active:
                    issues.append("Too much forward lean — try to keep chest up")
                scores["torso"] = 0.2

        # ── Knee valgus — FPPA in degrees (Tier-1 injury-critical, weighted 1.4x) ──
        if valgus is not None and in_transit:
            if valgus > VALGUS_BAD_DEG:
                issues.append("Knees caving in — push them out over your toes")
                scores["valgus"] = 0.2
                joint_feedback["left_knee"] = "incorrect"
                joint_feedback["right_knee"] = "incorrect"
            elif valgus > VALGUS_WARN_DEG:
                issues.append("Slight knee cave — focus on pushing knees outward")
                scores["valgus"] = 0.6

        # ── Ankle dorsiflexion (T11 — new) ──
        # Research: parallel needs 25-35° dorsiflexion (included angle 55-65°).
        # Flag <20° dorsiflexion (>70° included angle) at bottom as limited mobility.
        if ankle_dorsi is not None and self._state == RepPhase.BOTTOM:
            if ankle_dorsi > 80:  # less than 10° dorsiflexion
                if not _heel_rise_active:
                    issues.append("Ankles tight — try a wider stance or elevate heels")
                scores["ankle_dorsi"] = 0.4
                joint_feedback["left_ankle"] = "warning"
                joint_feedback["right_ankle"] = "warning"
            elif ankle_dorsi > 70:  # less than 20° dorsiflexion
                if not _heel_rise_active:
                    issues.append("Ankles a bit tight — try widening your stance")
                scores["ankle_dorsi"] = 0.7

        # ── Good morning squat detection (T12 — new) ──
        # If during ascent the torso angle grows by >10° vs. the bottom snapshot,
        # the hips are outrunning the chest — classic "good morning squat" pattern.
        if (self._state == RepPhase.GOING_UP and torso is not None
                and self._bottom_trunk_angle is not None
                and torso - self._bottom_trunk_angle > 10):
            issues.append("Good morning pattern — drive chest up with the hips")
            scores["good_morning"] = 0.3
            joint_feedback["left_shoulder"] = "incorrect"
            joint_feedback["right_shoulder"] = "incorrect"

        # ── Heel rise ──
        heel_rise = angles.get("heel_rise")
        if heel_rise is not None and heel_rise > HEEL_RISE_THRESHOLD:
            issues.append("Heels lifting — keep weight on heels")
            scores["heel_rise"] = 0.4
            joint_feedback["left_ankle"] = "warning"
            joint_feedback["right_ankle"] = "warning"

        # ── Butt wink (pelvis tuck at bottom) ──
        if (self._state == RepPhase.BOTTOM and self._mid_descent_hip_angle is not None
                and hip is not None):
            hip_drop = self._mid_descent_hip_angle - hip
            if hip_drop > 20:
                issues.append("Pelvis tucking under — try stopping just above this depth")
                scores["butt_wink"] = 0.3
            elif hip_drop > 15:
                issues.append("Slight pelvis tuck — that depth may be your limit for now")
                scores["butt_wink"] = 0.6

        if not scores:
            return 0.0, joint_feedback, issues

        # Valgus Tier-1 weight raised from 1.0 to 1.4 (injury-critical per Hewett 2005)
        WEIGHTS = {
            "knee": 1.0, "hip": 1.5, "torso": 1.0, "valgus": 1.4,
            "ankle_dorsi": 0.7, "good_morning": 1.2,
            "heel_rise": 0.8, "butt_wink": 0.8,
        }
        weighted_sum = sum(scores[k] * WEIGHTS.get(k, 1.0) for k in scores)
        weight_total = sum(WEIGHTS.get(k, 1.0) for k in scores)
        total = weighted_sum / weight_total

        return max(0.0, min(1.0, total)), joint_feedback, issues
