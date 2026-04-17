"""
Dedicated lateral raise detector with state machine and form assessment.

Angles tracked:
  - Shoulder abduction (elbow-shoulder-hip): primary rep counting angle
  - Elbow angle (shoulder-elbow-wrist): should stay slightly bent (~150-170°)
  - Torso lean (shoulder-hip vertical angle): should stay upright
  - Wrist vs elbow height: anti-cheat for wrist flicking
  - Ear-shoulder distance: anti-cheat for shrugging

Camera: FRONT VIEW — full upper body visible.
State machine: TOP (arms at sides) -> GOING_DOWN (raising) -> BOTTOM (T-shape) -> GOING_UP -> TOP (1 rep)
Note: uses REP_DIRECTION_INCREASING — angle increases from sides to T-shape.
Under INCREASING semantics:
  TOP         = arms at sides (low abduction, primary near REP_COMPLETE_TOP=20°)
  GOING_DOWN  = raising arms (abduction rising toward T)
  BOTTOM      = arms at T-shape (max abduction, primary ≥ REP_DEPTH_THRESHOLD=70°)
  GOING_UP    = lowering arms back to sides (abduction falling)
"""

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .base_detector import (
    RobustExerciseDetector,
    RepPhase,
    REP_DIRECTION_INCREASING,
    SessionState,
)
from ..pose_estimation.base import PoseResult
from ..utils.constants import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
    LEFT_EAR, RIGHT_EAR,
)
from ..utils.geometry import calculate_angle


# ── Thresholds ──
# Elbow angle: should stay slightly bent throughout
ELBOW_GOOD = 150          # minimum acceptable bend
ELBOW_WARNING = 140       # lever-shortening cheat starts
ELBOW_BAD = 130           # significant elbow bend

# Torso lean from vertical
TORSO_GOOD = 10
TORSO_WARNING = 15
TORSO_BAD = 20

# Shoulder abduction limits
ABDUCTION_OVER = 95       # above 90° = traps taking over

# Shrug detection: % shrinkage of ear-shoulder distance
SHRUG_WARNING_PCT = 0.15
SHRUG_BAD_PCT = 0.25

# ── Movement-gated session FSM ──
LAT_RAISE_ABD_AT_BOTTOM = 25.0       # arms at sides
LAT_RAISE_ABD_ASCENDING = 40.0       # committed to raise
LAT_RAISE_MOTION_WINDOW_S = 2.0
LAT_RAISE_MOTION_MIN_RANGE = 20.0
LAT_RAISE_REST_AT_BOTTOM_S = 8.0     # idle at sides → RESTING
LAT_RAISE_REST_ABOVE_BOTTOM_S = 60.0


class LateralRaiseDetector(RobustExerciseDetector):
    EXERCISE_NAME = "lateral_raise"
    PRIMARY_ANGLE_TOP = 90.0      # T-shape at top
    PRIMARY_ANGLE_BOTTOM = 10.0   # arms at sides
    DESCENT_THRESHOLD = 10.0
    ASCENT_THRESHOLD = 10.0
    MIN_FORM_GATE = 0.3

    SESSION_POSTURE_GATE = None
    VISIBILITY_GATE_LANDMARKS = (
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_ELBOW, RIGHT_ELBOW,
        LEFT_WRIST, RIGHT_WRIST,
        LEFT_HIP, RIGHT_HIP,
    )
    VISIBILITY_GATE_THRESHOLD = 0.6  # front view = standard

    REP_DIRECTION = REP_DIRECTION_INCREASING  # angle INCREASES sides→T
    REP_COMPLETE_TOP = 20.0       # arms back at sides = ready for next rep
    REP_START_THRESHOLD = 30.0    # past this = committed to raise
    REP_DEPTH_THRESHOLD = 70.0    # must reach near-horizontal
    REP_BOTTOM_CLEAR = 60.0       # descending past this = leaving top
    MIN_REAL_PRIMARY_DEG = 0.0    # abduction can be ~0
    MAX_PRIMARY_JUMP_DEG = 50.0   # shoulder moves slower than elbow

    def __init__(self, **kwargs):
        # Instance state BEFORE super().__init__() — reset() calls .clear()
        self._at_bottom_since: Optional[float] = None
        self._above_bottom_since: Optional[float] = None
        self._abduction_history: Deque[Tuple[float, float]] = deque()
        self._baseline_ear_shoulder_dist: Optional[float] = None
        self._ear_shoulder_samples: List[float] = []
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self._at_bottom_since = None
        self._above_bottom_since = None
        self._abduction_history.clear()
        self._baseline_ear_shoulder_dist = None
        self._ear_shoulder_samples = []

    # ── Angle computation ──────────────────────────────────────────────
    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        # Primary: shoulder abduction (angle at shoulder between elbow and hip)
        abduction = self._avg_angle(
            pose,
            (LEFT_ELBOW, LEFT_SHOULDER, LEFT_HIP),
            (RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP),
            name="abduction",
        )

        # Elbow angle: should stay slightly bent
        elbow = self._avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
            (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
            name="elbow",
        )

        # Torso lean from vertical
        torso = self._compute_torso_lean(pose)

        # Wrist above elbow check (image coords, only meaningful near top of rep)
        wrist_above_elbow = self._compute_wrist_above_elbow(pose, abduction)

        # Shrug detection: ear-shoulder distance
        shrug_ratio = self._compute_shrug_ratio(pose)

        return {
            "primary": abduction, "abduction": abduction,
            "elbow": elbow, "torso": torso,
            "wrist_above_elbow": wrist_above_elbow,
            "shrug_ratio": shrug_ratio,
        }

    def _compute_torso_lean(self, pose: PoseResult) -> Optional[float]:
        """Torso lean from vertical — average of visible sides."""
        angles = []
        for s_idx, h_idx in [(LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP)]:
            if pose.is_visible(s_idx) and pose.is_visible(h_idx):
                s = np.array(pose.get_world_landmark(s_idx))
                h = np.array(pose.get_world_landmark(h_idx))
                diff = s - h
                vertical = np.array([0, 1, 0])
                cos = np.dot(diff, vertical) / (np.linalg.norm(diff) + 1e-8)
                angles.append(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
        return sum(angles) / len(angles) if angles else None

    def _compute_wrist_above_elbow(
        self, pose: PoseResult, abduction: Optional[float]
    ) -> Optional[float]:
        """Check if wrists are above elbows at the top of the rep.
        Returns positive value if wrist is above elbow (bad — flicking cheat).
        Only meaningful when arms are raised (abduction > 60°)."""
        if abduction is None or abduction < 60:
            return None  # not at top yet, don't check

        diffs = []
        for w_idx, e_idx in [(LEFT_WRIST, LEFT_ELBOW), (RIGHT_WRIST, RIGHT_ELBOW)]:
            if pose.is_visible(w_idx) and pose.is_visible(e_idx):
                # Image coords: lower Y = higher in frame
                wrist_y = pose.landmarks[w_idx][1]
                elbow_y = pose.landmarks[e_idx][1]
                # Positive = wrist above elbow (bad)
                diffs.append(float(elbow_y - wrist_y))
        if not diffs:
            return None
        return max(diffs)  # worst side

    def _compute_shrug_ratio(self, pose: PoseResult) -> Optional[float]:
        """Compute current ear-shoulder distance relative to baseline.
        Returns ratio of distance shrinkage (0 = no shrink, 1 = fully collapsed)."""
        dists = []
        for e_idx, s_idx in [(LEFT_EAR, LEFT_SHOULDER), (RIGHT_EAR, RIGHT_SHOULDER)]:
            if pose.is_visible(e_idx, 0.3) and pose.is_visible(s_idx):
                ear = np.array(pose.get_world_landmark(e_idx))
                shoulder = np.array(pose.get_world_landmark(s_idx))
                dists.append(float(np.linalg.norm(ear - shoulder)))

        if not dists:
            return None

        current_dist = sum(dists) / len(dists)

        # Build baseline from first 10 valid samples
        if self._baseline_ear_shoulder_dist is None:
            self._ear_shoulder_samples.append(current_dist)
            if len(self._ear_shoulder_samples) >= 10:
                self._baseline_ear_shoulder_dist = (
                    sum(self._ear_shoulder_samples) / len(self._ear_shoulder_samples)
                )
                self._ear_shoulder_samples = []
            return None  # still building baseline

        # Ratio: how much has it shrunk? 0 = same, positive = shrunk
        if self._baseline_ear_shoulder_dist < 1e-6:
            return None
        shrinkage = 1.0 - (current_dist / self._baseline_ear_shoulder_dist)
        return max(0.0, shrinkage)  # clamp negative (got longer = fine)

    # ── Movement-gated session FSM ─────────────────────────────────────
    def _update_session_state(self, now: float, visibility_ok: bool) -> None:
        abduction = self._get_smooth("abduction")
        if abduction is None:
            super()._update_session_state(now, visibility_ok)
            return

        is_at_bottom = abduction <= LAT_RAISE_ABD_AT_BOTTOM
        is_ascending = abduction > LAT_RAISE_ABD_ASCENDING

        if is_at_bottom:
            if self._at_bottom_since is None:
                self._at_bottom_since = now
            self._above_bottom_since = None
        else:
            if self._above_bottom_since is None:
                self._above_bottom_since = now
            self._at_bottom_since = None

        self._abduction_history.append((now, abduction))
        window_start = now - LAT_RAISE_MOTION_WINDOW_S
        while self._abduction_history and self._abduction_history[0][0] < window_start:
            self._abduction_history.popleft()

        if len(self._abduction_history) >= 2:
            values = [v for _, v in self._abduction_history]
            motion_range = max(values) - min(values)
        else:
            motion_range = 0.0
        has_active_motion = motion_range >= LAT_RAISE_MOTION_MIN_RANGE

        s = self._session_state

        if not visibility_ok:
            if s == SessionState.ACTIVE:
                self._unknown_streak += 1
                if self._unknown_streak >= self.UNKNOWN_GRACE_FRAMES:
                    self._close_active_set_or_rollback(now)
                    self._abduction_history.clear()
            return
        self._unknown_streak = 0

        if s == SessionState.IDLE:
            self._session_state = SessionState.SETUP
            return

        if s == SessionState.SETUP:
            if is_ascending and has_active_motion:
                self._session_state = SessionState.ACTIVE
                self._reset_robust_rep_fsm()
                self._rep_start_time = now
                self._seed_rep_fsm_from_pre_active(now)
            return

        if s == SessionState.RESTING:
            if is_ascending and has_active_motion:
                self._session_state = SessionState.ACTIVE
                self._reset_robust_rep_fsm()
                self._rep_start_time = now
            return

        if s == SessionState.ACTIVE:
            if (
                self._at_bottom_since is not None
                and now - self._at_bottom_since >= LAT_RAISE_REST_AT_BOTTOM_S
            ):
                self._close_active_set_or_rollback(now)
                self._abduction_history.clear()
                return
            if (
                self._above_bottom_since is not None
                and now - self._above_bottom_since >= LAT_RAISE_REST_ABOVE_BOTTOM_S
            ):
                self._close_active_set_or_rollback(now)
                self._abduction_history.clear()
                return

    # ── Start position / missing parts ─────────────────────────────────
    def _check_start_position(
        self, angles: Dict[str, Optional[float]]
    ) -> Tuple[bool, List[str]]:
        issues = []
        abduction = angles.get("abduction")
        if abduction is None:
            return False, ["Stand facing the camera \u2014 arms at your sides with dumbbells visible"]
        if abduction > 30:
            issues.append("Lower arms to your sides to begin")
        return (len(issues) == 0, issues)

    def _get_missing_parts(self, angles: Dict[str, Optional[float]]) -> List[str]:
        if angles.get("abduction") is None:
            return ["arms and shoulders"]
        return ["body"]

    # ── Form assessment ────────────────────────────────────────────────
    def _assess_form(
        self, angles: Dict[str, Optional[float]]
    ) -> Tuple[float, Dict[str, str], List[str]]:
        joint_feedback: Dict[str, str] = {}
        issues: List[str] = []
        scores: Dict[str, float] = {}

        abduction = angles.get("abduction")
        elbow = angles.get("elbow")
        torso = angles.get("torso")
        wrist_above = angles.get("wrist_above_elbow")
        shrug = angles.get("shrug_ratio")

        # ── Elbow bend (lever cheat — most critical, weight 2.0) ──
        if elbow is not None:
            if elbow >= ELBOW_GOOD:
                joint_feedback["left_elbow"] = "correct"
                joint_feedback["right_elbow"] = "correct"
                scores["elbow_bend"] = 1.0
            elif elbow >= ELBOW_WARNING:
                joint_feedback["left_elbow"] = "warning"
                joint_feedback["right_elbow"] = "warning"
                issues.append("Keep a slight bend \u2014 don't curl the elbows")
                scores["elbow_bend"] = 0.5
            else:
                joint_feedback["left_elbow"] = "incorrect"
                joint_feedback["right_elbow"] = "incorrect"
                issues.append("Arms too bent \u2014 straighten elbows, use lighter weight")
                scores["elbow_bend"] = 0.2

        # ── Wrist above elbow at top (flick cheat — weight 1.5) ──
        if wrist_above is not None and wrist_above > 0.01:
            joint_feedback["left_wrist"] = "incorrect"
            joint_feedback["right_wrist"] = "incorrect"
            issues.append("Lead with elbows, not wrists \u2014 don't flick the weight up")
            scores["wrist_flick"] = 0.3

        # ── Torso upright (weight 1.2) ──
        if torso is not None:
            if torso <= TORSO_GOOD:
                joint_feedback["left_hip"] = "correct"
                joint_feedback["right_hip"] = "correct"
                scores["torso"] = 1.0
            elif torso <= TORSO_WARNING:
                joint_feedback["left_hip"] = "warning"
                joint_feedback["right_hip"] = "warning"
                issues.append("Stay upright \u2014 don't swing your body")
                scores["torso"] = 0.5
            else:
                joint_feedback["left_hip"] = "incorrect"
                joint_feedback["right_hip"] = "incorrect"
                issues.append("Too much body swing \u2014 reduce weight or slow down")
                scores["torso"] = 0.2

        # ── Shrugging (weight 1.0) ──
        if shrug is not None:
            if shrug < SHRUG_WARNING_PCT:
                joint_feedback["left_shoulder"] = joint_feedback.get("left_shoulder", "correct")
                joint_feedback["right_shoulder"] = joint_feedback.get("right_shoulder", "correct")
                scores["shrug"] = 1.0
            elif shrug < SHRUG_BAD_PCT:
                joint_feedback["left_shoulder"] = "warning"
                joint_feedback["right_shoulder"] = "warning"
                issues.append("Relax your traps \u2014 don't shrug shoulders up")
                scores["shrug"] = 0.5
            else:
                joint_feedback["left_shoulder"] = "incorrect"
                joint_feedback["right_shoulder"] = "incorrect"
                issues.append("Shrugging too much \u2014 lower your shoulders away from ears")
                scores["shrug"] = 0.2

        # ── Not above 90° (weight 0.8) ──
        if abduction is not None and abduction > ABDUCTION_OVER:
            issues.append("Don't raise past shoulder height \u2014 traps take over above 90\u00b0")
            scores["above_90"] = 0.4

        if not scores:
            return 0.0, joint_feedback, issues

        WEIGHTS = {
            "elbow_bend": 2.0, "wrist_flick": 1.5, "torso": 1.2,
            "shrug": 1.0, "above_90": 0.8,
        }
        weighted_sum = sum(scores[k] * WEIGHTS.get(k, 1.0) for k in scores)
        weight_total = sum(WEIGHTS.get(k, 1.0) for k in scores)
        total = weighted_sum / weight_total

        return max(0.0, min(1.0, total)), joint_feedback, issues
