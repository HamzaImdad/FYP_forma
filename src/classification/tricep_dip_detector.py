"""
Dedicated tricep dip detector — bench dip variant (hands on bench behind,
legs extended forward in L-shape).

Angles tracked:
  - Elbow (shoulder-elbow-wrist): primary rep counting + depth
  - Shoulder extension (elbow-shoulder-hip): injury-critical safety metric
  - Torso lean: should stay nearly vertical for bench dip
  - Hip angle (shoulder-hip-knee): L-shape maintenance
  - Knee (hip-knee-ankle): legs should be extended
  - Shoulder-below-elbow: secondary impingement proxy

Camera: side view.
State machine: TOP (arms extended) -> GOING_DOWN -> BOTTOM (dipped) -> GOING_UP -> TOP (1 rep)
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
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
)
from ..utils.geometry import calculate_angle


# ── Thresholds ──
ELBOW_EXTENDED = 155
ELBOW_DIPPED = 90       # Target depth
ELBOW_HALF = 110        # Partial dip
ELBOW_TOO_DEEP = 80     # <80° = excessive depth (research: forces impingement risk)

# Shoulder extension ROM — THE CRITICAL SAFETY METRIC for dips.
# McKenzie et al. (2022, 3D motion capture): bench dips reach 101.35% of max
# passive shoulder extension ROM (~89-90°), bar dips 88.03% (~78°), ring dips
# 68.88% (~62°). Brookbush Institute identifies 60° as the safety threshold
# above which anterior capsule strain and subacromial compression rise sharply.
SHOULDER_EXT_SAFE = 50      # <50° = safe zone
SHOULDER_EXT_MODERATE = 65  # 50-65° = moderate risk
# >65° = HIGH impingement risk

# Torso lean for BENCH DIP (nearly vertical, slight backward lean):
TORSO_LEAN_GOOD = 10       # <=10° = good (nearly vertical)
TORSO_LEAN_OK = 20         # 10-20° = acceptable
# >20° = excessive lean

# Hip angle (shoulder-hip-knee) for bench dip L-shape
HIP_ANGLE_GOOD_MIN = 80    # bottom position
HIP_ANGLE_GOOD_MAX = 110   # top position
HIP_TOO_OPEN = 130         # legs too straight relative to torso

# Knee extension (hip-knee-ankle) — legs should be extended
KNEE_EXTENDED = 170
KNEE_BENT_WARN = 150

# Side-view visibility threshold
DIP_VIS_THRESHOLD = 0.3

# Movement-gated session FSM
DIP_ELBOW_AT_TOP = 145.0
DIP_ELBOW_DESCEND = 130.0
DIP_MOTION_WINDOW_S = 2.0
DIP_MOTION_MIN_RANGE = 20.0
DIP_REST_AT_TOP_S = 8.0
DIP_REST_BELOW_TOP_S = 60.0


class TricepDipDetector(RobustExerciseDetector):
    EXERCISE_NAME = "tricep_dip"
    PRIMARY_ANGLE_TOP = ELBOW_EXTENDED
    PRIMARY_ANGLE_BOTTOM = ELBOW_DIPPED
    DESCENT_THRESHOLD = 10.0
    ASCENT_THRESHOLD = 10.0
    MIN_FORM_GATE = 0.3

    # ── Robust FSM config (bench dip, side view) ───────────────────────
    # Visibility-only gate with side-view override. Bench dip has hands
    # on bench behind, legs forward — body in L-shape from side view.
    SESSION_POSTURE_GATE = None
    VISIBILITY_GATE_LANDMARKS = (
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_ELBOW, RIGHT_ELBOW,
        LEFT_WRIST, RIGHT_WRIST,
        LEFT_HIP, RIGHT_HIP,
    )
    VISIBILITY_GATE_THRESHOLD = DIP_VIS_THRESHOLD

    REP_DIRECTION = REP_DIRECTION_DECREASING
    REP_COMPLETE_TOP = 145.0      # back at extended zone
    REP_START_THRESHOLD = 140.0   # committed to dip
    REP_DEPTH_THRESHOLD = 100.0   # must reach ~90° depth (with margin)
    REP_BOTTOM_CLEAR = 110.0
    MIN_REAL_PRIMARY_DEG = 30.0
    MAX_PRIMARY_JUMP_DEG = 80.0

    def __init__(self, **kwargs):
        # Instance state BEFORE super().__init__() — reset() touches these
        self._at_top_since: Optional[float] = None
        self._below_top_since: Optional[float] = None
        self._elbow_history: Deque[Tuple[float, float]] = deque()
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self._at_top_since = None
        self._below_top_since = None
        self._elbow_history.clear()

    # ── Side-view helpers ──────────────────────────────────────────────

    def _lenient_side_angle(
        self,
        pose: PoseResult,
        a_idx: int,
        b_idx: int,
        c_idx: int,
        threshold: float = DIP_VIS_THRESHOLD,
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

    def _check_visibility_gate(self, pose) -> bool:
        """Accept if EITHER side's shoulder-elbow-wrist chain is visible."""
        left_ok = all(
            pose.is_visible(idx, DIP_VIS_THRESHOLD)
            for idx in (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
        )
        right_ok = all(
            pose.is_visible(idx, DIP_VIS_THRESHOLD)
            for idx in (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        )
        return left_ok or right_ok

    # ── Movement-gated session FSM ─────────────────────────────────────

    def _update_session_state(self, now: float, visibility_ok: bool) -> None:
        """Elbow-based motion detection — same pattern as bicep_curl."""
        elbow = self._get_smooth("elbow")
        if elbow is None:
            super()._update_session_state(now, visibility_ok)
            return

        is_at_top = elbow >= DIP_ELBOW_AT_TOP
        is_dipping = elbow < DIP_ELBOW_DESCEND

        # Update top/below-top timers
        if is_at_top:
            if self._at_top_since is None:
                self._at_top_since = now
            self._below_top_since = None
        else:
            if self._below_top_since is None:
                self._below_top_since = now
            self._at_top_since = None

        # Rolling elbow history
        self._elbow_history.append((now, elbow))
        window_start = now - DIP_MOTION_WINDOW_S
        while self._elbow_history and self._elbow_history[0][0] < window_start:
            self._elbow_history.popleft()

        if len(self._elbow_history) >= 2:
            values = [v for _, v in self._elbow_history]
            motion_range = max(values) - min(values)
        else:
            motion_range = 0.0
        has_active_motion = motion_range >= DIP_MOTION_MIN_RANGE

        s = self._session_state

        # Visibility loss handling
        if not visibility_ok:
            if s == SessionState.ACTIVE:
                self._unknown_streak += 1
                if self._unknown_streak >= self.UNKNOWN_GRACE_FRAMES:
                    self._close_active_set_or_rollback(now)
                    self._elbow_history.clear()
            return
        self._unknown_streak = 0

        # IDLE → SETUP: body visible
        if s == SessionState.IDLE:
            self._session_state = SessionState.SETUP
            return

        # SETUP → ACTIVE: dipping motion detected
        if s == SessionState.SETUP:
            if is_dipping and has_active_motion:
                self._session_state = SessionState.ACTIVE
                self._reset_robust_rep_fsm()
                self._rep_start_time = now
                self._seed_rep_fsm_from_pre_active(now)
            return

        # RESTING → ACTIVE: same motion gate, no trace seeding
        if s == SessionState.RESTING:
            if is_dipping and has_active_motion:
                self._session_state = SessionState.ACTIVE
                self._reset_robust_rep_fsm()
                self._rep_start_time = now
            return

        # ACTIVE → RESTING: idle timeout
        if s == SessionState.ACTIVE:
            if (self._at_top_since is not None
                    and now - self._at_top_since >= DIP_REST_AT_TOP_S):
                self._close_active_set_or_rollback(now)
                self._elbow_history.clear()
                return
            if (self._below_top_since is not None
                    and now - self._below_top_since >= DIP_REST_BELOW_TOP_S):
                self._close_active_set_or_rollback(now)
                self._elbow_history.clear()
                return

    # ── Angle computation ─────────────────────────────────────────────

    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        elbow = self._lenient_avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
            (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
            name="elbow",
        )
        # Shoulder extension ROM (angle between upper arm and torso)
        shoulder_extension = self._lenient_avg_angle(
            pose,
            (LEFT_ELBOW, LEFT_SHOULDER, LEFT_HIP),
            (RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP),
            name="shoulder_extension",
        )
        torso_lean = self._compute_torso_lean(pose)
        shoulder_below_elbow = self._compute_shoulder_below_elbow(pose)

        # Hip angle (shoulder-hip-knee) — L-shape check
        hip_angle = self._lenient_avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
            (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
            name="hip",
        )

        # Knee extension (hip-knee-ankle) — legs should be extended
        knee = self._lenient_avg_angle(
            pose,
            (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
            (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
            name="knee",
        )

        return {
            "primary": elbow, "elbow": elbow,
            "shoulder_extension": shoulder_extension,
            "torso_lean": torso_lean,
            "shoulder_below_elbow": shoulder_below_elbow,
            "hip_angle": hip_angle,
            "knee": knee,
        }

    def _compute_torso_lean(self, pose: PoseResult) -> Optional[float]:
        """Torso lean: angle of shoulder-hip line from pure vertical.
        Bench dip torso should stay nearly vertical."""
        angles = []
        for s_idx, h_idx in [(LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP)]:
            if pose.is_visible(s_idx, DIP_VIS_THRESHOLD) and pose.is_visible(h_idx, DIP_VIS_THRESHOLD):
                s = np.array(pose.get_world_landmark(s_idx))
                h = np.array(pose.get_world_landmark(h_idx))
                diff = h - s  # points from shoulder down to hip
                norm = float(np.linalg.norm(diff))
                if norm < 0.1:
                    continue
                vertical = np.array([0.0, 1.0, 0.0])  # MediaPipe: +Y = down
                cos = float(np.dot(diff, vertical) / norm)
                angles.append(float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))))
        return sum(angles) / len(angles) if angles else None

    def _compute_shoulder_below_elbow(self, pose: PoseResult) -> Optional[float]:
        """Secondary impingement proxy. Returns (shoulder_y - elbow_y) in world Y.
        MediaPipe: +Y = down. Positive value = shoulder DROPPED below elbow."""
        drops = []
        for s_idx, e_idx in [(LEFT_SHOULDER, LEFT_ELBOW), (RIGHT_SHOULDER, RIGHT_ELBOW)]:
            if pose.is_visible(s_idx, DIP_VIS_THRESHOLD) and pose.is_visible(e_idx, DIP_VIS_THRESHOLD):
                s_y = pose.get_world_landmark(s_idx)[1]
                e_y = pose.get_world_landmark(e_idx)[1]
                drops.append(float(s_y - e_y))
        return max(drops) if drops else None

    # ── Start position + missing parts ────────────────────────────────

    def _check_start_position(self, angles: Dict[str, Optional[float]]) -> Tuple[bool, List[str]]:
        issues = []
        elbow = angles.get("elbow")
        hip_angle = angles.get("hip_angle")
        if elbow is None:
            return False, ["Position camera to your side — arms must be visible"]
        if elbow < ELBOW_EXTENDED - 20:
            issues.append("Extend arms fully to start — press up on the bench")
        if hip_angle is not None and hip_angle > HIP_TOO_OPEN:
            issues.append("Sit at edge of bench — legs extended forward in L-shape")
        return (len(issues) == 0, issues)

    def _get_missing_parts(self, angles: Dict[str, Optional[float]]) -> List[str]:
        if angles.get("elbow") is None:
            return ["arms and elbows"]
        return ["body"]

    # ── Form assessment ───────────────────────────────────────────────

    def _assess_form(self, angles: Dict[str, Optional[float]]) -> Tuple[float, Dict[str, str], List[str]]:
        joint_feedback = {}
        issues = []
        scores = {}

        elbow = angles.get("elbow")
        shoulder_ext = angles.get("shoulder_extension")
        torso_lean = angles.get("torso_lean")
        shoulder_below_elbow = angles.get("shoulder_below_elbow")
        hip_angle = angles.get("hip_angle")
        knee = angles.get("knee")

        in_transit = self._state in (RepPhase.BOTTOM, RepPhase.GOING_UP, RepPhase.GOING_DOWN)

        # Pre-compute suppression: shoulder safety overrides elbow depth messages
        _shoulder_danger = shoulder_ext is not None and shoulder_ext >= SHOULDER_EXT_MODERATE

        # ── Elbow depth ──
        if elbow is not None:
            if in_transit:
                if elbow < ELBOW_TOO_DEEP:             # <80 — excessive depth
                    joint_feedback["left_elbow"] = "incorrect"
                    joint_feedback["right_elbow"] = "incorrect"
                    if not _shoulder_danger:
                        issues.append("Going too deep — come up a bit to protect your shoulders")
                    scores["elbow"] = 0.3
                elif elbow <= ELBOW_DIPPED + 5:        # 80-95 — good depth
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 1.0
                elif elbow <= ELBOW_HALF:              # 95-110 — partial
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    if not _shoulder_danger:
                        issues.append("Go a bit deeper — bend elbows a bit more")
                    scores["elbow"] = 0.6
                else:
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    scores["elbow"] = 0.4
            else:
                if elbow >= ELBOW_EXTENDED - 10:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 1.0
                else:
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    issues.append("Extend arms fully at top")
                    scores["elbow"] = 0.6

        # ── Shoulder extension ROM (Tier-1 injury-critical, weighted 1.6x) ──
        # McKenzie 2022: >65° shoulder extension = high impingement risk.
        if shoulder_ext is not None and in_transit:
            if shoulder_ext < SHOULDER_EXT_SAFE:       # <50 — safe
                joint_feedback["left_shoulder"] = "correct"
                joint_feedback["right_shoulder"] = "correct"
                scores["shoulder_ext"] = 1.0
            elif shoulder_ext < SHOULDER_EXT_MODERATE: # 50-65 — moderate risk
                joint_feedback["left_shoulder"] = "warning"
                joint_feedback["right_shoulder"] = "warning"
                issues.append("Shoulders stretching far — try reducing your depth a little")
                scores["shoulder_ext"] = 0.5
            else:                                       # >=65 — high risk
                joint_feedback["left_shoulder"] = "incorrect"
                joint_feedback["right_shoulder"] = "incorrect"
                issues.append("Shoulders under strain — come up and reduce your depth")
                scores["shoulder_ext"] = 0.1

        # ── Torso lean (bench dip: nearly vertical) ──
        if torso_lean is not None:
            if torso_lean <= TORSO_LEAN_GOOD:          # <=10° — good
                scores["torso_lean"] = 1.0
            elif torso_lean <= TORSO_LEAN_OK:          # 10-20° — acceptable
                scores["torso_lean"] = 0.8
            else:                                       # >20° — excessive
                issues.append("Keep torso more upright — don't lean forward")
                scores["torso_lean"] = 0.4

        # ── Shoulder-below-elbow secondary proxy ──
        if shoulder_below_elbow is not None and shoulder_below_elbow > 0.02:
            issues.append("Shoulders sinking below elbows — control depth")
            scores["shoulder_sink"] = 0.4

        # ── Hip angle — L-shape maintenance ──
        if hip_angle is not None:
            if HIP_ANGLE_GOOD_MIN <= hip_angle <= HIP_ANGLE_GOOD_MAX:
                joint_feedback["left_hip"] = "correct"
                joint_feedback["right_hip"] = "correct"
                scores["hip_angle"] = 1.0
            elif hip_angle > HIP_TOO_OPEN:
                joint_feedback["left_hip"] = "warning"
                joint_feedback["right_hip"] = "warning"
                issues.append("Bend at hips more — maintain L-shape position")
                scores["hip_angle"] = 0.5
            elif hip_angle < HIP_ANGLE_GOOD_MIN - 15:  # <65° — too closed
                joint_feedback["left_hip"] = "warning"
                joint_feedback["right_hip"] = "warning"
                issues.append("Hips too low — keep body in L-shape")
                scores["hip_angle"] = 0.5
            else:
                joint_feedback["left_hip"] = "correct"
                joint_feedback["right_hip"] = "correct"
                scores["hip_angle"] = 0.8

        # ── Knee extension — legs should be straight ──
        if knee is not None:
            if knee >= KNEE_EXTENDED:
                joint_feedback["left_knee"] = "correct"
                joint_feedback["right_knee"] = "correct"
                scores["knee"] = 1.0
            elif knee >= KNEE_BENT_WARN:
                joint_feedback["left_knee"] = "warning"
                joint_feedback["right_knee"] = "warning"
                issues.append("Straighten legs — keep knees extended")
                scores["knee"] = 0.6
            else:
                joint_feedback["left_knee"] = "incorrect"
                joint_feedback["right_knee"] = "incorrect"
                issues.append("Legs too bent — extend them for proper bench dip form")
                scores["knee"] = 0.3

        if not scores:
            return 0.0, joint_feedback, issues

        WEIGHTS = {
            "elbow": 1.0,
            "shoulder_ext": 1.6,     # Tier-1 injury-critical (McKenzie 2022)
            "torso_lean": 0.8,
            "shoulder_sink": 1.1,
            "hip_angle": 0.9,
            "knee": 0.7,
        }
        weighted_sum = sum(scores[k] * WEIGHTS.get(k, 1.0) for k in scores)
        weight_total = sum(WEIGHTS.get(k, 1.0) for k in scores)
        total = weighted_sum / weight_total

        return max(0.0, min(1.0, total)), joint_feedback, issues
