"""
Dedicated overhead press detector with state machine and form assessment.

Angles tracked:
  - Elbow (shoulder-elbow-wrist): primary rep counting
  - Lockout angle (full extension at top)
  - Torso lean (should stay upright, not arch back)

State machine: BOTTOM (bar at shoulders) -> GOING_UP -> TOP (locked out) -> GOING_DOWN -> BOTTOM (1 rep)
Note: inverted from most exercises — starts at bottom, counts when returning to bottom.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base_detector import (
    RobustExerciseDetector,
    RepPhase,
    REP_DIRECTION_INCREASING,
)
from ..pose_estimation.base import PoseResult
from ..utils.constants import (
    NOSE,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
)


# ── Thresholds ──
# Research: full lockout is 175-180° elbow; active scapular shrug at top.
# Previous ELBOW_LOCKED=160° allowed 20° of un-extension to register as "locked".
# Tightened to 170° as state-machine gate, with separate ELBOW_LOCKOUT_MIN=165°
# used in form scoring to warn users who bar their lockout short.
ELBOW_LOCKED = 170      # State-machine top gate — research: 175-180°
ELBOW_LOCKOUT_MIN = 165 # Form-scoring warning threshold (< this = incomplete lockout)
ELBOW_SHOULDER = 85     # Bar at shoulder level (research: 85-95°, was 80)
ELBOW_HALF = 125        # Partial press (was 120)

# Torso lean. Research: 5-10° backward lean is necessary (to clear head for
# vertical bar path), 10-15° caution zone, >15° is lumbar hyperextension risk.
# Excessive backward lean in competitive overhead pressing was so injurious it
# caused the lift to be removed from Olympic weightlifting.
TORSO_GOOD = 10         # Research: 5-10° backward lean acceptable
TORSO_WARNING = 12      # was 20 — caution zone starts at 10-15°
TORSO_BAD = 18          # was 30 — research: >15° = error (lumbar risk)


class OverheadPressDetector(RobustExerciseDetector):
    EXERCISE_NAME = "overhead_press"
    # NOTE: overhead press uses REP_DIRECTION_INCREASING. Under the robust
    # FSM, state semantics for an increasing-direction exercise are:
    #   TOP         = primary at/near "start position" (bar at shoulders, low angle)
    #   GOING_DOWN  = primary heading toward depth (pressing up, angle rising)
    #   BOTTOM      = primary reached depth (locked out overhead, max angle)
    #   GOING_UP    = primary returning to start (lowering, angle falling)
    # This cleanly inverts from the previous base-class semantics where the
    # labels were reversed and form-scoring had to compensate with
    # at_lockout = state in {BOTTOM, GOING_DOWN} (which was a pre-existing
    # mislabel). Under the robust FSM that same check IS correct because
    # BOTTOM = lockout and GOING_DOWN = pressing up toward lockout.
    PRIMARY_ANGLE_TOP = ELBOW_LOCKED
    PRIMARY_ANGLE_BOTTOM = ELBOW_SHOULDER
    DESCENT_THRESHOLD = 10.0
    ASCENT_THRESHOLD = 10.0

    # ── Robust FSM config ───────────────────────────────────────────────
    SESSION_POSTURE_GATE = None
    VISIBILITY_GATE_LANDMARKS = (
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_ELBOW, RIGHT_ELBOW,
        LEFT_WRIST, RIGHT_WRIST,
        LEFT_HIP, RIGHT_HIP,
    )
    VISIBILITY_GATE_THRESHOLD = 0.6

    REP_DIRECTION = REP_DIRECTION_INCREASING
    # Start zone: elbow at/below shoulders (85-95°). Commit zone: lockout
    # reached 160°+ then back to below 100° for next rep.
    REP_COMPLETE_TOP = 100.0       # elbow back at/below this = at start
    REP_START_THRESHOLD = 105.0    # past this on press = committed to press
    REP_DEPTH_THRESHOLD = 160.0    # must reach this for valid lockout
    REP_BOTTOM_CLEAR = 150.0       # past this on return = leaving lockout
    MIN_REAL_PRIMARY_DEG = 20.0
    MAX_PRIMARY_JUMP_DEG = 80.0

    def __init__(self, **kwargs):
        # Must set before super().__init__ — base __init__ calls self.reset()
        self._wrist_x_history: deque = deque(maxlen=30)
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self._wrist_x_history.clear()

    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        elbow = self._avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
            (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
            name="elbow",
        )
        torso = self._compute_torso_lean(pose)
        head_through = self._compute_head_through(pose)
        bar_drift = self._compute_bar_drift(pose)

        return {
            "primary": elbow, "elbow": elbow, "torso": torso,
            "head_through": head_through, "bar_drift": bar_drift,
        }

    def _compute_head_through(self, pose: PoseResult) -> Optional[float]:
        """Push-head-through proxy: nose X relative to shoulder X midpoint.
        Positive = head has moved forward (anterior) past shoulders,
        Negative = head still behind/at shoulder plane.
        At lockout, the head should pass between the arms (small positive value)."""
        if not pose.is_visible(NOSE):
            return None
        ls = pose.get_world_landmark(LEFT_SHOULDER) if pose.is_visible(LEFT_SHOULDER) else None
        rs = pose.get_world_landmark(RIGHT_SHOULDER) if pose.is_visible(RIGHT_SHOULDER) else None
        if ls is None and rs is None:
            return None
        nose_z = pose.get_world_landmark(NOSE)[2]
        if ls is not None and rs is not None:
            shoulder_z = (ls[2] + rs[2]) / 2
        else:
            shoulder_z = (ls or rs)[2]
        # In MediaPipe world coords, -Z is in front of camera (toward lens).
        # "Head through" means nose moved further in -Z than the shoulder plane.
        return float(shoulder_z - nose_z)  # positive = nose forward of shoulders

    def _compute_bar_drift(self, pose: PoseResult) -> Optional[float]:
        """Track horizontal wrist drift during press. Returns X range of the current frame window."""
        xs = []
        for w_idx in [LEFT_WRIST, RIGHT_WRIST]:
            if pose.is_visible(w_idx):
                xs.append(pose.get_world_landmark(w_idx)[0])
        if xs:
            self._wrist_x_history.append(sum(xs) / len(xs))
        if len(self._wrist_x_history) < 5:
            return None
        return float(max(self._wrist_x_history) - min(self._wrist_x_history))

    def _compute_torso_lean(self, pose: PoseResult) -> Optional[float]:
        """Compute torso lean from vertical."""
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

    def _check_start_position(self, angles: Dict[str, Optional[float]]) -> Tuple[bool, List[str]]:
        issues = []
        elbow = angles.get("elbow")
        if elbow is None:
            return False, ["Stand facing camera — arms must be visible"]
        # Accept bar at shoulders (65-95°) OR arms locked out (150-175°)
        at_shoulders = 65 <= elbow <= 95
        locked_out = 150 <= elbow <= 175
        if at_shoulders or locked_out:
            return True, []
        if elbow < 65:
            issues.append("Raise bar to shoulder height to begin")
        elif elbow > 175:
            issues.append("Arms too far back — bring bar to shoulders")
        else:
            issues.append("Hold bar at shoulder height or lock out overhead to begin")
        return False, issues

    def _get_missing_parts(self, angles: Dict[str, Optional[float]]) -> List[str]:
        if angles.get("elbow") is None:
            return ["arms"]
        return ["body"]

    def _assess_form(self, angles: Dict[str, Optional[float]]) -> Tuple[float, Dict[str, str], List[str]]:
        joint_feedback = {}
        issues = []
        scores = {}

        elbow = angles.get("elbow")
        torso = angles.get("torso")
        head_through = angles.get("head_through")
        bar_drift = angles.get("bar_drift")

        # Under RobustExerciseDetector with REP_DIRECTION_INCREASING, state
        # semantics for overhead press are:
        #   TOP         = bar at shoulders (primary angle near REP_COMPLETE_TOP=100°)
        #   GOING_DOWN  = pressing up toward lockout (primary rising, 100°→170°)
        #   BOTTOM      = fully locked out overhead (primary ≥ REP_DEPTH_THRESHOLD=160°)
        #   GOING_UP    = lowering back to shoulders (primary falling, 170°→100°)
        #
        # Form check: scoring should rely on the ACTUAL elbow value rather
        # than the state label, because the press takes many frames during
        # which the state is GOING_DOWN but elbow hasn't reached lockout
        # yet. The state-only check (old code) would fire "lock out fully"
        # warnings during the press-up phase before the user had a chance
        # to reach lockout.
        at_lockout = elbow is not None and elbow >= ELBOW_LOCKOUT_MIN
        at_shoulders = elbow is not None and elbow <= ELBOW_SHOULDER + 10

        # ── Elbow / lockout ──
        if elbow is not None:
            if at_lockout:
                if elbow >= ELBOW_LOCKED:                  # >= 170 — full lockout
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 1.0
                elif elbow >= ELBOW_LOCKOUT_MIN:           # 165-170 — near lockout, soft cue
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    issues.append("Lock out fully — straighten elbows and shrug up")
                    scores["elbow"] = 0.75
                elif elbow >= ELBOW_HALF:                  # 125-165 — incomplete
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    issues.append("Lock out arms fully overhead")
                    scores["elbow"] = 0.5
                else:
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    scores["elbow"] = 0.3
            else:
                # At shoulders — bar should be in start position
                if elbow <= ELBOW_SHOULDER + 10:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 1.0
                else:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 0.8

        # ── Torso lean (weighted 1.5x — lumbar hyperextension is dangerous) ──
        if torso is not None:
            if torso <= TORSO_GOOD:
                joint_feedback["left_hip"] = "correct"
                joint_feedback["right_hip"] = "correct"
                scores["torso"] = 1.0
            elif torso <= TORSO_WARNING:
                joint_feedback["left_hip"] = "warning"
                joint_feedback["right_hip"] = "warning"
                issues.append("Brace your core — keep your back upright")
                scores["torso"] = 0.5
            else:
                joint_feedback["left_hip"] = "incorrect"
                joint_feedback["right_hip"] = "incorrect"
                issues.append("Too much back arch — brace core tighter, consider less weight")
                scores["torso"] = 0.2

        # ── Push head through at lockout (T29) ──
        # At lockout, nose should have moved forward (anterior) so the bar sits
        # directly over mid-foot/shoulders. Negative head_through = head still
        # behind shoulders = "chicken neck" at lockout.
        if head_through is not None and at_lockout and scores.get("torso", 1.0) > 0.5:
            if head_through < -0.03:  # nose still >3cm behind shoulder plane
                issues.append("Push head through at the top — head between the arms")
                scores["head_through"] = 0.6

        # ── Bar path drift (T30) ──
        # Press should travel nearly vertically. If wrist X varies >5cm across
        # the tracking window, the bar is drifting forward/backward.
        if bar_drift is not None and bar_drift > 0.05:
            issues.append("Keep bar path vertical — reduce forward/backward drift")
            scores["bar_drift"] = 0.6

        if not scores:
            return 0.0, joint_feedback, issues

        WEIGHTS = {
            "elbow": 1.0, "torso": 1.5,
            "head_through": 0.5, "bar_drift": 0.6,
        }
        weighted_sum = sum(scores[k] * WEIGHTS.get(k, 1.0) for k in scores)
        weight_total = sum(WEIGHTS.get(k, 1.0) for k in scores)
        total = weighted_sum / weight_total

        return max(0.0, min(1.0, total)), joint_feedback, issues
