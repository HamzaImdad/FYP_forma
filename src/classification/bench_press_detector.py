"""
Dedicated bench press detector with state machine and form assessment.

Angles tracked:
  - Elbow (shoulder-elbow-wrist): primary rep counting + depth
  - Elbow symmetry: left vs right should match
  - Shoulder angle (elbow-shoulder-hip): arm flare control

State machine: TOP (arms extended) -> GOING_DOWN -> BOTTOM (bar at chest) -> GOING_UP -> TOP (1 rep)
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base_detector import (
    RobustExerciseDetector,
    RepPhase,
    REP_DIRECTION_DECREASING,
)
from ..pose_estimation.base import PoseResult
from ..utils.constants import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
    LEFT_EAR, RIGHT_EAR,
)


# ── Thresholds ──
# Research: full lockout is 170-180° elbow (was 155° — too loose, allowed
# incomplete lockouts to count). Bar-to-chest is 85-95° (was 80° — allowed
# over-deep presses to register as normal depth).
ELBOW_EXTENDED = 170    # was 155 — research lockout (T20)
ELBOW_BOTTOM = 90       # was 80 — research: bar-to-chest 85-95° (T26)
ELBOW_HALF = 115        # was 110

# Shoulder-flare zones (research 4-zone model):
#   <30° = inefficiency (excessive tuck, loses chest loading)
#   30-45° = ideal (textbook)
#   45-75° = safe zone (NSCA, Rippetoe, Zatsiorsky consensus)
#   75-85° = caution (increasing shoulder stress)
#   ≥85° = danger (impingement — Green & Comfort 2007)
SHOULDER_INEFFICIENT_MAX = 30
SHOULDER_IDEAL_MIN = 30
SHOULDER_IDEAL_MAX = 45
SHOULDER_SAFE_MAX = 75
SHOULDER_CAUTION_MAX = 85   # tightened from 95 — research danger starts at 85

# Symmetry
SYMMETRY_GOOD = 15      # degrees difference allowed
SYMMETRY_WARNING = 25

# Forearm verticality (grip-width mismatch indicator)
FOREARM_VERT_WARN = 15  # degrees deviation from vertical

# Bar path (J-curve research — McLaughlin 1984: 5-8cm horizontal over the rep)
BAR_PATH_HISTORY = 40   # frames of wrist x-position to track


class BenchPressDetector(RobustExerciseDetector):
    EXERCISE_NAME = "bench_press"
    PRIMARY_ANGLE_TOP = ELBOW_EXTENDED
    MIN_FORM_GATE = 0.3
    PRIMARY_ANGLE_BOTTOM = ELBOW_BOTTOM
    DESCENT_THRESHOLD = 10.0
    ASCENT_THRESHOLD = 10.0

    # ── Robust FSM config ───────────────────────────────────────────────
    # Visibility-only gate — bench press presents as "lying" which is a
    # PostureClassifier label that SHOULD fire (torso horizontal, legs
    # extended, hip-shoulder aligned). But LYING detection hasn't been
    # validated against real bench press sessions in the codebase. Safer
    # to use visibility-only: if both elbows, both wrists, and both
    # shoulders are visible, the user has the bar in their hands — close
    # enough to gate rep counting.
    SESSION_POSTURE_GATE = None
    VISIBILITY_GATE_LANDMARKS = (
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_ELBOW, RIGHT_ELBOW,
        LEFT_WRIST, RIGHT_WRIST,
    )
    VISIBILITY_GATE_THRESHOLD = 0.6

    REP_DIRECTION = REP_DIRECTION_DECREASING
    REP_COMPLETE_TOP = 160.0      # back at lockout zone
    REP_START_THRESHOLD = 155.0   # committed to descent
    REP_DEPTH_THRESHOLD = 100.0   # must reach bar-near-chest
    REP_BOTTOM_CLEAR = 110.0      # leaving bottom on press-up
    MIN_REAL_PRIMARY_DEG = 30.0
    MAX_PRIMARY_JUMP_DEG = 80.0

    def __init__(self, **kwargs):
        # NOTE: these must be set BEFORE super().__init__ because base class
        # calls self.reset() which accesses them.
        self._wrist_x_history: deque = deque(maxlen=BAR_PATH_HISTORY)
        self._baseline_shoulder_ear_dist: Optional[float] = None
        self._j_curve_last_reported_rep: int = 0
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self._wrist_x_history.clear()
        self._baseline_shoulder_ear_dist = None
        self._j_curve_last_reported_rep = 0

    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        elbow = self._avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
            (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
            name="elbow",
        )

        # Individual elbows for symmetry
        left_elbow = self._angle_at(pose, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
        right_elbow = self._angle_at(pose, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        # T25: if one side invisible, still report the other side's visibility penalty
        # rather than silently dropping the symmetry check.
        if left_elbow is not None and right_elbow is not None:
            symmetry = abs(left_elbow - right_elbow)
        else:
            symmetry = None  # will score with a visibility discount downstream

        shoulder = self._avg_angle(
            pose,
            (LEFT_ELBOW, LEFT_SHOULDER, LEFT_HIP),
            (RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP),
            name="shoulder",
        )

        forearm_tilt = self._compute_forearm_verticality(pose)
        shoulder_ear_delta = self._compute_shoulder_ear_delta(pose)
        self._track_bar_path(pose)

        return {
            "primary": elbow, "elbow": elbow,
            "symmetry": symmetry, "shoulder": shoulder,
            "forearm_tilt": forearm_tilt,
            "shoulder_ear_delta": shoulder_ear_delta,
        }

    def _compute_forearm_verticality(self, pose: PoseResult) -> Optional[float]:
        """Compute max(left, right) forearm deviation from vertical gravity.
        Research: >15° deviation = grip-width mismatch."""
        max_dev = None
        for e_idx, w_idx in [(LEFT_ELBOW, LEFT_WRIST), (RIGHT_ELBOW, RIGHT_WRIST)]:
            if not (pose.is_visible(e_idx) and pose.is_visible(w_idx)):
                continue
            e = np.array(pose.get_world_landmark(e_idx))
            w = np.array(pose.get_world_landmark(w_idx))
            forearm = e - w  # pointing from wrist up to elbow
            norm = np.linalg.norm(forearm)
            if norm < 0.05:
                continue
            # Angle to vertical (Y axis). MediaPipe world coords: Y increases downward,
            # so a vertical forearm going "up" has diff Y = negative.
            vertical = np.array([0.0, -1.0, 0.0])
            cos = float(np.dot(forearm, vertical) / norm)
            dev = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
            if max_dev is None or dev > max_dev:
                max_dev = dev
        return max_dev

    def _compute_shoulder_ear_delta(self, pose: PoseResult) -> Optional[float]:
        """Proxy for scapular protraction: change in shoulder-to-ear distance vs baseline.
        Increasing distance = shoulder rolling anteriorly / losing retraction.
        Returns (current - baseline) in meters; None if baseline not yet set."""
        dists = []
        for s_idx, ear_idx in [(LEFT_SHOULDER, LEFT_EAR), (RIGHT_SHOULDER, RIGHT_EAR)]:
            if pose.is_visible(s_idx) and pose.is_visible(ear_idx):
                s = np.array(pose.get_world_landmark(s_idx))
                ear = np.array(pose.get_world_landmark(ear_idx))
                dists.append(float(np.linalg.norm(s - ear)))
        if not dists:
            return None
        current = sum(dists) / len(dists)
        if self._baseline_shoulder_ear_dist is None:
            self._baseline_shoulder_ear_dist = current
            return 0.0
        return current - self._baseline_shoulder_ear_dist

    def _track_bar_path(self, pose: PoseResult) -> None:
        """Record wrist X position for J-curve path tracking."""
        xs = []
        for w_idx in [LEFT_WRIST, RIGHT_WRIST]:
            if pose.is_visible(w_idx):
                xs.append(pose.get_world_landmark(w_idx)[0])
        if xs:
            self._wrist_x_history.append(sum(xs) / len(xs))

    def _check_start_position(self, angles: Dict[str, Optional[float]]) -> Tuple[bool, List[str]]:
        issues = []
        elbow = angles.get("elbow")
        if elbow is None:
            return False, ["Lie on bench — arms must be visible above"]
        if elbow < ELBOW_EXTENDED - 25:
            issues.append("Extend arms fully — press bar to lockout")
        return (len(issues) == 0, issues)

    def _get_missing_parts(self, angles: Dict[str, Optional[float]]) -> List[str]:
        if angles.get("elbow") is None:
            return ["arms"]
        return ["body"]

    def _assess_form(self, angles: Dict[str, Optional[float]]) -> Tuple[float, Dict[str, str], List[str]]:
        joint_feedback = {}
        issues = []
        scores = {}

        elbow = angles.get("elbow")
        symmetry = angles.get("symmetry")
        shoulder = angles.get("shoulder")
        forearm_tilt = angles.get("forearm_tilt")
        shoulder_ear_delta = angles.get("shoulder_ear_delta")

        in_transit = self._state in (RepPhase.BOTTOM, RepPhase.GOING_UP, RepPhase.GOING_DOWN)

        # Pre-compute suppression: when arms are very uneven, avg depth is misleading
        _symmetry_bad = symmetry is not None and symmetry > SYMMETRY_WARNING

        # ── Elbow depth ──
        if elbow is not None:
            if in_transit:
                if elbow <= ELBOW_BOTTOM + 5:          # <= 95 — full depth
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 1.0
                elif elbow <= ELBOW_HALF:              # 95-115 — acceptable partial
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 0.8
                elif elbow <= 140:                     # 115-140 — too shallow
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    if not _symmetry_bad:
                        issues.append("Bring the bar lower — aim to touch your chest")
                    scores["elbow"] = 0.5
                else:
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    if not _symmetry_bad:
                        issues.append("Go deeper — lower the bar closer to your chest")
                    scores["elbow"] = 0.3
            else:
                # Top — lockout check
                if elbow >= ELBOW_EXTENDED - 5:        # >= 165 — locked
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 1.0
                else:
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    issues.append("Extend arms fully at the top")
                    scores["elbow"] = 0.6

        # ── Symmetry (T25 — handle one-side invisible gracefully) ──
        if symmetry is not None:
            if symmetry <= SYMMETRY_GOOD:
                scores["symmetry"] = 1.0
            elif symmetry <= SYMMETRY_WARNING:
                issues.append("Uneven press — one arm ahead of the other")
                scores["symmetry"] = 0.5
            else:
                issues.append("Arms quite uneven — focus on pressing both sides evenly")
                scores["symmetry"] = 0.2
        else:
            # One arm not visible — apply a gentle confidence discount rather
            # than silently skipping. The user's camera angle may be suboptimal.
            scores["symmetry"] = 0.85

        # ── Shoulder flare (4-zone research model, weighted 1.4x) ──
        if shoulder is not None:
            if shoulder < SHOULDER_INEFFICIENT_MAX:        # <30 — excessive tuck
                joint_feedback["left_shoulder"] = "warning"
                joint_feedback["right_shoulder"] = "warning"
                issues.append("Arms too tucked — chest not fully loaded")
                scores["shoulder"] = 0.55
            elif shoulder <= SHOULDER_IDEAL_MAX:           # 30-45 — ideal
                joint_feedback["left_shoulder"] = "correct"
                joint_feedback["right_shoulder"] = "correct"
                scores["shoulder"] = 1.0
            elif shoulder <= SHOULDER_SAFE_MAX:            # 45-75 — safe zone
                joint_feedback["left_shoulder"] = "correct"
                joint_feedback["right_shoulder"] = "correct"
                scores["shoulder"] = 0.85
            elif shoulder <= SHOULDER_CAUTION_MAX:         # 75-85 — caution
                joint_feedback["left_shoulder"] = "warning"
                joint_feedback["right_shoulder"] = "warning"
                issues.append("Elbows flaring — tuck them in a bit closer")
                scores["shoulder"] = 0.5
            else:                                          # >=85 — danger
                joint_feedback["left_shoulder"] = "incorrect"
                joint_feedback["right_shoulder"] = "incorrect"
                issues.append("Elbows too wide — tuck them in to protect your shoulders")
                scores["shoulder"] = 0.2

        # ── Forearm verticality (T22) ──
        if forearm_tilt is not None and in_transit:
            if forearm_tilt > FOREARM_VERT_WARN:
                issues.append("Forearms not vertical — grip width mismatch")
                scores["forearm"] = 0.6

        # ── Scapular protraction (T24 — proxy via shoulder-to-ear distance) ──
        if shoulder_ear_delta is not None and abs(shoulder_ear_delta) > 0.03:
            # Distance grew >3cm vs baseline — shoulder rolling anteriorly
            if shoulder_ear_delta > 0.03:
                issues.append("Keep shoulders pulled back — maintain scapular retraction")
                scores["scapular"] = 0.5

        # ── J-curve bar path (T23) — coaching cue at rep completion ──
        # Only evaluate when we have a fresh rep to report.
        if self._rep_count > self._j_curve_last_reported_rep and len(self._wrist_x_history) >= 10:
            x_range = max(self._wrist_x_history) - min(self._wrist_x_history)
            # Research (McLaughlin): 5-8 cm horizontal travel. Straight-up press
            # (<2cm) is a coaching opportunity, not an error.
            if x_range < 0.02:
                issues.append("Bar path very straight — at lockout, press bar back toward shoulders")
            self._j_curve_last_reported_rep = self._rep_count
            self._wrist_x_history.clear()

        if not scores:
            return 0.0, joint_feedback, issues

        WEIGHTS = {
            "elbow": 1.0, "symmetry": 1.0, "shoulder": 1.4,
            "forearm": 0.6, "scapular": 0.8,
        }
        weighted_sum = sum(scores[k] * WEIGHTS.get(k, 1.0) for k in scores)
        weight_total = sum(WEIGHTS.get(k, 1.0) for k in scores)
        total = weighted_sum / weight_total

        return max(0.0, min(1.0, total)), joint_feedback, issues
