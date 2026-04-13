"""
Dedicated lunge detector with state machine and form assessment.

Angles tracked:
  - Front knee (hip-knee-ankle): primary rep counting angle
  - Back knee: should approach floor
  - Torso: should stay upright
  - Front shin: should be roughly vertical

State machine: TOP (standing) -> GOING_DOWN -> BOTTOM (lunged) -> GOING_UP -> TOP (1 rep)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .base_detector import (
    RobustExerciseDetector,
    RepPhase,
    REP_DIRECTION_DECREASING,
)
from ..pose_estimation.base import PoseResult
from ..utils.constants import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
)


# ── Thresholds ──
# Kotiuk et al. 2023 measured 98° front-knee flexion at bodyweight lunge bottom.
# Research: front knee 80-100° at bottom, rear knee 90-110° at bottom.
KNEE_STANDING = 175     # Research: 170-180° standing lockout (was 160)
KNEE_LUNGED = 90        # Front knee at 90° — mid-range of research
KNEE_HALF = 115         # Upper bound of acceptable partial

# Rear knee should hover just above the ground (~90-110° bent).
BACK_KNEE_GOOD = 100    # Back knee approaching floor
BACK_KNEE_WARNING = 115 # was 130 — research says rear knee should reach 90-110°

# Torso (research: standard lunge 0-20° forward lean, hip-focused 20-40°)
TORSO_GOOD = 15
TORSO_WARNING = 30
TORSO_BAD = 45

# Lateral pelvic tilt (Trendelenburg / gluteus medius weakness indicator).
# Research: 0-5° acceptable, >8° warning, >10° error.
PELVIC_TILT_WARN = 8
PELVIC_TILT_BAD = 10


class LungeDetector(RobustExerciseDetector):
    EXERCISE_NAME = "lunge"
    PRIMARY_ANGLE_TOP = KNEE_STANDING
    PRIMARY_ANGLE_BOTTOM = KNEE_LUNGED
    MIN_FORM_GATE = 0.3
    DESCENT_THRESHOLD = 15.0
    ASCENT_THRESHOLD = 10.0

    FRONT_LEG_LOCK_FRAMES = 22  # was 15 — ~0.75s at 30fps, prevents mid-rep flipping

    # ── Robust FSM config ───────────────────────────────────────────────
    # Visibility-only gate: legs must be visible. Front-leg hysteresis
    # (below) runs INSIDE _compute_angles — primary angle is already the
    # dynamically-locked front knee by the time the rep FSM sees it, so
    # the robust FSM treats lunge like any other decreasing-angle
    # exercise without conflicting with the lock.
    SESSION_POSTURE_GATE = None
    VISIBILITY_GATE_LANDMARKS = (
        LEFT_HIP, RIGHT_HIP,
        LEFT_KNEE, RIGHT_KNEE,
        LEFT_ANKLE, RIGHT_ANKLE,
    )
    VISIBILITY_GATE_THRESHOLD = 0.6

    REP_DIRECTION = REP_DIRECTION_DECREASING
    # Front knee: 175° standing → 90° lunged. Tuned strict to reject
    # mid-range hovering (common while setting up the stance).
    REP_COMPLETE_TOP = 160.0
    REP_START_THRESHOLD = 155.0
    REP_DEPTH_THRESHOLD = 110.0
    REP_BOTTOM_CLEAR = 120.0
    MIN_REAL_PRIMARY_DEG = 30.0
    MAX_PRIMARY_JUMP_DEG = 60.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._front_leg = None  # "left" or "right"
        self._front_leg_lock = 0  # frames remaining in current lock

    def reset(self):
        super().reset()
        self._front_leg = None
        self._front_leg_lock = 0

    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        left_knee = self._angle_at(pose, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
        right_knee = self._angle_at(pose, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)

        # Determine which leg is in front (lower knee angle = front leg)
        # Use hysteresis: once decided, lock for FRONT_LEG_LOCK_FRAMES frames
        if left_knee is not None and right_knee is not None:
            if self._front_leg_lock > 0:
                self._front_leg_lock -= 1
            else:
                # Re-evaluate: only switch if difference is significant (>15°)
                diff = left_knee - right_knee
                if diff < -15:
                    self._front_leg = "left"
                    self._front_leg_lock = self.FRONT_LEG_LOCK_FRAMES
                elif diff > 15:
                    self._front_leg = "right"
                    self._front_leg_lock = self.FRONT_LEG_LOCK_FRAMES
                elif self._front_leg is None:
                    # First detection — use smaller threshold
                    self._front_leg = "left" if left_knee < right_knee else "right"
                    self._front_leg_lock = self.FRONT_LEG_LOCK_FRAMES

            if self._front_leg == "left":
                front_knee = left_knee
                back_knee = right_knee
            else:
                front_knee = right_knee
                back_knee = left_knee
        elif left_knee is not None:
            self._front_leg = "left"
            front_knee = left_knee
            back_knee = None
        elif right_knee is not None:
            self._front_leg = "right"
            front_knee = right_knee
            back_knee = None
        else:
            front_knee = None
            back_knee = None

        torso = self._compute_torso_lean(pose)
        pelvic_tilt = self._compute_pelvic_tilt(pose)

        return {
            "primary": front_knee,
            "front_knee": front_knee,
            "back_knee": back_knee,
            "torso": torso,
            "pelvic_tilt": pelvic_tilt,
        }

    def _compute_pelvic_tilt(self, pose: PoseResult) -> Optional[float]:
        """Lateral pelvic tilt — angle between hip line and horizontal.
        Research: 0-5° acceptable, >8° warning (gluteus medius weakness),
        >10° error (Trendelenburg pattern). Uses world coordinates."""
        if not (pose.is_visible(LEFT_HIP) and pose.is_visible(RIGHT_HIP)):
            return None
        lh = np.array(pose.get_world_landmark(LEFT_HIP))
        rh = np.array(pose.get_world_landmark(RIGHT_HIP))
        dy = abs(lh[1] - rh[1])
        horizontal_dist = float(np.sqrt((lh[0] - rh[0]) ** 2 + (lh[2] - rh[2]) ** 2))
        if horizontal_dist < 0.05:  # hips too close together to estimate — bad pose
            return None
        return float(np.degrees(np.arctan2(dy, horizontal_dist)))

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
        fk = angles.get("front_knee")
        if fk is None:
            return False, ["Stand facing camera — legs must be visible"]
        if fk < KNEE_STANDING - 20:
            issues.append("Stand up straight to begin")
        return (len(issues) == 0, issues)

    def _get_missing_parts(self, angles: Dict[str, Optional[float]]) -> List[str]:
        if angles.get("front_knee") is None:
            return ["legs"]
        return ["body"]

    def _assess_form(self, angles: Dict[str, Optional[float]]) -> Tuple[float, Dict[str, str], List[str]]:
        joint_feedback = {}
        issues = []
        scores = {}

        front_knee = angles.get("front_knee")
        back_knee = angles.get("back_knee")
        torso = angles.get("torso")

        fl = self._front_leg or "left"
        bl = "right" if fl == "left" else "left"

        # Pre-compute suppression flags
        pelvic_tilt = angles.get("pelvic_tilt")
        _pelvis_unstable = pelvic_tilt is not None and pelvic_tilt > PELVIC_TILT_BAD

        # ── Front knee depth ──
        if front_knee is not None:
            if self._state in (RepPhase.BOTTOM, RepPhase.GOING_UP, RepPhase.GOING_DOWN):
                if front_knee <= KNEE_LUNGED + 10:
                    joint_feedback[f"{fl}_knee"] = "correct"
                    scores["front_knee"] = 1.0
                elif front_knee <= KNEE_HALF:
                    joint_feedback[f"{fl}_knee"] = "warning"
                    if not _pelvis_unstable:
                        issues.append("Go deeper — front thigh should be parallel to floor")
                    scores["front_knee"] = 0.5
                else:
                    joint_feedback[f"{fl}_knee"] = "correct"
                    scores["front_knee"] = 0.8
            else:
                joint_feedback[f"{fl}_knee"] = "correct"
                scores["front_knee"] = 1.0

        # ── Back knee depth ──
        if back_knee is not None and self._state in (RepPhase.BOTTOM, RepPhase.GOING_UP, RepPhase.GOING_DOWN):
            if back_knee <= BACK_KNEE_GOOD:
                joint_feedback[f"{bl}_knee"] = "correct"
                scores["back_knee"] = 1.0
            elif back_knee <= BACK_KNEE_WARNING:
                joint_feedback[f"{bl}_knee"] = "warning"
                issues.append("Drop back knee closer to floor")
                scores["back_knee"] = 0.5
            else:
                joint_feedback[f"{bl}_knee"] = "warning"
                scores["back_knee"] = 0.7

        # ── Torso upright (weighted 1.3x) ──
        if torso is not None:
            if torso <= TORSO_GOOD:
                joint_feedback["left_shoulder"] = "correct"
                joint_feedback["right_shoulder"] = "correct"
                scores["torso"] = 1.0
            elif torso <= TORSO_WARNING:
                joint_feedback["left_shoulder"] = "warning"
                joint_feedback["right_shoulder"] = "warning"
                issues.append("Keep torso upright — stay tall through your spine")
                scores["torso"] = 0.5
            else:
                joint_feedback["left_shoulder"] = "incorrect"
                joint_feedback["right_shoulder"] = "incorrect"
                issues.append("Too much forward lean — lift your chest")
                scores["torso"] = 0.2

        # ── Lateral pelvic tilt (T15 — Trendelenburg / glute med weakness) ──
        pelvic_tilt = angles.get("pelvic_tilt")
        if pelvic_tilt is not None:
            if pelvic_tilt > PELVIC_TILT_BAD:
                issues.append("Hips tilting to one side — engage glutes and core")
                scores["pelvic_tilt"] = 0.3
                joint_feedback["left_hip"] = "incorrect"
                joint_feedback["right_hip"] = "incorrect"
            elif pelvic_tilt > PELVIC_TILT_WARN:
                issues.append("Slight hip drop — keep pelvis level")
                scores["pelvic_tilt"] = 0.6

        if not scores:
            return 0.0, joint_feedback, issues

        WEIGHTS = {"front_knee": 1.0, "back_knee": 1.0, "torso": 1.3, "pelvic_tilt": 1.2}
        weighted_sum = sum(scores[k] * WEIGHTS.get(k, 1.0) for k in scores)
        weight_total = sum(WEIGHTS.get(k, 1.0) for k in scores)
        total = weighted_sum / weight_total

        return max(0.0, min(1.0, total)), joint_feedback, issues
