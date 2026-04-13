"""
Dedicated bicep curl detector with state machine and form assessment.

Angles tracked:
  - Elbow (shoulder-elbow-wrist): primary rep counting angle
  - Upper arm stability (shoulder-elbow vertical angle)
  - Torso swing (shoulder-hip vertical angle)

State machine: TOP (extended) -> GOING_DOWN (curling) -> BOTTOM (curled) -> GOING_UP -> TOP (1 rep)
Note: "down" here means curling UP the weight (elbow angle decreases).
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
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
    LEFT_INDEX, RIGHT_INDEX,
)


# ── Thresholds ──
# Research: Oliveira et al. 2009 — full contraction at elbow 30-50° (top of curl).
# Bottom should be 165-175° (slight bend maintained — full 180° lockout stresses
# biceps tendon under heavy eccentric load). Previous ELBOW_EXTENDED=155° was too
# loose and left the state machine stranded when users didn't fully extend.
ELBOW_EXTENDED = 165    # State-machine gate — research: slight bend maintained
ELBOW_CURLED = 45       # Fully curled (research: 30-50° full contraction)
ELBOW_TOP_INCOMPLETE = 60  # >60° at bottom state = incomplete contraction (form scoring)
ELBOW_HALF = 90         # Partial curl

# Upper arm should stay vertical (pinned to side).
# Research: 0-10° forward drift = strict, 10-20° mild cheating, >20° excessive.
UPPER_ARM_GOOD = 10     # was 20 — research strict
UPPER_ARM_WARNING = 20  # was 35 — mild cheating threshold
UPPER_ARM_BAD = 30      # was 50 — >30° transitions into front raise

# Torso swing (using momentum instead of bicep).
# Research: <5° strict, 5-10° mild swing, >10-15° excessive momentum.
TORSO_GOOD = 5          # was 10 — research strict
TORSO_WARNING = 10      # was 20 — mild swing
TORSO_BAD = 15          # was 30 — excessive momentum / low back strain risk


class BicepCurlDetector(RobustExerciseDetector):
    EXERCISE_NAME = "bicep_curl"
    PRIMARY_ANGLE_TOP = ELBOW_EXTENDED
    PRIMARY_ANGLE_BOTTOM = ELBOW_CURLED
    DESCENT_THRESHOLD = 15.0
    ASCENT_THRESHOLD = 10.0
    MIN_FORM_GATE = 0.3

    # ── Robust FSM config ───────────────────────────────────────────────
    # Visibility-only gate: arm chain + torso anchor must be in frame.
    SESSION_POSTURE_GATE = None
    VISIBILITY_GATE_LANDMARKS = (
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_ELBOW, RIGHT_ELBOW,
        LEFT_WRIST, RIGHT_WRIST,
        LEFT_HIP, RIGHT_HIP,
    )
    VISIBILITY_GATE_THRESHOLD = 0.6

    REP_DIRECTION = REP_DIRECTION_DECREASING
    # Elbow: 165° extended → 45° curled → 165° extended. Tuned strict to
    # avoid phantom reps from mid-range micro-bends (common when holding
    # dumbbells at sides between sets).
    REP_COMPLETE_TOP = 150.0      # back at extended-zone
    REP_START_THRESHOLD = 145.0   # must pass this to commit to curl
    REP_DEPTH_THRESHOLD = 55.0    # curl must reach this for full contraction
    REP_BOTTOM_CLEAR = 65.0       # past this on uncurl = leaving bottom
    MIN_REAL_PRIMARY_DEG = 20.0   # biceps can reach ~30-40° naturally
    MAX_PRIMARY_JUMP_DEG = 80.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # T43: track previous torso angle for angular-velocity detection
        self._prev_torso: Optional[float] = None

    def reset(self):
        super().reset()
        self._prev_torso = None

    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        elbow = self._avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
            (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
            name="elbow",
        )

        upper_arm = self._compute_upper_arm_stability(pose)
        torso = self._compute_torso_swing(pose)
        wrist_neutral = self._compute_wrist_neutral(pose)

        # T43: frame-to-frame torso angular velocity (degrees per frame).
        # At ~30fps, >5°/frame = momentum use per research.
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
        """Wrist neutral angle (elbow→wrist→index). Research: 170-180° neutral.
        MediaPipe index landmarks are noisy; visibility-gated. Flag <160° = wrist curl."""
        min_angle = None
        for e_idx, w_idx, i_idx in [
            (LEFT_ELBOW, LEFT_WRIST, LEFT_INDEX),
            (RIGHT_ELBOW, RIGHT_WRIST, RIGHT_INDEX),
        ]:
            angle = self._angle_at(pose, e_idx, w_idx, i_idx)
            if angle is not None:
                if min_angle is None or angle < min_angle:
                    min_angle = angle
        return min_angle

    def _compute_upper_arm_stability(self, pose: PoseResult) -> Optional[float]:
        """Compute how much upper arm deviates from vertical (shoulder-elbow line)."""
        angles = []
        for s_idx, e_idx in [(LEFT_SHOULDER, LEFT_ELBOW), (RIGHT_SHOULDER, RIGHT_ELBOW)]:
            if pose.is_visible(s_idx) and pose.is_visible(e_idx):
                s = np.array(pose.get_world_landmark(s_idx))
                e = np.array(pose.get_world_landmark(e_idx))
                diff = e - s
                vertical = np.array([0, 1, 0])  # down is positive y in MediaPipe world coords
                cos = np.dot(diff, vertical) / (np.linalg.norm(diff) + 1e-8)
                angles.append(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
        return sum(angles) / len(angles) if angles else None

    def _compute_torso_swing(self, pose: PoseResult) -> Optional[float]:
        """Compute torso lean from vertical (shoulder-hip line)."""
        angles = []
        for s_idx, h_idx in [(LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP)]:
            if pose.is_visible(s_idx) and pose.is_visible(h_idx):
                s = np.array(pose.get_world_landmark(s_idx))
                h = np.array(pose.get_world_landmark(h_idx))
                diff = s - h  # shoulder above hip
                vertical = np.array([0, 1, 0])
                cos = np.dot(diff, vertical) / (np.linalg.norm(diff) + 1e-8)
                angles.append(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
        return sum(angles) / len(angles) if angles else None

    def _check_start_position(self, angles: Dict[str, Optional[float]]) -> Tuple[bool, List[str]]:
        issues = []
        elbow = angles.get("elbow")
        if elbow is None:
            return False, ["Stand facing camera — arms must be visible"]
        if elbow < ELBOW_EXTENDED - 30:
            issues.append("Extend arms fully to begin — let weights hang at sides")
        return (len(issues) == 0, issues)

    def _get_missing_parts(self, angles: Dict[str, Optional[float]]) -> List[str]:
        if angles.get("elbow") is None:
            return ["arms and elbows"]
        return ["body"]

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

        # Pre-compute suppression: velocity catches momentum better than absolute angle
        _momentum_active = torso_velocity is not None and torso_velocity > 5.0

        # ── Elbow ROM (T44 — explicit top validation at BOTTOM state) ──
        if elbow is not None:
            if self._state == RepPhase.BOTTOM:
                # Fully curled — T44 incomplete contraction check
                if elbow <= ELBOW_CURLED + 5:         # <= 50 — full contraction
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 1.0
                elif elbow <= ELBOW_TOP_INCOMPLETE:   # 50-60 — acceptable
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 0.85
                else:                                  # >60 — incomplete
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    issues.append("Curl higher — full contraction at the top")
                    scores["elbow"] = 0.5
            elif in_curl:
                # In transit (going down/up between extended and curled)
                joint_feedback["left_elbow"] = "correct"
                joint_feedback["right_elbow"] = "correct"
                scores["elbow"] = 0.9
            else:
                # At top (extended) — research: 165-175° slight bend maintained
                if elbow >= ELBOW_EXTENDED - 10:
                    joint_feedback["left_elbow"] = "correct"
                    joint_feedback["right_elbow"] = "correct"
                    scores["elbow"] = 1.0
                else:
                    joint_feedback["left_elbow"] = "warning"
                    joint_feedback["right_elbow"] = "warning"
                    issues.append("Extend arms fully at the bottom")
                    scores["elbow"] = 0.6

        # ── Upper arm stability (most important — weighted 1.5x) ──
        if upper_arm is not None:
            if upper_arm <= UPPER_ARM_GOOD:
                joint_feedback["left_shoulder"] = "correct"
                joint_feedback["right_shoulder"] = "correct"
                scores["upper_arm"] = 1.0
            elif upper_arm <= UPPER_ARM_WARNING:
                joint_feedback["left_shoulder"] = "warning"
                joint_feedback["right_shoulder"] = "warning"
                issues.append("Pin elbows to your sides — upper arm is moving")
                scores["upper_arm"] = 0.5
            else:
                joint_feedback["left_shoulder"] = "incorrect"
                joint_feedback["right_shoulder"] = "incorrect"
                issues.append("Elbows swinging — keep upper arms still")
                scores["upper_arm"] = 0.2

        # ── Torso swing (cheating with momentum) ──
        if torso is not None:
            if torso <= TORSO_GOOD:
                joint_feedback["left_hip"] = "correct"
                joint_feedback["right_hip"] = "correct"
                scores["torso"] = 1.0
            elif torso <= TORSO_WARNING:
                joint_feedback["left_hip"] = "warning"
                joint_feedback["right_hip"] = "warning"
                if not _momentum_active:
                    issues.append("Slight body swing — stay more upright")
                scores["torso"] = 0.5
            else:
                joint_feedback["left_hip"] = "incorrect"
                joint_feedback["right_hip"] = "incorrect"
                if not _momentum_active:
                    issues.append("Too much body swing — reduce weight or slow down")
                scores["torso"] = 0.2

        # ── Torso angular velocity (T43 — catches momentum even at low absolute angle) ──
        # Research: >5°/frame at ~30fps = momentum use regardless of absolute lean.
        if torso_velocity is not None and torso_velocity > 5.0:
            issues.append("Using momentum — try slowing down and controlling the weight")
            scores["torso_velocity"] = 0.4

        # ── Wrist neutral (T42) ──
        if wrist_neutral is not None and wrist_neutral < 160:
            issues.append("Keep wrists straight — avoid curling them")
            scores["wrist"] = 0.5

        if not scores:
            return 0.0, joint_feedback, issues

        WEIGHTS = {
            "elbow": 1.0, "upper_arm": 1.8, "torso": 1.2,
            "torso_velocity": 1.0, "wrist": 0.6,
        }
        weighted_sum = sum(scores[k] * WEIGHTS.get(k, 1.0) for k in scores)
        weight_total = sum(WEIGHTS.get(k, 1.0) for k in scores)
        total = weighted_sum / weight_total

        return max(0.0, min(1.0, total)), joint_feedback, issues
