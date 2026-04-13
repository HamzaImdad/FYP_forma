"""
Dedicated tricep dip detector with state machine and form assessment.

Angles tracked:
  - Elbow (shoulder-elbow-wrist): primary rep counting + depth
  - Shoulder (elbow-shoulder-hip): forward lean / shoulder depression
  - Hip position: should stay close to bench

State machine: TOP (arms extended) -> GOING_DOWN -> BOTTOM (dipped) -> GOING_UP -> TOP (1 rep)
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
)


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

# Torso forward lean (T48 — research tiers):
#   0-15° = triceps emphasis
#   15-30° = balanced loading
#   30-45° = chest emphasis (not wrong, just different target)
#   >45° = excessive lean
TORSO_LEAN_TRICEPS = 15
TORSO_LEAN_BALANCED = 30
TORSO_LEAN_CHEST = 45

# Frontal-plane elbow flare (T49 — research):
#   0-15° = triceps-focused
#   20-45° = natural
#   >50° = excessive
FLARE_TUCK = 15
FLARE_NATURAL = 45
FLARE_EXCESSIVE = 50


class TricepDipDetector(RobustExerciseDetector):
    EXERCISE_NAME = "tricep_dip"
    PRIMARY_ANGLE_TOP = ELBOW_EXTENDED
    PRIMARY_ANGLE_BOTTOM = ELBOW_DIPPED
    DESCENT_THRESHOLD = 10.0
    ASCENT_THRESHOLD = 10.0
    MIN_FORM_GATE = 0.3

    # ── Robust FSM config ───────────────────────────────────────────────
    # Visibility-only gate. User confirmed parallel-bar variant (torso
    # vertical, legs extended below). That geometry is *close* to the
    # PostureClassifier's STANDING rule, but STANDING is empirically
    # unreliable on real MediaPipe output, so we use visibility on the
    # full arm chain plus hip as the gate.
    SESSION_POSTURE_GATE = None
    VISIBILITY_GATE_LANDMARKS = (
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_ELBOW, RIGHT_ELBOW,
        LEFT_WRIST, RIGHT_WRIST,
        LEFT_HIP, RIGHT_HIP,
    )
    VISIBILITY_GATE_THRESHOLD = 0.6

    REP_DIRECTION = REP_DIRECTION_DECREASING
    REP_COMPLETE_TOP = 145.0      # back at extended zone
    REP_START_THRESHOLD = 140.0   # committed to dip
    REP_DEPTH_THRESHOLD = 100.0   # must reach ~90° depth (with margin)
    REP_BOTTOM_CLEAR = 110.0
    MIN_REAL_PRIMARY_DEG = 30.0
    MAX_PRIMARY_JUMP_DEG = 80.0

    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        elbow = self._avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
            (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
            name="elbow",
        )
        # NOTE: the elbow-shoulder-hip angle is actually shoulder extension ROM
        # (angle between upper arm and torso). Previously this was labeled
        # "shoulder" and conflated with forward lean — renamed for clarity.
        shoulder_extension = self._avg_angle(
            pose,
            (LEFT_ELBOW, LEFT_SHOULDER, LEFT_HIP),
            (RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP),
            name="shoulder_extension",
        )
        torso_lean = self._compute_torso_lean(pose)
        elbow_flare = self._compute_elbow_flare(pose)
        shoulder_below_elbow = self._compute_shoulder_below_elbow(pose)
        hip_closeness = self._compute_hip_closeness(pose)
        return {
            "primary": elbow, "elbow": elbow,
            "shoulder_extension": shoulder_extension,
            "torso_lean": torso_lean,
            "elbow_flare": elbow_flare,
            "shoulder_below_elbow": shoulder_below_elbow,
            "hip": hip_closeness,
        }

    def _compute_torso_lean(self, pose: PoseResult) -> Optional[float]:
        """Forward lean: angle of shoulder-hip line from pure vertical (T48).
        Separate from shoulder extension ROM — this measures TORSO tilt."""
        angles = []
        for s_idx, h_idx in [(LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP)]:
            if pose.is_visible(s_idx) and pose.is_visible(h_idx):
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

    def _compute_elbow_flare(self, pose: PoseResult) -> Optional[float]:
        """Frontal-plane elbow flare (T49): max of left/right upper-arm outward tilt
        when the arm is projected onto the frontal (XY) plane.
        0° = arm hangs straight down; 45° = elbow flared out to the side."""
        max_flare = None
        for s_idx, e_idx in [(LEFT_SHOULDER, LEFT_ELBOW), (RIGHT_SHOULDER, RIGHT_ELBOW)]:
            if not (pose.is_visible(s_idx) and pose.is_visible(e_idx)):
                continue
            s = np.array(pose.get_world_landmark(s_idx))
            e = np.array(pose.get_world_landmark(e_idx))
            # Project onto XY plane (drop Z), measure angle from -Y (straight down)
            dx = abs(e[0] - s[0])
            dy = e[1] - s[1]  # +Y = down in MediaPipe world
            if dy < 0.01:
                continue
            angle = float(np.degrees(np.arctan2(dx, dy)))
            if max_flare is None or angle > max_flare:
                max_flare = angle
        return max_flare

    def _compute_shoulder_below_elbow(self, pose: PoseResult) -> Optional[float]:
        """T50: secondary impingement proxy. Returns (shoulder_y - elbow_y) in world Y.
        MediaPipe: +Y = down. Positive value = shoulder DROPPED below elbow = excessive
        depth / loss of scapular control."""
        drops = []
        for s_idx, e_idx in [(LEFT_SHOULDER, LEFT_ELBOW), (RIGHT_SHOULDER, RIGHT_ELBOW)]:
            if pose.is_visible(s_idx) and pose.is_visible(e_idx):
                s_y = pose.get_world_landmark(s_idx)[1]
                e_y = pose.get_world_landmark(e_idx)[1]
                drops.append(float(s_y - e_y))
        return max(drops) if drops else None

    def _compute_hip_closeness(self, pose: PoseResult) -> Optional[float]:
        """Compute how far hips drift from shoulder vertical line (should stay close)."""
        offsets = []
        for s_idx, h_idx in [(LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP)]:
            if pose.is_visible(s_idx) and pose.is_visible(h_idx):
                s = np.array(pose.get_world_landmark(s_idx))
                h = np.array(pose.get_world_landmark(h_idx))
                # Horizontal distance (x-axis) — hips should be close to under shoulders
                offsets.append(abs(float(h[0] - s[0])))
        return sum(offsets) / len(offsets) if offsets else None

    def _check_start_position(self, angles: Dict[str, Optional[float]]) -> Tuple[bool, List[str]]:
        issues = []
        elbow = angles.get("elbow")
        if elbow is None:
            return False, ["Position camera to see arms — elbows must be visible"]
        if elbow < ELBOW_EXTENDED - 20:
            issues.append("Extend arms fully to start — press up on the bench")
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
        shoulder_ext = angles.get("shoulder_extension")
        torso_lean = angles.get("torso_lean")
        elbow_flare = angles.get("elbow_flare")
        shoulder_below_elbow = angles.get("shoulder_below_elbow")
        hip = angles.get("hip")

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

        # ── Shoulder extension ROM (T46 — Tier-1 injury-critical, weighted 1.6x) ──
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

        # ── Torso forward lean tiers (T48) ──
        if torso_lean is not None:
            if torso_lean <= TORSO_LEAN_TRICEPS:
                scores["torso_lean"] = 1.0
            elif torso_lean <= TORSO_LEAN_BALANCED:
                scores["torso_lean"] = 0.9
            elif torso_lean <= TORSO_LEAN_CHEST:
                scores["torso_lean"] = 0.8  # chest-emphasis variant, not an error
            else:
                issues.append("Too much forward lean — shifts work away from triceps")
                scores["torso_lean"] = 0.4

        # ── Frontal-plane elbow flare (T49) ──
        if elbow_flare is not None and in_transit:
            if elbow_flare <= FLARE_TUCK:              # <=15 — tucked, triceps-focused
                scores["flare"] = 1.0
            elif elbow_flare <= FLARE_NATURAL:         # 15-45 — natural
                scores["flare"] = 0.85
            elif elbow_flare <= FLARE_EXCESSIVE:       # 45-50 — watch
                issues.append("Elbows starting to flare — tuck them in")
                scores["flare"] = 0.5
            else:                                       # >50 — excessive
                issues.append("Elbows flaring wide — tuck them closer to your body")
                scores["flare"] = 0.3

        # ── Shoulder-below-elbow secondary proxy (T50) ──
        # If the shoulder drops below the elbow (Y_shoulder > Y_elbow in MediaPipe),
        # the user has sunk too far — secondary impingement indicator.
        if shoulder_below_elbow is not None and shoulder_below_elbow > 0.02:
            issues.append("Shoulders sinking below elbows — control depth")
            scores["shoulder_sink"] = 0.4

        # ── Hip closeness (should stay near bench/shoulders) ──
        if hip is not None:
            if hip <= 0.05:  # within 5cm
                joint_feedback["left_hip"] = "correct"
                joint_feedback["right_hip"] = "correct"
                scores["hip"] = 1.0
            elif hip <= 0.12:
                joint_feedback["left_hip"] = "warning"
                joint_feedback["right_hip"] = "warning"
                issues.append("Keep hips close to the bench")
                scores["hip"] = 0.5
            else:
                joint_feedback["left_hip"] = "incorrect"
                joint_feedback["right_hip"] = "incorrect"
                issues.append("Hips too far from bench — stay close")
                scores["hip"] = 0.2

        if not scores:
            return 0.0, joint_feedback, issues

        WEIGHTS = {
            "elbow": 1.0,
            "shoulder_ext": 1.6,     # Tier-1 injury-critical (McKenzie 2022)
            "torso_lean": 0.9,
            "flare": 1.0,
            "shoulder_sink": 1.1,
            "hip": 0.9,
        }
        weighted_sum = sum(scores[k] * WEIGHTS.get(k, 1.0) for k in scores)
        weight_total = sum(WEIGHTS.get(k, 1.0) for k in scores)
        total = weighted_sum / weight_total

        return max(0.0, min(1.0, total)), joint_feedback, issues
