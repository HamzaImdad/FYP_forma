"""
Dedicated deadlift detector with state machine and form assessment.

Angles tracked:
  - Hip (shoulder-hip-knee): primary angle for hip hinge pattern
  - Knee (hip-knee-ankle): slight bend, not excessive
  - Back straightness (shoulder-hip alignment)

State machine: TOP (standing) -> GOING_DOWN (hinge) -> BOTTOM -> GOING_UP -> TOP (1 rep)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .base_detector import (
    RobustExerciseDetector,
    RepPhase,
    REP_DIRECTION_DECREASING,
    SessionState,
)
from ..pose_estimation.base import PoseResult
from ..utils.constants import (
    NOSE, LEFT_EAR, RIGHT_EAR,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
)
from ..utils.geometry import calculate_angle


# Visibility threshold for deadlift — MediaPipe from a side view often
# drops the far-side (and sometimes near-side) landmarks to ~0.4–0.5.
# We accept anything above 0.4 so the hip/knee chain is computable from
# any camera angle.
DEADLIFT_VIS_THRESHOLD = 0.4


# ── Thresholds ──
# Hip angle at liftoff / bottom of pull (shoulder-hip-knee included angle):
# Escamilla et al. (2000) and Swinton et al. (2011) measured ~100-115° of hip
# *flexion* at liftoff for conventional deadlifts, which translates to an
# included angle of 65-80°. Our state-machine gate uses 85° (upper bound + 5°
# margin) so users reaching a legitimate 70-80° bottom trigger BOTTOM state,
# while stiff-legged partial pulls (bottoming above 85°) correctly fail to count.
HIP_STANDING = 155
HIP_BOTTOM = 85          # State-machine gate — liftoff position (research: 65-80°)
HIP_PARTIAL = 110        # Upper bound of "acceptable partial hinge" in form scoring
HIP_SHALLOW = 140        # Beyond this = not really a hinge

KNEE_MAX_BEND = 130      # Knees should stay relatively straight (relaxed for bodyweight)
KNEE_MIN = 155           # Ideal slight bend
KNEE_SQUAT = 110         # Too much knee bend = squatting not hinging

# ── Movement-gated session FSM thresholds ──
# FORM LOCKED fires only on a true hip hinge: hip committed past the rep-start
# threshold AND knees still near-straight. This discriminates a real deadlift
# setup from sitting down on a bench (where knees drop with the hips).
HINGE_HIP_COMMIT = 140.0   # hip must drop below this to count as a hinge
HINGE_KNEE_MIN = 155.0     # knees must stay above this (else it's a sit/squat)
AT_TOP_HIP_MIN = 160.0     # above this = clearly standing at lockout
REST_TOP_SECONDS = 8.0     # standing tall for this long → RESTING
REST_BOTTOM_SECONDS = 60.0 # hinged/sat below top this long → RESTING

# Back straightness: EAR-to-SHOULDER vertical ratio (image coordinates)
# Measures how far ear is above shoulder, normalized by torso length.
# Uses image-space Y for reliability — avoids unreliable world-coord depth.
BACK_RATIO_GOOD = 0.05     # Ear clearly above shoulder line (straight back)
BACK_RATIO_WARNING = -0.05  # Ear dropping toward shoulder (starting to round)
BACK_RATIO_BAD = -0.15      # Ear below shoulder level (significant rounding)


class DeadliftDetector(RobustExerciseDetector):
    EXERCISE_NAME = "deadlift"
    PRIMARY_ANGLE_TOP = HIP_STANDING
    PRIMARY_ANGLE_BOTTOM = HIP_BOTTOM
    DESCENT_THRESHOLD = 15.0
    ASCENT_THRESHOLD = 10.0

    # ── Robust FSM config ───────────────────────────────────────────────
    # Visibility-only gate: full torso + legs must be in frame for a
    # deadlift. Hip angle drives the rep FSM; knees and back are form
    # checks.
    SESSION_POSTURE_GATE = None
    VISIBILITY_GATE_LANDMARKS = (
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_HIP, RIGHT_HIP,
        LEFT_KNEE, RIGHT_KNEE,
    )
    # Lowered from 0.6 so side-view landmarks (which hover ~0.4–0.6 on
    # the near side) can still latch into ACTIVE. Paired with the custom
    # _check_visibility_gate override which accepts one full side.
    VISIBILITY_GATE_THRESHOLD = DEADLIFT_VIS_THRESHOLD

    REP_DIRECTION = REP_DIRECTION_DECREASING
    # Rep-FSM thresholds on hip angle. Standing lockout ~155°, bottom of
    # hinge ~85°. We commit when hip has dropped past 140° (committed
    # hinge) and recovered back past 145° with peak <= 105° (valid depth).
    REP_COMPLETE_TOP = 145.0      # back at top = locked out
    REP_START_THRESHOLD = 140.0   # past this = committed to hinge
    REP_DEPTH_THRESHOLD = 105.0   # hip must reach this for valid depth
    REP_BOTTOM_CLEAR = 115.0      # past this on ascent = leaving bottom
    MIN_REAL_PRIMARY_DEG = 30.0
    MAX_PRIMARY_JUMP_DEG = 60.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prev_shoulder_y = None
        self._prev_hip_y = None
        # Movement-gated session FSM timers
        self._at_top_since: Optional[float] = None
        self._below_top_since: Optional[float] = None
        self._last_hinge_time: float = 0.0

    def reset(self):
        super().reset()
        self._prev_shoulder_y = None
        self._prev_hip_y = None
        self._at_top_since = None
        self._below_top_since = None
        self._last_hinge_time = 0.0

    # ── Movement-gated session FSM ──────────────────────────────────────
    # Overrides the base visibility-only gate. FORM LOCKED fires only when
    # the user commits to a real hip hinge (hip past REP_START_THRESHOLD AND
    # knees still near-straight). RESTING fires after 8s standing idle at
    # lockout OR after 60s below lockout (gave up / sat down).
    def _update_session_state(self, now: float, visibility_ok: bool) -> None:
        hip = self._get_smooth("hip")
        knee = self._get_smooth("knee")

        # If either signal is unavailable, defer to the base visibility gate.
        if hip is None or knee is None:
            super()._update_session_state(now, visibility_ok)
            return

        is_true_hinge = (hip < HINGE_HIP_COMMIT and knee >= HINGE_KNEE_MIN)
        is_at_top = hip >= AT_TOP_HIP_MIN

        # Update at-top / below-top continuous-window timers
        if is_at_top:
            if self._at_top_since is None:
                self._at_top_since = now
            self._below_top_since = None
        else:
            if self._below_top_since is None:
                self._below_top_since = now
            self._at_top_since = None

        if is_true_hinge:
            self._last_hinge_time = now

        s = self._session_state
        pose = self._last_pose
        in_stance = bool(pose is not None and self._is_in_stance(pose, visibility_ok))

        # ── ACTIVE → RESTING ──
        # Primary close: standing at lockout for REST_TOP_SECONDS (8s) —
        # matches the squat pattern. This is what catches "user finished
        # their set and is just standing there". Secondary close: below
        # top for REST_BOTTOM_SECONDS (60s) — user gave up mid-rep / sat
        # down. Fallback: 3s out-of-stance debounce (landmarks lost /
        # walked off camera).
        if s == SessionState.ACTIVE:
            if (
                self._at_top_since is not None
                and now - self._at_top_since >= REST_TOP_SECONDS
            ):
                self._close_active_set_or_rollback(now)
                return
            if (
                self._below_top_since is not None
                and now - self._below_top_since >= REST_BOTTOM_SECONDS
            ):
                self._close_active_set_or_rollback(now)
                return
            if self._should_close_for_out_of_stance(now, in_stance):
                self._close_active_set_or_rollback(now)
            else:
                self._unknown_streak = 0
            return

        if not visibility_ok:
            return
        self._unknown_streak = 0

        # ── IDLE → SETUP: first time the body is visible ──
        if s == SessionState.IDLE:
            self._session_state = SessionState.SETUP
            return

        # ── SETUP → ACTIVE: only on a true hinge (seed first rep from trace) ──
        if s == SessionState.SETUP:
            if is_true_hinge:
                self._session_state = SessionState.ACTIVE
                self._reset_robust_rep_fsm()
                self._rep_start_time = now
                self._out_of_stance_since = None
                # First rep may have started during SETUP bootstrap — recover it.
                self._seed_rep_fsm_from_pre_active(now)
            return

        # ── RESTING → ACTIVE: fresh start, do NOT seed from stale trace ──
        if s == SessionState.RESTING:
            if is_true_hinge:
                self._session_state = SessionState.ACTIVE
                self._reset_robust_rep_fsm()
                self._rep_start_time = now
                self._out_of_stance_since = None
            return

    # ── Side-view-friendly visibility gate ──────────────────────────────
    # Default gate requires ALL of (L+R shoulder, L+R hip, L+R knee) visible,
    # which excludes pure side-on cameras because the far side's landmarks
    # are inferred by MediaPipe and drop below the 0.6 threshold. Accept the
    # gate if EITHER side's shoulder-hip-knee chain is fully visible; the
    # angle calculations (_angle_at / _avg_angle) already fall back to the
    # visible side when the other returns None.
    def _check_visibility_gate(self, pose) -> bool:
        left_ok = all(
            pose.is_visible(idx, self.VISIBILITY_GATE_THRESHOLD)
            for idx in (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE)
        )
        right_ok = all(
            pose.is_visible(idx, self.VISIBILITY_GATE_THRESHOLD)
            for idx in (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE)
        )
        return left_ok or right_ok

    def _should_count_rep(self, elapsed: float, angles: Dict[str, Optional[float]]) -> bool:
        """Only count rep if form score averaged above 0.3 during the rep.

        Prevents random movements (bending to pick something up, sitting down,
        front raises) from counting as deadlift reps.
        """
        if not super()._should_count_rep(elapsed, angles):
            return False
        if self._current_rep_scores:
            avg = sum(self._current_rep_scores) / len(self._current_rep_scores)
            if avg < 0.3:
                return False
        return True

    def _lenient_side_angle(
        self,
        pose: PoseResult,
        a_idx: int,
        b_idx: int,
        c_idx: int,
        threshold: float = DEADLIFT_VIS_THRESHOLD,
    ) -> Optional[float]:
        """Compute world-coordinate angle at vertex b, accepting landmarks
        whose visibility is at least `threshold` (default 0.4 for deadlift,
        vs. the global MIN_VISIBILITY=0.5 used by the base class's
        `_angle_at`). Lowered floor lets near-side landmarks pass from a
        pure side view where MediaPipe often reports ~0.4–0.5 confidence.
        """
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
        falling back to whichever side is available. Stores L/R values for
        asymmetry detection (same contract as base `_avg_angle`)."""
        left = self._lenient_side_angle(pose, *left_triple)
        right = self._lenient_side_angle(pose, *right_triple)
        self._lr_angles[name] = (left, right)
        if left is not None and right is not None:
            return (left + right) / 2
        return left if left is not None else right

    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        # Use lenient angle computation so side-view (and 3/4 view) work:
        # near-side landmarks from MediaPipe frequently hover in the
        # 0.4–0.6 visibility range, which the default 0.5 floor would
        # reject. The world-coordinate angle itself is accurate from any
        # camera angle once MediaPipe's pose rig has located the joint.
        hip = self._lenient_avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
            (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
            name="hip",
        )
        knee = self._lenient_avg_angle(
            pose,
            (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
            (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
            name="knee",
        )

        # Back straightness: spine curvature proxy (NOSE -> mid-shoulder -> mid-hip)
        back = self._compute_back_angle(pose)

        # Hip shoot detection (hips rising faster than shoulders)
        hip_shoot = self._check_hip_shoot(pose)

        return {"primary": hip, "hip": hip, "knee": knee, "back": back, "hip_shoot": hip_shoot}

    def _compute_back_angle(self, pose: PoseResult) -> Optional[float]:
        """Measure back straightness via EAR-to-SHOULDER vertical displacement ratio.

        Uses IMAGE coordinates (Y axis) instead of unreliable world-coordinate
        depth.  Returns a ratio: positive = ear above shoulder (straight back),
        negative = ear at/below shoulder (rounding).  Normalized by torso length
        so the metric is camera-distance-invariant.

        This replaces the old NOSE→mid_shoulder→mid_hip angle which was
        unreliable from laptop cameras (83° variance while standing, inverted
        at bottom of hinge).
        """
        # Average left/right ear Y (image space, 0=top 1=bottom)
        ear_y = self._avg_image_y(pose, LEFT_EAR, RIGHT_EAR)
        shoulder_y = self._avg_image_y(pose, LEFT_SHOULDER, RIGHT_SHOULDER)
        hip_y = self._avg_image_y(pose, LEFT_HIP, RIGHT_HIP)

        if ear_y is None or shoulder_y is None or hip_y is None:
            return None

        torso_length = abs(hip_y - shoulder_y)
        if torso_length < 0.02:  # too small to measure
            return None

        # Positive = ear above shoulder (good), negative = ear dropping (rounding)
        ratio = (shoulder_y - ear_y) / torso_length
        return ratio

    @staticmethod
    def _avg_image_y(pose: PoseResult, idx_left: int, idx_right: int) -> Optional[float]:
        """Average image-space Y coordinate of left/right landmarks."""
        l_vis = pose.is_visible(idx_left)
        r_vis = pose.is_visible(idx_right)
        if l_vis and r_vis:
            return (pose.landmarks[idx_left, 1] + pose.landmarks[idx_right, 1]) / 2
        if l_vis:
            return float(pose.landmarks[idx_left, 1])
        if r_vis:
            return float(pose.landmarks[idx_right, 1])
        return None

    def _check_hip_shoot(self, pose: PoseResult) -> Optional[float]:
        """Detect if hips rise faster than shoulders during ascent (stripper pull).
        Returns ratio of hip_rise/shoulder_rise. >1.5 = hip shoot."""
        ls = pose.get_world_landmark(LEFT_SHOULDER) if pose.is_visible(LEFT_SHOULDER) else None
        rs = pose.get_world_landmark(RIGHT_SHOULDER) if pose.is_visible(RIGHT_SHOULDER) else None
        lh = pose.get_world_landmark(LEFT_HIP) if pose.is_visible(LEFT_HIP) else None
        rh = pose.get_world_landmark(RIGHT_HIP) if pose.is_visible(RIGHT_HIP) else None

        if (ls is None and rs is None) or (lh is None and rh is None):
            self._prev_shoulder_y = None
            self._prev_hip_y = None
            return None

        shoulder_y = float(np.mean([l[1] for l in [ls, rs] if l is not None]))
        hip_y = float(np.mean([l[1] for l in [lh, rh] if l is not None]))

        ratio = None
        if self._prev_shoulder_y is not None and self._prev_hip_y is not None:
            shoulder_rise = self._prev_shoulder_y - shoulder_y  # positive = rising (Y decreases going up in MediaPipe)
            hip_rise = self._prev_hip_y - hip_y
            if shoulder_rise > 0.001:  # shoulders actually moving up
                ratio = hip_rise / shoulder_rise

        self._prev_shoulder_y = shoulder_y
        self._prev_hip_y = hip_y
        return ratio

    def _check_start_position(self, angles: Dict[str, Optional[float]]) -> Tuple[bool, List[str]]:
        issues = []
        hip = angles.get("hip")
        if hip is None:
            return False, ["Stand facing the camera — full body must be visible"]
        if hip < HIP_STANDING - 20:
            issues.append("Stand up straight to begin")
        return (len(issues) == 0, issues)

    def _get_missing_parts(self, angles: Dict[str, Optional[float]]) -> List[str]:
        if angles.get("hip") is None:
            return ["hips and shoulders"]
        return ["body"]

    def _assess_form(self, angles: Dict[str, Optional[float]]) -> Tuple[float, Dict[str, str], List[str]]:
        joint_feedback = {}
        issues = []
        scores = {}

        hip = angles.get("hip")
        knee = angles.get("knee")
        back = angles.get("back")  # now an EAR-shoulder ratio, not an angle

        # ── Hip hinge depth ──
        hip_wants_deeper = False
        if hip is not None:
            if self._state in (RepPhase.BOTTOM, RepPhase.GOING_UP, RepPhase.GOING_DOWN):
                if hip <= HIP_BOTTOM:
                    joint_feedback["left_hip"] = "correct"
                    joint_feedback["right_hip"] = "correct"
                    scores["hip"] = 1.0
                elif hip <= HIP_PARTIAL:
                    joint_feedback["left_hip"] = "correct"
                    joint_feedback["right_hip"] = "correct"
                    scores["hip"] = 0.85
                elif hip <= HIP_SHALLOW:
                    joint_feedback["left_hip"] = "warning"
                    joint_feedback["right_hip"] = "warning"
                    issues.append("Hinge a bit deeper — push hips back")
                    scores["hip"] = 0.6
                    hip_wants_deeper = True
                else:
                    joint_feedback["left_hip"] = "warning"
                    joint_feedback["right_hip"] = "warning"
                    issues.append("Push your hips back and hinge forward")
                    scores["hip"] = 0.3
                    hip_wants_deeper = True
            else:
                # At top — should be fully extended
                if hip >= HIP_STANDING - 10:
                    joint_feedback["left_hip"] = "correct"
                    joint_feedback["right_hip"] = "correct"
                    scores["hip"] = 1.0
                else:
                    joint_feedback["left_hip"] = "warning"
                    joint_feedback["right_hip"] = "warning"
                    issues.append("Stand up fully at the top")
                    scores["hip"] = 0.7

        # ── Knee bend ──
        # Only warn about knee bend if we're NOT also telling user to hinge deeper
        # (deeper hinge naturally requires some knee bend — avoid contradiction)
        if knee is not None:
            if knee >= KNEE_MIN:
                joint_feedback["left_knee"] = "correct"
                joint_feedback["right_knee"] = "correct"
                scores["knee"] = 1.0
            elif knee >= KNEE_MAX_BEND:
                joint_feedback["left_knee"] = "correct"
                joint_feedback["right_knee"] = "correct"
                scores["knee"] = 0.8  # minor bend is OK, especially bodyweight
            elif knee >= KNEE_SQUAT:
                joint_feedback["left_knee"] = "warning"
                joint_feedback["right_knee"] = "warning"
                if not hip_wants_deeper:  # don't contradict "hinge deeper"
                    issues.append("Try to keep legs a bit straighter")
                scores["knee"] = 0.5
            else:
                joint_feedback["left_knee"] = "incorrect"
                joint_feedback["right_knee"] = "incorrect"
                issues.append("Too much knee bend — push hips back instead")
                scores["knee"] = 0.2

        # ── Back straightness (EAR-to-shoulder ratio) ──
        # back is now a ratio: positive = ear above shoulder (good),
        # negative = ear dropping (rounding). See _compute_back_angle().
        if back is not None:
            if back >= BACK_RATIO_GOOD:
                joint_feedback["left_shoulder"] = "correct"
                joint_feedback["right_shoulder"] = "correct"
                scores["back"] = 1.0
            elif back >= BACK_RATIO_WARNING:
                joint_feedback["left_shoulder"] = "correct"
                joint_feedback["right_shoulder"] = "correct"
                scores["back"] = 0.8
            elif back >= BACK_RATIO_BAD:
                joint_feedback["left_shoulder"] = "warning"
                joint_feedback["right_shoulder"] = "warning"
                issues.append("Keep chest up — back starting to round")
                scores["back"] = 0.5
            else:
                joint_feedback["left_shoulder"] = "incorrect"
                joint_feedback["right_shoulder"] = "incorrect"
                issues.append("Back rounding — straighten up, chest proud")
                scores["back"] = 0.2

        # ── Hip shoot (during ascent) ──
        hip_shoot = angles.get("hip_shoot")
        if hip_shoot is not None and self._state == RepPhase.GOING_UP:
            if hip_shoot > 2.0:
                issues.append("Lead with your chest — hips shooting up")
                scores["hip_shoot"] = 0.3
            elif hip_shoot > 1.5:
                issues.append("Drive chest and hips up together")
                scores["hip_shoot"] = 0.6

        if not scores:
            return 0.0, joint_feedback, issues

        # Balanced weights — back no longer dominates unfairly
        WEIGHTS = {"hip": 1.2, "knee": 0.8, "back": 1.0, "hip_shoot": 1.0}
        weighted_sum = sum(scores[k] * WEIGHTS.get(k, 1.0) for k in scores)
        weight_total = sum(WEIGHTS.get(k, 1.0) for k in scores)
        total = weighted_sum / weight_total

        return max(0.0, min(1.0, total)), joint_feedback, issues
