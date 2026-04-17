"""
Dedicated side plank detector — static hold, no rep counting.

Angles tracked:
  - Body line (shoulder-hip-ankle): should be ~175-180° (straight line)
  - Hip deviation (perpendicular distance from hip to shoulder-ankle line)
  - Hip rotation (L/R hip x-spread in image coords)
  - Elbow under shoulder (world-coord x-offset ratio)
  - Head/neck alignment (ear deviation from spine line)
  - Knee angle (legs straight)

Camera: SIDE VIEW — facing the front of the body, full body head to feet.
Tracks hold duration instead of reps (IS_STATIC = True).
Detects which side (left or right) is the supporting side.
"""

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .base_detector import (
    RobustExerciseDetector,
    SessionState,
)
from ..pose_estimation.base import PoseResult
from ..utils.constants import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_EAR, RIGHT_EAR,
)
from ..utils.geometry import calculate_angle


# ── Body line thresholds ──
BODY_LINE_GOOD = 175
BODY_LINE_WARNING = 170
BODY_LINE_BAD = 160

# ── Hip deviation (perpendicular distance as % of body length) ──
HIP_SAG_GOOD = 0.03        # < 3% = perfect
HIP_SAG_WARNING = 0.06     # 3-6% = acceptable but warn
HIP_SAG_BAD = 0.10         # > 10% = severe, pause timer
HIP_PIKE_WARNING = 0.05    # > 5% above line = piking

# ── Hip rotation (image x-spread of L/R hip) ──
HIP_ROTATION_GOOD = 0.03   # hips stacked
HIP_ROTATION_WARNING = 0.06
HIP_ROTATION_BAD = 0.10

# ── Elbow under shoulder ──
ELBOW_OFFSET_GOOD = 0.15   # x-offset / upper_arm_length
ELBOW_OFFSET_WARNING = 0.25
ELBOW_ANGLE_MIN = 60       # supporting arm elbow angle
ELBOW_ANGLE_MAX = 120

# ── Head/neck alignment (% of ear-shoulder distance) ──
HEAD_OFFSET_GOOD = 0.20
HEAD_OFFSET_WARNING = 0.30

# ── Knee angle ──
KNEE_STRAIGHT = 170

# ── Fatigue drift ──
BASELINE_FRAMES = 10
DRIFT_WARN_DEG = 5
DRIFT_STOP_DEG = 10

# ── Side-view visibility ──
SIDE_PLANK_VIS_THRESHOLD = 0.3

# ── Static hold position gate ──
SIDE_PLANK_BODY_LINE_MIN = 160.0
SIDE_PLANK_BODY_LINE_MAX = 195.0


class SidePlankDetector(RobustExerciseDetector):
    EXERCISE_NAME = "side_plank"
    IS_STATIC = True
    PRIMARY_ANGLE_TOP = 180.0     # straight line (perfect)
    PRIMARY_ANGLE_BOTTOM = 155.0  # form breakdown

    SESSION_POSTURE_GATE = None
    VISIBILITY_GATE_LANDMARKS = (
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_ELBOW, RIGHT_ELBOW,
        LEFT_HIP, RIGHT_HIP,
        LEFT_ANKLE, RIGHT_ANKLE,
    )
    VISIBILITY_GATE_THRESHOLD = SIDE_PLANK_VIS_THRESHOLD
    MIN_REAL_PRIMARY_DEG = 0.0
    MAX_PRIMARY_JUMP_DEG = 90.0

    def __init__(self, **kwargs):
        # Instance state BEFORE super().__init__() — reset() calls .clear()
        self._supporting_side: Optional[str] = None
        self._hip_deviation_buf: Deque[float] = deque(maxlen=10)
        self._baseline_body_line: Optional[float] = None
        self._baseline_sample_buf: List[float] = []
        self._drift_stop_warned: bool = False
        # Position-gated FSM timers (mirrors plank_detector pattern)
        self._side_plank_position_since: Optional[float] = None
        self._not_side_plank_since: Optional[float] = None
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self._supporting_side = None
        self._hip_deviation_buf.clear()
        self._baseline_body_line = None
        self._baseline_sample_buf = []
        self._side_plank_position_since = None
        self._not_side_plank_since = None
        self._drift_stop_warned = False

    # ── Side-view visibility gate ──────────────────────────────────────
    def _check_visibility_gate(self, pose) -> bool:
        """Accept if EITHER side's shoulder-hip-ankle chain is visible."""
        left_ok = all(
            pose.is_visible(idx, SIDE_PLANK_VIS_THRESHOLD)
            for idx in (LEFT_SHOULDER, LEFT_HIP, LEFT_ANKLE)
        )
        right_ok = all(
            pose.is_visible(idx, SIDE_PLANK_VIS_THRESHOLD)
            for idx in (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_ANKLE)
        )
        return left_ok or right_ok

    # ── Lenient angle helpers (side-view) ──────────────────────────────
    def _lenient_side_angle(
        self,
        pose: PoseResult,
        a_idx: int,
        b_idx: int,
        c_idx: int,
        threshold: float = SIDE_PLANK_VIS_THRESHOLD,
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

    # ── Position-gated session FSM (mirrors plank_detector pattern) ───
    def _update_session_state(self, now: float, visibility_ok: bool) -> None:
        """Side plank is IS_STATIC — gate on being in side plank position
        (body roughly straight + on forearm), not just visibility."""
        body_line = self._get_smooth("body_line")

        # Determine if user is in side plank position
        in_side_plank = (
            body_line is not None
            and SIDE_PLANK_BODY_LINE_MIN <= body_line <= SIDE_PLANK_BODY_LINE_MAX
        )

        # Update position timers
        if in_side_plank:
            if self._side_plank_position_since is None:
                self._side_plank_position_since = now
            self._not_side_plank_since = None
        else:
            if self._not_side_plank_since is None:
                self._not_side_plank_since = now
            self._side_plank_position_since = None

        s = self._session_state

        # Visibility loss handling
        if not visibility_ok:
            if s == SessionState.ACTIVE:
                self._unknown_streak += 1
                if self._unknown_streak >= self.UNKNOWN_GRACE_FRAMES:
                    self._close_active_set_or_rollback(now)
            return
        self._unknown_streak = 0

        # IDLE → SETUP: body visible
        if s == SessionState.IDLE:
            self._session_state = SessionState.SETUP
            return

        # SETUP → ACTIVE: in side plank position sustained for 1.5s
        if s == SessionState.SETUP:
            if (self._side_plank_position_since is not None
                    and now - self._side_plank_position_since >= 1.5):
                self._session_state = SessionState.ACTIVE
            return

        # RESTING → ACTIVE: same as SETUP
        if s == SessionState.RESTING:
            if (self._side_plank_position_since is not None
                    and now - self._side_plank_position_since >= 1.5):
                self._session_state = SessionState.ACTIVE
            return

        # ACTIVE → RESTING: out of side plank position for 5s
        if s == SessionState.ACTIVE:
            if (self._not_side_plank_since is not None
                    and now - self._not_side_plank_since >= 5.0):
                self._close_active_set_or_rollback(now)
                return

    # ── Which-side detection ───────────────────────────────────────────
    def _detect_supporting_side(self, pose: PoseResult) -> Optional[str]:
        """Detect which side is supporting (lower shoulder in image = supporting).
        In image coords, higher Y = lower in frame = closer to floor."""
        left_vis = pose.is_visible(LEFT_SHOULDER, SIDE_PLANK_VIS_THRESHOLD)
        right_vis = pose.is_visible(RIGHT_SHOULDER, SIDE_PLANK_VIS_THRESHOLD)
        if not left_vis and not right_vis:
            return None
        if left_vis and right_vis:
            left_y = pose.landmarks[LEFT_SHOULDER][1]
            right_y = pose.landmarks[RIGHT_SHOULDER][1]
            return "left" if left_y > right_y else "right"
        return "left" if left_vis else "right"

    # ── Angle computation ──────────────────────────────────────────────
    def _compute_angles(self, pose: PoseResult) -> Dict[str, Optional[float]]:
        # Detect supporting side
        side = self._detect_supporting_side(pose)
        if side is not None:
            self._supporting_side = side

        # Primary: body line (shoulder-hip-ankle)
        body_line = self._lenient_avg_angle(
            pose,
            (LEFT_SHOULDER, LEFT_HIP, LEFT_ANKLE),
            (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_ANKLE),
            name="body_line",
        )

        # Hip deviation (perpendicular distance, normalized by body length)
        hip_dev = self._compute_hip_deviation(pose)

        # Hip rotation
        hip_rotation = self._compute_hip_rotation(pose)

        # Elbow under shoulder
        elbow_offset, elbow_angle = self._compute_elbow_position(pose)

        # Head/neck alignment
        head_offset = self._compute_head_alignment(pose)

        # Knee angle
        knee = self._lenient_avg_angle(
            pose,
            (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
            (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
            name="knee",
        )

        # Fatigue drift tracking (baseline capture + drift computation)
        if (self._form_validated and body_line is not None
                and self._baseline_body_line is None):
            self._baseline_sample_buf.append(body_line)
            if len(self._baseline_sample_buf) >= BASELINE_FRAMES:
                self._baseline_body_line = (
                    sum(self._baseline_sample_buf) / len(self._baseline_sample_buf)
                )
                self._baseline_sample_buf = []

        drift = None
        if self._baseline_body_line is not None and body_line is not None:
            drift = abs(body_line - self._baseline_body_line)

        return {
            "primary": body_line, "body_line": body_line,
            "hip_deviation": hip_dev,
            "hip_rotation": hip_rotation,
            "elbow_offset": elbow_offset,
            "elbow_angle": elbow_angle,
            "head_offset": head_offset,
            "knee": knee,
            "drift": drift,
        }

    def _compute_hip_deviation(self, pose: PoseResult) -> Optional[float]:
        """Perpendicular distance from hip to shoulder-ankle line,
        normalized by shoulder-to-ankle distance (body length).
        Positive = sagging (hip below line), Negative = piking (hip above line).
        Uses world coordinates."""
        deviations = []
        for s_idx, h_idx, a_idx in [
            (LEFT_SHOULDER, LEFT_HIP, LEFT_ANKLE),
            (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_ANKLE),
        ]:
            if all(pose.is_visible(i, SIDE_PLANK_VIS_THRESHOLD) for i in (s_idx, h_idx, a_idx)):
                s = np.array(pose.get_world_landmark(s_idx))
                h = np.array(pose.get_world_landmark(h_idx))
                a = np.array(pose.get_world_landmark(a_idx))

                sa = a - s
                sh = h - s
                sa_len = np.linalg.norm(sa)
                if sa_len < 1e-6:
                    continue

                # Project hip onto shoulder-ankle line
                t = np.dot(sh, sa) / (sa_len ** 2)
                t = np.clip(t, 0, 1)
                projected = s + t * sa
                perp = h - projected
                # Signed deviation: positive Y = hip sagging
                signed_dev = float(perp[1])
                # Normalize by body length
                normalized = signed_dev / sa_len
                deviations.append(normalized)

        if not deviations:
            return None

        result = sum(deviations) / len(deviations)

        # Temporal smoothing
        self._hip_deviation_buf.append(result)
        if len(self._hip_deviation_buf) >= 3:
            return sum(self._hip_deviation_buf) / len(self._hip_deviation_buf)
        return result

    def _compute_hip_rotation(self, pose: PoseResult) -> Optional[float]:
        """Detect hip rotation by measuring L/R hip x-spread in image coords.
        In a true side plank, both hips are stacked (spread ≈ 0)."""
        if (pose.is_visible(LEFT_HIP, SIDE_PLANK_VIS_THRESHOLD)
                and pose.is_visible(RIGHT_HIP, SIDE_PLANK_VIS_THRESHOLD)):
            left_x = pose.landmarks[LEFT_HIP][0]
            right_x = pose.landmarks[RIGHT_HIP][0]
            return abs(float(left_x - right_x))
        return None

    def _compute_elbow_position(
        self, pose: PoseResult
    ) -> Tuple[Optional[float], Optional[float]]:
        """Check if supporting elbow is directly under shoulder.
        Returns (offset_ratio, elbow_angle).
        offset_ratio = |elbow_x - shoulder_x| / upper_arm_length"""
        if self._supporting_side is None:
            return None, None

        if self._supporting_side == "left":
            s_idx, e_idx, w_idx = LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
        else:
            s_idx, e_idx, w_idx = RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST

        if not all(pose.is_visible(i, SIDE_PLANK_VIS_THRESHOLD) for i in (s_idx, e_idx)):
            return None, None

        s = np.array(pose.get_world_landmark(s_idx))
        e = np.array(pose.get_world_landmark(e_idx))
        upper_arm_len = float(np.linalg.norm(s - e))
        if upper_arm_len < 1e-6:
            return None, None

        x_offset = abs(float(s[0] - e[0]))
        offset_ratio = x_offset / upper_arm_len

        # Elbow angle (if wrist visible)
        elbow_angle = None
        if pose.is_visible(w_idx, SIDE_PLANK_VIS_THRESHOLD):
            elbow_angle = calculate_angle(
                pose.get_world_landmark(s_idx),
                pose.get_world_landmark(e_idx),
                pose.get_world_landmark(w_idx),
            )

        return offset_ratio, elbow_angle

    def _compute_head_alignment(self, pose: PoseResult) -> Optional[float]:
        """Perpendicular distance from ear to spine line (hip→shoulder extended),
        normalized by ear-to-shoulder distance. Uses world coords."""
        if self._supporting_side is None:
            return None

        # Use the TOP side ear (more visible from front camera angle)
        if self._supporting_side == "left":
            ear_idx, s_idx, h_idx = RIGHT_EAR, RIGHT_SHOULDER, RIGHT_HIP
        else:
            ear_idx, s_idx, h_idx = LEFT_EAR, LEFT_SHOULDER, LEFT_HIP

        # Fall back to whichever side is visible
        if not all(pose.is_visible(i, 0.2) for i in (ear_idx, s_idx, h_idx)):
            # Try the other side
            if self._supporting_side == "left":
                ear_idx, s_idx, h_idx = LEFT_EAR, LEFT_SHOULDER, LEFT_HIP
            else:
                ear_idx, s_idx, h_idx = RIGHT_EAR, RIGHT_SHOULDER, RIGHT_HIP
            if not all(pose.is_visible(i, 0.2) for i in (ear_idx, s_idx, h_idx)):
                return None

        ear = np.array(pose.get_world_landmark(ear_idx))
        s = np.array(pose.get_world_landmark(s_idx))
        h = np.array(pose.get_world_landmark(h_idx))

        # Spine direction: hip → shoulder
        spine = s - h
        spine_len = np.linalg.norm(spine)
        if spine_len < 1e-6:
            return None

        # Ear relative to hip
        ear_from_hip = ear - h

        # Project ear onto spine line
        t = np.dot(ear_from_hip, spine) / (spine_len ** 2)
        projected = h + t * spine
        perp = ear - projected
        perp_dist = float(np.linalg.norm(perp))

        # Normalize by ear-to-shoulder distance
        ear_shoulder_dist = float(np.linalg.norm(ear - s))
        if ear_shoulder_dist < 1e-6:
            return None

        return perp_dist / ear_shoulder_dist

    # ── Start position / missing parts ─────────────────────────────────
    def _check_start_position(
        self, angles: Dict[str, Optional[float]]
    ) -> Tuple[bool, List[str]]:
        issues = []
        body_line = angles.get("body_line")
        elbow_angle = angles.get("elbow_angle")

        if body_line is None:
            return False, [
                "Get into side plank position \u2014 full body must be visible from the side"
            ]

        if body_line < BODY_LINE_BAD:
            issues.append("Straighten your body \u2014 align shoulders, hips, and ankles")
        if elbow_angle is not None and not (ELBOW_ANGLE_MIN <= elbow_angle <= ELBOW_ANGLE_MAX):
            issues.append("Support yourself on your forearm \u2014 elbow under shoulder")

        return (len(issues) == 0, issues)

    def _get_missing_parts(self, angles: Dict[str, Optional[float]]) -> List[str]:
        if angles.get("body_line") is None:
            return ["shoulders, hips, or ankles"]
        return ["body"]

    # ── Form assessment ────────────────────────────────────────────────
    def _assess_form(
        self, angles: Dict[str, Optional[float]]
    ) -> Tuple[float, Dict[str, str], List[str]]:
        joint_feedback: Dict[str, str] = {}
        issues: List[str] = []
        scores: Dict[str, float] = {}

        body_line = angles.get("body_line")
        hip_dev = angles.get("hip_deviation")
        hip_rot = angles.get("hip_rotation")
        elbow_offset = angles.get("elbow_offset")
        elbow_angle = angles.get("elbow_angle")
        head_offset = angles.get("head_offset")
        knee = angles.get("knee")
        drift = angles.get("drift")

        # Pre-compute: drift is the better fatigue signal
        _drift_active = drift is not None and drift >= DRIFT_WARN_DEG

        # ── Hip sag/pike (perpendicular distance — weight 2.0) ──
        if hip_dev is not None:
            abs_dev = abs(hip_dev)
            if hip_dev > 0:  # sagging
                if abs_dev < HIP_SAG_GOOD:
                    scores["hip_sag"] = 1.0
                elif abs_dev < HIP_SAG_WARNING:
                    if not _drift_active:
                        issues.append("Hips sagging \u2014 lift them in line with shoulders")
                    joint_feedback["left_hip"] = "warning"
                    joint_feedback["right_hip"] = "warning"
                    scores["hip_sag"] = 0.5
                else:
                    if not _drift_active:
                        issues.append("Hips sagging too much \u2014 tighten core, lift hips")
                    joint_feedback["left_hip"] = "incorrect"
                    joint_feedback["right_hip"] = "incorrect"
                    scores["hip_sag"] = 0.2
            else:  # piking (negative = above line)
                if abs_dev < HIP_SAG_GOOD:
                    scores["hip_sag"] = 1.0
                elif abs_dev < HIP_PIKE_WARNING:
                    scores["hip_sag"] = 0.8  # mild pike less common, soft warn
                else:
                    issues.append("Hips too high \u2014 lower them in line with body")
                    joint_feedback["left_hip"] = "warning"
                    joint_feedback["right_hip"] = "warning"
                    scores["hip_sag"] = 0.4

        # ── Body line angle (weight 1.5) ──
        if body_line is not None:
            if body_line >= BODY_LINE_GOOD:
                joint_feedback["left_hip"] = joint_feedback.get("left_hip", "correct")
                joint_feedback["right_hip"] = joint_feedback.get("right_hip", "correct")
                scores["body_line"] = 1.0
            elif body_line >= BODY_LINE_WARNING:
                scores["body_line"] = 0.6
            elif body_line >= BODY_LINE_BAD:
                scores["body_line"] = 0.3
            else:
                scores["body_line"] = 0.1

        # ── Hip rotation (weight 1.5) ──
        if hip_rot is not None:
            if hip_rot < HIP_ROTATION_GOOD:
                scores["hip_rotation"] = 1.0
            elif hip_rot < HIP_ROTATION_WARNING:
                issues.append("Hips rotating \u2014 keep them stacked, don't twist")
                scores["hip_rotation"] = 0.5
            else:
                issues.append("Body rotating too much \u2014 face forward, stack hips")
                scores["hip_rotation"] = 0.2

        # ── Elbow under shoulder (weight 1.3) ──
        if elbow_offset is not None:
            elbow_angle_ok = (
                elbow_angle is not None
                and ELBOW_ANGLE_MIN <= elbow_angle <= ELBOW_ANGLE_MAX
            )
            if elbow_offset < ELBOW_OFFSET_GOOD and elbow_angle_ok:
                joint_feedback["left_elbow"] = "correct"
                joint_feedback["right_elbow"] = "correct"
                scores["elbow_position"] = 1.0
            elif elbow_offset < ELBOW_OFFSET_WARNING:
                joint_feedback["left_elbow"] = "warning"
                joint_feedback["right_elbow"] = "warning"
                issues.append("Shift elbow directly under shoulder")
                scores["elbow_position"] = 0.6
            else:
                joint_feedback["left_elbow"] = "incorrect"
                joint_feedback["right_elbow"] = "incorrect"
                issues.append("Elbow too far from shoulder \u2014 reposition")
                scores["elbow_position"] = 0.3

        # ── Head/neck neutral (weight 1.0) ──
        if head_offset is not None:
            if head_offset < HEAD_OFFSET_GOOD:
                scores["head_neck"] = 1.0
            elif head_offset < HEAD_OFFSET_WARNING:
                issues.append("Keep head in line with spine \u2014 look straight ahead")
                scores["head_neck"] = 0.6
            else:
                issues.append("Head dropping \u2014 align with your body")
                scores["head_neck"] = 0.3

        # ── Legs straight (weight 1.0) ──
        if knee is not None and knee < KNEE_STRAIGHT:
            issues.append("Straighten legs \u2014 keep knees locked")
            scores["knee"] = 0.5
            joint_feedback["left_knee"] = "warning"
            joint_feedback["right_knee"] = "warning"

        # ── Fatigue drift (weight 1.1) ──
        if drift is not None:
            if drift >= DRIFT_STOP_DEG:
                if not self._drift_stop_warned:
                    self._drift_stop_warned = True
                issues.append("Form breaking down \u2014 consider resting")
                scores["drift"] = 0.3
            elif drift >= DRIFT_WARN_DEG:
                issues.append("Form starting to drift \u2014 stay tight")
                scores["drift"] = 0.7

        if not scores:
            return 0.0, joint_feedback, issues

        WEIGHTS = {
            "hip_sag": 2.0, "body_line": 1.5, "hip_rotation": 1.5,
            "elbow_position": 1.3, "head_neck": 1.0, "knee": 1.0,
            "drift": 1.1,
        }
        weighted_sum = sum(scores[k] * WEIGHTS.get(k, 1.0) for k in scores)
        weight_total = sum(WEIGHTS.get(k, 1.0) for k in scores)
        total = weighted_sum / weight_total

        return max(0.0, min(1.0, total)), joint_feedback, issues
