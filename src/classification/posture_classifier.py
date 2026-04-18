"""
Rule-based posture classifier.

Outputs a discrete body-position label per frame from MediaPipe BlazePose
landmarks: standing / sitting / kneeling / plank / lying / unknown.

Used by PushUpDetector to drive the outer session FSM
(IDLE / SETUP / ACTIVE / RESTING). Cheap pure-numpy rules, no ML model.
"""

from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, Optional

import numpy as np

from ..feature_extraction.angles import compute_joint_angles
from ..pose_estimation.base import PoseResult
from ..utils.constants import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    MIN_VISIBILITY,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from ..utils.geometry import calculate_angle, vertical_angle


# ──────────────────────────────────────────────────────────────────────────────
# Tunable thresholds (single source of truth — tune from this block)
# ──────────────────────────────────────────────────────────────────────────────

# Torso angle vs vertical: 0° = vertical, 90° = horizontal
TORSO_STANDING_MAX_DEG = 30.0
TORSO_SITTING_MAX_DEG = 45.0
TORSO_KNEELING_MAX_DEG = 45.0
TORSO_PLANK_MIN_DEG = 50.0        # was 60 — be forgiving of off-axis cameras
TORSO_LYING_MIN_DEG = 60.0

# Knee flexion: 180° = leg straight, 90° = deep bend
KNEE_STANDING_MIN = 160.0
KNEE_LYING_MIN = 150.0
KNEE_SITTING_MIN = 70.0
KNEE_SITTING_MAX = 130.0
KNEE_KNEELING_MAX = 100.0

# Body line shoulder-hip-ankle: 0° = perfectly collinear
PLANK_BODY_LINE_MAX_DEG = 40.0    # was 25 — tolerate slight hip sag/pike in FSM gate
PLANK_BODY_LINE_HIGH_CONF_DEG = 20.0

# World-coordinate Y offsets in meters — MediaPipe world: Y axis = image Y,
# so +Y points DOWN (toward ground). Positive Y-delta = point A below point B.
HIP_ABOVE_KNEE_MIN = 0.05         # standing: hip clearly above knee
KNEE_ABOVE_ANKLE_MIN = 0.15       # standing: knee clearly above ankle
HIP_KNEE_SIT_MAX_ABS = 0.15       # sitting: hip near knee level
ANKLE_KNEE_KNEEL_MAX_ABS = 0.15   # kneeling: shins near floor
HIP_ABOVE_KNEE_KNEEL_MIN = 0.02   # kneeling upright: hips just above knees
WRIST_BELOW_SHOULDER_MIN = 0.03   # plank: hands lower than shoulders (loose)
WRIST_ANKLE_GROUND_MAX = 0.50     # was 0.25 — depends on arm length + stance
HIP_BELOW_SHOULDER_MAX_PLANK = 0.25  # was 0.10 — hip can be above or near shoulder
HIP_SHOULDER_LEVEL_MAX_LYING = 0.20

MIN_OVERALL_VISIBILITY = 0.5

# Plank classification per-landmark visibility. 0.7 was originally chosen
# to block a sitting-user phantom-plank (torso_angle ~92°, ankles
# fabricated below hip), but on mobile side-view push-ups the far-side
# knee/ankle bounces 0.4-0.6 — never stable above 0.7, so PLANK never
# fires and the session never enters ACTIVE. Lowered to 0.4; the
# shoulder_above_hip >= 8cm guard below is what actually discriminates
# real planks from the seated-torso phantom case.
PLANK_LANDMARK_VISIBILITY = 0.4

# Real-plank shoulder-hip vertical separation. In a true plank, the
# shoulders float noticeably above the hips (the torso rests on the arms
# above hip level). When MediaPipe hallucinates a horizontal torso from
# a sitting user, the shoulder-y ends up within a few cm of hip-y
# (hip-centered origin). Require at least 8cm of vertical separation.
PLANK_SHOULDER_ABOVE_HIP_MIN = 0.08


# ──────────────────────────────────────────────────────────────────────────────
# Public types
# ──────────────────────────────────────────────────────────────────────────────


class PostureLabel(str, Enum):
    STANDING = "standing"
    SITTING = "sitting"
    KNEELING = "kneeling"
    PLANK = "plank"
    LYING = "lying"
    UNKNOWN = "unknown"


@dataclass
class PostureResult:
    label: PostureLabel
    confidence: float
    features: Dict[str, Optional[float]] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _safe_midpoint(
    pose: PoseResult, left_idx: int, right_idx: int
) -> Optional[np.ndarray]:
    """Average L/R world landmarks. Falls back to whichever side is visible."""
    left_ok = pose.is_visible(left_idx, MIN_VISIBILITY)
    right_ok = pose.is_visible(right_idx, MIN_VISIBILITY)
    if left_ok and right_ok:
        return (pose.get_world_landmark(left_idx) + pose.get_world_landmark(right_idx)) / 2.0
    if left_ok:
        return pose.get_world_landmark(left_idx)
    if right_ok:
        return pose.get_world_landmark(right_idx)
    return None


def _mean_y(pt: Optional[np.ndarray]) -> Optional[float]:
    return None if pt is None else float(pt[1])


# ──────────────────────────────────────────────────────────────────────────────
# Classifier
# ──────────────────────────────────────────────────────────────────────────────


class PostureClassifier:
    """
    Stateless rule classifier + stateful hysteresis smoother.

    Call `update(pose)` once per frame from the detector; it returns a
    smoothed PostureResult that's stable against single-frame noise.
    """

    def __init__(
        self,
        smoothing_window: int = 10,
        hysteresis_frames: int = 5,
    ) -> None:
        self._ring: Deque[PostureLabel] = deque(maxlen=smoothing_window)
        self._stable_label: PostureLabel = PostureLabel.UNKNOWN
        self._pending_label: Optional[PostureLabel] = None
        self._pending_count: int = 0
        self._hysteresis_frames = hysteresis_frames
        self._last_features: Dict[str, Optional[float]] = {}
        self._last_raw_conf: float = 0.0
        self._last_raw_label: PostureLabel = PostureLabel.UNKNOWN

    # ── Public API ────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._ring.clear()
        self._stable_label = PostureLabel.UNKNOWN
        self._pending_label = None
        self._pending_count = 0
        self._last_features = {}
        self._last_raw_conf = 0.0
        self._last_raw_label = PostureLabel.UNKNOWN

    def classify(self, pose: PoseResult) -> PostureResult:
        """Raw single-frame classification (no smoothing)."""
        return self._classify_raw(pose)

    def update(self, pose: PoseResult) -> PostureResult:
        """Smoothed classification — call once per frame."""
        raw = self._classify_raw(pose)
        self._last_features = raw.features
        self._last_raw_conf = raw.confidence
        self._last_raw_label = raw.label
        self._ring.append(raw.label)

        # Bootstrap: pick mode once we have enough samples
        if self._stable_label == PostureLabel.UNKNOWN and len(self._ring) >= 5:
            self._stable_label = self._mode_label()

        # Hysteresis: switch only after N consecutive matching raw labels
        elif raw.label != self._stable_label:
            if raw.label == self._pending_label:
                self._pending_count += 1
            else:
                self._pending_label = raw.label
                self._pending_count = 1
            if self._pending_count >= self._hysteresis_frames:
                self._stable_label = self._pending_label  # type: ignore[assignment]
                self._pending_count = 0
                self._pending_label = None
        else:
            self._pending_count = 0
            self._pending_label = None

        # Confidence weighted by ring agreement with stable label
        agreement = self._ring.count(self._stable_label) / max(len(self._ring), 1)
        weighted_conf = self._last_raw_conf * agreement
        return PostureResult(
            label=self._stable_label,
            confidence=weighted_conf,
            features=self._last_features,
        )

    # ── Internals ─────────────────────────────────────────────────────────

    def _mode_label(self) -> PostureLabel:
        if not self._ring:
            return PostureLabel.UNKNOWN
        most_common = Counter(self._ring).most_common(1)[0][0]
        return most_common

    def _classify_raw(self, pose: PoseResult) -> PostureResult:
        feats: Dict[str, Optional[float]] = {}

        # Visibility guard
        overall_vis = float(np.mean(pose.landmarks[:, 3]))
        feats["overall_visibility"] = overall_vis
        if overall_vis < MIN_OVERALL_VISIBILITY:
            return PostureResult(PostureLabel.UNKNOWN, 0.0, feats)

        # Midpoints (world coords)
        mid_shoulder = _safe_midpoint(pose, LEFT_SHOULDER, RIGHT_SHOULDER)
        mid_hip = _safe_midpoint(pose, LEFT_HIP, RIGHT_HIP)
        mid_knee = _safe_midpoint(pose, LEFT_KNEE, RIGHT_KNEE)
        mid_ankle = _safe_midpoint(pose, LEFT_ANKLE, RIGHT_ANKLE)
        mid_wrist = _safe_midpoint(pose, LEFT_WRIST, RIGHT_WRIST)

        if mid_shoulder is None or mid_hip is None:
            return PostureResult(PostureLabel.UNKNOWN, 0.0, feats)

        shoulder_y = float(mid_shoulder[1])
        hip_y = float(mid_hip[1])
        knee_y = _mean_y(mid_knee)
        ankle_y = _mean_y(mid_ankle)
        wrist_y = _mean_y(mid_wrist)

        feats["shoulder_y"] = shoulder_y
        feats["hip_y"] = hip_y
        feats["knee_y"] = knee_y
        feats["ankle_y"] = ankle_y
        feats["wrist_y"] = wrist_y

        # Torso angle vs vertical
        torso_angle = vertical_angle(mid_shoulder, mid_hip)
        feats["torso_angle_deg"] = torso_angle

        # Knee flexion (mean of L/R)
        joint_angles = compute_joint_angles(pose, use_world=True)
        l_knee = joint_angles.get("left_knee")
        r_knee = joint_angles.get("right_knee")
        knee_vals = [v for v in (l_knee, r_knee) if v is not None]
        knee_flex = float(np.mean(knee_vals)) if knee_vals else None
        feats["knee_flexion_deg"] = knee_flex

        # Body line deviation: 0° = shoulder-hip-ankle collinear
        body_line_dev: Optional[float] = None
        if mid_ankle is not None:
            body_line_angle = calculate_angle(mid_shoulder, mid_hip, mid_ankle)
            if body_line_angle is not None:
                body_line_dev = 180.0 - body_line_angle
        feats["body_line_deviation_deg"] = body_line_dev

        # Plank-specific bools — side-on camera often occludes one wrist,
        # so only require ONE visible wrist (whichever side we can see).
        any_wrist_visible = pose.is_visible(LEFT_WRIST, MIN_VISIBILITY) or pose.is_visible(
            RIGHT_WRIST, MIN_VISIBILITY
        )
        wrist_below_shoulder = (
            wrist_y is not None
            and any_wrist_visible
            and wrist_y > shoulder_y + WRIST_BELOW_SHOULDER_MIN
        )
        # Unknown wrist position → treat as "not ruling plank out"
        wrist_unknown = wrist_y is None or not any_wrist_visible
        wrist_ankle_ground = (
            wrist_y is not None
            and ankle_y is not None
            and abs(wrist_y - ankle_y) < WRIST_ANKLE_GROUND_MAX
        )
        feats["wrist_below_shoulder"] = 1.0 if wrist_below_shoulder else 0.0
        feats["wrist_ankle_ground"] = 1.0 if wrist_ankle_ground else 0.0
        feats["wrist_unknown"] = 1.0 if wrist_unknown else 0.0

        # Standing/sitting/kneeling helper bools
        hip_above_knee = (
            knee_y is not None and hip_y < knee_y - HIP_ABOVE_KNEE_MIN
        )
        hip_above_knee_kneeling = (
            knee_y is not None and hip_y < knee_y - HIP_ABOVE_KNEE_KNEEL_MIN
        )
        hip_near_knee_y = (
            knee_y is not None and abs(hip_y - knee_y) < HIP_KNEE_SIT_MAX_ABS
        )

        # ── Rule 1: PLANK ──────────────────────────────────────────────
        # Primary signals: torso horizontal + shoulder-hip-ankle reasonably
        # straight + hip not dangling below shoulder.
        # Wrist-below-shoulder is used as a BOOST to distinguish plank from
        # lying face-down, but is NOT required — a side-on camera routinely
        # occludes wrists, and during the DOWN phase of a push-up the shoulder
        # drops close to the wrist anyway, so wrist checks are unreliable.
        body_horizontal = (
            torso_angle is not None and torso_angle >= TORSO_PLANK_MIN_DEG
        )
        body_line_ok = (
            body_line_dev is not None and body_line_dev <= PLANK_BODY_LINE_MAX_DEG
        )
        hip_not_dangling = hip_y <= shoulder_y + HIP_BELOW_SHOULDER_MAX_PLANK

        # Full-body visibility gate: a plank is a whole-body pose. If the
        # legs aren't actually visible, MediaPipe hallucinates knee/ankle
        # landmarks and the torso-angle/body-line checks become meaningless
        # — the user sitting at a desk with only their torso in frame can
        # produce torso_angle ~92° and body_line_dev ~5° and look like a
        # perfect plank. Require confident per-landmark visibility on
        # legs (PLANK_LANDMARK_VISIBILITY, stricter than MIN_VISIBILITY).
        knee_visible = pose.is_visible(
            LEFT_KNEE, PLANK_LANDMARK_VISIBILITY
        ) or pose.is_visible(RIGHT_KNEE, PLANK_LANDMARK_VISIBILITY)
        ankle_visible = pose.is_visible(
            LEFT_ANKLE, PLANK_LANDMARK_VISIBILITY
        ) or pose.is_visible(RIGHT_ANKLE, PLANK_LANDMARK_VISIBILITY)
        legs_visible = knee_visible and ankle_visible

        # Geometric sanity: in a real plank the shoulders float above
        # the hips (pushed up by the arms). When MediaPipe hallucinates
        # a horizontal torso from a seated user, shoulder_y sits almost
        # exactly on top of hip_y (world origin). Require at least 8cm
        # of vertical separation to rule that case out.
        shoulder_above_hip = shoulder_y <= hip_y - PLANK_SHOULDER_ABOVE_HIP_MIN

        if (body_horizontal and body_line_ok and hip_not_dangling
                and legs_visible and shoulder_above_hip):
            # Boost confidence if wrist signals also line up; otherwise still
            # report PLANK (the session FSM treats 0.5+ as valid).
            if wrist_below_shoulder and wrist_ankle_ground:
                conf = 0.9 if body_line_dev <= PLANK_BODY_LINE_HIGH_CONF_DEG else 0.8
            elif wrist_unknown:
                conf = 0.7 if body_line_dev <= PLANK_BODY_LINE_HIGH_CONF_DEG else 0.6
            else:
                # Wrist visible but in a non-plank position (above shoulder
                # or far from ankle) — slight penalty but still PLANK.
                conf = 0.6 if body_line_dev <= PLANK_BODY_LINE_HIGH_CONF_DEG else 0.5
            return PostureResult(PostureLabel.PLANK, conf, feats)

        # ── Rule 2: STANDING ───────────────────────────────────────────
        if (
            torso_angle is not None
            and torso_angle <= TORSO_STANDING_MAX_DEG
            and knee_flex is not None
            and knee_flex >= KNEE_STANDING_MIN
            and hip_above_knee
            and ankle_y is not None
            and knee_y is not None
            and knee_y < ankle_y - KNEE_ABOVE_ANKLE_MIN
        ):
            return PostureResult(PostureLabel.STANDING, 0.9, feats)

        # ── Rule 3: KNEELING ───────────────────────────────────────────
        if (
            torso_angle is not None
            and torso_angle <= TORSO_KNEELING_MAX_DEG
            and knee_flex is not None
            and knee_flex < KNEE_KNEELING_MAX
            and hip_above_knee_kneeling
            and ankle_y is not None
            and knee_y is not None
            and abs(ankle_y - knee_y) < ANKLE_KNEE_KNEEL_MAX_ABS
        ):
            return PostureResult(PostureLabel.KNEELING, 0.8, feats)

        # ── Rule 4: SITTING ────────────────────────────────────────────
        if (
            torso_angle is not None
            and torso_angle <= TORSO_SITTING_MAX_DEG
            and knee_flex is not None
            and KNEE_SITTING_MIN <= knee_flex <= KNEE_SITTING_MAX
            and hip_near_knee_y
            and ankle_y is not None
            and knee_y is not None
            and knee_y < ankle_y
        ):
            return PostureResult(PostureLabel.SITTING, 0.8, feats)

        # ── Rule 5: LYING ──────────────────────────────────────────────
        if (
            torso_angle is not None
            and torso_angle >= TORSO_LYING_MIN_DEG
            and knee_flex is not None
            and knee_flex >= KNEE_LYING_MIN
            and abs(hip_y - shoulder_y) < HIP_SHOULDER_LEVEL_MAX_LYING
            and (not wrist_below_shoulder or not wrist_ankle_ground)
        ):
            return PostureResult(PostureLabel.LYING, 0.7, feats)

        return PostureResult(PostureLabel.UNKNOWN, 0.0, feats)
