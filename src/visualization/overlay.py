"""
Skeleton overlay renderer for pose visualization.
Draws landmarks, connections, angle mesh zones, and feedback text on BGR frames.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..pose_estimation.base import PoseResult
from ..classification.base import ClassificationResult
from ..feedback.feedback_engine import FeedbackEngine, COLORS
from ..utils.constants import (
    SKELETON_CONNECTIONS, NUM_LANDMARKS, MIN_VISIBILITY,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    JOINT_ANGLES,
)


# Default drawing settings
LANDMARK_RADIUS = 5
CONNECTION_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# Mesh zone settings
MESH_HALF_WIDTH = 14
MESH_SUBDIVISIONS = 3      # reduced from 5 for speed
MESH_ALPHA = 0.30           # single blend pass


def draw_skeleton(
    frame: np.ndarray,
    pose: PoseResult,
    joint_colors: Dict[int, Tuple[int, int, int]] = None,
    skeleton_colors=None,
) -> np.ndarray:
    """Draw pose landmarks and skeleton connections on a BGR frame."""
    h, w = frame.shape[:2]
    default_color = COLORS["neutral"]

    if skeleton_colors:
        for idx_a, idx_b, color in skeleton_colors:
            if not (pose.is_visible(idx_a, MIN_VISIBILITY) and pose.is_visible(idx_b, MIN_VISIBILITY)):
                continue
            pt_a = _to_pixel(pose.landmarks[idx_a], w, h)
            pt_b = _to_pixel(pose.landmarks[idx_b], w, h)
            cv2.line(frame, pt_a, pt_b, color, CONNECTION_THICKNESS)
    else:
        for idx_a, idx_b in SKELETON_CONNECTIONS:
            if not (pose.is_visible(idx_a, MIN_VISIBILITY) and pose.is_visible(idx_b, MIN_VISIBILITY)):
                continue
            pt_a = _to_pixel(pose.landmarks[idx_a], w, h)
            pt_b = _to_pixel(pose.landmarks[idx_b], w, h)
            color = default_color
            if joint_colors:
                ca = joint_colors.get(idx_a, default_color)
                cb = joint_colors.get(idx_b, default_color)
                if ca == COLORS["incorrect"] or cb == COLORS["incorrect"]:
                    color = COLORS["incorrect"]
                elif ca == COLORS["correct"] and cb == COLORS["correct"]:
                    color = COLORS["correct"]
            cv2.line(frame, pt_a, pt_b, color, CONNECTION_THICKNESS)

    for i in range(NUM_LANDMARKS):
        if not pose.is_visible(i, MIN_VISIBILITY):
            continue
        pt = _to_pixel(pose.landmarks[i], w, h)
        color = joint_colors.get(i, default_color) if joint_colors else default_color
        cv2.circle(frame, pt, LANDMARK_RADIUS, color, -1)
        cv2.circle(frame, pt, LANDMARK_RADIUS, (0, 0, 0), 1)

    return frame


# ── Angle Mesh Zones (optimized: single overlay blend) ───────────────────

def draw_angle_zones(
    frame: np.ndarray,
    pose: PoseResult,
    joint_feedback: Dict[str, str],
) -> np.ndarray:
    """Draw mesh grid zones around monitored angle regions.

    Zero-copy: draws directly on frame with no overlay blending.
    """
    if not joint_feedback:
        return frame

    h, w = frame.shape[:2]
    drawn = set()

    for joint_name, status in joint_feedback.items():
        if joint_name in drawn or joint_name not in JOINT_ANGLES:
            continue
        drawn.add(joint_name)

        idx_a, idx_b, idx_c = JOINT_ANGLES[joint_name]
        if not all(pose.is_visible(i, MIN_VISIBILITY) for i in (idx_a, idx_b, idx_c)):
            continue

        color = COLORS.get(status, COLORS["neutral"])
        fa = np.array(_to_pixel(pose.landmarks[idx_a], w, h), dtype=np.float64)
        fb = np.array(_to_pixel(pose.landmarks[idx_b], w, h), dtype=np.float64)
        fc = np.array(_to_pixel(pose.landmarks[idx_c], w, h), dtype=np.float64)

        # Draw mesh grid along both bone segments
        _draw_bone_mesh(frame, fa, fb, color)
        _draw_bone_mesh(frame, fb, fc, color)

        # Arc at vertex
        _draw_angle_arc(frame, fa, fb, fc, color)

    return frame


def _draw_bone_mesh(
    frame: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    color: Tuple[int, int, int],
) -> None:
    """Draw a mesh grid along a bone segment directly on frame (no copy)."""
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 2.0:
        return

    d = direction / length
    perp = np.array([-d[1], d[0]])
    hw = MESH_HALF_WIDTH
    n = MESH_SUBDIVISIONS

    # Build left/right/center points
    left = []
    right = []
    center = []
    for i in range(n + 1):
        t = i / n
        taper = 1.0 - 0.3 * abs(t - 0.5) * 2
        w = hw * taper
        c = p1 + t * direction
        left.append((int(c[0] - perp[0] * w), int(c[1] - perp[1] * w)))
        right.append((int(c[0] + perp[0] * w), int(c[1] + perp[1] * w)))
        center.append((int(c[0]), int(c[1])))

    # Draw mesh grid lines directly (no fill, no copy)
    for i in range(n):
        cv2.line(frame, left[i], left[i + 1], color, 1)
        cv2.line(frame, right[i], right[i + 1], color, 1)
        cv2.line(frame, center[i], center[i + 1], color, 1)
        cv2.line(frame, left[i], right[i], color, 1)
        cv2.line(frame, left[i], center[i + 1], color, 1)
        cv2.line(frame, right[i], center[i + 1], color, 1)
    cv2.line(frame, left[n], right[n], color, 1)


def _draw_angle_arc(
    frame: np.ndarray,
    pt_a: np.ndarray,
    pt_b: np.ndarray,
    pt_c: np.ndarray,
    color: Tuple[int, int, int],
    radius: int = 18,
) -> None:
    """Draw a small arc at the vertex (pt_b)."""
    ba = pt_a - pt_b
    bc = pt_c - pt_b
    if np.linalg.norm(ba) < 1.0 or np.linalg.norm(bc) < 1.0:
        return

    angle_a = np.degrees(np.arctan2(-ba[1], ba[0]))
    angle_c = np.degrees(np.arctan2(-bc[1], bc[0]))

    start = min(angle_a, angle_c)
    end = max(angle_a, angle_c)
    if end - start > 180:
        start, end = end, start + 360

    cv2.ellipse(frame, (int(pt_b[0]), int(pt_b[1])), (radius, radius),
                0, -end, -start, color, 2)


# ── Feedback Panel & Labels ──────────────────────────────────────────────

def draw_feedback_panel(
    frame: np.ndarray,
    text: str,
    is_correct: bool = True,
    rep_count: int = 0,
    fps: float = 0.0,
) -> np.ndarray:
    """Draw a semi-transparent feedback panel at the top of the frame."""
    h, w = frame.shape[:2]
    panel_height = 80

    panel_color = np.array((0, 80, 0) if is_correct else (0, 0, 80), dtype=np.uint8)
    # Blend directly without copy: darken existing pixels and add panel tint
    roi = frame[0:panel_height, :]
    cv2.addWeighted(roi, 0.4, np.full_like(roi, panel_color), 0.6, 0, roi)

    status = "GOOD FORM" if is_correct else "CHECK FORM"
    status_color = COLORS["correct"] if is_correct else COLORS["incorrect"]
    cv2.putText(frame, status, (10, 30), FONT, 0.8, status_color, 2)

    first_line = (text or "").split("\n")[0][:60]
    cv2.putText(frame, first_line, (10, 55), FONT, 0.5, (255, 255, 255), 1)

    if rep_count > 0:
        rep_text = f"Reps: {rep_count}"
        text_size = cv2.getTextSize(rep_text, FONT, 0.7, 2)[0]
        cv2.putText(frame, rep_text, (w - text_size[0] - 10, 30), FONT, 0.7, (255, 255, 255), 2)

    if fps > 0:
        fps_text = f"{fps:.0f} FPS"
        text_size = cv2.getTextSize(fps_text, FONT, 0.5, 1)[0]
        cv2.putText(frame, fps_text, (w - text_size[0] - 10, 55), FONT, 0.5, (200, 200, 200), 1)

    return frame


def draw_exercise_label(frame: np.ndarray, exercise: str) -> np.ndarray:
    """Draw current exercise name at bottom of frame."""
    h, w = frame.shape[:2]
    label = exercise.replace("_", " ").title()
    text_size = cv2.getTextSize(label, FONT, 0.7, 2)[0]
    x = (w - text_size[0]) // 2
    y = h - 15
    cv2.rectangle(frame, (x - 5, y - text_size[1] - 5), (x + text_size[0] + 5, y + 5), (0, 0, 0), -1)
    cv2.putText(frame, label, (x, y), FONT, 0.7, (255, 255, 255), 2)
    return frame


def _to_pixel(landmark: np.ndarray, w: int, h: int) -> Tuple[int, int]:
    """Convert normalized landmark (x, y) to pixel coordinates."""
    return (int(landmark[0] * w), int(landmark[1] * h))
