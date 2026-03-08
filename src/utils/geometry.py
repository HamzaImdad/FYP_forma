"""
Pure math functions for angle calculation, distances, and alignment checks.
"""

from typing import Optional

import numpy as np


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
    """
    Calculate the angle at vertex b formed by points a-b-c.
    Returns angle in degrees (0-180), or None if points are too close together.
    """
    ba = a - b
    bc = c - b
    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)
    if ba_norm < 1e-6 or bc_norm < 1e-6:
        return None
    cosine = np.dot(ba, bc) / (ba_norm * bc_norm)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(a - b))


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Midpoint between two points."""
    return (a + b) / 2.0


def vertical_angle(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """
    Angle of segment a->b relative to the vertical axis.
    Returns degrees: 0 = perfectly vertical, 90 = horizontal.
    Returns None if points are too close together.
    In world coordinates, y-axis is vertical (negative = up).
    """
    diff = b - a
    diff_norm = np.linalg.norm(diff)
    if diff_norm < 1e-6:
        return None
    vertical = np.array([0, -1, 0]) if len(diff) == 3 else np.array([0, -1])
    cosine = np.dot(diff, vertical) / diff_norm
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def lateral_symmetry(left_val: float, right_val: float) -> float:
    """
    Symmetry ratio between left and right side values.
    Returns 0-1 where 1 = perfectly symmetric.
    """
    max_val = max(abs(left_val), abs(right_val), 1e-8)
    return 1.0 - abs(left_val - right_val) / max_val
