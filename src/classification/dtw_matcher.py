"""
Dynamic Time Warping (DTW) template matching for rep-level form comparison.

Each rep produces a (T, 99) sequence of torso-normalized landmarks (see
LandmarkFeatureExtractor.extract_landmarks). DTWMatcher compares that sequence
against pre-built reference templates of "ideal" reps and returns:
  - an overall similarity score in [0, 1]
  - the worst-matching joint group (mnmDTW)

Uses dtaidistance when available; falls back to a numpy DTW otherwise.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from dtaidistance import dtw_ndim
    _HAS_DTAIDISTANCE = True
except Exception:
    _HAS_DTAIDISTANCE = False


# 11 symmetric joint groups (left_idx, right_idx) from LandmarkFeatureExtractor.
# Each entry covers two landmarks; the DTW group spans x/y/z of both = 6 dims.
JOINT_GROUPS: Dict[str, List[int]] = {
    "shoulders": [11, 12],
    "elbows": [13, 14],
    "wrists": [15, 16],
    "hips": [23, 24],
    "knees": [25, 26],
    "ankles": [27, 28],
    "feet": [31, 32],
}


@dataclass
class DTWScore:
    distance: float                          # best DTW distance vs any template
    similarity: float                        # in [0, 1], 1.0 = perfect match
    per_group_distances: Dict[str, float] = field(default_factory=dict)
    worst_joint: Optional[str] = None


def _pairwise_dtw_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """Pure-numpy multivariate DTW (Euclidean cost, no window).

    a: (T_a, D), b: (T_b, D). Returns the accumulated distance normalized
    by path length (average per-step distance).
    """
    T_a, T_b = a.shape[0], b.shape[0]
    if T_a == 0 or T_b == 0:
        return float("inf")
    # Cost matrix: ||a_i - b_j||_2
    diffs = a[:, None, :] - b[None, :, :]
    cost = np.sqrt(np.einsum("ijk,ijk->ij", diffs, diffs))
    acc = np.full((T_a + 1, T_b + 1), np.inf, dtype=np.float64)
    acc[0, 0] = 0.0
    for i in range(1, T_a + 1):
        for j in range(1, T_b + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(
                acc[i - 1, j],
                acc[i, j - 1],
                acc[i - 1, j - 1],
            )
    path_len = T_a + T_b
    return float(acc[T_a, T_b] / max(path_len, 1))


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """DTW distance between two multivariate sequences (T, D).

    Prefers dtaidistance (C, fast); falls back to numpy. Result is roughly
    scale-comparable in both paths: average per-step Euclidean cost.
    """
    if a.size == 0 or b.size == 0:
        return float("inf")
    if _HAS_DTAIDISTANCE:
        try:
            a32 = np.ascontiguousarray(a, dtype=np.float64)
            b32 = np.ascontiguousarray(b, dtype=np.float64)
            raw = dtw_ndim.distance(a32, b32)
            # dtaidistance returns total path cost; normalize by rough path length
            return float(raw / max(len(a) + len(b), 1))
        except Exception as e:
            logger.warning(f"dtaidistance failed, falling back to numpy DTW: {e}")
    return _pairwise_dtw_numpy(a, b)


def _similarity_from_distance(distance: float, scale: float = 0.15) -> float:
    """Map a DTW distance to a similarity in [0, 1] via exp decay.

    `scale` controls how quickly the similarity drops with distance. 0.15 was
    picked empirically to put typical good-match distances near 0.85 and clear
    mismatches below 0.4 on torso-normalized MediaPipe landmarks.
    """
    if not np.isfinite(distance):
        return 0.0
    return float(np.exp(-distance / max(scale, 1e-6)))


class DTWMatcher:
    """Rep-level DTW comparator for one or more exercises.

    Templates are loaded lazily from a directory. Each exercise's templates
    file is an .npz with:
        templates: object array of (T_i, 99) float32 sequences, one per template
        lengths:   (N,) int32 (kept for downstream tooling)

    Use `compare(exercise, rep_sequence)` to score a (T, 99) rep.
    """

    def __init__(self, templates_dir: str):
        self.templates_dir = templates_dir
        self._templates: Dict[str, List[np.ndarray]] = {}

    def has_template(self, exercise: str) -> bool:
        if exercise in self._templates:
            return True
        self._load_exercise(exercise)
        return exercise in self._templates and len(self._templates[exercise]) > 0

    def _load_exercise(self, exercise: str) -> None:
        path = os.path.join(self.templates_dir, f"{exercise}.npz")
        if not os.path.exists(path):
            self._templates[exercise] = []
            return
        try:
            data = np.load(path, allow_pickle=True)
            raw = data["templates"]
            templates = [np.asarray(t, dtype=np.float32) for t in raw]
            templates = [t for t in templates if t.ndim == 2 and t.shape[1] == 99 and t.shape[0] >= 3]
            self._templates[exercise] = templates
            logger.info(f"DTWMatcher loaded {len(templates)} templates for {exercise}")
        except Exception as e:
            logger.warning(f"DTWMatcher failed to load {path}: {e}")
            self._templates[exercise] = []

    def compare(self, exercise: str, rep_sequence: np.ndarray) -> Optional[DTWScore]:
        """Compare rep_sequence (T, 99) against templates for `exercise`.

        Returns None if no templates are loaded for the exercise, or if the
        sequence is too short to be meaningful.
        """
        if rep_sequence is None or rep_sequence.ndim != 2 or rep_sequence.shape[0] < 3:
            return None
        if not self.has_template(exercise):
            return None

        templates = self._templates[exercise]
        rep = np.ascontiguousarray(rep_sequence, dtype=np.float32)

        # Whole-body DTW: pick best-matching template
        best_dist = float("inf")
        best_idx = -1
        for i, tmpl in enumerate(templates):
            d = _dtw_distance(rep, tmpl)
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx < 0:
            return None

        best_template = templates[best_idx]
        similarity = _similarity_from_distance(best_dist)

        # mnmDTW: per-joint-group distances against the best-matching template
        per_group: Dict[str, float] = {}
        worst_joint: Optional[str] = None
        worst_score = float("inf")
        for group_name, indices in JOINT_GROUPS.items():
            # Extract x,y,z for each landmark in the group: shape (T, len(indices)*3)
            col_slices = []
            for idx in indices:
                col_slices.extend([idx * 3, idx * 3 + 1, idx * 3 + 2])
            a_sub = rep[:, col_slices]
            b_sub = best_template[:, col_slices]
            # If either sub-sequence is entirely zero (fully occluded), skip
            if not np.any(a_sub) or not np.any(b_sub):
                continue
            d = _dtw_distance(a_sub, b_sub)
            per_group[group_name] = d
            group_sim = _similarity_from_distance(d)
            if group_sim < worst_score:
                worst_score = group_sim
                worst_joint = group_name

        return DTWScore(
            distance=best_dist,
            similarity=similarity,
            per_group_distances=per_group,
            worst_joint=worst_joint if worst_score < 0.6 else None,
        )
