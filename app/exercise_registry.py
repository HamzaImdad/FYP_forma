"""
FORMA exercise family registry — single source of truth for how the 10
supported exercises are tracked.

Three families drive every downstream decision (goal calculators, plan-day
auto-complete, Custom Builder UI, Plan Architect tool schemas, session
summary screens):

  - rep_count  : progress = more reps per set     (pushup, pullup, lunge,
                                                    bicep_curl, tricep_dip)
  - weighted   : progress = heavier weight        (squat [weight optional],
                                                    bench_press, deadlift,
                                                    overhead_press)
  - time_hold  : progress = longer hold duration  (plank)

The mirror at app/web/src/lib/exerciseRegistry.ts MUST stay in sync — a
parity test in tests/test_exercise_registry_parity.py enforces that.
"""

from __future__ import annotations

import os
from typing import Dict, Literal, TypedDict

Family = Literal["rep_count", "weighted", "time_hold"]


class ExerciseMeta(TypedDict):
    family: Family
    default_sets: int
    weight_optional: bool


EXERCISE_FAMILIES: Dict[str, ExerciseMeta] = {
    # rep_count — bodyweight, progress = more reps per set
    "pushup":         {"family": "rep_count", "default_sets": 3, "weight_optional": False},
    "pullup":         {"family": "rep_count", "default_sets": 3, "weight_optional": False},
    "bicep_curl":     {"family": "rep_count", "default_sets": 3, "weight_optional": False},
    "tricep_dip":     {"family": "rep_count", "default_sets": 3, "weight_optional": False},
    "lunge":          {"family": "rep_count", "default_sets": 3, "weight_optional": False},
    # weighted — progress = heavier weight for same/similar reps.
    # squat is `weight_optional`: target_weight_kg == 0 means bodyweight.
    "squat":          {"family": "weighted",  "default_sets": 5, "weight_optional": True},
    "bench_press":    {"family": "weighted",  "default_sets": 5, "weight_optional": False},
    "deadlift":       {"family": "weighted",  "default_sets": 5, "weight_optional": False},
    "overhead_press": {"family": "weighted",  "default_sets": 5, "weight_optional": False},
    # time_hold — progress = longer hold
    "plank":          {"family": "time_hold", "default_sets": 1, "weight_optional": False},
    # ── New exercises ──
    "crunch":         {"family": "rep_count",  "default_sets": 3, "weight_optional": False},
    "lateral_raise":  {"family": "rep_count",  "default_sets": 3, "weight_optional": False},
    "side_plank":     {"family": "time_hold",  "default_sets": 3, "weight_optional": False},
}


def family_of(exercise: str) -> Family:
    """Return the family of a known exercise. Raises KeyError if unknown."""
    meta = EXERCISE_FAMILIES.get(exercise)
    if not meta:
        raise KeyError(f"unknown exercise {exercise!r}")
    return meta["family"]


def is_weight_optional(exercise: str) -> bool:
    """True for weighted exercises where target_weight_kg == 0 is legal
    (currently only squat — bodyweight mode)."""
    meta = EXERCISE_FAMILIES.get(exercise)
    return bool(meta and meta.get("weight_optional"))


def default_sets(exercise: str) -> int:
    meta = EXERCISE_FAMILIES.get(exercise)
    if not meta:
        raise KeyError(f"unknown exercise {exercise!r}")
    return meta["default_sets"]


# ── Form-score thresholds (0–1 scale) ────────────────────────────────────
# GOOD_REP_THRESHOLD: cutoff for "this rep counts" — used by volume goals,
#   plan-day auto-complete, dashboard "good reps" display.
# CLEAN_REP_THRESHOLD: higher bar for the Clean Machine badge.
#
# GOOD_REP_THRESHOLD is env-overridable so we can flip back to the legacy
# 0.7 behavior without a redeploy if the 0.6 rollout surfaces regressions.
def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


GOOD_REP_THRESHOLD: float = _env_float("FORMA_GOOD_REP_THRESHOLD", 0.6)
CLEAN_REP_THRESHOLD: float = _env_float("FORMA_CLEAN_REP_THRESHOLD", 0.8)
