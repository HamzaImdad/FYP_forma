"""
Parity tests for the Python + TypeScript exercise-family registries.

We intentionally don't run the TS compiler here — we parse the .ts file as
text and pull out the EXERCISE_FAMILIES literal. This keeps the test
lightweight and dependency-free, and still catches every realistic drift
(added exercise on one side, flipped family, mismatched default_sets,
weight_optional disagreement, threshold constants out of sync).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.exercise_registry import (
    CLEAN_REP_THRESHOLD,
    EXERCISE_FAMILIES,
    GOOD_REP_THRESHOLD,
    family_of,
    is_weight_optional,
)
from src.utils.constants import EXERCISES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TS_REGISTRY = PROJECT_ROOT / "app" / "web" / "src" / "lib" / "exerciseRegistry.ts"


def _parse_ts_registry(text: str) -> dict:
    """Extract EXERCISE_FAMILIES as a plain dict from the TS source.

    Matches entries shaped like:
        pushup:  { family: "rep_count", default_sets: 3, weight_optional: false },
    """
    entry_re = re.compile(
        r"(?P<name>[a-z_]+)\s*:\s*\{\s*"
        r"family\s*:\s*\"(?P<family>rep_count|weighted|time_hold)\"\s*,\s*"
        r"default_sets\s*:\s*(?P<sets>\d+)\s*,\s*"
        r"weight_optional\s*:\s*(?P<wo>true|false)\s*,?\s*\}"
    )
    out: dict = {}
    for m in entry_re.finditer(text):
        out[m.group("name")] = {
            "family": m.group("family"),
            "default_sets": int(m.group("sets")),
            "weight_optional": m.group("wo") == "true",
        }
    return out


def _parse_ts_constant(text: str, name: str) -> float:
    m = re.search(rf"export\s+const\s+{re.escape(name)}\s*=\s*([0-9.]+)\s*;", text)
    if not m:
        raise AssertionError(f"{name} not found in TS registry")
    return float(m.group(1))


@pytest.fixture(scope="module")
def ts_text() -> str:
    assert TS_REGISTRY.exists(), f"TS registry missing: {TS_REGISTRY}"
    return TS_REGISTRY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ts_registry(ts_text: str) -> dict:
    reg = _parse_ts_registry(ts_text)
    assert reg, "TS registry parsed as empty — check the regex / file shape"
    return reg


def test_python_registry_covers_all_exercises():
    """Every exercise in src/utils/constants.EXERCISES has a registry entry."""
    missing = set(EXERCISES) - set(EXERCISE_FAMILIES.keys())
    extra = set(EXERCISE_FAMILIES.keys()) - set(EXERCISES)
    assert not missing, f"missing from Python registry: {sorted(missing)}"
    assert not extra, f"unknown exercise in Python registry: {sorted(extra)}"


def test_ts_registry_matches_python_keys(ts_registry: dict):
    missing_in_ts = set(EXERCISE_FAMILIES) - set(ts_registry)
    missing_in_py = set(ts_registry) - set(EXERCISE_FAMILIES)
    assert not missing_in_ts, f"missing in TS: {sorted(missing_in_ts)}"
    assert not missing_in_py, f"missing in Python: {sorted(missing_in_py)}"


def test_ts_registry_values_match_python(ts_registry: dict):
    for name, py_meta in EXERCISE_FAMILIES.items():
        ts_meta = ts_registry[name]
        assert py_meta["family"] == ts_meta["family"], (
            f"{name}: family drift — py={py_meta['family']} ts={ts_meta['family']}"
        )
        assert py_meta["default_sets"] == ts_meta["default_sets"], (
            f"{name}: default_sets drift — py={py_meta['default_sets']} "
            f"ts={ts_meta['default_sets']}"
        )
        assert py_meta["weight_optional"] == ts_meta["weight_optional"], (
            f"{name}: weight_optional drift"
        )


def test_thresholds_match(ts_text: str):
    ts_good = _parse_ts_constant(ts_text, "GOOD_REP_THRESHOLD")
    ts_clean = _parse_ts_constant(ts_text, "CLEAN_REP_THRESHOLD")
    # Compare against the Python *module defaults*, not the env-overridden
    # runtime values — we're enforcing source-of-truth parity, not runtime.
    from app import exercise_registry as reg

    # The python constants may be env-overridden in dev; re-read defaults
    # by parsing the Python file the same way we parse TS.
    py_text = Path(reg.__file__).read_text(encoding="utf-8")
    py_good = float(
        re.search(r'"FORMA_GOOD_REP_THRESHOLD",\s*([0-9.]+)\)', py_text).group(1)
    )
    py_clean = float(
        re.search(r'"FORMA_CLEAN_REP_THRESHOLD",\s*([0-9.]+)\)', py_text).group(1)
    )
    assert ts_good == py_good, f"GOOD_REP_THRESHOLD drift: py={py_good} ts={ts_good}"
    assert ts_clean == py_clean, (
        f"CLEAN_REP_THRESHOLD drift: py={py_clean} ts={ts_clean}"
    )
    # And confirm the runtime values are sane floats in [0, 1].
    assert 0.0 < GOOD_REP_THRESHOLD <= 1.0
    assert 0.0 < CLEAN_REP_THRESHOLD <= 1.0
    assert GOOD_REP_THRESHOLD <= CLEAN_REP_THRESHOLD, (
        "good threshold must be <= clean threshold"
    )


def test_family_of_and_helpers():
    assert family_of("pushup") == "rep_count"
    assert family_of("deadlift") == "weighted"
    assert family_of("plank") == "time_hold"
    with pytest.raises(KeyError):
        family_of("burpee")
    # squat is the only weight-optional exercise
    assert is_weight_optional("squat") is True
    assert is_weight_optional("deadlift") is False
    assert is_weight_optional("pushup") is False
