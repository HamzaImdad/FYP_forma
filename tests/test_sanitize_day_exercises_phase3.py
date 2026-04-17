"""
Phase 3 sanitizer tests — family-aware validation of plan day exercises.

Exercises the branching in sanitize_day_exercises:
  • rep_count  → required target_reps + target_sets (weight/duration stripped)
  • weighted   → required target_weight_kg + target_reps + target_sets
                 (squat allows weight=0, other weighted drop on missing)
  • time_hold  → required target_duration_sec (reps/weight stripped)
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest

os.environ["FORMA_PHASE3_ENABLED"] = "1"

from app.database import ExerVisionDB  # noqa: E402
from app.chat_tools import sanitize_day_exercises  # noqa: E402


@pytest.fixture()
def db(tmp_path):
    d = ExerVisionDB(str(tmp_path / "test.db"))
    conn = d._get_conn()
    try:
        conn.execute(
            "INSERT INTO users (id, email, password_hash, display_name, created_at)"
            " VALUES (1, 't@t.t', 'x', 'T', ?)",
            (datetime.now().isoformat(),),
        )
        conn.commit()
    finally:
        conn.close()
    return d


# ── rep_count family ──────────────────────────────────────────────────

def test_rep_count_keeps_reps_sets_strips_weight(db):
    out, _warn = sanitize_day_exercises(
        [{"exercise": "pushup", "target_reps": 20, "target_sets": 3,
          "target_weight_kg": 50, "target_duration_sec": 99, "notes": "hi"}],
        user_id=1, db=db,
    )
    assert len(out) == 1
    e = out[0]
    assert e["exercise"] == "pushup"
    assert e["family"] == "rep_count"
    assert e["target_reps"] == 20
    assert e["target_sets"] == 3
    assert "target_weight_kg" not in e
    assert "target_duration_sec" not in e


def test_rep_count_clamps_reps_over_cap(db):
    out, warn = sanitize_day_exercises(
        [{"exercise": "pushup", "target_reps": 10_000, "target_sets": 3}],
        user_id=1, db=db,
    )
    assert len(out) == 1
    # Default cap for pushup is 60 (per _DEFAULT_REP_CEILING) when no user
    # history exists; anything bigger gets clamped.
    assert out[0]["target_reps"] <= 60
    assert any("clamped" in w for w in warn)


# ── weighted family ───────────────────────────────────────────────────

def test_weighted_requires_weight(db):
    # Deadlift with no target_weight_kg → dropped.
    out, warn = sanitize_day_exercises(
        [{"exercise": "deadlift", "target_reps": 3, "target_sets": 5}],
        user_id=1, db=db,
    )
    assert out == []
    assert any("missing target_weight_kg" in w for w in warn)


def test_weighted_keeps_all_three_fields(db):
    out, _warn = sanitize_day_exercises(
        [{"exercise": "deadlift", "target_reps": 3, "target_sets": 5,
          "target_weight_kg": 80, "target_duration_sec": 999}],
        user_id=1, db=db,
    )
    assert len(out) == 1
    e = out[0]
    assert e["family"] == "weighted"
    assert e["target_weight_kg"] == 80
    assert e["target_reps"] == 3
    assert e["target_sets"] == 5
    assert "target_duration_sec" not in e


def test_squat_allows_bodyweight(db):
    # target_weight_kg=0 is legal for squat (bodyweight mode).
    out, _warn = sanitize_day_exercises(
        [{"exercise": "squat", "target_reps": 10, "target_sets": 3,
          "target_weight_kg": 0}],
        user_id=1, db=db,
    )
    assert len(out) == 1
    assert out[0]["family"] == "weighted"
    assert out[0]["target_weight_kg"] == 0


def test_squat_keeps_explicit_weight(db):
    out, _warn = sanitize_day_exercises(
        [{"exercise": "squat", "target_reps": 5, "target_sets": 5,
          "target_weight_kg": 60}],
        user_id=1, db=db,
    )
    assert out[0]["target_weight_kg"] == 60


def test_weighted_clamps_oversized_weight(db):
    out, warn = sanitize_day_exercises(
        [{"exercise": "deadlift", "target_reps": 3, "target_sets": 3,
          "target_weight_kg": 9999}],
        user_id=1, db=db,
    )
    assert out[0]["target_weight_kg"] == 500
    assert any("clamped to 500" in w for w in warn)


# ── time_hold family ──────────────────────────────────────────────────

def test_plank_requires_duration(db):
    out, warn = sanitize_day_exercises(
        [{"exercise": "plank", "target_reps": 10, "target_sets": 3}],
        user_id=1, db=db,
    )
    assert out == []
    assert any("missing target_duration_sec" in w for w in warn)


def test_plank_strips_reps_and_weight(db):
    out, _warn = sanitize_day_exercises(
        [{"exercise": "plank", "target_duration_sec": 60,
          "target_reps": 99, "target_weight_kg": 50}],
        user_id=1, db=db,
    )
    assert len(out) == 1
    e = out[0]
    assert e["family"] == "time_hold"
    assert e["target_duration_sec"] == 60
    assert "target_reps" not in e
    assert "target_weight_kg" not in e
    # target_sets defaults to plank's default (1)
    assert e["target_sets"] == 1


def test_plank_clamps_massive_duration(db):
    out, warn = sanitize_day_exercises(
        [{"exercise": "plank", "target_duration_sec": 9999}],
        user_id=1, db=db,
    )
    assert out[0]["target_duration_sec"] == 600
    assert any("clamped to 600" in w for w in warn)


def test_unknown_exercise_dropped(db):
    out, warn = sanitize_day_exercises(
        [{"exercise": "burpee", "target_reps": 10, "target_sets": 3}],
        user_id=1, db=db,
    )
    assert out == []
    assert any("unknown exercise" in w for w in warn)


def test_every_sanitized_row_has_family_field(db):
    mixed = [
        {"exercise": "pushup", "target_reps": 20, "target_sets": 3},
        {"exercise": "deadlift", "target_reps": 3, "target_sets": 5,
         "target_weight_kg": 80},
        {"exercise": "plank", "target_duration_sec": 60},
    ]
    out, _warn = sanitize_day_exercises(mixed, user_id=1, db=db)
    assert [e["family"] for e in out] == ["rep_count", "weighted", "time_hold"]
