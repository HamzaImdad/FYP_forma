"""
Phase 2 plan-day auto-complete integration tests.

Exercises the `_exercise_passes` family branching and the
"all exercises must pass" rule added to on_session_complete.

We seed a plan + plan_day with one exercise from each family (pushup +
deadlift + plank), then seed sessions with varying coverage and call
`_exercise_passes` directly for each exercise spec to confirm pass/fail.
The full `on_session_complete` integration would require the socketio
runtime which isn't worth spinning up for these assertions.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

os.environ["FORMA_PHASE2_ENABLED"] = "1"

from app.database import ExerVisionDB  # noqa: E402
from app.exercise_registry import GOOD_REP_THRESHOLD  # noqa: E402
from app.server import _exercise_passes  # noqa: E402


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


def _today_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _today_ts() -> str:
    return datetime.now().isoformat()


def _add_session(db, **kw) -> int:
    conn = db._get_conn()
    try:
        cur = conn.execute(
            """INSERT INTO sessions
                   (exercise, classifier, date, duration_sec, total_reps,
                    good_reps, avg_form_score, weight_kg, user_id)
               VALUES (?, 'rule_based', ?, ?, ?, ?, ?, ?, 1)""",
            (
                kw["exercise"], kw.get("date", _today_ts()),
                kw.get("duration_sec", 0), kw.get("total_reps", 0),
                kw.get("good_reps", 0), kw.get("avg_form_score", 0.0),
                kw.get("weight_kg"),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def _add_set(db, session_id, set_num, reps_count, weight_kg=None) -> None:
    conn = db._get_conn()
    try:
        conn.execute(
            """INSERT INTO sets (session_id, set_num, reps_count, weight_kg)
               VALUES (?, ?, ?, ?)""",
            (session_id, set_num, reps_count, weight_kg),
        )
        conn.commit()
    finally:
        conn.close()


def _add_reps(db, session_id, set_num, form_scores) -> None:
    conn = db._get_conn()
    try:
        for i, s in enumerate(form_scores, start=1):
            conn.execute(
                """INSERT INTO reps (session_id, rep_num, form_score, quality,
                                     issues, duration, set_num)
                   VALUES (?, ?, ?, 'moderate', '[]', 0, ?)""",
                (session_id, i, s, set_num),
            )
        conn.commit()
    finally:
        conn.close()


# ── rep_count family ──────────────────────────────────────────────────

def test_rep_count_passes_when_good_reps_meet_target(db):
    _add_session(db, exercise="pushup", good_reps=75, total_reps=80)
    ok = _exercise_passes(
        db._get_conn(), 1, _today_iso(),
        {"exercise": "pushup", "target_reps": 25, "target_sets": 3},
    )
    assert ok is True


def test_rep_count_fails_when_good_reps_short(db):
    _add_session(db, exercise="pushup", good_reps=40, total_reps=80)
    ok = _exercise_passes(
        db._get_conn(), 1, _today_iso(),
        {"exercise": "pushup", "target_reps": 25, "target_sets": 3},
    )
    # needs 75, only has 40 good — fails even though total was 80
    assert ok is False


# ── weighted family ───────────────────────────────────────────────────

def test_weighted_passes_on_single_clean_set_at_weight(db):
    sid = _add_session(db, exercise="deadlift", weight_kg=80)
    _add_set(db, sid, 1, 3, weight_kg=80)
    _add_reps(db, sid, 1, [0.85, 0.82, 0.78])
    ok = _exercise_passes(
        db._get_conn(), 1, _today_iso(),
        {
            "exercise": "deadlift", "target_weight_kg": 75,
            "target_reps": 3, "target_sets": 5,
        },
    )
    assert ok is True


def test_weighted_fails_when_weight_too_low(db):
    sid = _add_session(db, exercise="deadlift", weight_kg=60)
    _add_set(db, sid, 1, 3, weight_kg=60)
    _add_reps(db, sid, 1, [0.9, 0.9, 0.9])
    ok = _exercise_passes(
        db._get_conn(), 1, _today_iso(),
        {
            "exercise": "deadlift", "target_weight_kg": 75,
            "target_reps": 3, "target_sets": 5,
        },
    )
    assert ok is False


def test_squat_bodyweight_target_zero_accepts_any_weight(db):
    # Squat with target_weight_kg == 0 means "bodyweight mode" — any
    # set with enough clean reps qualifies, even NULL weight.
    sid = _add_session(db, exercise="squat", weight_kg=None)
    _add_set(db, sid, 1, 10, weight_kg=None)
    _add_reps(db, sid, 1, [0.9] * 10)
    ok = _exercise_passes(
        db._get_conn(), 1, _today_iso(),
        {
            "exercise": "squat", "target_weight_kg": 0,
            "target_reps": 10, "target_sets": 3,
        },
    )
    assert ok is True


# ── time_hold family ──────────────────────────────────────────────────

def test_plank_passes_when_clean_hold_meets_target(db):
    _add_session(
        db, exercise="plank", duration_sec=65,
        avg_form_score=GOOD_REP_THRESHOLD + 0.1,
    )
    ok = _exercise_passes(
        db._get_conn(), 1, _today_iso(),
        {"exercise": "plank", "target_duration_sec": 60},
    )
    assert ok is True


def test_plank_fails_when_form_below_threshold(db):
    _add_session(
        db, exercise="plank", duration_sec=90,
        avg_form_score=GOOD_REP_THRESHOLD - 0.1,
    )
    ok = _exercise_passes(
        db._get_conn(), 1, _today_iso(),
        {"exercise": "plank", "target_duration_sec": 60},
    )
    assert ok is False


def test_unknown_exercise_fails_closed(db):
    ok = _exercise_passes(
        db._get_conn(), 1, _today_iso(),
        {"exercise": "burpee", "target_reps": 10, "target_sets": 3},
    )
    assert ok is False
