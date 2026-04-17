"""
Phase 2 goal-engine tests — exercises the behavior shifts that happen
when FORMA_PHASE2_ENABLED is true:

  • _calc_volume sums good_reps, not total_reps
  • _calc_duration gates on avg_form_score
  • _calc_strength returns MAX(sets.weight_kg) only when clean-rep count
    in that set >= target_reps
  • auto-archive flips 8-day-old completed goals to 'archived'
  • MAX_ACTIVE_GOALS is 20 and archived goals don't count

All tests use a file-backed sqlite DB in a temp dir so we can seed rows
with arbitrary dates.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Force phase 2 on for the module — belt-and-braces even though the
# default is True. Set before importing the module-under-test so the
# module-level constant picks it up.
os.environ["FORMA_PHASE2_ENABLED"] = "1"

from app import settings  # noqa: E402
from app import goal_engine  # noqa: E402
from app.database import ExerVisionDB  # noqa: E402
from app.exercise_registry import GOOD_REP_THRESHOLD  # noqa: E402


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """Fresh DB per test, writing to a tmp file so we don't touch prod."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("FORMA_DB_PATH", str(db_path))
    # ExerVisionDB reads FORMA_DB_PATH at construction time.
    d = ExerVisionDB(str(db_path))
    # Seed a user so foreign keys resolve.
    conn = d._get_conn()
    try:
        conn.execute(
            """INSERT INTO users (id, email, password_hash, display_name, created_at)
               VALUES (1, 't@t.t', 'x', 'T', ?)""",
            (datetime.now().isoformat(),),
        )
        conn.commit()
    finally:
        conn.close()
    return d


def _seed_session(
    db,
    *,
    user_id: int,
    exercise: str,
    date_iso: str,
    total_reps: int = 0,
    good_reps: int = 0,
    avg_form_score: float = 0.0,
    duration_sec: float = 0.0,
    weight_kg: float | None = None,
) -> int:
    conn = db._get_conn()
    try:
        cur = conn.execute(
            """INSERT INTO sessions
                   (exercise, classifier, date, duration_sec, total_reps,
                    good_reps, avg_form_score, weight_kg, user_id)
               VALUES (?, 'rule_based', ?, ?, ?, ?, ?, ?, ?)""",
            (
                exercise, date_iso, duration_sec,
                total_reps, good_reps, avg_form_score, weight_kg, user_id,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def _seed_set(
    db,
    *,
    session_id: int,
    set_num: int,
    reps_count: int,
    weight_kg: float | None = None,
) -> None:
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


def _seed_reps(
    db,
    *,
    session_id: int,
    set_num: int,
    form_scores: list[float],
) -> None:
    conn = db._get_conn()
    try:
        for i, s in enumerate(form_scores, start=1):
            conn.execute(
                """INSERT INTO reps
                       (session_id, rep_num, form_score, quality, issues,
                        duration, set_num)
                   VALUES (?, ?, ?, 'moderate', '[]', 0, ?)""",
                (session_id, i, s, set_num),
            )
        conn.commit()
    finally:
        conn.close()


# ── _calc_volume reads good_reps ──────────────────────────────────────

def test_volume_goal_counts_good_reps_not_total(db):
    today = datetime.now().isoformat()
    _seed_session(
        db, user_id=1, exercise="pushup", date_iso=today,
        total_reps=30, good_reps=20,
    )
    created = (datetime.now() - timedelta(hours=1)).isoformat(timespec="seconds")
    conn = db._get_conn()
    try:
        cur = conn.execute(
            """INSERT INTO goals
                   (user_id, title, goal_type, target_value, current_value,
                    unit, status, created_at, exercise, period)
               VALUES (1, 'Pushup 50', 'volume', 50, 0, 'reps', 'active', ?, 'pushup', 'once')""",
            (created,),
        )
        goal_id = int(cur.lastrowid)
        conn.commit()
    finally:
        conn.close()
    g = goal_engine.recompute_goal(goal_id, user_id=1, db=db)
    assert g is not None
    # Counts good_reps (20), not total_reps (30).
    assert g.current_value == 20


# ── _calc_duration gates on avg_form_score ────────────────────────────

def test_duration_goal_ignores_low_form_sessions(db):
    today = datetime.now().isoformat()
    # Two plank holds: one below threshold (ignored), one above.
    _seed_session(
        db, user_id=1, exercise="plank", date_iso=today,
        duration_sec=90, avg_form_score=GOOD_REP_THRESHOLD - 0.1,
    )
    _seed_session(
        db, user_id=1, exercise="plank", date_iso=today,
        duration_sec=60, avg_form_score=GOOD_REP_THRESHOLD + 0.1,
    )
    created = (datetime.now() - timedelta(hours=1)).isoformat(timespec="seconds")
    conn = db._get_conn()
    try:
        cur = conn.execute(
            """INSERT INTO goals
                   (user_id, title, goal_type, target_value, current_value,
                    unit, status, created_at, exercise, period)
               VALUES (1, 'Plank 2min', 'duration', 120, 0, 'seconds', 'active', ?, 'plank', 'once')""",
            (created,),
        )
        goal_id = int(cur.lastrowid)
        conn.commit()
    finally:
        conn.close()
    g = goal_engine.recompute_goal(goal_id, user_id=1, db=db)
    assert g is not None
    # Only the clean 60s hold counts.
    assert g.current_value == 60


# ── _calc_strength ────────────────────────────────────────────────────

def test_strength_goal_needs_clean_reps_at_weight(db):
    today = datetime.now().isoformat()
    sid = _seed_session(
        db, user_id=1, exercise="deadlift", date_iso=today,
        total_reps=10, good_reps=7, weight_kg=75,
    )
    # Set 1: 75kg, 3 reps all clean — qualifies.
    _seed_set(db, session_id=sid, set_num=1, reps_count=3, weight_kg=75)
    _seed_reps(db, session_id=sid, set_num=1, form_scores=[0.85, 0.82, 0.78])
    # Set 2: 80kg, 3 reps but only 1 clean — does NOT qualify.
    _seed_set(db, session_id=sid, set_num=2, reps_count=3, weight_kg=80)
    _seed_reps(db, session_id=sid, set_num=2, form_scores=[0.85, 0.4, 0.3])

    created = (datetime.now() - timedelta(hours=1)).isoformat(timespec="seconds")
    conn = db._get_conn()
    try:
        cur = conn.execute(
            """INSERT INTO goals
                   (user_id, title, goal_type, target_value, current_value,
                    unit, status, created_at, exercise, target_reps, period)
               VALUES (1, 'Deadlift 90kg x3', 'strength', 90, 0, 'kg', 'active', ?, 'deadlift', 3, 'once')""",
            (created,),
        )
        goal_id = int(cur.lastrowid)
        conn.commit()
    finally:
        conn.close()
    g = goal_engine.recompute_goal(goal_id, user_id=1, db=db)
    assert g is not None
    # Set 2 had only 1 clean rep < target_reps=3, so it fails the filter.
    # Set 1 (75kg, 3 clean reps) qualifies → MAX(weight_kg)=75.
    assert g.current_value == 75


def test_strength_goal_without_target_reps_returns_zero(db):
    today = datetime.now().isoformat()
    sid = _seed_session(
        db, user_id=1, exercise="bench_press", date_iso=today,
        total_reps=3, good_reps=3, weight_kg=60,
    )
    _seed_set(db, session_id=sid, set_num=1, reps_count=3, weight_kg=60)
    _seed_reps(db, session_id=sid, set_num=1, form_scores=[0.9, 0.9, 0.9])

    created = (datetime.now() - timedelta(hours=1)).isoformat(timespec="seconds")
    conn = db._get_conn()
    try:
        cur = conn.execute(
            """INSERT INTO goals
                   (user_id, title, goal_type, target_value, current_value,
                    unit, status, created_at, exercise, target_reps, period)
               VALUES (1, 'Bench 70', 'strength', 70, 0, 'kg', 'active', ?, 'bench_press', NULL, 'once')""",
            (created,),
        )
        goal_id = int(cur.lastrowid)
        conn.commit()
    finally:
        conn.close()
    g = goal_engine.recompute_goal(goal_id, user_id=1, db=db)
    assert g is not None
    assert g.current_value == 0


def test_create_strength_goal_rejects_missing_target_reps(db):
    with pytest.raises(goal_engine.InvalidGoal):
        goal_engine.create_goal(
            1,
            title="Deadlift",
            goal_type="strength",
            target_value=90,
            unit="kg",
            exercise="deadlift",
            period="once",
            db=db,
        )


# ── auto-archive ──────────────────────────────────────────────────────

def test_stale_completed_goals_get_archived_after_7_days(db):
    eight_days_ago = (
        datetime.now() - timedelta(days=8)
    ).isoformat(timespec="seconds")
    six_days_ago = (
        datetime.now() - timedelta(days=6)
    ).isoformat(timespec="seconds")
    long_ago = (datetime.now() - timedelta(days=30)).isoformat(timespec="seconds")
    conn = db._get_conn()
    try:
        conn.execute(
            """INSERT INTO goals (id, user_id, title, goal_type, target_value,
                                  current_value, unit, status, created_at,
                                  completed_at, period)
               VALUES (100, 1, 'Old win', 'volume', 10, 10, 'reps',
                       'completed', ?, ?, 'once')""",
            (long_ago, eight_days_ago),
        )
        conn.execute(
            """INSERT INTO goals (id, user_id, title, goal_type, target_value,
                                  current_value, unit, status, created_at,
                                  completed_at, period)
               VALUES (101, 1, 'Recent win', 'volume', 10, 10, 'reps',
                       'completed', ?, ?, 'once')""",
            (long_ago, six_days_ago),
        )
        conn.commit()
    finally:
        conn.close()

    goal_engine.recompute_all_goals(1, db=db)

    conn = db._get_conn()
    try:
        rows = {
            r["id"]: r["status"]
            for r in conn.execute(
                "SELECT id, status FROM goals WHERE user_id = 1"
            ).fetchall()
        }
    finally:
        conn.close()
    assert rows[100] == "archived"  # 8-day-old flipped
    assert rows[101] == "completed"  # 6-day-old kept


# ── cap = 20, archived excluded ───────────────────────────────────────

def test_max_active_goals_is_20_and_archived_excluded(db):
    assert goal_engine.MAX_ACTIVE_GOALS == 20
    # 20 active goals + 1 archived → still "20 active", cap check
    # compares against 20.
    now = datetime.now().isoformat(timespec="seconds")
    conn = db._get_conn()
    try:
        for i in range(20):
            conn.execute(
                """INSERT INTO goals (user_id, title, goal_type, target_value,
                                      unit, status, created_at)
                   VALUES (1, ?, 'volume', 10, 'reps', 'active', ?)""",
                (f"g{i}", now),
            )
        conn.execute(
            """INSERT INTO goals (user_id, title, goal_type, target_value,
                                  unit, status, created_at, completed_at)
               VALUES (1, 'archived', 'volume', 10, 'reps', 'archived', ?, ?)""",
            (now, now),
        )
        conn.commit()
    finally:
        conn.close()
    # 20 active → 21st should bounce.
    with pytest.raises(goal_engine.GoalCapExceeded):
        goal_engine.create_goal(
            1,
            title="too many",
            goal_type="volume",
            target_value=10,
            unit="reps",
            period="once",
            db=db,
        )
