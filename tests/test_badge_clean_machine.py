"""
Phase 4 test — Clean Machine badge.

Predicate: any completed (non-rest) plan day where every rep across every
session that matches the day's exercises scored >= CLEAN_REP_THRESHOLD.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from app import badge_engine
from app.database import ExerVisionDB
from app.exercise_registry import CLEAN_REP_THRESHOLD


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


def _seed_plan_with_completed_day(db, exercises, day_date):
    conn = db._get_conn()
    try:
        cur = conn.execute(
            """INSERT INTO plans (user_id, title, summary, start_date, end_date,
                                  status, created_at)
               VALUES (1, 'Test', '', ?, ?, 'active', ?)""",
            (day_date, day_date, datetime.now().isoformat()),
        )
        plan_id = int(cur.lastrowid)
        cur = conn.execute(
            """INSERT INTO plan_days (plan_id, day_date, is_rest, exercises,
                                      completed, completed_at)
               VALUES (?, ?, 0, ?, 1, ?)""",
            (
                plan_id, day_date,
                json.dumps([{"exercise": ex} for ex in exercises]),
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        return plan_id, int(cur.lastrowid)
    finally:
        conn.close()


def _seed_session_with_reps(db, exercise, date_iso, form_scores):
    conn = db._get_conn()
    try:
        cur = conn.execute(
            """INSERT INTO sessions
                   (exercise, classifier, date, duration_sec, total_reps,
                    good_reps, avg_form_score, user_id)
               VALUES (?, 'rule_based', ?, 0, ?, ?, ?, 1)""",
            (
                exercise, date_iso, len(form_scores),
                sum(1 for s in form_scores if s >= 0.6),
                sum(form_scores) / len(form_scores) if form_scores else 0,
            ),
        )
        sid = int(cur.lastrowid)
        for i, s in enumerate(form_scores, start=1):
            conn.execute(
                """INSERT INTO reps (session_id, rep_num, form_score, quality,
                                     issues, duration, set_num)
                   VALUES (?, ?, ?, 'moderate', '[]', 0, 1)""",
                (sid, i, s),
            )
        conn.commit()
        return sid
    finally:
        conn.close()


def test_awarded_when_every_rep_above_threshold(db):
    day_date = "2026-04-16"
    _seed_plan_with_completed_day(db, ["pushup"], day_date)
    # 3 reps, all above CLEAN_REP_THRESHOLD (0.8)
    _seed_session_with_reps(db, "pushup", day_date, [0.85, 0.9, 0.82])
    badges = badge_engine.check_badges(1, db=db)
    keys = {b.badge_key for b in badges}
    assert "clean_machine" in keys


def test_not_awarded_when_one_rep_below_threshold(db):
    day_date = "2026-04-16"
    _seed_plan_with_completed_day(db, ["pushup"], day_date)
    # 3 reps, one below threshold
    _seed_session_with_reps(db, "pushup", day_date, [0.85, 0.9, 0.7])
    badges = badge_engine.check_badges(1, db=db)
    keys = {b.badge_key for b in badges}
    assert "clean_machine" not in keys


def test_not_awarded_when_no_reps_logged(db):
    day_date = "2026-04-16"
    _seed_plan_with_completed_day(db, ["pushup"], day_date)
    # No session seeded.
    badges = badge_engine.check_badges(1, db=db)
    keys = {b.badge_key for b in badges}
    assert "clean_machine" not in keys


def test_requires_threshold_exactly(db):
    day_date = "2026-04-16"
    _seed_plan_with_completed_day(db, ["plank"], day_date)
    _seed_session_with_reps(db, "plank", day_date, [CLEAN_REP_THRESHOLD])
    badges = badge_engine.check_badges(1, db=db)
    keys = {b.badge_key for b in badges}
    assert "clean_machine" in keys


def test_multi_exercise_day_all_must_pass(db):
    day_date = "2026-04-16"
    _seed_plan_with_completed_day(db, ["pushup", "squat"], day_date)
    # pushup clean, squat below → fail.
    _seed_session_with_reps(db, "pushup", day_date, [0.9, 0.85])
    _seed_session_with_reps(db, "squat", day_date, [0.75, 0.82])
    badges = badge_engine.check_badges(1, db=db)
    keys = {b.badge_key for b in badges}
    assert "clean_machine" not in keys


def test_awarded_only_once(db):
    day_date = "2026-04-16"
    _seed_plan_with_completed_day(db, ["pushup"], day_date)
    _seed_session_with_reps(db, "pushup", day_date, [0.9, 0.9])
    badge_engine.check_badges(1, db=db)
    second = badge_engine.check_badges(1, db=db)
    assert not any(b.badge_key == "clean_machine" for b in second)
