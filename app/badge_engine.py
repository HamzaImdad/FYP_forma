"""
FORMA Session-4 badge engine.

Stateless rule evaluator. `check_badges(user_id)` runs each badge rule in
cheap-to-expensive order, inserts any newly earned badges via the UNIQUE
constraint on `user_badges`, and returns the list of badges that actually
landed *this run* (skipping duplicates and ties).

Called from app/server.py on_session_complete after every workout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.database import ExerVisionDB
from app.insights import personal_records


BADGE_KEYS: List[str] = [
    "first_session",
    "streak_7",
    "streak_30",
    "streak_100",
    "clean_form_session",
    "volume_100",
    "volume_1000",
    "volume_10000",
    "all_exercises",
    "plank_1min",
    "plank_2min",
    "comeback",
    "plan_complete",
]

BADGE_META: Dict[str, Dict[str, str]] = {
    "first_session":      {"title": "First step",           "description": "Logged your first session."},
    "streak_7":           {"title": "One-week streak",      "description": "Trained 7 days in a row."},
    "streak_30":          {"title": "30-day streak",        "description": "Trained 30 days in a row."},
    "streak_100":         {"title": "Century streak",       "description": "Trained 100 days in a row."},
    "clean_form_session": {"title": "Clean form",           "description": "A full session with an average score of 85+."},
    "volume_100":         {"title": "100 reps",             "description": "100 reps lifetime."},
    "volume_1000":        {"title": "1,000 reps",           "description": "1,000 reps lifetime."},
    "volume_10000":       {"title": "10,000 reps",          "description": "10,000 reps lifetime."},
    "all_exercises":      {"title": "Tour of FORMA",        "description": "At least one session of each of the 10 exercises."},
    "plank_1min":         {"title": "One-minute plank",     "description": "Held a clean plank for 60+ seconds."},
    "plank_2min":         {"title": "Two-minute plank",     "description": "Held a clean plank for 120+ seconds."},
    "comeback":           {"title": "Welcome back",         "description": "First session after 14+ days off."},
    "plan_complete":      {"title": "Plan complete",        "description": "Finished every day of a workout plan."},
}


@dataclass
class Badge:
    user_id: int
    badge_key: str
    earned_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        meta = BADGE_META.get(self.badge_key, {})
        return {
            "user_id": self.user_id,
            "badge_key": self.badge_key,
            "earned_at": self.earned_at,
            "metadata": self.metadata,
            "title": meta.get("title", self.badge_key),
            "description": meta.get("description", ""),
        }


def _db(db: Optional[ExerVisionDB]) -> ExerVisionDB:
    return db or ExerVisionDB()


# ── Rule helpers ─────────────────────────────────────────────────────

def _latest_and_prev_session(conn, user_id: int):
    rows = conn.execute(
        """SELECT id, exercise, total_reps, avg_form_score, date
           FROM sessions WHERE user_id = ? ORDER BY date DESC LIMIT 2""",
        (user_id,),
    ).fetchall()
    return rows


def _total_volume(conn, user_id: int) -> int:
    row = conn.execute(
        "SELECT COALESCE(SUM(total_reps),0) AS n FROM sessions WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    return int(row["n"] or 0)


def _distinct_exercises(conn, user_id: int) -> int:
    row = conn.execute(
        "SELECT COUNT(DISTINCT exercise) AS n FROM sessions WHERE user_id = ? AND total_reps >= 0",
        (user_id,),
    ).fetchone()
    return int(row["n"] or 0)


def _current_streak(conn, user_id: int) -> int:
    """Consecutive-day streak ending today or yesterday."""
    rows = conn.execute(
        """SELECT DISTINCT DATE(date) AS d FROM sessions
           WHERE user_id = ? ORDER BY d DESC""",
        (user_id,),
    ).fetchall()
    if not rows:
        return 0
    dates = []
    for r in rows:
        try:
            dates.append(datetime.strptime(r["d"], "%Y-%m-%d").date())
        except (TypeError, ValueError):
            continue
    if not dates:
        return 0
    today = datetime.now().date()
    gap_head = (today - dates[0]).days
    if gap_head > 1:
        return 0
    streak = 1
    for i in range(1, len(dates)):
        if (dates[i - 1] - dates[i]).days == 1:
            streak += 1
        else:
            break
    return streak


def _best_form_score(conn, user_id: int):
    row = conn.execute(
        """SELECT id, avg_form_score FROM sessions
           WHERE user_id = ? ORDER BY avg_form_score DESC LIMIT 1""",
        (user_id,),
    ).fetchone()
    return row


def _longest_plank_sec(conn, user_id: int) -> float:
    row = conn.execute(
        """SELECT COALESCE(MAX(duration_sec),0) AS m FROM sessions
           WHERE user_id = ? AND exercise = 'plank'""",
        (user_id,),
    ).fetchone()
    return float(row["m"] or 0)


def _comeback_gap(conn, user_id: int):
    """Return (gap_days, session_id) if the latest session came after a 14+ day gap."""
    rows = conn.execute(
        """SELECT id, DATE(date) AS d FROM sessions
           WHERE user_id = ? ORDER BY date DESC LIMIT 2""",
        (user_id,),
    ).fetchall()
    if len(rows) < 2:
        return None
    try:
        d1 = datetime.strptime(rows[0]["d"], "%Y-%m-%d").date()
        d2 = datetime.strptime(rows[1]["d"], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None
    gap = (d1 - d2).days
    if gap >= 14:
        return {"gap_days": gap, "session_id": int(rows[0]["id"])}
    return None


def _any_plan_completed(conn, user_id: int):
    """Return plan id if the user has any plan where every non-rest day is completed."""
    plans = conn.execute(
        "SELECT id FROM plans WHERE user_id = ?",
        (user_id,),
    ).fetchall()
    for p in plans:
        row = conn.execute(
            """SELECT
                   COUNT(*) AS total,
                   SUM(CASE WHEN completed = 1 OR is_rest = 1 THEN 1 ELSE 0 END) AS done
               FROM plan_days WHERE plan_id = ?""",
            (int(p["id"]),),
        ).fetchone()
        total = int(row["total"] or 0)
        done = int(row["done"] or 0)
        if total > 0 and total == done:
            return int(p["id"])
    return None


# ── Orchestrator ─────────────────────────────────────────────────────

def check_badges(user_id: int, db: Optional[ExerVisionDB] = None) -> List[Badge]:
    db = _db(db)
    earned: List[Badge] = []

    # Pre-fetch what we need in one connection, then release it per-rule for
    # the helpers that need fresh reads (they reopen).
    conn = db._get_conn()
    try:
        latest = conn.execute(
            """SELECT id, exercise, total_reps, avg_form_score, duration_sec, date
               FROM sessions WHERE user_id = ? ORDER BY date DESC LIMIT 1""",
            (user_id,),
        ).fetchone()
        session_count = int(
            conn.execute(
                "SELECT COUNT(*) AS n FROM sessions WHERE user_id = ?",
                (user_id,),
            ).fetchone()["n"] or 0
        )
        total_volume = _total_volume(conn, user_id)
        distinct_exercises = _distinct_exercises(conn, user_id)
        best_form = _best_form_score(conn, user_id)
        longest_plank = _longest_plank_sec(conn, user_id)
        streak = _current_streak(conn, user_id)
        comeback = _comeback_gap(conn, user_id)
        plan_done = _any_plan_completed(conn, user_id)
    finally:
        conn.close()

    def _try(key: str, metadata: Dict[str, Any]) -> None:
        if db.has_badge(user_id, key):
            return
        if db.insert_badge(user_id, key, metadata):
            earned.append(
                Badge(
                    user_id=user_id,
                    badge_key=key,
                    earned_at=datetime.now().isoformat(timespec="seconds"),
                    metadata=metadata,
                )
            )

    # first_session — any completed session
    if session_count >= 1 and latest is not None:
        _try("first_session", {"session_id": int(latest["id"])})

    # streaks
    if streak >= 7:
        _try("streak_7", {"streak_days": streak})
    if streak >= 30:
        _try("streak_30", {"streak_days": streak})
    if streak >= 100:
        _try("streak_100", {"streak_days": streak})

    # clean form (avg_form_score is stored 0..1)
    if best_form is not None and (best_form["avg_form_score"] or 0) >= 0.85:
        _try(
            "clean_form_session",
            {
                "session_id": int(best_form["id"]),
                "score": round(float(best_form["avg_form_score"] or 0) * 100, 1),
            },
        )

    # volume
    if total_volume >= 100:
        _try("volume_100", {"total_reps": total_volume})
    if total_volume >= 1000:
        _try("volume_1000", {"total_reps": total_volume})
    if total_volume >= 10000:
        _try("volume_10000", {"total_reps": total_volume})

    # all 10 exercises
    if distinct_exercises >= 10:
        _try("all_exercises", {"distinct": distinct_exercises})

    # plank
    if longest_plank >= 60:
        _try("plank_1min", {"seconds": round(longest_plank, 1)})
    if longest_plank >= 120:
        _try("plank_2min", {"seconds": round(longest_plank, 1)})

    # comeback
    if comeback is not None:
        _try("comeback", comeback)

    # plan complete — also flip plan.status to 'completed'
    if plan_done is not None:
        _try("plan_complete", {"plan_id": plan_done})
        try:
            db.set_plan_status(plan_done, "completed")
        except Exception:
            pass

    return earned
