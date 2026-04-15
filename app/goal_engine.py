"""
FORMA Session-4 goal engine.

Stateless-ish module. Owns the Goal/Milestone dataclasses, the create/recompute
lifecycle, and the 8 per-type "how much of this goal has the user done?" calculators.

Called from app/server.py on_session_complete after every workout. Also exposed
as a tool to the plan-creator chatbot (via create_goal) and read through the
/api/goals REST endpoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from app.database import ExerVisionDB
from app.insights import personal_records, resolve_muscle_groups


GOAL_TYPES = {"volume", "quality", "consistency", "skill", "duration", "balance"}
VALID_UNITS = {"reps", "form_score", "sessions", "seconds", "days", "muscle_groups"}
MAX_ACTIVE_GOALS = 10

_MILESTONE_FRACTIONS = (0.25, 0.50, 0.75, 1.00)


class GoalCapExceeded(ValueError):
    pass


class InvalidGoal(ValueError):
    pass


@dataclass
class Milestone:
    id: int
    goal_id: int
    label: str
    threshold_value: float
    reached: bool
    reached_at: Optional[str]
    just_reached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "goal_id": self.goal_id,
            "label": self.label,
            "threshold_value": self.threshold_value,
            "reached": self.reached,
            "reached_at": self.reached_at,
            "just_reached": self.just_reached,
        }


@dataclass
class Goal:
    id: int
    user_id: int
    title: str
    description: Optional[str]
    goal_type: str
    exercise: Optional[str]
    target_value: float
    current_value: float
    unit: str
    period: Optional[str]
    deadline: Optional[str]
    status: str
    created_at: str
    completed_at: Optional[str]
    milestones: List[Milestone] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "description": self.description,
            "goal_type": self.goal_type,
            "exercise": self.exercise,
            "target_value": self.target_value,
            "current_value": self.current_value,
            "unit": self.unit,
            "period": self.period,
            "deadline": self.deadline,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "progress": (
                min(1.0, self.current_value / self.target_value)
                if self.target_value
                else 0.0
            ),
            "milestones": [m.to_dict() for m in self.milestones],
        }


# ── Helpers ──────────────────────────────────────────────────────────

def _db(db: Optional[ExerVisionDB]) -> ExerVisionDB:
    return db or ExerVisionDB()


def _goal_from_row(row: Dict[str, Any]) -> Goal:
    ms = [
        Milestone(
            id=int(m["id"]),
            goal_id=int(m["goal_id"]),
            label=str(m["label"]),
            threshold_value=float(m["threshold_value"]),
            reached=bool(m.get("reached", 0)),
            reached_at=m.get("reached_at"),
        )
        for m in row.get("milestones", [])
    ]
    return Goal(
        id=int(row["id"]),
        user_id=int(row["user_id"]),
        title=str(row["title"]),
        description=row.get("description"),
        goal_type=str(row["goal_type"]),
        exercise=row.get("exercise"),
        target_value=float(row["target_value"]),
        current_value=float(row.get("current_value") or 0),
        unit=str(row["unit"]),
        period=row.get("period"),
        deadline=row.get("deadline"),
        status=str(row.get("status") or "active"),
        created_at=str(row["created_at"]),
        completed_at=row.get("completed_at"),
        milestones=ms,
    )


def _window_cutoff(period: Optional[str]) -> Optional[str]:
    """Return ISO cutoff for 'week'/'month'. None means 'no lower bound'."""
    if period == "week":
        return (datetime.now() - timedelta(days=7)).isoformat()
    if period == "month":
        return (datetime.now() - timedelta(days=30)).isoformat()
    return None


# ── Calculators ──────────────────────────────────────────────────────

def _calc_volume(conn, goal: Goal) -> float:
    params: list = [goal.user_id]
    q = "SELECT COALESCE(SUM(total_reps),0) AS n FROM sessions WHERE user_id = ?"
    if goal.exercise:
        q += " AND exercise = ?"
        params.append(goal.exercise)
    if goal.period in ("week", "month"):
        q += " AND date >= ?"
        params.append(_window_cutoff(goal.period))
    else:
        # period=once → cumulative since goal.created_at
        q += " AND date >= ?"
        params.append(goal.created_at)
    return float(conn.execute(q, params).fetchone()["n"] or 0)


def _calc_quality(conn, goal: Goal) -> float:
    params: list = [goal.user_id]
    q = (
        "SELECT COALESCE(AVG(NULLIF(avg_form_score,0)),0) AS s "
        "FROM sessions WHERE user_id = ?"
    )
    if goal.exercise:
        q += " AND exercise = ?"
        params.append(goal.exercise)
    if goal.period in ("week", "month"):
        q += " AND date >= ?"
        params.append(_window_cutoff(goal.period))
    else:
        q += " AND date >= ?"
        params.append(goal.created_at)
    # avg_form_score is stored 0..1 — scale to 0..100 for human-readable targets
    raw = float(conn.execute(q, params).fetchone()["s"] or 0)
    return round(raw * 100, 2)


def _calc_consistency(conn, goal: Goal) -> float:
    params: list = [goal.user_id]
    q = (
        "SELECT COUNT(DISTINCT DATE(date)) AS d "
        "FROM sessions WHERE user_id = ?"
    )
    if goal.exercise:
        q += " AND exercise = ?"
        params.append(goal.exercise)
    if goal.period in ("week", "month"):
        q += " AND date >= ?"
        params.append(_window_cutoff(goal.period))
    else:
        q += " AND date >= ?"
        params.append(goal.created_at)
    return float(conn.execute(q, params).fetchone()["d"] or 0)


def _calc_skill(conn, goal: Goal) -> float:
    """Best single-set rep count for this exercise since goal creation."""
    params: list = [goal.user_id, goal.created_at]
    q = (
        "SELECT COALESCE(MAX(st.reps_count),0) AS m "
        "FROM sets st JOIN sessions s ON s.id = st.session_id "
        "WHERE s.user_id = ? AND s.date >= ?"
    )
    if goal.exercise:
        q += " AND s.exercise = ?"
        params.append(goal.exercise)
    return float(conn.execute(q, params).fetchone()["m"] or 0)


def _calc_duration(conn, goal: Goal) -> float:
    """Longest plank hold in seconds since goal creation."""
    exercise = goal.exercise or "plank"
    row = conn.execute(
        """SELECT COALESCE(MAX(duration_sec), 0) AS m
           FROM sessions
           WHERE user_id = ? AND exercise = ? AND date >= ?""",
        (goal.user_id, exercise, goal.created_at),
    ).fetchone()
    return float(row["m"] or 0)


def _calc_balance(conn, goal: Goal) -> float:
    """Distinct big muscle groups covered in the window."""
    params: list = [goal.user_id]
    q = "SELECT DISTINCT exercise FROM sessions WHERE user_id = ?"
    if goal.period in ("week", "month"):
        q += " AND date >= ?"
        params.append(_window_cutoff(goal.period))
    else:
        q += " AND date >= ?"
        params.append(goal.created_at)
    exercises = [r["exercise"] for r in conn.execute(q, params).fetchall()]
    groups: set = set()
    for ex in exercises:
        for g in resolve_muscle_groups(ex or ""):
            groups.add(g)
    return float(len(groups))


_CALCULATORS = {
    "volume": _calc_volume,
    "quality": _calc_quality,
    "consistency": _calc_consistency,
    "skill": _calc_skill,
    "duration": _calc_duration,
    "balance": _calc_balance,
}


def _calculate_current(goal: Goal, db: ExerVisionDB) -> float:
    calc = _CALCULATORS.get(goal.goal_type)
    if calc is None:
        return goal.current_value
    conn = db._get_conn()
    try:
        return float(calc(conn, goal))
    finally:
        conn.close()


# ── Core API ─────────────────────────────────────────────────────────

def create_goal(
    user_id: int,
    *,
    title: str,
    goal_type: str,
    target_value: float,
    unit: str,
    exercise: Optional[str] = None,
    deadline: Optional[str] = None,
    period: Optional[str] = None,
    description: Optional[str] = None,
    db: Optional[ExerVisionDB] = None,
) -> Goal:
    if goal_type not in GOAL_TYPES:
        raise InvalidGoal(f"unknown goal_type '{goal_type}'")
    if unit not in VALID_UNITS:
        raise InvalidGoal(f"unknown unit '{unit}'")
    if not title or not title.strip():
        raise InvalidGoal("title is required")
    try:
        target_value = float(target_value)
    except (TypeError, ValueError):
        raise InvalidGoal("target_value must be numeric")
    if target_value <= 0:
        raise InvalidGoal("target_value must be positive")

    db = _db(db)
    if db.count_active_goals(user_id) >= MAX_ACTIVE_GOALS:
        raise GoalCapExceeded(
            f"You already have {MAX_ACTIVE_GOALS} active goals. "
            "Complete or delete one first."
        )

    goal_id = db.create_goal(
        user_id,
        title=title.strip(),
        goal_type=goal_type,
        target_value=target_value,
        unit=unit,
        exercise=exercise,
        deadline=deadline,
        period=period,
        description=description,
    )
    # Auto-generate 25/50/75/100% milestones
    for frac in _MILESTONE_FRACTIONS:
        label = f"{int(frac * 100)}%"
        threshold = round(target_value * frac, 4)
        db.create_milestone(goal_id, label, threshold)

    # Seed the goal with its current value immediately — so a user who has
    # already done some of the work toward this goal sees it right away.
    return recompute_goal(goal_id, user_id=user_id, db=db)


def recompute_goal(
    goal_id: int,
    *,
    user_id: Optional[int] = None,
    db: Optional[ExerVisionDB] = None,
) -> Optional[Goal]:
    """Recalculate current_value, fire milestones, persist, return refreshed Goal."""
    db = _db(db)
    # Use get_goal if we can verify ownership, otherwise fall back to a
    # direct read (used from recompute_all_goals where user_id is already known).
    row: Optional[Dict[str, Any]]
    if user_id is not None:
        row = db.get_goal(goal_id, user_id)
    else:
        conn = db._get_conn()
        try:
            g = conn.execute(
                "SELECT * FROM goals WHERE id = ?", (int(goal_id),)
            ).fetchone()
            if not g:
                return None
            ms = conn.execute(
                "SELECT * FROM milestones WHERE goal_id = ? ORDER BY threshold_value ASC",
                (int(goal_id),),
            ).fetchall()
        finally:
            conn.close()
        row = {**dict(g), "milestones": [dict(m) for m in ms]}
    if not row:
        return None

    goal = _goal_from_row(row)
    if goal.status != "active":
        return goal

    new_value = _calculate_current(goal, db)
    goal.current_value = new_value

    # Fire milestones that the new value has now passed
    for m in goal.milestones:
        if not m.reached and new_value + 1e-9 >= m.threshold_value:
            db.mark_milestone_reached(m.id)
            m.reached = True
            m.reached_at = datetime.now().isoformat(timespec="seconds")
            m.just_reached = True

    completed = new_value + 1e-9 >= goal.target_value
    if completed:
        now = datetime.now().isoformat(timespec="seconds")
        db.set_goal_current(goal.id, new_value, status="completed", completed_at=now)
        goal.status = "completed"
        goal.completed_at = now
    else:
        db.set_goal_current(goal.id, new_value)

    return goal


def recompute_all_goals(
    user_id: int, db: Optional[ExerVisionDB] = None
) -> List[Goal]:
    db = _db(db)
    active = db.list_goals(user_id, status="active")
    out: List[Goal] = []
    for row in active:
        g = recompute_goal(row["id"], user_id=user_id, db=db)
        if g is not None:
            out.append(g)
    return out


# ── Goal templates ───────────────────────────────────────────────────

_TEMPLATES: List[Dict[str, Any]] = [
    {
        "key": "volume_pushup_200_week",
        "title": "200 push-ups per week",
        "goal_type": "volume",
        "exercise": "pushup",
        "target_value": 200,
        "unit": "reps",
        "period": "week",
        "description": "Build push-up volume with a weekly target.",
    },
    {
        "key": "consistency_3_week",
        "title": "Train 3 days a week",
        "goal_type": "consistency",
        "exercise": None,
        "target_value": 3,
        "unit": "sessions",
        "period": "week",
        "description": "Show up three times a week. Any exercise counts.",
    },
    {
        "key": "duration_plank_120",
        "title": "Hold a 2-minute plank",
        "goal_type": "duration",
        "exercise": "plank",
        "target_value": 120,
        "unit": "seconds",
        "period": "once",
        "description": "Work your way up to a clean 2-minute hold.",
    },
    {
        "key": "quality_form_80",
        "title": "Average form score 80+",
        "goal_type": "quality",
        "exercise": None,
        "target_value": 80,
        "unit": "form_score",
        "period": "month",
        "description": "Clean up your technique. Month-long average.",
    },
    {
        "key": "skill_pushup_10_consec",
        "title": "10 push-ups in one set",
        "goal_type": "skill",
        "exercise": "pushup",
        "target_value": 10,
        "unit": "reps",
        "period": "once",
        "description": "Ten clean push-ups in a single set — no pauses.",
    },
    {
        "key": "balance_all_groups_week",
        "title": "Train every muscle group this week",
        "goal_type": "balance",
        "exercise": None,
        "target_value": 6,
        "unit": "muscle_groups",
        "period": "week",
        "description": "Hit chest, back, shoulders, arms, legs, and core in 7 days.",
    },
]


def goal_templates() -> List[Dict[str, Any]]:
    return [dict(t) for t in _TEMPLATES]
