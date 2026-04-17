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
from app.exercise_registry import GOOD_REP_THRESHOLD
from app.insights import personal_records, resolve_muscle_groups
from app.settings import PHASE2_ENABLED


# Redesign Phase 2 adds "strength" — heaviest single set with at least
# `target_reps` clean reps. Used for the weighted family (deadlift,
# bench_press, overhead_press, squat with weight).
GOAL_TYPES = {
    "volume", "quality", "consistency", "skill",
    "duration", "balance", "plan_progress",
    "strength",
}
VALID_UNITS = {
    "reps", "form_score", "sessions", "seconds", "days", "muscle_groups",
    "kg",  # strength goals
}
# Redesign Phase 2: 10 → 20. Paired with auto-archive of completed goals
# in recompute_all_goals so stale entries don't occupy cap slots.
MAX_ACTIVE_GOALS = 20

_MILESTONE_FRACTIONS = (0.25, 0.50, 0.75, 1.00)

# Custom labels for plan_progress goals — so the Milestones page reads as
# "Week 1 done / Halfway there / Final stretch / Plan complete" instead of
# the generic "25% / 50% / 75% / 100%".
_PLAN_PROGRESS_LABELS = {
    0.25: "First quarter done",
    0.50: "Halfway there",
    0.75: "Final stretch",
    1.00: "Plan complete",
}


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
    plan_id: Optional[int] = None
    # Redesign Phase 2 — only used by strength goals today ("heaviest single
    # clean set where at least target_reps reps scored >= GOOD_REP_THRESHOLD").
    target_reps: Optional[int] = None
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
            "plan_id": self.plan_id,
            "target_reps": self.target_reps,
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
        plan_id=(int(row["plan_id"]) if row.get("plan_id") is not None else None),
        target_reps=(
            int(row["target_reps"]) if row.get("target_reps") is not None else None
        ),
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
    """Sum of good-form reps for this user/exercise/window.

    Redesign Phase 2: counts `good_reps` (form_score >= GOOD_REP_THRESHOLD)
    instead of `total_reps`. Sloppy reps no longer inflate volume goals.
    Legacy behavior available via FORMA_PHASE2_ENABLED=false for rollback.
    """
    params: list = [goal.user_id]
    col = "good_reps" if PHASE2_ENABLED else "total_reps"
    q = f"SELECT COALESCE(SUM({col}),0) AS n FROM sessions WHERE user_id = ?"
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
    """Longest clean hold in seconds since goal creation.

    Redesign Phase 2: gates on avg_form_score >= GOOD_REP_THRESHOLD so a
    sloppy 2-minute plank doesn't count as progress. Legacy behavior (no
    form gate) via FORMA_PHASE2_ENABLED=false.
    """
    exercise = goal.exercise or "plank"
    if PHASE2_ENABLED:
        row = conn.execute(
            """SELECT COALESCE(MAX(duration_sec), 0) AS m
               FROM sessions
               WHERE user_id = ? AND exercise = ? AND date >= ?
                 AND avg_form_score >= ?""",
            (goal.user_id, exercise, goal.created_at, GOOD_REP_THRESHOLD),
        ).fetchone()
    else:
        row = conn.execute(
            """SELECT COALESCE(MAX(duration_sec), 0) AS m
               FROM sessions
               WHERE user_id = ? AND exercise = ? AND date >= ?""",
            (goal.user_id, exercise, goal.created_at),
        ).fetchone()
    return float(row["m"] or 0)


def _calc_strength(conn, goal: Goal) -> float:
    """Heaviest single set where at least goal.target_reps reps were clean.

    For the weighted family (deadlift, bench_press, overhead_press, squat
    with weight). Joins sets → sessions → reps; requires:
      • sessions.exercise == goal.exercise
      • sessions.date >= goal.created_at
      • sets.weight_kg > 0  (ignores the bodyweight mode)
      • count of reps on that set with form_score >= threshold
          >= goal.target_reps
    Returns the MAX(sets.weight_kg) across qualifying sets, else 0.
    """
    if not goal.exercise:
        return 0.0
    target_reps = int(goal.target_reps or 0)
    if target_reps <= 0:
        # Strength goals must carry target_reps. Without it we can't
        # distinguish "3-rep max at 90kg" from "10-rep max at 90kg".
        return 0.0
    row = conn.execute(
        """SELECT COALESCE(MAX(st.weight_kg), 0) AS m
           FROM sets st
           JOIN sessions s ON s.id = st.session_id
           WHERE s.user_id = ?
             AND s.exercise = ?
             AND s.date >= ?
             AND st.weight_kg IS NOT NULL AND st.weight_kg > 0
             AND (
               SELECT COUNT(*) FROM reps r
               WHERE r.session_id = s.id
                 AND r.set_num = st.set_num
                 AND r.form_score >= ?
             ) >= ?""",
        (
            goal.user_id,
            goal.exercise,
            goal.created_at,
            GOOD_REP_THRESHOLD,
            target_reps,
        ),
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


def _calc_plan_progress(conn, goal: Goal) -> float:
    """Completed non-rest plan_days for the goal's linked plan."""
    if goal.plan_id is None:
        return goal.current_value
    row = conn.execute(
        """SELECT COUNT(*) AS n FROM plan_days
           WHERE plan_id = ? AND completed = 1 AND is_rest = 0""",
        (int(goal.plan_id),),
    ).fetchone()
    return float(row["n"] or 0)


_CALCULATORS = {
    "volume": _calc_volume,
    "quality": _calc_quality,
    "consistency": _calc_consistency,
    "skill": _calc_skill,
    "duration": _calc_duration,
    "balance": _calc_balance,
    "plan_progress": _calc_plan_progress,
    # Redesign Phase 2 — weighted-family progress ("90 kg for 3 clean reps")
    "strength": _calc_strength,
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
    plan_id: Optional[int] = None,
    target_reps: Optional[int] = None,
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

    # Strength goals must carry both exercise and target_reps — the
    # calculator needs "heaviest set that hit N clean reps".
    if goal_type == "strength":
        if not exercise:
            raise InvalidGoal("strength goals require an exercise")
        if target_reps is None or int(target_reps) <= 0:
            raise InvalidGoal(
                "strength goals require target_reps (how many reps at the "
                "target weight)"
            )
        target_reps = int(target_reps)

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
        plan_id=plan_id,
        target_reps=target_reps,
    )
    # Auto-generate 25/50/75/100% milestones. Plan-progress goals get warmer
    # human labels ("Halfway there") so the Milestones timeline reads well.
    for frac in _MILESTONE_FRACTIONS:
        if goal_type == "plan_progress":
            label = _PLAN_PROGRESS_LABELS[frac]
        else:
            label = f"{int(frac * 100)}%"
        threshold = round(target_value * frac, 4)
        db.create_milestone(goal_id, label, threshold)

    # Seed the goal with its current value immediately — so a user who has
    # already done some of the work toward this goal sees it right away.
    return recompute_goal(goal_id, user_id=user_id, db=db)


# ── Plan-goal helper (shared by chat tool + REST /api/plans) ─────────

def _plan_volume_totals(plan: Dict[str, Any]) -> Dict[str, int]:
    totals: Dict[str, int] = {}
    for d in plan.get("days", []):
        if d.get("is_rest"):
            continue
        for e in d.get("exercises", []) or []:
            ex = e.get("exercise")
            if not ex:
                continue
            reps = int(e.get("target_reps") or 0) * int(e.get("target_sets") or 0)
            if reps > 0:
                totals[ex] = totals.get(ex, 0) + reps
    return totals


def _plan_active_day_count(plan: Dict[str, Any]) -> int:
    return sum(1 for d in plan.get("days", []) if not d.get("is_rest"))


def create_plan_goals_for_plan(
    user_id: int, plan_id: int, *, db: Optional[ExerVisionDB] = None,
) -> Dict[str, Any]:
    """Auto-create the two plan-level goals every saved plan always gets:
    one plan_progress + one consistency. Per-exercise goals (volume /
    strength / duration) are now opt-in — Custom Builder shows a
    GoalSuggestionModal and the Plan Architect asks in-conversation.

    Redesign Phase 3: this is a deliberate narrowing from the old path
    which spam-created one volume goal PER EXERCISE and hit MAX_ACTIVE_GOALS
    after two plans. Legacy behavior preserved behind
    FORMA_PHASE3_ENABLED=false so we can revert if needed.

    Never raises — goal cap / invalid goal failures collapse silently so
    the caller can still report partial success.
    """
    from app.settings import PHASE3_ENABLED

    db = _db(db)
    plan = db.get_plan_full(plan_id, user_id)
    if not plan:
        return {"error": "not_found"}

    created: List[Dict[str, Any]] = []
    active_days = _plan_active_day_count(plan)

    # Legacy spam path (kept for rollback) — one volume goal per distinct
    # exercise in the plan, then consistency + plan_progress.
    if not PHASE3_ENABLED:
        volumes = _plan_volume_totals(plan)
        for ex, total_reps in sorted(volumes.items()):
            try:
                g = create_goal(
                    user_id,
                    title=f"{plan['title']} — {ex.replace('_', ' ')}",
                    goal_type="volume",
                    target_value=float(total_reps),
                    unit="reps",
                    exercise=ex,
                    period="once",
                    description=f"Finish all the {ex.replace('_', ' ')} volume in this plan.",
                    plan_id=plan_id,
                    db=db,
                )
                created.append(g.to_dict())
            except GoalCapExceeded:
                return {
                    "error": "goal_cap",
                    "message": "Goal cap reached — couldn't create every plan goal.",
                    "created": created,
                }
            except InvalidGoal as e:
                created.append({"error": str(e), "exercise": ex})

    # Phase-3 narrowed path: always just these two plan-level goals. Titles
    # are suffixed so the two goals don't render as near-duplicates in the
    # Milestones UI (same base title + different metric).
    if active_days > 0:
        try:
            g = create_goal(
                user_id,
                title=f"{plan['title']} — daily progress",
                goal_type="plan_progress",
                target_value=float(active_days),
                unit="days",
                period="once",
                description="Progress through this plan, day by day.",
                plan_id=plan_id,
                db=db,
            )
            created.append(g.to_dict())
        except (GoalCapExceeded, InvalidGoal):
            pass

        try:
            g = create_goal(
                user_id,
                title=f"{plan['title']} — consistency",
                goal_type="consistency",
                target_value=float(active_days),
                unit="sessions",
                period="once",
                description="Complete every non-rest day in this plan.",
                plan_id=plan_id,
                db=db,
            )
            created.append(g.to_dict())
        except (GoalCapExceeded, InvalidGoal):
            pass

    return {"goals_created": created}


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
    # Redesign Phase 2: sweep stale completed goals into 'archived' before
    # we recompute. Once archived they don't count toward MAX_ACTIVE_GOALS
    # and drop out of default list views. Only runs when phase 2 is on.
    if PHASE2_ENABLED:
        _archive_stale_completed(user_id, db)
    active = db.list_goals(user_id, status="active")
    out: List[Goal] = []
    for row in active:
        g = recompute_goal(row["id"], user_id=user_id, db=db)
        if g is not None:
            out.append(g)
    return out


# Goals that have been `status='completed'` for longer than this window
# get auto-flipped to `archived` on the next recompute pass.
_ARCHIVE_AFTER_DAYS = 7


def _archive_stale_completed(
    user_id: int, db: ExerVisionDB
) -> int:
    """Flip completed goals older than 7 days to 'archived'.

    Returns the number of rows affected. Safe to call on every recompute;
    the WHERE clause filters stable rows.
    """
    cutoff = (datetime.now() - timedelta(days=_ARCHIVE_AFTER_DAYS)).isoformat()
    conn = db._get_conn()
    try:
        cur = conn.execute(
            """UPDATE goals SET status = 'archived'
               WHERE user_id = ?
                 AND status = 'completed'
                 AND completed_at IS NOT NULL
                 AND completed_at < ?""",
            (int(user_id), cutoff),
        )
        conn.commit()
        return int(cur.rowcount or 0)
    finally:
        conn.close()


# ── Session-5: manual goal edit with milestone re-anchor ─────────────

def update_goal_with_redrive(
    goal_id: int,
    user_id: int,
    patch: Dict[str, Any],
    *,
    db: Optional[ExerVisionDB] = None,
) -> Optional[Goal]:
    """Apply a whitelisted goal patch. If target_value changes, re-anchor
    the UNREACHED milestones at 25/50/75/100% of the new target, preserving
    any milestones the user has already hit at their original threshold.
    Then recompute_goal so the new current_value can fire any milestones
    the new thresholds now put behind progress.

    Returns the refreshed Goal, or None if the goal isn't owned by user_id.
    """
    db = _db(db)
    existing = db.get_goal(goal_id, user_id)
    if not existing:
        return None

    old_target = float(existing.get("target_value") or 0)
    new_target: Optional[float] = None
    if "target_value" in patch and patch["target_value"] is not None:
        try:
            new_target = float(patch["target_value"])
        except (TypeError, ValueError):
            raise InvalidGoal("target_value must be numeric")
        if new_target <= 0:
            raise InvalidGoal("target_value must be positive")

    # Apply the whitelisted patch via the existing db.update_goal.
    # Redesign Phase 2: target_reps is editable for strength goals.
    write_fields: Dict[str, Any] = {}
    for k in ("title", "description", "target_value", "status", "target_reps"):
        if k in patch and patch[k] is not None:
            write_fields[k] = patch[k]
    if write_fields:
        db.update_goal(goal_id, user_id, **write_fields)

    # Re-anchor milestones if the target changed.
    if new_target is not None and abs(new_target - old_target) > 1e-9:
        goal_type = str(existing.get("goal_type") or "")
        for m in existing.get("milestones") or []:
            if bool(m.get("reached")):
                continue  # frozen
            frac = None
            if old_target > 0:
                frac = round(float(m.get("threshold_value") or 0) / old_target, 4)
            # Snap to the canonical fractions if close (handles 0.25/0.5/0.75/1.0).
            if frac is not None:
                for canon in _MILESTONE_FRACTIONS:
                    if abs(frac - canon) < 1e-3:
                        frac = canon
                        break
            if frac is None:
                frac = 0.25
            new_threshold = round(new_target * frac, 4)
            new_label = None
            if goal_type == "plan_progress":
                new_label = _PLAN_PROGRESS_LABELS.get(frac)
            else:
                new_label = f"{int(round(frac * 100))}%"
            db.update_milestone_threshold(
                int(m["id"]), new_threshold, label=new_label
            )

    return recompute_goal(goal_id, user_id=user_id, db=db)


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
