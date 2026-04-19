"""
FORMA chat tools — OpenAI function-calling implementations.

Every tool takes `user_id` as its first argument, which the chat endpoint
injects from the JWT; it is NOT passed by the LLM. Tools return
JSON-serializable dicts/lists that the engine stringifies, wraps in
<tool_response> tags, and feeds back to the model.

Cross-user protection: every query in this module filters by user_id.
get_session_detail verifies ownership via ExerVisionDB.get_session_detail,
which returns None for unowned sessions — the tool surfaces that as
"session not found in your history" rather than leaking existence.

Session 4 extends this with plan-creator tools (propose_plan, revise_plan,
save_plan, create_plan_goals). Keep this file stable so the split is clean.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.database import ExerVisionDB
from app.insights import (
    generate_insights,
    muscle_balance,
    personal_records,
)
from src.utils.constants import EXERCISES

logger = logging.getLogger(__name__)


# ── Tool registry (OpenAI function-calling schemas) ───────────────────

EXERCISE_ENUM = list(EXERCISES)
PERIOD_ENUM = ["today", "week", "month", "all"]


PERSONAL_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_stats",
            "description": (
                "Return the user's aggregate training stats (workouts, reps, "
                "avg form score, total time, current streak) for a time period."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "enum": PERIOD_ENUM,
                        "description": "Time window. Defaults to 'week'.",
                    },
                    "exercise": {
                        "type": "string",
                        "enum": EXERCISE_ENUM,
                        "description": "Optional. Filter by a single exercise.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_sessions",
            "description": (
                "List the user's most recent workout sessions, newest first. "
                "Each session returns: id, exercise, date, duration_sec "
                "(seconds the session lasted, float), total_reps, good_reps, "
                "avg_form_score, weight_kg."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "exercise": {
                        "type": "string",
                        "enum": EXERCISE_ENUM,
                        "description": "Optional exercise filter.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max sessions to return (1-30). Default 10.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_session_detail",
            "description": (
                "Fetch the full per-rep breakdown for one session the user "
                "owns. Returns id, exercise, date, duration_sec (seconds), "
                "total_reps, avg_form_score, fatigue_index, consistency_score, "
                "plus per-rep and per-set arrays. Returns {error: 'not_found'} "
                "if the id is not in their history — never leak existence of "
                "other users' sessions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "integer"},
                },
                "required": ["session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_insights",
            "description": (
                "Run the FORMA insights engine (15 rules) and return ranked "
                "observations about the user's training. Preferred tool for "
                "'what's my weakness', 'how am I doing', 'am I improving'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "exercise": {
                        "type": "string",
                        "enum": EXERCISE_ENUM,
                        "description": "Filter insights to one exercise.",
                    },
                    "period": {
                        "type": "string",
                        "enum": PERIOD_ENUM,
                        "description": "Defaults to 'month'.",
                    },
                    "limit": {"type": "integer", "description": "Max insights (default 10)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_muscle_balance",
            "description": (
                "Return reps per muscle group (chest/back/shoulders/arms/legs/core) "
                "over the last N days. Use to answer 'am I balanced' or 'what have "
                "I been neglecting'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Default 7."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_personal_records",
            "description": "Return the user's personal records (biggest session, cleanest form, longest hold, longest streak).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weakness_report",
            "description": (
                "Produce a focused weakness report: top recurring issues + "
                "fatigue curve + depth trend for an exercise (or all exercises)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "exercise": {
                        "type": "string",
                        "enum": EXERCISE_ENUM,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_progress_comparison",
            "description": (
                "Compare this-period vs last-period KPIs (reps, form score, "
                "time trained, session count). Use for 'how did I do this week'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "enum": ["week", "month"],
                        "description": "Default 'week'.",
                    },
                    "exercise": {"type": "string", "enum": EXERCISE_ENUM},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_goals",
            "description": (
                "List the user's active SMART goals. Session 4 populates this "
                "table — for now this returns an empty list."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_badges",
            "description": (
                "Return the user's earned badges AND the locked badges they "
                "haven't unlocked yet. Use when the user asks 'what badges "
                "have I earned', 'what's next', or 'how close am I to X'."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_lifetime_totals",
            "description": (
                "Return lifetime career aggregates: total sessions, total "
                "reps, total good reps (form >= 0.6), total training time, "
                "lifetime average form score, first-session date, and the "
                "list of exercises the user has trained. Use for 'total reps "
                "ever', 'how long have I been training', 'what have I done'."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_plan_detail",
            "description": (
                "Return full detail of ONE plan — days, exercises per day, "
                "rest days, and adherence (days completed / total days). If "
                "plan_id is omitted, returns the user's currently active "
                "plan. Use for 'what does my plan look like', 'how am I "
                "doing vs my plan', or to inspect a specific plan by id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_id": {
                        "type": "integer",
                        "description": "Plan id. Omit to use the active plan.",
                    },
                },
                "required": [],
            },
        },
    },
]


# ── Helper: streak count ──────────────────────────────────────────────


def _streak_days(db: ExerVisionDB, user_id: int) -> int:
    conn = db._get_conn()
    try:
        rows = conn.execute(
            """SELECT DISTINCT DATE(date) AS d FROM sessions
               WHERE user_id = ? ORDER BY d DESC""",
            (int(user_id),),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return 0
    today = datetime.now().date()
    dates = [datetime.fromisoformat(r["d"]).date() for r in rows]
    if (today - dates[0]) > timedelta(days=1):
        return 0
    streak = 1
    cursor = dates[0]
    for d in dates[1:]:
        if cursor - d == timedelta(days=1):
            streak += 1
            cursor = d
        else:
            break
    return streak


# ── Tool implementations ──────────────────────────────────────────────


def get_stats(
    user_id: int,
    db: ExerVisionDB,
    period: str = "week",
    exercise: Optional[str] = None,
) -> Dict[str, Any]:
    if period not in PERIOD_ENUM:
        period = "week"
    if exercise and exercise not in EXERCISES:
        exercise = None
    stats = db.get_stats(user_id, period=period, exercise=exercise)
    stats["streak_days"] = _streak_days(db, user_id)
    stats["period"] = period
    if exercise:
        stats["exercise"] = exercise
    return stats


def get_recent_sessions(
    user_id: int,
    db: ExerVisionDB,
    exercise: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    try:
        limit = max(1, min(30, int(limit)))
    except (TypeError, ValueError):
        limit = 10
    if exercise and exercise not in EXERCISES:
        exercise = None
    rows = db.get_sessions(user_id, exercise=exercise, limit=limit)
    cleaned = [
        {
            "id": int(r["id"]),
            "exercise": r["exercise"],
            "date": r["date"],
            "duration_sec": round(float(r.get("duration_sec") or 0), 1),
            "total_reps": int(r.get("total_reps") or 0),
            # Redesign Phase 4 — distinguish good from total so the coach
            # can say "28 good / 35 total" instead of just "35".
            "good_reps": int(r.get("good_reps") or 0),
            "avg_form_score": round(float(r.get("avg_form_score") or 0), 3),
            "weight_kg": r.get("weight_kg"),
        }
        for r in rows
    ]
    return {"sessions": cleaned, "count": len(cleaned)}


def get_session_detail(
    user_id: int, db: ExerVisionDB, session_id: int
) -> Dict[str, Any]:
    try:
        sid = int(session_id)
    except (TypeError, ValueError):
        return {"error": "not_found", "message": "I don't see that session in your history."}
    detail = db.get_session_detail(user_id, sid)
    if detail is None:
        # Never leak existence — same response for "not yours" and "doesn't exist".
        return {
            "error": "not_found",
            "message": "I don't see that session in your history.",
        }
    # Trim heavy per-rep issue arrays to keep the tool response small.
    reps = detail.get("reps", []) or []
    trimmed_reps = [
        {
            "rep_num": int(r.get("rep_num") or 0),
            "form_score": round(float(r.get("form_score") or 0), 3),
            "quality": r.get("quality"),
            "issues": r.get("issues", [])[:3],
            "peak_angle": r.get("peak_angle"),
            "ecc_sec": r.get("ecc_sec"),
            "con_sec": r.get("con_sec"),
            "set_num": r.get("set_num"),
        }
        for r in reps[:40]
    ]
    return {
        "id": int(detail["id"]),
        "exercise": detail["exercise"],
        "date": detail["date"],
        "duration_sec": detail.get("duration_sec"),
        "total_reps": detail.get("total_reps"),
        "avg_form_score": detail.get("avg_form_score"),
        "fatigue_index": detail.get("fatigue_index"),
        "consistency_score": detail.get("consistency_score"),
        "reps": trimmed_reps,
        "sets": [
            {
                "set_num": int(s.get("set_num") or 0),
                "reps_count": int(s.get("reps_count") or 0),
                "avg_form_score": s.get("avg_form_score"),
                "score_dropoff": s.get("score_dropoff"),
                "rest_before_sec": s.get("rest_before_sec"),
                "failure_type": s.get("failure_type"),
            }
            for s in (detail.get("sets") or [])
        ],
    }


def get_insights_tool(
    user_id: int,
    db: ExerVisionDB,
    exercise: Optional[str] = None,
    period: str = "month",
    limit: int = 10,
) -> Dict[str, Any]:
    if exercise and exercise not in EXERCISES:
        exercise = None
    if period not in PERIOD_ENUM:
        period = "month"
    try:
        limit = max(1, min(20, int(limit)))
    except (TypeError, ValueError):
        limit = 10
    items = generate_insights(user_id, exercise=exercise, period=period, limit=limit, db=db)
    return {"insights": [i.to_dict() for i in items], "count": len(items)}


def get_muscle_balance_tool(
    user_id: int, db: ExerVisionDB, days: int = 7
) -> Dict[str, Any]:
    try:
        days = max(1, min(90, int(days)))
    except (TypeError, ValueError):
        days = 7
    groups = muscle_balance(user_id, days=days, db=db)
    return {"days": days, "groups": groups}


def get_personal_records_tool(user_id: int, db: ExerVisionDB) -> Dict[str, Any]:
    return personal_records(user_id, db=db)


def get_weakness_report(
    user_id: int,
    db: ExerVisionDB,
    exercise: Optional[str] = None,
) -> Dict[str, Any]:
    if exercise and exercise not in EXERCISES:
        exercise = None
    top_issues = db.get_top_issues(user_id, exercise=exercise, limit=5)
    insights = generate_insights(user_id, exercise=exercise, period="month", limit=20, db=db)
    targeted = [
        i.to_dict()
        for i in insights
        if i.id.startswith(("recurring_issue", "fatigue_curve", "depth_cutting"))
    ]
    return {
        "exercise": exercise,
        "top_issues": top_issues,
        "targeted_insights": targeted,
        "summary": {
            "issue_count": len(top_issues),
            "insight_count": len(targeted),
        },
    }


def get_progress_comparison(
    user_id: int,
    db: ExerVisionDB,
    period: str = "week",
    exercise: Optional[str] = None,
) -> Dict[str, Any]:
    if exercise and exercise not in EXERCISES:
        exercise = None
    wow = db.get_wow_deltas(user_id, exercise=exercise)

    def _delta(curr: float, prev: float) -> Dict[str, Any]:
        diff = curr - prev
        pct = (diff / prev * 100) if prev else None
        return {"current": curr, "previous": prev, "delta": diff, "pct_change": pct}

    return {
        "period": period,
        "exercise": exercise,
        "reps": _delta(wow["this_week"]["reps"], wow["last_week"]["reps"]),
        "form": _delta(
            round(wow["this_week"]["form"], 3), round(wow["last_week"]["form"], 3)
        ),
        "time_sec": _delta(
            round(wow["this_week"]["time"], 1), round(wow["last_week"]["time"], 1)
        ),
        "sessions": _delta(wow["this_week"]["sessions"], wow["last_week"]["sessions"]),
    }


def get_goals_tool(user_id: int, db: ExerVisionDB) -> Dict[str, Any]:
    """Session-4: return the user's active goals with progress + milestones."""
    goals = db.list_goals(user_id, status="active")
    out = []
    for g in goals:
        target = float(g.get("target_value") or 0)
        current = float(g.get("current_value") or 0)
        out.append(
            {
                "id": g["id"],
                "title": g["title"],
                "goal_type": g.get("goal_type"),
                "exercise": g.get("exercise"),
                "target_value": target,
                "current_value": current,
                "unit": g.get("unit"),
                "period": g.get("period"),
                "progress": min(1.0, current / target) if target else 0.0,
                "status": g.get("status"),
                "milestones": [
                    {
                        "label": m.get("label"),
                        "threshold_value": m.get("threshold_value"),
                        "reached": bool(m.get("reached", False)),
                        "reached_at": m.get("reached_at"),
                    }
                    for m in g.get("milestones", [])
                ],
            }
        )
    return {"goals": out}


def get_badges_tool(user_id: int, db: ExerVisionDB) -> Dict[str, Any]:
    """Earned + locked badges. Locked list = all BADGE_KEYS minus earned."""
    from app.badge_engine import BADGE_KEYS, BADGE_META

    earned_rows = db.list_badges(user_id)
    earned_keys = {r["badge_key"] for r in earned_rows}
    earned = [
        {
            "badge_key": r["badge_key"],
            "title": BADGE_META.get(r["badge_key"], {}).get("title", r["badge_key"]),
            "description": BADGE_META.get(r["badge_key"], {}).get("description", ""),
            "earned_at": r["earned_at"],
            "metadata": r.get("metadata") or {},
        }
        for r in earned_rows
    ]
    locked = [
        {
            "badge_key": k,
            "title": BADGE_META.get(k, {}).get("title", k),
            "description": BADGE_META.get(k, {}).get("description", ""),
        }
        for k in BADGE_KEYS
        if k not in earned_keys
    ]
    return {"earned": earned, "locked": locked, "earned_count": len(earned), "total_count": len(BADGE_KEYS)}


def get_lifetime_totals_tool(user_id: int, db: ExerVisionDB) -> Dict[str, Any]:
    """Career aggregates across all sessions (no time filter)."""
    conn = db._get_conn()
    try:
        row = conn.execute(
            """SELECT
                 COUNT(*) AS n_sessions,
                 COALESCE(SUM(total_reps), 0) AS total_reps,
                 COALESCE(SUM(good_reps), 0) AS total_good_reps,
                 COALESCE(SUM(duration_sec), 0.0) AS total_time_sec,
                 COALESCE(AVG(NULLIF(avg_form_score, 0)), 0.0) AS avg_form_score,
                 MIN(date) AS first_session_date
               FROM sessions WHERE user_id = ?""",
            (int(user_id),),
        ).fetchone()
        exs = conn.execute(
            """SELECT exercise, COUNT(*) AS n, SUM(total_reps) AS reps
               FROM sessions WHERE user_id = ?
               GROUP BY exercise ORDER BY n DESC""",
            (int(user_id),),
        ).fetchall()
    finally:
        conn.close()
    return {
        "total_sessions": int(row["n_sessions"] or 0),
        "total_reps": int(row["total_reps"] or 0),
        "total_good_reps": int(row["total_good_reps"] or 0),
        "total_time_sec": round(float(row["total_time_sec"] or 0.0), 1),
        "avg_form_score": round(float(row["avg_form_score"] or 0.0), 3),
        "first_session_date": row["first_session_date"],
        "exercises_trained": [
            {"exercise": r["exercise"], "sessions": int(r["n"]), "reps": int(r["reps"] or 0)}
            for r in exs
        ],
    }


def get_plan_detail_tool(
    user_id: int, db: ExerVisionDB, plan_id: Optional[int] = None
) -> Dict[str, Any]:
    """Full plan detail + adherence. If plan_id omitted, uses the active plan."""
    if plan_id is None:
        all_plans = db.list_plans(user_id)
        active = next((p for p in all_plans if p.get("status") == "active"), None)
        if not active:
            return {"error": "no_active_plan"}
        plan_id = int(active["id"])
    try:
        plan_id = int(plan_id)
    except (TypeError, ValueError):
        return {"error": "invalid_plan_id"}
    full = db.get_plan_full(plan_id, user_id)
    if not full:
        return {"error": "plan_not_found"}
    days = full.get("days") or []
    total_days = len(days)
    completed_days = sum(1 for d in days if d.get("completed"))
    non_rest_total = sum(1 for d in days if not d.get("is_rest"))
    non_rest_completed = sum(1 for d in days if d.get("completed") and not d.get("is_rest"))
    return {
        "id": full["id"],
        "title": full.get("title"),
        "summary": full.get("summary"),
        "start_date": full.get("start_date"),
        "end_date": full.get("end_date"),
        "status": full.get("status"),
        "active": full.get("status") == "active",
        "days": [
            {
                "day_date": d.get("day_date"),
                "is_rest": bool(d.get("is_rest")),
                "completed": bool(d.get("completed")),
                "completed_at": d.get("completed_at"),
                "exercises": d.get("exercises") or [],
            }
            for d in days
        ],
        "progress": {
            "days_total": total_days,
            "days_completed": completed_days,
            "training_days_total": non_rest_total,
            "training_days_completed": non_rest_completed,
            "adherence_pct": (
                round(100.0 * non_rest_completed / non_rest_total, 1)
                if non_rest_total else 0.0
            ),
        },
    }


# ── Dispatcher ────────────────────────────────────────────────────────


def make_dispatcher(
    user_id: int, db: ExerVisionDB
) -> Callable[[str, Dict[str, Any]], Any]:
    """Return a closure that maps tool name → implementation with user_id bound.

    The chat engine calls dispatcher(fn_name, args_dict). We shield tools from
    unexpected args and never trust the LLM to pass user_id — it's closed over
    from the HTTP request context.
    """

    def dispatch(fn_name: str, args: Dict[str, Any]) -> Any:
        logger.info("chat tool %s user=%d args=%s", fn_name, user_id, args)
        if fn_name == "get_stats":
            return get_stats(
                user_id, db,
                period=args.get("period", "week"),
                exercise=args.get("exercise"),
            )
        if fn_name == "get_recent_sessions":
            return get_recent_sessions(
                user_id, db,
                exercise=args.get("exercise"),
                limit=args.get("limit", 10),
            )
        if fn_name == "get_session_detail":
            return get_session_detail(user_id, db, session_id=args.get("session_id", 0))
        if fn_name == "get_insights":
            return get_insights_tool(
                user_id, db,
                exercise=args.get("exercise"),
                period=args.get("period", "month"),
                limit=args.get("limit", 10),
            )
        if fn_name == "get_muscle_balance":
            return get_muscle_balance_tool(user_id, db, days=args.get("days", 7))
        if fn_name == "get_personal_records":
            return get_personal_records_tool(user_id, db)
        if fn_name == "get_weakness_report":
            return get_weakness_report(user_id, db, exercise=args.get("exercise"))
        if fn_name == "get_progress_comparison":
            return get_progress_comparison(
                user_id, db,
                period=args.get("period", "week"),
                exercise=args.get("exercise"),
            )
        if fn_name == "get_goals":
            return get_goals_tool(user_id, db)
        if fn_name == "get_badges":
            return get_badges_tool(user_id, db)
        if fn_name == "get_lifetime_totals":
            return get_lifetime_totals_tool(user_id, db)
        if fn_name == "get_plan_detail":
            return get_plan_detail_tool(user_id, db, plan_id=args.get("plan_id"))
        return {"error": "unknown_tool", "name": fn_name}

    return dispatch


# ── Session 4: plan-creator tools ─────────────────────────────────────

import copy
import json as _json
from datetime import datetime as _dt, timedelta as _td

from app.exercise_registry import EXERCISE_FAMILIES

KNOWN_EXERCISES: set = set(EXERCISE_FAMILIES.keys())

# Per-exercise rep/duration ceilings when we have no user history to anchor against.
# Plank and side_plank ceilings are seconds (time_hold family); others are reps.
_DEFAULT_REP_CEILING: Dict[str, int] = {
    "pushup": 60,
    "squat": 60,
    "lunge": 60,
    "bicep_curl": 60,
    "tricep_dip": 60,
    "pullup": 20,
    "deadlift": 30,
    "crunch": 40,
    "lateral_raise": 30,
    "plank": 300,
    "side_plank": 180,
}


class PlanRejected(ValueError):
    pass


def _user_recent_max_reps(db: ExerVisionDB, user_id: int, exercise: str) -> Optional[int]:
    """Max rep count across any single set of this exercise in the last 90 days."""
    conn = db._get_conn()
    try:
        cutoff = (_dt.now() - _td(days=90)).isoformat()
        row = conn.execute(
            """SELECT COALESCE(MAX(st.reps_count), 0) AS m
               FROM sets st JOIN sessions s ON s.id = st.session_id
               WHERE s.user_id = ? AND s.exercise = ? AND s.date >= ?""",
            (int(user_id), exercise, cutoff),
        ).fetchone()
        m = int(row["m"] or 0)
        return m if m > 0 else None
    finally:
        conn.close()


def _reps_cap(db: ExerVisionDB, user_id: int, exercise: str) -> int:
    recent = _user_recent_max_reps(db, user_id, exercise)
    if recent:
        return max(10, int(round(recent * 1.5)))
    return _DEFAULT_REP_CEILING.get(exercise, 30)


def sanitize_day_exercises(
    exs_in: Any,
    user_id: int,
    db: ExerVisionDB,
    *,
    label: str = "day",
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Clamp a single day's exercise list against KNOWN_EXERCISES, the
    per-exercise rep ceilings, and — when FORMA_PHASE3_ENABLED is True
    (default) — the exercise family's required shape.

    Shared by the chatbot's plan sanitizer AND the manual PATCH
    /api/plans/:id/days/:day_id route, so manual edits get the same rules
    as chatbot-proposed plans.

    Redesign Phase 3 output per family (stamps `family` on every row):
      rep_count  : {exercise, family, target_reps, target_sets, notes}
      weighted   : {exercise, family, target_weight_kg, target_reps,
                    target_sets, notes}  — squat allows target_weight_kg=0
                    (bodyweight); other weighted exercises drop on weight<=0
      time_hold  : {exercise, family, target_duration_sec, target_sets, notes}

    Phase-3 off (legacy): emits the old rep/sets-only shape.
    """
    from app.settings import PHASE3_ENABLED
    from app.exercise_registry import EXERCISE_FAMILIES, family_of

    warnings: List[str] = []
    sanitized: List[Dict[str, Any]] = []
    if not isinstance(exs_in, list):
        return sanitized, warnings

    if not PHASE3_ENABLED:
        # ── Legacy path: no family branching, reps+sets only ──────────
        for e in exs_in:
            if not isinstance(e, dict):
                continue
            ex_name = str(e.get("exercise") or "").strip().lower()
            if ex_name not in KNOWN_EXERCISES:
                warnings.append(f"{label}: unknown exercise '{ex_name}' removed")
                continue
            try:
                target_reps = int(e.get("target_reps") or 0)
            except (TypeError, ValueError):
                target_reps = 0
            try:
                target_sets = int(e.get("target_sets") or 0)
            except (TypeError, ValueError):
                target_sets = 0
            cap = _reps_cap(db, user_id, ex_name)
            if target_reps <= 0:
                target_reps = min(10, cap)
            if target_reps > cap:
                warnings.append(
                    f"{label} {ex_name}: target_reps {target_reps} "
                    f"clamped to {cap}"
                )
                target_reps = cap
            if target_sets <= 0:
                target_sets = 3
            if target_sets > 6:
                warnings.append(
                    f"{label} {ex_name}: target_sets {target_sets} clamped to 6"
                )
                target_sets = 6
            notes = str(e.get("notes") or "").strip()[:160]
            sanitized.append({
                "exercise": ex_name,
                "target_reps": target_reps,
                "target_sets": target_sets,
                "notes": notes,
            })
        return sanitized, warnings

    # ── Phase 3: family-aware branching ──────────────────────────────
    for e in exs_in:
        if not isinstance(e, dict):
            continue
        ex_name = str(e.get("exercise") or "").strip().lower()
        if ex_name not in EXERCISE_FAMILIES:
            warnings.append(f"{label}: unknown exercise '{ex_name}' removed")
            continue
        fam = family_of(ex_name)
        meta = EXERCISE_FAMILIES[ex_name]
        notes = str(e.get("notes") or "").strip()[:160]

        if fam == "rep_count":
            try:
                target_reps = int(e.get("target_reps") or 0)
            except (TypeError, ValueError):
                target_reps = 0
            try:
                target_sets = int(e.get("target_sets") or 0)
            except (TypeError, ValueError):
                target_sets = 0
            cap = _reps_cap(db, user_id, ex_name)
            if target_reps <= 0:
                target_reps = min(10, cap)
            if target_reps > cap:
                warnings.append(
                    f"{label} {ex_name}: target_reps {target_reps} "
                    f"clamped to {cap}"
                )
                target_reps = cap
            if target_sets <= 0:
                target_sets = meta["default_sets"]
            if target_sets > 6:
                warnings.append(
                    f"{label} {ex_name}: target_sets {target_sets} clamped to 6"
                )
                target_sets = 6
            sanitized.append({
                "exercise": ex_name,
                "family": fam,
                "target_reps": target_reps,
                "target_sets": target_sets,
                "notes": notes,
            })

        elif fam == "weighted":
            try:
                target_reps = int(e.get("target_reps") or 0)
            except (TypeError, ValueError):
                target_reps = 0
            try:
                target_sets = int(e.get("target_sets") or 0)
            except (TypeError, ValueError):
                target_sets = 0
            try:
                target_weight = float(e.get("target_weight_kg") or 0)
            except (TypeError, ValueError):
                target_weight = 0.0
            if target_weight < 0:
                target_weight = 0.0
            # squat is weight_optional — target_weight_kg=0 is legal
            # (bodyweight mode). Every other weighted exercise must carry
            # a positive weight; drop the row if missing.
            if not meta["weight_optional"] and target_weight <= 0:
                warnings.append(
                    f"{label} {ex_name}: missing target_weight_kg — dropped"
                )
                continue
            if target_reps <= 0:
                target_reps = 5
            if target_reps > 30:
                warnings.append(
                    f"{label} {ex_name}: target_reps {target_reps} clamped to 30"
                )
                target_reps = 30
            if target_sets <= 0:
                target_sets = meta["default_sets"]
            if target_sets > 8:
                warnings.append(
                    f"{label} {ex_name}: target_sets {target_sets} clamped to 8"
                )
                target_sets = 8
            # sanity-cap weight too
            if target_weight > 500:
                warnings.append(
                    f"{label} {ex_name}: target_weight_kg {target_weight} "
                    f"clamped to 500"
                )
                target_weight = 500.0
            sanitized.append({
                "exercise": ex_name,
                "family": fam,
                "target_weight_kg": round(target_weight, 2),
                "target_reps": target_reps,
                "target_sets": target_sets,
                "notes": notes,
            })

        elif fam == "time_hold":
            try:
                target_duration = int(e.get("target_duration_sec") or 0)
            except (TypeError, ValueError):
                target_duration = 0
            if target_duration <= 0:
                warnings.append(
                    f"{label} {ex_name}: missing target_duration_sec — dropped"
                )
                continue
            if target_duration < 5:
                target_duration = 5
            if target_duration > 600:
                warnings.append(
                    f"{label} {ex_name}: target_duration_sec "
                    f"{target_duration} clamped to 600"
                )
                target_duration = 600
            try:
                target_sets = int(e.get("target_sets") or 0)
            except (TypeError, ValueError):
                target_sets = 0
            if target_sets <= 0:
                target_sets = meta["default_sets"]
            sanitized.append({
                "exercise": ex_name,
                "family": fam,
                "target_duration_sec": target_duration,
                "target_sets": target_sets,
                "notes": notes,
            })
    return sanitized, warnings


def _sanitize_plan(
    draft: Dict[str, Any], user_id: int, db: ExerVisionDB
) -> Tuple[Dict[str, Any], List[str]]:
    """Normalize + clamp a plan draft. Raises PlanRejected on hard failures."""
    warnings: List[str] = []
    title = str(draft.get("title") or "Workout plan").strip()[:80]
    summary = str(draft.get("summary") or "").strip()[:500]
    start_date = str(draft.get("start_date") or _dt.now().strftime("%Y-%m-%d"))
    try:
        target_weeks = int(draft.get("target_weeks") or 0)
    except (TypeError, ValueError):
        target_weeks = 0
    days_in = draft.get("days") or []
    if not isinstance(days_in, list) or not days_in:
        raise PlanRejected("plan has no days")
    if len(days_in) > 60:
        raise PlanRejected("plan too long (max 60 days)")

    # Span check: if target_weeks is set, days MUST cover target_weeks * 7
    # consecutive calendar dates starting from start_date. This catches the
    # LLM-generates-one-template-week failure mode.
    if target_weeks > 0:
        expected = target_weeks * 7
        if len(days_in) != expected:
            raise PlanRejected(
                f"target_weeks={target_weeks} requires exactly {expected} "
                f"day entries, got {len(days_in)}. Generate one entry per "
                f"calendar date from {start_date} for {expected} days, "
                f"marking non-training days as is_rest=true."
            )
        try:
            start_dt = _dt.fromisoformat(start_date).date()
        except ValueError:
            raise PlanRejected(f"invalid start_date {start_date!r} (use YYYY-MM-DD)")
        expected_dates = {(start_dt + _td(days=i)).isoformat() for i in range(expected)}
        got_dates = {
            str((d or {}).get("day_date") or "").strip()
            for d in days_in if isinstance(d, dict)
        }
        missing = expected_dates - got_dates
        extra = got_dates - expected_dates
        if missing or extra:
            problems = []
            if missing:
                problems.append(f"missing dates: {sorted(missing)[:5]}")
            if extra:
                problems.append(f"unexpected dates: {sorted(extra)[:5]}")
            raise PlanRejected(
                f"day_dates do not form a contiguous {expected}-day span "
                f"starting {start_date}. {'; '.join(problems)}"
            )

    sanitized_days: List[Dict[str, Any]] = []
    for idx, d in enumerate(days_in):
        if not isinstance(d, dict):
            warnings.append(f"day {idx} skipped (not an object)")
            continue
        day_date = str(d.get("day_date") or "").strip()
        if not day_date:
            warnings.append(f"day {idx} missing day_date — skipped")
            continue
        is_rest = bool(d.get("is_rest"))
        if is_rest:
            sanitized_days.append(
                {
                    "day_date": day_date,
                    "is_rest": True,
                    "exercises": [],
                }
            )
            continue
        sanitized_exs, day_warnings = sanitize_day_exercises(
            d.get("exercises") or [], user_id, db, label=f"day {idx}"
        )
        warnings.extend(day_warnings)
        sanitized_days.append(
            {
                "day_date": day_date,
                "is_rest": False,
                "exercises": sanitized_exs,
            }
        )
    if not sanitized_days:
        raise PlanRejected("plan has no valid days after sanitization")
    out: Dict[str, Any] = {
        "title": title,
        "summary": summary,
        "start_date": start_date,
        "days": sanitized_days,
    }
    if target_weeks > 0:
        out["target_weeks"] = target_weeks
    return out, warnings


def _apply_modifications(
    draft: Dict[str, Any], mods: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply a `modifications` dict from the LLM to a draft in place.

    Supported keys (all optional):
      title, summary, start_date     — scalar replacements
      set_days                       — list of days (replaces days entirely)
      add_days                       — list of days appended at the end
      remove_day_dates               — list of day_date strings to drop
      swap_exercise                  — {"from": "squat", "to": "lunge"}
      scale_volume                   — float multiplier on target_reps
      set_target_weight              — {"exercise": "deadlift", "weight_kg": 80}
                                       (Phase 3 — weighted family)
      set_target_duration            — {"exercise": "plank", "seconds": 90}
                                       (Phase 3 — time_hold family)
    """
    if not isinstance(mods, dict):
        return draft
    out = copy.deepcopy(draft)
    if "title" in mods and mods["title"]:
        out["title"] = str(mods["title"])[:80]
    if "summary" in mods and mods["summary"]:
        out["summary"] = str(mods["summary"])[:500]
    if "start_date" in mods and mods["start_date"]:
        out["start_date"] = str(mods["start_date"])
    if "set_days" in mods and isinstance(mods["set_days"], list):
        out["days"] = mods["set_days"]
    if "add_days" in mods and isinstance(mods["add_days"], list):
        out.setdefault("days", []).extend(mods["add_days"])
    if "remove_day_dates" in mods and isinstance(mods["remove_day_dates"], list):
        drop = {str(d) for d in mods["remove_day_dates"]}
        out["days"] = [d for d in out.get("days", []) if d.get("day_date") not in drop]
    swap = mods.get("swap_exercise") or {}
    if isinstance(swap, dict) and swap.get("from") and swap.get("to"):
        fr = str(swap["from"]).lower()
        to = str(swap["to"]).lower()
        for day in out.get("days", []):
            for e in day.get("exercises", []) or []:
                if e.get("exercise") == fr:
                    e["exercise"] = to
    scale = mods.get("scale_volume")
    try:
        scale_f = float(scale) if scale is not None else None
    except (TypeError, ValueError):
        scale_f = None
    if scale_f and scale_f > 0:
        for day in out.get("days", []):
            for e in day.get("exercises", []) or []:
                try:
                    e["target_reps"] = max(
                        1, int(round(float(e.get("target_reps", 0)) * scale_f))
                    )
                except (TypeError, ValueError):
                    pass
    # Phase 3 — family-specific setters.
    stw = mods.get("set_target_weight") or {}
    if isinstance(stw, dict) and stw.get("exercise") and stw.get("weight_kg") is not None:
        target_ex = str(stw["exercise"]).lower()
        try:
            target_w = float(stw["weight_kg"])
        except (TypeError, ValueError):
            target_w = None
        if target_w is not None and target_w >= 0:
            for day in out.get("days", []):
                for e in day.get("exercises", []) or []:
                    if e.get("exercise") == target_ex:
                        e["target_weight_kg"] = target_w
    std = mods.get("set_target_duration") or {}
    if isinstance(std, dict) and std.get("exercise") and std.get("seconds") is not None:
        target_ex = str(std["exercise"]).lower()
        try:
            target_s = int(std["seconds"])
        except (TypeError, ValueError):
            target_s = None
        if target_s is not None and target_s > 0:
            for day in out.get("days", []):
                for e in day.get("exercises", []) or []:
                    if e.get("exercise") == target_ex:
                        e["target_duration_sec"] = target_s
    return out


def _plan_volume_by_exercise(plan: Dict[str, Any]) -> Dict[str, int]:
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


def _plan_active_days(plan: Dict[str, Any]) -> int:
    return sum(1 for d in plan.get("days", []) if not d.get("is_rest"))


def _draft_read(db: ExerVisionDB, user_id: int, conversation_id: int) -> Optional[str]:
    """Phase 3: read from per-user drafts first, fall back to legacy
    conversation-scoped drafts so mid-migration conversations don't
    appear to have lost their draft."""
    from app.settings import PHASE3_ENABLED

    if PHASE3_ENABLED:
        row = db.get_user_plan_draft(user_id)
        if row and row.get("draft_json"):
            return row["draft_json"]
    return db.get_conversation_plan_draft(conversation_id)


def _draft_write(
    db: ExerVisionDB,
    user_id: int,
    conversation_id: int,
    draft_json: Optional[str],
) -> None:
    from app.settings import PHASE3_ENABLED

    if PHASE3_ENABLED:
        if draft_json is None:
            db.clear_user_plan_draft(user_id)
        else:
            db.set_user_plan_draft(
                user_id,
                draft_json,
                source="chat",
                conversation_id=conversation_id,
            )
    # Also clear the legacy slot so old clients don't read a stale draft.
    db.set_conversation_plan_draft(conversation_id, draft_json)


def propose_plan(
    user_id: int,
    db: ExerVisionDB,
    conversation_id: int,
    *,
    title: str,
    start_date: str,
    days: list,
    target_weeks: int,
    summary: str = "",
) -> Dict[str, Any]:
    draft_in = {
        "title": title,
        "summary": summary,
        "start_date": start_date,
        "target_weeks": target_weeks,
        "days": days,
    }
    try:
        sanitized, warnings = _sanitize_plan(draft_in, user_id, db)
    except PlanRejected as e:
        return {"error": "plan_rejected", "message": str(e)}
    _draft_write(db, user_id, conversation_id, _json.dumps(sanitized))
    return {"draft": sanitized, "warnings": warnings}


def revise_plan(
    user_id: int,
    db: ExerVisionDB,
    conversation_id: int,
    *,
    modifications: Dict[str, Any],
) -> Dict[str, Any]:
    raw = _draft_read(db, user_id, conversation_id)
    if not raw:
        return {
            "error": "no_draft",
            "message": "Call propose_plan before revise_plan.",
        }
    try:
        draft = _json.loads(raw)
    except _json.JSONDecodeError:
        return {"error": "corrupt_draft"}
    modified = _apply_modifications(draft, modifications or {})
    try:
        sanitized, warnings = _sanitize_plan(modified, user_id, db)
    except PlanRejected as e:
        return {"error": "plan_rejected", "message": str(e)}
    _draft_write(db, user_id, conversation_id, _json.dumps(sanitized))
    return {"draft": sanitized, "warnings": warnings}


def save_plan(
    user_id: int,
    db: ExerVisionDB,
    conversation_id: int,
    *,
    title: Optional[str] = None,
    summary: Optional[str] = None,
) -> Dict[str, Any]:
    raw = _draft_read(db, user_id, conversation_id)
    if not raw:
        return {"error": "no_draft", "message": "Nothing to save — call propose_plan first."}
    try:
        draft = _json.loads(raw)
    except _json.JSONDecodeError:
        return {"error": "corrupt_draft"}
    if not draft.get("days"):
        return {"error": "empty_plan"}
    plan_title = (title or draft.get("title") or "Workout plan").strip()[:80]
    plan_summary = (summary or draft.get("summary") or "").strip()[:500]
    start_date = draft.get("start_date") or _dt.now().strftime("%Y-%m-%d")
    end_date = draft["days"][-1].get("day_date") or start_date
    plan_id = db.create_plan(
        user_id,
        title=plan_title,
        summary=plan_summary,
        start_date=start_date,
        end_date=end_date,
        created_by_chat=True,
        conversation_id=conversation_id,
    )
    for d in draft["days"]:
        db.create_plan_day(
            plan_id,
            d["day_date"],
            _json.dumps(d.get("exercises") or []),
            is_rest=bool(d.get("is_rest")),
        )
    # Clear drafts (both scopes) so nothing lingers.
    _draft_write(db, user_id, conversation_id, None)
    saved_exercises = sorted({
        ex.get("exercise")
        for d in draft["days"]
        for ex in (d.get("exercises") or [])
        if ex.get("exercise")
    })
    return {
        "plan_id": plan_id,
        "title": plan_title,
        "days_saved": len(draft["days"]),
        "saved_summary": {
            "exercises": saved_exercises,
            "day_count": len(draft["days"]),
            "first_day": draft["days"][0].get("day_date"),
            "last_day": draft["days"][-1].get("day_date"),
        },
    }


def create_plan_goals(
    user_id: int, db: ExerVisionDB, *, plan_id: int
) -> Dict[str, Any]:
    """Auto-generate volume + consistency + plan_progress goals for a saved plan.

    Thin wrapper over goal_engine.create_plan_goals_for_plan so the REST path
    (POST /api/plans) and the chatbot tool share the same logic.
    """
    from app import goal_engine as _ge

    return _ge.create_plan_goals_for_plan(user_id, plan_id, db=db)


# OpenAI function schemas for the plan-creator chatbot.
#
# Redesign Phase 3: per-exercise fields are now family-aware —
#   rep_count (pushup/pullup/lunge/bicep_curl/tricep_dip):
#     target_reps + target_sets
#   weighted (squat/bench_press/deadlift/overhead_press):
#     target_weight_kg + target_reps + target_sets
#     (squat allows target_weight_kg=0 for bodyweight)
#   time_hold (plank):
#     target_duration_sec (target_sets optional, default 1)
# The sanitizer (sanitize_day_exercises) enforces per-family requirements
# at the server — these schemas only loosely hint; validation is real.
_PLAN_NEW_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "propose_plan",
            "description": (
                "Draft a workout plan as a list of days. Returns the sanitized "
                "draft so the frontend renders a live preview card. Does NOT "
                "save the plan to the database — call save_plan for that. "
                "Per-exercise fields are family-dependent (see system prompt): "
                "rep_count uses target_reps+target_sets, weighted adds "
                "target_weight_kg, time_hold uses target_duration_sec only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "start_date": {
                        "type": "string",
                        "description": "ISO date (YYYY-MM-DD) for day 1",
                    },
                    "target_weeks": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                        "description": (
                            "Planned duration in weeks. The days array MUST "
                            "contain exactly target_weeks * 7 consecutive "
                            "calendar dates starting from start_date. "
                            "Non-training days are entries with is_rest=true. "
                            "The server rejects the call if the span is wrong."
                        ),
                    },
                    "days": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "day_date": {"type": "string"},
                                "is_rest": {"type": "boolean"},
                                "exercises": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "exercise": {
                                                "type": "string",
                                                "enum": sorted(KNOWN_EXERCISES),
                                            },
                                            "target_reps": {
                                                "type": "integer",
                                                "description": "rep_count + weighted only",
                                            },
                                            "target_sets": {"type": "integer"},
                                            "target_weight_kg": {
                                                "type": "number",
                                                "description": (
                                                    "weighted only. 0 means "
                                                    "bodyweight (squat only)."
                                                ),
                                            },
                                            "target_duration_sec": {
                                                "type": "integer",
                                                "description": "time_hold only (plank)",
                                            },
                                            "notes": {"type": "string"},
                                        },
                                        "required": ["exercise"],
                                    },
                                },
                            },
                            "required": ["day_date", "is_rest", "exercises"],
                        },
                    },
                },
                "required": ["title", "start_date", "target_weeks", "days"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "revise_plan",
            "description": (
                "Update the current plan draft using a modifications object. "
                "Supported keys: title, summary, start_date, set_days, add_days, "
                "remove_day_dates, swap_exercise ({from,to}), scale_volume (float), "
                "set_target_weight ({exercise, weight_kg}), "
                "set_target_duration ({exercise, seconds})."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "modifications": {
                        "type": "object",
                        "description": "Patch-style changes to apply",
                    }
                },
                "required": ["modifications"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_plan",
            "description": (
                "Persist the current sanitized draft as a real plan. Call this "
                "only after the user has explicitly approved the preview. "
                "Server will also auto-create the two plan-level goals "
                "(plan_progress + consistency). Per-exercise goals are opt-in "
                "— ask the user and call create_goal for each they want."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_goal",
            "description": (
                "Create ONE goal for the user. Call after save_plan if the "
                "user agrees to track a specific goal (e.g. 'yes track my "
                "pushup volume and deadlift strength'). Pick goal_type by "
                "exercise family: rep_count→volume, weighted→strength, "
                "time_hold→duration. plan_id is optional — include it to "
                "link the goal to the just-saved plan so it archives when "
                "the plan completes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "goal_type": {
                        "type": "string",
                        "enum": [
                            "volume", "strength", "duration",
                            "consistency", "quality",
                        ],
                    },
                    "target_value": {"type": "number"},
                    "unit": {
                        "type": "string",
                        "enum": [
                            "reps", "kg", "seconds",
                            "sessions", "days", "form_score",
                        ],
                    },
                    "exercise": {
                        "type": "string",
                        "enum": sorted(KNOWN_EXERCISES),
                    },
                    "target_reps": {
                        "type": "integer",
                        "description": "required for strength goals",
                    },
                    "description": {"type": "string"},
                    "period": {"type": "string", "enum": ["once", "week", "month"]},
                    "plan_id": {"type": "integer"},
                },
                "required": ["title", "goal_type", "target_value", "unit"],
            },
        },
    },
]

PLAN_TOOL_SCHEMAS: List[Dict[str, Any]] = PERSONAL_TOOL_SCHEMAS + _PLAN_NEW_SCHEMAS


def make_plan_dispatcher(
    user_id: int,
    db: ExerVisionDB,
    conversation_id: int,
    *,
    on_plan_saved: Optional[Callable[[int], None]] = None,
) -> Callable[[str, Dict[str, Any]], Any]:
    """Like make_dispatcher, but also handles the plan-creator tools.

    conversation_id is closed over so the plan draft lives on the conversation
    row — the LLM cannot forge it and revise_plan has something to mutate
    between turns.

    If `on_plan_saved` is provided, it is invoked with the new plan_id after a
    successful save_plan tool call so the HTTP layer can emit a socket event.
    """
    base = make_dispatcher(user_id, db)

    def dispatch(fn_name: str, args: Dict[str, Any]) -> Any:
        if fn_name == "propose_plan":
            try:
                tw = args.get("target_weeks")
                try:
                    tw = int(tw) if tw is not None else 0
                except (TypeError, ValueError):
                    tw = 0
                if tw < 1 or tw > 8:
                    return {
                        "error": "plan_rejected",
                        "message": (
                            "target_weeks must be an integer 1-8. Re-collect "
                            "TIMEFRAME_WEEKS from the user and retry."
                        ),
                    }
                return propose_plan(
                    user_id, db, conversation_id,
                    title=args.get("title") or "Workout plan",
                    summary=args.get("summary") or "",
                    start_date=args.get("start_date")
                    or _dt.now().strftime("%Y-%m-%d"),
                    days=args.get("days") or [],
                    target_weeks=tw,
                )
            except PlanRejected as e:
                return {"error": "plan_rejected", "message": str(e)}
        if fn_name == "revise_plan":
            return revise_plan(
                user_id, db, conversation_id,
                modifications=args.get("modifications") or {},
            )
        if fn_name == "save_plan":
            result = save_plan(
                user_id, db, conversation_id,
                title=args.get("title"),
                summary=args.get("summary"),
            )
            if on_plan_saved is not None and isinstance(result, dict):
                pid = result.get("plan_id")
                if isinstance(pid, int) and pid > 0:
                    try:
                        on_plan_saved(pid)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("on_plan_saved hook failed: %s", e)
            return result
        if fn_name == "create_goal":
            # Redesign Phase 3: post-save per-exercise goal creation, opt-in.
            # Wraps goal_engine.create_goal; returns error codes on cap /
            # validation so the LLM can recover.
            from app import goal_engine as _ge

            kwargs: Dict[str, Any] = {
                "title": args.get("title") or "",
                "goal_type": args.get("goal_type") or "",
                "target_value": args.get("target_value"),
                "unit": args.get("unit") or "",
            }
            for k in ("exercise", "description", "period"):
                if args.get(k) is not None:
                    kwargs[k] = args.get(k)
            if args.get("target_reps") is not None:
                try:
                    kwargs["target_reps"] = int(args.get("target_reps"))
                except (TypeError, ValueError):
                    pass
            if args.get("plan_id") is not None:
                try:
                    kwargs["plan_id"] = int(args.get("plan_id"))
                except (TypeError, ValueError):
                    pass
            try:
                g = _ge.create_goal(user_id, db=db, **kwargs)
                return {"goal": g.to_dict()}
            except _ge.GoalCapExceeded as e:
                return {"error": "goal_cap", "message": str(e)}
            except _ge.InvalidGoal as e:
                return {"error": "invalid_goal", "message": str(e)}
        return base(fn_name, args)

    return dispatch
