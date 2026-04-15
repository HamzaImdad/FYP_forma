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
from typing import Any, Callable, Dict, List, Optional

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
            "description": "List the user's most recent workout sessions, newest first.",
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
                "owns. Returns {error: 'not_found'} if the id is not in "
                "their history — never leak existence of other users' sessions."
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
    # Session 4 populates the goals table. Return an empty list so the
    # plan-creator chatbot can still call the tool without crashing.
    return {"goals": [], "note": "Goals system arrives in Session 4."}


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
        return {"error": "unknown_tool", "name": fn_name}

    return dispatch
