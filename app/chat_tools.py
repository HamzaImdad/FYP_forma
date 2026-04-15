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


# ── Session 4: plan-creator tools ─────────────────────────────────────

import copy
import json as _json
from datetime import datetime as _dt, timedelta as _td

KNOWN_EXERCISES: set = {
    "squat", "lunge", "deadlift", "bench_press", "overhead_press",
    "pullup", "pushup", "plank", "bicep_curl", "tricep_dip",
}

# Per-exercise rep ceilings when we have no user history to anchor against.
_DEFAULT_REP_CEILING: Dict[str, int] = {
    "pushup": 60,
    "squat": 60,
    "lunge": 60,
    "bicep_curl": 60,
    "tricep_dip": 60,
    "pullup": 20,
    "bench_press": 30,
    "overhead_press": 30,
    "deadlift": 30,
    "plank": 300,  # seconds, not reps — used when exercise is plank
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


def _sanitize_plan(
    draft: Dict[str, Any], user_id: int, db: ExerVisionDB
) -> Tuple[Dict[str, Any], List[str]]:
    """Normalize + clamp a plan draft. Raises PlanRejected on hard failures."""
    warnings: List[str] = []
    title = str(draft.get("title") or "Workout plan").strip()[:80]
    summary = str(draft.get("summary") or "").strip()[:500]
    start_date = str(draft.get("start_date") or _dt.now().strftime("%Y-%m-%d"))
    days_in = draft.get("days") or []
    if not isinstance(days_in, list) or not days_in:
        raise PlanRejected("plan has no days")
    if len(days_in) > 20:
        raise PlanRejected("plan too long (max 20 days)")

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
        exs_in = d.get("exercises") or []
        if is_rest:
            sanitized_days.append(
                {
                    "day_date": day_date,
                    "is_rest": True,
                    "exercises": [],
                }
            )
            continue
        sanitized_exs: List[Dict[str, Any]] = []
        for e in exs_in:
            if not isinstance(e, dict):
                continue
            ex_name = str(e.get("exercise") or "").strip().lower()
            if ex_name not in KNOWN_EXERCISES:
                warnings.append(
                    f"day {idx}: unknown exercise '{ex_name}' removed"
                )
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
                    f"day {idx} {ex_name}: target_reps {target_reps} "
                    f"clamped to {cap}"
                )
                target_reps = cap
            if target_sets <= 0:
                target_sets = 3
            if target_sets > 6:
                warnings.append(
                    f"day {idx} {ex_name}: target_sets {target_sets} clamped to 6"
                )
                target_sets = 6
            notes = str(e.get("notes") or "").strip()[:160]
            sanitized_exs.append(
                {
                    "exercise": ex_name,
                    "target_reps": target_reps,
                    "target_sets": target_sets,
                    "notes": notes,
                }
            )
        sanitized_days.append(
            {
                "day_date": day_date,
                "is_rest": False,
                "exercises": sanitized_exs,
            }
        )
    if not sanitized_days:
        raise PlanRejected("plan has no valid days after sanitization")
    return (
        {
            "title": title,
            "summary": summary,
            "start_date": start_date,
            "days": sanitized_days,
        },
        warnings,
    )


def _apply_modifications(
    draft: Dict[str, Any], mods: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply a `modifications` dict from the LLM to a draft in place.

    Supported keys (all optional):
      title, summary, start_date — scalar replacements
      set_days         — list of days (replaces days entirely)
      add_days         — list of days appended at the end
      remove_day_dates — list of day_date strings to drop
      swap_exercise    — {"from": "squat", "to": "lunge"}
      scale_volume     — float multiplier on target_reps across the plan
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
                    e["target_reps"] = max(1, int(round(float(e.get("target_reps", 0)) * scale_f)))
                except (TypeError, ValueError):
                    pass
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


def propose_plan(
    user_id: int,
    db: ExerVisionDB,
    conversation_id: int,
    *,
    title: str,
    start_date: str,
    days: list,
    summary: str = "",
) -> Dict[str, Any]:
    draft_in = {
        "title": title,
        "summary": summary,
        "start_date": start_date,
        "days": days,
    }
    try:
        sanitized, warnings = _sanitize_plan(draft_in, user_id, db)
    except PlanRejected as e:
        return {"error": "plan_rejected", "message": str(e)}
    db.set_conversation_plan_draft(conversation_id, _json.dumps(sanitized))
    return {"draft": sanitized, "warnings": warnings}


def revise_plan(
    user_id: int,
    db: ExerVisionDB,
    conversation_id: int,
    *,
    modifications: Dict[str, Any],
) -> Dict[str, Any]:
    raw = db.get_conversation_plan_draft(conversation_id)
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
    db.set_conversation_plan_draft(conversation_id, _json.dumps(sanitized))
    return {"draft": sanitized, "warnings": warnings}


def save_plan(
    user_id: int,
    db: ExerVisionDB,
    conversation_id: int,
    *,
    title: Optional[str] = None,
    summary: Optional[str] = None,
) -> Dict[str, Any]:
    raw = db.get_conversation_plan_draft(conversation_id)
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
    # Clear the draft so revise_plan can't accidentally re-save it.
    db.set_conversation_plan_draft(conversation_id, None)
    return {
        "plan_id": plan_id,
        "title": plan_title,
        "days_saved": len(draft["days"]),
    }


def create_plan_goals(
    user_id: int, db: ExerVisionDB, *, plan_id: int
) -> Dict[str, Any]:
    """Auto-generate one volume goal per distinct exercise + one consistency goal."""
    plan = db.get_plan_full(plan_id, user_id)
    if not plan:
        return {"error": "not_found"}
    from app import goal_engine as _ge

    created: List[Dict[str, Any]] = []
    volumes = _plan_volume_by_exercise(plan)
    for ex, total_reps in sorted(volumes.items()):
        try:
            g = _ge.create_goal(
                user_id,
                title=f"{plan['title']} — {ex.replace('_', ' ')}",
                goal_type="volume",
                target_value=float(total_reps),
                unit="reps",
                exercise=ex,
                period="once",
                description=f"Finish all the {ex.replace('_', ' ')} volume in this plan.",
                db=db,
            )
            created.append(g.to_dict())
        except _ge.GoalCapExceeded:
            return {
                "error": "goal_cap",
                "message": "Goal cap reached — couldn't create every plan goal.",
                "created": created,
            }
        except _ge.InvalidGoal as e:
            created.append({"error": str(e), "exercise": ex})
    # Plus one consistency goal (active days over the plan window)
    try:
        active_days = _plan_active_days(plan)
        if active_days > 0:
            g = _ge.create_goal(
                user_id,
                title=f"{plan['title']} — finish every workout day",
                goal_type="consistency",
                target_value=float(active_days),
                unit="sessions",
                period="once",
                description="Complete every non-rest day in this plan.",
                db=db,
            )
            created.append(g.to_dict())
    except _ge.GoalCapExceeded:
        pass
    except _ge.InvalidGoal:
        pass
    return {"goals_created": created}


# OpenAI function schemas for the plan-creator chatbot
_PLAN_NEW_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "propose_plan",
            "description": (
                "Draft a workout plan as a list of days. Returns the sanitized "
                "draft so the frontend renders a live preview card. Does NOT "
                "save the plan to the database — call save_plan for that."
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
                                            "target_reps": {"type": "integer"},
                                            "target_sets": {"type": "integer"},
                                            "notes": {"type": "string"},
                                        },
                                        "required": [
                                            "exercise",
                                            "target_reps",
                                            "target_sets",
                                        ],
                                    },
                                },
                            },
                            "required": ["day_date", "is_rest", "exercises"],
                        },
                    },
                },
                "required": ["title", "start_date", "days"],
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
                "remove_day_dates, swap_exercise ({from,to}), scale_volume (float)."
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
                "only after the user has explicitly approved the preview."
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
            "name": "create_plan_goals",
            "description": (
                "For a saved plan, auto-create one volume goal per exercise + "
                "one consistency goal. Call after save_plan."
            ),
            "parameters": {
                "type": "object",
                "properties": {"plan_id": {"type": "integer"}},
                "required": ["plan_id"],
            },
        },
    },
]

PLAN_TOOL_SCHEMAS: List[Dict[str, Any]] = PERSONAL_TOOL_SCHEMAS + _PLAN_NEW_SCHEMAS


def make_plan_dispatcher(
    user_id: int, db: ExerVisionDB, conversation_id: int
) -> Callable[[str, Dict[str, Any]], Any]:
    """Like make_dispatcher, but also handles the plan-creator tools.

    conversation_id is closed over so the plan draft lives on the conversation
    row — the LLM cannot forge it and revise_plan has something to mutate
    between turns.
    """
    base = make_dispatcher(user_id, db)

    def dispatch(fn_name: str, args: Dict[str, Any]) -> Any:
        if fn_name == "propose_plan":
            try:
                return propose_plan(
                    user_id, db, conversation_id,
                    title=args.get("title") or "Workout plan",
                    summary=args.get("summary") or "",
                    start_date=args.get("start_date")
                    or _dt.now().strftime("%Y-%m-%d"),
                    days=args.get("days") or [],
                )
            except PlanRejected as e:
                return {"error": "plan_rejected", "message": str(e)}
        if fn_name == "revise_plan":
            return revise_plan(
                user_id, db, conversation_id,
                modifications=args.get("modifications") or {},
            )
        if fn_name == "save_plan":
            return save_plan(
                user_id, db, conversation_id,
                title=args.get("title"),
                summary=args.get("summary"),
            )
        if fn_name == "create_plan_goals":
            try:
                plan_id = int(args.get("plan_id") or 0)
            except (TypeError, ValueError):
                return {"error": "invalid_plan_id"}
            if plan_id <= 0:
                return {"error": "invalid_plan_id"}
            return create_plan_goals(user_id, db, plan_id=plan_id)
        return base(fn_name, args)

    return dispatch
