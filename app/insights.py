"""
FORMA insights engine.

Stateless rule-based analysis of a user's training history.  Produces a list
of `Insight` dataclass instances that power the dashboard's insight card, the
per-exercise deep dive, and (in Session 3) the personal chatbot's
`get_insights` tool.  No Flask coupling — just takes a `user_id` and the
`ExerVisionDB` instance.  Insights are never cached; they're re-generated on
every request so the user always sees the latest signal.

Import surface (Session 3 will call these):

    from app.insights import (
        Insight,
        generate_insights,
        MUSCLE_GROUP_MAP,
        resolve_muscle_groups,
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, date
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Muscle group lookup ─────────────────────────────────────────────────

# Raw muscle names live in app/exercise_data.json.  Map every distinct name to
# one of six large groups that the radar / imbalance rules care about.
MUSCLE_GROUP_MAP: Dict[str, str] = {
    # chest
    "chest": "chest",
    "upper chest": "chest",
    "pectoralis major": "chest",
    "pecs": "chest",
    # back
    "lats": "back",
    "upper back": "back",
    "lower back": "back",
    "traps": "back",
    "rhomboids": "back",
    # shoulders
    "shoulders": "shoulders",
    "deltoids": "shoulders",
    "rear delts": "shoulders",
    # arms
    "biceps": "arms",
    "triceps": "arms",
    "forearms": "arms",
    # legs
    "quadriceps": "legs",
    "quads": "legs",
    "hamstrings": "legs",
    "glutes": "legs",
    "calves": "legs",
    "adductors": "legs",
    # core
    "core": "core",
    "abs": "core",
    "obliques": "core",
    "abdominals": "core",
}

_ALL_GROUPS = ("chest", "back", "shoulders", "arms", "legs", "core")


def _group_for(raw: str) -> Optional[str]:
    if not raw:
        return None
    return MUSCLE_GROUP_MAP.get(raw.strip().lower())


@lru_cache(maxsize=1)
def _load_exercise_data() -> Dict[str, Any]:
    path = Path(__file__).parent / "exercise_data.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_muscle_groups(exercise: str) -> List[str]:
    """Return the distinct big-muscle groups an exercise hits."""
    meta = _load_exercise_data().get(exercise, {})
    raw = meta.get("muscles", []) or []
    out: List[str] = []
    for m in raw:
        g = _group_for(m)
        if g and g not in out:
            out.append(g)
    return out


# ── Insight data class ──────────────────────────────────────────────────


@dataclass
class Insight:
    id: str  # stable id like "fatigue_set3_pushup"
    category: str  # progress | weakness | consistency | volume | recovery | milestone
    severity: str  # info | notice | warn | celebrate
    text: str  # plain-English sentence
    exercise: Optional[str]  # None = cross-exercise
    source_session_ids: List[int] = field(default_factory=list)
    source_rep_ids: List[int] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_SEVERITY_RANK = {"celebrate": 0, "warn": 1, "notice": 2, "info": 3}
_CATEGORY_RANK = {
    "weakness": 0,
    "progress": 1,
    "consistency": 2,
    "volume": 3,
    "recovery": 4,
    "milestone": 5,
}


# ── Data context (single DB load for all rules) ────────────────────────


@dataclass
class InsightContext:
    user_id: int
    now: datetime
    sessions: List[Dict[str, Any]]  # last 90 days + a 'date_dt' helper
    reps: List[Dict[str, Any]]  # every rep in those sessions
    sets: List[Dict[str, Any]]  # every set in those sessions
    all_session_count: int  # total sessions for the user across all time
    all_time_best: Dict[str, Any]  # PR baseline across all time
    exercise_filter: Optional[str]


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        try:
            return datetime.fromisoformat(s.split(".")[0])
        except ValueError:
            return None


def _build_context(db, user_id: int, exercise: Optional[str]) -> InsightContext:
    """Load ~90 days of session data in one shot, plus all-time PR baseline."""
    now = datetime.now()
    lookback = (now - timedelta(days=90)).isoformat()

    conn = db._get_conn()
    try:
        # Recent sessions
        q = "SELECT * FROM sessions WHERE user_id = ? AND date >= ?"
        params: List[Any] = [user_id, lookback]
        if exercise:
            q += " AND exercise = ?"
            params.append(exercise)
        q += " ORDER BY date ASC"
        rows = conn.execute(q, params).fetchall()
        sessions: List[Dict[str, Any]] = []
        for r in rows:
            s = dict(r)
            s["date_dt"] = _parse_dt(s.get("date"))
            try:
                s["muscle_groups_list"] = json.loads(s.get("muscle_groups") or "[]")
            except json.JSONDecodeError:
                s["muscle_groups_list"] = []
            sessions.append(s)

        session_ids = [s["id"] for s in sessions]

        reps: List[Dict[str, Any]] = []
        sets: List[Dict[str, Any]] = []
        if session_ids:
            placeholders = ",".join("?" * len(session_ids))
            rep_rows = conn.execute(
                f"SELECT * FROM reps WHERE session_id IN ({placeholders}) "
                f"ORDER BY session_id ASC, rep_num ASC",
                session_ids,
            ).fetchall()
            for rr in rep_rows:
                d = dict(rr)
                try:
                    d["issues_list"] = json.loads(d.get("issues") or "[]")
                except json.JSONDecodeError:
                    d["issues_list"] = []
                reps.append(d)
            set_rows = conn.execute(
                f"SELECT * FROM sets WHERE session_id IN ({placeholders}) "
                f"ORDER BY session_id ASC, set_num ASC",
                session_ids,
            ).fetchall()
            sets = [dict(s) for s in set_rows]

        # All-time totals + PR baseline
        total_row = conn.execute(
            "SELECT COUNT(*) AS n FROM sessions WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        all_session_count = int(total_row["n"] if total_row else 0)

        pr_row = conn.execute(
            """SELECT
                   MAX(total_reps)       AS max_reps,
                   MAX(avg_form_score)   AS max_form,
                   MAX(duration_sec)     AS max_duration
               FROM sessions WHERE user_id = ?""",
            (user_id,),
        ).fetchone()
        best: Dict[str, Any] = dict(pr_row) if pr_row else {}
        # Longest plank specifically
        plank_row = conn.execute(
            """SELECT id, duration_sec, avg_form_score, date
               FROM sessions
               WHERE user_id = ? AND exercise = 'plank'
               ORDER BY duration_sec DESC LIMIT 1""",
            (user_id,),
        ).fetchone()
        if plank_row:
            best["longest_plank_sec"] = plank_row["duration_sec"]
            best["longest_plank_id"] = plank_row["id"]
    finally:
        conn.close()

    return InsightContext(
        user_id=user_id,
        now=now,
        sessions=sessions,
        reps=reps,
        sets=sets,
        all_session_count=all_session_count,
        all_time_best=best,
        exercise_filter=exercise,
    )


# ── Helpers ──────────────────────────────────────────────────────────────


def _score_pts(score: Optional[float]) -> float:
    """Normalize a stored form score (0-1 scale) to a 0-100 integer feel."""
    if score is None:
        return 0.0
    s = float(score)
    if s <= 1.0:
        return s * 100.0
    return s


def _mean(xs: Iterable[float]) -> Optional[float]:
    data = [x for x in xs if x is not None]
    if not data:
        return None
    return sum(data) / len(data)


def _slope(xs: List[float]) -> float:
    """Unweighted linear regression slope of a series. Returns 0 on short series."""
    n = len(xs)
    if n < 2:
        return 0.0
    try:
        import numpy as np  # type: ignore

        return float(np.polyfit(range(n), xs, 1)[0])
    except Exception:
        mean_x = (n - 1) / 2
        mean_y = sum(xs) / n
        num = sum((i - mean_x) * (y - mean_y) for i, y in enumerate(xs))
        den = sum((i - mean_x) ** 2 for i in range(n))
        return num / den if den else 0.0


def _reps_for_session(ctx: InsightContext, session_id: int) -> List[Dict]:
    return [r for r in ctx.reps if r["session_id"] == session_id]


def _sets_for_session(ctx: InsightContext, session_id: int) -> List[Dict]:
    return [s for s in ctx.sets if s["session_id"] == session_id]


def _iso() -> str:
    return datetime.now().isoformat()


# ── Rules ────────────────────────────────────────────────────────────────


def rule_first_session(ctx: InsightContext) -> Optional[Insight]:
    if ctx.all_session_count != 1:
        return None
    # Only fires if the single session is in the window
    if not ctx.sessions:
        return None
    s = ctx.sessions[-1]
    reps = int(s.get("total_reps") or 0)
    if s.get("exercise") == "plank":
        secs = int(s.get("duration_sec") or 0)
        txt = f"Nice start — your first FORMA session is in the books ({secs}s hold)."
    else:
        txt = f"Nice start — {reps} reps on your first FORMA session."
    return Insight(
        id="first_session",
        category="milestone",
        severity="celebrate",
        text=txt,
        exercise=s.get("exercise"),
        source_session_ids=[int(s["id"])],
        data={"reps": reps, "exercise": s.get("exercise")},
        created_at=_iso(),
    )


def rule_streak(ctx: InsightContext) -> Optional[Insight]:
    """Consecutive days with ≥1 session, ending today or yesterday."""
    if ctx.all_session_count < 2:
        return None
    dates = sorted(
        {s["date_dt"].date() for s in ctx.sessions if s.get("date_dt")},
        reverse=True,
    )
    if not dates:
        return None
    today = ctx.now.date()
    # A streak counts if the latest session is today or yesterday
    if (today - dates[0]) > timedelta(days=1):
        return None
    streak = 1
    cursor = dates[0]
    for d in dates[1:]:
        if cursor - d == timedelta(days=1):
            streak += 1
            cursor = d
        else:
            break
    if streak < 2:
        return None
    return Insight(
        id="streak",
        category="consistency",
        severity="celebrate" if streak >= 3 else "info",
        text=f"{streak} days in a row — momentum is yours.",
        exercise=None,
        data={"streak_days": streak},
        created_at=_iso(),
    )


def rule_regression(ctx: InsightContext) -> Optional[Insight]:
    this_week, last_week = _split_weeks(ctx)
    if len(this_week) < 3 or len(last_week) < 3:
        return None
    a = _mean([_score_pts(s["avg_form_score"]) for s in this_week]) or 0
    b = _mean([_score_pts(s["avg_form_score"]) for s in last_week]) or 0
    drop = b - a
    if drop < 5:
        return None
    return Insight(
        id="regression_week",
        category="progress",
        severity="warn",
        text=f"Form slipped {drop:.0f} points this week vs last — take a closer look at your set technique.",
        exercise=ctx.exercise_filter,
        source_session_ids=[int(s["id"]) for s in this_week],
        data={"this_week_avg": round(a, 1), "last_week_avg": round(b, 1), "drop": round(drop, 1)},
        created_at=_iso(),
    )


def rule_improvement(ctx: InsightContext) -> Optional[Insight]:
    this_week, last_week = _split_weeks(ctx)
    if len(this_week) < 3 or len(last_week) < 3:
        return None
    a = _mean([_score_pts(s["avg_form_score"]) for s in this_week]) or 0
    b = _mean([_score_pts(s["avg_form_score"]) for s in last_week]) or 0
    gain = a - b
    if gain < 5:
        return None
    return Insight(
        id="improvement_week",
        category="progress",
        severity="celebrate",
        text=f"Form up {gain:.0f} points this week — you're going in the right direction.",
        exercise=ctx.exercise_filter,
        source_session_ids=[int(s["id"]) for s in this_week],
        data={"this_week_avg": round(a, 1), "last_week_avg": round(b, 1), "gain": round(gain, 1)},
        created_at=_iso(),
    )


def _split_weeks(ctx: InsightContext) -> Tuple[List[Dict], List[Dict]]:
    this_start = ctx.now - timedelta(days=7)
    last_start = ctx.now - timedelta(days=14)
    this_week = [s for s in ctx.sessions if s.get("date_dt") and s["date_dt"] >= this_start]
    last_week = [
        s
        for s in ctx.sessions
        if s.get("date_dt") and last_start <= s["date_dt"] < this_start
    ]
    return this_week, last_week


def rule_fatigue_curve(ctx: InsightContext) -> Optional[Insight]:
    """Avg score at position N ≥ 10 pts below position 1, over ≥5 sessions."""
    # Group reps by session, then by rep_num
    sessions_by_id = {s["id"]: s for s in ctx.sessions}
    position_scores: Dict[int, List[float]] = {}
    qualifying_sessions = set()
    for r in ctx.reps:
        sess = sessions_by_id.get(r["session_id"])
        if sess is None:
            continue
        if ctx.exercise_filter and sess.get("exercise") != ctx.exercise_filter:
            continue
        rn = r.get("rep_num")
        score = r.get("form_score")
        if rn is None or score is None:
            continue
        pts = _score_pts(score)
        position_scores.setdefault(int(rn), []).append(pts)
        qualifying_sessions.add(r["session_id"])
    if len(qualifying_sessions) < 5:
        return None
    pos1 = _mean(position_scores.get(1, []))
    if pos1 is None:
        return None
    worst_pos = None
    worst_drop = 0.0
    for pos, scores in sorted(position_scores.items()):
        if pos <= 1 or len(scores) < 3:
            continue
        avg = _mean(scores) or 0
        drop = pos1 - avg
        if drop >= 10 and drop > worst_drop:
            worst_drop = drop
            worst_pos = pos
    if worst_pos is None:
        return None
    ex = ctx.exercise_filter or "training"
    return Insight(
        id=f"fatigue_{ex}_{worst_pos}",
        category="weakness",
        severity="warn",
        text=(
            f"You fall apart after rep {worst_pos - 1} — consider shorter sets "
            f"to hold form longer."
        ),
        exercise=ctx.exercise_filter,
        source_session_ids=sorted(int(x) for x in qualifying_sessions)[:10],
        data={"position": worst_pos, "drop": round(worst_drop, 1), "pos1_avg": round(pos1, 1)},
        created_at=_iso(),
    )


def rule_recurring_issue(ctx: InsightContext) -> Optional[Insight]:
    """Same issue in ≥60% of last 30 reps for the filter."""
    reps = [r for r in ctx.reps]
    if ctx.exercise_filter:
        sess_ids = {s["id"] for s in ctx.sessions if s.get("exercise") == ctx.exercise_filter}
        reps = [r for r in reps if r["session_id"] in sess_ids]
    reps = reps[-30:]
    if len(reps) < 10:
        return None
    counts: Dict[str, int] = {}
    sources: Dict[str, List[int]] = {}
    for r in reps:
        for issue in r.get("issues_list", []):
            counts[issue] = counts.get(issue, 0) + 1
            sources.setdefault(issue, []).append(int(r["id"]))
    if not counts:
        return None
    issue, count = max(counts.items(), key=lambda kv: kv[1])
    ratio = count / len(reps)
    if ratio < 0.6:
        return None
    return Insight(
        id=f"recurring_{issue}_{ctx.exercise_filter or 'any'}",
        category="weakness",
        severity="warn",
        text=f"{issue.replace('_', ' ').capitalize()} appeared in {count} of your last {len(reps)} reps — your top thing to clean up.",
        exercise=ctx.exercise_filter,
        source_session_ids=sorted({int(r["session_id"]) for r in reps})[:10],
        source_rep_ids=sources[issue][:20],
        data={"issue": issue, "count": count, "total": len(reps), "ratio": round(ratio, 2)},
        created_at=_iso(),
    )


def rule_tempo_rush(ctx: InsightContext) -> Optional[Insight]:
    """Average ecc & con < 0.5s over the last 10 sessions — user is rushing."""
    recent = [s for s in ctx.sessions[-10:] if s.get("exercise") != "plank"]
    if len(recent) < 3:
        return None
    sess_ids = {s["id"] for s in recent}
    reps = [r for r in ctx.reps if r["session_id"] in sess_ids]
    ecc_vals = [r["ecc_sec"] for r in reps if r.get("ecc_sec") is not None]
    con_vals = [r["con_sec"] for r in reps if r.get("con_sec") is not None]
    if len(ecc_vals) < 8 or len(con_vals) < 8:
        return None
    ecc_mean = sum(ecc_vals) / len(ecc_vals)
    con_mean = sum(con_vals) / len(con_vals)
    if ecc_mean >= 0.5 or con_mean >= 0.5:
        return None
    return Insight(
        id=f"tempo_rush_{ctx.exercise_filter or 'any'}",
        category="weakness",
        severity="notice",
        text=f"You're rushing reps — slow the lowering phase (averaging {ecc_mean:.1f}s down, {con_mean:.1f}s up).",
        exercise=ctx.exercise_filter,
        source_session_ids=[int(s["id"]) for s in recent],
        data={"ecc_mean": round(ecc_mean, 2), "con_mean": round(con_mean, 2)},
        created_at=_iso(),
    )


def rule_depth_cutting(ctx: InsightContext) -> Optional[Insight]:
    """Peak angle trends shallower across a session (later reps have smaller ROM)."""
    # Check the most recent non-plank session per exercise filter.
    candidates = [
        s for s in ctx.sessions[-5:]
        if s.get("exercise") != "plank"
        and (not ctx.exercise_filter or s.get("exercise") == ctx.exercise_filter)
    ]
    for s in reversed(candidates):
        reps = [
            r for r in _reps_for_session(ctx, s["id"])
            if r.get("peak_angle") is not None
        ]
        if len(reps) < 6:
            continue
        half = len(reps) // 2
        first = [r["peak_angle"] for r in reps[:half]]
        last = [r["peak_angle"] for r in reps[-half:]]
        first_mean = sum(first) / len(first)
        last_mean = sum(last) / len(last)
        # A higher peak_angle = less depth (flexed less); shallower = increased angle
        drift = last_mean - first_mean
        if drift > 8:
            ex = s.get("exercise", "your lift")
            return Insight(
                id=f"depth_cutting_{s['id']}",
                category="weakness",
                severity="notice",
                text=f"Your {ex.replace('_', ' ')}s got shallower toward the end of that session — reset your range of motion.",
                exercise=s.get("exercise"),
                source_session_ids=[int(s["id"])],
                source_rep_ids=[int(r["id"]) for r in reps],
                data={"first_mean": round(first_mean, 1), "last_mean": round(last_mean, 1), "drift": round(drift, 1)},
                created_at=_iso(),
            )
    return None


def rule_rest_short(ctx: InsightContext) -> Optional[Insight]:
    """Sets with short rest correlate with lower form score."""
    sets = [s for s in ctx.sets if s.get("rest_before_sec") and s.get("avg_form_score")]
    if ctx.exercise_filter:
        sess_ids = {x["id"] for x in ctx.sessions if x.get("exercise") == ctx.exercise_filter}
        sets = [s for s in sets if s["session_id"] in sess_ids]
    short = [s for s in sets if (s["rest_before_sec"] or 0) > 0 and (s["rest_before_sec"] or 0) < 60]
    long_ = [s for s in sets if (s["rest_before_sec"] or 0) >= 60]
    if len(short) < 5 or len(long_) < 5:
        return None
    s_avg = _mean([_score_pts(s["avg_form_score"]) for s in short]) or 0
    l_avg = _mean([_score_pts(s["avg_form_score"]) for s in long_]) or 0
    if (l_avg - s_avg) < 5:
        return None
    return Insight(
        id="rest_short",
        category="recovery",
        severity="notice",
        text=f"Your form is {l_avg - s_avg:.0f} points better when you rest at least 60 seconds between sets.",
        exercise=ctx.exercise_filter,
        data={"short_avg": round(s_avg, 1), "long_avg": round(l_avg, 1), "short_n": len(short), "long_n": len(long_)},
        created_at=_iso(),
    )


def rule_muscle_imbalance(ctx: InsightContext) -> Optional[Insight]:
    """A major muscle group has been untouched in the last 7 days (and was before)."""
    week_start = ctx.now - timedelta(days=7)
    older_start = ctx.now - timedelta(days=30)
    recent_groups: set = set()
    historical_groups: set = set()
    for s in ctx.sessions:
        groups = s.get("muscle_groups_list") or []
        # Fall back to resolving from exercise if the session pre-dates Session-1.
        if not groups:
            groups = resolve_muscle_groups(s.get("exercise", ""))
        big = {_group_for(g) or g for g in groups}
        big = {g for g in big if g in _ALL_GROUPS}
        if s.get("date_dt") and s["date_dt"] >= week_start:
            recent_groups |= big
        if s.get("date_dt") and s["date_dt"] >= older_start:
            historical_groups |= big
    missing = historical_groups - recent_groups
    # Prefer the biggest, most movement-central groups
    priority = ["legs", "back", "chest", "shoulders", "core", "arms"]
    for g in priority:
        if g in missing:
            return Insight(
                id=f"imbalance_{g}",
                category="recovery",
                severity="notice",
                text=f"You haven't trained {g} in the last week — add one session to stay balanced.",
                exercise=None,
                data={"missing_group": g, "recent": sorted(recent_groups)},
                created_at=_iso(),
            )
    return None


def rule_overtraining(ctx: InsightContext) -> Optional[Insight]:
    """Same muscle group trained 4+ consecutive days."""
    by_day: Dict[date, set] = {}
    for s in ctx.sessions:
        dt = s.get("date_dt")
        if not dt:
            continue
        groups = s.get("muscle_groups_list") or resolve_muscle_groups(s.get("exercise", ""))
        big = {_group_for(g) or g for g in groups}
        big = {g for g in big if g in _ALL_GROUPS}
        by_day.setdefault(dt.date(), set()).update(big)
    if not by_day:
        return None
    days_sorted = sorted(by_day.keys(), reverse=True)
    # For each group, count the longest run ending at the most recent date.
    for group in _ALL_GROUPS:
        run = 0
        last = None
        for d in days_sorted:
            if group not in by_day[d]:
                break
            if last is None:
                run = 1
            elif last - d == timedelta(days=1):
                run += 1
            else:
                break
            last = d
        if run >= 4:
            return Insight(
                id=f"overtrain_{group}",
                category="recovery",
                severity="warn",
                text=f"You've hit {group} {run} days straight — consider resting tomorrow.",
                exercise=None,
                data={"group": group, "run_days": run},
                created_at=_iso(),
            )
    return None


def rule_consistency_up(ctx: InsightContext) -> Optional[Insight]:
    """Consistency score trending up across the last four weeks."""
    weekly_means: List[float] = []
    for w in range(3, -1, -1):
        start = ctx.now - timedelta(days=(w + 1) * 7)
        end = ctx.now - timedelta(days=w * 7)
        vals = [
            s.get("consistency_score") or 0
            for s in ctx.sessions
            if s.get("date_dt") and start <= s["date_dt"] < end and (s.get("consistency_score") or 0) > 0
        ]
        if not vals:
            weekly_means.append(float("nan"))
        else:
            weekly_means.append(sum(vals) / len(vals))
    valid = [v for v in weekly_means if v == v]  # drop nan
    if len(valid) < 3:
        return None
    slope = _slope(valid)
    if slope <= 0.02:
        return None
    return Insight(
        id="consistency_up",
        category="consistency",
        severity="celebrate",
        text="Your reps are more consistent than they were two weeks ago — good pacing.",
        exercise=None,
        data={"slope": round(slope, 3), "weeks": len(valid)},
        created_at=_iso(),
    )


def rule_pr_reps(ctx: InsightContext) -> Optional[Insight]:
    """Most recent session broke the all-time reps-in-a-session record."""
    if not ctx.sessions:
        return None
    latest = ctx.sessions[-1]
    if latest.get("exercise") == "plank":
        return None
    all_time_max = ctx.all_time_best.get("max_reps") or 0
    if not all_time_max:
        return None
    if (latest.get("total_reps") or 0) >= all_time_max and (latest.get("total_reps") or 0) >= 8:
        # Only celebrate if latest IS the record (not prior)
        return Insight(
            id=f"pr_reps_{latest['id']}",
            category="milestone",
            severity="celebrate",
            text=f"Personal record: {latest['total_reps']} reps in one session.",
            exercise=latest.get("exercise"),
            source_session_ids=[int(latest["id"])],
            data={"reps": int(latest["total_reps"])},
            created_at=_iso(),
        )
    return None


def rule_pr_form(ctx: InsightContext) -> Optional[Insight]:
    """Latest session matches all-time best avg_form_score and is recent."""
    if not ctx.sessions:
        return None
    latest = ctx.sessions[-1]
    latest_pts = _score_pts(latest.get("avg_form_score"))
    best = _score_pts(ctx.all_time_best.get("max_form"))
    if latest_pts < 70 or latest_pts < best - 0.5:
        return None
    # If the latest session achieves best (within 0.5 pts), it's a PR day
    return Insight(
        id=f"pr_form_{latest['id']}",
        category="milestone",
        severity="celebrate",
        text=f"Cleanest session yet — {latest_pts:.0f}/100.",
        exercise=latest.get("exercise"),
        source_session_ids=[int(latest["id"])],
        data={"score": round(latest_pts, 1)},
        created_at=_iso(),
    )


def rule_pr_plank(ctx: InsightContext) -> Optional[Insight]:
    if not ctx.sessions:
        return None
    # Did the most recent plank set a new hold record?
    plank_sessions = [s for s in ctx.sessions if s.get("exercise") == "plank"]
    if not plank_sessions:
        return None
    latest = plank_sessions[-1]
    best = ctx.all_time_best.get("longest_plank_sec") or 0
    dur = latest.get("duration_sec") or 0
    if dur < 30 or dur < best - 0.5:
        return None
    mins = int(dur // 60)
    secs = int(dur % 60)
    label = f"{mins}:{secs:02d}" if mins else f"{secs}s"
    return Insight(
        id=f"pr_plank_{latest['id']}",
        category="milestone",
        severity="celebrate",
        text=f"Longest clean plank yet — {label}.",
        exercise="plank",
        source_session_ids=[int(latest["id"])],
        data={"duration_sec": round(dur, 1)},
        created_at=_iso(),
    )


# ── Top-level API ──────────────────────────────────────────────────────


ALL_RULES = (
    rule_first_session,
    rule_streak,
    rule_regression,
    rule_improvement,
    rule_fatigue_curve,
    rule_recurring_issue,
    rule_tempo_rush,
    rule_depth_cutting,
    rule_rest_short,
    rule_muscle_imbalance,
    rule_overtraining,
    rule_consistency_up,
    rule_pr_reps,
    rule_pr_form,
    rule_pr_plank,
)


def _sort_key(ins: Insight) -> Tuple[int, int, str]:
    return (
        _SEVERITY_RANK.get(ins.severity, 9),
        _CATEGORY_RANK.get(ins.category, 9),
        ins.id,
    )


def generate_insights(
    user_id: int,
    exercise: Optional[str] = None,
    period: str = "month",
    limit: int = 10,
    db=None,
) -> List[Insight]:
    """Generate ranked insights for a user.

    Stateless: every call re-reads the DB. Safe to call from Flask, a chatbot
    tool, a cron job, etc.

    Args:
        user_id: the authenticated user.
        exercise: optional filter — only fire rules scoped to this exercise.
        period: "week" | "month" | "all" — controls the lookback window for
            non-all-time rules. Currently used as a hint; most rules use their
            own natural window (week/4-week/30-rep) so `period` mostly affects
            the context load span. Defaults to "month".
        limit: max insights to return.
        db: an ExerVisionDB instance. If None, a fresh one is created.
    """
    if db is None:
        from app.database import ExerVisionDB  # local import to avoid cycles

        db = ExerVisionDB()

    ctx = _build_context(db, user_id, exercise)
    out: List[Insight] = []
    for rule in ALL_RULES:
        try:
            ins = rule(ctx)
            if ins is not None:
                out.append(ins)
        except Exception as e:
            logger.warning("Insight rule %s crashed: %s", rule.__name__, e)
    out.sort(key=_sort_key)
    return out[:limit]


def personal_records(user_id: int, db=None) -> Dict[str, Any]:
    """Return a PR summary used by the dashboard overview card."""
    if db is None:
        from app.database import ExerVisionDB

        db = ExerVisionDB()

    conn = db._get_conn()
    try:
        biggest = conn.execute(
            """SELECT id, exercise, total_reps, date FROM sessions
               WHERE user_id = ? AND total_reps > 0 AND exercise != 'plank'
               ORDER BY total_reps DESC, date DESC LIMIT 1""",
            (user_id,),
        ).fetchone()
        best_form = conn.execute(
            """SELECT id, exercise, avg_form_score, date FROM sessions
               WHERE user_id = ? AND total_reps > 0
               ORDER BY avg_form_score DESC, date DESC LIMIT 1""",
            (user_id,),
        ).fetchone()
        longest_plank = conn.execute(
            """SELECT id, duration_sec, date FROM sessions
               WHERE user_id = ? AND exercise = 'plank'
               ORDER BY duration_sec DESC LIMIT 1""",
            (user_id,),
        ).fetchone()
        # Longest streak in the past 90 days (approx — anchored to unique dates)
        rows = conn.execute(
            """SELECT DISTINCT DATE(date) AS d FROM sessions
               WHERE user_id = ? ORDER BY d ASC""",
            (user_id,),
        ).fetchall()
    finally:
        conn.close()

    # Longest streak calculation
    dates = [_parse_dt(r["d"]) for r in rows]
    dates = [d.date() for d in dates if d is not None]
    longest = 0
    if dates:
        run = 1
        longest = 1
        for i in range(1, len(dates)):
            if dates[i] - dates[i - 1] == timedelta(days=1):
                run += 1
                longest = max(longest, run)
            else:
                run = 1

    def _pr(row, key):
        if not row:
            return None
        return {k: row[k] for k in row.keys()}

    return {
        "biggest_session": _pr(biggest, "biggest"),
        "best_form_day": _pr(best_form, "best_form"),
        "longest_plank": _pr(longest_plank, "longest_plank"),
        "longest_streak": longest,
    }


def muscle_balance(user_id: int, days: int = 7, db=None) -> Dict[str, int]:
    """Return total reps per big muscle group over the last `days` days."""
    if db is None:
        from app.database import ExerVisionDB

        db = ExerVisionDB()

    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    conn = db._get_conn()
    try:
        rows = conn.execute(
            """SELECT exercise, total_reps, duration_sec, muscle_groups
               FROM sessions WHERE user_id = ? AND date >= ?""",
            (user_id, cutoff),
        ).fetchall()
    finally:
        conn.close()

    out = {g: 0 for g in _ALL_GROUPS}
    for r in rows:
        reps = int(r["total_reps"] or 0)
        if r["exercise"] == "plank":
            # count plank holds as core "effort" by mapping seconds → units
            reps = max(1, int((r["duration_sec"] or 0) / 5))
        try:
            stored = json.loads(r["muscle_groups"] or "[]")
        except json.JSONDecodeError:
            stored = []
        groups = stored or resolve_muscle_groups(r["exercise"] or "")
        big = []
        for g in groups:
            mapped = _group_for(g) or (g if g in _ALL_GROUPS else None)
            if mapped and mapped not in big:
                big.append(mapped)
        for g in big:
            out[g] += reps
    return out
