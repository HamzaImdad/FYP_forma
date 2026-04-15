"""
SQLite persistence layer for FORMA session history.

Stores users, workout sessions, per-rep details, and set breakdowns.
Provides query methods for dashboard aggregation (daily/weekly/monthly).
All query methods accept a user_id argument so data stays scoped per user.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


# Per-rep columns added in Session-1
_REP_EXT_COLS = (
    ("peak_angle", "REAL"),
    ("ecc_sec", "REAL"),
    ("con_sec", "REAL"),
    ("score_min", "REAL"),
    ("score_max", "REAL"),
    ("set_num", "INTEGER"),
)

# Per-set columns added in Session-1
_SET_EXT_COLS = (
    ("rest_before_sec", "REAL DEFAULT 0"),
    ("avg_form_score", "REAL DEFAULT 0"),
    ("score_dropoff", "REAL DEFAULT 0"),
    ("failure_type", "TEXT DEFAULT 'clean_stop'"),
)

# Per-session columns added in Session-1
_SESSION_EXT_COLS = (
    ("user_id", "INTEGER REFERENCES users(id)"),
    ("total_rest_sec", "REAL DEFAULT 0"),
    ("work_to_rest_ratio", "REAL DEFAULT 0"),
    ("consistency_score", "REAL DEFAULT 0"),
    ("fatigue_index", "REAL DEFAULT 0"),
    ("muscle_groups", "TEXT DEFAULT '[]'"),
)


class ExerVisionDB:
    """SQLite database for persisting workout session history."""

    def __init__(self, db_path: str = "exervision.db"):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        """Create tables if they don't exist and run idempotent migrations."""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    avatar_url TEXT,
                    height_cm REAL,
                    weight_kg REAL,
                    age INTEGER,
                    experience_level TEXT DEFAULT 'beginner',
                    training_goal TEXT DEFAULT 'strength',
                    coaching_tone TEXT DEFAULT 'neutral',
                    created_at TEXT NOT NULL,
                    last_login TEXT
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exercise TEXT NOT NULL,
                    classifier TEXT NOT NULL DEFAULT 'rule_based',
                    date TEXT NOT NULL,
                    duration_sec REAL NOT NULL DEFAULT 0,
                    total_reps INTEGER NOT NULL DEFAULT 0,
                    good_reps INTEGER NOT NULL DEFAULT 0,
                    avg_form_score REAL NOT NULL DEFAULT 0,
                    weight_kg REAL
                );

                CREATE TABLE IF NOT EXISTS reps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    rep_num INTEGER NOT NULL,
                    form_score REAL NOT NULL DEFAULT 0,
                    quality TEXT NOT NULL DEFAULT 'moderate',
                    issues TEXT NOT NULL DEFAULT '[]',
                    duration REAL NOT NULL DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS sets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    set_num INTEGER NOT NULL,
                    reps_count INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_date ON sessions(date);
                CREATE INDEX IF NOT EXISTS idx_sessions_exercise ON sessions(exercise);
                CREATE INDEX IF NOT EXISTS idx_reps_session ON reps(session_id);
                CREATE INDEX IF NOT EXISTS idx_sets_session ON sets(session_id);
            """)
            # Idempotent column migrations (ALTER is no-op if column exists).
            self._ensure_columns(conn, "sessions", _SESSION_EXT_COLS)
            self._ensure_columns(conn, "reps", _REP_EXT_COLS)
            self._ensure_columns(conn, "sets", _SET_EXT_COLS)
            # Legacy weight_kg migration — kept for DBs that predate Session-1
            try:
                conn.execute("ALTER TABLE sessions ADD COLUMN weight_kg REAL")
            except sqlite3.OperationalError:
                pass
            # Extra indexes for user-scoped queries
            try:
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, date)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sessions_user_exercise ON sessions(user_id, exercise)"
                )
            except sqlite3.OperationalError:
                pass
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _ensure_columns(conn: sqlite3.Connection, table: str, cols) -> None:
        """Add any missing columns to `table`. Idempotent — skips duplicates."""
        existing = {
            row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for name, decl in cols:
            if name in existing:
                continue
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
            except sqlite3.OperationalError:
                pass

    # ── Users ───────────────────────────────────────────────────────────

    def create_user(self, email: str, password_hash: str, display_name: str) -> int:
        conn = self._get_conn()
        try:
            now = datetime.now().isoformat()
            cur = conn.execute(
                """INSERT INTO users (email, password_hash, display_name, created_at)
                   VALUES (?, ?, ?, ?)""",
                (email.strip().lower(), password_hash, display_name.strip(), now),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ?",
                (email.strip().lower(),),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM users WHERE id = ?", (int(user_id),)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def update_user_profile(self, user_id: int, fields: Dict[str, Any]) -> None:
        allowed = {
            "display_name", "avatar_url", "height_cm", "weight_kg", "age",
            "experience_level", "training_goal", "coaching_tone",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        conn = self._get_conn()
        try:
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            params = list(updates.values()) + [int(user_id)]
            conn.execute(f"UPDATE users SET {set_clause} WHERE id = ?", params)
            conn.commit()
        finally:
            conn.close()

    def update_last_login(self, user_id: int) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE users SET last_login = ? WHERE id = ?",
                (datetime.now().isoformat(), int(user_id)),
            )
            conn.commit()
        finally:
            conn.close()

    def reassign_orphan_sessions(self, user_id: int) -> int:
        """Claim every legacy session row that predates Session-1 auth."""
        conn = self._get_conn()
        try:
            cur = conn.execute(
                "UPDATE sessions SET user_id = ? WHERE user_id IS NULL",
                (int(user_id),),
            )
            conn.commit()
            return int(cur.rowcount or 0)
        finally:
            conn.close()

    # ── Sessions ────────────────────────────────────────────────────────

    def save_session(self, summary: Dict[str, Any], user_id: int) -> int:
        """Save a complete workout session. Returns the session ID."""
        conn = self._get_conn()
        try:
            now = datetime.now().isoformat()

            cursor = conn.execute(
                """INSERT INTO sessions (
                       exercise, classifier, date, duration_sec,
                       total_reps, good_reps, avg_form_score, weight_kg,
                       user_id, total_rest_sec, work_to_rest_ratio,
                       consistency_score, fatigue_index, muscle_groups
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    summary.get("exercise", "unknown"),
                    summary.get("classifier", "rule_based"),
                    now,
                    summary.get("duration_sec", 0),
                    summary.get("total_reps", 0),
                    summary.get("good_reps", 0),
                    summary.get("avg_form_score", 0),
                    summary.get("weight_kg"),
                    int(user_id),
                    float(summary.get("total_rest_sec", 0) or 0),
                    float(summary.get("work_to_rest_ratio", 0) or 0),
                    float(summary.get("consistency_score", 0) or 0),
                    float(summary.get("fatigue_index", 0) or 0),
                    json.dumps(summary.get("muscle_groups", [])),
                ),
            )
            session_id = cursor.lastrowid

            # Save per-rep details — Session-1 fields included when present.
            reps = summary.get("reps", [])
            for rep in reps:
                issues = rep.get("issues", [])
                conn.execute(
                    """INSERT INTO reps (
                           session_id, rep_num, form_score, quality, issues, duration,
                           peak_angle, ecc_sec, con_sec, score_min, score_max, set_num
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        rep.get("rep_num", rep.get("rep", 0)),
                        rep.get("form_score", rep.get("score", 0)),
                        rep.get("quality", "moderate"),
                        json.dumps(issues),
                        rep.get("duration", 0),
                        rep.get("peak_angle"),
                        rep.get("ecc_sec"),
                        rep.get("con_sec"),
                        rep.get("score_min"),
                        rep.get("score_max"),
                        rep.get("set_num"),
                    ),
                )

            # Save per-set rows. Prefer the rich `sets` list; fall back to
            # `reps_per_set` for legacy callers.
            set_rows = summary.get("sets")
            if isinstance(set_rows, list) and set_rows:
                for i, s in enumerate(set_rows):
                    if not isinstance(s, dict):
                        continue
                    conn.execute(
                        """INSERT INTO sets (
                               session_id, set_num, reps_count,
                               rest_before_sec, avg_form_score, score_dropoff, failure_type
                           ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            session_id,
                            int(s.get("set_num", i + 1)),
                            int(s.get("reps_count", 0)),
                            float(s.get("rest_before_sec", 0) or 0),
                            float(s.get("avg_form_score", 0) or 0),
                            float(s.get("score_dropoff", 0) or 0),
                            str(s.get("failure_type", "clean_stop")),
                        ),
                    )
            else:
                reps_per_set = summary.get("reps_per_set", [])
                for i, reps_count in enumerate(reps_per_set):
                    conn.execute(
                        """INSERT INTO sets (session_id, set_num, reps_count)
                           VALUES (?, ?, ?)""",
                        (session_id, i + 1, reps_count),
                    )

            conn.commit()
            return session_id
        finally:
            conn.close()

    def get_sessions(
        self,
        user_id: int,
        exercise: Optional[str] = None,
        period: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict]:
        """Query sessions with optional filters, scoped to the user."""
        conn = self._get_conn()
        try:
            query = "SELECT * FROM sessions WHERE user_id = ?"
            params: list = [int(user_id)]

            if exercise:
                query += " AND exercise = ?"
                params.append(exercise)

            date_filter = self._period_to_date(period)
            if date_filter:
                query += " AND date >= ?"
                params.append(date_filter)

            query += " ORDER BY date DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_session_detail(self, user_id: int, session_id: int) -> Optional[Dict]:
        """Get full session with reps and sets. Returns None if unowned."""
        conn = self._get_conn()
        try:
            session = conn.execute(
                "SELECT * FROM sessions WHERE id = ? AND user_id = ?",
                (int(session_id), int(user_id)),
            ).fetchone()
            if not session:
                return None

            result = dict(session)

            reps = conn.execute(
                "SELECT * FROM reps WHERE session_id = ? ORDER BY rep_num",
                (int(session_id),),
            ).fetchall()
            result["reps"] = [
                {**dict(r), "issues": json.loads(r["issues"])} for r in reps
            ]

            sets = conn.execute(
                "SELECT * FROM sets WHERE session_id = ? ORDER BY set_num",
                (int(session_id),),
            ).fetchall()
            result["sets"] = [dict(s) for s in sets]

            return result
        finally:
            conn.close()

    def get_stats(
        self,
        user_id: int,
        period: Optional[str] = None,
        exercise: Optional[str] = None,
    ) -> Dict:
        """Aggregate stats for dashboard summary cards (user-scoped)."""
        conn = self._get_conn()
        try:
            query = """
                SELECT
                    COUNT(*) as total_workouts,
                    COALESCE(SUM(total_reps), 0) as total_reps,
                    COALESCE(AVG(avg_form_score), 0) as avg_score,
                    COALESCE(SUM(duration_sec), 0) as total_time_sec,
                    COALESCE(SUM(good_reps), 0) as total_good_reps
                FROM sessions
                WHERE user_id = ?
            """
            params: list = [int(user_id)]

            if exercise:
                query += " AND exercise = ?"
                params.append(exercise)

            date_filter = self._period_to_date(period)
            if date_filter:
                query += " AND date >= ?"
                params.append(date_filter)

            row = conn.execute(query, params).fetchone()
            return {
                "total_workouts": row["total_workouts"],
                "total_reps": row["total_reps"],
                "avg_score": round(row["avg_score"] or 0, 3),
                "total_time_sec": round(row["total_time_sec"] or 0, 1),
                "total_good_reps": row["total_good_reps"],
            }
        finally:
            conn.close()

    def get_today_totals(self, user_id: int, exercise: str) -> Dict:
        """Today's totals for one exercise, scoped to the user."""
        conn = self._get_conn()
        try:
            cutoff = self._period_to_date("today")
            sessions = conn.execute(
                """SELECT id, total_reps, avg_form_score FROM sessions
                   WHERE user_id = ? AND exercise = ? AND date >= ?
                   ORDER BY date ASC""",
                (int(user_id), exercise, cutoff),
            ).fetchall()

            session_ids = [s["id"] for s in sessions]
            sets_rows: List = []
            if session_ids:
                placeholders = ",".join("?" * len(session_ids))
                sets_rows = conn.execute(
                    f"""SELECT session_id, set_num, reps_count FROM sets
                        WHERE session_id IN ({placeholders})
                        ORDER BY session_id ASC, set_num ASC""",
                    session_ids,
                ).fetchall()

            reps_per_set = [r["reps_count"] for r in sets_rows]
            total_reps = sum(s["total_reps"] for s in sessions)
            weighted_score = 0.0
            if total_reps > 0:
                weighted_score = sum(
                    (s["total_reps"] or 0) * (s["avg_form_score"] or 0)
                    for s in sessions
                ) / total_reps

            return {
                "exercise": exercise,
                "sessions_today": len(sessions),
                "sets_today": len(reps_per_set),
                "reps_today": total_reps,
                "reps_per_set_today": reps_per_set,
                "avg_form_score": round(weighted_score, 3),
            }
        finally:
            conn.close()

    def get_daily_scores(
        self,
        user_id: int,
        days: int = 30,
        exercise: Optional[str] = None,
    ) -> Dict:
        """Daily average form scores for line chart."""
        conn = self._get_conn()
        try:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            query = """
                SELECT DATE(date) as day,
                       AVG(avg_form_score) as avg_score,
                       SUM(total_reps) as total_reps,
                       COUNT(*) as session_count
                FROM sessions
                WHERE user_id = ? AND date >= ?
            """
            params: list = [int(user_id), cutoff]

            if exercise:
                query += " AND exercise = ?"
                params.append(exercise)

            query += " GROUP BY DATE(date) ORDER BY day"

            rows = conn.execute(query, params).fetchall()
            return {
                "dates": [r["day"] for r in rows],
                "scores": [round(r["avg_score"] or 0, 3) for r in rows],
                "reps": [r["total_reps"] for r in rows],
                "session_counts": [r["session_count"] for r in rows],
            }
        finally:
            conn.close()

    def get_exercise_distribution(
        self, user_id: int, period: Optional[str] = None
    ) -> List[Dict]:
        """Exercise breakdown for pie/doughnut chart."""
        conn = self._get_conn()
        try:
            query = """
                SELECT exercise, COUNT(*) as count, SUM(total_reps) as total_reps
                FROM sessions
                WHERE user_id = ?
            """
            params: list = [int(user_id)]

            date_filter = self._period_to_date(period)
            if date_filter:
                query += " AND date >= ?"
                params.append(date_filter)

            query += " GROUP BY exercise ORDER BY count DESC"

            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ── Dashboard deep-dive helpers (Session 2) ─────────────────────────

    def get_heatmap(self, user_id: int, days: int = 84) -> List[Dict]:
        """Activity heatmap: reps count + session count per calendar day."""
        conn = self._get_conn()
        try:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            rows = conn.execute(
                """SELECT DATE(date) AS day,
                          SUM(total_reps) AS reps_count,
                          COUNT(*) AS session_count,
                          SUM(duration_sec) AS duration_sec
                   FROM sessions
                   WHERE user_id = ? AND date >= ?
                   GROUP BY DATE(date)
                   ORDER BY day""",
                (int(user_id), cutoff),
            ).fetchall()
            return [
                {
                    "date": r["day"],
                    "reps_count": int(r["reps_count"] or 0),
                    "session_count": int(r["session_count"] or 0),
                    "duration_sec": float(r["duration_sec"] or 0),
                }
                for r in rows
            ]
        finally:
            conn.close()

    def get_top_issues(
        self, user_id: int, exercise: Optional[str] = None, limit: int = 5
    ) -> List[Dict]:
        """Aggregate issue frequencies across a user's reps."""
        conn = self._get_conn()
        try:
            q = (
                "SELECT r.issues FROM reps r "
                "JOIN sessions s ON r.session_id = s.id "
                "WHERE s.user_id = ?"
            )
            params: list = [int(user_id)]
            if exercise:
                q += " AND s.exercise = ?"
                params.append(exercise)
            rows = conn.execute(q, params).fetchall()
            counts: Dict[str, int] = {}
            for r in rows:
                try:
                    for issue in json.loads(r["issues"] or "[]"):
                        if not issue:
                            continue
                        counts[issue] = counts.get(issue, 0) + 1
                except json.JSONDecodeError:
                    continue
            top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]
            return [{"issue": k, "count": v} for k, v in top]
        finally:
            conn.close()

    def get_exercise_deep_dive(
        self, user_id: int, exercise: str, days: int = 30
    ) -> Dict:
        """Bundle everything the per-exercise deep-dive panel needs."""
        conn = self._get_conn()
        try:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            sessions = conn.execute(
                """SELECT id, date, duration_sec, total_reps, good_reps, avg_form_score
                   FROM sessions
                   WHERE user_id = ? AND exercise = ? AND date >= ?
                   ORDER BY date ASC""",
                (int(user_id), exercise, cutoff),
            ).fetchall()
            session_ids = [s["id"] for s in sessions]

            rep_rows: list = []
            if session_ids:
                placeholders = ",".join("?" * len(session_ids))
                rep_rows = conn.execute(
                    f"""SELECT session_id, rep_num, form_score, quality,
                               peak_angle, ecc_sec, con_sec, set_num
                        FROM reps
                        WHERE session_id IN ({placeholders})
                        ORDER BY session_id ASC, rep_num ASC""",
                    session_ids,
                ).fetchall()
        finally:
            conn.close()

        scores = [
            {
                "session_id": s["id"],
                "date": s["date"],
                "avg_form_score": float(s["avg_form_score"] or 0),
                "total_reps": int(s["total_reps"] or 0),
                "duration_sec": float(s["duration_sec"] or 0),
            }
            for s in sessions
        ]

        # Quality breakdown per session
        quality_by_session: Dict[int, Dict[str, int]] = {
            int(s["id"]): {"good": 0, "moderate": 0, "bad": 0} for s in sessions
        }
        tempo: List[Dict] = []
        depth: List[Dict] = []
        fatigue_pos: Dict[int, List[float]] = {}

        for r in rep_rows:
            sid = int(r["session_id"])
            q = (r["quality"] or "moderate").lower()
            bucket = quality_by_session.setdefault(
                sid, {"good": 0, "moderate": 0, "bad": 0}
            )
            if q in bucket:
                bucket[q] += 1
            else:
                bucket["moderate"] += 1
            if r["ecc_sec"] is not None or r["con_sec"] is not None:
                tempo.append(
                    {
                        "session_id": sid,
                        "rep_num": int(r["rep_num"] or 0),
                        "ecc_sec": float(r["ecc_sec"] or 0),
                        "con_sec": float(r["con_sec"] or 0),
                    }
                )
            if r["peak_angle"] is not None:
                depth.append(
                    {
                        "session_id": sid,
                        "rep_num": int(r["rep_num"] or 0),
                        "peak_angle": float(r["peak_angle"]),
                    }
                )
            rn = r["rep_num"]
            sc = r["form_score"]
            if rn is not None and sc is not None:
                fatigue_pos.setdefault(int(rn), []).append(float(sc))

        quality_breakdown = [
            {
                "session_id": s["id"],
                "date": s["date"],
                "good": quality_by_session[int(s["id"])]["good"],
                "moderate": quality_by_session[int(s["id"])]["moderate"],
                "bad": quality_by_session[int(s["id"])]["bad"],
            }
            for s in sessions
        ]
        fatigue_curve = [
            {
                "rep_num": pos,
                "avg_score": round(sum(v) / len(v), 3) if v else 0,
                "sample": len(v),
            }
            for pos, v in sorted(fatigue_pos.items())
        ]
        return {
            "exercise": exercise,
            "scores": scores,
            "quality_breakdown": quality_breakdown,
            "tempo": tempo,
            "depth": depth,
            "fatigue_curve": fatigue_curve,
        }

    def get_wow_deltas(self, user_id: int, exercise: Optional[str] = None) -> Dict:
        """Week-over-week KPI snapshot (reps, avg form, time trained)."""
        conn = self._get_conn()
        try:
            now = datetime.now()
            this_start = (now - timedelta(days=7)).isoformat()
            last_start = (now - timedelta(days=14)).isoformat()

            def agg(start_iso: str, end_iso: str) -> Dict:
                q = (
                    "SELECT COALESCE(SUM(total_reps),0) AS reps,"
                    " COALESCE(AVG(NULLIF(avg_form_score,0)),0) AS form,"
                    " COALESCE(SUM(duration_sec),0) AS time,"
                    " COUNT(*) AS sessions"
                    " FROM sessions WHERE user_id = ? AND date >= ? AND date < ?"
                )
                params = [int(user_id), start_iso, end_iso]
                if exercise:
                    q += " AND exercise = ?"
                    params.append(exercise)
                r = conn.execute(q, params).fetchone()
                return {
                    "reps": int(r["reps"] or 0),
                    "form": float(r["form"] or 0),
                    "time": float(r["time"] or 0),
                    "sessions": int(r["sessions"] or 0),
                }

            this_week = agg(this_start, now.isoformat())
            last_week = agg(last_start, this_start)
            return {"this_week": this_week, "last_week": last_week}
        finally:
            conn.close()

    def _period_to_date(self, period: Optional[str]) -> Optional[str]:
        """Convert period string to ISO date cutoff."""
        if not period or period == "all":
            return None
        now = datetime.now()
        if period == "today":
            return now.replace(hour=0, minute=0, second=0).isoformat()
        elif period == "week":
            return (now - timedelta(days=7)).isoformat()
        elif period == "month":
            return (now - timedelta(days=30)).isoformat()
        return None
