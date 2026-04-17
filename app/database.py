"""
SQLite persistence layer for FORMA session history.

Stores users, workout sessions, per-rep details, and set breakdowns.
Provides query methods for dashboard aggregation (daily/weekly/monthly).
All query methods accept a user_id argument so data stays scoped per user.
"""

import json
import os
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
    # Redesign Phase 1: per-set weight for weighted-family exercises.
    # Nullable — pre-migration rows get backfilled from sessions.weight_kg.
    ("weight_kg", "REAL"),
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

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.getenv("DB_PATH", "exervision.db")
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        # check_same_thread=False so eventlet greenlets can reuse connections
        # across greenlet context-switches. WAL mode (below) keeps concurrent
        # reads safe; SQLite serializes writes at the file level.
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
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

                -- Session-3 chatbot tables
                CREATE TABLE IF NOT EXISTS chat_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    UNIQUE(user_id, date),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chat_conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    mode TEXT NOT NULL,
                    title TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_call_id TEXT,
                    citations TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES chat_conversations(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chat_conv_user ON chat_conversations(user_id, mode, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_chat_msg_conv ON chat_messages(conversation_id, id);

                -- Session-4 goals + milestones + plans + badges
                CREATE TABLE IF NOT EXISTS goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    goal_type TEXT NOT NULL,
                    exercise TEXT,
                    target_value REAL NOT NULL,
                    current_value REAL DEFAULT 0,
                    unit TEXT NOT NULL,
                    period TEXT,
                    deadline TEXT,
                    status TEXT DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS milestones (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal_id INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    threshold_value REAL NOT NULL,
                    reached INTEGER DEFAULT 0,
                    reached_at TEXT,
                    FOREIGN KEY (goal_id) REFERENCES goals(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    created_by_chat INTEGER DEFAULT 0,
                    conversation_id INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS plan_days (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_id INTEGER NOT NULL,
                    day_date TEXT NOT NULL,
                    is_rest INTEGER DEFAULT 0,
                    exercises TEXT NOT NULL,
                    completed INTEGER DEFAULT 0,
                    completed_at TEXT,
                    FOREIGN KEY (plan_id) REFERENCES plans(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS user_badges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    badge_key TEXT NOT NULL,
                    earned_at TEXT NOT NULL,
                    metadata TEXT,
                    UNIQUE (user_id, badge_key),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                -- Redesign Phase 3: per-user plan draft (was per-conversation).
                -- At most one in-flight draft per user. Survives closing the
                -- chat tab; the new conversation can offer "resume draft?".
                CREATE TABLE IF NOT EXISTS user_plan_drafts (
                    user_id INTEGER PRIMARY KEY,
                    draft_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    source TEXT,
                    conversation_id INTEGER,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
            """)
            # Idempotent column migrations (ALTER is no-op if column exists).
            self._ensure_columns(conn, "sessions", _SESSION_EXT_COLS)
            self._ensure_columns(conn, "reps", _REP_EXT_COLS)
            self._ensure_columns(conn, "sets", _SET_EXT_COLS)
            # Session-4: plan_draft is a nullable JSON column on chat_conversations
            # used by the plan-creator chatbot to persist the in-flight plan draft
            # between propose_plan / revise_plan tool calls.
            self._ensure_columns(
                conn, "chat_conversations", (("plan_draft", "TEXT"),)
            )
            # plan_id FK on goals — links auto-created goals back to their plan
            # so recompute can count completed plan_days directly (plan_progress type)
            # and plan deletion cascades cleanly.
            self._ensure_columns(conn, "goals", (("plan_id", "INTEGER"),))
            # Redesign Phase 1: target_reps on goals — required by the new
            # `strength` goal type ("hit target_weight_kg for target_reps
            # clean reps"). Nullable; only strength goals populate it.
            self._ensure_columns(conn, "goals", (("target_reps", "INTEGER"),))
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
            # Session-4 indexes for goals / milestones / plans / badges
            try:
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_goals_user_status ON goals(user_id, status)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_milestones_goal_reached ON milestones(goal_id, reached)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_plans_user_status ON plans(user_id, status)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_plan_days_plan_date ON plan_days(plan_id, day_date)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_user_badges_user ON user_badges(user_id)"
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

    # ── Session-3: chat usage + conversations ──────────────────────────

    CHAT_DAILY_TOKEN_BUDGET = 50_000

    def _today_str(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def record_token_usage(
        self, user_id: int, input_tokens: int, output_tokens: int
    ) -> None:
        """Increment today's input/output token counters for a user."""
        if input_tokens <= 0 and output_tokens <= 0:
            return
        conn = self._get_conn()
        try:
            day = self._today_str()
            conn.execute(
                """INSERT INTO chat_usage (user_id, date, input_tokens, output_tokens)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(user_id, date) DO UPDATE SET
                       input_tokens = input_tokens + excluded.input_tokens,
                       output_tokens = output_tokens + excluded.output_tokens""",
                (int(user_id), day, int(max(0, input_tokens)), int(max(0, output_tokens))),
            )
            conn.commit()
        finally:
            conn.close()

    def get_token_usage_today(self, user_id: int) -> Dict[str, int]:
        """Return today's usage for the user (0/0 if no row yet)."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT input_tokens, output_tokens FROM chat_usage
                   WHERE user_id = ? AND date = ?""",
                (int(user_id), self._today_str()),
            ).fetchone()
            if row is None:
                return {"input_tokens": 0, "output_tokens": 0, "total": 0}
            input_t = int(row["input_tokens"] or 0)
            output_t = int(row["output_tokens"] or 0)
            return {
                "input_tokens": input_t,
                "output_tokens": output_t,
                "total": input_t + output_t,
            }
        finally:
            conn.close()

    def check_token_budget(self, user_id: int) -> Dict[str, Any]:
        """Return {allowed, used, budget, fraction, degrade} for the user.

        allowed=False once total (input+output) exceeds CHAT_DAILY_TOKEN_BUDGET.
        degrade=True between 50% and 100% of the budget — Session-3 uses this
        to auto-downgrade from gpt-4o → gpt-4o-mini.
        """
        usage = self.get_token_usage_today(user_id)
        total = usage["total"]
        budget = self.CHAT_DAILY_TOKEN_BUDGET
        return {
            "allowed": total < budget,
            "used": total,
            "budget": budget,
            "fraction": total / budget if budget else 1.0,
            "degrade": total >= budget // 2,
        }

    def create_conversation(
        self, user_id: int, mode: str, title: Optional[str] = None
    ) -> int:
        """Create a new chat conversation; returns its id."""
        conn = self._get_conn()
        try:
            now = datetime.now().isoformat()
            cur = conn.execute(
                """INSERT INTO chat_conversations (user_id, mode, title, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (int(user_id), str(mode), title, now, now),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def list_conversations(self, user_id: int, mode: Optional[str] = None) -> List[Dict]:
        conn = self._get_conn()
        try:
            q = (
                "SELECT c.id, c.mode, c.title, c.created_at, c.updated_at,"
                " (SELECT COUNT(*) FROM chat_messages m WHERE m.conversation_id = c.id) AS message_count"
                " FROM chat_conversations c WHERE c.user_id = ?"
            )
            params: list = [int(user_id)]
            if mode:
                q += " AND c.mode = ?"
                params.append(mode)
            q += " ORDER BY c.updated_at DESC"
            rows = conn.execute(q, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def load_conversation(
        self, conversation_id: int, user_id: int
    ) -> Optional[Dict]:
        """Return {conversation, messages} or None if not owned by the user."""
        conn = self._get_conn()
        try:
            conv = conn.execute(
                "SELECT * FROM chat_conversations WHERE id = ? AND user_id = ?",
                (int(conversation_id), int(user_id)),
            ).fetchone()
            if not conv:
                return None
            msg_rows = conn.execute(
                """SELECT id, role, content, tool_call_id, citations, created_at
                   FROM chat_messages
                   WHERE conversation_id = ?
                   ORDER BY id ASC""",
                (int(conversation_id),),
            ).fetchall()
            messages: List[Dict] = []
            for m in msg_rows:
                d = dict(m)
                try:
                    d["citations"] = json.loads(d["citations"] or "[]")
                except (ValueError, TypeError):
                    d["citations"] = []
                messages.append(d)
            return {"conversation": dict(conv), "messages": messages}
        finally:
            conn.close()

    def append_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        tool_call_id: Optional[str] = None,
        citations: Optional[List[int]] = None,
    ) -> int:
        """Insert a chat message and bump the conversation's updated_at."""
        conn = self._get_conn()
        try:
            now = datetime.now().isoformat()
            cur = conn.execute(
                """INSERT INTO chat_messages
                   (conversation_id, role, content, tool_call_id, citations, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    int(conversation_id),
                    str(role),
                    str(content or ""),
                    tool_call_id,
                    json.dumps(list(citations or [])),
                    now,
                ),
            )
            conn.execute(
                "UPDATE chat_conversations SET updated_at = ? WHERE id = ?",
                (now, int(conversation_id)),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def delete_conversation(self, conversation_id: int, user_id: int) -> bool:
        """Delete a conversation if owned by the user. Cascade deletes messages."""
        conn = self._get_conn()
        try:
            cur = conn.execute(
                "DELETE FROM chat_conversations WHERE id = ? AND user_id = ?",
                (int(conversation_id), int(user_id)),
            )
            conn.commit()
            return (cur.rowcount or 0) > 0
        finally:
            conn.close()

    def update_conversation_title(
        self, conversation_id: int, user_id: int, title: str
    ) -> None:
        """Set the title for a conversation if owned by the user."""
        conn = self._get_conn()
        try:
            conn.execute(
                """UPDATE chat_conversations SET title = ?, updated_at = ?
                   WHERE id = ? AND user_id = ?""",
                (
                    (title or "")[:120],
                    datetime.now().isoformat(),
                    int(conversation_id),
                    int(user_id),
                ),
            )
            conn.commit()
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

    # ── Session-4: plan draft column on chat_conversations ─────────────

    def set_conversation_plan_draft(
        self, conversation_id: int, draft_json: Optional[str]
    ) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE chat_conversations SET plan_draft = ? WHERE id = ?",
                (draft_json, int(conversation_id)),
            )
            conn.commit()
        finally:
            conn.close()

    def get_conversation_plan_draft(
        self, conversation_id: int
    ) -> Optional[str]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT plan_draft FROM chat_conversations WHERE id = ?",
                (int(conversation_id),),
            ).fetchone()
            return row["plan_draft"] if row else None
        finally:
            conn.close()

    # ── Redesign Phase 3: per-user plan drafts ─────────────────────────
    # At most one in-flight plan draft per user, survives tab close.
    # Replaces the conversation-scoped draft above — old methods kept for
    # backwards-compat during the rollout + one-shot migration.

    def set_user_plan_draft(
        self,
        user_id: int,
        draft_json: str,
        *,
        source: Optional[str] = None,
        conversation_id: Optional[int] = None,
    ) -> None:
        conn = self._get_conn()
        try:
            now = datetime.now().isoformat(timespec="seconds")
            conn.execute(
                """INSERT INTO user_plan_drafts
                       (user_id, draft_json, updated_at, source, conversation_id)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(user_id) DO UPDATE SET
                       draft_json = excluded.draft_json,
                       updated_at = excluded.updated_at,
                       source = excluded.source,
                       conversation_id = excluded.conversation_id""",
                (
                    int(user_id),
                    draft_json,
                    now,
                    source,
                    int(conversation_id) if conversation_id is not None else None,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_user_plan_draft(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Return the user's in-flight draft row (or None).

        Shape: {"draft_json", "updated_at", "source", "conversation_id"}.
        Callers parse draft_json themselves.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT draft_json, updated_at, source, conversation_id
                   FROM user_plan_drafts WHERE user_id = ?""",
                (int(user_id),),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def clear_user_plan_draft(self, user_id: int) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                "DELETE FROM user_plan_drafts WHERE user_id = ?", (int(user_id),)
            )
            conn.commit()
        finally:
            conn.close()

    # ── Session-4: goals ───────────────────────────────────────────────

    def create_goal(
        self,
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
    ) -> int:
        conn = self._get_conn()
        try:
            now = datetime.now().isoformat(timespec="seconds")
            cur = conn.execute(
                """INSERT INTO goals
                       (user_id, title, description, goal_type, exercise,
                        target_value, current_value, unit, period, deadline,
                        status, created_at, plan_id, target_reps)
                   VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, 'active', ?, ?, ?)""",
                (
                    int(user_id),
                    title.strip(),
                    description,
                    goal_type,
                    exercise,
                    float(target_value),
                    unit,
                    period,
                    deadline,
                    now,
                    int(plan_id) if plan_id is not None else None,
                    int(target_reps) if target_reps is not None else None,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def count_active_goals(self, user_id: int) -> int:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM goals WHERE user_id = ? AND status = 'active'",
                (int(user_id),),
            ).fetchone()
            return int(row["n"] or 0)
        finally:
            conn.close()

    def list_goals(
        self, user_id: int, status: Optional[str] = None
    ) -> List[Dict]:
        conn = self._get_conn()
        try:
            q = "SELECT * FROM goals WHERE user_id = ?"
            params: list = [int(user_id)]
            if status:
                q += " AND status = ?"
                params.append(status)
            q += " ORDER BY (status = 'active') DESC, created_at DESC"
            rows = conn.execute(q, params).fetchall()
            goals = [dict(r) for r in rows]
            if not goals:
                return []
            ids = [g["id"] for g in goals]
            placeholders = ",".join("?" * len(ids))
            ms = conn.execute(
                f"""SELECT * FROM milestones
                    WHERE goal_id IN ({placeholders})
                    ORDER BY goal_id ASC, threshold_value ASC""",
                ids,
            ).fetchall()
            ms_by_goal: Dict[int, List[Dict]] = {}
            for m in ms:
                md = dict(m)
                md["reached"] = bool(md.get("reached", 0))
                ms_by_goal.setdefault(md["goal_id"], []).append(md)
            for g in goals:
                g["milestones"] = ms_by_goal.get(g["id"], [])
            return goals
        finally:
            conn.close()

    def get_goal(self, goal_id: int, user_id: int) -> Optional[Dict]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM goals WHERE id = ? AND user_id = ?",
                (int(goal_id), int(user_id)),
            ).fetchone()
            if not row:
                return None
            goal = dict(row)
            ms = conn.execute(
                "SELECT * FROM milestones WHERE goal_id = ? ORDER BY threshold_value ASC",
                (int(goal_id),),
            ).fetchall()
            goal["milestones"] = [
                {**dict(m), "reached": bool(dict(m).get("reached", 0))} for m in ms
            ]
            return goal
        finally:
            conn.close()

    def update_goal(self, goal_id: int, user_id: int, **fields) -> bool:
        # Redesign Phase 2 adds `target_reps` (strength goals) to the
        # editable set.
        allowed = {"title", "description", "target_value", "status", "target_reps"}
        updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not updates:
            return False
        conn = self._get_conn()
        try:
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            params = list(updates.values()) + [int(goal_id), int(user_id)]
            cur = conn.execute(
                f"UPDATE goals SET {set_clause} WHERE id = ? AND user_id = ?",
                params,
            )
            conn.commit()
            return (cur.rowcount or 0) > 0
        finally:
            conn.close()

    def set_goal_current(
        self,
        goal_id: int,
        current_value: float,
        status: Optional[str] = None,
        completed_at: Optional[str] = None,
    ) -> None:
        conn = self._get_conn()
        try:
            if status is not None:
                conn.execute(
                    """UPDATE goals SET current_value = ?, status = ?, completed_at = ?
                       WHERE id = ?""",
                    (float(current_value), status, completed_at, int(goal_id)),
                )
            else:
                conn.execute(
                    "UPDATE goals SET current_value = ? WHERE id = ?",
                    (float(current_value), int(goal_id)),
                )
            conn.commit()
        finally:
            conn.close()

    def delete_goal(self, goal_id: int, user_id: int) -> bool:
        conn = self._get_conn()
        try:
            cur = conn.execute(
                "DELETE FROM goals WHERE id = ? AND user_id = ?",
                (int(goal_id), int(user_id)),
            )
            conn.commit()
            return (cur.rowcount or 0) > 0
        finally:
            conn.close()

    # ── Session-4: milestones ──────────────────────────────────────────

    def create_milestone(
        self, goal_id: int, label: str, threshold_value: float
    ) -> int:
        conn = self._get_conn()
        try:
            cur = conn.execute(
                """INSERT INTO milestones (goal_id, label, threshold_value, reached)
                   VALUES (?, ?, ?, 0)""",
                (int(goal_id), label, float(threshold_value)),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def mark_milestone_reached(self, milestone_id: int) -> None:
        conn = self._get_conn()
        try:
            now = datetime.now().isoformat(timespec="seconds")
            conn.execute(
                """UPDATE milestones SET reached = 1, reached_at = ?
                   WHERE id = ? AND reached = 0""",
                (now, int(milestone_id)),
            )
            conn.commit()
        finally:
            conn.close()

    def list_milestones(self, user_id: int) -> List[Dict]:
        """Return every milestone for the user's goals, with goal context joined."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT m.*, g.title AS goal_title, g.goal_type, g.exercise
                   FROM milestones m
                   JOIN goals g ON g.id = m.goal_id
                   WHERE g.user_id = ?
                   ORDER BY m.reached DESC, m.reached_at DESC, g.created_at DESC,
                            m.threshold_value ASC""",
                (int(user_id),),
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["reached"] = bool(d.get("reached", 0))
                out.append(d)
            return out
        finally:
            conn.close()

    def list_goal_milestones(self, goal_id: int) -> List[Dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM milestones WHERE goal_id = ? ORDER BY threshold_value ASC",
                (int(goal_id),),
            ).fetchall()
            return [
                {**dict(r), "reached": bool(dict(r).get("reached", 0))} for r in rows
            ]
        finally:
            conn.close()

    def update_milestone_threshold(
        self, milestone_id: int, threshold_value: float, label: Optional[str] = None
    ) -> None:
        """Re-anchor an UNREACHED milestone to a new threshold. No-op if reached."""
        conn = self._get_conn()
        try:
            if label is not None:
                conn.execute(
                    """UPDATE milestones SET threshold_value = ?, label = ?
                       WHERE id = ? AND reached = 0""",
                    (float(threshold_value), label, int(milestone_id)),
                )
            else:
                conn.execute(
                    """UPDATE milestones SET threshold_value = ?
                       WHERE id = ? AND reached = 0""",
                    (float(threshold_value), int(milestone_id)),
                )
            conn.commit()
        finally:
            conn.close()

    # ── Session-4: plans ───────────────────────────────────────────────

    def create_plan(
        self,
        user_id: int,
        *,
        title: str,
        summary: Optional[str],
        start_date: str,
        end_date: str,
        created_by_chat: bool = False,
        conversation_id: Optional[int] = None,
    ) -> int:
        conn = self._get_conn()
        try:
            now = datetime.now().isoformat(timespec="seconds")
            cur = conn.execute(
                """INSERT INTO plans
                       (user_id, title, summary, start_date, end_date, status,
                        created_by_chat, conversation_id, created_at)
                   VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?)""",
                (
                    int(user_id),
                    title.strip(),
                    summary,
                    start_date,
                    end_date,
                    1 if created_by_chat else 0,
                    conversation_id,
                    now,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def create_plan_day(
        self,
        plan_id: int,
        day_date: str,
        exercises_json: str,
        is_rest: bool = False,
    ) -> int:
        conn = self._get_conn()
        try:
            cur = conn.execute(
                """INSERT INTO plan_days
                       (plan_id, day_date, is_rest, exercises, completed)
                   VALUES (?, ?, ?, ?, 0)""",
                (
                    int(plan_id),
                    day_date,
                    1 if is_rest else 0,
                    exercises_json,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def list_plans(self, user_id: int) -> List[Dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT * FROM plans WHERE user_id = ?
                   ORDER BY (status = 'active') DESC, created_at DESC""",
                (int(user_id),),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_plan(self, plan_id: int, user_id: int) -> Optional[Dict]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM plans WHERE id = ? AND user_id = ?",
                (int(plan_id), int(user_id)),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_plan_full(self, plan_id: int, user_id: int) -> Optional[Dict]:
        conn = self._get_conn()
        try:
            plan = conn.execute(
                "SELECT * FROM plans WHERE id = ? AND user_id = ?",
                (int(plan_id), int(user_id)),
            ).fetchone()
            if not plan:
                return None
            days = conn.execute(
                "SELECT * FROM plan_days WHERE plan_id = ? ORDER BY day_date ASC",
                (int(plan_id),),
            ).fetchall()
            plan_d = dict(plan)
            day_list = []
            for d in days:
                dd = dict(d)
                dd["is_rest"] = bool(dd.get("is_rest", 0))
                dd["completed"] = bool(dd.get("completed", 0))
                try:
                    dd["exercises"] = json.loads(dd.get("exercises") or "[]")
                except (ValueError, TypeError):
                    dd["exercises"] = []
                day_list.append(dd)
            plan_d["days"] = day_list
            return plan_d
        finally:
            conn.close()

    def get_plan_day(self, plan_day_id: int, user_id: int) -> Optional[Dict]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT pd.*, p.user_id AS plan_user_id, p.id AS plan_id_join
                   FROM plan_days pd
                   JOIN plans p ON p.id = pd.plan_id
                   WHERE pd.id = ? AND p.user_id = ?""",
                (int(plan_day_id), int(user_id)),
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            d["is_rest"] = bool(d.get("is_rest", 0))
            d["completed"] = bool(d.get("completed", 0))
            try:
                d["exercises"] = json.loads(d.get("exercises") or "[]")
            except (ValueError, TypeError):
                d["exercises"] = []
            return d
        finally:
            conn.close()

    def mark_plan_day_completed(self, plan_day_id: int, user_id: int) -> bool:
        """Mark the day complete; returns True on success. Verifies ownership."""
        conn = self._get_conn()
        try:
            # Ownership check via JOIN
            owned = conn.execute(
                """SELECT pd.id FROM plan_days pd
                   JOIN plans p ON p.id = pd.plan_id
                   WHERE pd.id = ? AND p.user_id = ?""",
                (int(plan_day_id), int(user_id)),
            ).fetchone()
            if not owned:
                return False
            now = datetime.now().isoformat(timespec="seconds")
            conn.execute(
                """UPDATE plan_days SET completed = 1, completed_at = ?
                   WHERE id = ?""",
                (now, int(plan_day_id)),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def plan_all_days_completed(self, plan_id: int) -> bool:
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT
                       COUNT(*) AS total,
                       SUM(CASE WHEN completed = 1 OR is_rest = 1 THEN 1 ELSE 0 END) AS done
                   FROM plan_days WHERE plan_id = ?""",
                (int(plan_id),),
            ).fetchone()
            total = int(row["total"] or 0)
            done = int(row["done"] or 0)
            return total > 0 and total == done
        finally:
            conn.close()

    def set_plan_status(self, plan_id: int, status: str) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE plans SET status = ? WHERE id = ?",
                (status, int(plan_id)),
            )
            conn.commit()
        finally:
            conn.close()

    def delete_plan(self, plan_id: int, user_id: int) -> bool:
        conn = self._get_conn()
        try:
            cur = conn.execute(
                "DELETE FROM plans WHERE id = ? AND user_id = ?",
                (int(plan_id), int(user_id)),
            )
            conn.commit()
            return (cur.rowcount or 0) > 0
        finally:
            conn.close()

    # ── Session-5: manual plan + plan_day edits ────────────────────────

    _PLAN_PATCH_COLS = ("title", "summary", "start_date", "end_date")
    _PLAN_DAY_PATCH_COLS = ("day_date", "is_rest", "exercises")

    def update_plan(self, plan_id: int, user_id: int, patch: Dict) -> Optional[Dict]:
        """PATCH whitelisted plan fields. Returns updated plan dict, or None if
        not owned. Caller supplies already-validated values.
        """
        cleaned: Dict[str, Any] = {}
        for key in self._PLAN_PATCH_COLS:
            if key in patch and patch[key] is not None:
                cleaned[key] = patch[key]
        if not cleaned:
            return self.get_plan(plan_id, user_id)
        conn = self._get_conn()
        try:
            owned = conn.execute(
                "SELECT id FROM plans WHERE id = ? AND user_id = ?",
                (int(plan_id), int(user_id)),
            ).fetchone()
            if not owned:
                return None
            set_clause = ", ".join(f"{k} = ?" for k in cleaned.keys())
            params = list(cleaned.values()) + [int(plan_id)]
            conn.execute(
                f"UPDATE plans SET {set_clause} WHERE id = ?",
                params,
            )
            conn.commit()
        finally:
            conn.close()
        return self.get_plan(plan_id, user_id)

    def update_plan_day(
        self,
        plan_id: int,
        plan_day_id: int,
        user_id: int,
        patch: Dict,
    ) -> Optional[Dict]:
        """PATCH whitelisted plan_day fields. Rejects edits to completed days.

        Returns (status, day) where status is one of:
          'ok'       : updated successfully, day = updated row
          'not_found': plan_day not owned by user
          'locked'   : day is already completed, refused
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT pd.* FROM plan_days pd
                   JOIN plans p ON p.id = pd.plan_id
                   WHERE pd.id = ? AND pd.plan_id = ? AND p.user_id = ?""",
                (int(plan_day_id), int(plan_id), int(user_id)),
            ).fetchone()
            if not row:
                return {"status": "not_found"}
            if int(row["completed"] or 0) == 1:
                return {"status": "locked"}
            cleaned: Dict[str, Any] = {}
            if "day_date" in patch and patch["day_date"]:
                cleaned["day_date"] = str(patch["day_date"])
            if "is_rest" in patch:
                cleaned["is_rest"] = 1 if patch["is_rest"] else 0
            if "exercises" in patch:
                # Caller has already sanitised — serialise as JSON.
                cleaned["exercises"] = json.dumps(patch["exercises"])
            if not cleaned:
                return {"status": "ok", "day": dict(row)}
            set_clause = ", ".join(f"{k} = ?" for k in cleaned.keys())
            params = list(cleaned.values()) + [int(plan_day_id)]
            conn.execute(
                f"UPDATE plan_days SET {set_clause} WHERE id = ?",
                params,
            )
            conn.commit()
            row2 = conn.execute(
                "SELECT * FROM plan_days WHERE id = ?",
                (int(plan_day_id),),
            ).fetchone()
            d = dict(row2) if row2 else {}
            if d:
                d["is_rest"] = bool(d.get("is_rest", 0))
                d["completed"] = bool(d.get("completed", 0))
                try:
                    d["exercises"] = json.loads(d.get("exercises") or "[]")
                except (ValueError, TypeError):
                    d["exercises"] = []
            return {"status": "ok", "day": d}
        finally:
            conn.close()

    def insert_plan_day(
        self,
        plan_id: int,
        user_id: int,
        *,
        day_date: str,
        is_rest: bool,
        exercises: Any,
    ) -> Optional[int]:
        """Append a new day to an existing plan. Returns new day_id or None."""
        conn = self._get_conn()
        try:
            owned = conn.execute(
                "SELECT id FROM plans WHERE id = ? AND user_id = ?",
                (int(plan_id), int(user_id)),
            ).fetchone()
            if not owned:
                return None
            cur = conn.execute(
                """INSERT INTO plan_days
                       (plan_id, day_date, is_rest, exercises, completed)
                   VALUES (?, ?, ?, ?, 0)""",
                (
                    int(plan_id),
                    str(day_date),
                    1 if is_rest else 0,
                    json.dumps(exercises or []),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def delete_plan_day(
        self, plan_id: int, plan_day_id: int, user_id: int
    ) -> str:
        """Delete a plan_day. Rejects if completed.

        Returns one of: 'ok', 'not_found', 'locked'.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT pd.id, pd.completed FROM plan_days pd
                   JOIN plans p ON p.id = pd.plan_id
                   WHERE pd.id = ? AND pd.plan_id = ? AND p.user_id = ?""",
                (int(plan_day_id), int(plan_id), int(user_id)),
            ).fetchone()
            if not row:
                return "not_found"
            if int(row["completed"] or 0) == 1:
                return "locked"
            conn.execute(
                "DELETE FROM plan_days WHERE id = ?",
                (int(plan_day_id),),
            )
            conn.commit()
            return "ok"
        finally:
            conn.close()

    def get_todays_plan_day(self, user_id: int) -> Optional[Dict]:
        """Return today's plan_day from the user's active plan, or None."""
        today_iso = datetime.now().strftime("%Y-%m-%d")
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT pd.*, p.id AS plan_id, p.title AS plan_title
                   FROM plan_days pd
                   JOIN plans p ON p.id = pd.plan_id
                   WHERE p.user_id = ? AND p.status = 'active'
                     AND DATE(pd.day_date) = ?
                   ORDER BY p.created_at DESC
                   LIMIT 1""",
                (int(user_id), today_iso),
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            d["is_rest"] = bool(d.get("is_rest", 0))
            d["completed"] = bool(d.get("completed", 0))
            try:
                d["exercises"] = json.loads(d.get("exercises") or "[]")
            except (ValueError, TypeError):
                d["exercises"] = []
            return d
        finally:
            conn.close()

    # ── Session-4: badges ──────────────────────────────────────────────

    def insert_badge(
        self,
        user_id: int,
        badge_key: str,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Insert a badge. Returns False if UNIQUE(user_id, badge_key) conflicts."""
        conn = self._get_conn()
        try:
            now = datetime.now().isoformat(timespec="seconds")
            try:
                conn.execute(
                    """INSERT INTO user_badges (user_id, badge_key, earned_at, metadata)
                       VALUES (?, ?, ?, ?)""",
                    (
                        int(user_id),
                        badge_key,
                        now,
                        json.dumps(metadata or {}),
                    ),
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False
        finally:
            conn.close()

    def list_badges(self, user_id: int) -> List[Dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT * FROM user_badges WHERE user_id = ?
                   ORDER BY earned_at DESC""",
                (int(user_id),),
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                try:
                    d["metadata"] = json.loads(d.get("metadata") or "{}")
                except (ValueError, TypeError):
                    d["metadata"] = {}
                out.append(d)
            return out
        finally:
            conn.close()

    def has_badge(self, user_id: int, badge_key: str) -> bool:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT 1 FROM user_badges WHERE user_id = ? AND badge_key = ?",
                (int(user_id), badge_key),
            ).fetchone()
            return row is not None
        finally:
            conn.close()
