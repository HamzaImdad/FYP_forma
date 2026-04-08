"""
SQLite persistence layer for ExerVision session history.

Stores workout sessions, per-rep details, and set breakdowns.
Provides query methods for dashboard aggregation (daily/weekly/monthly).
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


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
        """Create tables if they don't exist."""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exercise TEXT NOT NULL,
                    classifier TEXT NOT NULL DEFAULT 'rule_based',
                    date TEXT NOT NULL,
                    duration_sec REAL NOT NULL DEFAULT 0,
                    total_reps INTEGER NOT NULL DEFAULT 0,
                    good_reps INTEGER NOT NULL DEFAULT 0,
                    avg_form_score REAL NOT NULL DEFAULT 0
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
            conn.commit()
        finally:
            conn.close()

    def save_session(self, summary: Dict[str, Any]) -> int:
        """Save a complete workout session. Returns the session ID."""
        conn = self._get_conn()
        try:
            now = datetime.now().isoformat()

            cursor = conn.execute(
                """INSERT INTO sessions (exercise, classifier, date, duration_sec,
                   total_reps, good_reps, avg_form_score)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    summary.get("exercise", "unknown"),
                    summary.get("classifier", "rule_based"),
                    now,
                    summary.get("duration_sec", 0),
                    summary.get("total_reps", 0),
                    summary.get("good_reps", 0),
                    summary.get("avg_form_score", 0),
                ),
            )
            session_id = cursor.lastrowid

            # Save per-rep details
            reps = summary.get("reps", [])
            for rep in reps:
                issues = rep.get("issues", [])
                conn.execute(
                    """INSERT INTO reps (session_id, rep_num, form_score, quality, issues, duration)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        rep.get("rep_num", rep.get("rep", 0)),
                        rep.get("form_score", rep.get("score", 0)),
                        rep.get("quality", "moderate"),
                        json.dumps(issues),
                        rep.get("duration", 0),
                    ),
                )

            # Save set breakdowns
            reps_per_set = summary.get("reps_per_set", [])
            for i, reps_count in enumerate(reps_per_set):
                conn.execute(
                    "INSERT INTO sets (session_id, set_num, reps_count) VALUES (?, ?, ?)",
                    (session_id, i + 1, reps_count),
                )

            conn.commit()
            return session_id
        finally:
            conn.close()

    def get_sessions(
        self,
        exercise: Optional[str] = None,
        period: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict]:
        """Query sessions with optional filters."""
        conn = self._get_conn()
        try:
            query = "SELECT * FROM sessions"
            params: list = []
            conditions = []

            if exercise:
                conditions.append("exercise = ?")
                params.append(exercise)

            date_filter = self._period_to_date(period)
            if date_filter:
                conditions.append("date >= ?")
                params.append(date_filter)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY date DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_session_detail(self, session_id: int) -> Optional[Dict]:
        """Get full session with reps and sets."""
        conn = self._get_conn()
        try:
            session = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if not session:
                return None

            result = dict(session)

            reps = conn.execute(
                "SELECT * FROM reps WHERE session_id = ? ORDER BY rep_num",
                (session_id,),
            ).fetchall()
            result["reps"] = [
                {**dict(r), "issues": json.loads(r["issues"])} for r in reps
            ]

            sets = conn.execute(
                "SELECT * FROM sets WHERE session_id = ? ORDER BY set_num",
                (session_id,),
            ).fetchall()
            result["sets"] = [dict(s) for s in sets]

            return result
        finally:
            conn.close()

    def get_stats(
        self, period: Optional[str] = None, exercise: Optional[str] = None
    ) -> Dict:
        """Aggregate stats for dashboard summary cards."""
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
            """
            params: list = []
            conditions = []

            if exercise:
                conditions.append("exercise = ?")
                params.append(exercise)

            date_filter = self._period_to_date(period)
            if date_filter:
                conditions.append("date >= ?")
                params.append(date_filter)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            row = conn.execute(query, params).fetchone()
            return {
                "total_workouts": row["total_workouts"],
                "total_reps": row["total_reps"],
                "avg_score": round(row["avg_score"], 3),
                "total_time_sec": round(row["total_time_sec"], 1),
                "total_good_reps": row["total_good_reps"],
            }
        finally:
            conn.close()

    def get_daily_scores(
        self, days: int = 30, exercise: Optional[str] = None
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
                WHERE date >= ?
            """
            params: list = [cutoff]

            if exercise:
                query += " AND exercise = ?"
                params.append(exercise)

            query += " GROUP BY DATE(date) ORDER BY day"

            rows = conn.execute(query, params).fetchall()
            return {
                "dates": [r["day"] for r in rows],
                "scores": [round(r["avg_score"], 3) for r in rows],
                "reps": [r["total_reps"] for r in rows],
                "session_counts": [r["session_count"] for r in rows],
            }
        finally:
            conn.close()

    def get_exercise_distribution(self, period: Optional[str] = None) -> List[Dict]:
        """Exercise breakdown for pie/doughnut chart."""
        conn = self._get_conn()
        try:
            query = """
                SELECT exercise, COUNT(*) as count, SUM(total_reps) as total_reps
                FROM sessions
            """
            params: list = []

            date_filter = self._period_to_date(period)
            if date_filter:
                query += " WHERE date >= ?"
                params.append(date_filter)

            query += " GROUP BY exercise ORDER BY count DESC"

            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
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
