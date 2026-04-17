"""
FORMA Redesign — one-shot backfill for historical data.

Three idempotent passes, safe to re-run:

  1. Recompute sessions.good_reps under the new GOOD_REP_THRESHOLD (0.6)
     by counting reps where form_score >= threshold. Pre-redesign
     pipeline used 0.7, so old sessions will gain a few good_reps.

  2. Populate sessions.avg_form_score where it's missing (NULL or 0)
     by averaging that session's rep form_scores.

  3. Populate sets.weight_kg from parent sessions.weight_kg for rows
     where the per-set column is NULL. Coarse approximation — users
     who switched weight mid-session pre-migration get the session's
     starting weight on every set. Fresh sessions after Phase 2 ship
     will write per-set weights directly.

Usage:
    "C:/Users/Hamza/AppData/Local/Programs/Python/Python310/python.exe" \
        -m scripts.phase_backfill

Flags:
    --dry-run       print planned changes, do not write
    --threshold X   override threshold for pass 1 (default: GOOD_REP_THRESHOLD)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.database import ExerVisionDB
from app.exercise_registry import GOOD_REP_THRESHOLD


def _count(conn, sql: str, *params) -> int:
    row = conn.execute(sql, params).fetchone()
    return int((row[0] if row is not None else 0) or 0)


def _print_header(title: str) -> None:
    print()
    print("-" * 64)
    print(title)
    print("-" * 64)


def run(dry_run: bool = False, threshold: float = GOOD_REP_THRESHOLD) -> None:
    db = ExerVisionDB()
    conn = db._get_conn()
    try:
        # ── Pass 1: sessions.good_reps ────────────────────────────────
        # Only recompute sessions where the per-rep data is complete
        # (reps rowcount == total_reps AND total_reps > 0). Partial-data
        # sessions keep their legacy good_reps (pre-Session-1 pipeline
        # sometimes marked nearly every rep good without writing per-rep
        # form scores — recomputing from rows would undercount).
        _print_header(
            f"Pass 1 — recompute sessions.good_reps @ threshold {threshold}"
        )
        total_sessions = _count(conn, "SELECT COUNT(*) FROM sessions")
        eligible = _count(
            conn,
            "SELECT COUNT(*) FROM sessions s "
            "WHERE s.total_reps > 0 "
            "  AND s.total_reps = "
            "      (SELECT COUNT(*) FROM reps r WHERE r.session_id = s.id)",
        )
        skipped = total_sessions - eligible
        before = _count(conn, "SELECT COALESCE(SUM(good_reps), 0) FROM sessions")
        eligible_before = _count(
            conn,
            "SELECT COALESCE(SUM(good_reps), 0) FROM sessions s "
            "WHERE s.total_reps > 0 "
            "  AND s.total_reps = "
            "      (SELECT COUNT(*) FROM reps r WHERE r.session_id = s.id)",
        )
        eligible_after = _count(
            conn,
            "SELECT COALESCE(SUM(n), 0) FROM ("
            "  SELECT COUNT(*) AS n FROM reps r "
            "  JOIN sessions s ON s.id = r.session_id "
            "  WHERE r.form_score >= ? "
            "    AND s.total_reps > 0 "
            "    AND s.total_reps = "
            "        (SELECT COUNT(*) FROM reps r2 WHERE r2.session_id = s.id)"
            "  GROUP BY r.session_id"
            ")",
            threshold,
        )
        print(f"  sessions total          : {total_sessions}")
        print(f"  eligible (complete reps): {eligible}")
        print(f"  skipped (partial/empty) : {skipped}")
        print(f"  eligible good_reps before: {eligible_before}")
        print(f"  eligible good_reps after : {eligible_after}")
        print(f"  delta (eligible only)    : {eligible_after - eligible_before:+d}")
        print(f"  SUM(good_reps) overall before: {before}")
        print(f"  SUM(good_reps) overall after : "
              f"{before - eligible_before + eligible_after}")
        if not dry_run:
            conn.execute(
                """UPDATE sessions SET good_reps = COALESCE((
                       SELECT COUNT(*) FROM reps r
                       WHERE r.session_id = sessions.id
                         AND r.form_score >= ?
                   ), 0)
                   WHERE total_reps > 0
                     AND total_reps = (
                       SELECT COUNT(*) FROM reps r2 WHERE r2.session_id = sessions.id
                     )""",
                (threshold,),
            )
            conn.commit()
            print("  [ok] written")

        # ── Pass 2: sessions.avg_form_score ───────────────────────────
        _print_header("Pass 2 — backfill missing sessions.avg_form_score")
        needs_fill = _count(
            conn,
            "SELECT COUNT(*) FROM sessions "
            "WHERE (avg_form_score IS NULL OR avg_form_score = 0) "
            "  AND EXISTS (SELECT 1 FROM reps r WHERE r.session_id = sessions.id)",
        )
        print(f"  sessions needing backfill: {needs_fill}")
        if not dry_run and needs_fill > 0:
            conn.execute(
                """UPDATE sessions SET avg_form_score = COALESCE((
                       SELECT AVG(r.form_score) FROM reps r
                       WHERE r.session_id = sessions.id
                   ), 0)
                   WHERE (avg_form_score IS NULL OR avg_form_score = 0)
                     AND EXISTS (
                       SELECT 1 FROM reps r WHERE r.session_id = sessions.id
                     )"""
            )
            conn.commit()
            print("  [ok] written")
        elif needs_fill == 0:
            print("  [ok] nothing to do")

        # ── Pass 3: sets.weight_kg ────────────────────────────────────
        _print_header("Pass 3 — backfill sets.weight_kg from parent session")
        sets_missing = _count(
            conn, "SELECT COUNT(*) FROM sets WHERE weight_kg IS NULL"
        )
        sets_with_session_weight = _count(
            conn,
            "SELECT COUNT(*) FROM sets st "
            "JOIN sessions s ON s.id = st.session_id "
            "WHERE st.weight_kg IS NULL "
            "  AND s.weight_kg IS NOT NULL AND s.weight_kg > 0",
        )
        print(f"  sets missing weight_kg               : {sets_missing}")
        print(f"  of which have session-level weight   : {sets_with_session_weight}")
        if not dry_run and sets_with_session_weight > 0:
            conn.execute(
                """UPDATE sets SET weight_kg = (
                       SELECT s.weight_kg FROM sessions s
                       WHERE s.id = sets.session_id
                   )
                   WHERE weight_kg IS NULL
                     AND session_id IN (
                       SELECT id FROM sessions
                       WHERE weight_kg IS NOT NULL AND weight_kg > 0
                     )"""
            )
            conn.commit()
            print("  [ok] written")
        elif sets_with_session_weight == 0:
            print("  [ok] nothing to do")

        _print_header("Summary")
        good_total = _count(
            conn, "SELECT COALESCE(SUM(good_reps), 0) FROM sessions"
        )
        total = _count(
            conn, "SELECT COALESCE(SUM(total_reps), 0) FROM sessions"
        )
        ratio = (good_total / total) if total else 0.0
        print(f"  good_reps / total_reps : {good_total} / {total}  ({ratio:.1%})")
        sets_w = _count(
            conn, "SELECT COUNT(*) FROM sets WHERE weight_kg IS NOT NULL"
        )
        sets_all = _count(conn, "SELECT COUNT(*) FROM sets")
        print(f"  sets with weight_kg    : {sets_w} / {sets_all}")
        print()
        if dry_run:
            print("DRY RUN — no changes written.")
        else:
            print("Backfill complete. Safe to re-run; second run is a no-op.")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--threshold", type=float, default=GOOD_REP_THRESHOLD)
    args = parser.parse_args()
    run(dry_run=args.dry_run, threshold=args.threshold)


if __name__ == "__main__":
    main()
