"""Replay a recorded session's elbow angles through the current PushUpDetector
state-machine thresholds and report how many reps *would* have counted.

This is a minimal state-machine re-simulation — it uses ONLY the elbow angle
from the trace (no pose objects) so it can't exercise exit-from-plank, form
scoring, or voice cues. For that run the live app. This script's only job is
to answer: with the current ELBOW_DOWN / ELBOW_UP / MIN_BOTTOM_FRAMES / etc.,
how many reps does the trace produce?

Usage:
    python scripts/one_off/replay_session_rep_count.py [session_path]

Defaults to the 2026-04-12 failed push-up session.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.pushup_detector import (  # noqa: E402
    ELBOW_UP, ELBOW_DOWN, MIN_BOTTOM_FRAMES, MIN_REP_DURATION,
)


def replay(session_dir: Path) -> dict:
    trace_path = session_dir / "trace.jsonl"
    rows = [json.loads(l) for l in trace_path.read_text().splitlines() if l.strip()]

    state = "up"
    bottom_frames = 0
    rep_start_t = 0.0
    rep_count = 0
    attempts = []
    cur_attempt = None

    for r in rows:
        if not r.get("form_validated"):
            continue
        angles = r.get("angles") or {}
        elbow = angles.get("elbow")
        if elbow is None:
            continue
        t = r.get("t", 0.0)
        f = r.get("frame", 0)

        prev = state

        if state == "up":
            if elbow < ELBOW_UP - 10:
                state = "going_down"
                rep_start_t = t
                bottom_frames = 0
                cur_attempt = {"start_f": f, "start_t": t, "min_e": elbow}
        elif state == "going_down":
            if cur_attempt is not None and elbow < cur_attempt["min_e"]:
                cur_attempt["min_e"] = elbow
            if elbow <= ELBOW_DOWN:
                state = "down"
                bottom_frames = 0
            elif elbow > ELBOW_UP:
                state = "up"
                if cur_attempt:
                    cur_attempt.update({"end_f": f, "end_t": t, "result": "too_shallow"})
                    attempts.append(cur_attempt)
                    cur_attempt = None
        elif state == "down":
            if cur_attempt is not None and elbow < cur_attempt["min_e"]:
                cur_attempt["min_e"] = elbow
            bottom_frames += 1
            if elbow > ELBOW_DOWN + 10 and bottom_frames >= MIN_BOTTOM_FRAMES:
                state = "going_up"
        elif state == "going_up":
            if elbow >= ELBOW_UP:
                elapsed = t - rep_start_t
                if elapsed >= MIN_REP_DURATION:
                    rep_count += 1
                    if cur_attempt:
                        cur_attempt.update({
                            "end_f": f, "end_t": t,
                            "result": "counted", "duration": round(elapsed, 2),
                        })
                        attempts.append(cur_attempt)
                        cur_attempt = None
                else:
                    if cur_attempt:
                        cur_attempt.update({
                            "end_f": f, "end_t": t,
                            "result": "too_fast", "duration": round(elapsed, 2),
                        })
                        attempts.append(cur_attempt)
                        cur_attempt = None
                state = "up"
            elif elbow < ELBOW_DOWN:
                state = "down"
                bottom_frames = 0

    return {
        "session": session_dir.name,
        "total_frames": len(rows),
        "rep_count": rep_count,
        "attempts": attempts,
        "thresholds": {
            "ELBOW_UP": ELBOW_UP,
            "ELBOW_DOWN": ELBOW_DOWN,
            "MIN_BOTTOM_FRAMES": MIN_BOTTOM_FRAMES,
            "MIN_REP_DURATION": MIN_REP_DURATION,
        },
    }


def main():
    default = PROJECT_ROOT / "data/sessions/2026-04-12_23-39-05_pushup_690a20"
    session_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    result = replay(session_dir)
    print(f"Session: {result['session']}")
    print(f"Total frames: {result['total_frames']}")
    print(f"Thresholds: {result['thresholds']}")
    print(f"\nReps counted: {result['rep_count']}")
    print(f"\nAll attempts ({len(result['attempts'])}):")
    for i, a in enumerate(result["attempts"], 1):
        res = a.get("result", "?")
        dur = a.get("duration", "-")
        print(
            f"  #{i:>2}  f{a['start_f']:>4}-{a.get('end_f', '?'):>4}  "
            f"t={a['start_t']:6.2f}-{a.get('end_t', 0):6.2f}s  "
            f"min_elbow={a['min_e']:5.1f}  dur={dur}  result={res}"
        )


if __name__ == "__main__":
    main()
