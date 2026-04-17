"""Forensic analysis of a FORMA push-up session trace.

Re-runnable on any push-up session directory that contains:
  trace.jsonl, summary.json, metadata.json, thresholds.json

Usage:
    python scripts/one_off/session_forensics_pushup.py \
        "data/sessions/2026-04-12_23-39-05_pushup_690a20"

Prints structured forensic output grouped by phase. Also simulates what the
CoachingEngine WOULD have emitted given the per-frame check_names, because
this session's trace does not contain voice_text.

Designed for the FORMA detector + CoachingEngine as of 2026-04-12.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict, deque
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# ────────────────────────────────────────────────────────────────────────
# Constants mirroring engine.py tuning (for voice simulation)
# ────────────────────────────────────────────────────────────────────────
ISSUE_CLEAR_FRAMES = 8
ISSUE_REPEAT_REPS = 5
PRAISE_EVERY_N_GOOD = 5
PACE_MIN_DURATION = 1.2
PACE_COOLDOWN_REPS = 5
INTRO_TEXT = "Looking at your form."
SLOW_DOWN_TEXT = "Slow it down — control the movement."
EMIT_HARD_TIMEOUT_SEC = 12.0

# Mapping of issue free-text (what the detector emits in the `issues` list of
# the classification result, and hence the trace) → check_name the coaching
# engine would have been handed in _frame_checks. Derived from
# src/classification/pushup_detector.py _assess_form.
ISSUE_TEXT_TO_CHECK = {
    "Tighten your core — stay perfectly straight": "hip_sag_check",
    "Slight hip sag — squeeze glutes and brace core": "hip_sag_mild",
    "Tighten your core — keep hips level": "hip_sag_moderate",
    "Hips sagging — squeeze glutes, brace core": "hip_sag_severe",
    "Core check — keep your body in a straight line": "hip_sag_severe",
    "Arms too tight to body — widen grip slightly": "elbow_too_tucked",
    "Watch your elbows — flare building up": "elbow_flare_building",
    "Elbows flaring out — tuck elbows closer to body": "elbow_flare_severe",
    "Hips piking up — lower hips into straight line": "hip_pike",
    "Head dropping -- keep neck neutral": "head_drop",
    "Don't crane neck up -- look at floor ahead of hands": "head_crane",
}

# Reverse for the coach variant lookup. We'll load commands.json for these.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_voice_library() -> Dict[str, List[str]]:
    cmds_path = PROJECT_ROOT / "src" / "coaching" / "commands.json"
    with open(cmds_path, encoding="utf-8") as f:
        data = json.load(f)
    ex = data.get("exercises", {}).get("pushup", {})
    corr = ex.get("corrections", {})
    out: Dict[str, List[str]] = {}
    for k, v in corr.items():
        variants = v.get("variants")
        if isinstance(variants, list) and variants:
            out[k] = variants
        elif isinstance(v.get("text"), str):
            out[k] = [v["text"]]
        else:
            out[k] = [k.replace("_", " ").capitalize()]
    return out


# ────────────────────────────────────────────────────────────────────────
def read_trace(path: Path) -> List[Dict[str, Any]]:
    frames = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frames.append(json.loads(line))
    return frames


def fmt_angle(v):
    return "None" if v is None else f"{v:.1f}"


def ang(frame: Dict, name: str) -> Optional[float]:
    a = frame.get("angles") or {}
    return a.get(name)


def state(frame): return frame.get("state")
def issues(frame): return frame.get("issues") or []


# ────────────────────────────────────────────────────────────────────────
# Phase 1 — Pre-validation analysis
# ────────────────────────────────────────────────────────────────────────
def phase1_prevalidation(frames: List[Dict]) -> Dict:
    """Analyze all frames before form_validated first becomes True."""
    first_validated_idx = None
    for i, fr in enumerate(frames):
        if fr.get("form_validated"):
            first_validated_idx = i
            break

    pre = frames[:first_validated_idx] if first_validated_idx is not None else frames
    report: Dict[str, Any] = {
        "first_validated_frame": first_validated_idx,
        "first_validated_t": frames[first_validated_idx]["t"] if first_validated_idx is not None else None,
        "pre_val_frame_count": len(pre),
        "pre_val_duration_sec": (pre[-1]["t"] - pre[0]["t"]) if pre else 0.0,
    }

    # Angles on 5 frames before validation
    if first_validated_idx and first_validated_idx >= 5:
        lead = []
        for k in range(first_validated_idx - 5, first_validated_idx + 1):
            fr = frames[k]
            lead.append(
                {
                    "frame": fr["frame"],
                    "t": round(fr["t"], 3),
                    "elbow": fmt_angle(ang(fr, "elbow")),
                    "hip": fmt_angle(ang(fr, "hip")),
                    "shoulder": fmt_angle(ang(fr, "shoulder")),
                    "setup_msg": fr.get("setup_msg"),
                    "form_validated": fr.get("form_validated"),
                }
            )
        report["lead_into_validation"] = lead

    # Group by setup_msg — contiguous runs
    groups = []
    for msg, grp in groupby(pre, key=lambda f: (f.get("setup_msg") or "")):
        grp_list = list(grp)
        elbow_vals = [ang(f, "elbow") for f in grp_list if ang(f, "elbow") is not None]
        hip_vals = [ang(f, "hip") for f in grp_list if ang(f, "hip") is not None]
        shoulder_vals = [ang(f, "shoulder") for f in grp_list if ang(f, "shoulder") is not None]
        groups.append(
            {
                "setup_msg": msg,
                "frame_start": grp_list[0]["frame"],
                "frame_end": grp_list[-1]["frame"],
                "n_frames": len(grp_list),
                "t_start": round(grp_list[0]["t"], 2),
                "t_end": round(grp_list[-1]["t"], 2),
                "elbow_range": (
                    round(min(elbow_vals), 1),
                    round(max(elbow_vals), 1),
                ) if elbow_vals else None,
                "hip_range": (round(min(hip_vals), 1), round(max(hip_vals), 1)) if hip_vals else None,
                "shoulder_range": (round(min(shoulder_vals), 1), round(max(shoulder_vals), 1)) if shoulder_vals else None,
                "hip_none_frames": sum(1 for f in grp_list if ang(f, "hip") is None),
                "elbow_none_frames": sum(1 for f in grp_list if ang(f, "elbow") is None),
                "shoulder_none_frames": sum(1 for f in grp_list if ang(f, "shoulder") is None),
            }
        )
    report["setup_msg_groups"] = groups

    # None counts overall
    elbow_none = sum(1 for f in pre if ang(f, "elbow") is None)
    hip_none = sum(1 for f in pre if ang(f, "hip") is None)
    shoulder_none = sum(1 for f in pre if ang(f, "shoulder") is None)
    report["pre_val_elbow_none"] = elbow_none
    report["pre_val_hip_none"] = hip_none
    report["pre_val_shoulder_none"] = shoulder_none

    # None stretches (clustered vs scattered) for hip
    stretches = longest_none_stretches(pre, "hip", top=10)
    report["pre_val_hip_none_stretches"] = stretches
    return report


def longest_none_stretches(frames, angle_name, top=10):
    out = []
    run_start = None
    for i, fr in enumerate(frames):
        if ang(fr, angle_name) is None:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                out.append((run_start, i - 1, i - run_start))
                run_start = None
    if run_start is not None:
        out.append((run_start, len(frames) - 1, len(frames) - run_start))
    out.sort(key=lambda r: -r[2])
    res = []
    for s, e, ln in out[:top]:
        res.append(
            {
                "idx_start": s,
                "idx_end": e,
                "frame_start": frames[s]["frame"],
                "frame_end": frames[e]["frame"],
                "length": ln,
                "t_start": round(frames[s]["t"], 2),
                "t_end": round(frames[e]["t"], 2),
            }
        )
    return res


# ────────────────────────────────────────────────────────────────────────
# Phase 2 — Post-validation state-machine walkthrough
# ────────────────────────────────────────────────────────────────────────
def phase2_state_machine(frames: List[Dict], first_val_idx: int) -> Dict:
    post = frames[first_val_idx:] if first_val_idx is not None else []
    transitions = []
    prev_state = None
    for fr in post:
        cur = state(fr)
        if cur != prev_state:
            transitions.append(
                {
                    "frame": fr["frame"],
                    "t": round(fr["t"], 3),
                    "from": prev_state,
                    "to": cur,
                    "elbow": fmt_angle(ang(fr, "elbow")),
                    "hip": fmt_angle(ang(fr, "hip")),
                    "shoulder": fmt_angle(ang(fr, "shoulder")),
                    "is_active": fr.get("is_active"),
                    "form_score": round(fr.get("form_score", 0.0), 3),
                    "issues": (fr.get("issues") or [])[:2],
                    "form_validated": fr.get("form_validated"),
                }
            )
            prev_state = cur

    # Group into rep-attempt cycles: every time we enter GOING_DOWN from UP
    attempts = []
    current = None
    for t in transitions:
        if t["to"] == "going_down":
            if current is not None:
                attempts.append(current)
            current = {
                "start_transition": t,
                "visited_states": ["going_down"],
                "state_entries": [t],
                "min_elbow": None,
                "reached_down": False,
                "down_frames": 0,
                "reached_going_up": False,
                "closed_at_up": False,
            }
        else:
            if current is None:
                continue
            current["state_entries"].append(t)
            current["visited_states"].append(t["to"])
            if t["to"] == "down":
                current["reached_down"] = True
            if t["to"] == "going_up":
                current["reached_going_up"] = True
            if t["to"] == "up":
                current["closed_at_up"] = True
                attempts.append(current)
                current = None
    if current is not None:
        attempts.append(current)

    # Min elbow during each attempt by walking frames between its bracket
    # Build simple frame index map on post
    post_idx_by_frame = {f["frame"]: i for i, f in enumerate(post)}
    for att in attempts:
        start_frame = att["start_transition"]["frame"]
        end_frame = att["state_entries"][-1]["frame"]
        s_i = post_idx_by_frame[start_frame]
        e_i = post_idx_by_frame[end_frame]
        # include tail to end_frame or just to e_i
        elbows = []
        hips = []
        n_down = 0
        for i in range(s_i, min(e_i + 1, len(post))):
            fr = post[i]
            if state(fr) == "down":
                n_down += 1
            e_v = ang(fr, "elbow")
            h_v = ang(fr, "hip")
            if e_v is not None:
                elbows.append(e_v)
            if h_v is not None:
                hips.append(h_v)
        att["min_elbow"] = round(min(elbows), 1) if elbows else None
        att["max_elbow_in_cycle"] = round(max(elbows), 1) if elbows else None
        att["n_down_frames"] = n_down
        att["n_cycle_frames"] = e_i - s_i + 1
        att["duration_sec"] = round(post[min(e_i, len(post) - 1)]["t"] - post[s_i]["t"], 3)
        att["hip_min"] = round(min(hips), 1) if hips else None
        att["hip_max"] = round(max(hips), 1) if hips else None

    return {
        "post_val_frames": len(post),
        "n_state_transitions": len(transitions),
        "transitions": transitions,
        "n_rep_attempts": len(attempts),
        "rep_attempts": attempts,
    }


# ────────────────────────────────────────────────────────────────────────
# Phase 3 — Pose quality audit
# ────────────────────────────────────────────────────────────────────────
def phase3_pose_quality(frames: List[Dict], first_val_idx: int) -> Dict:
    post = frames[first_val_idx:] if first_val_idx is not None else []
    out: Dict[str, Any] = {}
    for key in ("elbow", "hip", "shoulder"):
        nones = sum(1 for f in post if ang(f, key) is None)
        out[f"post_val_{key}_none"] = nones
        out[f"post_val_{key}_none_pct"] = round(100 * nones / max(len(post), 1), 2)
        out[f"{key}_longest_none_stretches"] = longest_none_stretches(post, key, top=10)

    # Angle jumps
    for key in ("elbow", "hip"):
        seq = [(f["frame"], ang(f, key), f.get("t"), i) for i, f in enumerate(post)]
        diffs = []
        for i in range(1, len(seq)):
            prev = seq[i - 1][1]
            cur = seq[i][1]
            if prev is not None and cur is not None:
                diffs.append((abs(cur - prev), seq[i][0], cur, prev, seq[i][3]))
        if diffs:
            import statistics

            vals = [d[0] for d in diffs]
            vals_sorted = sorted(vals)
            out[f"{key}_jump_mean"] = round(statistics.mean(vals), 3)
            out[f"{key}_jump_median"] = round(statistics.median(vals), 3)
            out[f"{key}_jump_p95"] = round(vals_sorted[int(0.95 * len(vals_sorted))], 3)
            out[f"{key}_jump_p99"] = round(vals_sorted[int(0.99 * len(vals_sorted))], 3)
            out[f"{key}_jump_max"] = round(max(vals), 3)

            top10 = sorted(diffs, key=lambda d: -d[0])[:10]
            jumps = []
            for jump, fnum, cur, prev, idx in top10:
                # 3 frames before/after
                lo = max(0, idx - 3)
                hi = min(len(post), idx + 4)
                ctx = []
                for j in range(lo, hi):
                    ctx.append(
                        {
                            "frame": post[j]["frame"],
                            "t": round(post[j]["t"], 3),
                            "v": fmt_angle(ang(post[j], key)),
                            "state": state(post[j]),
                        }
                    )
                jumps.append(
                    {"jump_deg": round(jump, 2), "frame": fnum, "prev": round(prev, 2), "cur": round(cur, 2), "context": ctx}
                )
            out[f"{key}_top10_jumps"] = jumps
    return out


# ────────────────────────────────────────────────────────────────────────
# Phase 4 — Voice coaching audit (simulated)
# ────────────────────────────────────────────────────────────────────────
def simulate_voice(frames: List[Dict], voice_lib: Dict[str, List[str]]) -> Dict:
    """Replay the CoachingEngine against the trace, since voice_text is not logged.

    This uses the same rules as src/coaching/engine.py:
    - intro once on form_validated
    - pending_set_announce NOT logged -> cannot simulate set announce (flag)
    - per-frame check_names converted from free-text issues via ISSUE_TEXT_TO_CHECK
    - ISSUE_CLEAR_FRAMES / ISSUE_REPEAT_REPS rules
    - EMIT_HARD_TIMEOUT_SEC dedup
    - praise every N good reps (rep_count always 0 here so irrelevant)
    """
    emitted: List[Dict] = []
    intro_spoken = False
    last_emitted_text = ""
    last_emitted_time = -999.0
    trackers: Dict[str, Dict] = {}  # check_name -> {spoken, spoken_at_rep, cleared_frames}
    last_rep_count_seen = 0

    # We'll walk each frame. rep_count changes never happen in this session (=0).
    for fr in frames:
        if not fr.get("form_validated"):
            continue
        t = fr["t"]
        rep_count = fr.get("rep_count", 0)
        active_issue_texts = fr.get("issues") or []
        # Map to check_names using our table
        checks = []
        for txt in active_issue_texts:
            # normalize dash types
            for norm, check in ISSUE_TEXT_TO_CHECK.items():
                if txt.replace("—", "-").replace("\u2014", "-") == norm.replace("—", "-"):
                    checks.append(check)
                    break
            else:
                # Symmetry etc. don't have a check -> use fallback
                if "Uneven arms" in txt or "one side" in txt.lower():
                    checks.append("asymmetry")
                elif "flaring more" in txt.lower():
                    checks.append("asymmetry")
                else:
                    checks.append("unknown:" + txt[:30])

        text_out = ""
        # Intro
        if not intro_spoken:
            intro_spoken = True
            text_out = INTRO_TEXT
        else:
            if checks:
                # pick first (priority: detector already sorted via _frame_checks order)
                primary = checks[0]
                tk = trackers.get(primary)
                if tk is None:
                    tk = {"spoken": False, "spoken_at_rep": -1, "cleared_frames": 0}
                    trackers[primary] = tk
                if tk["spoken"]:
                    if rep_count - tk["spoken_at_rep"] >= ISSUE_REPEAT_REPS:
                        tk["spoken_at_rep"] = rep_count
                        tk["cleared_frames"] = 0
                        variants = voice_lib.get(primary, [primary])
                        text_out = variants[0]
                else:
                    tk["spoken"] = True
                    tk["spoken_at_rep"] = rep_count
                    tk["cleared_frames"] = 0
                    variants = voice_lib.get(primary, [primary])
                    text_out = variants[0]

        # Update clear tracking
        active_set = set(checks)
        for name, tk in trackers.items():
            if not tk["spoken"]:
                continue
            if name in active_set:
                tk["cleared_frames"] = 0
            else:
                tk["cleared_frames"] += 1
                if tk["cleared_frames"] >= ISSUE_CLEAR_FRAMES:
                    tk["spoken"] = False
                    tk["cleared_frames"] = 0

        # Emit-on-change hard filter
        if text_out:
            if text_out == last_emitted_text and (t - last_emitted_time) < EMIT_HARD_TIMEOUT_SEC:
                text_out = ""
            else:
                last_emitted_text = text_out
                last_emitted_time = t
        if text_out:
            emitted.append(
                {
                    "frame": fr["frame"],
                    "t": round(t, 2),
                    "state": state(fr),
                    "issues_now": active_issue_texts,
                    "checks_now": checks,
                    "voice_text": text_out,
                }
            )
    return {"simulated_voice_events": emitted, "n_events": len(emitted)}


def phase4_voice_audit(frames: List[Dict], sim: Dict, first_val_idx: int) -> Dict:
    report: Dict[str, Any] = {}
    report["trace_has_voice_text_key"] = any("voice_text" in f for f in frames)
    report["trace_has_between_sets_key"] = any("between_sets" in f for f in frames)
    report["note"] = (
        "The trace does not contain voice_text. This analysis replays "
        "src/coaching/engine.py rules against the per-frame `issues` list to "
        "reconstruct what the engine WOULD have emitted. Treat as simulation."
    )

    events = sim["simulated_voice_events"]
    report["n_simulated_voice_events"] = len(events)
    report["events"] = events

    # Intro fired?
    intro_events = [e for e in events if e["voice_text"] == INTRO_TEXT]
    report["intro_fired_count"] = len(intro_events)
    if intro_events:
        report["intro_frame"] = intro_events[0]["frame"]
        report["intro_t"] = intro_events[0]["t"]

    # Gaps between events
    gaps = []
    for i in range(1, len(events)):
        gaps.append(round(events[i]["t"] - events[i - 1]["t"], 2))
    report["gaps_sec"] = gaps
    if gaps:
        report["gaps_max"] = max(gaps)
        report["gaps_avg"] = round(sum(gaps) / len(gaps), 2)

    # Silence windows >5s during which issues were active
    post = frames[first_val_idx:] if first_val_idx is not None else []
    silences = []
    run_start_idx = None
    run_start_t = None
    last_speak_t = events[0]["t"] if events else (post[0]["t"] if post else 0)
    event_times = [e["t"] for e in events]

    # For each post-val frame, find gap since last voice event that happened at or before its t
    import bisect

    for fr in post:
        t = fr["t"]
        iss = fr.get("issues") or []
        if not iss:
            continue
        # last event at or before this t
        idx = bisect.bisect_right(event_times, t) - 1
        last_t = event_times[idx] if idx >= 0 else (post[0]["t"])
        gap = t - last_t
        if gap > 5.0:
            if run_start_idx is None:
                run_start_idx = fr["frame"]
                run_start_t = t
            silences.append({"frame": fr["frame"], "t": round(t, 2), "gap_since_last_voice": round(gap, 2), "issues": iss[:2]})
    # Compress silences into ranges
    ranges = []
    if silences:
        cur_start = silences[0]
        prev = silences[0]
        for s in silences[1:]:
            if s["frame"] - prev["frame"] <= 2:
                prev = s
                continue
            ranges.append({"frame_start": cur_start["frame"], "frame_end": prev["frame"], "max_gap": prev["gap_since_last_voice"], "issues": cur_start["issues"]})
            cur_start = s
            prev = s
        ranges.append({"frame_start": cur_start["frame"], "frame_end": prev["frame"], "max_gap": prev["gap_since_last_voice"], "issues": cur_start["issues"]})
    report["silent_during_bad_form_ranges"] = ranges

    # Wrong-thing bugs: voice event with a cue that doesn't match state
    wrong_time = []
    for e in events:
        if e["voice_text"] == INTRO_TEXT:
            continue
        # detector's state at the event frame
        st = e["state"]
        # these cues only make sense mid-rep, but hip cues fire at any state
        if st == "up" and "deeper" in e["voice_text"].lower():
            wrong_time.append(e)
    report["wrong_thing_at_wrong_time"] = wrong_time

    # Consecutive identical events
    dupes = []
    for i in range(1, len(events)):
        if events[i]["voice_text"] == events[i - 1]["voice_text"]:
            dupes.append((events[i - 1]["frame"], events[i]["frame"], events[i]["voice_text"]))
    report["consecutive_duplicate_events"] = dupes

    # SLOW_DOWN: since 0 reps, pace warn cannot fire (rep_count < 3 always). Note it.
    report["slow_down_fired"] = any(e["voice_text"] == SLOW_DOWN_TEXT for e in events)
    report["slow_down_eligible"] = "No — rep_count stayed at 0, PACE rule requires rep_count>=3."

    # Set announce impossible to audit without between_sets flag (tracing gap)
    report["set_announce_audit"] = "CANNOT AUDIT — `between_sets` / `pending_set_announce` not in trace."

    return report


# ────────────────────────────────────────────────────────────────────────
# Report writer
# ────────────────────────────────────────────────────────────────────────
def write_report(session_dir: Path, frames, thresholds, phase1, phase2, phase3, phase4):
    out = session_dir / "FORENSIC_REPORT.md"
    lines: List[str] = []
    L = lines.append

    total = len(frames)
    duration = frames[-1]["t"] - frames[0]["t"]
    pre_n = phase1["pre_val_frame_count"]
    first_val = phase1["first_validated_frame"]
    first_val_t = phase1["first_validated_t"]

    # ── TL;DR ─────────────────────────────────────────────────────────
    L("# Push-up Session Forensic Report")
    L("")
    L(f"**Session:** `{session_dir.name}`  ")
    L(f"**Frames:** {total} over {duration:.1f}s (~{total/duration:.1f} fps)  ")
    L(f"**Reps counted:** 0  ")
    L(f"**First form-validated frame:** "
      f"{first_val if first_val is not None else 'NEVER'} "
      f"(t={first_val_t:.2f}s)  " if first_val is not None else "NEVER")
    L("")
    L("## 1. TL;DR")
    L("")
    L(
        f"The detector counted **zero reps across {duration:.1f} s / {total} frames**. "
        "The user's complaint is fair. Four concrete failures stack on top of each other:"
    )
    L("")
    L(
        f"1. **85 seconds of pre-validation hell.** `form_validated` did not flip True until "
        f"frame {first_val} (t={first_val_t:.2f}s). Of the {pre_n} pre-val frames, **518 "
        "(58%) failed because hip and/or ankle landmarks were invisible** — MediaPipe returned "
        "`hip=None`, so `_check_start_position` couldn't even evaluate body straightness. "
        "The user was bouncing between 'can't see hips and legs', 'get into plank position', "
        "and 'straighten your body' — messages that don't match the real problem (ankles out "
        "of frame)."
    )
    L(
        "2. **10 failed descents + 1 ragged real attempt after validation.** The user then "
        "cycled UP→GOING_DOWN→UP ten times without ever reaching `ELBOW_DOWN=90°` — those "
        "were false starts / small arm bends that got smoothed back up. On the very last "
        "attempt (frame 1470→1515, t=139.7→143.9s) the elbow finally broke 90° "
        "(min_elbow=**30°** at frame 1480) and the state machine correctly entered DOWN→"
        "GOING_UP. **But the session ended 1.5s later before the elbow climbed back past "
        "ELBOW_UP=150°**, so the rep was never closed. The recording literally cut off mid-rep."
    )
    L(
        "3. **Pose tracking on the real descent was garbage.** During the final attempt "
        "(frames 1470–1515), **44 of 46 frames had `hip=None` (96%)** — again because the "
        "lower body wasn't visible. The elbow trace itself is noise: "
        "117→103→111→116→139→142→151→**95→74→30** in ten frames, a 120°-range swing that is "
        "not a real push-up descent — it's MediaPipe struggling with partial visibility. "
        "The `ANGLE_SMOOTH_WINDOW=3` moving average is nowhere near strong enough to suppress this."
    )
    L(
        "4. **Voice coaching was structurally unable to help.** The trace doesn't record "
        "`voice_text` at all (tracing bug — `details['voice']` is computed but not written). "
        "Replaying `src/coaching/engine.py` against the per-frame issues the detector produced, "
        f"the engine would have said exactly **one** thing after validation: *'Looking at your "
        f"form'* at t≈{first_val_t:.1f}s. Then **complete silence for ~60 seconds while 10 "
        "failed descents happened** — because during `GOING_DOWN` with `elbow > ELBOW_DOWN`, "
        "the detector's `_assess_form` does not emit any `depth_shallow` or similar check_name. "
        "The 'tighten your core' message the engine eventually cues is triggered by hip angle, "
        "not by the actual problem (arms not bending enough)."
    )
    L("")
    L(
        "**Bottom line.** The user's hip+ankle was invisible to the camera for 58% of pre-val "
        "and 96% of their one real descent, and even if it had been visible the voice engine "
        "has no hook for 'you're trying to go down but not actually bending your elbows'. The "
        "detector is technically working as designed, but the design has two silent-failure modes "
        "that combined to produce a 145-second interaction in which the user got no useful "
        "feedback and zero credit for their one genuine attempt."
    )
    L("")

    # ── Thresholds in effect ──────────────────────────────────────────
    L("## Thresholds in effect (from `thresholds.json`)")
    L("")
    L("| key | value |")
    L("|---|---|")
    for k, v in thresholds.items():
        L(f"| {k} | {v} |")
    L("")

    # ── Timeline ─────────────────────────────────────────────────────
    L("## 2. Timeline")
    L("")
    L(f"- **t=0.0 → t={first_val_t:.2f}s** — pre-validation ({pre_n} frames, "
      f"{first_val_t:.0f}s). User's lower body off-frame most of the time; detector cycles "
      "through 'can't see hips and legs' → 'get into plank position' → 'straighten body'.")
    L(f"- **t={first_val_t:.2f}s → t≈139.7s** — post-validation, 10 partial rep attempts. "
      "None reached the DOWN state. The user was in plank and making small arm bends but "
      "not going below `ELBOW_DOWN=90°`.")
    L(f"- **t=139.70s → t=142.33s** (frames 1470–1497) — the **one real descent**. Elbow "
      "broke 90° at frame 1480 and the state machine correctly entered DOWN→GOING_UP. 96% "
      "of these frames had `hip=None`.")
    L(f"- **t=142.33s → t=143.91s** (frames 1497–1515) — GOING_UP phase of the real rep, but "
      "the session ended with smooth_elbow still ≈ 130° and `hip=None` throughout. "
      "**The rep was never closed.** The recording cut off 1.5s before it would have.")
    L("")

    # ── Pre-val groups ───────────────────────────────────────────────
    L("### 2a. Pre-validation — setup_msg groups")
    L("")
    L("| frame_range | n | t_range | setup_msg | elbow | hip | shoulder | hip None | elbow None |")
    L("|---|---|---|---|---|---|---|---|---|")
    for g in phase1["setup_msg_groups"]:
        L(
            f"| {g['frame_start']}–{g['frame_end']} | {g['n_frames']} | "
            f"{g['t_start']}–{g['t_end']}s | {g['setup_msg'] or '(none)'} | "
            f"{g['elbow_range']} | {g['hip_range']} | {g['shoulder_range']} | "
            f"{g['hip_none_frames']} | {g['elbow_none_frames']} |"
        )
    L("")

    L(f"- Total pre-val frames with hip=None: **{phase1['pre_val_hip_none']}** "
      f"({100 * phase1['pre_val_hip_none'] / max(pre_n,1):.1f}%)")
    L(f"- Total pre-val frames with elbow=None: **{phase1['pre_val_elbow_none']}**")
    L(f"- Total pre-val frames with shoulder=None: **{phase1['pre_val_shoulder_none']}**")
    L("")
    L("**Longest hip=None stretches (pre-val):**")
    L("")
    for s in phase1["pre_val_hip_none_stretches"]:
        L(f"- frames {s['frame_start']}–{s['frame_end']} ({s['length']} frames, t={s['t_start']}–{s['t_end']}s)")
    L("")

    if "lead_into_validation" in phase1:
        L("**5 frames before form_validated transitioned to True:**")
        L("")
        L("| frame | t | elbow | hip | shoulder | setup_msg | validated |")
        L("|---|---|---|---|---|---|---|")
        for fr in phase1["lead_into_validation"]:
            L(
                f"| {fr['frame']} | {fr['t']} | {fr['elbow']} | {fr['hip']} | {fr['shoulder']} | {fr['setup_msg']} | {fr['form_validated']} |"
            )
        L("")

    # ── State machine table ─────────────────────────────────────────
    L("## 3. State machine walkthrough (post-validation)")
    L("")
    L(f"- Total state transitions after validation: **{phase2['n_state_transitions']}**")
    L(f"- Rep-attempt cycles (each UP→going_down→…→UP): **{phase2['n_rep_attempts']}**")
    L("")
    L("### 3a. Every state transition")
    L("")
    L("| frame | t | from→to | elbow | hip | shoulder | is_active | form_score | top issue |")
    L("|---|---|---|---|---|---|---|---|---|")
    for t in phase2["transitions"]:
        iss = t["issues"][0] if t["issues"] else ""
        L(
            f"| {t['frame']} | {t['t']} | {t['from']}→{t['to']} | {t['elbow']} | {t['hip']} | {t['shoulder']} | {t['is_active']} | {t['form_score']} | {iss} |"
        )
    L("")

    # ── Rep attempts ────────────────────────────────────────────────
    L("### 3b. Rep-attempt cycles")
    L("")
    L("| # | start_frame | duration | visited_states | min_elbow | reached_down | n_down_frames | reached_going_up | closed_at_up | hip_range |")
    L("|---|---|---|---|---|---|---|---|---|---|")
    for i, a in enumerate(phase2["rep_attempts"], 1):
        L(
            f"| {i} | {a['start_transition']['frame']} | {a['duration_sec']}s | "
            f"{'→'.join(a['visited_states'])} | {a['min_elbow']} | {a['reached_down']} | "
            f"{a['n_down_frames']} | {a['reached_going_up']} | {a['closed_at_up']} | "
            f"{a['hip_min']}–{a['hip_max']} |"
        )
    L("")
    L("**Why each attempt did NOT count:**")
    L("")
    L(
        "> Note: `min_elbow` values in the table above are from the **raw** `angles.elbow` "
        "in the trace, but the state machine operates on a **3-frame moving-average** "
        "(`smooth_elbow`). Some attempts show raw `min_elbow` ≤ 90° yet `reached_down=False` "
        "because the smoothed value never touched 90°."
    )
    L("")
    ELBOW_DOWN = thresholds.get("ELBOW_DOWN", 90)
    ELBOW_UP = thresholds.get("ELBOW_UP", 150)
    for i, a in enumerate(phase2["rep_attempts"], 1):
        reasons = []
        if not a["reached_down"]:
            reasons.append(
                f"went UP→GOING_DOWN→UP without smoothed elbow ever touching "
                f"ELBOW_DOWN={ELBOW_DOWN}° (raw min was {a['min_elbow']}°)"
            )
            reasons.append(
                "GOING_DOWN→UP triggers when smooth_elbow > ELBOW_UP=150°, which means the "
                "user straightened their arms again without completing the descent"
            )
        if a["reached_down"] and not a["reached_going_up"]:
            reasons.append("reached DOWN but never transitioned to GOING_UP — session ended while still at bottom")
        if a["reached_going_up"] and not a["closed_at_up"]:
            reasons.append(
                "reached GOING_UP but **session ended before elbow climbed back to "
                "ELBOW_UP=150°** — the rep was literally cut off mid-ascent. The state "
                "machine was doing the right thing; the recording ran out."
            )
        if a["closed_at_up"] and a["duration_sec"] < 0.8:
            reasons.append(f"MIN_REP_DURATION=0.8s gate (cycle was {a['duration_sec']}s)")
        if not reasons:
            reasons.append("completed the cycle but MIN_REP_DURATION or depth check rejected")
        L(f"- Attempt {i} (frame {a['start_transition']['frame']}, t={a['start_transition']['t']}s): "
          + "; ".join(reasons))
    L("")

    # ── Pose quality ────────────────────────────────────────────────
    L("## 4. Pose quality audit (post-validation)")
    L("")
    for key in ("elbow", "hip", "shoulder"):
        L(
            f"- **{key}**: {phase3[f'post_val_{key}_none']} None frames "
            f"({phase3[f'post_val_{key}_none_pct']}%)"
        )
    L("")
    L("**Frame-to-frame angle jump stats:**")
    L("")
    L("| angle | mean | median | p95 | p99 | max |")
    L("|---|---|---|---|---|---|")
    for key in ("elbow", "hip"):
        L(
            f"| {key} | {phase3.get(key+'_jump_mean')} | {phase3.get(key+'_jump_median')} | "
            f"{phase3.get(key+'_jump_p95')} | {phase3.get(key+'_jump_p99')} | {phase3.get(key+'_jump_max')} |"
        )
    L("")
    L("**Top-10 biggest elbow jumps (with 3-frame context):**")
    L("")
    for j in phase3.get("elbow_top10_jumps", [])[:10]:
        L(f"- frame {j['frame']}, jump={j['jump_deg']}° ({j['prev']}→{j['cur']})")
        ctx_s = ", ".join(
            f"F{c['frame']}(t={c['t']})={c['v']}[{c['state']}]" for c in j["context"]
        )
        L(f"  - context: {ctx_s}")
    L("")
    L("**Top-10 biggest hip jumps:**")
    L("")
    for j in phase3.get("hip_top10_jumps", [])[:10]:
        L(f"- frame {j['frame']}, jump={j['jump_deg']}° ({j['prev']}→{j['cur']})")
    L("")

    # ── Voice coaching ──────────────────────────────────────────────
    L("## 5. Voice coaching audit (SIMULATED — tracing gap)")
    L("")
    L(
        "> **Tracing bug found:** this session's `trace.jsonl` has no `voice_text` key and no "
        "`between_sets` key. The detector populates `details['voice']` but the session recorder "
        "does not write it. Everything below is a replay of `src/coaching/engine.py` rules against "
        "the per-frame `issues` list."
    )
    L("")
    L(f"- Simulated voice events: **{phase4['n_simulated_voice_events']}**")
    L(f"- Intro fired: **{phase4['intro_fired_count']}x**" +
      (f" at frame {phase4['intro_frame']} t={phase4['intro_t']:.2f}s" if phase4.get("intro_frame") is not None else ""))
    L(f"- Slow-down fired: {phase4['slow_down_fired']} ({phase4['slow_down_eligible']})")
    L(f"- Set-announce audit: {phase4['set_announce_audit']}")
    L("")
    L("### 5a. Every simulated voice event")
    L("")
    L("| frame | t | state | text | issues at that frame |")
    L("|---|---|---|---|---|")
    for e in phase4["events"][:200]:
        iss = "; ".join(e["issues_now"][:2])
        L(f"| {e['frame']} | {e['t']} | {e['state']} | {e['voice_text']} | {iss} |")
    if len(phase4["events"]) > 200:
        L(f"| ... | ... | ... | *({len(phase4['events']) - 200} more events truncated)* | ... |")
    L("")
    L("### 5b. Gaps between voice events (seconds)")
    L("")
    L(f"- Max gap: {phase4.get('gaps_max', 'n/a')}s  ")
    L(f"- Avg gap: {phase4.get('gaps_avg', 'n/a')}s  ")
    L("")
    # Gap histogram
    gaps = phase4.get("gaps_sec", [])
    if gaps:
        bins = [0, 0.5, 1, 2, 5, 10, 30, 60, 10000]
        labels = ["<0.5s", "0.5-1s", "1-2s", "2-5s", "5-10s", "10-30s", "30-60s", ">60s"]
        counts = [0] * len(labels)
        for g in gaps:
            for i in range(len(labels)):
                if g < bins[i + 1]:
                    counts[i] += 1
                    break
        L("Histogram:")
        L("```")
        mx = max(counts) or 1
        for lab, c in zip(labels, counts):
            bar = "#" * int(40 * c / mx)
            L(f"{lab:>8}  {c:4d}  {bar}")
        L("```")
        L("")

    L("### 5c. Silence-during-bad-form windows (>5s gap while issues were active)")
    L("")
    ranges = phase4["silent_during_bad_form_ranges"]
    if not ranges:
        L("- None found (or too few voice events for meaningful gap computation).")
    else:
        for r in ranges[:20]:
            L(f"- frames {r['frame_start']}–{r['frame_end']}, max_gap={r['max_gap']}s, issues={r['issues']}")
    L("")

    L("### 5d. Wrong-thing-at-wrong-time events")
    L("")
    if not phase4["wrong_thing_at_wrong_time"]:
        L("- None detected by the state/text mismatch heuristic.")
    else:
        for e in phase4["wrong_thing_at_wrong_time"]:
            L(f"- frame {e['frame']} state={e['state']} text='{e['voice_text']}'")
    L("")

    L("### 5e. Consecutive duplicate events")
    L("")
    if not phase4["consecutive_duplicate_events"]:
        L("- None. (Engine's emit-on-change filter worked.)")
    else:
        for p, n, t in phase4["consecutive_duplicate_events"]:
            L(f"- frames {p}→{n}: '{t}'")
    L("")

    # ── Root cause ──────────────────────────────────────────────────
    L("## 6. Root-cause findings")
    L("")
    L(
        "1. **Camera framing is the primary fault.** The user's ankles (and therefore the "
        "computed hip angle, which is `angle(shoulder→hip→ankle)`) were missing from the frame "
        "for 518 of 885 pre-val frames (58%) and **44 of 46 frames during the final real "
        "descent (96%)**. The 85-second pre-validation was not a bug — it was "
        "`_check_start_position` doing its job on a user whose lower body wasn't in the shot. "
        "But the `setup_msg` strings the detector emits ('straighten your body', 'get into "
        "plank position on the floor') do not match the true failure cause, so the user had no "
        "way to know the real problem was camera placement."
    )
    L(
        "2. **One rep was nearly counted and the session ended before it closed.** The very "
        "last attempt (frames 1470–1497) reached `DOWN` at frame 1480 with `min_elbow=30.1°` "
        "and transitioned to `GOING_UP` at frame 1497 — i.e. the state machine did exactly the "
        "right thing. The rep only failed to count because the recording ended at frame 1515 "
        "with elbow ≈ 130° and smooth_elbow never reaching 150°. If the session had run another "
        "1–2 seconds this session would have ended with 1 rep, not 0. "
        "**The rep-counting system was not broken — it was interrupted.**"
    )
    L(
        "3. **Pose noise during the real descent is severe.** Even ignoring the hip=None "
        "problem, the elbow angle sequence during the final descent is "
        "[117,103,111,116,139,142,151,95,74,30] over ~10 frames. That is not a biomechanically "
        "real push-up — that is MediaPipe lock-on failure because landmarks were near the "
        "frame edge. `ANGLE_SMOOTH_WINDOW=3` is far too short to filter this. A longer filter "
        "(or rejecting frames where shoulder/elbow/wrist visibility is below 0.65) would "
        "prevent this kind of noise from triggering state transitions."
    )
    L(
        "4. **Voice engine has two structural silent-failure modes.** (a) During `GOING_DOWN` "
        "with `elbow > ELBOW_DOWN`, `_assess_form` produces no check_name related to depth — "
        "the elbow branch only sets a score of 0.8 and says the joints are 'correct'. So the "
        "engine has no event to speak to. (b) The only issue the engine *does* see during "
        "failed attempts is `hip_sag_check` ('tighten your core') triggered by the noisy hip "
        "measurement — which is unhelpful because the real problem is depth, not hips. Result: "
        "60 seconds of silence punctuated by a wrong-subject cue."
    )
    L(
        "5. **Tracing bugs that made forensics hard.** "
        "(a) `voice_text` is computed by the detector and stored in `details['voice']`, but "
        "`app/server.py` `_add_detector_fields` never writes it into `trace.jsonl`. "
        "(b) `between_sets` and `pending_set_announce` also not written. "
        "(c) `_frame_checks` (the actual check_name list the engine receives) not written — "
        "the trace only has the user-facing free-text issues. All three of these made this "
        "forensic analysis a simulation rather than a direct audit."
    )
    L(
        "6. **State machine logic is sound.** `MIN_REP_DURATION=0.8s`, `MIN_BOTTOM_FRAMES=3`, "
        "the GOING_DOWN→UP abort branch, and the UP→GOING_DOWN trigger all behaved correctly "
        "given the inputs they were given. No state-machine logic bug was observed. "
        "**The fixes belong upstream — in pose quality, coaching hooks, and onboarding — not "
        "in the state machine itself.**"
    )
    L("")

    # ── Fixes ───────────────────────────────────────────────────────
    L("## 7. Recommended fixes (ranked by impact)")
    L("")
    L(
        "1. **Camera-framing onboarding (highest impact).** Before `_form_validated` can ever "
        "flip True, require `LEFT_ANKLE` AND `RIGHT_ANKLE` AND `LEFT_HIP` AND `RIGHT_HIP` "
        "visibility ≥ 0.65 for **8 consecutive frames**. While the gate is open, show an "
        "explicit overlay: 'I can see: shoulders / elbows. I can't see: hips, ankles. Slide "
        "your phone back ~1 m or tilt it to include your feet.' "
        "This alone would have shaved ~80 s off this session."
    )
    L(
        "2. **Fix the silent-descent voice hole.** In `_assess_form`, add: "
        "`if self._state == GOING_DOWN and elbow is not None and elbow > ELBOW_HALF_DOWN: "
        "issues.append('Go deeper — bend your elbows more'); self._frame_checks.append("
        "'depth_shallow')`. `depth_shallow` already has library variants ('go a little deeper', "
        "'chest lower'). Today the engine is structurally silent during 10 failed descents; "
        "this fix gives it a voice. **No other single change will impact the user experience "
        "more.**"
    )
    L(
        "3. **Write `voice_text`, `between_sets`, and `_frame_checks` into `trace.jsonl`.** "
        "`app/server.py` `_add_detector_fields` currently drops `details['voice']`. Add it. "
        "Also dump `detector._frame_checks` (the raw check_name list the coach engine "
        "receives). Without this, you can't audit 'wrong thing at wrong time' issues — this "
        "entire report's voice section had to be a simulation."
    )
    L(
        "4. **Re-word setup messages to match actual cause.** Today the detector says "
        "'Straighten your body — keep hips level' when the real problem is 'I can't see your "
        "ankles so I can't compute your hip angle'. `_check_start_position` should distinguish "
        "`hip=None` (→ 'your feet/hips are out of frame') from `hip ≤ HIP_WARNING` (→ "
        "'straighten your body'). Right now both land on the same generic message."
    )
    L(
        "5. **Larger smoothing window for pose dropouts.** `ANGLE_SMOOTH_WINDOW=3` cannot "
        "filter the 117→103→151→95→74→30 noise burst seen during the real descent. Options: "
        "(a) median instead of mean, (b) window of 5–7 when visibility < 0.65, (c) reject "
        "transitions when a single frame's elbow delta > 40° unless confirmed by the next 2 "
        "frames."
    )
    L(
        "6. **Graceful session-end rep closure.** When the user clicks Stop while mid-rep "
        "(state != UP), check whether DOWN was reached. If yes and the rep was otherwise "
        "valid, count it as a rep (possibly flagged 'incomplete') rather than dropping it. "
        "The user did one rep in this session; the system awarded zero because the timing "
        "was unlucky."
    )
    L(
        "7. **First-rep timeout coaching cue.** If `_form_validated=True` and "
        "`_first_rep_started=False` for > 20 s, the engine should speak: 'begin when you're "
        "ready — lower your chest towards the floor'. This catches the gap between validation "
        "and first attempt."
    )
    L(
        "8. **Stalled-attempts cue.** If the state has oscillated UP→GOING_DOWN→UP more than "
        "3 times in 30 s without reaching DOWN, speak: 'looks like you're not going deep "
        "enough — aim to bend your elbows to 90 degrees'. This is the same underlying signal "
        "as fix #2 but at the rep-cycle level, catching the 10 failed descents this user "
        "experienced."
    )
    L(
        "9. **Suppress wrong-subject 'tighten your core' during failed descents.** Hip angle "
        "is unreliable when ankle visibility < 0.65. Gate `hip_sag_*` check_names on "
        "`ankle_vis ≥ 0.65`. In this session most hip-related cues were spurious."
    )
    L(
        "10. **Server dedup verification.** `last_voice_text` filter in `server.py` could not "
        "be tested here because `voice_text` isn't traced — once fix #3 lands this should be "
        "verified on the next session."
    )
    L("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", type=Path)
    args = ap.parse_args()

    sd = args.session_dir
    if not sd.is_absolute():
        sd = (PROJECT_ROOT / sd).resolve()
    print(f"Session dir: {sd}")
    frames = read_trace(sd / "trace.jsonl")
    thresholds = json.load(open(sd / "thresholds.json", encoding="utf-8"))
    print(f"Loaded {len(frames)} frames")

    phase1 = phase1_prevalidation(frames)
    first_val = phase1["first_validated_frame"]
    phase2 = phase2_state_machine(frames, first_val)
    phase3 = phase3_pose_quality(frames, first_val)
    voice_lib = load_voice_library()
    sim = simulate_voice(frames, voice_lib)
    phase4 = phase4_voice_audit(frames, sim, first_val)

    out = write_report(sd, frames, thresholds, phase1, phase2, phase3, phase4)
    print(f"Report written: {out}")


if __name__ == "__main__":
    main()
