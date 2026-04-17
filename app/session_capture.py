"""
Session capture — saves per-frame detector trace + metadata to disk for
offline analysis.

Every exercise session creates a new folder under data/sessions/ containing:
    trace.jsonl      one JSON object per processed frame (appended live)
    summary.json     final session summary (rep breakdown, issues, scores)
    thresholds.json  snapshot of the detector's module-level constants at
                     the time of capture (so the session stays interpretable
                     even if thresholds are retuned later)
    metadata.json    session id, exercise, classifier, start/end timestamps

The trace is the primary data source. It lets you (or an LLM analyst) see
exactly what the detector saw and decided at every frame — which reps counted,
which issues fired at which frame, which state transitions happened, and how
smoothly the form score evolved.
"""

from __future__ import annotations

import ast
import inspect
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    """JSON serializer fallback for numpy scalars / arrays / other."""
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    return str(obj)


class SessionCapture:
    """Writes per-frame detector trace + session metadata to disk."""

    def __init__(
        self,
        project_root: Path,
        exercise: str,
        classifier: str,
        detector: Any,
    ):
        self.project_root = Path(project_root)
        self.exercise = exercise
        self.classifier = classifier
        self.detector = detector
        self.session_id = (
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            + f"_{exercise}_{uuid.uuid4().hex[:6]}"
        )
        self.session_dir = self.project_root / "data" / "sessions" / self.session_id
        self._trace_file = None
        self.start_time: Optional[float] = None
        self.frame_count = 0

    def start(self) -> None:
        """Create the session folder and open trace.jsonl for append."""
        try:
            self.session_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning("Could not create session dir %s: %s", self.session_dir, e)
            return

        try:
            self._trace_file = open(
                self.session_dir / "trace.jsonl", "w", encoding="utf-8"
            )
        except Exception as e:
            logger.warning("Could not open trace file: %s", e)
            self._trace_file = None

        self.start_time = time.time()

        # Snapshot detector threshold constants at capture time. This is what
        # makes a session reproducible — if we retune HIP_BOTTOM tomorrow, the
        # session saved today still records what it was scored under.
        try:
            thresholds = self._snapshot_thresholds()
            (self.session_dir / "thresholds.json").write_text(
                json.dumps(thresholds, indent=2, default=_json_default),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("Could not write thresholds snapshot: %s", e)

        # Write stub metadata (finalized in end())
        self._write_metadata({
            "session_id": self.session_id,
            "exercise": self.exercise,
            "classifier": self.classifier,
            "detector_class": type(self.detector).__name__ if self.detector else None,
            "start_time": self.start_time,
            "start_iso": datetime.now().isoformat(timespec="seconds"),
        })

        logger.info("Session capture started: %s", self.session_id)

    def write_frame(self, trace: Dict[str, Any]) -> None:
        """Append a single frame's trace as one JSON line."""
        if self._trace_file is None or self.start_time is None:
            return
        trace = dict(trace)  # shallow copy so we don't mutate caller's dict
        trace.setdefault("frame", self.frame_count)
        trace.setdefault("t", round(time.time() - self.start_time, 3))
        try:
            self._trace_file.write(json.dumps(trace, default=_json_default) + "\n")
        except Exception as e:
            logger.debug("Trace write failed: %s", e)
            return
        self.frame_count += 1

    def end(self, summary: Dict[str, Any]) -> None:
        """Close trace file and save final summary + end metadata."""
        if self._trace_file is not None:
            try:
                self._trace_file.flush()
                self._trace_file.close()
            except Exception:
                pass
            self._trace_file = None

        end_time = time.time()

        try:
            (self.session_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, default=_json_default),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("Could not write summary: %s", e)

        # Update metadata with end info (read existing, merge, write back)
        metadata_path = self.session_dir / "metadata.json"
        try:
            existing: Dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        existing.update({
            "end_time": end_time,
            "end_iso": datetime.now().isoformat(timespec="seconds"),
            "duration_sec": round(end_time - (self.start_time or end_time), 2),
            "frame_count": self.frame_count,
        })
        self._write_metadata(existing)

        logger.info(
            "Session capture ended: %s (%d frames, %.1fs)",
            self.session_id,
            self.frame_count,
            existing.get("duration_sec", 0),
        )

    def _write_metadata(self, data: Dict[str, Any]) -> None:
        try:
            (self.session_dir / "metadata.json").write_text(
                json.dumps(data, indent=2, default=_json_default), encoding="utf-8"
            )
        except Exception as e:
            logger.debug("metadata write failed: %s", e)

    def _snapshot_thresholds(self) -> Dict[str, Any]:
        """Pull tunable threshold constants from the detector's module.

        Uses AST to find only the names *defined* at module level in the
        detector's own source file — filters out imported landmark indices
        like LEFT_SHOULDER=11 which come in via `from ..utils.constants import`.
        Captures things like HIP_BOTTOM, KNEE_PARALLEL, SHOULDER_IDEAL_MAX.
        """
        if self.detector is None:
            return {}
        mod = inspect.getmodule(self.detector)
        if mod is None:
            return {}

        defined_names = _find_module_defined_names(mod)
        if not defined_names:
            return {}

        snapshot: Dict[str, Any] = {}
        for name, value in vars(mod).items():
            if name not in defined_names:
                continue
            if name.startswith("_") or not name.isupper():
                continue
            if isinstance(value, (int, float, str, bool)):
                snapshot[name] = value
            elif isinstance(value, (list, tuple)) and all(
                isinstance(v, (int, float, str, bool)) for v in value
            ):
                snapshot[name] = list(value)
        return snapshot


def _find_module_defined_names(module: Any) -> Set[str]:
    """Parse the module's source file and return the set of names that
    appear on the left-hand side of a top-level assignment. Excludes imports
    so we only see constants defined in the file itself."""
    try:
        source = inspect.getsource(module)
    except (OSError, TypeError):
        return set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def build_frame_trace(
    pipeline: Any,
    result: Any,
    detector: Any,
) -> Dict[str, Any]:
    """Build a trace dict for one processed frame.

    Pulls from the pipeline (rep count, exercise, fps), the classification
    result (form score, issues, joint feedback), and the detector (angles,
    state, form validation status). Keeps the dict compact — floats rounded,
    None values omitted where practical.
    """
    trace: Dict[str, Any] = {
        "exercise": pipeline.exercise,
        "rep_count": pipeline.rep_count,
        "fps": round(pipeline.fps, 1),
    }

    if result is not None:
        trace["form_score"] = round(getattr(result, "form_score", 0.0) or 0.0, 3)
        trace["is_active"] = bool(getattr(result, "is_active", False))
        trace["is_correct"] = bool(getattr(result, "is_correct", False))
        conf = getattr(result, "confidence", None)
        trace["confidence"] = round(conf, 3) if isinstance(conf, (int, float)) else 0.0

        jf = getattr(result, "joint_feedback", None)
        if jf:
            trace["joint_feedback"] = dict(jf)

        details = getattr(result, "details", None)
        if isinstance(details, dict):
            phase = details.get("phase")
            if phase:
                trace["phase"] = phase
            progress = details.get("progress")
            if progress:
                trace["progress"] = progress
            # Issues are stored as issue_0/issue_1/issue_2 in details
            issues = [v for k, v in details.items() if k.startswith("issue_")]
            if issues:
                trace["issues"] = issues
            setup_msg = details.get("setup")
            if setup_msg:
                trace["setup_msg"] = setup_msg
            voice = details.get("voice")
            if voice:
                trace["voice_text"] = voice

    if detector is not None:
        # Base-class detectors expose a dict of last computed angles
        last_angles = getattr(detector, "last_angles", None)
        if isinstance(last_angles, dict) and last_angles:
            trace["angles"] = {
                k: round(v, 2) if isinstance(v, (int, float)) else v
                for k, v in last_angles.items()
                if v is not None and k != "primary"  # 'primary' duplicates another entry
            }
        else:
            # PushUpDetector (standalone) uses individual fields
            angles_compat: Dict[str, float] = {}
            for attr, name in [
                ("last_elbow_angle", "elbow"),
                ("last_hip_angle", "hip"),
                ("last_shoulder_angle", "shoulder"),
            ]:
                val = getattr(detector, attr, None)
                if isinstance(val, (int, float)):
                    angles_compat[name] = round(val, 2)
            if angles_compat:
                trace["angles"] = angles_compat

        state = getattr(detector, "_state", None)
        if state is not None:
            trace["state"] = str(state)
        form_validated = getattr(detector, "_form_validated", None)
        if form_validated is not None:
            trace["form_validated"] = bool(form_validated)
        hold_duration = getattr(detector, "_hold_duration", None)
        if isinstance(hold_duration, (int, float)) and hold_duration > 0:
            trace["hold_duration"] = round(hold_duration, 2)

        # Push-up session FSM + posture classifier state for debugging
        session_state = getattr(detector, "_session_state", None)
        if session_state is not None:
            trace["session_state"] = getattr(session_state, "value", str(session_state))
        current_posture = getattr(detector, "_current_posture", None)
        if current_posture is not None:
            trace["posture"] = getattr(current_posture, "value", str(current_posture))
        posture_clf = getattr(detector, "_posture_classifier", None)
        if posture_clf is not None:
            feats = getattr(posture_clf, "_last_features", None)
            if isinstance(feats, dict) and feats:
                # Round floats, keep Nones as null
                trace["posture_features"] = {
                    k: round(v, 3) if isinstance(v, (int, float)) else v
                    for k, v in feats.items()
                }
            raw_conf = getattr(posture_clf, "_last_raw_conf", None)
            if isinstance(raw_conf, (int, float)):
                trace["posture_raw_conf"] = round(raw_conf, 3)
        between_sets = getattr(detector, "_between_sets", None)
        if between_sets is not None:
            trace["between_sets"] = bool(between_sets)
        frame_checks = getattr(detector, "_frame_checks", None)
        if frame_checks:
            trace["frame_checks"] = list(frame_checks)

    return trace
