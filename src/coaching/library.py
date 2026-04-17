"""
Coaching command library — loads commands.json and provides lookup for detectors.

Usage:
    from src.coaching import CoachingLibrary

    lib = CoachingLibrary()

    # Get a specific correction
    cmd = lib.correction("squat", "depth_shallow")
    print(cmd.text)      # "Go deeper — thighs should reach parallel"
    print(cmd.severity)  # "warning"
    print(cmd.joints)    # ["left_knee", "right_knee"]

    # Get a phase cue
    text = lib.phase_cue("squat", "bottom")  # "Good depth! Drive back up"

    # Get breathing cue
    text = lib.breathing("squat", "descend")  # "Big breath in — brace before you descend"

    # Get random encouragement
    text = lib.encourage("good_rep")  # varies: "Nailed it", "Solid rep", etc.

    # Get milestone text
    text = lib.milestone_reps(10)     # "10 reps — strong set"
    text = lib.milestone_duration(30) # "30 seconds — halfway to a good hold"

    # Format a template
    text = lib.format("plank", "fatigue", "drift_warning", drift=8)
    # "Form starting to break (8° drift) — stay tight"
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Command:
    """A single coaching command with metadata."""
    id: str
    text: str
    severity: str = "info"          # info, encourage, warning, error, safety
    joints: List[str] = field(default_factory=list)
    phase: List[str] = field(default_factory=list)
    category: str = ""              # depth, alignment, safety, stability, etc.

    def format(self, **kwargs) -> str:
        """Format text with dynamic values (e.g., drift degrees, duration)."""
        try:
            return self.text.format(**kwargs)
        except (KeyError, IndexError):
            return self.text


# Singleton — loaded once, shared across all detectors
_INSTANCE: Optional["CoachingLibrary"] = None


class CoachingLibrary:
    """Central coaching command store loaded from commands.json."""

    def __init__(self):
        self._data: Dict = {}
        self._load()

    @classmethod
    def instance(cls) -> "CoachingLibrary":
        """Get or create the singleton instance."""
        global _INSTANCE
        if _INSTANCE is None:
            _INSTANCE = cls()
        return _INSTANCE

    def _load(self):
        json_path = Path(__file__).parent / "commands.json"
        with open(json_path, "r", encoding="utf-8") as f:
            self._data = json.load(f)

    def reload(self):
        """Hot-reload commands from disk (for development)."""
        self._load()

    # ── Correction lookup ──────────────────────────────────────────────

    def correction(self, exercise: str, check_name: str) -> Optional[Command]:
        """Get a correction command for a specific exercise and check.
        For v2 format (variants list), returns the first variant as text.
        """
        ex = self._data.get("exercises", {}).get(exercise, {})
        corrections = ex.get("corrections", {})
        entry = corrections.get(check_name)
        if not entry:
            return None
        # v2: "variants" list; v1 fallback: "text" string
        variants = entry.get("variants", [])
        if variants:
            text = variants[0] if isinstance(variants, list) else str(variants)
        else:
            text = entry.get("text", "")
        return Command(
            id=entry.get("id", f"{exercise}_{check_name}"),
            text=text,
            severity=entry.get("severity", "warning"),
            joints=entry.get("joints", []),
            phase=entry.get("phase", []),
            category=entry.get("category", ""),
        )

    def correction_text(self, exercise: str, check_name: str) -> str:
        """Shortcut — just the text, or empty string if not found."""
        cmd = self.correction(exercise, check_name)
        return cmd.text if cmd else ""

    # ── Phase cues ─────────────────────────────────────────────────────

    def phase_cue(self, exercise: str, phase: str) -> str:
        """Get the positive phase cue for a given movement phase.
        v2: returns first variant from 'variants' list, v1 fallback: 'text' string.
        """
        ex = self._data.get("exercises", {}).get(exercise, {})
        cues = ex.get("phase_cues", {})
        entry = cues.get(phase, {})
        if isinstance(entry, dict):
            variants = entry.get("variants", [])
            if variants:
                return variants[0] if isinstance(variants, list) else str(variants)
            return entry.get("text", "Good form")
        return str(entry)

    # ── Breathing ──────────────────────────────────────────────────────

    def breathing(self, exercise: str, phase: str) -> str:
        """Get breathing cue for an exercise phase (descend/ascend/steady)."""
        ex = self._data.get("exercises", {}).get(exercise, {})
        breathing = ex.get("breathing", {})
        if phase in breathing:
            return breathing[phase]
        # Fall back to global
        return self._data.get("global", {}).get("breathing", {}).get("general", "")

    # ── Tempo ──────────────────────────────────────────────────────────

    def tempo(self, exercise: str, issue: str = "too_fast") -> str:
        """Get tempo feedback for an exercise."""
        ex = self._data.get("exercises", {}).get(exercise, {})
        tempo = ex.get("tempo", {})
        if issue in tempo:
            return tempo[issue]
        return self._data.get("global", {}).get("tempo", {}).get(issue, "")

    # ── Encouragement ──────────────────────────────────────────────────

    def encourage(self, quality: str = "good_rep") -> str:
        """Get a random encouragement phrase for a rep quality level."""
        enc = self._data.get("global", {}).get("encourage", {})
        options = enc.get(quality, [])
        if not options:
            return "Good"
        return random.choice(options)

    def encourage_exercise(self, exercise: str) -> str:
        """Get exercise-specific encouragement (e.g., plank has its own set)."""
        ex = self._data.get("exercises", {}).get(exercise, {})
        options = ex.get("encourage", [])
        if options:
            return random.choice(options)
        return self.encourage("good_rep")

    # ── Milestones ─────────────────────────────────────────────────────

    def milestone_reps(self, reps: int) -> Optional[str]:
        """Get milestone text for a rep count, or None if not a milestone."""
        milestones = self._data.get("global", {}).get("milestone", {}).get("reps", {})
        return milestones.get(str(reps))

    def milestone_duration(self, seconds: int) -> Optional[str]:
        """Get milestone text for a hold duration, or None."""
        # Check exercise-specific milestones first (for plank)
        # This is called with the exercise context by the caller
        milestones = self._data.get("global", {}).get("milestone", {}).get("duration", {})
        return milestones.get(str(seconds))

    def milestone_duration_exercise(self, exercise: str, seconds: int) -> Optional[str]:
        """Get exercise-specific milestone, falling back to global."""
        ex = self._data.get("exercises", {}).get(exercise, {})
        ex_milestones = ex.get("milestone", {})
        text = ex_milestones.get(str(seconds))
        if text:
            return text
        return self.milestone_duration(seconds)

    # ── Fatigue / safety ───────────────────────────────────────────────

    def fatigue(self, exercise: str, level: str = "mild") -> str:
        """Get fatigue warning. Checks exercise-specific first, then global."""
        ex = self._data.get("exercises", {}).get(exercise, {})
        ex_fatigue = ex.get("fatigue", {})
        if level in ex_fatigue:
            entry = ex_fatigue[level]
            return entry["text"] if isinstance(entry, dict) else str(entry)
        return self._data.get("global", {}).get("fatigue", {}).get(level, "")

    def safety(self, issue: str = "pain") -> str:
        """Get a global safety message."""
        return self._data.get("global", {}).get("safety", {}).get(issue, "")

    # ── Setup ──────────────────────────────────────────────────────────

    def setup(self, exercise: str, index: int = 0) -> str:
        """Get setup instruction for an exercise."""
        ex = self._data.get("exercises", {}).get(exercise, {})
        setup_list = ex.get("setup", [])
        if 0 <= index < len(setup_list):
            return setup_list[index]["text"]
        return f"Get into {exercise.replace('_', ' ')} position"

    def setup_all(self, exercise: str) -> List[str]:
        """Get all setup instructions for an exercise."""
        ex = self._data.get("exercises", {}).get(exercise, {})
        return [s["text"] for s in ex.get("setup", [])]

    # ── Rep results ────────────────────────────────────────────────────

    def rep_result(self, quality: str) -> str:
        """Get rep result text for a quality level (good/moderate/bad)."""
        results = self._data.get("global", {}).get("rep_result", {})
        return results.get(quality, "")

    # ── Generic format (for templates with {drift}, {duration}, etc.) ─

    def format(self, exercise: str, section: str, key: str, **kwargs) -> str:
        """Format a command template with dynamic values."""
        ex = self._data.get("exercises", {}).get(exercise, {})
        section_data = ex.get(section, {})
        entry = section_data.get(key)
        if entry is None:
            return ""
        text = entry["text"] if isinstance(entry, dict) else str(entry)
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError):
            return text

    # ── Bulk access ────────────────────────────────────────────────────

    def all_corrections(self, exercise: str) -> Dict[str, Command]:
        """Get all corrections for an exercise as {check_name: Command}."""
        ex = self._data.get("exercises", {}).get(exercise, {})
        corrections = ex.get("corrections", {})
        result = {}
        for name, entry in corrections.items():
            variants = entry.get("variants", [])
            text = variants[0] if isinstance(variants, list) and variants else entry.get("text", "")
            result[name] = Command(
                id=entry.get("id", f"{exercise}_{name}"),
                text=text,
                severity=entry.get("severity", "warning"),
                joints=entry.get("joints", []),
                phase=entry.get("phase", []),
                category=entry.get("category", ""),
            )
        return result

    def exercises(self) -> List[str]:
        """List all exercises in the library."""
        return list(self._data.get("exercises", {}).keys())

    def stats(self) -> Dict[str, int]:
        """Return command counts per exercise and total."""
        counts = {}
        total = 0
        for ex_name, ex_data in self._data.get("exercises", {}).items():
            n = len(ex_data.get("corrections", {}))
            n += len(ex_data.get("setup", []))
            n += len(ex_data.get("phase_cues", {}))
            n += len(ex_data.get("breathing", {}))
            n += len(ex_data.get("tempo", {}))
            n += len(ex_data.get("fatigue", {}))
            n += len(ex_data.get("encourage", []))
            n += len(ex_data.get("milestone", {}))
            n += len(ex_data.get("visibility", []))
            counts[ex_name] = n
            total += n

        # Global commands
        g = self._data.get("global", {})
        global_n = 0
        for key in ("encourage", "breathing", "tempo", "fatigue", "safety", "rep_result", "asymmetry", "visibility"):
            val = g.get(key, {})
            if isinstance(val, dict):
                for v in val.values():
                    if isinstance(v, list):
                        global_n += len(v)
                    else:
                        global_n += 1
        counts["_global"] = global_n
        total += global_n
        counts["_total"] = total
        return counts
