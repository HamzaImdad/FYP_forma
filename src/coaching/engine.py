"""
Coaching Engine — minimal voice scheduler for the push-up coach.

Two prompts, driven purely by session-state edges:

    * → SETUP   → "Get into push-up position" (variant)
    * → ACTIVE  → "Form locked" (variant)
    * → RESTING → silent (user just stopped — don't shout)
    * → IDLE    → silent (pose lost)

Variant rotation ensures the same phrase never repeats twice in a row.
No form-specific cues, no escalation, no fix-ack, no streak rewards.
Per-rep form evaluation for push-ups isn't reliable enough yet, so the
engine says nothing about hip/elbow/head. Silence is the approval
signal; "form locked" is the only positive confirmation.
"""

import random
from typing import Dict


PROMPT_GET_INTO_POSITION = (
    "Get into push-up position",
    "Find your push-up position",
    "Let's start — get into position",
    "Position up when you're ready",
    "Into position whenever you're ready",
)

PROMPT_FORM_LOCKED = (
    "Form locked",
    "Locked in",
    "Good — let's go",
    "Set — begin",
    "Ready",
    "There it is",
)


class CoachingEngine:
    """Stateless variant-rotating prompt emitter. One per detector."""

    def __init__(self, exercise: str = "pushup"):
        self.exercise = exercise
        self._last_session_state: str = "idle"
        self._last_prompt: Dict[str, str] = {}

    def reset(self) -> None:
        self._last_session_state = "idle"
        self._last_prompt = {}

    def frame_feedback(self, session_state: str = "idle", **_legacy) -> str:
        """Emit a prompt on SETUP / ACTIVE entry; silent elsewhere.

        Accepts and ignores every legacy keyword (`score`, `issues`,
        `phase`, `rep_count`, `is_rep_complete`, `form_validated`) so
        existing callers keep working while the voice layer is simple.
        """
        prev = self._last_session_state
        self._last_session_state = session_state
        if prev == session_state:
            return ""

        if session_state == "setup":
            return self._pick("get_into_position", PROMPT_GET_INTO_POSITION)
        if session_state == "active":
            return self._pick("form_locked", PROMPT_FORM_LOCKED)
        return ""

    def _pick(self, pool_name: str, pool: tuple) -> str:
        if not pool:
            return ""
        last = self._last_prompt.get(pool_name)
        choices = [p for p in pool if p != last] or list(pool)
        pick = random.choice(choices)
        self._last_prompt[pool_name] = pick
        return pick

    def stats(self) -> Dict:
        return {
            "exercise": self.exercise,
            "session_state": self._last_session_state,
            "last_prompt": dict(self._last_prompt),
        }
