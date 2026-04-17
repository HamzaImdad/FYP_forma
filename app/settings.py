"""
FORMA redesign feature flags.

Read once at import time from env. Flip via .env / OS env and restart
the Flask process. Kept as a separate module so:
  1. Tests can monkeypatch settings without touching app.server.
  2. Downstream modules (goal_engine, server) import one symbol each.

Redesign phases:

  PHASE2_ENABLED — controls:
    • _calc_volume summing good_reps (True) vs total_reps (False, legacy)
    • _calc_duration form-score gate
    • The new strength goal_type calculator
    • Plan-day auto-complete using the per-day family-aware aggregator
    When False, the old behavior is preserved — safe A/B ramp.

  PHASE3_ENABLED — controls the creation-path rewrites (family-aware
  sanitizer, narrowed auto-goal creation, plan drafts per-user).
"""

from __future__ import annotations

import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


PHASE2_ENABLED: bool = _env_bool("FORMA_PHASE2_ENABLED", True)
PHASE3_ENABLED: bool = _env_bool("FORMA_PHASE3_ENABLED", True)
