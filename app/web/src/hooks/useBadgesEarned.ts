// Session 4: listen for `badges_earned` — the server fires this when
// on_session_complete finds newly earned badges from badge_engine.

import { useEffect } from "react";
import { acquireSharedSocket, releaseSharedSocket } from "./sharedSocket";
import type { Badge } from "@/lib/plansApi";

export type BadgesEarnedPayload = { badges: Badge[] };

export function useBadgesEarned(
  onEarned: (payload: BadgesEarnedPayload) => void,
) {
  useEffect(() => {
    let cancelled = false;
    try {
      const socket = acquireSharedSocket();
      const handler = (p: BadgesEarnedPayload) => {
        if (!cancelled) onEarned(p);
      };
      socket.on("badges_earned", handler);
      return () => {
        cancelled = true;
        socket.off("badges_earned", handler);
        releaseSharedSocket();
      };
    } catch {
      return () => {};
    }
  }, [onEarned]);
}
