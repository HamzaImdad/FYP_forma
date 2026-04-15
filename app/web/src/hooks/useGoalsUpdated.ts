// Session 4: listen for the server's `goals_updated` event. Fires after
// every completed session with the latest `recompute_all_goals` payload.

import { useEffect } from "react";
import { acquireSharedSocket, releaseSharedSocket } from "./sharedSocket";
import type { Goal } from "@/lib/plansApi";

export type GoalsUpdatedPayload = { goals: Goal[] };

export function useGoalsUpdated(
  onUpdated: (payload: GoalsUpdatedPayload) => void,
) {
  useEffect(() => {
    let cancelled = false;
    try {
      const socket = acquireSharedSocket();
      const handler = (p: GoalsUpdatedPayload) => {
        if (!cancelled) onUpdated(p);
      };
      socket.on("goals_updated", handler);
      return () => {
        cancelled = true;
        socket.off("goals_updated", handler);
        releaseSharedSocket();
      };
    } catch {
      return () => {};
    }
  }, [onUpdated]);
}
