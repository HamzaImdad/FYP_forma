// Session 4: listen for `milestones_reached` — the server fires this when
// on_session_complete notices that a goal's current_value crossed one of
// its 25/50/75/100 thresholds.

import { useEffect } from "react";
import { acquireSharedSocket, releaseSharedSocket } from "./sharedSocket";
import type { MilestoneWithGoal } from "@/lib/plansApi";

export type MilestonesReachedPayload = { milestones: MilestoneWithGoal[] };

export function useMilestonesReached(
  onReached: (payload: MilestonesReachedPayload) => void,
) {
  useEffect(() => {
    let cancelled = false;
    try {
      const socket = acquireSharedSocket();
      const handler = (p: MilestonesReachedPayload) => {
        if (!cancelled) onReached(p);
      };
      socket.on("milestones_reached", handler);
      return () => {
        cancelled = true;
        socket.off("milestones_reached", handler);
        releaseSharedSocket();
      };
    } catch {
      return () => {};
    }
  }, [onReached]);
}
