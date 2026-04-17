// Session 5: listen for `plan_day_completed`. Emitted from handle_end_session
// when a user's aggregated today-reps for an exercise meet or exceed the
// planned volume on today's plan day. TodaysPlanStrip + PlanShell subscribe
// to refetch so the day tile flips to "done ✓" without a manual refresh.

import { useEffect } from "react";
import { acquireSharedSocket, releaseSharedSocket } from "./sharedSocket";

export type PlanDayCompletedPayload = {
  plan_id: number;
  day_id: number;
  exercise: string;
  reps_today: number;
  planned: number;
};

export function usePlanDayCompleted(
  onCompleted: (payload: PlanDayCompletedPayload) => void,
) {
  useEffect(() => {
    let cancelled = false;
    try {
      const socket = acquireSharedSocket();
      const handler = (p: PlanDayCompletedPayload) => {
        if (!cancelled) onCompleted(p);
      };
      socket.on("plan_day_completed", handler);
      return () => {
        cancelled = true;
        socket.off("plan_day_completed", handler);
        releaseSharedSocket();
      };
    } catch {
      return () => {};
    }
  }, [onCompleted]);
}
