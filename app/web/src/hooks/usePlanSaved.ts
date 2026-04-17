// Session 5: listen for the server's `plan_saved` event. Fires after a
// new plan is persisted, whether through the Plan Architect chatbot
// (chat_tools.save_plan → dispatcher hook) or the custom-builder form
// (POST /api/plans).

import { useEffect } from "react";
import { acquireSharedSocket, releaseSharedSocket } from "./sharedSocket";

export type PlanSavedPayload = {
  plan_id: number;
  title: string | null;
  start_date: string | null;
  end_date: string | null;
  source: "chat" | "custom";
};

export function usePlanSaved(onSaved: (payload: PlanSavedPayload) => void) {
  useEffect(() => {
    let cancelled = false;
    try {
      const socket = acquireSharedSocket();
      const handler = (p: PlanSavedPayload) => {
        if (!cancelled) onSaved(p);
      };
      socket.on("plan_saved", handler);
      return () => {
        cancelled = true;
        socket.off("plan_saved", handler);
        releaseSharedSocket();
      };
    } catch {
      return () => {};
    }
  }, [onSaved]);
}
