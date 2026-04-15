// Subscribe to the server's `session_completed` Socket.IO event so the
// dashboard can refetch the overview whenever the user finishes a workout.
//
// This opens a long-lived passive socket that only listens for the one
// event. We keep it separate from WorkoutPage's socket because the dashboard
// needs to hear about sessions completed in any tab, not just while the user
// has the workout page open.

import { useEffect } from "react";
import { io, type Socket } from "socket.io-client";

let sharedSocket: Socket | null = null;
let refCount = 0;

function acquire(): Socket {
  if (!sharedSocket) {
    sharedSocket = io({
      transports: ["websocket", "polling"],
      reconnection: true,
      reconnectionAttempts: 5,
      timeout: 20000,
      autoConnect: true,
    });
  }
  refCount += 1;
  return sharedSocket;
}

function release() {
  refCount = Math.max(0, refCount - 1);
  if (refCount === 0 && sharedSocket) {
    sharedSocket.disconnect();
    sharedSocket = null;
  }
}

export type SessionCompletedPayload = {
  exercise: string;
  session_id?: number;
  total_reps?: number;
  avg_form_score?: number;
};

export function useSessionCompleted(
  onCompleted: (payload: SessionCompletedPayload) => void,
) {
  useEffect(() => {
    let cancelled = false;
    let socket: Socket | null = null;
    try {
      socket = acquire();
      const handler = (p: SessionCompletedPayload) => {
        if (!cancelled) onCompleted(p);
      };
      socket.on("session_completed", handler);
      return () => {
        cancelled = true;
        socket?.off("session_completed", handler);
        release();
      };
    } catch {
      // Socket connection refused because cookie missing or server down —
      // silently bail so the dashboard still renders from HTTP data.
      return () => {};
    }
  }, [onCompleted]);
}
