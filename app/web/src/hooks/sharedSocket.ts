// Shared Socket.IO client with reference counting. Hooks like
// useSessionCompleted, useGoalsUpdated, useMilestonesReached, and
// useBadgesEarned all share this one underlying socket so opening the
// dashboard doesn't fan out into N separate WebSocket connections.

import { io, type Socket } from "socket.io-client";

let sharedSocket: Socket | null = null;
let refCount = 0;

export function acquireSharedSocket(): Socket {
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

export function releaseSharedSocket() {
  refCount = Math.max(0, refCount - 1);
  if (refCount === 0 && sharedSocket) {
    sharedSocket.disconnect();
    sharedSocket = null;
  }
}
