// FORMA chat API client — thin wrapper around /api/chat/* + a hand-rolled
// SSE parser. We can't use native EventSource because it doesn't support
// POST bodies; we stream the response via fetch+reader instead.

import { api } from "./api";

export type ChatRole = "user" | "assistant" | "system";

export type ChatMessageDTO = {
  id?: number;
  role: ChatRole;
  content: string;
  citations?: number[];
  created_at?: string;
};

export type ConversationSummary = {
  id: number;
  mode: "guide" | "personal" | "plan";
  title: string | null;
  created_at: string;
  updated_at: string;
  message_count: number;
};

export type ConversationDetail = {
  conversation: ConversationSummary;
  messages: ChatMessageDTO[];
};

export type StreamEvent =
  | { type: "meta"; conversation_id?: number; model?: string }
  | { type: "chunk"; text: string }
  | { type: "done" }
  | { type: "error"; error: string; message?: string };

export type StreamHandler = (event: StreamEvent) => void;

export async function streamChat(
  endpoint: string,
  body: Record<string, unknown>,
  handler: StreamHandler,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(endpoint, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok) {
    let errorCode = `http_${res.status}`;
    try {
      const j = await res.json();
      errorCode = String(j?.error ?? errorCode);
    } catch {
      // ignore
    }
    handler({ type: "error", error: errorCode });
    return;
  }
  if (!res.body) {
    handler({ type: "error", error: "no_body" });
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Split on SSE event boundaries (double newline).
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const raw = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");

      let eventName = "message";
      const dataLines: string[] = [];
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) eventName = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (dataLines.length === 0) continue;
      const dataRaw = dataLines.join("\n");
      let parsed: unknown;
      try {
        parsed = JSON.parse(dataRaw);
      } catch {
        parsed = dataRaw;
      }

      if (eventName === "chunk") {
        handler({ type: "chunk", text: typeof parsed === "string" ? parsed : String(parsed) });
      } else if (eventName === "meta") {
        const meta = (parsed ?? {}) as Record<string, unknown>;
        handler({
          type: "meta",
          conversation_id:
            typeof meta.conversation_id === "number" ? meta.conversation_id : undefined,
          model: typeof meta.model === "string" ? meta.model : undefined,
        });
      } else if (eventName === "done") {
        handler({ type: "done" });
      } else if (eventName === "error") {
        const err = (parsed ?? {}) as Record<string, unknown>;
        handler({
          type: "error",
          error: String(err.error ?? "unknown"),
          message: err.message != null ? String(err.message) : undefined,
        });
      }
    }
  }
  handler({ type: "done" });
}

export const chatApi = {
  listConversations: (mode: string = "personal") =>
    api<{ conversations: ConversationSummary[] }>(
      `/api/chat/conversations?mode=${encodeURIComponent(mode)}`,
    ).then((r) => r.conversations),

  loadConversation: (id: number) =>
    api<ConversationDetail>(`/api/chat/conversations/${id}`),

  deleteConversation: (id: number) =>
    api<{ ok: boolean }>(`/api/chat/conversations/${id}`, { method: "DELETE" }),

  usage: () =>
    api<{ allowed: boolean; used: number; budget: number; fraction: number; degrade: boolean }>(
      "/api/chat/usage",
    ),
};
