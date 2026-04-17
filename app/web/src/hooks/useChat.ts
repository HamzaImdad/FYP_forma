// useChat — encapsulates the streaming chat lifecycle for one conversation.
//
// State: messages, streamingText, pending, error, conversationId.
// Sends to /api/chat/guide, /api/chat/personal, or /api/chat/plan based on mode.
// On done, the streaming text is committed to messages as an assistant row.
// On error, the error is surfaced and streaming is cleared.

import { useCallback, useEffect, useRef, useState } from "react";
import { streamChat, type ChatMessageDTO, chatApi } from "@/lib/chatApi";

export type ChatMode = "guide" | "personal" | "plan" | "public";

export type UseChatOptions = {
  mode: ChatMode;
  conversationId?: number | null;
  initialMessages?: ChatMessageDTO[];
  // Session 4: called whenever a streaming turn finishes (done or error).
  // Plans page uses this to refetch /api/chat/conversations/:id/plan_draft
  // and render the live preview card.
  onTurnDone?: (conversationId: number | null) => void;
};

const ENDPOINT_BY_MODE: Record<ChatMode, string> = {
  guide: "/api/chat/guide",
  personal: "/api/chat/personal",
  plan: "/api/chat/plan",
  public: "/api/chat/public",
};

export function useChat({
  mode,
  conversationId,
  initialMessages,
  onTurnDone,
}: UseChatOptions) {
  const [messages, setMessages] = useState<ChatMessageDTO[]>(initialMessages ?? []);
  const [streamingText, setStreamingText] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeConversationId, setActiveConversationId] = useState<number | null>(
    conversationId ?? null,
  );
  const abortRef = useRef<AbortController | null>(null);
  const activeIdRef = useRef<number | null>(activeConversationId);
  activeIdRef.current = activeConversationId;
  const onTurnDoneRef = useRef(onTurnDone);
  onTurnDoneRef.current = onTurnDone;
  // Ids whose messages are already faithfully represented in local state.
  // Prevents the load-on-id-change effect from racing with an in-flight
  // stream (meta assigns the id mid-send → reload would overwrite the
  // optimistic user turn and duplicate the assistant reply once `done`
  // commits on top).
  const localSyncedIds = useRef<Set<number>>(new Set());

  useEffect(() => {
    setActiveConversationId(conversationId ?? null);
  }, [conversationId]);

  // Load past conversation history when the id changes (personal + plan modes).
  // Skip when this id was assigned mid-send — local state is already in sync
  // with what the server has, and reloading would race with the live stream.
  useEffect(() => {
    if (mode === "guide" || mode === "public" || activeConversationId == null) return;
    if (localSyncedIds.current.has(activeConversationId)) return;
    let cancelled = false;
    chatApi
      .loadConversation(activeConversationId)
      .then((d) => {
        if (cancelled) return;
        setMessages(
          d.messages
            .filter((m) => m.role === "user" || m.role === "assistant")
            .map((m) => ({ ...m })),
        );
        setStreamingText("");
        setError(null);
        localSyncedIds.current.add(activeConversationId);
      })
      .catch(() => {
        if (!cancelled) setError("conversation_load_failed");
      });
    return () => {
      cancelled = true;
    };
  }, [mode, activeConversationId]);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setMessages([]);
    setStreamingText("");
    setPending(false);
    setError(null);
    setActiveConversationId(null);
  }, []);

  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || pending) return;
      setError(null);

      // Optimistic user message
      const userMsg: ChatMessageDTO = { role: "user", content: trimmed };
      const next = [...messages, userMsg];
      setMessages(next);
      setStreamingText("");
      setPending(true);

      const controller = new AbortController();
      abortRef.current = controller;

      const endpoint = ENDPOINT_BY_MODE[mode];
      const body: Record<string, unknown> = {
        messages: next.map((m) => ({ role: m.role, content: m.content })),
      };
      if (mode !== "guide" && mode !== "public" && activeConversationId != null) {
        body.conversation_id = activeConversationId;
      }

      let buffer = "";
      try {
        await streamChat(
          endpoint,
          body,
          (event) => {
            if (event.type === "meta") {
              if (event.conversation_id != null) {
                localSyncedIds.current.add(event.conversation_id);
                setActiveConversationId(event.conversation_id);
              }
            } else if (event.type === "chunk") {
              buffer += event.text;
              setStreamingText(buffer);
            } else if (event.type === "done") {
              // Commit the streaming text as the final assistant turn.
              // Dedup guard: StrictMode dev double-invokes plus any SSE
              // edge-case re-fire used to append the same reply twice.
              if (buffer.trim()) {
                setMessages((prev) => {
                  const last = prev[prev.length - 1];
                  if (last && last.role === "assistant" && last.content === buffer) {
                    return prev;
                  }
                  return [...prev, { role: "assistant", content: buffer }];
                });
              }
              setStreamingText("");
              setPending(false);
              onTurnDoneRef.current?.(activeIdRef.current);
            } else if (event.type === "error") {
              setError(event.error);
              if (buffer.trim()) {
                setMessages((prev) => {
                  const last = prev[prev.length - 1];
                  if (last && last.role === "assistant" && last.content === buffer) {
                    return prev;
                  }
                  return [...prev, { role: "assistant", content: buffer }];
                });
              }
              setStreamingText("");
              setPending(false);
              onTurnDoneRef.current?.(activeIdRef.current);
            }
          },
          controller.signal,
        );
      } catch (e) {
        console.warn("chat stream aborted", e);
        setError("stream_failed");
        setPending(false);
      }
    },
    [messages, mode, activeConversationId, pending],
  );

  const injectAssistantHint = useCallback((hint: string) => {
    // Used by real-time awareness — inject a system-level nudge from the
    // client side so the LLM responds proactively on the next turn.
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: hint },
    ]);
  }, []);

  useEffect(
    () => () => {
      abortRef.current?.abort();
    },
    [],
  );

  return {
    messages,
    streamingText,
    pending,
    error,
    conversationId: activeConversationId,
    send,
    reset,
    injectAssistantHint,
  };
}
