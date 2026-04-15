// ChatShell — two-column layout for the personal chatbot: conversation
// list on the left, thread + input on the right. Also used by the dashboard's
// floating coach panel in compact mode (sidebar hidden).

import { useCallback, useEffect, useState } from "react";
import { MessageThread } from "./MessageThread";
import { MessageInput } from "./MessageInput";
import { SuggestedPrompts, type SuggestedContext } from "./SuggestedPrompts";
import { ConversationList } from "./ConversationList";
import { useChat } from "@/hooks/useChat";
import { useSessionCompleted } from "@/hooks/useSessionCompleted";

type Props = {
  mode: "guide" | "personal" | "plan";
  showSidebar?: boolean;
  initialContext?: SuggestedContext;
  title?: string;
  tagline?: string;
  authed?: boolean;
  onSignupClick?: () => void;
  // Session 4: Plans page hooks this so the live plan preview can refetch
  // the conversation's plan_draft column whenever a streaming turn completes.
  onTurnDone?: (conversationId: number | null) => void;
  inputPlaceholder?: string;
};

const ERROR_COPY: Record<string, string> = {
  daily_token_budget_exceeded:
    "You've used your daily chat budget. Try again tomorrow or shorten your questions.",
  openai_key_missing:
    "The coach is offline — the server is missing its OpenAI key. Set OPENAI_API_KEY in .env to enable it.",
  conversation_not_found: "That conversation doesn't exist.",
  stream_failed: "Connection dropped mid-answer. Try again.",
  empty_messages: "Type a message first.",
  http_401: "You need to sign in to use the personal coach.",
  http_429: "Too many requests — slow down for a moment.",
};

export function ChatShell({
  mode,
  showSidebar = true,
  initialContext = "guest",
  title,
  tagline,
  authed = false,
  onSignupClick,
  onTurnDone,
  inputPlaceholder,
}: Props) {
  const [activeConversationId, setActiveConversationId] = useState<number | null>(null);
  const [conversationReloadKey, setConversationReloadKey] = useState(0);
  const [inputDraft, setInputDraft] = useState("");

  const {
    messages,
    streamingText,
    pending,
    error,
    conversationId,
    send,
    reset,
    injectAssistantHint,
  } = useChat({ mode, conversationId: activeConversationId, onTurnDone });

  // Personal mode: when a session completes, surface a proactive nudge.
  useSessionCompleted(
    useCallback(
      (payload) => {
        if (mode !== "personal" || !authed) return;
        injectAssistantHint(
          `Nice work — you just wrapped ${payload.exercise}. Want a quick breakdown of how it went?`,
        );
      },
      [mode, authed, injectAssistantHint],
    ),
  );

  const handleNew = useCallback(() => {
    setActiveConversationId(null);
    reset();
    setConversationReloadKey((k) => k + 1);
  }, [reset]);

  const handleSelect = useCallback((id: number) => {
    setActiveConversationId(id);
  }, []);

  const handleSend = useCallback(
    (text: string) => {
      send(text);
      setInputDraft("");
      // Bump the sidebar refresh on the first message so new conversations
      // show up without requiring a navigation.
      setConversationReloadKey((k) => k + 1);
    },
    [send],
  );

  // Sync the conversation id once the server assigns one (first send).
  useEffect(() => {
    if (mode !== "guide" && conversationId != null && activeConversationId == null) {
      setActiveConversationId(conversationId);
    }
  }, [mode, conversationId, activeConversationId]);

  const context: SuggestedContext =
    mode === "guide"
      ? "guest"
      : messages.length === 0
        ? initialContext
        : "has-data";

  const emptyStateContent = (() => {
    if (mode === "guide") {
      return {
        title: title ?? "MEET YOUR FORMA GUIDE",
        tagline:
          tagline ?? "Ask anything about FORMA, what it tracks, and how it works.",
      };
    }
    if (mode === "plan") {
      return {
        title: title ?? "PLAN ARCHITECT",
        tagline:
          tagline ??
          "Tell me how you train and I'll build an adaptive workout plan.",
      };
    }
    return {
      title: title ?? "YOUR PERSONAL COACH",
      tagline:
        tagline ??
        "I read every rep you've logged. Ask me anything about your training.",
    };
  })();

  const emptyState = (
    <div className="max-w-xl text-center">
      <div
        className="text-[color:var(--color-ink)] mb-3"
        style={{
          fontFamily: "var(--font-display)",
          fontSize: "2.4rem",
          letterSpacing: "0.04em",
        }}
      >
        {emptyStateContent.title}
      </div>
      <p
        className="italic text-[color:var(--color-ink-3)] text-lg"
        style={{ fontFamily: "var(--font-serif)" }}
      >
        {emptyStateContent.tagline}
      </p>
    </div>
  );

  return (
    <div className="flex h-full min-h-0 w-full">
      {showSidebar && mode === "personal" && (
        <ConversationList
          activeId={activeConversationId ?? conversationId ?? null}
          onSelect={handleSelect}
          onNew={handleNew}
          reloadKey={conversationReloadKey}
        />
      )}
      <div className="flex-1 flex flex-col min-w-0">
        <MessageThread
          messages={messages}
          streamingText={streamingText}
          pending={pending && !streamingText}
          emptyState={emptyState}
        />
        {error && (
          <div className="px-6 md:px-10 py-3 bg-[color:var(--color-bad)]/10 border-t border-[color:var(--color-bad)]/30 text-[color:var(--color-bad)] text-[12px] uppercase tracking-[0.16em]">
            {ERROR_COPY[error] ?? error}
          </div>
        )}
        {mode === "guide" && !authed && messages.length >= 2 && onSignupClick && (
          <div className="px-6 md:px-10 py-4 border-t border-[color:var(--rule)] bg-[color:var(--color-raised)]/60">
            <button
              type="button"
              onClick={onSignupClick}
              className="w-full text-center px-6 py-3 bg-[color:var(--color-gold)] text-[color:var(--color-page)] text-[11px] uppercase tracking-[0.24em] hover:bg-[color:var(--color-gold-hover)] transition-colors rounded-[3px]"
            >
              Sign up to get your personal coach
            </button>
          </div>
        )}
        <SuggestedPrompts
          context={context}
          onPick={(text) => setInputDraft(text)}
          disabled={pending}
        />
        <MessageInput
          disabled={pending}
          onSend={handleSend}
          initialValue={inputDraft}
          placeholder={
            inputPlaceholder ??
            (mode === "guide"
              ? "Ask about FORMA…"
              : mode === "plan"
                ? "Describe your schedule and goals…"
                : "Ask about your training…")
          }
        />
      </div>
    </div>
  );
}
