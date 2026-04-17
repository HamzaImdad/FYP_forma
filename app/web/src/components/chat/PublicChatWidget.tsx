// PublicChatWidget — floating guide on every logged-out public page.
//
// Anchored bottom-right. Click the gold bubble → opens a 380x560 chat panel.
// Streams from /api/chat/public (via useChat mode="public"), which is backed
// by app/public_chat.py's KB retrieval over data/website_kb.json.
//
// Visibility is owned by App.tsx: this component is only mounted when the
// visitor is logged out AND not on the /login route. Nothing inside here
// checks auth or pathname.

import { useCallback, useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { useChat } from "@/hooks/useChat";

const ERROR_COPY: Record<string, string> = {
  openai_key_missing:
    "The guide is offline for maintenance. Try the About page in the meantime.",
  daily_token_budget_exceeded:
    "The guide is busy right now — try again in a few minutes.",
  stream_failed: "Connection dropped. Try your question again.",
  http_429: "Too many questions — slow down for a moment.",
};

const SUGGESTIONS = [
  "What is FORMA?",
  "How does form scoring work?",
  "Which exercises does it cover?",
  "Do I need any equipment?",
];

export function PublicChatWidget() {
  const [open, setOpen] = useState(false);
  const { messages, streamingText, pending, error, send } = useChat({ mode: "public" });
  const [draft, setDraft] = useState("");
  const threadRef = useRef<HTMLDivElement>(null);

  // Autoscroll to bottom on new messages / streaming
  useEffect(() => {
    const el = threadRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages, streamingText, open]);

  // Escape to close
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent | globalThis.KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("keydown", onKey as EventListener);
    return () => document.removeEventListener("keydown", onKey as EventListener);
  }, [open]);

  const handleSubmit = useCallback(
    (e?: FormEvent) => {
      e?.preventDefault();
      const t = draft.trim();
      if (!t || pending) return;
      send(t);
      setDraft("");
    },
    [draft, pending, send],
  );

  const handleSuggestion = useCallback(
    (text: string) => {
      if (pending) return;
      send(text);
    },
    [pending, send],
  );

  const onKeyDownArea = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const errorText = error ? (ERROR_COPY[error] ?? "Something went wrong. Try again.") : null;
  const empty = messages.length === 0 && !streamingText && !pending;

  return (
    <>
      {/* Floating launcher button */}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={
          "fixed bottom-4 right-4 sm:bottom-6 sm:right-6 z-[600] h-14 w-14 rounded-full flex items-center justify-center " +
          "bg-[color:var(--color-gold)] text-[color:var(--color-page)] " +
          "shadow-[0_12px_40px_-8px_rgba(174,231,16,0.45)] " +
          "hover:bg-[color:var(--color-gold-hover)] transition-all duration-200 " +
          "ring-1 ring-[color:var(--color-gold-soft)]/40 " +
          (open ? "scale-90" : "scale-100 hover:scale-105")
        }
        aria-label={open ? "Close FORMA guide" : "Open FORMA guide"}
        aria-expanded={open}
      >
        {open ? (
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path
              d="M6 6l12 12M18 6L6 18"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
            />
          </svg>
        ) : (
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path
              d="M21 12a8 8 0 0 1-11.3 7.3L4 21l1.7-5.7A8 8 0 1 1 21 12z"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinejoin="round"
            />
          </svg>
        )}
      </button>

      {/* Chat panel */}
      <div
        className={
          "fixed bottom-20 right-3 left-3 sm:bottom-24 sm:right-6 sm:left-auto z-[600] " +
          "sm:w-[380px] max-h-[calc(100vh-7rem)] sm:max-h-[calc(100vh-8rem)] " +
          "rounded-[8px] overflow-hidden flex flex-col " +
          "bg-[color:var(--color-page)] border border-[color:var(--rule)] " +
          "shadow-[0_32px_80px_-20px_rgba(0,0,0,0.5)] " +
          "origin-bottom-right transition-all duration-200 " +
          (open
            ? "opacity-100 translate-y-0 scale-100 pointer-events-auto"
            : "opacity-0 translate-y-4 scale-95 pointer-events-none")
        }
        role="dialog"
        aria-label="FORMA website guide"
        aria-hidden={!open}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 bg-[color:var(--color-contrast)] text-[color:var(--color-ink-on-dark)] border-b border-[color:var(--color-ink-on-dark)]/10">
          <div className="flex items-center gap-3">
            <span
              className="block h-2 w-2 rounded-full bg-[color:var(--color-gold-soft)] shadow-[0_0_8px_rgba(174,231,16,0.6)]"
              aria-hidden="true"
            />
            <div>
              <div className="font-[family-name:var(--font-display)] text-sm tracking-[0.14em] uppercase">
                Ask FORMA
              </div>
              <div className="font-[family-name:var(--font-mono)] text-[0.55rem] uppercase tracking-[0.22em] text-[color:var(--color-gold-soft)]">
                Website guide
              </div>
            </div>
          </div>
          <button
            type="button"
            onClick={() => setOpen(false)}
            className="text-[color:var(--color-ink-on-dark-2)] hover:text-[color:var(--color-gold-soft)] transition-colors"
            aria-label="Minimize chat"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path
                d="M6 12h12"
                stroke="currentColor"
                strokeWidth="2.2"
                strokeLinecap="round"
              />
            </svg>
          </button>
        </div>

        {/* Messages */}
        <div
          ref={threadRef}
          className="flex-1 overflow-y-auto px-5 py-5 space-y-3 bg-[color:var(--color-page)]"
        >
          {empty ? (
            <div>
              <p className="font-[family-name:var(--font-serif)] italic text-[color:var(--color-ink-2)] mb-4">
                Questions about FORMA? Ask the guide. It reads the site so you don't have to.
              </p>
              <div className="space-y-2">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => handleSuggestion(s)}
                    className="block w-full text-left px-3 py-2 border border-[color:var(--rule)] text-[0.82rem] text-[color:var(--color-ink-2)] hover:border-[color:var(--color-gold)] hover:text-[color:var(--color-ink)] transition-colors rounded-[3px]"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <>
              {messages
                .filter((m) => m.role === "user" || m.role === "assistant")
                .map((m, i) => (
                  <MessageBubble
                    key={i}
                    role={m.role as "user" | "assistant"}
                    text={m.content}
                  />
                ))}
              {pending && streamingText && <MessageBubble role="assistant" text={streamingText} streaming />}
              {pending && !streamingText && (
                <div className="flex items-center gap-2 text-[0.8rem] text-[color:var(--color-ink-2)] font-[family-name:var(--font-sans)]">
                  <span className="inline-block h-1.5 w-1.5 rounded-full bg-[color:var(--color-gold)] animate-pulse" />
                  <span>Thinking…</span>
                </div>
              )}
            </>
          )}
          {errorText && (
            <div
              className="px-3 py-2 border-l-2 text-[0.8rem]"
              style={{
                borderColor: "var(--color-bad)",
                color: "var(--color-bad)",
                background: "rgba(248,113,113,0.08)",
              }}
            >
              {errorText}
            </div>
          )}
        </div>

        {/* Input */}
        <form
          onSubmit={handleSubmit}
          className="p-3 border-t border-[color:var(--rule)] bg-[color:var(--color-page)]"
        >
          <div className="flex items-end gap-2">
            <textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={onKeyDownArea}
              placeholder="Ask about FORMA…"
              rows={1}
              className="flex-1 resize-none bg-[color:var(--color-raised)] border border-[color:var(--rule)] rounded-[3px] px-3 py-2 text-[0.88rem] text-[color:var(--color-ink)] placeholder:text-[color:var(--color-ink-2)]/60 focus:outline-none focus:border-[color:var(--color-gold)] transition-colors max-h-32"
              disabled={pending}
            />
            <button
              type="submit"
              disabled={pending || !draft.trim()}
              className="h-9 px-3 bg-[color:var(--color-gold)] text-[color:var(--color-page)] rounded-[3px] text-[0.7rem] uppercase tracking-[0.14em] font-medium hover:bg-[color:var(--color-gold-hover)] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              aria-label="Send"
            >
              Send
            </button>
          </div>
          <div className="mt-2 font-[family-name:var(--font-mono)] text-[0.55rem] uppercase tracking-[0.18em] text-[color:var(--color-ink-2)]/70">
            AI-generated · Not medical advice
          </div>
        </form>
      </div>
    </>
  );
}

function MessageBubble({
  role,
  text,
  streaming,
}: {
  role: "user" | "assistant";
  text: string;
  streaming?: boolean;
}) {
  return (
    <div className={`flex ${role === "user" ? "justify-end" : "justify-start"}`}>
      <div
        className={
          "max-w-[85%] px-3 py-2 rounded-[6px] text-[0.85rem] leading-[1.5] whitespace-pre-wrap " +
          (role === "user"
            ? "bg-[color:var(--color-gold)]/12 border border-[color:var(--color-gold)]/30 text-[color:var(--color-ink)]"
            : "bg-[color:var(--color-raised)] border border-[color:var(--rule)] text-[color:var(--color-ink-2)]")
        }
      >
        {text}
        {streaming && <span className="inline-block ml-0.5 h-3 w-[2px] bg-[color:var(--color-gold)] animate-pulse" />}
      </div>
    </div>
  );
}
