// Chat message thread — renders bubbles, auto-scrolls to the bottom while
// tokens stream in, and turns inline `[session #N]` markers into clickable
// citations that navigate to the Session 2 SessionDetailPanel.

import { useEffect, useMemo, useRef } from "react";
import { Link } from "react-router-dom";
import type { ChatMessageDTO } from "@/lib/chatApi";

type Props = {
  messages: ChatMessageDTO[];
  streamingText?: string;
  pending?: boolean;
  emptyState?: React.ReactNode;
};

const CITE_RE = /\[session\s*#?\s*(\d+)\]/gi;

function renderContent(content: string): React.ReactNode {
  if (!content) return null;
  const nodes: React.ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  CITE_RE.lastIndex = 0;
  while ((match = CITE_RE.exec(content)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(content.slice(lastIndex, match.index));
    }
    const id = match[1];
    nodes.push(
      <Link
        key={`cite-${match.index}-${id}`}
        to={`/dashboard/session/${id}`}
        className="inline-flex items-center gap-1 px-1.5 py-0.5 mx-0.5 rounded-sm border border-[color:var(--color-gold)]/40 text-[color:var(--color-gold)] hover:bg-[color:var(--color-gold)]/10 text-[0.82em] uppercase tracking-[0.12em] no-underline"
      >
        session {id}
      </Link>,
    );
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < content.length) {
    nodes.push(content.slice(lastIndex));
  }
  return nodes;
}

export function MessageThread({ messages, streamingText, pending, emptyState }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);

  const visible = useMemo(
    () => messages.filter((m) => m.role === "user" || m.role === "assistant"),
    [messages],
  );

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [visible, streamingText, pending]);

  if (visible.length === 0 && !streamingText && !pending) {
    return (
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-6 md:px-10 py-10 flex items-center justify-center"
      >
        {emptyState}
      </div>
    );
  }

  return (
    <div
      ref={scrollRef}
      className="flex-1 overflow-y-auto px-6 md:px-10 py-8 space-y-5"
    >
      {visible.map((m, idx) => {
        const isUser = m.role === "user";
        return (
          <div
            key={m.id ?? `msg-${idx}`}
            className={`flex ${isUser ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[82%] rounded-[10px] px-5 py-4 leading-[1.6] text-[0.97rem] ${
                isUser
                  ? "bg-[color:var(--color-gold)]/15 text-[color:var(--color-ink)] border border-[color:var(--color-gold)]/30"
                  : "bg-[color:var(--color-raised)] text-[color:var(--color-ink)] border border-[color:var(--rule)]"
              }`}
            >
              <div className="whitespace-pre-wrap">{renderContent(m.content)}</div>
            </div>
          </div>
        );
      })}

      {(streamingText || pending) && (
        <div className="flex justify-start">
          <div className="max-w-[82%] rounded-[10px] px-5 py-4 leading-[1.6] text-[0.97rem] bg-[color:var(--color-raised)] text-[color:var(--color-ink)] border border-[color:var(--rule)]">
            {streamingText ? (
              <>
                <div className="whitespace-pre-wrap">{renderContent(streamingText)}</div>
                <span className="inline-block w-[2px] h-4 ml-1 align-middle bg-[color:var(--color-gold)] animate-pulse" />
              </>
            ) : (
              <div className="flex items-center gap-1.5 h-4">
                {[0, 150, 300].map((d) => (
                  <span
                    key={d}
                    className="block h-[6px] w-[6px] rounded-full bg-[color:var(--color-gold)]/70 animate-pulse"
                    style={{ animationDelay: `${d}ms` }}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
