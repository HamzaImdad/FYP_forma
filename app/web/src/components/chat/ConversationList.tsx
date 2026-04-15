// Sidebar list of past conversations. Personal mode only — the guide
// chatbot is stateless and doesn't surface history to logged-out visitors.

import { useCallback, useEffect, useState } from "react";
import { chatApi, type ConversationSummary } from "@/lib/chatApi";

type Props = {
  activeId: number | null;
  onSelect: (id: number) => void;
  onNew: () => void;
  reloadKey?: number;
};

function formatUpdatedAt(iso: string): string {
  try {
    const d = new Date(iso);
    const now = new Date();
    const diffMs = now.getTime() - d.getTime();
    const day = 24 * 60 * 60 * 1000;
    if (diffMs < day && d.getDate() === now.getDate()) {
      return d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    }
    if (diffMs < 7 * day) {
      return d.toLocaleDateString([], { weekday: "short" });
    }
    return d.toLocaleDateString([], { month: "short", day: "numeric" });
  } catch {
    return iso;
  }
}

export function ConversationList({ activeId, onSelect, onNew, reloadKey }: Props) {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(() => {
    chatApi
      .listConversations("personal")
      .then(setConversations)
      .catch(() => setConversations([]))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh, reloadKey]);

  const handleDelete = useCallback(
    async (id: number, e: React.MouseEvent) => {
      e.stopPropagation();
      if (!confirm("Delete this conversation? This cannot be undone.")) return;
      try {
        await chatApi.deleteConversation(id);
        setConversations((prev) => prev.filter((c) => c.id !== id));
        if (activeId === id) onNew();
      } catch (err) {
        console.warn("delete conversation failed", err);
      }
    },
    [activeId, onNew],
  );

  return (
    <aside className="w-[280px] border-r border-[color:var(--rule)] bg-[color:var(--color-raised)]/40 flex flex-col">
      <div className="px-5 pt-5 pb-3">
        <div className="text-[10px] uppercase tracking-[0.24em] text-[color:var(--color-gold)]">
          Coach
        </div>
        <div
          className="mt-1 text-[color:var(--color-ink)]"
          style={{ fontFamily: "var(--font-display)", fontSize: "1.6rem", letterSpacing: "0.04em" }}
        >
          CONVERSATIONS
        </div>
      </div>
      <button
        type="button"
        onClick={onNew}
        className="mx-5 mb-3 px-4 py-2 text-[11px] uppercase tracking-[0.18em] border border-[color:var(--color-gold)] text-[color:var(--color-gold)] hover:bg-[color:var(--color-gold)] hover:text-[color:var(--color-page)] transition-colors rounded-[3px]"
      >
        + New conversation
      </button>
      <div className="flex-1 overflow-y-auto px-3 pb-4">
        {loading ? (
          <div className="px-3 py-4 text-[11px] uppercase tracking-[0.2em] text-[color:var(--color-ink-4)]">
            Loading…
          </div>
        ) : conversations.length === 0 ? (
          <div className="px-3 py-4 text-[11px] text-[color:var(--color-ink-4)] italic" style={{ fontFamily: "var(--font-serif)" }}>
            No saved conversations yet. Ask your coach anything.
          </div>
        ) : (
          <ul className="space-y-1">
            {conversations.map((c) => (
              <li key={c.id}>
                <button
                  type="button"
                  onClick={() => onSelect(c.id)}
                  className={`group w-full text-left px-3 py-2.5 rounded-[4px] transition-colors ${
                    activeId === c.id
                      ? "bg-[color:var(--color-gold)]/15 border border-[color:var(--color-gold)]/40"
                      : "hover:bg-[color:var(--color-raised)] border border-transparent"
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0 flex-1">
                      <div className="text-[0.85rem] text-[color:var(--color-ink)] truncate">
                        {c.title || "Untitled"}
                      </div>
                      <div className="text-[10px] uppercase tracking-[0.16em] text-[color:var(--color-ink-4)] mt-1">
                        {formatUpdatedAt(c.updated_at)} · {c.message_count} msg
                      </div>
                    </div>
                    <span
                      role="button"
                      aria-label="Delete conversation"
                      onClick={(e) => handleDelete(c.id, e)}
                      className="opacity-0 group-hover:opacity-100 text-[color:var(--color-ink-4)] hover:text-[color:var(--color-bad)] text-xs transition-opacity px-1 cursor-pointer"
                    >
                      ✕
                    </span>
                  </div>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </aside>
  );
}
