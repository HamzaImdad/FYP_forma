// Sidebar list of past conversations. Personal mode only — the guide
// chatbot is stateless and doesn't surface history to logged-out visitors.

import { useCallback, useEffect, useState } from "react";
import { chatApi, type ConversationSummary } from "@/lib/chatApi";

type Props = {
  activeId: number | null;
  onSelect: (id: number) => void;
  onNew: () => void;
  reloadKey?: number;
  // Mobile drawer mode — when set, sidebar renders as a slide-in drawer
  // on narrow screens. Desktop (md+) behavior is unchanged.
  mobileOpen?: boolean;
  onClose?: () => void;
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

export function ConversationList({
  activeId,
  onSelect,
  onNew,
  reloadKey,
  mobileOpen = false,
  onClose,
}: Props) {
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

  // Close drawer on Escape (mobile only)
  useEffect(() => {
    if (!mobileOpen || !onClose) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mobileOpen, onClose]);

  const handleSelectAndClose = useCallback(
    (id: number) => {
      onSelect(id);
      onClose?.();
    },
    [onSelect, onClose],
  );

  const handleNewAndClose = useCallback(() => {
    onNew();
    onClose?.();
  }, [onNew, onClose]);

  return (
    <>
      {/* Mobile backdrop — only visible when drawer is open on < md */}
      {onClose && (
        <div
          onClick={onClose}
          aria-hidden="true"
          className={`md:hidden fixed inset-0 z-40 bg-black/60 backdrop-blur-sm transition-opacity duration-200 ${
            mobileOpen ? "opacity-100 pointer-events-auto" : "opacity-0 pointer-events-none"
          }`}
        />
      )}
      <aside
        className={`border-r border-[color:var(--rule)] bg-[color:var(--color-raised)]/95 md:bg-[color:var(--color-raised)]/40 flex flex-col
          ${onClose
            ? `fixed md:static top-0 bottom-0 left-0 z-50 w-[min(320px,88vw)] md:w-[280px] transition-transform duration-250 ease-out md:translate-x-0 ${
                mobileOpen ? "translate-x-0" : "-translate-x-full"
              }`
            : "w-[280px]"}`}
      >
        <div className="flex items-start justify-between px-5 pt-5 pb-3">
          <div>
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
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              aria-label="Close conversations"
              className="md:hidden -mr-2 min-h-11 min-w-11 inline-flex items-center justify-center text-[color:var(--color-ink-3)] hover:text-[color:var(--color-ink)] transition-colors"
            >
              ✕
            </button>
          )}
        </div>
        <button
          type="button"
          onClick={handleNewAndClose}
          className="mx-5 mb-3 px-4 py-2 text-[11px] uppercase tracking-[0.18em] border border-[color:var(--color-gold)] text-[color:var(--color-gold)] hover:bg-[color:var(--color-gold)] hover:text-[color:var(--color-page)] transition-colors rounded-[3px] min-h-11"
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
                    onClick={() => handleSelectAndClose(c.id)}
                    className={`group w-full text-left px-3 py-2.5 rounded-[4px] transition-colors min-h-11 ${
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
                        className="opacity-60 md:opacity-0 md:group-hover:opacity-100 text-[color:var(--color-ink-4)] hover:text-[color:var(--color-bad)] text-xs transition-opacity px-1 cursor-pointer"
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
    </>
  );
}
