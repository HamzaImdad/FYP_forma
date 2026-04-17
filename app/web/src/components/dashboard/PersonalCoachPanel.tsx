// Dashboard floating coach panel. Bottom-right gold pill opens a slide-out
// chat drawer reusing ChatShell in compact mode (no conversation sidebar).

import { useState } from "react";
import { ChatShell } from "@/components/chat/ChatShell";

export function PersonalCoachPanel() {
  const [open, setOpen] = useState(false);

  return (
    <>
      {!open && (
        <button
          type="button"
          onClick={() => setOpen(true)}
          aria-label="Open coach"
          className="fixed bottom-4 right-4 sm:bottom-6 sm:right-6 z-[60] min-h-11 px-4 sm:px-5 py-3 bg-[color:var(--color-gold)] text-[color:var(--color-page)] text-[10px] sm:text-[11px] uppercase tracking-[0.18em] sm:tracking-[0.22em] font-medium rounded-full shadow-[0_6px_24px_rgba(174,231,16,0.35)] hover:bg-[color:var(--color-gold-hover)] transition-colors"
        >
          Ask your coach
        </button>
      )}

      {open && (
        <div
          className="fixed inset-0 z-[55] bg-black/30"
          onClick={() => setOpen(false)}
          aria-hidden="true"
        />
      )}

      <aside
        className={`fixed top-0 right-0 z-[56] h-full w-full sm:max-w-[480px] bg-[color:var(--color-page)] border-l border-[color:var(--rule)] shadow-[-8px_0_32px_rgba(0,0,0,0.12)] transition-transform duration-300 ease-out flex flex-col ${
          open ? "translate-x-0" : "translate-x-full pointer-events-none"
        }`}
      >
        <header className="px-6 py-5 border-b border-[color:var(--rule)] flex items-start justify-between gap-4">
          <div>
            <div className="text-[10px] uppercase tracking-[0.24em] text-[color:var(--color-gold)]">
              Coach
            </div>
            <div
              className="text-[color:var(--color-ink)] mt-1"
              style={{ fontFamily: "var(--font-display)", fontSize: "1.5rem", letterSpacing: "0.03em" }}
            >
              ASK YOUR COACH
            </div>
          </div>
          <button
            type="button"
            onClick={() => setOpen(false)}
            aria-label="Close coach"
            className="shrink-0 w-9 h-9 flex items-center justify-center rounded-full border border-[color:var(--rule)] text-[color:var(--color-ink-2)] hover:text-[color:var(--color-ink)] hover:border-[color:var(--color-gold)] transition-colors"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <path d="M3 3L13 13M13 3L3 13" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
            </svg>
          </button>
        </header>
        <div className="flex-1 min-h-0 flex">
          <ChatShell
            mode="personal"
            authed
            showSidebar={false}
            title="COACH ON DECK"
            tagline="Fresh off the floor — ask anything."
          />
        </div>
      </aside>
    </>
  );
}
