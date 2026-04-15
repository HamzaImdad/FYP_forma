// Dashboard floating coach panel. Bottom-right gold pill opens a slide-out
// chat drawer reusing ChatShell in compact mode (no conversation sidebar).

import { useState } from "react";
import { ChatShell } from "@/components/chat/ChatShell";

export function PersonalCoachPanel() {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-label={open ? "Close coach" : "Open coach"}
        className="fixed bottom-6 right-6 z-[60] px-5 py-3 bg-[color:var(--color-gold)] text-[color:var(--color-page)] text-[11px] uppercase tracking-[0.22em] font-medium rounded-full shadow-[0_6px_24px_rgba(184,134,74,0.35)] hover:bg-[color:var(--color-gold-hover)] transition-colors"
      >
        {open ? "Close coach" : "Ask your coach"}
      </button>

      {open && (
        <div
          className="fixed inset-0 z-[55] bg-black/30"
          onClick={() => setOpen(false)}
          aria-hidden="true"
        />
      )}

      <aside
        className={`fixed top-0 right-0 z-[56] h-full w-full max-w-[480px] bg-[color:var(--color-page)] border-l border-[color:var(--rule)] shadow-[-8px_0_32px_rgba(0,0,0,0.12)] transition-transform duration-300 ease-out flex flex-col ${
          open ? "translate-x-0" : "translate-x-full pointer-events-none"
        }`}
      >
        <header className="px-6 py-5 border-b border-[color:var(--rule)]">
          <div className="text-[10px] uppercase tracking-[0.24em] text-[color:var(--color-gold)]">
            Coach
          </div>
          <div
            className="text-[color:var(--color-ink)] mt-1"
            style={{ fontFamily: "var(--font-display)", fontSize: "1.5rem", letterSpacing: "0.03em" }}
          >
            ASK YOUR COACH
          </div>
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
