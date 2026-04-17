// FORMA chatbot page — Session 3 deliverable.
//
// Logged-out: simplified guide chatbot (no sidebar, no history).
// Logged-in:  personal chatbot with conversation sidebar, tool-use, and
//             real-time session-completed hints.

import { useNavigate } from "react-router-dom";
import { ChatShell } from "@/components/chat/ChatShell";
import { GradientGlow } from "@/components/sections/GradientGlow";
import { useAuth } from "@/context/AuthContext";

export function ChatbotPage() {
  const { user, loading } = useAuth();
  const navigate = useNavigate();

  if (loading) {
    return (
      <div className="min-h-[70vh] flex items-center justify-center text-[color:var(--color-ink-4)] text-[11px] uppercase tracking-[0.24em]">
        Loading your coach…
      </div>
    );
  }

  const authed = !!user;
  const firstName = user?.display_name?.split(" ")[0];

  return (
    <div className="relative max-w-[1440px] mx-auto px-6 md:px-10 pt-[calc(var(--nav-height)+2rem)] pb-10 overflow-hidden">
      <GradientGlow position="top-right" intensity="medium" />
      <GradientGlow position="bottom-left" intensity="subtle" />
      <header className="relative mb-6">
        <div className="text-[10px] uppercase tracking-[0.24em] text-[color:var(--color-gold)]">
          {authed ? "Personal coach" : "Guide"}
        </div>
        <h1
          className="text-[color:var(--color-ink)] mt-2"
          style={{
            fontFamily: "var(--font-display)",
            fontSize: "var(--fs-h1)",
            lineHeight: 0.95,
            letterSpacing: "0.03em",
          }}
        >
          {authed && firstName ? `${firstName}'s coach` : "Ask FORMA"}
        </h1>
        <p
          className="text-[color:var(--color-ink-3)] italic mt-2"
          style={{ fontFamily: "var(--font-serif)", fontSize: "1.25rem" }}
        >
          {authed
            ? "Every rep in your history, one question away."
            : "Anything you want to know — no sign-up required."}
        </p>
      </header>

      <section
        className="border border-[color:var(--rule)] rounded-[6px] bg-[color:var(--color-page)] overflow-hidden flex flex-col"
        style={{ height: "calc(100vh - var(--nav-height) - 12rem)", minHeight: 560 }}
      >
        <ChatShell
          mode={authed ? "personal" : "guide"}
          authed={authed}
          showSidebar={authed}
          onSignupClick={() => navigate("/signup")}
        />
      </section>
    </div>
  );
}
