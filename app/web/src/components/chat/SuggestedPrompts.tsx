// Suggested prompts — chips that populate the input. Content is context-aware:
// unauthenticated visitors see marketing/education questions; logged-in users
// with no sessions see onboarding prompts; active users see data-grounded
// questions about their training. After a session ends, we swap in "how was
// my session?" style prompts.

import type { MouseEvent } from "react";

export type SuggestedContext =
  | "guest"
  | "no-sessions"
  | "has-data"
  | "post-session";

const PROMPTS: Record<SuggestedContext, { label: string; text: string }[]> = {
  guest: [
    { label: "What is FORMA?", text: "What is FORMA and how does it work?" },
    { label: "Exercises", text: "What exercises does FORMA support?" },
    { label: "Privacy", text: "Is my workout data kept private?" },
    { label: "Form scoring", text: "How does the form scoring work?" },
  ],
  "no-sessions": [
    { label: "Where do I start?", text: "I've never trained before — where should I start?" },
    { label: "Camera setup", text: "How do I set up my camera for the best form reading?" },
    { label: "Ten exercises", text: "What are the ten exercises FORMA tracks?" },
    { label: "First workout", text: "What's a good first workout for a beginner?" },
  ],
  "has-data": [
    { label: "This week", text: "How did I do this week compared to last week?" },
    { label: "Biggest weakness", text: "What's my biggest weakness right now?" },
    { label: "Last session", text: "Walk me through my last session." },
    { label: "Recovery", text: "Do I need a rest day?" },
  ],
  "post-session": [
    { label: "How was it?", text: "How was my last session?" },
    { label: "What to fix", text: "What should I fix before my next session?" },
    { label: "Compare", text: "How did that session compare to my last one?" },
    { label: "Next up", text: "What should I train next?" },
  ],
};

type Props = {
  context: SuggestedContext;
  onPick: (text: string) => void;
  disabled?: boolean;
};

export function SuggestedPrompts({ context, onPick, disabled }: Props) {
  const prompts = PROMPTS[context] ?? PROMPTS.guest;
  return (
    <div className="flex flex-wrap gap-2 px-6 md:px-10 pb-4 pt-2">
      {prompts.map((p) => (
        <button
          key={p.label}
          type="button"
          disabled={disabled}
          onClick={(e: MouseEvent<HTMLButtonElement>) => {
            e.preventDefault();
            onPick(p.text);
          }}
          className="text-[11px] uppercase tracking-[0.16em] px-3 py-2 border border-[color:var(--rule)] rounded-full hover:border-[color:var(--color-gold)] hover:text-[color:var(--color-gold)] text-[color:var(--color-ink-3)] transition-colors disabled:opacity-40"
        >
          {p.label}
        </button>
      ))}
    </div>
  );
}
