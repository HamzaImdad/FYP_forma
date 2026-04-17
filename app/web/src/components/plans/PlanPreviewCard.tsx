// PlanPreviewCard — live preview of the in-flight plan draft from either
// the Plan Architect chatbot or the Custom Plan Builder form.
// Session-5 adds an explicit "Approve & Save" CTA so the user no longer
// has to type "yes, save it" to the LLM.

import { useState } from "react";
import type { PlanDraft } from "@/lib/plansApi";
import { PlanDayTile } from "./PlanDayTile";

type Props = {
  draft: PlanDraft | null;
  loading?: boolean;
  // When draft.source === "chat", this dispatches a user turn back to the
  // chat so the LLM calls save_plan. When draft.source === "custom", the
  // parent already handled the save — this prop is unused.
  onApproveChat?: () => void;
  approving?: boolean;
};

export function PlanPreviewCard({ draft, loading, onApproveChat, approving }: Props) {
  const [localApproving, setLocalApproving] = useState(false);
  if (!draft) {
    return (
      <div className="border border-[color:var(--rule)] rounded-[4px] p-6 bg-[color:var(--color-raised)]/40 min-h-[260px] flex items-center justify-center">
        <p
          className="italic text-[color:var(--color-ink-3)] text-center max-w-xs"
          style={{ fontFamily: "var(--font-serif)" }}
        >
          {loading
            ? "Drafting your plan…"
            : "Tell the architect about your schedule and a live preview will appear here."}
        </p>
      </div>
    );
  }

  const activeCount = draft.days.filter((d) => !d.is_rest).length;
  const restCount = draft.days.length - activeCount;

  return (
    <div className="border border-[color:var(--rule)] rounded-[4px] bg-[color:var(--color-raised)]/40">
      <div className="px-5 pt-5 pb-3 border-b border-[color:var(--rule)]">
        <div className="flex items-baseline justify-between gap-3">
          <h3
            className="text-[color:var(--color-ink)] truncate"
            style={{
              fontFamily: "var(--font-display)",
              fontSize: "1.6rem",
              letterSpacing: "0.04em",
            }}
          >
            {draft.title || "WORKOUT PLAN"}
          </h3>
          <div className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)]">
            draft
          </div>
        </div>
        {draft.summary && (
          <p
            className="italic text-[color:var(--color-ink-3)] text-[13px] mt-1"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            {draft.summary}
          </p>
        )}
        <div className="mt-3 text-[11px] uppercase tracking-[0.16em] text-[color:var(--color-ink-3)]">
          {draft.days.length} days · {activeCount} active · {restCount} rest
        </div>
      </div>

      <div className="p-4 grid grid-cols-1 sm:grid-cols-2 gap-3 max-h-[540px] overflow-auto">
        {draft.days.map((d, i) => (
          <PlanDayTile
            key={`${d.day_date}-${i}`}
            day={{
              id: 0,
              plan_id: 0,
              day_date: d.day_date,
              is_rest: d.is_rest,
              exercises: d.exercises,
              completed: false,
              completed_at: null,
            }}
            compact
          />
        ))}
      </div>

      {draft.source === "chat" && onApproveChat && (
        <div className="px-5 py-3 border-t border-[color:var(--rule)] flex justify-end">
          <button
            type="button"
            disabled={approving || localApproving}
            onClick={() => {
              setLocalApproving(true);
              try {
                onApproveChat();
              } finally {
                // parent resets draft state; keep the visual disabled just
                // long enough to avoid a double-click.
                window.setTimeout(() => setLocalApproving(false), 800);
              }
            }}
            className="text-[11px] uppercase tracking-[0.22em] px-5 py-3 bg-[color:var(--color-ink)] text-[color:var(--color-page)] hover:bg-[color:var(--color-gold)] transition-colors rounded-[3px] disabled:opacity-50"
          >
            {approving || localApproving ? "Saving…" : "Approve & save"}
          </button>
        </div>
      )}
    </div>
  );
}
