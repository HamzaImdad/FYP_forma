// PlanPreviewCard — live preview of the in-flight plan draft from the
// plan-creator chatbot. Re-renders whenever PlanShell refetches the draft.

import type { PlanDraft } from "@/lib/plansApi";
import { PlanDayTile } from "./PlanDayTile";

type Props = {
  draft: PlanDraft | null;
  loading?: boolean;
};

export function PlanPreviewCard({ draft, loading }: Props) {
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
              day_date: d.day_date,
              is_rest: d.is_rest,
              exercises: d.exercises,
              completed: false,
            }}
            compact
          />
        ))}
      </div>
    </div>
  );
}
