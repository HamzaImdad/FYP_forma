// PlanShell — /plans page layout: split screen with the plan-creator
// chatbot on the left and live preview + active plan on the right.

import { useCallback, useEffect, useState } from "react";
import { plansApi, type PlanDraft } from "@/lib/plansApi";
import { PlanCreatorChat } from "./PlanCreatorChat";
import { PlanPreviewCard } from "./PlanPreviewCard";
import { ActivePlanCard } from "./ActivePlanCard";
import { GoalTemplatePicker } from "./GoalTemplatePicker";
import { useGoalsUpdated } from "@/hooks/useGoalsUpdated";

export function PlanShell() {
  const [draft, setDraft] = useState<PlanDraft | null>(null);
  const [draftLoading, setDraftLoading] = useState(false);
  const [activeReloadKey, setActiveReloadKey] = useState(0);
  const [pickerOpen, setPickerOpen] = useState(false);

  const fetchDraft = useCallback(async (conversationId: number | null) => {
    if (conversationId == null) return;
    setDraftLoading(true);
    try {
      const next = await plansApi.getPlanDraft(conversationId);
      setDraft(next);
    } catch (e) {
      console.warn("plan draft fetch failed", e);
    } finally {
      setDraftLoading(false);
    }
  }, []);

  const handleTurnDone = useCallback(
    (conversationId: number | null) => {
      // Refetch the draft and bump the ActivePlanCard so a freshly saved
      // plan appears without a manual reload.
      void fetchDraft(conversationId);
      setActiveReloadKey((k) => k + 1);
    },
    [fetchDraft],
  );

  // Real-time sync: when the server recomputes goals (e.g. after a session),
  // bump the ActivePlanCard too so day-complete progress reflects.
  useGoalsUpdated(
    useCallback(() => {
      setActiveReloadKey((k) => k + 1);
    }, []),
  );

  useEffect(() => {
    setDraft(null);
  }, []);

  return (
    <div className="min-h-screen bg-[color:var(--color-page)] pt-[72px]">
      <div className="max-w-[1440px] mx-auto px-6 md:px-10 py-10">
        <header className="mb-8 flex items-end justify-between gap-6 flex-wrap">
          <div>
            <h1
              className="text-[color:var(--color-ink)]"
              style={{
                fontFamily: "var(--font-display)",
                fontSize: "clamp(2.6rem, 5vw, 4rem)",
                letterSpacing: "0.04em",
                lineHeight: 1,
              }}
            >
              PLANS
            </h1>
            <p
              className="italic text-[color:var(--color-ink-3)] mt-2"
              style={{ fontFamily: "var(--font-serif)", fontSize: "1.2rem" }}
            >
              An adaptive workout plan, built around your real baseline.
            </p>
          </div>
          <button
            type="button"
            onClick={() => setPickerOpen(true)}
            className="text-[11px] uppercase tracking-[0.22em] px-5 py-3 border border-[color:var(--color-gold)] text-[color:var(--color-gold)] hover:bg-[color:var(--color-gold)] hover:text-[color:var(--color-page)] transition-colors rounded-[3px]"
          >
            + Quick goal
          </button>
        </header>

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          {/* Left: chat architect */}
          <div className="border border-[color:var(--rule)] rounded-[4px] bg-[color:var(--color-page)] h-[720px] min-h-0 overflow-hidden flex flex-col">
            <PlanCreatorChat onTurnDone={handleTurnDone} />
          </div>

          {/* Right: preview + active plan */}
          <div className="flex flex-col gap-6 min-w-0">
            <PlanPreviewCard draft={draft} loading={draftLoading} />
            <ActivePlanCard reloadKey={activeReloadKey} />
          </div>
        </div>
      </div>

      <GoalTemplatePicker
        open={pickerOpen}
        onClose={() => setPickerOpen(false)}
        onCreated={() => setActiveReloadKey((k) => k + 1)}
      />
    </div>
  );
}
