// PlanShell — /plans page layout.
//
// Session 5 rebuild: the left column now switches between the Plan Architect
// chatbot and the Custom Exercise Plan builder. The right column keeps the
// live preview card on top and replaces the single "Active plan" card with
// a PlanHistoryTabs block so the user can browse Active / Past / Archived
// plans without leaving the page.

import { useCallback, useRef, useState } from "react";
import { plansApi, type PlanDraft } from "@/lib/plansApi";
import { PlanCreatorChat } from "./PlanCreatorChat";
import { CustomPlanBuilder } from "./CustomPlanBuilder";
import { PlanPreviewCard } from "./PlanPreviewCard";
import { ActivePlanCard } from "./ActivePlanCard";
import { PlanHistoryTabs } from "./PlanHistoryTabs";
import { GoalTemplatePicker } from "./GoalTemplatePicker";
import { useGoalsUpdated } from "@/hooks/useGoalsUpdated";
import { usePlanSaved } from "@/hooks/usePlanSaved";
import { usePlanDayCompleted } from "@/hooks/usePlanDayCompleted";

type LeftTab = "architect" | "custom";

export function PlanShell() {
  const [draft, setDraft] = useState<PlanDraft | null>(null);
  const [draftLoading, setDraftLoading] = useState(false);
  const [activeReloadKey, setActiveReloadKey] = useState(0);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [leftTab, setLeftTab] = useState<LeftTab>("architect");
  const [approving, setApproving] = useState(false);

  // Exposed by ChatShell — lets us dispatch an "approve and save" turn
  // directly from the preview card.
  const chatSendRef = useRef<
    | { send: (text: string) => void; pending: boolean }
    | null
  >(null);

  const fetchDraft = useCallback(async (conversationId: number | null) => {
    if (conversationId == null) return;
    setDraftLoading(true);
    try {
      const next = await plansApi.getPlanDraft(conversationId);
      if (next) {
        setDraft({ ...next, source: "chat" });
      } else {
        setDraft(null);
      }
    } catch (e) {
      console.warn("plan draft fetch failed", e);
    } finally {
      setDraftLoading(false);
    }
  }, []);

  const handleTurnDone = useCallback(
    (conversationId: number | null) => {
      void fetchDraft(conversationId);
      setActiveReloadKey((k) => k + 1);
      // Clear approving state as soon as the assistant turn finishes, so the
      // preview button unlocks even if save_plan quietly succeeded.
      setApproving(false);
    },
    [fetchDraft],
  );

  const handleCustomDraftChange = useCallback((next: PlanDraft | null) => {
    setDraft(next);
  }, []);

  const bumpReload = useCallback(() => {
    setActiveReloadKey((k) => k + 1);
  }, []);

  const handleCustomSaved = useCallback(() => {
    setDraft(null);
    bumpReload();
  }, [bumpReload]);

  // Real-time sync: session completion or any other plan change.
  useGoalsUpdated(
    useCallback(() => {
      setActiveReloadKey((k) => k + 1);
    }, []),
  );

  usePlanSaved(
    useCallback(() => {
      setActiveReloadKey((k) => k + 1);
      setDraft(null);
      // Switch back to the architect tab so the newly-saved plan takes focus.
      setLeftTab("architect");
    }, []),
  );

  // Session-5: plan day auto-complete → refetch everything so Active tile
  // flips to "done ✓" and Past/Archived tabs pick up plan.status=completed.
  usePlanDayCompleted(
    useCallback(() => {
      setActiveReloadKey((k) => k + 1);
    }, []),
  );

  const handleApproveChat = useCallback(() => {
    const handle = chatSendRef.current;
    if (!handle || handle.pending) return;
    setApproving(true);
    handle.send("Approve and save this plan.");
  }, []);

  const leftTabCls = (t: LeftTab) =>
    `px-4 py-2 text-[10px] uppercase tracking-[0.2em] border-b-2 transition-colors ${
      leftTab === t
        ? "border-[color:var(--color-gold)] text-[color:var(--color-ink)]"
        : "border-transparent text-[color:var(--color-ink-3)] hover:text-[color:var(--color-ink)]"
    }`;

  return (
    <div className="min-h-screen bg-[color:var(--color-page)] pt-[64px] sm:pt-[72px]">
      <div className="max-w-[1440px] mx-auto px-4 sm:px-6 md:px-10 py-6 md:py-10">
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
          {/* Left: architect / custom builder tabs */}
          <div className="border border-[color:var(--rule)] rounded-[4px] bg-[color:var(--color-page)] h-[720px] min-h-0 overflow-hidden flex flex-col">
            <div className="flex border-b border-[color:var(--rule)]">
              <button
                type="button"
                className={leftTabCls("architect")}
                onClick={() => setLeftTab("architect")}
              >
                Plan Architect
              </button>
              <button
                type="button"
                className={leftTabCls("custom")}
                onClick={() => setLeftTab("custom")}
              >
                Custom Builder
              </button>
            </div>
            <div className="flex-1 min-h-0 overflow-hidden">
              {leftTab === "architect" ? (
                <PlanCreatorChat
                  onTurnDone={handleTurnDone}
                  sendRef={chatSendRef}
                />
              ) : (
                <CustomPlanBuilder
                  onDraftChange={handleCustomDraftChange}
                  onSaved={handleCustomSaved}
                />
              )}
            </div>
          </div>

          {/* Right: preview + active plan + history tabs */}
          <div className="flex flex-col gap-6 min-w-0">
            <PlanPreviewCard
              draft={draft}
              loading={draftLoading}
              onApproveChat={handleApproveChat}
              approving={approving}
            />
            <ActivePlanCard reloadKey={activeReloadKey} onChanged={bumpReload} />
            <PlanHistoryTabs reloadKey={activeReloadKey} onChanged={bumpReload} />
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
