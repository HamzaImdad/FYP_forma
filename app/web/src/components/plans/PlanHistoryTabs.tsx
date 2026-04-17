// PlanHistoryTabs — Active / Past / Archived tabbed list of the user's
// plans, below the ActivePlanCard on /plans. Active tab defers to the
// ActivePlanCard (rendered by PlanShell); Past + Archived tabs render
// compact PlanListCard rows.

import { useCallback, useEffect, useState } from "react";
import type { Plan } from "@/lib/plansApi";
import { plansApi } from "@/lib/plansApi";
import { PlanListCard } from "./PlanListCard";
import { useGoalsUpdated } from "@/hooks/useGoalsUpdated";
import { usePlanSaved } from "@/hooks/usePlanSaved";

type Tab = "active" | "past" | "archived";

type Props = {
  reloadKey: number;
  onChanged: () => void;
};

export function PlanHistoryTabs({ reloadKey, onChanged }: Props) {
  const [tab, setTab] = useState<Tab>("active");
  const [plans, setPlans] = useState<Plan[]>([]);
  const [loading, setLoading] = useState(true);

  const refetch = useCallback(async () => {
    setLoading(true);
    try {
      const list = await plansApi.listPlans();
      setPlans(list);
    } catch (e) {
      console.warn("listPlans failed", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refetch();
  }, [refetch, reloadKey]);

  useGoalsUpdated(useCallback(() => void refetch(), [refetch]));
  usePlanSaved(useCallback(() => void refetch(), [refetch]));

  const byTab: Record<Tab, Plan[]> = {
    active: plans.filter((p) => p.status === "active"),
    past: plans.filter((p) => p.status === "completed"),
    archived: plans.filter((p) => p.status === "archived" || p.status === "paused"),
  };

  const tabCls = (t: Tab) =>
    `px-4 py-2 text-[10px] uppercase tracking-[0.2em] border-b-2 transition-colors ${
      tab === t
        ? "border-[color:var(--color-gold)] text-[color:var(--color-ink)]"
        : "border-transparent text-[color:var(--color-ink-3)] hover:text-[color:var(--color-ink)]"
    }`;

  const handleChanged = useCallback(() => {
    void refetch();
    onChanged();
  }, [refetch, onChanged]);

  return (
    <div className="border border-[color:var(--rule)] rounded-[4px] bg-[color:var(--color-raised)]/40">
      <div className="flex border-b border-[color:var(--rule)]">
        <button type="button" className={tabCls("active")} onClick={() => setTab("active")}>
          Active · {byTab.active.length}
        </button>
        <button type="button" className={tabCls("past")} onClick={() => setTab("past")}>
          Past · {byTab.past.length}
        </button>
        <button type="button" className={tabCls("archived")} onClick={() => setTab("archived")}>
          Archived · {byTab.archived.length}
        </button>
      </div>

      <div className="p-4 space-y-2 max-h-[360px] overflow-auto">
        {loading && (
          <div className="text-[color:var(--color-ink-3)] italic text-sm">Loading…</div>
        )}

        {!loading && tab === "active" && byTab.active.length === 0 && (
          <p
            className="italic text-[color:var(--color-ink-3)] text-[13px]"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            No active plan yet. Build one with the architect or the custom builder on the left.
          </p>
        )}
        {!loading && tab === "past" && byTab.past.length === 0 && (
          <p
            className="italic text-[color:var(--color-ink-3)] text-[13px]"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            No completed plans yet — finish every day of an active plan to see it here.
          </p>
        )}
        {!loading && tab === "archived" && byTab.archived.length === 0 && (
          <p
            className="italic text-[color:var(--color-ink-3)] text-[13px]"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            Nothing archived. Pause or archive a plan to keep it here for later.
          </p>
        )}

        {!loading &&
          byTab[tab].map((p) => (
            <PlanListCard key={p.id} plan={p} onChanged={handleChanged} />
          ))}
      </div>
    </div>
  );
}
