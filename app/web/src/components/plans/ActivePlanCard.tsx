// ActivePlanCard — shows the user's current active plan with progress
// and clickable PlanDayTiles.

import { useEffect, useState } from "react";
import { plansApi, type Plan } from "@/lib/plansApi";
import { PlanDayTile } from "./PlanDayTile";

type Props = {
  reloadKey?: number;
};

export function ActivePlanCard({ reloadKey = 0 }: Props) {
  const [plan, setPlan] = useState<Plan | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    plansApi
      .listPlans()
      .then(async (plans) => {
        if (cancelled) return;
        const active = plans.find((p) => p.status === "active") ?? plans[0] ?? null;
        if (!active) {
          setPlan(null);
          return;
        }
        const full = await plansApi.getPlan(active.id);
        if (cancelled) return;
        setPlan(full);
      })
      .catch((e) => {
        if (!cancelled) setError(e?.message ?? "Failed to load plans");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [reloadKey]);

  const handleComplete = async (planDayId: number) => {
    if (!plan) return;
    try {
      const refreshed = await plansApi.completePlanDay(plan.id, planDayId);
      setPlan(refreshed);
    } catch (e) {
      console.warn("completePlanDay failed", e);
    }
  };

  const handleDelete = async () => {
    if (!plan) return;
    if (!window.confirm(`Delete plan "${plan.title}"?`)) return;
    try {
      await plansApi.deletePlan(plan.id);
      setPlan(null);
    } catch (e) {
      console.warn("deletePlan failed", e);
    }
  };

  if (loading) {
    return (
      <div className="border border-[color:var(--rule)] rounded-[4px] p-6 bg-[color:var(--color-raised)]/40 text-[color:var(--color-ink-3)] italic text-sm">
        Loading your plan…
      </div>
    );
  }

  if (error) {
    return (
      <div className="border border-[color:var(--color-bad)]/30 rounded-[4px] p-6 bg-[color:var(--color-bad)]/5 text-[color:var(--color-bad)] text-sm">
        {error}
      </div>
    );
  }

  if (!plan) {
    return (
      <div className="border border-[color:var(--rule)] rounded-[4px] p-6 bg-[color:var(--color-raised)]/40">
        <div
          className="text-[color:var(--color-ink)] mb-1"
          style={{ fontFamily: "var(--font-display)", fontSize: "1.4rem", letterSpacing: "0.04em" }}
        >
          NO ACTIVE PLAN
        </div>
        <p
          className="italic text-[color:var(--color-ink-3)]"
          style={{ fontFamily: "var(--font-serif)" }}
        >
          Build one with the architect on the left.
        </p>
      </div>
    );
  }

  const days = plan.days ?? [];
  const totalActive = days.filter((d) => !d.is_rest).length;
  const done = days.filter((d) => d.completed).length;
  const pct = totalActive > 0 ? Math.min(1, done / totalActive) : 0;

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
            {plan.title}
          </h3>
          <button
            type="button"
            onClick={handleDelete}
            className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)] hover:text-[color:var(--color-bad)]"
          >
            Delete
          </button>
        </div>
        {plan.summary && (
          <p
            className="italic text-[color:var(--color-ink-3)] text-[13px] mt-1"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            {plan.summary}
          </p>
        )}
        <div className="mt-3">
          <div className="flex justify-between text-[10px] uppercase tracking-[0.16em] text-[color:var(--color-ink-3)] mb-1">
            <span>
              {done} / {totalActive} done
            </span>
            <span>{Math.round(pct * 100)}%</span>
          </div>
          <div className="h-[3px] bg-[color:var(--color-ink)]/10 overflow-hidden rounded-full">
            <div
              className="h-full bg-[color:var(--color-gold)]"
              style={{ width: `${pct * 100}%` }}
            />
          </div>
        </div>
      </div>

      <div className="p-4 grid grid-cols-1 sm:grid-cols-2 gap-3 max-h-[540px] overflow-auto">
        {days.map((d) => (
          <PlanDayTile
            key={d.id}
            day={d}
            planId={plan.id}
            completable
            onComplete={handleComplete}
            compact
          />
        ))}
      </div>
    </div>
  );
}
