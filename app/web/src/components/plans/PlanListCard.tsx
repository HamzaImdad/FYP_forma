// PlanListCard — compact row for the PlanHistoryTabs list. Shows title,
// date range, adherence % (completed non-rest days / total non-rest days),
// and status-contextual actions (unarchive, delete, activate).

import { useState } from "react";
import type { Plan, PlanStatus } from "@/lib/plansApi";
import { plansApi, describePlanExercise } from "@/lib/plansApi";
import { EditPlanModal } from "./EditPlanModal";

type Props = {
  plan: Plan;                  // Plan (may have .days attached or not)
  onChanged: () => void;       // parent re-fetch
};

function fmtDate(iso: string | null): string {
  if (!iso) return "";
  try {
    const d = new Date(`${iso}T00:00:00`);
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  } catch {
    return iso;
  }
}

function adherence(plan: Plan): { done: number; total: number; pct: number } {
  const days = plan.days ?? [];
  const active = days.filter((d) => !d.is_rest);
  const total = active.length;
  const done = active.filter((d) => d.completed).length;
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  return { done, total, pct };
}

export function PlanListCard({ plan, onChanged }: Props) {
  const [busy, setBusy] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [editOpen, setEditOpen] = useState(false);
  const [fullPlan, setFullPlan] = useState<Plan | null>(
    plan.days && plan.days.length > 0 ? plan : null,
  );

  const { done, total, pct } = adherence(fullPlan ?? plan);

  const setStatus = async (status: PlanStatus) => {
    setBusy(true);
    try {
      await plansApi.updatePlanStatus(plan.id, status);
      onChanged();
    } catch (e) {
      console.warn("updatePlanStatus failed", e);
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async () => {
    if (!window.confirm(`Delete plan "${plan.title}"? This can't be undone.`)) return;
    setBusy(true);
    try {
      await plansApi.deletePlan(plan.id);
      onChanged();
    } catch (e) {
      console.warn("deletePlan failed", e);
    } finally {
      setBusy(false);
    }
  };

  const handleToggle = async () => {
    if (expanded) {
      setExpanded(false);
      return;
    }
    if (!fullPlan || !fullPlan.days) {
      try {
        const full = await plansApi.getPlan(plan.id);
        setFullPlan(full);
      } catch (e) {
        console.warn("getPlan failed", e);
      }
    }
    setExpanded(true);
  };

  return (
    <div className="border border-[color:var(--rule)] rounded-[3px] bg-[color:var(--color-page)]">
      <button
        type="button"
        onClick={handleToggle}
        className="w-full text-left px-4 py-3 flex items-center justify-between gap-3 hover:bg-[color:var(--color-raised)]/40 transition-colors"
      >
        <div className="min-w-0 flex-1">
          <div
            className="text-[color:var(--color-ink)] text-[13px] uppercase tracking-[0.18em] truncate"
            style={{ fontFamily: "var(--font-display)" }}
          >
            {plan.title}
          </div>
          <div className="text-[10px] uppercase tracking-[0.16em] text-[color:var(--color-ink-3)] mt-0.5">
            {fmtDate(plan.start_date)} → {fmtDate(plan.end_date)} · {done}/{total} done · {pct}%
          </div>
        </div>
        <div className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)]">
          {expanded ? "Hide" : "View"}
        </div>
      </button>

      {expanded && fullPlan?.days && fullPlan.days.length > 0 && (
        <div className="px-4 pb-4 pt-1 space-y-1 max-h-[240px] overflow-auto border-t border-[color:var(--rule)]">
          {fullPlan.days.map((d) => (
            <div
              key={d.id}
              className="flex items-center justify-between text-[11px] text-[color:var(--color-ink-3)]"
            >
              <span className="truncate">
                {fmtDate(d.day_date)} ·{" "}
                {d.is_rest
                  ? "Rest"
                  : (d.exercises ?? [])
                      .map((e) => describePlanExercise(e))
                      .join(", ") || "—"}
              </span>
              <span className={d.completed ? "text-[color:var(--color-gold)]" : ""}>
                {d.completed ? "✓" : d.is_rest ? "·" : "○"}
              </span>
            </div>
          ))}
        </div>
      )}

      <div className="px-4 py-2 border-t border-[color:var(--rule)] flex gap-3 text-[10px] uppercase tracking-[0.18em]">
        <button
          type="button"
          onClick={() => setEditOpen(true)}
          disabled={busy}
          className="text-[color:var(--color-ink-3)] hover:text-[color:var(--color-ink)] disabled:opacity-50"
        >
          Edit
        </button>
        {plan.status === "archived" && (
          <button
            type="button"
            onClick={() => setStatus("active")}
            disabled={busy}
            className="text-[color:var(--color-gold)] hover:underline disabled:opacity-50"
          >
            Unarchive
          </button>
        )}
        {plan.status === "paused" && (
          <button
            type="button"
            onClick={() => setStatus("active")}
            disabled={busy}
            className="text-[color:var(--color-gold)] hover:underline disabled:opacity-50"
          >
            Reactivate
          </button>
        )}
        {plan.status === "completed" && (
          <span className="text-[color:var(--color-ink-3)]">Completed</span>
        )}
        <button
          type="button"
          onClick={handleDelete}
          disabled={busy}
          className="text-[color:var(--color-ink-3)] hover:text-[color:var(--color-bad)] ml-auto disabled:opacity-50"
        >
          Delete
        </button>
      </div>

      {editOpen && (
        <EditPlanModal
          plan={fullPlan ?? plan}
          onClose={() => setEditOpen(false)}
          onSaved={() => onChanged()}
        />
      )}
    </div>
  );
}
