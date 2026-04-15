// ActiveGoalsCard — compact top-3 active goals for the dashboard header.
// Refetches on goals_updated so it stays in sync with session-end.

import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { plansApi, type Goal } from "@/lib/plansApi";
import { useGoalsUpdated } from "@/hooks/useGoalsUpdated";
import { GoalTemplatePicker } from "@/components/plans/GoalTemplatePicker";

export function ActiveGoalsCard() {
  const [goals, setGoals] = useState<Goal[]>([]);
  const [loading, setLoading] = useState(true);
  const [pickerOpen, setPickerOpen] = useState(false);

  const fetchGoals = useCallback(() => {
    plansApi
      .listGoals("active")
      .then((gs) => {
        setGoals(gs.slice(0, 3));
      })
      .catch(() => setGoals([]))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchGoals();
  }, [fetchGoals]);

  useGoalsUpdated(useCallback(() => fetchGoals(), [fetchGoals]));

  return (
    <div className="border border-[color:var(--rule)] rounded-[4px] p-5 bg-[color:var(--color-raised)]/40">
      <div className="flex items-baseline justify-between gap-3 mb-3">
        <h3
          className="text-[color:var(--color-ink-3)] uppercase tracking-[0.22em]"
          style={{ fontFamily: "var(--font-display)", fontSize: "0.85rem" }}
        >
          Active goals
        </h3>
        <div className="flex gap-3">
          <button
            type="button"
            onClick={() => setPickerOpen(true)}
            className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-gold)] hover:text-[color:var(--color-gold-hover)]"
          >
            + Add
          </button>
          <Link
            to="/milestones"
            className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)] hover:text-[color:var(--color-ink)]"
          >
            All →
          </Link>
        </div>
      </div>

      {loading ? (
        <div className="italic text-[color:var(--color-ink-3)] text-[13px]">Loading…</div>
      ) : goals.length === 0 ? (
        <p
          className="italic text-[color:var(--color-ink-3)] text-[13px]"
          style={{ fontFamily: "var(--font-serif)" }}
        >
          No active goals — pick a template to get started.
        </p>
      ) : (
        <ul className="space-y-3">
          {goals.map((g) => {
            const pct = g.target_value > 0 ? Math.min(1, g.current_value / g.target_value) : 0;
            return (
              <li key={g.id}>
                <div className="flex items-baseline justify-between gap-3 mb-1">
                  <span
                    className="text-[color:var(--color-ink)] truncate text-[14px]"
                    style={{ fontFamily: "var(--font-display)", letterSpacing: "0.04em" }}
                  >
                    {g.title.toUpperCase()}
                  </span>
                  <span
                    className="text-[color:var(--color-gold)] tabular-nums text-[12px]"
                    style={{ fontFamily: "var(--font-display)" }}
                  >
                    {Math.round(g.current_value)}/{Math.round(g.target_value)}
                  </span>
                </div>
                <div className="h-[3px] bg-[color:var(--color-ink)]/10 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[color:var(--color-gold)]"
                    style={{ width: `${pct * 100}%` }}
                  />
                </div>
              </li>
            );
          })}
        </ul>
      )}

      <GoalTemplatePicker
        open={pickerOpen}
        onClose={() => setPickerOpen(false)}
        onCreated={fetchGoals}
      />
    </div>
  );
}
