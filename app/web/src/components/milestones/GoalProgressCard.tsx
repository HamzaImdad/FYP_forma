// GoalProgressCard — one active-goal card with progress bar, milestone
// markers at 25/50/75/100, and linear-extrapolation projected completion.

import { useMemo } from "react";
import type { Goal } from "@/lib/plansApi";
import { plansApi } from "@/lib/plansApi";

type Props = {
  goal: Goal;
  onChanged?: () => void;
};

function prettyType(t: Goal["goal_type"]): string {
  return (
    {
      volume: "Volume",
      quality: "Quality",
      consistency: "Consistency",
      skill: "Skill",
      duration: "Duration",
      balance: "Balance",
    } as Record<string, string>
  )[t] ?? t;
}

function formatValue(v: number, unit: string): string {
  if (unit === "form_score") return `${Math.round(v)}`;
  if (unit === "seconds") return `${Math.round(v)}s`;
  return `${Math.round(v)}`;
}

function projectedCompletion(goal: Goal): string | null {
  // Linear extrapolation from created_at → now using current pace.
  // Only meaningful for cumulative / once-period goals.
  if (goal.period === "week" || goal.period === "month") {
    return null;
  }
  try {
    const start = new Date(goal.created_at).getTime();
    const now = Date.now();
    const days = Math.max(1, (now - start) / (1000 * 60 * 60 * 24));
    const rate = goal.current_value / days;
    if (rate <= 0) return null;
    const remaining = Math.max(0, goal.target_value - goal.current_value);
    const daysLeft = remaining / rate;
    if (!isFinite(daysLeft) || daysLeft > 365) return null;
    const eta = new Date(now + daysLeft * 24 * 60 * 60 * 1000);
    return `ETA ${eta.toLocaleDateString(undefined, { month: "short", day: "numeric" })}`;
  } catch {
    return null;
  }
}

export function GoalProgressCard({ goal, onChanged }: Props) {
  const pct = goal.target_value > 0 ? Math.min(1, goal.current_value / goal.target_value) : 0;
  const markers = useMemo(
    () =>
      goal.milestones
        .map((m) => ({
          pct: goal.target_value > 0 ? m.threshold_value / goal.target_value : 0,
          reached: m.reached,
          label: m.label,
        }))
        .filter((m) => m.pct <= 1.0001 && m.pct >= 0),
    [goal.milestones, goal.target_value],
  );

  const eta = projectedCompletion(goal);

  const handleDelete = async () => {
    if (!window.confirm(`Delete goal "${goal.title}"?`)) return;
    try {
      await plansApi.deleteGoal(goal.id);
      onChanged?.();
    } catch (e) {
      console.warn("deleteGoal failed", e);
    }
  };

  const handlePause = async () => {
    try {
      const next = goal.status === "paused" ? "active" : "paused";
      await plansApi.patchGoal(goal.id, { status: next });
      onChanged?.();
    } catch (e) {
      console.warn("patchGoal failed", e);
    }
  };

  return (
    <div className="border border-[color:var(--rule)] rounded-[4px] p-5 bg-[color:var(--color-raised)]/40">
      <div className="flex items-baseline justify-between gap-3 mb-1">
        <div className="min-w-0">
          <h3
            className="text-[color:var(--color-ink)] truncate"
            style={{ fontFamily: "var(--font-display)", fontSize: "1.4rem", letterSpacing: "0.04em" }}
          >
            {goal.title.toUpperCase()}
          </h3>
          <div className="text-[10px] uppercase tracking-[0.2em] text-[color:var(--color-ink-3)]">
            {prettyType(goal.goal_type)}
            {goal.exercise ? ` · ${goal.exercise.replace(/_/g, " ")}` : ""}
            {goal.period ? ` · ${goal.period}` : ""}
            {goal.status !== "active" ? ` · ${goal.status}` : ""}
          </div>
        </div>
        <div className="flex gap-2 shrink-0">
          <button
            type="button"
            onClick={handlePause}
            className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)] hover:text-[color:var(--color-ink)]"
          >
            {goal.status === "paused" ? "Resume" : "Pause"}
          </button>
          <button
            type="button"
            onClick={handleDelete}
            className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)] hover:text-[color:var(--color-bad)]"
          >
            Delete
          </button>
        </div>
      </div>

      <div className="flex items-baseline justify-between gap-3 mt-2 mb-2">
        <div
          className="text-[color:var(--color-gold)] tabular-nums"
          style={{ fontFamily: "var(--font-display)", fontSize: "1.8rem" }}
        >
          {formatValue(goal.current_value, goal.unit)}
          <span
            className="text-[color:var(--color-ink-3)] ml-2"
            style={{ fontSize: "0.9rem" }}
          >
            / {formatValue(goal.target_value, goal.unit)} {goal.unit}
          </span>
        </div>
        {eta && (
          <div className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)]">
            {eta}
          </div>
        )}
      </div>

      <div className="relative h-[6px] bg-[color:var(--color-ink)]/10 rounded-full overflow-visible">
        <div
          className="absolute inset-y-0 left-0 bg-[color:var(--color-gold)] rounded-full"
          style={{ width: `${pct * 100}%` }}
        />
        {markers.map((m, i) => (
          <div
            key={i}
            className={`absolute top-1/2 -translate-y-1/2 w-[10px] h-[10px] rounded-full border-2 ${
              m.reached
                ? "bg-[color:var(--color-gold)] border-[color:var(--color-gold)]"
                : "bg-[color:var(--color-page)] border-[color:var(--color-ink-3)]"
            }`}
            style={{ left: `calc(${m.pct * 100}% - 5px)` }}
            title={m.label}
          />
        ))}
      </div>

      {goal.description && (
        <p
          className="italic text-[color:var(--color-ink-3)] text-[13px] mt-3"
          style={{ fontFamily: "var(--font-serif)" }}
        >
          {goal.description}
        </p>
      )}
    </div>
  );
}
