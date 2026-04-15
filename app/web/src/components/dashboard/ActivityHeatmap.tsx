// GitHub-style activity heatmap — 12 weeks × 7 days. Intensity scales with
// reps count per day. Click a cell to filter the session history list.

import { useMemo } from "react";
import type { HeatmapCell } from "../../lib/dashboardApi";

type Props = {
  cells: HeatmapCell[];
  onPickDay: (date: string | null) => void;
  activeDate: string | null;
};

function toISODate(d: Date): string {
  const yr = d.getFullYear();
  const mo = String(d.getMonth() + 1).padStart(2, "0");
  const dy = String(d.getDate()).padStart(2, "0");
  return `${yr}-${mo}-${dy}`;
}

export function ActivityHeatmap({ cells, onPickDay, activeDate }: Props) {
  const { grid, max } = useMemo(() => {
    const lookup = new Map<string, HeatmapCell>();
    for (const c of cells) lookup.set(c.date, c);
    // 12 weeks × 7 days ending today
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const dayOfWeek = today.getDay(); // 0 = Sunday
    // Anchor Sunday of this week
    const anchor = new Date(today);
    anchor.setDate(today.getDate() - dayOfWeek);
    const weeks: { date: string; cell: HeatmapCell | null }[][] = [];
    let maxReps = 0;
    for (let w = 11; w >= 0; w--) {
      const col: { date: string; cell: HeatmapCell | null }[] = [];
      for (let d = 0; d < 7; d++) {
        const dt = new Date(anchor);
        dt.setDate(anchor.getDate() - w * 7 + d);
        const iso = toISODate(dt);
        const cell = lookup.get(iso) ?? null;
        if (cell && cell.reps_count > maxReps) maxReps = cell.reps_count;
        col.push({ date: iso, cell });
      }
      weeks.push(col);
    }
    return { grid: weeks, max: maxReps };
  }, [cells]);

  function intensityClass(cell: HeatmapCell | null): string {
    if (!cell || cell.reps_count === 0) return "bg-[color:var(--color-sunken)]";
    if (max === 0) return "bg-[color:var(--color-sunken)]";
    const ratio = cell.reps_count / max;
    if (ratio > 0.75) return "bg-[color:var(--color-gold)]";
    if (ratio > 0.5) return "bg-[color:var(--color-gold-soft)]";
    if (ratio > 0.25) return "bg-[color:var(--color-gold-soft)]/70";
    return "bg-[color:var(--color-gold-soft)]/40";
  }

  return (
    <section className="bg-[color:var(--color-raised)]/60 border border-[color:var(--rule)] rounded-sm px-8 py-8">
      <div className="flex items-baseline justify-between mb-5">
        <h2
          className="text-[color:var(--color-ink)]"
          style={{ fontFamily: "var(--font-display)", fontSize: "1.6rem", letterSpacing: "0.06em" }}
        >
          Activity · 12 weeks
        </h2>
        {activeDate && (
          <button
            onClick={() => onPickDay(null)}
            className="text-[10px] uppercase tracking-[0.24em] text-[color:var(--color-gold)]"
          >
            Clear filter · {activeDate}
          </button>
        )}
      </div>
      <div className="flex gap-[3px] overflow-x-auto pb-1">
        {grid.map((col, ci) => (
          <div key={ci} className="flex flex-col gap-[3px]">
            {col.map(({ date, cell }) => {
              const isActive = activeDate === date;
              const interactive = cell !== null;
              return (
                <button
                  key={date}
                  disabled={!interactive}
                  onClick={() => onPickDay(isActive ? null : date)}
                  title={
                    cell
                      ? `${date} · ${cell.reps_count} reps · ${cell.session_count} session${cell.session_count === 1 ? "" : "s"}`
                      : date
                  }
                  className={`w-4 h-4 rounded-[2px] transition-all ${intensityClass(cell)} ${
                    isActive ? "ring-2 ring-[color:var(--color-orange)]" : ""
                  } ${interactive ? "cursor-pointer hover:brightness-110" : "cursor-default"}`}
                />
              );
            })}
          </div>
        ))}
      </div>
    </section>
  );
}
