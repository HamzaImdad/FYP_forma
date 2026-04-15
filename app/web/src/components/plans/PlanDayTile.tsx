// PlanDayTile — one calendar cell inside a plan preview or active plan.
// Shows date, exercises (or REST), and a mark-complete button for past days.

import { Link } from "react-router-dom";
import type { PlanDay, PlanDayExercise } from "@/lib/plansApi";

type Props = {
  day: Pick<PlanDay, "day_date" | "is_rest" | "exercises" | "completed" | "id">;
  planId?: number;
  completable?: boolean;
  onComplete?: (planDayId: number) => void;
  compact?: boolean;
};

function prettyDate(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, {
      weekday: "short",
      month: "short",
      day: "numeric",
    });
  } catch {
    return iso;
  }
}

export function PlanDayTile({
  day,
  planId,
  completable = false,
  onComplete,
  compact = false,
}: Props) {
  const pretty = prettyDate(day.day_date);
  const primary = (day.exercises?.[0] as PlanDayExercise | undefined) ?? null;

  return (
    <div
      className={`border border-[color:var(--rule)] rounded-[4px] p-3 bg-[color:var(--color-raised)]/40 ${
        day.completed ? "opacity-70" : ""
      } ${compact ? "text-[12px]" : "text-[13px]"}`}
    >
      <div className="flex items-baseline justify-between gap-2 mb-2">
        <div
          className="text-[10px] uppercase tracking-[0.2em] text-[color:var(--color-ink-3)]"
          style={{ fontFamily: "var(--font-display)" }}
        >
          {pretty}
        </div>
        {day.completed && (
          <div className="text-[10px] tracking-[0.16em] text-[color:var(--color-good)] uppercase">
            done
          </div>
        )}
      </div>

      {day.is_rest ? (
        <div
          className="italic text-[color:var(--color-ink-3)]"
          style={{ fontFamily: "var(--font-serif)" }}
        >
          Rest day
        </div>
      ) : (day.exercises ?? []).length === 0 ? (
        <div className="text-[color:var(--color-ink-3)] italic">No work</div>
      ) : (
        <ul className="space-y-1">
          {(day.exercises ?? []).map((e, i) => (
            <li
              key={`${e.exercise}-${i}`}
              className="flex items-baseline justify-between gap-2"
            >
              <span className="text-[color:var(--color-ink)] capitalize">
                {e.exercise.replace(/_/g, " ")}
              </span>
              <span
                className="text-[color:var(--color-gold)] tabular-nums"
                style={{ fontFamily: "var(--font-display)" }}
              >
                {e.target_sets}×{e.target_reps}
              </span>
            </li>
          ))}
        </ul>
      )}

      {!day.is_rest && primary && planId != null && (
        <div className="mt-3 flex flex-wrap gap-2">
          <Link
            to={`/workout/${primary.exercise}?plan_day_id=${day.id}`}
            className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-gold)] hover:text-[color:var(--color-gold-hover)]"
          >
            Start →
          </Link>
          {completable && !day.completed && onComplete && (
            <button
              type="button"
              onClick={() => onComplete(day.id)}
              className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)] hover:text-[color:var(--color-ink)]"
            >
              Mark done
            </button>
          )}
        </div>
      )}
    </div>
  );
}
