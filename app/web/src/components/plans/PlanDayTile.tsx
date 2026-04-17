// PlanDayTile — one calendar cell inside a plan preview or active plan.
// Shows date, exercises (or REST), and a mark-complete button for past days.

import { useState } from "react";
import { Link } from "react-router-dom";
import type { Plan, PlanDay, PlanDayExercise } from "@/lib/plansApi";
import { exerciseFamilyOf } from "@/lib/plansApi";
import { EditPlanDayModal } from "./EditPlanDayModal";

function formatTarget(e: PlanDayExercise): string {
  const fam = exerciseFamilyOf(e);
  if (fam === "weighted") {
    const w = (e as { target_weight_kg?: number }).target_weight_kg ?? 0;
    const reps = (e as { target_reps: number }).target_reps;
    const sets = e.target_sets;
    return w > 0 ? `${sets}x${reps} @${w}kg` : `${sets}x${reps}`;
  }
  if (fam === "time_hold") {
    const s = (e as { target_duration_sec: number }).target_duration_sec;
    return `${e.target_sets}x${s}s`;
  }
  const reps = (e as { target_reps: number }).target_reps;
  return `${e.target_sets}x${reps}`;
}

type Props = {
  day: PlanDay;
  planId?: number;
  completable?: boolean;
  onComplete?: (planDayId: number) => void;
  compact?: boolean;
  /** Session-5: show pencil button that opens EditPlanDayModal. */
  editable?: boolean;
  /** Called with the refreshed plan after an edit/delete. */
  onUpdated?: (plan: Plan) => void;
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
  editable = false,
  onUpdated,
}: Props) {
  const pretty = prettyDate(day.day_date);
  const primary = (day.exercises?.[0] as PlanDayExercise | undefined) ?? null;
  const [editOpen, setEditOpen] = useState(false);
  const canEdit = editable && !day.completed && planId != null;

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
        <div className="flex items-center gap-2">
          {canEdit && (
            <button
              type="button"
              onClick={() => setEditOpen(true)}
              aria-label="Edit day"
              className="text-[11px] text-[color:var(--color-ink-3)] hover:text-[color:var(--color-gold)]"
              title="Edit day"
            >
              ✎
            </button>
          )}
          {day.completed && (
            <div className="text-[10px] tracking-[0.16em] text-[color:var(--color-good)] uppercase">
              done
            </div>
          )}
        </div>
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
                {formatTarget(e)}
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

      {editOpen && planId != null && (
        <EditPlanDayModal
          planId={planId}
          day={day}
          onClose={() => setEditOpen(false)}
          onSaved={(plan) => onUpdated?.(plan)}
          onDeleted={(plan) => onUpdated?.(plan)}
        />
      )}
    </div>
  );
}
