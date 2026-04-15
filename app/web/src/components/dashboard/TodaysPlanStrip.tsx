// TodaysPlanStrip — banner on the dashboard showing today's workout from
// the active plan (if any). Silent if no plan_day is scheduled for today.

import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { plansApi, type PlanDay, type PlanDayExercise } from "@/lib/plansApi";
import { useGoalsUpdated } from "@/hooks/useGoalsUpdated";

type TodayPlanDay = PlanDay & { plan_title?: string };

export function TodaysPlanStrip() {
  const [day, setDay] = useState<TodayPlanDay | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchDay = useCallback(() => {
    plansApi
      .getTodaysPlanDay()
      .then((d) => setDay(d))
      .catch(() => setDay(null))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchDay();
  }, [fetchDay]);

  // Refresh when the plan gets a day-complete update.
  useGoalsUpdated(useCallback(() => fetchDay(), [fetchDay]));

  if (loading || !day) return null;

  const first = (day.exercises?.[0] as PlanDayExercise | undefined) ?? null;

  return (
    <div className="border border-[color:var(--color-gold)] rounded-[4px] px-5 py-4 bg-[color:var(--color-gold)]/10 flex items-baseline justify-between gap-4 flex-wrap">
      <div>
        <div
          className="text-[10px] uppercase tracking-[0.22em] text-[color:var(--color-gold)]"
          style={{ fontFamily: "var(--font-display)" }}
        >
          Today's plan
        </div>
        <div
          className="text-[color:var(--color-ink)] mt-1"
          style={{
            fontFamily: "var(--font-display)",
            fontSize: "1.4rem",
            letterSpacing: "0.04em",
          }}
        >
          {day.is_rest ? (
            "REST DAY"
          ) : (
            <>
              {(day.exercises ?? [])
                .map(
                  (e) =>
                    `${e.target_sets}×${e.target_reps} ${e.exercise.replace(/_/g, " ")}`,
                )
                .join(" · ")
                .toUpperCase()}
            </>
          )}
        </div>
        {day.plan_title && (
          <p
            className="italic text-[color:var(--color-ink-3)] text-[12px] mt-1"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            from “{day.plan_title}”
          </p>
        )}
      </div>
      {!day.is_rest && first && (
        <Link
          to={`/workout/${first.exercise}?plan_day_id=${day.id}`}
          className="text-[11px] uppercase tracking-[0.22em] px-5 py-3 bg-[color:var(--color-gold)] text-[color:var(--color-page)] hover:bg-[color:var(--color-gold-hover)] transition-colors rounded-[3px]"
        >
          Start workout
        </Link>
      )}
      {day.completed && (
        <div className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-good)]">
          Done ✓
        </div>
      )}
    </div>
  );
}
