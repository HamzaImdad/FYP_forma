// TodaysPlanStrip — banner on the dashboard showing today's workout from
// the active plan (if any). Silent if no plan_day is scheduled for today.

import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  plansApi,
  describePlanExercise,
  type PlanDay,
  type PlanDayExercise,
} from "@/lib/plansApi";
import { useGoalsUpdated } from "@/hooks/useGoalsUpdated";
import { usePlanDayCompleted } from "@/hooks/usePlanDayCompleted";

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
  usePlanDayCompleted(useCallback(() => fetchDay(), [fetchDay]));

  if (loading || !day) return null;

  const first = (day.exercises?.[0] as PlanDayExercise | undefined) ?? null;

  // Phase 4 — build per-exercise chip list from the backend status + the
  // plan day spec. Backend returns `per_exercise_status` on today's day;
  // fall back to a dumb-but-not-broken view when it's missing.
  const statusByEx = new Map<
    string,
    { passed: boolean; progressPct: number }
  >();
  for (const s of day.per_exercise_status ?? []) {
    let pct = 0;
    if (s.family === "rep_count") {
      const target = s.progress["target_total_reps"] || 0;
      const cur = s.progress["current_good_reps"] || 0;
      pct = target > 0 ? Math.min(1, cur / target) : 0;
    } else if (s.family === "weighted") {
      const target = s.progress["target_weight_kg"] || 0;
      const cur = s.progress["current_max_weight_kg"] || 0;
      pct = target > 0 ? Math.min(1, cur / target) : cur > 0 ? 1 : 0;
    } else if (s.family === "time_hold") {
      const target = s.progress["target_duration_sec"] || 0;
      const cur = s.progress["current_max_duration_sec"] || 0;
      pct = target > 0 ? Math.min(1, cur / target) : 0;
    }
    statusByEx.set(s.exercise, { passed: s.passed, progressPct: pct });
  }

  return (
    <div className="border border-[color:var(--color-gold)] rounded-[4px] px-5 py-4 bg-[color:var(--color-gold)]/10 flex flex-col gap-3">
      <div className="flex items-baseline justify-between gap-4 flex-wrap">
        <div>
          <div
            className="text-[10px] uppercase tracking-[0.22em] text-[color:var(--color-gold)]"
            style={{ fontFamily: "var(--font-display)" }}
          >
            Today's workout
          </div>
          <div
            className="text-[color:var(--color-ink)] mt-1"
            style={{
              fontFamily: "var(--font-display)",
              fontSize: "1.4rem",
              letterSpacing: "0.04em",
            }}
          >
            {day.is_rest
              ? "REST DAY"
              : (day.exercises ?? [])
                  .map((e) => describePlanExercise(e))
                  .join(" · ")
                  .toUpperCase()}
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

      {/* Per-exercise chip row — Phase 4. Green when passed,
          amber when partial, neutral when untouched. */}
      {!day.is_rest && (day.exercises ?? []).length > 0 && (
        <div className="flex flex-wrap gap-2">
          {(day.exercises ?? []).map((ex) => {
            const s = statusByEx.get(ex.exercise);
            const tone = s?.passed
              ? "border-[color:var(--color-good)] text-[color:var(--color-good)] bg-[color:var(--color-good)]/10"
              : s && s.progressPct >= 0.5
                ? "border-[color:var(--color-warn)] text-[color:var(--color-warn)] bg-[color:var(--color-warn)]/10"
                : "border-[color:var(--rule)] text-[color:var(--color-ink-2)]";
            return (
              <Link
                key={ex.exercise}
                to={`/workout/${ex.exercise}?plan_day_id=${day.id}`}
                className={`text-[10px] uppercase tracking-[0.16em] px-3 py-1.5 border rounded-[3px] transition-colors hover:text-[color:var(--color-ink)] ${tone}`}
              >
                {s?.passed ? "✓ " : ""}
                {describePlanExercise(ex)}
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}
