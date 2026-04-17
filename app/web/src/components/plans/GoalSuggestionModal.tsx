// Redesign Phase 3 — goal suggestion modal that pops after Custom Builder
// saves a plan. Shows one pre-checked suggestion per exercise in the plan,
// derived from that exercise's final targets. User can uncheck, skip, or
// accept — no silent auto-creation.
//
// The plan-level goals (plan_progress + consistency) are already created
// server-side on save; this modal only adds OPT-IN per-exercise goals.

import { useMemo, useState } from "react";
import { plansApi, type CreateGoalBody } from "@/lib/plansApi";
import type { ExerciseInput } from "./customPlanLogic";

type Props = {
  planId: number;
  exercises: ExerciseInput[];
  onClose: () => void;
};

type Suggestion = {
  key: string;
  label: string;
  body: CreateGoalBody;
};

function prettyExercise(ex: string): string {
  return ex.replace(/_/g, " ");
}

function suggestionsFromInputs(
  exercises: ExerciseInput[],
  planId: number,
): Suggestion[] {
  const out: Suggestion[] = [];
  for (const e of exercises) {
    if (e.family === "rep_count") {
      // Target = final single-set reps × sets. That's the honest "finish
      // the plan's peak workout" bar.
      const target = Math.max(1, e.targetReps * e.sets);
      out.push({
        key: `volume-${e.exercise}`,
        label: `${prettyExercise(e.exercise)}: ${target} good reps`,
        body: {
          title: `${prettyExercise(e.exercise)} volume`,
          goal_type: "volume",
          target_value: target,
          unit: "reps",
          exercise: e.exercise,
          period: "once",
          description: `Finish all the ${prettyExercise(e.exercise)} volume on this plan's peak day.`,
          plan_id: planId,
        },
      });
    } else if (e.family === "weighted") {
      if (e.targetWeightKg > 0) {
        out.push({
          key: `strength-${e.exercise}`,
          label: `${prettyExercise(e.exercise)}: ${e.targetWeightKg}kg x ${e.targetReps} clean reps`,
          body: {
            title: `${prettyExercise(e.exercise)} strength`,
            goal_type: "strength",
            target_value: e.targetWeightKg,
            target_reps: e.targetReps,
            unit: "kg",
            exercise: e.exercise,
            period: "once",
            description: `Lift ${e.targetWeightKg}kg for ${e.targetReps} clean reps in one set.`,
            plan_id: planId,
          },
        });
      } else {
        // bodyweight squat — suggest a volume goal instead.
        const target = Math.max(1, e.targetReps * e.sets);
        out.push({
          key: `volume-${e.exercise}`,
          label: `${prettyExercise(e.exercise)}: ${target} bodyweight reps`,
          body: {
            title: `${prettyExercise(e.exercise)} volume`,
            goal_type: "volume",
            target_value: target,
            unit: "reps",
            exercise: e.exercise,
            period: "once",
            description: `Bodyweight ${prettyExercise(e.exercise)} volume target.`,
            plan_id: planId,
          },
        });
      }
    } else {
      out.push({
        key: `duration-${e.exercise}`,
        label: `${prettyExercise(e.exercise)}: ${e.targetSec}s clean hold`,
        body: {
          title: `${prettyExercise(e.exercise)} duration`,
          goal_type: "duration",
          target_value: e.targetSec,
          unit: "seconds",
          exercise: e.exercise,
          period: "once",
          description: `Hold ${prettyExercise(e.exercise)} for ${e.targetSec}s with good form.`,
          plan_id: planId,
        },
      });
    }
  }
  return out;
}

export function GoalSuggestionModal({ planId, exercises, onClose }: Props) {
  const suggestions = useMemo(
    () => suggestionsFromInputs(exercises, planId),
    [exercises, planId],
  );
  const [checked, setChecked] = useState<Record<string, boolean>>(
    () => Object.fromEntries(suggestions.map((s) => [s.key, true])),
  );
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAccept = async () => {
    setError(null);
    setSubmitting(true);
    const picked = suggestions.filter((s) => checked[s.key]);
    try {
      for (const s of picked) {
        await plansApi.createGoal(s.body);
      }
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to add goals");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-[600] flex items-center justify-center bg-black/60 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
    >
      <div className="w-full max-w-[520px] mx-6 bg-[color:var(--color-raised)] border border-[color:var(--rule)] rounded-[4px] shadow-[0_32px_80px_rgba(0,0,0,0.6)] overflow-hidden">
        <div className="px-6 pt-6 pb-3 border-b border-[color:var(--rule)]">
          <div
            className="text-[color:var(--color-ink)]"
            style={{ fontFamily: "var(--font-display)", fontSize: "1.3rem", letterSpacing: "0.04em" }}
          >
            PLAN SAVED - TRACK ANY GOALS?
          </div>
          <p
            className="italic text-[color:var(--color-ink-3)] mt-1 text-[12px]"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            Your plan's progress is already tracked. Opt into per-exercise goals
            to see 25/50/75/100% milestones for them too.
          </p>
        </div>

        <div className="px-6 py-4 max-h-[45vh] overflow-auto space-y-2">
          {suggestions.length === 0 && (
            <div className="text-[12px] text-[color:var(--color-ink-3)] italic">
              No per-exercise goals to suggest.
            </div>
          )}
          {suggestions.map((s) => (
            <label
              key={s.key}
              className="flex items-center gap-3 py-2 px-3 border border-[color:var(--rule)] rounded-[3px] hover:border-[color:var(--color-gold)]/40 transition-colors cursor-pointer"
            >
              <input
                type="checkbox"
                checked={!!checked[s.key]}
                onChange={(e) =>
                  setChecked((c) => ({ ...c, [s.key]: e.target.checked }))
                }
                className="accent-[color:var(--color-gold)]"
              />
              <span className="text-[13px] text-[color:var(--color-ink)]">
                {s.label}
              </span>
            </label>
          ))}
          {error && (
            <div className="text-[12px] text-[color:var(--color-bad)] border border-[color:var(--color-bad)]/30 bg-[color:var(--color-bad)]/5 rounded-[3px] px-3 py-2 mt-3">
              {error}
            </div>
          )}
        </div>

        <div className="px-6 py-4 border-t border-[color:var(--rule)] flex gap-3 justify-end">
          <button
            type="button"
            onClick={onClose}
            disabled={submitting}
            className="text-[11px] uppercase tracking-[0.22em] px-4 py-2 border border-[color:var(--rule)] text-[color:var(--color-ink-2)] hover:text-[color:var(--color-ink)] transition-colors rounded-[3px] disabled:opacity-50"
          >
            Skip
          </button>
          <button
            type="button"
            onClick={handleAccept}
            disabled={submitting || suggestions.length === 0}
            className="text-[11px] uppercase tracking-[0.22em] px-5 py-2 bg-[color:var(--color-gold)] text-[color:var(--color-page)] hover:bg-[color:var(--color-gold-hover)] transition-colors rounded-[3px] disabled:opacity-50"
          >
            {submitting ? "Adding..." : "Add selected"}
          </button>
        </div>
      </div>
    </div>
  );
}
