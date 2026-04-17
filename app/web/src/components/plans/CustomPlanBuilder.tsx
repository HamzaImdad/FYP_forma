// CustomPlanBuilder — Redesign Phase 3.
//
// Two modes:
//   single : one exercise, family-adaptive input (reps / weight+reps / seconds)
//   multi  : list of 2-N exercises, round-robin scheduled across training days
//
// After save, opens GoalSuggestionModal so the user can opt into a couple
// of specific per-exercise goals (the plan itself already auto-creates
// plan_progress + consistency on the backend).

import { useMemo, useState } from "react";
import type { PlanDraft } from "@/lib/plansApi";
import { plansApi } from "@/lib/plansApi";
import {
  EXERCISE_FAMILIES,
  isWeightOptional,
  type Family,
} from "@/lib/exerciseRegistry";
import {
  SUPPORTED_EXERCISES,
  autoTitle,
  blankExerciseInput,
  generateProgressivePlan,
  todayIso,
  type CustomPlanInput,
  type ExerciseInput,
  type RepCountInput,
  type TimeHoldInput,
  type WeightedInput,
} from "./customPlanLogic";
import { GoalSuggestionModal } from "./GoalSuggestionModal";

type Props = {
  onDraftChange: (draft: PlanDraft | null) => void;
  onSaved: () => void;
};

const TIMEFRAME_OPTIONS = [14, 21, 30, 45, 60] as const;
const DAYS_PER_WEEK_OPTIONS = [3, 4, 5, 6, 7] as const;

function prettyLabel(ex: string): string {
  return ex.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

// ── Styling helpers — keep visual style matching the rest of the app ──

const labelCls =
  "block text-[10px] uppercase tracking-[0.2em] text-[color:var(--color-ink-3)] mb-1.5";
const inputCls =
  "w-full bg-transparent border border-[color:var(--rule)] focus:border-[color:var(--color-gold)] outline-none px-3 py-2 text-[14px] text-[color:var(--color-ink)] rounded-[2px]";
const segmentBtnClass = (active: boolean) =>
  `px-3 py-1.5 text-[11px] uppercase tracking-[0.18em] border transition-colors rounded-[2px] ${
    active
      ? "bg-[color:var(--color-gold)] text-[color:var(--color-page)] border-[color:var(--color-gold)]"
      : "text-[color:var(--color-ink-3)] border-[color:var(--rule)] hover:text-[color:var(--color-ink)]"
  }`;

// ── Per-family exercise row (family-adaptive inputs) ─────────────────

function ExerciseRow({
  value,
  onChange,
  onRemove,
  canRemove,
}: {
  value: ExerciseInput;
  onChange: (next: ExerciseInput) => void;
  onRemove?: () => void;
  canRemove: boolean;
}) {
  const family: Family = value.family;
  const weightOptional = isWeightOptional(value.exercise);

  function setExerciseName(nextName: string) {
    // Switching exercise can cross families — rebuild the input from the
    // blank template so field shapes stay consistent.
    onChange(blankExerciseInput(nextName));
  }

  return (
    <div className="border border-[color:var(--rule)] rounded-[3px] p-4 space-y-3 bg-[color:var(--color-raised)]/40">
      <div className="flex items-start gap-3">
        <div className="flex-1">
          <label className={labelCls}>Exercise</label>
          <select
            value={value.exercise}
            onChange={(e) => setExerciseName(e.target.value)}
            className={inputCls}
          >
            {SUPPORTED_EXERCISES.map((ex) => (
              <option key={ex} value={ex} style={{ color: "#0A0A0A", background: "#F5F5F5" }}>
                {prettyLabel(ex)} - {EXERCISE_FAMILIES[ex].family.replace("_", " ")}
              </option>
            ))}
          </select>
        </div>
        {canRemove && onRemove && (
          <button
            type="button"
            onClick={onRemove}
            aria-label="Remove exercise"
            className="mt-6 shrink-0 text-[11px] uppercase tracking-[0.14em] text-[color:var(--color-ink-3)] hover:text-[color:var(--color-bad)] transition-colors px-2 py-1"
          >
            Remove
          </button>
        )}
      </div>

      {family === "rep_count" && (
        <div className="grid grid-cols-3 gap-3">
          <div>
            <label className={labelCls}>Current reps</label>
            <input
              type="number" min={1}
              value={(value as RepCountInput).baselineReps}
              onChange={(e) =>
                onChange({
                  ...(value as RepCountInput),
                  baselineReps: Number(e.target.value) || 0,
                })
              }
              className={inputCls}
            />
          </div>
          <div>
            <label className={labelCls}>Target reps</label>
            <input
              type="number" min={1}
              value={(value as RepCountInput).targetReps}
              onChange={(e) =>
                onChange({
                  ...(value as RepCountInput),
                  targetReps: Number(e.target.value) || 0,
                })
              }
              className={inputCls}
            />
          </div>
          <div>
            <label className={labelCls}>Sets</label>
            <input
              type="number" min={1} max={6}
              value={value.sets}
              onChange={(e) =>
                onChange({
                  ...(value as RepCountInput),
                  sets: Math.min(6, Math.max(1, Number(e.target.value) || 1)),
                })
              }
              className={inputCls}
            />
          </div>
        </div>
      )}

      {family === "weighted" && (
        <>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className={labelCls}>
                Current weight (kg){weightOptional ? " - 0 = bodyweight" : ""}
              </label>
              <input
                type="number" min={0} step={2.5}
                value={(value as WeightedInput).baselineWeightKg}
                onChange={(e) =>
                  onChange({
                    ...(value as WeightedInput),
                    baselineWeightKg: Math.max(0, Number(e.target.value) || 0),
                  })
                }
                className={inputCls}
              />
            </div>
            <div>
              <label className={labelCls}>Target weight (kg)</label>
              <input
                type="number" min={0} step={2.5}
                value={(value as WeightedInput).targetWeightKg}
                onChange={(e) =>
                  onChange({
                    ...(value as WeightedInput),
                    targetWeightKg: Math.max(0, Number(e.target.value) || 0),
                  })
                }
                className={inputCls}
              />
            </div>
          </div>
          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className={labelCls}>Current reps/set</label>
              <input
                type="number" min={1}
                value={(value as WeightedInput).baselineReps}
                onChange={(e) =>
                  onChange({
                    ...(value as WeightedInput),
                    baselineReps: Number(e.target.value) || 0,
                  })
                }
                className={inputCls}
              />
            </div>
            <div>
              <label className={labelCls}>Target reps/set</label>
              <input
                type="number" min={1}
                value={(value as WeightedInput).targetReps}
                onChange={(e) =>
                  onChange({
                    ...(value as WeightedInput),
                    targetReps: Number(e.target.value) || 0,
                  })
                }
                className={inputCls}
              />
            </div>
            <div>
              <label className={labelCls}>Sets</label>
              <input
                type="number" min={1} max={8}
                value={value.sets}
                onChange={(e) =>
                  onChange({
                    ...(value as WeightedInput),
                    sets: Math.min(8, Math.max(1, Number(e.target.value) || 1)),
                  })
                }
                className={inputCls}
              />
            </div>
          </div>
        </>
      )}

      {family === "time_hold" && (
        <div className="grid grid-cols-3 gap-3">
          <div>
            <label className={labelCls}>Current hold (s)</label>
            <input
              type="number" min={5}
              value={(value as TimeHoldInput).baselineSec}
              onChange={(e) =>
                onChange({
                  ...(value as TimeHoldInput),
                  baselineSec: Math.max(5, Number(e.target.value) || 0),
                })
              }
              className={inputCls}
            />
          </div>
          <div>
            <label className={labelCls}>Target hold (s)</label>
            <input
              type="number" min={5}
              value={(value as TimeHoldInput).targetSec}
              onChange={(e) =>
                onChange({
                  ...(value as TimeHoldInput),
                  targetSec: Math.max(5, Number(e.target.value) || 0),
                })
              }
              className={inputCls}
            />
          </div>
          <div>
            <label className={labelCls}>Sets</label>
            <input
              type="number" min={1} max={3}
              value={value.sets}
              onChange={(e) =>
                onChange({
                  ...(value as TimeHoldInput),
                  sets: Math.min(3, Math.max(1, Number(e.target.value) || 1)),
                })
              }
              className={inputCls}
            />
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main builder ─────────────────────────────────────────────────────

export function CustomPlanBuilder({ onDraftChange, onSaved }: Props) {
  const [mode, setMode] = useState<"single" | "multi">("single");
  const [exercises, setExercises] = useState<ExerciseInput[]>(
    () => [blankExerciseInput("pushup")],
  );
  const [totalDays, setTotalDays] = useState<number>(30);
  const [daysPerWeek, setDaysPerWeek] = useState<3 | 4 | 5 | 6 | 7>(5);
  const [startDate, setStartDate] = useState<string>(todayIso());
  const [titleOverride, setTitleOverride] = useState<string>("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewed, setPreviewed] = useState(false);
  const [suggestionForPlan, setSuggestionForPlan] = useState<{
    planId: number;
    exercises: ExerciseInput[];
  } | null>(null);

  const computedTitle = useMemo<string>(() => {
    const input: CustomPlanInput = {
      mode, exercises, totalDays, daysPerWeek, startDate, title: "",
    };
    return titleOverride.trim() || autoTitle(input);
  }, [titleOverride, mode, exercises, totalDays, daysPerWeek, startDate]);

  const buildInput = (): CustomPlanInput => ({
    mode,
    exercises,
    totalDays,
    daysPerWeek,
    startDate,
    title: computedTitle,
  });

  const validate = (): string | null => {
    if (exercises.length === 0) return "Add at least one exercise.";
    if (mode === "single" && exercises.length !== 1) {
      return "Single-exercise mode needs exactly one exercise.";
    }
    for (const e of exercises) {
      if (e.family === "rep_count") {
        if (e.targetReps <= e.baselineReps) {
          return `${prettyLabel(e.exercise)}: target reps must be higher than baseline.`;
        }
      } else if (e.family === "weighted") {
        const isSquatBw = isWeightOptional(e.exercise) && e.targetWeightKg === 0;
        if (!isSquatBw) {
          if (e.targetWeightKg <= 0) {
            return `${prettyLabel(e.exercise)}: enter a target weight (or 0 for bodyweight squat).`;
          }
          if (e.targetWeightKg < e.baselineWeightKg) {
            return `${prettyLabel(e.exercise)}: target weight must be >= baseline.`;
          }
        }
        if (e.targetReps <= 0) {
          return `${prettyLabel(e.exercise)}: target reps must be positive.`;
        }
      } else {
        if (e.targetSec <= e.baselineSec) {
          return `${prettyLabel(e.exercise)}: target hold must be longer than baseline.`;
        }
      }
    }
    return null;
  };

  const handlePreview = () => {
    setError(null);
    const err = validate();
    if (err) { setError(err); return; }
    const draft = generateProgressivePlan(buildInput());
    onDraftChange(draft);
    setPreviewed(true);
  };

  const handleSave = async () => {
    setError(null);
    const err = validate();
    if (err) { setError(err); return; }
    setSaving(true);
    try {
      const draft = generateProgressivePlan(buildInput());
      const end_date = draft.days[draft.days.length - 1]?.day_date ?? startDate;
      const saved = await plansApi.createPlan({
        title: draft.title,
        summary: draft.summary,
        start_date: draft.start_date,
        end_date,
        days: draft.days,
        // Phase 3: narrow auto-create (plan_progress + consistency only).
        // The modal below offers per-exercise goals as opt-in.
        auto_create_goals: true,
      });
      onDraftChange(null);
      setPreviewed(false);
      setSuggestionForPlan({ planId: saved.id, exercises });
      onSaved();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to save plan");
    } finally {
      setSaving(false);
    }
  };

  const updateExerciseAt = (idx: number, next: ExerciseInput) => {
    setExercises((xs) => xs.map((e, i) => (i === idx ? next : e)));
  };

  const addExercise = () => {
    // Pick the first exercise not already in the list, defaulting to squat.
    const used = new Set(exercises.map((e) => e.exercise));
    const pick = SUPPORTED_EXERCISES.find((ex) => !used.has(ex)) ?? "squat";
    setExercises((xs) => [...xs, blankExerciseInput(pick)]);
  };

  const removeExerciseAt = (idx: number) => {
    setExercises((xs) => xs.filter((_, i) => i !== idx));
  };

  const onChangeMode = (next: "single" | "multi") => {
    setMode(next);
    if (next === "single" && exercises.length > 1) {
      setExercises([exercises[0]]);
    }
  };

  return (
    <div className="h-full flex flex-col overflow-auto">
      <div className="px-6 pt-6 pb-4 border-b border-[color:var(--rule)]">
        <div
          className="text-[color:var(--color-ink)]"
          style={{ fontFamily: "var(--font-display)", fontSize: "1.8rem", letterSpacing: "0.04em" }}
        >
          CUSTOM PLAN BUILDER
        </div>
        <p
          className="italic text-[color:var(--color-ink-3)] mt-1 text-[13px]"
          style={{ fontFamily: "var(--font-serif)" }}
        >
          Pick one exercise for a focused challenge, or stack a few for a routine.
          FORMA writes every training day for you.
        </p>
      </div>

      <div className="flex-1 overflow-auto px-6 py-5 space-y-5">
        {/* Mode toggle */}
        <div>
          <label className={labelCls}>Mode</label>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => onChangeMode("single")}
              className={segmentBtnClass(mode === "single")}
            >
              Single exercise
            </button>
            <button
              type="button"
              onClick={() => onChangeMode("multi")}
              className={segmentBtnClass(mode === "multi")}
            >
              Multi-exercise routine
            </button>
          </div>
        </div>

        {/* Exercise list */}
        <div className="space-y-3">
          {exercises.map((ex, i) => (
            <ExerciseRow
              key={`${ex.exercise}-${i}`}
              value={ex}
              onChange={(next) => updateExerciseAt(i, next)}
              onRemove={() => removeExerciseAt(i)}
              canRemove={mode === "multi" && exercises.length > 1}
            />
          ))}
          {mode === "multi" && exercises.length < SUPPORTED_EXERCISES.length && (
            <button
              type="button"
              onClick={addExercise}
              className="w-full text-[11px] uppercase tracking-[0.18em] border border-dashed border-[color:var(--rule)] text-[color:var(--color-ink-3)] hover:border-[color:var(--color-gold)] hover:text-[color:var(--color-gold)] transition-colors rounded-[3px] py-3"
            >
              + Add exercise
            </button>
          )}
        </div>

        {/* Schedule */}
        <div>
          <label className={labelCls}>Timeframe</label>
          <div className="flex gap-2 flex-wrap">
            {TIMEFRAME_OPTIONS.map((opt) => (
              <button
                key={opt}
                type="button"
                onClick={() => setTotalDays(opt)}
                className={segmentBtnClass(totalDays === opt)}
              >
                {opt} days
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className={labelCls}>Days per week</label>
          <div className="flex gap-2 flex-wrap">
            {DAYS_PER_WEEK_OPTIONS.map((opt) => (
              <button
                key={opt}
                type="button"
                onClick={() => setDaysPerWeek(opt as 3 | 4 | 5 | 6 | 7)}
                className={segmentBtnClass(daysPerWeek === opt)}
              >
                {opt}x
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className={labelCls}>Start date</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value || todayIso())}
              className={inputCls}
            />
          </div>
          <div>
            <label className={labelCls}>Plan title</label>
            <input
              type="text"
              value={titleOverride}
              placeholder={computedTitle}
              onChange={(e) => setTitleOverride(e.target.value)}
              className={inputCls}
            />
          </div>
        </div>

        {error && (
          <div className="text-[12px] text-[color:var(--color-bad)] border border-[color:var(--color-bad)]/30 bg-[color:var(--color-bad)]/5 rounded-[3px] px-3 py-2">
            {error}
          </div>
        )}
      </div>

      <div className="px-6 py-4 border-t border-[color:var(--rule)] flex gap-3">
        <button
          type="button"
          onClick={handlePreview}
          disabled={saving}
          className="flex-1 text-[11px] uppercase tracking-[0.22em] border border-[color:var(--rule)] text-[color:var(--color-ink)] hover:border-[color:var(--color-gold)] hover:text-[color:var(--color-gold)] transition-colors rounded-[3px] py-3 disabled:opacity-50"
        >
          Preview plan
        </button>
        <button
          type="button"
          onClick={handleSave}
          disabled={saving}
          className="flex-1 text-[11px] uppercase tracking-[0.22em] bg-[color:var(--color-gold)] text-[color:var(--color-page)] hover:bg-[color:var(--color-gold-hover)] transition-colors rounded-[3px] py-3 disabled:opacity-50"
        >
          {saving ? "Saving..." : previewed ? "Save plan" : "Save plan"}
        </button>
      </div>

      {suggestionForPlan && (
        <GoalSuggestionModal
          planId={suggestionForPlan.planId}
          exercises={suggestionForPlan.exercises}
          onClose={() => setSuggestionForPlan(null)}
        />
      )}
    </div>
  );
}
