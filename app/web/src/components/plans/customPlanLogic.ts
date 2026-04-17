// Redesign Phase 3 — Custom Plan Builder logic.
//
// Single-exercise mode  : one exercise, progressive ramp across all
//                         training days, family-specific metric (reps /
//                         weight / seconds).
// Multi-exercise mode   : N exercises, round-robin-ish distribution so
//                         each training day contains 1..N of them, and
//                         each exercise still rides its own ramp from
//                         baseline to target across the days it actually
//                         appears on.
//
// `generateProgressivePlan` is pure: inputs → PlanDraft. No network, no
// state. Keep it that way so the preview card and tests stay cheap.

import {
  EXERCISE_FAMILIES,
  familyOf,
  type Family,
} from "@/lib/exerciseRegistry";
import type {
  PlanDayExercise,
  PlanDraft,
  PlanDraftDay,
} from "@/lib/plansApi";

// ── Types ────────────────────────────────────────────────────────────

export const SUPPORTED_EXERCISES = Object.keys(EXERCISE_FAMILIES) as readonly string[];
export type SupportedExercise = (typeof SUPPORTED_EXERCISES)[number];

export type RepCountInput = {
  family: "rep_count";
  exercise: string;
  baselineReps: number;
  targetReps: number;
  sets: number;
};

export type WeightedInput = {
  family: "weighted";
  exercise: string;
  baselineWeightKg: number;  // 0 = bodyweight (squat only)
  baselineReps: number;
  targetWeightKg: number;
  targetReps: number;
  sets: number;
};

export type TimeHoldInput = {
  family: "time_hold";
  exercise: string;
  baselineSec: number;
  targetSec: number;
  sets: number;
};

export type ExerciseInput = RepCountInput | WeightedInput | TimeHoldInput;

export type CustomPlanInput = {
  mode: "single" | "multi";
  exercises: ExerciseInput[];
  totalDays: number;
  daysPerWeek: 3 | 4 | 5 | 6 | 7;
  startDate: string;
  title: string;
};

// ── Scheduling ───────────────────────────────────────────────────────

const WORKOUT_MASKS: Record<number, readonly number[]> = {
  3: [1, 0, 1, 0, 1, 0, 0],
  4: [1, 0, 1, 0, 1, 0, 1],
  5: [1, 1, 1, 0, 1, 1, 0],
  6: [1, 1, 1, 1, 1, 1, 0],
  7: [1, 1, 1, 1, 1, 1, 1],
};

function isTrainingDay(dayIndex: number, daysPerWeek: number): boolean {
  const mask = WORKOUT_MASKS[daysPerWeek] ?? WORKOUT_MASKS[5];
  return mask[dayIndex % 7] === 1;
}

function addDays(startIso: string, offset: number): string {
  const d = new Date(`${startIso}T00:00:00`);
  d.setDate(d.getDate() + offset);
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${dd}`;
}

export function todayIso(): string {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${dd}`;
}

function prettyExercise(ex: string): string {
  return ex.replace(/_/g, " ");
}

// ── Family-specific ramp generators (pure) ───────────────────────────

/** Linear reps ramp, rounded to whole reps, clamped to [1, ∞). */
export function rampReps(
  baseline: number,
  target: number,
  workoutDayCount: number,
): number[] {
  if (workoutDayCount <= 0) return [];
  if (workoutDayCount === 1) return [Math.max(1, Math.round(target))];
  const step = (target - baseline) / (workoutDayCount - 1);
  const out: number[] = [];
  for (let i = 0; i < workoutDayCount; i++) {
    out.push(Math.max(1, Math.round(baseline + step * i)));
  }
  return out;
}

/** Linear weight ramp, rounded to nearest 2.5 kg (gym plate math). */
export function rampWeight(
  baselineKg: number,
  targetKg: number,
  workoutDayCount: number,
): number[] {
  const round25 = (v: number) => Math.max(0, Math.round(v / 2.5) * 2.5);
  if (workoutDayCount <= 0) return [];
  if (workoutDayCount === 1) return [round25(targetKg)];
  const step = (targetKg - baselineKg) / (workoutDayCount - 1);
  const out: number[] = [];
  for (let i = 0; i < workoutDayCount; i++) {
    out.push(round25(baselineKg + step * i));
  }
  return out;
}

/** Linear seconds ramp, rounded to nearest 5s. Clamped to [5, ∞). */
export function rampSeconds(
  baselineSec: number,
  targetSec: number,
  workoutDayCount: number,
): number[] {
  const round5 = (v: number) => Math.max(5, Math.round(v / 5) * 5);
  if (workoutDayCount <= 0) return [];
  if (workoutDayCount === 1) return [round5(targetSec)];
  const step = (targetSec - baselineSec) / (workoutDayCount - 1);
  const out: number[] = [];
  for (let i = 0; i < workoutDayCount; i++) {
    out.push(round5(baselineSec + step * i));
  }
  return out;
}

// ── Multi-exercise distribution ──────────────────────────────────────

/**
 * For N exercises over W workout days, assign a subset of exercise
 * indices to each day.
 *
 * Single-exercise mode (N=1) : every day has [0].
 * Small routines (N <= 3)    : every day has all exercises (full-body).
 * Larger routines (N >= 4)   : round-robin pairs so each day's list is a
 *                              digestible subset; every exercise appears
 *                              on workoutDayCount / ceil(N/2) days.
 */
export function distributeMulti(
  exerciseCount: number,
  workoutDayCount: number,
): number[][] {
  if (exerciseCount <= 0 || workoutDayCount <= 0) return [];
  if (exerciseCount === 1) {
    return Array.from({ length: workoutDayCount }, () => [0]);
  }
  if (exerciseCount <= 3) {
    const all = Array.from({ length: exerciseCount }, (_, i) => i);
    return Array.from({ length: workoutDayCount }, () => [...all]);
  }
  const days: number[][] = [];
  for (let i = 0; i < workoutDayCount; i++) {
    const a = (i * 2) % exerciseCount;
    const b = (i * 2 + 1) % exerciseCount;
    const bucket = a === b ? [a] : [a, b];
    days.push(bucket);
  }
  return days;
}

// ── Main: generate a progressive plan draft ──────────────────────────

function _buildItem(
  input: ExerciseInput,
  occurrenceIdx: number,
  rampLen: number,
): PlanDayExercise {
  // occurrenceIdx is this exercise's Kth appearance (0-indexed) in the
  // whole plan. Each exercise rides its own independent ramp so multi-
  // exercise plans still progress each exercise evenly.
  const i = Math.min(Math.max(occurrenceIdx, 0), Math.max(rampLen - 1, 0));
  if (input.family === "rep_count") {
    const reps = rampReps(input.baselineReps, input.targetReps, rampLen);
    return {
      exercise: input.exercise,
      family: "rep_count",
      target_reps: reps[i] ?? input.targetReps,
      target_sets: input.sets,
    };
  }
  if (input.family === "weighted") {
    const weights = rampWeight(
      input.baselineWeightKg,
      input.targetWeightKg,
      rampLen,
    );
    const reps = rampReps(input.baselineReps, input.targetReps, rampLen);
    return {
      exercise: input.exercise,
      family: "weighted",
      target_weight_kg: weights[i] ?? input.targetWeightKg,
      target_reps: reps[i] ?? input.targetReps,
      target_sets: input.sets,
    };
  }
  const seconds = rampSeconds(input.baselineSec, input.targetSec, rampLen);
  return {
    exercise: input.exercise,
    family: "time_hold",
    target_duration_sec: seconds[i] ?? input.targetSec,
    target_sets: input.sets,
  };
}

function _summarize(input: CustomPlanInput): string {
  const { totalDays, daysPerWeek, exercises, mode } = input;
  if (mode === "single" && exercises.length === 1) {
    const e = exercises[0];
    if (e.family === "rep_count") {
      return `Progressive ${prettyExercise(e.exercise)} plan — ${e.baselineReps} → ${e.targetReps} reps per set over ${totalDays} days (${daysPerWeek}x/week).`;
    }
    if (e.family === "weighted") {
      return `Progressive ${prettyExercise(e.exercise)} plan — ${e.baselineWeightKg}kg x ${e.baselineReps} → ${e.targetWeightKg}kg x ${e.targetReps} over ${totalDays} days (${daysPerWeek}x/week).`;
    }
    return `Progressive ${prettyExercise(e.exercise)} plan — ${e.baselineSec}s → ${e.targetSec}s hold over ${totalDays} days (${daysPerWeek}x/week).`;
  }
  const names = exercises.map((e) => prettyExercise(e.exercise)).join(", ");
  return `${exercises.length}-exercise routine (${names}) over ${totalDays} days, ${daysPerWeek}x/week.`;
}

export function generateProgressivePlan(input: CustomPlanInput): PlanDraft {
  const { totalDays, daysPerWeek, startDate, title, exercises } = input;

  const trainingDayIdxs: number[] = [];
  for (let i = 0; i < totalDays; i++) {
    if (isTrainingDay(i, daysPerWeek)) trainingDayIdxs.push(i);
  }
  const workoutDayCount = trainingDayIdxs.length;

  const perDayExerciseIdxs = distributeMulti(
    exercises.length,
    workoutDayCount,
  );

  // How many total appearances each exercise gets, so per-exercise ramps
  // can size their step correctly.
  const appearanceCount: number[] = exercises.map(() => 0);
  for (const bucket of perDayExerciseIdxs) {
    for (const exIdx of bucket) appearanceCount[exIdx]++;
  }

  const seenCount: number[] = exercises.map(() => 0);

  const days: PlanDraftDay[] = [];
  let trainingCursor = 0;
  for (let i = 0; i < totalDays; i++) {
    const day_date = addDays(startDate, i);
    if (!isTrainingDay(i, daysPerWeek)) {
      days.push({ day_date, is_rest: true, exercises: [] });
      continue;
    }
    const bucket = perDayExerciseIdxs[trainingCursor] ?? [0];
    trainingCursor++;
    const dayExercises: PlanDayExercise[] = bucket.map((exIdx) => {
      const ex = exercises[exIdx];
      const occurrence = seenCount[exIdx];
      seenCount[exIdx] = occurrence + 1;
      return _buildItem(ex, occurrence, appearanceCount[exIdx]);
    });
    days.push({ day_date, is_rest: false, exercises: dayExercises });
  }

  return {
    title: title.trim() || autoTitle(input),
    summary: _summarize(input),
    start_date: startDate,
    days,
    source: "custom",
  };
}

export function autoTitle(input: CustomPlanInput): string {
  const { mode, exercises, totalDays } = input;
  if (mode === "single" && exercises.length === 1) {
    const e = exercises[0];
    if (e.family === "rep_count") {
      return `${e.targetReps} ${prettyExercise(e.exercise)} in ${totalDays} days`;
    }
    if (e.family === "weighted") {
      const w = e.targetWeightKg;
      const weightTag = w > 0 ? ` ${w}kg` : " bodyweight";
      return `${prettyExercise(e.exercise)}${weightTag} in ${totalDays} days`;
    }
    return `${e.targetSec}s ${prettyExercise(e.exercise)} in ${totalDays} days`;
  }
  return `${exercises.length}-exercise routine (${totalDays} days)`;
}

// ── Factory helpers for the UI ───────────────────────────────────────

export function blankExerciseInput(exercise: string): ExerciseInput {
  const fam: Family = familyOf(exercise);
  const sets = EXERCISE_FAMILIES[exercise]?.default_sets ?? 3;
  if (fam === "rep_count") {
    return {
      family: "rep_count", exercise,
      baselineReps: 10, targetReps: 25, sets,
    };
  }
  if (fam === "weighted") {
    return {
      family: "weighted", exercise,
      baselineWeightKg: 0, baselineReps: 5,
      targetWeightKg: 0, targetReps: 5, sets,
    };
  }
  return {
    family: "time_hold", exercise,
    baselineSec: 30, targetSec: 60, sets,
  };
}
