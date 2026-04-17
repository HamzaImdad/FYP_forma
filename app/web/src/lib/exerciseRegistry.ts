// FORMA exercise family registry — TypeScript mirror of app/exercise_registry.py.
// Keep the two in sync; tests/test_exercise_registry_parity.py asserts parity.
//
// Three families drive every downstream decision (goal display, plan-day
// rendering, Custom Builder UI, session summary screens):
//   - rep_count : progress = more reps per set
//   - weighted  : progress = heavier weight
//   - time_hold : progress = longer hold duration

export type Family = "rep_count" | "weighted" | "time_hold";

export type ExerciseMeta = {
  family: Family;
  default_sets: number;
  weight_optional: boolean;
};

export const EXERCISE_FAMILIES: Record<string, ExerciseMeta> = {
  // rep_count — bodyweight, progress = more reps per set
  pushup:         { family: "rep_count", default_sets: 3, weight_optional: false },
  pullup:         { family: "rep_count", default_sets: 3, weight_optional: false },
  bicep_curl:     { family: "rep_count", default_sets: 3, weight_optional: false },
  tricep_dip:     { family: "rep_count", default_sets: 3, weight_optional: false },
  lunge:          { family: "rep_count", default_sets: 3, weight_optional: false },
  // weighted — progress = heavier weight for same/similar reps.
  // squat is weight_optional: target_weight_kg === 0 means bodyweight.
  squat:          { family: "weighted",  default_sets: 5, weight_optional: true  },
  deadlift:       { family: "weighted",  default_sets: 5, weight_optional: false },
  // time_hold — progress = longer hold
  plank:          { family: "time_hold", default_sets: 1, weight_optional: false },
  // ── New exercises ──
  crunch:         { family: "rep_count",  default_sets: 3, weight_optional: false },
  lateral_raise:  { family: "rep_count",  default_sets: 3, weight_optional: false },
  side_plank:     { family: "time_hold",  default_sets: 3, weight_optional: false },
};

export function familyOf(exercise: string): Family {
  const meta = EXERCISE_FAMILIES[exercise];
  if (!meta) throw new Error(`unknown exercise ${exercise}`);
  return meta.family;
}

export function isWeightOptional(exercise: string): boolean {
  return !!EXERCISE_FAMILIES[exercise]?.weight_optional;
}

export function defaultSets(exercise: string): number {
  const meta = EXERCISE_FAMILIES[exercise];
  if (!meta) throw new Error(`unknown exercise ${exercise}`);
  return meta.default_sets;
}

// ── Form-score thresholds (0–1 scale) ────────────────────────────────────
// Must match the Python defaults. Env overrides apply only on the server
// side; the frontend uses the compile-time constants for display/labels.
export const GOOD_REP_THRESHOLD = 0.6;
export const CLEAN_REP_THRESHOLD = 0.8;
