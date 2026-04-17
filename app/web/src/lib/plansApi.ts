// Typed wrappers for the Session-4 goals / milestones / plans / badges
// endpoints. Keeps the React components free of URL strings.

import { api } from "./api";

// ── Goals ────────────────────────────────────────────────────────────

export type GoalMilestone = {
  id: number;
  goal_id: number;
  label: string;
  threshold_value: number;
  reached: boolean;
  reached_at: string | null;
  just_reached?: boolean;
};

export type GoalType =
  | "volume"
  | "quality"
  | "consistency"
  | "skill"
  | "duration"
  | "balance"
  | "plan_progress"
  // Redesign Phase 2 — heaviest clean set at `target_reps` reps.
  | "strength";

export type GoalStatus =
  | "active"
  | "completed"
  | "failed"
  | "paused"
  // Redesign Phase 2 — auto-archive flips 7-day-old completed goals here.
  | "archived";

export type Goal = {
  id: number;
  user_id: number;
  title: string;
  description: string | null;
  goal_type: GoalType;
  exercise: string | null;
  target_value: number;
  current_value: number;
  unit: string;
  period: string | null;
  deadline: string | null;
  status: GoalStatus;
  created_at: string;
  completed_at: string | null;
  plan_id?: number | null;
  // Redesign Phase 2 — populated only for strength goals.
  target_reps?: number | null;
  progress: number;
  milestones: GoalMilestone[];
};

export type GoalTemplate = {
  key: string;
  title: string;
  goal_type: GoalType;
  exercise: string | null;
  target_value: number;
  unit: string;
  period: string | null;
  description: string;
};

export type CreateGoalBody = {
  title: string;
  goal_type: GoalType;
  target_value: number;
  unit: string;
  exercise?: string | null;
  period?: string | null;
  deadline?: string | null;
  description?: string | null;
  // Redesign Phase 2 — required for strength goals.
  target_reps?: number | null;
  // Redesign Phase 3 — link opt-in goals back to the plan that
  // suggested them so they archive together.
  plan_id?: number | null;
};

export type MilestoneWithGoal = GoalMilestone & {
  goal_title: string;
  goal_type: string;
  exercise: string | null;
};

// ── Plans ────────────────────────────────────────────────────────────

// Redesign Phase 3 — exercise items in a plan day are a discriminated
// union keyed on `family`. The backend sanitizer stamps `family` on every
// row so the UI can branch on it without a second registry lookup. Legacy
// rows written before Phase 3 may have no family field — treat them as
// rep_count (see isLegacyExercise helper below).
export type RepCountPlanExercise = {
  exercise: string;
  family: "rep_count";
  target_reps: number;
  target_sets: number;
  notes?: string;
};

export type WeightedPlanExercise = {
  exercise: string;
  family: "weighted";
  target_weight_kg: number; // 0 for squat = bodyweight
  target_reps: number;
  target_sets: number;
  notes?: string;
};

export type TimeHoldPlanExercise = {
  exercise: string;
  family: "time_hold";
  target_duration_sec: number;
  target_sets: number;
  notes?: string;
};

// Legacy (pre-Phase-3) rows that never got a `family` stamp. Rendered as
// rep_count by the UI helpers below. Kept as a separate variant so code
// that does care can detect them.
export type LegacyPlanExercise = {
  exercise: string;
  family?: undefined;
  target_reps: number;
  target_sets: number;
  notes?: string;
};

export type PlanDayExercise =
  | RepCountPlanExercise
  | WeightedPlanExercise
  | TimeHoldPlanExercise
  | LegacyPlanExercise;

// Shared render helpers. `family` is optional on legacy rows; these helpers
// treat a missing family as rep_count so old UI code keeps working during
// the Phase-3 rollout.
export function exerciseFamilyOf(
  e: PlanDayExercise,
): "rep_count" | "weighted" | "time_hold" {
  return e.family ?? "rep_count";
}

export function describePlanExercise(e: PlanDayExercise): string {
  const fam = exerciseFamilyOf(e);
  const pretty = e.exercise.replace(/_/g, " ");
  if (fam === "weighted") {
    const w = (e as WeightedPlanExercise).target_weight_kg;
    const reps = (e as WeightedPlanExercise).target_reps;
    const sets = (e as WeightedPlanExercise).target_sets;
    const weightTag = w && w > 0 ? ` @ ${w}kg` : "";
    return `${sets}x${reps} ${pretty}${weightTag}`;
  }
  if (fam === "time_hold") {
    const s = (e as TimeHoldPlanExercise).target_duration_sec;
    const sets = (e as TimeHoldPlanExercise).target_sets;
    return `${sets}x${s}s ${pretty}`;
  }
  const r = e as RepCountPlanExercise | LegacyPlanExercise;
  return `${r.target_sets}x${r.target_reps} ${pretty}`;
}

// Redesign Phase 4 — /api/plans/today attaches this so the dashboard
// strip can render per-exercise chips with pass/partial state.
export type PerExerciseStatus = {
  exercise: string;
  family: "rep_count" | "weighted" | "time_hold" | null;
  passed: boolean;
  progress: Record<string, number>;
};

export type PlanDay = {
  id: number;
  plan_id: number;
  day_date: string;
  is_rest: boolean;
  exercises: PlanDayExercise[];
  completed: boolean;
  completed_at: string | null;
  // Only present on GET /api/plans/today.
  per_exercise_status?: PerExerciseStatus[];
};

export type Plan = {
  id: number;
  user_id: number;
  title: string;
  summary: string | null;
  start_date: string;
  end_date: string;
  status: string;
  created_by_chat: number;
  conversation_id: number | null;
  created_at: string;
  days?: PlanDay[];
};

export type PlanDraftDay = {
  day_date: string;
  is_rest: boolean;
  exercises: PlanDayExercise[];
};

export type PlanDraftSource = "chat" | "custom";

export type PlanDraft = {
  title: string;
  summary: string;
  start_date: string;
  days: PlanDraftDay[];
  // Session-5: identifies which creation path produced this draft. Chat drafts
  // are saved via a user turn ("Approve and save this plan.") — the LLM calls
  // save_plan. Custom drafts are saved directly via POST /api/plans.
  source?: PlanDraftSource;
};

export type PlanStatus = "active" | "paused" | "archived" | "completed";

// ── Badges ───────────────────────────────────────────────────────────

export type Badge = {
  badge_key: string;
  title: string;
  description: string;
  earned: boolean;
  earned_at: string | null;
  metadata: Record<string, unknown>;
};

// ── Client ───────────────────────────────────────────────────────────

export const plansApi = {
  // Goals
  listGoals: (status?: string) =>
    api<{ goals: Goal[] }>(
      `/api/goals${status ? `?status=${encodeURIComponent(status)}` : ""}`,
    ).then((r) => r.goals),

  listGoalTemplates: () =>
    api<{ templates: GoalTemplate[] }>("/api/goals/templates").then((r) => r.templates),

  createGoal: (body: CreateGoalBody) =>
    api<{ goal: Goal }>("/api/goals", {
      method: "POST",
      body: JSON.stringify(body),
    }).then((r) => r.goal),

  patchGoal: (id: number, body: Partial<Goal>) =>
    api<{ goal: Goal }>(`/api/goals/${id}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }).then((r) => r.goal),

  deleteGoal: (id: number) =>
    api<{ ok: boolean }>(`/api/goals/${id}`, { method: "DELETE" }),

  // Milestones
  listMilestones: () =>
    api<{ milestones: MilestoneWithGoal[] }>("/api/milestones").then(
      (r) => r.milestones,
    ),

  // Badges
  listBadges: () =>
    api<{ badges: Badge[] }>("/api/badges").then((r) => r.badges),

  // Plans
  listPlans: (status?: PlanStatus) =>
    api<{ plans: Plan[] }>(
      `/api/plans${status ? `?status=${encodeURIComponent(status)}` : ""}`,
    ).then((r) => r.plans),

  getPlan: (id: number) =>
    api<{ plan: Plan }>(`/api/plans/${id}`).then((r) => r.plan),

  getTodaysPlanDay: () =>
    api<{ plan_day: (PlanDay & { plan_title?: string }) | null }>(
      "/api/plans/today",
    ).then((r) => r.plan_day),

  createPlan: (body: {
    title: string;
    summary?: string;
    start_date: string;
    end_date: string;
    days: PlanDraftDay[];
    auto_create_goals?: boolean;
  }) =>
    api<{ plan: Plan }>("/api/plans", {
      method: "POST",
      body: JSON.stringify(body),
    }).then((r) => r.plan),

  updatePlanStatus: (id: number, status: PlanStatus) =>
    api<{ plan: Plan }>(`/api/plans/${id}/status`, {
      method: "PATCH",
      body: JSON.stringify({ status }),
    }).then((r) => r.plan),

  deletePlan: (id: number) =>
    api<{ ok: boolean }>(`/api/plans/${id}`, { method: "DELETE" }),

  completePlanDay: (planId: number, planDayId: number) =>
    api<{ plan: Plan }>(`/api/plans/${planId}/days/${planDayId}/complete`, {
      method: "POST",
    }).then((r) => r.plan),

  // Manual edits (Session-5) — chatbot-created AND custom plans share these.
  updatePlan: (
    id: number,
    patch: Partial<Pick<Plan, "title" | "summary" | "start_date" | "end_date">>,
  ) =>
    api<{ plan: Plan }>(`/api/plans/${id}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    }).then((r) => r.plan),

  updatePlanDay: (
    planId: number,
    planDayId: number,
    patch: {
      day_date?: string;
      is_rest?: boolean;
      exercises?: PlanDayExercise[];
    },
  ) =>
    api<{ plan: Plan; warnings: string[] }>(
      `/api/plans/${planId}/days/${planDayId}`,
      {
        method: "PATCH",
        body: JSON.stringify(patch),
      },
    ),

  insertPlanDay: (
    planId: number,
    body: {
      day_date: string;
      is_rest: boolean;
      exercises: PlanDayExercise[];
    },
  ) =>
    api<{ plan: Plan; day_id: number; warnings: string[] }>(
      `/api/plans/${planId}/days`,
      {
        method: "POST",
        body: JSON.stringify(body),
      },
    ),

  deletePlanDay: (planId: number, planDayId: number) =>
    api<{ plan: Plan }>(`/api/plans/${planId}/days/${planDayId}`, {
      method: "DELETE",
    }).then((r) => r.plan),

  // Plan draft (for live preview during plan-creator chat)
  getPlanDraft: (conversationId: number) =>
    api<{ draft: PlanDraft | null }>(
      `/api/chat/conversations/${conversationId}/plan_draft`,
    ).then((r) => r.draft),

  // Redesign Phase 3 — per-user draft (survives tab close). Used by both
  // Custom Builder and Plan Architect to offer "resume draft?" on entry.
  getUserDraft: () =>
    api<{
      draft: PlanDraft | null;
      updated_at?: string;
      source?: "chat" | "custom";
      conversation_id?: number | null;
    }>("/api/plans/draft"),

  discardUserDraft: () =>
    api<{ ok: boolean }>("/api/plans/draft", { method: "DELETE" }),
};
