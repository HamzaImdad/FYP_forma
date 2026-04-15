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

export type Goal = {
  id: number;
  user_id: number;
  title: string;
  description: string | null;
  goal_type: "volume" | "quality" | "consistency" | "skill" | "duration" | "balance";
  exercise: string | null;
  target_value: number;
  current_value: number;
  unit: string;
  period: string | null;
  deadline: string | null;
  status: "active" | "completed" | "failed" | "paused";
  created_at: string;
  completed_at: string | null;
  progress: number;
  milestones: GoalMilestone[];
};

export type GoalTemplate = {
  key: string;
  title: string;
  goal_type: Goal["goal_type"];
  exercise: string | null;
  target_value: number;
  unit: string;
  period: string | null;
  description: string;
};

export type CreateGoalBody = {
  title: string;
  goal_type: Goal["goal_type"];
  target_value: number;
  unit: string;
  exercise?: string | null;
  period?: string | null;
  deadline?: string | null;
  description?: string | null;
};

export type MilestoneWithGoal = GoalMilestone & {
  goal_title: string;
  goal_type: string;
  exercise: string | null;
};

// ── Plans ────────────────────────────────────────────────────────────

export type PlanDayExercise = {
  exercise: string;
  target_reps: number;
  target_sets: number;
  notes?: string;
};

export type PlanDay = {
  id: number;
  plan_id: number;
  day_date: string;
  is_rest: boolean;
  exercises: PlanDayExercise[];
  completed: boolean;
  completed_at: string | null;
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

export type PlanDraft = {
  title: string;
  summary: string;
  start_date: string;
  days: Array<{
    day_date: string;
    is_rest: boolean;
    exercises: PlanDayExercise[];
  }>;
};

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
  listPlans: () =>
    api<{ plans: Plan[] }>("/api/plans").then((r) => r.plans),

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
    days: Array<{
      day_date: string;
      is_rest: boolean;
      exercises: PlanDayExercise[];
    }>;
  }) =>
    api<{ plan: Plan }>("/api/plans", {
      method: "POST",
      body: JSON.stringify(body),
    }).then((r) => r.plan),

  deletePlan: (id: number) =>
    api<{ ok: boolean }>(`/api/plans/${id}`, { method: "DELETE" }),

  completePlanDay: (planId: number, planDayId: number) =>
    api<{ plan: Plan }>(`/api/plans/${planId}/days/${planDayId}/complete`, {
      method: "POST",
    }).then((r) => r.plan),

  // Plan draft (for live preview during plan-creator chat)
  getPlanDraft: (conversationId: number) =>
    api<{ draft: PlanDraft | null }>(
      `/api/chat/conversations/${conversationId}/plan_draft`,
    ).then((r) => r.draft),
};
