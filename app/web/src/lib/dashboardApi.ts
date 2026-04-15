// Dashboard API client — typed wrappers around Session-2 endpoints.

import { api } from "./api";

export type Insight = {
  id: string;
  category:
    | "progress"
    | "weakness"
    | "consistency"
    | "volume"
    | "recovery"
    | "milestone";
  severity: "info" | "notice" | "warn" | "celebrate";
  text: string;
  exercise: string | null;
  source_session_ids: number[];
  source_rep_ids: number[];
  data: Record<string, unknown>;
  created_at: string;
};

export type KpiDelta = {
  current: number;
  previous: number;
  delta: number;
  pct: number | null;
};

export type DashboardOverview = {
  today: {
    reps: number;
    avg_form_score: number;
    time_sec: number;
    session_count: number;
    streak_days: number;
  };
  wow_deltas: {
    reps: KpiDelta;
    form: KpiDelta;
    time: KpiDelta;
    sessions: KpiDelta;
  };
  top_insights: Insight[];
  personal_records: {
    biggest_session: { id: number; exercise: string; total_reps: number; date: string } | null;
    best_form_day: { id: number; exercise: string; avg_form_score: number; date: string } | null;
    longest_plank: { id: number; duration_sec: number; date: string } | null;
    longest_streak: number;
  };
  muscle_balance: Record<string, number>;
  totals: { all_sessions: number };
};

export type ExerciseDeepDive = {
  exercise: string;
  scores: { session_id: number; date: string; avg_form_score: number; total_reps: number; duration_sec: number }[];
  quality_breakdown: { session_id: number; date: string; good: number; moderate: number; bad: number }[];
  tempo: { session_id: number; rep_num: number; ecc_sec: number; con_sec: number }[];
  depth: { session_id: number; rep_num: number; peak_angle: number }[];
  fatigue_curve: { rep_num: number; avg_score: number; sample: number }[];
  top_issues: { issue: string; count: number }[];
  insights: Insight[];
  muscles: string[];
};

export type HeatmapCell = {
  date: string;
  reps_count: number;
  session_count: number;
  duration_sec: number;
};

export type SessionRow = {
  id: number;
  exercise: string;
  classifier: string;
  date: string;
  duration_sec: number;
  total_reps: number;
  good_reps: number;
  avg_form_score: number;
  user_id?: number | null;
  weight_kg?: number | null;
};

export type SessionDetail = SessionRow & {
  reps: {
    id: number;
    rep_num: number;
    form_score: number;
    quality: string;
    issues: string[];
    duration: number;
    peak_angle?: number | null;
    ecc_sec?: number | null;
    con_sec?: number | null;
    score_min?: number | null;
    score_max?: number | null;
    set_num?: number | null;
  }[];
  sets: {
    id: number;
    set_num: number;
    reps_count: number;
    rest_before_sec?: number | null;
    avg_form_score?: number | null;
    score_dropoff?: number | null;
    failure_type?: string | null;
  }[];
  total_rest_sec?: number | null;
  consistency_score?: number | null;
  fatigue_index?: number | null;
  muscle_groups?: string | null;
};

export const dashboardApi = {
  overview: () => api<DashboardOverview>("/api/dashboard/overview"),
  insights: (params: { exercise?: string; period?: string; limit?: number } = {}) => {
    const q = new URLSearchParams();
    if (params.exercise) q.set("exercise", params.exercise);
    if (params.period) q.set("period", params.period);
    if (params.limit) q.set("limit", String(params.limit));
    const s = q.toString();
    return api<{ insights: Insight[] }>(`/api/dashboard/insights${s ? `?${s}` : ""}`);
  },
  exercise: (exercise: string) =>
    api<ExerciseDeepDive>(`/api/dashboard/exercise/${exercise}`),
  heatmap: (days = 84) =>
    api<{ days: number; cells: HeatmapCell[] }>(`/api/dashboard/heatmap?days=${days}`),
  muscleBalance: (days = 7) =>
    api<{ days: number; groups: Record<string, number> }>(
      `/api/dashboard/muscle-balance?days=${days}`,
    ),
  sessions: (limit = 20, offset = 0, exercise?: string) => {
    const q = new URLSearchParams();
    q.set("limit", String(limit));
    q.set("offset", String(offset));
    if (exercise) q.set("exercise", exercise);
    return api<SessionRow[]>(`/api/dashboard/sessions?${q.toString()}`);
  },
  sessionDetail: (id: number) =>
    api<SessionDetail>(`/api/dashboard/session/${id}`),
};
