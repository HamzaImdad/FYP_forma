// FORMA Dashboard — Session 2 deliverable.
//
// Layout: TodayRibbon → InsightCard → PersonalRecordsStrip → FormTrendChart
// → ActivityHeatmap + MuscleBalanceRadar (side by side) → ExerciseChipRow
// + DeepDivePanel → SessionHistoryList. URL-driven exercise selection and
// heatmap date filter so back-button works and links are shareable.

import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { dashboardApi, type DashboardOverview, type HeatmapCell } from "../lib/dashboardApi";
import { api } from "../lib/api";
import { TodayRibbon } from "../components/dashboard/TodayRibbon";
import { InsightCard } from "../components/dashboard/InsightCard";
import { PersonalRecordsStrip } from "../components/dashboard/PersonalRecordsStrip";
import { FormTrendChart } from "../components/dashboard/FormTrendChart";
import { ExerciseChipRow } from "../components/dashboard/ExerciseChipRow";
import { DeepDivePanel } from "../components/dashboard/DeepDivePanel";
import { ActivityHeatmap } from "../components/dashboard/ActivityHeatmap";
import { MuscleBalanceRadar } from "../components/dashboard/MuscleBalanceRadar";
import { SessionHistoryList } from "../components/dashboard/SessionHistoryList";
import { PersonalCoachPanel } from "../components/dashboard/PersonalCoachPanel";
import { ActiveGoalsCard } from "../components/dashboard/ActiveGoalsCard";
import { TodaysPlanStrip } from "../components/dashboard/TodaysPlanStrip";
import { useSessionCompleted } from "../hooks/useSessionCompleted";
import { useAuth } from "../context/AuthContext";

type ExerciseMeta = { id: string; display_name: string };
type DistributionRow = { exercise: string; count: number; total_reps: number };

export function DashboardPage() {
  const { user } = useAuth();
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const selectedExercise = searchParams.get("exercise");
  const filterDate = searchParams.get("day");

  const [overview, setOverview] = useState<DashboardOverview | null>(null);
  const [overviewError, setOverviewError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const [exercises, setExercises] = useState<ExerciseMeta[]>([]);
  const [distribution, setDistribution] = useState<DistributionRow[]>([]);
  const [heatmap, setHeatmap] = useState<HeatmapCell[]>([]);

  const fetchOverview = useCallback(() => {
    dashboardApi
      .overview()
      .then((d) => {
        setOverview(d);
        setOverviewError(null);
      })
      .catch((e) => {
        setOverview(null);
        setOverviewError(e?.message ?? "Failed to load dashboard");
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchOverview();
    api<{ exercises: ExerciseMeta[] }>("/api/exercises")
      .then((d) => setExercises(d.exercises))
      .catch(() => setExercises([]));
    api<DistributionRow[]>("/api/dashboard/distribution")
      .then((rows) => setDistribution(Array.isArray(rows) ? rows : []))
      .catch(() => setDistribution([]));
    dashboardApi
      .heatmap(84)
      .then((r) => setHeatmap(r.cells))
      .catch(() => setHeatmap([]));
  }, [fetchOverview]);

  // Real-time sync: when any session completes, refetch overview + heatmap.
  useSessionCompleted(
    useCallback(() => {
      fetchOverview();
      dashboardApi.heatmap(84).then((r) => setHeatmap(r.cells)).catch(() => {});
    }, [fetchOverview]),
  );

  // Exercises that have data in this user's history
  const chips = useMemo(() => {
    const counts = new Map<string, number>();
    for (const r of distribution) counts.set(r.exercise, r.count);
    return exercises.map((e) => ({
      id: e.id,
      display_name: e.display_name,
      has_data: (counts.get(e.id) ?? 0) > 0,
    }));
  }, [exercises, distribution]);

  const setExercise = useCallback(
    (id: string | null) => {
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev);
        if (id) next.set("exercise", id);
        else next.delete("exercise");
        return next;
      }, { replace: true });
    },
    [setSearchParams],
  );

  const setFilterDate = useCallback(
    (d: string | null) => {
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev);
        if (d) next.set("day", d);
        else next.delete("day");
        return next;
      }, { replace: true });
    },
    [setSearchParams],
  );

  const openSession = useCallback(
    (id: number) => navigate(`/dashboard/session/${id}`),
    [navigate],
  );

  if (loading) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center text-[color:var(--color-ink-4)] text-[11px] uppercase tracking-[0.24em]">
        Loading your training story…
      </div>
    );
  }

  if (overviewError || !overview) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center text-center px-6">
        <div>
          <p className="text-[color:var(--color-bad)] mb-4 uppercase tracking-[0.2em] text-xs">
            {overviewError ?? "Dashboard unavailable"}
          </p>
          <button
            onClick={fetchOverview}
            className="text-[11px] uppercase tracking-[0.24em] text-[color:var(--color-gold)]"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const totalSessions = overview.totals?.all_sessions ?? 0;
  const showEmptyState = totalSessions === 0;
  const selectedMeta = selectedExercise
    ? exercises.find((e) => e.id === selectedExercise)
    : null;

  return (
    <div className="max-w-[1440px] mx-auto px-6 md:px-10 pt-[calc(var(--nav-height)+2.5rem)] pb-24 space-y-10">
      <header>
        <div className="text-[10px] uppercase tracking-[0.24em] text-[color:var(--color-gold)]">
          Dashboard
        </div>
        <h1
          className="text-[color:var(--color-ink)] mt-2"
          style={{ fontFamily: "var(--font-display)", fontSize: "var(--fs-h1)", lineHeight: 0.95, letterSpacing: "0.03em" }}
        >
          {user?.display_name ? `${user.display_name}'s floor` : "Training floor"}
        </h1>
        <p
          className="text-[color:var(--color-ink-3)] italic mt-2"
          style={{ fontFamily: "var(--font-serif)", fontSize: "1.25rem" }}
        >
          {showEmptyState
            ? "A blank page — your first session writes the first line."
            : "Every rep you've ever logged, read back to you."}
        </p>
      </header>

      {showEmptyState ? (
        <section className="border border-[color:var(--rule)] rounded-sm bg-[color:var(--color-raised)]/60 px-10 py-20 text-center">
          <h2
            className="mb-4 text-[color:var(--color-ink)]"
            style={{ fontFamily: "var(--font-display)", fontSize: "2.5rem", letterSpacing: "0.04em" }}
          >
            LOG YOUR FIRST SESSION
          </h2>
          <p
            className="italic text-[color:var(--color-ink-3)] mb-8 max-w-xl mx-auto"
            style={{ fontFamily: "var(--font-serif)", fontSize: "1.2rem" }}
          >
            FORMA reads the form off your body, not a form you fill in. Pick an exercise and train — the dashboard writes itself.
          </p>
          <button
            onClick={() => navigate("/exercises")}
            className="px-8 py-3 bg-[color:var(--color-gold)] text-[color:var(--color-page)] text-[11px] uppercase tracking-[0.24em] hover:bg-[color:var(--color-gold-hover)] transition-colors rounded-sm"
          >
            Start training
          </button>
        </section>
      ) : (
        <>
          <TodaysPlanStrip />

          <TodayRibbon today={overview.today} wow={overview.wow_deltas} />

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <InsightCard insights={overview.top_insights} />
            </div>
            <div className="lg:col-span-1 space-y-6">
              <ActiveGoalsCard />
              <MuscleBalanceRadar groups={overview.muscle_balance} />
            </div>
          </div>

          <PersonalRecordsStrip records={overview.personal_records} />

          <FormTrendChart exercises={chips} />

          <ActivityHeatmap
            cells={heatmap}
            activeDate={filterDate}
            onPickDay={setFilterDate}
          />

          <ExerciseChipRow
            exercises={chips}
            selected={selectedExercise}
            onSelect={setExercise}
          />

          {selectedExercise && selectedMeta && (
            <DeepDivePanel
              exercise={selectedExercise}
              displayName={selectedMeta.display_name}
              onOpenSession={openSession}
            />
          )}

          <SessionHistoryList filterDate={filterDate} onOpen={openSession} />
        </>
      )}
      <PersonalCoachPanel />
    </div>
  );
}
