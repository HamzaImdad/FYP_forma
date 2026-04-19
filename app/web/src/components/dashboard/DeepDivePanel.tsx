// Per-exercise deep dive. Lazy-loads /api/dashboard/exercise/<exercise> when
// a chip is clicked, then renders a stack of compact charts: issue ranking,
// rep-quality stacked bars, tempo scatter, depth ladder, fatigue curve, and a
// captioned insight list. Swaps in HoldTimeline for plank instead of
// tempo/depth.

import { useEffect, useState } from "react";
import { Bar, Line, Scatter } from "react-chartjs-2";
import "../../lib/chartRegistry";
import { FORMA_CHART_COLORS as C } from "../../lib/chartRegistry";
import { dashboardApi, type ExerciseDeepDive } from "../../lib/dashboardApi";
import { InsightCard } from "./InsightCard";

type Props = {
  exercise: string;
  displayName: string;
  onOpenSession: (id: number) => void;
};

export function DeepDivePanel({ exercise, displayName, onOpenSession }: Props) {
  const [data, setData] = useState<ExerciseDeepDive | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    setData(null);
    dashboardApi
      .exercise(exercise)
      .then(setData)
      .catch((e) => setError(e?.message ?? "failed to load"))
      .finally(() => setLoading(false));
  }, [exercise]);

  if (loading) {
    return (
      <div className="border border-[color:var(--rule)] rounded-sm p-10 text-center text-[color:var(--color-ink-4)] text-[11px] uppercase tracking-[0.24em]">
        Loading {displayName} detail…
      </div>
    );
  }
  if (error || !data) {
    return (
      <div className="border border-[color:var(--rule)] rounded-sm p-10 text-center text-[color:var(--color-bad)]">
        Could not load detail ({error ?? "unknown"}).
      </div>
    );
  }

  // Static holds (plank + side_plank) — show hold timeline instead of
  // tempo/depth charts.
  const isPlank = exercise === "plank" || exercise === "side_plank";
  const hasReps = data.scores.some((s) => s.total_reps > 0);
  if (!hasReps && !isPlank) {
    return (
      <div className="border border-[color:var(--rule)] rounded-sm p-10 text-center">
        <p
          className="italic text-[color:var(--color-ink-3)]"
          style={{ fontFamily: "var(--font-serif)", fontSize: "1.25rem" }}
        >
          No {displayName} sessions in the last 60 days yet. Train one to unlock the detail view.
        </p>
      </div>
    );
  }

  return (
    <section className="space-y-6 bg-[color:var(--color-raised)]/40 border border-[color:var(--rule)] rounded-sm p-6 md:p-8">
      <header className="flex items-baseline justify-between flex-wrap gap-3">
        <div>
          <div className="text-[10px] uppercase tracking-[0.24em] text-[color:var(--color-gold)]">
            Deep dive
          </div>
          <h2
            className="text-[color:var(--color-ink)]"
            style={{ fontFamily: "var(--font-display)", fontSize: "2.2rem", letterSpacing: "0.04em" }}
          >
            {displayName}
          </h2>
        </div>
        <div className="text-[10px] uppercase tracking-[0.24em] text-[color:var(--color-ink-4)]">
          {data.scores.length} sessions · {data.muscles.join(", ") || "unmapped"}
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <IssueRanking issues={data.top_issues} />
        <RepQualityStack
          series={data.quality_breakdown}
          onOpenSession={onOpenSession}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {isPlank ? (
          <HoldTimeline
            scores={data.scores}
            onOpenSession={onOpenSession}
          />
        ) : (
          <>
            <TempoStrip tempo={data.tempo} />
            <DepthLadder depth={data.depth} />
          </>
        )}
        <FatigueCurveChart curve={data.fatigue_curve} />
      </div>

      <InsightCard
        insights={data.insights}
        title={`Reading your ${displayName.toLowerCase()}s`}
        emptyText={`We need a few more ${displayName.toLowerCase()} sessions before we can call anything out.`}
      />
    </section>
  );
}

// ── IssueRanking (horizontal bar) ──

function IssueRanking({ issues }: { issues: { issue: string; count: number }[] }) {
  if (issues.length === 0) {
    return (
      <Card title="Top issues">
        <Empty text="No recurring issues detected. Clean so far." />
      </Card>
    );
  }
  return (
    <Card title="Top issues">
      <div className="h-48">
        <Bar
          data={{
            labels: issues.map((i) => i.issue),
            datasets: [
              {
                data: issues.map((i) => i.count),
                backgroundColor: C.orange,
                borderRadius: 2,
                barThickness: 18,
              },
            ],
          }}
          options={{
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { backgroundColor: C.ink } },
            scales: {
              x: { grid: { color: C.rule }, ticks: { color: C.ink3 } },
              y: { grid: { display: false }, ticks: { color: C.ink2, font: { size: 11 } } },
            },
          }}
        />
      </div>
    </Card>
  );
}

// ── RepQualityStack ──

function RepQualityStack({
  series,
  onOpenSession,
}: {
  series: ExerciseDeepDive["quality_breakdown"];
  onOpenSession: (id: number) => void;
}) {
  const data = series.slice(-15);
  if (data.length === 0) {
    return (
      <Card title="Rep quality per session">
        <Empty text="No quality data yet." />
      </Card>
    );
  }
  return (
    <Card title="Rep quality per session">
      <div className="h-48">
        <Bar
          data={{
            labels: data.map((d) =>
              new Date(d.date).toLocaleDateString(undefined, { month: "short", day: "numeric" }),
            ),
            datasets: [
              { label: "Good", data: data.map((d) => d.good), backgroundColor: C.good, stack: "q" },
              { label: "Moderate", data: data.map((d) => d.moderate), backgroundColor: C.warn, stack: "q" },
              { label: "Needs work", data: data.map((d) => d.bad), backgroundColor: C.bad, stack: "q" },
            ],
          }}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            onClick: (_evt, elements) => {
              if (elements[0]) {
                const idx = elements[0].index;
                onOpenSession(data[idx].session_id);
              }
            },
            onHover: (evt, elements) => {
              const target = evt.native?.target as HTMLElement | undefined;
              if (target) {
                target.style.cursor = elements[0] ? "pointer" : "default";
              }
            },
            plugins: {
              legend: { position: "bottom", labels: { boxWidth: 10, color: C.ink3, font: { size: 10 } } },
              tooltip: { backgroundColor: C.ink, callbacks: { footer: () => "Click to open session" } },
            },
            scales: {
              x: { stacked: true, grid: { display: false }, ticks: { color: C.ink3, maxRotation: 0 } },
              y: { stacked: true, grid: { color: C.rule }, ticks: { color: C.ink3 } },
            },
          }}
        />
      </div>
    </Card>
  );
}

// ── TempoStrip (scatter ecc/con) ──

function TempoStrip({ tempo }: { tempo: ExerciseDeepDive["tempo"] }) {
  if (tempo.length === 0) {
    return (
      <Card title="Tempo (eccentric vs concentric)">
        <Empty text="No tempo data captured yet. New sessions will populate this." />
      </Card>
    );
  }
  return (
    <Card title="Tempo">
      <p className="text-[10px] uppercase tracking-[0.2em] text-[color:var(--color-ink-4)] -mt-2 mb-3">
        Healthy band: 0.6–1.5s each way
      </p>
      <div className="h-48">
        <Scatter
          data={{
            datasets: [
              {
                label: "Reps",
                data: tempo.map((t) => ({ x: t.ecc_sec, y: t.con_sec })),
                backgroundColor: "rgba(174,231,16,0.7)",
                pointRadius: 5,
              },
            ],
          }}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { backgroundColor: C.ink } },
            scales: {
              x: {
                title: { display: true, text: "Eccentric (s)", color: C.ink3, font: { size: 10 } },
                min: 0,
                grid: { color: C.rule },
                ticks: { color: C.ink3 },
              },
              y: {
                title: { display: true, text: "Concentric (s)", color: C.ink3, font: { size: 10 } },
                min: 0,
                grid: { color: C.rule },
                ticks: { color: C.ink3 },
              },
            },
          }}
        />
      </div>
    </Card>
  );
}

// ── DepthLadder ──

function DepthLadder({ depth }: { depth: ExerciseDeepDive["depth"] }) {
  if (depth.length === 0) {
    return (
      <Card title="Depth (peak angle per rep)">
        <Empty text="No depth data captured yet. New sessions will populate this." />
      </Card>
    );
  }
  return (
    <Card title="Depth">
      <p className="text-[10px] uppercase tracking-[0.2em] text-[color:var(--color-ink-4)] -mt-2 mb-3">
        Lower = deeper range of motion
      </p>
      <div className="h-48">
        <Line
          data={{
            labels: depth.map((_, i) => i + 1),
            datasets: [
              {
                label: "Peak angle",
                data: depth.map((d) => d.peak_angle),
                borderColor: C.orange,
                backgroundColor: "rgba(174,231,16,0.1)",
                fill: true,
                tension: 0.3,
                pointRadius: 3,
                borderWidth: 2,
              },
            ],
          }}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { backgroundColor: C.ink } },
            scales: {
              x: { grid: { display: false }, ticks: { color: C.ink3 } },
              y: { grid: { color: C.rule }, ticks: { color: C.ink3 } },
            },
          }}
        />
      </div>
    </Card>
  );
}

// ── FatigueCurveChart ──

function FatigueCurveChart({ curve }: { curve: ExerciseDeepDive["fatigue_curve"] }) {
  if (curve.length < 2) {
    return (
      <Card title="Fatigue curve">
        <Empty text="Need at least two rep-position samples. Bank more sessions." />
      </Card>
    );
  }
  return (
    <Card title="Fatigue curve">
      <p className="text-[10px] uppercase tracking-[0.2em] text-[color:var(--color-ink-4)] -mt-2 mb-3">
        Avg form score by rep position across all sessions
      </p>
      <div className="h-48">
        <Line
          data={{
            labels: curve.map((c) => `#${c.rep_num}`),
            datasets: [
              {
                label: "Avg score",
                data: curve.map((c) => Math.round(c.avg_score * 100)),
                borderColor: C.gold,
                backgroundColor: "rgba(174,231,16,0.15)",
                fill: true,
                tension: 0.25,
                pointRadius: 2,
                borderWidth: 2,
              },
            ],
          }}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { backgroundColor: C.ink } },
            scales: {
              x: { grid: { display: false }, ticks: { color: C.ink3, maxRotation: 0 } },
              y: { min: 0, max: 100, grid: { color: C.rule }, ticks: { color: C.ink3 } },
            },
          }}
        />
      </div>
    </Card>
  );
}

// ── HoldTimeline (plank) ──

function HoldTimeline({
  scores,
  onOpenSession,
}: {
  scores: ExerciseDeepDive["scores"];
  onOpenSession: (id: number) => void;
}) {
  if (scores.length === 0) {
    return (
      <Card title="Hold timeline">
        <Empty text="No plank sessions yet." />
      </Card>
    );
  }
  const max = Math.max(...scores.map((s) => s.duration_sec), 1);
  return (
    <Card title="Hold timeline">
      <p className="text-[10px] uppercase tracking-[0.2em] text-[color:var(--color-ink-4)] -mt-2 mb-3">
        Bar length = duration. Color = avg form.
      </p>
      <ul className="space-y-2 max-h-56 overflow-y-auto pr-2">
        {scores.map((s) => {
          const width = (s.duration_sec / max) * 100;
          const pct = Math.round(s.avg_form_score * 100);
          const color =
            pct >= 75
              ? "var(--color-good)"
              : pct >= 55
                ? "var(--color-warn)"
                : "var(--color-bad)";
          return (
            <li key={s.session_id}>
              <button
                onClick={() => onOpenSession(s.session_id)}
                className="w-full text-left group"
              >
                <div className="flex items-baseline justify-between text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)] mb-1">
                  <span>{new Date(s.date).toLocaleDateString(undefined, { month: "short", day: "numeric" })}</span>
                  <span>
                    {Math.round(s.duration_sec)}s · {pct}/100
                  </span>
                </div>
                <div className="h-3 bg-[color:var(--color-sunken)] rounded-sm overflow-hidden">
                  <div
                    className="h-full transition-all group-hover:brightness-110"
                    style={{ width: `${width}%`, backgroundColor: color }}
                  />
                </div>
              </button>
            </li>
          );
        })}
      </ul>
    </Card>
  );
}

// ── Shared card wrapper ──

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-[color:var(--color-page)] border border-[color:var(--rule)] rounded-sm p-5">
      <h3
        className="mb-3 text-[color:var(--color-ink)]"
        style={{ fontFamily: "var(--font-display)", fontSize: "1.05rem", letterSpacing: "0.08em" }}
      >
        {title}
      </h3>
      {children}
    </div>
  );
}

function Empty({ text }: { text: string }) {
  return (
    <div className="h-40 flex items-center justify-center text-center">
      <p
        className="italic text-[color:var(--color-ink-4)]"
        style={{ fontFamily: "var(--font-serif)", fontSize: "1rem" }}
      >
        {text}
      </p>
    </div>
  );
}
