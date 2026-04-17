import { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";
import "../../lib/chartRegistry";
import { FORMA_CHART_COLORS as C } from "../../lib/chartRegistry";
import { api } from "../../lib/api";

type DailyScores = {
  dates: string[];
  scores: number[];
  reps: number[];
  session_counts: number[];
};

type Props = {
  exercises: { id: string; display_name: string; has_data: boolean }[];
};

export function FormTrendChart({ exercises }: Props) {
  const [selected, setSelected] = useState<string>("");
  const [data, setData] = useState<DailyScores | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    const q = selected ? `?days=30&exercise=${selected}` : "?days=30";
    api<DailyScores>(`/api/dashboard/scores${q}`)
      .then((d) => setData(d))
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [selected]);

  const total = data?.dates?.length ?? 0;

  return (
    <section className="bg-[color:var(--color-raised)]/60 border border-[color:var(--rule)] rounded-sm px-8 py-8">
      <div className="flex items-baseline justify-between flex-wrap gap-3 mb-6">
        <div>
          <h2
            className="text-[color:var(--color-ink)]"
            style={{ fontFamily: "var(--font-display)", fontSize: "1.6rem", letterSpacing: "0.06em" }}
          >
            Form trend
          </h2>
          <p
            className="text-[color:var(--color-ink-3)] italic mt-1"
            style={{ fontFamily: "var(--font-serif)", fontSize: "1rem" }}
          >
            30-day rolling average · baseline 70
          </p>
        </div>
        <select
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          className="bg-[color:var(--color-page)] border border-[color:var(--rule-strong)] px-4 py-2 text-[11px] uppercase tracking-[0.2em] text-[color:var(--color-ink)] focus:outline-none focus:border-[color:var(--color-gold)] rounded-sm"
        >
          <option value="" style={{ color: "#0A0A0A", background: "#F5F5F5" }}>All exercises</option>
          {exercises.filter((e) => e.has_data).map((e) => (
            <option key={e.id} value={e.id} style={{ color: "#0A0A0A", background: "#F5F5F5" }}>
              {e.display_name}
            </option>
          ))}
        </select>
      </div>

      {loading ? (
        <div className="h-64 flex items-center justify-center text-[color:var(--color-ink-4)] text-[11px] uppercase tracking-[0.2em]">
          Loading trend…
        </div>
      ) : total < 2 ? (
        <div className="h-64 flex items-center justify-center">
          <p
            className="text-[color:var(--color-ink-3)] italic text-center"
            style={{ fontFamily: "var(--font-serif)", fontSize: "1.15rem" }}
          >
            {total === 0
              ? "No training days yet. Finish a session to begin your trend."
              : `${2 - total} more session${2 - total === 1 ? "" : "s"} unlock your form trend.`}
          </p>
        </div>
      ) : (
        <div className="h-64">
          <Line
            data={{
              labels: data!.dates.map((d) =>
                new Date(d).toLocaleDateString(undefined, { month: "short", day: "numeric" }),
              ),
              datasets: [
                {
                  label: "Form score",
                  data: data!.scores.map((s) => Math.round(s * 100)),
                  borderColor: C.gold,
                  backgroundColor: "rgba(174,231,16,0.15)",
                  fill: true,
                  tension: 0.3,
                  pointRadius: 4,
                  pointBackgroundColor: C.gold,
                  pointHoverRadius: 6,
                  borderWidth: 2.5,
                },
                {
                  label: "Baseline",
                  data: data!.dates.map(() => 70),
                  borderColor: C.ink4,
                  borderDash: [4, 6],
                  borderWidth: 1,
                  pointRadius: 0,
                  fill: false,
                },
              ],
            }}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: { display: false },
                tooltip: {
                  backgroundColor: C.ink,
                  titleColor: "#F0F0F0",
                  bodyColor: "#F0F0F0",
                  padding: 10,
                  cornerRadius: 2,
                },
              },
              scales: {
                y: {
                  min: 0,
                  max: 100,
                  grid: { color: C.rule },
                  ticks: { color: C.ink3, font: { size: 11 } },
                },
                x: {
                  grid: { display: false },
                  ticks: { color: C.ink3, font: { size: 11 }, maxRotation: 0 },
                },
              },
            }}
          />
        </div>
      )}
    </section>
  );
}
