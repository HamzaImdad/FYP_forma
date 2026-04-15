import { PolarArea } from "react-chartjs-2";
import "../../lib/chartRegistry";
import { FORMA_CHART_COLORS as C } from "../../lib/chartRegistry";

type Props = {
  groups: Record<string, number>;
};

const ORDER = ["chest", "back", "shoulders", "arms", "legs", "core"];

export function MuscleBalanceRadar({ groups }: Props) {
  const total = ORDER.reduce((a, k) => a + (groups[k] || 0), 0);
  if (total === 0) {
    return (
      <section className="bg-[color:var(--color-raised)]/60 border border-[color:var(--rule)] rounded-sm px-8 py-8">
        <h2
          className="text-[color:var(--color-ink)] mb-3"
          style={{ fontFamily: "var(--font-display)", fontSize: "1.6rem", letterSpacing: "0.06em" }}
        >
          Muscle balance
        </h2>
        <p
          className="italic text-[color:var(--color-ink-3)]"
          style={{ fontFamily: "var(--font-serif)", fontSize: "1.1rem" }}
        >
          No sessions in the last 7 days — radar appears after your next workout.
        </p>
      </section>
    );
  }

  const values = ORDER.map((g) => groups[g] || 0);
  const gap = ORDER.find((g) => (groups[g] || 0) === 0);

  return (
    <section className="bg-[color:var(--color-raised)]/60 border border-[color:var(--rule)] rounded-sm px-8 py-8">
      <div className="flex items-baseline justify-between mb-3">
        <h2
          className="text-[color:var(--color-ink)]"
          style={{ fontFamily: "var(--font-display)", fontSize: "1.6rem", letterSpacing: "0.06em" }}
        >
          Muscle balance · 7d
        </h2>
        {gap && (
          <span className="text-[10px] uppercase tracking-[0.24em] text-[color:var(--color-bad)]">
            Gap · {gap}
          </span>
        )}
      </div>
      <div className="h-72">
        <PolarArea
          data={{
            labels: ORDER.map((g) => g.toUpperCase()),
            datasets: [
              {
                data: values,
                backgroundColor: [
                  "rgba(184,134,74,0.55)",
                  "rgba(212,69,20,0.55)",
                  "rgba(184,134,74,0.35)",
                  "rgba(212,69,20,0.35)",
                  "rgba(184,134,74,0.65)",
                  "rgba(212,69,20,0.65)",
                ],
                borderColor: C.ruleStrong,
                borderWidth: 1,
              },
            ],
          }}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: {
                position: "right",
                labels: { color: C.ink2, font: { size: 11 }, boxWidth: 12 },
              },
              tooltip: { backgroundColor: C.ink },
            },
            scales: {
              r: {
                ticks: { display: false },
                grid: { color: C.rule },
                angleLines: { color: C.rule },
              },
            },
          }}
        />
      </div>
    </section>
  );
}
