import type { DashboardOverview } from "../../lib/dashboardApi";

type Props = {
  records: DashboardOverview["personal_records"];
};

function fmtDate(iso: string | undefined): string {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  } catch {
    return "—";
  }
}

function fmtHold(sec: number | undefined): string {
  if (!sec) return "—";
  const s = Math.round(sec);
  const m = Math.floor(s / 60);
  const r = s % 60;
  return m ? `${m}:${r.toString().padStart(2, "0")}` : `${r}s`;
}

type TileProps = {
  label: string;
  value: string;
  caption: string;
};

function PRTile({ label, value, caption }: TileProps) {
  return (
    <div className="px-6 py-6 border border-[color:var(--rule)] rounded-sm bg-[color:var(--color-page)]">
      <div className="text-[10px] uppercase tracking-[0.24em] text-[color:var(--color-gold)] mb-2">
        PR · {label}
      </div>
      <div
        className="text-[color:var(--color-ink)]"
        style={{ fontFamily: "var(--font-display)", fontSize: "2.25rem", lineHeight: 0.95 }}
      >
        {value}
      </div>
      <div
        className="mt-3 text-[color:var(--color-ink-3)] italic"
        style={{ fontFamily: "var(--font-serif)", fontSize: "0.95rem" }}
      >
        {caption}
      </div>
    </div>
  );
}

export function PersonalRecordsStrip({ records }: Props) {
  const biggest = records.biggest_session;
  const bestForm = records.best_form_day;
  const plank = records.longest_plank;
  return (
    <section>
      <h2
        className="text-[color:var(--color-ink)] mb-4"
        style={{ fontFamily: "var(--font-display)", fontSize: "1.6rem", letterSpacing: "0.06em" }}
      >
        Personal records
      </h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <PRTile
          label="Biggest session"
          value={biggest ? `${biggest.total_reps}` : "—"}
          caption={biggest ? `${biggest.exercise.replace("_", " ")} · ${fmtDate(biggest.date)}` : "not yet recorded"}
        />
        <PRTile
          label="Cleanest form"
          value={bestForm ? `${Math.round((bestForm.avg_form_score ?? 0) * 100)}` : "—"}
          caption={bestForm ? `${bestForm.exercise.replace("_", " ")} · ${fmtDate(bestForm.date)}` : "not yet recorded"}
        />
        <PRTile
          label="Longest hold"
          value={plank ? fmtHold(plank.duration_sec) : "—"}
          caption={plank ? `plank · ${fmtDate(plank.date)}` : "no plank session yet"}
        />
        <PRTile
          label="Longest streak"
          value={`${records.longest_streak}d`}
          caption={records.longest_streak >= 2 ? "consecutive training days" : "train tomorrow to start one"}
        />
      </div>
    </section>
  );
}
