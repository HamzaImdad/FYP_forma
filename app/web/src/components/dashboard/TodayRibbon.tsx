import type { DashboardOverview, KpiDelta } from "../../lib/dashboardApi";

type Props = {
  today: DashboardOverview["today"];
  wow: DashboardOverview["wow_deltas"];
};

function formatTime(sec: number): string {
  const s = Math.round(sec);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const r = s % 60;
  if (m < 60) return `${m}m ${r.toString().padStart(2, "0")}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${(m % 60).toString().padStart(2, "0")}m`;
}

function formatPct(pct: number | null): string {
  if (pct === null || !isFinite(pct)) return "—";
  const sign = pct > 0 ? "▲" : pct < 0 ? "▼" : "·";
  return `${sign} ${Math.abs(Math.round(pct))}%`;
}

function deltaClass(value: number, better: "higher" | "lower"): string {
  if (value === 0) return "text-[color:var(--color-ink-3)]";
  const good = better === "higher" ? value > 0 : value < 0;
  return good ? "text-[color:var(--color-good)]" : "text-[color:var(--color-bad)]";
}

type TileProps = {
  label: string;
  value: string;
  subvalue?: string;  // Phase 4 — "NN good" under the big reps number
  delta: KpiDelta;
  deltaLabel?: string;
  better?: "higher" | "lower";
};

function Tile({ label, value, subvalue, delta, deltaLabel, better = "higher" }: TileProps) {
  return (
    <div className="relative px-8 py-7 bg-[color:var(--color-raised)]/60 border border-[color:var(--rule)] rounded-sm overflow-hidden">
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-[color:var(--color-gold-soft)]/60 to-transparent" />
      <div className="text-[10px] uppercase tracking-[0.24em] text-[color:var(--color-ink-3)]">
        {label}
      </div>
      <div
        className="mt-3 text-[color:var(--color-ink)]"
        style={{ fontFamily: "var(--font-display)", fontSize: "3.5rem", lineHeight: 0.9 }}
      >
        {value}
      </div>
      {subvalue && (
        <div className="mt-1 text-[11px] uppercase tracking-[0.18em] text-[color:var(--color-gold)]">
          {subvalue}
        </div>
      )}
      <div className={`mt-3 text-[11px] uppercase tracking-[0.2em] font-medium ${deltaClass(delta.delta, better)}`}>
        {formatPct(delta.pct)}
        <span className="ml-2 text-[color:var(--color-ink-4)] normal-case tracking-[0.1em]">
          {deltaLabel ?? "vs last week"}
        </span>
      </div>
    </div>
  );
}

export function TodayRibbon({ today, wow }: Props) {
  const scorePts = Math.round((today.avg_form_score ?? 0) * 100);
  return (
    <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      <Tile
        label="Reps today"
        value={String(today.reps)}
        subvalue={
          typeof today.good_reps === "number"
            ? `${today.good_reps} good`
            : undefined
        }
        delta={wow.reps}
      />
      <Tile
        label="Avg form"
        value={today.avg_form_score ? `${scorePts}` : "—"}
        delta={{
          current: wow.form.current * 100,
          previous: wow.form.previous * 100,
          delta: wow.form.delta * 100,
          pct: wow.form.pct,
        }}
      />
      <Tile
        label="Time trained"
        value={formatTime(today.time_sec)}
        delta={wow.time}
      />
      <Tile
        label="Streak"
        value={`${today.streak_days}d`}
        delta={{
          current: today.streak_days,
          previous: 0,
          delta: today.streak_days,
          pct: null,
        }}
        deltaLabel="active streak"
      />
    </section>
  );
}
