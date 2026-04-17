import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { dashboardApi, type SessionRow } from "../../lib/dashboardApi";

type Props = {
  filterDate: string | null; // yyyy-mm-dd
  onOpen: (id: number) => void;
};

type HistoryMode = "flat" | "grouped";

const PAGE = 15;

type ExerciseGroup = {
  exercise: string;
  sessions: SessionRow[];
  bestForm: number;
};

function bucketByExercise(rows: SessionRow[]): ExerciseGroup[] {
  const map = new Map<string, SessionRow[]>();
  for (const r of rows) {
    const k = r.exercise;
    const list = map.get(k) ?? [];
    list.push(r);
    map.set(k, list);
  }
  return [...map.entries()]
    .map(([exercise, sessions]) => ({
      exercise,
      sessions,
      bestForm: Math.max(...sessions.map((s) => s.avg_form_score ?? 0), 0),
    }))
    .sort((a, b) => b.sessions.length - a.sessions.length);
}

export function SessionHistoryList({ filterDate, onOpen }: Props) {
  const [rows, setRows] = useState<SessionRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sentinel = useRef<HTMLDivElement | null>(null);

  const [searchParams, setSearchParams] = useSearchParams();
  const mode: HistoryMode =
    searchParams.get("history") === "grouped" ? "grouped" : "flat";
  const setMode = (next: HistoryMode) => {
    const p = new URLSearchParams(searchParams);
    if (next === "flat") p.delete("history");
    else p.set("history", next);
    setSearchParams(p, { replace: true });
  };
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  // Reset when filter changes
  useEffect(() => {
    setRows([]);
    setDone(false);
    setError(null);
  }, [filterDate]);

  // Load next page
  useEffect(() => {
    if (done || loading) return;
    let cancelled = false;
    setLoading(true);
    dashboardApi
      .sessions(PAGE, rows.length)
      .then((batch) => {
        if (cancelled) return;
        if (batch.length < PAGE) setDone(true);
        setRows((prev) => [...prev, ...batch]);
      })
      .catch((e) => !cancelled && setError(e?.message ?? "failed"))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
    // deliberately exclude `rows.length` so this only fires on explicit paginate
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filterDate]);

  const filtered = filterDate
    ? rows.filter((r) => (r.date || "").startsWith(filterDate))
    : rows;

  const groups = useMemo(() => bucketByExercise(filtered), [filtered]);

  const renderRow = (s: SessionRow) => {
    const pct = Math.round((s.avg_form_score ?? 0) * 100);
    const incomplete = (s.total_reps ?? 0) === 0 && s.exercise !== "plank";
    return (
      <li key={s.id}>
        <button
          onClick={() => onOpen(s.id)}
          className="w-full grid grid-cols-12 gap-3 py-3 text-left hover:bg-[color:var(--color-page)] transition-colors px-2 rounded-sm"
        >
          <span className="col-span-3 text-[11px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)]">
            {new Date(s.date).toLocaleDateString(undefined, {
              month: "short",
              day: "numeric",
              hour: "numeric",
              minute: "2-digit",
            })}
          </span>
          <span
            className="col-span-3 text-[color:var(--color-ink)]"
            style={{
              fontFamily: "var(--font-display)",
              letterSpacing: "0.06em",
            }}
          >
            {s.exercise.replace(/_/g, " ").toUpperCase()}
          </span>
          <span className="col-span-2 text-[color:var(--color-ink-2)] text-sm">
            {incomplete ? "incomplete" : `${s.total_reps} reps`}
          </span>
          <span className="col-span-2 text-[color:var(--color-ink-2)] text-sm">
            {Math.round(s.duration_sec)}s
          </span>
          <span
            className={`col-span-2 text-sm text-right ${
              pct >= 75
                ? "text-[color:var(--color-good)]"
                : pct >= 55
                  ? "text-[color:var(--color-warn)]"
                  : "text-[color:var(--color-bad)]"
            }`}
          >
            {pct > 0 ? `${pct}/100` : "—"}
          </span>
        </button>
      </li>
    );
  };

  function loadMore() {
    if (done || loading) return;
    setLoading(true);
    dashboardApi
      .sessions(PAGE, rows.length)
      .then((batch) => {
        if (batch.length < PAGE) setDone(true);
        setRows((prev) => [...prev, ...batch]);
      })
      .catch((e) => setError(e?.message ?? "failed"))
      .finally(() => setLoading(false));
  }

  return (
    <section className="bg-[color:var(--color-raised)]/60 border border-[color:var(--rule)] rounded-sm px-6 py-7">
      <div className="flex items-baseline justify-between mb-4 flex-wrap gap-3">
        <h2
          className="text-[color:var(--color-ink)]"
          style={{ fontFamily: "var(--font-display)", fontSize: "1.6rem", letterSpacing: "0.06em" }}
        >
          Session history
        </h2>
        <div className="flex items-center gap-3">
          {filterDate && (
            <span className="text-[10px] uppercase tracking-[0.24em] text-[color:var(--color-gold)]">
              Filtered · {filterDate}
            </span>
          )}
          <div
            className="inline-flex border border-[color:var(--rule)] rounded-sm overflow-hidden"
            role="tablist"
            aria-label="History view mode"
          >
            <button
              type="button"
              role="tab"
              aria-selected={mode === "flat"}
              onClick={() => setMode("flat")}
              className={`px-3 py-1.5 text-[10px] uppercase tracking-[0.22em] transition-colors ${
                mode === "flat"
                  ? "bg-[color:var(--color-gold)] text-[color:var(--color-page)]"
                  : "text-[color:var(--color-ink-3)] hover:text-[color:var(--color-ink)]"
              }`}
            >
              Flat
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={mode === "grouped"}
              onClick={() => setMode("grouped")}
              className={`px-3 py-1.5 text-[10px] uppercase tracking-[0.22em] transition-colors border-l border-[color:var(--rule)] ${
                mode === "grouped"
                  ? "bg-[color:var(--color-gold)] text-[color:var(--color-page)]"
                  : "text-[color:var(--color-ink-3)] hover:text-[color:var(--color-ink)]"
              }`}
            >
              By exercise
            </button>
          </div>
        </div>
      </div>

      {error && <p className="text-[color:var(--color-bad)] text-sm mb-3">Error: {error}</p>}

      {filtered.length === 0 && !loading ? (
        <p
          className="italic text-[color:var(--color-ink-3)]"
          style={{ fontFamily: "var(--font-serif)", fontSize: "1.1rem" }}
        >
          {filterDate
            ? "No sessions on that day."
            : "No past sessions yet. Your history appears here as soon as you log one."}
        </p>
      ) : mode === "flat" ? (
        <ul className="divide-y divide-[color:var(--rule)]">
          {filtered.map(renderRow)}
        </ul>
      ) : (
        <ul className="divide-y divide-[color:var(--rule)]">
          {groups.map((g) => {
            const isOpen = expanded[g.exercise] ?? false;
            const pct = Math.round(g.bestForm * 100);
            return (
              <li key={g.exercise}>
                <button
                  type="button"
                  onClick={() =>
                    setExpanded((prev) => ({
                      ...prev,
                      [g.exercise]: !isOpen,
                    }))
                  }
                  className="w-full grid grid-cols-12 gap-3 py-3 text-left hover:bg-[color:var(--color-page)] transition-colors px-2 rounded-sm items-baseline"
                >
                  <span className="col-span-1 text-[color:var(--color-ink-3)]">
                    {isOpen ? "▾" : "▸"}
                  </span>
                  <span
                    className="col-span-5 text-[color:var(--color-ink)]"
                    style={{
                      fontFamily: "var(--font-display)",
                      letterSpacing: "0.06em",
                    }}
                  >
                    {g.exercise.replace(/_/g, " ").toUpperCase()}
                  </span>
                  <span className="col-span-3 text-[color:var(--color-ink-2)] text-sm">
                    {g.sessions.length}{" "}
                    {g.sessions.length === 1 ? "session" : "sessions"}
                  </span>
                  <span
                    className={`col-span-3 text-sm text-right ${
                      pct >= 75
                        ? "text-[color:var(--color-good)]"
                        : pct >= 55
                          ? "text-[color:var(--color-warn)]"
                          : "text-[color:var(--color-ink-3)]"
                    }`}
                  >
                    {pct > 0 ? `best ${pct}/100` : "—"}
                  </span>
                </button>
                {isOpen && (
                  <ul className="divide-y divide-[color:var(--rule)] pl-6 pb-2">
                    {g.sessions.map(renderRow)}
                  </ul>
                )}
              </li>
            );
          })}
        </ul>
      )}

      {!done && (
        <div ref={sentinel} className="pt-5 text-center">
          <button
            onClick={loadMore}
            disabled={loading}
            className="text-[11px] uppercase tracking-[0.24em] text-[color:var(--color-gold)] hover:text-[color:var(--color-gold-hover)] transition-colors disabled:opacity-50"
          >
            {loading ? "Loading…" : "Load more"}
          </button>
        </div>
      )}
    </section>
  );
}
