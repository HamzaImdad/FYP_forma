// Drill-down view for a single session. Rendered at /dashboard/session/:id.

import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { dashboardApi, type SessionDetail } from "../../lib/dashboardApi";

function fmtDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

function severityColor(score: number): string {
  if (score >= 0.75) return "var(--color-good)";
  if (score >= 0.55) return "var(--color-warn)";
  return "var(--color-bad)";
}

export function SessionDetailPanel() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [data, setData] = useState<SessionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedRep, setExpandedRep] = useState<number | null>(null);

  useEffect(() => {
    if (!id) return;
    setLoading(true);
    setError(null);
    dashboardApi
      .sessionDetail(Number(id))
      .then(setData)
      .catch((e) => setError(e?.message ?? "failed to load"))
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center text-[color:var(--color-ink-4)] text-[11px] uppercase tracking-[0.24em]">
        Loading session…
      </div>
    );
  }
  if (error || !data) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center text-center">
        <div>
          <p className="text-[color:var(--color-bad)] mb-4">
            {error ?? "Session not found"}
          </p>
          <Link
            to="/dashboard"
            className="text-[11px] uppercase tracking-[0.24em] text-[color:var(--color-gold)]"
          >
            Back to dashboard
          </Link>
        </div>
      </div>
    );
  }

  const pct = Math.round((data.avg_form_score ?? 0) * 100);
  const reps = data.reps ?? [];
  const sets = data.sets ?? [];

  return (
    <div className="max-w-[1100px] mx-auto px-6 md:px-10 pt-[calc(var(--nav-height)+3rem)] pb-24">
      <button
        onClick={() => navigate(-1)}
        className="text-[11px] uppercase tracking-[0.24em] text-[color:var(--color-ink-3)] hover:text-[color:var(--color-gold)] mb-6"
      >
        ← Back
      </button>

      <header className="mb-10">
        <div className="text-[10px] uppercase tracking-[0.24em] text-[color:var(--color-gold)]">
          Session #{data.id} · {fmtDate(data.date)}
        </div>
        <h1
          className="text-[color:var(--color-ink)] mt-3"
          style={{ fontFamily: "var(--font-display)", fontSize: "3.5rem", lineHeight: 0.95, letterSpacing: "0.04em" }}
        >
          {data.exercise.replace(/_/g, " ").toUpperCase()}
        </h1>
        <div className="mt-5 flex flex-wrap gap-8 text-[11px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)]">
          <span>{data.total_reps} reps</span>
          <span>{Math.round(data.duration_sec)}s</span>
          <span>{reps.length} tracked reps</span>
          <span>{sets.length} sets</span>
          <span style={{ color: severityColor(data.avg_form_score ?? 0) }}>
            {pct}/100 avg form
          </span>
          {data.consistency_score ? (
            <span>{Math.round((data.consistency_score ?? 0) * 100)}% consistent</span>
          ) : null}
        </div>
      </header>

      {sets.length > 0 && (
        <section className="mb-10">
          <h2
            className="mb-4 text-[color:var(--color-ink)]"
            style={{ fontFamily: "var(--font-display)", fontSize: "1.4rem", letterSpacing: "0.06em" }}
          >
            Sets
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {sets.map((s) => (
              <div
                key={s.id}
                className="border border-[color:var(--rule)] px-4 py-4 rounded-sm bg-[color:var(--color-raised)]/60"
              >
                <div className="text-[10px] uppercase tracking-[0.2em] text-[color:var(--color-gold)]">
                  Set {s.set_num}
                </div>
                <div
                  className="mt-2 text-[color:var(--color-ink)]"
                  style={{ fontFamily: "var(--font-display)", fontSize: "1.8rem", lineHeight: 0.95 }}
                >
                  {s.reps_count} reps
                </div>
                {s.avg_form_score !== undefined && s.avg_form_score !== null && (
                  <div className="text-[11px] text-[color:var(--color-ink-3)] mt-2">
                    {Math.round((s.avg_form_score ?? 0) * 100)}/100 avg ·{" "}
                    {Math.round(s.rest_before_sec ?? 0)}s rest
                  </div>
                )}
                {s.failure_type && (
                  <div className="text-[9px] uppercase tracking-[0.18em] text-[color:var(--color-ink-4)] mt-1">
                    {s.failure_type.replace(/_/g, " ")}
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      <section>
        <h2
          className="mb-4 text-[color:var(--color-ink)]"
          style={{ fontFamily: "var(--font-display)", fontSize: "1.4rem", letterSpacing: "0.06em" }}
        >
          Reps
        </h2>
        {reps.length === 0 ? (
          <p
            className="italic text-[color:var(--color-ink-3)]"
            style={{ fontFamily: "var(--font-serif)", fontSize: "1.1rem" }}
          >
            No per-rep telemetry for this session.
          </p>
        ) : (
          <ul className="divide-y divide-[color:var(--rule)]">
            {reps.map((r) => {
              const rpct = Math.round((r.form_score ?? 0) * 100);
              const isOpen = expandedRep === r.id;
              return (
                <li key={r.id}>
                  <button
                    onClick={() => setExpandedRep(isOpen ? null : r.id)}
                    className="w-full grid grid-cols-12 py-3 gap-3 text-left hover:bg-[color:var(--color-raised)]/50 transition-colors px-3 rounded-sm"
                  >
                    <span
                      className="col-span-1 text-[color:var(--color-gold)]"
                      style={{ fontFamily: "var(--font-display)", fontSize: "1.1rem" }}
                    >
                      #{r.rep_num}
                    </span>
                    <span className="col-span-2 text-[11px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)]">
                      set {r.set_num ?? "—"}
                    </span>
                    <span
                      className="col-span-2 text-sm"
                      style={{ color: severityColor(r.form_score ?? 0) }}
                    >
                      {rpct}/100
                    </span>
                    <span className="col-span-2 text-sm text-[color:var(--color-ink-3)]">
                      ecc {r.ecc_sec ? r.ecc_sec.toFixed(2) : "—"}s
                    </span>
                    <span className="col-span-2 text-sm text-[color:var(--color-ink-3)]">
                      con {r.con_sec ? r.con_sec.toFixed(2) : "—"}s
                    </span>
                    <span className="col-span-3 text-sm text-[color:var(--color-ink-3)]">
                      {r.peak_angle != null ? `${r.peak_angle.toFixed(1)}°` : ""}{" "}
                      {r.issues?.length > 0 && (
                        <em className="text-[color:var(--color-bad)]">
                          · {r.issues[0]}
                        </em>
                      )}
                    </span>
                  </button>
                  {isOpen && r.issues.length > 0 && (
                    <div className="px-3 pb-3 text-[color:var(--color-ink-2)] text-sm">
                      {r.issues.map((i, ix) => (
                        <div key={ix} className="italic" style={{ fontFamily: "var(--font-serif)" }}>
                          — {i}
                        </div>
                      ))}
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </section>
    </div>
  );
}
