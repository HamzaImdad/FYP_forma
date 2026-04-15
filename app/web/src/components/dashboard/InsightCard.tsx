import type { Insight } from "../../lib/dashboardApi";

type Props = {
  insights: Insight[];
  title?: string;
  emptyText?: string;
};

const SEVERITY_ACCENT: Record<Insight["severity"], string> = {
  celebrate: "var(--color-gold)",
  warn: "var(--color-orange)",
  notice: "var(--color-warn)",
  info: "var(--color-ink-2)",
};

const SEVERITY_LABEL: Record<Insight["severity"], string> = {
  celebrate: "WIN",
  warn: "FIX",
  notice: "NOTE",
  info: "FYI",
};

export function InsightCard({ insights, title = "What your body is telling us", emptyText }: Props) {
  return (
    <section className="relative bg-[color:var(--color-raised)]/60 border border-[color:var(--rule)] rounded-sm px-8 py-8">
      <div className="flex items-baseline justify-between mb-6">
        <h2
          className="text-[color:var(--color-ink)]"
          style={{ fontFamily: "var(--font-display)", fontSize: "1.6rem", letterSpacing: "0.06em" }}
        >
          {title}
        </h2>
        <span className="text-[10px] uppercase tracking-[0.24em] text-[color:var(--color-ink-4)]">
          Insights · {insights.length}
        </span>
      </div>
      {insights.length === 0 ? (
        <p
          className="text-[color:var(--color-ink-3)] italic"
          style={{ fontFamily: "var(--font-serif)", fontSize: "1.35rem", lineHeight: 1.4 }}
        >
          {emptyText ?? "Keep training and insights will appear here. We need a few sessions before we can tell you anything useful."}
        </p>
      ) : (
        <ul className="space-y-5">
          {insights.map((ins) => (
            <li
              key={ins.id}
              className="border-l-2 pl-5 py-1"
              style={{ borderColor: SEVERITY_ACCENT[ins.severity] }}
            >
              <div
                className="text-[10px] uppercase tracking-[0.24em] mb-1"
                style={{ color: SEVERITY_ACCENT[ins.severity] }}
              >
                {SEVERITY_LABEL[ins.severity]} · {ins.category}
              </div>
              <p
                className="text-[color:var(--color-ink)] italic leading-snug"
                style={{ fontFamily: "var(--font-serif)", fontSize: "1.35rem" }}
              >
                {ins.text}
              </p>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
