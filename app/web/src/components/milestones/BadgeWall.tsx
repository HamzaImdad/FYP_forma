// BadgeWall — 13-badge grid from /api/badges. Earned badges in gold with
// date, locked badges grayscaled with tooltip-style description.

import type { Badge } from "@/lib/plansApi";

type Props = {
  badges: Badge[];
};

function relative(iso: string | null): string {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  } catch {
    return "";
  }
}

export function BadgeWall({ badges }: Props) {
  return (
    <div className="border border-[color:var(--rule)] rounded-[4px] bg-[color:var(--color-raised)]/40">
      <div className="px-5 pt-5 pb-3 border-b border-[color:var(--rule)]">
        <h2
          className="text-[color:var(--color-ink)]"
          style={{
            fontFamily: "var(--font-display)",
            fontSize: "1.6rem",
            letterSpacing: "0.04em",
          }}
        >
          BADGE WALL
        </h2>
        <p
          className="italic text-[color:var(--color-ink-3)] text-[13px] mt-1"
          style={{ fontFamily: "var(--font-serif)" }}
        >
          {badges.filter((b) => b.earned).length} of {badges.length} earned so far.
        </p>
      </div>
      <ul className="p-5 grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
        {badges.map((b) => (
          <li
            key={b.badge_key}
            className={`border rounded-[4px] p-4 flex flex-col gap-1 transition-opacity ${
              b.earned
                ? "border-[color:var(--color-gold)] bg-[color:var(--color-gold)]/10"
                : "border-[color:var(--rule)] bg-[color:var(--color-page)] opacity-55"
            }`}
            title={b.description}
          >
            <div
              className={`${
                b.earned
                  ? "text-[color:var(--color-gold)]"
                  : "text-[color:var(--color-ink-3)]"
              }`}
              style={{
                fontFamily: "var(--font-display)",
                fontSize: "1.1rem",
                letterSpacing: "0.06em",
              }}
            >
              {b.title.toUpperCase()}
            </div>
            <p
              className="italic text-[color:var(--color-ink-3)] text-[11px] leading-snug"
              style={{ fontFamily: "var(--font-serif)" }}
            >
              {b.description}
            </p>
            {b.earned && b.earned_at && (
              <div className="text-[10px] uppercase tracking-[0.16em] text-[color:var(--color-gold)] mt-auto pt-1">
                earned {relative(b.earned_at)}
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
