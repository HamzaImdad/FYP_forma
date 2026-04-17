import type { ReactNode } from "react";
import { Link } from "react-router-dom";

export function FeatureCard({
  index,
  title,
  body,
  image,
  href,
  onDark = false,
}: {
  index?: string;
  title: string;
  body: ReactNode;
  image?: string;
  href?: string;
  onDark?: boolean;
}) {
  const inkMain = onDark ? "var(--color-ink-on-dark)" : "var(--color-ink)";
  const inkBody = onDark ? "var(--color-ink-on-dark-2)" : "var(--color-ink-2)";
  const gold = onDark ? "var(--color-gold-soft)" : "var(--color-gold)";
  const surface = onDark
    ? "bg-[color:var(--color-contrast)]/60 border-[color:var(--color-ink-on-dark)]/10"
    : "bg-[color:var(--color-page)] border-[color:var(--rule)]";

  const inner = (
    <>
      {image ? (
        <div className="relative aspect-[4/3] overflow-hidden rounded-[4px] mb-6 bg-[color:var(--color-sunken)]">
          <img
            src={image}
            alt=""
            loading="lazy"
            className="h-full w-full object-cover transition-transform duration-700 ease-[var(--ease-out-editorial)] group-hover:scale-[1.04]"
          />
          {onDark ? (
            <div className="absolute inset-0 bg-gradient-to-t from-[color:var(--color-contrast)]/60 to-transparent" />
          ) : null}
        </div>
      ) : null}
      {index ? (
        <span
          className="block font-[family-name:var(--font-mono)] text-xs mb-3"
          style={{ color: gold }}
        >
          {index}
        </span>
      ) : null}
      <h3
        className="font-[family-name:var(--font-display)] text-2xl md:text-3xl leading-[0.95] mb-4"
        style={{ color: inkMain }}
      >
        {title}
      </h3>
      <div
        className="text-[0.95rem] leading-[1.6]"
        style={{ color: inkBody }}
      >
        {body}
      </div>
      {href ? (
        <span
          className="mt-6 inline-flex items-center gap-2 text-xs uppercase tracking-[0.14em] transition-colors"
          style={{ color: gold }}
        >
          Explore <span aria-hidden>→</span>
        </span>
      ) : null}
    </>
  );

  const base = `group block p-8 rounded-[6px] border transition-colors ${surface}`;

  if (href) {
    return (
      <Link to={href} data-reveal className={base}>
        {inner}
      </Link>
    );
  }
  return (
    <div data-reveal className={base}>
      {inner}
    </div>
  );
}
