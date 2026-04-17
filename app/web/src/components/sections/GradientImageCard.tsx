import type { ReactNode } from "react";
import { Link } from "react-router-dom";

type Aspect = "portrait" | "square" | "wide" | "tall";

const ASPECT: Record<Aspect, string> = {
  portrait: "aspect-[4/5]",
  square: "aspect-square",
  wide: "aspect-[16/10]",
  tall: "aspect-[3/4]",
};

/**
 * Designmonks-style photography card: a colored gradient container holds
 * stacked image frames with rounded corners and soft shadows. Text sits
 * above the stack. Use this for feature grids, case-study tiles, anywhere
 * we want a photo to feel curated rather than pasted.
 */
export function GradientImageCard({
  images,
  color,
  aspect = "portrait",
  title,
  body,
  href,
  kicker,
  className = "",
}: {
  images: string[];
  color: string;
  aspect?: Aspect;
  title?: ReactNode;
  body?: ReactNode;
  href?: string;
  kicker?: string;
  className?: string;
}) {
  const inner = (
    <>
      <div
        className={`relative rounded-2xl overflow-hidden p-3 ${ASPECT[aspect]}`}
        style={{
          background: `linear-gradient(160deg, ${color} 0%, ${color}cc 70%, ${color}99 100%)`,
        }}
      >
        <div className="absolute inset-3 flex flex-col gap-2">
          {images.map((src, i) => (
            <div
              key={`${src}-${i}`}
              className="relative flex-1 overflow-hidden rounded-[10px] shadow-[0_18px_40px_rgba(13,13,13,0.35)] ring-1 ring-white/10"
            >
              <img
                src={src}
                alt=""
                loading="lazy"
                className="absolute inset-0 h-full w-full object-cover transition-transform duration-[700ms] ease-[var(--ease-out-editorial)] group-hover:scale-[1.06]"
              />
            </div>
          ))}
        </div>
        <div
          className="absolute inset-0 pointer-events-none"
          aria-hidden="true"
          style={{
            background:
              "radial-gradient(ellipse 120% 80% at 50% 120%, rgba(13,13,13,0.45) 0%, transparent 70%)",
          }}
        />
      </div>
      {(kicker || title || body) && (
        <div className="mt-6">
          {kicker ? (
            <span className="block font-[family-name:var(--font-mono)] text-[0.65rem] uppercase tracking-[0.18em] text-[color:var(--color-gold)] mb-3">
              {kicker}
            </span>
          ) : null}
          {title ? (
            <h3 className="font-[family-name:var(--font-display)] text-2xl md:text-3xl leading-[0.95] text-[color:var(--color-ink)] mb-3">
              {title}
            </h3>
          ) : null}
          {body ? (
            <div className="text-[0.95rem] text-[color:var(--color-ink-2)] leading-[1.6]">
              {body}
            </div>
          ) : null}
          {href ? (
            <span className="mt-5 inline-flex items-center gap-2 text-xs uppercase tracking-[0.14em] text-[color:var(--color-gold)]">
              Explore <span aria-hidden>→</span>
            </span>
          ) : null}
        </div>
      )}
    </>
  );

  const base = `group block ${className}`;
  if (href) {
    return (
      <Link to={href} className={base}>
        {inner}
      </Link>
    );
  }
  return <div className={base}>{inner}</div>;
}
