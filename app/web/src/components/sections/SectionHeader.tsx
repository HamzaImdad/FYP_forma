import type { ReactNode } from "react";

export function SectionHeader({
  eyebrow,
  title,
  italic,
  body,
  onDark = false,
  align = "left",
  className = "",
}: {
  eyebrow: string;
  title: ReactNode;
  italic?: ReactNode;
  body?: ReactNode;
  onDark?: boolean;
  align?: "left" | "center";
  className?: string;
}) {
  const gold = onDark ? "var(--color-gold-soft)" : "var(--color-gold)";
  const ink = onDark ? "var(--color-ink-on-dark)" : "var(--color-ink)";
  const ink2 = onDark ? "var(--color-ink-on-dark-2)" : "var(--color-ink-2)";
  const alignCls =
    align === "center" ? "mx-auto text-center items-center" : "text-left items-start";

  return (
    <header className={`flex flex-col max-w-3xl ${alignCls} ${className}`}>
      <span
        data-reveal
        className="block text-xs uppercase tracking-[0.24em] mb-4"
        style={{ color: gold }}
      >
        {eyebrow}
      </span>
      <h2
        data-reveal
        className="font-[family-name:var(--font-display)] text-[clamp(2.5rem,6vw,5rem)] leading-[0.92]"
        style={{ color: ink }}
      >
        {title}
        {italic ? (
          <>
            <br />
            <em
              className="not-italic font-[family-name:var(--font-serif)] italic"
              style={{ color: gold }}
            >
              {italic}
            </em>
          </>
        ) : null}
      </h2>
      {body ? (
        <p
          data-reveal
          className="mt-6 max-w-2xl leading-[1.6]"
          style={{ color: ink2 }}
        >
          {body}
        </p>
      ) : null}
    </header>
  );
}
