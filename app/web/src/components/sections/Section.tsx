import type { ReactNode } from "react";

type Variant = "light" | "raised" | "dark";

const VARIANTS: Record<Variant, string> = {
  light: "bg-[color:var(--color-page)] text-[color:var(--color-ink)]",
  raised:
    "bg-[color:var(--color-raised)] text-[color:var(--color-ink)] border-y border-[color:var(--rule)]",
  dark: "bg-[color:var(--color-contrast)] text-[color:var(--color-ink-on-dark)]",
};

export function Section({
  children,
  variant = "light",
  className = "",
  innerClassName = "",
  id,
}: {
  children: ReactNode;
  variant?: Variant;
  className?: string;
  innerClassName?: string;
  id?: string;
}) {
  return (
    <section id={id} className={`relative ${VARIANTS[variant]} ${className}`}>
      <div className={`mx-auto max-w-[1440px] px-6 md:px-10 py-32 ${innerClassName}`}>
        {children}
      </div>
    </section>
  );
}
