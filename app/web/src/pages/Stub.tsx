import { Link } from "react-router-dom";

export function Stub({
  title,
  phase,
  description,
}: {
  title: string;
  phase: string;
  description: string;
}) {
  return (
    <main className="relative z-[2] min-h-screen flex items-center justify-center px-6 pt-[72px]">
      <div className="max-w-3xl w-full text-center">
        <span className="block text-xs uppercase tracking-[0.24em] text-[color:var(--color-gold)] mb-6">
          {phase}
        </span>
        <h1 className="font-[family-name:var(--font-display)] text-[clamp(3rem,8vw,6rem)] leading-[0.92] text-[color:var(--color-ink)]">
          {title}
        </h1>
        <p className="mt-6 font-[family-name:var(--font-serif)] italic text-2xl text-[color:var(--color-ink-2)]">
          {description}
        </p>
        <div className="mt-12 flex items-center justify-center gap-4">
          <Link
            to="/"
            className="inline-flex items-center gap-2 px-7 py-3 border border-[color:var(--color-ink)]/20 text-xs uppercase tracking-[0.14em] text-[color:var(--color-ink-2)] hover:text-[color:var(--color-ink)] transition-colors"
          >
            ← Home
          </Link>
        </div>
      </div>
    </main>
  );
}
