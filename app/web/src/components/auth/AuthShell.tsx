import type { ReactNode } from "react";
import { Link } from "react-router-dom";

type Props = {
  kicker: string;
  title: string;
  tagline: string;
  children: ReactNode;
  footer: ReactNode;
};

export function AuthShell({ kicker, title, tagline, children, footer }: Props) {
  return (
    <main className="relative min-h-screen flex items-center justify-center px-6 py-16 bg-[color:var(--color-page)]">
      {/* subtle editorial backdrop */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden>
        <div
          className="absolute -top-40 -left-40 w-[520px] h-[520px] rounded-full opacity-[0.08] blur-[120px]"
          style={{ background: "var(--color-gold)" }}
        />
        <div
          className="absolute -bottom-48 -right-48 w-[560px] h-[560px] rounded-full opacity-[0.06] blur-[140px]"
          style={{ background: "var(--color-orange)" }}
        />
      </div>

      <div className="relative z-[1] w-full max-w-[480px]">
        <Link
          to="/"
          className="block text-center text-[10px] uppercase tracking-[0.4em] text-[color:var(--color-ink-2)] hover:text-[color:var(--color-gold)] transition-colors"
        >
          ← FORMA
        </Link>

        <div className="mt-10 text-center">
          <span className="block text-[10px] uppercase tracking-[0.32em] text-[color:var(--color-gold)]">
            {kicker}
          </span>
          <h1 className="mt-4 font-[family-name:var(--font-display)] text-[clamp(3.2rem,7vw,4.8rem)] leading-[0.88] text-[color:var(--color-ink)] tracking-tight">
            {title}
          </h1>
          <p className="mt-3 font-[family-name:var(--font-serif)] italic text-[1.15rem] text-[color:var(--color-ink-2)]">
            {tagline}
          </p>
        </div>

        <div className="mt-12">{children}</div>

        <div className="mt-10 text-center font-[family-name:var(--font-serif)] italic text-[0.95rem] text-[color:var(--color-ink-2)]">
          {footer}
        </div>
      </div>
    </main>
  );
}

type FieldProps = {
  label: string;
  name: string;
  type?: string;
  value: string;
  onChange: (v: string) => void;
  autoComplete?: string;
  placeholder?: string;
  disabled?: boolean;
};

export function AuthField({
  label,
  name,
  type = "text",
  value,
  onChange,
  autoComplete,
  placeholder,
  disabled,
}: FieldProps) {
  return (
    <label className="block mb-7">
      <span className="block text-[10px] uppercase tracking-[0.22em] text-[color:var(--color-ink-2)] mb-2">
        {label}
      </span>
      <input
        type={type}
        name={name}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        autoComplete={autoComplete}
        placeholder={placeholder}
        disabled={disabled}
        className="w-full bg-transparent border-0 border-b-2 border-[color:var(--color-ink)]/25
                   py-3 text-[1.15rem] text-[color:var(--color-ink)]
                   placeholder:text-[color:var(--color-ink-2)]/50
                   focus:outline-none focus:border-[color:var(--color-gold)]
                   transition-colors font-[family-name:var(--font-sans)]"
      />
    </label>
  );
}

type SubmitProps = {
  label: string;
  loading?: boolean;
  disabled?: boolean;
};

export function AuthSubmit({ label, loading, disabled }: SubmitProps) {
  return (
    <button
      type="submit"
      disabled={disabled || loading}
      className="group w-full py-4 mt-2 bg-[color:var(--color-gold)] text-[color:var(--color-page)]
                 font-[family-name:var(--font-display)] text-[1.35rem] tracking-[0.06em]
                 uppercase transition-all duration-200
                 hover:bg-[color:var(--color-gold-hover)] hover:shadow-[0_8px_30px_-10px_rgba(184,134,74,0.5)]
                 disabled:opacity-50 disabled:cursor-not-allowed"
      style={{ transitionTimingFunction: "var(--ease-out-editorial)" }}
    >
      {loading ? (
        <span className="inline-block animate-pulse">…</span>
      ) : (
        <span>{label}</span>
      )}
    </button>
  );
}

export function AuthError({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <div
      className="mb-6 px-4 py-3 border-l-2 font-[family-name:var(--font-sans)] text-[0.9rem]"
      style={{
        borderColor: "var(--color-bad)",
        color: "var(--color-bad)",
        background: "rgba(184,52,28,0.06)",
      }}
    >
      {message}
    </div>
  );
}
