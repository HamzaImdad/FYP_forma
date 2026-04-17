import type { ReactNode } from "react";
import { VideoBackdrop } from "../sections/VideoBackdrop";

// 2-column split layout for /login (handles both sign-in and sign-up tabs).
// Nav is rendered by App.tsx (no longer in the fullscreen exclusion list), so
// AuthShell only owns the region below the 72px nav.
export function AuthShell({ children }: { children: ReactNode }) {
  return (
    <main className="relative grid md:grid-cols-2 bg-[color:var(--color-page)] pt-[64px] sm:pt-[72px] md:min-h-screen">
      {/* Mobile video banner — shows above the form on < md */}
      <div className="md:hidden relative h-[220px] w-full overflow-hidden">
        <VideoBackdrop
          src="/static/videos/auth-loop.mp4"
          poster="/static/images/hero_wide.jpg"
          anchor="bottom"
          intensity={0.72}
          className="absolute inset-0"
          objectPosition="center 30%"
        >
          <div className="absolute bottom-6 left-6 right-6 text-[color:var(--color-ink-on-dark)]">
            <span className="block text-[9px] uppercase tracking-[0.28em] text-[color:var(--color-gold-soft)]">
              Train with form.
            </span>
            <p className="mt-2 font-[family-name:var(--font-serif)] italic text-lg leading-tight">
              Quiet, honest, in your living room.
            </p>
          </div>
        </VideoBackdrop>
      </div>

      {/* Form column — creative gradient: bloom bleeding from the seam
          (right edge, matching where the video column starts) plus a subtle
          corner glow at the bottom-left and a faint vertical accent line. */}
      <div className="relative flex items-center justify-center px-6 md:px-12 py-16 md:py-20 overflow-hidden">
        {/* Large radial bleeding from the right seam — unifies the two columns */}
        <div
          aria-hidden="true"
          className="absolute top-1/2 -right-[15%] -translate-y-1/2 pointer-events-none rounded-full"
          style={{
            width: "720px",
            height: "720px",
            background:
              "radial-gradient(circle, rgba(174,231,16,0.22) 0%, rgba(174,231,16,0.08) 30%, transparent 65%)",
            filter: "blur(90px)",
          }}
        />
        {/* Secondary soft glow bottom-left */}
        <div
          aria-hidden="true"
          className="absolute -bottom-[20%] -left-[10%] pointer-events-none rounded-full"
          style={{
            width: "480px",
            height: "480px",
            background:
              "radial-gradient(circle, rgba(194,240,74,0.08) 0%, transparent 70%)",
            filter: "blur(80px)",
          }}
        />
        {/* Grain texture */}
        <div
          aria-hidden="true"
          className="absolute inset-0 pointer-events-none opacity-[0.15] mix-blend-overlay"
          style={{ backgroundImage: "var(--grain)" }}
        />
        <div className="relative w-full max-w-[460px]">{children}</div>
      </div>

      {/* Video column — desktop only. Left edge dissolves into the form
          column's dark via a heavy horizontal gradient so there's no seam. */}
      <div className="relative hidden md:block overflow-hidden">
        <VideoBackdrop
          src="/static/videos/auth-loop.mp4"
          poster="/static/images/hero_wide.jpg"
          anchor="right"
          intensity={0.62}
          className="absolute inset-0"
          objectPosition="center 30%"
        >
          <div className="absolute bottom-12 left-12 right-12 max-w-sm text-[color:var(--color-ink-on-dark)]">
            <span className="block text-[10px] uppercase tracking-[0.32em] text-[color:var(--color-gold-soft)]">
              Train with form.
            </span>
            <p className="mt-4 font-[family-name:var(--font-serif)] italic text-[1.65rem] leading-[1.15]">
              Quiet, honest, in your living room.
            </p>
            <p className="mt-6 text-[10px] uppercase tracking-[0.32em] font-mono text-[color:var(--color-ink-on-dark-2)]">
              FORMA · 2026
            </p>
          </div>
        </VideoBackdrop>

        {/* Seam-dissolve: heavy black fade on the left edge that blends the
            video into the form column. Goes from full page-black on the left
            to transparent at ~50% so the subject stays visible. */}
        <div
          aria-hidden="true"
          className="absolute inset-0 pointer-events-none z-[5]"
          style={{
            background:
              "linear-gradient(90deg, #0A0A0A 0%, rgba(10,10,10,0.92) 12%, rgba(10,10,10,0.55) 28%, rgba(10,10,10,0.18) 50%, transparent 75%)",
          }}
        />

        {/* Subtle top + bottom vignettes for polish */}
        <div
          aria-hidden="true"
          className="absolute inset-0 pointer-events-none z-[5]"
          style={{
            background:
              "linear-gradient(180deg, rgba(10,10,10,0.35) 0%, transparent 18%, transparent 82%, rgba(10,10,10,0.55) 100%)",
          }}
        />
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
                 hover:bg-[color:var(--color-gold-hover)] hover:shadow-[0_8px_30px_-10px_rgba(174,231,16,0.4)]
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
        background: "rgba(248,113,113,0.08)",
      }}
    >
      {message}
    </div>
  );
}
