import { Link } from "react-router-dom";

const MARQUEE = "Train with form. · Real-time AI feedback. · Eleven exercises, one trainer. · No wearables, no cloud. · Just your webcam.";

export function Footer() {
  return (
    <footer className="relative bg-[color:var(--color-contrast)] text-[color:var(--color-ink-on-dark)] mt-32 overflow-hidden">
      {/* ── MARQUEE BANNER ───────────────────────────────────────── */}
      <div
        className="relative border-y border-[color:var(--color-ink-on-dark)]/5 bg-[color:var(--color-contrast)] py-1.5 overflow-hidden"
        aria-hidden="true"
      >
        <div
          className="marquee-track"
          style={{ ["--marquee-duration" as string]: "96s" }}
        >
          {Array.from({ length: 6 }).map((_, i) => (
            <span
              key={i}
              className="flex-shrink-0 mx-6 font-[family-name:var(--font-mono)] text-[0.65rem] md:text-[0.7rem] tracking-[0.22em] uppercase whitespace-nowrap text-[color:var(--color-ink-on-dark)]/25"
            >
              {MARQUEE}
              <span className="mx-4 text-[color:var(--color-gold-soft)]/30">✦</span>
            </span>
          ))}
        </div>
      </div>

      {/* ── MAIN FOOTER BODY ─────────────────────────────────────── */}
      <div className="relative mx-auto max-w-[1440px] px-6 md:px-10 pt-24 pb-12">
        {/* Brand row — logo + tagline */}
        <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-10 pb-16 border-b border-[color:var(--color-ink-on-dark)]/10">
          <div>
            <Link
              to="/"
              className="font-[family-name:var(--font-display)] text-6xl md:text-7xl tracking-[0.18em] text-[color:var(--color-ink-on-dark)] inline-block leading-none"
            >
              F
              <em className="font-[family-name:var(--font-serif)] italic text-[color:var(--color-gold-soft)]">
                O
              </em>
              RMA
            </Link>
            <p className="mt-5 font-[family-name:var(--font-serif)] italic text-2xl md:text-3xl text-[color:var(--color-ink-on-dark)] max-w-md leading-snug">
              Train with form.
            </p>
          </div>

          <div className="md:text-right space-y-2">
            <p className="text-xs uppercase tracking-[0.18em] text-[color:var(--color-gold-soft)]">
              BSc Computer Science
            </p>
            <p className="text-xs uppercase tracking-[0.18em] text-[color:var(--color-ink-on-dark-2)]">
              University of Greenwich · Final Year Project
            </p>
          </div>
        </div>

        {/* Link grid */}
        <div className="grid grid-cols-2 md:grid-cols-3 gap-10 md:gap-14 pt-16 pb-20">
          <FooterCol
            title="Product"
            items={[
              { to: "/", label: "Home" },
              { to: "/exercises", label: "Exercises" },
              { to: "/dashboard", label: "Dashboard" },
              { to: "/about", label: "About" },
            ]}
          />
          <FooterCol
            title="Features"
            items={[
              { to: "/voice-coaching", label: "Voice Coaching" },
              { to: "/chatbot", label: "AI Chatbot" },
              { to: "/plans", label: "Plans & Goals" },
              { to: "/milestones", label: "Milestones" },
            ]}
          />
          <FooterCol
            title="Pipeline"
            items={[
              { to: "/about", label: "MediaPipe BlazePose" },
              { to: "/about", label: "Dedicated Detectors" },
              { to: "/about", label: "Form Score" },
              { to: "/about", label: "The Project" },
            ]}
          />
        </div>

        {/* Bottom bar */}
        <div className="pt-8 border-t border-[color:var(--color-ink-on-dark)]/10 flex flex-col md:flex-row items-center justify-between gap-4 text-[0.7rem] uppercase tracking-[0.18em] text-[color:var(--color-ink-on-dark-2)]">
          <span>© 2026 FORMA · Train with form.</span>
          <span className="hidden md:inline">MediaPipe · Flask · React · Socket.IO</span>
          <button
            type="button"
            onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
            className="inline-flex items-center gap-2 hover:text-[color:var(--color-gold-soft)] transition-colors"
          >
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-[color:var(--color-gold-soft)]" />
            Back to top <span aria-hidden>↑</span>
          </button>
        </div>
      </div>

      {/* ── GIANT BRAND WATERMARK ────────────────────────────────── */}
      <div className="relative overflow-hidden h-[180px] md:h-[320px] lg:h-[420px] -mt-4">
        {/* Gold radial glow */}
        <div
          className="absolute inset-0"
          aria-hidden="true"
          style={{
            background:
              "radial-gradient(ellipse 90% 110% at 50% 130%, rgba(174,231,16,0.15) 0%, rgba(174,231,16,0.06) 40%, rgba(174,231,16,0.02) 70%, transparent 100%)",
          }}
        />
        {/* Subtle grain texture via repeating gradient for tactility */}
        <div
          className="absolute inset-0 opacity-[0.15] mix-blend-overlay pointer-events-none"
          aria-hidden="true"
          style={{
            background:
              "repeating-linear-gradient(45deg, transparent 0px, transparent 2px, rgba(255,255,255,0.02) 2px, rgba(255,255,255,0.02) 3px)",
          }}
        />
        {/* Giant wordmark */}
        <div
          className="absolute left-1/2 -translate-x-1/2 bottom-[-14%] md:bottom-[-18%] lg:bottom-[-20%] whitespace-nowrap pointer-events-none select-none"
          aria-hidden="true"
        >
          <span
            className="font-[family-name:var(--font-display)] leading-none tracking-[0.05em]"
            style={{
              fontSize: "clamp(8rem, 28vw, 22rem)",
              background:
                "linear-gradient(180deg, rgba(240,240,240,0.18) 0%, rgba(240,240,240,0.10) 35%, rgba(174,231,16,0.12) 65%, rgba(174,231,16,0.04) 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
            }}
          >
            F
            <em
              className="not-italic font-[family-name:var(--font-serif)] italic"
              style={{
                background:
                  "linear-gradient(180deg, rgba(174,231,16,0.45) 0%, rgba(174,231,16,0.2) 50%, rgba(174,231,16,0.05) 100%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                backgroundClip: "text",
              }}
            >
              O
            </em>
            RMA
          </span>
        </div>
      </div>
    </footer>
  );
}

type FooterLink = { to: string; label: string };

function FooterCol({ title, items }: { title: string; items: FooterLink[] }) {
  return (
    <div>
      <h4 className="font-[family-name:var(--font-mono)] text-[0.65rem] uppercase tracking-[0.22em] text-[color:var(--color-gold-soft)] mb-6">
        {title}
      </h4>
      <ul className="space-y-3">
        {items.map((item) => (
          <li key={`${item.to}-${item.label}`}>
            <Link
              to={item.to}
              className="group inline-flex items-center gap-2 text-sm text-[color:var(--color-ink-on-dark)]/80 hover:text-[color:var(--color-gold-soft)] transition-colors"
            >
              <span className="h-px w-0 bg-[color:var(--color-gold-soft)] transition-all duration-300 group-hover:w-4" />
              {item.label}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
