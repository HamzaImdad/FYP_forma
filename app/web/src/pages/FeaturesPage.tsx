import { useRef } from "react";
import { Link } from "react-router-dom";
import { useRevealAnimations } from "@/lib/useRevealAnimations";
import { CinematicHero } from "@/components/sections/CinematicHero";

/* ──────────────────────────────────────────────────────────────
   Features page — "futuristic tech + fitness"

   Aesthetic: dark dominant, electric lime (#AEE710) accent.
   Tech grid background, gradient mesh orbs, scanline overlays,
   HUD corner brackets, glowing borders on hover.
   No new npm deps — CSS effects + GSAP entrance reveals
   (wired via useRevealAnimations on the scope ref).
   ────────────────────────────────────────────────────────────── */

type FeatureVisualKind =
  | "body"
  | "grid"
  | "dial"
  | "chart"
  | "timeline"
  | "badges"
  | "chat"
  | "bubble"
  | "lock";

type Feature = {
  num: string;
  tag: string;
  title: string;
  body: string;
  bullets: string[];
  visual: FeatureVisualKind;
};

const FEATURES: Feature[] = [
  {
    num: "01",
    tag: "Pose detection",
    title: "Every joint,\nevery frame.",
    body: "MediaPipe BlazePose tracks 33 body landmarks in world-space at the speed of your webcam. Each frame goes through angle computation, state machine, detector — all in under 50 ms.",
    bullets: ["33 body landmarks", "World-space coordinates", "60 fps feedback loop"],
    visual: "body",
  },
  {
    num: "02",
    tag: "Eleven detectors",
    title: "Specialised, not\ngeneral.",
    body: "A squat detector watches knee depth and torso lean. A curl detector watches upper-arm stability and torso swing. No one-size-fits-all ML model — each exercise gets its own state machine and its own thresholds.",
    bullets: ["Per-exercise state machines", "Biomechanics-backed thresholds", "Deterministic and debuggable"],
    visual: "grid",
  },
  {
    num: "03",
    tag: "Form score",
    title: "Words,\nnot numbers.",
    body: "Green when your alignment holds. Orange when it slips. Red when something needs immediate correction. One cue at a time — never a wall of text — and silence when you're good.",
    bullets: ["Colour-coded joints", "Plain-language cues", "Silence is a feature"],
    visual: "dial",
  },
  {
    num: "04",
    tag: "Dashboard",
    title: "Drill down\nto the rep.",
    body: "Every session logs to your history. Chips surface patterns (\"your squat depth is trending shallower\"), deep-dives open the session, the session opens to the individual rep. You can always see why.",
    bullets: ["Per-rep logging", "Insight chips", "Session → set → rep drill-down"],
    visual: "chart",
  },
  {
    num: "05",
    tag: "Plans & goals",
    title: "Tell it your goal.\nGet a plan.",
    body: "Set a goal — volume, frequency, a target lift — and FORMA generates a plan that adapts as you progress. Auto-generated milestones at 25, 50, 75, 100 percent. No spreadsheet.",
    bullets: ["Six goal types", "Adaptive week structure", "25 / 50 / 75 / 100 milestones"],
    visual: "timeline",
  },
  {
    num: "06",
    tag: "Milestones",
    title: "Progress you\ncan feel.",
    body: "Thirteen badges, each tied to a real achievement — first session, first ten sessions, first perfect form score, first 100-rep day. Celebration moments that aren't patronising, because they're earned.",
    bullets: ["13 unique badges", "Earned, not gifted", "Real-time celebration toasts"],
    visual: "badges",
  },
  {
    num: "07",
    tag: "AI coach",
    title: "A coach that\nknows your history.",
    body: "Ask FORMA about your training. \"How's my squat depth trending?\" \"What was my best bench session?\" The logged-in coach uses your own data through secure tool calls — not generic advice, your actual numbers.",
    bullets: ["Tool-use over your sessions", "Streamed responses", "Medical disclaimer, always"],
    visual: "chat",
  },
  {
    num: "08",
    tag: "Website guide",
    title: "A guide for\nfirst-time visitors.",
    body: "New here? The floating guide at the bottom-right answers questions about FORMA itself — what it does, what you need, how privacy works. It reads from an ingested knowledge base of this site.",
    bullets: ["Floats on every public page", "Site-wide knowledge base", "Streaming OpenAI responses"],
    visual: "bubble",
  },
  {
    num: "09",
    tag: "Privacy",
    title: "Your body,\nyour machine.",
    body: "No video uploaded. No frames sent to a server. No silhouettes, no biometric templates, no cloud copy. Pose estimation runs locally in your browser — the only thing the backend ever sees is the numeric summary you choose to save.",
    bullets: ["On-device pose estimation", "No video ever leaves", "Opt-in summaries only"],
    visual: "lock",
  },
];

export function FeaturesPage() {
  const scopeRef = useRef<HTMLDivElement>(null);
  useRevealAnimations(scopeRef);

  return (
    <div
      ref={scopeRef}
      className="bg-[color:var(--color-contrast)] text-[color:var(--color-ink-on-dark)]"
    >
      {/* ══ HERO — cinematic ══ */}
      <CinematicHero
        image="/static/images/cinematic/cinematic_back.jpg"
        anchor="left"
        minHeight="min-h-[92vh]"
      >
        <span
          data-reveal
          className="block font-[family-name:var(--font-mono)] text-[0.7rem] uppercase tracking-[0.32em] text-[color:var(--color-gold-soft)] mb-8"
        >
          / / / FEATURES
        </span>
        <h1
          data-reveal
          className="font-[family-name:var(--font-display)] leading-[0.88] text-[color:var(--color-ink-on-dark)]"
          style={{ fontSize: "clamp(3rem, 9vw, 7.5rem)" }}
        >
          Every rep,
          <br />
          <em className="not-italic font-[family-name:var(--font-serif)] italic text-[color:var(--color-gold-soft)]">
            engineered.
          </em>
        </h1>
        <p
          data-reveal
          className="mt-8 font-[family-name:var(--font-serif)] italic text-xl md:text-2xl text-[color:var(--color-ink-on-dark-2)] leading-[1.4]"
        >
          One product, built around the way you actually train. Pose detection to plans,
          dashboards to dedicated detectors — nine pieces that work the same way every session.
        </p>
        <div data-reveal className="mt-10 flex flex-wrap gap-4">
          <Link
            to="/login?tab=signup"
            className="inline-flex items-center gap-2 px-8 py-4 bg-[color:var(--color-gold)] text-[#0A0A0A] text-xs uppercase tracking-[0.14em] font-medium rounded-[2px] hover:bg-[color:var(--color-gold-soft)] transition-colors shadow-[0_0_60px_-10px_rgba(174,231,16,0.5)]"
          >
            Sign in to try →
          </Link>
          <Link
            to="/how-it-works"
            className="inline-flex items-center gap-2 px-8 py-4 border border-[color:var(--color-ink-on-dark)]/30 text-[color:var(--color-ink-on-dark)] text-xs uppercase tracking-[0.14em] font-medium rounded-[2px] hover:border-[color:var(--color-gold-soft)] hover:text-[color:var(--color-gold-soft)] transition-colors"
          >
            How it works
          </Link>
        </div>

        {/* Anchor nav */}
        <div
          data-reveal
          className="mt-20 pt-8 border-t border-[color:var(--color-ink-on-dark)]/10 flex flex-wrap gap-x-8 gap-y-3 font-[family-name:var(--font-mono)] text-[0.65rem] uppercase tracking-[0.22em]"
        >
          {FEATURES.map((f) => (
            <a
              key={f.num}
              href={`#f-${f.num}`}
              className="text-[color:var(--color-ink-on-dark-2)] hover:text-[color:var(--color-gold-soft)] transition-colors"
            >
              <span className="text-[color:var(--color-gold-soft)]">{f.num}</span>
              <span className="mx-2">·</span>
              {f.tag}
            </a>
          ))}
        </div>
      </CinematicHero>

      {/* ══ FEATURE SECTIONS — with cinematic accent strips at pauses ══ */}
      {FEATURES.map((f, i) => (
        <div key={f.num}>
          <FeatureBlock feature={f} flipped={i % 2 === 1} />
          {i === 2 && (
            <CinematicStrip
              image="/static/images/cinematic/cinematic_bicep.jpg"
              text="One cue at a time."
              subtext="Silence is a feature."
            />
          )}
          {i === 5 && (
            <CinematicStrip
              image="/static/images/cinematic/cinematic_silhouette.jpg"
              text="Your data, your body."
              subtext="On-device pose estimation."
            />
          )}
        </div>
      ))}

      {/* ══ CLOSING CTA — cinematic ══ */}
      <CinematicHero
        image="/static/images/cinematic/cinematic_deadlift.jpg"
        anchor="center"
        minHeight="min-h-[72vh]"
      >
        <span
          data-reveal
          className="block font-[family-name:var(--font-mono)] text-[0.7rem] uppercase tracking-[0.32em] text-[color:var(--color-gold-soft)] mb-8"
        >
          / / / NEXT
        </span>
        <h2
          data-reveal
          className="font-[family-name:var(--font-display)] leading-[0.9] text-[color:var(--color-ink-on-dark)]"
          style={{ fontSize: "clamp(2.8rem, 8vw, 6.5rem)" }}
        >
          Start training.
          <br />
          <em className="not-italic font-[family-name:var(--font-serif)] italic text-[color:var(--color-gold-soft)]">
            See the difference.
          </em>
        </h2>
        <p
          data-reveal
          className="mt-8 font-[family-name:var(--font-serif)] italic text-xl text-[color:var(--color-ink-on-dark-2)]"
        >
          Everything on this page is running in the browser right now. Press the button.
        </p>
        <div className="mt-12 flex flex-wrap gap-4 justify-center">
          <Link
            to="/login?tab=signup"
            className="inline-flex items-center gap-2 px-10 py-5 bg-[color:var(--color-gold)] text-[#0A0A0A] text-sm uppercase tracking-[0.16em] font-medium rounded-[2px] hover:bg-[color:var(--color-gold-hover)] transition-colors shadow-[0_0_60px_-10px_rgba(174,231,16,0.5)]"
          >
            Create an account →
          </Link>
        </div>
      </CinematicHero>
    </div>
  );
}

/* ══════════ FeatureBlock — one section per feature ══════════ */

function FeatureBlock({ feature, flipped }: { feature: Feature; flipped: boolean }) {
  return (
    <section
      id={`f-${feature.num}`}
      className="relative overflow-hidden border-t border-[rgba(174,231,16,0.06)] group/section hover:border-[rgba(174,231,16,0.15)] transition-colors duration-500"
    >
      <TechGridBackground faint />

      {/* Creative gradient: a soft radial bloom sitting near the visual's center,
          dimmer corner glow on the text side. Position flips with the layout. */}
      <div
        aria-hidden="true"
        className={`absolute top-1/2 -translate-y-1/2 pointer-events-none rounded-full z-[1] ${
          flipped ? "left-[15%]" : "right-[15%]"
        }`}
        style={{
          width: "620px",
          height: "620px",
          background:
            "radial-gradient(circle, rgba(174,231,16,0.22) 0%, rgba(174,231,16,0.08) 30%, transparent 65%)",
          filter: "blur(80px)",
          willChange: "transform",
          transform: "translateZ(0)",
          contain: "strict",
        }}
      />
      <div
        aria-hidden="true"
        className={`absolute top-[20%] pointer-events-none rounded-full z-[1] ${
          flipped ? "right-[5%]" : "left-[5%]"
        }`}
        style={{
          width: "360px",
          height: "360px",
          background:
            "radial-gradient(circle, rgba(194,240,74,0.10) 0%, transparent 70%)",
          filter: "blur(70px)",
          willChange: "transform",
          transform: "translateZ(0)",
          contain: "strict",
        }}
      />

      <div className="relative z-[2] mx-auto max-w-[1440px] px-6 md:px-10 py-28 md:py-36">
        <div
          className={`grid gap-16 lg:gap-24 lg:grid-cols-2 items-center ${flipped ? "lg:[&>*:first-child]:order-2" : ""}`}
        >
          <div>
            <div data-reveal className="flex items-baseline gap-6 mb-8">
              <span
                className="font-[family-name:var(--font-display)] text-6xl md:text-8xl leading-[0.8] text-[color:var(--color-gold-soft)] opacity-80"
                style={{ textShadow: "0 0 40px rgba(174,231,16,0.35)" }}
              >
                {feature.num}
              </span>
              <span className="font-[family-name:var(--font-mono)] text-[0.65rem] uppercase tracking-[0.28em] text-[color:var(--color-ink-on-dark-2)]">
                {feature.tag}
              </span>
            </div>
            <h2
              data-reveal
              className="font-[family-name:var(--font-display)] leading-[0.92] text-[color:var(--color-ink-on-dark)] mb-8 whitespace-pre-line"
              style={{ fontSize: "clamp(2.2rem, 5vw, 4.5rem)" }}
            >
              {feature.title.split("\n").map((line, i) => (
                <span key={i} className="block">
                  {i === feature.title.split("\n").length - 1 ? (
                    <em className="not-italic font-[family-name:var(--font-serif)] italic text-[color:var(--color-gold-soft)]">
                      {line}
                    </em>
                  ) : (
                    line
                  )}
                </span>
              ))}
            </h2>
            <p
              data-reveal
              className="text-[color:var(--color-ink-on-dark-2)] leading-[1.7] text-lg max-w-xl mb-8"
            >
              {feature.body}
            </p>
            <ul data-reveal className="space-y-3">
              {feature.bullets.map((b) => (
                <li
                  key={b}
                  className="flex items-center gap-3 font-[family-name:var(--font-mono)] text-[0.78rem] uppercase tracking-[0.14em] text-[color:var(--color-ink-on-dark-2)]"
                >
                  <span className="block h-px w-8 bg-[color:var(--color-gold-soft)]" />
                  {b}
                </li>
              ))}
            </ul>
          </div>

          <div data-reveal className="relative">
            <FeatureVisual kind={feature.visual} />
          </div>
        </div>
      </div>
    </section>
  );
}

/* ══════════ Visual mockups — CSS + inline SVG only ══════════ */

function FeatureVisual({ kind }: { kind: FeatureVisualKind }) {
  switch (kind) {
    case "body":
      return <BodyVisual />;
    case "grid":
      return <ExerciseGridVisual />;
    case "dial":
      return <ScoreDialVisual />;
    case "chart":
      return <DashboardChartVisual />;
    case "timeline":
      return <TimelineVisual />;
    case "badges":
      return <BadgesVisual />;
    case "chat":
      return <ChatVisual subject="history" />;
    case "bubble":
      return <ChatVisual subject="guide" />;
    case "lock":
      return <LockVisual />;
  }
}

function VisualFrame({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="relative aspect-[4/5] md:aspect-[5/6] rounded-[6px] overflow-hidden border border-[rgba(174,231,16,0.1)] bg-[color:var(--color-contrast-2)] shadow-[0_40px_120px_-40px_rgba(0,0,0,0.6),0_0_60px_-20px_rgba(174,231,16,0.18)]"
      style={{ contentVisibility: "auto", containIntrinsicSize: "720px" }}
    >
      {/* Gradient overlay */}
      <div
        className="absolute inset-0 opacity-40"
        style={{
          background:
            "linear-gradient(135deg, rgba(174,231,16,0.1), transparent 50%, rgba(174,231,16,0.06))",
        }}
      />
      {/* Scanline overlay */}
      <div
        className="absolute inset-0 pointer-events-none opacity-30"
        style={{
          background:
            "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(174,231,16,0.03) 2px, rgba(174,231,16,0.03) 4px)",
          willChange: "transform",
          transform: "translateZ(0)",
        }}
      />
      {/* HUD corner brackets */}
      <div className="absolute top-2 left-2 w-4 h-4 border-t border-l border-[rgba(174,231,16,0.3)]" />
      <div className="absolute top-2 right-2 w-4 h-4 border-t border-r border-[rgba(174,231,16,0.3)]" />
      <div className="absolute bottom-2 left-2 w-4 h-4 border-b border-l border-[rgba(174,231,16,0.3)]" />
      <div className="absolute bottom-2 right-2 w-4 h-4 border-b border-r border-[rgba(174,231,16,0.3)]" />
      {children}
    </div>
  );
}

function BodyVisual() {
  // Stylised wireframe body with 33 landmarks as dots + lines
  const dots: Array<[number, number]> = [
    [50, 12], // head
    [50, 22], // neck
    [38, 26], [62, 26], // shoulders
    [30, 44], [70, 44], // elbows
    [24, 60], [76, 60], // wrists
    [44, 40], [56, 40], // spine
    [44, 58], [56, 58], // hips
    [42, 76], [58, 76], // knees
    [40, 92], [60, 92], // ankles
  ];
  const lines: Array<[number, number]> = [
    [0, 1], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7],
    [2, 8], [3, 9], [8, 10], [9, 11], [10, 12], [11, 13],
    [12, 14], [13, 15],
  ];
  return (
    <VisualFrame>
      <svg
        viewBox="0 0 100 100"
        className="absolute inset-0 h-full w-full"
        preserveAspectRatio="xMidYMid meet"
        aria-hidden="true"
      >
        {lines.map(([a, b], i) => (
          <line
            key={i}
            x1={dots[a][0]}
            y1={dots[a][1]}
            x2={dots[b][0]}
            y2={dots[b][1]}
            stroke="rgba(174,231,16,0.55)"
            strokeWidth="0.3"
          />
        ))}
        {dots.map(([x, y], i) => (
          <g key={i}>
            <circle cx={x} cy={y} r="0.9" fill="#AEE710" />
            <circle cx={x} cy={y} r="2.2" fill="rgba(174,231,16,0.15)" />
          </g>
        ))}
      </svg>
      <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between font-[family-name:var(--font-mono)] text-[0.55rem] uppercase tracking-[0.2em] text-[color:var(--color-gold-soft)]">
        <span>33 landmarks · world space</span>
        <span className="text-[color:var(--color-ink-on-dark-2)]">60 fps</span>
      </div>
    </VisualFrame>
  );
}

function ExerciseGridVisual() {
  const slugs = [
    "ex_squat",
    "ex_deadlift",
    "ex_pullup",
    "ex_pushup",
    "ex_plank",
    "ex_bicep_curl",
    "ex_tricep_dip",
    "ex_crunch",
    "ex_lateral_raise",
    "ex_side_plank",
  ];
  return (
    <VisualFrame>
      <div className="absolute inset-0 p-5 grid grid-cols-3 gap-2">
        {slugs.slice(0, 9).map((slug, i) => (
          <div
            key={slug}
            className="relative overflow-hidden rounded-[3px] ring-1 ring-[color:var(--color-gold-soft)]/20"
          >
            <img
              src={`/static/images/${slug}.jpg`}
              alt=""
              aria-hidden="true"
              className="absolute inset-0 h-full w-full object-cover"
              loading="lazy"
            />
            <div className="absolute inset-0 bg-[color:var(--color-contrast)]/65" />
            {i === 4 && (
              <>
                <div className="absolute inset-0 ring-2 ring-[color:var(--color-gold-soft)] shadow-[inset_0_0_40px_rgba(174,231,16,0.3)]" />
                <span className="absolute bottom-1 left-1 font-[family-name:var(--font-mono)] text-[0.5rem] uppercase tracking-wider text-[color:var(--color-gold-soft)]">
                  ACTIVE
                </span>
              </>
            )}
          </div>
        ))}
      </div>
    </VisualFrame>
  );
}

function ScoreDialVisual() {
  const value = 87;
  const circumference = 2 * Math.PI * 42;
  const dashOffset = circumference * (1 - value / 100);
  return (
    <VisualFrame>
      <div className="absolute inset-0 flex flex-col items-center justify-center pb-24">
        <div className="relative w-[64%] aspect-square flex items-center justify-center">
          <svg
            viewBox="0 0 100 100"
            className="absolute inset-0 w-full h-full"
            aria-hidden="true"
          >
            <circle
              cx="50"
              cy="50"
              r="42"
              fill="none"
              stroke="rgba(255,255,255,0.06)"
              strokeWidth="5"
            />
            <circle
              cx="50"
              cy="50"
              r="42"
              fill="none"
              stroke="#AEE710"
              strokeWidth="5"
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={dashOffset}
              transform="rotate(-90 50 50)"
              style={{ filter: "drop-shadow(0 0 8px rgba(174,231,16,0.5))" }}
            />
          </svg>
          <div className="relative text-center pointer-events-none">
            <div className="font-[family-name:var(--font-display)] text-7xl text-[color:var(--color-ink-on-dark)] leading-none">
              {value}
            </div>
            <div className="mt-2 font-[family-name:var(--font-mono)] text-[0.6rem] uppercase tracking-[0.22em] text-[color:var(--color-gold-soft)]">
              Form score
            </div>
          </div>
        </div>
        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 w-4/5 rounded-[3px] border-l-2 border-[color:var(--color-gold-soft)] bg-[color:var(--color-contrast)]/80 px-4 py-3">
          <span className="block font-[family-name:var(--font-mono)] text-[0.5rem] uppercase tracking-[0.2em] text-[color:var(--color-gold-soft)]">
            Cue
          </span>
          <span className="mt-1 block font-[family-name:var(--font-serif)] italic text-lg text-[color:var(--color-ink-on-dark)]">
            Hips dropping.
          </span>
        </div>
      </div>
    </VisualFrame>
  );
}

function DashboardChartVisual() {
  const bars = [32, 58, 41, 72, 85, 63, 90, 76, 88];
  return (
    <VisualFrame>
      <div className="absolute inset-0 p-6 flex flex-col">
        <div className="flex items-center justify-between mb-4">
          <span className="font-[family-name:var(--font-mono)] text-[0.55rem] uppercase tracking-[0.22em] text-[color:var(--color-gold-soft)]">
            Weekly volume
          </span>
          <span className="font-[family-name:var(--font-mono)] text-[0.55rem] uppercase tracking-[0.22em] text-[color:var(--color-ink-on-dark-2)]">
            + 14%
          </span>
        </div>
        <div className="flex-1 flex items-end gap-2">
          {bars.map((h, i) => (
            <div
              key={i}
              className="flex-1 rounded-t-[2px]"
              style={{
                height: `${h}%`,
                background: `linear-gradient(180deg, rgba(174,231,16,${0.15 + (h / 100) * 0.6}) 0%, rgba(174,231,16,0.08) 100%)`,
                boxShadow: "inset 0 1px 0 rgba(174,231,16,0.4)",
              }}
            />
          ))}
        </div>
        <div className="mt-4 pt-4 border-t border-[color:var(--color-ink-on-dark)]/10 space-y-2">
          {[
            { label: "Squat depth", val: "+2°", trend: "up" },
            { label: "Bench symmetry", val: "96%", trend: "flat" },
            { label: "Pull-up ROM", val: "-4°", trend: "down" },
          ].map((row) => (
            <div
              key={row.label}
              className="flex items-center justify-between font-[family-name:var(--font-mono)] text-[0.55rem] uppercase tracking-[0.16em]"
            >
              <span className="text-[color:var(--color-ink-on-dark-2)]">{row.label}</span>
              <span
                className={
                  row.trend === "up"
                    ? "text-[color:var(--color-gold-soft)]"
                    : row.trend === "down"
                      ? "text-[color:var(--color-bad)]"
                      : "text-[color:var(--color-ink-on-dark-2)]"
                }
              >
                {row.val}
              </span>
            </div>
          ))}
        </div>
      </div>
    </VisualFrame>
  );
}

function TimelineVisual() {
  const milestones = [
    { pct: 10, label: "Start" },
    { pct: 32, label: "25%", done: true },
    { pct: 54, label: "50%", active: true },
    { pct: 76, label: "75%" },
    { pct: 93, label: "100%" },
  ];
  return (
    <VisualFrame>
      <div className="absolute inset-0 flex flex-col justify-center px-8">
        <span className="font-[family-name:var(--font-mono)] text-[0.55rem] uppercase tracking-[0.22em] text-[color:var(--color-gold-soft)] mb-10">
          Goal · Squat 100 reps / week
        </span>
        <div className="relative h-[2px] bg-[color:var(--color-ink-on-dark)]/12 rounded-full">
          <div
            className="absolute left-0 top-0 h-full bg-[color:var(--color-gold-soft)] rounded-full"
            style={{ width: "54%" }}
          />
          {milestones.map((m) => (
            <div
              key={m.label}
              className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2"
              style={{ left: `${m.pct}%` }}
            >
              <div
                className={
                  "h-3 w-3 rounded-full border " +
                  (m.done
                    ? "bg-[color:var(--color-gold-soft)] border-[color:var(--color-gold-soft)]"
                    : m.active
                      ? "bg-[color:var(--color-gold)] border-[color:var(--color-gold)] shadow-[0_0_12px_rgba(174,231,16,0.7)]"
                      : "bg-[color:var(--color-contrast-2)] border-[color:var(--color-ink-on-dark)]/30")
                }
              />
              <span className="absolute top-5 left-1/2 -translate-x-1/2 font-[family-name:var(--font-mono)] text-[0.5rem] uppercase tracking-wider text-[color:var(--color-ink-on-dark-2)]">
                {m.label}
              </span>
            </div>
          ))}
        </div>
        <div className="mt-20 flex items-center justify-between">
          <span className="font-[family-name:var(--font-mono)] text-[0.55rem] uppercase tracking-[0.18em] text-[color:var(--color-ink-on-dark-2)]">
            Week 3 · 54 / 100
          </span>
          <span className="font-[family-name:var(--font-mono)] text-[0.55rem] uppercase tracking-[0.18em] text-[color:var(--color-gold-soft)]">
            On pace
          </span>
        </div>
      </div>
    </VisualFrame>
  );
}

function BadgesVisual() {
  const badges = [
    { label: "First Session", earned: true },
    { label: "10 Sessions", earned: true },
    { label: "Perfect Form", earned: true },
    { label: "100 Reps / Day" },
    { label: "Consistency 7", earned: true },
    { label: "Deep Squat" },
    { label: "Form Hunter" },
    { label: "Silent Set" },
    { label: "All Ten" },
  ];
  return (
    <VisualFrame>
      <div className="absolute inset-0 p-8 grid grid-cols-3 grid-rows-3 gap-4 place-items-center">
        {badges.map((b, i) => (
          <div
            key={i}
            className={
              "relative h-14 w-14 md:h-16 md:w-16 rounded-full border flex items-center justify-center " +
              (b.earned
                ? "border-[color:var(--color-gold-soft)] bg-[color:var(--color-gold-soft)]/10 shadow-[0_0_20px_rgba(174,231,16,0.3)]"
                : "border-[color:var(--color-ink-on-dark)]/12 bg-transparent")
            }
            title={b.label}
          >
            <span
              className={
                "font-[family-name:var(--font-display)] text-sm " +
                (b.earned
                  ? "text-[color:var(--color-gold-soft)]"
                  : "text-[color:var(--color-ink-on-dark-2)]/50")
              }
            >
              {String(i + 1).padStart(2, "0")}
            </span>
          </div>
        ))}
      </div>
      <div className="absolute bottom-4 left-0 right-0 text-center font-[family-name:var(--font-mono)] text-[0.55rem] uppercase tracking-[0.22em] text-[color:var(--color-gold-soft)]">
        5 / 13 earned
      </div>
    </VisualFrame>
  );
}

function ChatVisual({ subject }: { subject: "history" | "guide" }) {
  const messages =
    subject === "history"
      ? [
          { role: "user", text: "How is my squat depth trending this month?" },
          {
            role: "bot",
            text: "Up 2° on average across 14 sessions. Best: Monday's set 3 (121° knee flexion).",
          },
          { role: "user", text: "Best bench session?" },
          { role: "bot", text: "April 8 — 4 sets, 87 form score, 0 cues." },
        ]
      : [
          { role: "user", text: "Does FORMA need any equipment?" },
          { role: "bot", text: "Just a webcam and a browser. Nothing else — no wearables, no uploads." },
          { role: "user", text: "What about privacy?" },
          { role: "bot", text: "Everything runs on your machine. No video ever leaves the browser." },
        ];
  return (
    <VisualFrame>
      <div className="absolute inset-0 flex flex-col">
        <div className="px-5 py-4 border-b border-[color:var(--color-ink-on-dark)]/10 flex items-center gap-3">
          <div className="h-2.5 w-2.5 rounded-full bg-[color:var(--color-gold-soft)] shadow-[0_0_8px_rgba(174,231,16,0.6)]" />
          <span className="font-[family-name:var(--font-mono)] text-[0.6rem] uppercase tracking-[0.22em] text-[color:var(--color-gold-soft)]">
            {subject === "history" ? "FORMA Coach" : "Ask FORMA"}
          </span>
        </div>
        <div className="flex-1 overflow-hidden p-5 space-y-3">
          {messages.map((m, i) => (
            <div
              key={i}
              className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={
                  "max-w-[82%] rounded-[8px] px-4 py-2.5 text-[0.78rem] leading-[1.4] " +
                  (m.role === "user"
                    ? "bg-[color:var(--color-gold-soft)]/12 border border-[color:var(--color-gold-soft)]/25 text-[color:var(--color-ink-on-dark)]"
                    : "bg-[color:var(--color-ink-on-dark)]/8 border border-[color:var(--color-ink-on-dark)]/12 text-[color:var(--color-ink-on-dark-2)]")
                }
              >
                {m.text}
              </div>
            </div>
          ))}
        </div>
        <div className="px-5 py-3 border-t border-[color:var(--color-ink-on-dark)]/10 flex items-center gap-3">
          <div className="flex-1 h-8 rounded-[4px] border border-[color:var(--color-ink-on-dark)]/15 bg-[color:var(--color-contrast)]/60" />
          <div className="h-8 w-8 rounded-[4px] bg-[color:var(--color-gold-soft)]/20 border border-[color:var(--color-gold-soft)]/40 flex items-center justify-center">
            <span className="text-[color:var(--color-gold-soft)] text-sm">→</span>
          </div>
        </div>
      </div>
    </VisualFrame>
  );
}

function LockVisual() {
  return (
    <VisualFrame>
      <div className="absolute inset-0 flex flex-col items-center justify-center gap-8">
        <svg
          viewBox="0 0 100 100"
          className="w-[50%] h-auto"
          aria-hidden="true"
        >
          {/* shackle */}
          <path
            d="M 35 45 L 35 32 Q 35 18 50 18 Q 65 18 65 32 L 65 45"
            fill="none"
            stroke="rgba(174,231,16,0.7)"
            strokeWidth="3.5"
            strokeLinecap="round"
          />
          {/* body */}
          <rect
            x="26"
            y="44"
            width="48"
            height="42"
            rx="4"
            fill="rgba(174,231,16,0.12)"
            stroke="#AEE710"
            strokeWidth="2.5"
            style={{ filter: "drop-shadow(0 0 16px rgba(174,231,16,0.5))" }}
          />
          {/* keyhole */}
          <circle cx="50" cy="60" r="4" fill="#AEE710" />
          <path
            d="M 50 60 L 50 72"
            stroke="#AEE710"
            strokeWidth="3"
            strokeLinecap="round"
          />
        </svg>
        <div className="text-center">
          <div className="font-[family-name:var(--font-mono)] text-[0.58rem] uppercase tracking-[0.24em] text-[color:var(--color-gold-soft)] mb-3">
            0 frames uploaded
          </div>
          <div className="font-[family-name:var(--font-serif)] italic text-xl text-[color:var(--color-ink-on-dark)] max-w-[16ch] leading-tight">
            Your body, your machine.
          </div>
        </div>
      </div>
    </VisualFrame>
  );
}

/* ══════════ Background primitives ══════════ */

// Soft atmospheric gradient mesh — replaces the grid the user didn't like.
// `faint` mode is used between feature blocks so they don't overpower.
function TechGridBackground({ faint = false }: { faint?: boolean }) {
  const o = faint ? 0.05 : 0.11;
  return (
    <div
      className="absolute inset-0 pointer-events-none"
      aria-hidden="true"
      style={{
        background: `
          radial-gradient(ellipse 60% 40% at 20% 0%, rgba(174,231,16,${o}) 0%, transparent 60%),
          radial-gradient(ellipse 55% 45% at 85% 100%, rgba(174,231,16,${o * 0.7}) 0%, transparent 60%),
          radial-gradient(circle at 50% 50%, rgba(194,240,74,${o * 0.25}) 0%, transparent 70%)
        `,
        willChange: "transform",
        transform: "translateZ(0)",
        contain: "strict",
      }}
    />
  );
}

/* ══════════ CinematicStrip — atmospheric break between feature blocks ══════════ */

function CinematicStrip({
  image,
  text,
  subtext,
}: {
  image: string;
  text: string;
  subtext: string;
}) {
  return (
    <section
      data-reveal
      className="relative h-[42vh] min-h-[340px] overflow-hidden"
    >
      <img
        src={image}
        alt=""
        aria-hidden="true"
        className="absolute inset-0 h-full w-full object-cover"
        style={{ filter: "contrast(1.1) brightness(0.7) saturate(0.85)" }}
      />
      <div
        className="absolute inset-0 pointer-events-none"
        aria-hidden="true"
        style={{
          background:
            "linear-gradient(90deg, rgba(10,10,10,0.95) 0%, rgba(10,10,10,0.55) 45%, rgba(10,10,10,0.75) 100%), linear-gradient(180deg, rgba(10,10,10,0.5) 0%, transparent 40%, rgba(10,10,10,0.8) 100%)",
        }}
      />
      <div
        aria-hidden="true"
        className="absolute top-1/2 left-[35%] -translate-y-1/2 pointer-events-none rounded-full"
        style={{
          width: "360px",
          height: "360px",
          background: "radial-gradient(circle, rgba(174,231,16,0.18) 0%, transparent 70%)",
          filter: "blur(70px)",
          willChange: "transform",
          transform: "translateZ(0)",
          contain: "strict",
        }}
      />
      <div
        aria-hidden="true"
        className="absolute inset-0 pointer-events-none opacity-[0.2] mix-blend-overlay"
        style={{ backgroundImage: "var(--grain)" }}
      />
      <div className="relative z-[2] h-full mx-auto max-w-[1440px] px-6 md:px-10 flex items-center">
        <div className="max-w-xl">
          <p
            className="font-[family-name:var(--font-display)] leading-[0.92] text-[color:var(--color-ink-on-dark)]"
            style={{ fontSize: "clamp(2rem, 4.5vw, 3.5rem)" }}
          >
            {text}
          </p>
          <p
            className="mt-4 font-[family-name:var(--font-serif)] italic text-xl md:text-2xl text-[color:var(--color-gold-soft)]"
          >
            {subtext}
          </p>
        </div>
      </div>
    </section>
  );
}
