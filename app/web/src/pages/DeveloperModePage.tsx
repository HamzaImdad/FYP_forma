import { useRef } from "react";
import { Link } from "react-router-dom";
import { PageHero } from "@/components/sections/PageHero";
import { Section } from "@/components/sections/Section";
import { SectionHeader } from "@/components/sections/SectionHeader";
import { MockWindow } from "@/components/sections/MockWindow";
import { useRevealAnimations } from "@/lib/useRevealAnimations";

type Reliability = "high" | "medium" | "low";

const RELIABILITY: { exercise: string; angles: { name: string; level: Reliability }[] }[] = [
  {
    exercise: "Squat",
    angles: [
      { name: "Knee", level: "high" },
      { name: "Hip", level: "high" },
      { name: "Torso lean", level: "medium" },
    ],
  },
  {
    exercise: "Deadlift",
    angles: [
      { name: "Hip hinge", level: "high" },
      { name: "Knee", level: "high" },
      { name: "Spine", level: "low" },
    ],
  },
  {
    exercise: "Bench Press",
    angles: [
      { name: "Elbow", level: "high" },
      { name: "Shoulder flare", level: "medium" },
      { name: "Symmetry", level: "high" },
    ],
  },
  {
    exercise: "Overhead Press",
    angles: [
      { name: "Elbow lockout", level: "high" },
      { name: "Torso back-arch", level: "medium" },
      { name: "Wrist", level: "low" },
    ],
  },
  {
    exercise: "Pull-up",
    angles: [
      { name: "Elbow", level: "high" },
      { name: "Chin height", level: "high" },
      { name: "Body swing", level: "medium" },
    ],
  },
  {
    exercise: "Push-up",
    angles: [
      { name: "Elbow", level: "high" },
      { name: "Hip line", level: "high" },
      { name: "Head pose", level: "medium" },
    ],
  },
  {
    exercise: "Plank",
    angles: [
      { name: "Body line", level: "high" },
      { name: "Hip sag", level: "high" },
      { name: "Shoulder", level: "medium" },
    ],
  },
  {
    exercise: "Lunge",
    angles: [
      { name: "Front knee", level: "high" },
      { name: "Back knee", level: "medium" },
      { name: "Torso upright", level: "high" },
    ],
  },
  {
    exercise: "Bicep Curl",
    angles: [
      { name: "Elbow", level: "high" },
      { name: "Upper arm drift", level: "medium" },
      { name: "Torso swing", level: "medium" },
    ],
  },
  {
    exercise: "Tricep Dip",
    angles: [
      { name: "Elbow", level: "high" },
      { name: "Forward lean", level: "medium" },
      { name: "Shoulder", level: "low" },
    ],
  },
];

const LEVEL_COLOR: Record<Reliability, { bg: string; text: string; label: string }> = {
  high: { bg: "var(--color-good)", text: "High", label: "reliable" },
  medium: { bg: "var(--color-warn)", text: "Medium", label: "caution" },
  low: { bg: "var(--color-bad)", text: "Low", label: "hard case" },
};

const LOOP = [
  { step: "01", title: "Record", body: "Capture a real session with the camera on." },
  { step: "02", title: "Replay", body: "Scrub the timeline with skeleton and angles overlaid." },
  { step: "03", title: "Spot failures", body: "Frame where detector disagreed with reality." },
  { step: "04", title: "Adjust thresholds", body: "Tune the per-exercise detector params." },
  { step: "05", title: "Re-record", body: "Verify the fix on fresh footage. Compare." },
];

export function DeveloperModePage() {
  const scopeRef = useRef<HTMLDivElement>(null);
  useRevealAnimations(scopeRef);

  return (
    <div ref={scopeRef}>
      <PageHero
        eyebrow="Developer Mode"
        title="See what the"
        italic="detector sees."
        subtitle="Skeletons, angles, confidence scores, per-frame logs. Everything the pipeline thinks, visible."
        image="/static/images/gym_equipment.jpg"
        overlay={0.7}
      />

      {/* SKELETON OVERLAY */}
      <Section variant="light">
        <SectionHeader
          eyebrow="Skeleton overlay"
          title="Every joint,"
          italic="every angle."
          body="Toggle Developer Mode and the detector's view lights up: 33 landmarks, live joint angles, confidence per point. If the skeleton looks wrong, the feedback is wrong — and now you can see it."
        />

        <div data-reveal className="mt-16">
          <MockWindow label="dev · push_up · live" tone="dark">
            <div className="relative aspect-[16/9] bg-[#0a0a0a]">
              <img
                src="/static/images/pushup.jpg"
                alt=""
                className="absolute inset-0 h-full w-full object-cover opacity-40"
                loading="lazy"
              />
              <div className="absolute inset-0 bg-[#0a0a0a]/30" />

              {/* Simulated skeleton overlay */}
              <svg
                viewBox="0 0 800 450"
                className="absolute inset-0 h-full w-full"
                style={{ mixBlendMode: "screen" }}
              >
                <defs>
                  <filter id="glow">
                    <feGaussianBlur stdDeviation="2.5" result="blur" />
                    <feMerge>
                      <feMergeNode in="blur" />
                      <feMergeNode in="SourceGraphic" />
                    </feMerge>
                  </filter>
                </defs>
                <g stroke="#2f7d5b" strokeWidth="3" fill="none" filter="url(#glow)">
                  {/* Arms */}
                  <line x1="200" y1="200" x2="340" y2="300" />
                  <line x1="340" y1="300" x2="430" y2="220" />
                  {/* Torso */}
                  <line x1="430" y1="220" x2="560" y2="230" />
                  {/* Legs */}
                  <line x1="560" y1="230" x2="660" y2="260" />
                  <line x1="660" y1="260" x2="740" y2="280" />
                  {/* Head */}
                  <circle cx="165" cy="180" r="22" />
                </g>
                <g fill="#AEE710">
                  {[
                    [165, 180],
                    [200, 200],
                    [340, 300],
                    [430, 220],
                    [560, 230],
                    [660, 260],
                    [740, 280],
                  ].map(([x, y]) => (
                    <circle key={`${x}-${y}`} cx={x} cy={y} r="5" />
                  ))}
                </g>
                {/* Angle labels */}
                <g
                  fontFamily="ui-monospace, monospace"
                  fontSize="13"
                  fill="#AEE710"
                  stroke="#0a0a0a"
                  strokeWidth="3"
                  paintOrder="stroke"
                >
                  <text x="355" y="340">
                    elbow 88°
                  </text>
                  <text x="575" y="215">
                    hip 172°
                  </text>
                  <text x="670" y="245">
                    knee 175°
                  </text>
                </g>
                {/* Red warning on hip */}
                <circle cx="560" cy="230" r="14" fill="none" stroke="#b8341c" strokeWidth="2.5">
                  <animate attributeName="r" from="10" to="22" dur="1.5s" repeatCount="indefinite" />
                  <animate
                    attributeName="opacity"
                    from="1"
                    to="0"
                    dur="1.5s"
                    repeatCount="indefinite"
                  />
                </circle>
                <text x="575" y="195" fontFamily="ui-monospace, monospace" fontSize="11" fill="#b8341c">
                  ⚠ hip_sag detected
                </text>
              </svg>

              {/* HUD */}
              <div className="absolute top-5 left-5 font-[family-name:var(--font-mono)] text-[0.7rem] text-[color:var(--color-ink-on-dark-2)] space-y-1">
                <div>frame · 00247</div>
                <div>exercise · push_up</div>
                <div>detector · conf 0.94</div>
                <div className="text-[color:var(--color-bad)]">flag · hip_sag</div>
              </div>
              <div className="absolute bottom-5 right-5 font-[family-name:var(--font-mono)] text-[0.7rem] text-[color:var(--color-ink-on-dark-2)]">
                FORMA · dev
              </div>
            </div>
          </MockWindow>
        </div>
      </Section>

      {/* TIMELINE REPLAY */}
      <Section variant="raised">
        <SectionHeader
          eyebrow="Timeline replay"
          title="Scrub the rep,"
          italic="find the frame."
          body="Every session recording saves the trace. Load any session in dev mode and you get a scrubber with angle plots over time, flag markers, and the ability to annotate frames where the detector was wrong."
        />

        <div data-reveal className="mt-16">
          <MockWindow label="dev · session · 2026-04-14_0832" tone="light">
            <div className="p-8 md:p-10">
              <div className="mb-6 flex items-baseline justify-between flex-wrap gap-4">
                <div>
                  <div className="text-xs uppercase tracking-[0.14em] text-[color:var(--color-gold)] mb-2">
                    Session · push_up · 3 sets · 47 reps
                  </div>
                  <div className="font-[family-name:var(--font-display)] text-2xl text-[color:var(--color-ink)]">
                    Hip angle · full session
                  </div>
                </div>
                <div className="font-[family-name:var(--font-mono)] text-xs text-[color:var(--color-ink-3)]">
                  frame 47 · 01:23 · hip 155°
                </div>
              </div>

              {/* Fake plot */}
              <div className="relative h-40 bg-[color:var(--color-sunken)] rounded-[3px] border border-[color:var(--rule)] overflow-hidden">
                <svg viewBox="0 0 1000 160" preserveAspectRatio="none" className="h-full w-full">
                  <defs>
                    <linearGradient id="area" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#AEE710" stopOpacity="0.35" />
                      <stop offset="100%" stopColor="#AEE710" stopOpacity="0" />
                    </linearGradient>
                  </defs>
                  {/* Reference line at 165 target */}
                  <line x1="0" y1="40" x2="1000" y2="40" stroke="#6b6760" strokeDasharray="3,4" />
                  <text x="8" y="34" fontSize="10" fill="#6b6760" fontFamily="ui-monospace, monospace">
                    target 165°
                  </text>
                  {/* Data path — synthetic wave */}
                  <path
                    d="M0,80 Q50,40 100,60 T200,70 T300,50 T400,90 T500,100 T600,55 T700,75 T800,60 T900,50 T1000,70 L1000,160 L0,160 Z"
                    fill="url(#area)"
                  />
                  <path
                    d="M0,80 Q50,40 100,60 T200,70 T300,50 T400,90 T500,100 T600,55 T700,75 T800,60 T900,50 T1000,70"
                    fill="none"
                    stroke="#AEE710"
                    strokeWidth="2"
                  />
                  {/* Flag marker at frame 47 */}
                  <line x1="470" y1="0" x2="470" y2="160" stroke="#b8341c" strokeWidth="2" />
                  <circle cx="470" cy="100" r="5" fill="#b8341c" />
                  <text
                    x="478"
                    y="18"
                    fontSize="10"
                    fill="#b8341c"
                    fontFamily="ui-monospace, monospace"
                  >
                    ⚠ frame 47
                  </text>
                </svg>
              </div>

              {/* Scrubber */}
              <div className="mt-5 flex items-center gap-4">
                <button className="font-[family-name:var(--font-mono)] text-xs uppercase tracking-[0.1em] text-[color:var(--color-ink-2)] border border-[color:var(--rule)] px-3 py-1.5 rounded-[2px]">
                  ▶ Play
                </button>
                <div className="flex-1 h-1 rounded-full bg-[color:var(--color-sunken)] relative">
                  <div className="absolute inset-y-0 left-0 rounded-full bg-[color:var(--color-gold)] w-[47%]" />
                  <div className="absolute top-1/2 left-[47%] -translate-x-1/2 -translate-y-1/2 h-3 w-3 rounded-full bg-[color:var(--color-ink)] border-2 border-[color:var(--color-page)]" />
                </div>
                <span className="font-[family-name:var(--font-mono)] text-[0.7rem] text-[color:var(--color-ink-3)]">
                  01:23 / 02:57
                </span>
              </div>

              {/* Annotation */}
              <div className="mt-6 rounded-[4px] border border-[color:var(--color-bad)]/30 bg-[color:var(--color-bad)]/5 p-5">
                <div className="text-[0.65rem] uppercase tracking-[0.14em] text-[color:var(--color-bad)] mb-2">
                  Annotation · frame 47
                </div>
                <p className="font-[family-name:var(--font-serif)] italic text-[color:var(--color-ink-2)] leading-[1.55]">
                  &ldquo;Detector said hip angle 155°, actually ~165°. Spine midpoint lost for 3
                  frames — consider smoothing window ↑.&rdquo;
                </p>
              </div>
            </div>
          </MockWindow>
        </div>
      </Section>

      {/* RELIABILITY MATRIX */}
      <Section variant="light">
        <SectionHeader
          eyebrow="Reliability matrix"
          title="What MediaPipe"
          italic="can and can't see."
          body="Not every joint is measurable from every angle. FORMA's reliability matrix documents which angles are trustworthy per exercise, so the detectors can degrade gracefully on the unreliable ones."
        />

        <div data-reveal className="mt-16 overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="text-left border-b border-[color:var(--rule)]">
                <th className="py-4 pr-4 text-xs uppercase tracking-[0.14em] text-[color:var(--color-ink-3)]">
                  Exercise
                </th>
                <th className="py-4 px-4 text-xs uppercase tracking-[0.14em] text-[color:var(--color-ink-3)]">
                  Angle 1
                </th>
                <th className="py-4 px-4 text-xs uppercase tracking-[0.14em] text-[color:var(--color-ink-3)]">
                  Angle 2
                </th>
                <th className="py-4 px-4 text-xs uppercase tracking-[0.14em] text-[color:var(--color-ink-3)]">
                  Angle 3
                </th>
              </tr>
            </thead>
            <tbody>
              {RELIABILITY.map((row) => (
                <tr key={row.exercise} className="border-b border-[color:var(--rule)]">
                  <td className="py-4 pr-4 font-[family-name:var(--font-display)] text-xl text-[color:var(--color-ink)]">
                    {row.exercise}
                  </td>
                  {row.angles.map((a) => (
                    <td key={a.name} className="py-4 px-4">
                      <div className="flex items-center gap-3">
                        <span
                          className="block h-3 w-3 rounded-full"
                          style={{ background: LEVEL_COLOR[a.level].bg }}
                          aria-label={LEVEL_COLOR[a.level].label}
                        />
                        <span className="text-sm text-[color:var(--color-ink-2)]">
                          {a.name}
                        </span>
                        <span className="text-[0.65rem] uppercase tracking-[0.1em] text-[color:var(--color-ink-3)]">
                          {LEVEL_COLOR[a.level].text}
                        </span>
                      </div>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div
          data-reveal
          className="mt-12 rounded-[6px] border border-[color:var(--color-warn)]/30 bg-[color:var(--color-warn)]/5 p-6 max-w-3xl"
        >
          <div className="flex items-start gap-4">
            <span className="font-[family-name:var(--font-mono)] text-xs text-[color:var(--color-warn)] mt-1">
              KNOWN HARD CASE
            </span>
            <p className="text-[color:var(--color-ink-2)] leading-[1.6]">
              Deadlift spine angle. The midpoint between shoulder and hip is unreliable when the
              torso is bent, so FORMA uses an ear-to-hip ratio approximation instead of a direct
              spine-angle read. If confidence drops, the detector stays silent rather than guess.
            </p>
          </div>
        </div>
      </Section>

      {/* ITERATION LOOP */}
      <Section variant="raised">
        <SectionHeader
          eyebrow="Iteration loop"
          title="Record,"
          italic="refine, repeat."
          body="Developer Mode isn't just for debugging — it's how FORMA's detectors get better. Every miscall you spot and annotate becomes the next threshold tune."
        />

        <div className="mt-16 grid gap-4 md:grid-cols-5">
          {LOOP.map((l) => (
            <div
              key={l.step}
              data-reveal
              className="rounded-[4px] border border-[color:var(--rule)] bg-[color:var(--color-page)] p-6"
            >
              <div className="font-[family-name:var(--font-mono)] text-xs text-[color:var(--color-gold)] mb-3">
                {l.step}
              </div>
              <h3 className="font-[family-name:var(--font-display)] text-xl text-[color:var(--color-ink)] leading-[0.95] mb-2">
                {l.title}
              </h3>
              <p className="text-sm text-[color:var(--color-ink-2)] leading-[1.55]">{l.body}</p>
            </div>
          ))}
        </div>
      </Section>

      {/* CTA */}
      <Section variant="dark">
        <div data-reveal className="text-center max-w-3xl mx-auto">
          <h2 className="font-[family-name:var(--font-display)] text-[clamp(2.5rem,7vw,6rem)] leading-[0.92]">
            Transparency
            <br />
            <em className="not-italic font-[family-name:var(--font-serif)] italic text-[color:var(--color-gold-soft)]">
              by default.
            </em>
          </h2>
          <p className="mt-6 font-[family-name:var(--font-serif)] italic text-xl text-[color:var(--color-ink-on-dark-2)]">
            Nothing hidden. If you're curious how the form score came out of your rep, look.
          </p>
          <div className="mt-10">
            <Link
              to="/exercises"
              className="inline-flex items-center gap-2 px-8 py-4 bg-[color:var(--color-gold-soft)] text-[#0A0A0A] text-xs uppercase tracking-[0.14em] font-medium rounded-[2px] hover:bg-[color:var(--color-gold)] transition-colors"
            >
              Enable Developer Mode
            </Link>
          </div>
        </div>
      </Section>
    </div>
  );
}
