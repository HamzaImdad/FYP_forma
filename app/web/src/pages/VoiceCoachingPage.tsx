import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Section } from "@/components/sections/Section";
import { SectionHeader } from "@/components/sections/SectionHeader";
import { FeatureCard } from "@/components/sections/FeatureCard";
import { MockWindow } from "@/components/sections/MockWindow";
import { CinematicHero } from "@/components/sections/CinematicHero";
import { GradientGlow } from "@/components/sections/GradientGlow";
import { useRevealAnimations } from "@/lib/useRevealAnimations";

const HIP_SAG_VARIANTS = [
  "Keep your hips up",
  "Tighten your core, hips level",
  "Don't let those hips drop",
  "Core tight, body straight",
];

const HEAD_DOWN_VARIANTS = [
  "Head straight, eyes down",
  "Keep your neck neutral",
  "Don't crane your head back",
];

const EXERCISE_CUES: Record<string, { error: string; phrases: string[] }[]> = {
  "Push-up": [
    { error: "hip_sag", phrases: HIP_SAG_VARIANTS },
    { error: "head_down", phrases: HEAD_DOWN_VARIANTS },
  ],
  Squat: [
    {
      error: "knee_valgus",
      phrases: ["Knees out, not in", "Drive your knees over your toes", "Push the floor apart"],
    },
    {
      error: "depth_shallow",
      phrases: ["Go a little deeper", "Thighs to parallel", "Sit down more"],
    },
  ],
  Plank: [
    {
      error: "hip_pike",
      phrases: ["Hips down, body straight", "Drop your hips a touch", "Flat line head to heel"],
    },
  ],
};

export function VoiceCoachingPage() {
  const scopeRef = useRef<HTMLDivElement>(null);
  const [cueIndex, setCueIndex] = useState(0);

  useRevealAnimations(scopeRef);

  useEffect(() => {
    const id = setInterval(() => {
      setCueIndex((i) => (i + 1) % HIP_SAG_VARIANTS.length);
    }, 2400);
    return () => clearInterval(id);
  }, []);

  return (
    <div ref={scopeRef}>
      <CinematicHero
        image="/static/images/cinematic/cinematic_battle.jpg"
        anchor="right"
        minHeight="min-h-[74vh]"
      >
        <span
          data-reveal
          className="block font-[family-name:var(--font-mono)] text-[0.7rem] uppercase tracking-[0.32em] text-[color:var(--color-gold-soft)] mb-8"
        >
          / / / VOICE COACHING · LIVE
        </span>
        <h1
          data-reveal
          className="font-[family-name:var(--font-display)] leading-[0.88] text-[color:var(--color-ink-on-dark)]"
          style={{ fontSize: "clamp(3rem, 8vw, 6.5rem)" }}
        >
          A coach in
          <br />
          <em className="not-italic font-[family-name:var(--font-serif)] italic text-[color:var(--color-gold-soft)]">
            your browser.
          </em>
        </h1>
        <p
          data-reveal
          className="mt-8 font-[family-name:var(--font-serif)] italic text-xl md:text-2xl text-[color:var(--color-ink-on-dark-2)] leading-[1.4]"
        >
          Hard-coded commands, instant cues, works offline. No cloud, no latency.
        </p>
      </CinematicHero>

      {/* WHY NOT AN LLM */}
      <Section variant="light" className="relative overflow-hidden">
        <GradientGlow position="top-right" intensity="medium" />
        <GradientGlow position="bottom-left" intensity="subtle" />
        <SectionHeader
          eyebrow="Why not an LLM"
          title="100 milliseconds"
          italic="or nothing."
          body={
            <>
              A real trainer corrects you in about <em>100 ms</em>. A cloud LLM round-trip takes
              500 ms to 2 seconds. The rep is over before the app finishes speaking. Latency kills
              the feature. So we don't use one.
            </>
          }
        />

        <div data-reveal className="mt-16 grid gap-6 md:grid-cols-2">
          <div className="rounded-[6px] border border-[color:var(--color-bad)]/30 bg-[color:var(--color-bad)]/5 p-8">
            <span className="block text-xs uppercase tracking-[0.14em] text-[color:var(--color-bad)] mb-3">
              LLM in the hot path
            </span>
            <div className="font-[family-name:var(--font-display)] text-5xl text-[color:var(--color-ink)] leading-[0.9]">
              500–2000<small className="text-lg ml-1">ms</small>
            </div>
            <p className="mt-3 text-sm text-[color:var(--color-ink-2)] leading-[1.6]">
              Round-trip to a cloud model. You've already finished the rep. The cue arrives late
              and useless.
            </p>
          </div>
          <div className="rounded-[6px] border border-[color:var(--color-good)]/30 bg-[color:var(--color-good)]/5 p-8">
            <span className="block text-xs uppercase tracking-[0.14em] text-[color:var(--color-good)] mb-3">
              FORMA's approach
            </span>
            <div className="font-[family-name:var(--font-display)] text-5xl text-[color:var(--color-ink)] leading-[0.9]">
              ~100<small className="text-lg ml-1">ms</small>
            </div>
            <p className="mt-3 text-sm text-[color:var(--color-ink-2)] leading-[1.6]">
              Hard-coded phrases fire the instant the detector flags a form break. Spoken through
              the browser's own voice, offline.
            </p>
          </div>
        </div>
      </Section>

      {/* EXAMPLE PHRASES */}
      <Section variant="raised">
        <SectionHeader
          eyebrow="Example phrases"
          title="Three to five variants"
          italic="per error."
          body="So it never sounds robotic. The coach rotates through alternate phrasings; the same error never triggers the same words twice in a row."
        />

        <div data-reveal className="mt-16 grid gap-10 lg:grid-cols-[1.1fr_0.9fr] items-center">
          <MockWindow label="push_up · hip_sag" tone="dark">
            <div className="p-10 min-h-[340px] flex flex-col justify-center">
              <div className="flex items-center gap-3 mb-8">
                <span className="relative flex h-3 w-3">
                  <span className="absolute inline-flex h-full w-full rounded-full bg-[color:var(--color-bad)] opacity-75 animate-ping" />
                  <span className="relative inline-flex h-3 w-3 rounded-full bg-[color:var(--color-bad)]" />
                </span>
                <span className="text-xs uppercase tracking-[0.14em] text-[color:var(--color-ink-on-dark-2)]">
                  Error detected · hip_sag
                </span>
              </div>
              <div
                key={cueIndex}
                className="font-[family-name:var(--font-serif)] italic text-3xl md:text-4xl text-[color:var(--color-ink-on-dark)] leading-[1.15] animate-[fadeInUp_0.45s_ease-out]"
              >
                &ldquo;{HIP_SAG_VARIANTS[cueIndex]}&rdquo;
              </div>
              <div className="mt-10 flex gap-2">
                {HIP_SAG_VARIANTS.map((_, i) => (
                  <span
                    key={i}
                    className="block h-[3px] flex-1 rounded-full transition-colors duration-500"
                    style={{
                      background:
                        i === cueIndex
                          ? "var(--color-gold-soft)"
                          : "rgba(199,195,187,0.18)",
                    }}
                  />
                ))}
              </div>
              <style>{`
                @keyframes fadeInUp { from { opacity: 0; transform: translateY(8px) } to { opacity: 1; transform: translateY(0) } }
              `}</style>
            </div>
          </MockWindow>

          <div>
            <pre className="font-[family-name:var(--font-mono)] text-xs leading-[1.65] p-6 rounded-[6px] bg-[color:var(--color-contrast)] text-[color:var(--color-ink-on-dark)] overflow-x-auto">
              {`{
  "push_up": {
    "hip_sag": [
      "Keep your hips up",
      "Tighten your core, hips level",
      "Don't let those hips drop",
      "Core tight, body straight"
    ],
    "head_down": [
      "Head straight, eyes down",
      "Keep your neck neutral",
      "Don't crane your head back"
    ]
  }
}`}
            </pre>
            <p className="mt-4 text-sm text-[color:var(--color-ink-3)] leading-[1.6]">
              One JSON file per exercise. Every form error the detectors flag has its own set of
              phrases. Debounced so the same cue never re-triggers within a cooldown window.
            </p>
          </div>
        </div>
      </Section>

      {/* EXERCISE COVERAGE */}
      <Section variant="light">
        <SectionHeader
          eyebrow="Coverage"
          title="Every exercise,"
          italic="every error."
          body="All eleven dedicated detectors feed the voice layer. These are a few of the cues FORMA speaks in real time."
        />

        <div data-reveal className="mt-16 grid gap-6 md:grid-cols-3">
          {Object.entries(EXERCISE_CUES).map(([exercise, cues]) => (
            <FeatureCard
              key={exercise}
              index={exercise}
              title={cues[0].error.replace(/_/g, " ")}
              body={
                <ul className="space-y-2">
                  {cues[0].phrases.map((p) => (
                    <li
                      key={p}
                      className="font-[family-name:var(--font-serif)] italic text-[1.05rem]"
                    >
                      &ldquo;{p}&rdquo;
                    </li>
                  ))}
                </ul>
              }
            />
          ))}
        </div>
      </Section>

      {/* OFFLINE */}
      <Section variant="dark">
        <div data-reveal className="max-w-3xl">
          <span className="block text-xs uppercase tracking-[0.24em] text-[color:var(--color-gold-soft)] mb-4">
            Works offline
          </span>
          <h2 className="font-[family-name:var(--font-display)] text-[clamp(2.5rem,6vw,5rem)] leading-[0.92]">
            No cloud.
            <br />
            <em className="not-italic font-[family-name:var(--font-serif)] italic text-[color:var(--color-gold-soft)]">
              No excuses.
            </em>
          </h2>
          <p className="mt-6 max-w-xl text-[color:var(--color-ink-on-dark-2)] leading-[1.6]">
            FORMA speaks through the browser's built-in Web Speech API. The phrase library ships
            with the app. Your form feedback works on a plane, in a basement gym, behind a hotel
            wall that kills your signal.
          </p>
          <div className="mt-10">
            <Link
              to="/exercises"
              className="inline-flex items-center gap-2 px-7 py-4 bg-[color:var(--color-gold-soft)] text-[#0A0A0A] text-xs uppercase tracking-[0.14em] font-medium rounded-[2px] hover:bg-[color:var(--color-gold)] transition-colors"
            >
              Start A Session
            </Link>
          </div>
        </div>
      </Section>
    </div>
  );
}
