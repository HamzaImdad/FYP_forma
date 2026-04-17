import { useRef } from "react";
import { Link } from "react-router-dom";
import { Section } from "@/components/sections/Section";
import { SectionHeader } from "@/components/sections/SectionHeader";
import { GradientImageCard } from "@/components/sections/GradientImageCard";
import { TextOverMedia } from "@/components/sections/TextOverMedia";
import { CinematicHero } from "@/components/sections/CinematicHero";
import { useRevealAnimations } from "@/lib/useRevealAnimations";

const PRINCIPLES = [
  {
    kicker: "Principle 01",
    title: "Specialised beats general",
    body: "A purpose-built state machine with the right angle thresholds will outperform a general-purpose ML model on a feedback task \u2014 every time. FORMA uses dedicated detectors per exercise, not a single model trying to cover all 11.",
  },
  {
    kicker: "Principle 02",
    title: "Silence is a feature",
    body: "Constant feedback is noise. FORMA only speaks when something is wrong. If your form is holding, the app stays out of your way. A good coach doesn't narrate every rep.",
  },
  {
    kicker: "Principle 03",
    title: "Honesty over flattery",
    body: "No vague encouragement. No false positives. Green means your form holds. Red means you need to fix it. If FORMA can't see clearly, it says so \u2014 it doesn't guess.",
  },
];

export function AboutPage() {
  const scopeRef = useRef<HTMLDivElement>(null);
  useRevealAnimations(scopeRef);

  return (
    <div ref={scopeRef}>
      <CinematicHero
        image="/static/images/cinematic/cinematic_rope.jpg"
        anchor="right"
        minHeight="min-h-[82vh]"
      >
        <span
          data-reveal
          className="block font-[family-name:var(--font-mono)] text-[0.7rem] uppercase tracking-[0.32em] text-[color:var(--color-gold-soft)] mb-8"
        >
          / / / ABOUT
        </span>
        <h1
          data-reveal
          className="font-[family-name:var(--font-display)] leading-[0.88] text-[color:var(--color-ink-on-dark)]"
          style={{ fontSize: "clamp(3rem, 8vw, 7rem)" }}
        >
          Form over
          <br />
          <em className="not-italic font-[family-name:var(--font-serif)] italic text-[color:var(--color-gold-soft)]">
            everything.
          </em>
        </h1>
        <p
          data-reveal
          className="mt-8 font-[family-name:var(--font-serif)] italic text-xl md:text-2xl text-[color:var(--color-ink-on-dark-2)] leading-[1.4]"
        >
          A final-year project about trust. Computer vision meets biomechanics — and the two are held accountable to each other.
        </p>
      </CinematicHero>

      {/* ORIGIN STORY */}
      <Section variant="light">
        <div className="grid gap-16 lg:grid-cols-[0.9fr_1.1fr] items-start">
          <div>
            <span className="block text-xs uppercase tracking-[0.24em] text-[color:var(--color-gold)] mb-4">
              The project
            </span>
            <h2
              data-reveal
              className="font-[family-name:var(--font-display)] text-[clamp(2.5rem,6vw,5rem)] leading-[0.92] text-[color:var(--color-ink)] mb-8"
            >
              Built at
              <br />
              <em className="not-italic font-[family-name:var(--font-serif)] italic text-[color:var(--color-gold)]">
                Greenwich.
              </em>
            </h2>
            <p
              data-reveal
              className="text-[color:var(--color-ink-2)] leading-[1.7] mb-5 max-w-xl"
            >
              FORMA is a BSc Computer Science final-year project at the University of Greenwich.
              It started with a simple question: <em>why is bad form so hard to catch on your own?</em>
            </p>
            <p
              data-reveal
              className="text-[color:var(--color-ink-2)] leading-[1.7] mb-5 max-w-xl"
            >
              You can film yourself, but you don't know what to look for. You can ask a trainer,
              but trainers cost money and aren't in your living room at 7am. You can guess from
              YouTube tutorials, but nothing tells you <em>your</em> form is slipping, rep by rep,
              as it happens.
            </p>
            <p
              data-reveal
              className="text-[color:var(--color-ink-2)] leading-[1.7] max-w-xl mb-8"
            >
              So the project became this: take the hardware everyone already has \u2014 a webcam
              \u2014 and build a trainer out of it. One that watches every landmark on every
              frame, knows what good form looks like, and tells you the moment something breaks.
            </p>
            <Link
              data-reveal
              to="/how-it-works"
              className="inline-flex items-center gap-2 px-6 py-3 border border-[color:var(--color-ink)]/20 text-[color:var(--color-ink)] text-[11px] uppercase tracking-[0.18em] font-medium rounded-[2px] hover:border-[color:var(--color-gold)] hover:text-[color:var(--color-gold)] transition-colors"
            >
              See how it works \u2192
            </Link>
          </div>

          <div data-reveal>
            <GradientImageCard
              images={["/static/images/training.jpg", "/static/images/gym_dark.jpg"]}
              color="#AEE710"
              aspect="tall"
              kicker="In the gym"
              title="Where it trains."
              body="Every detector was tuned with real footage \u2014 push-ups first, then squats, deadlifts, the rest. Lighting, angles, body types. The thresholds come from biomechanics literature; the refinements come from what actually holds up on camera."
            />
          </div>
        </div>
      </Section>

      {/* PRINCIPLES */}
      <Section variant="light">
        <SectionHeader
          eyebrow="Principles"
          title="What FORMA"
          italic="believes."
          body="Three ideas ran through every design decision. They explain why the app feels the way it does."
        />

        <div className="mt-16 space-y-16">
          {PRINCIPLES.map((p, i) => (
            <div
              key={p.title}
              data-reveal
              className={`grid gap-12 md:grid-cols-2 items-center ${
                i % 2 === 1 ? "md:[&>*:first-child]:order-2" : ""
              }`}
            >
              <GradientImageCard
                images={
                  i === 0
                    ? ["/static/images/ex_squat.jpg", "/static/images/ex_deadlift.jpg"]
                    : i === 1
                      ? ["/static/images/atmosphere.jpg", "/static/images/stretch.jpg"]
                      : ["/static/images/ex_pushup.jpg", "/static/images/ex_plank.jpg"]
                }
                color={i === 0 ? "#AEE710" : i === 1 ? "#F0F0F0" : "#C2F04A"}
                aspect="portrait"
              />
              <div>
                <span className="block font-[family-name:var(--font-mono)] text-[0.65rem] uppercase tracking-[0.18em] text-[color:var(--color-gold)] mb-4">
                  {p.kicker}
                </span>
                <h3 className="font-[family-name:var(--font-display)] text-[clamp(2rem,4vw,3.5rem)] leading-[0.92] text-[color:var(--color-ink)] mb-6">
                  {p.title}
                </h3>
                <p className="text-[color:var(--color-ink-2)] leading-[1.7] max-w-xl">
                  {p.body}
                </p>
              </div>
            </div>
          ))}
        </div>
      </Section>

      {/* STACK */}
      <Section variant="raised">
        <SectionHeader
          eyebrow="Under the hood"
          title="The stack."
        />

        <div data-reveal className="mt-12 grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          {[
            {
              layer: "Vision",
              tech: "MediaPipe BlazePose",
              body: "Google's real-time pose estimator. 33 landmarks, world coordinates, visibility per point.",
            },
            {
              layer: "Detection",
              tech: "Python + NumPy",
              body: "Per-exercise state machines and form checks. No ML at inference \u2014 deterministic, fast, debuggable.",
            },
            {
              layer: "Backend",
              tech: "Flask + Socket.IO",
              body: "Streams webcam frames to the detector and pushes feedback back to the browser in real time.",
            },
            {
              layer: "Frontend",
              tech: "React 19 + GSAP",
              body: "TypeScript, Tailwind 4, Framer Motion, Lenis smooth scroll. Editorial typography and scroll-driven motion.",
            },
          ].map((s) => (
            <div
              key={s.layer}
              className="rounded-[6px] border border-[color:var(--rule)] bg-[color:var(--color-page)] p-6"
            >
              <div className="font-[family-name:var(--font-mono)] text-[0.65rem] uppercase tracking-[0.18em] text-[color:var(--color-gold)] mb-3">
                {s.layer}
              </div>
              <h3 className="font-[family-name:var(--font-display)] text-2xl text-[color:var(--color-ink)] leading-[0.95] mb-3">
                {s.tech}
              </h3>
              <p className="text-sm text-[color:var(--color-ink-2)] leading-[1.55]">{s.body}</p>
            </div>
          ))}
        </div>
      </Section>

      {/* COVERAGE */}
      <Section variant="light">
        <SectionHeader
          eyebrow="Coverage"
          title="Eleven exercises,"
          italic="one project."
          body="FORMA's detectors cover the movements most people actually train \u2014 compound lifts, bodyweight staples, and a few isolation moves."
        />

        <div data-reveal className="mt-16 grid gap-4 grid-cols-2 md:grid-cols-5">
          {[
            ["Squat", "ex_squat"],
            ["Deadlift", "ex_deadlift"],
            ["Lunge", "ex_lunge"],
            ["Pull-up", "ex_pullup"],
            ["Push-up", "ex_pushup"],
            ["Plank", "ex_plank"],
            ["Bicep Curl", "ex_bicep_curl"],
            ["Tricep Dip", "ex_tricep_dip"],
            ["V-Up Crunch", "ex_crunch"],
            ["Lateral Raise", "ex_lateral_raise"],
            ["Side Plank", "ex_side_plank"],
          ].map(([name, slug]) => (
            <div
              key={slug}
              className="relative aspect-square overflow-hidden rounded-[6px] ring-1 ring-[color:var(--rule)] group"
            >
              <img
                src={`/static/images/${slug}.jpg`}
                alt={name}
                loading="lazy"
                className="absolute inset-0 h-full w-full object-cover transition-transform duration-700 ease-[var(--ease-out-editorial)] group-hover:scale-[1.06]"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-[color:var(--color-contrast)]/85 via-[color:var(--color-contrast)]/20 to-transparent" />
              <div className="absolute inset-x-0 bottom-0 p-4">
                <span className="font-[family-name:var(--font-display)] text-lg text-[color:var(--color-ink-on-dark)]">
                  {name}
                </span>
              </div>
            </div>
          ))}
        </div>
      </Section>

      {/* CLOSING */}
      <TextOverMedia
        image="/static/images/cta_bg.jpg"
        anchor="center"
        intensity={0.85}
        imgOpacity={0.5}
        className="bg-[color:var(--color-contrast)] text-[color:var(--color-ink-on-dark)]"
      >
        <div className="mx-auto max-w-[1440px] px-6 md:px-10 py-32 text-center">
          <h2
            data-reveal
            className="font-[family-name:var(--font-display)] text-[clamp(3rem,8vw,7rem)] leading-[0.92]"
          >
            Train
            <em className="not-italic font-[family-name:var(--font-serif)] italic text-[color:var(--color-gold-soft)]">
              {" "}with form.
            </em>
          </h2>
          <p
            data-reveal
            className="mt-8 max-w-2xl mx-auto font-[family-name:var(--font-serif)] italic text-2xl text-[color:var(--color-ink-on-dark)]"
          >
            Everything you've read is running in the browser right now. Press the button.
          </p>
          <div className="mt-12">
            <Link
              to="/exercises"
              className="inline-flex items-center gap-2 px-8 py-4 bg-[color:var(--color-gold-soft)] text-[#0A0A0A] text-xs uppercase tracking-[0.14em] font-medium rounded-[2px] hover:bg-[color:var(--color-gold)] transition-colors"
            >
              Start Your First Session
            </Link>
          </div>
        </div>
      </TextOverMedia>
    </div>
  );
}
