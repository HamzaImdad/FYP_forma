import { useRef } from "react";
import { Link } from "react-router-dom";
import { Section } from "@/components/sections/Section";
import { SectionHeader } from "@/components/sections/SectionHeader";
import { TextOverMedia } from "@/components/sections/TextOverMedia";
import { CinematicHero } from "@/components/sections/CinematicHero";
import { useRevealAnimations } from "@/lib/useRevealAnimations";

const PIPELINE = [
  {
    step: "01",
    title: "Webcam frame",
    body: "Captured at full webcam framerate. No special hardware, no wearables, no markers. Any laptop with a camera works \u2014 FORMA meets you where you already train.",
    detail: "Each frame is grabbed directly from the browser's MediaStream. No upload, no buffer, no round-trip.",
  },
  {
    step: "02",
    title: "MediaPipe BlazePose",
    body: "Google's pose estimator tracks 33 body landmarks in real time, with world-space coordinates and a confidence score per point.",
    detail: "BlazePose runs fully in the browser via WebAssembly. The model weights ship with the page; nothing is fetched at runtime.",
  },
  {
    step: "03",
    title: "Joint angles",
    body: "Vector math turns raw landmarks into biomechanical angles \u2014 knee flexion, hip hinge, elbow extension, spine neutrality.",
    detail: "Three-point angles computed per joint, filtered through a small rolling window to smooth jitter without introducing lag.",
  },
  {
    step: "04",
    title: "Dedicated detector",
    body: "A state machine per exercise: TOP \u2192 GOING_DOWN \u2192 BOTTOM \u2192 GOING_UP \u2192 TOP. No shared logic between squat and deadlift.",
    detail: "Each detector knows what bad form looks like for its specific exercise. Squat watches knee depth and torso lean. Bench watches elbow flare and bar path. The thresholds come from peer-reviewed biomechanics.",
  },
  {
    step: "05",
    title: "Form score",
    body: "A weighted score comes out of the detector every frame, based on the form checks that matter for that specific exercise.",
    detail: "Green when your alignment holds. Orange when it slips. Red when something needs immediate correction \u2014 so you always know where you stand without reading numbers.",
  },
  {
    step: "06",
    title: "Feedback",
    body: "Words, not numbers. \u201cHips dropping.\u201d \u201cGo a little deeper.\u201d Spoken through the browser's own voice when the rep needs it.",
    detail: "One correction at a time. Silence when you're good. FORMA only speaks when something is wrong \u2014 a good coach doesn't narrate every rep.",
  },
];

export function HowItWorksPage() {
  const scopeRef = useRef<HTMLDivElement>(null);
  useRevealAnimations(scopeRef);

  return (
    <div ref={scopeRef}>
      <CinematicHero
        image="/static/images/cinematic/cinematic_dumbbells.jpg"
        anchor="left"
        minHeight="min-h-[82vh]"
      >
        <span
          data-reveal
          className="block font-[family-name:var(--font-mono)] text-[0.7rem] uppercase tracking-[0.32em] text-[color:var(--color-gold-soft)] mb-8"
        >
          / / / HOW IT WORKS
        </span>
        <h1
          data-reveal
          className="font-[family-name:var(--font-display)] leading-[0.88] text-[color:var(--color-ink-on-dark)]"
          style={{ fontSize: "clamp(3rem, 8vw, 7rem)" }}
        >
          From webcam
          <br />
          <em className="not-italic font-[family-name:var(--font-serif)] italic text-[color:var(--color-gold-soft)]">
            to form cue.
          </em>
        </h1>
        <p
          data-reveal
          className="mt-8 font-[family-name:var(--font-serif)] italic text-xl md:text-2xl text-[color:var(--color-ink-on-dark-2)] leading-[1.4]"
        >
          Six stages run every frame. Start to finish, under 50 ms on a consumer laptop. Nothing leaves your machine.
        </p>
      </CinematicHero>

      {/* THE PIPELINE (detailed) */}
      <Section variant="light">
        <SectionHeader
          eyebrow="The pipeline"
          title="Six stages,"
          italic="every frame."
          body="Each stage is deterministic, fast, and debuggable. No black-box ML at inference. If something goes wrong, the trace shows exactly where."
        />

        <div className="mt-20 space-y-6">
          {PIPELINE.map((p, i) => (
            <div
              key={p.step}
              data-reveal
              className={`grid gap-8 md:grid-cols-[auto_1fr_1.2fr] items-start rounded-[6px] border border-[color:var(--rule)] bg-[color:var(--color-raised)] p-8 md:p-12 hover:border-[color:var(--color-gold)]/40 transition-colors ${i % 2 === 1 ? "md:bg-[color:var(--color-page)]" : ""}`}
            >
              <div className="font-[family-name:var(--font-display)] text-[5rem] md:text-[7rem] leading-[0.8] text-[color:var(--color-gold)] min-w-[6rem]">
                {p.step}
              </div>
              <div>
                <h3 className="font-[family-name:var(--font-display)] text-3xl md:text-4xl text-[color:var(--color-ink)] leading-[0.95] mb-4">
                  {p.title}
                </h3>
                <p className="text-[color:var(--color-ink-2)] leading-[1.65] max-w-md">
                  {p.body}
                </p>
              </div>
              <div className="md:pl-8 md:border-l md:border-[color:var(--rule)]">
                <span className="block font-[family-name:var(--font-mono)] text-[0.65rem] uppercase tracking-[0.18em] text-[color:var(--color-gold)] mb-3">
                  Under the hood
                </span>
                <p className="text-[0.95rem] text-[color:var(--color-ink-2)] leading-[1.65]">
                  {p.detail}
                </p>
              </div>
            </div>
          ))}
        </div>
      </Section>

      {/* PERFORMANCE */}
      <Section variant="raised">
        <div className="grid gap-16 lg:grid-cols-[0.9fr_1.1fr] items-center">
          <div>
            <span className="block text-xs uppercase tracking-[0.24em] text-[color:var(--color-gold)] mb-4">
              Performance
            </span>
            <h2
              data-reveal
              className="font-[family-name:var(--font-display)] text-[clamp(2.5rem,6vw,5rem)] leading-[0.92] text-[color:var(--color-ink)] mb-8"
            >
              Under
              <br />
              <em className="not-italic font-[family-name:var(--font-serif)] italic text-[color:var(--color-gold)]">
                50 milliseconds.
              </em>
            </h2>
            <p
              data-reveal
              className="text-[color:var(--color-ink-2)] leading-[1.7] mb-5 max-w-xl"
            >
              A full pipeline pass \u2014 capture, pose estimation, angle math, detection,
              scoring, feedback \u2014 completes in under 50 ms on a consumer laptop. That
              means you see the correction before you finish the rep.
            </p>
            <p
              data-reveal
              className="text-[color:var(--color-ink-2)] leading-[1.7] max-w-xl"
            >
              Nothing round-trips to a server. Nothing waits in a queue. The entire loop
              lives on your machine, in your browser, running at webcam framerate.
            </p>
          </div>
          <div
            data-reveal
            className="grid grid-cols-3 gap-4"
          >
            {[
              { value: "<50", unit: "ms", label: "Per frame" },
              { value: "60", unit: "fps", label: "Realtime" },
              { value: "33", unit: "pts", label: "Landmarks" },
            ].map((s) => (
              <div
                key={s.label}
                className="rounded-[6px] border border-[color:var(--rule)] bg-[color:var(--color-page)] p-6 text-center"
              >
                <div className="font-[family-name:var(--font-display)] text-5xl text-[color:var(--color-ink)] leading-[0.9]">
                  {s.value}
                  <small className="text-[0.4em] ml-1 text-[color:var(--color-gold)] align-super font-[family-name:var(--font-sans)]">
                    {s.unit}
                  </small>
                </div>
                <div className="mt-3 font-[family-name:var(--font-mono)] text-[0.65rem] uppercase tracking-[0.18em] text-[color:var(--color-ink-2)]">
                  {s.label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* ON-DEVICE / PRIVACY */}
      <Section variant="light">
        <div className="grid gap-16 lg:grid-cols-[1.1fr_0.9fr] items-center">
          <div data-reveal className="relative aspect-[4/3] overflow-hidden rounded-[6px] ring-1 ring-[color:var(--rule)]">
            <img
              src="/static/images/training.jpg"
              alt="Private in-home training"
              className="absolute inset-0 h-full w-full object-cover"
            />
            <div className="absolute inset-0 bg-gradient-to-br from-[color:var(--color-contrast)]/70 via-transparent to-transparent" />
            <div className="absolute top-6 left-6 text-[color:var(--color-ink-on-dark)]">
              <span className="block font-[family-name:var(--font-mono)] text-[0.6rem] uppercase tracking-[0.22em] text-[color:var(--color-gold-soft)]">
                Private by design
              </span>
              <p className="mt-3 font-[family-name:var(--font-serif)] italic text-2xl leading-tight max-w-[14ch]">
                Nothing leaves your machine.
              </p>
            </div>
          </div>
          <div>
            <span className="block text-xs uppercase tracking-[0.24em] text-[color:var(--color-gold)] mb-4">
              On-device processing
            </span>
            <h2
              data-reveal
              className="font-[family-name:var(--font-display)] text-[clamp(2.2rem,5vw,4rem)] leading-[0.95] text-[color:var(--color-ink)] mb-8"
            >
              Your body,
              <br />
              <em className="not-italic font-[family-name:var(--font-serif)] italic text-[color:var(--color-gold)]">
                your machine.
              </em>
            </h2>
            <p data-reveal className="text-[color:var(--color-ink-2)] leading-[1.7] mb-5">
              No video is uploaded. No frames are sent to a server. No silhouettes, no
              biometric templates, no cloud copy of your session. Pose estimation runs
              locally; the only thing the backend ever sees is the numeric session summary
              you choose to save.
            </p>
            <p data-reveal className="text-[color:var(--color-ink-2)] leading-[1.7]">
              When FORMA can't see clearly, it says so \u2014 it doesn't guess. When it
              doesn't need to speak, it stays quiet. Trust is built by doing less, not more.
            </p>
          </div>
        </div>
      </Section>

      {/* DETECTORS CALLOUT */}
      <Section variant="raised">
        <div data-reveal className="text-center max-w-3xl mx-auto">
          <span className="block text-xs uppercase tracking-[0.24em] text-[color:var(--color-gold)] mb-4">
            Dedicated detectors
          </span>
          <h2 className="font-[family-name:var(--font-display)] text-[clamp(2.2rem,5vw,4rem)] leading-[0.95] text-[color:var(--color-ink)] mb-6">
            Eleven exercises,
            <br />
            <em className="not-italic font-[family-name:var(--font-serif)] italic text-[color:var(--color-gold)]">
              eleven detectors.
            </em>
          </h2>
          <p className="text-[color:var(--color-ink-2)] leading-[1.7] mb-10">
            Every exercise gets its own state machine and its own form checks. A squat
            detector does not share logic with a deadlift detector. Specialised beats
            general, every time.
          </p>
          <div className="flex flex-wrap gap-3 justify-center">
            <Link
              to="/exercises"
              className="inline-flex items-center gap-2 px-6 py-3 border border-[color:var(--color-ink)]/20 text-[color:var(--color-ink)] text-[11px] uppercase tracking-[0.18em] font-medium rounded-[2px] hover:border-[color:var(--color-gold)] hover:text-[color:var(--color-gold)] transition-colors"
            >
              Browse the eleven \u2192
            </Link>
            <Link
              to="/features"
              className="inline-flex items-center gap-2 px-6 py-3 border border-[color:var(--color-ink)]/20 text-[color:var(--color-ink)] text-[11px] uppercase tracking-[0.18em] font-medium rounded-[2px] hover:border-[color:var(--color-gold)] hover:text-[color:var(--color-gold)] transition-colors"
            >
              See the features \u2192
            </Link>
          </div>
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
            Try it
            <em className="not-italic font-[family-name:var(--font-serif)] italic text-[color:var(--color-gold-soft)]">
              {" "}yourself.
            </em>
          </h2>
          <p
            data-reveal
            className="mt-8 max-w-2xl mx-auto font-[family-name:var(--font-serif)] italic text-2xl text-[color:var(--color-ink-on-dark)]"
          >
            Everything you've read is running in the browser right now.
          </p>
          <div className="mt-12">
            <Link
              to="/login?tab=signup"
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
