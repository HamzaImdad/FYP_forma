import { useEffect, useRef, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { motion, useScroll, useTransform } from "framer-motion";
import { useGSAP } from "@gsap/react";
import { gsap, registerGsap, ScrollTrigger } from "@/lib/gsap";
import { TextOverMedia } from "@/components/sections/TextOverMedia";
import { VideoBackdrop } from "@/components/sections/VideoBackdrop";
import { GradientGlow } from "@/components/sections/GradientGlow";
import { useAuth } from "@/context/AuthContext";

registerGsap();

const STATS = [
  { value: "11", label: "Exercises", caption: "Covered by dedicated detectors" },
  { value: "33", label: "Landmarks", caption: "Tracked every frame, world-space" },
  { value: "60", suffix: "fps", label: "Real-time", caption: "Feedback while you move" },
];

const HOW = [
  {
    num: "01",
    title: "Pose Detection",
    body: (
      <>
        <em>MediaPipe BlazePose</em> tracks 33 body landmarks in real time from your webcam — no
        wearables, no markers, no setup.
      </>
    ),
  },
  {
    num: "02",
    title: "Form Analysis",
    body: (
      <>
        Joint angles, body ratios and symmetry metrics are computed every frame, then compared
        against <em>biomechanical thresholds</em> from peer-reviewed literature.
      </>
    ),
  },
  {
    num: "03",
    title: "Dedicated Detector",
    body: (
      <>
        Each exercise has its own state machine and form thresholds. Squat doesn't share logic with
        deadlift. <em>Specialised, not general.</em>
      </>
    ),
  },
  {
    num: "04",
    title: "Live Feedback",
    body: (
      <>
        Colour-coded skeleton overlay, on-screen cues and a continuous form score tell you{" "}
        <em>exactly what to fix</em>, rep by rep, set by set.
      </>
    ),
  },
];

const FEATURES = [
  {
    img: "/static/images/feature_ai.jpg",
    idx: "01 / 04",
    title: "Real-time, no lag.",
    body: (
      <>
        Frames are processed at full speed with GPU acceleration. The feedback you see is the form
        you're holding <em>right now</em> — not three seconds ago.
      </>
    ),
  },
  {
    img: "/static/images/feature_form.jpg",
    idx: "02 / 04",
    title: "Joint-level honesty.",
    body: (
      <>
        Green when your alignment holds. Orange when it slips. Red when something needs{" "}
        <em>immediate correction</em>. No vague encouragement, no false positives.
      </>
    ),
  },
  {
    img: "/static/images/training.jpg",
    idx: "03 / 04",
    title: "No equipment, no excuses.",
    body: (
      <>
        Works with just your webcam. No wearables, no sensors, no calibration ritual. Any laptop or
        desktop with a camera is <em>all you need</em>.
      </>
    ),
  },
  {
    img: "/static/images/atmosphere.jpg",
    idx: "04 / 04",
    title: "Sessions you can review.",
    body: (
      <>
        Per-rep form scores, set breakdowns, common-mistake heatmaps. Every session is logged so
        you can <em>see yourself improve</em> over weeks, not days.
      </>
    ),
  },
];

const EXERCISES = [
  ["01", "Squat", "ex_squat"],
  ["02", "Deadlift", "ex_deadlift"],
  ["03", "Pull-up", "ex_pullup"],
  ["04", "Push-up", "ex_pushup"],
  ["05", "Plank", "ex_plank"],
  ["06", "Bicep Curl", "ex_bicep_curl"],
  ["07", "Tricep Dip", "ex_tricep_dip"],
  ["08", "V-Up Crunch", "ex_crunch"],
  ["09", "Lateral Raise", "ex_lateral_raise"],
  ["10", "Side Plank", "ex_side_plank"],
];

const TRAINER_PILLARS: { title: string; body: ReactNode }[] = [
  {
    title: "One correction at a time",
    body: (
      <>
        The moment your form slips, <em>one cue</em>. No walls of text. No five warnings at once.
        Fix the important thing, then move on.
      </>
    ),
  },
  {
    title: "Words, not numbers",
    body: (
      <>
        "Hips dropping." Not "Score: 0.62." You don't need <em>math mid-rep</em>. FORMA speaks
        like a coach, not a dashboard.
      </>
    ),
  },
  {
    title: "Silence means you're good",
    body: (
      <>
        When your form holds, FORMA <em>stays out of your way</em>. No nagging, no cheerleading,
        no constant affirmations. Just space to train.
      </>
    ),
  },
  {
    title: "Patient before the first rep",
    body: (
      <>
        Getting into position is not a failure. FORMA waits until you're ready, counts{" "}
        <em>3–2–1</em>, then starts watching.
      </>
    ),
  },
];

const PRODUCT_FEATURES: {
  title: string;
  body: string;
  to: string;
  image: string;
  tag: string;
}[] = [
  {
    title: "Voice Coaching",
    body: "A coach in your browser. No cloud, no lag.",
    to: "/voice-coaching",
    image: "/static/images/atmosphere.jpg",
    tag: "Live",
  },
  {
    title: "AI Chatbot",
    body: "Ask FORMA about your training history.",
    to: "/chatbot",
    image: "/static/images/feature_ai.jpg",
    tag: "Ask",
  },
  {
    title: "Plans & Goals",
    body: "Tell it your goal. Get a plan.",
    to: "/plans",
    image: "/static/images/training.jpg",
    tag: "Plan",
  },
  {
    title: "Milestones",
    body: "Progress you can feel, not guilt.",
    to: "/milestones",
    image: "/static/images/gym_dark.jpg",
    tag: "Track",
  },
];

const SCIENCE: { label: string; num: string; title: string; body: string }[] = [
  {
    label: "Pose estimation",
    num: "33",
    title: "Body landmarks, every frame",
    body: "MediaPipe BlazePose detects 33 points in world space — hips, knees, elbows, ears, the full skeleton. Tracked at full webcam framerate, hip-centered, in meters.",
  },
  {
    label: "Biomechanics",
    num: "10+",
    title: "Peer-reviewed thresholds",
    body: "Every detector's angle thresholds trace back to research on joint range and injury risk. Not guessed, not generic — pulled from the literature, validated frame by frame.",
  },
  {
    label: "Architecture",
    num: "1:1",
    title: "One detector per exercise",
    body: "Squat logic doesn't share code with deadlift. Each exercise has its own state machine, its own form checks, its own idea of what 'good' means. Specialised, not general.",
  },
];

export function Home() {
  const { user } = useAuth();
  // Logged-in visitors drop straight into a push-up session (our most
  // reliable detector); logged-out visitors are bounced to signup so they
  // don't hit a protected route with no context.
  const ctaTarget = user ? "/workout/pushup" : "/login?tab=signup";
  const exerciseHref = (slug: string) =>
    user ? `/workout/${slug}` : "/login?tab=signup";
  const heroRef = useRef<HTMLDivElement>(null);
  const heroImgRef = useRef<HTMLVideoElement>(null);
  const wordmarkRef = useRef<HTMLHeadingElement>(null);

  // GSAP: pin hero, parallax bg, stagger wordmark chars, draw rule line
  useGSAP(
    () => {
      const chars = gsap.utils.toArray<HTMLSpanElement>(".fh-char");
      gsap.fromTo(
        chars,
        { y: 120, opacity: 0 },
        {
          y: 0,
          opacity: 1,
          duration: 1.1,
          ease: "power3.out",
          stagger: 0.08,
          delay: 0.2,
        },
      );

      gsap.fromTo(
        ".hero-kicker,.hero-tagline,.hero-sub,.hero-actions",
        { y: 24, opacity: 0 },
        {
          y: 0,
          opacity: 1,
          duration: 0.9,
          ease: "power3.out",
          stagger: 0.12,
          delay: 0.7,
        },
      );

      gsap.fromTo(
        ".hero-rule",
        { scaleX: 0 },
        {
          scaleX: 1,
          duration: 1.4,
          ease: "power3.inOut",
          delay: 0.9,
          transformOrigin: "left center",
        },
      );

      // Scroll-pinned hero: slow parallax on the background image while content fades out
      if (heroImgRef.current && heroRef.current) {
        gsap.to(heroImgRef.current, {
          yPercent: 18,
          ease: "none",
          scrollTrigger: {
            trigger: heroRef.current,
            start: "top top",
            end: "bottom top",
            scrub: true,
          },
        });
        gsap.to(".forma-hero__content", {
          opacity: 0,
          y: -40,
          ease: "none",
          scrollTrigger: {
            trigger: heroRef.current,
            start: "top top",
            end: "bottom 30%",
            scrub: true,
          },
        });
      }

      // Section label reveals on scroll — subtle translateY only; content is
      // always visible so Playwright/reduced-motion/no-JS still see it.
      gsap.utils.toArray<HTMLElement>("[data-reveal]").forEach((el) => {
        gsap.from(el, {
          y: 24,
          duration: 0.9,
          ease: "power3.out",
          immediateRender: false,
          scrollTrigger: {
            trigger: el,
            start: "top 88%",
            toggleActions: "play none none none",
          },
        });
      });
    },
    { scope: heroRef },
  );

  // Refresh ScrollTrigger after fonts/images settle so positions are correct
  useEffect(() => {
    const t = setTimeout(() => ScrollTrigger.refresh(), 400);
    return () => clearTimeout(t);
  }, []);

  // Framer-motion driven stats reveal
  const { scrollYProgress } = useScroll({
    target: heroRef,
    offset: ["end end", "end start"],
  });
  const statsY = useTransform(scrollYProgress, [0, 1], [0, -40]);

  return (
    <div ref={heroRef}>
      {/* ── HERO ─────────────────────────────────────────────────────── */}
      <section
        className="forma-hero relative h-screen min-h-[720px] w-full overflow-hidden"
        aria-label="FORMA"
      >
        <VideoBackdrop
          src="/static/videos/hero-loop.mp4"
          poster="/static/images/hero_wide.jpg"
          overlayGradient="linear-gradient(115deg, rgba(10,10,10,0.96) 0%, rgba(10,10,10,0.78) 35%, rgba(10,10,10,0.42) 70%, rgba(10,10,10,0.72) 100%), linear-gradient(180deg, rgba(10,10,10,0.35) 0%, transparent 35%, rgba(10,10,10,0.85) 100%)"
          videoRef={heroImgRef}
          videoClassName="!h-[115%] [filter:contrast(1.12)_brightness(0.82)_saturate(0.9)]"
          className="absolute inset-0"
        />
        {/* Green rim accent on subject side (right) */}
        <div
          aria-hidden="true"
          className="absolute top-1/2 right-[12%] -translate-y-1/2 pointer-events-none rounded-full z-[1]"
          style={{
            width: "460px",
            height: "460px",
            background: "radial-gradient(circle, rgba(174,231,16,0.18) 0%, transparent 70%)",
            filter: "blur(70px)",
          }}
        />
        {/* Grain overlay for film texture */}
        <div
          aria-hidden="true"
          className="absolute inset-0 pointer-events-none opacity-[0.22] mix-blend-overlay z-[1]"
          style={{ backgroundImage: "var(--grain)" }}
        />

        <div className="forma-hero__content relative z-[2] h-full mx-auto max-w-[1440px] px-6 md:px-10 flex flex-col justify-center">
          <span className="hero-kicker text-[0.7rem] uppercase tracking-[0.24em] text-[color:var(--color-ink-on-dark-2)] mb-6">
            BSc Computer Science · University of Greenwich
          </span>

          <h1
            ref={wordmarkRef}
            className="font-[family-name:var(--font-display)] leading-[0.88] text-[color:var(--color-ink-on-dark)] select-none"
            style={{ fontSize: "var(--fs-display)" }}
            aria-label="FORMA"
          >
            <span className="fh-char inline-block">F</span>
            <span className="fh-char inline-block text-[color:var(--color-gold-soft)] font-[family-name:var(--font-serif)] italic">
              O
            </span>
            <span className="fh-char inline-block">R</span>
            <span className="fh-char inline-block">M</span>
            <span className="fh-char inline-block">A</span>
          </h1>

          <span
            className="hero-rule block h-[2px] w-[min(520px,42vw)] mt-10 bg-[color:var(--color-gold-soft)]"
            aria-hidden="true"
          />

          <p className="hero-tagline mt-8 font-[family-name:var(--font-serif)] italic text-3xl md:text-4xl text-[color:var(--color-ink-on-dark)]">
            Train with form.
          </p>
          <p className="hero-sub mt-5 max-w-xl text-base md:text-lg text-[color:var(--color-ink-on-dark-2)] leading-[1.6]">
            Real-time AI feedback on your exercise technique. Eleven exercises, thirty-three body
            landmarks, one trainer that never looks away.
          </p>

          <div className="hero-actions mt-10 flex flex-wrap gap-4">
            <Link
              to={ctaTarget}
              className="inline-flex items-center gap-2 px-7 py-4 bg-[color:var(--color-gold)] text-[#0A0A0A] text-xs uppercase tracking-[0.14em] font-medium rounded-[2px] hover:bg-[color:var(--color-gold-soft)] transition-colors"
            >
              {user ? "Start Training" : "Sign up to train"}
            </Link>
            <Link
              to="/about"
              className="inline-flex items-center gap-2 px-7 py-4 border border-[color:var(--color-ink-on-dark)]/40 text-[color:var(--color-ink-on-dark)] text-xs uppercase tracking-[0.14em] font-medium rounded-[2px] hover:border-[color:var(--color-gold-soft)] hover:text-[color:var(--color-gold-soft)] transition-colors"
            >
              How It Works
            </Link>
          </div>
        </div>

        <div
          className="absolute bottom-10 right-10 z-[2] hidden md:flex items-center gap-4 text-[color:var(--color-ink-on-dark-2)]"
          aria-hidden="true"
        >
          <span className="text-[0.625rem] uppercase tracking-[0.24em]">Scroll</span>
          <span className="block h-px w-16 bg-[color:var(--color-ink-on-dark-2)]" />
        </div>
      </section>

      {/* ── STATS BAND ───────────────────────────────────────────────── */}
      <motion.section
        className="relative bg-[color:var(--color-raised)] border-y border-[color:var(--rule)] overflow-hidden"
        style={{ y: statsY }}
      >
        <GradientGlow position="top-left" intensity="medium" />
        <GradientGlow position="bottom-right" intensity="subtle" />
        <div className="relative mx-auto max-w-[1440px] px-6 md:px-10 py-24 grid gap-12 md:grid-cols-3">
          {STATS.map((stat) => (
            <div key={stat.label} data-reveal className="text-center md:text-left">
              <div className="font-[family-name:var(--font-display)] text-[clamp(4rem,8vw,7rem)] leading-[0.88] text-[color:var(--color-ink)]">
                {stat.value}
                {stat.suffix ? (
                  <small className="text-[0.35em] ml-1 text-[color:var(--color-gold)] align-super font-[family-name:var(--font-sans)]">
                    {stat.suffix}
                  </small>
                ) : null}
              </div>
              <div className="mt-3 text-xs uppercase tracking-[0.14em] text-[color:var(--color-ink-2)]">
                {stat.label}
              </div>
              <div className="mt-2 text-sm text-[color:var(--color-ink-3)]">{stat.caption}</div>
            </div>
          ))}
        </div>
      </motion.section>

      {/* ── HOW IT WORKS ─────────────────────────────────────────────── */}
      <section className="relative overflow-hidden">
        <GradientGlow position="top-right" intensity="medium" />
        <GradientGlow position="center-left" intensity="subtle" />
        <div className="relative mx-auto max-w-[1440px] px-6 md:px-10 py-32">
        <header className="max-w-3xl mb-20">
          <span
            data-reveal
            className="block text-xs uppercase tracking-[0.24em] text-[color:var(--color-gold)] mb-4"
          >
            How It Works
          </span>
          <h2
            data-reveal
            className="font-[family-name:var(--font-display)] text-[clamp(2.5rem,6vw,5rem)] leading-[0.92] text-[color:var(--color-ink)]"
          >
            Smart analysis,
            <br />
            <em className="not-italic font-[family-name:var(--font-serif)] italic text-[color:var(--color-gold)]">
              every rep.
            </em>
          </h2>
        </header>
        <ol className="divide-y divide-[color:var(--rule)] border-t border-[color:var(--rule)]">
          {HOW.map((row) => (
            <li
              key={row.num}
              data-reveal
              className="grid gap-6 md:grid-cols-[auto_1fr_2fr] items-baseline py-10"
            >
              <span className="font-[family-name:var(--font-mono)] text-sm text-[color:var(--color-gold)]">
                {row.num}
              </span>
              <div className="font-[family-name:var(--font-display)] text-3xl text-[color:var(--color-ink)]">
                {row.title}
              </div>
              <p className="text-[color:var(--color-ink-2)] leading-[1.6] max-w-2xl">{row.body}</p>
            </li>
          ))}
        </ol>
        </div>
      </section>

      {/* ── WHY FORMA ────────────────────────────────────────────────── */}
      <section className="relative bg-[color:var(--color-raised)] border-y border-[color:var(--rule)] overflow-hidden">
        <GradientGlow position="top-left" intensity="medium" />
        <GradientGlow position="bottom-right" intensity="subtle" />
        <div className="mx-auto max-w-[1440px] px-6 md:px-10 py-32">
          <header className="max-w-3xl mb-20">
            <span
              data-reveal
              className="block text-xs uppercase tracking-[0.24em] text-[color:var(--color-gold)] mb-4"
            >
              Why FORMA
            </span>
            <h2
              data-reveal
              className="font-[family-name:var(--font-display)] text-[clamp(2.5rem,6vw,5rem)] leading-[0.92] text-[color:var(--color-ink)]"
            >
              Built for the
              <br />
              <em className="not-italic font-[family-name:var(--font-serif)] italic text-[color:var(--color-gold)]">
                way you train.
              </em>
            </h2>
          </header>
          <div className="space-y-24">
            {FEATURES.map((f, i) => (
              <div
                key={f.idx}
                data-reveal
                className={
                  "grid gap-12 md:grid-cols-2 items-center " +
                  (i % 2 === 1 ? "md:[&>*:first-child]:order-2" : "")
                }
              >
                <div className="relative aspect-[4/3] overflow-hidden rounded-[6px] shadow-[0_24px_64px_rgba(0,0,0,0.4)]">
                  <img src={f.img} alt="" loading="lazy" className="h-full w-full object-cover" />
                </div>
                <div>
                  <span className="block text-xs uppercase tracking-[0.14em] text-[color:var(--color-gold)] mb-4">
                    {f.idx}
                  </span>
                  <h3 className="font-[family-name:var(--font-display)] text-[clamp(2rem,4vw,3rem)] leading-[0.92] text-[color:var(--color-ink)] mb-6">
                    {f.title}
                  </h3>
                  <p className="text-[color:var(--color-ink-2)] leading-[1.6] max-w-xl">{f.body}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── EXERCISES PREVIEW ────────────────────────────────────────── */}
      <section className="relative overflow-hidden">
        <GradientGlow position="top-center" intensity="subtle" />
        <GradientGlow position="center-right" intensity="medium" />
        <div className="relative mx-auto max-w-[1440px] px-6 md:px-10 py-32">
        <header className="max-w-3xl mb-20">
          <span
            data-reveal
            className="block text-xs uppercase tracking-[0.24em] text-[color:var(--color-gold)] mb-4"
          >
            The Index
          </span>
          <h2
            data-reveal
            className="font-[family-name:var(--font-display)] text-[clamp(2.5rem,6vw,5rem)] leading-[0.92] text-[color:var(--color-ink)]"
          >
            Eleven exercises,
            <br />
            <em className="not-italic font-[family-name:var(--font-serif)] italic text-[color:var(--color-gold)]">
              one platform.
            </em>
          </h2>
          <p
            data-reveal
            className="mt-6 max-w-2xl text-[color:var(--color-ink-2)] leading-[1.6]"
          >
            From compound lifts to bodyweight movements. Each exercise has its own dedicated
            detector, its own biomechanical thresholds, and its own form cues.{" "}
            <em className="text-[color:var(--color-gold)]">Specialised, not general.</em>
          </p>
        </header>

        <div className="grid gap-6 grid-cols-2 md:grid-cols-3 lg:grid-cols-5">
          {EXERCISES.map(([num, name, slug]) => (
            <Link
              key={slug}
              to={exerciseHref(slug)}
              data-reveal
              className="group block"
            >
              <div className="relative aspect-[3/4] overflow-hidden rounded-[6px] bg-[color:var(--color-sunken)]">
                <img
                  src={`/static/images/${slug}.jpg`}
                  alt={name}
                  loading="lazy"
                  className="h-full w-full object-cover transition-transform duration-700 ease-[var(--ease-out-editorial)] group-hover:scale-105"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-[color:var(--color-contrast)]/70 via-transparent to-transparent" />
              </div>
              <div className="mt-4 flex items-baseline justify-between">
                <span className="font-[family-name:var(--font-mono)] text-xs text-[color:var(--color-gold)]">
                  {num}
                </span>
                <span className="font-[family-name:var(--font-display)] text-lg text-[color:var(--color-ink)]">
                  {name}
                </span>
              </div>
            </Link>
          ))}
        </div>

        <div className="mt-16 flex justify-center">
          <Link
            to={ctaTarget}
            className="inline-flex items-center gap-2 px-7 py-4 bg-[color:var(--color-gold)] text-[color:var(--color-page)] text-xs uppercase tracking-[0.14em] font-medium rounded-[2px] hover:bg-[color:var(--color-gold-hover)] transition-colors"
          >
            {user ? "Start a session" : "Create an account"}
          </Link>
        </div>
        </div>
      </section>

      {/* ── BUILT LIKE A TRAINER ─────────────────────────────────────── */}
      <TextOverMedia
        image="/static/images/atmosphere.jpg"
        anchor="left"
        intensity={0.88}
        imgOpacity={0.55}
        className="bg-[color:var(--color-contrast)] text-[color:var(--color-ink-on-dark)] border-y border-[color:var(--color-ink-on-dark)]/5"
      >
        <div className="mx-auto max-w-[1440px] px-6 md:px-10 py-32">
          <header className="max-w-3xl mb-20">
            <span
              data-reveal
              className="block text-xs uppercase tracking-[0.24em] text-[color:var(--color-gold-soft)] mb-4"
            >
              Built like a trainer
            </span>
            <h2
              data-reveal
              className="font-[family-name:var(--font-display)] text-[clamp(2.5rem,6vw,5rem)] leading-[0.92]"
            >
              Patient.
              <br />
              <em className="not-italic font-[family-name:var(--font-serif)] italic text-[color:var(--color-gold-soft)]">
                Quiet. Honest.
              </em>
            </h2>
            <p
              data-reveal
              className="mt-6 max-w-2xl text-[color:var(--color-ink-on-dark)] leading-[1.6]"
            >
              A good coach doesn't yell. Doesn't recite numbers. Doesn't pile on corrections while
              you're mid-rep. FORMA was designed with the same restraint.
            </p>
          </header>

          <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-4">
            {TRAINER_PILLARS.map((p, i) => (
              <div
                key={p.title}
                data-reveal
                className="relative border-t border-[color:var(--color-gold-soft)]/40 pt-8 bg-[color:var(--color-contrast)]/55 backdrop-blur-[2px] rounded-sm p-6 -mx-2"
              >
                <div className="font-[family-name:var(--font-mono)] text-xs text-[color:var(--color-gold-soft)] mb-4">
                  {String(i + 1).padStart(2, "0")}
                </div>
                <h3 className="font-[family-name:var(--font-display)] text-2xl md:text-3xl leading-[0.95] mb-4 text-[color:var(--color-ink-on-dark)]">
                  {p.title}
                </h3>
                <p className="text-[color:var(--color-ink-on-dark)] leading-[1.55]">{p.body}</p>
              </div>
            ))}
          </div>
        </div>
      </TextOverMedia>

      {/* ── EVERYTHING FORMA DOES ────────────────────────────────────── */}
      <section className="relative overflow-hidden">
        <GradientGlow position="top-left" intensity="medium" />
        <GradientGlow position="bottom-right" intensity="subtle" />
        <div className="relative mx-auto max-w-[1440px] px-6 md:px-10 py-32">
        <header className="max-w-3xl mb-20">
          <span
            data-reveal
            className="block text-xs uppercase tracking-[0.24em] text-[color:var(--color-gold)] mb-4"
          >
            Everything FORMA does for you
          </span>
          <h2
            data-reveal
            className="font-[family-name:var(--font-display)] text-[clamp(2.5rem,6vw,5rem)] leading-[0.92] text-[color:var(--color-ink)]"
          >
            One product.
            <br />
            <em className="not-italic font-[family-name:var(--font-serif)] italic text-[color:var(--color-gold)]">
              Five ways to train.
            </em>
          </h2>
          <p
            data-reveal
            className="mt-6 max-w-2xl text-[color:var(--color-ink-2)] leading-[1.6]"
          >
            Real-time form feedback is only the start. FORMA remembers your sessions, talks back
            when you ask, writes you training plans, and opens up its own internals when you want
            to see how it thinks.
          </p>
        </header>

        <div className="grid gap-6 md:grid-cols-2">
          {PRODUCT_FEATURES.map((f) => (
            <Link
              key={f.to}
              to={f.to}
              data-reveal
              className="group relative overflow-hidden rounded-[6px] border border-[color:var(--rule)] bg-[color:var(--color-raised)] transition-colors hover:border-[color:var(--color-gold)]/50"
            >
              <div className="relative overflow-hidden aspect-[16/10]">
                <img
                  src={f.image}
                  alt=""
                  loading="lazy"
                  className="h-full w-full object-cover transition-transform duration-700 ease-[var(--ease-out-editorial)] group-hover:scale-[1.04]"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-[color:var(--color-contrast)]/70 via-[color:var(--color-contrast)]/20 to-transparent" />
                <span className="absolute top-5 left-5 text-[0.6rem] uppercase tracking-[0.18em] text-[color:var(--color-gold-soft)] border border-[color:var(--color-gold-soft)]/40 px-2.5 py-1 rounded-full">
                  {f.tag}
                </span>
              </div>
              <div className="p-8">
                <h3 className="font-[family-name:var(--font-display)] text-3xl text-[color:var(--color-ink)] leading-[0.95] mb-3">
                  {f.title}
                </h3>
                <p className="text-[color:var(--color-ink-2)] leading-[1.55]">{f.body}</p>
                <span className="mt-6 inline-flex items-center gap-2 text-xs uppercase tracking-[0.14em] text-[color:var(--color-gold)] group-hover:text-[color:var(--color-gold-hover)] transition-colors">
                  Explore <span aria-hidden>→</span>
                </span>
              </div>
            </Link>
          ))}
        </div>
        </div>
      </section>

      {/* ── THE SCIENCE ──────────────────────────────────────────────── */}
      <section className="relative bg-[color:var(--color-raised)] border-y border-[color:var(--rule)] overflow-hidden">
        <GradientGlow position="center-right" intensity="medium" />
        <GradientGlow position="bottom-left" intensity="subtle" />
        <div className="relative mx-auto max-w-[1440px] px-6 md:px-10 py-32">
          <header className="max-w-3xl mb-20">
            <span
              data-reveal
              className="block text-xs uppercase tracking-[0.24em] text-[color:var(--color-gold)] mb-4"
            >
              The science
            </span>
            <h2
              data-reveal
              className="font-[family-name:var(--font-display)] text-[clamp(2.5rem,6vw,5rem)] leading-[0.92] text-[color:var(--color-ink)]"
            >
              Not guesswork.
              <br />
              <em className="not-italic font-[family-name:var(--font-serif)] italic text-[color:var(--color-gold)]">
                Research-backed.
              </em>
            </h2>
          </header>

          <div className="grid gap-10 md:grid-cols-3">
            {SCIENCE.map((s) => (
              <div
                key={s.label}
                data-reveal
                className="border-t-2 border-[color:var(--color-gold)]/40 pt-8"
              >
                <div className="font-[family-name:var(--font-mono)] text-[0.65rem] uppercase tracking-[0.18em] text-[color:var(--color-gold)] mb-4">
                  {s.label}
                </div>
                <div className="font-[family-name:var(--font-display)] text-[clamp(3rem,5vw,5rem)] leading-[0.88] text-[color:var(--color-ink)] mb-4">
                  {s.num}
                </div>
                <h3 className="font-[family-name:var(--font-display)] text-2xl text-[color:var(--color-ink)] leading-[1] mb-3">
                  {s.title}
                </h3>
                <p className="text-[color:var(--color-ink-2)] leading-[1.6]">{s.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CLOSING CTA ──────────────────────────────────────────────── */}
      <section className="relative overflow-hidden bg-[color:var(--color-contrast)] text-[color:var(--color-ink-on-dark)]">
        <img
          src="/static/images/cta_bg.jpg"
          alt=""
          aria-hidden="true"
          className="absolute inset-0 h-full w-full object-cover opacity-30"
          loading="lazy"
        />
        <div className="relative mx-auto max-w-[1440px] px-6 md:px-10 py-32 text-center">
          <h2
            data-reveal
            className="font-[family-name:var(--font-display)] text-[clamp(3rem,8vw,7rem)] leading-[0.92]"
          >
            Train
            <em className="not-italic font-[family-name:var(--font-serif)] italic text-[color:var(--color-gold-soft)]">
              {" "}with form.
            </em>
          </h2>
          <div className="mt-12">
            <Link
              to={ctaTarget}
              className="inline-flex items-center gap-2 px-8 py-4 bg-[color:var(--color-gold)] text-[#0A0A0A] text-xs uppercase tracking-[0.14em] font-medium rounded-[2px] hover:bg-[color:var(--color-gold-soft)] transition-colors"
            >
              {user ? "Start a session" : "Start Your First Session"}
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
