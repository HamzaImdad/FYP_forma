import { useEffect, useRef, type ReactNode } from "react";
import { useGSAP } from "@gsap/react";
import { gsap, registerGsap, ScrollTrigger } from "@/lib/gsap";

registerGsap();

export function PageHero({
  eyebrow,
  title,
  italic,
  subtitle,
  image,
  overlay = 0.55,
  actions,
}: {
  eyebrow?: string;
  title: string;
  italic?: string;
  subtitle?: ReactNode;
  image: string;
  overlay?: number;
  actions?: ReactNode;
}) {
  const heroRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  useGSAP(
    () => {
      gsap.fromTo(
        ".ph-eyebrow, .ph-title, .ph-rule, .ph-sub, .ph-actions",
        { y: 28, opacity: 0 },
        {
          y: 0,
          opacity: 1,
          duration: 0.95,
          ease: "power3.out",
          stagger: 0.1,
          delay: 0.15,
        },
      );

      if (imgRef.current && heroRef.current) {
        gsap.to(imgRef.current, {
          yPercent: 16,
          ease: "none",
          scrollTrigger: {
            trigger: heroRef.current,
            start: "top top",
            end: "bottom top",
            scrub: true,
          },
        });
        gsap.to(".ph-content", {
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
    },
    { scope: heroRef },
  );

  useEffect(() => {
    const t = setTimeout(() => ScrollTrigger.refresh(), 400);
    return () => clearTimeout(t);
  }, []);

  return (
    <section
      ref={heroRef}
      className="relative h-screen min-h-[640px] w-full overflow-hidden"
    >
      <div className="absolute inset-0">
        <img
          ref={imgRef}
          src={image}
          alt=""
          className="h-[115%] w-full object-cover will-change-transform"
          loading="eager"
        />
        <div
          className="absolute inset-0"
          style={{
            background: `linear-gradient(100deg, rgba(13,13,13,${Math.min(overlay + 0.25, 0.95)}) 0%, rgba(13,13,13,${overlay + 0.05}) 45%, rgba(13,13,13,${Math.max(overlay - 0.1, 0.2)}) 100%)`,
          }}
        />
        <div
          className="absolute inset-0"
          style={{
            background:
              "linear-gradient(0deg, rgba(13,13,13,0.55) 0%, rgba(13,13,13,0.15) 35%, transparent 65%)",
          }}
        />
      </div>

      <div
        className="ph-content relative z-[2] h-full mx-auto max-w-[1440px] px-6 md:px-10 flex flex-col justify-center [&_h1]:[text-shadow:0_2px_20px_rgba(0,0,0,0.7),0_1px_4px_rgba(0,0,0,0.5)] [&_p]:[text-shadow:0_1px_12px_rgba(0,0,0,0.6)] [&_span]:[text-shadow:0_1px_8px_rgba(0,0,0,0.55)]"
      >
        {eyebrow ? (
          <span className="ph-eyebrow text-[0.7rem] uppercase tracking-[0.24em] text-[color:var(--color-ink-on-dark-2)] mb-6">
            {eyebrow}
          </span>
        ) : null}

        <h1
          className="ph-title font-[family-name:var(--font-display)] leading-[0.9] text-[color:var(--color-ink-on-dark)]"
          style={{ fontSize: "clamp(3.5rem, 10vw, 8rem)" }}
        >
          {title}
          {italic ? (
            <>
              {" "}
              <em className="not-italic font-[family-name:var(--font-serif)] italic text-[color:var(--color-gold-soft)]">
                {italic}
              </em>
            </>
          ) : null}
        </h1>

        <span
          className="ph-rule block h-[2px] w-[min(520px,42vw)] mt-8 bg-[color:var(--color-gold-soft)]"
          aria-hidden="true"
        />

        {subtitle ? (
          <p className="ph-sub mt-8 max-w-2xl font-[family-name:var(--font-serif)] italic text-2xl md:text-3xl text-[color:var(--color-ink-on-dark)]">
            {subtitle}
          </p>
        ) : null}

        {actions ? <div className="ph-actions mt-10 flex flex-wrap gap-4">{actions}</div> : null}
      </div>

      <div
        className="absolute bottom-10 right-10 z-[2] hidden md:flex items-center gap-4 text-[color:var(--color-ink-on-dark-2)]"
        aria-hidden="true"
      >
        <span className="text-[0.625rem] uppercase tracking-[0.24em]">Scroll</span>
        <span className="block h-px w-16 bg-[color:var(--color-ink-on-dark-2)]" />
      </div>
    </section>
  );
}
