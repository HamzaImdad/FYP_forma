import type { ReactNode, CSSProperties } from "react";

/* ──────────────────────────────────────────────────────────────
   CinematicHero — chiaroscuro hero pattern

   Full-bleed dark image with multi-stop gradient overlays that push
   the edges into black, a subject-side lift to make the subject
   "shine", a lime rim accent, grain, and content sitting in the lit
   quadrant. Inspired by A24 / Nike dark-campaign still photography.

   Anchor controls which side the content + light zone sits on.
   ────────────────────────────────────────────────────────────── */

type Anchor = "left" | "right" | "center";

export function CinematicHero({
  image,
  anchor = "left",
  minHeight = "min-h-[88vh]",
  children,
  className = "",
  imgStyle,
  accentHue = "rgba(174,231,16,0.22)",
}: {
  image: string;
  anchor?: Anchor;
  minHeight?: string;
  children: ReactNode;
  className?: string;
  imgStyle?: CSSProperties;
  accentHue?: string;
}) {
  // Anchor determines where the spotlight sits and how content flows.
  const spotlight: Record<Anchor, string> = {
    left: "radial-gradient(ellipse 65% 90% at 75% 50%, transparent 0%, rgba(10,10,10,0.35) 35%, rgba(10,10,10,0.85) 70%, #0A0A0A 95%)",
    right: "radial-gradient(ellipse 65% 90% at 25% 50%, transparent 0%, rgba(10,10,10,0.35) 35%, rgba(10,10,10,0.85) 70%, #0A0A0A 95%)",
    center: "radial-gradient(ellipse 70% 90% at 50% 45%, transparent 0%, rgba(10,10,10,0.4) 40%, rgba(10,10,10,0.9) 80%, #0A0A0A 100%)",
  };

  const contentJustify: Record<Anchor, string> = {
    left: "justify-start",
    right: "justify-end",
    center: "justify-center",
  };

  const accentPos: Record<Anchor, string> = {
    left: "top-1/2 right-[8%] -translate-y-1/2",
    right: "top-1/2 left-[8%] -translate-y-1/2",
    center: "top-[30%] left-1/2 -translate-x-1/2",
  };

  return (
    <section
      className={`relative w-full overflow-hidden bg-[color:var(--color-page)] ${minHeight} ${className}`}
    >
      {/* Base image — boost contrast, crush blacks */}
      <img
        src={image}
        alt=""
        aria-hidden="true"
        className="absolute inset-0 h-full w-full object-cover"
        style={{
          filter: "contrast(1.1) brightness(0.85) saturate(0.85)",
          ...imgStyle,
        }}
      />

      {/* Vignette: bottom-to-top crush */}
      <div
        className="absolute inset-0 pointer-events-none"
        aria-hidden="true"
        style={{
          background:
            "linear-gradient(180deg, rgba(10,10,10,0.45) 0%, rgba(10,10,10,0.15) 40%, rgba(10,10,10,0.85) 95%, #0A0A0A 100%)",
        }}
      />

      {/* Spotlight: the lit "subject zone" — opposite of where the content sits */}
      <div
        className="absolute inset-0 pointer-events-none"
        aria-hidden="true"
        style={{ background: spotlight[anchor] }}
      />

      {/* Rim accent: a faint green glow at the subject edge */}
      <div
        aria-hidden="true"
        className={`absolute ${accentPos[anchor]} pointer-events-none rounded-full`}
        style={{
          width: "420px",
          height: "420px",
          background: `radial-gradient(circle, ${accentHue} 0%, transparent 70%)`,
          filter: "blur(60px)",
          opacity: 0.7,
        }}
      />

      {/* Grain */}
      <div
        aria-hidden="true"
        className="absolute inset-0 pointer-events-none opacity-[0.25] mix-blend-overlay"
        style={{ backgroundImage: "var(--grain)" }}
      />

      {/* Content */}
      <div
        className={`relative z-[2] h-full mx-auto max-w-[1440px] px-6 md:px-10 flex flex-col ${contentJustify[anchor]}`}
      >
        <div className={`${anchor === "right" ? "ml-auto" : ""} ${anchor === "center" ? "mx-auto text-center" : ""} max-w-2xl py-32`}>
          {children}
        </div>
      </div>
    </section>
  );
}
