/* ──────────────────────────────────────────────────────────────
   GradientGlow — reusable lime radial accent

   Drops a soft, blurred green radial behind a section or card.
   Decorative only (aria-hidden). The exact effect from the
   CinematicHero rim-light, extracted for use anywhere.

   Usage:
     <section className="relative overflow-hidden">
       <GradientGlow position="top-right" intensity="medium" />
       {content...}
     </section>
   ────────────────────────────────────────────────────────────── */

type Position =
  | "top-left"
  | "top-right"
  | "top-center"
  | "center-left"
  | "center-right"
  | "center"
  | "bottom-left"
  | "bottom-right"
  | "bottom-center";

type Intensity = "subtle" | "medium" | "strong";

const POS: Record<Position, string> = {
  "top-left":      "top-[-12%] left-[-8%]",
  "top-right":     "top-[-12%] right-[-8%]",
  "top-center":    "top-[-15%] left-1/2 -translate-x-1/2",
  "center-left":   "top-1/2 left-[-10%] -translate-y-1/2",
  "center-right":  "top-1/2 right-[-10%] -translate-y-1/2",
  "center":        "top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2",
  "bottom-left":   "bottom-[-15%] left-[-8%]",
  "bottom-right":  "bottom-[-15%] right-[-8%]",
  "bottom-center": "bottom-[-20%] left-1/2 -translate-x-1/2",
};

const INTENSITY: Record<Intensity, { size: number; opacity: number; core: number }> = {
  subtle: { size: 420, opacity: 0.10, core: 0.15 },
  medium: { size: 560, opacity: 0.16, core: 0.22 },
  strong: { size: 720, opacity: 0.24, core: 0.32 },
};

export function GradientGlow({
  position = "top-right",
  intensity = "medium",
  hue = "174,231,16",
  blur = 70,
  className = "",
}: {
  position?: Position;
  intensity?: Intensity;
  /** RGB triple as string, e.g. "174,231,16" */
  hue?: string;
  blur?: number;
  className?: string;
}) {
  const { size, opacity, core } = INTENSITY[intensity];
  return (
    <div
      aria-hidden="true"
      className={`absolute ${POS[position]} pointer-events-none rounded-full z-[0] ${className}`}
      style={{
        width: `${size}px`,
        height: `${size}px`,
        background: `radial-gradient(circle, rgba(${hue},${core}) 0%, rgba(${hue},${opacity * 0.4}) 35%, transparent 70%)`,
        filter: `blur(${blur}px)`,
        opacity: 1,
      }}
    />
  );
}
