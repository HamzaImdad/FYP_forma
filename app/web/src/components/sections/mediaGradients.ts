export type MediaGradientAnchor = "left" | "right" | "center" | "bottom" | "top";

export const MEDIA_GRADIENTS: Record<MediaGradientAnchor, (intensity: number) => string> = {
  left: (i) =>
    `linear-gradient(90deg, rgba(13,13,13,${i}) 0%, rgba(13,13,13,${i * 0.85}) 35%, rgba(13,13,13,${i * 0.5}) 65%, rgba(13,13,13,${Math.max(i * 0.25, 0.15)}) 100%)`,
  right: (i) =>
    `linear-gradient(270deg, rgba(13,13,13,${i}) 0%, rgba(13,13,13,${i * 0.85}) 35%, rgba(13,13,13,${i * 0.5}) 65%, rgba(13,13,13,${Math.max(i * 0.25, 0.15)}) 100%)`,
  bottom: (i) =>
    `linear-gradient(0deg, rgba(13,13,13,${i}) 0%, rgba(13,13,13,${i * 0.75}) 40%, rgba(13,13,13,${i * 0.35}) 75%, rgba(13,13,13,${Math.max(i * 0.15, 0.1)}) 100%)`,
  top: (i) =>
    `linear-gradient(180deg, rgba(13,13,13,${i}) 0%, rgba(13,13,13,${i * 0.75}) 40%, rgba(13,13,13,${i * 0.35}) 75%, rgba(13,13,13,${Math.max(i * 0.15, 0.1)}) 100%)`,
  center: (i) =>
    `radial-gradient(ellipse 80% 90% at 50% 50%, rgba(13,13,13,${i}) 0%, rgba(13,13,13,${i * 0.65}) 55%, rgba(13,13,13,${Math.max(i * 0.25, 0.2)}) 100%)`,
};
