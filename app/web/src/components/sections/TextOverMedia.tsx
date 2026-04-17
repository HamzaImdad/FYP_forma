import type { CSSProperties, ReactNode } from "react";
import { MEDIA_GRADIENTS, type MediaGradientAnchor } from "./mediaGradients";

type Anchor = MediaGradientAnchor;

const GRADIENTS = MEDIA_GRADIENTS;

export function TextOverMedia({
  image,
  alt = "",
  anchor = "left",
  intensity = 0.78,
  imgClassName = "",
  className = "",
  children,
  imgOpacity = 1,
  style,
}: {
  image: string;
  alt?: string;
  anchor?: Anchor;
  intensity?: number;
  imgClassName?: string;
  className?: string;
  children: ReactNode;
  imgOpacity?: number;
  style?: CSSProperties;
}) {
  return (
    <div
      className={`relative overflow-hidden ${className}`}
      style={style}
      data-text-over-media="true"
    >
      <img
        src={image}
        alt={alt}
        aria-hidden={alt ? undefined : true}
        className={`absolute inset-0 h-full w-full object-cover ${imgClassName}`}
        style={{ opacity: imgOpacity }}
        loading="lazy"
      />
      <div
        className="absolute inset-0 pointer-events-none"
        style={{ background: GRADIENTS[anchor](intensity) }}
        aria-hidden="true"
      />
      <div
        className="relative z-[2]"
        style={
          {
            ["--safety-shadow" as string]: "0 2px 16px rgba(0,0,0,0.55), 0 1px 4px rgba(0,0,0,0.45)",
          } as CSSProperties
        }
      >
        <div className="[&_h1]:[text-shadow:var(--safety-shadow)] [&_h2]:[text-shadow:var(--safety-shadow)] [&_h3]:[text-shadow:var(--safety-shadow)] [&_p]:[text-shadow:var(--safety-shadow)] [&_span]:[text-shadow:var(--safety-shadow)] [&_li]:[text-shadow:var(--safety-shadow)] [&_em]:[text-shadow:var(--safety-shadow)]">
          {children}
        </div>
      </div>
    </div>
  );
}
