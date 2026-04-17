import { useEffect, useRef, useState, type CSSProperties, type ReactNode, type Ref } from "react";
import { MEDIA_GRADIENTS, type MediaGradientAnchor } from "./mediaGradients";

type Props = {
  src: string;
  poster: string;
  anchor?: MediaGradientAnchor;
  intensity?: number;
  overlayGradient?: string;
  videoRef?: Ref<HTMLVideoElement>;
  className?: string;
  videoClassName?: string;
  objectPosition?: string;
  children?: ReactNode;
  style?: CSSProperties;
};

export function VideoBackdrop({
  src,
  poster,
  anchor = "bottom",
  intensity = 0.78,
  overlayGradient,
  videoRef,
  className = "",
  videoClassName = "",
  objectPosition,
  children,
  style,
}: Props) {
  const localVideoRef = useRef<HTMLVideoElement | null>(null);
  const [mode, setMode] = useState<"video" | "poster">("video");

  useEffect(() => {
    const video = localVideoRef.current;
    if (!video) return;

    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const applyMotionPref = () => {
      if (mq.matches) {
        video.pause();
        setMode("poster");
      } else {
        setMode("video");
        video.play().catch(() => setMode("poster"));
      }
    };
    applyMotionPref();
    mq.addEventListener("change", applyMotionPref);
    return () => mq.removeEventListener("change", applyMotionPref);
  }, [src]);

  const assignVideoRef = (el: HTMLVideoElement | null) => {
    localVideoRef.current = el;
    if (typeof videoRef === "function") videoRef(el);
    else if (videoRef && "current" in videoRef)
      (videoRef as { current: HTMLVideoElement | null }).current = el;
  };

  const gradient = overlayGradient ?? MEDIA_GRADIENTS[anchor](intensity);

  return (
    <div className={`overflow-hidden ${className || "relative"}`} style={style}>
      {mode === "video" ? (
        <video
          ref={assignVideoRef}
          src={src}
          poster={poster}
          autoPlay
          muted
          loop
          playsInline
          preload="metadata"
          aria-hidden="true"
          onError={() => setMode("poster")}
          className={`absolute inset-0 h-full w-full object-cover will-change-transform ${videoClassName}`}
          style={objectPosition ? { objectPosition } : undefined}
        />
      ) : (
        <img
          src={poster}
          alt=""
          aria-hidden="true"
          className={`absolute inset-0 h-full w-full object-cover ${videoClassName}`}
          style={objectPosition ? { objectPosition } : undefined}
        />
      )}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{ background: gradient }}
        aria-hidden="true"
      />
      {children ? (
        <div
          className="relative z-[2]"
          style={
            {
              ["--safety-shadow" as string]:
                "0 2px 16px rgba(0,0,0,0.55), 0 1px 4px rgba(0,0,0,0.45)",
            } as CSSProperties
          }
        >
          <div className="[&_h1]:[text-shadow:var(--safety-shadow)] [&_h2]:[text-shadow:var(--safety-shadow)] [&_h3]:[text-shadow:var(--safety-shadow)] [&_p]:[text-shadow:var(--safety-shadow)] [&_span]:[text-shadow:var(--safety-shadow)] [&_li]:[text-shadow:var(--safety-shadow)] [&_em]:[text-shadow:var(--safety-shadow)]">
            {children}
          </div>
        </div>
      ) : null}
    </div>
  );
}
