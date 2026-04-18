import { useEffect } from "react";

import type { Exercise, PhoneOrientation } from "../../types/exercise";

type Props = {
  exercise: Exercise;
  current: PhoneOrientation;
  onOverride: () => void;
};

export function OrientationGate({ exercise, current, onOverride }: Props) {
  const preferred = exercise.preferredOrientation;
  const desiredLandscape = preferred === "landscape";

  useEffect(() => {
    const lockable = (screen as Screen & {
      orientation?: { lock?: (o: string) => Promise<void> };
    }).orientation;
    if (lockable?.lock) {
      lockable.lock(preferred).catch(() => {
        /* iOS Safari rejects silently — that's fine, we still show the prompt. */
      });
    }
  }, [preferred]);

  return (
    <div className="absolute inset-0 z-[450] flex items-center justify-center bg-black/92 backdrop-blur-sm p-6">
      <div className="w-full max-w-md text-center">
        <div className="text-[10px] uppercase tracking-[0.24em] text-[color:var(--color-gold-soft)] mb-6">
          Rotate your phone
        </div>

        <div className="flex justify-center mb-8" aria-hidden="true">
          <div
            className="relative h-24 w-14 rounded-[12px] border-2 border-white/70 bg-white/5"
            style={{
              transform: desiredLandscape ? "rotate(90deg)" : "rotate(0deg)",
              transition: "transform 0.9s cubic-bezier(0.65, 0, 0.35, 1)",
              animation: "orientation-gate-rotate 2.4s ease-in-out infinite",
            }}
          >
            <div className="absolute left-1/2 top-1.5 -translate-x-1/2 h-1 w-6 rounded-full bg-white/40" />
            <div className="absolute inset-2 rounded-sm bg-gradient-to-br from-[color:var(--color-gold-soft)]/40 to-transparent" />
          </div>
        </div>

        <h2 className="font-[family-name:var(--font-display)] text-3xl md:text-4xl text-white tracking-[0.04em] leading-tight">
          Turn your phone {desiredLandscape ? "sideways" : "upright"}
        </h2>

        <p className="mt-4 font-[family-name:var(--font-serif)] italic text-lg text-white/75 leading-relaxed">
          {desiredLandscape
            ? `${exercise.name} reads best in landscape — your body fits end-to-end in the frame.`
            : `${exercise.name} reads best in portrait — you'll have the full standing height you need.`}
        </p>

        <button
          type="button"
          onClick={onOverride}
          className="mt-10 inline-flex items-center gap-2 px-6 py-2.5 border border-white/25 text-[10px] uppercase tracking-[0.18em] text-white/60 hover:text-white hover:border-white/50 transition-colors"
        >
          Use {current} anyway
        </button>

        <p className="mt-3 text-[10px] uppercase tracking-[0.18em] text-white/30">
          Your body may be cropped
        </p>
      </div>

      <style>{`
        @keyframes orientation-gate-rotate {
          0%, 40%   { transform: rotate(${desiredLandscape ? "0deg" : "90deg"}); }
          60%, 100% { transform: rotate(${desiredLandscape ? "90deg" : "0deg"}); }
        }
      `}</style>
    </div>
  );
}
