// CelebrationToast — mounted once at the app root. Subscribes to
// milestones_reached + badges_earned Socket.IO events and shows a
// warmly-animated toast for each one in sequence.
//
// No extra dependencies: CSS keyframes for the confetti, framer-motion
// for the toast itself (already in the project).

import { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useMilestonesReached } from "@/hooks/useMilestonesReached";
import { useBadgesEarned } from "@/hooks/useBadgesEarned";
import { usePlanSaved } from "@/hooks/usePlanSaved";

type ToastEvent = {
  id: string;
  kind: "milestone" | "badge" | "plan";
  title: string;
  body: string;
};

const TOAST_MS = 3500;

export function CelebrationToast() {
  const [queue, setQueue] = useState<ToastEvent[]>([]);
  const [active, setActive] = useState<ToastEvent | null>(null);

  const push = useCallback((ev: ToastEvent) => {
    setQueue((q) => [...q, ev]);
  }, []);

  useMilestonesReached(
    useCallback(
      (payload) => {
        payload.milestones.forEach((m) => {
          push({
            id: `milestone-${m.id}-${Date.now()}`,
            kind: "milestone",
            title: `${m.label} · ${m.goal_title}`,
            body: `You just crossed ${m.label} of this goal.`,
          });
        });
      },
      [push],
    ),
  );

  useBadgesEarned(
    useCallback(
      (payload) => {
        payload.badges.forEach((b) => {
          push({
            id: `badge-${b.badge_key}-${Date.now()}`,
            kind: "badge",
            title: b.title,
            body: b.description,
          });
        });
      },
      [push],
    ),
  );

  usePlanSaved(
    useCallback(
      (payload) => {
        const title = payload.title ?? "Workout plan";
        const started = payload.start_date
          ? `Starts ${payload.start_date}`
          : "Saved to your plans";
        push({
          id: `plan-${payload.plan_id}-${Date.now()}`,
          kind: "plan",
          title,
          body: `${started}. Open the Plans page any time to adjust it.`,
        });
      },
      [push],
    ),
  );

  // Drain the queue one at a time (no timer logic here).
  useEffect(() => {
    if (active || queue.length === 0) return;
    const [next, ...rest] = queue;
    setActive(next);
    setQueue(rest);
  }, [active, queue]);

  // Owns the dismiss timer — keyed only on the active toast's id so it
  // fires exactly once per toast and survives unrelated re-renders that
  // previously clobbered the timeout.
  useEffect(() => {
    if (!active) return;
    const id = window.setTimeout(() => setActive(null), TOAST_MS);
    return () => window.clearTimeout(id);
  }, [active?.id]);

  return (
    <>
      <style>{`
        @keyframes forma-confetti-fall {
          0% { transform: translate3d(var(--fx, 0), -20px, 0) rotate(0deg); opacity: 0; }
          15% { opacity: 1; }
          100% { transform: translate3d(var(--fx, 0), 80px, 0) rotate(720deg); opacity: 0; }
        }
      `}</style>
      <div className="fixed inset-x-0 top-[88px] z-[90] flex justify-center pointer-events-none">
        <AnimatePresence>
          {active && (
            <motion.div
              key={active.id}
              initial={{ y: -20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              exit={{ y: -20, opacity: 0 }}
              transition={{ type: "spring", stiffness: 260, damping: 22 }}
              className="relative bg-[color:var(--color-page)] border border-[color:var(--color-gold)] shadow-[0_24px_60px_-20px_rgba(174,231,16,0.3)] rounded-[4px] px-6 py-4 pr-9 max-w-md pointer-events-auto"
            >
              <button
                type="button"
                onClick={() => setActive(null)}
                aria-label="Dismiss"
                className="absolute top-2 right-2 w-6 h-6 flex items-center justify-center text-[color:var(--color-ink-3)] hover:text-[color:var(--color-ink)] text-base leading-none pointer-events-auto z-[1]"
              >
                ×
              </button>
              {/* Confetti */}
              <div className="absolute inset-x-0 -top-2 h-0">
                {Array.from({ length: 8 }).map((_, i) => {
                  const fx = `${(i - 3.5) * 12}px`;
                  const delay = i * 0.08;
                  const hue = i % 2 === 0 ? "var(--color-gold)" : "var(--color-gold-soft)";
                  return (
                    <span
                      key={i}
                      style={{
                        position: "absolute",
                        left: "50%",
                        top: 0,
                        width: 6,
                        height: 10,
                        background: hue,
                        borderRadius: 1,
                        // @ts-expect-error — CSS custom property
                        "--fx": fx,
                        animation: `forma-confetti-fall 1.6s ease-out ${delay}s forwards`,
                      }}
                    />
                  );
                })}
              </div>
              <div
                className="text-[10px] uppercase tracking-[0.22em] text-[color:var(--color-gold)]"
                style={{ fontFamily: "var(--font-display)" }}
              >
                {active.kind === "badge"
                  ? "Badge earned"
                  : active.kind === "plan"
                    ? "Plan saved"
                    : "Milestone reached"}
              </div>
              <div
                className="text-[color:var(--color-ink)] mt-1"
                style={{
                  fontFamily: "var(--font-display)",
                  fontSize: "1.4rem",
                  letterSpacing: "0.04em",
                }}
              >
                {active.title.toUpperCase()}
              </div>
              <p
                className="italic text-[color:var(--color-ink-3)] mt-1 text-[13px]"
                style={{ fontFamily: "var(--font-serif)" }}
              >
                {active.body}
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </>
  );
}
