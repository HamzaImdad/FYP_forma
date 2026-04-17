// MilestonesShell — /milestones page layout:
// 1) Active GoalProgressCards
// 2) BadgeWall
// 3) Chronological timeline of reached milestones

import { useCallback, useEffect, useState } from "react";
import {
  plansApi,
  type Goal,
  type Badge,
  type MilestoneWithGoal,
} from "@/lib/plansApi";
import { GoalProgressCard } from "./GoalProgressCard";
import { BadgeWall } from "./BadgeWall";
import { GoalTemplatePicker } from "@/components/plans/GoalTemplatePicker";
import { useGoalsUpdated } from "@/hooks/useGoalsUpdated";
import { useMilestonesReached } from "@/hooks/useMilestonesReached";
import { useBadgesEarned } from "@/hooks/useBadgesEarned";

function prettyDate(iso: string | null | undefined): string {
  if (!iso) return "";
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  } catch {
    return iso;
  }
}

export function MilestonesShell() {
  const [goals, setGoals] = useState<Goal[]>([]);
  const [badges, setBadges] = useState<Badge[]>([]);
  const [milestones, setMilestones] = useState<MilestoneWithGoal[]>([]);
  const [loading, setLoading] = useState(true);
  const [pickerOpen, setPickerOpen] = useState(false);

  const fetchAll = useCallback(() => {
    setLoading(true);
    Promise.all([
      plansApi.listGoals(),
      plansApi.listBadges(),
      plansApi.listMilestones(),
    ])
      .then(([g, b, m]) => {
        setGoals(g);
        setBadges(b);
        setMilestones(m);
      })
      .catch((e) => console.warn("milestones fetch failed", e))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  useGoalsUpdated(useCallback(() => fetchAll(), [fetchAll]));
  useMilestonesReached(useCallback(() => fetchAll(), [fetchAll]));
  useBadgesEarned(useCallback(() => fetchAll(), [fetchAll]));

  const active = goals.filter((g) => g.status === "active");
  const reached = milestones.filter((m) => m.reached);
  // Redesign Phase 4 — split completed vs archived so the user can see
  // the recent wins AND find old ones that auto-archived after 7 days.
  const completed = goals.filter((g) => g.status === "completed");
  const archived = goals.filter((g) => g.status === "archived");

  return (
    <div className="min-h-screen bg-[color:var(--color-page)] pt-[64px] sm:pt-[72px]">
      <div className="max-w-[1200px] mx-auto px-4 sm:px-6 md:px-10 py-6 md:py-10">
        <header className="mb-8 flex items-end justify-between gap-6 flex-wrap">
          <div>
            <h1
              className="text-[color:var(--color-ink)]"
              style={{
                fontFamily: "var(--font-display)",
                fontSize: "clamp(2.6rem, 5vw, 4rem)",
                letterSpacing: "0.04em",
                lineHeight: 1,
              }}
            >
              MILESTONES
            </h1>
            <p
              className="italic text-[color:var(--color-ink-3)] mt-2"
              style={{ fontFamily: "var(--font-serif)", fontSize: "1.2rem" }}
            >
              Goals in progress, badges earned, and every milestone behind you.
            </p>
          </div>
          <button
            type="button"
            onClick={() => setPickerOpen(true)}
            className="text-[11px] uppercase tracking-[0.22em] px-5 py-3 bg-[color:var(--color-gold)] text-[color:var(--color-page)] hover:bg-[color:var(--color-gold-hover)] transition-colors rounded-[3px]"
          >
            + New goal
          </button>
        </header>

        {loading ? (
          <div className="italic text-[color:var(--color-ink-3)]">Loading…</div>
        ) : (
          <div className="space-y-10">
            {/* Active goals */}
            <section>
              <h2
                className="text-[color:var(--color-ink-3)] uppercase tracking-[0.22em] mb-4"
                style={{ fontFamily: "var(--font-display)", fontSize: "0.95rem" }}
              >
                Active goals
              </h2>
              {active.length === 0 ? (
                <div className="border border-dashed border-[color:var(--rule)] rounded-[4px] p-6 text-center">
                  <p
                    className="italic text-[color:var(--color-ink-3)]"
                    style={{ fontFamily: "var(--font-serif)" }}
                  >
                    No active goals yet. Pick a template to get started.
                  </p>
                </div>
              ) : (
                <div className="grid gap-4 md:grid-cols-2">
                  {active.map((g) => (
                    <GoalProgressCard key={g.id} goal={g} onChanged={fetchAll} />
                  ))}
                </div>
              )}
            </section>

            {/* Recently completed (still visible for ~7 days before
                auto-archive; shown faded so they don't compete with
                active ones). */}
            {completed.length > 0 && (
              <section>
                <h2
                  className="text-[color:var(--color-ink-3)] uppercase tracking-[0.22em] mb-4"
                  style={{ fontFamily: "var(--font-display)", fontSize: "0.95rem" }}
                >
                  Recently completed
                </h2>
                <div className="grid gap-4 md:grid-cols-2 opacity-75">
                  {completed.map((g) => (
                    <GoalProgressCard key={g.id} goal={g} onChanged={fetchAll} />
                  ))}
                </div>
              </section>
            )}

            {/* Badges */}
            <section>
              <BadgeWall badges={badges} />
            </section>

            {/* Archived goals (post-7-day sweep). Rendered as a
                compressed list since these are frozen history. */}
            {archived.length > 0 && (
              <section>
                <h2
                  className="text-[color:var(--color-ink-3)] uppercase tracking-[0.22em] mb-4"
                  style={{ fontFamily: "var(--font-display)", fontSize: "0.95rem" }}
                >
                  Archived
                </h2>
                <ul className="space-y-1">
                  {archived.map((g) => (
                    <li
                      key={g.id}
                      className="flex items-baseline justify-between gap-3 text-[12px] text-[color:var(--color-ink-3)] border-b border-[color:var(--rule)] pb-1"
                    >
                      <span className="truncate">
                        {g.title} ·{" "}
                        <span className="italic">
                          {g.goal_type.replace("_", " ")}
                        </span>
                      </span>
                      <span className="text-[10px] uppercase tracking-[0.18em]">
                        {prettyDate(g.completed_at)}
                      </span>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {/* Timeline */}
            <section>
              <h2
                className="text-[color:var(--color-ink-3)] uppercase tracking-[0.22em] mb-4"
                style={{ fontFamily: "var(--font-display)", fontSize: "0.95rem" }}
              >
                Timeline
              </h2>
              {reached.length === 0 ? (
                <div className="border border-dashed border-[color:var(--rule)] rounded-[4px] p-6 text-center">
                  <p
                    className="italic text-[color:var(--color-ink-3)]"
                    style={{ fontFamily: "var(--font-serif)" }}
                  >
                    Your milestone timeline will fill up as you train.
                  </p>
                </div>
              ) : (
                <ul className="border-l border-[color:var(--rule)] pl-6 space-y-4">
                  {reached.map((m) => (
                    <li key={m.id} className="relative">
                      <span className="absolute -left-[30px] top-[6px] w-[10px] h-[10px] rounded-full bg-[color:var(--color-gold)] border-2 border-[color:var(--color-page)]" />
                      <div className="flex items-baseline justify-between gap-3">
                        <div
                          className="text-[color:var(--color-ink)]"
                          style={{
                            fontFamily: "var(--font-display)",
                            fontSize: "1.15rem",
                            letterSpacing: "0.04em",
                          }}
                        >
                          {m.label} · {m.goal_title.toUpperCase()}
                        </div>
                        <div className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)]">
                          {prettyDate(m.reached_at)}
                        </div>
                      </div>
                      <p
                        className="italic text-[color:var(--color-ink-3)] text-[13px] mt-1"
                        style={{ fontFamily: "var(--font-serif)" }}
                      >
                        {m.goal_type}
                        {m.exercise ? ` · ${m.exercise.replace(/_/g, " ")}` : ""}
                      </p>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          </div>
        )}
      </div>

      <GoalTemplatePicker
        open={pickerOpen}
        onClose={() => setPickerOpen(false)}
        onCreated={fetchAll}
      />
    </div>
  );
}
