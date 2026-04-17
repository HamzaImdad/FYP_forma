// ProfilePage — editable identity/body stats (Card 1) + read-only lifetime
// stats summary (Card 2). Composed entirely from existing endpoints:
// /api/auth/profile PATCH for saves, /api/dashboard/overview + /api/badges
// + /api/dashboard/distribution for the stats card.

import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import { dashboardApi, type DashboardOverview } from "@/lib/dashboardApi";
import { plansApi, type Badge } from "@/lib/plansApi";
import { useAuth, type User } from "@/context/AuthContext";

const EXPERIENCE_OPTIONS: User["experience_level"][] = [
  "beginner",
  "intermediate",
  "advanced",
];
const GOAL_OPTIONS: User["training_goal"][] = [
  "strength",
  "size",
  "endurance",
  "skill",
];
const TONE_OPTIONS: User["coaching_tone"][] = [
  "gentle",
  "neutral",
  "drill_sergeant",
];

type ExerciseDistribution = {
  exercise: string;
  count: number;
  total_reps: number;
};

type StatValue = { label: string; value: string };

function formatDuration(seconds: number): string {
  if (!seconds) return "0s";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m`;
  return `${Math.round(seconds)}s`;
}

export function ProfilePage() {
  const { user, refresh } = useAuth();

  const [displayName, setDisplayName] = useState(user?.display_name ?? "");
  const [avatarUrl, setAvatarUrl] = useState(user?.avatar_url ?? "");
  const [heightCm, setHeightCm] = useState<number | "">(user?.height_cm ?? "");
  const [weightKg, setWeightKg] = useState<number | "">(user?.weight_kg ?? "");
  const [age, setAge] = useState<number | "">(user?.age ?? "");
  const [experience, setExperience] = useState<User["experience_level"]>(
    user?.experience_level ?? "beginner",
  );
  const [goal, setGoal] = useState<User["training_goal"]>(
    user?.training_goal ?? "strength",
  );
  const [tone, setTone] = useState<User["coaching_tone"]>(
    user?.coaching_tone ?? "neutral",
  );
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  const [overview, setOverview] = useState<DashboardOverview | null>(null);
  const [badges, setBadges] = useState<Badge[]>([]);
  const [distribution, setDistribution] = useState<ExerciseDistribution[]>([]);

  useEffect(() => {
    if (!user) return;
    setDisplayName(user.display_name);
    setAvatarUrl(user.avatar_url ?? "");
    setHeightCm(user.height_cm ?? "");
    setWeightKg(user.weight_kg ?? "");
    setAge(user.age ?? "");
    setExperience(user.experience_level);
    setGoal(user.training_goal);
    setTone(user.coaching_tone);
  }, [user]);

  useEffect(() => {
    dashboardApi.overview().then(setOverview).catch(() => setOverview(null));
    plansApi.listBadges().then(setBadges).catch(() => setBadges([]));
    api<ExerciseDistribution[]>("/api/dashboard/distribution")
      .then(setDistribution)
      .catch(() => setDistribution([]));
  }, []);

  const handleSave = useCallback(async () => {
    setSaving(true);
    setSaveMessage(null);
    setSaveError(null);
    try {
      await api("/api/auth/profile", {
        method: "PATCH",
        body: JSON.stringify({
          display_name: displayName.trim(),
          avatar_url: avatarUrl.trim() || null,
          height_cm: heightCm === "" ? null : Number(heightCm),
          weight_kg: weightKg === "" ? null : Number(weightKg),
          age: age === "" ? null : Number(age),
          experience_level: experience,
          training_goal: goal,
          coaching_tone: tone,
        }),
      });
      await refresh();
      setSaveMessage("Saved.");
    } catch (e) {
      setSaveError((e as Error)?.message ?? "Save failed.");
    } finally {
      setSaving(false);
    }
  }, [
    displayName,
    avatarUrl,
    heightCm,
    weightKg,
    age,
    experience,
    goal,
    tone,
    refresh,
  ]);

  const stats = useMemo<StatValue[]>(() => {
    if (!overview) return [];
    const t = overview.today;
    const pr = overview.personal_records;
    const totalSessions = overview.totals.all_sessions ?? 0;
    const badgesEarned = badges.filter((b) => b.earned).length;
    const favorite =
      distribution.length > 0
        ? distribution.reduce((a, b) =>
            (a.count ?? 0) >= (b.count ?? 0) ? a : b,
          )
        : null;
    return [
      { label: "Total sessions", value: String(totalSessions) },
      { label: "Reps today", value: String(t?.reps ?? 0) },
      {
        label: "Longest streak",
        value: `${pr?.longest_streak ?? 0}d`,
      },
      {
        label: "Best plank",
        value: pr?.longest_plank
          ? formatDuration(pr.longest_plank.duration_sec)
          : "—",
      },
      {
        label: "Biggest session",
        value: pr?.biggest_session
          ? `${pr.biggest_session.total_reps} reps`
          : "—",
      },
      {
        label: "Best form score",
        value: pr?.best_form_day
          ? `${Math.round(pr.best_form_day.avg_form_score * 100)}`
          : "—",
      },
      {
        label: "Favorite exercise",
        value: favorite
          ? favorite.exercise.replace(/_/g, " ").toUpperCase()
          : "—",
      },
      { label: "Badges", value: `${badgesEarned} / ${badges.length || 13}` },
    ];
  }, [overview, badges, distribution]);

  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center text-[color:var(--color-ink-3)]">
        Loading…
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[color:var(--color-page)] pt-[72px]">
      <div className="max-w-[960px] mx-auto px-6 md:px-10 py-10">
        <header className="mb-8">
          <h1
            className="text-[color:var(--color-ink)]"
            style={{
              fontFamily: "var(--font-display)",
              fontSize: "clamp(2.6rem, 5vw, 4rem)",
              letterSpacing: "0.04em",
              lineHeight: 1,
            }}
          >
            PROFILE
          </h1>
          <p
            className="italic text-[color:var(--color-ink-3)] mt-2"
            style={{ fontFamily: "var(--font-serif)", fontSize: "1.2rem" }}
          >
            Who you are, what you've done.
          </p>
        </header>

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
          {/* Card 1 — editable identity */}
          <section className="border border-[color:var(--rule)] rounded-[4px] bg-[color:var(--color-raised)]/40 p-6">
            <div className="text-[10px] uppercase tracking-[0.22em] text-[color:var(--color-gold)] mb-1">
              Identity
            </div>
            <div
              className="text-[color:var(--color-ink)] mb-5"
              style={{
                fontFamily: "var(--font-display)",
                fontSize: "1.6rem",
                letterSpacing: "0.04em",
              }}
            >
              DETAILS
            </div>

            <div className="space-y-4">
              <label className="block">
                <div className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)] mb-1">
                  Display name
                </div>
                <input
                  type="text"
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  className="w-full border border-[color:var(--rule)] rounded-[3px] px-3 py-2 bg-[color:var(--color-page)] text-[color:var(--color-ink)] focus:outline-none focus:border-[color:var(--color-gold)]"
                />
              </label>
              <label className="block">
                <div className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)] mb-1">
                  Email
                </div>
                <input
                  type="email"
                  value={user.email}
                  readOnly
                  disabled
                  className="w-full border border-[color:var(--rule)] rounded-[3px] px-3 py-2 bg-[color:var(--color-raised)]/60 text-[color:var(--color-ink-3)]"
                />
              </label>
              <label className="block">
                <div className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)] mb-1">
                  Avatar URL
                </div>
                <input
                  type="url"
                  value={avatarUrl}
                  placeholder="https://…"
                  onChange={(e) => setAvatarUrl(e.target.value)}
                  className="w-full border border-[color:var(--rule)] rounded-[3px] px-3 py-2 bg-[color:var(--color-page)] text-[color:var(--color-ink)] focus:outline-none focus:border-[color:var(--color-gold)]"
                />
              </label>

              <div className="grid grid-cols-3 gap-3">
                <label className="block">
                  <div className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)] mb-1">
                    Height (cm)
                  </div>
                  <input
                    type="number"
                    value={heightCm}
                    onChange={(e) =>
                      setHeightCm(e.target.value === "" ? "" : Number(e.target.value))
                    }
                    className="w-full border border-[color:var(--rule)] rounded-[3px] px-3 py-2 bg-[color:var(--color-page)] text-[color:var(--color-ink)] tabular-nums focus:outline-none focus:border-[color:var(--color-gold)]"
                  />
                </label>
                <label className="block">
                  <div className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)] mb-1">
                    Weight (kg)
                  </div>
                  <input
                    type="number"
                    value={weightKg}
                    onChange={(e) =>
                      setWeightKg(e.target.value === "" ? "" : Number(e.target.value))
                    }
                    className="w-full border border-[color:var(--rule)] rounded-[3px] px-3 py-2 bg-[color:var(--color-page)] text-[color:var(--color-ink)] tabular-nums focus:outline-none focus:border-[color:var(--color-gold)]"
                  />
                </label>
                <label className="block">
                  <div className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)] mb-1">
                    Age
                  </div>
                  <input
                    type="number"
                    value={age}
                    onChange={(e) =>
                      setAge(e.target.value === "" ? "" : Number(e.target.value))
                    }
                    className="w-full border border-[color:var(--rule)] rounded-[3px] px-3 py-2 bg-[color:var(--color-page)] text-[color:var(--color-ink)] tabular-nums focus:outline-none focus:border-[color:var(--color-gold)]"
                  />
                </label>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <label className="block">
                  <div className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)] mb-1">
                    Experience
                  </div>
                  <select
                    value={experience}
                    onChange={(e) =>
                      setExperience(e.target.value as User["experience_level"])
                    }
                    className="w-full border border-[color:var(--rule)] rounded-[3px] px-3 py-2 bg-[color:var(--color-page)] text-[color:var(--color-ink)] capitalize focus:outline-none focus:border-[color:var(--color-gold)]"
                  >
                    {EXPERIENCE_OPTIONS.map((o) => (
                      <option key={o} value={o} style={{ color: "#0A0A0A", background: "#F5F5F5" }}>
                        {o}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="block">
                  <div className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)] mb-1">
                    Training goal
                  </div>
                  <select
                    value={goal}
                    onChange={(e) =>
                      setGoal(e.target.value as User["training_goal"])
                    }
                    className="w-full border border-[color:var(--rule)] rounded-[3px] px-3 py-2 bg-[color:var(--color-page)] text-[color:var(--color-ink)] capitalize focus:outline-none focus:border-[color:var(--color-gold)]"
                  >
                    {GOAL_OPTIONS.map((o) => (
                      <option key={o} value={o} style={{ color: "#0A0A0A", background: "#F5F5F5" }}>
                        {o}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="block">
                  <div className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)] mb-1">
                    Coach tone
                  </div>
                  <select
                    value={tone}
                    onChange={(e) =>
                      setTone(e.target.value as User["coaching_tone"])
                    }
                    className="w-full border border-[color:var(--rule)] rounded-[3px] px-3 py-2 bg-[color:var(--color-page)] text-[color:var(--color-ink)] capitalize focus:outline-none focus:border-[color:var(--color-gold)]"
                  >
                    {TONE_OPTIONS.map((o) => (
                      <option key={o} value={o} style={{ color: "#0A0A0A", background: "#F5F5F5" }}>
                        {o.replace("_", " ")}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              <div className="flex items-center justify-end gap-3 pt-2">
                {saveMessage && (
                  <span className="text-[11px] uppercase tracking-[0.18em] text-[color:var(--color-good)]">
                    {saveMessage}
                  </span>
                )}
                {saveError && (
                  <span className="text-[11px] text-[color:var(--color-bad)]">
                    {saveError}
                  </span>
                )}
                <button
                  type="button"
                  onClick={handleSave}
                  disabled={saving}
                  className="text-[11px] uppercase tracking-[0.22em] px-5 py-3 bg-[color:var(--color-gold)] text-[color:var(--color-page)] hover:bg-[color:var(--color-gold-hover)] rounded-[3px] disabled:opacity-50"
                >
                  {saving ? "Saving…" : "Save changes"}
                </button>
              </div>
            </div>
          </section>

          {/* Card 2 — lifetime stats */}
          <section className="border border-[color:var(--rule)] rounded-[4px] bg-[color:var(--color-raised)]/40 p-6">
            <div className="text-[10px] uppercase tracking-[0.22em] text-[color:var(--color-gold)] mb-1">
              Lifetime
            </div>
            <div
              className="text-[color:var(--color-ink)] mb-5"
              style={{
                fontFamily: "var(--font-display)",
                fontSize: "1.6rem",
                letterSpacing: "0.04em",
              }}
            >
              EVERYTHING YOU'VE DONE
            </div>

            {stats.length === 0 ? (
              <div className="text-[color:var(--color-ink-3)] italic text-sm">
                Loading stats…
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-4">
                {stats.map((s) => (
                  <div key={s.label}>
                    <div
                      className="text-[color:var(--color-gold)] tabular-nums"
                      style={{
                        fontFamily: "var(--font-display)",
                        fontSize: "1.8rem",
                        letterSpacing: "0.02em",
                      }}
                    >
                      {s.value}
                    </div>
                    <div className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)] mt-0.5">
                      {s.label}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
