import { useRef, useState, type ChangeEvent, type KeyboardEvent } from "react";
import { useNavigate } from "react-router-dom";
import { Section } from "@/components/sections/Section";
import { SectionHeader } from "@/components/sections/SectionHeader";
import { CinematicHero } from "@/components/sections/CinematicHero";
import { GradientGlow } from "@/components/sections/GradientGlow";
import { useRevealAnimations } from "@/lib/useRevealAnimations";
import { EXERCISES, type Exercise } from "@/types/exercise";

const DEFAULT_WEIGHT_KG = 60;
const weightStorageKey = (slug: string) => `forma:last_weight:${slug}`;

function readLastWeight(slug: string): number {
  try {
    const raw = window.localStorage.getItem(weightStorageKey(slug));
    if (raw) {
      const n = parseFloat(raw);
      if (Number.isFinite(n) && n > 0) return n;
    }
  } catch {
    // ignore
  }
  return DEFAULT_WEIGHT_KG;
}

function writeLastWeight(slug: string, weight: number): void {
  try {
    window.localStorage.setItem(weightStorageKey(slug), String(weight));
  } catch {
    // ignore
  }
}

const MUSCLES: Record<string, string[]> = {
  pushup: ["Chest", "Shoulders", "Triceps", "Core"],
  squat: ["Quads", "Glutes", "Hamstrings", "Core"],
  lunge: ["Quads", "Glutes", "Hamstrings", "Calves"],
  deadlift: ["Hamstrings", "Glutes", "Back", "Traps", "Core"],
  pullup: ["Lats", "Biceps", "Back", "Core"],
  plank: ["Core", "Shoulders", "Glutes"],
  bicep_curl: ["Biceps", "Forearms"],
  tricep_dip: ["Triceps", "Chest", "Shoulders"],
  crunch: ["Abs", "Hip Flexors", "Core"],
  lateral_raise: ["Lateral Deltoid", "Anterior Deltoid", "Traps"],
  side_plank: ["Obliques", "Core", "Shoulders", "Glutes"],
};

export function ExercisesPage() {
  const scopeRef = useRef<HTMLDivElement>(null);
  useRevealAnimations(scopeRef);

  return (
    <div ref={scopeRef}>
      <CinematicHero
        image="/static/images/cinematic/cinematic_barbell.jpg"
        anchor="left"
        minHeight="min-h-[74vh]"
      >
        <span
          data-reveal
          className="block font-[family-name:var(--font-mono)] text-[0.7rem] uppercase tracking-[0.32em] text-[color:var(--color-gold-soft)] mb-8"
        >
          / / / THE LIBRARY · 11 EXERCISES
        </span>
        <h1
          data-reveal
          className="font-[family-name:var(--font-display)] leading-[0.88] text-[color:var(--color-ink-on-dark)]"
          style={{ fontSize: "clamp(3rem, 8vw, 6.5rem)" }}
        >
          Eleven exercises,
          <br />
          <em className="not-italic font-[family-name:var(--font-serif)] italic text-[color:var(--color-gold-soft)]">
            one coach.
          </em>
        </h1>
        <p
          data-reveal
          className="mt-8 font-[family-name:var(--font-serif)] italic text-xl md:text-2xl text-[color:var(--color-ink-on-dark-2)] leading-[1.4]"
        >
          Every exercise FORMA tracks. Pick one and train — form feedback runs live in your browser.
        </p>
      </CinematicHero>

      <Section variant="light" className="relative overflow-hidden">
        <GradientGlow position="top-right" intensity="medium" />
        <GradientGlow position="bottom-left" intensity="subtle" />
        <SectionHeader
          eyebrow="The full library"
          title="Built for the lifts"
          italic="that matter."
          body="Compound lifts, bodyweight staples, isolation moves, and two static holds. Each one has its own dedicated form detector — no generic classifier, no guessing. Tap any card to begin."
        />

        <div
          data-reveal
          className="mt-20 grid gap-6 sm:grid-cols-2 lg:grid-cols-3"
        >
          {EXERCISES.map((ex) => (
            <ExerciseCard key={ex.slug} exercise={ex} />
          ))}
        </div>
      </Section>
    </div>
  );
}

function ExerciseCard({ exercise }: { exercise: Exercise }) {
  const navigate = useNavigate();
  const [weight, setWeight] = useState<string>(() =>
    exercise.isWeighted ? String(readLastWeight(exercise.slug)) : "",
  );

  const muscles = MUSCLES[exercise.slug] ?? [];
  const imgSrc = `/static/images/ex_${exercise.slug}.jpg`;

  const handleStart = () => {
    if (exercise.isWeighted) {
      const parsed = parseFloat(weight);
      if (!Number.isFinite(parsed) || parsed <= 0) return;
      writeLastWeight(exercise.slug, parsed);
      navigate(`/workout/${exercise.slug}?weight=${parsed}`);
    } else {
      navigate(`/workout/${exercise.slug}`);
    }
  };

  const onWeightChange = (e: ChangeEvent<HTMLInputElement>) => {
    setWeight(e.target.value);
  };

  const onWeightKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") handleStart();
  };

  const canStart = exercise.isWeighted ? parseFloat(weight) > 0 : true;

  return (
    <article className="group relative flex flex-col overflow-hidden bg-[color:var(--color-page)] border border-[color:var(--rule)] rounded-[4px] transition-all duration-500 hover:border-[color:var(--color-gold)] hover:shadow-[0_32px_64px_-24px_rgba(0,0,0,0.4)]">
      {/* Image */}
      <div className="relative h-64 overflow-hidden">
        <img
          src={imgSrc}
          alt={exercise.name}
          loading="lazy"
          className="absolute inset-0 h-full w-full object-cover transition-transform duration-[900ms] ease-[var(--ease-out-editorial)] group-hover:scale-[1.05]"
        />
        <div
          aria-hidden="true"
          className="absolute inset-0 bg-gradient-to-t from-[rgba(13,13,13,0.78)] via-[rgba(13,13,13,0.15)] to-transparent"
        />
        <div className="absolute top-4 left-4 flex items-center gap-2">
          {exercise.primary && (
            <span className="text-[9px] uppercase tracking-[0.2em] px-2 py-1 bg-[color:var(--color-gold)] text-[color:var(--color-page)] font-medium">
              Default
            </span>
          )}
          {exercise.isWeighted && (
            <span className="text-[9px] uppercase tracking-[0.2em] px-2 py-1 bg-[color:var(--color-page)]/75 text-[color:var(--color-ink)] border border-[color:var(--rule-strong)] backdrop-blur-sm">
              Weighted
            </span>
          )}
        </div>
        <div className="absolute bottom-4 left-5 right-5">
          <h3
            className="font-[family-name:var(--font-display)] text-[color:var(--color-ink-on-dark)] leading-[0.95] tracking-[0.02em] [text-shadow:0_2px_12px_rgba(0,0,0,0.6)]"
            style={{ fontSize: "clamp(1.75rem, 3vw, 2.5rem)" }}
          >
            {exercise.name}
          </h3>
        </div>
      </div>

      {/* Body */}
      <div className="flex flex-1 flex-col p-6 gap-5">
        <p className="font-[family-name:var(--font-serif)] italic text-[color:var(--color-ink-2)] leading-[1.5]">
          {exercise.tagline}
        </p>

        {muscles.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {muscles.map((m) => (
              <span
                key={m}
                className="text-[10px] uppercase tracking-[0.14em] px-2 py-1 border border-[color:var(--rule)] text-[color:var(--color-ink-2)]"
              >
                {m}
              </span>
            ))}
          </div>
        )}

        {/* Action row */}
        <div className="mt-auto pt-4 border-t border-[color:var(--rule)]">
          {exercise.isWeighted ? (
            <div className="flex items-stretch gap-3">
              <label className="flex items-baseline gap-2 flex-1 border-b border-[color:var(--color-ink)]/40 focus-within:border-[color:var(--color-gold)] transition-colors">
                <input
                  type="number"
                  inputMode="decimal"
                  min={0}
                  step={0.5}
                  value={weight}
                  onChange={onWeightChange}
                  onKeyDown={onWeightKey}
                  aria-label={`${exercise.name} weight in kilograms`}
                  className="w-full bg-transparent font-[family-name:var(--font-display)] text-2xl tabular-nums text-[color:var(--color-ink)] focus:outline-none py-1"
                />
                <span className="font-[family-name:var(--font-display)] text-[11px] uppercase tracking-[0.16em] text-[color:var(--color-ink-2)]">
                  KG
                </span>
              </label>
              <button
                type="button"
                onClick={handleStart}
                disabled={!canStart}
                className="px-5 py-2.5 bg-[color:var(--color-gold)] text-[color:var(--color-page)] text-[11px] uppercase tracking-[0.16em] font-medium rounded-[2px] hover:bg-[color:var(--color-gold-hover)] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                Start →
              </button>
            </div>
          ) : (
            <button
              type="button"
              onClick={handleStart}
              className="w-full px-5 py-3 bg-[color:var(--color-gold)] text-[color:var(--color-page)] text-[11px] uppercase tracking-[0.16em] font-medium rounded-[2px] hover:bg-[color:var(--color-gold-hover)] transition-colors"
            >
              Start Workout →
            </button>
          )}
        </div>
      </div>
    </article>
  );
}
