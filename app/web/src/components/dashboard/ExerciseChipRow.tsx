// 10 exercise chips with signature photos. Clicking an active chip loads the
// DeepDivePanel below. Grayed-out chips have zero sessions.

type Chip = {
  id: string;
  display_name: string;
  has_data: boolean;
};

type Props = {
  exercises: Chip[];
  selected: string | null;
  onSelect: (id: string | null) => void;
};

const IMAGES: Record<string, string> = {
  squat: "/static/images/ex_squat.jpg",
  deadlift: "/static/images/ex_deadlift.jpg",
  pullup: "/static/images/ex_pullup.jpg",
  pushup: "/static/images/ex_pushup.jpg",
  plank: "/static/images/ex_plank.jpg",
  bicep_curl: "/static/images/ex_bicep_curl.jpg",
  tricep_dip: "/static/images/ex_tricep_dip.jpg",
  crunch: "/static/images/ex_crunch.jpg",
  lateral_raise: "/static/images/ex_lateral_raise.jpg",
  side_plank: "/static/images/ex_side_plank.jpg",
};

export function ExerciseChipRow({ exercises, selected, onSelect }: Props) {
  return (
    <section>
      <div className="flex items-baseline justify-between mb-4">
        <h2
          className="text-[color:var(--color-ink)]"
          style={{ fontFamily: "var(--font-display)", fontSize: "1.6rem", letterSpacing: "0.06em" }}
        >
          Drill into one lift
        </h2>
        {selected && (
          <button
            onClick={() => onSelect(null)}
            className="text-[10px] uppercase tracking-[0.24em] text-[color:var(--color-ink-3)] hover:text-[color:var(--color-gold)] transition-colors"
          >
            Close detail
          </button>
        )}
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
        {exercises.map((ex) => {
          const active = selected === ex.id;
          const disabled = !ex.has_data;
          return (
            <button
              key={ex.id}
              disabled={disabled}
              onClick={() => onSelect(active ? null : ex.id)}
              title={disabled ? "not yet trained" : `Drill into ${ex.display_name}`}
              className={`relative aspect-[4/3] overflow-hidden border rounded-sm group text-left transition-all duration-300 ${
                active
                  ? "border-[color:var(--color-gold)] shadow-[0_0_0_2px_rgba(174,231,16,0.35)]"
                  : "border-[color:var(--rule)]"
              } ${disabled ? "opacity-40 cursor-not-allowed grayscale" : "hover:border-[color:var(--color-gold-soft)]"}`}
            >
              <img
                src={IMAGES[ex.id] ?? "/static/images/ex_squat.jpg"}
                alt=""
                className="absolute inset-0 w-full h-full object-cover"
                loading="lazy"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/85 via-black/30 to-transparent" />
              <div className="absolute inset-x-0 bottom-0 p-3">
                <div
                  className="text-[color:var(--color-ink-on-dark)] leading-none"
                  style={{
                    fontFamily: "var(--font-display)",
                    fontSize: "1.15rem",
                    letterSpacing: "0.06em",
                  }}
                >
                  {ex.display_name}
                </div>
                {disabled && (
                  <div className="text-[9px] uppercase tracking-[0.2em] text-[color:var(--color-ink-on-dark-2)] mt-1">
                    not yet trained
                  </div>
                )}
              </div>
            </button>
          );
        })}
      </div>
    </section>
  );
}
