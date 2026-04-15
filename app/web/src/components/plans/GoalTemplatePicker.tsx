// GoalTemplatePicker — modal for creating a SMART goal from a template.
// Accessible from PlansPage and DashboardPage's ActiveGoalsCard.

import { useEffect, useState } from "react";
import { plansApi, type GoalTemplate } from "@/lib/plansApi";

type Props = {
  open: boolean;
  onClose: () => void;
  onCreated?: () => void;
};

export function GoalTemplatePicker({ open, onClose, onCreated }: Props) {
  const [templates, setTemplates] = useState<GoalTemplate[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyKey, setBusyKey] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setLoading(true);
    plansApi
      .listGoalTemplates()
      .then((ts) => {
        if (!cancelled) setTemplates(ts);
      })
      .catch((e) => {
        if (!cancelled) setError(e?.message ?? "Failed to load templates");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open]);

  if (!open) return null;

  const handleAdd = async (t: GoalTemplate) => {
    setBusyKey(t.key);
    setError(null);
    try {
      await plansApi.createGoal({
        title: t.title,
        goal_type: t.goal_type,
        target_value: t.target_value,
        unit: t.unit,
        exercise: t.exercise,
        period: t.period,
        description: t.description,
      });
      onCreated?.();
      onClose();
    } catch (e) {
      const err = e as Error & { code?: string };
      if (err.code === "goal_cap") {
        setError("You already have 10 active goals. Complete or delete one first.");
      } else {
        setError(err.message ?? "Failed to create goal");
      }
    } finally {
      setBusyKey(null);
    }
  };

  return (
    <div
      className="fixed inset-0 z-[70] bg-[color:var(--color-ink)]/40 backdrop-blur-sm flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-[color:var(--color-page)] border border-[color:var(--rule)] rounded-[4px] max-w-2xl w-full max-h-[80vh] overflow-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="px-6 pt-6 pb-4 border-b border-[color:var(--rule)] flex items-baseline justify-between">
          <h2
            className="text-[color:var(--color-ink)]"
            style={{
              fontFamily: "var(--font-display)",
              fontSize: "1.8rem",
              letterSpacing: "0.04em",
            }}
          >
            PICK A GOAL
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="text-[11px] uppercase tracking-[0.2em] text-[color:var(--color-ink-3)] hover:text-[color:var(--color-ink)]"
          >
            Close
          </button>
        </div>
        {error && (
          <div className="px-6 py-3 bg-[color:var(--color-bad)]/10 border-b border-[color:var(--color-bad)]/30 text-[color:var(--color-bad)] text-[12px]">
            {error}
          </div>
        )}
        {loading ? (
          <div className="p-6 text-[color:var(--color-ink-3)] italic">Loading templates…</div>
        ) : (
          <ul className="p-6 grid gap-4 md:grid-cols-2">
            {templates.map((t) => (
              <li
                key={t.key}
                className="border border-[color:var(--rule)] rounded-[4px] p-4 bg-[color:var(--color-raised)]/40 flex flex-col"
              >
                <h3
                  className="text-[color:var(--color-ink)]"
                  style={{ fontFamily: "var(--font-display)", fontSize: "1.3rem", letterSpacing: "0.04em" }}
                >
                  {t.title.toUpperCase()}
                </h3>
                <p
                  className="italic text-[color:var(--color-ink-3)] text-[13px] mt-1 mb-3 flex-1"
                  style={{ fontFamily: "var(--font-serif)" }}
                >
                  {t.description}
                </p>
                <button
                  type="button"
                  onClick={() => handleAdd(t)}
                  disabled={busyKey === t.key}
                  className="text-[10px] uppercase tracking-[0.22em] px-4 py-2 bg-[color:var(--color-gold)] text-[color:var(--color-page)] hover:bg-[color:var(--color-gold-hover)] transition-colors rounded-[3px] disabled:opacity-50"
                >
                  {busyKey === t.key ? "Adding…" : "Add goal"}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
