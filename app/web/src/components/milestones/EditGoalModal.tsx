// EditGoalModal — edit a goal's title, description, and target_value.
// Target-change triggers a server-side re-anchor of unreached milestones
// at 25/50/75/100% of the new target. Reached milestones are preserved.

import { useEffect, useState } from "react";
import { plansApi, type Goal } from "@/lib/plansApi";

type Props = {
  goal: Goal;
  onClose: () => void;
  onSaved: (goal: Goal) => void;
};

export function EditGoalModal({ goal, onClose, onSaved }: Props) {
  const [title, setTitle] = useState(goal.title);
  const [description, setDescription] = useState(goal.description ?? "");
  const [targetValue, setTargetValue] = useState<number>(goal.target_value);
  // Phase 4 — strength goals carry target_reps ("how many clean reps at
  // the target weight"). Editable only for strength type.
  const [targetReps, setTargetReps] = useState<number>(goal.target_reps ?? 0);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isStrength = goal.goal_type === "strength";

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  const handleSave = async () => {
    if (!title.trim()) {
      setError("Title is required.");
      return;
    }
    if (!(targetValue > 0)) {
      setError("Target must be greater than zero.");
      return;
    }
    if (isStrength && !(targetReps > 0)) {
      setError("Strength goals need a positive target_reps.");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const patch: Partial<Goal> = {
        title: title.trim(),
        description: description.trim(),
        target_value: targetValue,
      };
      if (isStrength) patch.target_reps = targetReps;
      const updated = await plansApi.patchGoal(goal.id, patch);
      onSaved(updated);
      onClose();
    } catch (e) {
      setError((e as Error)?.message ?? "Save failed.");
    } finally {
      setSaving(false);
    }
  };

  const targetChanged = Math.abs(targetValue - goal.target_value) > 1e-9;
  const reachedCount = goal.milestones.filter((m) => m.reached).length;
  const unreachedCount = goal.milestones.length - reachedCount;

  return (
    <div
      className="fixed inset-0 z-[70] bg-black/40 backdrop-blur-sm flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-[520px] bg-[color:var(--color-page)] border border-[color:var(--rule)] rounded-[4px] shadow-[0_12px_40px_-8px_rgba(0,0,0,0.4)]"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="px-6 pt-6 pb-4 border-b border-[color:var(--rule)]">
          <div className="text-[10px] uppercase tracking-[0.22em] text-[color:var(--color-gold)]">
            Edit
          </div>
          <div
            className="text-[color:var(--color-ink)] mt-1"
            style={{
              fontFamily: "var(--font-display)",
              fontSize: "1.5rem",
              letterSpacing: "0.04em",
            }}
          >
            GOAL
          </div>
        </header>
        <div className="p-6 space-y-4">
          <label className="block">
            <div className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)] mb-1">
              Title
            </div>
            <input
              type="text"
              value={title}
              maxLength={120}
              onChange={(e) => setTitle(e.target.value)}
              className="w-full border border-[color:var(--rule)] rounded-[3px] px-3 py-2 bg-[color:var(--color-page)] text-[color:var(--color-ink)] focus:outline-none focus:border-[color:var(--color-gold)]"
            />
          </label>
          <label className="block">
            <div className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)] mb-1">
              Description
            </div>
            <textarea
              value={description}
              maxLength={500}
              rows={3}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full border border-[color:var(--rule)] rounded-[3px] px-3 py-2 bg-[color:var(--color-page)] text-[color:var(--color-ink)] focus:outline-none focus:border-[color:var(--color-gold)] resize-none"
            />
          </label>
          <label className="block">
            <div className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)] mb-1">
              Target ({goal.unit})
            </div>
            <input
              type="number"
              min={0.01}
              step={goal.unit === "kg" ? 2.5 : 1}
              value={targetValue}
              onChange={(e) => setTargetValue(parseFloat(e.target.value) || 0)}
              className="w-full border border-[color:var(--rule)] rounded-[3px] px-3 py-2 bg-[color:var(--color-page)] text-[color:var(--color-ink)] focus:outline-none focus:border-[color:var(--color-gold)] tabular-nums"
            />
          </label>

          {isStrength && (
            <label className="block">
              <div className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)] mb-1">
                Clean reps at target weight
              </div>
              <input
                type="number"
                min={1}
                step={1}
                value={targetReps}
                onChange={(e) => setTargetReps(parseInt(e.target.value) || 0)}
                className="w-full border border-[color:var(--rule)] rounded-[3px] px-3 py-2 bg-[color:var(--color-page)] text-[color:var(--color-ink)] focus:outline-none focus:border-[color:var(--color-gold)] tabular-nums"
              />
              <p
                className="italic text-[color:var(--color-ink-3)] text-[11px] mt-1"
                style={{ fontFamily: "var(--font-serif)" }}
              >
                Strength progress counts only sets where this many reps scored
                ≥ 0.6.
              </p>
            </label>
          )}

          {targetChanged && (
            <div className="border border-[color:var(--color-gold)]/30 bg-[color:var(--color-gold)]/5 rounded-[3px] p-3 text-[11px] text-[color:var(--color-ink)]">
              <div className="uppercase tracking-[0.18em] text-[10px] text-[color:var(--color-gold)] mb-1">
                Milestones will re-anchor
              </div>
              <div>
                {reachedCount} reached milestone
                {reachedCount === 1 ? "" : "s"} stays frozen.
                {unreachedCount > 0
                  ? ` ${unreachedCount} unreached will move to ${[25, 50, 75, 100]
                      .filter((_, i) => i >= 4 - unreachedCount)
                      .map((p) => `${Math.round((targetValue * p) / 100)}`)
                      .join(" / ")}.`
                  : ""}
              </div>
            </div>
          )}

          {error && (
            <div className="text-[color:var(--color-bad)] text-[12px]">
              {error}
            </div>
          )}
        </div>
        <footer className="px-6 py-4 border-t border-[color:var(--rule)] flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            disabled={saving}
            className="text-[11px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)] hover:text-[color:var(--color-ink)] px-4 py-2"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleSave}
            disabled={saving}
            className="text-[11px] uppercase tracking-[0.18em] text-[color:var(--color-page)] bg-[color:var(--color-gold)] hover:bg-[color:var(--color-gold-hover)] px-4 py-2 rounded-[3px] disabled:opacity-50"
          >
            {saving ? "Saving…" : "Save"}
          </button>
        </footer>
      </div>
    </div>
  );
}
