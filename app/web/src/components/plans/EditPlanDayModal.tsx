// EditPlanDayModal — edit a single plan_day's date, rest flag, and exercises.
// Completed days are locked (backend rejects with 409 and this modal hides the
// pencil upstream, so this modal is only reached for editable days).

import { useEffect, useState } from "react";
import {
  plansApi,
  type PlanDay,
  type PlanDayExercise,
  type Plan,
} from "@/lib/plansApi";

const KNOWN_EXERCISES: string[] = [
  "squat",
  "lunge",
  "deadlift",
  "pullup",
  "pushup",
  "plank",
  "bicep_curl",
  "tricep_dip",
  "crunch",
  "lateral_raise",
  "side_plank",
];

type Props = {
  planId: number;
  day: PlanDay;
  onClose: () => void;
  onSaved: (plan: Plan) => void;
  onDeleted?: (plan: Plan) => void;
};

type RowDraft = {
  exercise: string;
  target_reps: number;
  target_sets: number;
  notes: string;
};

function toRow(ex: PlanDayExercise): RowDraft {
  // Phase 3: time_hold rows don't carry target_reps — this editor is
  // rep-count-centric for now, so we show 0. Family-aware editing comes
  // in Phase 4.
  const reps =
    "target_reps" in ex && typeof ex.target_reps === "number"
      ? ex.target_reps
      : 0;
  return {
    exercise: ex.exercise,
    target_reps: reps,
    target_sets: ex.target_sets,
    notes: ex.notes ?? "",
  };
}

export function EditPlanDayModal({
  planId,
  day,
  onClose,
  onSaved,
  onDeleted,
}: Props) {
  const [dayDate, setDayDate] = useState(day.day_date);
  const [isRest, setIsRest] = useState(day.is_rest);
  const [rows, setRows] = useState<RowDraft[]>(
    (day.exercises ?? []).map(toRow),
  );
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  const addRow = () => {
    setRows((r) => [
      ...r,
      { exercise: "pushup", target_reps: 10, target_sets: 3, notes: "" },
    ]);
  };

  const removeRow = (idx: number) => {
    setRows((r) => r.filter((_, i) => i !== idx));
  };

  const updateRow = (idx: number, patch: Partial<RowDraft>) => {
    setRows((r) => r.map((row, i) => (i === idx ? { ...row, ...patch } : row)));
  };

  const handleSave = async () => {
    if (!dayDate) {
      setError("Day date is required.");
      return;
    }
    if (!isRest && rows.length === 0) {
      setError("Add at least one exercise, or mark as rest day.");
      return;
    }
    setSaving(true);
    setError(null);
    setWarnings([]);
    try {
      const result = await plansApi.updatePlanDay(planId, day.id, {
        day_date: dayDate,
        is_rest: isRest,
        exercises: isRest ? [] : rows,
      });
      if (result.warnings?.length) setWarnings(result.warnings);
      onSaved(result.plan);
      onClose();
    } catch (e) {
      const msg = (e as Error)?.message ?? "Save failed.";
      setError(msg);
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!window.confirm("Delete this day from the plan?")) return;
    setDeleting(true);
    setError(null);
    try {
      const updated = await plansApi.deletePlanDay(planId, day.id);
      onDeleted?.(updated);
      onClose();
    } catch (e) {
      setError((e as Error)?.message ?? "Delete failed.");
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-[70] bg-black/40 backdrop-blur-sm flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-[600px] max-h-[90vh] bg-[color:var(--color-page)] border border-[color:var(--rule)] rounded-[4px] shadow-[0_12px_40px_-8px_rgba(0,0,0,0.4)] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="px-6 pt-6 pb-4 border-b border-[color:var(--rule)] shrink-0">
          <div className="text-[10px] uppercase tracking-[0.22em] text-[color:var(--color-gold)]">
            Edit day
          </div>
          <div
            className="text-[color:var(--color-ink)] mt-1"
            style={{
              fontFamily: "var(--font-display)",
              fontSize: "1.5rem",
              letterSpacing: "0.04em",
            }}
          >
            PLAN DAY
          </div>
        </header>

        <div className="p-6 space-y-4 overflow-auto">
          <div className="grid grid-cols-2 gap-3">
            <label className="block">
              <div className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)] mb-1">
                Date
              </div>
              <input
                type="date"
                value={dayDate}
                onChange={(e) => setDayDate(e.target.value)}
                className="w-full border border-[color:var(--rule)] rounded-[3px] px-3 py-2 bg-[color:var(--color-page)] text-[color:var(--color-ink)] focus:outline-none focus:border-[color:var(--color-gold)]"
              />
            </label>
            <label className="flex items-end gap-2 pb-1">
              <input
                type="checkbox"
                checked={isRest}
                onChange={(e) => setIsRest(e.target.checked)}
                className="accent-[color:var(--color-gold)] h-4 w-4"
              />
              <div className="text-[11px] uppercase tracking-[0.18em] text-[color:var(--color-ink)]">
                Rest day
              </div>
            </label>
          </div>

          {!isRest && (
            <div>
              <div className="flex items-baseline justify-between mb-2">
                <div className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)]">
                  Exercises
                </div>
                <button
                  type="button"
                  onClick={addRow}
                  className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-gold)] hover:text-[color:var(--color-gold-hover)]"
                >
                  + add
                </button>
              </div>
              <div className="space-y-2">
                {rows.length === 0 && (
                  <div className="text-[12px] italic text-[color:var(--color-ink-3)]">
                    No exercises yet — click "+ add".
                  </div>
                )}
                {rows.map((row, idx) => (
                  <div
                    key={idx}
                    className="border border-[color:var(--rule)] rounded-[3px] p-3 bg-[color:var(--color-raised)]/30"
                  >
                    <div className="grid grid-cols-[1fr_64px_64px_auto] gap-2 items-center">
                      <select
                        value={row.exercise}
                        onChange={(e) =>
                          updateRow(idx, { exercise: e.target.value })
                        }
                        className="border border-[color:var(--rule)] rounded-[2px] px-2 py-1.5 text-[12px] bg-[color:var(--color-page)] text-[color:var(--color-ink)] capitalize"
                      >
                        {KNOWN_EXERCISES.map((ex) => (
                          <option key={ex} value={ex} style={{ color: "#0A0A0A", background: "#F5F5F5" }}>
                            {ex.replace(/_/g, " ")}
                          </option>
                        ))}
                      </select>
                      <input
                        type="number"
                        min={1}
                        value={row.target_sets}
                        onChange={(e) =>
                          updateRow(idx, {
                            target_sets: parseInt(e.target.value) || 1,
                          })
                        }
                        className="border border-[color:var(--rule)] rounded-[2px] px-2 py-1.5 text-[12px] bg-[color:var(--color-page)] text-[color:var(--color-ink)] text-center tabular-nums"
                        title="Sets"
                      />
                      <input
                        type="number"
                        min={1}
                        value={row.target_reps}
                        onChange={(e) =>
                          updateRow(idx, {
                            target_reps: parseInt(e.target.value) || 1,
                          })
                        }
                        className="border border-[color:var(--rule)] rounded-[2px] px-2 py-1.5 text-[12px] bg-[color:var(--color-page)] text-[color:var(--color-ink)] text-center tabular-nums"
                        title="Reps"
                      />
                      <button
                        type="button"
                        onClick={() => removeRow(idx)}
                        className="text-[10px] uppercase tracking-[0.16em] text-[color:var(--color-bad)] hover:text-[color:var(--color-bad)] px-2"
                        aria-label="Remove exercise"
                      >
                        ✕
                      </button>
                    </div>
                    <div className="flex justify-between text-[9px] uppercase tracking-[0.16em] text-[color:var(--color-ink-3)] mt-1 px-1">
                      <span>exercise</span>
                      <span className="text-right w-[70px]">sets</span>
                      <span className="text-right w-[70px]">reps</span>
                      <span className="w-6" />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {warnings.length > 0 && (
            <div className="border border-[color:var(--color-gold)]/30 bg-[color:var(--color-gold)]/5 rounded-[3px] p-3 text-[11px] text-[color:var(--color-ink)]">
              <div className="uppercase tracking-[0.18em] text-[10px] text-[color:var(--color-gold)] mb-1">
                Adjusted
              </div>
              <ul className="space-y-0.5">
                {warnings.map((w, i) => (
                  <li key={i}>• {w}</li>
                ))}
              </ul>
            </div>
          )}

          {error && (
            <div className="text-[color:var(--color-bad)] text-[12px]">
              {error}
            </div>
          )}
        </div>

        <footer className="px-6 py-4 border-t border-[color:var(--rule)] flex items-center justify-between gap-2 shrink-0">
          <button
            type="button"
            onClick={handleDelete}
            disabled={saving || deleting}
            className="text-[11px] uppercase tracking-[0.18em] text-[color:var(--color-bad)] hover:underline px-2 py-2"
          >
            {deleting ? "Deleting…" : "Delete day"}
          </button>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onClose}
              disabled={saving || deleting}
              className="text-[11px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)] hover:text-[color:var(--color-ink)] px-4 py-2"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={handleSave}
              disabled={saving || deleting}
              className="text-[11px] uppercase tracking-[0.18em] text-[color:var(--color-page)] bg-[color:var(--color-gold)] hover:bg-[color:var(--color-gold-hover)] px-4 py-2 rounded-[3px] disabled:opacity-50"
            >
              {saving ? "Saving…" : "Save"}
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}
