// EditPlanModal — edit a saved plan's metadata (title, summary, dates).
// Used on ActivePlanCard and PlanHistoryTabs so users can tweak chatbot-
// created or custom plans without going back to the chatbot.

import { useEffect, useState } from "react";
import { plansApi, type Plan } from "@/lib/plansApi";

type Props = {
  plan: Plan;
  onClose: () => void;
  onSaved: (plan: Plan) => void;
};

export function EditPlanModal({ plan, onClose, onSaved }: Props) {
  const [title, setTitle] = useState(plan.title);
  const [summary, setSummary] = useState(plan.summary ?? "");
  const [startDate, setStartDate] = useState(plan.start_date);
  const [endDate, setEndDate] = useState(plan.end_date);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
    setSaving(true);
    setError(null);
    try {
      const updated = await plansApi.updatePlan(plan.id, {
        title: title.trim(),
        summary: summary.trim(),
        start_date: startDate,
        end_date: endDate,
      });
      onSaved(updated);
      onClose();
    } catch (e) {
      setError((e as Error)?.message ?? "Save failed.");
    } finally {
      setSaving(false);
    }
  };

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
            PLAN DETAILS
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
              Summary
            </div>
            <textarea
              value={summary}
              maxLength={500}
              rows={3}
              onChange={(e) => setSummary(e.target.value)}
              className="w-full border border-[color:var(--rule)] rounded-[3px] px-3 py-2 bg-[color:var(--color-page)] text-[color:var(--color-ink)] focus:outline-none focus:border-[color:var(--color-gold)] resize-none"
            />
          </label>
          <div className="grid grid-cols-2 gap-3">
            <label className="block">
              <div className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)] mb-1">
                Start date
              </div>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full border border-[color:var(--rule)] rounded-[3px] px-3 py-2 bg-[color:var(--color-page)] text-[color:var(--color-ink)] focus:outline-none focus:border-[color:var(--color-gold)]"
              />
            </label>
            <label className="block">
              <div className="text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-3)] mb-1">
                End date
              </div>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-full border border-[color:var(--rule)] rounded-[3px] px-3 py-2 bg-[color:var(--color-page)] text-[color:var(--color-ink)] focus:outline-none focus:border-[color:var(--color-gold)]"
              />
            </label>
          </div>
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
