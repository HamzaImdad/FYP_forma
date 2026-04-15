// Chat input — textarea with auto-grow, Enter-to-send, Shift+Enter newline,
// 2000-char soft cap (server enforces).

import { useCallback, useEffect, useRef, useState } from "react";

type Props = {
  disabled?: boolean;
  onSend: (text: string) => void;
  placeholder?: string;
  initialValue?: string;
};

const MAX_CHARS = 2000;

export function MessageInput({ disabled, onSend, placeholder, initialValue }: Props) {
  const [value, setValue] = useState(initialValue ?? "");
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (initialValue !== undefined) setValue(initialValue);
  }, [initialValue]);

  const autoGrow = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, []);

  useEffect(() => {
    autoGrow();
  }, [value, autoGrow]);

  const submit = () => {
    const text = value.trim();
    if (!text || disabled) return;
    onSend(text);
    setValue("");
  };

  return (
    <div className="border-t border-[color:var(--rule)] bg-[color:var(--color-page)] px-6 md:px-10 py-4">
      <div className="flex items-end gap-3">
        <textarea
          ref={ref}
          value={value}
          onChange={(e) => setValue(e.target.value.slice(0, MAX_CHARS))}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          placeholder={placeholder ?? "Ask FORMA about your training…"}
          disabled={disabled}
          rows={1}
          className="flex-1 resize-none bg-transparent border border-[color:var(--rule)] rounded-[6px] px-4 py-3 text-[0.97rem] leading-[1.55] text-[color:var(--color-ink)] placeholder:text-[color:var(--color-ink-4)] focus:outline-none focus:border-[color:var(--color-gold)] disabled:opacity-50 font-[family-name:var(--font-body)]"
          style={{ maxHeight: 200 }}
        />
        <button
          type="button"
          onClick={submit}
          disabled={disabled || !value.trim()}
          className="h-[44px] px-5 bg-[color:var(--color-gold)] text-[color:var(--color-page)] text-[11px] uppercase tracking-[0.24em] font-medium rounded-[4px] hover:bg-[color:var(--color-gold-hover)] transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Send
        </button>
      </div>
      <div className="mt-2 flex justify-between text-[10px] uppercase tracking-[0.18em] text-[color:var(--color-ink-4)]">
        <span>Enter to send · Shift+Enter for newline</span>
        <span className={value.length > MAX_CHARS * 0.9 ? "text-[color:var(--color-gold)]" : ""}>
          {value.length}/{MAX_CHARS}
        </span>
      </div>
    </div>
  );
}
