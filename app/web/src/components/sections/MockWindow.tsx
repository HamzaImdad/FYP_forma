import type { ReactNode } from "react";

export function MockWindow({
  label,
  children,
  tone = "light",
  className = "",
}: {
  label?: string;
  children: ReactNode;
  tone?: "light" | "dark";
  className?: string;
}) {
  const isDark = tone === "dark";
  const surface = isDark
    ? "bg-[#141414] text-[color:var(--color-ink-on-dark)] border-[color:var(--color-ink-on-dark)]/10"
    : "bg-[color:var(--color-page)] text-[color:var(--color-ink)] border-[color:var(--rule)]";
  const bar = isDark
    ? "bg-[#0b0b0b] border-[color:var(--color-ink-on-dark)]/10"
    : "bg-[color:var(--color-raised)] border-[color:var(--rule)]";
  const dotColors = ["#F87171", "#AEE710", "#4ADE80"];

  return (
    <div
      className={`relative rounded-[10px] border shadow-[0_32px_80px_rgba(0,0,0,0.45)] overflow-hidden ${surface} ${className}`}
    >
      <div className={`flex items-center gap-3 px-5 py-3 border-b ${bar}`}>
        <div className="flex items-center gap-2">
          {dotColors.map((c) => (
            <span
              key={c}
              className="block h-[11px] w-[11px] rounded-full"
              style={{ background: c, opacity: 0.85 }}
            />
          ))}
        </div>
        {label ? (
          <span
            className="ml-3 text-[0.7rem] font-[family-name:var(--font-mono)] tracking-[0.1em] uppercase"
            style={{ color: isDark ? "rgba(199,195,187,0.7)" : "rgba(60,58,54,0.65)" }}
          >
            {label}
          </span>
        ) : null}
      </div>
      <div className="relative">{children}</div>
    </div>
  );
}
