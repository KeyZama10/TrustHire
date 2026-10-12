import clsx from "clsx";
import type { ReactNode } from "react";

export const riskColor = (r: number) => (r >= 0.65 ? "#ef4444" : r >= 0.4 ? "#f59e0b" : "#10b981");
export const trustColor = (t: number) => (t >= 70 ? "#10b981" : t >= 45 ? "#f59e0b" : "#ef4444");

export function Card({ title, right, children, className }: {
  title?: ReactNode; right?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={clsx("rounded-xl border border-slate-800 bg-slate-900/70 p-4", className)}>
      {(title || right) && (
        <header className="mb-3 flex items-center justify-between gap-2">
          <h2 className="text-sm font-semibold tracking-wide text-slate-300">{title}</h2>
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

export function Button({ children, onClick, disabled, variant = "primary", className, type = "button" }: {
  children: ReactNode; onClick?: () => void; disabled?: boolean;
  variant?: "primary" | "ghost" | "danger"; className?: string; type?: "button" | "submit";
}) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={clsx(
        "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg px-3.5 py-2 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-40",
        variant === "primary" && "bg-teal-600 text-white hover:bg-teal-500",
        variant === "ghost" && "border border-slate-700 text-slate-200 hover:bg-slate-800",
        variant === "danger" && "bg-red-600 text-white hover:bg-red-500",
        className,
      )}
    >
      {children}
    </button>
  );
}

export function Badge({ children, tone = "slate" }: { children: ReactNode; tone?: "slate" | "green" | "amber" | "red" | "teal" }) {
  const tones = {
    slate: "bg-slate-800 text-slate-300",
    green: "bg-emerald-900/60 text-emerald-300",
    amber: "bg-amber-900/60 text-amber-300",
    red: "bg-red-900/60 text-red-300",
    teal: "bg-teal-900/60 text-teal-300",
  };
  return <span className={clsx("rounded-full px-2 py-0.5 text-xs font-medium", tones[tone])}>{children}</span>;
}

export function TrustGauge({ value, size = 150 }: { value: number | null | undefined; size?: number }) {
  const v = value == null || Number.isNaN(value) ? null : value;
  const r = size / 2 - 12;
  const c = 2 * Math.PI * r * 0.75;
  const pct = v == null ? 0 : Math.max(0, Math.min(100, v)) / 100;
  const col = v == null ? "#475569" : trustColor(v);
  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="rotate-[135deg]">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#1e293b" strokeWidth={12}
          strokeDasharray={`${c} ${2 * Math.PI * r}`} strokeLinecap="round" />
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={col} strokeWidth={12}
          strokeDasharray={`${c * pct} ${2 * Math.PI * r}`} strokeLinecap="round"
          style={{ transition: "stroke-dasharray .6s ease, stroke .6s" }} />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-4xl font-bold tabular-nums" style={{ color: col }}>{v == null ? "—" : Math.round(v)}</span>
        <span className="text-xs uppercase tracking-wider text-slate-400">trust score</span>
      </div>
    </div>
  );
}

export function Bar({ value, color }: { value: number; color?: string }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
      <div className="h-full rounded-full transition-all duration-500"
        style={{ width: `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`, background: color ?? riskColor(value) }} />
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="py-6 text-center text-sm text-slate-500">{children}</p>;
}
