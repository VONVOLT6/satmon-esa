import type { ReactNode } from "react";
import { Info } from "lucide-react";

export const COLORS = {
  ours: "#38e8ff",
  ours2: "#3b82f6",
  paper: "#7c8db5",
  unpruned: "#a78bfa",
  good: "#34d399",
  warn: "#fbbf24",
  bad: "#fb7185",
  grid: "rgba(148,197,255,0.08)",
  axis: "#8fa3c7",
};

export function Section({ id, letter, title, subtitle, children }: { id: string; letter: string; title: string; subtitle?: string; children: ReactNode }) {
  return (
    <section id={id} className="scroll-mt-32 fade-in lg:scroll-mt-20">
      <div className="mb-4 flex items-end gap-4">
        <span className="font-display text-3xl text-cyan-300/30">{letter}</span>
        <div>
          <h2 className="text-xl font-semibold text-white sm:text-2xl">{title}</h2>
          {subtitle && <p className="mt-1 text-sm text-slate-400">{subtitle}</p>}
        </div>
      </div>
      <div className="space-y-5">{children}</div>
    </section>
  );
}

export function Panel({ title, eyebrow, right, children, className = "" }: { title?: string; eyebrow?: string; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <div className={`glass p-4 sm:p-5 ${className}`}>
      {(title || eyebrow || right) && (
        <div className="mb-3 flex flex-wrap items-start justify-between gap-2">
          <div>
            {eyebrow && <div className="eyebrow mb-1">{eyebrow}</div>}
            {title && <h3 className="text-base font-semibold text-slate-100">{title}</h3>}
          </div>
          {right}
        </div>
      )}
      {children}
    </div>
  );
}

export function Explain({ shows, means }: { shows: ReactNode; means: ReactNode }) {
  return (
    <div className="mt-4 grid gap-3 border-t border-cyan-300/10 pt-4 text-sm leading-relaxed md:grid-cols-2">
      <div>
        <div className="mb-1 flex items-center gap-1.5 font-semibold text-cyan-200"><Info className="h-3.5 w-3.5" />What does this graph show?</div>
        <p className="text-slate-300">{shows}</p>
      </div>
      <div>
        <div className="mb-1 font-semibold text-emerald-300">What does the result mean?</div>
        <p className="text-slate-300">{means}</p>
      </div>
    </div>
  );
}

export function Stat({ label, value, sub, tone = "default" }: { label: string; value: ReactNode; sub?: ReactNode; tone?: "default" | "good" | "warn" | "accent" }) {
  const toneClass = { default: "text-white", good: "text-emerald-300", warn: "text-amber-300", accent: "text-cyan-300" }[tone];
  return (
    <div className="glass p-4">
      <div className="text-[11px] font-medium uppercase tracking-wider text-slate-400">{label}</div>
      <div className={`mt-1.5 font-mono text-2xl font-semibold ${toneClass}`}>{value}</div>
      {sub && <div className="mt-1 text-xs text-slate-400">{sub}</div>}
    </div>
  );
}

export function Chip({ children, tone = "cyan" }: { children: ReactNode; tone?: "cyan" | "good" | "warn" | "muted" | "violet" }) {
  const cls = {
    cyan: "border-cyan-300/30 bg-cyan-400/10 text-cyan-200",
    good: "border-emerald-300/30 bg-emerald-400/10 text-emerald-200",
    warn: "border-amber-300/30 bg-amber-400/10 text-amber-200",
    muted: "border-slate-500/30 bg-slate-500/10 text-slate-300",
    violet: "border-violet-300/30 bg-violet-400/10 text-violet-200",
  }[tone];
  return <span className={`chip ${cls}`}>{children}</span>;
}

export function Toggle<T extends string>({ value, options, onChange }: { value: T; options: { value: T; label: string }[]; onChange: (v: T) => void }) {
  return (
    <div className="inline-flex rounded-lg border border-cyan-300/10 bg-slate-900/60 p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          onClick={() => onChange(o.value)}
          className={`rounded-md px-2.5 py-1 text-xs font-medium transition ${value === o.value ? "bg-cyan-400/15 text-cyan-200" : "text-slate-400 hover:text-slate-200"}`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function ChartTooltip({ active, payload, label, digits = 3, unit = "" }: { active?: boolean; payload?: { name: string; value: number; color: string }[]; label?: string | number; digits?: number; unit?: string }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass-strong px-3 py-2 text-xs">
      {label !== undefined && <div className="mb-1 font-semibold text-slate-200">{label}</div>}
      {payload.map((p) => (
        <div key={p.name} className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full" style={{ background: p.color }} />
          <span className="text-slate-400">{p.name}</span>
          <span className="ml-auto font-mono text-slate-100">{typeof p.value === "number" ? p.value.toFixed(digits) : p.value}{unit}</span>
        </div>
      ))}
    </div>
  );
}

export const axisProps = { stroke: COLORS.axis, tick: { fill: COLORS.axis, fontSize: 12 }, tickLine: false, axisLine: { stroke: "rgba(148,197,255,0.15)" } };

export function Insight({ compared, indicates, improvement, why, tradeoffs }: { compared: ReactNode; indicates: ReactNode; improvement: ReactNode; why: ReactNode; tradeoffs: ReactNode }) {
  const rows: [string, ReactNode, string][] = [
    ["What is being compared?", compared, "text-cyan-200"],
    ["What does the graph indicate?", indicates, "text-cyan-200"],
    ["What did our approach improve?", improvement, "text-emerald-300"],
    ["Why it matters for satellites", why, "text-sky-300"],
    ["Trade-offs", tradeoffs, "text-amber-300"],
  ];
  return (
    <div className="mt-4 grid gap-x-6 gap-y-3 border-t border-cyan-300/10 pt-4 text-sm leading-relaxed md:grid-cols-2 xl:grid-cols-3">
      {rows.map(([q, a, tone]) => (
        <div key={q}>
          <div className={`mb-0.5 text-xs font-semibold uppercase tracking-wide ${tone}`}>{q}</div>
          <p className="text-slate-300">{a}</p>
        </div>
      ))}
    </div>
  );
}

export function Missing({ title, needed }: { title: string; needed: ReactNode }) {
  return (
    <div className="glass flex flex-col items-start gap-2 border-dashed p-5">
      <Chip tone="muted">Data not yet provided</Chip>
      <div className="text-base font-semibold text-slate-200">{title}</div>
      <p className="text-sm leading-relaxed text-slate-400">{needed}</p>
    </div>
  );
}
