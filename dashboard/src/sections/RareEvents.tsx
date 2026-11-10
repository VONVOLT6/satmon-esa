import { Bar, BarChart, CartesianGrid, LabelList, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { axisProps, ChartTooltip, COLORS, Explain, Panel, Section, Stat } from "../components/ui";
import { ci, contrib, pct, signed } from "../data/contributions";
import { fmt } from "../data/model";
import type { SectionProps } from "./types";

const { h6 } = contrib;

export default function RareEvents({ id, letter, title }: SectionProps) {
  const rows = [
    { name: "Rare events (of 36)", key: "rare" as const },
    { name: "Anomalies (of 29)", key: "anomaly" as const },
    { name: "False-alarm events", key: "false_alarms" as const },
  ].map(({ name, key }) => ({ name, "Always-on LSTM": h6.caught.B0[key], "With telecommand gates": h6.caught.R[key], "Gates alone, no LSTM": h6.gate_alone[key] }));
  const changes = [
    ["Rare events only", h6.test.rare],
    ["Anomalies only", h6.test.anomaly],
    ["All events (H6)", h6.test.all],
  ] as const;
  return (
    <Section id={id} letter={letter} title={title} subtitle="Rare events are mostly planned operations, and operations start with telecommands.">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="Rare events caught" value={`${h6.caught.B0.rare.toFixed(0)} → ${h6.caught.R.rare.toFixed(1)}`} sub="of 36 in the test period, mean of 5 seeds" tone="good" />
        <Stat label="F0.5, all events" value={`${fmt(h6.test.all.A)} → ${fmt(h6.test.all.B)}`} sub={`${signed(h6.test.all.diff)}, 95% CI ${ci(h6.test.all)}`} tone="warn" />
        <Stat label="False alarms added" value={`${h6.caught.B0.false_alarms.toFixed(0)} → ${h6.caught.R.false_alarms.toFixed(0)}`} sub={`anomalies-only F0.5 ${signed(h6.test.anomaly.diff)}`} />
        <Stat label="Time inside a gate" value={pct(h6.gate_share)} sub={`${h6.chosen.W / 3600} h after priority ≥ ${h6.chosen.p} commands`} tone="accent" />
      </div>
      <div className="grid gap-5 xl:grid-cols-2">
        <Panel title="What each system catches" eyebrow="Test period, mean of 5 seeds">
          <div className="h-[290px]">
            <ResponsiveContainer>
              <BarChart data={rows} margin={{ top: 20, right: 12, left: 0, bottom: 4 }}>
                <CartesianGrid vertical={false} stroke={COLORS.grid} />
                <XAxis dataKey="name" {...axisProps} />
                <YAxis {...axisProps} allowDecimals={false} />
                <Tooltip content={<ChartTooltip digits={1} />} />
                <Legend />
                <Bar dataKey="Always-on LSTM" fill={COLORS.paper} radius={[5, 5, 0, 0]} maxBarSize={44}><LabelList dataKey="Always-on LSTM" position="top" fill="#94a3b8" fontSize={11} formatter={(v: number) => +v.toFixed(1)} /></Bar>
                <Bar dataKey="With telecommand gates" fill={COLORS.ours} radius={[5, 5, 0, 0]} maxBarSize={44}><LabelList dataKey="With telecommand gates" position="top" fill={COLORS.ours} fontSize={11} formatter={(v: number) => +v.toFixed(1)} /></Bar>
                <Bar dataKey="Gates alone, no LSTM" fill="#475569" radius={[5, 5, 0, 0]} maxBarSize={44}><LabelList dataKey="Gates alone, no LSTM" position="top" fill="#94a3b8" fontSize={11} formatter={(v: number) => +v.toFixed(1)} /></Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <Explain
            shows={<>For {h6.chosen.W / 3600} hours after each high-priority telecommand, the LSTM may alarm at a lower error floor ({h6.chosen.f_r} instead of 0.007). The grey bars alarm on every gate with no LSTM at all, for comparison.</>}
            means="The gates more than double the rare events found, with no anomaly lost and one false alarm added. The gates alone find even more rare events but raise 15 false alarms, so telecommands carry the signal and the LSTM check inside the gate is the part to improve."
          />
        </Panel>
        <Panel title="Score change against the baseline" eyebrow="95% CI, paired bootstrap over test months and seeds">
          <div className="space-y-5 py-4">
            {changes.map(([name, b]) => {
              const lo = -0.15, hi = 0.55, x = (v: number) => `${((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * 100}%`;
              const sure = b.ci95[0] > 0;
              return (
                <div key={name}>
                  <div className="mb-1 flex justify-between text-sm"><span className="text-slate-200">{name}</span><span className="font-mono text-slate-300">{signed(b.diff)} {ci(b)}</span></div>
                  <div className="relative h-6 rounded bg-slate-900/60">
                    <div className="absolute inset-y-0 w-px bg-slate-400" style={{ left: x(0) }} />
                    <div className={`absolute top-1/2 h-1 -translate-y-1/2 rounded ${sure ? "bg-emerald-400" : "bg-amber-400"}`} style={{ left: x(b.ci95[0]), width: `calc(${x(b.ci95[1])} - ${x(b.ci95[0])})` }} />
                    <div className={`absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full ${sure ? "bg-emerald-300" : "bg-amber-300"}`} style={{ left: x(b.diff) }} />
                  </div>
                </div>
              );
            })}
            <div className="relative h-4 font-mono text-[11px] text-slate-500">
              {[-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5].map((v) => (
                <span key={v} className="absolute -translate-x-1/2" style={{ left: `${((v + 0.15) / 0.7) * 100}%` }}>{v === 0 ? "0" : signed(v, 1)}</span>
              ))}
            </div>
          </div>
          <Explain
            shows="How much each F0.5 score changed, with its 95% interval. The vertical line is zero, meaning no change."
            means={<>Rare-event detection improves clearly: its interval is above zero. The overall score rises too, but its interval crosses zero, so by our protocol the gain is not confirmed. Development data promised more ({fmt(h6.dev_B0)} → {fmt(h6.dev_chosen)}) than the test delivered.</>}
          />
        </Panel>
      </div>
    </Section>
  );
}
