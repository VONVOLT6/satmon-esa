import { Bar, BarChart, CartesianGrid, Cell, LabelList, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { axisProps, ChartTooltip, COLORS, Explain, Panel, Section, Stat } from "../components/ui";
import { contrib } from "../data/contributions";
import type { SectionProps } from "./types";

const { c4 } = contrib;
const POLICIES = [
  ["P0", "P0 page every alarm", COLORS.paper],
  ["P1", "P1 C2 sensor rule", COLORS.ours],
  ["P2", "P2 learned cost-aware", COLORS.warn],
] as const;

export default function ResponseC4({ id, letter, title }: SectionProps) {
  const costRows = POLICIES.map(([k, name, color]) => ({ name, color, cost: c4.central[k] }));
  const winRows = ["2", "5", "10"].map((d) => ({ name: `D = ${d}`, "P1 C2 sensor rule": c4.wins_by_D[d].P1, "P2 learned cost-aware": c4.wins_by_D[d].P2 }));
  const cls = c4.alarm_classes;
  return (
    <Section id={id} letter={letter} title={title} subtitle="For each alarm, choose: page an operator now, open a maintenance ticket, or ignore it. Replayed on the recorded timeline.">
      <div className="rounded-xl border border-amber-300/25 bg-amber-400/5 p-4 text-sm text-amber-100/90">
        ESA-ADB has no cost or response records, so this replays the alert policy only, over a declared grid of 36 cost settings: the cost of ignoring an anomaly (M),
        of delaying it (D), of a false page (F) and of a missed sensor fault (S). It is not a simulation of the spacecraft.
      </div>
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="Test alarms (5 seeds pooled)" value={Object.values(cls).reduce((a, b) => a + b, 0)} sub={`${cls.anomaly} anomaly · ${cls.rare} rare · ${cls.sensor} sensor · ${cls.false} false`} />
        <Stat label="Cost, page everything" value={c4.central.P0.toFixed(1)} sub="central setting M=50, D=5, F=1, S=1" />
        <Stat label="Cost, C2 sensor rule" value={c4.central.P1.toFixed(1)} sub={`ranked first at ${c4.ranked_first.P1} of ${c4.grid_points} settings`} tone="good" />
        <Stat label="Learned policy beats both" value={`${c4.P2_beats_both} of ${c4.grid_points}`} sub="protocol required ≥ 80%: not supported" tone="warn" />
      </div>
      <div className="grid gap-5 xl:grid-cols-2">
        <Panel title="Response cost at the central setting" eyebrow="Per test period, mean of 5 seeds · oracle = 0">
          <div className="h-[250px]">
            <ResponsiveContainer>
              <BarChart data={costRows} layout="vertical" margin={{ top: 4, right: 48, left: 20, bottom: 4 }}>
                <CartesianGrid horizontal={false} stroke={COLORS.grid} />
                <XAxis type="number" {...axisProps} />
                <YAxis type="category" dataKey="name" {...axisProps} width={160} />
                <Tooltip content={<ChartTooltip digits={1} />} />
                <Bar dataKey="cost" name="Cost" maxBarSize={32} radius={[0, 5, 5, 0]}>
                  {costRows.map((r) => <Cell key={r.name} fill={r.color} />)}
                  <LabelList dataKey="cost" position="right" formatter={(v: number) => v.toFixed(1)} fill="#cbd5e1" fontSize={11} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
        <Panel title="Which policy wins across the cost grid" eyebrow="Settings ranked first, of 12 per delay cost · P0 never wins">
          <div className="h-[250px]">
            <ResponsiveContainer>
              <BarChart data={winRows} margin={{ top: 16, right: 12, left: 0, bottom: 4 }}>
                <CartesianGrid vertical={false} stroke={COLORS.grid} />
                <XAxis dataKey="name" {...axisProps} />
                <YAxis {...axisProps} domain={[0, 12]} allowDecimals={false} />
                <Tooltip content={<ChartTooltip digits={0} />} />
                <Legend />
                <Bar dataKey="P1 C2 sensor rule" fill={COLORS.ours} radius={[5, 5, 0, 0]} maxBarSize={44}><LabelList dataKey="P1 C2 sensor rule" position="top" fill="#cbd5e1" fontSize={11} /></Bar>
                <Bar dataKey="P2 learned cost-aware" fill={COLORS.warn} radius={[5, 5, 0, 0]} maxBarSize={44}><LabelList dataKey="P2 learned cost-aware" position="top" fill="#cbd5e1" fontSize={11} /></Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>
      <Panel>
        <Explain
          shows="Left: what each policy would have cost on the faulty test period at the central cost setting. Right: across the grid, how often each policy is the cheapest, split by the cost of delaying a real anomaly (D)."
          means={<>The learned policy only wins when delay is cheap; it sometimes treats real anomalies as ticket-worthy, which is costly when delay matters. The simple rule "ticket it if C2 says sensor fault, otherwise page" is the most robust and cuts cost from {c4.central.P0.toFixed(1)} to {c4.central.P1.toFixed(1)}: a reliable sensor-fault label matters more than a learned cost model.</>}
        />
      </Panel>
    </Section>
  );
}
