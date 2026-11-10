import { Bar, BarChart, CartesianGrid, LabelList, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { axisProps, ChartTooltip, COLORS, Explain, Panel, Section, Stat } from "../components/ui";
import { ci, contrib, signed } from "../data/contributions";
import { fmt } from "../data/model";
import type { SectionProps } from "./types";

const { c2 } = contrib;
const KINDS = ["bias", "noise", "drift", "dropout", "stuck"];

export default function ReliabilityC2({ id, letter, title }: SectionProps) {
  const scoreRows = [
    { name: "With 50 injected sensor faults", "Always-on LSTM": c2.injected.A, "With C2": c2.injected.B },
    { name: "Clean test, no faults", "Always-on LSTM": c2.clean.A, "With C2": c2.clean.B },
  ];
  const faultRows = KINDS.map((k) => {
    const a = c2.attribution[k];
    return { name: k, "Attributed to the sensor": a.attributed, "Alarm, not attributed": a.alarmed - a.attributed, "No alarm at all": a.faults - a.alarmed };
  });
  return (
    <Section id={id} letter={letter} title={title} subtitle="Tell a broken sensor apart from a real spacecraft problem, using the five peer channels of the same group.">
      <div className="rounded-xl border border-amber-300/25 bg-amber-400/5 p-4 text-sm text-amber-100/90">
        ESA-ADB has no sensor-fault labels, so 50 single-channel faults (10 each of stuck, drift, bias, noise and dropout) were injected into each period. Claims hold for these injected faults only.
      </div>
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="F0.5 with injected faults" value={`${fmt(c2.injected.A)} → ${fmt(c2.injected.B)}`} sub={`${signed(c2.injected.diff)}, 95% CI ${ci(c2.injected)}`} tone="good" />
        <Stat label="F0.5 on clean data" value={`${fmt(c2.clean.A)} → ${fmt(c2.clean.B)}`} sub={`${signed(c2.clean.diff)}, CI ${ci(c2.clean)}: no harm`} />
        <Stat label="Development sensor-fault F1" value={fmt(c2.dev_sensor_f1)} sub={`decision threshold ${c2.threshold.toFixed(3)}`} tone="accent" />
        <Stat label="Stuck sensors raising an alarm" value={`${c2.attribution.stuck.alarmed} of ${c2.attribution.stuck.faults}`} sub="the weak spot: frozen values look normal" tone="warn" />
      </div>
      <div className="grid gap-5 xl:grid-cols-2">
        <Panel title="System-alarm score" eyebrow="Test period, scored against real events only, 5 seeds">
          <div className="h-[290px]">
            <ResponsiveContainer>
              <BarChart data={scoreRows} margin={{ top: 20, right: 12, left: 0, bottom: 4 }}>
                <CartesianGrid vertical={false} stroke={COLORS.grid} />
                <XAxis dataKey="name" {...axisProps} />
                <YAxis {...axisProps} domain={[0, 1]} />
                <Tooltip content={<ChartTooltip digits={3} />} />
                <Legend />
                <Bar dataKey="Always-on LSTM" fill={COLORS.paper} radius={[5, 5, 0, 0]} maxBarSize={64}><LabelList dataKey="Always-on LSTM" position="top" formatter={(v: number) => fmt(v)} fill="#94a3b8" fontSize={11} /></Bar>
                <Bar dataKey="With C2" fill={COLORS.ours} radius={[5, 5, 0, 0]} maxBarSize={64}><LabelList dataKey="With C2" position="top" formatter={(v: number) => fmt(v)} fill={COLORS.ours} fontSize={11} fontWeight={700} /></Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <Explain
            shows="The detection score for real events, with and without C2, on the faulty test period and on the untouched one."
            means="Without C2, every injected fault the LSTM notices becomes a false system alarm and the score halves. C2 recovers most of that, and on clean data it changes almost nothing (one real alarm removed in one seed)."
          />
        </Panel>
        <Panel title="What happened to each injected fault" eyebrow="Test period, 5 seeds × 10 faults per type">
          <div className="h-[290px]">
            <ResponsiveContainer>
              <BarChart data={faultRows} layout="vertical" margin={{ top: 8, right: 16, left: 10, bottom: 4 }}>
                <CartesianGrid horizontal={false} stroke={COLORS.grid} />
                <XAxis type="number" {...axisProps} domain={[0, 50]} />
                <YAxis type="category" dataKey="name" {...axisProps} width={64} />
                <Tooltip content={<ChartTooltip digits={0} />} />
                <Legend />
                <Bar dataKey="Attributed to the sensor" stackId="f" fill={COLORS.ours} maxBarSize={30} />
                <Bar dataKey="Alarm, not attributed" stackId="f" fill={COLORS.warn} maxBarSize={30} />
                <Bar dataKey="No alarm at all" stackId="f" fill="#334155" maxBarSize={30} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <Explain
            shows="For each fault type: whether the LSTM raised an alarm, and whether C2 then blamed the sensor."
            means="Bias, noise, drift and dropout faults are almost always caught and correctly blamed on the sensor. Stuck sensors often produce no alarm at all, because a frozen value can look normal."
          />
        </Panel>
      </div>
    </Section>
  );
}
