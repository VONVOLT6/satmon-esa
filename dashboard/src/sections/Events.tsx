import { Area, CartesianGrid, Cell, ComposedChart, Legend, Line, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AlertTriangle, CheckCircle2, Moon } from "lucide-react";
import { axisProps, ChartTooltip, COLORS, Explain, Panel, Section, Stat } from "../components/ui";
import { events, example } from "../data/model";
import type { SectionProps } from "./types";

function Donut({ title, detected, missed, color }: { title: string; detected: number; missed: number; color: string }) {
  const total = detected + missed;
  const pct = Math.round((detected / total) * 100);
  const data = [{ name: "Detected", value: detected }, { name: "Missed", value: missed }];
  return (
    <div className="flex flex-col items-center">
      <div className="relative h-[200px] w-[200px]">
        <ResponsiveContainer>
          <PieChart>
            <Pie data={data} dataKey="value" innerRadius={62} outerRadius={88} startAngle={90} endAngle={-270} stroke="none" paddingAngle={2}>
              <Cell fill={color} />
              <Cell fill="rgba(251,113,133,0.35)" />
            </Pie>
            <Tooltip content={<ChartTooltip digits={0} />} />
          </PieChart>
        </ResponsiveContainer>
        <div className="pointer-events-none absolute inset-0 grid place-items-center text-center">
          <div>
            <div className="font-mono text-3xl font-semibold text-white">{pct}%</div>
            <div className="text-[11px] text-slate-400">{detected} of {total} found</div>
          </div>
        </div>
      </div>
      <div className="mt-2 text-sm font-semibold text-slate-200">{title}</div>
    </div>
  );
}

export default function Events({ id, letter, title }: SectionProps) {
  const a = events.Anomaly;
  const r = events["Rare Event"];
  const points = example.points.map((p) => ({ ...p, band: p.labelled ? 1 : 0 }));
  const values = example.points.map((p) => p.actual);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const yDomain: [number, number] = [Math.floor(lo * 100) / 100, Math.ceil(hi * 100) / 100];
  const withBands = points.map((p) => ({ ...p, labelBand: p.labelled ? yDomain[1] : null, alarmBand: p.alarm ? yDomain[0] + (yDomain[1] - yDomain[0]) * 0.06 : null }));

  return (
    <Section id={id} letter={letter} title={title} subtitle="Event by event: what was caught and what was missed. Telemanom-ESA-Pruned, 21 months, seed 42.">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="Anomalies detected" value={`${a.detected} / ${a.events}`} tone="good" sub={`${a.missed} missed`} />
        <Stat label="Rare events detected" value={`${r.detected} / ${r.events}`} tone="warn" sub={`${r.missed} missed`} />
        <Stat label="Alarms raised" value={events.alarms_total} sub="across the 7 test years" />
        <Stat label="Alarms outside labelled events" value={events.alarms_on_no_event} tone="good" sub="no false alarms on nominal time" />
      </div>

      <div className="grid gap-5 xl:grid-cols-[1.1fr_1fr]">
        <Panel title="Detection rate by event type" eyebrow="Event-level · seed 42">
          <div className="flex flex-wrap justify-around gap-6 py-2">
            <Donut title="Anomalies" detected={a.detected} missed={a.missed} color={COLORS.good} />
            <Donut title="Rare nominal events" detected={r.detected} missed={r.missed} color={COLORS.warn} />
          </div>
          <Explain
            shows="The share of labelled events of each type that the model found during 2007-2013."
            means={<>The model catches {a.detected} of {a.events} real anomalies ({Math.round((a.detected / a.events) * 100)}%) but only {r.detected} of {r.events} rare nominal events. Rare events are unusual but harmless and often look normal to a forecaster. This is the clearest weakness to improve.</>}
          />
        </Panel>

        <Panel title="Event-level outcome table" eyebrow="What a confusion matrix looks like for events">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs uppercase tracking-wider text-slate-400">
                <th className="pb-2 font-medium">Labelled as</th>
                <th className="pb-2 text-right font-medium">Detected</th>
                <th className="pb-2 text-right font-medium">Missed</th>
                <th className="pb-2 text-right font-medium">Total</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              <tr className="border-t border-cyan-300/10"><td className="py-2.5 font-sans text-slate-200">Anomaly</td><td className="text-right text-emerald-300">{a.detected}</td><td className="text-right text-rose-300">{a.missed}</td><td className="text-right text-slate-300">{a.events}</td></tr>
              <tr className="border-t border-cyan-300/10"><td className="py-2.5 font-sans text-slate-200">Rare nominal event</td><td className="text-right text-emerald-300">{r.detected}</td><td className="text-right text-rose-300">{r.missed}</td><td className="text-right text-slate-300">{r.events}</td></tr>
              <tr className="border-t border-cyan-300/10"><td className="py-2.5 font-sans text-slate-200">No event (nominal time)</td><td className="text-right text-rose-300" title="alarms that hit no labelled event">{events.alarms_on_no_event} false alarms</td><td className="text-right text-slate-500">n/a</td><td className="text-right text-slate-500">n/a</td></tr>
            </tbody>
          </table>
          <div className="mt-5 grid gap-3 text-sm sm:grid-cols-2">
            <div className="rounded-xl border border-emerald-300/20 bg-emerald-400/5 p-3">
              <div className="mb-1 flex items-center gap-2 font-semibold text-emerald-200"><AlertTriangle className="h-4 w-4" />Anomaly</div>
              <p className="text-xs leading-relaxed text-slate-300">Something actually went wrong on the spacecraft or in a subsystem. These are what operators must not miss.</p>
            </div>
            <div className="rounded-xl border border-amber-300/20 bg-amber-400/5 p-3">
              <div className="mb-1 flex items-center gap-2 font-semibold text-amber-200"><Moon className="h-4 w-4" />Rare nominal event</div>
              <p className="text-xs leading-relaxed text-slate-300">Unusual but planned or harmless behaviour, such as a manoeuvre or a rare mode. ESA engineers labelled them, and the benchmark counts them as events too.</p>
            </div>
          </div>
          <Explain
            shows="A confusion-matrix style table built from whole events, not individual data points, because the official metric is event-wise."
            means={<>No alarm landed on nominal time, which is why precision is near 1.0. All of the model's mistakes are misses, mostly of rare events.</>}
          />
        </Panel>
      </div>

      <Panel title={`What a detection looks like: event ${example.event} on ${example.channel}`} eyebrow={`Real test data · ${example.start.slice(0, 10)} · hover to inspect`}>
        <div className="h-[340px]">
          <ResponsiveContainer>
            <ComposedChart data={withBands} margin={{ top: 8, right: 16, left: 4, bottom: 4 }}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="t" {...axisProps} minTickGap={70} tickFormatter={(t: string) => t.slice(5, 16).replace("T", " ")} />
              <YAxis domain={yDomain} {...axisProps} tickFormatter={(v: number) => v.toFixed(2)} width={48} />
              <Tooltip content={<ChartTooltip digits={4} />} />
              <Legend />
              <Area dataKey="labelBand" name="Labelled anomaly (ESA engineers)" fill="rgba(251,113,133,0.18)" stroke="none" baseValue={yDomain[0]} isAnimationActive={false} connectNulls={false} />
              <Area dataKey="alarmBand" name="Model alarm" fill="rgba(56,232,255,0.55)" stroke="none" baseValue={yDomain[0]} isAnimationActive={false} connectNulls={false} />
              <Line dataKey="actual" name="Actual telemetry" stroke="#e2e8f0" dot={false} strokeWidth={1.2} isAnimationActive={false} />
              <Line dataKey="forecast" name="LSTM forecast" stroke={COLORS.ours} dot={false} strokeWidth={1.2} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <Explain
          shows="Real telemetry from the test period (white) and the model's one-step forecast (cyan). The pink band is an anomaly labelled by ESA engineers, and the cyan strip at the bottom is when our model's alarm was on."
          means={<><CheckCircle2 className="mr-1 inline h-3.5 w-3.5 text-emerald-300" />Normally the forecast tracks the signal closely. When the channel suddenly drops and jumps, the forecast cannot follow, the error grows past the threshold and the alarm switches on during the labelled anomaly.</>}
        />
      </Panel>
    </Section>
  );
}
