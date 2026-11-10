import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { axisProps, ChartTooltip, COLORS, Explain, Panel, Section, Stat } from "../components/ui";
import { ci, contrib, pct, signed } from "../data/contributions";
import { fmt } from "../data/model";
import type { SectionProps } from "./types";

const { c1 } = contrib;
const VARIANTS = [
  ["full", "Full screener", COLORS.ours],
  ["conf+impact", "Confidence + impact", COLORS.ours2],
  ["conf", "Confidence only", COLORS.unpruned],
] as const;

export default function AdaptiveC1({ id, letter, title }: SectionProps) {
  const t = c1.timing;
  const timeRows = [
    { name: "Always-on LSTM", "LSTM": t.full_lstm_seconds, "Cheap screener": 0 },
    { name: "C1 adaptive", "LSTM": t.routed_lstm_seconds, "Cheap screener": t.cheap_tier_seconds },
  ];
  return (
    <Section id={id} letter={letter} title={title} subtitle="A cheap screening tier watches every reading; the LSTM runs only in bursts around what it flags.">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="Test F0.5, C1 vs always-on" value={`${fmt(c1.test.B)} vs ${fmt(c1.test.A)}`} sub={`difference ${signed(c1.test.diff)}, 95% CI ${ci(c1.test)}`} tone="good" />
        <Stat label="LSTM share of test data" value={pct(c1.lstm_share)} sub="mean of 5 seeds, padding included" tone="accent" />
        <Stat label="Inference time, seed 42" value={`${t.full_lstm_seconds.toFixed(0)} s → ${(t.routed_lstm_seconds + t.cheap_tier_seconds).toFixed(0)} s`} sub={`${t.bursts} LSTM bursts`} />
        <Stat label="End-to-end speed-up" value={`${t.speedup_total.toFixed(1)}×`} sub="same events, same false alarms" tone="good" />
      </div>

      <div className="grid gap-5 xl:grid-cols-2">
        <Panel title="Detection score vs LSTM work" eyebrow="Development period, mean of 5 seeds">
          <div className="h-[290px]">
            <ResponsiveContainer>
              <LineChart margin={{ top: 12, right: 20, left: 0, bottom: 16 }}>
                <CartesianGrid stroke={COLORS.grid} />
                <XAxis type="number" dataKey="share" scale="log" domain={[0.01, 1.05]} ticks={[0.02, 0.05, 0.1, 0.2, 0.5, 1]}
                  tickFormatter={(v: number) => `${Math.round(v * 100)}%`} {...axisProps}
                  label={{ value: "share of data the LSTM runs on (log scale)", position: "insideBottom", offset: -10, fill: COLORS.axis, fontSize: 12 }} />
                <YAxis type="number" dataKey="f05" domain={[0.7, 0.8]} tickFormatter={(v: number) => v.toFixed(2)} {...axisProps} />
                <Tooltip content={<ChartTooltip digits={3} />} />
                <Legend verticalAlign="top" height={28} />
                <ReferenceLine y={c1.baseline_dev} stroke={COLORS.paper} strokeDasharray="5 4" label={{ value: "always-on LSTM", fill: "#94a3b8", fontSize: 11, position: "insideTopLeft" }} />
                {VARIANTS.map(([k, name, color]) => (
                  <Line key={k} name={name} data={c1.dev[k].map((p) => ({ share: p.share, f05: p.f05, tau: p.tau }))} dataKey="f05"
                    stroke={color} strokeWidth={2} dot={{ r: 4 }} isAnimationActive={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <Explain
            shows={<>Each point is one screening threshold τ (0.995 to 0.9999). Moving left, the screener sends less data to the LSTM.</>}
            means={<>Down to about 12% LSTM share the score stays at the always-on level. The protocol picked the cheapest such setting, τ = {c1.chosen}. Below that, detections start to drop.</>}
          />
        </Panel>

        <Panel title="Inference time on the 7-year test period" eyebrow="Seed 42, RTX 5060 laptop GPU">
          <div className="h-[290px]">
            <ResponsiveContainer>
              <BarChart data={timeRows} layout="vertical" margin={{ top: 8, right: 24, left: 20, bottom: 4 }}>
                <CartesianGrid horizontal={false} stroke={COLORS.grid} />
                <XAxis type="number" {...axisProps} unit=" s" />
                <YAxis type="category" dataKey="name" {...axisProps} width={110} />
                <Tooltip content={<ChartTooltip digits={0} unit=" s" />} />
                <Legend />
                <Bar dataKey="LSTM" stackId="t" fill={COLORS.ours} maxBarSize={42} />
                <Bar dataKey="Cheap screener" stackId="t" fill={COLORS.warn} maxBarSize={42} radius={[0, 5, 5, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <Explain
            shows="Wall-clock time to produce alarms for 2007-2013, measured in one session for both systems."
            means={<>The routed system needs {t.routed_lstm_seconds.toFixed(0)} s of LSTM time plus {t.cheap_tier_seconds.toFixed(0)} s of screening, against {t.full_lstm_seconds.toFixed(0)} s. It catches exactly the same events on every seed, so the saving is free.</>}
          />
        </Panel>
      </div>
    </Section>
  );
}
