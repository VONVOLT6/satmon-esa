import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, LabelList, PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { axisProps, ChartTooltip, COLORS, Explain, Panel, Section, Toggle } from "../components/ui";
import { fmt, label, METRICS, ours, type Dataset, type Model, type Subset } from "../data/model";
import type { SectionProps } from "./types";

export default function Metrics({ id, letter, title }: SectionProps) {
  const [dataset, setDataset] = useState<Dataset>("21_months");
  const [model, setModel] = useState<Model>("Telemanom-ESA-Pruned");
  const [subset, setSubset] = useState<Subset>("all");

  const rows = METRICS.map((m) => {
    const s = ours(dataset, model, m.key, subset);
    return { name: m.short, label: m.label, value: s.mean, min: s.min, max: s.max, help: m.help };
  });
  const radar = METRICS.map((m) => ({
    metric: m.short,
    ESA: ours(dataset, "Telemanom-ESA", m.key, subset).mean,
    Pruned: ours(dataset, "Telemanom-ESA-Pruned", m.key, subset).mean,
  }));
  const isDefault = dataset === "21_months" && model === "Telemanom-ESA-Pruned" && subset === "all";

  return (
    <Section id={id} letter={letter} title={title} subtitle="All five official ESA-ADB metrics. Use the toggles to switch training length, variant and which events count.">
      <div className="flex flex-wrap gap-2">
        <Toggle value={dataset} onChange={setDataset} options={[{ value: "21_months", label: "21 months" }, { value: "84_months", label: "84 months" }]} />
        <Toggle value={model} onChange={setModel} options={[{ value: "Telemanom-ESA-Pruned", label: "Pruned" }, { value: "Telemanom-ESA", label: "Unpruned" }]} />
        <Toggle value={subset} onChange={setSubset} options={[{ value: "all", label: "Anomalies + rare events" }, { value: "anomaly", label: "Anomalies only" }]} />
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
        {rows.map((r) => (
          <div key={r.name} className="glass p-4">
            <div className="text-[11px] font-medium uppercase tracking-wider text-slate-400">{r.label}</div>
            <div className="mt-1.5 font-mono text-2xl font-semibold text-cyan-300">{fmt(r.value)}</div>
            <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-800"><div className="h-full rounded-full bg-gradient-to-r from-blue-500 to-cyan-300" style={{ width: `${r.value * 100}%` }} /></div>
            <p className="mt-2 text-[11px] leading-snug text-slate-400">{r.help}</p>
          </div>
        ))}
      </div>

      <div className="grid gap-5 xl:grid-cols-[1.3fr_1fr]">
        <Panel title={`${model}, ${label(dataset)}`} eyebrow={subset === "all" ? "Anomalies and rare events · mean of 5 seeds" : "Anomalies only · mean of 5 seeds"}>
          <div className="h-[300px]">
            <ResponsiveContainer>
              <BarChart data={rows} margin={{ top: 24, right: 12, left: 0, bottom: 4 }}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="name" {...axisProps} interval={0} tick={{ fill: COLORS.axis, fontSize: 11 }} />
                <YAxis domain={[0, 1]} {...axisProps} />
                <Tooltip content={<ChartTooltip />} />
                <Bar dataKey="value" name="Score" radius={[6, 6, 0, 0]} maxBarSize={70}>
                  {rows.map((r) => <Cell key={r.name} fill={r.name === "F0.5" ? COLORS.ours : COLORS.ours2} />)}
                  <LabelList dataKey="value" position="top" formatter={(v: number) => fmt(v)} fill="#e2f8ff" fontSize={12} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <Explain
            shows="Each official metric for the selected setting, averaged over the 5 training runs. The F0.5 bar (bright) is the main score."
            means={isDefault
              ? <>For the 21-month Pruned model, precision is almost perfect ({fmt(rows[0].value)}): nearly every alarm is real. Recall is {fmt(rows[1].value)}, so about half of all events are missed. Most of those are harmless rare events, as Section E shows. Switch to "Anomalies only" to see recall rise to {fmt(ours("21_months", "Telemanom-ESA-Pruned", "EW_recall", "anomaly").mean)}.</>
              : <>Event-wise F0.5 here is {fmt(rows[2].value)}, with precision {fmt(rows[0].value)} and recall {fmt(rows[1].value)}. Compare with the default view (21 months, Pruned, all events) to see the effect of each switch.</>}
          />
        </Panel>

        <Panel title="Metric profile: Unpruned vs Pruned" eyebrow={`${label(dataset)} · ${subset === "all" ? "all events" : "anomalies only"}`}>
          <div className="h-[300px]">
            <ResponsiveContainer>
              <RadarChart data={radar} outerRadius="72%">
                <PolarGrid stroke="rgba(148,197,255,0.15)" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: COLORS.axis, fontSize: 11 }} />
                <PolarRadiusAxis domain={[0, 1]} tick={false} axisLine={false} />
                <Radar name="Unpruned" dataKey="ESA" stroke={COLORS.unpruned} fill={COLORS.unpruned} fillOpacity={0.18} />
                <Radar name="Pruned" dataKey="Pruned" stroke={COLORS.ours} fill={COLORS.ours} fillOpacity={0.25} />
                <Tooltip content={<ChartTooltip />} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
          <Explain
            shows="The shape of each variant's results across all five metrics. A bigger area means better overall."
            means={<>Pruned leads on precision ({fmt(radar[0].Pruned)} vs {fmt(radar[0].ESA)}) and on the main F0.5 score ({fmt(radar[2].Pruned)} vs {fmt(radar[2].ESA)}). Unpruned keeps higher recall ({fmt(radar[1].ESA)} vs {fmt(radar[1].Pruned)}), because it alarms on almost everything.</>}
          />
        </Panel>
      </div>
    </Section>
  );
}
