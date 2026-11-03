import { useState } from "react";
import { Bar, BarChart, CartesianGrid, LabelList, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { axisProps, ChartTooltip, COLORS, Explain, Panel, Section, Toggle } from "../components/ui";
import { DATASETS, fmt, label, MODELS, ours, paper, seedScore, SEEDS, type Dataset, type Model } from "../data/model";
import type { SectionProps } from "./types";

export default function Stability({ id, letter, title }: SectionProps) {
  const [model, setModel] = useState<Model>("Telemanom-ESA-Pruned");
  const yMax = model === "Telemanom-ESA-Pruned" ? 1 : 0.25;

  const rows = SEEDS.map((s) => ({
    name: `Seed ${s}`,
    "21 months": seedScore("21_months", s, model),
    "84 months": seedScore("84_months", s, model),
  }));
  const stat = (d: Dataset) => ours(d, model);

  return (
    <Section id={id} letter={letter} title={title} subtitle="Each setting was trained 5 times from different random starting points. A trustworthy result should not depend on one lucky run.">
      <Toggle value={model} onChange={setModel} options={[{ value: "Telemanom-ESA-Pruned", label: "Telemanom-ESA-Pruned" }, { value: "Telemanom-ESA", label: "Telemanom-ESA (unpruned)" }]} />

      <div className="grid gap-4 md:grid-cols-2">
        {DATASETS.map((d) => {
          const s = stat(d);
          return (
            <div key={d} className="glass p-4">
              <div className="eyebrow mb-2">{label(d)} · event-wise F0.5</div>
              <div className="grid grid-cols-4 gap-3 text-center">
                {[["Mean", s.mean], ["Min", s.min], ["Max", s.max], ["Std dev", s.sd]].map(([k, v]) => (
                  <div key={k as string}>
                    <div className="text-[11px] uppercase tracking-wider text-slate-400">{k}</div>
                    <div className={`font-mono text-lg font-semibold ${k === "Mean" ? "text-cyan-300" : "text-slate-100"}`}>{fmt(v as number)}</div>
                  </div>
                ))}
              </div>
              <div className="mt-2 text-center text-xs text-slate-400">Paper (single run): <span className="font-mono text-slate-300">{fmt(paper(d, model))}</span></div>
            </div>
          );
        })}
      </div>

      <Panel title={`${model}: event-wise F0.5 for every training run`} eyebrow="Dashed lines = published values">
        <div className="h-[340px]">
          <ResponsiveContainer>
            <BarChart data={rows} margin={{ top: 24, right: 120, left: 0, bottom: 4 }} barGap={6}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="name" {...axisProps} />
              <YAxis domain={[0, yMax]} {...axisProps} />
              <Tooltip content={<ChartTooltip />} />
              <Legend />
              <ReferenceLine y={paper("21_months", model)} stroke={COLORS.ours2} strokeDasharray="5 4" label={{ value: `paper 21 m: ${fmt(paper("21_months", model))}`, fill: "#93c5fd", fontSize: 11, position: "right", dy: model === "Telemanom-ESA-Pruned" ? 8 : 0 }} />
              <ReferenceLine y={paper("84_months", model)} stroke={COLORS.ours} strokeDasharray="5 4" label={{ value: `paper 84 m: ${fmt(paper("84_months", model))}`, fill: COLORS.ours, fontSize: 11, position: "right", dy: model === "Telemanom-ESA-Pruned" ? -8 : 0 }} />
              <Bar dataKey="21 months" fill={COLORS.ours2} radius={[5, 5, 0, 0]} maxBarSize={44}><LabelList dataKey="21 months" position="top" formatter={(v: number) => fmt(v)} fill="#bfdbfe" fontSize={10} /></Bar>
              <Bar dataKey="84 months" fill={COLORS.ours} radius={[5, 5, 0, 0]} maxBarSize={44}><LabelList dataKey="84 months" position="top" formatter={(v: number) => fmt(v)} fill={COLORS.ours} fontSize={10} /></Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <Explain
          shows="The main score of each of the 5 training runs (seeds), for both training lengths, with the paper's published values as dashed lines."
          means={model === "Telemanom-ESA-Pruned"
            ? <>Pruned is extremely stable: all 5 runs at 21 months score {fmt(stat("21_months").min)}, and at 84 months they range {fmt(stat("84_months").min)} to {fmt(stat("84_months").max)}. The 5 models make different predictions but catch the same events, so the 0.007 threshold, not the random start, decides the result. Two 84-month runs scored lower because each raised one false alarm.</>
            : <>Unpruned varies much more between runs: {fmt(stat("21_months").min)} to {fmt(stat("21_months").max)} at 21 months and {fmt(stat("84_months").min)} to {fmt(stat("84_months").max)} at 84 months. Comparisons with this variant need the mean and spread, not a single run.</>}
        />
      </Panel>

      <Panel title="Per-seed table" eyebrow="Event-wise F0.5 · anomalies and rare events">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[520px] text-sm">
            <thead>
              <tr className="text-left text-xs uppercase tracking-wider text-slate-400">
                <th className="pb-2 font-medium">Setting</th>
                {SEEDS.map((s) => <th key={s} className="pb-2 text-right font-medium">Seed {s}</th>)}
                <th className="pb-2 text-right font-medium text-cyan-300">Mean</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {DATASETS.flatMap((d) => MODELS.map((m) => (
                <tr key={`${d}${m}`} className="border-t border-cyan-300/10">
                  <td className="py-2 font-sans text-slate-200">{m.replace("Telemanom-", "")} · {label(d)}</td>
                  {SEEDS.map((s) => <td key={s} className="text-right text-slate-300">{fmt(seedScore(d, s, m))}</td>)}
                  <td className="text-right font-semibold text-cyan-300">{fmt(ours(d, m).mean)}</td>
                </tr>
              )))}
            </tbody>
          </table>
        </div>
      </Panel>
    </Section>
  );
}
