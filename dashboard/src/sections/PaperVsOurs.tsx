import { Bar, BarChart, CartesianGrid, ErrorBar, LabelList, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { axisProps, ChartTooltip, Chip, COLORS, Explain, Panel, Section, Stat } from "../components/ui";
import { DATASETS, fmt, label, ours, paper, signed, verdict } from "../data/model";
import type { SectionProps } from "./types";

const MODEL = "Telemanom-ESA-Pruned" as const;

export default function PaperVsOurs({ id, letter, title }: SectionProps) {
  const rows = DATASETS.map((d) => {
    const s = ours(d, MODEL);
    return { name: `Trained on ${label(d)}`, dataset: d, paper: paper(d, MODEL), ours: s.mean, range: [s.mean - s.min, s.max - s.mean], min: s.min, max: s.max };
  });
  const [r21, r84] = rows;
  const d21 = r21.ours - r21.paper;
  const d84 = r84.ours - r84.paper;

  return (
    <Section id={id} letter={letter} title={title} subtitle={`${MODEL}, corrected event-wise F0.5, anomalies and rare events, mean of 5 seeds.`}>
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label={`Published · ${label("21_months")}`} value={fmt(r21.paper)} sub="Kotowski et al. 2024, Supp. Table 13" />
        <Stat label={`Ours · ${label("21_months")}`} value={fmt(r21.ours)} tone="accent" sub={<span>Difference <b className="text-emerald-300">{signed(d21)}</b> · <Chip tone="good">{verdict(d21).text}</Chip></span>} />
        <Stat label={`Published · ${label("84_months")}`} value={fmt(r84.paper)} sub="Kotowski et al. 2024, Supp. Table 13" />
        <Stat label={`Ours · ${label("84_months")}`} value={fmt(r84.ours)} tone="accent" sub={<span>Difference <b className="text-emerald-300">{signed(d84)}</b> · <Chip tone="good">{verdict(d84).text}</Chip></span>} />
      </div>

      <Panel title="Published result vs our reproduction" eyebrow="Interactive · hover for values">
        <div className="h-[340px]">
          <ResponsiveContainer>
            <BarChart data={rows} margin={{ top: 28, right: 16, left: 0, bottom: 4 }} barGap={10}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="name" {...axisProps} />
              <YAxis domain={[0, 1]} {...axisProps} label={{ value: "Event-wise F0.5", angle: -90, position: "insideLeft", fill: COLORS.axis, fontSize: 12 }} />
              <Tooltip content={<ChartTooltip />} />
              <Legend wrapperStyle={{ paddingTop: 8 }} />
              <Bar dataKey="paper" name="Published paper" fill={COLORS.paper} radius={[6, 6, 0, 0]} maxBarSize={90}>
                <LabelList dataKey="paper" position="top" formatter={(v: number) => fmt(v)} fill="#cbd5e1" fontSize={12} />
              </Bar>
              <Bar dataKey="ours" name="Our reproduction (mean, bars = min to max)" fill={COLORS.ours} radius={[6, 6, 0, 0]} maxBarSize={90}>
                <ErrorBar dataKey="range" width={8} stroke="#e2f8ff" strokeWidth={1.5} />
                <LabelList dataKey="ours" position="top" offset={14} formatter={(v: number) => fmt(v)} fill={COLORS.ours} fontSize={12} fontWeight={700} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <Explain
          shows={<>The paper's published score (grey) next to our score (cyan) for the same model, data and official metric, for both training lengths. The thin bars show the range across our 5 training runs.</>}
          means={<>At 21 months we scored {fmt(r21.ours)} against the published {fmt(r21.paper)} ({signed(d21)}). At 84 months we scored {fmt(r84.ours)} against {fmt(r84.paper)} ({signed(d84)}). Both are inside our ±0.03 acceptance band, so we successfully reproduced the paper's best detector, and slightly exceeded it at 21 months.</>}
        />
      </Panel>
    </Section>
  );
}
