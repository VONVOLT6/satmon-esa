import { Bar, BarChart, CartesianGrid, Cell, LabelList, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { axisProps, ChartTooltip, COLORS, Explain, Panel, Section } from "../components/ui";
import { alarmBreakdown, DATASETS, fmt, label, MODELS, ours, paper, paper84Detail, type Dataset, type Model } from "../data/model";
import type { SectionProps } from "./types";

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;

export default function BaselineVsPruned({ id, letter, title }: SectionProps) {
  const scoreRows = DATASETS.map((d) => ({
    name: label(d),
    "ESA · ours": ours(d, "Telemanom-ESA").mean,
    "ESA · paper": paper(d, "Telemanom-ESA"),
    "Pruned · ours": ours(d, "Telemanom-ESA-Pruned").mean,
    "Pruned · paper": paper(d, "Telemanom-ESA-Pruned"),
  }));

  const alarmsFor = (d: Dataset, m: Model) => {
    const rows = alarmBreakdown.filter((r) => r.dataset === d && r.model === m);
    return { alarms: mean(rows.map((r) => r.alarms)), falseAlarms: mean(rows.map((r) => r.false_alarms)), found: mean(rows.map((r) => r.events_found)) };
  };
  const alarmRows = DATASETS.flatMap((d) =>
    MODELS.map((m) => {
      const a = alarmsFor(d, m);
      return { name: `${m === "Telemanom-ESA" ? "ESA" : "Pruned"} · ${label(d)}`, model: m, "Alarms on a labelled event": a.alarms - a.falseAlarms, "False alarms": a.falseAlarms, found: a.found };
    }),
  );
  const esa21 = alarmsFor("21_months", "Telemanom-ESA");
  const pr21 = alarmsFor("21_months", "Telemanom-ESA-Pruned");

  const prRows = DATASETS.flatMap((d) =>
    MODELS.map((m) => ({ name: `${m === "Telemanom-ESA" ? "ESA" : "Pruned"} · ${label(d)}`, Precision: ours(d, m, "EW_precision").mean, Recall: ours(d, m, "EW_recall").mean })),
  );

  return (
    <Section id={id} letter={letter} title={title} subtitle="One trained model, two alarm rules: with and without a minimum error size of 0.007.">
      <Panel title="Event-wise F0.5: Telemanom-ESA vs Telemanom-ESA-Pruned" eyebrow="Ours (mean of 5 seeds) and published">
        <div className="h-[320px]">
          <ResponsiveContainer>
            <BarChart data={scoreRows} margin={{ top: 24, right: 16, left: 0, bottom: 4 }} barGap={4}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="name" {...axisProps} tickFormatter={(v) => `Trained on ${v}`} />
              <YAxis domain={[0, 1]} {...axisProps} />
              <Tooltip content={<ChartTooltip />} />
              <Legend />
              <Bar dataKey="ESA · paper" fill={COLORS.paper} fillOpacity={0.55} radius={[5, 5, 0, 0]} maxBarSize={56}><LabelList dataKey="ESA · paper" position="top" formatter={(v: number) => fmt(v)} fill="#94a3b8" fontSize={11} /></Bar>
              <Bar dataKey="ESA · ours" fill={COLORS.unpruned} radius={[5, 5, 0, 0]} maxBarSize={56}><LabelList dataKey="ESA · ours" position="top" formatter={(v: number) => fmt(v)} fill="#c4b5fd" fontSize={11} /></Bar>
              <Bar dataKey="Pruned · paper" fill={COLORS.paper} radius={[5, 5, 0, 0]} maxBarSize={56}><LabelList dataKey="Pruned · paper" position="top" formatter={(v: number) => fmt(v)} fill="#94a3b8" fontSize={11} /></Bar>
              <Bar dataKey="Pruned · ours" fill={COLORS.ours} radius={[5, 5, 0, 0]} maxBarSize={56}><LabelList dataKey="Pruned · ours" position="top" formatter={(v: number) => fmt(v)} fill={COLORS.ours} fontSize={11} fontWeight={700} /></Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <Explain
          shows="The main score for both alarm rules at both training lengths, ours next to the paper's."
          means={<>Pruning lifts the score from about {fmt(ours("21_months", "Telemanom-ESA").mean, 2)} to {fmt(ours("21_months", "Telemanom-ESA-Pruned").mean, 2)}. Our Pruned results match the paper. Our unpruned results are lower than the paper's ({fmt(ours("84_months", "Telemanom-ESA").mean)} vs {fmt(paper("84_months", "Telemanom-ESA"))} at 84 months). That is a known open issue, explained below.</>}
        />
      </Panel>

      <div className="grid gap-5 xl:grid-cols-2">
        <Panel title="Why: how many alarms each rule raises" eyebrow="Mean of 5 seeds, test 2007-2013">
          <div className="h-[300px]">
            <ResponsiveContainer>
              <BarChart data={alarmRows} layout="vertical" margin={{ top: 4, right: 24, left: 8, bottom: 4 }}>
                <CartesianGrid horizontal={false} />
                <XAxis type="number" {...axisProps} />
                <YAxis type="category" dataKey="name" width={118} {...axisProps} />
                <Tooltip content={<ChartTooltip digits={0} />} />
                <Legend />
                <Bar dataKey="Alarms on a labelled event" stackId="a" fill={COLORS.good} />
                <Bar dataKey="False alarms" stackId="a" fill={COLORS.bad} radius={[0, 5, 5, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <Explain
            shows="Every alarm the detector raised in the 7 test years, split into alarms that overlapped a labelled event (green) and alarms that hit nothing (red)."
            means={<>Without pruning, the model raises about {Math.round(esa21.alarms)} alarms at 21 months, of which {Math.round(esa21.falseAlarms)} are false. With the 0.007 minimum error it raises {Math.round(pr21.alarms)}, with {Math.round(pr21.falseAlarms)} false. Small forecast wobbles no longer trigger alarms, which is why precision, and the F0.5 score, jump.</>}
          />
        </Panel>

        <Panel title="The trade-off: precision vs recall" eyebrow="Mean of 5 seeds">
          <div className="h-[300px]">
            <ResponsiveContainer>
              <BarChart data={prRows} margin={{ top: 4, right: 16, left: 0, bottom: 4 }} barGap={3}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="name" {...axisProps} interval={0} tick={{ fill: COLORS.axis, fontSize: 11 }} />
                <YAxis domain={[0, 1]} {...axisProps} />
                <Tooltip content={<ChartTooltip />} />
                <Legend />
                <Bar dataKey="Precision" fill={COLORS.ours} radius={[4, 4, 0, 0]} maxBarSize={34}>
                  {prRows.map((r) => <Cell key={r.name} fillOpacity={r.name.startsWith("ESA") ? 0.55 : 1} />)}
                </Bar>
                <Bar dataKey="Recall" fill={COLORS.warn} radius={[4, 4, 0, 0]} maxBarSize={34}>
                  {prRows.map((r) => <Cell key={r.name} fillOpacity={r.name.startsWith("ESA") ? 0.55 : 1} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <Explain
            shows="Precision (how many alarms were real) and recall (how many real events were found) for each variant."
            means={<>The minimum error threshold trades recall for precision. Unpruned finds about {fmt(ours("21_months", "Telemanom-ESA", "EW_recall").mean, 2)} of events but most alarms are false. Pruned is almost never wrong ({fmt(ours("21_months", "Telemanom-ESA-Pruned", "EW_precision").mean)}) but finds {fmt(ours("21_months", "Telemanom-ESA-Pruned", "EW_recall").mean, 2)}. Because F0.5 weights precision more, Pruned wins.</>}
          />
        </Panel>
      </div>

      <Panel eyebrow="Open issue" title="Why our unpruned score is below the paper's">
        <p className="text-sm leading-relaxed text-slate-300">
          At 84 months the paper reports precision {fmt(paper84Detail["Telemanom-ESA"].EW_precision)} and recall {fmt(paper84Detail["Telemanom-ESA"].EW_recall)} for
          Telemanom-ESA. We get the same recall ({fmt(ours("84_months", "Telemanom-ESA", "EW_recall").mean)}) but lower precision ({fmt(ours("84_months", "Telemanom-ESA", "EW_precision").mean)}).
          So we find the same events but raise more short false alarms. The likely causes are small numerical differences between our PyTorch
          port and the original TensorFlow code, and run-to-run variation: our 5 runs span {fmt(ours("84_months", "Telemanom-ESA").min)} to {fmt(ours("84_months", "Telemanom-ESA").max)}, while the paper reports a single run.
          Tuning the minimum error on a labelled development period, never the test set, is in progress as our first improvement.
        </p>
      </Panel>
    </Section>
  );
}
