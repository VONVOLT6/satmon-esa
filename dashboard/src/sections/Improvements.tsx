import type { ReactNode } from "react";
import { Activity, ArrowRight, Gauge, Layers, Scale, Target, TrendingUp } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, LabelList, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { axisProps, ChartTooltip, Chip, COLORS, Insight, Panel, Section } from "../components/ui";
import { change, MODEL_COLOR, MODEL_LABEL, n0, pct, RF, rf, RF_METRICS, TEAM, type ModelKey } from "../data/channel9";
import type { SectionProps } from "./types";

interface Card {
  title: string;
  from: string;
  to: string;
  delta: string;
  good: boolean;
  compare: string;
  team?: boolean;
  original: ReactNode;
  added: ReactNode;
  result: ReactNode;
  why: ReactNode;
}

function ImprovementCard({ c }: { c: Card }) {
  return (
    <div className="glass flex flex-col p-4">
      <div className="flex items-start justify-between gap-2">
        <div className="text-[11px] font-medium uppercase tracking-wider text-slate-400">{c.title}</div>
        {c.team && <Chip tone="violet">team-reported</Chip>}
      </div>
      <div className="mt-2 flex items-center gap-2 font-mono">
        <span className="text-lg text-slate-400">{c.from}</span>
        <ArrowRight className="h-4 w-4 text-slate-500" />
        <span className="text-2xl font-semibold text-cyan-300">{c.to}</span>
      </div>
      <div className="mt-1 flex items-center gap-2 text-xs">
        <span className={`rounded-full px-2 py-0.5 font-mono ${c.good ? "bg-emerald-400/10 text-emerald-300" : "bg-amber-400/10 text-amber-300"}`}>{c.delta}</span>
        <span className="text-slate-500">{c.compare}</span>
      </div>
      <dl className="mt-4 space-y-2 text-[13px] leading-relaxed">
        {([["Original approach", c.original], ["What we added", c.added], ["Experimental result", c.result], ["Why it matters", c.why]] as const).map(([k, v]) => (
          <div key={k}>
            <dt className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">{k}</dt>
            <dd className="text-slate-300">{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

const CONTRIBUTIONS = [
  { icon: Activity, title: "Temporal feature enhancement", model: "10" as ModelKey, text: "Adds deviation from the 30- and 120-observation rolling means and a variability ratio, so the model sees how far a reading strays from its recent behaviour." },
  { icon: TrendingUp, title: "Drift-aware feature enhancement", model: "13" as ModelKey, text: "Adds long-term slope, the gap between short and long averages, and cumulative deviation, so slow degradation that builds over time becomes visible." },
  { icon: Gauge, title: "Threshold calibration", model: undefined, text: "Each model's alarm threshold is chosen on validation data to maximise F1 while keeping false positives at or below 1%, then applied unchanged to the test set." },
  { icon: Target, title: "Incident-level evaluation", model: undefined, text: "Scores whole incidents as well as single observations, which matches how operators work: one alarm per real incident is what matters." },
];

export default function Improvements({ id, letter, title }: SectionProps) {
  const P = (k: ModelKey) => `${(rf(k).precision * 100).toFixed(2)}%`;
  const inc = TEAM.incident_detection;
  const size = TEAM.model_size_mb;
  const cP = change("precision", "13");
  const cF = change("f1", "13");
  const cA = change("pr_auc", "13");
  const cR = change("fpr", "13");
  const sizePct = ((size["13"] - size["7"]) / size["7"]) * 100;

  const cards: Card[] = [
    { title: "Precision", from: P("7"), to: P("13"), delta: `+${((rf("13").precision - rf("7").precision) * 100).toFixed(2)} pts (${pct(cP.pct)})`, good: true, compare: "7 → 13 features",
      original: "7 basic features: current value, change, and two rolling means and spreads.",
      added: "Temporal deviation features (10) and drift-aware features (13).",
      result: <>{P("7")} of alarms correct → {P("13")}.</>,
      why: "More of the alarms operators see are real, so less time is lost checking false ones." },
    { title: "F1-score", from: rf("7").f1.toFixed(4), to: rf("13").f1.toFixed(4), delta: `${pct(cF.pct)}`, good: true, compare: "7 → 13 features",
      original: "Balance of precision and recall with 7 features.",
      added: "The 6 extra temporal and drift features.",
      result: <>A small rise, {rf("7").f1.toFixed(4)} → {rf("13").f1.toFixed(4)}, driven mostly by precision.</>,
      why: "F1 summarises catching anomalies and avoiding false alarms in one number." },
    { title: "PR-AUC", from: rf("7").pr_auc.toFixed(4), to: rf("13").pr_auc.toFixed(4), delta: `${pct(cA.pct)}`, good: true, compare: "7 → 13 features",
      original: "How well 7 features rank anomalous above normal readings, across all thresholds.",
      added: "Features describing deviation and long-term drift.",
      result: <>The largest relative gain: {rf("7").pr_auc.toFixed(4)} → {rf("13").pr_auc.toFixed(4)}.</>,
      why: "A better ranking gives operators more room to pick a threshold that suits their workload." },
    { title: "False-positive rate", from: rf("7").fpr.toFixed(5), to: rf("13").fpr.toFixed(5), delta: `${pct(cR.pct)}`, good: true, compare: "7 → 13 features (shown as 0.0017 → 0.0015 at 4 decimals)",
      original: <>{n0(rf("7").fp)} normal test observations wrongly flagged.</>,
      added: "Temporal and drift features, with the threshold calibrated on validation.",
      result: <>{n0(rf("13").fp)} wrongly flagged: {n0(rf("7").fp - rf("13").fp)} fewer false alarms.</>,
      why: "False alarms cause alarm fatigue. Cutting them keeps operators responsive to real problems." },
    { title: "Incident detection", from: `${inc["10"].detected}/${inc.total_incidents}`, to: `${inc["13"].detected}/${inc.total_incidents}`, delta: `+${inc["13"].detected - inc["10"].detected} incidents`, good: true, compare: "10 → 13 features", team: true,
      original: "Incident-level results were not reported for the 7-feature original. The comparison here is 10 vs 13 features.",
      added: "Drift-aware features on top of the temporal ones.",
      result: <>All {inc.total_incidents} incidents detected, against {inc["10"].detected} with 10 features.</>,
      why: "Operators act on incidents. Missing a whole incident matters far more than missing a few readings inside one." },
    { title: "Model size", from: `${size["7"].toFixed(2)} MB`, to: `${size["13"].toFixed(2)} MB`, delta: `${pct(sizePct)}`, good: true, compare: "7 → 13 features", team: true,
      original: <>The saved 7-feature model takes {size["7"]} MB.</>,
      added: "The 6 extra temporal and drift features (13 inputs in total).",
      result: <>{size["13"]} MB: smaller despite having more input features.</>,
      why: "Smaller models are easier to deploy on limited on-board or edge hardware." },
  ];

  const chart = RF_METRICS.map((m) => {
    const sign = m.higherIsBetter ? 1 : -1;
    return { name: m.higherIsBetter ? m.label : "False-positive rate reduction", "10 features": sign * change(m.key, "10").pct, "13 features": sign * change(m.key, "13").pct };
  });

  return (
    <Section id={id} letter={letter} title={title} subtitle="Measured gains of our proposed feature sets over the original detector (Mission 2, channel 9, sample-level, one run).">
      <div className="grid gap-3 md:grid-cols-3">
        {(["7", "10", "13"] as ModelKey[]).map((k) => (
          <div key={k} className="glass flex items-center gap-3 p-3">
            <span className="h-9 w-1.5 rounded-full" style={{ background: MODEL_COLOR[k] }} />
            <div>
              <div className="text-sm font-semibold text-slate-100">{MODEL_LABEL[k]}{k === "7" && <span className="ml-2 text-xs font-normal text-slate-400">baseline</span>}{k === "13" && <span className="ml-2 text-xs font-normal text-emerald-300">proposed</span>}</div>
              <div className="text-xs text-slate-400">{rf(k).feature_names.length} inputs · threshold {rf(k).threshold.toFixed(4)}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">{cards.map((c) => <ImprovementCard key={c.title} c={c} />)}</div>

      <Panel title="Improvement over the original model, metric by metric" eyebrow="Percent change vs 7 features · positive = better">
        <div className="h-[320px]">
          <ResponsiveContainer>
            <BarChart data={chart} margin={{ top: 24, right: 12, left: 0, bottom: 4 }} barGap={4}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="name" {...axisProps} interval={0} tick={{ fill: COLORS.axis, fontSize: 11 }} />
              <YAxis {...axisProps} unit="%" label={{ value: "Change vs 7 features", angle: -90, position: "insideLeft", fill: COLORS.axis, fontSize: 12 }} />
              <ReferenceLine y={0} stroke="rgba(148,197,255,0.35)" />
              <Tooltip content={<ChartTooltip digits={2} unit="%" />} />
              <Legend />
              {(["10", "13"] as ModelKey[]).map((k) => (
                <Bar key={k} dataKey={`${k} features`} fill={MODEL_COLOR[k]} radius={[5, 5, 0, 0]} maxBarSize={48}>
                  {chart.map((r) => <Cell key={r.name} fillOpacity={(r[`${k} features` as "10 features"] as number) < 0 ? 0.45 : 1} />)}
                  <LabelList dataKey={`${k} features`} position="top" formatter={(v: number) => pct(v)} fill="#cbd5e1" fontSize={10} />
                </Bar>
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
        <Insight
          compared="The percentage change of each score for the 10 and 13-feature models, relative to the original 7-feature model. False-positive rate is flipped so that a taller bar always means better."
          indicates={<>The 13-feature model improves every score. The biggest relative gains are PR-AUC ({pct(cA.pct)}) and fewer false positives ({pct(-cR.pct)}). The 10-feature model does not improve recall ({pct(change("recall", "10").pct, 2)}).</>}
          improvement={<>{n0(rf("7").fp - rf("13").fp)} fewer false alarms and {n0(rf("13").tp - rf("7").tp)} more anomalous observations caught on the same test data.</>}
          why="Drift and trend features help the model recognise slow degradation, which short-window features can miss. Satellite faults often develop this way."
          tradeoffs={<>Recall barely moves ({pct(change("recall", "13").pct)}) and stays near {Math.round(rf("13").recall * 100)}%. Absolute gains are small, and this is one run on one channel with no confidence intervals.</>}
        />
      </Panel>

      <Panel title="Research contributions behind these results" eyebrow="What was added to the original pipeline">
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {CONTRIBUTIONS.map(({ icon: Icon, title: t, model, text }) => (
            <div key={t} className="rounded-xl border border-cyan-300/10 bg-slate-900/40 p-4">
              <span className="mb-3 grid h-9 w-9 place-items-center rounded-lg border border-cyan-300/25 bg-cyan-400/10"><Icon className="h-4 w-4 text-cyan-300" /></span>
              <div className="text-sm font-semibold text-slate-100">{t}</div>
              {model && <div className="mt-0.5 text-xs" style={{ color: MODEL_COLOR[model] }}>{MODEL_LABEL[model]}</div>}
              <p className="mt-2 text-[13px] leading-relaxed text-slate-400">{text}</p>
            </div>
          ))}
        </div>
      </Panel>

      <div className="grid gap-5 lg:grid-cols-2">
        <Panel eyebrow="Honest reading" title="What did not improve" right={<Scale className="h-5 w-5 text-amber-300/80" />}>
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-slate-300">
            <li>Recall is essentially unchanged: {rf("7").recall.toFixed(4)} → {rf("13").recall.toFixed(4)}. Roughly three in four anomalous observations are still missed.</li>
            <li>The 10-feature model's recall is slightly <i>lower</i> than the original's ({rf("10").recall.toFixed(4)} vs {rf("7").recall.toFixed(4)}).</li>
            <li>Speed did not improve. Reported latency is {TEAM.prediction_latency_ms["7"]} → {TEAM.prediction_latency_ms["13"]} ms per 1,000 observations, and 10 features is the slowest at {TEAM.prediction_latency_ms["10"]} ms. In the result files' whole-test-set timing, 13 features took {RF.runtime_seconds.always_13_feature.toFixed(2)} s against {RF.runtime_seconds.baseline_7_feature.toFixed(2)} s.</li>
            <li>The adaptive hybrid does not beat always running 13 features (F1 {RF.adaptive_test.hybrid.f1.toFixed(4)} vs {RF.adaptive_test.always_13.f1.toFixed(4)}).</li>
          </ul>
        </Panel>
        <Panel eyebrow="Before a paper claim" title="What is still needed" right={<Layers className="h-5 w-5 text-cyan-300/70" />}>
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-slate-300">
            <li>Repeated runs (several random seeds) and confidence intervals for every metric.</li>
            <li>More channels and missions than Mission 2, channel 9.</li>
            <li>Incident-level results for the 7-feature original, so the incident gain is measured against the baseline.</li>
            <li>The baseline vs calibrated threshold results and monthly recall, with their result files.</li>
          </ul>
        </Panel>
      </div>
    </Section>
  );
}
