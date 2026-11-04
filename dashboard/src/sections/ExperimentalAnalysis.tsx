import { Bar, BarChart, CartesianGrid, Cell, ComposedChart, LabelList, Legend, Line, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { FlaskConical } from "lucide-react";
import { axisProps, ChartTooltip, Chip, COLORS, Insight, Missing, Panel, Section } from "../components/ui";
import { change, MODEL_COLOR, MODEL_KEYS, MODEL_LABEL, MODEL_SHORT, n0, pct, RF, rf, TEAM, type ModelKey } from "../data/channel9";
import type { SectionProps } from "./types";

const f3 = (x: number) => x.toFixed(3);
const f4 = (x: number) => x.toFixed(4);

export function ScopeNote() {
  const e = RF.experiment;
  return (
    <div className="rounded-xl border border-violet-300/25 bg-violet-400/5 p-4 text-sm leading-relaxed text-violet-100/90">
      <div className="mb-1 flex items-center gap-2 font-semibold text-violet-200"><FlaskConical className="h-4 w-4" />A separate experiment</div>
      These results come from the team's feature-engineering study: Random Forest detectors on <b>{e.dataset}</b>, split {e.split.toLowerCase()},
      with {n0(e.test_observations)} test observations ({n0(e.test_anomalies)} anomalous). Metrics are <b>sample-level</b>: every observation
      is scored separately. They measure a different task from the event-wise Mission 1 results in sections A-H and must not be compared with
      them. Each value comes from a single run. Charts 1-4 read the result files at commit <span className="font-mono text-xs">{RF.source.commit}</span>.
      Values marked <Chip tone="violet">team-reported</Chip> were supplied from the team's screenshots and are not in those files.
    </div>
  );
}

function Matrix({ k }: { k: ModelKey }) {
  const m = rf(k);
  const cell = (label: string, value: number, tone: string, diff?: number, goodWhenLower?: boolean) => (
    <div className={`rounded-lg border p-3 ${tone}`}>
      <div className="text-[10px] uppercase tracking-wide text-slate-400">{label}</div>
      <div className="font-mono text-sm font-semibold text-slate-100">{n0(value)}</div>
      {diff !== undefined && diff !== 0 && (
        <div className={`font-mono text-[11px] ${(goodWhenLower ? diff < 0 : diff > 0) ? "text-emerald-300" : "text-amber-300"}`}>
          {diff > 0 ? "+" : "−"}{n0(Math.abs(diff))} vs 7 features
        </div>
      )}
    </div>
  );
  const base = rf("7");
  const d = (x: number, y: number) => (k === "7" ? undefined : x - y);
  return (
    <div className="glass p-4">
      <div className="mb-3 text-sm font-semibold" style={{ color: MODEL_COLOR[k] }}>{MODEL_LABEL[k]}</div>
      <div className="mb-1 grid grid-cols-[58px_minmax(0,1fr)_minmax(0,1fr)] gap-1.5 text-[10px] uppercase tracking-wide text-slate-500">
        <span />
        <span className="text-center">Predicted normal</span>
        <span className="text-center">Predicted anomaly</span>
      </div>
      <div className="grid grid-cols-[58px_minmax(0,1fr)_minmax(0,1fr)] items-center gap-1.5">
        <span className="text-[10px] uppercase tracking-wide text-slate-500">Actual normal</span>
        {cell("True negative", m.tn, "border-slate-500/20 bg-slate-500/5", d(m.tn, base.tn))}
        {cell("False positive", m.fp, "border-rose-300/25 bg-rose-400/5", d(m.fp, base.fp), true)}
        <span className="text-[10px] uppercase tracking-wide text-slate-500">Actual anomaly</span>
        {cell("False negative", m.fn, "border-amber-300/25 bg-amber-400/5", d(m.fn, base.fn), true)}
        {cell("True positive", m.tp, "border-emerald-300/25 bg-emerald-400/5", d(m.tp, base.tp))}
      </div>
    </div>
  );
}

export default function ExperimentalAnalysis({ id, letter, title }: SectionProps) {
  const perf = (["precision", "recall", "f1", "pr_auc"] as const).map((key) => ({
    name: { precision: "Precision", recall: "Recall", f1: "F1-score", pr_auc: "PR-AUC" }[key],
    ...Object.fromEntries(MODEL_KEYS.map((k) => [MODEL_SHORT[k], rf(k)[key]])),
  }));
  const fprRows = MODEL_KEYS.map((k) => ({ name: MODEL_SHORT[k], fpr: rf(k).fpr * 100, color: MODEL_COLOR[k] }));
  const p13 = change("precision", "13");
  const f13 = change("f1", "13");
  const a13 = change("pr_auc", "13");
  const fpr13 = change("fpr", "13");
  const r13 = change("recall", "13");

  const scatter = [
    ...MODEL_KEYS.map((k) => ({ name: MODEL_LABEL[k], fpr: rf(k).fpr * 100, precision: rf(k).precision, color: MODEL_COLOR[k] })),
    { name: "Adaptive hybrid (7 → 13)", fpr: RF.adaptive_test.hybrid.fpr * 100, precision: RF.adaptive_test.hybrid.precision, color: COLORS.warn },
  ];

  const margins = RF.routing_validation.filter((r) => r.model === "adaptive hybrid").map((r) => ({ margin: `±${r.margin}`, f1: r.f1, escalation: r.escalation_pct ?? 0 }));
  const valBase = RF.routing_validation.find((r) => r.model === "7-feature baseline")!;
  const valAlways = RF.routing_validation.find((r) => r.model === "13-feature always")!;

  const inc = TEAM.incident_detection;
  const incidentRows = (["10", "13"] as const).map((k) => ({ name: MODEL_SHORT[k], Detected: inc[k].detected, Missed: inc[k].missed }));
  const sizeRows = MODEL_KEYS.map((k) => ({ name: MODEL_SHORT[k], size: TEAM.model_size_mb[k], color: MODEL_COLOR[k] }));
  const latencyRows = MODEL_KEYS.map((k) => ({ name: MODEL_SHORT[k], latency: TEAM.prediction_latency_ms[k], color: MODEL_COLOR[k] }));
  const sizeChange = ((TEAM.model_size_mb["13"] - TEAM.model_size_mb["7"]) / TEAM.model_size_mb["7"]) * 100;
  const latChange = ((TEAM.prediction_latency_ms["13"] - TEAM.prediction_latency_ms["7"]) / TEAM.prediction_latency_ms["7"]) * 100;
  const lat10 = ((TEAM.prediction_latency_ms["10"] - TEAM.prediction_latency_ms["7"]) / TEAM.prediction_latency_ms["7"]) * 100;

  return (
    <Section id={id} letter={letter} title={title} subtitle="Six analyses of the original 7-feature, expanded 10-feature and extended 13-feature Random Forest detectors.">
      <ScopeNote />

      <Panel title="1 · Three-model performance comparison" eyebrow="Test set · sample-level · hover for values">
        <div className="grid gap-5 xl:grid-cols-[1.6fr_1fr]">
          <div className="h-[320px]">
            <ResponsiveContainer>
              <BarChart data={perf} margin={{ top: 24, right: 12, left: 0, bottom: 4 }} barGap={4}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="name" {...axisProps} />
                <YAxis domain={[0, 1]} {...axisProps} label={{ value: "Score (higher is better)", angle: -90, position: "insideLeft", fill: COLORS.axis, fontSize: 12 }} />
                <Tooltip content={<ChartTooltip digits={4} />} />
                <Legend />
                {MODEL_KEYS.map((k) => (
                  <Bar key={k} dataKey={MODEL_SHORT[k]} fill={MODEL_COLOR[k]} radius={[5, 5, 0, 0]} maxBarSize={46}>
                    <LabelList dataKey={MODEL_SHORT[k]} position="top" formatter={(v: number) => f3(v)} fill="#cbd5e1" fontSize={10} />
                  </Bar>
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="h-[320px]">
            <ResponsiveContainer>
              <BarChart data={fprRows} margin={{ top: 24, right: 12, left: 0, bottom: 4 }}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="name" {...axisProps} />
                <YAxis {...axisProps} unit="%" tickFormatter={(v: number) => v.toFixed(2)} width={52} label={{ value: "False-positive rate", angle: -90, position: "insideLeft", fill: COLORS.axis, fontSize: 11 }} />
                <Tooltip content={<ChartTooltip digits={4} unit="%" />} />
                <Bar dataKey="fpr" name="False-positive rate" radius={[5, 5, 0, 0]} maxBarSize={56}>
                  {fprRows.map((r) => <Cell key={r.name} fill={r.color} />)}
                  <LabelList dataKey="fpr" position="top" formatter={(v: number) => `${v.toFixed(3)}%`} fill="#cbd5e1" fontSize={10} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <Insight
          compared="Precision, recall, F1-score, PR-AUC (left) and false-positive rate (right) for the original 7-feature model, the expanded 10-feature model (adds short-term deviation features) and the extended 13-feature model (adds long-term drift features)."
          indicates={<>The 10-feature model raises precision, F1 and PR-AUC but not recall ({f4(rf("10").recall)} vs {f4(rf("7").recall)}). The 13-feature model is the best of the three on every score shown.</>}
          improvement={<>Against the 7-feature original: precision {pct(p13.pct)}, F1 {pct(f13.pct)}, PR-AUC {pct(a13.pct)}, recall {pct(r13.pct)} and false-positive rate {pct(fpr13.pct)}.</>}
          why="When the detector raises an alarm, operators can trust it more, and fewer normal readings are wrongly flagged."
          tradeoffs={<>Recall stays near {Math.round(rf("13").recall * 100)}% for all three models: most anomalous observations are still missed. The F1 gain is small, and the values come from one run on one channel.</>}
        />
      </Panel>

      <Panel title="2 · Confusion matrix comparison" eyebrow={`Test set · ${n0(RF.experiment.test_observations)} observations · differences vs 7 features`}>
        <div className="grid gap-4 lg:grid-cols-2 2xl:grid-cols-3">{MODEL_KEYS.map((k) => <Matrix key={k} k={k} />)}</div>
        <Insight
          compared="Every test observation sorted into four outcomes: normal readings correctly left alone (true negatives), normal readings wrongly flagged (false positives), anomalies missed (false negatives) and anomalies caught (true positives)."
          indicates={<>False positives fall steadily: {n0(rf("7").fp)} → {n0(rf("10").fp)} → {n0(rf("13").fp)}. Caught anomalies dip slightly at 10 features ({n0(rf("10").tp)}) and rise at 13 ({n0(rf("13").tp)}).</>}
          improvement={<>The 13-feature model raises {n0(rf("7").fp - rf("13").fp)} fewer false alarms and catches {n0(rf("13").tp - rf("7").tp)} more anomalous observations than the original.</>}
          why="Each false positive is an alarm an operator has to check for nothing. Removing hundreds of them reduces alarm fatigue."
          tradeoffs={<>The bottom-left box dominates: about {n0(rf("13").fn)} anomalous observations are still missed by every model. The improvement is in false alarms, not in finding more anomalies.</>}
        />
      </Panel>

      <div className="grid gap-5 xl:grid-cols-2">
        <Panel title="3 · Precision vs false-positive rate" eyebrow="Test set · top-left is best">
          <div className="h-[300px]">
            <ResponsiveContainer>
              <ScatterChart margin={{ top: 16, right: 24, left: 4, bottom: 16 }}>
                <CartesianGrid />
                <XAxis type="number" dataKey="fpr" name="False-positive rate" domain={[0.14, 0.17]} {...axisProps} tickFormatter={(v: number) => `${v.toFixed(3)}%`} label={{ value: "False-positive rate · lower is better", position: "insideBottom", offset: -10, fill: COLORS.axis, fontSize: 12 }} />
                <YAxis type="number" dataKey="precision" name="Precision" domain={[0.79, 0.83]} {...axisProps} tickFormatter={(v: number) => v.toFixed(2)} width={44} label={{ value: "Precision", angle: -90, position: "insideLeft", fill: COLORS.axis, fontSize: 12 }} />
                <ZAxis range={[160, 160]} />
                <Tooltip cursor={{ strokeDasharray: "3 3" }} content={({ active, payload }) => active && payload?.length ? (
                  <div className="glass-strong px-3 py-2 text-xs">
                    <div className="mb-1 font-semibold text-slate-100">{payload[0].payload.name}</div>
                    <div className="text-slate-300">Precision <span className="font-mono">{f4(payload[0].payload.precision)}</span></div>
                    <div className="text-slate-300">False-positive rate <span className="font-mono">{payload[0].payload.fpr.toFixed(4)}%</span></div>
                  </div>) : null} />
                <Legend verticalAlign="top" height={28} />
                {scatter.map((p) => <Scatter key={p.name} name={p.name} data={[p]} fill={p.color} />)}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <Insight
            compared="How often each model's alarms are correct (precision) against how often it alarms on normal data (false-positive rate). It also includes the adaptive hybrid, which sends uncertain cases from the 7-feature model to the 13-feature model."
            indicates="Moving from 7 to 10 to 13 features shifts the point up and to the left: more trustworthy alarms and fewer false ones. The hybrid lands almost on top of the 13-feature model."
            improvement={<>False-positive rate falls from {(rf("7").fpr * 100).toFixed(3)}% to {(rf("13").fpr * 100).toFixed(3)}% while precision rises to {f3(rf("13").precision)}. The hybrid has the lowest false-positive rate ({(RF.adaptive_test.hybrid.fpr * 100).toFixed(4)}%).</>}
            why="Fewer false alarms without losing detections is exactly what spacecraft operations need."
            tradeoffs="All points sit within a narrow band (precision 0.80-0.82). The axes are zoomed in to make the differences visible."
          />
        </Panel>

        <Panel title="4 · Threshold calibration" eyebrow="Calibrated on validation data only">
          <div className="h-[230px]">
            <ResponsiveContainer>
              <ComposedChart data={margins} margin={{ top: 16, right: 8, left: 0, bottom: 16 }}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="margin" {...axisProps} label={{ value: "Routing margin around the 7-feature threshold", position: "insideBottom", offset: -10, fill: COLORS.axis, fontSize: 12 }} />
                <YAxis yAxisId="f1" domain={[0.52, 0.55]} {...axisProps} tickFormatter={(v: number) => v.toFixed(3)} width={48} />
                <YAxis yAxisId="esc" orientation="right" {...axisProps} unit="%" width={44} />
                <Tooltip content={<ChartTooltip digits={4} />} />
                <Legend verticalAlign="top" height={28} />
                <Bar yAxisId="esc" dataKey="escalation" name="Escalated to 13 features (%)" fill="rgba(251,191,36,0.35)" radius={[4, 4, 0, 0]} maxBarSize={40} />
                <Line yAxisId="f1" dataKey="f1" name="Hybrid F1 (validation)" stroke={COLORS.ours} strokeWidth={2} dot={{ r: 4 }} />
                <ReferenceLine yAxisId="f1" y={valBase.f1} stroke={MODEL_COLOR["7"]} strokeDasharray="5 4" label={{ value: `7 features ${f3(valBase.f1)}`, fill: "#a5b4d4", fontSize: 10, position: "insideBottomLeft" }} />
                <ReferenceLine yAxisId="f1" y={valAlways.f1} stroke={COLORS.good} strokeDasharray="5 4" label={{ value: `13 always ${f3(valAlways.f1)}`, fill: COLORS.good, fontSize: 10, position: "insideTopLeft" }} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <table className="mt-2 w-full text-xs">
            <thead><tr className="text-left uppercase tracking-wider text-slate-400"><th className="pb-1 font-medium">Model</th><th className="pb-1 text-right font-medium">Calibrated alarm threshold</th></tr></thead>
            <tbody className="font-mono">{MODEL_KEYS.map((k) => (
              <tr key={k} className="border-t border-cyan-300/10"><td className="py-1.5 font-sans text-slate-200">{MODEL_LABEL[k]}</td><td className="text-right text-slate-300">{f4(rf(k).threshold)}</td></tr>
            ))}</tbody>
          </table>
          <div className="mt-3"><Missing title="Baseline vs calibrated threshold: precision, recall and F1" needed="The exact values and threshold settings for this comparison are not in the experiment result files. Add the result file behind the screenshot and this chart will be built from it." /></div>
          <Insight
            compared={<>Alarm thresholds chosen on validation data to maximise F1 with at most {RF.experiment.maximum_false_positive_rate * 100}% false positives. The chart shows how wide a "grey zone" around the 7-feature threshold should be sent to the 13-feature model.</>}
            indicates="Widening the margin escalates more observations (bars) and slowly raises validation F1 (line). The ±0.1 margin scored best and was locked in before testing."
            improvement={<>The hybrid reaches validation F1 {f3(Math.max(...margins.map((m) => m.f1)))} against {f3(valBase.f1)} for 7 features, while escalating at most {Math.max(...margins.map((m) => m.escalation)).toFixed(2)}% of observations.</>}
            why="Calibrating on validation data, never the test set, keeps the evaluation honest. Escalating only uncertain cases is the idea behind adaptive computation."
            tradeoffs={<>The hybrid never beats always running the 13-feature model ({f3(valAlways.f1)} on validation). Distance from the threshold is a rough stand-in for uncertainty.</>}
          />
        </Panel>
      </div>

      <Missing title="5 · Monthly anomaly detection recall" needed="The monthly recall values and their exact dates (baseline vs calibrated threshold, seven months) are not in the experiment result files, and they must not be estimated from the screenshot. Add the result file with the monthly values and this line chart will be built from it." />

      <Panel title="6 · Incident detection and computational efficiency" eyebrow="Team-reported values">
        <div className="grid gap-5 xl:grid-cols-3">
          <div>
            <div className="mb-2 flex items-center justify-between"><span className="text-sm font-semibold text-slate-200">Incidents detected</span><Chip tone="violet">team-reported</Chip></div>
            <div className="space-y-3">
              {(["10", "13"] as const).map((k) => {
                const v = inc[k];
                const share = (v.detected / inc.total_incidents) * 100;
                return (
                  <div key={k} className="rounded-xl border border-cyan-300/10 bg-slate-900/40 p-3">
                    <div className="flex items-baseline justify-between">
                      <span className="text-sm" style={{ color: MODEL_COLOR[k] }}>{MODEL_LABEL[k]}</span>
                      <span className="font-mono text-lg font-semibold text-white">{v.detected}/{inc.total_incidents}</span>
                    </div>
                    <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-800"><div className="h-full rounded-full" style={{ width: `${share}%`, background: MODEL_COLOR[k] }} /></div>
                    <div className="mt-1 flex justify-between text-[11px] text-slate-400"><span>{share.toFixed(1)}% detected</span><span>{v.missed} missed</span></div>
                  </div>
                );
              })}
              <p className="text-[11px] text-slate-500">{inc.note}</p>
            </div>
            <div className="mt-2 h-[150px]">
              <ResponsiveContainer>
                <BarChart data={incidentRows} layout="vertical" margin={{ top: 4, right: 16, left: 4, bottom: 4 }}>
                  <XAxis type="number" domain={[0, inc.total_incidents]} {...axisProps} />
                  <YAxis type="category" dataKey="name" width={80} {...axisProps} />
                  <Tooltip content={<ChartTooltip digits={0} />} />
                  <Bar dataKey="Detected" stackId="i" fill={COLORS.good} />
                  <Bar dataKey="Missed" stackId="i" fill={COLORS.bad} radius={[0, 5, 5, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div>
            <div className="mb-2 flex items-center justify-between"><span className="text-sm font-semibold text-slate-200">Model size (MB)</span><Chip tone="violet">team-reported</Chip></div>
            <div className="h-[300px]">
              <ResponsiveContainer>
                <BarChart data={sizeRows} margin={{ top: 24, right: 8, left: 0, bottom: 4 }}>
                  <CartesianGrid vertical={false} />
                  <XAxis dataKey="name" {...axisProps} />
                  <YAxis {...axisProps} width={44} tickFormatter={(v: number) => v.toFixed(1)} label={{ value: "MB", angle: -90, position: "insideLeft", fill: COLORS.axis, fontSize: 11 }} />
                  <Tooltip content={<ChartTooltip digits={2} unit=" MB" />} />
                  <Bar dataKey="size" name="Model size" radius={[5, 5, 0, 0]} maxBarSize={56}>
                    {sizeRows.map((r) => <Cell key={r.name} fill={r.color} />)}
                    <LabelList dataKey="size" position="top" formatter={(v: number) => `${v.toFixed(2)} MB`} fill="#cbd5e1" fontSize={11} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div>
            <div className="mb-2 flex items-center justify-between"><span className="text-sm font-semibold text-slate-200">Prediction latency (ms)</span><Chip tone="violet">team-reported</Chip></div>
            <div className="h-[300px]">
              <ResponsiveContainer>
                <BarChart data={latencyRows} margin={{ top: 24, right: 8, left: 0, bottom: 4 }}>
                  <CartesianGrid vertical={false} />
                  <XAxis dataKey="name" {...axisProps} />
                  <YAxis {...axisProps} unit=" ms" width={52} domain={[0, 10]} />
                  <Tooltip content={<ChartTooltip digits={3} unit=" ms" />} />
                  <Bar dataKey="latency" name="Prediction latency" radius={[5, 5, 0, 0]} maxBarSize={56}>
                    {latencyRows.map((r) => <Cell key={r.name} fill={r.color} />)}
                    <LabelList dataKey="latency" position="top" formatter={(v: number) => `${v.toFixed(3)} ms`} fill="#cbd5e1" fontSize={11} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
        <p className="mt-2 text-xs leading-relaxed text-slate-500">
          {TEAM.latency_note} In that whole-test-set run, scoring took {RF.runtime_seconds.baseline_7_feature.toFixed(2)} s with 7 features and{" "}
          {RF.runtime_seconds.always_13_feature.toFixed(2)} s with 13 features.
        </p>
        <Insight
          compared="Whole incidents caught (rather than single observations), the size of each saved model, and how long one prediction takes."
          indicates={<>The 13-feature model detects all {inc.total_incidents} incidents, against {inc["10"].detected} for the 10-feature model. It is also the smallest model ({TEAM.model_size_mb["13"]} MB). Its latency ({TEAM.prediction_latency_ms["13"]} ms) is about the same as the original's ({TEAM.prediction_latency_ms["7"]} ms).</>}
          improvement={<>{inc["13"].detected - inc["10"].detected} more incidents detected than the 10-feature model, a {pct(sizeChange)} smaller model than the original, and latency within {pct(latChange)} of the original.</>}
          why="Operators respond to incidents, not single data points. Catching every incident at least once matters more than flagging every anomalous reading. Small, fast models suit on-board processors and ground systems watching many satellites."
          tradeoffs={<>No incident count was reported for the 7-feature original, so this does not show an improvement over the original. The 10-feature model was the slowest ({pct(lat10)} vs 7 features). In the full test-set timing from the result files, the 13-feature model was slower than the original.</>}
        />
      </Panel>
    </Section>
  );
}
