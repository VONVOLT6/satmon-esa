import { BrainCircuit, CheckCircle2, CircleDashed, Database, FlaskConical, Ruler } from "lucide-react";
import { Chip, Panel, Section } from "../components/ui";
import { dataset, label, SEEDS, totalTestEvents } from "../data/model";
import type { SectionProps } from "./types";

const n = (x: number) => x.toLocaleString("en-US");

export default function Overview({ id, letter, title }: SectionProps) {
  return (
    <Section id={id} letter={letter} title={title} subtitle="What we set out to do, on what data, and how it is judged.">
      <Panel eyebrow="Project" title="Trust-Aware, Impact-Driven and Resource-Efficient Satellite Health Monitoring">
        <p className="text-sm leading-relaxed text-slate-300">
          <b className="text-slate-100">Satellite telemetry anomaly detection</b> means automatically spotting sensor readings (temperatures,
          voltages, currents and so on) that stop behaving the way they normally do. Engineers want two things at once: catch real problems
          early, and raise as few false alarms as possible, because every alarm costs someone's time.
        </p>
        <p className="mt-3 text-sm leading-relaxed text-slate-300">
          <b className="text-slate-100">Objective.</b> First reproduce a published, benchmarked detector exactly, to get a trustworthy
          baseline. Then add our own contributions on top and measure them against it with the same official scoring.
        </p>
      </Panel>

      <div className="grid gap-5 lg:grid-cols-2">
        <Panel eyebrow="Dataset" title="ESA Anomaly Detection Benchmark (ESA-ADB)" right={<Database className="h-5 w-5 text-cyan-300/70" />}>
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
            <dt className="text-slate-400">Source</dt><dd className="text-slate-200">{dataset.name}</dd>
            <dt className="text-slate-400">Mission</dt><dd className="text-slate-200">{dataset.mission}, real spacecraft, labelled by ESA engineers</dd>
            <dt className="text-slate-400">Channels</dt><dd className="font-mono text-slate-200">{dataset.channels.join(", ")}</dd>
            <dt className="text-slate-400">Sampling</dt><dd className="text-slate-200">one reading every {dataset.sampling_seconds} s</dd>
            {(["21_months", "84_months"] as const).map((d) => (
              <div key={d} className="contents">
                <dt className="text-slate-400">Train ({label(d)})</dt>
                <dd className="text-slate-200">{dataset.train_periods[d][0]} to {dataset.train_periods[d][1]} · {n(dataset.train_rows[d])} rows</dd>
              </div>
            ))}
            <dt className="text-slate-400">Test</dt>
            <dd className="text-slate-200">{dataset.test_period[0]} to {dataset.test_period[1]} · {n(dataset.test_rows)} rows · {totalTestEvents} events ({dataset.test_events.Anomaly} anomalies, {dataset.test_events["Rare Event"]} rare events)</dd>
          </dl>
        </Panel>

        <Panel eyebrow="Models" title="Baseline and variants" right={<BrainCircuit className="h-5 w-5 text-cyan-300/70" />}>
          <ul className="space-y-3 text-sm">
            <li className="flex gap-3">
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-300" />
              <div><b className="text-slate-100">Telemanom-ESA</b> <Chip tone="good">reproduced</Chip><p className="mt-1 text-slate-400">Two-layer LSTM (80 units) forecasts the next 10 readings of all 6 channels from the last 250 (about 2 hours). A dynamic threshold on the smoothed forecast error raises alarms. No minimum error size.</p></div>
            </li>
            <li className="flex gap-3">
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-300" />
              <div><b className="text-slate-100">Telemanom-ESA-Pruned</b> <Chip tone="good">reproduced</Chip><p className="mt-1 text-slate-400">Same model, but forecast errors below 0.007 are ignored. The paper's best Mission 1 detector.</p></div>
            </li>
            <li className="flex gap-3">
              <CircleDashed className="mt-0.5 h-4 w-4 shrink-0 text-amber-300" />
              <div><b className="text-slate-100">Improved variants (ours)</b> <Chip tone="warn">not yet evaluated</Chip><p className="mt-1 text-slate-400">A data-tuned error floor and contributions C1-C3 (Section H) are in progress. No test-set results exist for them yet, so none are shown as improvements.</p></div>
            </li>
          </ul>
        </Panel>
      </div>

      <Panel eyebrow="Evaluation" title="How results are judged" right={<Ruler className="h-5 w-5 text-cyan-300/70" />}>
        <div className="grid gap-4 text-sm md:grid-cols-3">
          <div><div className="mb-1 flex items-center gap-2 font-semibold text-slate-100"><FlaskConical className="h-4 w-4 text-cyan-300" />Official split</div><p className="text-slate-400">Train only on data before the split; test on {dataset.test_period[0].slice(0, 4)}-2013, which no model ever sees during training or tuning.</p></div>
          <div><div className="mb-1 font-semibold text-slate-100">Official metric code</div><p className="text-slate-400">Scores come from ESA-ADB's own <span className="font-mono">ESAScores</span>, unmodified, after 10 hand-calculated checks. The main score is the corrected event-wise F0.5: whole events, false alarms weighted more than misses.</p></div>
          <div><div className="mb-1 font-semibold text-slate-100">{SEEDS.length} seeds per setting</div><p className="text-slate-400">Each setting was trained {SEEDS.length} times from different random starts (seeds {SEEDS.join(", ")}), so we report the average and spread, not one lucky run.</p></div>
        </div>
        <p className="mt-4 rounded-lg border border-amber-300/20 bg-amber-400/5 p-3 text-xs leading-relaxed text-amber-100/80">
          All scores here are <b>event-wise</b> (whole anomaly events). They cannot be compared with sample-level precision, recall, F1 or PR-AUC
          from experiments that count individual 30-second points, use another mission or channel, or use a random split.
        </p>
      </Panel>
    </Section>
  );
}
