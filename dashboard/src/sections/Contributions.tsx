import { Cpu, GitBranch, ShieldCheck } from "lucide-react";
import { Chip, Panel, Section } from "../components/ui";
import { events, fmt, ours } from "../data/model";
import type { SectionProps } from "./types";

const pruned = () => ours("21_months", "Telemanom-ESA-Pruned");

const CONTRIBUTIONS = [
  {
    code: "C1",
    name: "Adaptive computation",
    icon: Cpu,
    baseline: "Runs the full LSTM forecaster on every 30-second reading of every channel, all the time.",
    limitation: "Most telemetry is normal and easy to judge, so full computation is spent where it changes nothing. Our seed study also shows the alarm decision, not the network, decides which events are caught.",
    proposal: "A lightweight detector screens all telemetry. The LSTM is activated only when the screener's confidence is low, the potential impact is high, or the pattern looks unfamiliar.",
    benefit: "Less computation for the same or better detection, which matters for on-board or large-fleet monitoring.",
    metrics: "Event-wise F0.5 (must not drop), share of windows sent to the LSTM, wall-clock and GPU time, a quality vs compute curve.",
  },
  {
    code: "C2",
    name: "Sensor reliability",
    icon: ShieldCheck,
    baseline: "Treats every channel as equally trustworthy. A faulty sensor looks the same as a real spacecraft problem.",
    limitation: "Sensor faults (stuck, drifting or noisy readings) can raise false alarms or hide real ones. The dataset has no sensor-fault labels.",
    proposal: "A trust-aware layer scores each sensor's reliability by comparing it with related channels in its group, and weighs alarms by that trust.",
    benefit: "Separates 'the sensor is broken' from 'the satellite has a problem', cutting false alarms caused by bad sensors.",
    metrics: "Per-fault-type F1 and false-alarm reduction on injected sensor faults, plus no loss of event-wise F0.5 on real events.",
  },
  {
    code: "C3",
    name: "Incident correlation",
    icon: GitBranch,
    baseline: "Raises alarms channel by channel, with no link between alarms that belong to the same incident.",
    limitation: "One incident can trigger alarms on several channels, flooding operators with alerts that are really one problem.",
    proposal: "A dependency graph learned from training data connects channels that move together, and groups related alarms into a single incident.",
    benefit: "Fewer, clearer alerts that point to the likely related channels.",
    metrics: "Grouping accuracy against ESA event IDs, alerts per event, and onset-order agreement. Evaluated on the full Mission 1 channel set.",
  },
];

const FIELDS = [
  ["baseline", "What the baseline does"],
  ["limitation", "Limitation we identified"],
  ["proposal", "What we propose to add"],
  ["benefit", "How it is expected to help"],
  ["metrics", "How we will prove it"],
] as const;

export default function Contributions({ id, letter, title }: SectionProps) {
  return (
    <Section id={id} letter={letter} title={title} subtitle="Planned work, built on the reproduced baseline. No results are claimed for these yet.">
      <div className="rounded-xl border border-amber-300/25 bg-amber-400/5 p-4 text-sm text-amber-100/90">
        <b>Status: proposed.</b> These contributions are designed but not yet implemented or evaluated. No improvement figures are shown for
        them, and none will be until they are tested against the baseline with the official metric.
      </div>

      <div className="grid gap-5 xl:grid-cols-3">
        {CONTRIBUTIONS.map(({ code, name, icon: Icon, ...rest }) => (
          <Panel key={code} className="flex flex-col">
            <div className="mb-4 flex items-start justify-between">
              <div className="flex items-center gap-3">
                <span className="grid h-10 w-10 place-items-center rounded-xl border border-cyan-300/25 bg-cyan-400/10"><Icon className="h-5 w-5 text-cyan-300" /></span>
                <div>
                  <div className="font-mono text-xs text-cyan-300/70">{code}</div>
                  <div className="text-base font-semibold text-white">{name}</div>
                </div>
              </div>
              <Chip tone="warn">Proposed</Chip>
            </div>
            <ol className="space-y-3">
              {FIELDS.map(([key, heading], i) => (
                <li key={key} className="flex gap-3 text-sm">
                  <span className="mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded-full bg-slate-800 font-mono text-[10px] text-slate-400">{i + 1}</span>
                  <div><div className="text-xs font-semibold uppercase tracking-wide text-slate-400">{heading}</div><p className="mt-0.5 leading-relaxed text-slate-300">{rest[key]}</p></div>
                </li>
              ))}
            </ol>
          </Panel>
        ))}
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <Panel eyebrow="Where we stand" title="What is achieved">
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-slate-300">
            <li>Reproduced the published Telemanom-ESA-Pruned detector: event-wise F0.5 {fmt(pruned().mean)} at 21 months, against the published 0.776.</li>
            <li>Results stable across 5 training seeds, and every number checked against the official scoring code.</li>
            <li>Found that the alarm decision, not the network, decides which events are caught. That is where C1-C3 operate.</li>
          </ul>
        </Panel>
        <Panel eyebrow="Where we stand" title="What remains">
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-slate-300">
            <li>Rare nominal events are mostly missed ({events["Rare Event"].detected} of {events["Rare Event"].events} found).</li>
            <li>Our unpruned variant raises more false alarms than the paper's single run. Development-period threshold tuning is in progress.</li>
            <li>Only the six lightweight channels of Mission 1 have been evaluated so far.</li>
            <li>C4 (cost-aware response planning) and C5 (ablation study) are also planned, after C1-C3.</li>
          </ul>
        </Panel>
      </div>
    </Section>
  );
}
