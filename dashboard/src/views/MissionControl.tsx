import { Activity, ArrowRight, BrainCircuit, Database, Gauge, Radio, ShieldCheck, Siren, Timer } from "lucide-react";
import Globe from "../components/Globe";
import { Chip } from "../components/ui";
import { dataset, fmt, label, ours, paper, SEEDS, totalTestEvents } from "../data/model";

const PHASES: { id: string; name: string; status: "done" | "next" | "planned" }[] = [
  { id: "P0", name: "Setup and metric check", status: "done" },
  { id: "P1", name: "Reproduce the paper's baseline", status: "done" },
  { id: "P2", name: "Freeze the evaluation protocol", status: "done" },
  { id: "P3", name: "C1: adaptive computation", status: "done" },
  { id: "P4", name: "C2 + C3: reliability and graphs", status: "done" },
  { id: "P5", name: "C4: cost-aware response", status: "done" },
  { id: "P6", name: "C5 ablation + rare events", status: "done" },
];

const PIPELINE = [
  { icon: Radio, title: "Telemetry in", text: `${dataset.channels.length} sensor channels, one reading every ${dataset.sampling_seconds} s` },
  { icon: BrainCircuit, title: "LSTM forecast", text: "Predicts the next readings from the last 2 hours" },
  { icon: Activity, title: "Forecast error", text: "Gap between prediction and the real reading, smoothed" },
  { icon: Gauge, title: "Dynamic threshold", text: "Flags errors far above the recent normal level" },
  { icon: Siren, title: "Alarm", text: "Raised for the channel and time span affected" },
];

export default function MissionControl({ onOpenResearch }: { onOpenResearch: () => void }) {
  const years = Number(dataset.test_period[1].slice(0, 4)) - Number(dataset.train_periods["84_months"][0].slice(0, 4));
  const headline = ours("21_months", "Telemanom-ESA-Pruned").mean;
  const headlinePaper = paper("21_months", "Telemanom-ESA-Pruned");
  const facts = [
    { icon: Database, value: dataset.mission, label: "European Space Agency mission data" },
    { icon: Radio, value: `${dataset.channels.length} channels`, label: `${dataset.channels[0].replace("_", " ")} to ${dataset.channels.at(-1)!.replace("_", " ")}` },
    { icon: Timer, value: `${years} years`, label: `${dataset.train_periods["84_months"][0].slice(0, 4)} to ${dataset.test_period[1].slice(0, 4)}, 30-second steps` },
    { icon: Siren, value: `${totalTestEvents} events`, label: `in the test years: ${dataset.test_events.Anomaly} anomalies, ${dataset.test_events["Rare Event"]} rare events` },
    { icon: ShieldCheck, value: `${SEEDS.length * 2} models`, label: "trained: 5 seeds × 2 training lengths" },
  ];

  return (
    <div className="relative min-h-[calc(100vh-3.5rem)] overflow-hidden">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(600px_400px_at_72%_45%,rgba(56,189,248,0.12),transparent_70%)]" />

      <div className="relative mx-auto grid max-w-[1500px] gap-6 px-4 pb-10 pt-6 sm:px-6 lg:min-h-[calc(100vh-3.5rem)] lg:grid-cols-[minmax(0,1fr)_minmax(0,1.25fr)] lg:items-center">
        <div className="relative z-10 order-2 lg:order-1">
          <div className="eyebrow mb-3 flex items-center gap-2"><span className="pulse-dot h-1.5 w-1.5 rounded-full bg-cyan-300" />ESA Anomaly Detection Benchmark · {dataset.mission}</div>
          <h1 className="font-display text-3xl leading-tight text-white sm:text-5xl">
            Satellite Telemetry
            <span className="block bg-gradient-to-r from-cyan-200 via-cyan-300 to-blue-400 bg-clip-text text-transparent">Anomaly Detection</span>
          </h1>
          <p className="mt-5 max-w-xl text-[15px] leading-relaxed text-slate-300">
            A satellite streams thousands of sensor readings to the ground. We trained a forecasting neural network on real European Space
            Agency (ESA) telemetry to flag readings that stop behaving normally, while keeping false alarms rare. We reproduced the published
            benchmark result, then tested trust-aware, resource-efficient additions on top of it: running the network 20× less often at the same accuracy, and telling broken sensors apart from real problems.
          </p>

          <div className="mt-6 flex flex-wrap items-center gap-3">
            <button
              onClick={onOpenResearch}
              className="group inline-flex items-center gap-2 rounded-xl border border-cyan-300/40 bg-cyan-400/15 px-5 py-2.5 text-sm font-semibold text-cyan-100 shadow-[0_0_30px_-8px_rgba(56,232,255,.7)] transition hover:bg-cyan-400/25"
            >
              Open research results <ArrowRight className="h-4 w-4 transition group-hover:translate-x-0.5" />
            </button>
            <div className="glass flex items-center gap-3 px-4 py-2">
              <div>
                <div className="text-[10px] uppercase tracking-wider text-slate-400">Event-wise F0.5, {label("21_months")}</div>
                <div className="font-mono text-sm"><span className="font-semibold text-cyan-300">{fmt(headline)}</span><span className="text-slate-500"> ours vs </span><span className="text-slate-300">{fmt(headlinePaper)}</span><span className="text-slate-500"> paper</span></div>
              </div>
            </div>
          </div>

          <div className="mt-8 grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-5">
            {facts.map(({ icon: Icon, value, label: text }) => (
              <div key={value} className="glass p-3">
                <Icon className="mb-2 h-4 w-4 text-cyan-300/80" />
                <div className="font-mono text-sm font-semibold text-white">{value}</div>
                <div className="mt-0.5 text-[11px] leading-snug text-slate-400">{text}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="relative order-1 h-[380px] sm:h-[520px] lg:order-2 lg:h-[min(78vh,760px)]">
          <Globe className="absolute inset-0" />
          <div className="absolute left-2 top-2 sm:left-6 sm:top-6">
            <Chip tone="muted">Illustrative visual · not a live satellite feed</Chip>
          </div>
          <div className="glass absolute bottom-2 right-2 w-[min(290px,calc(100%-1rem))] p-4 sm:bottom-6 sm:right-6">
            <div className="eyebrow mb-2">Project status</div>
            <ul className="space-y-1.5 text-xs">
              {PHASES.map((p) => (
                <li key={p.id} className="flex items-center gap-2">
                  <span className={`h-2 w-2 rounded-full ${p.status === "done" ? "bg-emerald-400" : p.status === "next" ? "pulse-dot bg-amber-300" : "bg-slate-600"}`} />
                  <span className="w-6 font-mono text-slate-400">{p.id}</span>
                  <span className={p.status === "planned" ? "text-slate-500" : "text-slate-200"}>{p.name}</span>
                  <span className={`ml-auto text-[10px] uppercase ${p.status === "done" ? "text-emerald-300" : p.status === "next" ? "text-amber-300" : "text-slate-500"}`}>{p.status}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      <div className="relative mx-auto max-w-[1500px] px-4 pb-14 sm:px-6">
        <div className="eyebrow mb-3">How the detector works</div>
        <div className="grid gap-3 md:grid-cols-5">
          {PIPELINE.map(({ icon: Icon, title, text }, i) => (
            <div key={title} className="glass relative p-4">
              <div className="mb-2 flex items-center gap-2">
                <span className="grid h-7 w-7 place-items-center rounded-lg bg-cyan-400/10"><Icon className="h-4 w-4 text-cyan-300" /></span>
                <span className="font-mono text-[11px] text-cyan-300/60">0{i + 1}</span>
              </div>
              <div className="text-sm font-semibold text-slate-100">{title}</div>
              <div className="mt-1 text-xs leading-relaxed text-slate-400">{text}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
