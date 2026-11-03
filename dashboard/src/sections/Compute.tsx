import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Cpu, HardDrive, MemoryStick, Microchip } from "lucide-react";
import { axisProps, ChartTooltip, COLORS, Explain, Panel, Section, Stat } from "../components/ui";
import { DATASETS, evaluation, hardware, SEEDS, training } from "../data/model";
import type { SectionProps } from "./types";

const SEED_COLORS = ["#38e8ff", "#3b82f6", "#a78bfa", "#34d399", "#fbbf24"];
const minutes = (s: number) => s / 60;
const avg = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;

export default function Compute({ id, letter, title }: SectionProps) {
  const t21 = training("21_months");
  const t84 = training("84_months");
  const timeRows = SEEDS.map((s, i) => ({
    name: `Seed ${s}`,
    "21 months": minutes(t21[i].train_seconds),
    "84 months": minutes(t84[i].train_seconds),
    epochs21: t21[i].epochs,
    epochs84: t84[i].epochs,
  }));
  const epochRows = SEEDS.map((s, i) => ({ name: `Seed ${s}`, "21 months": t21[i].epochs, "84 months": t84[i].epochs }));

  const curveLen = Math.max(...t21.map((r) => r.val_loss.length));
  const curves = Array.from({ length: curveLen }, (_, e) => {
    const row: Record<string, number | null> = { epoch: e + 1 };
    t21.forEach((r) => { row[`Seed ${r.seed}`] = r.val_loss[e] ?? null; });
    return row;
  });

  const evals = DATASETS.flatMap((d) => evaluation(d));
  const predict = evals.map((e) => e.predict_seconds).filter((x): x is number => x !== null);
  const thrESA = evals.map((e) => e["threshold_seconds|Telemanom-ESA"]).filter((x): x is number => x !== null);
  const thrPruned = evals.map((e) => e["threshold_seconds|Telemanom-ESA-Pruned"]).filter((x): x is number => x !== null);
  const samples = evals.find((e) => e.samples)?.samples ?? 0;

  const rng = (xs: number[]) => `${Math.min(...xs).toFixed(1)}–${Math.max(...xs).toFixed(1)}`;

  return (
    <Section id={id} letter={letter} title={title} subtitle="Measured on one laptop. Times come from the training summaries and evaluation logs of each run.">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="Training time · 21 months" value={`${rng(t21.map((r) => minutes(r.train_seconds)))} min`} sub={`${Math.min(...t21.map((r) => r.epochs))}–${Math.max(...t21.map((r) => r.epochs))} epochs`} tone="accent" />
        <Stat label="Training time · 84 months" value={`${rng(t84.map((r) => minutes(r.train_seconds)))} min`} sub={`${Math.min(...t84.map((r) => r.epochs))}–${Math.max(...t84.map((r) => r.epochs))} epochs`} tone="accent" />
        <Stat label="Test prediction" value={`${rng(predict.map(minutes))} min`} sub={`${samples.toLocaleString("en-US")} forecasts, 7 years`} />
        <Stat label="Alarm thresholding" value={`${rng([...thrESA, ...thrPruned].map(minutes))} min`} sub={`per variant · mean unpruned ${minutes(avg(thrESA)).toFixed(1)}, Pruned ${minutes(avg(thrPruned)).toFixed(1)}`} />
      </div>

      <div className="grid gap-5 xl:grid-cols-2">
        <Panel title="Training time per run" eyebrow="Minutes, until early stopping">
          <div className="h-[280px]">
            <ResponsiveContainer>
              <BarChart data={timeRows} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="name" {...axisProps} />
                <YAxis {...axisProps} label={{ value: "minutes", angle: -90, position: "insideLeft", fill: COLORS.axis, fontSize: 12 }} />
                <Tooltip content={<ChartTooltip digits={1} unit=" min" />} />
                <Legend />
                <Bar dataKey="21 months" fill={COLORS.ours2} radius={[5, 5, 0, 0]} maxBarSize={36} />
                <Bar dataKey="84 months" fill={COLORS.ours} radius={[5, 5, 0, 0]} maxBarSize={36} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <Explain
            shows="How long each training run took on the GPU before early stopping ended it."
            means={<>Every run trains in under 11 minutes. The 84-month runs are not slower even with 4× the data, because each epoch is capped at 1,000 batches and they stopped earlier ({Math.min(...t84.map((r) => r.epochs))}–{Math.max(...t84.map((r) => r.epochs))} epochs vs {Math.min(...t21.map((r) => r.epochs))}–{Math.max(...t21.map((r) => r.epochs))}).</>}
          />
        </Panel>

        <Panel title="Epochs until early stopping" eyebrow="Patience 20, minimum improvement 0.0003">
          <div className="h-[280px]">
            <ResponsiveContainer>
              <BarChart data={epochRows} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="name" {...axisProps} />
                <YAxis {...axisProps} />
                <Tooltip content={<ChartTooltip digits={0} unit=" epochs" />} />
                <Legend />
                <Bar dataKey="21 months" fill={COLORS.ours2} radius={[5, 5, 0, 0]} maxBarSize={36} />
                <Bar dataKey="84 months" fill={COLORS.ours} radius={[5, 5, 0, 0]} maxBarSize={36} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <Explain
            shows="The number of training passes (1,000 batches each) before the validation loss stopped improving for 20 passes in a row."
            means="The longer training set gives a smaller validation loss, so the fixed improvement threshold is reached sooner. This is the same stopping rule as the original code."
          />
        </Panel>
      </div>

      <Panel title="Validation loss during training, 21 months" eyebrow="One line per seed · lower is better">
        <div className="h-[300px]">
          <ResponsiveContainer>
            <LineChart data={curves} margin={{ top: 8, right: 16, left: 4, bottom: 4 }}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="epoch" {...axisProps} label={{ value: "epoch", position: "insideBottomRight", offset: -2, fill: COLORS.axis, fontSize: 12 }} />
              <YAxis {...axisProps} domain={["auto", "auto"]} tickFormatter={(v: number) => v.toFixed(3)} width={56} />
              <Tooltip content={<ChartTooltip digits={4} />} />
              <Legend />
              {t21.map((r, i) => <Line key={r.seed} dataKey={`Seed ${r.seed}`} stroke={SEED_COLORS[i]} dot={false} strokeWidth={1.6} connectNulls={false} isAnimationActive={false} />)}
            </LineChart>
          </ResponsiveContainer>
        </div>
        <Explain
          shows="How the forecasting error on held-out validation data fell as each of the 5 runs trained."
          means="All runs follow the same curve and level off at a similar loss. Training behaves consistently, and the lines end where early stopping kicked in."
        />
      </Panel>

      <Panel title="Computational setup" eyebrow="All runs on this machine">
        <div className="grid gap-3 text-sm sm:grid-cols-2 xl:grid-cols-4">
          {[[Microchip, "GPU", hardware.gpu], [Cpu, "CPU", hardware.cpu], [MemoryStick, "Memory", `${hardware.ram_gb} GB RAM`], [HardDrive, "Framework", hardware.framework]].map(([Icon, k, v]) => {
            const I = Icon as typeof Cpu;
            return (
              <div key={k as string} className="flex items-start gap-3 rounded-xl border border-cyan-300/10 bg-slate-900/40 p-3">
                <I className="mt-0.5 h-4 w-4 text-cyan-300" />
                <div><div className="text-xs text-slate-400">{k as string}</div><div className="text-slate-200">{v as string}</div></div>
              </div>
            );
          })}
        </div>
        <p className="mt-3 text-xs text-slate-400">The paper's own Telemanom-ESA training took 13,115 s at 84 months on different hardware, so timings are indicative only.</p>
      </Panel>
    </Section>
  );
}
