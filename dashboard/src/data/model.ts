// Typed access to results.json. Every number the dashboard shows is read or derived here,
// so components never hold their own copies of experimental values.
import raw from "./results.json";

export type Dataset = "21_months" | "84_months";
export type Model = "Telemanom-ESA" | "Telemanom-ESA-Pruned";
export type Subset = "all" | "anomaly";
export type MetricKey = "EW_precision" | "EW_recall" | "EW_F_0.50" | "alarming_precision" | "AFF_F_0.50";

export interface Stats { mean: number; sd: number; min: number; max: number; values: number[] }
interface TrainingRun { epochs: number; train_seconds: number; best_val: number; device: string; torch: string; val_loss: number[]; train_loss: number[] }
interface EvalRun { predict_seconds: number | null; samples: number | null; seed?: number; [key: string]: number | null | undefined }
interface AlarmRow { dataset: Dataset; seed: number; model: string; "F0.5": number; alarms: number; false_alarms: number; events_found: number; median_alarm_min: number; flagged_hours: number }
interface ExamplePoint { t: string; actual: number; forecast: number | null; labelled: number; alarm: number }

const data = raw as unknown as {
  paper: Record<Dataset | "42_months", Record<Model, number>>;
  paper_84_detail: Record<Model, { EW_precision: number; EW_recall: number }>;
  seeds: number[];
  runs: Record<Dataset, Record<string, Record<Model, Record<Subset, Record<MetricKey, number>>>>>;
  summary: Record<string, Record<MetricKey, Stats>>;
  training: Record<Dataset, Record<string, TrainingRun>>;
  evaluation: Record<Dataset, Record<string, EvalRun>>;
  events_21_months_seed42_pruned: { Anomaly: { events: number; detected: number; missed: number }; "Rare Event": { events: number; detected: number; missed: number }; alarms_total: number; alarms_on_no_event: number };
  alarm_breakdown: AlarmRow[];
  dataset: { name: string; mission: string; channels: string[]; subsystem: string; sampling_seconds: number; train_rows: Record<Dataset, number>; train_periods: Record<Dataset, [string, string]>; test_period: [string, string]; test_rows: number; test_events: { Anomaly: number; "Rare Event": number } };
  hardware: { gpu: string; cpu: string; ram_gb: number; framework: string };
  example: { event: string; channel: string; category: string; start: string; end: string; points: ExamplePoint[] };
};

export const DATASETS: Dataset[] = ["21_months", "84_months"];
export const MODELS: Model[] = ["Telemanom-ESA", "Telemanom-ESA-Pruned"];
export const SEEDS = data.seeds;
export const METRICS: { key: MetricKey; label: string; short: string; help: string }[] = [
  { key: "EW_precision", label: "Event-wise precision", short: "Precision", help: "Of the alarms raised, the share that were real events, corrected for time spent in false alarm." },
  { key: "EW_recall", label: "Event-wise recall", short: "Recall", help: "Of the labelled events, the share the model found." },
  { key: "EW_F_0.50", label: "Event-wise F0.5", short: "F0.5", help: "The main score: combines precision and recall, weighting precision more, so false alarms cost more than misses." },
  { key: "alarming_precision", label: "Alarming precision", short: "Alarming prec.", help: "Penalises raising several separate alarms for one event. 1.0 means one alarm per detected event." },
  { key: "AFF_F_0.50", label: "Affiliation-based F0.5", short: "Affiliation F0.5", help: "Rewards alarms that fall close in time to the real event, even if not exactly on it." },
];

export const label = (d: Dataset) => (d === "21_months" ? "21 months" : "84 months");
export const modelLabel = (m: Model) => (m === "Telemanom-ESA" ? "Telemanom-ESA" : "Telemanom-ESA-Pruned");

export const paper = (d: Dataset, m: Model) => data.paper[d][m];
export const paper84Detail = data.paper_84_detail;
export const summary = (d: Dataset, m: Model, s: Subset = "all") => data.summary[`${d}|${m}|${s}`];
export const ours = (d: Dataset, m: Model, k: MetricKey = "EW_F_0.50", s: Subset = "all") => summary(d, m, s)[k];
export const seedScore = (d: Dataset, seed: number, m: Model, k: MetricKey = "EW_F_0.50", s: Subset = "all") =>
  data.runs[d][String(seed)][m][s][k];

export const r3 = (x: number) => Math.round(x * 1000) / 1000;
export const fmt = (x: number, digits = 3) => x.toFixed(digits);
export const signed = (x: number, digits = 3) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(digits);

// Acceptance band set before the runs: within ±0.03 of the published value counts as reproduced.
export const BAND = 0.03;
export function verdict(diff: number) {
  if (diff > 0.005) return { text: "Exceeded published", tone: "good" as const };
  if (diff >= -BAND) return { text: "Matched (within ±0.03)", tone: "good" as const };
  return { text: "Below published", tone: "warn" as const };
}

export const training = (d: Dataset) => SEEDS.map((s) => ({ seed: s, ...data.training[d][String(s)] }));
export const evaluation = (d: Dataset): (EvalRun & { seed: number })[] => SEEDS.map((s) => ({ ...data.evaluation[d][String(s)], seed: s }));
export const events = data.events_21_months_seed42_pruned;
export const alarmBreakdown = data.alarm_breakdown;
export const dataset = data.dataset;
export const hardware = data.hardware;
export const example = data.example;

export const totalTestEvents = dataset.test_events.Anomaly + dataset.test_events["Rare Event"];
