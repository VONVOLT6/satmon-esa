// Typed access to channel9.json: the teammate's Random Forest feature-engineering experiment
// (ESA-ADB Mission 2, channel_9, sample-level metrics). Kept apart from model.ts on purpose:
// these numbers must never be mixed with the event-wise Mission 1 results.
import raw from "./channel9.json";

export type ModelKey = "7" | "10" | "13";
export interface RfModel { name: string; features: number; feature_names: string[]; threshold: number; precision: number; recall: number; f1: number; pr_auc: number; fpr: number; tp: number; fp: number; tn: number; fn: number; test_observations: number }
interface Scores { precision: number; recall: number; f1: number; fpr: number; tp: number; fp: number; tn: number; fn: number }
interface RoutingRow { model: string; margin: number | null; escalation_pct: number | null; precision: number; recall: number; f1: number; fpr: number }

const data = raw as unknown as {
  source: { repo: string; commit: string; folder: string };
  experiment: { dataset: string; split: string; threshold_selection: string; maximum_false_positive_rate: number; batch_size: number; test_observations: number; test_anomalies: number };
  models: Record<ModelKey, RfModel>;
  routing_validation: RoutingRow[];
  adaptive_test: { margin: number; escalated: number; escalation_pct: number; baseline_7: Scores; always_13: Scores; hybrid: Scores };
  runtime_seconds: { baseline_7_feature: number; always_13_feature: number; adaptive_hybrid: number };
  team_reported: {
    source: string;
    incident_detection: { total_incidents: number; "10": { detected: number; missed: number }; "13": { detected: number; missed: number }; note: string };
    model_size_mb: Record<ModelKey, number>;
    prediction_latency_ms: Record<ModelKey, number>;
    latency_unit: string;
    latency_note: string;
    not_provided: string[];
  };
};

export const RF = data;
export const MODEL_KEYS: ModelKey[] = ["7", "10", "13"];
export const MODEL_LABEL: Record<ModelKey, string> = { "7": "Original · 7 features", "10": "Expanded · 10 features", "13": "Extended · 13 features" };
export const MODEL_SHORT: Record<ModelKey, string> = { "7": "7 features", "10": "10 features", "13": "13 features" };
export const MODEL_COLOR: Record<ModelKey, string> = { "7": "#7c8db5", "10": "#3b82f6", "13": "#38e8ff" };
export const rf = (k: ModelKey) => data.models[k];

export type RfMetric = "precision" | "recall" | "f1" | "pr_auc" | "fpr";
export const RF_METRICS: { key: RfMetric; label: string; higherIsBetter: boolean }[] = [
  { key: "precision", label: "Precision", higherIsBetter: true },
  { key: "recall", label: "Recall", higherIsBetter: true },
  { key: "f1", label: "F1-score", higherIsBetter: true },
  { key: "pr_auc", label: "PR-AUC", higherIsBetter: true },
  { key: "fpr", label: "False-positive rate", higherIsBetter: false },
];

/** Change of a metric relative to the 7-feature original: absolute difference and percent. */
export function change(metric: RfMetric, to: ModelKey, from: ModelKey = "7") {
  const a = rf(from)[metric];
  const b = rf(to)[metric];
  const diff = b - a;
  const pct = (diff / a) * 100;
  const better = RF_METRICS.find((m) => m.key === metric)!.higherIsBetter ? diff > 0 : diff < 0;
  return { from: a, to: b, diff, pct, better };
}

export const pct = (x: number, digits = 1) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(digits)}%`;
export const n0 = (x: number) => Math.round(x).toLocaleString("en-US");

export const TEAM = data.team_reported;
