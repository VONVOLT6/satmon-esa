// Typed access to contributions.json (built by ../src/export_contributions_data.py from results/*.json).
import raw from "./contributions.json";

export interface Boot { A: number; B: number; diff: number; ci95: [number, number] }
interface DevPoint { tau: number; f05: number; share: number }
interface GroupStats { precision: number; recall: number; f1: number; incidents_per_event: number; mixed_incidents: number; incidents_total: number }
export interface AblationRow {
  F05_injected: number; F05_clean: number; lstm_share_injected: number; lstm_share_clean: number;
  alarms: number; items: number; mixed_items: number; cost_central: number; mean_regret: number;
}

export const contrib = raw as unknown as {
  h1: { floors: number[]; dev_f05: number[]; chosen: number; baseline: number; test: { all: Boot; anomaly: Boot }; verdict: string };
  c1: {
    taus: number[]; chosen: number; baseline_dev: number; dev: Record<"full" | "conf+impact" | "conf", DevPoint[]>;
    test: Boot; test_anomaly: Boot; lstm_share: number; verdict: string;
    timing: { full_lstm_seconds: number; routed_lstm_seconds: number; cheap_tier_seconds: number; bursts: number; speedup_total: number };
  };
  c2: {
    injected: Boot; clean: Boot; threshold: number; dev_sensor_f1: number; verdicts: { H3a: string; H3b: string };
    attribution: Record<string, { faults: number; alarmed: number; attributed: number }>;
  };
  c3: { methods: Record<"G0" | "G1" | "G2" | "G2_subsystem_prior", GroupStats>; chosen: { G1: { W: number }; G2: { W: number; rho: number } };
        alerts: number; labelled_alerts: number; bootstrap: { diff: number; ci95: [number, number] }; verdict: string };
  c4: {
    central: Record<"P0" | "P1" | "P2" | "oracle", number>; ranked_first: Record<"P0" | "P1" | "P2", number>;
    P2_beats_both: number; grid_points: number; wins_by_D: Record<string, Record<"P0" | "P1" | "P2", number>>;
    bootstrap: { P1: number; P2: number; diff: number; ci95: [number, number] }; alarm_classes: Record<string, number>; verdict: string;
  };
  c5: { configs: Record<string, [string, boolean, boolean, boolean]>; table: Record<string, AblationRow>; bootstrap: Record<string, Boot>; W: number };
  h6: {
    chosen: { p: number; W: number; f_r: number }; gate_share: number; dev_B0: number; dev_chosen: number;
    test: { all: Boot; anomaly: Boot; rare: Boot };
    caught: Record<"B0" | "R", { rare: number; anomaly: number; false_alarms: number }>;
    gate_alone: { rare: number; anomaly: number; false_alarms: number; f05: number }; verdict: string;
  };
};

export const signed = (x: number, digits = 3) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(digits)}`;
export const ci = (b: { ci95: [number, number] }, digits = 3) => `[${signed(b.ci95[0], digits)}, ${signed(b.ci95[1], digits)}]`;
export const pct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`;
