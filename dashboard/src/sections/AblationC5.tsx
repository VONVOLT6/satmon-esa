import { Chip, Explain, Panel, Section } from "../components/ui";
import { ci, contrib, pct, signed } from "../data/contributions";
import { fmt } from "../data/model";
import type { SectionProps } from "./types";

const { c5 } = contrib;
const ROWS: [string, string][] = [
  ["A0", "Always-on LSTM (baseline)"], ["A1", "Cheap tier alone, no LSTM"], ["A2", "C1, confidence only"], ["A3", "C1, confidence + impact"],
  ["A4", "C1 full"], ["A5", "C1 + C2"], ["A6", "C1 + C3"], ["A7", "C1 + C4"], ["A8", "Full system C1 + C2 + C3 + C4"],
  ["L1", "A8 minus C1"], ["L2", "A8 minus C2"], ["L3", "A8 minus C3"], ["L4", "A8 minus C4, C2 rule instead"],
];
const COLS: [keyof (typeof c5.table)["A0"], string, (v: number) => string, "high" | "low"][] = [
  ["F05_injected", "F0.5 faulty", (v) => fmt(v), "high"],
  ["F05_clean", "F0.5 clean", (v) => fmt(v), "high"],
  ["lstm_share_injected", "LSTM share", (v) => pct(v), "low"],
  ["items", "Operator items", (v) => v.toFixed(1), "low"],
  ["cost_central", "Cost", (v) => v.toFixed(1), "low"],
  ["mean_regret", "Mean regret", (v) => v.toFixed(1), "low"],
];

export default function AblationC5({ id, letter, title }: SectionProps) {
  const best = (key: (typeof COLS)[number][0], dir: "high" | "low") => {
    const vals = ROWS.map(([k]) => c5.table[k][key]);
    return dir === "high" ? Math.max(...vals) : Math.min(...vals);
  };
  return (
    <Section id={id} letter={letter} title={title} subtitle="All contributions in one pipeline, each switched off in turn. Faulty test period with 50 injected sensor faults, 5 seeds.">
      <Panel title="Every configuration" eyebrow="Best value in each column highlighted" right={<Chip tone="violet">Descriptive</Chip>}>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] text-sm">
            <thead>
              <tr className="border-b border-cyan-300/10 text-left text-xs uppercase tracking-wide text-slate-400">
                <th className="py-2 pr-3">Config</th><th className="py-2 pr-3">What is on</th>
                {COLS.map(([, h]) => <th key={h} className="py-2 pr-3 text-right">{h}</th>)}
              </tr>
            </thead>
            <tbody>
              {ROWS.map(([k, name]) => (
                <tr key={k} className={`border-b border-white/5 ${k === "L4" ? "bg-cyan-400/10" : ""}`}>
                  <td className="py-2 pr-3 font-mono text-cyan-300">{k}</td>
                  <td className={`py-2 pr-3 ${k === "L4" ? "font-semibold text-white" : "text-slate-300"}`}>{name}</td>
                  {COLS.map(([key, h, f, dir]) => {
                    const v = c5.table[k][key];
                    const isBest = Math.abs(v - best(key, dir)) < 1e-9;
                    return <td key={h} className={`py-2 pr-3 text-right font-mono ${isBest ? "text-emerald-300" : "text-slate-200"}`}>{f(v)}</td>;
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-slate-400">
          Interval for the full system against the baseline on faulty data: F0.5 {signed(c5.bootstrap["A8-A0"].diff)}, 95% CI {ci(c5.bootstrap["A8-A0"])}.
          Keeping the learned C4 instead of the C2 rule costs {signed(c5.bootstrap["cost_A8-L4"].diff, 1)}, CI {ci(c5.bootstrap["cost_A8-L4"], 1)}.
          Operator items are alarms, or incidents when C3 groups alarms within {c5.W.toLocaleString("en-US")} samples. No incident mixed a real event with a sensor fault.
        </p>
        <Explain
          shows="Each row is a system built from a different set of contributions, scored on detection, compute, operator load and response cost."
          means={<>C2 alone provides the detection gain. C1 removes {pct(1 - c5.table.L4.lstm_share_injected, 0)} of the LSTM work at no cost. Time grouping (C3) trims operator items. The learned C4 policy makes things worse, so the best combination is <b>L4</b>: C1 + C2 + time grouping + the simple C2 ticket rule.</>}
        />
      </Panel>
    </Section>
  );
}
