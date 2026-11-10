import { CartesianGrid, Line, LineChart, ReferenceDot, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Activity, Coins, Cpu, FlaskConical, GitBranch, Radio, ShieldCheck } from "lucide-react";
import { axisProps, ChartTooltip, Chip, COLORS, Explain, Panel, Section } from "../components/ui";
import { ci, contrib, pct, signed } from "../data/contributions";
import { fmt } from "../data/model";
import type { SectionProps } from "./types";

const { h1, c1, c2, c3, c4, c5, h6 } = contrib;
type Tone = "good" | "warn" | "muted" | "violet";

const CARDS: { code: string; name: string; icon: typeof Cpu; status: string; tone: Tone; headline: string; detail: string; where: string }[] = [
  {
    code: "C1", name: "Adaptive computation", icon: Cpu, status: "Confirmed", tone: "good",
    headline: `${c1.timing.speedup_total.toFixed(1)}× faster, same F0.5`,
    detail: `A cheap screener watches all telemetry; the LSTM runs on ${pct(c1.lstm_share)} of the test data. F0.5 ${fmt(c1.test.B)} vs ${fmt(c1.test.A)}.`,
    where: "Section I",
  },
  {
    code: "C2", name: "Sensor reliability", icon: ShieldCheck, status: "Confirmed", tone: "good",
    headline: `F0.5 ${fmt(c2.injected.A)} → ${fmt(c2.injected.B)}`,
    detail: `With 50 injected sensor faults, broken-sensor alarms are moved out of the system stream (${signed(c2.injected.diff)}, 95% CI ${ci(c2.injected)}). No harm on clean data.`,
    where: "Section J",
  },
  {
    code: "C3", name: "Incident correlation", icon: GitBranch, status: "Not supported", tone: "muted",
    headline: `${c3.methods.G0.incidents_per_event.toFixed(1)} → ${c3.methods.G1.incidents_per_event.toFixed(2)} alerts per event`,
    detail: `Time grouping works on all 76 channels; the proposed learned graph was worse (pairwise F1 ${fmt(c3.methods.G2.f1)} vs ${fmt(c3.methods.G1.f1)}).`,
    where: "Section K",
  },
  {
    code: "C4", name: "Cost-aware response", icon: Coins, status: "Not supported", tone: "muted",
    headline: `Cost ${c4.central.P0.toFixed(1)} → ${c4.central.P1.toFixed(1)} with the C2 rule`,
    detail: `The learned policy wins only ${c4.P2_beats_both} of ${c4.grid_points} cost settings; the simple C2 ticket rule beats paging every alarm at all of them.`,
    where: "Section L",
  },
  {
    code: "C5", name: "Ablation", icon: FlaskConical, status: "Descriptive", tone: "violet",
    headline: "Best: C1 + C2 + time grouping + C2 rule",
    detail: `On faulty data: F0.5 ${fmt(c5.table.A0.F05_injected)} → ${fmt(c5.table.L4.F05_injected)}, LSTM on ${pct(c5.table.L4.lstm_share_injected)}, cost ${c5.table.A0.cost_central.toFixed(1)} → ${c5.table.L4.cost_central.toFixed(1)}.`,
    where: "Section M",
  },
  {
    code: "H6", name: "Rare events", icon: Radio, status: "Not confirmed", tone: "warn",
    headline: `Rare events caught ${h6.caught.B0.rare.toFixed(0)} → ${h6.caught.R.rare.toFixed(1)} of 36`,
    detail: `High-priority telecommands open a window where the LSTM alarms at a lower floor. Overall F0.5 ${signed(h6.test.all.diff)}, 95% CI ${ci(h6.test.all)} includes 0.`,
    where: "Section N",
  },
  {
    code: "H1", name: "Calibrated error floor", icon: Activity, status: "Not confirmed", tone: "warn",
    headline: `F0.5 ${fmt(h1.test.all.A)} → ${fmt(h1.test.all.B)}`,
    detail: `Floor ${h1.chosen} chosen on development data instead of the paper's ${h1.baseline}. Difference ${signed(h1.test.all.diff)}, 95% CI ${ci(h1.test.all)} includes 0.`,
    where: "Below",
  },
];

export default function Contributions({ id, letter, title }: SectionProps) {
  const floorRows = h1.floors.map((f, i) => ({ floor: f, "Development F0.5": h1.dev_f05[i] }));
  return (
    <Section id={id} letter={letter} title={title} subtitle="Every contribution was designed and tuned on development data, then run once on the 2007-2013 test set (PROTOCOL.md).">
      <div className="rounded-xl border border-cyan-300/20 bg-cyan-400/5 p-4 text-sm leading-relaxed text-cyan-50/90">
        <b>How to read the verdicts.</b> "Confirmed" means the 95% confidence interval of the change against the baseline lies entirely on the good side, using a paired
        bootstrap over test months and 5 training seeds. "Not confirmed" means the point estimate is better but the interval includes zero. "Not supported" means the
        proposed method did not beat a simpler alternative. Negative results are reported, not hidden.
      </div>

      <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
        {CARDS.map(({ code, name, icon: Icon, status, tone, headline, detail, where }) => (
          <Panel key={code} className="flex flex-col">
            <div className="mb-3 flex items-start justify-between gap-2">
              <div className="flex items-center gap-3">
                <span className="grid h-10 w-10 place-items-center rounded-xl border border-cyan-300/25 bg-cyan-400/10"><Icon className="h-5 w-5 text-cyan-300" /></span>
                <div>
                  <div className="font-mono text-xs text-cyan-300/70">{code}</div>
                  <div className="text-base font-semibold text-white">{name}</div>
                </div>
              </div>
              <Chip tone={tone}>{status}</Chip>
            </div>
            <div className="font-mono text-lg font-semibold text-slate-100">{headline}</div>
            <p className="mt-2 flex-1 text-sm leading-relaxed text-slate-300">{detail}</p>
            <div className="mt-3 text-xs text-slate-500">{where}</div>
          </Panel>
        ))}
      </div>

      <Panel title="Calibrated error floor (H1)" eyebrow="Development period, mean of 5 seeds">
        <div className="h-[260px]">
          <ResponsiveContainer>
            <LineChart data={floorRows} margin={{ top: 12, right: 20, left: 0, bottom: 4 }}>
              <CartesianGrid vertical={false} stroke={COLORS.grid} />
              <XAxis dataKey="floor" {...axisProps} label={{ value: "min_error_value floor", position: "insideBottomRight", offset: -2, fill: COLORS.axis, fontSize: 12 }} />
              <YAxis {...axisProps} domain={[0, 1]} tickFormatter={(v: number) => v.toFixed(1)} />
              <Tooltip content={<ChartTooltip digits={3} />} />
              <Line dataKey="Development F0.5" stroke={COLORS.ours} strokeWidth={2} dot={{ r: 4 }} isAnimationActive={false} />
              <ReferenceDot x={h1.chosen} y={h1.dev_f05[h1.floors.indexOf(h1.chosen)]} r={7} fill={COLORS.good} stroke="none" label={{ value: "chosen 0.005", fill: COLORS.good, fontSize: 11, position: "top" }} />
              <ReferenceDot x={h1.baseline} y={h1.dev_f05[h1.floors.indexOf(h1.baseline)]} r={6} fill={COLORS.paper} stroke="none" label={{ value: "paper 0.007", fill: "#94a3b8", fontSize: 11, position: "bottom" }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <Explain
          shows="The detection score on the development period for each candidate error floor, the setting that discards small forecast errors before alarms are raised."
          means={<>0.005 scored best on every seed in development. On the test set it scored {fmt(h1.test.all.B)} against {fmt(h1.test.all.A)}, but with only 65 test events the interval {ci(h1.test.all)} includes zero, so the gain is not confirmed.</>}
        />
      </Panel>
    </Section>
  );
}
