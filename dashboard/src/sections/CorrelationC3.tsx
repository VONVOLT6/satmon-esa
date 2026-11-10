import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { axisProps, ChartTooltip, COLORS, Explain, Panel, Section, Stat } from "../components/ui";
import { contrib } from "../data/contributions";
import { fmt } from "../data/model";
import type { SectionProps } from "./types";

const { c3 } = contrib;
const ROWS = [
  ["G0", "G0 no grouping", COLORS.paper],
  ["G1", "G1 time only", COLORS.ours],
  ["G2", "G2 learned graph (proposed)", COLORS.warn],
  ["G2_subsystem_prior", "G2 with known subsystems", COLORS.unpruned],
] as const;

export default function CorrelationC3({ id, letter, title }: SectionProps) {
  const rows = ROWS.map(([k, name, color]) => ({ name, color, load: c3.methods[k].incidents_per_event, f1: c3.methods[k].f1 }));
  return (
    <Section id={id} letter={letter} title={title} subtitle="All 76 Mission 1 channels: merge the alerts of one incident so operators see one item, not dozens.">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="Channel alerts, test period" value={c3.alerts.toLocaleString("en-US")} sub={`${c3.labelled_alerts.toLocaleString("en-US")} on labelled events`} />
        <Stat label="Alerts per event, ungrouped" value={c3.methods.G0.incidents_per_event.toFixed(1)} sub="what an operator faces today" tone="warn" />
        <Stat label="With time grouping (G1)" value={c3.methods.G1.incidents_per_event.toFixed(2)} sub={`pairwise F1 ${fmt(c3.methods.G1.f1)}`} tone="good" />
        <Stat label="Learned graph G2 − G1" value={`${c3.bootstrap.diff.toFixed(3)} F1`} sub={`95% CI [${c3.bootstrap.ci95[0].toFixed(3)}, ${c3.bootstrap.ci95[1].toFixed(3)}]: worse`} />
      </div>
      <div className="grid gap-5 xl:grid-cols-2">
        {([["load", "Operator load", "Incidents per labelled event · lower is better", 1], ["f1", "Grouping correctness", "Pairwise F1 against ESA event IDs · higher is better", 3]] as const).map(([key, t, eyebrow, digits]) => (
          <Panel key={key} title={t} eyebrow={eyebrow}>
            <div className="h-[260px]">
              <ResponsiveContainer>
                <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 48, left: 20, bottom: 4 }}>
                  <CartesianGrid horizontal={false} stroke={COLORS.grid} />
                  <XAxis type="number" {...axisProps} />
                  <YAxis type="category" dataKey="name" {...axisProps} width={170} />
                  <Tooltip content={<ChartTooltip digits={digits} />} />
                  <Bar dataKey={key} name={t} maxBarSize={30} radius={[0, 5, 5, 0]}>
                    {rows.map((r) => <Cell key={r.name} fill={r.color} />)}
                    <LabelList dataKey={key} position="right" formatter={(v: number) => v.toFixed(digits)} fill="#cbd5e1" fontSize={11} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Panel>
        ))}
      </div>
      <Panel>
        <Explain
          shows={<>Cheap per-channel detectors raise alerts on all 76 channels. They are grouped by time alone (G1, W = {c3.chosen.G1.W} samples), by time plus a dependency graph learned from training data (G2, W = {c3.chosen.G2.W}, ρ = {c3.chosen.G2.rho}), or by time plus ESA's subsystem list.</>}
          means="Time grouping already cuts 30 alerts per event to 1.5 with almost no mistakes, because separate incidents rarely overlap in time. The learned graph never mixes two events but splits most real incidents apart: training data has too few incidents to learn reliable links. The known subsystem structure does better than the learned one."
        />
      </Panel>
    </Section>
  );
}
