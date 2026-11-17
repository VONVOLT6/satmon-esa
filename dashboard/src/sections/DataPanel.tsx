import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Layers } from "lucide-react";
import { axisProps, ChartTooltip, COLORS, Explain, Panel } from "../components/ui";
import raw from "../data/data_overview.json";

const data = raw as unknown as {
  channels: string[];
  splits: { name: string; period: string; rows: number; labelled_pct: number; anomalies: number; rare: number; use: string }[];
  preprocessing: {
    train_rows: number; labelled_removed: number; labelled_removed_pct: number; nominal_train_rows: number; nominal_validation_rows: number;
    validation_from: string; test_events_all_channels: number; test_events: number; anomalies_total: number; anomalies_under_2min: number; rare_median_minutes: number;
  };
  timeline: ({ t: string } & Record<string, number | string>)[];
  events: { start: string; end: string; category: string }[];
  duration_hist: { bin: string; Anomaly: number; "Rare event": number }[];
};

const n = (x: number) => x.toLocaleString("en-US");
const ms = (d: string) => new Date(d).getTime();
const LINE = ["#38e8ff", "#3b82f6", "#60a5fa", "#a78bfa", "#22d3ee", "#93c5fd"];
const ANOM = "#fb7185";
const RARE = "#a78bfa";

export default function DataPanel() {
  const p = data.preprocessing;
  const timeline = data.timeline.map((r) => ({ ...r, x: ms(r.t) }));
  const years = Array.from({ length: 8 }, (_, i) => ms(`${2000 + 2 * i}-01-01`));
  const steps: [string, string][] = [
    ["Resample", "Each sensor is put on ESA's 30-second grid by carrying the last value forward. Short labelled events are kept, so none is lost. Our row count matches ESA's official one exactly."],
    ["Label", "Every 30-second step of every sensor is marked normal, anomaly or rare event, from ESA engineers' annotations."],
    ["Clean training data", `The model learns normal behaviour only, so ${n(p.labelled_removed)} labelled training rows (${p.labelled_removed_pct.toFixed(1)}%) are removed. That leaves ${n(p.nominal_train_rows)} rows to train on and ${n(p.nominal_validation_rows)} to validate on (the last three months).`],
    ["Scale", "Each sensor is standardised with the training mean and standard deviation, so all six share one scale."],
    ["Window", "The LSTM reads 250 steps (about 2 hours) and predicts the next 10. Windows are cut only inside clean stretches."],
  ];
  return (
    <Panel eyebrow="Data" title="Data splits and preprocessing" right={<Layers className="h-5 w-5 text-cyan-300/70" />}>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] text-sm">
          <thead>
            <tr className="border-b border-cyan-300/10 text-left text-xs uppercase tracking-wide text-slate-400">
              <th className="py-2 pr-3">Split</th><th className="py-2 pr-3">Period</th><th className="py-2 pr-3 text-right">Rows</th>
              <th className="py-2 pr-3 text-right">Labelled</th><th className="py-2 pr-3 text-right">Anomalies</th><th className="py-2 pr-3 text-right">Rare events</th><th className="py-2 pr-3">Used for</th>
            </tr>
          </thead>
          <tbody>
            {data.splits.map((s) => (
              <tr key={s.name} className="border-b border-white/5">
                <td className="py-2 pr-3 text-slate-100">{s.name}</td>
                <td className="py-2 pr-3 font-mono text-xs text-slate-300">{s.period}</td>
                <td className="py-2 pr-3 text-right font-mono text-slate-200">{n(s.rows)}</td>
                <td className="py-2 pr-3 text-right font-mono text-slate-200">{s.labelled_pct.toFixed(1)}%</td>
                <td className="py-2 pr-3 text-right font-mono text-slate-200">{s.anomalies}</td>
                <td className="py-2 pr-3 text-right font-mono text-slate-200">{s.rare}</td>
                <td className="py-2 pr-3 text-slate-400">{s.use}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-slate-500">The 84-month training period overlaps the development period, so all tuning uses the 21-month models, which never see development data.</p>

      <ol className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-5">
        {steps.map(([t, d], i) => (
          <li key={t} className="rounded-xl border border-cyan-300/10 bg-slate-900/40 p-3 text-sm">
            <div className="mb-1 flex items-center gap-2 font-semibold text-slate-100"><span className="grid h-5 w-5 place-items-center rounded-full bg-cyan-400/15 font-mono text-[10px] text-cyan-200">{i + 1}</span>{t}</div>
            <p className="leading-relaxed text-slate-400">{d}</p>
          </li>
        ))}
      </ol>

      <div className="mt-5 grid gap-5 xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <div>
          <div className="mb-1 text-sm font-semibold text-slate-200">The six sensors over 14 years, labelled events shaded</div>
          <div className="h-[280px]">
            <ResponsiveContainer>
              <LineChart data={timeline} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
                <CartesianGrid vertical={false} stroke={COLORS.grid} />
                <XAxis dataKey="x" type="number" scale="time" domain={["dataMin", "dataMax"]} ticks={years}
                  tickFormatter={(v: number) => String(new Date(v).getFullYear())} {...axisProps} />
                <YAxis {...axisProps} width={40} tickFormatter={(v: number) => v.toFixed(1)} domain={[-1.5, 1.5]} allowDataOverflow
                  label={{ value: "weekly mean, z-score", angle: -90, position: "insideLeft", fill: COLORS.axis, fontSize: 11 }} />
                <Tooltip content={<ChartTooltip digits={2} />} labelFormatter={(v: number) => new Date(v).toISOString().slice(0, 10)} />
                {data.events.map((e, i) => (
                  <ReferenceArea key={i} x1={ms(e.start)} x2={ms(e.end)} fill={e.category === "Anomaly" ? ANOM : RARE} fillOpacity={0.25} stroke="none" ifOverflow="extendDomain" />
                ))}
                <ReferenceLine x={ms("2001-10-01")} stroke="#94a3b8" strokeDasharray="4 4" label={{ value: "dev", fill: "#94a3b8", fontSize: 11, position: "insideTopRight" }} />
                <ReferenceLine x={ms("2007-01-01")} stroke="#94a3b8" strokeDasharray="4 4" label={{ value: "test", fill: "#94a3b8", fontSize: 11, position: "insideTopRight" }} />
                {data.channels.map((c, i) => <Line key={c} dataKey={c} stroke={LINE[i]} dot={false} strokeWidth={1} isAnimationActive={false} />)}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-1 flex gap-4 text-xs text-slate-400">
            <span className="flex items-center gap-1.5"><span className="h-2.5 w-4 rounded-sm" style={{ background: ANOM, opacity: 0.5 }} />Anomaly</span>
            <span className="flex items-center gap-1.5"><span className="h-2.5 w-4 rounded-sm" style={{ background: RARE, opacity: 0.5 }} />Rare event</span>
            <span>Lines: channels 41-46</span>
          </div>
        </div>
        <div>
          <div className="mb-1 text-sm font-semibold text-slate-200">How long events last, 2000-2013</div>
          <div className="h-[280px]">
            <ResponsiveContainer>
              <BarChart data={data.duration_hist} margin={{ top: 8, right: 8, left: 0, bottom: 4 }}>
                <CartesianGrid vertical={false} stroke={COLORS.grid} />
                <XAxis dataKey="bin" {...axisProps} interval={0} tick={{ fill: COLORS.axis, fontSize: 10 }} />
                <YAxis {...axisProps} width={30} allowDecimals={false} />
                <Tooltip content={<ChartTooltip digits={0} unit=" events" />} />
                <Legend />
                <Bar dataKey="Anomaly" stackId="d" fill={ANOM} maxBarSize={40} />
                <Bar dataKey="Rare event" stackId="d" fill={RARE} maxBarSize={40} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
      <Explain
        shows="Left: what the raw telemetry looks like across the whole mission, with every labelled event marked. Right: how long the labelled events last."
        means={<>Labelled data is scarce (under 3% of rows), and the sensors drift and shift level over the years. {p.anomalies_under_2min} of {p.anomalies_total} anomalies last under 2 minutes, while rare events last about {(p.rare_median_minutes / 60).toFixed(1)} hours (median), which is why the model catches anomalies far more easily. {p.test_events_all_channels} of {p.test_events} test events move all six sensors together, the basis of C2's broken-sensor check.</>}
      />
    </Panel>
  );
}
