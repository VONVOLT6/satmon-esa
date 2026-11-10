import { useEffect, useState } from "react";
import Overview from "../sections/Overview";
import PaperVsOurs from "../sections/PaperVsOurs";
import BaselineVsPruned from "../sections/BaselineVsPruned";
import Metrics from "../sections/Metrics";
import Events from "../sections/Events";
import Stability from "../sections/Stability";
import Compute from "../sections/Compute";
import Contributions from "../sections/Contributions";
import AdaptiveC1 from "../sections/AdaptiveC1";
import ReliabilityC2 from "../sections/ReliabilityC2";
import CorrelationC3 from "../sections/CorrelationC3";
import ResponseC4 from "../sections/ResponseC4";
import AblationC5 from "../sections/AblationC5";
import RareEvents from "../sections/RareEvents";

type Group = { group?: string };
const SECTIONS: ({ id: string; letter: string; title: string; el: typeof Overview } & Group)[] = [
  { id: "overview", group: "ESA-ADB reproduction · Mission 1", letter: "A", title: "Research overview", el: Overview },
  { id: "paper", letter: "B", title: "Paper vs our reproduction", el: PaperVsOurs },
  { id: "pruned", letter: "C", title: "Baseline vs Pruned", el: BaselineVsPruned },
  { id: "metrics", letter: "D", title: "Performance metrics", el: Metrics },
  { id: "events", letter: "E", title: "Anomaly detection analysis", el: Events },
  { id: "stability", letter: "F", title: "Training stability", el: Stability },
  { id: "compute", letter: "G", title: "Training and compute", el: Compute },
  { id: "contributions", group: "Our contributions · Mission 1", letter: "H", title: "Contributions at a glance", el: Contributions },
  { id: "c1", letter: "I", title: "C1 Adaptive computation", el: AdaptiveC1 },
  { id: "c2", letter: "J", title: "C2 Sensor reliability", el: ReliabilityC2 },
  { id: "c3", letter: "K", title: "C3 Incident correlation", el: CorrelationC3 },
  { id: "c4", letter: "L", title: "C4 Cost-aware response", el: ResponseC4 },
  { id: "c5", letter: "M", title: "C5 Ablation", el: AblationC5 },
  { id: "rare", letter: "N", title: "Rare events (H6)", el: RareEvents },
];

export default function Research() {
  const [active, setActive] = useState(SECTIONS[0].id);

  useEffect(() => {
    // The active section is the last one whose top has scrolled past the header;
    // at the very bottom of the page it is the last section.
    const onScroll = () => {
      const atBottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4;
      let current = SECTIONS[0].id;
      for (const s of SECTIONS) {
        const el = document.getElementById(s.id);
        if (el && el.getBoundingClientRect().top <= 140) current = s.id;
      }
      setActive(atBottom ? SECTIONS[SECTIONS.length - 1].id : current);
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const jump = (id: string) => document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });

  return (
    <div className="mx-auto grid max-w-[1500px] gap-6 px-4 py-6 sm:px-6 lg:grid-cols-[220px_minmax(0,1fr)]">
      <aside className="sticky top-14 z-30 -mx-4 min-w-0 bg-space-950/85 px-4 py-2 backdrop-blur sm:-mx-6 sm:px-6 lg:top-20 lg:mx-0 lg:h-[calc(100vh-6rem)] lg:bg-transparent lg:p-0 lg:backdrop-blur-none">
        <div className="eyebrow mb-3 hidden lg:block">Research results</div>
        <nav className="flex gap-1 overflow-x-auto pb-1 lg:flex-col lg:overflow-visible">
          {SECTIONS.map((s) => [
            s.group && <div key={`${s.id}-group`} className="hidden px-3 pb-1 pt-3 text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500 first:pt-0 lg:block">{s.group}</div>,
            <button
              key={s.id}
              onClick={() => jump(s.id)}
              className={`flex shrink-0 items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition ${
                active === s.id ? "bg-cyan-400/10 text-cyan-100" : "text-slate-400 hover:bg-white/5 hover:text-slate-200"
              }`}
            >
              <span className={`font-mono text-xs ${active === s.id ? "text-cyan-300" : "text-slate-500"}`}>{s.letter}</span>
              {s.title}
            </button>,
          ])}
        </nav>
        <p className="mt-6 hidden text-xs leading-relaxed text-slate-500 lg:block">
          Every value on this page is read from the saved experiment outputs in <span className="font-mono text-slate-400">results/</span> and{" "}
          <span className="font-mono text-slate-400">models/</span>. Published values come from the ESA-ADB paper (Kotowski et al., 2024).
        </p>
      </aside>
      <div className="min-w-0 space-y-14 pb-20">
        {SECTIONS.map(({ id, letter, title, group, el: Content }) => (
          <div key={id} className="space-y-14">
            {group && id !== SECTIONS[0].id && (
              <div className="flex items-center gap-4 pt-6">
                <div className="h-px flex-1 bg-gradient-to-r from-transparent via-violet-300/30 to-transparent" />
                <span className="eyebrow !text-violet-300/80">{group}</span>
                <div className="h-px flex-1 bg-gradient-to-r from-transparent via-violet-300/30 to-transparent" />
              </div>
            )}
            <Content id={id} letter={letter} title={title} />
          </div>
        ))}
      </div>
    </div>
  );
}
