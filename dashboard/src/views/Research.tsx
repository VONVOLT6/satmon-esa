import { useEffect, useState } from "react";
import Overview from "../sections/Overview";
import PaperVsOurs from "../sections/PaperVsOurs";
import BaselineVsPruned from "../sections/BaselineVsPruned";
import Metrics from "../sections/Metrics";
import Events from "../sections/Events";
import Stability from "../sections/Stability";
import Compute from "../sections/Compute";
import Contributions from "../sections/Contributions";
import ExperimentalAnalysis from "../sections/ExperimentalAnalysis";
import Improvements from "../sections/Improvements";

type Group = { group?: string };
const SECTIONS: ({ id: string; letter: string; title: string; el: typeof Overview } & Group)[] = [
  { id: "overview", group: "ESA-ADB reproduction · Mission 1", letter: "A", title: "Research overview", el: Overview },
  { id: "paper", letter: "B", title: "Paper vs our reproduction", el: PaperVsOurs },
  { id: "pruned", letter: "C", title: "Baseline vs Pruned", el: BaselineVsPruned },
  { id: "metrics", letter: "D", title: "Performance metrics", el: Metrics },
  { id: "events", letter: "E", title: "Anomaly detection analysis", el: Events },
  { id: "stability", letter: "F", title: "Training stability", el: Stability },
  { id: "compute", letter: "G", title: "Training and compute", el: Compute },
  { id: "contributions", letter: "H", title: "Proposed contributions", el: Contributions },
  { id: "experiments", group: "Feature experiments · Mission 2", letter: "I", title: "Experimental analysis", el: ExperimentalAnalysis },
  { id: "improvements", letter: "J", title: "Improvements achieved", el: Improvements },
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
