import { useEffect, useState } from "react";
import { BarChart3, Satellite } from "lucide-react";
import MissionControl from "./views/MissionControl";
import Research from "./views/Research";

type View = "mission" | "research";
const viewFromHash = (): View => (window.location.hash.startsWith("#/research") ? "research" : "mission");

export default function App() {
  const [view, setView] = useState<View>(viewFromHash);

  useEffect(() => {
    const onHash = () => setView(viewFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const go = (v: View) => {
    window.location.hash = v === "research" ? "#/research" : "#/mission";
    window.scrollTo({ top: 0 });
  };

  return (
    <div className="relative min-h-screen">
      <div className="stars" />
      <header className="fixed inset-x-0 top-0 z-40 border-b border-cyan-300/10 bg-space-950/70 backdrop-blur-xl">
        <div className="mx-auto flex h-14 max-w-[1500px] items-center justify-between px-4 sm:px-6">
          <button onClick={() => go("mission")} className="flex items-center gap-2.5 text-left">
            <span className="grid h-8 w-8 place-items-center rounded-lg border border-cyan-300/30 bg-cyan-400/10">
              <Satellite className="h-4 w-4 text-cyan-300" />
            </span>
            <span className="leading-tight">
              <span className="block font-display text-sm tracking-[0.18em] text-white">SATMON</span>
              <span className="block font-mono text-[10px] text-cyan-300/60">ESA-ADB · Telemetry anomaly research</span>
            </span>
          </button>
          <nav className="flex items-center gap-1 rounded-xl border border-cyan-300/10 bg-slate-900/50 p-1">
            {([
              ["mission", "Mission Control", Satellite],
              ["research", "Research Results", BarChart3],
            ] as const).map(([v, text, Icon]) => (
              <button
                key={v}
                onClick={() => go(v)}
                className={`flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs font-medium transition sm:text-sm ${
                  view === v ? "bg-cyan-400/15 text-cyan-200 shadow-[0_0_20px_-6px_rgba(56,232,255,.6)]" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <Icon className="h-4 w-4" />
                <span className="hidden sm:inline">{text}</span>
              </button>
            ))}
          </nav>
        </div>
      </header>
      <main className="relative z-10 pt-14">{view === "mission" ? <MissionControl onOpenResearch={() => go("research")} /> : <Research />}</main>
    </div>
  );
}
