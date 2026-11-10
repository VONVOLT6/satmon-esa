"""Charts for contributions C1-C5, built from saved result files only (no recomputation).

Usage: python src/make_contribution_figures.py           (results/figures/, with titles)
       python src/make_contribution_figures.py --paper   (paper/figures/, no titles: the paper's captions carry them)
Writes 7_c1_adaptive_computation.png ... 12_rare_events.png
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thresholding import ROOT

PAPER = "--paper" in sys.argv
OUT = ROOT / ("paper" if PAPER else "results") / "figures"
RES = ROOT / "results"
PROPOSED = "#2a78d6"   # categorical slot 1: proposed / ours
SECOND = "#eb6834"     # categorical slot 2
BASE = "#a3a29d"       # neutral: baseline / comparison
FAINT = "#dedcd6"      # neutral light: "nothing happened"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": False, "legend.frameon": False})


def suptitle(fig, text, **kw):
    if not PAPER:
        fig.suptitle(text, **kw)


def load(name):
    return json.loads((RES / name).read_text())


def label_bars(ax, bars, fmt, inside=False, dx=0.0):
    for b in bars:
        w = b.get_width()
        ax.text(w + dx, b.get_y() + b.get_height() / 2, fmt(w), va="center", ha="left", fontsize=9, color=INK)


def fig_c1():
    dev, h2 = load("c1_dev.json"), load("h2_result.json")
    taus = dev["taus"]
    share = [dev["summary"][f"{t}|full"]["lstm_share"] * 100 for t in taus]
    f05 = [dev["summary"][f"{t}|full"]["F0.5"] for t in taus]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})

    a.plot(share, f05, color=PROPOSED, lw=2, marker="o", ms=8, mec="white", mew=2, zorder=3, label="C1 router, development period")
    a.axhline(dev["baseline_dev_F0.5"], color=BASE, lw=1.5, ls="--", zorder=2)
    a.text(1.9, dev["baseline_dev_F0.5"] + 0.003, "always-on LSTM", ha="left", va="bottom", fontsize=9, color=INK2)
    pts = dict(zip(taus, zip(share, f05)))
    a.annotate("τ = 0.999 (chosen)", pts[0.999], xytext=(6, -18), textcoords="offset points", ha="left",
               fontsize=9, fontweight="bold", color=INK)
    a.annotate("τ = 0.998", pts[0.998], xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9, color=INK2)
    a.annotate("τ = 0.995", pts[0.995], xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9, color=INK2)
    a.annotate("τ = 0.9995 and 0.9999", pts[0.9995], xytext=(6, -16), textcoords="offset points", ha="left",
               fontsize=9, color=INK2)
    a.set_xscale("log")
    a.set_xticks([2, 5, 10, 20, 50, 100], ["2%", "5%", "10%", "20%", "50%", "100%"])
    a.set_xlabel("Share of data the LSTM runs on (log scale)")
    a.set_ylabel("Event-wise F0.5")
    a.set_ylim(0.70, 0.80)
    a.set_title("Development: detection score vs LSTM work", loc="left", fontsize=11)

    t = h2["timing_seed42"]
    a_bar = b.barh([1], [t["full_lstm_seconds"]], color=BASE, height=0.5)
    b.barh([0], [t["routed_lstm_seconds"]], color=PROPOSED, height=0.5, label="LSTM on bursts")
    b.barh([0], [t["cheap_tier_seconds"]], left=[t["routed_lstm_seconds"] + 4], color=SECOND, height=0.5,
           label="Cheap screening tier")
    b.text(t["full_lstm_seconds"] + 10, 1, f"{t['full_lstm_seconds']:.0f} s", va="center", fontsize=9)
    total = t["routed_lstm_seconds"] + t["cheap_tier_seconds"]
    b.text(total + 18, 0, f"{total:.0f} s  ({t['speedup_total']:.1f}× faster)", va="center", fontsize=9, fontweight="bold")
    b.set_yticks([1, 0], ["Always-on LSTM", "C1 adaptive"])
    b.set_xlabel("Inference time on the 7-year test period, seed 42 (s)")
    b.set_xlim(0, 1000)
    b.legend(loc="lower right", fontsize=9)
    b.set_title(f"Test: same F0.5 ({h2['bootstrap']['all']['B']:.3f}), LSTM on {h2['mean_lstm_share'] * 100:.1f}% of data",
                loc="left", fontsize=11)
    suptitle(fig, "C1 adaptive computation: the LSTM runs only where a cheap screener sees something", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "7_c1_adaptive_computation.png", dpi=160)
    plt.close(fig)


def fig_c2():
    h3 = load("h3_result.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
    conds = [("injected", "Test with 50 injected\nsensor faults"), ("clean", "Clean test,\nno faults")]
    x = np.arange(len(conds))
    b0 = [h3[c]["A"] for c, _ in conds]
    c2 = [h3[c]["B"] for c, _ in conds]
    a.bar(x - 0.19, b0, 0.36, color=BASE, label="Always-on LSTM (B0)")
    a.bar(x + 0.19, c2, 0.36, color=PROPOSED, label="With C2")
    for i, (c, _) in enumerate(conds):
        a.text(x[i] - 0.19, b0[i] + 0.015, f"{b0[i]:.3f}", ha="center", fontsize=9)
        a.text(x[i] + 0.19, c2[i] + 0.015, f"{c2[i]:.3f}", ha="center", fontsize=9, fontweight="bold")
        lo, hi = h3[c]["ci95"]
        a.text(x[i], max(b0[i], c2[i]) + 0.08, f"difference {h3[c]['diff']:+.3f}\n95% CI [{lo:+.3f}, {hi:+.3f}]",
               ha="center", fontsize=8.5, color=INK2)
    a.set_xticks(x, [l for _, l in conds])
    a.set_ylim(0, 1)
    a.set_ylabel("Event-wise F0.5, real events only")
    a.legend(loc="upper left", fontsize=9)
    a.set_title("System-alarm score (5 seeds)", loc="left", fontsize=11)

    att = h3["attribution"]
    kinds = ["bias", "noise", "drift", "dropout", "stuck"]
    y = np.arange(len(kinds))[::-1]
    ok = np.array([att[k]["attributed"] for k in kinds])
    wrong = np.array([att[k]["alarmed"] - att[k]["attributed"] for k in kinds])
    none = np.array([att[k]["faults"] - att[k]["alarmed"] for k in kinds])
    seg = dict(height=0.6, edgecolor="white", linewidth=2)
    b.barh(y, ok, color=PROPOSED, label="Alarm, attributed to the sensor", **seg)
    b.barh(y, wrong, left=ok, color=SECOND, label="Alarm, not attributed", **seg)
    b.barh(y, none, left=ok + wrong, color=FAINT, label="No LSTM alarm at all", **seg)
    for yi, k, o in zip(y, kinds, ok):
        b.text(o / 2, yi, f"{o}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    b.set_yticks(y, kinds)
    b.set_xlabel("Injected faults (5 seeds × 10 per type)")
    b.set_xlim(0, 52)
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8.5)
    b.set_title("What happened to each injected fault", loc="left", fontsize=11)
    suptitle(fig, "C2 sensor reliability: broken-sensor alarms leave the system-alarm stream", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "8_c2_sensor_reliability.png", dpi=160)
    plt.close(fig)


def fig_c3():
    m = load("h4_result.json")["methods"]
    rows = [("G0", "G0  no grouping"), ("G1", "G1  time only"), ("G2", "G2  learned graph (proposed)"),
            ("G2_subsystem_prior", "G2 with known subsystems")]
    y = np.arange(len(rows))[::-1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    colors = [BASE, PROPOSED, SECOND, BASE]
    load_ = [m[k]["incidents_per_event"] for k, _ in rows]
    f1 = [m[k]["f1"] for k, _ in rows]
    bars = a.barh(y, load_, color=colors, height=0.6)
    for bar, v in zip(bars, load_):
        a.text(v + 0.4, bar.get_y() + 0.3, f"{v:.2f}", va="center", fontsize=9)
    a.set_yticks(y, [l for _, l in rows])
    a.set_xlabel("Incidents an operator sees per labelled event (lower is better)")
    a.set_title("Operator load", loc="left", fontsize=11)
    bars = b.barh(y, f1, color=colors, height=0.6)
    for bar, v in zip(bars, f1):
        b.text(v + 0.01, bar.get_y() + 0.3, f"{v:.3f}", va="center", fontsize=9)
    b.set_xlim(0, 1.1)
    b.set_xlabel("Pairwise grouping F1 (higher is better)")
    b.set_title("Grouping correctness", loc="left", fontsize=11)
    suptitle(fig, "C3 incident correlation, all 76 channels, test period: time-only grouping wins", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "9_c3_incident_correlation.png", dpi=160)
    plt.close(fig)


def fig_c4():
    h5 = load("h5_result.json")
    central = next(r for r in h5["grid"] if (r["M"], r["D"], r["F"], r["S"]) == (50, 5, 1, 1))
    names = {"P0": "P0  page every alarm", "P1": "P1  C2 sensor rule", "P2": "P2  learned cost-aware"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.1]})
    pols = ["P0", "P1", "P2"]
    y = np.arange(3)[::-1]
    colors = {"P0": BASE, "P1": PROPOSED, "P2": SECOND}
    bars = a.barh(y, [central["cost"][p] for p in pols], color=[colors[p] for p in pols], height=0.6)
    for bar, p in zip(bars, pols):
        a.text(bar.get_width() + 1, bar.get_y() + 0.3, f"{central['cost'][p]:.1f}", va="center", fontsize=9)
    a.set_yticks(y, [names[p] for p in pols])
    a.set_xlabel("Cost per test period at M = 50, D = 5, F = 1, S = 1")
    a.set_title("Central cost point (oracle = 0)", loc="left", fontsize=11)

    ds = [2, 5, 10]
    x = np.arange(len(ds))
    w = 0.36
    for i, p in enumerate(["P1", "P2"]):
        n = [sum(r["best"] == p for r in h5["grid"] if r["D"] == d) for d in ds]
        bars = b.bar(x + (i - 0.5) * (w + 0.02), n, w, color=colors[p], label=names[p])
        for bar, v in zip(bars, n):
            b.text(bar.get_x() + w / 2, v + 0.2, str(v), ha="center", fontsize=9)
    b.set_xticks(x, [f"D = {d}" for d in ds])
    b.set_xlabel("Cost of delaying a real anomaly (D)")
    b.set_ylabel("Grid points ranked first (of 12)")
    b.set_ylim(0, 13)
    b.legend(loc="upper left", fontsize=9)
    b.set_title("Which policy wins across the 36-point grid (P0 never does)", loc="left", fontsize=11)
    suptitle(fig, "C4 cost-aware response replay: the simple C2 rule beats the learned policy", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "10_c4_response_cost.png", dpi=160)
    plt.close(fig)


def fig_c5():
    t = load("c5_result.json")["table"]
    rows = [("A0", "A0  always-on LSTM"), ("A4", "A4  C1"), ("A5", "A5  C1 + C2"), ("A6", "A6  C1 + C3"),
            ("A7", "A7  C1 + C4"), ("A8", "A8  full system"), ("L4", "L4  C1 + C2 + C3 + C2 rule")]
    metrics = [("F05_injected", "F0.5, faulty test\n(higher is better)", "{:.3f}", (0, 0.75)),
               ("lstm_share_injected", "LSTM share\n(lower is better)", "{:.0%}", (0, 1.25)),
               ("items", "Operator items\n(lower is better)", "{:.0f}", (0, 125)),
               ("cost_central", "Response cost, central\n(lower is better)", "{:.1f}", (0, 85))]
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 4.2), sharey=True)
    for ax, (key, title, fmt, lim) in zip(axes, metrics):
        vals = [t[k][key] for k, _ in rows]
        colors = [PROPOSED if k == "L4" else (BASE if k == "A0" else "#7fa9e0") for k, _ in rows]
        bars = ax.barh(y, vals, color=colors, height=0.62)
        for bar, v in zip(bars, vals):
            ax.text(v + lim[1] * 0.02, bar.get_y() + 0.31, fmt.format(v), va="center", fontsize=9)
        ax.set_xlim(*lim)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.set_title(title, loc="left", fontsize=10)
    axes[0].set_yticks(y, [l for _, l in rows])
    suptitle(fig, "C5 ablation on the faulty test period (5 seeds): L4 is the best combination", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "11_c5_ablation.png", dpi=160)
    plt.close(fig)


def fig_rare():
    r = load("h6_result.json")
    seeds = list(r["per_seed"])
    mean = lambda sys_, cat, k: float(np.mean([r["per_seed"][s][sys_][cat][k] for s in seeds]))
    systems = [("B0", "Always-on LSTM (B0)", BASE), ("R", "B0 + telecommand-gated path", PROPOSED)]
    ga = r["gate_alone"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.3, 1]})
    cats = [("rare", "Rare events\n(of 36)"), ("anomaly", "Anomalies\n(of 29)"), ("all", "False-alarm\nevents")]
    x = np.arange(len(cats))
    w = 0.26
    for i, (key, name, col) in enumerate(systems + [("gate", "Gate alone, no LSTM (comparison)", FAINT)]):
        vals = [(ga[c]["fp"] if c == "all" else ga[c]["tp"]) if key == "gate" else
                (mean(key, c, "fp") if c == "all" else mean(key, c, "tp")) for c, _ in cats]
        bars = a.bar(x + (i - 1) * (w + 0.02), vals, w, color=col, label=name)
        for bar, v in zip(bars, vals):
            a.text(bar.get_x() + w / 2, v + 0.5, f"{v:.1f}".rstrip("0").rstrip("."), ha="center", fontsize=9)
    a.set_xticks(x, [l for _, l in cats])
    a.set_ylabel("Events, test period (mean of 5 seeds)")
    a.set_ylim(0, 29)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8.5)
    a.set_title("What each system catches", loc="left", fontsize=11)

    rows = [("rare", "Rare events only"), ("anomaly", "Anomalies only"), ("all", "All events (H6)")]
    y = np.arange(len(rows))[::-1]
    for (c, label), yi in zip(rows, y):
        bb = r["bootstrap"][c]
        lo, hi = bb["ci95"]
        b.plot([lo, hi], [yi, yi], color=PROPOSED, lw=2)
        b.plot([bb["diff"]], [yi], "o", color=PROPOSED, ms=8, mec="white", mew=2)
        b.text(hi + 0.02, yi, f"{bb['diff']:+.3f}  [{lo:+.3f}, {hi:+.3f}]", va="center", fontsize=9)
    b.axvline(0, color=INK2, lw=1)
    b.set_yticks(y, [l for _, l in rows])
    b.set_xlim(-0.15, 0.85)
    b.set_xlabel("F0.5 change vs B0, with 95% CI (paired monthly bootstrap)")
    b.set_title("Score change", loc="left", fontsize=11)
    suptitle(fig, "Rare events: telecommand gates more than double rare-event detections, but H6 is not confirmed",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "12_rare_events.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    figs = [fig_c1, fig_c2, fig_c5, fig_rare] if PAPER else [fig_c1, fig_c2, fig_c3, fig_c4, fig_c5, fig_rare]
    for f in figs:
        f()
        print("done", f.__name__)
