"""Charts and tables for the results write-up, built from saved results only.

Usage: python src/make_figures.py
Writes results/figures/*.png and results/figures/tables.json
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate import PAPER
from metrics import load_truth, test_events
from thresholding import ROOT, WINDOW

OUT = ROOT / "results" / "figures"
MODELS = ["Telemanom-ESA", "Telemanom-ESA-Pruned"]
DATASETS = ["21_months", "84_months"]
LABEL = {"21_months": "21 months", "84_months": "84 months"}
OURS, PAPER_COLOUR, UNPRUNED = "#2b6cb0", "#a0aec0", "#90cdf4"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def load_runs(dataset):
    runs = []
    for f in sorted((ROOT / "results").glob(f"telemanom_{dataset}_seed*_last.json")):
        runs.append(json.loads(f.read_text()))
    return runs


def values(runs, model, metric, subset="all"):
    return np.array([r["configs"][model][subset][metric] for r in runs])


def fig_ours_vs_paper(all_runs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, model in zip(axes, MODELS):
        x = np.arange(len(DATASETS))
        ours = [values(all_runs[d], model, "EW_F_0.50") for d in DATASETS]
        means = [v.mean() for v in ours]
        lo = [m - v.min() for m, v in zip(means, ours)]
        hi = [v.max() - m for m, v in zip(means, ours)]
        paper = [PAPER[d][model] for d in DATASETS]
        ax.bar(x - 0.2, paper, 0.38, color=PAPER_COLOUR, label="Paper (Kotowski et al. 2024)")
        ax.bar(x + 0.2, means, 0.38, color=OURS, yerr=[lo, hi], capsize=4,
               label=f"Ours, mean of 5 seeds (bars: min-max)")
        for i in range(len(DATASETS)):
            ax.text(x[i] - 0.2, paper[i] + 0.015, f"{paper[i]:.3f}", ha="center", fontsize=9)
            ax.text(x[i] + 0.2, means[i] + hi[i] + 0.015, f"{means[i]:.3f}", ha="center", fontsize=9,
                    color=OURS, fontweight="bold")
        ax.set_xticks(x, [f"Trained on {LABEL[d]}" for d in DATASETS])
        ax.set_title(model)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Corrected event-wise F0.5 (higher is better)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("Reproduced Telemanom-ESA vs the published ESA-ADB results (Mission 1, channels 41-46)")
    fig.tight_layout()
    fig.savefig(OUT / "1_ours_vs_paper.png", dpi=160)
    plt.close(fig)


def fig_metrics(all_runs, dataset):
    metrics = [("EW_precision", "Event-wise precision"), ("EW_recall", "Event-wise recall"),
               ("EW_F_0.50", "Event-wise F0.5"), ("alarming_precision", "Alarming precision"),
               ("AFF_F_0.50", "Affiliation F0.5")]
    runs = all_runs[dataset]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 3.8))
    for ax, (key, title) in zip(axes, metrics):
        means = [values(runs, m, key).mean() for m in MODELS]
        ax.bar([0, 1], means, color=[UNPRUNED, OURS], width=0.6)
        for i, v in enumerate(means):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
        ax.set_xticks([0, 1], ["Telemanom-ESA", "Telemanom-ESA\nPruned"], fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(title)
    fig.suptitle(f"All official metrics, trained on {LABEL[dataset]} (mean of 5 seeds, anomalies and rare events)")
    fig.tight_layout()
    fig.savefig(OUT / f"2_all_metrics_{dataset}.png", dpi=160)
    plt.close(fig)


def fig_seeds(all_runs):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for j, d in enumerate(DATASETS):
        runs = all_runs[d]
        for k, model in enumerate(MODELS):
            v = values(runs, model, "EW_F_0.50")
            xpos = j * 2.5 + k
            ax.scatter(np.full(len(v), xpos) + np.linspace(-0.12, 0.12, len(v)), v, color=OURS, zorder=3)
            ax.hlines(PAPER[d][model], xpos - 0.3, xpos + 0.3, color="#c53030", lw=2, zorder=2)
    ax.set_xticks([0, 1, 2.5, 3.5], ["ESA\n21 months", "ESA-Pruned\n21 months", "ESA\n84 months", "ESA-Pruned\n84 months"])
    ax.set_ylabel("Corrected event-wise F0.5")
    ax.set_ylim(0, 1)
    ax.scatter([], [], color=OURS, label="Our runs (one dot per seed)")
    ax.plot([], [], color="#c53030", lw=2, label="Paper")
    ax.legend(frameon=False, loc="center right")
    ax.set_title("Run-to-run stability: five training seeds per setting")
    fig.tight_layout()
    fig.savefig(OUT / "3_seed_stability.png", dpi=160)
    plt.close(fig)


def fig_training(dataset):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for run in sorted((ROOT / "models").glob(f"telemanom_{dataset}_seed*")):
        h = json.loads((run / "history.json").read_text())
        seed = run.name.split("seed")[-1]
        ax.plot([e["epoch"] for e in h], [e["val_loss"] for e in h], label=f"seed {seed}")
    ax.set_xlabel("Epoch (1000 training batches each)")
    ax.set_ylabel("Validation loss (MSE, scaled units)")
    ax.set_title(f"Training curves, {LABEL[dataset]}: each run stops when validation loss stops improving")
    ax.legend(frameon=False, ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"4_training_curves_{dataset}.png", dpi=160)
    plt.close(fig)


def runs_of(mask):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2] - 1))


def event_table(index, flags, truth):
    """Which labelled events were caught, and how many alarms hit no labelled event."""
    alarms = runs_of(flags.max(axis=1) > 0)
    alarm_times = [(index[a], index[b]) for a, b in alarms]
    caught = {}
    hit = np.zeros(len(alarms), bool)
    for eid, rows in truth.groupby("ID"):
        found = False
        for _, r in rows.iterrows():
            for k, (s, e) in enumerate(alarm_times):
                if s <= r["EndTime"] and e >= r["StartTime"]:
                    found, hit[k] = True, True
        caught[eid] = found
    cats = truth.groupby("ID")["Category"].first()
    rows = {}
    for cat in ["Anomaly", "Rare Event"]:
        ids = cats[cats == cat].index
        rows[cat] = {"events": len(ids), "detected": int(sum(caught[i] for i in ids)),
                     "missed": int(sum(not caught[i] for i in ids))}
    rows["alarms_total"] = len(alarms)
    rows["alarms_on_no_event"] = int((~hit).sum())
    return rows, caught


def fig_example(dataset="21_months", seed=42):
    run = ROOT / "models" / f"telemanom_{dataset}_seed{seed}"
    with np.load(ROOT / "data" / "processed" / "test.npz") as d:
        index, vals, channels = d["index"], d["values"], [str(c) for c in d["channels"]]
    y_hat = np.load(run / "y_hat_last.npy", mmap_mode="r")
    flags = np.load(run / "flags_last_Telemanom-ESA-Pruned.npy")
    truth = test_events(load_truth(channels), index)
    truth = truth[truth["Category"].isin(["Anomaly", "Rare Event"])]
    table, caught = event_table(index, flags, truth)

    # a detected anomaly lasting a few hours, so the plot is readable
    spans = truth.groupby("ID").agg(start=("StartTime", "min"), end=("EndTime", "max"), cat=("Category", "first"),
                                    channel=("Channel", "first"))
    spans["hours"] = (spans["end"] - spans["start"]).dt.total_seconds() / 3600
    pick = spans[(spans["cat"] == "Anomaly") & spans.index.map(caught) & spans["hours"].between(0.5, 48)]
    eid = pick.sort_values("hours").index[len(pick) // 2]
    ev = spans.loc[eid]
    c = channels.index(ev["channel"])
    pad = max(pd.Timedelta(hours=12), (ev["end"] - ev["start"]) * 2)
    t = pd.to_datetime(index)
    sel = np.flatnonzero((t >= ev["start"] - pad) & (t <= ev["end"] + pad))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    a1.plot(t[sel], vals[sel, c], color="#2d3748", lw=0.8, label="Actual telemetry")
    pred_idx = sel[sel >= WINDOW] - WINDOW
    a1.plot(t[pred_idx + WINDOW], y_hat[pred_idx, c], color=OURS, lw=0.8, alpha=0.8, label="LSTM forecast")
    for _, r in truth[(truth["ID"] == eid) & (truth["Channel"] == ev["channel"])].iterrows():
        a1.axvspan(r["StartTime"], r["EndTime"], color="#feb2b2", alpha=0.5, label="Labelled anomaly (ESA engineers)")
    a1.set_ylabel(ev["channel"])
    handles, labels = a1.get_legend_handles_labels()
    keep = dict(zip(labels, handles))
    a1.legend(keep.values(), keep.keys(), frameon=False, fontsize=8, loc="upper left")
    a1.set_title(f"Example detection: event {eid} on {ev['channel']} (model trained on 21 months, seed 42)")
    a2.fill_between(t[sel], 0, flags[sel, c], step="post", color="#c53030")
    a2.set_ylabel("Alarm")
    a2.set_yticks([0, 1], ["off", "on"])
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "5_example_detection.png", dpi=160)
    plt.close(fig)
    return table


def fig_events(table):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    cats = ["Anomaly", "Rare Event"]
    det = [table[c]["detected"] for c in cats]
    mis = [table[c]["missed"] for c in cats]
    ax.barh(cats, det, color=OURS, label="Detected")
    ax.barh(cats, mis, left=det, color="#feb2b2", edgecolor="#e53e3e", label="Missed")
    for i in range(len(cats)):
        if det[i] >= 8:
            ax.text(det[i] / 2, i, f"{det[i]} detected", ha="center", va="center", color="white", fontweight="bold")
        else:
            ax.text(det[i] + 0.4, i - 0.32, f"{det[i]} detected", ha="left", va="center", color=OURS, fontweight="bold")
        ax.text(det[i] + mis[i] / 2, i, f"{mis[i]} missed", ha="center", va="center")
    ax.invert_yaxis()
    ax.set_xlabel("Labelled events in the 2007-2013 test period")
    ax.set_title("Telemanom-ESA-Pruned (21 months, seed 42): events detected\n"
                 f"{table['alarms_total']} alarms raised, {table['alarms_on_no_event']} of them on no labelled event",
                 fontsize=10)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "6_events_detected.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_runs = {d: load_runs(d) for d in DATASETS}
    fig_ours_vs_paper(all_runs)
    for d in DATASETS:
        fig_metrics(all_runs, d)
        fig_training(d)
    fig_seeds(all_runs)
    events = fig_example()
    fig_events(events)

    tables = {"events_21_months_seed42_pruned": events, "summary": {}}
    for d in DATASETS:
        for m in MODELS:
            for subset in ["all", "anomaly"]:
                tables["summary"][f"{d}|{m}|{subset}"] = {
                    k: {"mean": float(values(all_runs[d], m, k, subset).mean()),
                        "min": float(values(all_runs[d], m, k, subset).min()),
                        "max": float(values(all_runs[d], m, k, subset).max()),
                        "seeds": len(all_runs[d])}
                    for k in ["EW_precision", "EW_recall", "EW_F_0.50", "alarming_precision", "AFF_F_0.50"]}
    (OUT / "tables.json").write_text(json.dumps(tables, indent=1))
    print(json.dumps(events, indent=1))
    for k, v in tables["summary"].items():
        print(k, {m: round(x["mean"], 3) for m, x in v.items()}, "n =", v["EW_F_0.50"]["seeds"])


if __name__ == "__main__":
    main()
