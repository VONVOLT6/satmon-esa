"""Dataset sizes, label statistics and a data-overview figure for the paper (Mission 1, channels 41-46).

Reads only the prepared arrays and the official label files. Writes results/data_summary.json and
results/figures/0_data_overview.png (plus a copy without its title in paper/figures/).

Usage: python src/data_summary.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import load_truth
from thresholding import ROOT
from train import VALIDATION_SPLITS, nominal_fragments

P = ROOT / "data" / "processed"
SPLITS = {"train_21": ("2000-01-01", "2001-10-01"), "development": ("2001-10-01", "2007-01-01"),
          "train_84": ("2000-01-01", "2007-01-01"), "test": ("2007-01-01", "2014-01-01")}
PROPOSED, SECOND, BASE, INK2 = "#2a78d6", "#eb6834", "#a3a29d", "#52514e"


def events_in(truth, start, end):
    t = truth[(truth["StartTime"] >= pd.Timestamp(start)) & (truth["EndTime"] <= pd.Timestamp(end))]
    t = t[t["Category"].isin(["Anomaly", "Rare Event"])]
    g = t.groupby("ID").agg(start=("StartTime", "min"), end=("EndTime", "max"), category=("Category", "first"),
                            channels=("Channel", "nunique"))
    g["minutes"] = (g["end"] - g["start"]).dt.total_seconds() / 60
    return g


def main():
    with np.load(P / "84_months.train.npz") as d:
        idx84, v84, f84, channels = d["index"], d["values"], d["flags"], [str(c) for c in d["channels"]]
    with np.load(P / "21_months.train.npz") as d:
        idx21, v21, f21 = d["index"], d["values"], d["flags"]
    with np.load(P / "test.npz") as d:
        idxt, vt, ft = d["index"], d["values"], d["flags"]
    truth = load_truth(channels)

    def labelled(flags):
        any_ = (flags > 0).any(axis=1)
        return {"rows": int(len(flags)), "labelled_rows": int(any_.sum()), "labelled_pct": float(any_.mean() * 100),
                "anomaly_rows": int((flags == 1).any(axis=1).sum()), "rare_rows": int((flags == 2).any(axis=1).sum())}

    dev_mask = idx84 >= np.datetime64("2001-10-01")
    summary = {"channels": channels, "sampling_seconds": 30,
               "splits": {"train_21": labelled(f21), "development": labelled(f84[dev_mask]),
                          "train_84": labelled(f84), "test": labelled(ft)}}
    tr, va = nominal_fragments(idx21, v21, f21, VALIDATION_SPLITS["21_months"])
    summary["train_21_nominal"] = {"fragments": len(tr) + len(va), "train_rows": int(sum(map(len, tr))),
                                   "validation_rows": int(sum(map(len, va))),
                                   "validation_from": VALIDATION_SPLITS["21_months"]}
    ev = {}
    for name, (a, b) in SPLITS.items():
        g = events_in(truth, a, b)
        ev[name] = {"events": int(len(g)), "anomalies": int((g["category"] == "Anomaly").sum()),
                    "rare_events": int((g["category"] == "Rare Event").sum()),
                    "median_minutes": float(g["minutes"].median()) if len(g) else None,
                    "max_hours": float(g["minutes"].max() / 60) if len(g) else None,
                    "all_six_channels": int((g["channels"] == 6).sum())}
    summary["events"] = ev
    summary["channel_stats_train_21"] = {c: {"mean": float(v21[:, i].mean()), "std": float(v21[:, i].std()),
                                             "min": float(v21[:, i].min()), "max": float(v21[:, i].max())}
                                         for i, c in enumerate(channels)}
    (ROOT / "results" / "data_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ["splits", "train_21_nominal", "events"]}, indent=1))

    # ---- figure: mission timeline + event durations ----
    idx = np.concatenate([idx84, idxt])
    vals = np.concatenate([v84, vt])
    mu, sd = v21.mean(axis=0), v21.std(axis=0) + 1e-9
    z = (vals - mu) / sd
    daily = pd.DataFrame(z, index=pd.to_datetime(idx)).resample("1D").mean()
    all_ev = events_in(truth, "2000-01-01", "2014-01-01")

    for paper in (False, True):
        fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [2.4, 1]})
        for i in range(daily.shape[1]):
            a.plot(daily.index, daily.iloc[:, i], lw=0.6, color=plt.cm.Blues(0.45 + 0.09 * i), alpha=0.9)
        for _, r in all_ev.iterrows():
            a.axvspan(r["start"], r["end"] + pd.Timedelta(hours=12), color=SECOND if r["category"] == "Anomaly" else PROPOSED,
                      alpha=0.35, lw=0)
        lo, hi = np.nanpercentile(daily.to_numpy(), [0.5, 99.5])
        a.set_ylim(lo - 0.5, hi + 0.5)
        for x, label in [("2001-10-01", "dev"), ("2007-01-01", "test")]:
            a.axvline(pd.Timestamp(x), color=INK2, lw=1, ls="--")
            a.text(pd.Timestamp(x), hi + 0.3, f" {label} →", fontsize=8.5, color=INK2, va="top")
        a.text(pd.Timestamp("2000-01-15"), hi + 0.3, "train", fontsize=8.5, color=INK2, va="top")
        a.xaxis.set_major_locator(mdates.YearLocator(2))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        a.set_ylabel("Daily mean, z-score\n(training mean and SD)")
        a.set_title("Channels 41-46 over the mission, labelled events shaded", loc="left", fontsize=10.5)
        a.plot([], [], color=SECOND, lw=6, alpha=0.5, label="Anomaly")
        a.plot([], [], color=PROPOSED, lw=6, alpha=0.5, label="Rare event")
        a.legend(loc="lower left", fontsize=8.5, frameon=False, ncol=2)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
            b.spines[s].set_visible(False)

        bins = np.logspace(0, np.log10(max(all_ev["minutes"].max(), 10) + 1), 16)
        groups = [all_ev.loc[all_ev["category"] == cat, "minutes"].clip(lower=1) for cat in ["Anomaly", "Rare Event"]]
        b.hist(groups, bins=bins, stacked=True, color=[SECOND, PROPOSED], edgecolor="white", linewidth=1,
               label=[f"Anomaly ({len(groups[0])})", f"Rare event ({len(groups[1])})"])
        b.set_xscale("log")
        b.set_xlabel("Event duration (minutes, log scale;\npoint events shown at 1 minute)")
        b.set_ylabel("Events")
        b.legend(fontsize=8.5, frameon=False)
        b.set_title("Event durations, 2000-2013", loc="left", fontsize=10.5)
        if not paper:
            fig.suptitle("ESA-ADB Mission 1, channels 41-46: data overview", x=0.01, ha="left", fontsize=12, fontweight="bold")
        fig.tight_layout()
        out = ROOT / ("paper" if paper else "results") / "figures" / "0_data_overview.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
