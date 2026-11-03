"""Collect the saved experiment results into one JSON file for the dashboard.

Every number the dashboard shows comes from this file, and this file comes only from saved
results: results/*.json, models/*/train_summary.json and history.json, the evaluation logs,
results/figures/tables.json and results/diagnose_unpruned.csv. Nothing is typed in by hand
except the published paper values (Kotowski et al. 2024, Supplementary Table 13 and Table 2).

Usage: python src/export_dashboard_data.py
Writes dashboard/src/data/results.json
"""
import csv
import json
import re

import numpy as np
import pandas as pd

from evaluate import PAPER
from metrics import load_truth, test_events
from thresholding import ROOT, WINDOW

OUT = ROOT / "dashboard" / "src" / "data" / "results.json"
DATASETS = ["21_months", "84_months"]
MODELS = ["Telemanom-ESA", "Telemanom-ESA-Pruned"]
METRICS = ["EW_precision", "EW_recall", "EW_F_0.50", "alarming_precision", "AFF_F_0.50"]
SEEDS = [42, 1, 2, 3, 4]
# Table 2 of the paper gives the full metric set only for 84 months
PAPER_84_DETAIL = {"Telemanom-ESA": {"EW_precision": 0.148, "EW_recall": 0.894},
                   "Telemanom-ESA-Pruned": {"EW_precision": 0.999, "EW_recall": 0.424}}


def stats(values):
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "values": [float(x) for x in v]}


def eval_timing(path):
    text = path.read_text(encoding="utf8", errors="replace")
    predicted = re.findall(r"predicted (\d+) samples in (\d+)s", text)
    timing = {"predict_seconds": int(predicted[0][1]) if predicted else None,
              "samples": int(predicted[0][0]) if predicted else None}
    for model in MODELS:
        found = re.findall(rf"^{re.escape(model)}: F0\.5 .*\[(\d+)s\]", text, flags=re.M)
        timing[f"threshold_seconds|{model}"] = int(found[0]) if found else None
    return timing


def example_series():
    """The example event in the report figure, as data for an interactive chart."""
    run = ROOT / "models" / "telemanom_21_months_seed42"
    with np.load(ROOT / "data" / "processed" / "test.npz") as d:
        index, values, channels = d["index"], d["values"], [str(c) for c in d["channels"]]
    y_hat = np.load(run / "y_hat_last.npy", mmap_mode="r")
    flags = np.load(run / "flags_last_Telemanom-ESA-Pruned.npy", mmap_mode="r")
    truth = test_events(load_truth(channels), index)
    event_id, channel = "id_18", "channel_41"
    rows = truth[(truth["ID"] == event_id) & (truth["Channel"] == channel)]
    start, end = rows["StartTime"].min(), rows["EndTime"].max()
    pad = max(pd.Timedelta(hours=12), (end - start) * 2)
    t = pd.to_datetime(index)
    sel = np.flatnonzero((t >= start - pad) & (t <= end + pad))
    c = channels.index(channel)
    step = max(1, len(sel) // 1500)
    points = []
    for i in sel[::step]:
        points.append({"t": str(t[i])[:16], "actual": round(float(values[i, c]), 5),
                       "forecast": round(float(y_hat[i - WINDOW, c]), 5) if i >= WINDOW else None,
                       "labelled": int(start <= t[i] <= end), "alarm": int(flags[i, c])})
    return {"event": event_id, "channel": channel, "category": str(rows["Category"].iloc[0]),
            "start": str(start), "end": str(end), "points": points}


def main():
    data = {"paper": PAPER, "paper_84_detail": PAPER_84_DETAIL, "seeds": SEEDS, "runs": {}, "summary": {},
            "training": {}, "evaluation": {}}

    for d in DATASETS:
        runs = {s: json.loads((ROOT / "results" / f"telemanom_{d}_seed{s}_last.json").read_text()) for s in SEEDS}
        data["runs"][d] = {str(s): r["configs"] for s, r in runs.items()}
        for m in MODELS:
            for subset in ["all", "anomaly"]:
                data["summary"][f"{d}|{m}|{subset}"] = {
                    k: stats([runs[s]["configs"][m][subset][k] for s in SEEDS]) for k in METRICS}
        data["training"][d] = {}
        data["evaluation"][d] = {}
        for s in SEEDS:
            run = ROOT / "models" / f"telemanom_{d}_seed{s}"
            summary = json.loads((run / "train_summary.json").read_text())
            history = json.loads((run / "history.json").read_text())
            data["training"][d][str(s)] = {
                "epochs": summary["epochs"], "train_seconds": summary["train_seconds"],
                "best_val": summary["best_val"], "device": summary["device"], "torch": summary["torch"],
                "val_loss": [round(h["val_loss"], 6) for h in history],
                "train_loss": [round(h["train_loss"], 6) for h in history]}
            data["evaluation"][d][str(s)] = eval_timing(ROOT / "results" / f"eval_{d}_seed{s}_last.log")

    tables = json.loads((ROOT / "results" / "figures" / "tables.json").read_text())
    data["events_21_months_seed42_pruned"] = tables["events_21_months_seed42_pruned"]

    with open(ROOT / "results" / "diagnose_unpruned.csv", encoding="utf8") as f:
        data["alarm_breakdown"] = [{k: (float(v) if k in ("F0.5", "median_alarm_min", "flagged_hours")
                                        else int(v) if k in ("seed", "alarms", "false_alarms", "events_found") else v)
                                    for k, v in row.items()} for row in csv.DictReader(f)]

    with np.load(ROOT / "data" / "processed" / "test.npz") as d:
        index, channels = d["index"], [str(c) for c in d["channels"]]
    truth = test_events(load_truth(channels), index)
    cats = truth.groupby("ID")["Category"].first()
    data["dataset"] = {
        "name": "ESA Anomaly Dataset (ESA-AD), Zenodo record 15237121", "mission": "Mission 1",
        "channels": channels, "subsystem": "subsystem_5", "sampling_seconds": 30,
        "train_rows": {"21_months": 1840321, "84_months": 7364161},  # checked against datasets.csv by prepare.py
        "train_periods": {"21_months": ["2000-01-01", "2001-10-01"], "84_months": ["2000-01-01", "2007-01-01"]},
        "test_period": [str(pd.Timestamp(index[0]))[:10], str(pd.Timestamp(index[-1]))[:10]],
        "test_rows": int(len(index)),
        "test_events": {"Anomaly": int((cats == "Anomaly").sum()), "Rare Event": int((cats == "Rare Event").sum())}}
    data["hardware"] = {"gpu": "NVIDIA GeForce RTX 5060 Laptop GPU (8 GB)", "cpu": "Intel Core Ultra 9 275HX (24 threads)",
                        "ram_gb": 15.4, "framework": "PyTorch 2.11 (CUDA 12.8)"}
    data["example"] = example_series()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")
    for k in ["21_months|Telemanom-ESA-Pruned|all", "84_months|Telemanom-ESA-Pruned|all",
              "21_months|Telemanom-ESA|all", "84_months|Telemanom-ESA|all"]:
        print(k, round(data["summary"][k]["EW_F_0.50"]["mean"], 3))
    print("events", data["dataset"]["test_events"], "example points", len(data["example"]["points"]))
    print("timing sample", data["evaluation"]["21_months"]["42"])


if __name__ == "__main__":
    main()
