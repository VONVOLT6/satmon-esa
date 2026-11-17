"""Dataset and preprocessing facts for the dashboard (Mission 1, channels 41-46).

Reads results/data_summary.json (from src/data_summary.py), the prepared arrays and the official labels.
Writes dashboard/src/data/data_overview.json: split sizes, preprocessing counts, a weekly z-scored
timeline of the six channels, labelled event spans and an event-duration histogram.

Usage: python src/export_data_overview.py
"""
import json

import numpy as np
import pandas as pd

from data_summary import events_in
from metrics import load_truth
from thresholding import ROOT

P = ROOT / "data" / "processed"
OUT = ROOT / "dashboard" / "src" / "data" / "data_overview.json"


def main():
    s = json.loads((ROOT / "results" / "data_summary.json").read_text())
    with np.load(P / "84_months.train.npz") as d:
        idx84, v84, channels = d["index"], d["values"], [str(c) for c in d["channels"]]
    with np.load(P / "21_months.train.npz") as d:
        v21 = d["values"]
    with np.load(P / "test.npz") as d:
        idxt, vt = d["index"], d["values"]

    mu, sd = v21.mean(axis=0), v21.std(axis=0) + 1e-9
    frame = pd.DataFrame((np.concatenate([v84, vt]) - mu) / sd, index=pd.to_datetime(np.concatenate([idx84, idxt])),
                         columns=channels)
    weekly = frame.resample("7D").mean().dropna()
    timeline = [{"t": t.strftime("%Y-%m-%d"), **{c: round(float(v), 3) for c, v in row.items()}}
                for t, row in weekly.iterrows()]

    truth = load_truth(channels)
    ev = events_in(truth, "2000-01-01", "2014-01-01")
    spans = [{"start": r["start"].strftime("%Y-%m-%d"), "end": (r["end"] + pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
              "category": r["category"]} for _, r in ev.iterrows()]

    edges = [0, 2, 10, 60, 240, 1440, 10080, 10 ** 9]
    labels = ["< 2 min", "2-10 min", "10-60 min", "1-4 h", "4-24 h", "1-7 days", "> 7 days"]
    hist = []
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        m = (ev["minutes"] >= lo) & (ev["minutes"] < hi)
        hist.append({"bin": lab, "Anomaly": int((m & (ev["category"] == "Anomaly")).sum()),
                     "Rare event": int((m & (ev["category"] == "Rare Event")).sum())})

    sp, e = s["splits"], s["events"]
    out = {
        "channels": channels,
        "splits": [
            {"name": "Training (21 months)", "period": "2000-01 to 2001-09", "rows": sp["train_21"]["rows"],
             "labelled_pct": sp["train_21"]["labelled_pct"], "anomalies": e["train_21"]["anomalies"],
             "rare": e["train_21"]["rare_events"], "use": "Learn normal behaviour (21-month models)"},
            {"name": "Development", "period": "2001-10 to 2006-12", "rows": sp["development"]["rows"],
             "labelled_pct": sp["development"]["labelled_pct"], "anomalies": e["development"]["anomalies"],
             "rare": e["development"]["rare_events"], "use": "Tune every new idea"},
            {"name": "Training (84 months)", "period": "2000-01 to 2006-12", "rows": sp["train_84"]["rows"],
             "labelled_pct": sp["train_84"]["labelled_pct"], "anomalies": e["train_84"]["anomalies"],
             "rare": e["train_84"]["rare_events"], "use": "Reproduce the paper's 84-month result"},
            {"name": "Test", "period": "2007-01 to 2013-12", "rows": sp["test"]["rows"],
             "labelled_pct": sp["test"]["labelled_pct"], "anomalies": e["test"]["anomalies"],
             "rare": e["test"]["rare_events"], "use": "Final check, once per idea"},
        ],
        "preprocessing": {
            "train_rows": sp["train_21"]["rows"], "labelled_removed": sp["train_21"]["labelled_rows"],
            "labelled_removed_pct": sp["train_21"]["labelled_pct"],
            "nominal_train_rows": s["train_21_nominal"]["train_rows"],
            "nominal_validation_rows": s["train_21_nominal"]["validation_rows"],
            "validation_from": s["train_21_nominal"]["validation_from"],
            "test_events_all_channels": e["test"]["all_six_channels"], "test_events": e["test"]["events"],
            "anomalies_total": int((ev["category"] == "Anomaly").sum()),
            "anomalies_under_2min": int(((ev["category"] == "Anomaly") & (ev["minutes"] < 2)).sum()),
            "rare_median_minutes": float(ev.loc[ev["category"] == "Rare Event", "minutes"].median()),
        },
        "timeline": timeline,
        "events": spans,
        "duration_hist": hist,
    }
    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB, {len(timeline)} weeks, {len(spans)} events)")


if __name__ == "__main__":
    main()
