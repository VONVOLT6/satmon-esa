"""Break down the unpruned variant's precision: how many alarms, how many hit nothing, how short they are.

Usage: python src/diagnose_unpruned.py
"""
import json

import numpy as np
import pandas as pd

from make_figures import event_table, runs_of
from metrics import load_truth, test_events
from thresholding import ROOT


def main():
    with np.load(ROOT / "data" / "processed" / "test.npz") as d:
        index, channels = d["index"], [str(c) for c in d["channels"]]
    truth = test_events(load_truth(channels), index)
    truth = truth[truth["Category"].isin(["Anomaly", "Rare Event"])]
    rows = []
    for dataset in ["21_months", "84_months"]:
        for run in sorted((ROOT / "models").glob(f"telemanom_{dataset}_seed*")):
            seed = int(run.name.split("seed")[-1])
            for name in ["Telemanom-ESA", "Telemanom-ESA-Pruned"]:
                flags = np.load(run / f"flags_last_{name}.npy")
                table, _ = event_table(index, flags, truth)
                alarms = runs_of(flags.max(axis=1) > 0)
                lengths = np.array([b - a + 1 for a, b in alarms]) * 0.5  # minutes, 30 s samples
                score = json.loads((ROOT / "results" / f"telemanom_{dataset}_seed{seed}_last.json").read_text())
                rows.append({"dataset": dataset, "seed": seed, "model": name,
                             "F0.5": round(score["configs"][name]["all"]["EW_F_0.50"], 3),
                             "alarms": table["alarms_total"], "false_alarms": table["alarms_on_no_event"],
                             "events_found": table["Anomaly"]["detected"] + table["Rare Event"]["detected"],
                             "median_alarm_min": float(np.median(lengths)) if len(lengths) else 0,
                             "flagged_hours": float(flags.max(axis=1).sum() * 0.5 / 60)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(ROOT / "results" / "diagnose_unpruned.csv", index=False)


if __name__ == "__main__":
    main()
