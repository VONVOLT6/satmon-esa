"""Why does the baseline miss rare events? Development period only (2001-10 to 2006-12), never the test set.

For each labelled development event: category, length, class, whether B0 (floor 0.007) catches it, and
the strongest signal inside it from the LSTM error and from the C1 cheap tier (confidence, novelty,
impact), each expressed as a rank against nominal development time.

Usage: python src/rare_diag.py [seed]
Writes results/rare_diag.csv
"""
import sys

import numpy as np
import pandas as pd

from c1_router import CheapTier
from metrics import load_truth, test_events
from thresholding import ROOT, WINDOW, flags_from

DEV_START = np.datetime64("2001-10-01")


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    run = ROOT / "models" / f"telemanom_21_months_seed{seed}"
    with np.load(ROOT / "data" / "processed" / "84_months.train.npz") as d:
        keep = d["index"] > DEV_START
        index, values, labels, channels = d["index"][keep], d["values"][keep], d["flags"][keep], [str(c) for c in d["channels"]]
    with np.load(ROOT / "data" / "processed" / "21_months.train.npz") as d:
        tv, tf = d["values"], d["flags"]
    with np.load(run / "scaling.npz") as d:
        means, stds = d["means"], d["stds"]
    truth = test_events(load_truth(channels), index)
    kinds = pd.read_csv(ROOT / "data" / "raw" / "ESA-Mission1" / "anomaly_types.csv").set_index("ID")
    truth = truth[truth["Category"].isin(["Anomaly", "Rare Event"])]

    y_hat = np.load(run / "y_hat_dev.npy")
    flags = flags_from(y_hat, values, 0.007)
    actual = values[WINDOW:WINDOW + len(y_hat)]
    err = np.zeros(len(values), np.float32)
    err[WINDOW:WINDOW + len(y_hat)] = (np.abs(actual - y_hat) / stds.astype(np.float32)).max(axis=1)
    err = pd.Series(err).ewm(span=30).mean().to_numpy()
    tier = CheapTier().fit(tv, tf, means, stds)
    sc = tier.scores(values)

    nominal = (labels == 0).all(axis=1)
    nominal[:WINDOW + 300] = False
    ref = {"lstm_err": np.sort(err[nominal]), "novelty": np.sort(sc["novelty"][nominal]),
           "confidence": np.sort(sc["confidence"][nominal])}
    t = pd.to_datetime(index)
    rows = []
    for eid, g in truth.groupby("ID"):
        s = int(np.searchsorted(t, g["StartTime"].min()))
        e = max(int(np.searchsorted(t, g["EndTime"].max(), side="right")), s + 1)
        row = {"ID": eid, "category": g["Category"].iloc[0], "class": kinds.loc[eid, "Class"],
               "locality": kinds.loc[eid, "Locality"], "length": kinds.loc[eid, "Length"],
               "start": str(t[s]), "samples": e - s, "channels": g["Channel"].nunique(),
               "B0_detected": bool(flags[s:e].any())}
        for k, r in ref.items():
            v = (err if k == "lstm_err" else sc[k])[s:e].max()
            row[f"{k}_rank"] = float(np.searchsorted(r, v, side="right") / len(r))
        row["impact_max"] = int(sc["impact"][s:e].max())
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["category", "start"])
    df.to_csv(ROOT / "results" / "rare_diag.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.drop(columns=["start"]).to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(df.groupby("category")[["B0_detected"]].agg(["sum", "count"]))


if __name__ == "__main__":
    main()
