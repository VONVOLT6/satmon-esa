"""Diagnose C1 on the development period: where do routed alarms differ from the full LSTM run?

Usage: python src/diag_c1.py 0.995
"""
import sys

import numpy as np

from c1_router import PRE, CheapTier, bursts, routed_flags, triggers
from metrics import load_truth, test_events
from stats import components, f05
from thresholding import ROOT, WINDOW, flags_from


def main():
    tau = float(sys.argv[1])
    run = ROOT / "models" / "telemanom_21_months_seed42"
    with np.load(ROOT / "data" / "processed" / "21_months.train.npz") as d:
        tv, tf = d["values"], d["flags"]
    with np.load(run / "scaling.npz") as d:
        tier = CheapTier().fit(tv, tf, d["means"], d["stds"])
    with np.load(ROOT / "data" / "processed" / "84_months.train.npz") as d:
        keep = d["index"] > np.datetime64("2001-10-01")
        idx, dv, ch = d["index"][keep], d["values"][keep], [str(c) for c in d["channels"]]
    truth = test_events(load_truth(ch), idx)
    y = np.load(run / "y_hat_dev.npy")

    full = flags_from(y, dv, 0.007)
    mask, valid = bursts(triggers(tier.scores(dv), tau), len(y))
    routed = routed_flags(y, dv, mask, valid)
    for name, fl in [("full", full), ("routed", routed)]:
        c = components(idx, fl, truth).sum()
        print(name, {k: round(float(v), 1) for k, v in c.items()}, {k: round(v, 3) for k, v in f05(c).items()})

    inside = np.zeros(len(dv), bool)
    inside[WINDOW:WINDOW + len(mask)] = mask
    ok = np.zeros(len(dv), bool)
    ok[WINDOW:WINDOW + len(valid)] = valid
    t = idx.astype("datetime64[ns]")
    sel_truth = truth[truth["Category"].isin(["Anomaly", "Rare Event"])]
    for eid, rows in sel_truth.groupby("ID"):
        sel = (t >= np.datetime64(rows.StartTime.min())) & (t <= np.datetime64(rows.EndTime.max()))
        f_hit, r_hit = full.max(1)[sel].any(), routed.max(1)[sel].any()
        if f_hit != r_hit:
            print(f"{'lost' if f_hit else 'gained'} {eid} {rows.Category.iloc[0]}: in burst {inside[sel].mean():.2f}, valid {ok[sel].mean():.2f}")

    # alarms raised by the routed run that the full run did not raise
    extra = (routed.max(1) > 0) & (full.max(1) == 0)
    starts = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8)]) == 1) + WINDOW
    if extra.any():
        pos = np.flatnonzero(extra)
        gaps = np.array([p - starts[starts <= p].max() for p in pos if (starts <= p).any()])
        print(f"extra alarm samples {extra.sum()}, distance from burst start: median {np.median(gaps):.0f}, "
              f"share within PRE+2100 of start {(gaps < PRE + 2100).mean():.2f}")


if __name__ == "__main__":
    main()
