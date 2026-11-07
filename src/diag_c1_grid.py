"""Seed 42, development period: F0.5 and LSTM share for burst look-back and post lengths.

Usage: python src/diag_c1_grid.py
"""
import numpy as np

from c1_router import CheapTier, bursts, routed_flags, triggers
from metrics import global_scores, load_truth, test_events
from thresholding import ROOT


def main():
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
    sc = tier.scores(dv)
    for tau in [0.999, 0.9995]:
        for lookback, post in [(0, 700), (2100, 2100), (6000, 6000), (12000, 12000)]:
            mask, valid = bursts(triggers(sc, tau), len(y), lookback, post)
            s = global_scores(idx, routed_flags(y, dv, mask, valid), truth, "all")
            print(f"tau {tau} lookback {lookback} post {post}: F0.5 {s['EW_F_0.50']:.3f} precision {s['EW_precision']:.3f} "
                  f"recall {s['EW_recall']:.3f} LSTM share {mask.mean() * 100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
