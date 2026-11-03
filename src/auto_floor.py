"""Telemanom-ESA-AutoFloor: replace the paper's hand-picked 0.007 error floor with one learned from data.

For each channel, the floor is the largest smoothed forecast error the trained model makes on
the nominal validation period (the last 3 months of the training data). A test window is only
considered for an alarm if its error exceeds anything the model produced on normal data.

No test data and no anomaly labels are used to set the floor. Labels only mark which validation
samples are nominal, exactly as the baseline already uses them to choose training data.

Usage: python src/auto_floor.py 21_months 42 1 2 3 4
"""
import json
import sys
import time

import numpy as np
import pandas as pd

from metrics import global_scores, load_truth, test_events
from thresholding import ROOT, WINDOW, config, flags_from

NAME = "Telemanom-ESA-AutoFloor"


def smoothed(errors):
    cfg = config(0.0)
    span = int(cfg.batch_size * cfg.smoothing_window_size * cfg.smoothing_perc)  # same span as errors.py
    return pd.DataFrame(errors).ewm(span=span).mean().values


def validation_floors(dataset, seed):
    import torch

    from evaluate import predict
    from train import VALIDATION_SPLITS, Telemanom, nominal_fragments

    run = ROOT / "models" / f"telemanom_{dataset}_seed{seed}"
    with np.load(ROOT / "data" / "processed" / f"{dataset}.train.npz") as d:
        index, values, flags = d["index"], d["values"], d["flags"]
    _, val = nominal_fragments(index, values, flags, VALIDATION_SPLITS[dataset])
    with np.load(run / "scaling.npz") as d:
        means, stds = d["means"], d["stds"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Telemanom(values.shape[1], values.shape[1]).to(device)
    model.load_state_dict(torch.load(run / "last.pt", map_location=device))
    peaks = []
    for fragment in val:
        if len(fragment) <= WINDOW + 2 * 105:  # too short to smooth meaningfully
            continue
        y_hat = predict(model, fragment.astype(np.float64), means, stds, device)
        e_s = smoothed(np.abs(y_hat - fragment[WINDOW:]))
        peaks.append(e_s[105:].max(axis=0))  # skip the smoothing warm-up, as errors.py does at the start
    return np.max(peaks, axis=0)


def main():
    dataset, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
    with np.load(ROOT / "data" / "processed" / "test.npz") as d:
        index, values, channels = d["index"], d["values"], [str(c) for c in d["channels"]]
    truth = test_events(load_truth(channels), index)
    for seed in seeds:
        run = ROOT / "models" / f"telemanom_{dataset}_seed{seed}"
        out = ROOT / "results" / f"autofloor_{dataset}_seed{seed}.json"
        if out.exists():
            print(f"seed {seed} already done", flush=True)
            continue
        t0 = time.time()
        floors = validation_floors(dataset, seed)
        y_hat = np.load(run / "y_hat_last.npy")
        flags = flags_from(y_hat, values, floors)
        np.save(run / f"flags_last_{NAME}.npy", flags)
        scores = {c: {k: float(v) for k, v in global_scores(index, flags, truth, c).items()} for c in ["all", "anomaly"]}
        out.write_text(json.dumps({"dataset": dataset, "seed": seed, "floors": floors.tolist(),
                                   "configs": {NAME: scores}}, indent=1))
        s = scores["all"]
        print(f"seed {seed}: floors {np.round(floors, 4).tolist()} -> F0.5 {s['EW_F_0.50']:.3f} "
              f"precision {s['EW_precision']:.3f} recall {s['EW_recall']:.3f} [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
