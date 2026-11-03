"""Tune the error floor (min_error_value) on a labelled development period, never on the test set.

The 21-month models were trained on 2000-01 to 2001-09. The rest of the training years,
2001-10-01 to 2007-01-01, are labelled, unseen by those models, and still before the test period,
so they serve as a development set. Each candidate floor is scored there with the official metric.
The floor with the best mean development F0.5 across seeds is then applied once to the test set.

Usage: python src/dev_tune.py 42 1 2 3 4
Progress: results/dev_tune.log   Results: results/dev_tune_21_months.json
"""
import json
import sys
import time

import numpy as np

from metrics import global_scores, load_truth, test_events
from thresholding import ROOT, flags_from

DATASET = "21_months"
DEV_START = np.datetime64("2001-10-01")
FLOORS = [0.0, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03]
OUT = ROOT / "results" / "dev_tune_21_months.json"
LOG = ROOT / "results" / "dev_tune.log"


def log(message):
    with open(LOG, "a", encoding="utf8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def dev_predictions(seed, index, values):
    import torch

    from evaluate import predict
    from train import Telemanom

    run = ROOT / "models" / f"telemanom_{DATASET}_seed{seed}"
    path = run / "y_hat_dev.npy"
    if path.exists():
        return np.load(path)
    with np.load(run / "scaling.npz") as d:
        means, stds = d["means"], d["stds"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Telemanom(values.shape[1], values.shape[1]).to(device)
    model.load_state_dict(torch.load(run / "last.pt", map_location=device))
    y_hat = predict(model, values.astype(np.float64), means, stds, device)
    np.save(path, y_hat)
    return y_hat


def main():
    seeds = [int(s) for s in sys.argv[1:]]
    state = json.loads(OUT.read_text()) if OUT.exists() else {"floors": FLOORS, "dev": {}, "test": {}}

    with np.load(ROOT / "data" / "processed" / "84_months.train.npz") as d:
        keep = d["index"] > DEV_START
        dev_index, dev_values, channels = d["index"][keep], d["values"][keep], [str(c) for c in d["channels"]]
    dev_truth = test_events(load_truth(channels), dev_index)
    n_dev = dev_truth[dev_truth["Category"].isin(["Anomaly", "Rare Event"])]["ID"].nunique()
    log(f"start: dev period {dev_index[0]} to {dev_index[-1]}, {len(dev_index)} samples, {n_dev} events; seeds {seeds}")

    for seed in seeds:
        y_hat = dev_predictions(seed, dev_index, dev_values)
        for floor in FLOORS:
            key = f"{seed}|{floor}"
            if key in state["dev"]:
                continue
            t0 = time.time()
            flags = flags_from(y_hat, dev_values, floor)
            score = global_scores(dev_index, flags, dev_truth, "all")
            state["dev"][key] = {k: float(v) for k, v in score.items()}
            OUT.write_text(json.dumps(state, indent=1))
            log(f"dev seed {seed} floor {floor}: F0.5 {score['EW_F_0.50']:.3f} precision {score['EW_precision']:.3f} "
                f"recall {score['EW_recall']:.3f} [{time.time() - t0:.0f}s]")

    mean_dev = {f: float(np.mean([state["dev"][f"{s}|{f}"]["EW_F_0.50"] for s in seeds])) for f in FLOORS}
    chosen = max(FLOORS, key=lambda f: mean_dev[f])
    state["mean_dev_F0.5"] = {str(f): v for f, v in mean_dev.items()}
    state["chosen_floor"] = chosen
    log(f"mean dev F0.5 by floor: {mean_dev}; chosen floor {chosen}")

    with np.load(ROOT / "data" / "processed" / "test.npz") as d:
        index, values = d["index"], d["values"]
    truth = test_events(load_truth(channels), index)
    for seed in seeds:
        if str(seed) in state["test"]:
            continue
        run = ROOT / "models" / f"telemanom_{DATASET}_seed{seed}"
        flags = flags_from(np.load(run / "y_hat_last.npy"), values, chosen)
        state["test"][str(seed)] = {c: {k: float(v) for k, v in global_scores(index, flags, truth, c).items()}
                                    for c in ["all", "anomaly"]}
        OUT.write_text(json.dumps(state, indent=1))
        s = state["test"][str(seed)]["all"]
        log(f"test seed {seed} floor {chosen}: F0.5 {s['EW_F_0.50']:.3f} precision {s['EW_precision']:.3f} "
            f"recall {s['EW_recall']:.3f}")
    log("finished")


if __name__ == "__main__":
    main()
