"""Score a trained model on the Mission 1 test period.

Runs the original errors.py thresholding unchanged and the official ESAScores.
Reports two configs from the same predictions:
  Telemanom-ESA         min_error_value = 0
  Telemanom-ESA-Pruned  min_error_value = 0.007

Usage: python src/evaluate.py 21_months --seed 42 [--weights last]
"""
import argparse
import json
import time

import numpy as np

from metrics import global_scores, load_truth, test_events
from thresholding import ROOT, WINDOW, flags_from

PREDICT_BATCH = 4096
TARGETS = {"Telemanom-ESA": ("all", 0.0), "Telemanom-ESA-Pruned": ("all", 0.007)}
# Supplementary Table 13, Mission 1 channels 41-46, by training length
PAPER = {"21_months": {"Telemanom-ESA": 0.122, "Telemanom-ESA-Pruned": 0.776},
         "42_months": {"Telemanom-ESA": 0.309, "Telemanom-ESA-Pruned": 0.776},
         "84_months": {"Telemanom-ESA": 0.178, "Telemanom-ESA-Pruned": 0.786}}


def predict(model, values, means, stds, device):
    """One-step-ahead predictions, in original units, for samples WINDOW onwards."""
    import torch

    scaled = torch.as_tensor((values - means) / stds, dtype=torch.float32, device=device)
    steps = torch.arange(WINDOW, device=device)
    out = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values) - WINDOW, PREDICT_BATCH):
            s = torch.arange(start, min(start + PREDICT_BATCH, len(values) - WINDOW), device=device).unsqueeze(1)
            out.append(model(scaled[s + steps])[:, 0, :].cpu().numpy())
    return np.concatenate(out) * stds + means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", default="21_months", nargs="?")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", default="last", choices=["last", "best"])
    args = parser.parse_args()
    # torch is imported here, not at the top, so the thresholding workers do not load it
    import torch
    from train import Telemanom

    run = ROOT / "models" / f"telemanom_{args.dataset}_seed{args.seed}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(ROOT / "data" / "processed" / "test.npz") as d:
        index, values, channels = d["index"], d["values"], [str(c) for c in d["channels"]]
    with np.load(run / "scaling.npz") as d:
        means, stds = d["means"], d["stds"]

    y_path = run / f"y_hat_{args.weights}.npy"
    if y_path.exists():
        y_hat = np.load(y_path)
    else:
        t0 = time.time()
        model = Telemanom(values.shape[1], values.shape[1]).to(device)
        model.load_state_dict(torch.load(run / f"{args.weights}.pt", map_location=device))
        y_hat = predict(model, values.astype(np.float64), means, stds, device)
        np.save(y_path, y_hat)
        print(f"predicted {len(y_hat)} samples in {time.time() - t0:.0f}s", flush=True)

    truth = test_events(load_truth(channels), index)
    n_all = truth[truth["Category"].isin(["Anomaly", "Rare Event"])]["ID"].nunique()
    n_anom = truth[truth["Category"] == "Anomaly"]["ID"].nunique()
    print(f"test: {len(index)} samples, {index[0]} to {index[-1]}, {n_all} events, {n_anom} anomalies", flush=True)

    results = {"dataset": args.dataset, "seed": args.seed, "weights": args.weights, "configs": {}}
    for name, (category, min_error_value) in TARGETS.items():
        t0 = time.time()
        flags_path = run / f"flags_{args.weights}_{name}.npy"
        if flags_path.exists():  # thresholding is the slow part, so a rerun reuses it
            flags = np.load(flags_path)
        else:
            flags = flags_from(y_hat, values, min_error_value)
            np.save(flags_path, flags)
        scores = {c: global_scores(index, flags, truth, c) for c in ["all", "anomaly"]}
        results["configs"][name] = {c: {k: float(v) for k, v in s.items()} for c, s in scores.items()}
        s = scores["all"]
        print(f"{name}: F0.5 {s['EW_F_0.50']:.3f} precision {s['EW_precision']:.3f} recall {s['EW_recall']:.3f} "
              f"alarming precision {s['alarming_precision']:.3f} affiliation F0.5 {s['AFF_F_0.50']:.3f} "
              f"(paper: {PAPER.get(args.dataset, {}).get(name)}) [{time.time() - t0:.0f}s]", flush=True)

    out = ROOT / "results" / f"telemanom_{args.dataset}_seed{args.seed}_{args.weights}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
