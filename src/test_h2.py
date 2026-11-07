"""H2 confirmatory test: C1 adaptive computation vs the always-on LSTM (PROTOCOL.md, addendum 1).

Uses the tau chosen on the development period by c1_dev.py. For each seed: route the test set,
score it, compare with B0 using the paired monthly-block bootstrap (non-inferiority margin 0.02),
and record the LSTM share. For seed 42 it also times the real LSTM on the routed bursts against
the full test period on the GPU. Writes results/h2_result.json and appends to results/test_log.md.

Usage: python src/test_h2.py
"""
import json
import subprocess
import time

import numpy as np

from c1_router import CheapTier, bursts, routed_flags, triggers
from metrics import load_truth, test_events
from stats import components, f05, paired_bootstrap
from thresholding import ROOT, WINDOW

SEEDS = [42, 1, 2, 3, 4]
MARGIN = 0.02


def time_lstm(seed, values, mask):
    """Seconds to run the LSTM on the whole test set and on the routed bursts only."""
    import torch

    from evaluate import predict
    from train import Telemanom

    run = ROOT / "models" / f"telemanom_21_months_seed{seed}"
    with np.load(run / "scaling.npz") as d:
        means, stds = d["means"], d["stds"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Telemanom(values.shape[1], values.shape[1]).to(device)
    model.load_state_dict(torch.load(run / "last.pt", map_location=device))
    v = values.astype(np.float64)
    predict(model, v[:20000], means, stds, device)  # warm-up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    predict(model, v, means, stds, device)
    torch.cuda.synchronize()
    full = time.perf_counter() - t0
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    t0 = time.perf_counter()
    for a, b in zip(edges[::2], edges[1::2]):  # y_hat index a..b needs samples a..b+WINDOW
        predict(model, v[a:b + WINDOW], means, stds, device)
    torch.cuda.synchronize()
    return full, time.perf_counter() - t0, int(len(edges) // 2)


def main():
    dev = json.loads((ROOT / "results" / "c1_dev.json").read_text())
    tau = dev["chosen_tau"]
    if tau is None:
        raise SystemExit("C1 failed at development: no tau met the rule, so there is nothing to test")
    with np.load(ROOT / "data" / "processed" / "21_months.train.npz") as d:
        train_values, train_flags = d["values"], d["flags"]
    with np.load(ROOT / "data" / "processed" / "test.npz") as d:
        index, values, channels = d["index"], d["values"], [str(c) for c in d["channels"]]
    truth = test_events(load_truth(channels), index)
    with np.load(ROOT / "models" / "telemanom_21_months_seed42" / "scaling.npz") as d:
        tier = CheapTier().fit(train_values, train_flags, d["means"], d["stds"])
    t0 = time.perf_counter()
    scores = tier.scores(values)
    cheap_seconds = time.perf_counter() - t0
    trig = triggers(scores, tau)

    result = {"hypothesis": "H2", "tau": tau, "per_seed": {}, "bootstrap": {}, "cheap_tier_seconds": cheap_seconds}
    parts = {c: ([], []) for c in ["all", "anomaly"]}
    for seed in SEEDS:
        run = ROOT / "models" / f"telemanom_21_months_seed{seed}"
        y_hat = np.load(run / "y_hat_last.npy")
        mask, valid = bursts(trig, len(y_hat))
        flags = routed_flags(y_hat, values, mask, valid)
        np.save(run / "flags_last_C1.npy", flags)
        result["per_seed"][str(seed)] = {"lstm_share": float(mask.mean())}
        for c in parts:
            b0 = components(index, np.load(run / "flags_last_Telemanom-ESA-Pruned.npy"), truth, c)
            c1 = components(index, flags, truth, c)
            parts[c][0].append(b0)
            parts[c][1].append(c1)
            result["per_seed"][str(seed)][c] = {"B0": f05(b0.sum()), "C1": f05(c1.sum()),
                                                "C1_counts": {k: float(v) for k, v in c1.sum().items()}}
        print(f"seed {seed}: C1 F0.5 {result['per_seed'][str(seed)]['all']['C1']['f05']:.4f}, LSTM share {mask.mean() * 100:.1f}%", flush=True)
    for c in parts:
        result["bootstrap"][c] = paired_bootstrap(*parts[c])
        b = result["bootstrap"][c]
        print(f"{c}: B0 {b['A']:.4f}  C1 {b['B']:.4f}  diff {b['diff']:+.4f}  95% CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]", flush=True)

    mask42, _ = bursts(trig, len(values) - WINDOW)
    full_s, routed_s, n_bursts = time_lstm(42, values, mask42)
    result["timing_seed42"] = {"full_lstm_seconds": full_s, "routed_lstm_seconds": routed_s, "bursts": n_bursts,
                               "cheap_tier_seconds": cheap_seconds,
                               "speedup_total": full_s / (routed_s + cheap_seconds)}
    shares = [result["per_seed"][str(s)]["lstm_share"] for s in SEEDS]
    result["mean_lstm_share"] = float(np.mean(shares))
    lo = result["bootstrap"]["all"]["ci95"][0]
    result["verdict"] = "non-inferior" if lo > -MARGIN else "not shown non-inferior"
    (ROOT / "results" / "h2_result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result["timing_seed42"], indent=1))

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    b = result["bootstrap"]["all"]
    with open(ROOT / "results" / "test_log.md", "a", encoding="utf8") as f:
        f.write(f"| {time.strftime('%Y-%m-%d')} | {commit} | H2 | C1 router tau {tau}, 21 months, 5 seeds | "
                f"F0.5 {b['B']:.4f} vs {b['A']:.4f}, diff {b['diff']:+.4f}, 95% CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}], "
                f"LSTM share {result['mean_lstm_share'] * 100:.1f}%: {result['verdict']} |\n")
    print("verdict:", result["verdict"])


if __name__ == "__main__":
    main()
