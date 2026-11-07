"""H1 confirmatory test: calibrated error floor vs the paper's 0.007 (PROTOCOL.md).

Run after `dev_tune.py 42 1 2 3 4` has selected the floor on the development period and scored it
on the test set. Compares it with B0 (Telemanom-ESA-Pruned, 0.007) using the paired monthly-block
bootstrap, writes results/h1_result.json and appends a line to results/test_log.md.

Usage: python src/test_h1.py
"""
import json
import subprocess
import time

import numpy as np

from metrics import load_truth, test_events
from stats import components, f05, paired_bootstrap
from thresholding import ROOT

SEEDS = [42, 1, 2, 3, 4]


def main():
    tune = json.loads((ROOT / "results" / "dev_tune_21_months.json").read_text())
    floor = tune["chosen_floor"]
    with np.load(ROOT / "data" / "processed" / "test.npz") as d:
        index, channels = d["index"], [str(c) for c in d["channels"]]
    truth = test_events(load_truth(channels), index)

    result = {"hypothesis": "H1", "chosen_floor": floor, "baseline_floor": 0.007,
              "mean_dev_F0.5": tune["mean_dev_F0.5"], "per_seed": {}, "bootstrap": {}}
    for category in ["all", "anomaly"]:
        parts_b0, parts_h1 = [], []
        for seed in SEEDS:
            run = ROOT / "models" / f"telemanom_21_months_seed{seed}"
            pb0 = components(index, np.load(run / "flags_last_Telemanom-ESA-Pruned.npy"), truth, category)
            ph1 = components(index, np.load(run / f"flags_last_floor{floor}.npy"), truth, category)
            parts_b0.append(pb0)
            parts_h1.append(ph1)
            result["per_seed"].setdefault(str(seed), {})[category] = {
                "B0": f05(pb0.sum()), "H1": f05(ph1.sum()),
                "B0_counts": {k: float(v) for k, v in pb0.sum().items()},
                "H1_counts": {k: float(v) for k, v in ph1.sum().items()}}
        result["bootstrap"][category] = paired_bootstrap(parts_b0, parts_h1)
        b = result["bootstrap"][category]
        print(f"{category}: B0 {b['A']:.4f}  H1 {b['B']:.4f}  diff {b['diff']:+.4f}  95% CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]", flush=True)

    ci = result["bootstrap"]["all"]["ci95"]
    result["verdict"] = ("improvement" if ci[0] > 0 else "worse" if ci[1] < 0 else "no clear difference")
    (ROOT / "results" / "h1_result.json").write_text(json.dumps(result, indent=1))

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    b = result["bootstrap"]["all"]
    with open(ROOT / "results" / "test_log.md", "a", encoding="utf8") as f:
        f.write(f"| {time.strftime('%Y-%m-%d')} | {commit} | H1 | floor {floor} vs 0.007, 21 months, 5 seeds | "
                f"F0.5 {b['B']:.4f} vs {b['A']:.4f}, diff {b['diff']:+.4f}, 95% CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]: {result['verdict']} |\n")
    print("verdict:", result["verdict"])


if __name__ == "__main__":
    main()
