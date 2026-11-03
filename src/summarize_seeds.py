"""Mean and spread of the baseline scores across seeds.

Usage: python src/summarize_seeds.py 21_months
"""
import argparse
import json

import numpy as np

from train import ROOT

METRICS = ["EW_F_0.50", "EW_precision", "EW_recall", "alarming_precision", "AFF_F_0.50"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", default="21_months", nargs="?")
    parser.add_argument("--weights", default="last")
    args = parser.parse_args()

    files = sorted(ROOT.glob(f"results/telemanom_{args.dataset}_seed*_{args.weights}.json"))
    runs = [json.loads(f.read_text()) for f in files]
    seeds = [r["seed"] for r in runs]
    rows = [f"# Baseline across seeds ({args.dataset}, {args.weights} weights)", "",
            f"Seeds: {', '.join(map(str, seeds))}. Values are mean ± sample standard deviation, with min and max.", ""]
    summary = {"dataset": args.dataset, "weights": args.weights, "seeds": seeds, "configs": {}}
    for config in runs[0]["configs"]:
        for subset in ["all", "anomaly"]:
            rows += [f"## {config}, {'anomalies and rare events' if subset == 'all' else 'anomalies only'}", "",
                     "| Metric | Mean ± SD | Min | Max | Per seed |", "|---|---|---|---|---|"]
            for m in METRICS:
                v = np.array([r["configs"][config][subset][m] for r in runs])
                sd = v.std(ddof=1) if len(v) > 1 else 0.0
                summary["configs"].setdefault(config, {}).setdefault(subset, {})[m] = {
                    "mean": v.mean(), "sd": sd, "min": v.min(), "max": v.max(), "values": v.tolist()}
                rows.append(f"| {m} | {v.mean():.3f} ± {sd:.3f} | {v.min():.3f} | {v.max():.3f} | "
                            f"{', '.join(f'{x:.3f}' for x in v)} |")
            rows.append("")
    out = ROOT / "results" / f"seed_summary_{args.dataset}_{args.weights}"
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    out.with_suffix(".md").write_text("\n".join(rows), encoding="utf8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
