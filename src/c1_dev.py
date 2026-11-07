"""C1 on the development period: score each router setting, then pick one by the frozen rule.

Rule (fixed before any development scores were seen, 2026-11-08): among the candidate tau values,
choose the one with the smallest mean LSTM share whose mean development F0.5 is at least the
full-LSTM baseline's (floor 0.007) minus 0.01. If none qualifies, C1 fails at development.

Also scores the ablations at every tau: confidence only, confidence + impact, and the full router.
Resumable; progress in results/c1_dev.log, results in results/c1_dev.json.

Usage: python src/c1_dev.py 42 1 2 3 4
"""
import json
import sys
import time

import numpy as np

from c1_router import CheapTier, bursts, routed_flags, triggers
from metrics import global_scores, load_truth, test_events
from thresholding import ROOT

DEV_START = np.datetime64("2001-10-01")
TAUS = [0.995, 0.998, 0.999, 0.9995, 0.9999]
VARIANTS = {"full": ("confidence", "novelty", "impact"), "conf+impact": ("confidence", "impact"), "conf": ("confidence",)}
TOLERANCE = 0.01
OUT = ROOT / "results" / "c1_dev.json"
LOG = ROOT / "results" / "c1_dev.log"


def log(message):
    with open(LOG, "a", encoding="utf8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def main():
    seeds = [int(s) for s in sys.argv[1:]]
    state = json.loads(OUT.read_text()) if OUT.exists() else {"taus": TAUS, "runs": {}}
    tune = json.loads((ROOT / "results" / "dev_tune_21_months.json").read_text())
    baseline = float(np.mean([tune["dev"][f"{s}|0.007"]["EW_F_0.50"] for s in seeds]))
    state["baseline_dev_F0.5"] = baseline

    with np.load(ROOT / "data" / "processed" / "21_months.train.npz") as d:
        train_values, train_flags = d["values"], d["flags"]
    with np.load(ROOT / "data" / "processed" / "84_months.train.npz") as d:
        keep = d["index"] > DEV_START
        index, values, channels = d["index"][keep], d["values"][keep], [str(c) for c in d["channels"]]
    truth = test_events(load_truth(channels), index)

    run0 = ROOT / "models" / f"telemanom_21_months_seed{seeds[0]}"
    with np.load(run0 / "scaling.npz") as d:  # scaling depends only on the data, so it is the same for every seed
        tier = CheapTier().fit(train_values, train_flags, d["means"], d["stds"])
    t0 = time.time()
    scores = tier.scores(values)
    state["cheap_us_per_sample"] = (time.time() - t0) / len(values) * 1e6
    log(f"start: {len(values)} dev samples, baseline dev F0.5 {baseline:.4f}, cheap tier {state['cheap_us_per_sample']:.2f} us/sample")

    for seed in seeds:
        y_hat = np.load(ROOT / "models" / f"telemanom_21_months_seed{seed}" / "y_hat_dev.npy")
        for tau in TAUS:
            for name, use in VARIANTS.items():
                key = f"{seed}|{tau}|{name}"
                if key in state["runs"]:
                    continue
                t1 = time.time()
                mask, valid = bursts(triggers(scores, tau, use), len(y_hat))
                flags = routed_flags(y_hat, values, mask, valid)
                s = global_scores(index, flags, truth, "all")
                state["runs"][key] = {"F0.5": float(s["EW_F_0.50"]), "precision": float(s["EW_precision"]),
                                      "recall": float(s["EW_recall"]), "lstm_share": float(mask.mean())}
                OUT.write_text(json.dumps(state, indent=1))
                log(f"seed {seed} tau {tau} {name}: F0.5 {s['EW_F_0.50']:.3f} LSTM share {mask.mean() * 100:.1f}% [{time.time() - t1:.0f}s]")

    summary = {}
    for tau in TAUS:
        for name in VARIANTS:
            rows = [state["runs"][f"{s}|{tau}|{name}"] for s in seeds]
            summary[f"{tau}|{name}"] = {"F0.5": float(np.mean([r["F0.5"] for r in rows])),
                                        "lstm_share": float(np.mean([r["lstm_share"] for r in rows]))}
    state["summary"] = summary
    ok = [t for t in TAUS if summary[f"{t}|full"]["F0.5"] >= baseline - TOLERANCE]
    state["chosen_tau"] = min(ok, key=lambda t: summary[f"{t}|full"]["lstm_share"]) if ok else None
    OUT.write_text(json.dumps(state, indent=1))
    log(f"summary {summary}; chosen tau {state['chosen_tau']}")
    log("finished")


if __name__ == "__main__":
    main()
