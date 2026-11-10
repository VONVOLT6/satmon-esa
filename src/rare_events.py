"""Telecommand-gated rare-event path R (PROTOCOL.md addendum 6, hypothesis H6).

After every telecommand of priority >= p, a gate stays open for W. Inside a gate, LSTM alarms count at a
lower error floor f_r. Outside gates the baseline B0 (floor 0.007) is unchanged. Output = B0 OR R.

Usage: python src/rare_events.py dev    (18-setting grid on the development period, 5 seeds)
       python src/rare_events.py test   (selected setting, run once on the test period)
Progress: results/rare.log. Flags at each floor are cached per seed in models/<run>/r_*.npy.
"""
import io
import itertools
import json
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd

from metrics import load_truth, test_events
from stats import components, f05, paired_bootstrap
from thresholding import ROOT, flags_from

SEEDS = [42, 1, 2, 3, 4]
DEV_START = np.datetime64("2001-10-01")
SAMPLE_S = 30
GRID = [dict(p=p, W=w, f_r=f) for p, w, f in itertools.product([2, 3], [1800, 3600, 10800], [0.0, 0.003, 0.005])]
LOG = ROOT / "results" / "rare.log"
TC_CACHE = ROOT / "data" / "processed" / "telecommands.pkl"


def log(message):
    with open(LOG, "a", encoding="utf8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    print(message, flush=True)


def telecommands():
    if TC_CACHE.exists():
        return pd.read_pickle(TC_CACHE)
    raw = ROOT / "data" / "raw" / "ESA-Mission1"
    pri = pd.read_csv(raw / "telecommands.csv").set_index("Telecommand")["Priority"]
    parts = []
    for name in pri.index:
        with zipfile.ZipFile(raw / "telecommands" / f"{name}.zip") as z:
            df = pd.read_pickle(io.BytesIO(z.read(name)))
        parts.append(pd.DataFrame({"t": df.index, "tc": name, "p": pri[name]}))
    tc = pd.concat(parts).sort_values("t").reset_index(drop=True)
    tc.to_pickle(TC_CACHE)
    return tc


def gate(index, tc, p, w_seconds):
    """Boolean per sample: inside [t, t + W) after a telecommand of priority >= p."""
    idx = index.astype("datetime64[ns]")
    t = tc.loc[tc["p"] >= p, "t"].to_numpy("datetime64[ns]")
    a = np.searchsorted(idx, t)
    b = np.searchsorted(idx, t + np.timedelta64(int(w_seconds), "s"))
    run = np.zeros(len(idx) + 1, np.int32)
    np.add.at(run, a, 1)
    np.add.at(run, b, -1)
    return np.cumsum(run[:-1]) > 0


def combine(b0, low, g):
    return np.maximum(b0, low * g[:, None].astype(low.dtype))


def score(index, flags, truth):
    out = {}
    for c in ["all", "anomaly", "rare"]:
        parts = components(index, flags, truth, c)
        out[c] = {**f05(parts.sum()), **{k: float(v) for k, v in parts.sum()[["tp", "fn", "fp"]].items()}}
    return out


def dev_flags(seed, values, floor):
    path = ROOT / "models" / f"telemanom_21_months_seed{seed}" / f"r_dev_{floor}.npy"
    if not path.exists():
        y_hat = np.load(path.parent / "y_hat_dev.npy")
        t0 = time.time()
        np.save(path, flags_from(y_hat, values, floor))
        log(f"seed {seed} dev floor {floor}: flags computed [{time.time() - t0:.0f}s]")
    return np.load(path)


def dev():
    with np.load(ROOT / "data" / "processed" / "84_months.train.npz") as d:
        keep = d["index"] > DEV_START
        index, values, channels = d["index"][keep], d["values"][keep], [str(c) for c in d["channels"]]
    truth = test_events(load_truth(channels), index)
    tc = telecommands()
    gates = {(g["p"], g["W"]): gate(index, tc, g["p"], g["W"]) for g in GRID}
    res = {"grid": GRID, "per_seed": {}, "gate_share": {f"{k[0]}|{k[1]}": float(v.mean()) for k, v in gates.items()}}
    for seed in SEEDS:
        b0 = dev_flags(seed, values, 0.007)
        lows = {f: dev_flags(seed, values, f) for f in [0.0, 0.003, 0.005]}
        res["per_seed"][str(seed)] = {"B0": score(index, b0, truth)}
        for g in GRID:
            key = f"{g['p']}|{g['W']}|{g['f_r']}"
            res["per_seed"][str(seed)][key] = score(index, combine(b0, lows[g["f_r"]], gates[(g["p"], g["W"])]), truth)
        r = res["per_seed"][str(seed)]
        best = max((k for k in r if k != "B0"), key=lambda k: r[k]["all"]["f05"])
        log(f"seed {seed}: B0 F0.5 {r['B0']['all']['f05']:.3f} (rare tp {r['B0']['rare']['tp']:.0f}); "
            f"best {best} F0.5 {r[best]['all']['f05']:.3f} (rare tp {r[best]['rare']['tp']:.0f}, fp {r[best]['all']['fp']:.0f})")

    mean = {}
    for g in GRID:
        key = f"{g['p']}|{g['W']}|{g['f_r']}"
        mean[key] = float(np.mean([res["per_seed"][str(s)][key]["all"]["f05"] for s in SEEDS]))
    b0_mean = float(np.mean([res["per_seed"][str(s)]["B0"]["all"]["f05"] for s in SEEDS]))
    # highest mean F0.5; ties -> higher f_r, shorter W, higher p
    chosen = max(GRID, key=lambda g: (round(mean[f"{g['p']}|{g['W']}|{g['f_r']}"], 10), g["f_r"], -g["W"], g["p"]))
    res.update(mean_f05=mean, B0_mean_f05=b0_mean, chosen=chosen)
    (ROOT / "results" / "rare_dev.json").write_text(json.dumps(res, indent=1))
    log("mean development F0.5 by setting (p|W s|f_r): " + ", ".join(f"{k} {v:.3f}" for k, v in mean.items()))
    log(f"B0 development F0.5 {b0_mean:.3f}; chosen {chosen} with {mean[f'{chosen['p']}|{chosen['W']}|{chosen['f_r']}']:.3f}")


def test_low_flags(seed, values, floor):
    run = ROOT / "models" / f"telemanom_21_months_seed{seed}"
    existing = {0.0: "flags_last_Telemanom-ESA.npy", 0.005: "flags_last_floor0.005.npy"}
    if floor in existing:
        return np.load(run / existing[floor])
    path = run / f"r_test_{floor}.npy"
    if not path.exists():
        np.save(path, flags_from(np.load(run / "y_hat_last.npy"), values, floor))
    return np.load(path)


def test():
    chosen = json.loads((ROOT / "results" / "rare_dev.json").read_text())["chosen"]
    with np.load(ROOT / "data" / "processed" / "test.npz") as d:
        index, values, channels = d["index"], d["values"], [str(c) for c in d["channels"]]
    truth = test_events(load_truth(channels), index)
    g = gate(index, telecommands(), chosen["p"], chosen["W"])
    res = {"hypothesis": "H6", "chosen": chosen, "gate_share": float(g.mean()), "per_seed": {}}
    parts = {c: ([], []) for c in ["all", "anomaly", "rare"]}
    gate_alone = np.repeat(g[:, None], values.shape[1], axis=1).astype(np.uint8)
    res["gate_alone"] = score(index, gate_alone, truth)
    for seed in SEEDS:
        b0 = np.load(ROOT / "models" / f"telemanom_21_months_seed{seed}" / "flags_last_Telemanom-ESA-Pruned.npy")
        sysf = combine(b0, test_low_flags(seed, values, chosen["f_r"]), g)
        res["per_seed"][str(seed)] = {"B0": score(index, b0, truth), "R": score(index, sysf, truth)}
        for c in parts:
            parts[c][0].append(components(index, b0, truth, c))
            parts[c][1].append(components(index, sysf, truth, c))
        r = res["per_seed"][str(seed)]
        log(f"test seed {seed}: B0 F0.5 {r['B0']['all']['f05']:.4f} -> {r['R']['all']['f05']:.4f}; rare events "
            f"{r['B0']['rare']['tp']:.0f} -> {r['R']['rare']['tp']:.0f}; false-alarm events {r['B0']['all']['fp']:.0f} -> {r['R']['all']['fp']:.0f}")
    res["bootstrap"] = {c: paired_bootstrap(*parts[c]) for c in parts}
    b = res["bootstrap"]["all"]
    res["verdict"] = "improvement" if b["ci95"][0] > 0 else "not shown"
    (ROOT / "results" / "h6_result.json").write_text(json.dumps(res, indent=1))
    for c, bb in res["bootstrap"].items():
        log(f"{c}: B0 {bb['A']:.4f}  B0+R {bb['B']:.4f}  diff {bb['diff']:+.4f}  95% CI [{bb['ci95'][0]:+.4f}, {bb['ci95'][1]:+.4f}]")
    ga = res["gate_alone"]["all"]
    log(f"gate alone: F0.5 {ga['f05']:.4f}, precision {ga['precision']:.3f}, recall {ga['recall']:.3f}; H6 {res['verdict']}")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    with open(ROOT / "results" / "test_log.md", "a", encoding="utf8") as fh:
        fh.write(f"| {time.strftime('%Y-%m-%d')} | {commit} | H6 | telecommand-gated rare-event path p>={chosen['p']}, "
                 f"W {chosen['W']} s, f_r {chosen['f_r']}, 21 months, 5 seeds | F0.5 {b['B']:.4f} vs {b['A']:.4f}, diff {b['diff']:+.4f}, "
                 f"95% CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]: {res['verdict']} |\n")


if __name__ == "__main__":
    {"dev": dev, "test": test}[sys.argv[1]]()
