"""Run C3 (PROTOCOL.md addendum 3). Stage "dev" builds alerts, the graph and tunes W and rho;
stage "test" applies the chosen settings once and tests H4.

Usage: python src/c3_run.py dev
       python src/c3_run.py test
Progress: results/c3.log. Alerts and the graph are cached in results/c3_cache.pkl.
"""
import json
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from c3_correlation import (ChannelDetectors, alert_event_ids, contingency, dependency_graph, group, load_full,
                            paired_event_bootstrap, summary)
from thresholding import ROOT

DEV_START, TEST_START = np.datetime64("2001-10-01"), np.datetime64("2007-01-01")
WINDOWS = [60, 240, 720, 2880]
RHOS = [0.1, 0.2, 0.3, 0.5, 0.7]
CACHE = ROOT / "results" / "c3_cache.pkl"
LOG = ROOT / "results" / "c3.log"


def log(message):
    with open(LOG, "a", encoding="utf8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    print(message, flush=True)


def build():
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    index, channels, values, labels, ids = load_full(ROOT)
    tr = slice(0, int(np.searchsorted(index, DEV_START)))
    dv = slice(tr.stop, int(np.searchsorted(index, TEST_START)))
    te = slice(dv.stop, len(index))
    t0 = time.time()
    det = ChannelDetectors().fit(values[tr], labels[tr])
    log(f"detectors fitted on {tr.stop:,} training rows [{time.time() - t0:.0f}s]")
    data = {"channels": channels}
    for name, sl in [("train", tr), ("dev", dv), ("test", te)]:
        t0 = time.time()
        alerts, activity = det.alerts(values[sl])
        alerts = [(c, s + sl.start, e + sl.start) for c, s, e in alerts]
        data[name] = {"alerts": alerts, "event_ids": alert_event_ids(alerts, ids)}
        if name == "train":
            data["graph"] = dependency_graph(activity)
        log(f"{name}: {len(alerts)} alerts, {int((data[name]['event_ids'] > 0).sum())} on labelled events [{time.time() - t0:.0f}s]")
    subsystem = pd.read_csv(ROOT / "data" / "raw" / "ESA-Mission1" / "channels.csv").set_index("Channel").loc[channels, "Subsystem"].to_numpy()
    data["subsystem_adjacency"] = subsystem[:, None] == subsystem[None, :]
    CACHE.write_bytes(pickle.dumps(data))
    return data


def dev():
    data = build()
    a, ev = data["dev"]["alerts"], data["dev"]["event_ids"]
    rows = []
    for w in WINDOWS:
        rows.append({"method": "G1", "W": w, "rho": None, **summary(group(a, w), ev)})
        for rho in RHOS:
            rows.append({"method": "G2", "W": w, "rho": rho, **summary(group(a, w, data["graph"] >= rho), ev)})
    df = pd.DataFrame(rows)
    best_g1 = df[df.method == "G1"].sort_values("f1", ascending=False).iloc[0]
    best_g2 = df[df.method == "G2"].sort_values("f1", ascending=False).iloc[0]
    out = {"G0": summary(np.arange(len(a)), ev), "grid": rows,
           "chosen": {"G1": {"W": int(best_g1.W)}, "G2": {"W": int(best_g2.W), "rho": float(best_g2.rho)}}}
    (ROOT / "results" / "c3_dev.json").write_text(json.dumps(out, indent=1, default=float))
    log(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    log(f"chosen: G1 W={int(best_g1.W)} (dev F1 {best_g1.f1:.3f}); G2 W={int(best_g2.W)} rho={best_g2.rho} (dev F1 {best_g2.f1:.3f})")


def test():
    data = build()
    chosen = json.loads((ROOT / "results" / "c3_dev.json").read_text())["chosen"]
    a, ev = data["test"]["alerts"], data["test"]["event_ids"]
    inc = {"G0": np.arange(len(a)), "G1": group(a, chosen["G1"]["W"]),
           "G2": group(a, chosen["G2"]["W"], data["graph"] >= chosen["G2"]["rho"]),
           "G2_subsystem_prior": group(a, chosen["G2"]["W"], data["subsystem_adjacency"])}
    res = {"hypothesis": "H4", "chosen": chosen, "alerts": len(a), "labelled_alerts": int((ev > 0).sum()),
           "methods": {k: summary(v, ev) for k, v in inc.items()}}
    t1, _ = contingency(inc["G1"], ev)
    t2, _ = contingency(inc["G2"], ev)
    res["bootstrap_G2_vs_G1"] = paired_event_bootstrap(t1, t2)
    b = res["bootstrap_G2_vs_G1"]
    res["verdict"] = "improvement" if b["ci95"][0] > 0 else "worse" if b["ci95"][1] < 0 else "no clear difference"
    (ROOT / "results" / "h4_result.json").write_text(json.dumps(res, indent=1, default=float))
    for k, v in res["methods"].items():
        log(f"test {k}: pairwise F1 {v['f1']:.3f} (P {v['precision']:.3f}, R {v['recall']:.3f}), "
            f"incidents per event {v['incidents_per_event']:.2f}, mixed incidents {v['mixed_incidents']}, incidents {v['incidents_total']}")
    log(f"G2 vs G1: diff {b['diff']:+.3f}, 95% CI [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}] -> {res['verdict']}")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    with open(ROOT / "results" / "test_log.md", "a", encoding="utf8") as fh:
        fh.write(f"| {time.strftime('%Y-%m-%d')} | {commit} | H4 | C3 graph grouping vs time-only grouping, all 76 channels | "
                 f"pairwise F1 {res['methods']['G2']['f1']:.3f} vs {res['methods']['G1']['f1']:.3f}, diff {b['diff']:+.3f}, "
                 f"95% CI [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]: {res['verdict']} |\n")


if __name__ == "__main__":
    {"dev": dev, "test": test}[sys.argv[1]]()
