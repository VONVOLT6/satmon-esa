"""C4 cost-aware response replay (PROTOCOL.md addendum 4).

Every alarm event of the always-on LSTM on the fault-injected periods gets one of three actions:
page an operator, open a maintenance ticket, or ignore. Policies are compared by the total cost
they would have caused on the recorded timeline, across a declared cost grid.

Usage: python src/c4_policy.py dev    (train the class model on development alarms)
       python src/c4_policy.py test   (replay on the test period once, test H5)
"""
import itertools
import json
import pickle
import subprocess
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from c2_reliability import alarm_events, event_features, inject, label_intervals, overlaps
from metrics import load_truth
from thresholding import ROOT

SEEDS = [42, 1, 2, 3, 4]
CLASSES = ["anomaly", "rare", "sensor", "false"]
ACTIONS = ["page", "ticket", "ignore"]
GRID = [dict(M=m, D=d, F=f, S=s, T=0.5) for m, d, f, s in itertools.product([20, 50, 100], [2, 5, 10], [1, 3], [1, 3])]
CENTRAL = dict(M=50, D=5, F=1, S=1, T=0.5)
LOG = ROOT / "results" / "c4.log"


def log(message):
    with open(LOG, "a", encoding="utf8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    print(message, flush=True)


def cost_matrix(g):
    """Rows = actions, columns = classes."""
    return np.array([[0, g["F"], g["F"], g["F"]],
                     [g["D"], g["T"], 0, g["T"]],
                     [g["M"], 0, g["S"], 0]], float)


def period(name):
    """Values with injected faults, real-label masks per category, faults and per-seed cached flags."""
    with np.load(ROOT / "models" / "telemanom_21_months_seed42" / "scaling.npz") as d:
        sigma = d["stds"].astype(np.float32)
    if name == "dev":
        with np.load(ROOT / "data" / "processed" / "84_months.train.npz") as d:
            keep = d["index"] > np.datetime64("2001-10-01")
            index, values, labels = d["index"][keep], d["values"][keep], d["flags"][keep]
        seed = 101
    else:
        with np.load(ROOT / "data" / "processed" / "test.npz") as d:
            index, values, labels = d["index"], d["values"], d["flags"]
        seed = 202
    values_inj, faults = inject(values, labels, sigma, seed=seed)
    return index, values_inj, labels, faults


def classes_of(events, labels, faults):
    anomaly = label_intervals(labels == 1)
    rare = label_intervals(labels == 2)
    fault_iv = [(f["start"], f["end"]) for f in faults]
    is_a, is_r, is_s = overlaps(events, anomaly), overlaps(events, rare), overlaps(events, fault_iv)
    return np.where(is_a, 0, np.where(is_r, 1, np.where(is_s, 2, 3)))


def alarm_table(name):
    with open(ROOT / "results" / "c2_model.pkl", "rb") as fh:
        c2 = pickle.load(fh)
    index, values_inj, labels, faults = period(name)
    rows = []
    for seed in SEEDS:
        flags = np.load(ROOT / "models" / f"telemanom_21_months_seed{seed}" / f"c2_{name}_flags.npy")
        events = alarm_events(flags)
        feats = event_features(events, flags, values_inj, c2["peer"])
        p_sensor = c2["classifier"].proba(feats)
        dur = np.log([max(e - s, 1) for s, e in events])
        x = np.c_[feats, p_sensor, dur]
        y = classes_of(events, labels, faults)
        month = np.array([str(np.datetime64(index[s], "M")) for s, _ in events])
        rows.append({"seed": seed, "x": x, "y": y, "c2_sensor": p_sensor >= c2["threshold"], "month": month})
    return rows


def policy_actions(name, table, model=None, g=None):
    if name == "P0":
        return np.zeros(len(table["y"]), int)
    if name == "P1":
        return np.where(table["c2_sensor"], 1, 0)
    if name == "oracle":
        return cost_matrix(g)[:, table["y"]].argmin(axis=0)
    proba = np.zeros((len(table["y"]), len(CLASSES)))
    proba[:, model["classes"]] = model["clf"].predict_proba(model["scaler"].transform(table["x"]))
    return (proba @ cost_matrix(g).T).argmin(axis=1)


def total_cost(actions, y, g):
    return float(cost_matrix(g)[actions, y].sum())


def dev():
    tables = alarm_table("dev")
    x = np.concatenate([t["x"] for t in tables])
    y = np.concatenate([t["y"] for t in tables])
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(x), y)
    model = {"scaler": scaler, "clf": clf, "classes": clf.classes_.tolist()}
    (ROOT / "results" / "c4_model.pkl").write_bytes(pickle.dumps(model))
    counts = {CLASSES[k]: int((y == k).sum()) for k in range(len(CLASSES))}
    log(f"dev alarms by class (5 seeds pooled): {counts}; classes in model {[CLASSES[k] for k in model['classes']]}")


def test():
    model = pickle.loads((ROOT / "results" / "c4_model.pkl").read_bytes())
    tables = alarm_table("test")
    policies = ["P0", "P1", "P2", "oracle"]
    grid_rows = []
    for g in GRID:
        cost = {p: float(np.mean([total_cost(policy_actions(p, t, model, g), t["y"], g) for t in tables])) for p in policies}
        regret = {p: cost[p] - cost["oracle"] for p in policies if p != "oracle"}
        grid_rows.append({**g, "cost": cost, "regret": regret, "best": min(regret, key=regret.get)})
    wins = sum(r["regret"]["P2"] < r["regret"]["P0"] and r["regret"]["P2"] < r["regret"]["P1"] for r in grid_rows)
    first = {p: sum(r["best"] == p for r in grid_rows) for p in ["P0", "P1", "P2"]}

    # paired monthly-block bootstrap of the P2 - P1 cost difference at the central grid point
    rng = np.random.default_rng(0)
    months = sorted(set(np.concatenate([t["month"] for t in tables])))
    per_month = np.zeros((len(tables), len(months), 2))
    for i, t in enumerate(tables):
        a1, a2 = policy_actions("P1", t, model, CENTRAL), policy_actions("P2", t, model, CENTRAL)
        cm = cost_matrix(CENTRAL)
        for j, m in enumerate(months):
            sel = t["month"] == m
            per_month[i, j] = [cm[a1[sel], t["y"][sel]].sum(), cm[a2[sel], t["y"][sel]].sum()]
    diffs = []
    for _ in range(5000):
        idx = rng.integers(0, len(months), len(months))
        s = per_month[:, idx].sum(axis=1).mean(axis=0)
        diffs.append(s[1] - s[0])
    obs = per_month.sum(axis=1).mean(axis=0)
    boot = {"P1": float(obs[0]), "P2": float(obs[1]), "diff": float(obs[1] - obs[0]),
            "ci95": [float(v) for v in np.percentile(diffs, [2.5, 97.5])]}

    verdict = "supported" if wins >= 0.8 * len(GRID) else "not supported"
    counts = {CLASSES[k]: int(sum((t["y"] == k).sum() for t in tables)) for k in range(len(CLASSES))}
    res = {"hypothesis": "H5", "grid": grid_rows, "P2_beats_both": wins, "grid_points": len(GRID),
           "ranked_first": first, "central_bootstrap_P2_minus_P1": boot, "test_alarm_classes": counts, "verdict": verdict}
    (ROOT / "results" / "h5_result.json").write_text(json.dumps(res, indent=1))
    c = next(r for r in grid_rows if all(r[k] == v for k, v in CENTRAL.items()))
    log(f"test alarm classes (5 seeds pooled): {counts}")
    log(f"central grid point cost per test period: {c['cost']}")
    log(f"P2 beats P0 and P1 at {wins}/{len(GRID)} grid points; ranked first: {first}")
    log(f"central bootstrap P2 - P1: {boot['diff']:+.1f}, 95% CI [{boot['ci95'][0]:+.1f}, {boot['ci95'][1]:+.1f}]; H5 {verdict}")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    with open(ROOT / "results" / "test_log.md", "a", encoding="utf8") as fh:
        fh.write(f"| {time.strftime('%Y-%m-%d')} | {commit} | H5 | C4 cost-aware policy replay on injected test, 36-point cost grid | "
                 f"P2 lower regret than P0 and P1 at {wins}/{len(GRID)} points; central P2-P1 cost {boot['diff']:+.1f} "
                 f"[{boot['ci95'][0]:+.1f}, {boot['ci95'][1]:+.1f}]: {verdict} |\n")


if __name__ == "__main__":
    {"dev": dev, "test": test}[sys.argv[1]]()
