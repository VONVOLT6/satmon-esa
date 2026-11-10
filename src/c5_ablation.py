"""C5 ablation: what each contribution adds in the combined pipeline (PROTOCOL.md addendum 5).

Usage: python src/c5_ablation.py dev    (fit the no-C2 variant of the C4 class model on development alarms)
       python src/c5_ablation.py test   (run every configuration once on the test period)
Progress: results/c5.log. Detector flags are cached per seed in models/<run>/c5_*.npy, so a run can resume.
"""
import json
import pickle
import subprocess
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from c1_router import CheapTier, bursts, routed_flags, triggers
from c2_reliability import alarm_events, event_features, inject, label_intervals, overlaps, repredict, system_flags
from c3_correlation import group
from c4_policy import CENTRAL, GRID, classes_of, cost_matrix
from c4_policy import period as c4_period
from metrics import load_truth, test_events
from stats import components, f05, paired_bootstrap
from thresholding import ROOT, WINDOW

SEEDS = [42, 1, 2, 3, 4]
VARIANTS = {"conf": ("confidence",), "conf+impact": ("confidence", "impact"), "full": ("confidence", "novelty", "impact")}
CONFIGS = {  # detector, C2, C3, C4
    "A0": ("B0", False, False, False),
    "A1": ("cheap", False, False, False),
    "A2": ("conf", False, False, False),
    "A3": ("conf+impact", False, False, False),
    "A4": ("full", False, False, False),
    "A5": ("full", True, False, False),
    "A6": ("full", False, True, False),
    "A7": ("full", False, False, True),
    "A8": ("full", True, True, True),
    "L1": ("B0", True, True, True),
    "L2": ("full", False, True, True),
    "L3": ("full", True, False, True),
    "L4": ("full", True, True, False),
}
LOG = ROOT / "results" / "c5.log"
NOC2_MODEL = ROOT / "results" / "c5_c4_noc2.pkl"


def log(message):
    with open(LOG, "a", encoding="utf8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    print(message, flush=True)


def run_dir(seed):
    return ROOT / "models" / f"telemanom_21_months_seed{seed}"


def noc2_features(events, flags):
    dur = np.log([max(e - s, 1) for s, e in events])
    n_ch = [float((flags[s:e].max(axis=0) > 0).sum()) for s, e in events]
    return np.c_[dur, n_ch].reshape(-1, 2)


def dev():
    _, values_inj, labels, faults = c4_period("dev")
    xs, ys = [], []
    for seed in SEEDS:
        flags = np.load(run_dir(seed) / "c2_dev_flags.npy")
        events = alarm_events(flags)
        xs.append(noc2_features(events, flags))
        ys.append(classes_of(events, labels, faults))
    x, y = np.concatenate(xs), np.concatenate(ys)
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(x), y)
    NOC2_MODEL.write_bytes(pickle.dumps({"scaler": scaler, "clf": clf, "classes": clf.classes_.tolist()}))
    log(f"dev: no-C2 C4 model fitted on {len(y)} development alarms (5 seeds pooled), classes {clf.classes_.tolist()}")


class Test:
    def __init__(self):
        with np.load(ROOT / "data" / "processed" / "21_months.train.npz") as d:
            train_values, train_flags = d["values"], d["flags"]
        with np.load(ROOT / "data" / "processed" / "test.npz") as d:
            self.index, self.values, self.labels = d["index"], d["values"], d["flags"]
            channels = [str(c) for c in d["channels"]]
        self.truth = test_events(load_truth(channels), self.index)
        with np.load(run_dir(42) / "scaling.npz") as d:
            sigma = d["stds"].astype(np.float32)
            tier = CheapTier().fit(train_values, train_flags, d["means"], d["stds"])
        self.values_inj, self.faults = inject(self.values, self.labels, sigma, seed=202)
        self.fault_iv = [(f["start"], f["end"]) for f in self.faults]
        self.real_iv = label_intervals(self.labels)
        self.tau = json.loads((ROOT / "results" / "c1_dev.json").read_text())["chosen_tau"]
        self.scores = {"clean": tier.scores(self.values), "injected": tier.scores(self.values_inj)}
        self.c2 = pickle.loads((ROOT / "results" / "c2_model.pkl").read_bytes())
        self.c4 = pickle.loads((ROOT / "results" / "c4_model.pkl").read_bytes())
        self.c4_noc2 = pickle.loads(NOC2_MODEL.read_bytes())
        self.window = json.loads((ROOT / "results" / "c3_dev.json").read_text())["chosen"]["G2"]["W"]

    def vals(self, cond):
        return self.values_inj if cond == "injected" else self.values

    def y_hat(self, seed, cond):
        if cond == "clean":
            return np.load(run_dir(seed) / "y_hat_last.npy")
        path = run_dir(seed) / "c5_y_hat_injected.npy"
        if not path.exists():
            from c2_run import load_lstm

            model, means, stds, device = load_lstm(seed)
            np.save(path, repredict(model, np.load(run_dir(seed) / "y_hat_last.npy"), self.values_inj, self.faults,
                                    means, stds, device))
        return np.load(path)

    def detector(self, seed, det, cond):
        """Alarm flags and LSTM share of one detector."""
        if det == "B0":
            name = "c2_test_flags.npy" if cond == "injected" else "flags_last_Telemanom-ESA-Pruned.npy"
            return np.load(run_dir(seed) / name), 1.0
        if det == "cheap":
            trig = triggers(self.scores[cond], self.tau)
            return np.repeat(trig[:, None], self.values.shape[1], axis=1).astype(np.uint8), 0.0
        trig = triggers(self.scores[cond], self.tau, VARIANTS[det])
        y = self.y_hat(seed, cond)
        mask, valid = bursts(trig, len(y))
        path = run_dir(seed) / f"c5_{cond}_{det}.npy"
        if not path.exists():
            t0 = time.time()
            np.save(path, routed_flags(y, self.vals(cond), mask, valid))
            log(f"seed {seed} {cond} {det}: flags computed [{time.time() - t0:.0f}s]")
        return np.load(path), float(mask.mean())

    def pipeline(self, flags, cond, c2_on, c3_on, c4_on):
        values = self.vals(cond)
        events = alarm_events(flags)
        feats = event_features(events, flags, values, self.c2["peer"])
        p = self.c2["classifier"].proba(feats)
        sensor = (p >= self.c2["threshold"]) if c2_on else np.zeros(len(events), bool)
        sysf = system_flags(flags, events, sensor) if c2_on else flags
        out = {"parts": components(self.index, sysf, self.truth), "alarms": len(events)}
        if cond == "clean":
            return out

        # operator load
        if c3_on:
            incidents = group([(int(sen), s, e) for (s, e), sen in zip(events, sensor)], self.window, np.eye(2, dtype=bool))
        else:
            incidents = np.arange(len(events))
        is_real, is_fault = overlaps(events, self.real_iv), overlaps(events, self.fault_iv)
        ids = np.unique(incidents)
        mixed = sum(bool(is_real[incidents == i].any() and is_fault[incidents == i].any()) for i in ids)
        out.update(items=len(ids), mixed=int(mixed))

        # response cost, per alarm event
        y = classes_of(events, self.labels, self.faults)
        dur = np.log([max(e - s, 1) for s, e in events])
        if c4_on and c2_on:
            m, x = self.c4, np.c_[feats, p, dur]
        elif c4_on:
            m, x = self.c4_noc2, noc2_features(events, flags)
        else:
            m = x = None

        def actions(g):
            if m is not None:
                proba = np.zeros((len(y), 4))
                if len(y):
                    proba[:, m["classes"]] = m["clf"].predict_proba(m["scaler"].transform(x))
                return (proba @ cost_matrix(g).T).argmin(axis=1)
            return np.where(sensor, 1, 0)  # P1 with C2 (tickets sensor faults), P0 without (sensor is all False)

        regrets = []
        for g in GRID:
            cm = cost_matrix(g)
            regrets.append(cm[actions(g), y].sum() - cm[:, y].min(axis=0).sum())
        cm = cost_matrix(CENTRAL)
        per_event = cm[actions(CENTRAL), y]
        month = np.array([str(np.datetime64(self.index[s], "M")) for s, _ in events])
        out.update(cost=float(per_event.sum()), regret=float(np.mean(regrets)),
                   month_cost={mo: float(per_event[month == mo].sum()) for mo in np.unique(month)})
        return out


def cost_bootstrap(a, b, n=5000):
    """Monthly-block bootstrap of the mean (over seeds) central-point cost difference b - a."""
    months = sorted({mo for r in a + b for mo in r["month_cost"]})
    arr = np.array([[[r["month_cost"].get(mo, 0.0) for mo in months] for r in side] for side in (a, b)])
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(months), len(months))
        s = arr[:, :, idx].sum(axis=2).mean(axis=1)
        diffs.append(s[1] - s[0])
    obs = arr.sum(axis=2).mean(axis=1)
    return {"A": float(obs[0]), "B": float(obs[1]), "diff": float(obs[1] - obs[0]),
            "ci95": [float(v) for v in np.percentile(diffs, [2.5, 97.5])]}


def test():
    t = Test()
    runs = {name: {"injected": [], "clean": []} for name in CONFIGS}
    shares = {name: {"injected": [], "clean": []} for name in CONFIGS}
    for seed in SEEDS:
        for cond in ["injected", "clean"]:
            cache = {}
            for name, (det, c2_on, c3_on, c4_on) in CONFIGS.items():
                if det not in cache:
                    cache[det] = t.detector(seed, det, cond)
                flags, share = cache[det]
                runs[name][cond].append(t.pipeline(flags, cond, c2_on, c3_on, c4_on))
                shares[name][cond].append(share)
            log(f"seed {seed} {cond}: " + ", ".join(f"{n} {f05(runs[n][cond][-1]['parts'].sum())['f05']:.3f}" for n in CONFIGS))

    table = {}
    for name in CONFIGS:
        inj, cln = runs[name]["injected"], runs[name]["clean"]
        table[name] = {
            "F05_injected": float(np.mean([f05(r["parts"].sum())["f05"] for r in inj])),
            "F05_clean": float(np.mean([f05(r["parts"].sum())["f05"] for r in cln])),
            "lstm_share_injected": float(np.mean(shares[name]["injected"])),
            "lstm_share_clean": float(np.mean(shares[name]["clean"])),
            "alarms": float(np.mean([r["alarms"] for r in inj])),
            "items": float(np.mean([r["items"] for r in inj])),
            "mixed_items": float(np.mean([r["mixed"] for r in inj])),
            "cost_central": float(np.mean([r["cost"] for r in inj])),
            "mean_regret": float(np.mean([r["regret"] for r in inj])),
        }
    boot = {}
    for other in ["A0", "L1", "L2", "L3", "L4"]:
        boot[f"A8-{other}"] = paired_bootstrap([r["parts"] for r in runs[other]["injected"]],
                                               [r["parts"] for r in runs["A8"]["injected"]])
    boot["cost_A8-L4"] = cost_bootstrap(runs["L4"]["injected"], runs["A8"]["injected"])
    res = {"addendum": 5, "tau": t.tau, "W": t.window, "configs": {k: list(v) for k, v in CONFIGS.items()},
           "table": table, "bootstrap": boot}
    (ROOT / "results" / "c5_result.json").write_text(json.dumps(res, indent=1))

    log("config | F0.5 injected | F0.5 clean | LSTM share | alarms | items | mixed | cost (central) | mean regret")
    for name, r in table.items():
        log(f"{name} | {r['F05_injected']:.3f} | {r['F05_clean']:.3f} | {r['lstm_share_injected'] * 100:.1f}% | {r['alarms']:.1f} | "
            f"{r['items']:.1f} | {r['mixed_items']:.1f} | {r['cost_central']:.1f} | {r['mean_regret']:.1f}")
    for k, b in boot.items():
        log(f"{k}: {b['diff']:+.4f} 95% CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    a8, b = table["A8"], boot["A8-A0"]
    with open(ROOT / "results" / "test_log.md", "a", encoding="utf8") as fh:
        fh.write(f"| {time.strftime('%Y-%m-%d')} | {commit} | C5 (descriptive) | ablation A0-A8, L1-L4 on injected and clean test, 5 seeds | "
                 f"A8 injected F0.5 {a8['F05_injected']:.4f} vs A0 {table['A0']['F05_injected']:.4f}, diff {b['diff']:+.4f} "
                 f"[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]; A8 LSTM share {a8['lstm_share_injected'] * 100:.1f}%, "
                 f"central cost {a8['cost_central']:.1f} vs {table['A0']['cost_central']:.1f} |\n")


if __name__ == "__main__":
    {"dev": dev, "test": test}[sys.argv[1]]()
