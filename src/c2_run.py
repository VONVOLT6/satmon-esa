"""Run C2 (PROTOCOL.md addendum 2). Stage "dev" trains and calibrates; stage "test" runs H3a/H3b once.

Usage: python src/c2_run.py dev
       python src/c2_run.py test
Progress: results/c2.log. Per-seed alarm flags are cached in models/<run>/ so a stage can resume.
"""
import json
import pickle
import subprocess
import sys
import time

import numpy as np

from c2_reliability import (FAULT_TYPES, Classifier, PeerModel, alarm_events, choose_threshold, event_features,
                            inject, label_intervals, overlaps, repredict, system_flags)
from metrics import global_scores, load_truth, test_events
from stats import components, f05, paired_bootstrap
from thresholding import ROOT, flags_from

SEEDS = [42, 1, 2, 3, 4]
FLOOR = 0.007
LOG = ROOT / "results" / "c2.log"


def log(message):
    with open(LOG, "a", encoding="utf8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    print(message, flush=True)


def load_lstm(seed):
    import torch

    from train import Telemanom

    run = ROOT / "models" / f"telemanom_21_months_seed{seed}"
    with np.load(run / "scaling.npz") as d:
        means, stds = d["means"], d["stds"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Telemanom(6, 6).to(device)
    model.load_state_dict(torch.load(run / "last.pt", map_location=device))
    return model, means, stds, device


def injected_flags(seed, tag, y_hat, values_inj, faults):
    path = ROOT / "models" / f"telemanom_21_months_seed{seed}" / f"c2_{tag}_flags.npy"
    if path.exists():
        return np.load(path)
    model, means, stds, device = load_lstm(seed)
    y_inj = repredict(model, y_hat, values_inj, faults, means, stds, device)
    flags = flags_from(y_inj, values_inj, FLOOR)
    np.save(path, flags)
    return flags


def common():
    with np.load(ROOT / "data" / "processed" / "21_months.train.npz") as d:
        tv, tf, channels = d["values"], d["flags"], [str(c) for c in d["channels"]]
    with np.load(ROOT / "models" / "telemanom_21_months_seed42" / "scaling.npz") as d:
        means, stds = d["means"], d["stds"]
    return PeerModel().fit(tv, tf, means, stds), stds.astype(np.float32), channels


def dev():
    peer, sigma, channels = common()
    with np.load(ROOT / "data" / "processed" / "84_months.train.npz") as d:
        keep = d["index"] > np.datetime64("2001-10-01")
        values, labels = d["values"][keep], d["flags"][keep]
    values_inj, faults = inject(values, labels, sigma, seed=101)
    real = label_intervals(labels)
    fault_iv = [(f["start"], f["end"]) for f in faults]
    feats, ys = [], []
    for seed in SEEDS:
        flags = injected_flags(seed, "dev", np.load(ROOT / "models" / f"telemanom_21_months_seed{seed}" / "y_hat_dev.npy"), values_inj, faults)
        events = alarm_events(flags)
        is_fault, is_real = overlaps(events, fault_iv), overlaps(events, real)
        keep_ev = is_fault ^ is_real
        f = event_features(events, flags, values_inj, peer)
        feats.append(f[keep_ev])
        ys.append(is_fault[keep_ev].astype(int))
        log(f"dev seed {seed}: {len(events)} alarms, {is_fault.sum()} on injected faults, {is_real.sum()} on real events")
    X, y = np.concatenate(feats), np.concatenate(ys)
    clf = Classifier().fit(X, y)
    p = clf.proba(X)
    threshold, f1 = choose_threshold(p, y)
    with open(ROOT / "results" / "c2_model.pkl", "wb") as fh:
        pickle.dump({"classifier": clf, "peer": peer, "threshold": threshold}, fh)
    summary = {"threshold": threshold, "dev_sensor_F1": f1, "dev_events": {"sensor": int(y.sum()), "real": int((y == 0).sum())},
               "real_suppressed": int(((p >= threshold) & (y == 0)).sum()), "faults": faults,
               "coefficients": dict(zip(__import__("c2_reliability").FEATURES, clf.model.coef_[0].round(3).tolist()))}
    (ROOT / "results" / "c2_dev.json").write_text(json.dumps(summary, indent=1))
    log(f"dev: threshold {threshold:.3f}, sensor F1 {f1:.3f}, real alarms suppressed {summary['real_suppressed']} of {(y == 0).sum()}")
    log(f"coefficients {summary['coefficients']}")


def test():
    with open(ROOT / "results" / "c2_model.pkl", "rb") as fh:
        saved = pickle.load(fh)
    clf, peer, threshold = saved["classifier"], saved["peer"], saved["threshold"]
    _, sigma, channels = common()
    with np.load(ROOT / "data" / "processed" / "test.npz") as d:
        index, values, labels = d["index"], d["values"], d["flags"]
    truth = test_events(load_truth(channels), index)
    values_inj, faults = inject(values, labels, sigma, seed=202)
    fault_iv = [(f["start"], f["end"]) for f in faults]

    res = {"hypotheses": ["H3a", "H3b"], "threshold": threshold, "per_seed": {}, "faults": faults}
    parts = {"injected": ([], []), "clean": ([], [])}
    attribution = {k: {"faults": 0, "alarmed": 0, "attributed": 0} for k in FAULT_TYPES}
    for seed in SEEDS:
        run = ROOT / "models" / f"telemanom_21_months_seed{seed}"
        for cond, flags, vals in [("injected", injected_flags(seed, "test", np.load(run / "y_hat_last.npy"), values_inj, faults), values_inj),
                                  ("clean", np.load(run / "flags_last_Telemanom-ESA-Pruned.npy"), values)]:
            events = alarm_events(flags)
            sensor = clf.proba(event_features(events, flags, vals, peer)) >= threshold
            sysf = system_flags(flags, events, sensor)
            pb, pc = components(index, flags, truth), components(index, sysf, truth)
            parts[cond][0].append(pb)
            parts[cond][1].append(pc)
            res["per_seed"].setdefault(str(seed), {})[cond] = {"B0": f05(pb.sum()), "C2": f05(pc.sum()),
                                                               "alarms": len(events), "sensor_labelled": int(sensor.sum())}
            if cond == "injected":
                for f in faults:
                    a = attribution[f["type"]]
                    a["faults"] += 1
                    alarmed = bool(flags[f["start"]:f["end"]].any())
                    a["alarmed"] += alarmed
                    hit = overlaps(events, [(f["start"], f["end"])])
                    a["attributed"] += bool(alarmed and (hit & sensor).any())
            log(f"test seed {seed} {cond}: B0 F0.5 {res['per_seed'][str(seed)][cond]['B0']['f05']:.4f}, "
                f"C2 {res['per_seed'][str(seed)][cond]['C2']['f05']:.4f}, {sensor.sum()} of {len(events)} alarms labelled sensor")
    res["attribution"] = attribution
    for cond in parts:
        res[cond] = paired_bootstrap(*parts[cond])
        b = res[cond]
        log(f"{cond}: B0 {b['A']:.4f}  C2 {b['B']:.4f}  diff {b['diff']:+.4f}  95% CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]")
    # Holm across H3a (one-sided improvement) and H3b (non-inferiority): report raw intervals and verdicts
    res["H3a_verdict"] = "improvement" if res["injected"]["ci95"][0] > 0 else "not shown"
    res["H3b_verdict"] = "no harm (non-inferior)" if res["clean"]["ci95"][0] > -0.02 else "not shown"
    (ROOT / "results" / "h3_result.json").write_text(json.dumps(res, indent=1))
    log(f"attribution {attribution}")
    log(f"verdicts: H3a {res['H3a_verdict']}, H3b {res['H3b_verdict']}")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    with open(ROOT / "results" / "test_log.md", "a", encoding="utf8") as fh:
        for cond, h in [("injected", "H3a"), ("clean", "H3b")]:
            b = res[cond]
            fh.write(f"| {time.strftime('%Y-%m-%d')} | {commit} | {h} | C2 reliability, {cond} test, 21 months, 5 seeds | "
                     f"F0.5 {b['B']:.4f} vs {b['A']:.4f}, diff {b['diff']:+.4f}, 95% CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]: "
                     f"{res[h + '_verdict']} |\n")


if __name__ == "__main__":
    {"dev": dev, "test": test}[sys.argv[1]]()
