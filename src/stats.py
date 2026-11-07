"""Confidence intervals for the corrected event-wise F0.5, by monthly block bootstrap.

The official ESAScores gives one number per run. To say whether two detectors really differ,
we split the test period into calendar-month blocks and break the metric into per-block parts:
detected events, missed events, false-alarm events, false-alarm seconds and nominal seconds.
Resampling months with replacement and recombining the parts gives a bootstrap distribution.
Comparisons are paired: both detectors are scored on the same resampled months.

The decomposition follows ESA_ADB_metrics.ESAScores line by line. `check` confirms it reproduces
the official precision, recall and F0.5 on saved runs before any interval is reported.

Usage: python src/stats.py check
"""
import sys

import numpy as np
import pandas as pd
import portion as P

from metrics import CATEGORIES, global_scores, load_truth, test_events
from thresholding import ROOT

NS = 1e9


def pred_events(timestamps, flags):
    """Same intervals as timeeval.metrics.utils.convert_time_series_to_events."""
    t = pd.to_datetime(timestamps)
    on = flags.max(axis=1) > 0
    edges = np.flatnonzero(np.diff(np.r_[False, on, False].astype(np.int8)))
    events = []
    for a, b in zip(edges[::2], edges[1::2]):
        events.append(P.closed(t[a], t[b - 1]) if b == len(t) else P.closedopen(t[a], t[b]))
    return P.Interval(*events)


def components(timestamps, flags, truth, category="all"):
    """Per-month parts of the corrected event-wise F0.5, as a DataFrame indexed by month."""
    t = pd.to_datetime(timestamps)
    full = P.closed(min(truth["StartTime"].min(), t[0]), max(truth["EndTime"].max(), t[-1]))
    events_pred = pred_events(timestamps, flags)
    selected = truth[truth["Category"].isin(CATEGORIES[category])]

    months = pd.period_range(full.lower, full.upper, freq="M")
    rows = {m: {"tp": 0, "fn": 0, "fp": 0, "fp_s": 0.0, "nom_s": 0.0} for m in months}
    month_of = lambda ts: pd.Timestamp(ts).to_period("M")

    matched = [False] * len(events_pred)
    for _, gt in selected.groupby("ID"):
        gt_iv = P.Interval(*[P.closed(r.StartTime, r.EndTime) for r in gt.itertuples()])
        hit = False
        for k, pred in enumerate(events_pred):
            if pred.upper < gt_iv.lower or pred.lower > gt_iv.upper:
                continue
            if any(not (pred & g).empty for g in gt_iv):
                matched[k] = True
                hit = True
        rows[month_of(gt_iv.lower)]["tp" if hit else "fn"] += 1

    events_gt = P.Interval(*[P.closed(r.StartTime, r.EndTime) for r in truth.itertuples()])
    for k, pred in enumerate(events_pred):
        if not matched[k] and (pred & events_gt).empty:
            rows[month_of(pred.lower)]["fp"] += 1

    nominal = P.closed(full.lower, full.upper) - events_gt
    false_time = nominal & events_pred
    for m in months:
        block = P.closedopen(max(m.start_time, full.lower), min((m + 1).start_time, full.upper)) | P.singleton(full.upper) \
            if m == months[-1] else P.closedopen(max(m.start_time, full.lower), (m + 1).start_time)
        rows[m]["nom_s"] = sum((iv.upper - iv.lower).value / NS for iv in (nominal & block))
        rows[m]["fp_s"] = sum((iv.upper - iv.lower).value / NS for iv in (false_time & block))
    return pd.DataFrame.from_dict(rows, orient="index")


def f05(parts):
    """Corrected event-wise precision, recall and F0.5 from summed parts, as in ESAScores."""
    tp, fn, fp = parts["tp"], parts["fn"], parts["fp"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    if precision > 0:
        precision *= 1 - parts["fp_s"] / parts["nom_s"]
    recall = tp / (tp + fn) if tp + fn else 0.0
    denom = 0.25 * precision + recall
    return {"precision": precision, "recall": recall, "f05": 1.25 * precision * recall / denom if denom else 0.0}


def paired_bootstrap(parts_a, parts_b, n=5000, seed=0):
    """parts_a/parts_b: lists of per-month DataFrames (one per seed) for detectors A and B.

    Returns observed mean F0.5 of each, their difference, and a 95% interval for B - A.
    Months are resampled once per replicate and shared by every seed and both detectors.
    """
    months = parts_a[0].index
    arrays_a = [p.loc[months].to_numpy() for p in parts_a]
    arrays_b = [p.loc[months].to_numpy() for p in parts_b]
    cols = list(parts_a[0].columns)

    def mean_f05(arrays, idx):
        return float(np.mean([f05(dict(zip(cols, arr[idx].sum(axis=0))))["f05"] for arr in arrays]))

    everything = np.arange(len(months))
    obs_a, obs_b = mean_f05(arrays_a, everything), mean_f05(arrays_b, everything)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(months), len(months))
        diffs[i] = mean_f05(arrays_b, idx) - mean_f05(arrays_a, idx)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"A": obs_a, "B": obs_b, "diff": obs_b - obs_a, "ci95": [float(lo), float(hi)],
            "p_B_not_better": float(np.mean(diffs <= 0)), "months": len(months), "replicates": n}


def check():
    """The decomposition must reproduce the official scores on saved runs."""
    with np.load(ROOT / "data" / "processed" / "test.npz") as d:
        index, channels = d["index"], [str(c) for c in d["channels"]]
    truth = test_events(load_truth(channels), index)
    ok = True
    for name in ["Telemanom-ESA-Pruned", "Telemanom-ESA"]:
        flags = np.load(ROOT / "models" / "telemanom_21_months_seed42" / f"flags_last_{name}.npy")
        for category in ["all", "anomaly"]:
            mine = f05(components(index, flags, truth, category).sum())
            official = global_scores(index, flags, truth, category)
            same = all(abs(mine[a] - official[b]) < 1e-9 for a, b in
                       [("precision", "EW_precision"), ("recall", "EW_recall"), ("f05", "EW_F_0.50")])
            ok &= same
            print(f"{'PASS' if same else 'FAIL'} {name} {category}: mine F0.5 {mine['f05']:.6f}, official {official['EW_F_0.50']:.6f}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    if sys.argv[1:] == ["check"]:
        check()
