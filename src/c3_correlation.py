"""C3 incident correlation on all 76 Mission 1 channels (PROTOCOL.md addendum 3).

1. Per-channel alerts: a univariate autoregressive forecaster per channel, fitted on nominal
   training data; the smoothed absolute residual (in training standard deviations) raises an alert
   above that channel's 99.99th training percentile.
2. Dependency graph: correlation between channels' residual activity in training (10-minute maxima).
3. Grouping: G0 every alert alone; G1 alerts within W samples of each other (time only);
   G2 alerts within W samples AND linked in the graph (proposed).
4. Scoring against ESA event IDs: pairwise grouping precision/recall/F1, incidents per event,
   incidents that mix events. Event-level bootstrap with resampling weights for the H4 interval.
"""
import json

import numpy as np
import pandas as pd

LAGS = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233])
SPAN = 30
QUANTILE = 0.9999
MIN_LEN = 5
MERGE_GAP = 60
BIN = 20  # 10-minute bins for the activity correlation


def load_full(root):
    d = root / "data" / "processed" / "full"
    index = np.load(d / "index.npy")
    channels = json.loads((d / "channels.json").read_text())
    shape = (len(index), len(channels))
    values = np.memmap(d / "values.f32", np.float32, "r", shape=shape)
    labels = np.memmap(d / "labels.u8", np.uint8, "r", shape=shape)
    ids = np.memmap(d / "ids.u16", np.uint16, "r", shape=shape)
    return index, channels, values, labels, ids


class ChannelDetectors:
    def fit(self, train_values, train_labels):
        n_ch = train_values.shape[1]
        self.mean = np.zeros(n_ch, np.float32)
        self.std = np.ones(n_ch, np.float32)
        self.w, self.res_std, self.threshold = [], np.ones(n_ch, np.float32), np.full(n_ch, np.inf, np.float32)
        lag_max = int(LAGS.max())
        for c in range(n_ch):
            v = np.asarray(train_values[:, c], np.float32)
            clean = np.asarray(train_labels[:, c]) == 0
            self.mean[c], self.std[c] = v[clean].mean(), v[clean].std() + 1e-6
            x = (v - self.mean[c]) / self.std[c]
            rows = np.arange(lag_max, len(x))
            rows = rows[clean[rows] & clean[rows - lag_max]]
            rows = rows[:: max(1, len(rows) // 200_000)]
            a = np.c_[np.stack([x[rows - lag] for lag in LAGS], axis=1), np.ones(len(rows), np.float32)]
            w, *_ = np.linalg.lstsq(a, x[rows], rcond=None)
            self.w.append(w.astype(np.float32))
            self.res_std[c] = (x[rows] - a @ w).std() + 1e-6
            act = self.activity_channel(v, c)
            ok = clean.copy()
            ok[:lag_max + SPAN] = False
            if ok.any():
                self.threshold[c] = np.quantile(act[ok], QUANTILE)
        return self

    def activity_channel(self, v, c):
        x = (np.asarray(v, np.float32) - self.mean[c]) / self.std[c]
        lag_max = int(LAGS.max())
        pred = np.zeros_like(x)
        a = np.stack([x[lag_max - lag:len(x) - lag] for lag in LAGS], axis=1)
        pred[lag_max:] = a @ self.w[c][:-1] + self.w[c][-1]
        z = np.abs(x - pred) / self.res_std[c]
        z[:lag_max] = 0
        return pd.Series(z).ewm(span=SPAN).mean().to_numpy(np.float32)

    def alerts(self, values):
        """List of (channel, start, end) with end exclusive, plus the activity matrix in 10-minute maxima."""
        out = []
        binned = []
        for c in range(values.shape[1]):
            act = self.activity_channel(values[:, c], c)
            on = act > self.threshold[c]
            edges = np.flatnonzero(np.diff(np.r_[False, on, False].astype(np.int8)))
            runs = list(zip(edges[::2], edges[1::2]))
            merged = []
            for s, e in runs:
                if merged and s - merged[-1][1] <= MERGE_GAP:
                    merged[-1] = (merged[-1][0], e)
                else:
                    merged.append((s, e))
            out += [(c, s, e) for s, e in merged if e - s >= MIN_LEN]
            n = len(act) // BIN * BIN
            binned.append((act[:n] / self.threshold[c]).reshape(-1, BIN).max(axis=1))
        return out, np.stack(binned, axis=1)


def dependency_graph(train_activity):
    """Pearson correlation of log activity between channels (training only)."""
    a = np.log1p(np.nan_to_num(train_activity, posinf=0))
    a = a[:, a.std(axis=0) > 0] if False else a
    std = a.std(axis=0)
    corr = np.corrcoef(np.where(std > 0, a, a + np.random.default_rng(0).normal(0, 1e-9, a.shape)), rowvar=False)
    np.fill_diagonal(corr, 1.0)
    return np.nan_to_num(corr)


def group(alerts, window, adjacency=None):
    """Union-find over alerts: link two alerts when their spans come within `window` samples
    (and, if adjacency is given, their channels are linked or identical). Returns incident ids."""
    n = len(alerts)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = sorted(range(n), key=lambda i: alerts[i][1])
    active = []  # (end + window, index) of alerts that can still link to later ones
    for i in order:
        c, s, e = alerts[i]
        active = [(reach, j) for reach, j in active if reach >= s]
        for _, j in active:
            cj = alerts[j][0]
            if adjacency is None or cj == c or adjacency[c, cj]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
        active.append((e + window, i))
    return np.array([find(i) for i in range(n)])


def alert_event_ids(alerts, ids):
    """Most frequent labelled event ID inside each alert, on the alert's own channel (0 = none)."""
    out = np.zeros(len(alerts), np.int64)
    for k, (c, s, e) in enumerate(alerts):
        seg = np.asarray(ids[s:e, c])
        seg = seg[seg > 0]
        if len(seg):
            out[k] = np.bincount(seg).argmax()
    return out


def contingency(incidents, event_ids):
    """Table rows = labelled events, columns = incidents, counts = alerts (labelled alerts only)."""
    m = event_ids > 0
    ev, inc = event_ids[m], incidents[m]
    ev_u, ev_i = np.unique(ev, return_inverse=True)
    in_u, in_i = np.unique(inc, return_inverse=True)
    table = np.zeros((len(ev_u), len(in_u)), np.int64)
    np.add.at(table, (ev_i, in_i), 1)
    return table, ev_u


def pairwise(table, weights=None):
    """Pairwise precision/recall/F1 of grouping, with optional event resampling weights."""
    w = np.ones(table.shape[0]) if weights is None else weights
    same = (table * (table - 1) / 2).sum(axis=1)                 # same-event pairs grouped together
    total = table.sum(axis=1) * (table.sum(axis=1) - 1) / 2       # all same-event pairs
    tp = (w * same).sum()
    fn = (w * (total - same)).sum()
    col = (w[:, None] * table).sum(axis=0)
    fp = ((col ** 2 - (w[:, None] ** 2 * table ** 2).sum(axis=0)) / 2).sum()  # cross-event pairs in one incident
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def summary(incidents, event_ids):
    table, _ = contingency(incidents, event_ids)
    stats = pairwise(table)
    stats["incidents_per_event"] = float((table > 0).sum(axis=1).mean()) if len(table) else 0.0
    stats["mixed_incidents"] = int(((table > 0).sum(axis=0) >= 2).sum())
    stats["incidents_total"] = int(len(np.unique(incidents)))
    stats["events_detected"] = int(len(table))
    return stats


def paired_event_bootstrap(table_a, table_b, n=5000, seed=0):
    """95% interval for F1(B) - F1(A), resampling labelled events (same rows in both tables)."""
    rng = np.random.default_rng(seed)
    k = table_a.shape[0]
    diffs = np.empty(n)
    for i in range(n):
        w = np.bincount(rng.integers(0, k, k), minlength=k).astype(float)
        diffs[i] = pairwise(table_b, w)["f1"] - pairwise(table_a, w)["f1"]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": pairwise(table_b)["f1"] - pairwise(table_a)["f1"], "ci95": [float(lo), float(hi)], "replicates": n}
