"""C2 sensor reliability: tell a broken sensor apart from a real spacecraft event (PROTOCOL.md addendum 2).

Channels 41-46 form one sensor group, and real events almost always move all of them together.
So for every alarm we ask whether one channel disagrees with its five peers, and whether its
signal looks stuck, noisy or out of range. A logistic model turns these features into
P(sensor fault). Alarms above a threshold become "sensor fault" and leave the system-alarm stream.

ESA-AD has no sensor-fault labels, so faults are injected; claims hold for injected faults only.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from thresholding import WINDOW

FAULT_TYPES = ["stuck", "drift", "bias", "noise", "dropout"]
MIN_LEN, MAX_LEN = 240, 2880   # 2 to 24 hours of 30-second samples
GAP = 3000


def inject(values, label_flags, sigma, per_type=10, seed=0):
    """Return a copy of values with single-channel faults, and the list of faults."""
    rng = np.random.default_rng(seed)
    out = values.copy()
    busy = (label_flags > 0).any(axis=1)
    taken = np.zeros(len(values), bool)
    faults = []
    for kind in FAULT_TYPES:
        placed = 0
        while placed < per_type:
            length = int(rng.integers(MIN_LEN, MAX_LEN + 1))
            s = int(rng.integers(WINDOW + GAP, len(values) - length - GAP))
            e = s + length
            if busy[s - GAP:e + GAP].any() or taken[s - GAP:e + GAP].any():
                continue
            c = int(rng.integers(values.shape[1]))
            if kind == "stuck":
                out[s:e, c] = out[s - 1, c]
                k = 0.0
            elif kind == "drift":
                k = float(rng.uniform(4, 8))
                out[s:e, c] += np.linspace(0, k * sigma[c], length, dtype=np.float32) * rng.choice([-1, 1])
            elif kind == "bias":
                k = float(rng.uniform(4, 8))
                out[s:e, c] += np.float32(k * sigma[c] * rng.choice([-1, 1]))
            elif kind == "noise":
                k = float(rng.uniform(2, 4))
                out[s:e, c] += rng.normal(0, k * sigma[c], length).astype(np.float32)
            else:  # dropout
                out[s:e, c] = 0.0
                k = 0.0
            taken[s:e] = True
            faults.append({"type": kind, "channel": c, "start": s, "end": e, "k": k})
            placed += 1
    return out, faults


def repredict(model, y_hat, values_inj, faults, means, stds, device):
    """Forecasts change only where a fault touches the 250-sample input window."""
    from evaluate import predict

    y = y_hat.copy()
    for f in faults:
        a, b = f["start"] - WINDOW, f["end"]  # y_hat indices whose input overlaps the fault
        y[a:b] = predict(model, values_inj[a:b + WINDOW].astype(np.float64), means, stds, device)
    return y


def alarm_events(flags):
    """Runs of any-channel alarm, as (start, end) sample indices, end exclusive."""
    on = flags.max(axis=1) > 0
    edges = np.flatnonzero(np.diff(np.r_[False, on, False].astype(np.int8)))
    return list(zip(edges[::2], edges[1::2]))


class PeerModel:
    """Each channel predicted linearly from the other channels in its group, fitted on nominal training data."""

    def fit(self, values, flags, means, stds):
        self.means, self.stds = means.astype(np.float32), stds.astype(np.float32)
        x = self.scale(values)
        clean = (flags > 0).any(axis=1) == 0
        x = x[clean][:: max(1, clean.sum() // 400_000)]
        n = x.shape[1]
        self.w, self.res_std = [], []
        for c in range(n):
            others = [j for j in range(n) if j != c]
            a = np.c_[x[:, others], np.ones(len(x), np.float32)]
            w, *_ = np.linalg.lstsq(a, x[:, c], rcond=None)
            self.w.append((others, w))
            self.res_std.append((x[:, c] - a @ w).std() + 1e-6)
        d = np.diff(x, axis=0)
        self.diff_std = d.std(axis=0) + 1e-6
        self.lo, self.hi = x.min(axis=0), x.max(axis=0)
        return self

    def scale(self, values):
        return ((values - self.means) / self.stds).astype(np.float32)

    def residuals(self, x):
        r = np.empty_like(x)
        for c, (others, w) in enumerate(self.w):
            r[:, c] = np.abs(x[:, c] - (x[:, others] @ w[:-1] + w[-1])) / self.res_std[c]
        return r


FEATURES = ["n_channels", "top_share", "peer_dominance", "peer_max", "stuck", "noise", "out_of_range"]


def event_features(events, flags, values, peer):
    x = peer.scale(values)
    rows = []
    for s, e in events:
        seg = x[s:e]
        per_ch = flags[s:e].sum(axis=0).astype(float)
        r = peer.residuals(seg).mean(axis=0)
        top = np.sort(r)[::-1]
        if e - s > 2:
            dstd = np.diff(seg, axis=0).std(axis=0) / peer.diff_std
        else:
            dstd = np.ones(seg.shape[1])
        span = peer.hi - peer.lo
        outside = ((seg < peer.lo - 0.1 * span) | (seg > peer.hi + 0.1 * span)).mean(axis=0)
        rows.append([
            float((per_ch > 0).sum()),
            float(per_ch.max() / max(per_ch.sum(), 1)),
            float(np.log((top[0] + 1e-3) / (np.median(top[1:]) + 1e-3))),
            float(np.log(top[0] + 1e-3)),
            float(np.log(dstd.min() + 1e-3)),
            float(np.log(dstd.max() + 1e-3)),
            float(outside.max()),
        ])
    return np.array(rows, dtype=float).reshape(-1, len(FEATURES))


def overlaps(events, intervals):
    """For each event, whether it overlaps any (start, end) interval."""
    if not intervals:
        return np.zeros(len(events), bool)
    starts = np.array([a for a, _ in intervals])
    ends = np.array([b for _, b in intervals])
    return np.array([bool(((starts < e) & (ends > s)).any()) for s, e in events])


def label_intervals(label_flags):
    on = (label_flags > 0).any(axis=1)
    edges = np.flatnonzero(np.diff(np.r_[False, on, False].astype(np.int8)))
    return list(zip(edges[::2], edges[1::2]))


class Classifier:
    def fit(self, feats, y):
        self.scaler = StandardScaler().fit(feats)
        self.model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(self.scaler.transform(feats), y)
        return self

    def proba(self, feats):
        if len(feats) == 0:
            return np.zeros(0)
        return self.model.predict_proba(self.scaler.transform(feats))[:, 1]


def choose_threshold(p, y, max_suppressed_real=0.05):
    """Best sensor-fault F1, keeping at most 5% of real-event alarms suppressed."""
    best = (1.01, -1.0)
    for t in np.unique(np.r_[p, 1.01]):
        pred = p >= t
        if (pred & (y == 0)).sum() > max_suppressed_real * max((y == 0).sum(), 1):
            continue
        tp, fp, fn = (pred & (y == 1)).sum(), (pred & (y == 0)).sum(), (~pred & (y == 1)).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best[1]:
            best = (float(t), float(f1))
    return best


def system_flags(flags, events, sensor):
    """Remove alarms attributed to a sensor fault from the system-alarm stream."""
    out = flags.copy()
    for (s, e), is_sensor in zip(events, sensor):
        if is_sensor:
            out[s:e] = 0
    return out
