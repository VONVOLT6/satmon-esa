"""C1 adaptive computation: a cheap screening tier decides when the LSTM runs.

Tier 1 (always on, cheap):
  confidence  a linear forecaster on 12 lags of all 6 channels; score = smoothed standardised residual
  novelty     Mahalanobis distance of 120-sample rolling means and spreads from the training range
  impact      how many channels deviate at once
Each score is turned into a rank against the nominal training data (0 to 1).

Tier 3 (on demand): Telemanom-ESA-Pruned. A trigger opens a burst of LSTM forecasts covering
LOOKBACK + PRE samples before the trigger (LOOKBACK added after the first development run, see
PROTOCOL.md addendum) and
POST after it, because errors.py judges each point against the
previous 2,100 smoothed errors. Alarms count only after the first PRE samples of a burst, once
that history is complete. Every sample inside a burst counts as LSTM compute, padding included.

The LSTM is stateless given its 250-sample input, so its output on a burst equals the cached
full-run forecast at those positions. Routing can therefore be simulated exactly from y_hat_*.npy.
"""
import numpy as np
import pandas as pd

from thresholding import WINDOW, flags_from

LAGS = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233])
ROLL = 120
SPAN = 30
PRE = 2400   # 2,100-sample threshold window plus warm-up of the error smoothing
POST = 2100      # samples after the trigger
LOOKBACK = 2100  # samples before the trigger that can still raise alarms
IMPACT_Z = 3.0


def _lagged(x, start, stop):
    """Lag features for rows start..stop-1 (rows must be >= max lag)."""
    cols = [x[start - lag:stop - lag] for lag in LAGS]
    return np.concatenate(cols + [np.ones((stop - start, 1), x.dtype)], axis=1)


def _ranks(sorted_ref, values):
    return np.searchsorted(sorted_ref, values, side="right") / len(sorted_ref)


class CheapTier:
    def fit(self, values, flags, means, stds):
        """values/flags: the training arrays; means/stds: the LSTM's scaling, reused here."""
        self.means, self.stds = means.astype(np.float32), stds.astype(np.float32)
        x = self._scale(values)
        clean = (flags > 0).any(axis=1) == 0
        lag_max = int(LAGS.max())
        # rows whose target and oldest input are both nominal (a cheap approximation at fragment edges)
        rows = np.arange(lag_max, len(x))
        rows = rows[clean[rows] & clean[rows - lag_max]]
        rows = rows[:: max(1, len(rows) // 400_000)]  # plenty for a 73-weight linear model
        a = np.concatenate([x[rows - lag] for lag in LAGS] + [np.ones((len(rows), 1), np.float32)], axis=1)
        self.w, *_ = np.linalg.lstsq(a, x[rows], rcond=None)
        self.resid_std = (x[rows] - a @ self.w).std(axis=0) + 1e-6

        ok = clean.copy()
        ok[: lag_max + ROLL] = False
        feats = self._features(x)
        self.nov_mean = feats[ok].mean(axis=0)
        self.nov_icov = np.linalg.pinv(np.cov(feats[ok], rowvar=False))
        conf, nov, _ = self._raw(x, feats)
        self.conf_ref = np.sort(conf[ok])
        self.nov_ref = np.sort(nov[ok])
        return self

    def _scale(self, values):
        return ((values - self.means) / self.stds).astype(np.float32)

    @staticmethod
    def _features(x):
        frame = pd.DataFrame(x)
        return np.concatenate([frame.rolling(ROLL, min_periods=1).mean().to_numpy(np.float32),
                               frame.rolling(ROLL, min_periods=1).std().fillna(0).to_numpy(np.float32)], axis=1)

    def _raw(self, x, feats):
        n, lag_max = len(x), int(LAGS.max())
        z = np.zeros_like(x)
        for a in range(lag_max, n, 500_000):
            b = min(a + 500_000, n)
            z[a:b] = (x[a:b] - _lagged(x, a, b) @ self.w) / self.resid_std
        smooth = pd.DataFrame(np.abs(z)).ewm(span=SPAN).mean().to_numpy(np.float32)
        d = feats - self.nov_mean
        nov = np.einsum("ij,jk,ik->i", d, self.nov_icov, d)
        return smooth.max(axis=1), nov, (smooth > IMPACT_Z).sum(axis=1)

    def scores(self, values):
        x = self._scale(values)
        conf, nov, impact = self._raw(x, self._features(x))
        return {"confidence": _ranks(self.conf_ref, conf), "novelty": _ranks(self.nov_ref, nov), "impact": impact}


def triggers(scores, tau, use=("confidence", "novelty", "impact"), impact_channels=3):
    hit = np.zeros(len(scores["confidence"]), bool)
    if "confidence" in use:
        hit |= scores["confidence"] >= tau
    if "novelty" in use:
        hit |= scores["novelty"] >= tau
    if "impact" in use:
        hit |= scores["impact"] >= impact_channels
    return hit


def bursts(trigger, n_pred, lookback=LOOKBACK, post=POST, warm=PRE):
    """LSTM mask and alarm-valid mask, both in y_hat index space (sample index minus WINDOW).

    A trigger at j runs the LSTM on [j - lookback - warm, j + post). The first `warm` samples only
    rebuild the error history errors.py needs; alarms count from j - lookback onwards.
    """
    j = np.flatnonzero(trigger[WINDOW:WINDOW + n_pred])
    run = np.zeros(n_pred + 1, np.int32)
    np.add.at(run, np.clip(j - lookback - warm, 0, n_pred), 1)
    np.add.at(run, np.clip(j + post, 0, n_pred), -1)
    mask = np.cumsum(run[:-1]) > 0
    valid = mask.copy()
    edges = np.flatnonzero(np.diff(np.r_[False, mask].astype(np.int8)) == 1)
    for s in edges:
        valid[s:s + warm] = False
    return mask, valid


def routed_flags(y_hat, values, mask, valid, floor=0.007):
    """Alarm flags when the LSTM only runs inside bursts. Outside them the forecast error is zero."""
    actual = values[WINDOW:WINDOW + len(y_hat)]
    y = np.where(mask[:, None], y_hat, actual)
    flags = flags_from(y, values, floor)
    keep = np.zeros(len(values), bool)
    keep[WINDOW:WINDOW + len(valid)] = valid
    flags[~keep] = 0
    return flags
