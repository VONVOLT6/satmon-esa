"""Score predictions with the official ESA-ADB metric code, unmodified.

The metric file is loaded directly from the cloned repo at commit 67194d3. A stub
package skips timeeval/__init__.py, which pulls in the whole Docker framework.
"""
import contextlib
import importlib
import io
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "baseline" / "timeeval" / "metrics"
RAW = ROOT / "data" / "raw" / "ESA-Mission1"
CATEGORIES = {"all": ["Rare Event", "Anomaly"], "anomaly": ["Anomaly"]}


def _load_esa_scores():
    pkg = types.ModuleType("esa_metrics")
    pkg.__path__ = [str(METRICS_DIR)]
    sys.modules["esa_metrics"] = pkg
    return importlib.import_module("esa_metrics.ESA_ADB_metrics").ESAScores


ESAScores = _load_esa_scores()


def load_truth(channels):
    labels = pd.read_csv(RAW / "labels.csv")
    for col in ["StartTime", "EndTime"]:
        labels[col] = pd.to_datetime(labels[col], utc=True).dt.tz_localize(None)
    kinds = pd.read_csv(RAW / "anomaly_types.csv")
    truth = labels.merge(kinds, on="ID")
    return truth[truth["Channel"].isin(channels)]


def test_events(truth, timestamps):
    """Rows fully inside the test period, the same filter as timeeval/core/experiments.py:347-348."""
    start, end = pd.Timestamp(timestamps[0]), pd.Timestamp(timestamps[-1])
    return truth[(truth["StartTime"] >= start) & (truth["EndTime"] <= end)]


def global_scores(timestamps, flags, truth, category="all"):
    """flags: array of shape (time, channels), 1 where a channel is flagged."""
    pred = pd.DataFrame({"Timestamp": pd.to_datetime(timestamps), "Score": flags.max(axis=1).astype(np.uint8)})
    metric = ESAScores(betas=0.5, select_labels={"Category": CATEGORIES[category]})
    with contextlib.redirect_stdout(io.StringIO()):  # the metric prints per-event lists
        return metric.score(truth.drop(columns=["Channel"]), pred)
