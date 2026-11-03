"""Run the original Telemanom-ESA errors.py thresholding on saved predictions.

Kept free of torch so the worker processes stay small. Each worker gets a single
channel: errors.py treats channels independently (per-column smoothing, ranges and
windows), so this gives the same flags as passing all channels at once.
"""
import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ERRORS_PY = ROOT / "baseline" / "TimeEval-algorithms" / "telemanom_esa" / "telemanom" / "errors.py"
WINDOW = 250  # must match train.py
HORIZON = 10
BATCH = 70  # errors.py uses this for its smoothing span and window steps, as the original does


def load_errors_module():
    spec = importlib.util.spec_from_file_location("telemanom_errors", ERRORS_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["telemanom_errors"] = module
    spec.loader.exec_module(module)
    return module


def config(min_error_value):
    return SimpleNamespace(
        smoothing_window_size=30, batch_size=BATCH, smoothing_perc=0.05, window_size=WINDOW,
        error_buffer=100, p=0.13, min_error_value=min_error_value,
        target_channel_indices=[0], prediction_window_size=HORIZON)


def make_channel(values):
    """The two attributes errors.py reads from the original Channel object."""
    data = values[np.newaxis]
    n_batches = int(np.ceil((len(values) - WINDOW) / BATCH))

    class Generator:
        def __init__(self):
            self.data = data

        def __len__(self):
            return n_batches

    return SimpleNamespace(id="channel", generator_test=Generator())


def threshold_channel(job):
    channel_idx, y_hat, values, min_error_value = job
    errors_module = load_errors_module()
    channel = make_channel(values)
    errors = errors_module.Errors(channel, SimpleNamespace(y_hat=y_hat), config(min_error_value), "run", save=False)
    errors.process_batches(channel, 0)
    return channel_idx, errors.E_seq


def flags_from(y_hat, values, min_error_value, workers=6):
    flags = np.zeros((len(values), values.shape[1]), dtype=np.uint8)
    jobs = [(c, y_hat[:, [c]], values[:, [c]], min_error_value) for c in range(values.shape[1])]
    with ProcessPoolExecutor(workers) as pool:
        for channel_idx, sequences in pool.map(threshold_channel, jobs):
            for start, end in sequences:
                flags[start + WINDOW:end + WINDOW, channel_idx] = 1  # same slice as detector.py:215, then padded
    return flags
