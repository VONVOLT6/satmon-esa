"""Train the Telemanom-ESA forecaster on Mission 1 channels 41-46.

PyTorch port of TimeEval-algorithms/telemanom_esa (TF 2.5.1), because that TF
version has no support for this laptop's RTX 5060. Settings match
mission1_experiments.py line 75 and the Keras defaults it relies on:
weight init, Adam epsilon, dropout placement, 1000 steps per epoch, patience 20.

The original keeps the last epoch's weights: ModelCheckpoint saves the best one,
then model.save() overwrites it. Both are saved here and `last.pt` is the faithful one.

Usage: python src/train.py 21_months --seed 42
Re-running the same command resumes from the last finished epoch.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_SPLITS = {"3_months": "2000-03-11", "10_months": "2000-09-01", "21_months": "2001-07-01",
                     "42_months": "2003-04-01", "84_months": "2006-10-01"}
WINDOW = 250
HORIZON = 10
BATCH = 70
MAX_STEPS = 1000
EPOCHS = 1000
PATIENCE = 20
MIN_DELTA = 0.0003
DROPOUT = 0.3
UNITS = 80


def init_lstm(lstm):
    """Keras LSTM defaults: glorot uniform, orthogonal recurrent, forget gate bias 1."""
    for name, param in lstm.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(param)
        elif "weight_hh" in name:
            nn.init.orthogonal_(param)
        elif "bias" in name:
            nn.init.zeros_(param)
            param.data[lstm.hidden_size:2 * lstm.hidden_size] = 1.0


class Telemanom(nn.Module):
    def __init__(self, n_in, n_out, units=UNITS, horizon=HORIZON, dropout=DROPOUT):
        super().__init__()
        self.n_out, self.horizon = n_out, horizon
        self.lstm1 = nn.LSTM(n_in, units, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(units, units, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.head = nn.Linear(units, n_out * horizon)
        init_lstm(self.lstm1)
        init_lstm(self.lstm2)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h, _ = self.lstm1(x)
        h, _ = self.lstm2(self.drop1(h))
        return self.head(self.drop2(h[:, -1])).view(-1, self.horizon, self.n_out)


def nominal_fragments(index, values, flags, split):
    """Runs of samples with no labelled channel, split at the validation date."""
    clean = (flags > 0).any(axis=1) == 0
    edges = np.flatnonzero(np.diff(np.r_[False, clean, False].astype(np.int8)))
    train, val = [], []
    cut = np.datetime64(split)
    for a, b in zip(edges[::2], edges[1::2]):  # b is exclusive
        start, end = index[a], index[b - 1]
        if end < cut:
            train.append(values[a:b])
        elif start > cut:
            val.append(values[a:b])
        else:
            k = int(np.searchsorted(index[a:b], cut))
            train.append(values[a:a + k + 1])  # the original's date slices share the split sample
            val.append(values[a + k:b])
    return train, val


def channel_scaling(train, values):
    joined = np.concatenate(train)
    means, stds = joined.mean(axis=0).astype(np.float64), joined.std(axis=0).astype(np.float64)
    binary = np.array([np.all(values[:, c] == values[:, c].astype(np.int64)) and len(np.unique(values[:, c])) == 2
                       for c in range(values.shape[1])])
    means = np.where(binary, values.min(axis=0), means)
    stds = np.where(binary, values.max(axis=0) - values.min(axis=0), stds)
    stds = np.where(stds == 0, 1, stds)
    return means, stds


class Windows:
    """Keeps the data on the GPU and cuts batches with indexing, which avoids a slow Python loop."""

    def __init__(self, fragments, means, stds, device):
        self.device = device
        scaled = [(f - means) / stds for f in fragments if len(f) > WINDOW + HORIZON]
        self.data = torch.as_tensor(np.concatenate(scaled), dtype=torch.float32, device=device)
        starts, offset = [], 0
        for f in scaled:
            starts.append(np.arange(offset, offset + len(f) - WINDOW - HORIZON))
            offset += len(f)
        self.starts = torch.as_tensor(np.concatenate(starts), device=device)
        self.input_steps = torch.arange(WINDOW, device=device)
        self.output_steps = torch.arange(WINDOW, WINDOW + HORIZON, device=device)

    def __len__(self):
        return len(self.starts)

    def batch(self, ids):
        s = self.starts[ids].unsqueeze(1)
        return self.data[s + self.input_steps], self.data[s + self.output_steps]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", default="21_months", nargs="?")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    run = ROOT / "models" / f"telemanom_{args.dataset}_seed{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    with np.load(ROOT / "data" / "processed" / f"{args.dataset}.train.npz") as d:
        index, values, flags = d["index"], d["values"], d["flags"]
    train, val = nominal_fragments(index, values, flags, VALIDATION_SPLITS[args.dataset])
    means, stds = channel_scaling(train, values)
    np.savez(run / "scaling.npz", means=means, stds=stds)

    train_w, val_w = Windows(train, means, stds, device), Windows(val, means, stds, device)
    print(f"device {device}, {len(train_w)} train windows, {len(val_w)} validation windows", flush=True)

    model = Telemanom(values.shape[1], values.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-7)
    loss_fn = nn.MSELoss()

    steps = min(MAX_STEPS, int(np.ceil(len(train_w) / BATCH)))
    order, cursor = rng.permutation(len(train_w)), 0
    history, best, stop_reference, stale, t0 = [], np.inf, np.inf, 0, time.time()
    state_path = run / "state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state.get("finished"):
            print("already finished, delete the run folder to retrain", flush=True)
            return
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        order, cursor, history = state["order"], state["cursor"], state["history"]
        best, stop_reference, stale = state["best"], state["stop_reference"], state["stale"]
        t0 -= history[-1]["elapsed_s"]
        print(f"resumed after epoch {len(history)}", flush=True)
    for epoch in range(len(history) + 1, args.max_epochs + 1):
        model.train()
        running = 0.0
        for _ in range(steps):
            if cursor + BATCH > len(order):
                order, cursor = rng.permutation(len(train_w)), 0
            ids = torch.as_tensor(order[cursor:cursor + BATCH], device=device)
            cursor += BATCH
            x, y = train_w.batch(ids)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item()

        model.eval()
        total, seen = 0.0, 0
        with torch.inference_mode():
            for start in range(0, len(val_w), 2048):
                ids = torch.arange(start, min(start + 2048, len(val_w)), device=device)
                x, y = val_w.batch(ids)
                total += loss_fn(model(x), y).item() * len(ids)
                seen += len(ids)
        val_loss = total / seen
        history.append({"epoch": epoch, "train_loss": running / steps, "val_loss": val_loss,
                        "elapsed_s": time.time() - t0})
        print(f"epoch {epoch} train {running / steps:.5f} val {val_loss:.5f} "
              f"[{time.time() - t0:.0f}s]", flush=True)

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), run / "best.pt")
        # Keras EarlyStopping (mode min) counts an improvement only when val_loss < reference - min_delta
        if val_loss + MIN_DELTA < stop_reference:
            stop_reference, stale = val_loss, 0
        else:
            stale += 1
        stopping = stale >= PATIENCE
        torch.save(model.state_dict(), run / "last.pt")
        (run / "history.json").write_text(json.dumps(history, indent=1))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "numpy_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                    "order": order, "cursor": cursor, "history": history, "best": best,
                    "stop_reference": stop_reference, "stale": stale, "finished": stopping},
                   state_path.with_suffix(".tmp"))
        state_path.with_suffix(".tmp").replace(state_path)  # a shutdown mid-write leaves the old state intact
        if stopping:
            print(f"early stop at epoch {epoch}", flush=True)
            break

    (run / "train_summary.json").write_text(json.dumps({
        "dataset": args.dataset, "seed": args.seed, "epochs": len(history), "best_val": best,
        "last_val": history[-1]["val_loss"], "train_seconds": time.time() - t0,
        "device": str(device), "torch": torch.__version__}, indent=1))
    print(f"done in {time.time() - t0:.0f}s, best val {best:.5f}", flush=True)


if __name__ == "__main__":
    main()
