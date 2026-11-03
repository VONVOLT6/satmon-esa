"""Quick checks on made-up data before the real data arrives.

Usage: python src/smoke_test.py
"""
import time

import numpy as np
import torch
from torch import nn

from evaluate import predict
from thresholding import WINDOW, flags_from
from train import BATCH, Telemanom, Windows, channel_scaling, nominal_fragments


def main():
    # 1. Fragments split on labels and on the validation date.
    index = np.datetime64("2000-01-01") + np.arange(1000) * np.timedelta64(30, "s")
    values = np.random.default_rng(0).normal(size=(1000, 6)).astype(np.float32)
    flags = np.zeros((1000, 6), np.uint8)
    flags[400:420, 2] = 1
    train, val = nominal_fragments(index, values, flags, str(index[700]))
    sizes = [len(f) for f in train], [len(f) for f in val]
    assert sizes == ([400, 281], [300]), sizes
    print("PASS fragments", sizes)

    # 2. Training on the GPU lowers the loss on a learnable signal, and speed is measured.
    t = np.arange(40000)
    signal = np.stack([np.sin(t / (20 + 7 * c)) for c in range(6)], axis=1).astype(np.float32)
    means, stds = channel_scaling([signal], signal)
    device = torch.device("cuda")
    w = Windows([signal], means, stds, device)
    model = Telemanom(6, 6).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-7)
    losses, t0 = [], time.time()
    rng = np.random.default_rng(0)
    for _ in range(300):
        x, y = w.batch(torch.as_tensor(rng.integers(0, len(w), BATCH), device=device))
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    rate = 300 / (time.time() - t0)
    assert np.mean(losses[-20:]) < np.mean(losses[:20]) * 0.5, (losses[0], losses[-1])
    print(f"PASS training, loss {np.mean(losses[:20]):.3f} to {np.mean(losses[-20:]):.3f}, {rate:.0f} steps/s, "
          f"so about {1000 / rate:.0f}s for the 1000 training steps in each epoch")

    # 3. Prediction output has the shape errors.py expects.
    y_hat = predict(model, signal.astype(np.float64), means, stds, device)
    assert y_hat.shape == (len(signal) - WINDOW, 6), y_hat.shape
    print("PASS prediction shape", y_hat.shape)

    # 4. The original errors.py flags a planted fault on the right channel, near the right time.
    n = 12000
    t = np.arange(n)
    clean = np.stack([np.sin(t / (50 + 9 * c)) for c in range(6)], axis=1).astype(np.float32)
    noise = np.random.default_rng(1).normal(scale=0.003, size=(n - WINDOW, 6))
    y_hat = clean[WINDOW:] + noise
    faulty = clean.copy()
    faulty[8000:8060, 3] += 1.5
    t0 = time.time()
    flags = flags_from(y_hat, faulty, 0.007)
    print(f"  thresholding took {time.time() - t0:.1f}s for {n} samples, 6 channels")
    hit = np.flatnonzero(flags[:, 3])
    others = np.count_nonzero(np.delete(flags, 3, axis=1))
    assert len(hit) and hit.min() <= 8060 and hit.max() >= 8000, hit[:5]
    print(f"PASS planted fault found: channel 3 flagged {hit.min()} to {hit.max()}, other channels flagged {others} samples")


if __name__ == "__main__":
    main()
