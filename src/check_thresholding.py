"""Check that thresholding.py reproduces flags saved by an earlier run.

Usage: python src/check_thresholding.py 42 Telemanom-ESA-Pruned 0.007
"""
import sys
import time

import numpy as np

from thresholding import ROOT, flags_from


def main():
    seed, name, min_error_value = sys.argv[1], sys.argv[2], float(sys.argv[3])
    run = ROOT / "models" / f"telemanom_21_months_seed{seed}"
    y_hat = np.load(run / "y_hat_last.npy")
    with np.load(ROOT / "data" / "processed" / "test.npz") as d:
        values = d["values"]
    t0 = time.time()
    new = flags_from(y_hat, values, min_error_value)
    old = np.load(run / f"flags_last_{name}.npy")
    same = np.array_equal(new, old)
    print(f"{'PASS' if same else 'FAIL'} seed {seed} {name}: {new.sum()} flagged now, {old.sum()} before, "
          f"{time.time() - t0:.0f}s")
    raise SystemExit(0 if same else 1)


if __name__ == "__main__":
    main()
