"""Train and score several seeds in a row, skipping any seed that already has results.

Runs as its own process so it keeps going if the terminal or chat session closes.
Progress goes to results/run_seeds.log.

Usage: python src/run_seeds.py 21_months 3 4 2
"""
import os
import subprocess
import sys
import time

from train import ROOT

PY = sys.executable
SRC = ROOT / "src"
RESULTS = ROOT / "results"


def log(message):
    with open(RESULTS / "run_seeds.log", "a", encoding="utf8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def step(script, dataset, seed, logfile, *extra):
    with open(RESULTS / logfile, "a", encoding="utf8") as out:
        return subprocess.call([PY, "-W", "ignore", str(SRC / script), dataset, "--seed", str(seed), *extra],
                               cwd=SRC, stdout=out, stderr=subprocess.STDOUT,
                               env={**os.environ, "PYTHONUTF8": "1"})


def main():
    dataset, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
    log(f"start {dataset} seeds {seeds}")
    for seed in seeds:
        if (RESULTS / f"telemanom_{dataset}_seed{seed}_last.json").exists():
            log(f"seed {seed} already scored, skipping")
            continue
        code = step("train.py", dataset, seed, f"train_{dataset}_seed{seed}.log")
        log(f"seed {seed} train exit {code}")
        if code != 0:
            continue
        code = step("evaluate.py", dataset, seed, f"eval_{dataset}_seed{seed}_last.log", "--weights", "last")
        log(f"seed {seed} evaluate exit {code}")
    log("finished")


if __name__ == "__main__":
    main()
