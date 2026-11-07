"""Run a confirmatory test once, in its own process so it survives the chat session.

H1 (default): score the selected floor on the test set, then the bootstrap comparison.
H2 (argument "h2"): route the test set with the chosen C1 setting, compare and time it.
Output goes to results/h1_run.log or results/h2_run.log.

Usage: python src/run_h1.py [h2]
"""
import os
import subprocess
import sys

from thresholding import ROOT

SRC = ROOT / "src"
H2 = sys.argv[1:] == ["h2"]
STEPS = [("test_h2.py", [])] if H2 else [("dev_tune.py", ["42", "1", "2", "3", "4"]), ("test_h1.py", [])]
env = {**os.environ, "PYTHONUTF8": "1"}
with open(ROOT / "results" / ("h2_run.log" if H2 else "h1_run.log"), "a", encoding="utf8") as log:
    for script, args in STEPS:
        log.write(f"== {script}\n")
        log.flush()
        code = subprocess.call([sys.executable, "-W", "ignore", str(SRC / script), *args], cwd=SRC, stdout=log, stderr=subprocess.STDOUT, env=env)
        log.write(f"== {script} exit {code}\n")
        log.flush()
        if code:
            break
