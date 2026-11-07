"""Run the H1 test once: score the selected floor on the test set, then the bootstrap comparison.

Runs as its own process so it survives the chat session. Output goes to results/h1_run.log.

Usage: python src/run_h1.py [h2]
"""
import os
import subprocess
import sys

from thresholding import ROOT

SRC = ROOT / "src"
env = {**os.environ, "PYTHONUTF8": "1"}
with open(ROOT / "results" / ("h2_run.log" if sys.argv[1:] == ["h2"] else "h1_run.log"), "a", encoding="utf8") as log:
    for script, args in [*([("dev_tune.py", ["42", "1", "2", "3", "4"]), ("test_h1.py", [])] if sys.argv[1:] != ["h2"] else [("test_h2.py", [])])]:
        log.write(f"== {script}\n")
        log.flush()
        code = subprocess.call([sys.executable, "-W", "ignore", str(SRC / script), *args], cwd=SRC, stdout=log, stderr=subprocess.STDOUT, env=env)
        log.write(f"== {script} exit {code}\n")
        log.flush()
        if code:
            break
