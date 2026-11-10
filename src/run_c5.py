"""Run the C5 test stage detached, so it survives a session restart. Usage: python src/run_c5.py"""
import os
import subprocess
import sys

from thresholding import ROOT

with open(ROOT / "results" / "c5_test.out", "a", encoding="utf8") as out:
    subprocess.call([sys.executable, "-W", "ignore", str(ROOT / "src" / "c5_ablation.py"), "test"], cwd=ROOT / "src",
                    stdout=out, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUTF8": "1"})
