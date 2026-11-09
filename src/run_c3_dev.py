"""Wait until all 76 channels are prepared, then run the C3 development stage. Runs on its own.

Usage: python src/run_c3_dev.py
"""
import os
import subprocess
import sys
import time

from thresholding import ROOT

done = ROOT / "data" / "processed" / "full" / "done.txt"
while not done.exists() or len(done.read_text().split()) < 76:
    time.sleep(60)
with open(ROOT / "results" / "c3_dev.out", "a", encoding="utf8") as out:
    subprocess.call([sys.executable, "-W", "ignore", str(ROOT / "src" / "c3_run.py"), "dev"], cwd=ROOT / "src",
                    stdout=out, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUTF8": "1"})
