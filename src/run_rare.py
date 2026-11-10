"""Run a rare_events.py stage detached, so it survives a session restart. Usage: python src/run_rare.py dev|test"""
import os
import subprocess
import sys

from thresholding import ROOT

stage = sys.argv[1]
with open(ROOT / "results" / f"rare_{stage}.out", "a", encoding="utf8") as out:
    subprocess.call([sys.executable, "-W", "ignore", str(ROOT / "src" / "rare_events.py"), stage], cwd=ROOT / "src",
                    stdout=out, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUTF8": "1"})
