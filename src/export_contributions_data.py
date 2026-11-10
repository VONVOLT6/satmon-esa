"""Collect the saved contribution results (H1-H6, C1-C5) into one JSON file for the dashboard.

Every value is read from the result files written by the test and development stages; nothing is typed in.

Usage: python src/export_contributions_data.py
Writes dashboard/src/data/contributions.json
"""
import json

import numpy as np

from thresholding import ROOT

RES = ROOT / "results"
OUT = ROOT / "dashboard" / "src" / "data" / "contributions.json"


def load(name):
    return json.loads((RES / name).read_text())


def boot(b):
    return {"A": b["A"], "B": b["B"], "diff": b["diff"], "ci95": b["ci95"]}


def main():
    h1, dev1 = load("h1_result.json"), load("dev_tune_21_months.json")
    c1dev, h2 = load("c1_dev.json"), load("h2_result.json")
    h3, c2dev = load("h3_result.json"), load("c2_dev.json")
    h4, h5, c5 = load("h4_result.json"), load("h5_result.json"), load("c5_result.json")
    h6, rdev = load("h6_result.json"), load("rare_dev.json")

    seeds = list(h6["per_seed"])
    mean6 = lambda sys_, cat, k: float(np.mean([h6["per_seed"][s][sys_][cat][k] for s in seeds]))

    out = {
        "h1": {
            "floors": [float(f) for f in dev1["floors"]],
            "dev_f05": [h1["mean_dev_F0.5"][str(float(f))] for f in dev1["floors"]],
            "chosen": h1["chosen_floor"], "baseline": h1["baseline_floor"],
            "test": {c: boot(h1["bootstrap"][c]) for c in ["all", "anomaly"]}, "verdict": h1["verdict"],
        },
        "c1": {
            "taus": c1dev["taus"], "chosen": c1dev["chosen_tau"], "baseline_dev": c1dev["baseline_dev_F0.5"],
            "dev": {v: [{"tau": t, "f05": c1dev["summary"][f"{t}|{v}"]["F0.5"],
                         "share": c1dev["summary"][f"{t}|{v}"]["lstm_share"]} for t in c1dev["taus"]]
                    for v in ["full", "conf+impact", "conf"]},
            "test": boot(h2["bootstrap"]["all"]), "test_anomaly": boot(h2["bootstrap"]["anomaly"]),
            "lstm_share": h2["mean_lstm_share"], "timing": h2["timing_seed42"], "verdict": h2["verdict"],
        },
        "c2": {
            "injected": boot(h3["injected"]), "clean": boot(h3["clean"]), "attribution": h3["attribution"],
            "threshold": h3["threshold"], "dev_sensor_f1": c2dev["dev_sensor_F1"],
            "verdicts": {"H3a": h3["H3a_verdict"], "H3b": h3["H3b_verdict"]},
        },
        "c3": {"methods": h4["methods"], "chosen": h4["chosen"], "alerts": h4["alerts"],
               "labelled_alerts": h4["labelled_alerts"], "bootstrap": h4["bootstrap_G2_vs_G1"], "verdict": h4["verdict"]},
        "c4": {
            "central": next(r["cost"] for r in h5["grid"] if (r["M"], r["D"], r["F"], r["S"]) == (50, 5, 1, 1)),
            "ranked_first": h5["ranked_first"], "P2_beats_both": h5["P2_beats_both"], "grid_points": h5["grid_points"],
            "wins_by_D": {str(d): {p: sum(r["best"] == p for r in h5["grid"] if r["D"] == d) for p in ["P0", "P1", "P2"]}
                          for d in [2, 5, 10]},
            "bootstrap": h5["central_bootstrap_P2_minus_P1"], "alarm_classes": h5["test_alarm_classes"], "verdict": h5["verdict"],
        },
        "c5": {"configs": c5["configs"], "table": c5["table"], "bootstrap": c5["bootstrap"], "W": c5["W"]},
        "h6": {
            "chosen": h6["chosen"], "gate_share": h6["gate_share"], "dev_B0": rdev["B0_mean_f05"],
            "dev_chosen": rdev["mean_f05"][f"{h6['chosen']['p']}|{h6['chosen']['W']}|{h6['chosen']['f_r']}"],
            "dev_grid": rdev["mean_f05"],
            "test": {c: boot(h6["bootstrap"][c]) for c in ["all", "anomaly", "rare"]},
            "caught": {sys_: {"rare": mean6(sys_, "rare", "tp"), "anomaly": mean6(sys_, "anomaly", "tp"),
                              "false_alarms": mean6(sys_, "all", "fp")} for sys_ in ["B0", "R"]},
            "gate_alone": {"rare": h6["gate_alone"]["rare"]["tp"], "anomaly": h6["gate_alone"]["anomaly"]["tp"],
                           "false_alarms": h6["gate_alone"]["all"]["fp"], "f05": h6["gate_alone"]["all"]["f05"]},
            "verdict": h6["verdict"],
        },
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
