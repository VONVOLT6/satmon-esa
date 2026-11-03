"""P0 gate: the official metric must behave as hand-calculated on small cases.

Usage: python src/test_metrics.py
"""
import numpy as np
import pandas as pd

from metrics import ESAScores


def run(events, pred, category=("Anomaly",)):
    truth = pd.DataFrame(events, columns=["ID", "StartTime", "EndTime", "Category"])
    truth["StartTime"] = pd.to_datetime(truth["StartTime"])
    truth["EndTime"] = pd.to_datetime(truth["EndTime"])
    frame = pd.DataFrame(pred, columns=["Timestamp", "Score"])
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], format="ISO8601")
    return ESAScores(betas=0.5, select_labels={"Category": list(category)},
                     full_range=(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-11"))).score(truth, frame)


EVENTS = [["a", "2020-01-02", "2020-01-03", "Anomaly"], ["b", "2020-01-06", "2020-01-07", "Anomaly"]]


def check(name, value, expected):
    ok = abs(value - expected) < 1e-9
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {value:.6f}, expected {expected:.6f}")
    return ok


def main():
    results = []

    # Finds both events exactly, no false alarms.
    r = run(EVENTS, [["2020-01-01", 0], ["2020-01-02", 1], ["2020-01-03", 0], ["2020-01-06", 1], ["2020-01-07", 0], ["2020-01-11", 0]])
    results += [check("perfect recall", r["EW_recall"], 1.0), check("perfect precision", r["EW_precision"], 1.0),
                check("perfect F0.5", r["EW_F_0.50"], 1.0)]

    # Finds one of two events.
    r = run(EVENTS, [["2020-01-01", 0], ["2020-01-02", 1], ["2020-01-03", 0], ["2020-01-11", 0]])
    results += [check("half recall", r["EW_recall"], 0.5), check("half precision stays 1", r["EW_precision"], 1.0),
                check("half F0.5", r["EW_F_0.50"], 1.25 * 1 * 0.5 / (0.25 + 0.5))]

    # Finds both, plus one false alarm lasting 1 of the 8 nominal days.
    # Precision = 2/3 before correction, times true-negative rate 7/8.
    r = run(EVENTS, [["2020-01-01", 0], ["2020-01-02", 1], ["2020-01-03", 0], ["2020-01-04", 1], ["2020-01-05", 0],
                     ["2020-01-06", 1], ["2020-01-07", 0], ["2020-01-11", 0]])
    p = 2 / 3 * 7 / 8
    results += [check("corrected precision", r["EW_precision"], p),
                check("corrected F0.5", r["EW_F_0.50"], 1.25 * p / (0.25 * p + 1))]

    # Alarm on a rare event while scoring anomalies only must not count as a false alarm.
    events = EVENTS + [["c", "2020-01-08", "2020-01-09", "Rare Event"]]
    r = run(events, [["2020-01-01", 0], ["2020-01-02", 1], ["2020-01-03", 0], ["2020-01-06", 1], ["2020-01-07", 0],
                     ["2020-01-08", 1], ["2020-01-09", 0], ["2020-01-11", 0]])
    results.append(check("rare event is not a false alarm", r["EW_precision"], 1.0))

    # Two detections of one event: alarming precision halves.
    r = run(EVENTS[:1], [["2020-01-01", 0], ["2020-01-02", 1], ["2020-01-02 06:00", 0], ["2020-01-02 12:00", 1],
                         ["2020-01-03", 0], ["2020-01-11", 0]])
    results.append(check("redundant detection lowers alarming precision", r["alarming_precision"], 0.5))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
