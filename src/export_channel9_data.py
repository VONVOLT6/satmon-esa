"""Collect the teammate's Mission 2 channel_9 Random Forest results for the dashboard.

Reads only the files in results/channel9_rf/ (copied unmodified from the teammate's repository,
see PROVENANCE.md) and writes dashboard/src/data/channel9.json. Charts whose data is not in
those files are not invented: team_reported.json holds the values the team supplied from their
screenshots, labelled as such, and anything still unknown is listed under "not_provided".

Usage: python src/export_channel9_data.py
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "channel9_rf"
OUT = ROOT / "dashboard" / "src" / "data" / "channel9.json"

MODEL_KEYS = {"baseline_7_features": "7", "temporal_10_features": "10", "temporal_drift_13_features": "13"}


def number(value):
    return None if value in ("", None) else float(value)


def main():
    three = json.loads((SRC / "three_model_comparison.json").read_text())
    adaptive = json.loads((SRC / "final_adaptive_test_results.json").read_text())

    models = {}
    for key, short in MODEL_KEYS.items():
        r = three["results"][key]
        (tn, fp), (fn, tp) = r["confusion_matrix"]
        models[short] = {"name": key, "features": r["number_of_features"], "feature_names": r["feature_names"],
                         "threshold": r["threshold"], "precision": r["precision"], "recall": r["recall"],
                         "f1": r["f1_score"], "pr_auc": r["pr_auc"], "fpr": r["false_positive_rate"],
                         "tp": tp, "fp": fp, "tn": tn, "fn": fn, "test_observations": r["test_observations"]}

    routing = []
    with open(SRC / "adaptive_routing_validation.csv", encoding="utf8") as f:
        for row in csv.DictReader(f):
            routing.append({"model": row["model"], "margin": number(row["margin"]),
                            "escalation_pct": number(row["escalation_percentage"]),
                            "precision": float(row["precision"]), "recall": float(row["recall"]),
                            "f1": float(row["f1"]), "fpr": float(row["fpr"])})

    def adaptive_block(key):
        b = adaptive[key]
        return {"precision": b["precision"], "recall": b["recall"], "f1": b["f1"], "fpr": b["fpr"],
                "tp": b["tp"], "fp": b["fp"], "tn": b["tn"], "fn": b["fn"]}

    data = {
        "source": {"repo": "https://github.com/siddarthtakale0-ctrl/saellteeproject", "commit": "37cefb4",
                   "folder": "results/channel9"},
        "experiment": {**three["experiment"], "test_observations": adaptive["test_observations"],
                       "test_anomalies": adaptive["test_anomalies"]},
        "models": models,
        "routing_validation": routing,
        "adaptive_test": {"margin": adaptive["escalation_margin_selected_on_validation"],
                          "escalated": adaptive["escalated_observations"],
                          "escalation_pct": adaptive["escalation_percentage"],
                          "baseline_7": adaptive_block("baseline_7_feature"),
                          "always_13": adaptive_block("always_13_feature"),
                          "hybrid": adaptive_block("adaptive_hybrid")},
        "runtime_seconds": adaptive["runtime_seconds"],
        "team_reported": json.loads((SRC / "team_reported.json").read_text()),
    }
    OUT.write_text(json.dumps(data, indent=1))
    print(f"wrote {OUT}")
    for k, m in models.items():
        print(k, {x: round(m[x], 4) for x in ("precision", "recall", "f1", "pr_auc", "fpr")})


if __name__ == "__main__":
    main()
