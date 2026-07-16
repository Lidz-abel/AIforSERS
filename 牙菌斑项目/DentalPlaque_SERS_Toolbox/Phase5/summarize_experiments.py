"""Create the unified Phase 5 experiment comparison matrix."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def baseline_row():
    result = json.loads(
        (ROOT / "Results/Phase4/mcss_hetero/phase4d_final_results.json").read_text(encoding="utf-8")
    )
    analysis = json.loads(
        (ROOT / "Results/Phase4/mcss_hetero/phase4d_analysis.json").read_text(encoding="utf-8")
    )
    return {
        "experiment_id": "exp_000_baseline",
        "members": 1,
        "ssl": False,
        "oof_auc": result["oof"]["metrics"]["roc_auc"],
        "oof_accuracy": result["oof"]["metrics"]["accuracy"],
        "oof_balanced_accuracy": result["oof"]["metrics"]["balanced_accuracy"],
        "oof_ece": result["oof"]["metrics"]["ece"],
        "oof_epistemic_error_auc": None,
        "legacy_test_auc": result["test"]["metrics"]["roc_auc"],
        "legacy_test_accuracy": result["test"]["metrics"]["accuracy"],
        "legacy_test_balanced_accuracy": result["test"]["metrics"]["balanced_accuracy"],
        "legacy_test_ece": result["test"]["metrics"]["ece"],
        "legacy_test_epistemic_error_auc": analysis["uncertainty_error_detection_auc"]["epistemic_uncertainty"],
    }


def ensemble_row(experiment_id: str, ssl: bool):
    path = ROOT / "Results/Phase5" / experiment_id / "final_results.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    return {
        "experiment_id": experiment_id,
        "members": 5,
        "ssl": ssl,
        "oof_auc": result["oof"]["metrics"]["roc_auc"],
        "oof_accuracy": result["oof"]["metrics"]["accuracy"],
        "oof_balanced_accuracy": result["oof"]["metrics"]["balanced_accuracy"],
        "oof_ece": result["oof"]["metrics"]["ece"],
        "oof_epistemic_error_auc": result["oof"]["epistemic_error_detection_auc"],
        "legacy_test_auc": result["test"]["metrics"]["roc_auc"],
        "legacy_test_accuracy": result["test"]["metrics"]["accuracy"],
        "legacy_test_balanced_accuracy": result["test"]["metrics"]["balanced_accuracy"],
        "legacy_test_ece": result["test"]["metrics"]["ece"],
        "legacy_test_epistemic_error_auc": result["test"]["uncertainty_error_detection_auc"]["ensemble_epistemic"],
    }


def main():
    rows = [baseline_row(), ensemble_row("exp_001_ensemble", False), ensemble_row("exp_002_ssl_ensemble", True)]
    frame = pd.DataFrame([row for row in rows if row is not None])
    for metric in ["oof_auc", "oof_accuracy", "oof_balanced_accuracy", "oof_ece"]:
        frame[f"delta_{metric}_vs_parent"] = frame[metric].diff()
    output = ROOT / "Results/Phase5/phase5_comparison_matrix.csv"
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
