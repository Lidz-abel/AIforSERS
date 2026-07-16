"""Register the completed Phase4D run as immutable Phase5 experiment P5-00."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_utils import initialize_run, load_config, resolve, update_manifest, update_registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Phase5/configs/exp_000_baseline.yaml")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    result_dir, _ = initialize_run(config, config_path)
    source = json.loads(resolve(config["experiment"]["source_result"]).read_text(encoding="utf-8"))
    summary = pd.DataFrame([
        {"set": "OOF development", "n": 41, **source["oof"]["metrics"]},
        {"set": "legacy observed holdout", "n": len(source["test"]["labels"]), **source["test"]["metrics"]},
    ])
    summary.to_csv(result_dir / "summary_matrix.csv", index=False)
    patient = pd.DataFrame({
        "patient_id": source["test"]["patient_ids"],
        "true_label": source["test"]["labels"],
        "probability": source["test"]["calibrated_probability"],
        "predicted_label": source["test"]["predicted_label"],
        "sampling_uncertainty": source["test"]["sampling_uncertainty"],
        "aleatoric_uncertainty": source["test"]["aleatoric_uncertainty"],
        "epistemic_uncertainty": source["test"]["epistemic_uncertainty"],
    })
    patient.to_csv(result_dir / "patient_predictions.csv", index=False)
    metrics = {
        "oof_auc": source["oof"]["metrics"]["roc_auc"],
        "oof_accuracy": source["oof"]["metrics"]["accuracy"],
        "test_auc": source["test"]["metrics"]["roc_auc"],
        "test_accuracy": source["test"]["metrics"]["accuracy"],
    }
    manifest = update_manifest(
        result_dir,
        status="complete",
        source_result=str(resolve(config["experiment"]["source_result"])),
        reproduced_metrics=metrics,
    )
    update_registry(config, manifest, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
