"""Initialize the immutable P5-05 explanation experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

TOOLBOX = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLBOX / "Phase5"))

from experiment_utils import initialize_run, load_config, resolve, update_manifest, update_registry


def main():
    config, config_path = load_config("Phase5/configs/exp_005_clinical_explainability.yaml")
    parent, _ = load_config("Phase5/configs/exp_004_heterogeneous.yaml")
    directory, _ = initialize_run(config, config_path, parent_config=parent)
    parent_manifest = json.loads(
        (resolve(config["parent_model"]["result_dir"]) / "run_manifest.json").read_text(encoding="utf-8")
    )
    if parent_manifest["config_sha256"] != config["parent_model"]["expected_config_sha256"]:
        raise RuntimeError("Frozen P5-04 config hash does not match")
    manifest = update_manifest(
        directory,
        status="explaining",
        parent_result=str(resolve(config["parent_model"]["result_dir"]) / "final_results.json"),
        prediction_policy="P5-04 predictions are immutable",
    )
    update_registry(config, manifest)
    print(f"Prepared: {directory}")


if __name__ == "__main__":
    main()
