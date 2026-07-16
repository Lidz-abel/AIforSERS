"""Formal fixed-weight heterogeneous ensemble experiment P5-04."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TOOLBOX = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLBOX / "Phase3" / "baseline"))
sys.path.insert(0, str(TOOLBOX / "Phase4" / "mcss_hetero"))
sys.path.insert(0, str(TOOLBOX / "Phase4" / "stability"))
sys.path.insert(0, str(TOOLBOX / "Phase5"))

from experiment_utils import initialize_run, load_config, resolve, sha256_file, update_manifest, update_registry
from train_phase4d import calibrate, fit_temperature, jsonable, metrics
from phase4b_utils import optimize_threshold


def validate_component(component: dict) -> Path:
    directory = resolve(component["result_dir"])
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise RuntimeError(f"Component is incomplete: {component['experiment_id']}")
    if manifest["config_sha256"] != component["expected_config_sha256"]:
        raise RuntimeError(f"Component config hash changed: {component['experiment_id']}")
    return directory


def load_oof_members(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = {}
    for fold in range(5):
        for member in range(5):
            result = json.loads((directory / f"oof_fold_{fold}_member_{member}.json").read_text(encoding="utf-8"))
            prediction = result["val_predictions"]
            for pid, label, probability in zip(
                prediction["patient_ids"], prediction["labels"], prediction["probabilities"]
            ):
                entry = rows.setdefault(int(pid), {"label": int(label), "probabilities": []})
                if entry["label"] != int(label):
                    raise RuntimeError(f"Inconsistent OOF label for patient {pid}")
                entry["probabilities"].append(float(probability))
    ids = np.asarray(sorted(rows), dtype=int)
    matrix = np.asarray([rows[int(pid)]["probabilities"] for pid in ids], dtype=float)
    if matrix.shape != (41, 5):
        raise RuntimeError(f"Expected a 41x5 OOF matrix, got {matrix.shape}")
    return ids, np.asarray([rows[int(pid)]["label"] for pid in ids], dtype=int), matrix


def load_test_members(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    member_probabilities = []
    reference_ids = reference_labels = None
    for member in range(5):
        result = json.loads((directory / f"final_member_{member}.json").read_text(encoding="utf-8"))
        prediction = result["test_predictions"]
        ids = np.asarray(prediction["patient_ids"], dtype=int)
        labels = np.asarray(prediction["labels"], dtype=int)
        if reference_ids is None:
            reference_ids, reference_labels = ids, labels
        elif not np.array_equal(ids, reference_ids) or not np.array_equal(labels, reference_labels):
            raise RuntimeError("Final member patient order differs")
        member_probabilities.append(np.asarray(prediction["probabilities"], dtype=float))
    patient_frame = pd.read_csv(directory / "patient_predictions.csv").set_index("patient_id")
    return reference_ids, reference_labels, np.stack(member_probabilities, axis=1), patient_frame


def align(ids: np.ndarray, target_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    positions = {int(pid): index for index, pid in enumerate(ids)}
    return np.asarray([values[positions[int(pid)]] for pid in target_ids])


def stratified_bootstrap(labels, probabilities, threshold, repeats, seed):
    rng = np.random.RandomState(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    auc, accuracy = [], []
    for _ in range(repeats):
        indices = np.concatenate([
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ])
        auc.append(roc_auc_score(labels[indices], probabilities[indices]))
        accuracy.append(np.mean((probabilities[indices] >= threshold) == labels[indices]))
    return {
        "auc_95": np.percentile(auc, [2.5, 97.5]).tolist(),
        "accuracy_95": np.percentile(accuracy, [2.5, 97.5]).tolist(),
    }


def risk_coverage(labels, probabilities, threshold, uncertainty):
    correct = (probabilities >= threshold) == labels
    rows = []
    for ranking, score, descending in [
        ("decision_margin", np.abs(probabilities - threshold), True),
        ("total_epistemic", uncertainty, False),
    ]:
        order = np.argsort(-score if descending else score)
        for coverage in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
            count = max(1, int(np.floor(len(labels) * coverage)))
            rows.append({
                "ranking": ranking,
                "requested_coverage": coverage,
                "n": count,
                "actual_coverage": count / len(labels),
                "selective_accuracy": float(correct[order[:count]].mean()),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Phase5/configs/exp_004_heterogeneous.yaml")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    parent, _ = load_config("Phase5/configs/exp_001_ensemble.yaml")
    output_dir, _ = initialize_run(config, config_path, parent_config=parent)
    output_dir = resolve(config["paths"]["results_dir"])
    component_dirs = [validate_component(component) for component in config["components"]]
    weights = np.asarray([component["weight"] for component in config["components"]], dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        raise RuntimeError("Component weights must sum to one")

    oof_components = [load_oof_members(directory) for directory in component_dirs]
    oof_ids, oof_labels = oof_components[0][0], oof_components[0][1]
    oof_matrices = []
    for ids, labels, matrix in oof_components:
        matrix = align(ids, oof_ids, matrix)
        aligned_labels = align(ids, oof_ids, labels)
        if not np.array_equal(aligned_labels, oof_labels):
            raise RuntimeError("OOF labels differ between architectures")
        oof_matrices.append(matrix)
    architecture_means = np.stack([matrix.mean(axis=1) for matrix in oof_matrices], axis=1)
    fused_oof_raw = architecture_means @ weights
    temperature = fit_temperature(fused_oof_raw, oof_labels, {
        "calibration": {"temperature_min": 0.2, "temperature_max": 5.0}
    })
    fused_oof = calibrate(fused_oof_raw, temperature)
    threshold_result = optimize_threshold(oof_labels, fused_oof, strategy="max_accuracy")
    threshold = float(threshold_result["threshold"])
    oof_metrics = metrics(oof_labels, fused_oof, threshold)
    all_oof_members = np.concatenate(oof_matrices, axis=1)
    oof_total_epistemic = all_oof_members.var(axis=1)
    oof_architecture_disagreement = np.average(
        (architecture_means - fused_oof_raw[:, None]) ** 2, axis=1, weights=weights
    )
    oof_errors = ((fused_oof >= threshold).astype(int) != oof_labels).astype(int)
    oof_uncertainty_auc = float(roc_auc_score(oof_errors, oof_total_epistemic))

    test_components = [load_test_members(directory) for directory in component_dirs]
    test_ids, test_labels = test_components[0][0], test_components[0][1]
    test_matrices, source_frames = [], []
    for ids, labels, matrix, frame in test_components:
        matrix = align(ids, test_ids, matrix)
        aligned_labels = align(ids, test_ids, labels)
        if not np.array_equal(aligned_labels, test_labels):
            raise RuntimeError("Test labels differ between architectures")
        test_matrices.append(matrix)
        source_frames.append(frame.loc[test_ids])
    test_architecture_means = np.stack([matrix.mean(axis=1) for matrix in test_matrices], axis=1)
    fused_test_raw = test_architecture_means @ weights
    fused_test = calibrate(fused_test_raw, temperature)
    test_prediction = (fused_test >= threshold).astype(int)
    test_metrics = metrics(test_labels, fused_test, threshold)
    all_test_members = np.concatenate(test_matrices, axis=1)
    test_total_epistemic = all_test_members.var(axis=1)
    test_architecture_disagreement = np.average(
        (test_architecture_means - fused_test_raw[:, None]) ** 2, axis=1, weights=weights
    )
    test_errors = (test_prediction != test_labels).astype(int)
    test_uncertainty_auc = float(roc_auc_score(test_errors, test_total_epistemic))
    sampling = np.average(
        np.stack([frame["sampling_uncertainty"].to_numpy() for frame in source_frames], axis=1),
        axis=1,
        weights=weights,
    )
    aleatoric = np.average(
        np.stack([frame["aleatoric_uncertainty"].to_numpy() for frame in source_frames], axis=1),
        axis=1,
        weights=weights,
    )

    dataset = np.load(resolve(config["paths"]["dataset_dir"]) / "spectra.npz", allow_pickle=True)
    patient_uids = [str(value) for value in dataset["patient_uids"]]
    patient_frame = pd.DataFrame({
        "patient_id": test_ids,
        "patient_uid": [patient_uids[int(pid)] for pid in test_ids],
        "true_label": test_labels,
        "probability": fused_test,
        "predicted_label": test_prediction,
        "correct": 1 - test_errors,
        "sampling_uncertainty": sampling,
        "aleatoric_uncertainty": aleatoric,
        "within_all_members_epistemic": test_total_epistemic,
        "between_architecture_disagreement": test_architecture_disagreement,
    })
    patient_frame.to_csv(output_dir / "patient_predictions.csv", index=False)
    pd.DataFrame(risk_coverage(oof_labels, fused_oof, threshold, oof_total_epistemic)).to_csv(
        output_dir / "risk_coverage.csv", index=False
    )
    summary = pd.DataFrame([
        {"set": "OOF development", "n": len(oof_labels), **oof_metrics},
        {"set": "legacy observed holdout", "n": len(test_labels), **test_metrics},
    ])
    summary.to_csv(output_dir / "summary_matrix.csv", index=False)
    bootstrap_repeats = int(config["fusion"]["bootstrap_repeats"])
    payload = {
        "experiment_id": config["experiment"]["id"],
        "test_status": config["experiment"]["test_status"],
        "components": [
            {
                **component,
                "final_result_sha256": sha256_file(directory / "final_results.json"),
            }
            for component, directory in zip(config["components"], component_dirs)
        ],
        "fusion": {"temperature": temperature, "threshold_result": threshold_result, "weights": weights},
        "oof": {
            "patient_ids": oof_ids,
            "labels": oof_labels,
            "probability": fused_oof,
            "predicted_label": (fused_oof >= threshold).astype(int),
            "metrics": oof_metrics,
            "total_epistemic": oof_total_epistemic,
            "architecture_disagreement": oof_architecture_disagreement,
            "uncertainty_error_detection_auc": oof_uncertainty_auc,
            "bootstrap": stratified_bootstrap(
                oof_labels, fused_oof, threshold, bootstrap_repeats, int(config["fusion"]["bootstrap_seed"])
            ),
        },
        "test": {
            "patient_ids": test_ids,
            "labels": test_labels,
            "probability": fused_test,
            "predicted_label": test_prediction,
            "metrics": test_metrics,
            "total_epistemic": test_total_epistemic,
            "architecture_disagreement": test_architecture_disagreement,
            "uncertainty_error_detection_auc": test_uncertainty_auc,
            "bootstrap": stratified_bootstrap(
                test_labels, fused_test, threshold, bootstrap_repeats, int(config["fusion"]["bootstrap_seed"])
            ),
        },
    }
    (output_dir / "final_results.json").write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    experiment_metrics = {
        "oof_auc": oof_metrics["roc_auc"],
        "oof_accuracy": oof_metrics["accuracy"],
        "test_auc": test_metrics["roc_auc"],
        "test_accuracy": test_metrics["accuracy"],
    }
    manifest = update_manifest(
        output_dir,
        status="complete",
        completed_metrics=experiment_metrics,
        component_config_hashes=[component["expected_config_sha256"] for component in config["components"]],
    )
    update_registry(config, manifest, experiment_metrics)
    print(
        f"P5-04 OOF AUC={oof_metrics['roc_auc']:.4f} Acc={oof_metrics['accuracy']:.4f} "
        f"BA={oof_metrics['balanced_accuracy']:.4f} uncertainty-AUC={oof_uncertainty_auc:.4f}",
        flush=True,
    )
    print(
        f"P5-04 legacy test AUC={test_metrics['roc_auc']:.4f} Acc={test_metrics['accuracy']:.4f} "
        f"BA={test_metrics['balanced_accuracy']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
