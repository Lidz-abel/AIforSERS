"""Build compact fold, summary, and patient result matrices for Phase 4D."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "Results" / "Phase4" / "mcss_hetero"


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return float(center - radius), float(center + radius)


def stratified_auc_interval(labels: np.ndarray, probabilities: np.ndarray, repeats: int = 10000):
    rng = np.random.RandomState(42)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    values = []
    for _ in range(repeats):
        indices = np.concatenate([
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ])
        values.append(roc_auc_score(labels[indices], probabilities[indices]))
    return tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))


def main():
    final = json.loads((RESULTS / "phase4d_final_results.json").read_text(encoding="utf-8"))
    threshold = float(final["oof"]["threshold_result"]["threshold"])
    temperature = float(final["oof"]["temperature"])

    fold_rows = []
    for fold in range(5):
        result = json.loads((RESULTS / f"oof_fold_{fold}.json").read_text(encoding="utf-8"))
        prediction = result["val_predictions"]
        labels = np.asarray(prediction["labels"], dtype=int)
        raw_probability = np.asarray(prediction["probabilities"], dtype=float)
        logits = np.log(np.clip(raw_probability, 1e-6, 1 - 1e-6) / np.clip(1 - raw_probability, 1e-6, 1))
        probability = 1.0 / (1.0 + np.exp(-logits / temperature))
        predicted = (probability >= threshold).astype(int)
        fold_rows.append({
            "fold": fold,
            "n_train": len(result["train_patient_ids"]),
            "n_val": len(labels),
            "val_positive": int(labels.sum()),
            "val_negative": int((labels == 0).sum()),
            "best_epoch": int(result["best_epoch"]),
            "best_val_nll": float(result["best_val_nll"]),
            "val_auc": float(roc_auc_score(labels, probability)),
            "val_accuracy": float(accuracy_score(labels, predicted)),
            "val_balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        })
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(RESULTS / "phase4d_fold_matrix.csv", index=False)

    dataset = np.load(ROOT / "Results" / "Phase3_QC" / "dataset" / "spectra.npz", allow_pickle=True)
    patient_uids = [str(value) for value in dataset["patient_uids"]]
    test = final["test"]
    patient_frame = pd.DataFrame({
        "patient_id": test["patient_ids"],
        "patient_uid": [patient_uids[int(pid)] for pid in test["patient_ids"]],
        "true_label": test["labels"],
        "probability": test["calibrated_probability"],
        "predicted_label": test["predicted_label"],
        "sampling_uncertainty": test["sampling_uncertainty"],
        "aleatoric_uncertainty": test["aleatoric_uncertainty"],
        "epistemic_uncertainty": test["epistemic_uncertainty"],
    })
    patient_frame["correct"] = (patient_frame["true_label"] == patient_frame["predicted_label"]).astype(int)
    patient_frame["error"] = 1 - patient_frame["correct"]
    patient_frame["decision_margin"] = np.abs(patient_frame["probability"] - threshold)
    patient_frame.to_csv(RESULTS / "phase4d_patient_matrix.csv", index=False)

    labels = patient_frame["true_label"].to_numpy()
    probabilities = patient_frame["probability"].to_numpy()
    predictions = patient_frame["predicted_label"].to_numpy()
    accuracy_ci = wilson_interval(int((labels == predictions).sum()), len(labels))
    sensitivity_ci = wilson_interval(int(predictions[labels == 1].sum()), int((labels == 1).sum()))
    specificity_ci = wilson_interval(int((1 - predictions[labels == 0]).sum()), int((labels == 0).sum()))
    auc_ci = stratified_auc_interval(labels, probabilities)

    uncertainty_detection = {}
    for column in ["sampling_uncertainty", "aleatoric_uncertainty", "epistemic_uncertainty"]:
        uncertainty_detection[column] = float(roc_auc_score(patient_frame["error"], patient_frame[column]))

    summary_rows = [
        {"set": "OOF development", "n": 41, **final["oof"]["metrics"]},
        {"set": "Locked test", "n": len(labels), **final["test"]["metrics"]},
    ]
    pd.DataFrame(summary_rows).to_csv(RESULTS / "phase4d_summary_matrix.csv", index=False)
    analysis = {
        "completed": True,
        "leakage_overlap": final["leakage_audit"]["overlap"],
        "threshold": threshold,
        "temperature": temperature,
        "test_confidence_intervals_95": {
            "auc_stratified_bootstrap": auc_ci,
            "accuracy_wilson": accuracy_ci,
            "sensitivity_wilson": sensitivity_ci,
            "specificity_wilson": specificity_ci,
        },
        "uncertainty_error_detection_auc": uncertainty_detection,
        "misclassified_patient_ids": patient_frame.loc[patient_frame["error"] == 1, "patient_id"].tolist(),
    }
    (RESULTS / "phase4d_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(fold_frame.to_string(index=False))
    print("\n", pd.DataFrame(summary_rows).to_string(index=False))
    print("\n", json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
