"""Five-member leakage-safe OOF deep ensemble for Phase 5 P5-01."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

TOOLBOX = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLBOX / "Phase4" / "mcss_hetero"))
sys.path.insert(0, str(TOOLBOX / "Phase5"))

from experiment_utils import initialize_run, load_config, resolve, update_manifest, update_registry
from patient_mcss_dataset import load_dataset, patient_labels
from train_phase4d import (
    calibrate,
    deterministic_predictions,
    fit_temperature,
    jsonable,
    make_dataset,
    make_loader,
    metrics,
    oof_split,
    optimize_threshold,
    outer_development_test,
    stochastic_patient_predictions,
    train_model,
)


def result_dir(config: dict) -> Path:
    return resolve(config["paths"]["results_dir"])


def prepare(config: dict, config_path: Path) -> Path:
    parent_id = config["experiment"]["parent"]
    parent, _ = load_config(f"Phase5/configs/{parent_id}.yaml")
    directory, manifest = initialize_run(config, config_path, parent_config=parent)
    manifest = update_manifest(directory, status="training", test_policy="legacy holdout; not confirmatory")
    update_registry(config, manifest)
    return directory


def member_seed(config: dict, fold: int, member: int) -> int:
    return int(config["seed"]) + 1000 + member * int(config["ensemble"]["seed_stride"]) + fold


def run_oof_member(
    config: dict,
    fold: int,
    member: int,
    max_epochs: int | None = None,
    initial_encoder_state: dict | None = None,
    pretraining_checkpoint: str | None = None,
) -> Path:
    directory = result_dir(config)
    output = directory / f"oof_fold_{fold}_member_{member}.json"
    if output.exists() and max_epochs is None:
        print(f"skip completed fold={fold} member={member}", flush=True)
        return output
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_dataset(resolve(config["paths"]["dataset_dir"]))
    labels = patient_labels(data)
    development, locked_test = outer_development_test(labels, config)
    train_ids, val_ids = oof_split(development, labels, fold, config)
    if set(train_ids.tolist()) & set(locked_test.tolist()) or set(val_ids.tolist()) & set(locked_test.tolist()):
        raise RuntimeError("Legacy holdout entered ensemble OOF training")
    seed = member_seed(config, fold, member)
    local_config = json.loads(json.dumps(config))
    if max_epochs is not None:
        local_config["training"]["epochs"] = int(max_epochs)
        local_config["training"]["early_stopping_patience"] = int(max_epochs)
    print(
        f"{config['experiment']['id']} OOF fold={fold} member={member} seed={seed} device={device} "
        f"train={len(train_ids)} val={len(val_ids)} locked={len(locked_test)}",
        flush=True,
    )
    model, best_epoch, best_nll, history = train_model(
        data,
        train_ids,
        val_ids,
        local_config,
        seed,
        device,
        initial_encoder_state=initial_encoder_state,
    )
    validation = make_dataset(data, val_ids, local_config, seed + 22, training=False)
    prediction = deterministic_predictions(
        model, make_loader(validation, local_config, False, seed + 102), device
    )
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "config_sha256": None, "fold": fold, "member": member},
        directory / f"oof_fold_{fold}_member_{member}.pt",
    )
    payload = {
        "fold": fold,
        "member": member,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_nll": best_nll,
        "pretraining_checkpoint": pretraining_checkpoint,
        "train_patient_ids": train_ids,
        "val_patient_ids": val_ids,
        "locked_test_patient_ids": locked_test,
        "val_predictions": prediction,
        "history": history,
    }
    output.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
    print(f"complete fold={fold} member={member} epoch={best_epoch} nll={best_nll:.4f}", flush=True)
    return output


def all_oof_paths(config: dict) -> list[Path]:
    directory = result_dir(config)
    return [
        directory / f"oof_fold_{fold}_member_{member}.json"
        for fold in range(int(config["split"]["oof_folds"]))
        for member in range(int(config["ensemble"]["members"]))
    ]


def aggregate_oof(config: dict) -> Path:
    missing = [path for path in all_oof_paths(config) if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} OOF member results")
    directory = result_dir(config)
    n_folds = int(config["split"]["oof_folds"])
    n_members = int(config["ensemble"]["members"])
    patient_ids, labels_all, probabilities_all, epistemic_all = [], [], [], []
    best_epochs = {member: [] for member in range(n_members)}
    fold_rows = []
    for fold in range(n_folds):
        member_results = [
            json.loads((directory / f"oof_fold_{fold}_member_{member}.json").read_text(encoding="utf-8"))
            for member in range(n_members)
        ]
        reference_ids = np.asarray(member_results[0]["val_predictions"]["patient_ids"], dtype=int)
        reference_labels = np.asarray(member_results[0]["val_predictions"]["labels"], dtype=int)
        member_probabilities = []
        for member, result in enumerate(member_results):
            ids = np.asarray(result["val_predictions"]["patient_ids"], dtype=int)
            if not np.array_equal(ids, reference_ids):
                raise RuntimeError(f"Patient order differs in fold {fold}")
            member_probabilities.append(np.asarray(result["val_predictions"]["probabilities"], dtype=float))
            best_epochs[member].append(int(result["best_epoch"]))
        matrix = np.stack(member_probabilities)
        ensemble_probability = matrix.mean(axis=0)
        patient_ids.append(reference_ids)
        labels_all.append(reference_labels)
        probabilities_all.append(ensemble_probability)
        epistemic_all.append(matrix.var(axis=0))
        fold_rows.append({
            "fold": fold,
            "n_val": len(reference_ids),
            "raw_auc": float(roc_auc_score(reference_labels, ensemble_probability)),
            "mean_best_epoch": float(np.mean([result["best_epoch"] for result in member_results])),
            "member_auc_mean": float(np.mean([
                roc_auc_score(reference_labels, member_probability) for member_probability in member_probabilities
            ])),
            "member_auc_std": float(np.std([
                roc_auc_score(reference_labels, member_probability) for member_probability in member_probabilities
            ], ddof=1)),
        })
    ids = np.concatenate(patient_ids)
    labels = np.concatenate(labels_all)
    raw_probability = np.concatenate(probabilities_all)
    ensemble_epistemic = np.concatenate(epistemic_all)
    if len(np.unique(ids)) != len(ids):
        raise RuntimeError("OOF patients are duplicated")
    data = load_dataset(resolve(config["paths"]["dataset_dir"]))
    all_labels = patient_labels(data)
    development, locked_test = outer_development_test(all_labels, config)
    if set(ids.tolist()) != set(development.tolist()) or set(ids.tolist()) & set(locked_test.tolist()):
        raise RuntimeError("OOF coverage/leakage audit failed")
    temperature = fit_temperature(raw_probability, labels, config)
    probability = calibrate(raw_probability, temperature)
    threshold_result = optimize_threshold(labels, probability, strategy=str(config["threshold"]["strategy"]))
    threshold = float(threshold_result["threshold"])
    oof_metrics = metrics(labels, probability, threshold)
    errors = ((probability >= threshold).astype(int) != labels).astype(int)
    uncertainty_auc = float(roc_auc_score(errors, ensemble_epistemic)) if len(np.unique(errors)) == 2 else float("nan")
    selected_epochs = {str(member): max(1, int(np.median(values))) for member, values in best_epochs.items()}
    payload = {
        "patient_ids": ids,
        "labels": labels,
        "raw_probability": raw_probability,
        "calibrated_probability": probability,
        "ensemble_epistemic_uncertainty": ensemble_epistemic,
        "temperature": temperature,
        "threshold_result": threshold_result,
        "metrics": oof_metrics,
        "epistemic_error_detection_auc": uncertainty_auc,
        "selected_final_epochs": selected_epochs,
        "locked_test_patient_ids": locked_test,
        "fold_matrix": fold_rows,
    }
    output = directory / "oof_ensemble.json"
    output.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
    pd.DataFrame(fold_rows).to_csv(directory / "oof_fold_matrix.csv", index=False)
    print(
        f"OOF ensemble AUC={oof_metrics['roc_auc']:.4f} Acc={oof_metrics['accuracy']:.4f} "
        f"BA={oof_metrics['balanced_accuracy']:.4f} uncertainty-AUC={uncertainty_auc:.4f}",
        flush=True,
    )
    return output


def run_final_member(
    config: dict,
    member: int,
    initial_encoder_state: dict | None = None,
    pretraining_checkpoint: str | None = None,
) -> Path:
    directory = result_dir(config)
    output = directory / f"final_member_{member}.json"
    if output.exists():
        print(f"skip completed final member={member}", flush=True)
        return output
    oof = json.loads((directory / "oof_ensemble.json").read_text(encoding="utf-8"))
    fixed_epochs = int(oof["selected_final_epochs"][str(member)])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_dataset(resolve(config["paths"]["dataset_dir"]))
    labels = patient_labels(data)
    development, locked_test = outer_development_test(labels, config)
    seed = member_seed(config, 900, member)
    print(
        f"{config['experiment']['id']} final member={member} seed={seed} epochs={fixed_epochs} device={device}",
        flush=True,
    )
    model, _, _, history = train_model(
        data,
        development,
        None,
        config,
        seed,
        device,
        fixed_epochs=fixed_epochs,
        initial_encoder_state=initial_encoder_state,
    )
    test_dataset = make_dataset(data, locked_test, config, seed + 33, training=False)
    prediction = stochastic_patient_predictions(
        model, make_loader(test_dataset, config, False, seed + 103), device, config
    )
    torch.save(
        {"model_state": model.state_dict(), "member": member, "epochs": fixed_epochs},
        directory / f"final_member_{member}.pt",
    )
    payload = {
        "member": member,
        "seed": seed,
        "epochs": fixed_epochs,
        "pretraining_checkpoint": pretraining_checkpoint,
        "test_predictions": prediction,
        "history": history,
    }
    output.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
    print(f"complete final member={member}", flush=True)
    return output


def all_final_paths(config: dict) -> list[Path]:
    directory = result_dir(config)
    return [directory / f"final_member_{member}.json" for member in range(int(config["ensemble"]["members"]))]


def finalize(config: dict) -> Path:
    missing = [path for path in all_final_paths(config) if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} final member results")
    directory = result_dir(config)
    oof = json.loads((directory / "oof_ensemble.json").read_text(encoding="utf-8"))
    members = [json.loads(path.read_text(encoding="utf-8")) for path in all_final_paths(config)]
    reference = members[0]["test_predictions"]
    patient_ids = np.asarray(reference["patient_ids"], dtype=int)
    labels = np.asarray(reference["labels"], dtype=int)
    raw_matrix = []
    for member in members:
        prediction = member["test_predictions"]
        if not np.array_equal(np.asarray(prediction["patient_ids"], dtype=int), patient_ids):
            raise RuntimeError("Final member patient order differs")
        raw_matrix.append(np.asarray(prediction["probabilities"], dtype=float))
    raw_matrix = np.stack(raw_matrix)
    raw_probability = raw_matrix.mean(axis=0)
    probability = calibrate(raw_probability, float(oof["temperature"]))
    threshold = float(oof["threshold_result"]["threshold"])
    prediction_label = (probability >= threshold).astype(int)
    test_metrics = metrics(labels, probability, threshold)
    ensemble_epistemic = raw_matrix.var(axis=0)
    sampling = np.mean([member["test_predictions"]["sampling_uncertainty"] for member in members], axis=0)
    aleatoric = np.mean([member["test_predictions"]["aleatoric_uncertainty"] for member in members], axis=0)
    mc_dropout_epistemic = np.mean(
        [member["test_predictions"]["epistemic_uncertainty"] for member in members], axis=0
    )
    errors = (prediction_label != labels).astype(int)
    uncertainty_auc = {
        "ensemble_epistemic": float(roc_auc_score(errors, ensemble_epistemic)) if len(np.unique(errors)) == 2 else float("nan"),
        "sampling": float(roc_auc_score(errors, sampling)) if len(np.unique(errors)) == 2 else float("nan"),
        "aleatoric": float(roc_auc_score(errors, aleatoric)) if len(np.unique(errors)) == 2 else float("nan"),
        "mc_dropout_epistemic": float(roc_auc_score(errors, mc_dropout_epistemic)) if len(np.unique(errors)) == 2 else float("nan"),
    }
    dataset = np.load(resolve(config["paths"]["dataset_dir"]) / "spectra.npz", allow_pickle=True)
    uids = [str(value) for value in dataset["patient_uids"]]
    patient_frame = pd.DataFrame({
        "patient_id": patient_ids,
        "patient_uid": [uids[pid] for pid in patient_ids],
        "true_label": labels,
        "probability": probability,
        "predicted_label": prediction_label,
        "correct": 1 - errors,
        "sampling_uncertainty": sampling,
        "aleatoric_uncertainty": aleatoric,
        "mc_dropout_epistemic": mc_dropout_epistemic,
        "ensemble_epistemic": ensemble_epistemic,
    })
    patient_frame.to_csv(directory / "patient_predictions.csv", index=False)
    summary = pd.DataFrame([
        {"set": "OOF development", "n": 41, **oof["metrics"]},
        {"set": "legacy observed holdout", "n": len(labels), **test_metrics},
    ])
    summary.to_csv(directory / "summary_matrix.csv", index=False)
    payload = {
        "experiment_id": config["experiment"]["id"],
        "test_status": config["experiment"]["test_status"],
        "leakage_audit": {"development_test_overlap": 0, "oof_unique_patients": 41},
        "oof": oof,
        "test": {
            "patient_ids": patient_ids,
            "labels": labels,
            "probability": probability,
            "predicted_label": prediction_label,
            "metrics": test_metrics,
            "uncertainty_error_detection_auc": uncertainty_auc,
        },
    }
    output = directory / "final_results.json"
    output.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
    experiment_metrics = {
        "oof_auc": oof["metrics"]["roc_auc"],
        "oof_accuracy": oof["metrics"]["accuracy"],
        "test_auc": test_metrics["roc_auc"],
        "test_accuracy": test_metrics["accuracy"],
    }
    manifest = update_manifest(directory, status="complete", completed_metrics=experiment_metrics)
    update_registry(config, manifest, experiment_metrics)
    print(
        f"FINAL ensemble AUC={test_metrics['roc_auc']:.4f} Acc={test_metrics['accuracy']:.4f} "
        f"BA={test_metrics['balanced_accuracy']:.4f}",
        flush=True,
    )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Phase5/configs/exp_001_ensemble.yaml")
    parser.add_argument("--mode", choices=["prepare", "oof-member", "aggregate-oof", "final-member", "finalize"], required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--member", type=int)
    parser.add_argument("--max-epochs", type=int)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    if args.mode == "prepare":
        prepare(config, config_path)
    elif args.mode == "oof-member":
        run_oof_member(config, args.fold, args.member, args.max_epochs)
    elif args.mode == "aggregate-oof":
        aggregate_oof(config)
    elif args.mode == "final-member":
        run_final_member(config, args.member)
    else:
        finalize(config)


if __name__ == "__main__":
    main()
