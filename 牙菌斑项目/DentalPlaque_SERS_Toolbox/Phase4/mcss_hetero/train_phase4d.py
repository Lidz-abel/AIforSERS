"""Leakage-safe OOF training and locked-test evaluation for Phase 4D."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

TOOLBOX = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLBOX / "Phase3" / "baseline"))
sys.path.insert(0, str(TOOLBOX / "Phase4" / "stability"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_utils import expected_calibration_error, resolve_path
from hetero_mil_model import HeteroscedasticMCSSMIL
from patient_mcss_dataset import GroupedPatientMCSSDataset, load_dataset, patient_labels
from phase4b_utils import create_patient_split, optimize_threshold


def load_config(path: str | None) -> dict:
    config_path = Path(path) if path else Path(__file__).with_name("phase4d_config.yaml")
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def outer_development_test(labels: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    split_cfg = cfg["split"]
    split = create_patient_split(
        labels,
        seed=int(split_cfg["outer_seed"]),
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        stratify=True,
    )
    development = np.sort(np.asarray(split["train_patients"] + split["val_patients"], dtype=np.int64))
    test = np.sort(np.asarray(split["test_patients"], dtype=np.int64))
    if set(development.tolist()) & set(test.tolist()):
        raise RuntimeError("Outer patient leakage detected")
    return development, test


def oof_split(development: np.ndarray, labels: np.ndarray, fold: int, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedKFold(
        n_splits=int(cfg["split"]["oof_folds"]),
        shuffle=True,
        random_state=int(cfg["split"]["outer_seed"]),
    )
    splits = list(splitter.split(development, labels[development]))
    train_pos, val_pos = splits[fold]
    train_ids, val_ids = development[train_pos], development[val_pos]
    if set(train_ids.tolist()) & set(val_ids.tolist()):
        raise RuntimeError("OOF patient leakage detected")
    return train_ids, val_ids


def make_dataset(data: dict, patient_ids: np.ndarray, cfg: dict, seed: int, training: bool):
    mcss = cfg["mcss"]
    return GroupedPatientMCSSDataset(
        data=data,
        patient_ids=patient_ids,
        bag_size=int(mcss["bag_size"]),
        bags_per_group=int(mcss["train_bags_per_patient"] if training else mcss["eval_bags_per_group"]),
        groups_per_patient=int(mcss["train_repeats_per_patient"] if training else mcss["eval_groups_per_patient"]),
        seed=seed,
        dynamic=training,
        sample_with_replacement=bool(mcss["sample_with_replacement"]),
        max_wavenumber_shift=int(mcss["max_wavenumber_shift"]),
        gaussian_noise_std=float(mcss["gaussian_noise_std"]),
    )


def make_loader(dataset, cfg: dict, training: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=training,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def heteroscedastic_patient_loss(mu, log_var, targets, epoch: int, cfg: dict):
    training = cfg["training"]
    smoothed = targets * (1.0 - float(training["label_smoothing"])) + 0.5 * float(training["label_smoothing"])
    deterministic = epoch <= int(training["warmup_deterministic_epochs"])
    if deterministic:
        probability = torch.sigmoid(mu).mean(dim=1)
        classification = F.binary_cross_entropy(probability.clamp(1e-6, 1 - 1e-6), smoothed)
        variance_penalty = mu.new_zeros(())
    else:
        samples = int(training["aleatoric_samples"])
        epsilon = torch.randn(samples, *mu.shape, device=mu.device)
        noisy_logits = mu.unsqueeze(0) + torch.exp(0.5 * log_var).unsqueeze(0) * epsilon
        probability = torch.sigmoid(noisy_logits).mean(dim=(0, 2))
        classification = F.binary_cross_entropy(probability.clamp(1e-6, 1 - 1e-6), smoothed)
        variance_penalty = float(training["variance_regularization"]) * torch.exp(log_var).mean()
    consistency = torch.sigmoid(mu).var(dim=1, unbiased=False).mean()
    total = classification + variance_penalty + float(training["consistency_weight"]) * consistency
    return total, classification.detach(), consistency.detach()


@torch.no_grad()
def deterministic_predictions(model, loader, device) -> dict:
    model.eval()
    probabilities = defaultdict(list)
    true_labels = {}
    for bags, labels, patient_ids in loader:
        mu, _ = model(bags.to(device, non_blocking=True))
        item_prob = torch.sigmoid(mu).mean(dim=1).cpu().numpy()
        for pid, label, probability in zip(patient_ids.numpy(), labels.numpy(), item_prob):
            probabilities[int(pid)].append(float(probability))
            true_labels[int(pid)] = int(label)
    ids = np.asarray(sorted(probabilities), dtype=np.int64)
    return {
        "patient_ids": ids,
        "labels": np.asarray([true_labels[int(pid)] for pid in ids], dtype=np.int64),
        "probabilities": np.asarray([np.mean(probabilities[int(pid)]) for pid in ids], dtype=np.float64),
    }


def train_model(
    data,
    train_ids,
    val_ids,
    cfg,
    seed,
    device,
    fixed_epochs: int | None = None,
    initial_encoder_state: dict | None = None,
):
    set_seed(seed)
    train_ds = make_dataset(data, train_ids, cfg, seed + 11, training=True)
    train_loader = make_loader(train_ds, cfg, training=True, seed=seed + 101)
    val_loader = None
    if val_ids is not None:
        val_ds = make_dataset(data, val_ids, cfg, seed + 22, training=False)
        val_loader = make_loader(val_ds, cfg, training=False, seed=seed + 102)

    model = HeteroscedasticMCSSMIL(cfg["model"]).to(device)
    if initial_encoder_state is not None:
        model.encoder.load_state_dict(initial_encoder_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg["training"]["scheduler_factor"]),
        patience=int(cfg["training"]["scheduler_patience"]),
    )
    epochs = int(fixed_epochs or cfg["training"]["epochs"])
    best_nll = math.inf
    best_epoch = epochs
    best_state = None
    patience = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        patient_items = 0
        for bags, labels, _ in train_loader:
            bags = bags.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            mu, log_var = model(bags)
            loss, _, _ = heteroscedastic_patient_loss(mu, log_var, labels, epoch, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["gradient_clip_norm"]))
            optimizer.step()
            loss_sum += float(loss.item()) * len(labels)
            patient_items += len(labels)

        row = {"epoch": epoch, "train_loss": loss_sum / max(patient_items, 1)}
        if val_loader is not None:
            prediction = deterministic_predictions(model, val_loader, device)
            val_nll = float(log_loss(prediction["labels"], prediction["probabilities"], labels=[0, 1]))
            val_auc = float(roc_auc_score(prediction["labels"], prediction["probabilities"]))
            row.update({"val_nll": val_nll, "val_auc": val_auc})
            scheduler.step(val_nll)
            if val_nll < best_nll - 1e-5:
                best_nll = val_nll
                best_epoch = epoch
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
                patience = 0
            else:
                patience += 1
            if epoch % 5 == 0 or epoch == 1:
                print(f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} val_nll={val_nll:.4f} val_auc={val_auc:.4f}", flush=True)
            if patience >= int(cfg["training"]["early_stopping_patience"]):
                history.append(row)
                break
        else:
            if epoch % 5 == 0 or epoch == 1:
                print(f"epoch={epoch:03d}/{epochs} train_loss={row['train_loss']:.4f}", flush=True)
        history.append(row)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_nll, history


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray, cfg: dict) -> float:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))

    def objective(temperature):
        calibrated = 1.0 / (1.0 + np.exp(-logits / temperature))
        return log_loss(labels, calibrated, labels=[0, 1])

    result = minimize_scalar(
        objective,
        bounds=(float(cfg["calibration"]["temperature_min"]), float(cfg["calibration"]["temperature_max"])),
        method="bounded",
    )
    return float(result.x)


def calibrate(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)) / temperature
    return 1.0 / (1.0 + np.exp(-logits))


@torch.no_grad()
def stochastic_patient_predictions(model, loader, device, cfg) -> dict:
    n_dropout = int(cfg["inference"]["mc_dropout_samples"])
    n_noise = int(cfg["inference"]["aleatoric_samples"])
    dropout_sum = defaultdict(lambda: np.zeros(n_dropout, dtype=np.float64))
    dropout_count = defaultdict(int)
    sampling_values = defaultdict(list)
    aleatoric_values = defaultdict(list)
    true_labels = {}

    for bags, labels, patient_ids in loader:
        bags = bags.to(device, non_blocking=True)
        model.eval()
        mu_det, _ = model(bags)
        deterministic_bags = torch.sigmoid(mu_det).cpu().numpy()
        for row, pid, label in zip(deterministic_bags, patient_ids.numpy(), labels.numpy()):
            sampling_values[int(pid)].extend(row.tolist())
            true_labels[int(pid)] = int(label)

        for sample_index in range(n_dropout):
            model.train()
            mu, log_var = model(bags)
            epsilon = torch.randn(n_noise, *mu.shape, device=device)
            noisy_probability = torch.sigmoid(mu.unsqueeze(0) + torch.exp(0.5 * log_var).unsqueeze(0) * epsilon)
            item_probability = noisy_probability.mean(dim=(0, 2)).cpu().numpy()
            item_aleatoric = noisy_probability.var(dim=0, unbiased=False).mean(dim=1).cpu().numpy()
            for pid, probability, uncertainty in zip(patient_ids.numpy(), item_probability, item_aleatoric):
                dropout_sum[int(pid)][sample_index] += float(probability)
                aleatoric_values[int(pid)].append(float(uncertainty))
        for pid in patient_ids.numpy():
            dropout_count[int(pid)] += 1

    model.eval()
    ids = np.asarray(sorted(true_labels), dtype=np.int64)
    mean_probability, sampling, aleatoric, epistemic = [], [], [], []
    for pid in ids:
        mc_patient = dropout_sum[int(pid)] / dropout_count[int(pid)]
        mean_probability.append(mc_patient.mean())
        epistemic.append(mc_patient.var())
        sampling.append(np.var(sampling_values[int(pid)]))
        aleatoric.append(np.mean(aleatoric_values[int(pid)]))
    return {
        "patient_ids": ids,
        "labels": np.asarray([true_labels[int(pid)] for pid in ids], dtype=np.int64),
        "probabilities": np.asarray(mean_probability),
        "sampling_uncertainty": np.asarray(sampling),
        "aleatoric_uncertainty": np.asarray(aleatoric),
        "epistemic_uncertainty": np.asarray(epistemic),
    }


def metrics(labels, probabilities, threshold) -> dict:
    prediction = (probabilities >= threshold).astype(np.int64)
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "ece": float(expected_calibration_error(labels, probabilities)),
        "sensitivity": float(prediction[labels == 1].mean()),
        "specificity": float((1 - prediction[labels == 0]).mean()),
    }


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


def run_oof_fold(cfg: dict, fold: int, max_epochs: int | None = None) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_dataset(resolve_path(cfg["paths"]["dataset_dir"]))
    labels = patient_labels(data)
    development, test = outer_development_test(labels, cfg)
    train_ids, val_ids = oof_split(development, labels, fold, cfg)
    if set(train_ids.tolist()) & set(test.tolist()) or set(val_ids.tolist()) & set(test.tolist()):
        raise RuntimeError("Locked test patient entered OOF training")
    seed = int(cfg["seed"]) + 1000 + fold
    local_cfg = copy.deepcopy(cfg)
    if max_epochs is not None:
        local_cfg["training"]["epochs"] = int(max_epochs)
        local_cfg["training"]["early_stopping_patience"] = int(max_epochs)
    print(f"Phase4D OOF fold={fold} device={device} train={len(train_ids)} val={len(val_ids)} locked_test={len(test)}", flush=True)
    model, best_epoch, best_nll, history = train_model(data, train_ids, val_ids, local_cfg, seed, device)
    val_ds = make_dataset(data, val_ids, local_cfg, seed + 22, training=False)
    val_prediction = deterministic_predictions(model, make_loader(val_ds, local_cfg, False, seed + 102), device)
    results_dir = resolve_path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = results_dir / f"oof_fold_{fold}.pt"
    torch.save({"model_state": model.state_dict(), "config": cfg, "best_epoch": best_epoch}, checkpoint)
    output = results_dir / f"oof_fold_{fold}.json"
    payload = {
        "phase": "Phase4D",
        "fold": fold,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_nll": best_nll,
        "train_patient_ids": train_ids,
        "val_patient_ids": val_ids,
        "locked_test_patient_ids": test,
        "val_predictions": val_prediction,
        "history": history,
    }
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2)
    print(f"OOF fold {fold} complete: best_epoch={best_epoch} val_nll={best_nll:.4f} saved={output}", flush=True)
    return output


def run_finalize(cfg: dict) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = resolve_path(cfg["paths"]["results_dir"])
    fold_results = []
    for fold in range(int(cfg["split"]["oof_folds"])):
        with open(results_dir / f"oof_fold_{fold}.json", "r", encoding="utf-8") as handle:
            fold_results.append(json.load(handle))

    ids = np.concatenate([np.asarray(result["val_predictions"]["patient_ids"]) for result in fold_results])
    labels_oof = np.concatenate([np.asarray(result["val_predictions"]["labels"]) for result in fold_results])
    probabilities_oof = np.concatenate([np.asarray(result["val_predictions"]["probabilities"]) for result in fold_results])
    if len(np.unique(ids)) != len(ids):
        raise RuntimeError("OOF patients are duplicated")
    data = load_dataset(resolve_path(cfg["paths"]["dataset_dir"]))
    labels = patient_labels(data)
    development, test_ids = outer_development_test(labels, cfg)
    if set(ids.tolist()) != set(development.tolist()):
        raise RuntimeError("OOF predictions do not cover the development set exactly")

    temperature = fit_temperature(probabilities_oof, labels_oof, cfg)
    calibrated_oof = calibrate(probabilities_oof, temperature)
    threshold_result = optimize_threshold(labels_oof, calibrated_oof, strategy=str(cfg["threshold"]["strategy"]))
    final_epochs = max(1, int(np.median([result["best_epoch"] for result in fold_results])))
    seed = int(cfg["seed"]) + 9000
    print(f"Final training device={device} development={len(development)} locked_test={len(test_ids)} epochs={final_epochs}", flush=True)
    model, _, _, history = train_model(data, development, None, cfg, seed, device, fixed_epochs=final_epochs)
    test_ds = make_dataset(data, test_ids, cfg, seed + 33, training=False)
    test_loader = make_loader(test_ds, cfg, False, seed + 103)
    test_prediction = stochastic_patient_predictions(model, test_loader, device, cfg)
    calibrated_test = calibrate(test_prediction["probabilities"], temperature)
    threshold = float(threshold_result["threshold"])
    test_metrics = metrics(test_prediction["labels"], calibrated_test, threshold)

    checkpoint = results_dir / "phase4d_final_model.pt"
    torch.save({"model_state": model.state_dict(), "config": cfg, "temperature": temperature, "threshold": threshold}, checkpoint)
    output = results_dir / "phase4d_final_results.json"
    payload = {
        "phase": "Phase4D",
        "method": "Grouped-MCSS-AttentionMeanMIL-Heteroscedastic",
        "leakage_audit": {
            "development_patient_ids": development,
            "locked_test_patient_ids": test_ids,
            "overlap": len(set(development.tolist()) & set(test_ids.tolist())),
            "oof_unique_patients": len(np.unique(ids)),
        },
        "oof": {
            "temperature": temperature,
            "threshold_result": threshold_result,
            "best_epochs": [result["best_epoch"] for result in fold_results],
            "selected_final_epochs": final_epochs,
            "metrics": metrics(labels_oof, calibrated_oof, threshold),
        },
        "test": {
            **test_prediction,
            "calibrated_probability": calibrated_test,
            "predicted_label": (calibrated_test >= threshold).astype(np.int64),
            "metrics": test_metrics,
        },
        "final_history": history,
        "config": cfg,
    }
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2)
    print(f"FINAL Test AUC={test_metrics['roc_auc']:.4f} Acc={test_metrics['accuracy']:.4f} BA={test_metrics['balanced_accuracy']:.4f}", flush=True)
    print(f"Saved final result: {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", choices=["oof-fold", "finalize"], required=True)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    start = time.time()
    if args.mode == "oof-fold":
        if args.fold is None or not 0 <= args.fold < int(cfg["split"]["oof_folds"]):
            raise ValueError("--fold is required and must be a valid OOF fold")
        run_oof_fold(cfg, args.fold, args.max_epochs)
    else:
        run_finalize(cfg)
    print(f"Elapsed: {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
