"""Train one Phase 4C MCSS-MIL split.

Usage from toolbox root:
  python Phase4/mcss/mcss_train.py --split_seed 42 --output Results/Phase4/mcss/split_42.json

This script trains one configured model on one patient-level split.  It does
not define or run an experiment matrix.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import random
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader

_toolbox = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_toolbox / "Phase3" / "baseline"))
sys.path.insert(0, str(_toolbox / "Phase4" / "deep_learning"))
sys.path.insert(0, str(_toolbox / "Phase4" / "stability"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_utils import binomial_ci, expected_calibration_error, resolve_path
from mcss_dataset import (
    MCSSBagDataset,
    audit_patient_split,
    load_mcss_dataset,
    make_patient_split,
    patient_labels_from_spectra,
)
from mcss_models import MCSSMILNet
from phase4b_utils import balanced_accuracy_score, optimize_threshold
from reliability import TemperatureScaling


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path is not None else Path(__file__).resolve().parent / "phase4c_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def make_bag_dataset(data: dict, patient_ids: np.ndarray, cfg: dict, seed: int, split_name: str):
    mcss = cfg["mcss"]
    dynamic = bool(mcss["dynamic_train_bags"]) and split_name == "train"
    bags_per_patient = (
        int(mcss["train_bags_per_patient"])
        if split_name == "train"
        else int(mcss["eval_bags_per_patient"])
    )
    return MCSSBagDataset(
        X_spectra=data["X_spectra"],
        labels=data["labels"],
        patient_index=data["patient_index"],
        patient_ids=patient_ids,
        bag_size=int(mcss["bag_size"]),
        bags_per_patient=bags_per_patient,
        seed=seed,
        dynamic=dynamic,
        sample_with_replacement=bool(mcss["sample_with_replacement"]),
    )


def build_model(cfg: dict) -> MCSSMILNet:
    m = cfg["model"]
    return MCSSMILNet(
        in_channels=int(m["in_channels"]),
        n_classes=int(m["n_classes"]),
        kernel_sizes=list(m["kernel_sizes"]),
        base_channels=int(m["base_channels"]),
        n_res_blocks=int(m["n_res_blocks"]),
        embedding_dim=int(m["embedding_dim"]),
        dropout_rate=float(m["dropout_rate"]),
        group_norm_groups=int(m["group_norm_groups"]),
        pooling=str(m["pooling"]),
        attention_hidden=int(m["attention_hidden"]),
    )


def train_one_epoch(model, loader, optimizer, device, label_smoothing: float = 0.0):
    model.train()
    total_loss = 0.0
    total_n = 0
    for bag, label, _patient_id in loader:
        bag = bag.to(device)
        label = label.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(bag)
        loss = F.cross_entropy(logits, label, label_smoothing=label_smoothing)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(label)
        total_n += len(label)
    return total_loss / max(total_n, 1)


@torch.no_grad()
def val_loss(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_n = 0
    for bag, label, _patient_id in loader:
        bag = bag.to(device)
        label = label.to(device)
        logits = model(bag)
        loss = F.cross_entropy(logits, label, reduction="sum")
        total_loss += float(loss.item())
        total_n += len(label)
    return total_loss / max(total_n, 1)


def aggregate_logits_by_patient(logits: np.ndarray, labels: np.ndarray, patient_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_pids = np.unique(patient_ids)
    patient_logits = []
    patient_labels = []
    for pid in unique_pids:
        mask = patient_ids == pid
        patient_logits.append(logits[mask].mean(axis=0))
        values = np.unique(labels[mask])
        if len(values) != 1:
            raise ValueError(f"Patient {pid} has inconsistent labels in bags: {values}")
        patient_labels.append(int(values[0]))
    return unique_pids.astype(np.int64), np.vstack(patient_logits), np.array(patient_labels, dtype=np.int64)


def collect_bag_logits(model, loader, device, n_mc_samples: int = 1, temperature_scaler=None):
    all_logits = []
    all_labels = []
    all_patient_ids = []

    for bag, label, patient_id in loader:
        bag = bag.to(device)
        if n_mc_samples > 1:
            logits_t = model.mc_dropout_sample(bag, n_samples=n_mc_samples)
            if temperature_scaler is not None:
                logits_t = logits_t / temperature_scaler.temperature.detach().to(logits_t.device)
            logits = logits_t.mean(dim=0)
        else:
            model.eval()
            with torch.no_grad():
                logits = model(bag)
                if temperature_scaler is not None:
                    logits = logits / temperature_scaler.temperature.detach().to(logits.device)

        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(label.numpy())
        all_patient_ids.append(patient_id.numpy())

    return (
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_patient_ids, axis=0),
    )


def evaluate_patient_level(
    model,
    loader,
    device,
    threshold: float,
    n_mc_samples: int,
    temperature_scaler=None,
) -> dict:
    bag_logits, bag_labels, bag_patient_ids = collect_bag_logits(
        model,
        loader,
        device,
        n_mc_samples=n_mc_samples,
        temperature_scaler=temperature_scaler,
    )
    patient_ids, patient_logits, y_true = aggregate_logits_by_patient(
        bag_logits, bag_labels, bag_patient_ids
    )
    probs = torch.softmax(torch.as_tensor(patient_logits), dim=1).numpy()[:, 1]
    y_pred = (probs >= threshold).astype(np.int64)

    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    tp = int((y_pred[y_true == 1] == 1).sum())
    tn = int((y_pred[y_true == 0] == 0).sum())

    return {
        "patient_ids": patient_ids,
        "true_label": y_true,
        "prob_positive": probs.astype(np.float64),
        "pred_label": y_pred,
        "roc_auc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) >= 2 else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": binomial_ci(tp, n_pos),
        "specificity": binomial_ci(tn, n_neg),
        "brier_score": float(brier_score_loss(y_true, probs)),
        "ece": float(expected_calibration_error(y_true, probs)),
    }


def fit_temperature(model, val_loader, device, n_mc_samples: int) -> tuple[TemperatureScaling | None, float | None]:
    bag_logits, bag_labels, bag_patient_ids = collect_bag_logits(
        model,
        val_loader,
        device,
        n_mc_samples=n_mc_samples,
        temperature_scaler=None,
    )
    _patient_ids, patient_logits_np, y_true_np = aggregate_logits_by_patient(
        bag_logits, bag_labels, bag_patient_ids
    )
    scaler = TemperatureScaling()
    temperature = scaler.fit(
        torch.as_tensor(patient_logits_np, dtype=torch.float32),
        torch.as_tensor(y_true_np, dtype=torch.long),
    )
    return scaler, float(temperature)


def metric_for_selection(metrics: dict, name: str) -> float:
    if name == "val_auc":
        return float(metrics["roc_auc"])
    if name == "val_accuracy":
        return float(metrics["accuracy"])
    if name == "val_balanced_accuracy":
        return float(metrics["balanced_accuracy"])
    raise ValueError(f"Unknown selection metric: {name}")


def numpy_to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: numpy_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numpy_to_jsonable(v) for v in obj]
    return obj


def run_one_split(cfg: dict, split_seed: int, model_seed: int | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_seed is None:
        model_seed = split_seed + 4000
    set_reproducibility(model_seed)

    dataset_dir = resolve_path(cfg["paths"]["dataset_dir"])
    data = load_mcss_dataset(dataset_dir)
    patient_labels = patient_labels_from_spectra(data["labels"], data["patient_index"])
    split = make_patient_split(
        patient_labels=patient_labels,
        seed=split_seed,
        train_ratio=float(cfg["split"]["train_ratio"]),
        val_ratio=float(cfg["split"]["val_ratio"]),
        test_ratio=float(cfg["split"]["test_ratio"]),
        stratify=bool(cfg["split"]["stratify"]),
    )
    split_audit = audit_patient_split(split)

    train_ds = make_bag_dataset(data, split.train_patients, cfg, seed=model_seed + 11, split_name="train")
    val_ds = make_bag_dataset(data, split.val_patients, cfg, seed=model_seed + 22, split_name="val")
    test_ds = make_bag_dataset(data, split.test_patients, cfg, seed=model_seed + 33, split_name="test")

    batch_size = int(cfg["training"]["batch_size"])
    train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True, seed=model_seed + 101)
    val_loader = make_loader(val_ds, batch_size=batch_size, shuffle=False, seed=model_seed + 102)
    test_loader = make_loader(test_ds, batch_size=batch_size, shuffle=False, seed=model_seed + 103)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(cfg["training"]["scheduler_factor"]),
        patience=int(cfg["training"]["scheduler_patience"]),
    )

    best_score = -np.inf
    best_epoch = 0
    best_state = None
    patience = 0
    history = []
    train_eval_samples = int(cfg["mc_dropout"]["train_eval_samples"]) if cfg["mc_dropout"]["enabled"] else 1

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        train_ds.set_epoch(epoch)
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            label_smoothing=float(cfg["training"]["label_smoothing"]),
        )
        v_loss = val_loss(model, val_loader, device)
        val_metrics = evaluate_patient_level(
            model,
            val_loader,
            device,
            threshold=0.5,
            n_mc_samples=train_eval_samples,
            temperature_scaler=None,
        )
        score = metric_for_selection(val_metrics, str(cfg["training"]["selection_metric"]))
        scheduler.step(score if np.isfinite(score) else 0.0)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": v_loss,
                "val_auc": val_metrics["roc_auc"],
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            }
        )

        if np.isfinite(score) and score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            patience = 0
        else:
            patience += 1

        if patience >= int(cfg["training"]["early_stopping_patience"]):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_samples = int(cfg["mc_dropout"]["final_samples"]) if cfg["mc_dropout"]["enabled"] else 1
    temp_scaler = None
    temperature = None
    if bool(cfg["calibration"]["enabled"]):
        temp_scaler, temperature = fit_temperature(model, val_loader, device, n_mc_samples=final_samples)

    val_final = evaluate_patient_level(
        model,
        val_loader,
        device,
        threshold=0.5,
        n_mc_samples=final_samples,
        temperature_scaler=temp_scaler,
    )
    threshold_result = optimize_threshold(
        val_final["true_label"],
        val_final["prob_positive"],
        strategy=str(cfg["threshold"]["strategy"]),
        sens_constraint=cfg["threshold"].get("sens_constraint"),
        spec_constraint=cfg["threshold"].get("spec_constraint"),
    )
    test_metrics = evaluate_patient_level(
        model,
        test_loader,
        device,
        threshold=float(threshold_result["threshold"]),
        n_mc_samples=final_samples,
        temperature_scaler=temp_scaler,
    )

    return numpy_to_jsonable(
        {
            "phase": "Phase4C",
            "method": "MCSS-GatedAttention-MIL",
            "split_seed": split_seed,
            "model_seed": model_seed,
            "device": str(device),
            "best_epoch": best_epoch,
            "best_val_score": best_score,
            "temperature": temperature,
            "threshold": threshold_result["threshold"],
            "threshold_result": threshold_result,
            "split_audit": split_audit,
            "counts": {
                "n_patients": len(patient_labels),
                "n_spectra": int(len(data["labels"])),
                "n_train_bags": len(train_ds),
                "n_val_bags": len(val_ds),
                "n_test_bags": len(test_ds),
            },
            "val_metrics": val_final,
            "test_metrics": test_metrics,
            "history": history,
            "config": cfg,
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--model_seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    start = time.time()
    result = run_one_split(cfg, split_seed=args.split_seed, model_seed=args.model_seed)
    elapsed = time.time() - start

    m = result["test_metrics"]
    print(f"Phase4C completed in {elapsed:.1f}s")
    print(
        "Test: "
        f"AUC={m['roc_auc']:.3f}, "
        f"Acc={m['accuracy']:.3f}, "
        f"BA={m['balanced_accuracy']:.3f}, "
        f"Sens={m['sensitivity']['value']:.3f}, "
        f"Spec={m['specificity']['value']:.3f}"
    )

    if args.output:
        out_path = resolve_path(args.output)
    else:
        out_dir = resolve_path(cfg["paths"]["results_dir"])
        out_path = out_dir / f"phase4c_split_{args.split_seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
