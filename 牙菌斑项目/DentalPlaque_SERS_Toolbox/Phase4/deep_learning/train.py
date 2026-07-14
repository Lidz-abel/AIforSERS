r"""Phase 4A: CC-SERSNet v1 Training & Evaluation Pipeline.

Train → MC Dropout Inference → Calibrate → Clinical Decision → Evaluate.

All evaluation is at the PATIENT level.
"""

from __future__ import annotations

import csv
import io
import json
import random
import sys
import time
import warnings
from pathlib import Path

# Work around Windows GBK encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

# Add Phase3/baseline to path for shared utils
_phase3_baseline = Path(__file__).resolve().parents[2] / "Phase3" / "baseline"
sys.path.insert(0, str(_phase3_baseline))

from baseline_utils import (
    binomial_ci,
    bootstrap_metric_ci,
    compute_calibration_curve_data,
    compute_metrics,
    expected_calibration_error,
    write_json,
)

from datasets import (
    SpectrumDataset,
    audit_splits,
    build_spectrum_masks,
    build_split_masks,
    load_phase4_dataset,
)
from models import MultiScaleEncoder
from reliability import (
    TemperatureScaling,
    aggregate_to_patient,
    clinical_decision,
    compute_clinical_confidence,
    compute_spectrum_reliability,
)

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "figure.facecolor": "white",
})


def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Training loop ────────────────────────────────────────────────────────
    """Compute patient-level AUC: aggregate spectrum probs per patient, then AUC."""
    from sklearn.metrics import roc_auc_score

    model.eval()
    all_probs = []
    all_pidx = []
    all_y = []

    with torch.no_grad():
        sample_idx = 0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            probs = F.softmax(model(X_batch), dim=1)[:, 1].cpu().numpy()
            n = len(probs)
            all_probs.extend(probs)
            all_y.extend(y_batch.numpy())
            all_pidx.extend(patient_index[sample_idx:sample_idx + n])
            sample_idx += n

    all_probs = np.array(all_probs)
    all_y = np.array(all_y)
    all_pidx = np.array(all_pidx)

    # Aggregate to target patients
    target_pids = np.where(patient_mask)[0]
    p_probs = np.array([all_probs[all_pidx == pid].mean() for pid in target_pids])
    p_labels = np.array([all_y[all_pidx == pid][0] for pid in target_pids])

    if len(np.unique(p_labels)) < 2:
        return float("nan")
    return roc_auc_score(p_labels, p_probs)


# ── Training loop ────────────────────────────────────────────────────────


def _patient_auc_from_model(model, dataloader, patient_index, patient_mask, device):
    """Compute patient-level AUC: aggregate spectrum probs per patient, then AUC."""
    from sklearn.metrics import roc_auc_score

    model.eval()
    all_probs = []
    all_pidx = []
    all_y = []

    with torch.no_grad():
        sample_idx = 0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            probs = F.softmax(model(X_batch), dim=1)[:, 1].cpu().numpy()
            n = len(probs)
            all_probs.extend(probs)
            all_y.extend(y_batch.numpy())
            all_pidx.extend(patient_index[sample_idx:sample_idx + n])
            sample_idx += n

    all_probs = np.array(all_probs)
    all_y = np.array(all_y)
    all_pidx = np.array(all_pidx)

    target_pids = np.where(patient_mask)[0]
    p_probs = np.array([all_probs[all_pidx == pid].mean() for pid in target_pids])
    p_labels = np.array([all_y[all_pidx == pid][0] for pid in target_pids])

    if len(np.unique(p_labels)) < 2:
        return float("nan")
    return roc_auc_score(p_labels, p_probs)


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_weight = 0.0
    has_weights = dataloader.dataset.sample_weight is not None

    for batch in dataloader:
        if has_weights:
            X_batch, y_batch, w_batch = batch
            w_batch = w_batch.to(device)
        else:
            X_batch, y_batch = batch
            w_batch = None

        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss_per_sample = F.cross_entropy(logits, y_batch, reduction="none")
        if w_batch is not None:
            loss = (loss_per_sample * w_batch).sum() / w_batch.sum()
        else:
            loss = loss_per_sample.mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
        total_weight += len(X_batch)
    return total_loss / total_weight if total_weight > 0 else 0.0


@torch.no_grad()
def val_loss(model, dataloader, device):
    model.eval()
    total = 0.0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        total += F.cross_entropy(logits, y_batch, reduction="sum").item()
    return total / len(dataloader.dataset)


# ── MC Dropout patient-level inference ────────────────────────────────────


def mc_patient_inference(
    model, dataloader, patient_index, patient_uids, labels, patient_mask, device, cfg
):
    """MC Dropout inference → patient-level aggregation.

    Returns patient-level reliability dict.
    """
    n_samples = cfg["mc_dropout"]["n_samples"]

    all_mc_probs = []
    all_pidx = []
    all_y = []

    sample_idx = 0
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        _, mc_probs = model.mc_predict_proba(X_batch, n_samples=n_samples)
        # mc_probs: [T, B, C]
        all_mc_probs.append(mc_probs.cpu())
        all_y.extend(y_batch.numpy())
        n_b = len(y_batch)
        all_pidx.extend(patient_index[sample_idx:sample_idx + n_b])
        sample_idx += n_b

    # Concatenate all spectra
    mc_probs_all = torch.cat(all_mc_probs, dim=1)  # [T, N_total, C]
    all_pidx = np.array(all_pidx)
    all_y = np.array(all_y)

    # Spectrum-level reliability
    spec_rel = compute_spectrum_reliability(mc_probs_all)

    # Patient-level aggregation
    pat_rel = aggregate_to_patient(
        spec_rel, all_pidx, patient_uids, all_y, patient_mask
    )

    return pat_rel, spec_rel, all_pidx


# ── Calibration (Temperature Scaling) ─────────────────────────────────────


def calibrate_on_val(model, val_loader, patient_index, patient_uids, labels,
                     val_patient_mask, device, cfg):
    """Fit temperature scaling on validation MC-posterior patient-level logits.

    Uses MC Dropout (same as test-time inference) for statistical consistency
    between calibration and evaluation pipelines.
    """
    if cfg["calibration"]["method"] != "temperature_scaling":
        return None

    n_samples = cfg["mc_dropout"]["n_samples"]

    all_mc_logits = []  # mean logits per spectrum over MC samples
    all_pidx = []
    all_y = []

    sample_idx = 0
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        # MC posterior: [T, B, C] logits
        mc_logits = model.mc_dropout_sample(X_batch, n_samples=n_samples).detach()
        # Mean logits over MC samples → [B, C]
        mean_logits = mc_logits.mean(dim=0).cpu()
        all_mc_logits.append(mean_logits)
        all_y.extend(y_batch.numpy())
        n_b = len(y_batch)
        all_pidx.extend(patient_index[sample_idx:sample_idx + n_b])
        sample_idx += n_b

    logits_all = torch.cat(all_mc_logits, dim=0)
    all_pidx = np.array(all_pidx)
    all_y = np.array(all_y)

    # Aggregate to patient-level
    target_pids = np.where(val_patient_mask)[0]
    p_logits = torch.stack([
        logits_all[all_pidx == pid].mean(dim=0) for pid in target_pids
    ])
    p_labels = torch.tensor([all_y[all_pidx == pid][0] for pid in target_pids])

    scaler = TemperatureScaling()
    T = scaler.fit(p_logits, p_labels)
    print(f"  Temperature (T): {T:.4f}")
    return scaler


# ── Full evaluation ──────────────────────────────────────────────────────


def _mc_patient_inference_calibrated(
    model, dataloader, patient_index, patient_uids, labels,
    patient_mask, device, cfg, temp_scaler,
):
    """MC Dropout inference with temperature scaling applied to RAW LOGITS."""
    n_samples = cfg["mc_dropout"]["n_samples"]

    all_mc_probs = []
    all_pidx = []
    all_y = []

    sample_idx = 0
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        # Get raw MC logits, apply temperature, then softmax
        mc_logits = model.mc_dropout_sample(X_batch, n_samples=n_samples).detach()  # [T, B, C]
        mc_logits_cal = mc_logits / temp_scaler.temperature.detach().to(mc_logits.device)
        mc_probs = F.softmax(mc_logits_cal, dim=-1)  # [T, B, C]
        all_mc_probs.append(mc_probs.cpu())
        all_y.extend(y_batch.numpy())
        n_b = len(y_batch)
        all_pidx.extend(patient_index[sample_idx:sample_idx + n_b])
        sample_idx += n_b

    mc_probs_all = torch.cat(all_mc_probs, dim=1)
    all_pidx = np.array(all_pidx)
    all_y = np.array(all_y)

    spec_rel = compute_spectrum_reliability(mc_probs_all)
    pat_rel = aggregate_to_patient(spec_rel, all_pidx, patient_uids, all_y, patient_mask)
    return pat_rel


def full_evaluation(model, dataloader, patient_index, patient_uids, labels,
                    patient_mask, device, cfg, temp_scaler, split_name="test"):
    """MC Dropout → aggregate → calibrate → clinical → metrics.

    Runs MC inference ONCE: calibrated path if temp_scaler is provided,
    uncalibrated otherwise.  No wasted computation.
    """
    # 1. MC Dropout inference (calibrated or uncalibrated, single pass)
    if temp_scaler is not None:
        pat_rel = _mc_patient_inference_calibrated(
            model, dataloader, patient_index, patient_uids, labels,
            patient_mask, device, cfg, temp_scaler,
        )
        calibrated = True
    else:
        pat_rel, _spec_rel, _all_pidx = mc_patient_inference(
            model, dataloader, patient_index, patient_uids, labels,
            patient_mask, device, cfg
        )
        calibrated = False

    # 3. Clinical confidence
    clin_conf = compute_clinical_confidence(
        pat_rel,
        weights=(
            cfg["clinical"]["prob_margin_weight"],
            cfg["clinical"]["entropy_norm_weight"],
            cfg["clinical"]["mi_norm_weight"],
            cfg["clinical"]["patient_agreement_weight"],
        ),
        high_thresh=cfg["clinical"]["high_confidence_threshold"],
        med_thresh=cfg["clinical"]["medium_confidence_threshold"],
    )

    # 4. Clinical recommendation
    recommendations = clinical_decision(pat_rel, clin_conf)

    # 5. Patient-level metrics
    y_true = pat_rel["true_label"]
    y_prob = pat_rel["prob_positive"]
    metrics = compute_metrics(y_true, y_prob)

    # CIs: bootstrap for continuous metrics, binomial for proportions
    from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
    from baseline_utils import sensitivity_score, specificity_score

    y_pred = (y_prob >= 0.5).astype(int)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    tp = int((y_pred[y_true == 1] == 1).sum())
    tn = int((y_pred[y_true == 0] == 0).sum())

    metric_fns = {
        "roc_auc": lambda yt, yp: roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else float("nan"),
        "accuracy": lambda yt, yp: accuracy_score(yt, (yp >= 0.5).astype(int)),
        "brier_score": brier_score_loss,
        "ece": expected_calibration_error,
    }
    # sensitivity and specificity use binomial CI — see below
    BINOMIAL_METRICS = {"sensitivity", "specificity"}

    metrics_ci = {}
    for m in cfg["evaluation"]["metrics"]:
        if m in BINOMIAL_METRICS:
            continue  # handled below
        if m in metric_fns:
            metrics_ci[m] = bootstrap_metric_ci(
                y_true, y_prob, metric_fns[m],
                n_bootstrap=cfg["evaluation"]["n_bootstrap"],
                alpha=cfg["evaluation"]["ci_alpha"],
                seed=cfg["seed"],
                metric_name=m,
            )

    # Binomial CI for sensitivity (tp / n_pos) and specificity (tn / n_neg)
    metrics_ci["sensitivity"] = binomial_ci(tp, n_pos, alpha=cfg["evaluation"]["ci_alpha"])
    metrics_ci["specificity"] = binomial_ci(tn, n_neg, alpha=cfg["evaluation"]["ci_alpha"])

    # 6. Stratified metrics by confidence group
    conf_metrics = {}
    for group in ["high", "medium", "low"]:
        mask = clin_conf["confidence_group"] == group
        if mask.sum() > 0 and len(np.unique(y_true[mask])) > 1:
            conf_metrics[group] = compute_metrics(y_true[mask], y_prob[mask])
            conf_metrics[group]["n_patients"] = int(mask.sum())
        else:
            conf_metrics[group] = {"n_patients": int(mask.sum())}

    # 7. Calibration curve
    calib_data = compute_calibration_curve_data(y_true, y_prob)

    return {
        "patient_rel": pat_rel,
        "clinical_conf": clin_conf,
        "recommendations": recommendations,
        "metrics": metrics,
        "metrics_ci": metrics_ci,
        "conf_metrics": conf_metrics,
        "calib_data": calib_data,
        "calibrated": calibrated,
    }


# ── Output writers ────────────────────────────────────────────────────────


def write_predictions_csv(path, eval_result, patient_uids):
    """Write predictions.csv with all patient-level fields."""
    pr = eval_result["patient_rel"]
    cc = eval_result["clinical_conf"]
    recs = eval_result["recommendations"]
    n = len(pr["patient_uid"])

    header = [
        "patient_uid", "true_label", "pred_label",
        "prob_negative", "prob_positive",
        "predictive_entropy", "expected_entropy", "mutual_information",
        "margin", "patient_agreement", "prob_variance",
        "clinical_confidence", "confidence_group",
        "recommendation", "correct", "split",
    ]

    rows = []
    for i in range(n):
        correct = int(pr["true_label"][i] == pr["pred_class"][i])
        rows.append([
            pr["patient_uid"][i], int(pr["true_label"][i]), int(pr["pred_class"][i]),
            round(float(pr["prob_negative"][i]), 6), round(float(pr["prob_positive"][i]), 6),
            round(float(pr["entropy_mean"][i]), 6),
            round(float(pr["expected_entropy_mean"][i]), 6),
            round(float(pr["mi_mean"][i]), 6),
            round(float(pr["margin_mean"][i]), 6),
            round(float(pr["patient_agreement"][i]), 6),
            round(float(pr["prob_variance"][i]), 6),
            round(float(cc["clinical_confidence"][i]), 6),
            cc["confidence_group"][i],
            recs[i], correct, "test",
        ])

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Predictions saved: {path}")


def write_reliability_csv(path, eval_result):
    """Write reliability_patient.csv."""
    pr = eval_result["patient_rel"]
    cc = eval_result["clinical_conf"]
    recs = eval_result["recommendations"]
    n = len(pr["patient_uid"])

    header = [
        "patient_uid", "true_label", "pred_label",
        "prob_positive", "prob_variance", "patient_agreement",
        "entropy_mean", "mi_mean", "margin_mean",
        "prob_margin_raw", "entropy_norm", "mi_norm",
        "clinical_confidence", "confidence_group", "recommendation",
    ]
    rows = []
    for i in range(n):
        rows.append([
            pr["patient_uid"][i], int(pr["true_label"][i]), int(pr["pred_class"][i]),
            round(float(pr["prob_positive"][i]), 6),
            round(float(pr["prob_variance"][i]), 6),
            round(float(pr["patient_agreement"][i]), 6),
            round(float(pr["entropy_mean"][i]), 6),
            round(float(pr["mi_mean"][i]), 6),
            round(float(pr["margin_mean"][i]), 6),
            round(float(cc["prob_margin_raw"][i]), 6),
            round(float(cc["entropy_norm"][i]), 6),
            round(float(cc["mi_norm"][i]), 6),
            round(float(cc["clinical_confidence"][i]), 6),
            cc["confidence_group"][i], recs[i],
        ])

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Reliability saved: {path}")


# ── Figures ───────────────────────────────────────────────────────────────


def plot_training_curve(log: list[dict], path: str):
    """Plot training/validation loss and patient AUC."""
    epochs = [r["epoch"] for r in log]
    train_loss = [r["train_loss"] for r in log]
    val_loss_vals = [r["val_loss"] for r in log]
    val_auc = [r.get("val_patient_auc", float("nan")) for r in log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    ax1.plot(epochs, train_loss, label="Train Loss", linewidth=1.2)
    ax1.plot(epochs, val_loss_vals, label="Val Loss", linewidth=1.2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_auc, "o-", markersize=3, linewidth=1.2, color="#2ca02c",
             label="Val Patient AUC")
    best_epoch = np.nanargmax(val_auc) if not all(np.isnan(val_auc)) else 0
    best_auc = val_auc[best_epoch]
    ax2.axvline(epochs[best_epoch], color="red", linestyle="--", alpha=0.5,
                label=f"Best epoch={epochs[best_epoch]} (AUC={best_auc:.4f})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Patient AUC")
    ax2.set_title("Validation Patient-Level AUC")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curve saved: {path}")


def plot_confidence_distribution(eval_result, path):
    """Histogram of clinical confidence scores colored by group."""
    cc = eval_result["clinical_conf"]
    fig, ax = plt.subplots(figsize=(5, 3.5))

    colors = {"high": "#2ca02c", "medium": "#ff7f0e", "low": "#d62728"}
    for group, color in colors.items():
        mask = cc["confidence_group"] == group
        if mask.sum() > 0:
            ax.hist(cc["clinical_confidence"][mask], bins=10, alpha=0.6,
                    color=color, label=f"{group} (n={mask.sum()})")

    ax.set_xlabel("Clinical Confidence")
    ax.set_ylabel("Patient Count")
    ax.set_title("Clinical Confidence Distribution (Test Set)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Confidence distribution saved: {path}")


def plot_calibration(eval_result, path):
    """Calibration curve."""
    calib = eval_result["calib_data"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(calib["prob_pred"], calib["prob_true"], "o-", color="#1f77b4",
            linewidth=1.5, markersize=6, label="CC-SERSNet-v1")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Perfect")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Calibration Curve (Test Set)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Calibration curve saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    cfg = load_config()
    seed = cfg["seed"]

    # ── Reproducibility ────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed: {seed}")

    # Resolve output paths relative to toolbox root (consistent with Phase3)
    from baseline_utils import toolbox_root, resolve_path
    root = toolbox_root()

    def out_path(key: str) -> Path:
        return resolve_path(cfg["outputs"][key])

    # ── Load data ──────────────────────────────────────────
    print("\n[1/5] Loading dataset...")
    data = load_phase4_dataset(cfg)
    audit_splits(data)

    patient_masks = build_split_masks(data["patient_uids"], data["splits"])
    spec_masks = build_spectrum_masks(data["patient_index"], patient_masks)

    # ── Build datasets & dataloaders ────────────────────────
    X = data["X_spectra"]
    y = data["labels"]

    train_ds = SpectrumDataset(X[spec_masks["train"]], y[spec_masks["train"]])
    val_ds = SpectrumDataset(X[spec_masks["val"]], y[spec_masks["val"]])
    test_ds = SpectrumDataset(X[spec_masks["test"]], y[spec_masks["test"]])

    # ── Patient-balanced sample weights (train only) ────────
    # Weight = 1 / n_spectra_for_this_patient → each patient contributes equally.
    train_pidx = data["patient_index"][spec_masks["train"]]
    unique_pids, inv, counts = np.unique(train_pidx, return_inverse=True, return_counts=True)
    patient_weight = 1.0 / counts.astype(np.float32)
    train_weights = patient_weight[inv]
    # Normalize so mean weight = 1.0
    train_weights = train_weights / train_weights.mean()
    train_ds.sample_weight = torch.as_tensor(train_weights, dtype=torch.float32)
    print(f"  Train sample weights: min={train_weights.min():.3f}, max={train_weights.max():.3f}, "
          f"mean={train_weights.mean():.3f}")

    batch_size = cfg["training"]["batch_size"]
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # ── Build model ─────────────────────────────────────────
    print("\n[2/5] Building CC-SERSNet-v1...")
    m_cfg = cfg["model"]
    model = MultiScaleEncoder(
        in_channels=m_cfg["in_channels"],
        n_wavenumber=m_cfg["n_wavenumber"],
        n_classes=m_cfg["n_classes"],
        kernel_sizes=m_cfg["kernel_sizes"],
        base_channels=m_cfg["base_channels"],
        n_res_blocks=m_cfg["n_res_blocks"],
        dropout_rate=m_cfg["dropout_rate"],
        group_norm_groups=m_cfg["group_norm_groups"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # ── Train ───────────────────────────────────────────────
    print("\n[3/5] Training...")
    t_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=t_cfg["scheduler_factor"],
        patience=t_cfg["scheduler_patience"],
    )

    val_pidx_for_train = data["patient_index"][spec_masks["val"]]
    train_log = []
    best_auc = -1.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, t_cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, device)
        v_loss = val_loss(model, val_loader, device)
        v_auc = _patient_auc_from_model(
            model, val_loader, val_pidx_for_train, patient_masks["val"], device
        )

        scheduler.step(v_auc if not np.isnan(v_auc) else 0.0)
        cur_lr = optimizer.param_groups[0]["lr"]

        train_log.append({
            "epoch": epoch, "train_loss": tr_loss, "val_loss": v_loss,
            "val_patient_auc": float(v_auc) if not np.isnan(v_auc) else None,
            "lr": cur_lr,
        })

        elapsed = time.time() - t0
        auc_str = f"{v_auc:.4f}" if not np.isnan(v_auc) else "nan"
        print(f"  Epoch {epoch:3d} | tr_loss={tr_loss:.4f} val_loss={v_loss:.4f} "
              f"val_auc={auc_str} lr={cur_lr:.6f} ({elapsed:.1f}s)")

        if not np.isnan(v_auc) and v_auc > best_auc:
            best_auc = v_auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= t_cfg["early_stopping_patience"]:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  Best epoch: {best_epoch}, best val AUC: {best_auc:.4f}")

    # Save model
    model_path = out_path("model_weights")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "best_epoch": best_epoch,
                "best_val_auc": best_auc, "config": cfg}, model_path)
    print(f"  Model saved: {model_path}")

    # Save training log
    log_path = out_path("training_log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss",
                                                "val_patient_auc", "lr"])
        writer.writeheader()
        writer.writerows(train_log)
    print(f"  Training log saved: {log_path}")

    # ── Calibrate on val ────────────────────────────────────
    print("\n[4/5] Calibrating...")
    temp_scaler = calibrate_on_val(
        model, val_loader, data["patient_index"][spec_masks["val"]],
        data["patient_uids"], data["labels"],
        patient_masks["val"], device, cfg,
    )

    # ── Evaluate on test ────────────────────────────────────
    print("\n[5/5] Evaluating on test set...")
    eval_result = full_evaluation(
        model, test_loader,
        data["patient_index"][spec_masks["test"]],
        data["patient_uids"], data["labels"],
        patient_masks["test"], device, cfg, temp_scaler,
    )

    # ── Print metrics ───────────────────────────────────────
    m = eval_result["metrics_ci"]
    print(f"\n{'='*50}")
    print("Test Set Results (Patient-Level)")
    print(f"{'='*50}")
    for metric_name in ["roc_auc", "accuracy", "sensitivity", "specificity",
                         "brier_score", "ece"]:
        if metric_name in m:
            v = m[metric_name]
            print(f"  {metric_name:15s}: {v['value']:.4f} [{v['ci_lower']:.4f}, {v['ci_upper']:.4f}]")

    # Confidence breakdown
    print(f"\n  Confidence groups:")
    for group in ["high", "medium", "low"]:
        cm = eval_result["conf_metrics"][group]
        n_p = cm.get("n_patients", 0)
        acc = cm.get("accuracy", None)
        acc_str = f"{acc:.4f}" if acc is not None else "N/A"
        print(f"    {group:6s}: {n_p} patients, accuracy={acc_str}")

    # Recommendation breakdown
    recs = eval_result["recommendations"]
    for rec in ["Report", "Doctor Review", "Further Examination"]:
        n_rec = (recs == rec).sum()
        print(f"    {rec:20s}: {n_rec} patients")

    # ── Write outputs ───────────────────────────────────────
    write_predictions_csv(str(out_path("predictions")), eval_result,
                          data["patient_uids"])
    write_reliability_csv(str(out_path("reliability_patient")), eval_result)

    # Metrics JSON
    metrics_out = {
        "model": "CC-SERSNet-v1",
        "seed": cfg["seed"],
        "best_val_auc": best_auc,
        "best_epoch": best_epoch,
        "n_parameters": n_params,
        "calibrated": eval_result["calibrated"],
        "temperature": float(temp_scaler.temperature.item()) if temp_scaler is not None else None,
        "test_metrics": {k: v for k, v in eval_result["metrics_ci"].items()},
        "confidence_metrics": {
            group: {kk: (float(vv) if not isinstance(vv, (dict, type(None))) else vv)
                    for kk, vv in cm.items()}
            for group, cm in eval_result["conf_metrics"].items()
        },
    }
    write_json(out_path("metrics"), metrics_out)
    print(f"Metrics saved: {out_path('metrics')}")

    # Figures
    plot_training_curve(train_log, str(out_path("training_curve")))
    plot_calibration(eval_result, str(out_path("calibration_curve")))
    plot_confidence_distribution(eval_result, str(out_path("confidence_dist")))

    print("\nDone.")


if __name__ == "__main__":
    main()
