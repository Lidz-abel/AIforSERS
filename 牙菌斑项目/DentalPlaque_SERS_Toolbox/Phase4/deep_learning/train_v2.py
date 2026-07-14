r"""Phase 4A v2: Stable Training with Full-Batch SGD + Strong Regularization.

Changes from v1:
  - Smaller model (~10K params: base_ch=16, 1 res block)
  - Full-batch SGD with momentum (exact gradients, no batch noise)
  - weight_decay=0.01 (100x stronger L2)
  - Cosine annealing (no noise-triggered LR drops)
  - Wider early stopping patience (50 epochs)
  - TensorBoard logging with gradient histograms
  - LOPO-CV mode for unbiased small-sample evaluation
"""

from __future__ import annotations

import csv
import io
import json
import sys
import time
import warnings
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

_phase3_baseline = Path(__file__).resolve().parents[2] / "Phase3" / "baseline"
sys.path.insert(0, str(_phase3_baseline))

from baseline_utils import (
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


def load_config_v2() -> dict:
    cfg_path = Path(__file__).resolve().parent / "config_v2.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Patient AUC ────────────────────────────────────────────────────────


def _patient_auc_from_model(model, dataloader, patient_index, patient_mask, device):
    from sklearn.metrics import roc_auc_score
    model.eval()
    all_probs, all_pidx, all_y = [], [], []
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
    all_probs, all_y, all_pidx = np.array(all_probs), np.array(all_y), np.array(all_pidx)
    target_pids = np.where(patient_mask)[0]
    p_probs = np.array([all_probs[all_pidx == pid].mean() for pid in target_pids])
    p_labels = np.array([all_y[all_pidx == pid][0] for pid in target_pids])
    if len(np.unique(p_labels)) < 2:
        return float("nan")
    return roc_auc_score(p_labels, p_probs)


# ── Full-batch training ─────────────────────────────────────────────────


def full_batch_train_epoch(model, X_train, y_train, optimizer, device, writer, epoch):
    """Single full-batch SGD step with gradient logging."""
    model.train()
    X_all = X_train.to(device)
    y_all = y_train.to(device)

    optimizer.zero_grad()
    logits = model(X_all)
    loss = F.cross_entropy(logits, y_all)
    loss.backward()

    # Log gradient norms per layer
    total_grad_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            g_norm = param.grad.data.norm(2).item()
            total_grad_norm += g_norm ** 2
            if writer is not None:
                writer.add_scalar(f"Gradients/{name}", g_norm, epoch)

    total_grad_norm = total_grad_norm ** 0.5
    if writer is not None:
        writer.add_scalar("Gradients/total_norm", total_grad_norm, epoch)

    optimizer.step()
    return loss.item()


@torch.no_grad()
def full_batch_val_loss(model, X_val, y_val, device):
    model.eval()
    logits = model(X_val.to(device))
    return F.cross_entropy(logits, y_val.to(device)).item()


# ── MC inference (reuse from train.py patterns) ──────────────────────────


def _mc_patient_inference_calibrated(
    model, dataloader, patient_index, patient_uids, labels,
    patient_mask, device, cfg, temp_scaler,
):
    n_samples = cfg["mc_dropout"]["n_samples"]
    all_mc_probs, all_pidx, all_y = [], [], []
    sample_idx = 0
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        mc_logits = model.mc_dropout_sample(X_batch, n_samples=n_samples).detach()
        mc_logits_cal = mc_logits / temp_scaler.temperature.detach().to(mc_logits.device)
        mc_probs = F.softmax(mc_logits_cal, dim=-1)
        all_mc_probs.append(mc_probs.cpu())
        all_y.extend(y_batch.numpy())
        n_b = len(y_batch)
        all_pidx.extend(patient_index[sample_idx:sample_idx + n_b])
        sample_idx += n_b
    mc_probs_all = torch.cat(all_mc_probs, dim=1)
    all_pidx, all_y = np.array(all_pidx), np.array(all_y)
    spec_rel = compute_spectrum_reliability(mc_probs_all)
    return aggregate_to_patient(spec_rel, all_pidx, patient_uids, all_y, patient_mask)


def mc_patient_inference(
    model, dataloader, patient_index, patient_uids, labels, patient_mask, device, cfg
):
    n_samples = cfg["mc_dropout"]["n_samples"]
    all_mc_probs, all_pidx, all_y = [], [], []
    sample_idx = 0
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        _, mc_probs = model.mc_predict_proba(X_batch, n_samples=n_samples)
        all_mc_probs.append(mc_probs.cpu())
        all_y.extend(y_batch.numpy())
        n_b = len(y_batch)
        all_pidx.extend(patient_index[sample_idx:sample_idx + n_b])
        sample_idx += n_b
    mc_probs_all = torch.cat(all_mc_probs, dim=1)
    all_pidx, all_y = np.array(all_pidx), np.array(all_y)
    spec_rel = compute_spectrum_reliability(mc_probs_all)
    return aggregate_to_patient(spec_rel, all_pidx, patient_uids, all_y, patient_mask)


# ── Calibration ─────────────────────────────────────────────────────────


def calibrate_on_val(model, val_loader, patient_index, patient_uids, labels,
                     val_patient_mask, device, cfg):
    if cfg["calibration"]["method"] != "temperature_scaling":
        return None
    model.eval()
    all_logits, all_pidx, all_y = [], [], []
    sample_idx = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch).cpu()
            all_logits.append(logits)
            all_y.extend(y_batch.numpy())
            n_b = len(y_batch)
            all_pidx.extend(patient_index[sample_idx:sample_idx + n_b])
            sample_idx += n_b
    logits_all = torch.cat(all_logits, dim=0)
    all_pidx, all_y = np.array(all_pidx), np.array(all_y)
    target_pids = np.where(val_patient_mask)[0]
    p_logits = torch.stack([logits_all[all_pidx == pid].mean(dim=0) for pid in target_pids])
    p_labels = torch.tensor([all_y[all_pidx == pid][0] for pid in target_pids])
    scaler = TemperatureScaling()
    T = scaler.fit(p_logits, p_labels)
    print(f"  Temperature (T): {T:.4f}")
    return scaler


# ── Full evaluation ─────────────────────────────────────────────────────


def full_evaluation(model, dataloader, patient_index, patient_uids, labels,
                    patient_mask, device, cfg, temp_scaler):
    if temp_scaler is not None:
        pat_rel = _mc_patient_inference_calibrated(
            model, dataloader, patient_index, patient_uids, labels,
            patient_mask, device, cfg, temp_scaler)
        calibrated = True
    else:
        pat_rel = mc_patient_inference(
            model, dataloader, patient_index, patient_uids, labels,
            patient_mask, device, cfg)
        calibrated = False

    clin_conf = compute_clinical_confidence(
        pat_rel,
        weights=(cfg["clinical"]["prob_margin_weight"],
                 cfg["clinical"]["entropy_norm_weight"],
                 cfg["clinical"]["mi_norm_weight"],
                 cfg["clinical"]["patient_agreement_weight"]),
        high_thresh=cfg["clinical"]["high_confidence_threshold"],
        med_thresh=cfg["clinical"]["medium_confidence_threshold"],
    )
    recommendations = clinical_decision(pat_rel, clin_conf)

    y_true = pat_rel["true_label"]
    y_prob = pat_rel["prob_positive"]
    metrics = compute_metrics(y_true, y_prob)

    from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
    from baseline_utils import sensitivity_score, specificity_score

    metric_fns = {
        "roc_auc": lambda yt, yp: roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else float("nan"),
        "accuracy": lambda yt, yp: accuracy_score(yt, (yp >= 0.5).astype(int)),
        "sensitivity": lambda yt, yp: sensitivity_score(yt, (yp >= 0.5).astype(int)),
        "specificity": lambda yt, yp: specificity_score(yt, (yp >= 0.5).astype(int)),
        "brier_score": brier_score_loss,
        "ece": expected_calibration_error,
    }
    metrics_ci = {}
    for m in cfg["evaluation"]["metrics"]:
        if m in metric_fns:
            metrics_ci[m] = bootstrap_metric_ci(
                y_true, y_prob, metric_fns[m],
                n_bootstrap=cfg["evaluation"]["n_bootstrap"],
                alpha=cfg["evaluation"]["ci_alpha"], seed=cfg["seed"],
                metric_name=m,
            )

    conf_metrics = {}
    for group in ["high", "medium", "low"]:
        mask = clin_conf["confidence_group"] == group
        if mask.sum() > 0 and len(np.unique(y_true[mask])) > 1:
            conf_metrics[group] = compute_metrics(y_true[mask], y_prob[mask])
            conf_metrics[group]["n_patients"] = int(mask.sum())
        else:
            conf_metrics[group] = {"n_patients": int(mask.sum())}

    calib_data = compute_calibration_curve_data(y_true, y_prob)

    return {
        "patient_rel": pat_rel, "clinical_conf": clin_conf,
        "recommendations": recommendations,
        "metrics": metrics, "metrics_ci": metrics_ci,
        "conf_metrics": conf_metrics, "calib_data": calib_data,
        "calibrated": calibrated,
    }


# ── Main training ───────────────────────────────────────────────────────


def train_v2(cfg, data, device, results_dir):
    """Train CC-SERSNet-v2 with full-batch SGD."""
    patient_masks = build_split_masks(data["patient_uids"], data["splits"])
    spec_masks = build_spectrum_masks(data["patient_index"], patient_masks)

    X = data["X_spectra"]
    y = data["labels"]

    # Full-batch: entire datasets as single tensors
    X_train_t = torch.as_tensor(X[spec_masks["train"]], dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.as_tensor(y[spec_masks["train"]], dtype=torch.long)
    X_val_t = torch.as_tensor(X[spec_masks["val"]], dtype=torch.float32).unsqueeze(1)
    y_val_t = torch.as_tensor(y[spec_masks["val"]], dtype=torch.long)
    X_test_t = torch.as_tensor(X[spec_masks["test"]], dtype=torch.float32).unsqueeze(1)

    # Dataloaders for val/test (needed for AUC computation and MC inference)
    batch_size = cfg["training"].get("batch_size", 0) or len(X_train_t)
    val_loader = DataLoader(SpectrumDataset(X[spec_masks["val"]], y[spec_masks["val"]]),
                            batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SpectrumDataset(X[spec_masks["test"]], y[spec_masks["test"]]),
                             batch_size=batch_size, shuffle=False)

    # ── Build model ──────────────────────────────────────
    m_cfg = cfg["model"]
    model = MultiScaleEncoder(
        in_channels=m_cfg["in_channels"], n_wavenumber=m_cfg["n_wavenumber"],
        n_classes=m_cfg["n_classes"], kernel_sizes=m_cfg["kernel_sizes"],
        base_channels=m_cfg["base_channels"], n_res_blocks=m_cfg["n_res_blocks"],
        dropout_rate=m_cfg["dropout_rate"],
        group_norm_groups=m_cfg["group_norm_groups"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # ── Optimizer ─────────────────────────────────────────
    t_cfg = cfg["training"]
    optimizer = torch.optim.SGD(
        model.parameters(), lr=t_cfg["learning_rate"],
        momentum=t_cfg["momentum"], weight_decay=t_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_cfg["scheduler_T_max"],
        eta_min=t_cfg["scheduler_eta_min"],
    )

    # ── TensorBoard ───────────────────────────────────────
    tb_dir = results_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    # ── Training loop ─────────────────────────────────────
    val_pidx = data["patient_index"][spec_masks["val"]]
    train_log = []
    best_auc = -1.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    print(f"  Full-batch SGD: {len(X_train_t)} spectra/step, 1 update/epoch")
    print(f"  weight_decay={t_cfg['weight_decay']}, momentum={t_cfg['momentum']}")

    for epoch in range(1, t_cfg["epochs"] + 1):
        t0 = time.time()

        tr_loss = full_batch_train_epoch(
            model, X_train_t, y_train_t, optimizer, device, writer, epoch)
        v_loss = full_batch_val_loss(model, X_val_t, y_val_t, device)

        # Patient AUC (every 5 epochs to save time)
        if epoch % 5 == 0 or epoch <= 10:
            v_auc = _patient_auc_from_model(model, val_loader, val_pidx,
                                             patient_masks["val"], device)
        else:
            v_auc = float("nan") if epoch > 1 else _patient_auc_from_model(
                model, val_loader, val_pidx, patient_masks["val"], device)

        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        # TensorBoard
        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("Loss/val", v_loss, epoch)
        writer.add_scalar("Loss/gap", tr_loss - v_loss, epoch)
        if not np.isnan(v_auc):
            writer.add_scalar("Metrics/val_patient_auc", v_auc, epoch)
        writer.add_scalar("Params/lr", cur_lr, epoch)

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
        elif not np.isnan(v_auc):
            patience_counter += 1
        # If v_auc is nan (epochs where we skip AUC), don't count toward patience

        if patience_counter >= t_cfg["early_stopping_patience"] and not np.isnan(v_auc):
            print(f"  Early stopping at epoch {epoch}")
            break

    writer.close()

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  Best epoch: {best_epoch}, best val AUC: {best_auc:.4f}")

    # Save model
    model_path = results_dir / "model.pt"
    torch.save({"model_state_dict": best_state, "best_epoch": best_epoch,
                "best_val_auc": best_auc, "config": cfg}, model_path)

    # Save log
    log_path = results_dir / "training_log.csv"
    with open(log_path, "w", encoding="utf-8", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss",
                                                    "val_patient_auc", "lr"])
        writer_csv.writeheader()
        writer_csv.writerows(train_log)

    # ── Calibrate ─────────────────────────────────────────
    print("\nCalibrating...")
    temp_scaler = calibrate_on_val(
        model, val_loader, data["patient_index"][spec_masks["val"]],
        data["patient_uids"], data["labels"], patient_masks["val"], device, cfg,
    )

    # ── Evaluate ──────────────────────────────────────────
    print("Evaluating on test set...")
    eval_result = full_evaluation(
        model, test_loader,
        data["patient_index"][spec_masks["test"]],
        data["patient_uids"], data["labels"],
        patient_masks["test"], device, cfg, temp_scaler,
    )

    return eval_result, train_log, best_epoch, best_auc, n_params, temp_scaler


def write_outputs(cfg, eval_result, train_log, best_epoch, best_auc, n_params, results_dir):
    """Write all output files."""
    from baseline_utils import toolbox_root, resolve_path

    # Predictions CSV
    pr = eval_result["patient_rel"]
    cc = eval_result["clinical_conf"]
    recs = eval_result["recommendations"]
    n = len(pr["patient_uid"])

    pred_path = results_dir / "predictions.csv"
    with open(pred_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "patient_uid", "true_label", "pred_label",
            "prob_negative", "prob_positive",
            "predictive_entropy", "expected_entropy", "mutual_information",
            "margin", "patient_agreement", "prob_variance",
            "clinical_confidence", "confidence_group",
            "recommendation", "correct", "split",
        ])
        for i in range(n):
            correct = int(pr["true_label"][i] == pr["pred_class"][i])
            writer.writerow([
                pr["patient_uid"][i], int(pr["true_label"][i]), int(pr["pred_class"][i]),
                round(float(pr["prob_negative"][i]), 6), round(float(pr["prob_positive"][i]), 6),
                round(float(pr["entropy_mean"][i]), 6),
                round(float(pr["expected_entropy_mean"][i]), 6),
                round(float(pr["mi_mean"][i]), 6),
                round(float(pr["margin_mean"][i]), 6),
                round(float(pr["patient_agreement"][i]), 6),
                round(float(pr["prob_variance"][i]), 6),
                round(float(cc["clinical_confidence"][i]), 6),
                cc["confidence_group"][i], recs[i], correct, "test",
            ])

    # Reliability CSV
    rel_path = results_dir / "reliability_patient.csv"
    with open(rel_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "patient_uid", "true_label", "pred_label",
            "prob_positive", "prob_variance", "patient_agreement",
            "entropy_mean", "mi_mean", "margin_mean",
            "prob_margin_raw", "entropy_norm", "mi_norm",
            "clinical_confidence", "confidence_group", "recommendation",
        ])
        for i in range(n):
            writer.writerow([
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

    # Metrics JSON
    metrics_out = {
        "model": "CC-SERSNet-v2", "seed": cfg["seed"],
        "best_val_auc": best_auc, "best_epoch": best_epoch,
        "n_parameters": n_params,
        "calibrated": eval_result["calibrated"],
        "test_metrics": {k: v for k, v in eval_result["metrics_ci"].items()},
    }
    write_json(results_dir / "metrics.json", metrics_out)

    # ── Figures ────────────────────────────────────────────
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Training curve: loss + AUC + LR + grad norm combined
    _plot_v2_dashboard(train_log, fig_dir / "training_dashboard.png")

    # Calibration
    calib = eval_result["calib_data"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(calib["prob_pred"], calib["prob_true"], "o-", color="#1f77b4",
            linewidth=1.5, markersize=6, label="CC-SERSNet-v2")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Predicted Probability"); ax.set_ylabel("Observed Frequency")
    ax.set_title("Calibration Curve (Test Set)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "calibration_curve.png", dpi=300)
    plt.close(fig)

    print(f"Outputs saved to {results_dir}")


def _plot_v2_dashboard(log, save_path):
    """Training dashboard for v2."""
    epochs = [r["epoch"] for r in log]
    tr_loss = [r["train_loss"] for r in log]
    v_loss = [r["val_loss"] for r in log]
    v_auc = [r.get("val_patient_auc") or float("nan") for r in log]
    lrs = [r["lr"] for r in log]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, tr_loss, linewidth=1.2, label="Train")
    ax.plot(epochs, v_loss, linewidth=1.2, label="Val")
    best_loss_idx = np.argmin(v_loss)
    ax.axvline(epochs[best_loss_idx], color="red", linestyle="--", alpha=0.4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Full-Batch SGD Loss (1 update/epoch)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # AUC
    ax = axes[0, 1]
    valid = ~np.isnan(v_auc)
    ax.plot(np.array(epochs)[valid], np.array(v_auc)[valid], "o-",
            markersize=3, linewidth=1.2, color="#2ca02c")
    best_auc_idx = np.nanargmax(v_auc)
    ax.axvline(epochs[best_auc_idx], color="red", linestyle="--", alpha=0.4)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("AUC")
    ax.set_title(f"Val Patient AUC (best={v_auc[best_auc_idx]:.4f})")
    ax.set_ylim(0.4, 1.05); ax.grid(True, alpha=0.3)

    # LR
    ax = axes[1, 0]
    ax.plot(epochs, lrs, linewidth=1.2, color="#9467bd")
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
    ax.set_title("Cosine Annealing LR Schedule")
    ax.grid(True, alpha=0.3)

    # Loss gap
    ax = axes[1, 1]
    gap = np.array(tr_loss) - np.array(v_loss)
    ax.fill_between(epochs, 0, gap, alpha=0.3, color="gray")
    ax.plot(epochs, gap, linewidth=1.2, color="purple")
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Train-Val Gap")
    ax.set_title("Overfitting Signal")
    ax.grid(True, alpha=0.3)

    fig.suptitle("CC-SERSNet v2 Training (Full-Batch SGD, 10K params)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── LOPO-CV mode ────────────────────────────────────────────────────────


def run_lopo_cv(cfg, data, device, results_dir):
    """Leave-One-Patient-Out cross-validation.

    For each of the 52 patients:
      - Train on the other 51 patients
      - Evaluate on the held-out patient
    Aggregate all 52 predictions for an unbiased estimate.
    """
    patient_uids = data["patient_uids"]
    n_patients = len(patient_uids)
    patient_labels_arr = np.array([
        data["labels"][data["patient_index"] == i][0] for i in range(n_patients)
    ])

    X_all = torch.as_tensor(data["X_spectra"], dtype=torch.float32).unsqueeze(1)
    y_all = torch.as_tensor(data["labels"], dtype=torch.long)
    p_idx_all = data["patient_index"]

    all_predictions = []
    aucs_per_fold = []

    print(f"\n{'='*60}")
    print(f"LOPO-CV: {n_patients} folds (1 patient held out each)")
    print(f"{'='*60}")

    for test_pid in range(n_patients):
        t0 = time.time()

        # Masks
        test_p_mask = p_idx_all == test_pid
        train_p_mask = ~test_p_mask

        X_tr = X_all[train_p_mask]
        y_tr = y_all[train_p_mask]
        X_te = X_all[test_p_mask]
        n_te = test_p_mask.sum()

        # Build model
        m_cfg = cfg["model"]
        model = MultiScaleEncoder(
            in_channels=m_cfg["in_channels"], n_wavenumber=m_cfg["n_wavenumber"],
            n_classes=m_cfg["n_classes"], kernel_sizes=m_cfg["kernel_sizes"],
            base_channels=m_cfg["base_channels"], n_res_blocks=m_cfg["n_res_blocks"],
            dropout_rate=m_cfg["dropout_rate"],
            group_norm_groups=m_cfg["group_norm_groups"],
        ).to(device)

        t_cfg = cfg["training"]
        optimizer = torch.optim.SGD(
            model.parameters(), lr=t_cfg["learning_rate"],
            momentum=t_cfg["momentum"], weight_decay=t_cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=t_cfg["scheduler_eta_min"],
        )

        # Train (fewer epochs for LOPO-CV speed)
        for epoch in range(1, 101):
            model.train()
            optimizer.zero_grad()
            logits = model(X_tr.to(device))
            loss = F.cross_entropy(logits, y_tr.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()

        # MC Dropout predict on held-out patient
        model.eval()
        mc_probs_list = []
        with torch.no_grad():
            for _ in range(cfg["mc_dropout"]["n_samples"]):
                model.train()  # enable dropout
                logits_te = model(X_te.to(device))
                mc_probs_list.append(F.softmax(logits_te, dim=1).cpu().unsqueeze(0))
                model.eval()
        mc_probs = torch.cat(mc_probs_list, dim=0).mean(dim=0)  # [N_te, 2]
        p_pos = mc_probs[:, 1].mean().item()  # patient mean

        true_label = int(patient_labels_arr[test_pid])
        pred_label = 1 if p_pos >= 0.5 else 0
        correct = int(true_label == pred_label)

        all_predictions.append({
            "patient_uid": patient_uids[test_pid],
            "true_label": true_label,
            "pred_label": pred_label,
            "prob_positive": p_pos,
            "correct": correct,
            "n_spectra": int(test_p_mask.sum()),
        })

        elapsed = time.time() - t0
        if (test_pid + 1) % 10 == 0 or test_pid == 0:
            # Compute running AUC
            y_true_so_far = np.array([p["true_label"] for p in all_predictions])
            y_prob_so_far = np.array([p["prob_positive"] for p in all_predictions])
            if len(np.unique(y_true_so_far)) > 1:
                from sklearn.metrics import roc_auc_score
                running_auc = roc_auc_score(y_true_so_far, y_prob_so_far)
                aucs_per_fold.append(running_auc)
            print(f"  Fold {test_pid+1:2d}/{n_patients}: "
                  f"true={true_label} pred={pred_label} P={p_pos:.4f} "
                  f"correct={correct} n_spec={n_te} ({elapsed:.1f}s)")

    # ── Aggregate LOPO-CV metrics ─────────────────────────
    y_true_all = np.array([p["true_label"] for p in all_predictions])
    y_prob_all = np.array([p["prob_positive"] for p in all_predictions])
    y_pred_all = np.array([p["pred_label"] for p in all_predictions])

    from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
    from baseline_utils import sensitivity_score, specificity_score

    cv_metrics = {
        "n_patients": n_patients,
        "roc_auc": roc_auc_score(y_true_all, y_prob_all),
        "accuracy": accuracy_score(y_true_all, y_pred_all),
        "sensitivity": sensitivity_score(y_true_all, y_pred_all),
        "specificity": specificity_score(y_true_all, y_pred_all),
        "brier_score": brier_score_loss(y_true_all, y_prob_all),
    }

    print(f"\n{'='*60}")
    print("LOPO-CV Results (52 patients, unbiased estimate)")
    print(f"{'='*60}")
    for k, v in cv_metrics.items():
        print(f"  {k:15s}: {v:.4f}")

    # Save
    cv_path = results_dir / "lopo_cv_predictions.csv"
    with open(cv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_uid", "true_label", "pred_label",
                          "prob_positive", "correct", "n_spectra"])
        for p in all_predictions:
            writer.writerow([p["patient_uid"], p["true_label"], p["pred_label"],
                              round(p["prob_positive"], 6), p["correct"], p["n_spectra"]])
    write_json(results_dir / "lopo_cv_metrics.json", cv_metrics)
    print(f"\nLOPO-CV saved: {cv_path}")

    return cv_metrics, all_predictions


# ── Entry point ──────────────────────────────────────────────────────────


def main():
    cfg = load_config_v2()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from baseline_utils import toolbox_root, resolve_path
    results_dir = resolve_path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Loading dataset...")
    data = load_phase4_dataset(cfg)
    audit_splits(data)

    # ── Standard train/val/test ────────────────────────────
    print("\n[2/5] Training CC-SERSNet-v2 (full-batch SGD)...")
    eval_result, train_log, best_epoch, best_auc, n_params, temp_scaler = train_v2(
        cfg, data, device, results_dir)

    print("\n[3/5] Writing standard outputs...")
    write_outputs(cfg, eval_result, train_log, best_epoch, best_auc, n_params, results_dir)

    # ── LOPO-CV ────────────────────────────────────────────
    print("\n[4/5] Running LOPO-CV (52 folds)...")
    cv_metrics, cv_predictions = run_lopo_cv(cfg, data, device, results_dir)

    # ── Summary ────────────────────────────────────────────
    m = eval_result["metrics_ci"]
    print(f"\n[5/5] Final Summary")
    print(f"{'='*60}")
    print(f"Standard split (seed=42):")
    print(f"  AUC={m['roc_auc']['value']:.4f} [{m['roc_auc']['ci_lower']:.4f}, {m['roc_auc']['ci_upper']:.4f}]")
    print(f"  Acc={m['accuracy']['value']:.4f}  Sens={m['sensitivity']['value']:.4f}  Spec={m['specificity']['value']:.4f}")
    print(f"LOPO-CV (52 folds, unbiased):")
    print(f"  AUC={cv_metrics['roc_auc']:.4f}")
    print(f"  Acc={cv_metrics['accuracy']:.4f}  Sens={cv_metrics['sensitivity']:.4f}  Spec={cv_metrics['specificity']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
