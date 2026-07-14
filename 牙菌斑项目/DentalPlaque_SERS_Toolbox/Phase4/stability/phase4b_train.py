r"""Phase 4B: Single-split training & evaluation for one experiment config.

Usage (from toolbox root):
  python Phase4/stability/phase4b_train.py --exp B0 --split_seed 42

Called by phase4b_run.py for all experiment × split combinations.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import warnings
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

# Add project paths
_toolbox = Path(__file__).resolve().parents[2]
_phase3 = _toolbox / "Phase3" / "baseline"
_phase4dl = _toolbox / "Phase4" / "deep_learning"
sys.path.insert(0, str(_phase3))
sys.path.insert(0, str(_phase4dl))
sys.path.insert(0, str(_toolbox / "Phase4" / "stability"))

from baseline_utils import binomial_ci, expected_calibration_error, write_json
from datasets import SpectrumDataset, load_phase4_dataset
from models import MultiScaleEncoder
from phase4b_utils import (
    aggregate_patient_probs,
    balanced_accuracy_score,
    compute_patient_balanced_weights,
    compute_patient_class_balanced_weights,
    create_patient_split,
    optimize_threshold,
)
from reliability import (
    TemperatureScaling,
    compute_spectrum_reliability,
)

warnings.filterwarnings("ignore")


# ── Config ──────────────────────────────────────────────────────────────────


def load_phase4b_config() -> dict:
    cfg_path = Path(__file__).resolve().parent / "phase4b_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Loss function ──────────────────────────────────────────────────────────


def compute_loss_weights(
    patient_index: np.ndarray,
    labels: np.ndarray,
    exp_cfg: dict,
) -> np.ndarray | None:
    """Compute per-sample weights based on experiment config."""
    has_class_balance = exp_cfg.get("class_balance_weight", False)
    has_label_smooth = exp_cfg.get("label_smoothing", 0.0) > 0

    if has_class_balance:
        weights = compute_patient_class_balanced_weights(patient_index, labels)
    else:
        weights = compute_patient_balanced_weights(patient_index)

    return weights


def train_epoch_weighted(model, dataloader, optimizer, device, label_smoothing=0.0):
    """Train one epoch with per-sample weights."""
    model.train()
    total_loss = 0.0
    total_weight = 0.0

    for X_batch, y_batch, w_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        w_batch = w_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)

        if label_smoothing > 0:
            # Label smoothing: soft targets
            n_classes = logits.size(1)
            smooth_targets = torch.full_like(logits, label_smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, y_batch.unsqueeze(1), 1.0 - label_smoothing)
            log_probs = F.log_softmax(logits, dim=1)
            loss_per_sample = -(smooth_targets * log_probs).sum(dim=1)
        else:
            loss_per_sample = F.cross_entropy(logits, y_batch, reduction="none")

        loss = (loss_per_sample * w_batch).sum() / w_batch.sum()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
        total_weight += len(X_batch)

    return total_loss / total_weight if total_weight > 0 else 0.0


@torch.no_grad()
def val_loss_weighted(model, dataloader, device):
    """Validation loss (unweighted)."""
    model.eval()
    total = 0.0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        total += F.cross_entropy(logits, y_batch, reduction="sum").item()
    return total / len(dataloader.dataset)


# ── Patient-level AUC on val set ───────────────────────────────────────────


def patient_auc_from_model(model, dataloader, patient_index, patient_mask, device):
    """Compute patient-level AUC by aggregating spectrum probs (mean) per patient."""
    from sklearn.metrics import roc_auc_score

    model.eval()
    all_probs = []
    all_pidx = []
    all_y = []

    sample_idx = 0
    with torch.no_grad():
        for batch in dataloader:
            X_batch = batch[0].to(device)
            probs = F.softmax(model(X_batch), dim=1)[:, 1].cpu().numpy()
            n = len(probs)
            all_probs.extend(probs)
            all_y.extend(batch[1].numpy())
            all_pidx.extend(patient_index[sample_idx : sample_idx + n])
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


# ── Patient-level metrics on val set (for model selection) ─────────────────


def compute_val_patient_metrics(
    model, dataloader, patient_index, patient_mask, device, agg_method="mean"
):
    """Compute patient-level metrics on validation set.

    Returns dict with: prob_positive, true_label, pred_class (at 0.5), auc, bal_acc, sens, spec.
    """
    from sklearn.metrics import roc_auc_score

    model.eval()
    all_probs = []
    all_pidx = []
    all_y = []

    sample_idx = 0
    with torch.no_grad():
        for batch in dataloader:
            X_batch = batch[0].to(device)
            probs = F.softmax(model(X_batch), dim=1)[:, 1].cpu().numpy()
            n = len(probs)
            all_probs.extend(probs)
            all_y.extend(batch[1].numpy())
            all_pidx.extend(patient_index[sample_idx : sample_idx + n])
            sample_idx += n

    all_probs = np.array(all_probs)
    all_y = np.array(all_y)
    all_pidx = np.array(all_pidx)

    target_pids = np.where(patient_mask)[0]
    p_probs = aggregate_patient_probs(all_probs, all_pidx, method=agg_method)
    # Match p_probs to target_pids order
    unique_all = np.unique(all_pidx)
    pid_to_pos = {pid: i for i, pid in enumerate(unique_all)}
    p_probs_ordered = np.array([p_probs[pid_to_pos[pid]] for pid in target_pids])
    p_labels = np.array([all_y[all_pidx == pid][0] for pid in target_pids])

    # AUC
    auc = roc_auc_score(p_labels, p_probs_ordered) if len(np.unique(p_labels)) >= 2 else float("nan")

    # Threshold-0.5 metrics
    p_preds_05 = (p_probs_ordered >= 0.5).astype(int)
    bal_acc = balanced_accuracy_score(p_labels, p_preds_05)
    pos_mask = p_labels == 1
    neg_mask = p_labels == 0
    sens = p_preds_05[pos_mask].mean() if pos_mask.sum() > 0 else float("nan")
    spec = (1.0 - p_preds_05[neg_mask]).mean() if neg_mask.sum() > 0 else float("nan")

    return {
        "prob_positive": p_probs_ordered,
        "true_label": p_labels,
        "pred_class_05": p_preds_05,
        "auc": float(auc) if not np.isnan(auc) else None,
        "balanced_accuracy": float(bal_acc) if not np.isnan(bal_acc) else None,
        "sensitivity": float(sens) if not np.isnan(sens) else None,
        "specificity": float(spec) if not np.isnan(spec) else None,
    }


def mc_val_inference(
    model, dataloader, patient_index, patient_mask, device, cfg, temp_scaler,
    agg_method="mean",
):
    """MC Dropout inference on validation set — same pipeline as test.

    Uses MC Dropout + temperature scaling + patient aggregation.
    Returns patient-level prob_positive, true_label, pred_class_05.
    """
    n_samples = cfg["mc_dropout"]["n_samples"]

    all_mc_probs = []
    all_pidx = []
    all_y = []

    sample_idx = 0
    for batch in dataloader:
        X_batch = batch[0].to(device)
        mc_logits = model.mc_dropout_sample(X_batch, n_samples=n_samples).detach()
        if temp_scaler is not None:
            mc_logits = mc_logits / temp_scaler.temperature.detach().to(mc_logits.device)
        mc_probs = F.softmax(mc_logits, dim=-1)
        all_mc_probs.append(mc_probs.cpu())
        all_y.extend(batch[1].numpy())
        n_b = len(batch[1])
        all_pidx.extend(patient_index[sample_idx : sample_idx + n_b])
        sample_idx += n_b

    mc_probs_all = torch.cat(all_mc_probs, dim=1)
    mean_probs = mc_probs_all.mean(dim=0)[:, 1].cpu().numpy()
    all_pidx = np.array(all_pidx)
    all_y = np.array(all_y)

    target_pids = np.where(patient_mask)[0]
    p_probs = aggregate_patient_probs(mean_probs, all_pidx, method=agg_method)
    unique_all = np.unique(all_pidx)
    pid_to_pos = {pid: i for i, pid in enumerate(unique_all)}
    p_probs_ordered = np.array([p_probs[pid_to_pos[pid]] for pid in target_pids])
    p_labels = np.array([all_y[all_pidx == pid][0] for pid in target_pids])
    p_preds_05 = (p_probs_ordered >= 0.5).astype(int)

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(p_labels, p_probs_ordered) if len(np.unique(p_labels)) >= 2 else float("nan")
    bal_acc = balanced_accuracy_score(p_labels, p_preds_05)
    pos_mask = p_labels == 1
    neg_mask = p_labels == 0
    sens = p_preds_05[pos_mask].mean() if pos_mask.sum() > 0 else float("nan")
    spec = (1.0 - p_preds_05[neg_mask]).mean() if neg_mask.sum() > 0 else float("nan")

    return {
        "prob_positive": p_probs_ordered,
        "true_label": p_labels,
        "pred_class_05": p_preds_05,
        "auc": float(auc) if not np.isnan(auc) else None,
        "balanced_accuracy": float(bal_acc) if not np.isnan(bal_acc) else None,
        "sensitivity": float(sens) if not np.isnan(sens) else None,
        "specificity": float(spec) if not np.isnan(spec) else None,
    }


# ── MC Dropout inference on test ───────────────────────────────────────────


def mc_test_inference(
    model, dataloader, patient_index, patient_mask, device, cfg, temp_scaler,
    agg_method="mean", threshold=0.5,
):
    """MC Dropout inference → patient aggregation → thresholded predictions."""
    n_samples = cfg["mc_dropout"]["n_samples"]

    all_mc_probs = []
    all_pidx = []
    all_y = []

    sample_idx = 0
    for batch in dataloader:
        X_batch = batch[0].to(device)
        # MC posterior with temperature scaling
        mc_logits = model.mc_dropout_sample(X_batch, n_samples=n_samples).detach()
        if temp_scaler is not None:
            mc_logits = mc_logits / temp_scaler.temperature.detach().to(mc_logits.device)
        mc_probs = F.softmax(mc_logits, dim=-1)  # [T, B, C]
        all_mc_probs.append(mc_probs.cpu())
        all_y.extend(batch[1].numpy())
        n_b = len(batch[1])
        all_pidx.extend(patient_index[sample_idx : sample_idx + n_b])
        sample_idx += n_b

    mc_probs_all = torch.cat(all_mc_probs, dim=1)  # [T, N, C]
    mean_probs = mc_probs_all.mean(dim=0)[:, 1].cpu().numpy()  # [N] — p(class=1)
    all_pidx = np.array(all_pidx)
    all_y = np.array(all_y)

    # Patient aggregation
    target_pids = np.where(patient_mask)[0]
    p_probs = aggregate_patient_probs(mean_probs, all_pidx, method=agg_method)
    unique_all = np.unique(all_pidx)
    pid_to_pos = {pid: i for i, pid in enumerate(unique_all)}
    p_probs_ordered = np.array([p_probs[pid_to_pos[pid]] for pid in target_pids])
    p_labels = np.array([all_y[all_pidx == pid][0] for pid in target_pids])

    # Apply threshold
    p_preds = (p_probs_ordered >= threshold).astype(int)

    # Spectrum reliability for MC uncertainty
    spec_rel = compute_spectrum_reliability(mc_probs_all)

    return {
        "prob_positive": p_probs_ordered,
        "true_label": p_labels,
        "pred_label": p_preds,
        "spec_rel": spec_rel,
        "all_pidx_test": all_pidx,
    }


# ── Fit temperature on val MC posterior ────────────────────────────────────


def calibrate_on_val_mc(model, val_loader, patient_index, val_patient_mask, device, cfg):
    """Fit temperature scaling on val MC posterior patient-level mean logits."""
    n_samples = cfg["mc_dropout"]["n_samples"]

    all_mc_logits = []
    all_pidx = []
    all_y = []

    sample_idx = 0
    for batch in val_loader:
        X_batch = batch[0].to(device)
        mc_logits = model.mc_dropout_sample(X_batch, n_samples=n_samples).detach()
        mean_logits = mc_logits.mean(dim=0).cpu()
        all_mc_logits.append(mean_logits)
        all_y.extend(batch[1].numpy())
        n_b = len(batch[1])
        all_pidx.extend(patient_index[sample_idx : sample_idx + n_b])
        sample_idx += n_b

    logits_all = torch.cat(all_mc_logits, dim=0)
    all_pidx = np.array(all_pidx)
    all_y = np.array(all_y)

    target_pids = np.where(val_patient_mask)[0]
    p_logits = torch.stack([
        logits_all[all_pidx == pid].mean(dim=0) for pid in target_pids
    ])
    p_labels = torch.tensor([all_y[all_pidx == pid][0] for pid in target_pids])

    scaler = TemperatureScaling()
    T = scaler.fit(p_logits, p_labels)
    return scaler, float(T)


# ── Main: train & evaluate one split ───────────────────────────────────────


def run_one_split(cfg: dict, exp_id: str, split_seed: int) -> dict:
    """Train and evaluate one experiment on one split. Returns results dict."""
    exp_cfg = cfg["experiments"][exp_id]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reproducibility
    torch.manual_seed(split_seed)
    np.random.seed(split_seed)

    # ── Load data ──────────────────────────────────────
    data = load_phase4_dataset({"paths": cfg["paths"]})
    n_patients = len(data["patient_uids"])

    # ── Create split ───────────────────────────────────
    split = create_patient_split(
        np.array([data["labels"][data["patient_index"] == i][0] for i in range(n_patients)]),
        seed=split_seed,
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        test_ratio=cfg["split"]["test_ratio"],
        stratify=cfg["split"]["stratify"],
    )

    # Build boolean masks
    train_mask = np.zeros(n_patients, dtype=bool)
    val_mask = np.zeros(n_patients, dtype=bool)
    test_mask = np.zeros(n_patients, dtype=bool)
    train_mask[split["train_patients"]] = True
    val_mask[split["val_patients"]] = True
    test_mask[split["test_patients"]] = True

    # Spectrum-level masks
    pidx_all = data["patient_index"]
    train_spec = train_mask[pidx_all]
    val_spec = val_mask[pidx_all]
    test_spec = test_mask[pidx_all]

    X = data["X_spectra"]
    y = data["labels"]

    # ── Compute sample weights ─────────────────────────
    train_weights = compute_loss_weights(
        pidx_all[train_spec], y[train_spec], exp_cfg
    )

    # ── Build datasets ─────────────────────────────────
    train_ds = SpectrumDataset(X[train_spec], y[train_spec], train_weights)
    val_ds = SpectrumDataset(X[val_spec], y[val_spec])
    test_ds = SpectrumDataset(X[test_spec], y[test_spec])

    batch_size = cfg["training"]["batch_size"]
    g = torch.Generator().manual_seed(split_seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # ── Build model ────────────────────────────────────
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

    # ── Train ──────────────────────────────────────────
    t_cfg = cfg["training"]
    label_smooth = exp_cfg.get("label_smoothing", 0.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t_cfg["learning_rate"], weight_decay=t_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=t_cfg["scheduler_factor"],
        patience=t_cfg["scheduler_patience"],
    )

    selection_metric = exp_cfg["selection_metric"]
    selection_constraint = exp_cfg.get("selection_sens_constraint")
    agg_method = exp_cfg["aggregation"]

    best_score = -np.inf
    best_epoch = 0
    best_state = None
    patience_counter = 0
    train_log = []

    for epoch in range(1, t_cfg["epochs"] + 1):
        tr_loss = train_epoch_weighted(model, train_loader, optimizer, device, label_smooth)
        v_loss = val_loss_weighted(model, val_loader, device)

        # Compute val patient metrics using MC Dropout (same stochastic process as test).
        # No temperature scaling yet (fitted after training), but MC ensures the relative
        # ranking across epochs is consistent with the final evaluation pipeline.
        val_metrics = mc_val_inference(
            model, val_loader, pidx_all[val_spec], val_mask, device, cfg,
            temp_scaler=None, agg_method=agg_method,
        )

        # Determine score for model selection
        if selection_metric == "val_auc":
            score = val_metrics["auc"]
        elif selection_metric == "val_balanced_accuracy":
            score = val_metrics["balanced_accuracy"]
        elif selection_metric == "val_accuracy":
            score = float((val_metrics["pred_class_05"] == val_metrics["true_label"]).mean())
        else:
            score = val_metrics["auc"]

        # Check sensitivity constraint
        if selection_constraint is not None:
            sens = val_metrics.get("sensitivity")
            if sens is None or np.isnan(sens) or sens < selection_constraint:
                score = -np.inf  # disqualify this epoch

        scheduler.step(score if score is not None and not np.isnan(score) else 0.0)

        train_log.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": v_loss,
            "val_patient_auc": val_metrics["auc"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
        })

        if score is not None and not np.isnan(score) and score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= t_cfg["early_stopping_patience"]:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Calibrate on val MC posterior ──────────────────
    temp_scaler, temperature = calibrate_on_val_mc(
        model, val_loader, pidx_all[val_spec], val_mask, device, cfg
    )

    # ── Optimize threshold on val MC-calibrated probs ───
    # MUST use MC + temperature scaling (same pipeline as test).
    val_final = mc_val_inference(
        model, val_loader, pidx_all[val_spec], val_mask, device, cfg,
        temp_scaler, agg_method=agg_method,
    )
    thresh_result = optimize_threshold(
        val_final["true_label"],
        val_final["prob_positive"],
        strategy=exp_cfg["threshold_strategy"],
        sens_constraint=exp_cfg.get("threshold_sens_constraint"),
    )
    best_threshold = thresh_result["threshold"]

    # ── MC inference on test ───────────────────────────
    test_result = mc_test_inference(
        model, test_loader, pidx_all[test_spec], test_mask, device, cfg,
        temp_scaler, agg_method=agg_method, threshold=best_threshold,
    )

    # ── Compute test metrics ───────────────────────────
    from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

    yt = test_result["true_label"]
    yp = test_result["prob_positive"]
    ypred = test_result["pred_label"]

    n_pos = int(yt.sum())
    n_neg = int(len(yt) - n_pos)
    tp = int((ypred[yt == 1] == 1).sum())
    tn = int((ypred[yt == 0] == 0).sum())

    metrics = {}
    metrics["roc_auc"] = float(roc_auc_score(yt, yp)) if len(np.unique(yt)) >= 2 else float("nan")
    metrics["accuracy"] = float(accuracy_score(yt, ypred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(yt, ypred))
    # Binomial CIs
    sens_ci = binomial_ci(tp, n_pos)
    spec_ci = binomial_ci(tn, n_neg)
    metrics["sensitivity"] = sens_ci
    metrics["specificity"] = spec_ci
    metrics["brier_score"] = float(brier_score_loss(yt, yp))
    metrics["ece"] = float(expected_calibration_error(yt, yp))

    return {
        "exp_id": exp_id,
        "split_seed": split_seed,
        "best_epoch": best_epoch,
        "best_val_score": float(best_score) if not np.isnan(best_score) else None,
        "temperature": temperature,
        "threshold": best_threshold,
        "threshold_result": thresh_result,
        "test_metrics": metrics,
        "n_train_patients": int(train_mask.sum()),
        "n_val_patients": int(val_mask.sum()),
        "n_test_patients": int(test_mask.sum()),
        "n_train_spectra": int(train_spec.sum()),
        "n_val_spectra": int(val_spec.sum()),
        "n_test_spectra": int(test_spec.sum()),
    }


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="Experiment ID (B0-B4)")
    parser.add_argument("--split_seed", type=int, required=True, help="Split seed (0-19)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    cfg = load_phase4b_config()

    print(f"Phase4B: exp={args.exp} split_seed={args.split_seed}")
    t0 = time.time()
    result = run_one_split(cfg, args.exp, args.split_seed)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Test AUC={result['test_metrics']['roc_auc']:.4f}, "
          f"BalAcc={result['test_metrics']['balanced_accuracy']:.4f}, "
          f"Sens={result['test_metrics']['sensitivity']['value']:.4f}, "
          f"Spec={result['test_metrics']['specificity']['value']:.4f}")

    if args.output:
        write_json(Path(args.output), result)
        print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
