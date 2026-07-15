"""Phase 4B: Shared utilities for stability validation and ablation study.

Patient aggregation, threshold optimization, loss weight computation,
balanced accuracy, and patient-level split creation.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


# ── Balanced accuracy ──────────────────────────────────────────────────────


def balanced_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """(sensitivity + specificity) / 2 — handles edge cases."""
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    sens = y_pred[pos_mask].mean() if pos_mask.sum() > 0 else float("nan")
    spec = (1.0 - y_pred[neg_mask]).mean() if neg_mask.sum() > 0 else float("nan")

    if np.isnan(sens) and np.isnan(spec):
        return float("nan")
    if np.isnan(sens):
        return float(spec)
    if np.isnan(spec):
        return float(sens)
    return float((sens + spec) / 2.0)


# ── Threshold optimization ─────────────────────────────────────────────────


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy: str = "max_balanced_accuracy",
    sens_constraint: float | None = None,
    spec_constraint: float | None = None,
) -> dict:
    """Find optimal decision threshold on validation data.

    Args:
        y_true: ground-truth labels (0/1).
        y_prob: predicted probabilities for class 1.
        strategy: one of {"fixed_0.5", "max_accuracy", "max_balanced_accuracy",
                          "max_youden", "max_specificity"}.
        sens_constraint: if not None, only consider thresholds where sensitivity
                         >= sens_constraint.
        spec_constraint: if not None, only consider thresholds where specificity
                         >= spec_constraint.

    Returns:
        {threshold, sensitivity, specificity, balanced_accuracy, youden, accuracy}
    """
    if strategy == "fixed_0.5":
        thresh = 0.5
        y_pred = (y_prob >= thresh).astype(int)
        return _threshold_result(y_true, y_pred, thresh)

    # Search grid
    thresholds = np.linspace(0.05, 0.95, 91)

    best = None
    best_score = -np.inf

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        res = _threshold_result(y_true, y_pred, t)

        # Check sensitivity constraint
        if sens_constraint is not None and res["sensitivity"] is not None and res["sensitivity"] < sens_constraint:
            continue

        # Check specificity constraint
        if spec_constraint is not None and res["specificity"] is not None and res["specificity"] < spec_constraint:
            continue

        if strategy == "max_accuracy":
            score = res["accuracy"]
        elif strategy == "max_balanced_accuracy":
            score = res["balanced_accuracy"]
        elif strategy == "max_youden":
            score = res["youden"]
        elif strategy == "max_specificity":
            score = res["specificity"]
        else:
            raise ValueError(f"Unknown threshold strategy: {strategy}")

        if np.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best = res

    if best is None:
        # Fallback to 0.5
        y_pred = (y_prob >= 0.5).astype(int)
        return _threshold_result(y_true, y_pred, 0.5)

    return best


def _threshold_result(y_true, y_pred, threshold):
    """Compute all metrics for a given threshold."""
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    tp = int((y_pred[pos_mask] == 1).sum()) if pos_mask.sum() > 0 else 0
    tn = int((y_pred[neg_mask] == 0).sum()) if neg_mask.sum() > 0 else 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())

    sens = tp / n_pos if n_pos > 0 else float("nan")
    spec = tn / n_neg if n_neg > 0 else float("nan")
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    youden = sens + spec - 1.0
    acc = float((y_pred == y_true).mean())

    return {
        "threshold": float(threshold),
        "sensitivity": float(sens) if not np.isnan(sens) else None,
        "specificity": float(spec) if not np.isnan(spec) else None,
        "balanced_accuracy": float(bal_acc) if not np.isnan(bal_acc) else None,
        "youden": float(youden),
        "accuracy": float(acc),
    }


# ── Patient aggregation ────────────────────────────────────────────────────


def aggregate_patient_probs(
    spectrum_probs: np.ndarray,
    patient_index: np.ndarray,
    method: str = "mean",
    trim_proportion: float = 0.1,
) -> np.ndarray:
    """Aggregate spectrum-level probabilities to patient-level.

    Args:
        spectrum_probs: [N] probability of class 1 for each spectrum.
        patient_index: [N] patient index per spectrum.
        method: "mean", "median", "trimmed_mean", "majority_vote".
        trim_proportion: proportion to trim from each tail (for trimmed_mean).

    Returns:
        patient_probs: [n_patients] aggregated probability.
    """
    from scipy.stats import trim_mean

    unique_pids = np.unique(patient_index)
    result = np.zeros(len(unique_pids), dtype=np.float32)

    for i, pid in enumerate(unique_pids):
        mask = patient_index == pid
        probs = spectrum_probs[mask]

        if method == "mean":
            result[i] = probs.mean()
        elif method == "median":
            result[i] = np.median(probs)
        elif method == "trimmed_mean":
            if len(probs) >= 5:
                result[i] = trim_mean(probs, trim_proportion)
            else:
                result[i] = probs.mean()  # fallback for tiny n
        elif method == "majority_vote":
            preds = (probs >= 0.5).astype(int)
            # Majority vote: use mean of positive fraction as "probability"
            majority_class = 1 if preds.sum() > len(preds) / 2 else 0
            result[i] = float(majority_class)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    return result


# ── Loss weight computation ────────────────────────────────────────────────


def compute_patient_balanced_weights(
    patient_index: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Weight = 1 / n_spectra_for_this_patient (each patient contributes equally).

    Args:
        patient_index: [N] patient index per spectrum.
        normalize: if True, normalize so mean weight = 1.0.

    Returns:
        weights: [N] sample weights.
    """
    unique_pids, inv, counts = np.unique(
        patient_index, return_inverse=True, return_counts=True
    )
    w = 1.0 / counts.astype(np.float32)
    weights = w[inv]
    if normalize:
        weights = weights / weights.mean()
    return weights


def compute_patient_class_balanced_weights(
    patient_index: np.ndarray,
    labels: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Patient weight × patient-level class weight.

    Class weights are computed from unique patient counts (not spectrum counts),
    because the unit of independence is the patient.

    class_weight[0] = n_patients / (2 * n_neg_patients)
    class_weight[1] = n_patients / (2 * n_pos_patients)
    """
    patient_w = compute_patient_balanced_weights(patient_index, normalize=False)

    # Patient-level class counts
    unique_pids = np.unique(patient_index)
    patient_labels = np.array([labels[patient_index == pid][0] for pid in unique_pids])
    n_patients = len(patient_labels)
    n_pos = max(int(patient_labels.sum()), 1)
    n_neg = max(n_patients - n_pos, 1)

    # Map patient-level class weight back to spectrum level
    pid_to_class_weight = {}
    for i, pid in enumerate(unique_pids):
        lbl = patient_labels[i]
        pid_to_class_weight[pid] = n_patients / (2.0 * n_pos) if lbl == 1 else n_patients / (2.0 * n_neg)

    class_w = np.array([pid_to_class_weight[pid] for pid in patient_index])

    weights = patient_w * class_w
    if normalize:
        weights = weights / weights.mean()
    return weights.astype(np.float32)


# ── Patient-level split ────────────────────────────────────────────────────


def create_patient_split(
    patient_labels: np.ndarray,
    seed: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    stratify: bool = True,
) -> dict:
    """Create stratified patient-level train/val/test split.

    Args:
        patient_labels: [n_patients] int labels.
        seed: random seed.
        train_ratio, val_ratio, test_ratio: split proportions.

    Returns:
        {train_patients: list, val_patients: list, test_patients: list}
    """
    n = len(patient_labels)
    indices = np.arange(n)

    # First split: train vs (val+test)
    test_val_ratio = val_ratio + test_ratio
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=test_val_ratio,
        random_state=seed,
        stratify=patient_labels if stratify else None,
    )

    # Second split: val vs test
    temp_labels = patient_labels[temp_idx]
    val_ratio_adjusted = val_ratio / test_val_ratio
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1.0 - val_ratio_adjusted,
        random_state=seed,
        stratify=temp_labels if stratify else None,
    )

    return {
        "train_patients": train_idx.tolist(),
        "val_patients": val_idx.tolist(),
        "test_patients": test_idx.tolist(),
    }
