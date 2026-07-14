"""Phase 4A: Spectrum-level and patient-level reliability metrics.

Spectrum-level (MC Dropout posterior):
  - p_mean, predictive_entropy, expected_entropy, mutual_information, margin

Patient-level aggregation:
  - mean probability, agreement, probability variance
  - mean entropy, mean MI, mean margin
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ── Spectrum-level reliability ───────────────────────────────────────────────


def compute_spectrum_reliability(
    mc_probs: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Compute per-spectrum reliability metrics from MC posterior samples.

    Args:
        mc_probs: [T, N, n_classes] softmax probabilities from T MC forward passes.

    Returns:
        Dict of numpy arrays, each shape [N,].
    """
    T, N, C = mc_probs.shape

    # Mean probability over MC samples
    p_mean = mc_probs.mean(dim=0)  # [N, C]

    # Predictive entropy: H(E[p])
    # H(p) = -sum_c p_c * log(p_c)
    eps = 1e-10
    pred_entropy = -(p_mean * torch.log(p_mean + eps)).sum(dim=1)  # [N]

    # Expected entropy: E[H(p)]
    # For each MC sample, compute entropy, then average over T
    per_sample_entropy = -(mc_probs * torch.log(mc_probs + eps)).sum(dim=2)  # [T, N]
    exp_entropy = per_sample_entropy.mean(dim=0)  # [N]

    # Mutual Information (model uncertainty): H(E[p]) - E[H(p)]
    mutual_info = pred_entropy - exp_entropy  # [N]

    # Margin: p_top1 - p_top2
    top2, _ = torch.topk(p_mean, 2, dim=1)
    margin = top2[:, 0] - top2[:, 1]  # [N]

    # Predicted class
    pred_class = p_mean.argmax(dim=1)  # [N]

    return {
        "p_positive": p_mean[:, 1].cpu().numpy().astype(np.float32),
        "p_negative": p_mean[:, 0].cpu().numpy().astype(np.float32),
        "pred_class": pred_class.cpu().numpy().astype(int),
        "predictive_entropy": pred_entropy.cpu().numpy().astype(np.float32),
        "expected_entropy": exp_entropy.cpu().numpy().astype(np.float32),
        "mutual_information": mutual_info.cpu().numpy().astype(np.float32),
        "margin": margin.cpu().numpy().astype(np.float32),
    }


# ── Patient-level aggregation ────────────────────────────────────────────────


def aggregate_to_patient(
    spec_rel: dict[str, np.ndarray],
    patient_index: np.ndarray,
    patient_uids: list[str],
    labels: np.ndarray,
    target_patients: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Aggregate spectrum-level reliability metrics to patient level.

    Args:
        spec_rel: dict from compute_spectrum_reliability.
        patient_index: [N_spectra] patient index per spectrum.
        patient_uids: list of all patient UIDs.
        labels: [N_spectra] per-spectrum labels.
        target_patients: optional bool mask or indices of patients to include.

    Returns:
        Dict of numpy arrays, each shape [n_patients,].
    """
    unique_pids = np.unique(patient_index)
    if target_patients is not None:
        unique_pids = unique_pids[target_patients[unique_pids]]

    n_p = len(unique_pids)
    result = {
        "patient_uid": np.array([patient_uids[pid] for pid in unique_pids]),
        "true_label": np.zeros(n_p, dtype=int),
        "prob_positive": np.zeros(n_p, dtype=np.float32),
        "prob_negative": np.zeros(n_p, dtype=np.float32),
        "pred_class": np.zeros(n_p, dtype=int),
        "entropy_mean": np.zeros(n_p, dtype=np.float32),
        "mi_mean": np.zeros(n_p, dtype=np.float32),
        "expected_entropy_mean": np.zeros(n_p, dtype=np.float32),
        "margin_mean": np.zeros(n_p, dtype=np.float32),
        "prob_variance": np.zeros(n_p, dtype=np.float32),
        "patient_agreement": np.zeros(n_p, dtype=np.float32),
        "n_spectra": np.zeros(n_p, dtype=int),
    }

    for i, pid in enumerate(unique_pids):
        mask = patient_index == pid
        n_spec = mask.sum()
        result["n_spectra"][i] = n_spec

        # Patient label (all spectra share same label)
        result["true_label"][i] = int(labels[mask][0])

        # Mean probabilities
        p_pos = spec_rel["p_positive"][mask].mean()
        result["prob_positive"][i] = p_pos
        result["prob_negative"][i] = 1.0 - p_pos

        # Patient-level prediction
        result["pred_class"][i] = 1 if p_pos >= 0.5 else 0

        # Mean reliability metrics
        result["entropy_mean"][i] = spec_rel["predictive_entropy"][mask].mean()
        result["mi_mean"][i] = spec_rel["mutual_information"][mask].mean()
        result["expected_entropy_mean"][i] = spec_rel["expected_entropy"][mask].mean()
        result["margin_mean"][i] = spec_rel["margin"][mask].mean()

        # Probability variance (higher = spectra disagree more)
        result["prob_variance"][i] = spec_rel["p_positive"][mask].var()

        # Patient agreement: fraction of spectra whose argmax = patient-level argmax
        pat_pred = result["pred_class"][i]
        spec_preds = spec_rel["pred_class"][mask]
        result["patient_agreement"][i] = (spec_preds == pat_pred).mean()

    return result


# ── Calibration ──────────────────────────────────────────────────────────────


class TemperatureScaling:
    """Temperature scaling for probability calibration.

    Fit a single temperature parameter T on validation logits/labels.
    calibrated_logit = raw_logit / T
    """

    def __init__(self):
        self.temperature = torch.nn.Parameter(torch.ones(1))

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 200,
    ) -> float:
        """Fit temperature on validation data (patient-level logits & labels).

        Returns the optimized temperature value.
        """
        self.temperature.data = torch.ones(1)
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def _eval():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(_eval)
        return self.temperature.item()

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Return calibrated logits."""
        with torch.no_grad():
            return logits / self.temperature


# ── Clinical confidence (rule-based) ────────────────────────────────────────


def compute_clinical_confidence(
    patient_rel: dict[str, np.ndarray],
    weights: tuple[float, float, float, float] = (0.30, 0.25, 0.20, 0.25),
    high_thresh: float = 0.75,
    med_thresh: float = 0.50,
) -> dict[str, np.ndarray]:
    """Compute rule-based clinical confidence score ∈ [0, 1].

    **WARNING: Exploratory metric — NOT a validated clinical threshold.**
    The weights and thresholds are preset heuristics and have NOT been
    calibrated against clinical outcomes or independent validation data.

    Components (each min-max normalized across patients):
      1. prob_margin: |2*P(positive) - 1| (high = confident prediction)
      2. normalized_entropy: 1 - entropy/entropy_max (low entropy = high confidence)
      3. normalized_MI: 1 - MI/MI_max (low MI = high confidence)
      4. patient_agreement: fraction of spectra agreeing with patient-level prediction

    Weighted sum → clinical_confidence.
    """
    n = len(patient_rel["prob_positive"])

    # 1. Probability margin
    prob_margin = np.abs(2.0 * patient_rel["prob_positive"] - 1.0)  # [0, 1]
    # Already in [0, 1], 1 = far from 0.5

    # 2. Normalized entropy (invert: low entropy → high confidence)
    ent = patient_rel["entropy_mean"]
    ent_max = np.log(2.0)  # max entropy for binary classification
    ent_norm = np.clip(1.0 - ent / ent_max, 0.0, 1.0)

    # 3. Normalized MI (invert: low MI → high confidence)
    # MI ≤ log(K) = log(2) for binary classification.  Use theoretical maximum
    # so that confidence scores are comparable across patient cohorts.
    mi = patient_rel["mi_mean"]
    mi_max = np.log(2.0)  # theoretical maximum for binary classification
    mi_norm = np.clip(1.0 - mi / mi_max, 0.0, 1.0)

    # 4. Patient agreement (already in [0, 1])
    agreement = patient_rel["patient_agreement"]

    # Weighted sum
    w_margin, w_ent, w_mi, w_agree = weights
    confidence = (
        w_margin * prob_margin
        + w_ent * ent_norm
        + w_mi * mi_norm
        + w_agree * agreement
    )
    confidence = np.clip(confidence, 0.0, 1.0)

    # Assign confidence group
    group = np.full(n, "low", dtype=object)
    group[confidence >= med_thresh] = "medium"
    group[confidence >= high_thresh] = "high"

    return {
        "clinical_confidence": confidence.astype(np.float32),
        "confidence_group": group,
        "prob_margin_raw": prob_margin.astype(np.float32),
        "entropy_norm": ent_norm.astype(np.float32),
        "mi_norm": mi_norm.astype(np.float32),
    }


# ── Clinical decision rule ──────────────────────────────────────────────────


def clinical_decision(
    patient_rel: dict[str, np.ndarray],
    confidence: dict[str, np.ndarray],
) -> np.ndarray:
    """Generate clinical recommendation per patient.

    **WARNING: Exploratory rule-based triage — NOT validated for clinical use.**
    Thresholds (high≥0.75, medium≥0.50, MI>0.3) are heuristics calibrated on
    a single split of 52 patients.  Do NOT use for actual clinical decisions.

    Rules:
      High confidence → "Report"
      Medium confidence → "Doctor Review"
      Low confidence or high MI → "Further Examination"
    """
    n = len(patient_rel["prob_positive"])
    recommendations = np.full(n, "Further Examination", dtype=object)

    conf_group = confidence["confidence_group"]
    prob_pos = patient_rel["prob_positive"]
    mi = patient_rel["mi_mean"]

    for i in range(n):
        if conf_group[i] == "high":
            recommendations[i] = "Report"
        elif conf_group[i] == "medium":
            # Medium confidence + positive → doctor should review
            recommendations[i] = "Doctor Review"
        else:
            # Low confidence → Further Examination
            recommendations[i] = "Further Examination"

        # Override: high MI (>0.3) always → Further Examination
        if mi[i] > 0.3:
            recommendations[i] = "Further Examination"

    return recommendations
