"""Phase 3B: Shared utilities for the ML baseline pipeline.

Loads data, builds feature matrices, computes metrics and uncertainty scores,
and handles triage threshold calibration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GridSearchCV
from xgboost import XGBClassifier

# ── Path helpers ────────────────────────────────────────────────────────────


def toolbox_root() -> Path:
    """Return the DentalPlaque_SERS_Toolbox directory."""
    return Path(__file__).resolve().parents[2]


def resolve_path(path_value: str, default: Path | None = None) -> Path:
    """Resolve a config path relative to toolbox root."""
    p = Path(path_value)
    if p.is_absolute():
        return p
    return toolbox_root() / p


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load baseline_config.yaml."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "baseline_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Data loading ────────────────────────────────────────────────────────────


def load_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load spectra.npz, wavenumber, and split file.

    Returns dict with keys:
        X_spectra, X_raw_spectra, labels, patient_index, patient_uids,
        spectrum_ids, wavenumber, splits
    """
    dataset_dir = resolve_path(cfg["paths"]["dataset_dir"])
    splits_dir = resolve_path(cfg["paths"]["splits_dir"])

    data = np.load(dataset_dir / "spectra.npz", allow_pickle=True)
    wavenumber = np.load(dataset_dir / "wavenumber.npy")

    split_file = cfg["paths"]["split_file"]
    with open(splits_dir / split_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    return {
        "X_spectra": data["X_spectra"],
        "X_raw_spectra": data["X_raw_spectra"],
        "labels": data["labels"],
        "patient_index": data["patient_index"],
        "patient_uids": list(data["patient_uids"]),
        "spectrum_ids": list(data["spectrum_ids"]),
        "wavenumber": wavenumber,
        "splits": splits,
    }


def get_split_masks(
    patient_uids: list[str], splits: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Build boolean masks for train/val/test patients.

    Returns {split_name: bool_array_of_patients}.
    """
    uid_to_idx = {uid: i for i, uid in enumerate(patient_uids)}
    masks = {}
    for split_name in ["train", "val", "test"]:
        mask = np.zeros(len(patient_uids), dtype=bool)
        for uid in splits[f"{split_name}_patients"]:
            if uid in uid_to_idx:
                mask[uid_to_idx[uid]] = True
        masks[split_name] = mask
    return masks


def get_patient_labels(patient_uids: list[str], labels: np.ndarray, patient_index: np.ndarray) -> np.ndarray:
    """Get label for each patient (first spectrum's label, all spectra per patient share the same label)."""
    n_patients = len(patient_uids)
    patient_labels = np.zeros(n_patients, dtype=int)
    for i in range(n_patients):
        idx = np.where(patient_index == i)[0][0]
        patient_labels[i] = labels[idx]
    return patient_labels


# ── Feature building ────────────────────────────────────────────────────────


def build_patient_median_features(
    X_spectra: np.ndarray,
    patient_index: np.ndarray,
    patient_uids: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Strategy A: collapse each patient to their median spectrum.

    Returns (X_patient, patient_ids).
    """
    n_patients = len(patient_uids)
    n_features = X_spectra.shape[1]
    X_patient = np.zeros((n_patients, n_features), dtype=np.float32)

    for i in range(n_patients):
        mask = patient_index == i
        X_patient[i] = np.median(X_spectra[mask], axis=0)

    return X_patient, list(patient_uids)


def build_spectrum_features(
    X_spectra: np.ndarray,
    labels: np.ndarray,
    patient_index: np.ndarray,
    patient_uids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Strategy B: return individual spectra with patient mapping.

    Returns (X, y, patient_idx, patient_ids).
    """
    return (
        X_spectra.copy(),
        labels.copy(),
        patient_index.copy(),
        list(patient_uids),
    )


# ── Metrics ─────────────────────────────────────────────────────────────────


def sensitivity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sensitivity = TP / (TP + FN). Returns NaN when no positive samples exist."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denom = tp + fn
    return tp / denom if denom > 0 else float("nan")


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Specificity = TN / (TN + FP). Returns NaN when no negative samples exist."""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    denom = tn + fp
    return tn / denom if denom > 0 else float("nan")


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error (ECE).

    Standard definition: weighted average of |observed_positive_rate - predicted_probability|
    across equal-width bins.  Bins are [lower, upper) for all but the last, which is [lower, upper]
    so that y_prob == 0.0 and y_prob == 1.0 are both captured.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        if i < n_bins - 1:
            in_bin = (y_prob >= lower) & (y_prob < upper)
        else:
            in_bin = (y_prob >= lower) & (y_prob <= upper)
        n_in_bin = np.sum(in_bin)
        if n_in_bin > 0:
            obs_pos_rate = np.mean(y_true[in_bin])          # fraction of actual positives
            pred_conf = np.mean(y_prob[in_bin])              # mean predicted probability
            ece += (n_in_bin / len(y_true)) * abs(obs_pos_rate - pred_conf)
    return float(ece)


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute all evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "roc_auc": auc,
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": sensitivity_score(y_true, y_pred),
        "specificity": specificity_score(y_true, y_pred),
        "brier_score": brier_score_loss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
    }


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    metric_name: str = "",
) -> dict[str, float]:
    """Bootstrap confidence interval for a metric at the patient level.

    Samples patients with replacement.  Only skips single-class resamples for
    AUC (which is undefined with one class).  Other metrics (accuracy, Brier,
    sensitivity, specificity) are well-defined on single-class samples and
    should be included for unbiased CI estimation.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    values = []
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt_sample = y_true[idx]
        # Only AUC requires both classes
        if metric_name == "roc_auc" and len(np.unique(yt_sample)) < 2:
            continue
        try:
            v = metric_fn(yt_sample, y_prob[idx])
            if not np.isnan(v):
                values.append(v)
        except (ValueError, ZeroDivisionError):
            continue

    if len(values) == 0:
        return {"value": float(metric_fn(y_true, y_prob)), "ci_lower": float("nan"), "ci_upper": float("nan")}

    values = np.array(values)
    lower = np.percentile(values, 100 * alpha / 2)
    upper = np.percentile(values, 100 * (1 - alpha / 2))
    return {
        "value": float(metric_fn(y_true, y_prob)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


# ── Uncertainty scoring ─────────────────────────────────────────────────────


def compute_uncertainty(
    y_prob: np.ndarray,
    prob_variance: np.ndarray | None = None,
    decision_values: np.ndarray | None = None,
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
    ref_stats: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Compute per-patient uncertainty score U in [0, 1].

    Components:
      1. Probability confidence: 1 - 2*|P - 0.5|  (near 0.5 = uncertain)
      2. Prediction variance: variance of P across patient's spectra
      3. Decision boundary distance: normalized |distance| (small = uncertain)

    Each component is percentile-normalized using ref_stats (from validation set),
    then combined via weighted sum.

    Parameters
    ----------
    y_prob : shape (n_patients,) probability of positive class
    prob_variance : shape (n_patients,) variance of P across spectra, or None
    decision_values : shape (n_patients,) raw decision function values, or None
    weights : (w_prob, w_var, w_boundary)
    ref_stats : dict with keys 'prob_conf', 'prob_var', 'boundary_dist'
                each mapping to sorted validation values for percentile lookup.
                If None, min-max normalization is used as fallback.

    Returns
    -------
    uncertainty : shape (n_patients,) values in [0, 1]
    """
    w_prob, w_var, w_boundary = weights
    n = len(y_prob)

    # Component 1: probability certainty — raw distance from 0.5
    # 2*|P-0.5| → 0 at P=0.5 (least certain), 1 at P=0 or 1 (most certain)
    prob_certainty_raw = 2.0 * np.abs(y_prob - 0.5)

    # Component 2: prediction variance (higher variance = more uncertain)
    if prob_variance is not None:
        var_score = prob_variance
    else:
        var_score = np.zeros(n)

    # Component 3: decision boundary distance
    if decision_values is not None:
        boundary_dist = np.abs(decision_values)
    else:
        boundary_dist = np.zeros(n)

    # Convert to uncertainty (invert certainty signals so higher = more uncertain)
    # prob_certainty_raw: high = certain → percentile normalize then invert
    prob_uncertainty = prob_certainty_raw
    # boundary_dist: high = far from boundary (certain) → must invert
    if np.any(boundary_dist > 0):
        if ref_stats is not None and "boundary_dist" in ref_stats:
            boundary_uncertainty = 1.0 - _percentile_normalize(boundary_dist, ref_stats["boundary_dist"])
        else:
            boundary_uncertainty = 1.0 - _minmax_normalize(boundary_dist)
    else:
        boundary_uncertainty = np.zeros(n)
    # var_score: high = uncertain → keep as-is
    if np.any(var_score > 0):
        if ref_stats is not None and "prob_var" in ref_stats:
            var_uncertainty = _percentile_normalize(var_score, ref_stats["prob_var"])
        else:
            var_uncertainty = _minmax_normalize(var_score)
    else:
        var_uncertainty = np.zeros(n)

    # Normalize prob_certainty_raw against val reference, then invert to uncertainty.
    # Raw = 2|P-0.5|, high when certain.  Percentile → fraction of val patients LESS certain.
    # 1.0 - percentile → high when uncertain (matches var_uncertainty & boundary_uncertainty).
    if ref_stats is not None and "prob_certainty" in ref_stats:
        prob_uncertainty = 1.0 - _percentile_normalize(prob_uncertainty, ref_stats["prob_certainty"])
    else:
        prob_uncertainty = 1.0 - _minmax_normalize(prob_uncertainty)

    # Weighted sum
    uncertainty = (
        w_prob * prob_uncertainty
        + w_var * var_uncertainty
        + w_boundary * boundary_uncertainty
    )

    return np.clip(uncertainty, 0.0, 1.0)


def _percentile_normalize(values: np.ndarray, ref_sorted: np.ndarray | None) -> np.ndarray:
    """Map values to [0, 1] via percentile in reference distribution."""
    if ref_sorted is None or len(ref_sorted) == 0:
        return _minmax_normalize(values)
    # For each value, find fraction of ref values below it
    result = np.zeros(len(values))
    for i, v in enumerate(values):
        result[i] = np.searchsorted(ref_sorted, v) / len(ref_sorted)
    return result


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-10:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def compute_uncertainty_components_val(
    y_prob_val: np.ndarray,
    prob_variance_val: np.ndarray | None,
    decision_values_val: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Compute reference distributions for each uncertainty component from validation set.

    Returns dict with sorted arrays for percentile-based normalization.
    """
    ref = {}
    # Store sorted certainty values (2|P-0.5|) from validation set.
    # compute_uncertainty will invert these via 1.0 - percentile.
    prob_certainty_raw = 2.0 * np.abs(y_prob_val - 0.5)
    ref["prob_certainty"] = np.sort(prob_certainty_raw)

    if prob_variance_val is not None:
        ref["prob_var"] = np.sort(prob_variance_val)

    if decision_values_val is not None:
        ref["boundary_dist"] = np.sort(np.abs(decision_values_val))

    return ref


def save_ref_stats(ref_stats: dict[str, np.ndarray], path: Path) -> None:
    """Save uncertainty reference distributions as npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **ref_stats)


def load_ref_stats(path: Path) -> dict[str, np.ndarray]:
    """Load uncertainty reference distributions from npz."""
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


# ── Triage thresholds ───────────────────────────────────────────────────────


def calibrate_triage_thresholds(
    uncertainty: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_coverage: float = 0.95,
) -> dict[str, float]:
    """Calibrate triage thresholds using percentile method.

    threshold_1 (confident upper bound): max U such that ≥ target_coverage
      of the model's TRUE POSITIVES (correctly classified positives) are in
      the confident zone.  This ensures that patients the model gets right
      are assigned low uncertainty.

    threshold_2: 80th percentile of all uncertainties (~top 20% → CT).

    Returns dict with keys threshold_1, threshold_2.
    """
    # Sort patients by uncertainty (ascending)
    sort_idx = np.argsort(uncertainty)
    sorted_u = uncertainty[sort_idx]
    sorted_y = y_true[sort_idx]
    sorted_p = y_prob[sort_idx]

    n = len(sorted_y)
    n_pos = np.sum(sorted_y == 1)

    # Count correctly-classified positives up to each uncertainty index
    cum_tp = np.cumsum((sorted_y == 1) & (sorted_p >= 0.5))
    total_tp = cum_tp[-1] if n > 0 else 0

    if total_tp == 0 or n_pos == 0:
        return {
            "threshold_1": float(np.percentile(uncertainty, 50)),
            "threshold_2": float(np.percentile(uncertainty, 80)),
        }

    # Find threshold_1: max U that covers ≥ target_coverage of true positives
    threshold_1_idx = 0
    for i in range(n):
        tp_coverage = cum_tp[i] / total_tp
        if tp_coverage >= target_coverage:
            threshold_1_idx = i
            break
    else:
        threshold_1_idx = n - 1

    threshold_1 = sorted_u[threshold_1_idx]

    # threshold_2: set at 80th percentile of all uncertainties.
    # Patients with U > threshold_2 (top ~20% most uncertain) → CT-recommended.
    threshold_2 = float(np.percentile(uncertainty, 80))

    # Safeguard: ensure threshold_1 < threshold_2
    if threshold_1 >= threshold_2:
        threshold_2 = min(threshold_1 + 0.05, 1.0)

    return {"threshold_1": float(threshold_1), "threshold_2": float(threshold_2)}


def assign_triage_zone(
    uncertainty: np.ndarray,
    threshold_1: float,
    threshold_2: float,
) -> np.ndarray:
    """Assign each patient to a triage zone.

    0 = confident, 1 = review, 2 = CT-recommended
    """
    zones = np.full(len(uncertainty), 2, dtype=int)  # default CT
    zones[uncertainty <= threshold_2] = 1  # review
    zones[uncertainty <= threshold_1] = 0  # confident
    return zones


# ── Model helpers ───────────────────────────────────────────────────────────


def grid_search_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    cfg: dict[str, Any],
) -> LogisticRegression:
    """Grid search for LogisticRegression with GroupKFold."""
    model_cfg = cfg["models"]["logistic_regression"]
    cv_cfg = cfg["cv"]
    n_splits = cv_cfg.get("n_splits", 5)

    param_grid = {"C": model_cfg.get("param_grid", {}).get("C", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0])}

    base = LogisticRegression(
        penalty=model_cfg.get("penalty", "l2"),
        solver=model_cfg.get("solver", "liblinear"),
        class_weight=model_cfg.get("class_weight", "balanced"),
        max_iter=model_cfg.get("max_iter", 5000),
        random_state=cfg["seed"],
    )

    cv = GroupKFold(n_splits=n_splits)
    gs = GridSearchCV(
        base, param_grid, scoring=cv_cfg.get("scoring", "roc_auc"),
        cv=cv, n_jobs=-1, verbose=0,
    )
    gs.fit(X_train, y_train, groups=groups)
    return gs.best_estimator_


def grid_search_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict[str, Any],
    scale_pos_weight: float,
) -> XGBClassifier:
    """Grid search for XGBoost with early stopping on validation set."""
    model_cfg = cfg["models"]["xgboost"]
    pg = model_cfg.get("param_grid", {})

    best_score = -1.0
    best_model: XGBClassifier | None = None

    n_estimators_list = pg.get("n_estimators", [100])
    max_depth_list = pg.get("max_depth", [3])
    lr_list = pg.get("learning_rate", [0.1])
    subsample_list = pg.get("subsample", [1.0])
    colsample_list = pg.get("colsample_bytree", [1.0])

    for n_est in n_estimators_list:
        for md in max_depth_list:
            for lr in lr_list:
                for ss in subsample_list:
                    for cs in colsample_list:
                        xgb_kwargs: dict[str, Any] = dict(
                            n_estimators=n_est,
                            max_depth=md,
                            learning_rate=lr,
                            subsample=ss,
                            colsample_bytree=cs,
                            scale_pos_weight=scale_pos_weight,
                            random_state=cfg["seed"],
                            eval_metric=model_cfg.get("eval_metric", "logloss"),
                            verbosity=0,
                        )
                        early_stop = model_cfg.get("early_stopping_rounds", None)
                        if early_stop is not None:
                            xgb_kwargs["early_stopping_rounds"] = early_stop
                        model = XGBClassifier(**xgb_kwargs)
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            verbose=False,
                        )
                        val_pred = model.predict_proba(X_val)[:, 1]
                        auc = roc_auc_score(y_val, val_pred)
                        if auc > best_score:
                            best_score = auc
                            best_model = model

    return best_model


def calibrate_model(
    model,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    method: str,
) -> Any:
    """Calibrate a model's probabilities.

    Args:
        model: trained classifier with predict_proba
        X_calib, y_calib: calibration data (validation set)
        method: 'isotonic', 'platt', or 'none'

    Returns:
        Calibrated classifier or original model if method='none'.
    """
    if method == "none":
        return model

    if method == "isotonic":
        # Use IsotonicRegression on the positive-class probabilities
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        # Fit on validation set
        raw_prob = model.predict_proba(X_calib)[:, 1]
        iso.fit(raw_prob, y_calib)
        return _IsotonicWrapper(model, iso)

    if method == "platt":
        return _platt_scaling(model, X_calib, y_calib)

    return model


def _platt_scaling(model, X_calib: np.ndarray, y_calib: np.ndarray):
    """Manual Platt scaling: fit logistic regression on raw model scores.

    Avoids sklearn's CalibratedClassifierCV which removed cv='prefit' in v1.6+.
    """
    raw_scores = get_decision_values(model, X_calib)
    # Reshape for sklearn
    X_scores = raw_scores.reshape(-1, 1)
    platt_lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000)
    platt_lr.fit(X_scores, y_calib)
    return _PlattWrapper(model, platt_lr)


class _PlattWrapper:
    """Wrapper that applies Platt (logistic) calibration to model scores."""

    def __init__(self, model, platt_lr: LogisticRegression):
        self.model = model
        self.platt_lr_ = platt_lr
        self.estimator = model
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_scores = get_decision_values(self.model, X)
        calibrated_pos = self.platt_lr_.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
        calibrated_pos = np.clip(calibrated_pos, 0.0, 1.0)
        return np.column_stack([1.0 - calibrated_pos, calibrated_pos])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        proba_pos = self.predict_proba(X)[:, 1]
        eps = 1e-12
        p = np.clip(proba_pos, eps, 1 - eps)
        return np.log(p / (1 - p))


class _IsotonicWrapper:
    """Wrapper that applies isotonic regression to model probabilities."""

    def __init__(self, model, iso: IsotonicRegression):
        self.model = model
        self.iso_ = iso
        self.estimator = model  # for sklearn compatibility
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.model.predict_proba(X)[:, 1]
        calibrated_pos = self.iso_.predict(raw)
        calibrated_pos = np.clip(calibrated_pos, 0.0, 1.0)
        calibrated_neg = 1.0 - calibrated_pos
        return np.column_stack([calibrated_neg, calibrated_pos])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return a decision-function-like value for boundary distance."""
        proba_pos = self.predict_proba(X)[:, 1]
        # Map probability to log-odds-like scale for better boundary distance
        eps = 1e-12
        p = np.clip(proba_pos, eps, 1 - eps)
        return np.log(p / (1 - p))


def get_decision_values(model, X: np.ndarray) -> np.ndarray:
    """Get decision function values for uncertainty scoring.

    For linear models: distance from decision hyperplane (standardized).
    For tree models: use log-odds from probabilities instead.
    """
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    else:
        # Fallback: convert probabilities to log-odds
        proba = model.predict_proba(X)[:, 1]
        eps = 1e-12
        p = np.clip(proba, eps, 1 - eps)
        return np.log(p / (1 - p))


# ── Calibration curve data ──────────────────────────────────────────────────


def compute_calibration_curve_data(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict[str, Any]:
    """Compute calibration curve for plotting."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    return {"prob_pred": prob_pred.tolist(), "prob_true": prob_true.tolist()}


# ── JSON helpers ────────────────────────────────────────────────────────────


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON with consistent formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    """Handle numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
