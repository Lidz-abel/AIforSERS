"""Phase 3B: Train baseline models with grid search, calibration, and triage.

For each strategy (patient-median, spectrum-level) and model (LR, XGBoost):
  1. Grid search hyperparameters on training set
  2. Calibrate probabilities on validation set
  3. Compute uncertainty scores on validation set
  4. Derive triage thresholds from validation uncertainty

Outputs:
  - training_results.json  (best params, val metrics for all combos)
  - triage_thresholds.json (thresholds per combo)
  - models/*.joblib        (trained model objects)
"""

from __future__ import annotations

import joblib
import warnings
from pathlib import Path

import numpy as np

from baseline_utils import (
    calibrate_model,
    calibrate_triage_thresholds,
    compute_metrics,
    compute_uncertainty,
    compute_uncertainty_components_val,
    get_decision_values,
    get_patient_labels,
    get_split_masks,
    grid_search_lr,
    grid_search_xgb,
    load_config,
    load_dataset,
    resolve_path,
    save_ref_stats,
    toolbox_root,
    write_json,
)

warnings.filterwarnings("ignore")


def train_strategy_a_patient_median(cfg: dict, data: dict) -> list[dict]:
    """Strategy A: train models on patient-median features."""
    results = []
    models_dir = resolve_path(cfg["paths"]["results_dir"]) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    features = np.load(resolve_path(cfg["paths"]["features_patient_median"]), allow_pickle=True)
    X = features["X_patient"]
    y = features["y_patient"]
    train_mask = features["train_mask"]
    val_mask = features["val_mask"]

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print(f"\n  Train: {len(y_train)} patients (pos={sum(y_train == 1)}, neg={sum(y_train == 0)})")
    print(f"  Val:   {len(y_val)} patients (pos={sum(y_val == 1)}, neg={sum(y_val == 0)})")

    # For GroupKFold we need group labels per sample; here each patient IS a group
    train_groups = np.arange(len(y_train))

    # ── Logistic Regression ──
    if cfg["models"]["logistic_regression"]["enabled"]:
        print("\n  [LR] Grid search...")
        lr = grid_search_lr(X_train, y_train, train_groups, cfg)
        print(f"    Best C: {lr.C}")

        # Calibrate
        calib_method = cfg["models"]["logistic_regression"]["calibration"]
        lr_cal = calibrate_model(lr, X_val, y_val, calib_method)
        print(f"    Calibration: {calib_method}")

        # Val predictions (patient level)
        val_prob = lr_cal.predict_proba(X_val)[:, 1]
        val_decision = get_decision_values(lr_cal, X_val)

        # Uncertainty (no spectrum variance for Strategy A)
        ref_stats = compute_uncertainty_components_val(val_prob, None, val_decision)
        val_uncertainty = compute_uncertainty(
            val_prob,
            prob_variance=None,
            decision_values=val_decision,
            weights=(
                cfg["uncertainty"]["prob_confidence_weight"],
                cfg["uncertainty"]["variance_weight"],
                cfg["uncertainty"]["boundary_distance_weight"],
            ),
            ref_stats=ref_stats,
        )

        # Triage thresholds
        thresholds = calibrate_triage_thresholds(
            val_uncertainty, y_val, val_prob,
            target_coverage=cfg["triage"]["target_coverage"],
        )
        print(f"    Triage thresholds: U_confident <= {thresholds['threshold_1']:.3f} < U_review <= {thresholds['threshold_2']:.3f} < CT")

        val_metrics = compute_metrics(y_val, val_prob)
        print(f"    Val ROC-AUC: {val_metrics['roc_auc']:.4f}")

        model_path = models_dir / "patient_median__logistic_regression.joblib"
        joblib.dump(lr_cal, model_path)

        ref_stats_path = models_dir / "patient_median__logistic_regression__ref_stats.npz"
        save_ref_stats(ref_stats, ref_stats_path)

        results.append({
            "strategy": "patient_median",
            "model": "logistic_regression",
            "best_params": {"C": lr.C},
            "calibration": calib_method,
            "val_metrics": val_metrics,
            "triage_thresholds": thresholds,
            "model_path": str(model_path),
            "ref_stats_path": str(ref_stats_path),
        })

    # ── XGBoost ──
    if cfg["models"]["xgboost"]["enabled"]:
        print("\n  [XGBoost] Grid search...")
        n_neg = sum(y_train == 0)
        n_pos = sum(y_train == 1)
        sw = n_neg / n_pos if n_pos > 0 else 1.0
        print(f"    scale_pos_weight: {sw:.3f}")

        xgb = grid_search_xgb(X_train, y_train, X_val, y_val, cfg, scale_pos_weight=sw)
        print(f"    Best params: n_est={xgb.n_estimators}, max_depth={xgb.max_depth}, lr={xgb.learning_rate}")

        # Calibrate
        calib_method = cfg["models"]["xgboost"]["calibration"]
        xgb_cal = calibrate_model(xgb, X_val, y_val, calib_method)
        print(f"    Calibration: {calib_method}")

        # Val predictions
        val_prob = xgb_cal.predict_proba(X_val)[:, 1]
        val_decision = get_decision_values(xgb_cal, X_val)

        ref_stats = compute_uncertainty_components_val(val_prob, None, val_decision)
        val_uncertainty = compute_uncertainty(
            val_prob,
            prob_variance=None,
            decision_values=val_decision,
            weights=(
                cfg["uncertainty"]["prob_confidence_weight"],
                cfg["uncertainty"]["variance_weight"],
                cfg["uncertainty"]["boundary_distance_weight"],
            ),
            ref_stats=ref_stats,
        )

        thresholds = calibrate_triage_thresholds(
            val_uncertainty, y_val, val_prob,
            target_coverage=cfg["triage"]["target_coverage"],
        )
        print(f"    Triage thresholds: U_confident <= {thresholds['threshold_1']:.3f} < U_review <= {thresholds['threshold_2']:.3f} < CT")

        val_metrics = compute_metrics(y_val, val_prob)
        print(f"    Val ROC-AUC: {val_metrics['roc_auc']:.4f}")

        model_path = models_dir / "patient_median__xgboost.joblib"
        joblib.dump(xgb_cal, model_path)

        ref_stats_path = models_dir / "patient_median__xgboost__ref_stats.npz"
        save_ref_stats(ref_stats, ref_stats_path)

        results.append({
            "strategy": "patient_median",
            "model": "xgboost",
            "best_params": {
                "n_estimators": int(xgb.n_estimators),
                "max_depth": int(xgb.max_depth),
                "learning_rate": float(xgb.learning_rate),
                "subsample": float(xgb.subsample) if hasattr(xgb, "subsample") else None,
                "colsample_bytree": float(xgb.colsample_bytree) if hasattr(xgb, "colsample_bytree") else None,
            },
            "calibration": calib_method,
            "scale_pos_weight": sw,
            "val_metrics": val_metrics,
            "triage_thresholds": thresholds,
            "model_path": str(model_path),
            "ref_stats_path": str(ref_stats_path),
        })

    return results


def train_strategy_b_spectrum_level(cfg: dict, data: dict) -> list[dict]:
    """Strategy B: train on individual spectra, aggregate to patient level."""
    results = []
    models_dir = resolve_path(cfg["paths"]["results_dir"]) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    features = np.load(resolve_path(cfg["paths"]["features_spectrum_level"]), allow_pickle=True)
    X = features["X_spectra"]
    y = features["y_spectra"]
    p_idx = features["patient_index"]
    train_mask = features["train_mask"]
    val_mask = features["val_mask"]

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    p_train = p_idx[train_mask]
    p_val = p_idx[val_mask]

    print(f"\n  Train: {len(y_train)} spectra (pos={sum(y_train == 1)}, neg={sum(y_train == 0)})")
    print(f"  Val:   {len(y_val)} spectra (pos={sum(y_val == 1)}, neg={sum(y_val == 0)})")

    # Get patient-level labels for validation
    patient_uids = data["patient_uids"]
    patient_labels = get_patient_labels(patient_uids, data["labels"], data["patient_index"])
    split_masks = get_split_masks(patient_uids, data["splits"])
    val_patient_mask = split_masks["val"]
    val_patient_labels = patient_labels[val_patient_mask]
    val_patient_indices = np.where(val_patient_mask)[0]

    # ── Logistic Regression ──
    if cfg["models"]["logistic_regression"]["enabled"]:
        print("\n  [LR] Grid search...")
        lr = grid_search_lr(X_train, y_train, p_train, cfg)
        print(f"    Best C: {lr.C}")

        calib_method = cfg["models"]["logistic_regression"]["calibration"]
        lr_cal = calibrate_model(lr, X_val, y_val, calib_method)
        print(f"    Calibration: {calib_method}")

        # Spectrum-level val predictions
        val_prob_spec = lr_cal.predict_proba(X_val)[:, 1]
        val_decision_spec = get_decision_values(lr_cal, X_val)

        # Aggregate to patient level
        val_prob_patient, val_true_patient, val_variance, val_pids = (
            _aggregate_by_patient(val_prob_spec, val_decision_spec, p_val, val_patient_indices, patient_labels)
        )

        val_decision_patient = np.array([
            np.mean(get_decision_values(lr_cal, X_val[p_val == pid]))
            for pid in val_patient_indices
        ])
        ref_stats = compute_uncertainty_components_val(val_prob_patient, val_variance, val_decision_patient)

        val_uncertainty = compute_uncertainty(
            val_prob_patient,
            prob_variance=val_variance,
            decision_values=val_decision_patient,
            weights=(
                cfg["uncertainty"]["prob_confidence_weight"],
                cfg["uncertainty"]["variance_weight"],
                cfg["uncertainty"]["boundary_distance_weight"],
            ),
            ref_stats=ref_stats,
        )

        thresholds = calibrate_triage_thresholds(
            val_uncertainty, val_true_patient, val_prob_patient,
            target_coverage=cfg["triage"]["target_coverage"],
        )
        print(f"    Triage thresholds: U_confident <= {thresholds['threshold_1']:.3f} < U_review <= {thresholds['threshold_2']:.3f} < CT")

        val_metrics = compute_metrics(val_true_patient, val_prob_patient)
        print(f"    Val ROC-AUC: {val_metrics['roc_auc']:.4f}")

        model_path = models_dir / "spectrum_level__logistic_regression.joblib"
        joblib.dump(lr_cal, model_path)

        ref_stats_path = models_dir / "spectrum_level__logistic_regression__ref_stats.npz"
        save_ref_stats(ref_stats, ref_stats_path)

        results.append({
            "strategy": "spectrum_level",
            "model": "logistic_regression",
            "best_params": {"C": lr.C},
            "calibration": calib_method,
            "val_metrics": val_metrics,
            "triage_thresholds": thresholds,
            "model_path": str(model_path),
            "ref_stats_path": str(ref_stats_path),
        })

    # ── XGBoost ──
    if cfg["models"]["xgboost"]["enabled"]:
        print("\n  [XGBoost] Grid search...")
        n_neg = sum(y_train == 0)
        n_pos = sum(y_train == 1)
        sw = n_neg / n_pos if n_pos > 0 else 1.0
        print(f"    scale_pos_weight: {sw:.3f}")

        xgb = grid_search_xgb(X_train, y_train, X_val, y_val, cfg, scale_pos_weight=sw)
        print(f"    Best params: n_est={xgb.n_estimators}, max_depth={xgb.max_depth}, lr={xgb.learning_rate}")

        calib_method = cfg["models"]["xgboost"]["calibration"]
        xgb_cal = calibrate_model(xgb, X_val, y_val, calib_method)
        print(f"    Calibration: {calib_method}")

        val_prob_spec = xgb_cal.predict_proba(X_val)[:, 1]

        # Aggregate to patient level
        val_prob_patient, val_true_patient, val_variance, _ = (
            _aggregate_by_patient(val_prob_spec, None, p_val, val_patient_indices, patient_labels)
        )
        val_decision_patient = np.array([
            np.mean(get_decision_values(xgb_cal, X_val[p_val == pid]))
            for pid in val_patient_indices
        ])

        ref_stats = compute_uncertainty_components_val(val_prob_patient, val_variance, val_decision_patient)

        val_uncertainty = compute_uncertainty(
            val_prob_patient,
            prob_variance=val_variance,
            decision_values=val_decision_patient,
            weights=(
                cfg["uncertainty"]["prob_confidence_weight"],
                cfg["uncertainty"]["variance_weight"],
                cfg["uncertainty"]["boundary_distance_weight"],
            ),
            ref_stats=ref_stats,
        )

        thresholds = calibrate_triage_thresholds(
            val_uncertainty, val_true_patient, val_prob_patient,
            target_coverage=cfg["triage"]["target_coverage"],
        )
        print(f"    Triage thresholds: U_confident <= {thresholds['threshold_1']:.3f} < U_review <= {thresholds['threshold_2']:.3f} < CT")

        val_metrics = compute_metrics(val_true_patient, val_prob_patient)
        print(f"    Val ROC-AUC: {val_metrics['roc_auc']:.4f}")

        model_path = models_dir / "spectrum_level__xgboost.joblib"
        joblib.dump(xgb_cal, model_path)

        ref_stats_path = models_dir / "spectrum_level__xgboost__ref_stats.npz"
        save_ref_stats(ref_stats, ref_stats_path)

        results.append({
            "strategy": "spectrum_level",
            "model": "xgboost",
            "best_params": {
                "n_estimators": int(xgb.n_estimators),
                "max_depth": int(xgb.max_depth),
                "learning_rate": float(xgb.learning_rate),
                "subsample": float(xgb.subsample) if hasattr(xgb, "subsample") else None,
                "colsample_bytree": float(xgb.colsample_bytree) if hasattr(xgb, "colsample_bytree") else None,
            },
            "calibration": calib_method,
            "scale_pos_weight": sw,
            "val_metrics": val_metrics,
            "triage_thresholds": thresholds,
            "model_path": str(model_path),
            "ref_stats_path": str(ref_stats_path),
        })

    return results


def _aggregate_by_patient(
    prob_spec: np.ndarray,
    decision_spec: np.ndarray | None,
    p_idx: np.ndarray,
    target_patient_indices: np.ndarray,
    patient_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate spectrum-level predictions to patient level."""
    n_patients = len(target_patient_indices)
    prob_patient = np.zeros(n_patients)
    variance_patient = np.zeros(n_patients)
    true_patient = np.zeros(n_patients, dtype=int)
    pids = np.zeros(n_patients, dtype=int)

    for i, pid in enumerate(target_patient_indices):
        mask = p_idx == pid
        prob_patient[i] = np.mean(prob_spec[mask])
        variance_patient[i] = np.var(prob_spec[mask])
        true_patient[i] = patient_labels[pid]
        pids[i] = pid

    return prob_patient, true_patient, variance_patient, pids


def main() -> None:
    cfg = load_config()
    root = toolbox_root()
    results_dir = resolve_path(cfg["paths"]["results_dir"])

    print("=" * 60)
    print("Phase 3B: Training Baseline Models")
    print("=" * 60)

    # Load data for patient metadata
    data = load_dataset(cfg)

    all_results = []

    # ── Strategy A: Patient-median ────────────────────────
    print("\n" + "─" * 40)
    print("Strategy A: Patient-Median Features")
    print("─" * 40)
    results_a = train_strategy_a_patient_median(cfg, data)
    all_results.extend(results_a)

    # ── Strategy B: Spectrum-level ────────────────────────
    print("\n" + "─" * 40)
    print("Strategy B: Spectrum-Level Features")
    print("─" * 40)
    results_b = train_strategy_b_spectrum_level(cfg, data)
    all_results.extend(results_b)

    # ── Summary ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Validation ROC-AUC Summary")
    print("=" * 60)
    for r in all_results:
        print(f"  {r['strategy']:20s} + {r['model']:20s}: {r['val_metrics']['roc_auc']:.4f}")

    # ── Save training results ─────────────────────────────
    out_training = resolve_path(cfg["outputs"]["training_results"])
    write_json(out_training, {"results": all_results, "seed": cfg["seed"]})
    print(f"\nTraining results saved: {out_training}")

    # ── Save triage thresholds ────────────────────────────
    triage_data = {}
    for r in all_results:
        key = f"{r['strategy']}__{r['model']}"
        triage_data[key] = r["triage_thresholds"]
    out_triage = resolve_path(cfg["outputs"]["triage_thresholds"])
    write_json(out_triage, triage_data)
    print(f"Triage thresholds saved: {out_triage}")

    print("\nDone.")


if __name__ == "__main__":
    main()
