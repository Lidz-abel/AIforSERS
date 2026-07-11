"""Phase 3B: Evaluate baseline models on test set.

Loads trained models, makes predictions, computes metrics with bootstrap CIs,
and generates all output files (predictions.csv, metrics.json, triage_report.csv,
calibration curve, uncertainty histogram).

IMPORTANT: All evaluation is at the PATIENT level.
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

from baseline_utils import (
    assign_triage_zone,
    bootstrap_metric_ci,
    compute_calibration_curve_data,
    compute_metrics,
    compute_uncertainty,
    expected_calibration_error,
    get_decision_values,
    get_patient_labels,
    get_split_masks,
    load_config,
    load_dataset,
    load_ref_stats,
    resolve_path,
    sensitivity_score,
    specificity_score,
    toolbox_root,
    write_json,
)

warnings.filterwarnings("ignore")

# Style
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})


def evaluate_combo(
    cfg: dict,
    data: dict,
    result: dict,
    features: dict,
    split_masks: dict,
    patient_labels: np.ndarray,
) -> dict:
    """Evaluate one model×strategy combo on test set."""
    strategy = result["strategy"]
    model_name = result["model"]

    # Load model
    model = joblib.load(result["model_path"])
    thresholds = result["triage_thresholds"]

    test_patient_mask = split_masks["test"]
    test_patient_indices = np.where(test_patient_mask)[0]
    y_true_test = patient_labels[test_patient_mask]

    if strategy == "patient_median":
        # Strategy A: patient-level features
        X_test = features["X_patient"][test_patient_mask]
        y_prob = model.predict_proba(X_test)[:, 1]
        decision_values = get_decision_values(model, X_test)
        prob_variance = None

    else:
        # Strategy B: spectrum-level features, aggregate to patient
        test_spectrum_mask = features["test_mask"]
        X_test_spec = features["X_spectra"][test_spectrum_mask]
        p_idx_test = features["patient_index"][test_spectrum_mask]

        y_prob_spec = model.predict_proba(X_test_spec)[:, 1]

        # Aggregate to patient level
        n_test_patients = len(test_patient_indices)
        y_prob = np.zeros(n_test_patients)
        prob_variance = np.zeros(n_test_patients)
        decision_values = np.zeros(n_test_patients)

        for i, pid in enumerate(test_patient_indices):
            mask = p_idx_test == pid
            y_prob[i] = np.mean(y_prob_spec[mask])
            prob_variance[i] = np.var(y_prob_spec[mask])
            decision_values[i] = np.mean(get_decision_values(model, X_test_spec[mask]))

    # Compute uncertainty using validation-set reference distributions
    # (saved during training — ensures same normalization scale as triage thresholds)
    ref_stats = load_ref_stats(Path(result["ref_stats_path"]))
    uncertainty = compute_uncertainty(
        y_prob,
        prob_variance=prob_variance,
        decision_values=decision_values,
        weights=(
            cfg["uncertainty"]["prob_confidence_weight"],
            cfg["uncertainty"]["variance_weight"],
            cfg["uncertainty"]["boundary_distance_weight"],
        ),
        ref_stats=ref_stats,
    )

    # Triage
    zones = assign_triage_zone(uncertainty, thresholds["threshold_1"], thresholds["threshold_2"])

    # Metrics with bootstrap CIs
    metric_fns = {
        "roc_auc": lambda yt, yp: roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else float("nan"),
        "accuracy": lambda yt, yp: accuracy_score(yt, (yp >= 0.5).astype(int)),
        "sensitivity": lambda yt, yp: sensitivity_score(yt, (yp >= 0.5).astype(int)),
        "specificity": lambda yt, yp: specificity_score(yt, (yp >= 0.5).astype(int)),
        "brier_score": brier_score_loss,
        "ece": expected_calibration_error,
    }

    metrics_ci = {}
    for metric_name in cfg["evaluation"]["metrics"]:
        if metric_name not in metric_fns:
            continue
        fn = metric_fns[metric_name]
        ci = bootstrap_metric_ci(
            y_true_test, y_prob, fn,
            n_bootstrap=cfg["evaluation"]["n_bootstrap"],
            alpha=cfg["evaluation"]["ci_alpha"],
            seed=cfg["seed"],
            metric_name=metric_name,
        )
        metrics_ci[metric_name] = ci

    # Stratified metrics per triage zone
    zone_metrics = {}
    zone_names = {0: "confident", 1: "review", 2: "ct_recommended"}
    for zid, zname in zone_names.items():
        z_mask = zones == zid
        n_z = int(z_mask.sum())
        if n_z == 0:
            # Empty zone: nothing is defined
            zone_metrics[zname] = {
                "n_patients": 0,
                "roc_auc": None, "accuracy": None, "sensitivity": None,
                "specificity": None, "brier_score": None, "ece": None,
            }
        else:
            zone_metrics[zname] = compute_metrics(y_true_test[z_mask], y_prob[z_mask])
            zone_metrics[zname]["n_patients"] = n_z

    # Calibration curve
    calib_data = compute_calibration_curve_data(y_true_test, y_prob)

    return {
        "strategy": strategy,
        "model": model_name,
        "y_true": y_true_test.tolist(),
        "y_prob": y_prob.tolist(),
        "uncertainty": uncertainty.tolist(),
        "triage_zone": zones.tolist(),
        "triage_zone_names": [zone_names[z] for z in zones],
        "metrics": metrics_ci,
        "zone_metrics": zone_metrics,
        "calibration_curve": calib_data,
        "thresholds": thresholds,
    }


def write_predictions_csv(
    path: Path,
    patient_uids: list[str],
    split_masks: dict,
    all_eval_results: list[dict],
) -> None:
    """Write predictions.csv with columns for all model combos.

    Columns: patient_uid, true_label, split,
             {strategy}__{model}__prob, {strategy}__{model}__uncertainty,
             {strategy}__{model}__triage_zone
    """
    test_patient_mask = split_masks["test"]
    test_uids = [uid for uid, m in zip(patient_uids, test_patient_mask) if m]
    n_test = len(test_uids)

    # Build header
    header = ["patient_uid", "true_label", "split"]
    for r in all_eval_results:
        prefix = f"{r['strategy']}__{r['model']}"
        header.extend([f"{prefix}__prob", f"{prefix}__uncertainty", f"{prefix}__triage_zone"])

    rows = []
    for i in range(n_test):
        row = [test_uids[i], int(all_eval_results[0]["y_true"][i]), "test"]
        for r in all_eval_results:
            row.extend([
                round(r["y_prob"][i], 6),
                round(r["uncertainty"][i], 6),
                r["triage_zone_names"][i],
            ])
        rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Predictions saved: {path}")


def _fmt_metric(val) -> str:
    """Format a metric value for CSV: number → rounded string, None/NaN → 'NA'."""
    if val is None:
        return "NA"
    if isinstance(val, float) and np.isnan(val):
        return "NA"
    return f"{val:.4f}"


def write_triage_report_csv(path: Path, all_eval_results: list[dict]) -> None:
    """Write triage_report.csv with per-zone breakdown for each model combo.

    Undefined metrics (empty zones, single-class zones) are written as 'NA'.
    """
    header = ["model_combo", "zone", "n_patients", "roc_auc", "accuracy",
              "sensitivity", "specificity", "brier_score", "ece"]
    rows = []

    for r in all_eval_results:
        combo = f"{r['strategy']}__{r['model']}"
        for zname, zmetrics in r["zone_metrics"].items():
            row = [
                combo, zname, zmetrics["n_patients"],
                _fmt_metric(zmetrics.get("roc_auc")),
                _fmt_metric(zmetrics.get("accuracy")),
                _fmt_metric(zmetrics.get("sensitivity")),
                _fmt_metric(zmetrics.get("specificity")),
                _fmt_metric(zmetrics.get("brier_score")),
                _fmt_metric(zmetrics.get("ece")),
            ]
            rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Triage report saved: {path}")


def plot_calibration_curve(all_eval_results: list[dict], save_path: Path) -> None:
    """Plot calibration curves for all model combos."""
    fig, ax = plt.subplots(figsize=(5, 4.5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, r in enumerate(all_eval_results):
        calib = r["calibration_curve"]
        label = f"{r['strategy']} + {r['model']}"
        ax.plot(calib["prob_pred"], calib["prob_true"], marker="o", ms=5,
                color=colors[i % len(colors)], label=label, linewidth=1.5)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Calibration Curves (Test Set)")
    ax.legend(fontsize=7, loc="best")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Calibration curve saved: {save_path}")


def plot_uncertainty_histogram(all_eval_results: list[dict], save_path: Path) -> None:
    """Plot uncertainty histograms with triage zones."""
    n_combos = len(all_eval_results)
    fig, axes = plt.subplots(1, n_combos, figsize=(4 * n_combos, 3.5))
    if n_combos == 1:
        axes = [axes]

    colors_zone = ["#2ca02c", "#ff7f0e", "#d62728"]  # green, orange, red

    for ax, r in zip(axes, all_eval_results):
        uncertainty = np.array(r["uncertainty"])
        zones = np.array(r["triage_zone"])
        t1 = r["thresholds"]["threshold_1"]
        t2 = r["thresholds"]["threshold_2"]

        for zid, zname, color in zip([0, 1, 2], ["Confident", "Review", "CT"], colors_zone):
            z_data = uncertainty[zones == zid]
            if len(z_data) > 0:
                ax.hist(z_data, bins=10, alpha=0.6, color=color, label=f"{zname} (n={len(z_data)})")

        ax.axvline(t1, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.axvline(t2, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_xlabel("Uncertainty Score")
        ax.set_ylabel("Patient Count")
        ax.set_title(f"{r['strategy']}\n{r['model']}", fontsize=8)
        ax.legend(fontsize=6)
        ax.set_xlim(0, 1)

    fig.suptitle("Uncertainty Distribution with Triage Zones (Test Set)", fontsize=11)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Uncertainty histogram saved: {save_path}")


def main() -> None:
    cfg = load_config()
    root = toolbox_root()
    results_dir = resolve_path(cfg["paths"]["results_dir"])
    figures_dir = resolve_path(cfg["paths"]["figures_dir"])

    print("=" * 60)
    print("Phase 3B: Evaluating Baseline Models on Test Set")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────
    print("\n[1/4] Loading data and models...")
    data = load_dataset(cfg)

    # Load training results
    training_path = resolve_path(cfg["outputs"]["training_results"])
    with open(training_path, "r", encoding="utf-8") as f:
        training_data = json.load(f)
    training_results = training_data["results"]

    # Load features
    features_a = np.load(resolve_path(cfg["paths"]["features_patient_median"]), allow_pickle=True)
    features_b = np.load(resolve_path(cfg["paths"]["features_spectrum_level"]), allow_pickle=True)

    patient_uids = data["patient_uids"]
    patient_labels = get_patient_labels(patient_uids, data["labels"], data["patient_index"])
    split_masks = get_split_masks(patient_uids, data["splits"])

    print(f"  Test patients: {split_masks['test'].sum()}")

    # ── Evaluate each combo ────────────────────────────────
    print("\n[2/4] Evaluating model combos...")
    all_eval_results = []

    for result in training_results:
        strategy = result["strategy"]
        features = features_a if strategy == "patient_median" else features_b

        print(f"  {strategy} + {result['model']}...")
        eval_result = evaluate_combo(
            cfg, data, result, features, split_masks, patient_labels,
        )
        all_eval_results.append(eval_result)

        # Print metrics
        m = eval_result["metrics"]
        print(f"    ROC-AUC: {m['roc_auc']['value']:.4f} [{m['roc_auc']['ci_lower']:.4f}, {m['roc_auc']['ci_upper']:.4f}]")
        print(f"    Accuracy: {m['accuracy']['value']:.4f}")
        print(f"    Sensitivity: {m['sensitivity']['value']:.4f}")
        print(f"    Specificity: {m['specificity']['value']:.4f}")
        print(f"    Brier: {m['brier_score']['value']:.4f}")
        print(f"    ECE: {m['ece']['value']:.4f}")

        # Zone breakdown
        for zname, zm in eval_result["zone_metrics"].items():
            print(f"    Zone '{zname}': {zm['n_patients']} patients")

    # ── Write predictions.csv ──────────────────────────────
    print("\n[3/4] Writing output files...")
    pred_path = resolve_path(cfg["outputs"]["predictions"])
    write_predictions_csv(pred_path, patient_uids, split_masks, all_eval_results)

    # ── Write metrics.json ─────────────────────────────────
    metrics_out = {}
    for r in all_eval_results:
        key = f"{r['strategy']}__{r['model']}"
        metrics_out[key] = {
            "metrics": r["metrics"],
            "zone_metrics": r["zone_metrics"],
        }
    metrics_path = resolve_path(cfg["outputs"]["metrics"])
    write_json(metrics_path, metrics_out)
    print(f"Metrics saved: {metrics_path}")

    # ── Write triage_report.csv ────────────────────────────
    triage_path = resolve_path(cfg["outputs"]["triage_report"])
    write_triage_report_csv(triage_path, all_eval_results)

    # ── Generate figures ────────────────────────────────────
    print("\n[4/4] Generating figures...")
    calib_path = resolve_path(cfg["outputs"]["calibration_curve"])
    plot_calibration_curve(all_eval_results, calib_path)

    uncert_path = resolve_path(cfg["outputs"]["uncertainty_histogram"])
    plot_uncertainty_histogram(all_eval_results, uncert_path)

    print("\n" + "=" * 60)
    print("Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
