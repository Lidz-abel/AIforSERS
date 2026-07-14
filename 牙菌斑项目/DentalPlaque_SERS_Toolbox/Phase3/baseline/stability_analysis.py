"""Phase 3C: Stability Validation via Repeated Patient-Level Splits.

Tests whether Phase3B results are stable across different random train/val/test
splits, or whether spectrum_level__xgboost winning was just lucky.

For each of N random seeds:
  1. Create stratified patient-level 60/20/20 split
  2. Build feature matrices for both strategies
  3. Train all 4 model×strategy combos (fixed best hyperparams from Phase3B)
  4. Evaluate on test set (patient-level metrics)

Aggregates mean ± std of each metric across all seeds.
"""

from __future__ import annotations

import csv
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from baseline_utils import (
    calibrate_model,
    compute_metrics,
    get_decision_values,
    get_patient_labels,
    load_config as load_baseline_config,
    load_dataset,
    resolve_path,
    toolbox_root,
    write_json,
)

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "figure.facecolor": "white",
})


def load_stability_config() -> dict:
    """Load stability_config.yaml."""
    import yaml
    cfg_path = Path(__file__).resolve().parent / "stability_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_patient_split(patient_labels, cfg, seed):
    """Create stratified patient-level train/val/test split."""
    n = len(patient_labels)
    indices = np.arange(n)
    train_r = cfg["split"]["train_ratio"]
    val_r = cfg["split"]["val_ratio"]

    train_idx, temp_idx = train_test_split(
        indices, test_size=1 - train_r, stratify=patient_labels,
        random_state=seed,
    )
    val_size = val_r / (val_r + cfg["split"]["test_ratio"])
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=1 - val_size,
        stratify=patient_labels[temp_idx],
        random_state=seed + 1,
    )

    masks = {}
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        masks[name] = mask
    return masks


def build_features_for_split(X_spectra, labels, patient_index, n_patients, split_masks):
    """Build both strategy features for a given split."""
    # Strategy A: patient-median
    X_patient = np.zeros((n_patients, X_spectra.shape[1]), dtype=np.float32)
    for i in range(n_patients):
        mask = patient_index == i
        X_patient[i] = np.median(X_spectra[mask], axis=0)
    y_patient = np.zeros(n_patients, dtype=int)
    for i in range(n_patients):
        idx = np.where(patient_index == i)[0][0]
        y_patient[i] = labels[idx]

    # Strategy B: spectrum-level masks
    spectrum_masks = {}
    for split_name in ["train", "val", "test"]:
        mask = np.zeros(len(X_spectra), dtype=bool)
        for i in range(len(X_spectra)):
            pid = patient_index[i]
            if split_masks[split_name][pid]:
                mask[i] = True
        spectrum_masks[split_name] = mask

    return {
        "strategy_a": {
            "X": X_patient, "y": y_patient,
            "train_mask": split_masks["train"],
            "val_mask": split_masks["val"],
            "test_mask": split_masks["test"],
        },
        "strategy_b": {
            "X": X_spectra, "y": labels, "patient_index": patient_index,
            "train_mask": spectrum_masks["train"],
            "val_mask": spectrum_masks["val"],
            "test_mask": spectrum_masks["test"],
        },
    }


def train_and_eval_one_split(features, cfg, data, split_masks):
    """Train all 4 combos on one split, return test metrics."""
    model_cfg = cfg["models"]
    results = []

    patient_labels = get_patient_labels(
        data["patient_uids"], data["labels"], data["patient_index"]
    )
    test_patient_mask = split_masks["test"]
    test_labels = patient_labels[test_patient_mask]

    # ── Strategy A ────────────────────────────────────────
    fa = features["strategy_a"]
    X_tr_a, y_tr_a = fa["X"][fa["train_mask"]], fa["y"][fa["train_mask"]]
    X_val_a, y_val_a = fa["X"][fa["val_mask"]], fa["y"][fa["val_mask"]]
    X_te_a = fa["X"][fa["test_mask"]]

    # LR + patient-median
    lr = LogisticRegression(
        C=model_cfg["logistic_regression"]["C"],
        penalty=model_cfg["logistic_regression"]["penalty"],
        solver=model_cfg["logistic_regression"]["solver"],
        class_weight=model_cfg["logistic_regression"]["class_weight"],
        max_iter=model_cfg["logistic_regression"]["max_iter"],
        random_state=cfg["seed"],
    )
    lr.fit(X_tr_a, y_tr_a)
    lr_cal = calibrate_model(lr, X_val_a, y_val_a, model_cfg["logistic_regression"]["calibration"])
    prob_a_lr = lr_cal.predict_proba(X_te_a)[:, 1]
    results.append({"strategy": "patient_median", "model": "logistic_regression",
                    "metrics": compute_metrics(test_labels, prob_a_lr)})

    # XGBoost + patient-median
    n_neg, n_pos = sum(y_tr_a == 0), sum(y_tr_a == 1)
    sw = n_neg / n_pos if n_pos > 0 else 1.0
    xgb_cfg = model_cfg["xgboost"]
    xgb = XGBClassifier(
        n_estimators=xgb_cfg["n_estimators"], max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"], subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
        scale_pos_weight=sw, random_state=cfg["seed"],
        eval_metric=xgb_cfg["eval_metric"], verbosity=0,
    )
    xgb.fit(X_tr_a, y_tr_a, eval_set=[(X_val_a, y_val_a)], verbose=False)
    xgb_cal = calibrate_model(xgb, X_val_a, y_val_a, xgb_cfg["calibration"])
    prob_a_xgb = xgb_cal.predict_proba(X_te_a)[:, 1]
    results.append({"strategy": "patient_median", "model": "xgboost",
                    "metrics": compute_metrics(test_labels, prob_a_xgb)})

    # ── Strategy B ────────────────────────────────────────
    fb = features["strategy_b"]
    X_tr_b, y_tr_b = fb["X"][fb["train_mask"]], fb["y"][fb["train_mask"]]
    X_val_b, y_val_b = fb["X"][fb["val_mask"]], fb["y"][fb["val_mask"]]
    p_te = fb["patient_index"][fb["test_mask"]]
    X_te_b = fb["X"][fb["test_mask"]]

    # LR + spectrum-level
    lr_b = LogisticRegression(
        C=model_cfg["logistic_regression"]["C"],
        penalty=model_cfg["logistic_regression"]["penalty"],
        solver=model_cfg["logistic_regression"]["solver"],
        class_weight=model_cfg["logistic_regression"]["class_weight"],
        max_iter=model_cfg["logistic_regression"]["max_iter"],
        random_state=cfg["seed"],
    )
    lr_b.fit(X_tr_b, y_tr_b)
    lr_b_cal = calibrate_model(lr_b, X_val_b, y_val_b, model_cfg["logistic_regression"]["calibration"])
    prob_spec = lr_b_cal.predict_proba(X_te_b)[:, 1]
    prob_b_lr = _aggregate_to_patient(prob_spec, p_te, test_patient_mask)
    results.append({"strategy": "spectrum_level", "model": "logistic_regression",
                    "metrics": compute_metrics(test_labels, prob_b_lr)})

    # XGBoost + spectrum-level
    n_neg_b, n_pos_b = sum(y_tr_b == 0), sum(y_tr_b == 1)
    sw_b = n_neg_b / n_pos_b if n_pos_b > 0 else 1.0
    xgb_b = XGBClassifier(
        n_estimators=xgb_cfg["n_estimators"], max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"], subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
        scale_pos_weight=sw_b, random_state=cfg["seed"],
        eval_metric=xgb_cfg["eval_metric"], verbosity=0,
    )
    xgb_b.fit(X_tr_b, y_tr_b, eval_set=[(X_val_b, y_val_b)], verbose=False)
    xgb_b_cal = calibrate_model(xgb_b, X_val_b, y_val_b, xgb_cfg["calibration"])
    prob_spec_xgb = xgb_b_cal.predict_proba(X_te_b)[:, 1]
    prob_b_xgb = _aggregate_to_patient(prob_spec_xgb, p_te, test_patient_mask)
    results.append({"strategy": "spectrum_level", "model": "xgboost",
                    "metrics": compute_metrics(test_labels, prob_b_xgb)})

    return results


def _aggregate_to_patient(prob_spec, p_idx, patient_mask):
    """Aggregate spectrum-level probs to patient level via mean."""
    target_pids = np.where(patient_mask)[0]
    prob_patient = np.array([np.mean(prob_spec[p_idx == pid]) for pid in target_pids])
    return prob_patient


def main():
    cfg = load_stability_config()
    data_cfg = load_baseline_config()
    data = load_dataset(data_cfg)
    root = toolbox_root()
    results_dir = resolve_path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)

    X_spectra = data["X_spectra"]
    labels = data["labels"]
    patient_index = data["patient_index"]
    patient_uids = data["patient_uids"]
    n_patients = len(patient_uids)

    patient_labels = get_patient_labels(patient_uids, labels, patient_index)

    n_repeats = cfg["n_repeats"]
    metrics_list = cfg["evaluation"]["metrics"]
    combo_names = [
        "patient_median__logistic_regression",
        "patient_median__xgboost",
        "spectrum_level__logistic_regression",
        "spectrum_level__xgboost",
    ]

    # Collect metrics across seeds
    all_seed_results = {c: {m: [] for m in metrics_list} for c in combo_names}

    print("=" * 60)
    print(f"Phase 3C: Stability Validation ({n_repeats} random splits)")
    print("=" * 60)

    base_seed = cfg["seed"]
    for run in range(n_repeats):
        seed = base_seed + run * 10
        t0 = time.time()

        # 1. Create split
        split_masks = create_patient_split(patient_labels, cfg, seed)

        # 2. Build features
        features = build_features_for_split(
            X_spectra, labels, patient_index, n_patients, split_masks
        )

        # 3. Train & eval
        combo_results = train_and_eval_one_split(features, cfg, data, split_masks)

        # 4. Collect
        for cr in combo_results:
            key = f"{cr['strategy']}__{cr['model']}"
            for m in metrics_list:
                all_seed_results[key][m].append(cr["metrics"][m])

        elapsed = time.time() - t0
        n_test = split_masks["test"].sum()
        print(f"  Seed {seed:3d}: test={n_test}p, {elapsed:.1f}s")

    # ── Aggregate ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Stability Results (mean ± std across {n_repeats} splits)")
    print(f"{'='*60}")

    summary_rows = []
    metric_display = {
        "roc_auc": "ROC-AUC", "accuracy": "Accuracy",
        "sensitivity": "Sensitivity", "specificity": "Specificity",
        "brier_score": "Brier ↓", "ece": "ECE ↓",
    }

    for combo in combo_names:
        parts = combo.split("__")
        strategy = parts[0]
        model = parts[1]
        print(f"\n  {strategy} + {model}:")
        row = {"combo": combo, "strategy": strategy, "model": model}
        for m in metrics_list:
            vals = np.array(all_seed_results[combo][m])
            # Remove NaN values (can happen for AUC with single-class test sets)
            valid = vals[~np.isnan(vals)]
            mean_v = np.mean(valid) if len(valid) > 0 else float("nan")
            std_v = np.std(valid) if len(valid) > 0 else float("nan")
            print(f"    {metric_display[m]:15s}: {mean_v:.4f} ± {std_v:.4f}  (n_valid={len(valid)}/{n_repeats})")
            row[m] = {"mean": float(mean_v), "std": float(std_v), "n_valid": len(valid)}
        summary_rows.append(row)

    # ── Save JSON ──────────────────────────────────────────
    out_json = resolve_path(cfg["outputs"]["stability_results"])
    write_json(out_json, {
        "n_repeats": n_repeats, "seed": base_seed,
        "combo_results": {
            combo: {m: {"mean": float(np.mean([v for v in all_seed_results[combo][m] if not np.isnan(v)])),
                        "std": float(np.std([v for v in all_seed_results[combo][m] if not np.isnan(v)]))}
                    for m in metrics_list}
            for combo in combo_names
        },
        "per_seed": {combo: {m: [float(v) for v in all_seed_results[combo][m]]
                             for m in metrics_list}
                     for combo in combo_names},
    })
    print(f"\nStability results saved: {out_json}")

    # ── Save CSV summary ───────────────────────────────────
    csv_path = resolve_path(cfg["outputs"]["stability_summary"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["combo", "strategy", "model"]
        for m in metrics_list:
            header.extend([f"{m}_mean", f"{m}_std"])
        writer.writerow(header)
        for row in summary_rows:
            line = [row["combo"], row["strategy"], row["model"]]
            for m in metrics_list:
                line.extend([f"{row[m]['mean']:.4f}", f"{row[m]['std']:.4f}"])
            writer.writerow(line)
    print(f"Stability summary saved: {csv_path}")

    # ── Boxplot ────────────────────────────────────────────
    _plot_stability_boxplot(all_seed_results, combo_names, metrics_list,
                            resolve_path(cfg["outputs"]["stability_boxplot"]))

    # ── Rank analysis ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("Best Model Frequency (by ROC-AUC)")
    print(f"{'='*60}")
    best_counts = {c: 0 for c in combo_names}
    for run_idx in range(n_repeats):
        aucs = {}
        for combo in combo_names:
            aucs[combo] = all_seed_results[combo]["roc_auc"][run_idx]
        best = max(aucs, key=lambda k: aucs[k] if not np.isnan(aucs[k]) else -1)
        best_counts[best] += 1
    for combo in combo_names:
        print(f"  {combo:45s}: {best_counts[combo]:2d}/{n_repeats}")

    print("\nDone.")


def _plot_stability_boxplot(all_seed_results, combo_names, metrics_list, save_path):
    """Boxplot comparing metric distributions across seeds for all combos."""
    n_metrics = len(metrics_list)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.2 * n_metrics, 4))

    if n_metrics == 1:
        axes = [axes]

    short_names = {
        "patient_median__logistic_regression": "Median+LR",
        "patient_median__xgboost": "Median+XGB",
        "spectrum_level__logistic_regression": "Spect+LR",
        "spectrum_level__xgboost": "Spect+XGB",
    }
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    metric_labels = {
        "roc_auc": "ROC-AUC", "accuracy": "Accuracy",
        "sensitivity": "Sensitivity", "specificity": "Specificity",
        "brier_score": "Brier Score", "ece": "ECE",
    }

    for ax, metric in zip(axes, metrics_list):
        data_to_plot = []
        for combo in combo_names:
            vals = [v for v in all_seed_results[combo][metric] if not np.isnan(v)]
            data_to_plot.append(vals)

        bp = ax.boxplot(data_to_plot, labels=[short_names[c] for c in combo_names],
                         patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(metric_labels.get(metric, metric), fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        if metric in ("brier_score", "ece"):
            ax.set_ylabel("lower is better")
        else:
            ax.set_ylabel("higher is better")

    fig.suptitle(f"Phase3C Stability: {len(next(iter(all_seed_results.values()))['roc_auc'])} Random Patient Splits",
                 fontsize=12)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Boxplot saved: {save_path}")


if __name__ == "__main__":
    main()
