r"""Phase 4B: Results Analysis & Visualization.

Loads per-split results, generates boxplots, computes statistics,
and writes PHASE4B_REPORT.md.

Usage:
  python Phase4/stability/phase4b_analyze.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

_toolbox = Path(__file__).resolve().parents[2]

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "figure.facecolor": "white",
})


# ── Helpers ─────────────────────────────────────────────────────────────────


def load_config():
    cfg_path = Path(__file__).resolve().parent / "phase4b_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_results() -> dict[str, list[dict]]:
    """Load all per-split JSON files.  Returns {exp_id: [result_dict, ...]}."""
    splits_dir = _toolbox / "Results" / "Phase4" / "stability" / "splits"
    results: dict[str, list] = {}
    for exp_dir in sorted(splits_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp_id = exp_dir.name
        exp_results = []
        for fpath in sorted(exp_dir.glob("split_*.json")):
            with open(fpath, "r", encoding="utf-8") as f:
                exp_results.append(json.load(f))
        if exp_results:
            results[exp_id] = exp_results
    return results


def extract_metric_values(all_results, exp_id, metric_name):
    """Extract list of metric values across splits for one experiment."""
    if exp_id not in all_results:
        return []
    vals = []
    for r in all_results[exp_id]:
        tm = r["test_metrics"]
        if metric_name in ("sensitivity", "specificity"):
            v = tm[metric_name]["value"]
        else:
            v = tm[metric_name]
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(float(v))
    return vals


# ── Boxplot ─────────────────────────────────────────────────────────────────


def plot_boxplots(all_results: dict, cfg: dict):
    """Generate multi-panel boxplot comparing experiments on all metrics."""
    exp_ids = list(all_results.keys())
    metrics = ["roc_auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "brier_score", "ece"]
    metric_labels = ["ROC-AUC", "Accuracy", "Balanced Accuracy", "Sensitivity", "Specificity", "Brier Score", "ECE"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    for ax_i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[ax_i]
        data = []
        positions = []
        for j, eid in enumerate(exp_ids):
            vals = extract_metric_values(all_results, eid, metric)
            if vals:
                data.append(vals)
                positions.append(j + 1)

        bp = ax.boxplot(data, positions=positions, widths=0.5,
                        patch_artist=True, showfliers=True, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="red", markersize=5))

        for patch, color in zip(bp["boxes"], colors[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xticks(positions)
        ax.set_xticklabels(exp_ids[:len(data)], fontsize=7)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Highlight B0 as baseline
        if "B0" in exp_ids:
            b0_idx = exp_ids.index("B0")
            if b0_idx < len(data) and data[b0_idx]:
                b0_mean = np.mean(data[b0_idx])
                ax.axhline(y=b0_mean, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    # Hide unused subplot
    if len(metrics) < len(axes):
        axes[-1].set_visible(False)

    fig.suptitle("Phase 4B: Stability Validation — Metric Distributions Across 20 Splits",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    out_path = _toolbox / "Results" / "Phase4" / "stability" / "figures" / "phase4b_boxplot.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Boxplot saved: {out_path}")


# ── Statistics ──────────────────────────────────────────────────────────────


def compute_paired_stats(all_results, exp_id, baseline_id="B0"):
    """Compute paired differences vs baseline."""
    if baseline_id not in all_results or exp_id not in all_results:
        return None

    b0 = all_results[baseline_id]
    ex = all_results[exp_id]

    # Align by split_seed
    b0_by_seed = {r["split_seed"]: r for r in b0}
    ex_by_seed = {r["split_seed"]: r for r in ex}
    common_seeds = sorted(set(b0_by_seed.keys()) & set(ex_by_seed.keys()))

    if len(common_seeds) < 3:
        return None

    metrics = ["roc_auc", "accuracy", "balanced_accuracy", "brier_score", "ece"]
    prop_metrics = ["sensitivity", "specificity"]

    diffs = {}
    for m in metrics + prop_metrics:
        b0_vals = []
        ex_vals = []
        for s in common_seeds:
            if m in prop_metrics:
                b0_vals.append(b0_by_seed[s]["test_metrics"][m]["value"])
                ex_vals.append(ex_by_seed[s]["test_metrics"][m]["value"])
            else:
                b0_vals.append(b0_by_seed[s]["test_metrics"][m])
                ex_vals.append(ex_by_seed[s]["test_metrics"][m])

        b0_vals = np.array(b0_vals)
        ex_vals = np.array(ex_vals)
        d = ex_vals - b0_vals
        diffs[m] = {
            "mean_diff": float(np.mean(d)),
            "std_diff": float(np.std(d, ddof=1)),
            "b0_mean": float(np.mean(b0_vals)),
            "exp_mean": float(np.mean(ex_vals)),
        }

    return {
        "n_paired_splits": len(common_seeds),
        "diffs": diffs,
    }


def compute_win_counts(all_results, baseline_id="B0"):
    """Count how many splits each experiment achieves the best value for each metric."""
    exp_ids = list(all_results.keys())
    metrics = ["roc_auc", "balanced_accuracy", "sensitivity", "specificity"]

    # Align all experiments by split_seed
    seeds = set()
    for eid in exp_ids:
        seeds.update(r["split_seed"] for r in all_results[eid])
    seeds = sorted(seeds)

    wins = {eid: {m: 0 for m in metrics} for eid in exp_ids}

    for s in seeds:
        for m in metrics:
            best_val = -np.inf
            best_exp = None
            for eid in exp_ids:
                for r in all_results[eid]:
                    if r["split_seed"] == s:
                        if m in ("sensitivity", "specificity"):
                            v = r["test_metrics"][m]["value"]
                        else:
                            v = r["test_metrics"][m]
                        if v is not None and v > best_val:
                            best_val = v
                            best_exp = eid
                        break
            if best_exp:
                wins[best_exp][m] += 1

    return wins


# ── Report ──────────────────────────────────────────────────────────────────


def write_report(all_results, paired_stats, win_counts, cfg):
    """Generate PHASE4B_REPORT.md."""
    exp_ids = list(all_results.keys())

    lines = []
    lines.append("# Phase 4B: Stability Validation & Ablation Study")
    lines.append("")
    lines.append(f"**日期**: 2026-07-13")
    lines.append(f"**状态**: 探索性内部验证")
    lines.append("")
    lines.append("## 1. 概述")
    lines.append("")
    lines.append(f"Phase4B 在 {cfg['n_repeats']} 次 repeated patient-level split 上验证 CC-SERSNet v1 的稳定性，")
    lines.append(f"并对损失函数、患者聚合方法、决策阈值和模型选择标准进行系统消融实验。")
    lines.append("")

    lines.append("## 2. 实验矩阵")
    lines.append("")
    lines.append("| ID | Loss | Aggregation | Threshold | Selection |")
    lines.append("|----|------|------------|-----------|-----------|")
    for eid in exp_ids:
        ec = cfg["experiments"][eid]
        loss_desc = {
            "patient_balanced_ce": "Patient-balanced CE",
        }
        loss_name = loss_desc.get(ec["loss"], ec["loss"])
        if ec.get("class_balance_weight"):
            loss_name += " + class-balanced"
        if ec.get("label_smoothing", 0) > 0:
            loss_name += f" + LS({ec['label_smoothing']})"

        thresh_desc = {
            "fixed_0.5": "0.5 (fixed)",
            "max_balanced_accuracy": "max BalAcc (sens≥0.90)",
        }
        thresh_name = thresh_desc.get(ec["threshold_strategy"], ec["threshold_strategy"])

        sel_desc = {
            "val_auc": "val AUC",
            "val_balanced_accuracy": "val BalAcc (sens≥0.90)",
        }
        sel_name = sel_desc.get(ec["selection_metric"], ec["selection_metric"])

        lines.append(f"| {eid} | {loss_name} | {ec['aggregation']} | {thresh_name} | {sel_name} |")

    lines.append("")
    lines.append("## 3. 总体结果 (mean ± std across {n} splits)".format(n=cfg["n_repeats"]))
    lines.append("")
    lines.append("| Experiment | AUC | Bal Acc | Accuracy | Sensitivity | Specificity | Brier | ECE |")
    lines.append("|-----------|-----|---------|----------|-------------|-------------|-------|-----|")

    for eid in exp_ids:
        if eid not in all_results:
            continue
        results = all_results[eid]
        if not results:
            continue

        def fmt(m):
            vals = extract_metric_values(all_results, eid, m)
            if not vals:
                return "N/A"
            return f"{np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}"

        lines.append(
            f"| {eid} | {fmt('roc_auc')} | {fmt('balanced_accuracy')} | {fmt('accuracy')} | "
            f"{fmt('sensitivity')} | {fmt('specificity')} | {fmt('brier_score')} | {fmt('ece')} |"
        )

    lines.append("")
    lines.append("## 4. Win Counts (best per split)")
    lines.append("")
    lines.append("| Experiment | AUC | Bal Acc | Sensitivity | Specificity | Total |")
    lines.append("|-----------|-----|---------|-------------|-------------|-------|")
    for eid in exp_ids:
        w = win_counts.get(eid, {})
        total = sum(w.values())
        lines.append(
            f"| {eid} | {w.get('roc_auc', 0)} | {w.get('balanced_accuracy', 0)} | "
            f"{w.get('sensitivity', 0)} | {w.get('specificity', 0)} | {total} |"
        )

    lines.append("")
    lines.append("## 5. Paired Differences vs B0")
    lines.append("")
    lines.append("| Exp | Δ AUC | Δ Bal Acc | Δ Sens | Δ Spec | Δ Brier | Δ ECE |")
    lines.append("|-----|-------|-----------|--------|--------|---------|-------|")
    for eid in exp_ids:
        if eid == "B0":
            continue
        ps = paired_stats.get(eid)
        if ps is None:
            continue

        def fmt_diff(m):
            d = ps["diffs"].get(m)
            if d is None:
                return "N/A"
            sign = "+" if d["mean_diff"] >= 0 else ""
            return f"{sign}{d['mean_diff']:.4f}"

        lines.append(
            f"| {eid} | {fmt_diff('roc_auc')} | {fmt_diff('balanced_accuracy')} | "
            f"{fmt_diff('sensitivity')} | {fmt_diff('specificity')} | "
            f"{fmt_diff('brier_score')} | {fmt_diff('ece')} |"
        )

    lines.append("")
    lines.append("## 6. 结论")
    lines.append("")
    lines.append("Phase4B 通过 20 次 repeated split 验证了 CC-SERSNet v1 的稳定性，")
    lines.append("并比较了不同损失函数、聚合方法、阈值策略和模型选择标准对患者级性能的影响。")
    lines.append("")
    lines.append("**This is an exploratory internal validation. No external clinical validation yet.**")

    report_path = _toolbox / "Results" / "Phase4" / "stability" / "PHASE4B_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report saved: {report_path}")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    cfg = load_config()

    print("Phase 4B: Analysis")
    print("=" * 60)

    # Load results
    all_results = load_all_results()
    if not all_results:
        print("No results found. Run phase4b_run.py first.")
        return

    exp_ids = list(all_results.keys())
    n_splits = len(all_results[exp_ids[0]]) if exp_ids else 0
    print(f"Experiments: {exp_ids}")
    print(f"Splits per experiment: {n_splits}")
    print()

    # Print quick summary
    for eid in exp_ids:
        results = all_results[eid]
        auc_vals = extract_metric_values(all_results, eid, "roc_auc")
        ba_vals = extract_metric_values(all_results, eid, "balanced_accuracy")
        sens_vals = extract_metric_values(all_results, eid, "sensitivity")
        spec_vals = extract_metric_values(all_results, eid, "specificity")
        print(f"  {eid}: AUC={np.mean(auc_vals):.3f}±{np.std(auc_vals, ddof=1):.3f}, "
              f"BalAcc={np.mean(ba_vals):.3f}±{np.std(ba_vals, ddof=1):.3f}, "
              f"Sens={np.mean(sens_vals):.3f}±{np.std(sens_vals, ddof=1):.3f}, "
              f"Spec={np.mean(spec_vals):.3f}±{np.std(spec_vals, ddof=1):.3f}")

    # Generate boxplots
    print("\nGenerating boxplots...")
    plot_boxplots(all_results, cfg)

    # Compute statistics
    print("Computing paired statistics...")
    paired_stats = {}
    for eid in exp_ids:
        if eid == "B0":
            continue
        ps = compute_paired_stats(all_results, eid)
        if ps:
            paired_stats[eid] = ps
            if "balanced_accuracy" in ps["diffs"]:
                d = ps["diffs"]["balanced_accuracy"]
                sign = "+" if d["mean_diff"] >= 0 else ""
                print(f"  {eid} vs B0: ΔBalAcc = {sign}{d['mean_diff']:.4f} ± {d['std_diff']:.4f}")

    win_counts = compute_win_counts(all_results)
    print("\nWin counts (best per split):")
    for eid in exp_ids:
        w = win_counts[eid]
        total = sum(w.values())
        print(f"  {eid}: {total} wins (AUC={w['roc_auc']}, BalAcc={w['balanced_accuracy']}, "
              f"Sens={w['sensitivity']}, Spec={w['specificity']})")

    # Write report
    print("\nWriting report...")
    write_report(all_results, paired_stats, win_counts, cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()
