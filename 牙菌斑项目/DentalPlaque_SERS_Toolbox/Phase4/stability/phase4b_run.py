r"""Phase 4B: Stability Validation Orchestrator.

Runs all experiments × splits sequentially.  Saves per-split results so that
interrupted runs can be resumed.

Usage:
  python Phase4/stability/phase4b_run.py              # run all
  python Phase4/stability/phase4b_run.py --exp B1     # single experiment
  python Phase4/stability/phase4b_run.py --resume      # skip completed splits
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_toolbox = Path(__file__).resolve().parents[2]

# ── Helpers ─────────────────────────────────────────────────────────────────


def load_config():
    cfg_path = Path(__file__).resolve().parent / "phase4b_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def results_dir() -> Path:
    return _toolbox / "Results" / "Phase4" / "stability"


def split_result_path(exp_id: str, split_seed: int) -> Path:
    """Per-split JSON file for intermediate caching."""
    d = results_dir() / "splits" / exp_id
    d.mkdir(parents=True, exist_ok=True)
    return d / f"split_{split_seed:02d}.json"


def load_split_result(exp_id: str, split_seed: int) -> dict | None:
    p = split_result_path(exp_id, split_seed)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_split_result(exp_id: str, split_seed: int, result: dict):
    p = split_result_path(exp_id, split_seed)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)


def run_one(cfg: dict, exp_id: str, split_seed: int) -> dict | None:
    """Call phase4b_train.py as subprocess.  Returns parsed result dict."""
    train_script = Path(__file__).resolve().parent / "phase4b_train.py"
    output_path = split_result_path(exp_id, split_seed)

    cmd = [
        sys.executable, str(train_script),
        "--exp", exp_id,
        "--split_seed", str(split_seed),
        "--output", str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        print(result.stdout)
        if result.returncode != 0:
            print(f"  ERROR (returncode={result.returncode})")
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return None
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 600s")
        return None

    return load_split_result(exp_id, split_seed)


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default=None, help="Single experiment ID (B0-B4)")
    parser.add_argument("--resume", action="store_true", help="Skip completed splits")
    parser.add_argument("--start_seed", type=int, default=0, help="First split seed")
    args = parser.parse_args()

    cfg = load_config()
    exp_ids = [args.exp] if args.exp else list(cfg["experiments"].keys())
    n_repeats = cfg["n_repeats"]

    print("=" * 60)
    print("Phase 4B: Stability Validation + Ablation Study")
    print("=" * 60)
    print(f"Experiments: {exp_ids}")
    print(f"Splits per experiment: {n_repeats}")
    print(f"Total runs: {len(exp_ids) * n_repeats}")
    print()

    all_results: dict[str, list] = {eid: [] for eid in exp_ids}

    for exp_id in exp_ids:
        print(f"\n{'=' * 60}")
        print(f"Experiment: {exp_id}")
        print(f"{'=' * 60}")
        exp_start = time.time()

        for split_seed in range(args.start_seed, n_repeats):
            # Resume: skip if already completed
            if args.resume:
                existing = load_split_result(exp_id, split_seed)
                if existing is not None:
                    print(f"  [{exp_id}] split_seed={split_seed:02d} — SKIPPED (already done)")
                    all_results[exp_id].append(existing)
                    continue

            print(f"  [{exp_id}] split_seed={split_seed:02d} — running...", end=" ", flush=True)
            t0 = time.time()
            result = run_one(cfg, exp_id, split_seed)
            elapsed = time.time() - t0

            if result is not None:
                save_split_result(exp_id, split_seed, result)
                all_results[exp_id].append(result)
                m = result["test_metrics"]
                print(f"OK ({elapsed:.0f}s) | AUC={m['roc_auc']:.3f} "
                      f"BalAcc={m['balanced_accuracy']:.3f} "
                      f"Sens={m['sensitivity']['value']:.3f} "
                      f"Spec={m['specificity']['value']:.3f}")
            else:
                print(f"FAILED — skipping")

        exp_elapsed = time.time() - exp_start
        n_done = len(all_results[exp_id])
        print(f"  {exp_id}: {n_done}/{n_repeats} completed in {exp_elapsed:.0f}s")

    # ── Aggregate results ─────────────────────────────
    print(f"\n{'=' * 60}")
    print("Aggregating results...")
    aggregated = {}
    for exp_id in exp_ids:
        results = all_results[exp_id]
        if not results:
            print(f"  {exp_id}: NO RESULTS")
            continue

        metric_names = ["roc_auc", "accuracy", "balanced_accuracy", "brier_score", "ece"]
        prop_metrics = ["sensitivity", "specificity"]

        summary = {"n_splits": len(results)}
        for m in metric_names:
            vals = [r["test_metrics"][m] for r in results if r["test_metrics"].get(m) is not None]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if vals:
                summary[m] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }

        for m in prop_metrics:
            vals = [r["test_metrics"][m]["value"] for r in results
                    if r["test_metrics"].get(m) is not None]
            if vals:
                summary[f"{m}_mean"] = float(np.mean(vals))
                summary[f"{m}_std"] = float(np.std(vals, ddof=1))

        # Threshold
        thresholds = [r["threshold"] for r in results]
        summary["threshold_mean"] = float(np.mean(thresholds))
        summary["threshold_std"] = float(np.std(thresholds, ddof=1))

        # Temperature
        temps = [r["temperature"] for r in results]
        summary["temperature_mean"] = float(np.mean(temps))
        summary["temperature_std"] = float(np.std(temps, ddof=1))

        aggregated[exp_id] = summary
        print(f"  {exp_id}: {summary['n_splits']} splits, "
              f"AUC={summary['roc_auc']['mean']:.4f}±{summary['roc_auc']['std']:.4f}, "
              f"BalAcc={summary['balanced_accuracy']['mean']:.4f}±{summary['balanced_accuracy']['std']:.4f}")

    # Save aggregated results
    out_dir = results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    agg_path = out_dir / "phase4b_results.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nAggregated results saved: {agg_path}")

    # Save summary CSV
    csv_path = out_dir / "phase4b_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv_mod.writer(f)
        header = ["exp_id", "n_splits"] + [f"{m}_{s}" for m in metric_names + prop_metrics
                                            for s in ["mean", "std"]]
        writer.writerow(header)
        for exp_id in exp_ids:
            if exp_id not in aggregated:
                continue
            s = aggregated[exp_id]
            row = [exp_id, s["n_splits"]]
            for m in metric_names + prop_metrics:
                row.append(s.get(f"{m}_mean", "NA"))
                row.append(s.get(f"{m}_std", "NA"))
            writer.writerow(row)
    print(f"Summary CSV saved: {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
