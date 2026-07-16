"""GPU worker used by tmux for parallel Phase 4D OOF training."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from train_phase4d import load_config, run_finalize, run_oof_fold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=["gpu6", "gpu7"], required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    folds = [0, 2, 4] if args.worker == "gpu6" else [1, 3]
    print(f"{args.worker} assigned OOF folds {folds}", flush=True)
    for fold in folds:
        run_oof_fold(cfg, fold)

    if args.worker == "gpu6":
        results_dir = Path(cfg["paths"]["results_dir"])
        if not results_dir.is_absolute():
            results_dir = Path(__file__).resolve().parents[2] / results_dir
        expected = [results_dir / f"oof_fold_{fold}.json" for fold in range(int(cfg["split"]["oof_folds"]))]
        while not all(path.exists() for path in expected):
            missing = [path.stem for path in expected if not path.exists()]
            print(f"Waiting for OOF workers: {missing}", flush=True)
            time.sleep(30)
        run_finalize(cfg)
    else:
        print("gpu7 OOF assignment complete; gpu6 will perform final training.", flush=True)


if __name__ == "__main__":
    main()
