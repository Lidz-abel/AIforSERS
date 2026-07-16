"""Two-GPU resumable worker for the P5-01 ensemble experiment."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from train_ensemble import (
    aggregate_oof,
    all_final_paths,
    all_oof_paths,
    finalize,
    load_config,
    result_dir,
    run_final_member,
    run_oof_member,
)


def wait_for(paths: list[Path], label: str) -> None:
    while not all(path.exists() for path in paths):
        missing = sum(not path.exists() for path in paths)
        print(f"waiting for {label}: {missing} remaining", flush=True)
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=["gpu6", "gpu7"], required=True)
    parser.add_argument("--config", default="Phase5/configs/exp_001_ensemble.yaml")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    parity = 0 if args.worker == "gpu6" else 1
    tasks = [
        (fold, member)
        for fold in range(int(config["split"]["oof_folds"]))
        for member in range(int(config["ensemble"]["members"]))
    ]
    assigned = [task for index, task in enumerate(tasks) if index % 2 == parity]
    print(f"{args.worker} OOF tasks={assigned}", flush=True)
    for fold, member in assigned:
        run_oof_member(config, fold, member)

    directory = result_dir(config)
    oof_result = directory / "oof_ensemble.json"
    if args.worker == "gpu6":
        wait_for(all_oof_paths(config), "OOF members")
        aggregate_oof(config)
    else:
        wait_for([oof_result], "OOF aggregation")

    final_members = [member for member in range(int(config["ensemble"]["members"])) if member % 2 == parity]
    print(f"{args.worker} final members={final_members}", flush=True)
    for member in final_members:
        run_final_member(config, member)

    if args.worker == "gpu6":
        wait_for(all_final_paths(config), "final members")
        finalize(config)
    print(f"{args.worker} complete", flush=True)


if __name__ == "__main__":
    main()
