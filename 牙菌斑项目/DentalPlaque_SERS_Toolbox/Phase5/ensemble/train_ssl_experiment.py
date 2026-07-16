"""Experiment-level orchestration helpers for P5-02 SSL ensemble."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

TOOLBOX = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLBOX / "Phase4" / "mcss_hetero"))
sys.path.insert(0, str(TOOLBOX / "Phase5"))

from experiment_utils import initialize_run, load_config, resolve, update_manifest, update_registry
from patient_mcss_dataset import load_dataset, patient_labels
from ssl_pretrain import load_encoder_state, pretrain_encoder
from train_ensemble import member_seed, result_dir, run_final_member, run_oof_member
from train_phase4d import oof_split, outer_development_test


def prepare_ssl(config: dict, config_path: Path) -> Path:
    parent, _ = load_config("Phase5/configs/exp_001_ensemble.yaml")
    directory, manifest = initialize_run(config, config_path, parent_config=parent)
    manifest = update_manifest(
        directory,
        status="pretraining",
        pretraining_leakage_policy="fold-local unlabeled spectra only",
        test_policy="legacy holdout; not confirmatory",
    )
    update_registry(config, manifest)
    return directory


def ssl_paths(config: dict, scope: str | int) -> tuple[Path, Path]:
    name = f"fold_{scope}" if isinstance(scope, int) else str(scope)
    directory = result_dir(config)
    return directory / f"ssl_{name}.pt", directory / f"ssl_{name}.json"


def pretrain_scope(config: dict, scope: str | int, max_epochs: int | None = None) -> Path:
    data = load_dataset(resolve(config["paths"]["dataset_dir"]))
    labels = patient_labels(data)
    development, locked_test = outer_development_test(labels, config)
    if isinstance(scope, int):
        train_ids, _ = oof_split(development, labels, scope, config)
        seed = int(config["seed"]) + 50_000 + scope
    elif scope == "development":
        train_ids = development
        seed = int(config["seed"]) + 59_000
    else:
        raise ValueError(f"Unknown SSL scope: {scope}")
    if set(train_ids.tolist()) & set(locked_test.tolist()):
        raise RuntimeError("Legacy holdout entered self-supervised pretraining")
    checkpoint, metadata = ssl_paths(config, scope)
    print(
        f"SSL scope={scope} patients={len(train_ids)} spectra="
        f"{int(np.isin(data['patient_index'], train_ids).sum())} locked_test={len(locked_test)}",
        flush=True,
    )
    return pretrain_encoder(
        data,
        train_ids,
        config,
        seed,
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        checkpoint,
        metadata,
        max_epochs=max_epochs,
    )


def run_ssl_oof_member(config: dict, fold: int, member: int, max_epochs: int | None = None) -> Path:
    checkpoint, _ = ssl_paths(config, fold)
    if not checkpoint.exists():
        raise RuntimeError(f"Missing fold-local SSL checkpoint: {checkpoint}")
    return run_oof_member(
        config,
        fold,
        member,
        max_epochs=max_epochs,
        initial_encoder_state=load_encoder_state(checkpoint),
        pretraining_checkpoint=str(checkpoint),
    )


def run_ssl_final_member(config: dict, member: int) -> Path:
    checkpoint, _ = ssl_paths(config, "development")
    if not checkpoint.exists():
        raise RuntimeError(f"Missing development SSL checkpoint: {checkpoint}")
    return run_final_member(
        config,
        member,
        initial_encoder_state=load_encoder_state(checkpoint),
        pretraining_checkpoint=str(checkpoint),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Phase5/configs/exp_002_ssl_ensemble.yaml")
    parser.add_argument("--mode", choices=["prepare", "pretrain"], required=True)
    parser.add_argument("--scope", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    if args.mode == "prepare":
        prepare_ssl(config, config_path)
    else:
        if args.scope is None:
            raise ValueError("--scope is required for pretraining")
        scope = int(args.scope) if args.scope.isdigit() else args.scope
        pretrain_scope(config, scope, args.max_epochs)


if __name__ == "__main__":
    main()
