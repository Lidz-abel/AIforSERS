"""Reproducibility and immutable experiment registry helpers for Phase 5."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import yaml


TOOLBOX = Path(__file__).resolve().parents[1]
REGISTRY = TOOLBOX / "Results" / "Phase5" / "experiment_registry.csv"


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else TOOLBOX / path


def load_config(path: str | Path) -> tuple[dict, Path]:
    config_path = resolve(path)
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle), config_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=TOOLBOX, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"


def flatten(data: dict, prefix: str = "") -> dict:
    result = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def config_diff(parent: dict, child: dict) -> list[dict]:
    parent_flat, child_flat = flatten(parent), flatten(child)
    rows = []
    for key in sorted(set(parent_flat) | set(child_flat)):
        if parent_flat.get(key) != child_flat.get(key):
            rows.append({"parameter": key, "parent": parent_flat.get(key), "current": child_flat.get(key)})
    return rows


def initialize_run(config: dict, config_path: Path, parent_config: dict | None = None) -> tuple[Path, dict]:
    result_dir = resolve(config["paths"]["results_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    config_hash = canonical_config_hash(config)
    frozen_path = result_dir / "resolved_config.yaml"
    rendered = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    if frozen_path.exists() and canonical_config_hash(yaml.safe_load(frozen_path.read_text(encoding="utf-8"))) != config_hash:
        raise RuntimeError(f"Immutable experiment config changed: {frozen_path}")
    frozen_path.write_text(rendered, encoding="utf-8")

    dataset_path = resolve(config["paths"]["dataset_dir"]) / "spectra.npz"
    manifest = {
        "experiment_id": config["experiment"]["id"],
        "parent": config["experiment"].get("parent"),
        "status": "initialized",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_source": str(config_path),
        "config_sha256": config_hash,
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    (result_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if parent_config is not None:
        (result_dir / "config_diff_from_parent.json").write_text(
            json.dumps(config_diff(parent_config, config), indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return result_dir, manifest


def update_manifest(result_dir: Path, **updates) -> dict:
    path = result_dir / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(updates)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def update_registry(config: dict, manifest: dict, metrics: dict | None = None) -> None:
    metrics = metrics or {}
    row = {
        "experiment_id": manifest["experiment_id"],
        "parent": manifest.get("parent"),
        "status": manifest.get("status"),
        "config_sha256": manifest["config_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "git_commit": manifest["git_commit"],
        "hypothesis": config["experiment"]["hypothesis"],
        "change_reason": config["experiment"]["change_reason"],
        "test_status": config["experiment"]["test_status"],
        "oof_auc": metrics.get("oof_auc"),
        "oof_accuracy": metrics.get("oof_accuracy"),
        "test_auc": metrics.get("test_auc"),
        "test_accuracy": metrics.get("test_accuracy"),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if REGISTRY.exists():
        frame = pd.read_csv(REGISTRY)
        frame = frame[frame["experiment_id"] != row["experiment_id"]]
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    else:
        frame = pd.DataFrame([row])
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values("experiment_id").to_csv(REGISTRY, index=False)
