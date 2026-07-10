from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def toolbox_root() -> Path:
    return Path(__file__).resolve().parents[2]


def project_root() -> Path:
    return toolbox_root().parent


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(path_value: str | None, default: Path) -> Path:
    if path_value is None:
        return default
    p = Path(path_value)
    if p.is_absolute():
        return p
    return toolbox_root() / p


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def natural_key(value: str) -> list[Any]:
    parts = re.split(r"(\d+)", value)
    return [int(p) if p.isdigit() else p for p in parts]


def safe_id(value: str) -> str:
    value = value.replace("\\", "_").replace("/", "_")
    return re.sub(r"\s+", "_", value)


def read_csv_header(file_path: Path, max_rows: int = 40) -> dict[str, str]:
    header: dict[str, str] = {}
    with open(file_path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader, start=1):
            if i > max_rows:
                break
            if len(row) >= 2 and row[0]:
                header[row[0]] = row[1]
    return header


def read_spectrum_csv(file_path: Path, csv_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    row_start = int(csv_cfg["row_start"])
    row_end = int(csv_cfg["row_end"])
    wn_col = int(csv_cfg["wavenumber_column"]) - 1
    int_col = int(csv_cfg["intensity_column"]) - 1
    expected = int(csv_cfg["expected_points"])

    wavenumber: list[float] = []
    intensity: list[float] = []

    with open(file_path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader, start=1):
            if row_idx < row_start:
                continue
            if row_idx > row_end:
                break
            if len(row) <= max(wn_col, int_col):
                raise ValueError(f"Too few columns at row {row_idx}: {file_path}")
            wavenumber.append(float(row[wn_col]))
            intensity.append(float(row[int_col]))

    if len(wavenumber) != expected:
        raise ValueError(
            f"Expected {expected} points, got {len(wavenumber)} in {file_path}"
        )

    return np.asarray(wavenumber, dtype=np.float32), np.asarray(intensity, dtype=np.float32)


def snv(spectra: np.ndarray) -> np.ndarray:
    spectra = np.asarray(spectra, dtype=np.float32)
    mu = spectra.mean(axis=1, keepdims=True)
    sigma = spectra.std(axis=1, keepdims=True)
    sigma[sigma == 0] = np.finfo(np.float32).eps
    return (spectra - mu) / sigma


def write_json(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

