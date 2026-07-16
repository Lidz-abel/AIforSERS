"""Build the deterministic P5-03 intensity plus first-derivative dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_utils import load_config, resolve, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Phase5/configs/exp_003_dual_view.yaml")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    source_dir = resolve(config["representation"]["source_dataset"])
    output_dir = resolve(config["paths"]["dataset_dir"])
    source_path = source_dir / "spectra.npz"
    source = np.load(source_path, allow_pickle=True)
    wavenumber = np.load(source_dir / "wavenumber.npy").astype(np.float64)
    intensity = source["X_spectra"].astype(np.float32)
    derivative_config = config["representation"]["derivative"]
    delta = float(np.median(np.abs(np.diff(wavenumber))))
    derivative = savgol_filter(
        intensity.astype(np.float64),
        window_length=int(derivative_config["window_length"]),
        polyorder=int(derivative_config["polyorder"]),
        deriv=int(derivative_config["deriv"]),
        delta=delta,
        axis=1,
        mode=str(derivative_config["mode"]),
    )
    epsilon = float(config["representation"]["derivative_normalization"]["epsilon"])
    derivative = derivative - derivative.mean(axis=1, keepdims=True)
    derivative = derivative / np.maximum(derivative.std(axis=1, keepdims=True), epsilon)
    dual_view = np.stack([intensity, derivative.astype(np.float32)], axis=1).astype(np.float32)
    if not np.isfinite(dual_view).all():
        raise RuntimeError("Dual-view dataset contains non-finite values")

    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {key: source[key] for key in source.files}
    arrays["X_spectra"] = dual_view
    output_path = output_dir / "spectra.npz"
    np.savez_compressed(output_path, **arrays)
    np.save(output_dir / "wavenumber.npy", wavenumber.astype(np.float32))
    metadata = {
        "representation": config["representation"],
        "config_source": str(config_path),
        "source_dataset": str(source_path),
        "source_sha256": sha256_file(source_path),
        "output_dataset": str(output_path),
        "output_sha256": sha256_file(output_path),
        "shape": list(dual_view.shape),
        "dtype": str(dual_view.dtype),
        "wavenumber_delta": delta,
        "channel_mean": dual_view.mean(axis=(0, 2)).astype(float).tolist(),
        "channel_std": dual_view.std(axis=(0, 2)).astype(float).tolist(),
    }
    (output_dir / "DERIVED_DATASET.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
