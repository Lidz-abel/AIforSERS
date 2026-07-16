from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.signal import find_peaks, savgol_filter
from scipy.sparse.linalg import spsolve

from utils import (
    ensure_dir,
    load_config,
    natural_key,
    project_root,
    read_csv_header,
    read_spectrum_csv,
    resolve_path,
    safe_id,
    snv,
    toolbox_root,
    write_json,
)


def airpls(
    spectra: np.ndarray,
    lambda_: float = 1e3,
    order: int = 2,
    wep: float = 0.05,
    p: float = 0.05,
    max_iter: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Python port of the MATLAB airPLS implementation used by the toolbox."""
    X = np.asarray(spectra, dtype=np.float64)
    m, n = X.shape
    edge_left = np.arange(int(np.ceil(n * wep)))
    edge_right = np.arange(int(np.floor(n - n * wep)) - 1, n)
    edge_idx = np.unique(np.concatenate([edge_left, edge_right]))

    diff_mat = sparse.csc_matrix(np.diff(np.eye(n), n=order, axis=0))
    penalty = lambda_ * (diff_mat.T @ diff_mat)
    baselines = np.zeros_like(X)

    for i in range(m):
        weights = np.ones(n, dtype=np.float64)
        x = X[i]
        for j in range(1, max_iter + 1):
            system = sparse.diags(weights, 0, shape=(n, n), format="csc") + penalty
            baseline = spsolve(system, weights * x)
            residual = x - baseline
            neg_sum = abs(float(residual[residual < 0].sum()))
            if neg_sum < 0.001 * float(np.abs(x).sum()):
                break
            weights[residual >= 0] = 0.0
            weights[edge_idx] = p
            neg_mask = residual < 0
            weights[neg_mask] = np.exp(j * np.abs(residual[neg_mask]) / neg_sum)
        baselines[i] = baseline

    return (X - baselines).astype(np.float32), baselines.astype(np.float32)


def preprocess_spectra(raw_spectra: np.ndarray, preprocess_cfg: dict) -> np.ndarray:
    """Build training/QC spectra using either SNV-only or MATLAB-like preprocessing."""
    pipeline = preprocess_cfg.get("pipeline", "snv_only").lower()
    normalization = preprocess_cfg.get("normalization", "snv").lower()

    if pipeline == "matlab":
        sg_cfg = preprocess_cfg.get("savgol", {})
        smooth = savgol_filter(
            raw_spectra,
            window_length=int(sg_cfg.get("window", 7)),
            polyorder=int(sg_cfg.get("order", 3)),
            axis=1,
        )
        baseline_cfg = preprocess_cfg.get("baseline", {})
        method = baseline_cfg.get("method", "airPLS").lower()
        if method != "airpls":
            raise ValueError(f"Unsupported baseline method: {method}")
        corrected, _ = airpls(
            smooth,
            lambda_=float(baseline_cfg.get("lambda", 1e3)),
            order=int(baseline_cfg.get("order", 2)),
            wep=float(baseline_cfg.get("wep", 0.05)),
            p=float(baseline_cfg.get("p", 0.05)),
            max_iter=int(baseline_cfg.get("max_iter", 50)),
        )
    elif pipeline in {"snv_only", "none"}:
        corrected = raw_spectra.copy()
    else:
        raise ValueError(f"Unsupported preprocessing pipeline: {pipeline}")

    if normalization == "snv":
        return snv(corrected)
    if normalization == "none":
        return corrected.astype(np.float32)
    raise ValueError(f"Unsupported normalization: {normalization}")


def apply_qc(
    spectra: np.ndarray,
    raw_spectra: np.ndarray,
    qc_cfg: dict,
) -> dict[str, np.ndarray]:
    """Apply MATLAB-style technical and structural QC to processed spectra."""
    tech_cfg = qc_cfg.get("technical", {})
    struct_cfg = qc_cfg.get("structural", {})
    peak_cfg = struct_cfg.get("find_peaks", {})

    snr_min = float(tech_cfg.get("snr_min", 5.0))
    saturation_value = float(tech_cfg.get("saturation_value", 65535.0))
    saturation_frac_max = float(tech_cfg.get("saturation_frac", 0.02))

    min_peak_number = int(struct_cfg.get("min_peak_number", 4))
    min_mean_prominence = float(struct_cfg.get("min_mean_prominence", 0.02))
    peak_prominence = float(peak_cfg.get("min_peak_prominence", 0.005))
    peak_distance = int(peak_cfg.get("min_peak_distance", 8))
    peak_height = float(peak_cfg.get("min_peak_height", 0.01))

    n_spec, n_points = spectra.shape
    passed = np.ones(n_spec, dtype=bool)
    snr = np.zeros(n_spec, dtype=np.float32)
    saturation = np.zeros(n_spec, dtype=np.float32)
    peak_number = np.zeros(n_spec, dtype=np.int32)
    mean_prominence = np.zeros(n_spec, dtype=np.float32)
    fail_reason = np.full(n_spec, "", dtype=object)

    for i in range(n_spec):
        s_proc = spectra[i]
        s_raw = raw_spectra[i]

        if not np.isfinite(s_proc).all() or not np.isfinite(s_raw).all():
            passed[i] = False
            fail_reason[i] = "NaN/Inf"
            continue

        signal = float(np.max(s_proc) - np.min(s_proc))
        noise = float(np.std(np.diff(s_proc)))
        snr[i] = signal / (noise + np.finfo(np.float32).eps)
        if snr[i] < snr_min:
            passed[i] = False
            fail_reason[i] = f"Low SNR ({snr[i]:.1f})"
            continue

        saturation[i] = float(np.sum(s_raw >= saturation_value) / n_points)
        if saturation[i] > saturation_frac_max:
            passed[i] = False
            fail_reason[i] = f"Saturated ({saturation[i] * 100:.1f}%)"
            continue

        _, props = find_peaks(
            s_proc,
            prominence=peak_prominence,
            distance=peak_distance,
            height=peak_height,
        )
        prominences = props.get("prominences", np.asarray([], dtype=np.float32))
        peak_number[i] = int(len(prominences))
        mean_prominence[i] = float(np.mean(prominences)) if len(prominences) else 0.0

        if peak_number[i] < min_peak_number:
            passed[i] = False
            fail_reason[i] = f"NPeaks={peak_number[i]} (<{min_peak_number})"
            continue

        if mean_prominence[i] < min_mean_prominence:
            passed[i] = False
            fail_reason[i] = (
                f"MeanProm={mean_prominence[i]:.4f} (<{min_mean_prominence:.3f})"
            )

    return {
        "pass": passed,
        "snr": snr,
        "saturation": saturation,
        "peak_number": peak_number,
        "mean_prominence": mean_prominence,
        "fail_reason": fail_reason,
    }


def filter_dataset_by_qc(
    patient_rows: list[dict],
    spectrum_rows: list[dict],
    X: np.ndarray,
    X_raw: np.ndarray,
    y: np.ndarray,
    patient_index: np.ndarray,
    spectrum_ids: list[str],
    qc_result: dict[str, np.ndarray],
) -> tuple[list[dict], list[dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Filter spectra by QC pass mask and compact patient/spectrum indices."""
    keep = qc_result["pass"]
    n_patients_raw = len(patient_rows)
    kept_counts = np.bincount(patient_index[keep], minlength=n_patients_raw)
    raw_counts = np.bincount(patient_index, minlength=n_patients_raw)
    keep_patient = kept_counts > 0

    old_to_new = {}
    new_patient_rows = []
    for old_idx, row in enumerate(patient_rows):
        if not keep_patient[old_idx]:
            continue
        new_idx = len(new_patient_rows)
        old_to_new[old_idx] = new_idx
        new_row = dict(row)
        new_row["patient_index"] = new_idx
        new_row["n_spectra_raw"] = int(raw_counts[old_idx])
        new_row["n_spectra"] = int(kept_counts[old_idx])
        new_row["n_spectra_removed_qc"] = int(raw_counts[old_idx] - kept_counts[old_idx])
        new_row["qc_pass_fraction"] = float(kept_counts[old_idx] / raw_counts[old_idx])
        new_patient_rows.append(new_row)

    kept_indices = np.flatnonzero(keep)
    new_spectrum_rows = []
    new_patient_index = np.zeros(len(kept_indices), dtype=np.int64)
    new_spectrum_ids = []
    for new_spec_idx, old_spec_idx in enumerate(kept_indices):
        old_row = spectrum_rows[int(old_spec_idx)]
        old_pid = int(old_row["patient_index"])
        new_pid = old_to_new[old_pid]
        new_row = dict(old_row)
        new_row["spectrum_index"] = new_spec_idx
        new_row["patient_index"] = new_pid
        new_row["qc_snr"] = float(qc_result["snr"][old_spec_idx])
        new_row["qc_saturation"] = float(qc_result["saturation"][old_spec_idx])
        new_row["qc_peak_number"] = int(qc_result["peak_number"][old_spec_idx])
        new_row["qc_mean_prominence"] = float(qc_result["mean_prominence"][old_spec_idx])
        new_row["qc_pass"] = True
        new_spectrum_rows.append(new_row)
        new_patient_index[new_spec_idx] = new_pid
        new_spectrum_ids.append(spectrum_ids[int(old_spec_idx)])

    return (
        new_patient_rows,
        new_spectrum_rows,
        X[keep],
        X_raw[keep],
        y[keep],
        new_patient_index,
        np.asarray(new_spectrum_ids, dtype=object),
    )


def build_dataset(config_path: Path | None = None) -> dict:
    cfg = load_config(config_path)
    data_root = resolve_path(
        cfg["paths"].get("data_root"), project_root() / "牙菌斑SERS光谱"
    )
    dataset_dir = resolve_path(cfg["paths"]["dataset_dir"], toolbox_root())
    ensure_dir(dataset_dir)

    labels = cfg["source"]["labels"]
    groups = cfg["source"]["include_groups"]
    pattern = cfg["source"]["file_pattern"]

    patient_rows = []
    spectrum_rows = []
    spectra_raw = []
    spectrum_labels = []
    spectrum_patient_indices = []
    spectrum_ids = []
    patient_uids = []
    reference_wn = None
    spectrum_index = 0

    for group in groups:
        group_dir = data_root / group
        if not group_dir.is_dir():
            raise FileNotFoundError(f"Missing group directory: {group_dir}")

        patient_dirs = [p for p in group_dir.iterdir() if p.is_dir()]
        patient_dirs = sorted(patient_dirs, key=lambda p: natural_key(p.name))

        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            patient_uid = f"{group}_{patient_id}"
            patient_index = len(patient_rows)
            label = int(labels[group])

            files = sorted(patient_dir.rglob(pattern), key=lambda p: natural_key(str(p)))
            if not files:
                continue

            n_before = len(spectra_raw)
            for local_idx, file_path in enumerate(files):
                wn, intensity = read_spectrum_csv(file_path, cfg["csv"])
                if reference_wn is None:
                    reference_wn = wn
                elif not np.allclose(reference_wn, wn, atol=1e-3):
                    raise ValueError(f"Wavenumber axis mismatch: {file_path}")

                header = read_csv_header(file_path)
                relative_path = file_path.relative_to(data_root)
                rel_patient = file_path.relative_to(patient_dir)
                day = rel_patient.parts[0] if len(rel_patient.parts) > 1 else ""
                spectrum_id = f"{safe_id(patient_uid)}__{spectrum_index:05d}"

                spectra_raw.append(intensity)
                spectrum_labels.append(label)
                spectrum_patient_indices.append(patient_index)
                spectrum_ids.append(spectrum_id)
                spectrum_rows.append(
                    {
                        "spectrum_index": spectrum_index,
                        "spectrum_id": spectrum_id,
                        "patient_index": patient_index,
                        "patient_uid": patient_uid,
                        "patient_id": patient_id,
                        "group": group,
                        "label": label,
                        "local_spectrum_index": local_idx,
                        "day": day,
                        "relative_path": str(relative_path),
                        "file_name": file_path.name,
                        "file_version": header.get("File Version", ""),
                        "date": header.get("Date", ""),
                        "integration_time_sec": header.get("integration times(sec)", ""),
                        "min_intensity": float(np.min(intensity)),
                        "max_intensity": float(np.max(intensity)),
                        "mean_intensity": float(np.mean(intensity)),
                    }
                )
                spectrum_index += 1

            n_spectra = len(spectra_raw) - n_before
            patient_uids.append(patient_uid)
            patient_rows.append(
                {
                    "patient_index": patient_index,
                    "patient_uid": patient_uid,
                    "patient_id": patient_id,
                    "group": group,
                    "label": label,
                    "n_spectra": n_spectra,
                    "source_folder": str(patient_dir.relative_to(data_root)),
                }
            )

    X_raw = np.vstack(spectra_raw).astype(np.float32)
    normalization = cfg["preprocessing"].get("normalization", "none").lower()
    X = preprocess_spectra(X_raw, cfg["preprocessing"])

    y = np.asarray(spectrum_labels, dtype=np.int64)
    patient_index = np.asarray(spectrum_patient_indices, dtype=np.int64)

    qc_summary = None
    if cfg.get("qc", {}).get("enabled", False):
        qc_result = apply_qc(X, X_raw, cfg["qc"])
        qc_report = pd.DataFrame(spectrum_rows)
        qc_report["qc_pass"] = qc_result["pass"]
        qc_report["qc_snr"] = qc_result["snr"]
        qc_report["qc_saturation"] = qc_result["saturation"]
        qc_report["qc_peak_number"] = qc_result["peak_number"]
        qc_report["qc_mean_prominence"] = qc_result["mean_prominence"]
        qc_report["qc_fail_reason"] = qc_result["fail_reason"]
        qc_report.to_csv(dataset_dir / "spectrum_qc_report.csv", index=False, encoding="utf-8-sig")

        n_before = int(len(y))
        (
            patient_rows,
            spectrum_rows,
            X,
            X_raw,
            y,
            patient_index,
            spectrum_ids_arr,
        ) = filter_dataset_by_qc(
            patient_rows,
            spectrum_rows,
            X,
            X_raw,
            y,
            patient_index,
            spectrum_ids,
            qc_result,
        )
        n_after = int(len(y))
        qc_summary = {
            "enabled": True,
            "method": cfg["qc"].get("method", "technical_structural"),
            "n_spectra_before": n_before,
            "n_spectra_after": n_after,
            "n_spectra_removed": n_before - n_after,
            "pass_fraction": float(n_after / n_before) if n_before else 0.0,
            "n_patients_before": int(len(np.unique(spectrum_patient_indices))),
            "n_patients_after": int(len(patient_rows)),
        }
    else:
        spectrum_ids_arr = np.asarray(spectrum_ids, dtype=object)
        qc_summary = {"enabled": False}

    patient_uids_arr = np.asarray([row["patient_uid"] for row in patient_rows], dtype=object)
    patient_df = pd.DataFrame(patient_rows)
    spectrum_df = pd.DataFrame(spectrum_rows)

    patient_df.to_csv(dataset_dir / "patient_metadata.csv", index=False, encoding="utf-8-sig")
    spectrum_df.to_csv(dataset_dir / "spectrum_metadata.csv", index=False, encoding="utf-8-sig")
    np.save(dataset_dir / "wavenumber.npy", reference_wn)

    save_kwargs = {
        "X_spectra": X,
        "labels": y,
        "patient_index": patient_index,
        "patient_uids": patient_uids_arr,
        "spectrum_ids": spectrum_ids_arr,
    }
    if cfg["preprocessing"].get("save_raw", True):
        save_kwargs["X_raw_spectra"] = X_raw
    np.savez_compressed(dataset_dir / "spectra.npz", **save_kwargs)

    summary = {
        "source_type": cfg["source"]["type"],
        "normalization": normalization,
        "n_patients": int(len(patient_df)),
        "n_spectra": int(len(spectrum_df)),
        "n_wavenumber": int(len(reference_wn)),
        "groups": {
            group: {
                "label": int(labels[group]),
                "n_patients": int((patient_df["group"] == group).sum()),
                "n_spectra": int((spectrum_df["group"] == group).sum()),
            }
            for group in groups
        },
        "wavenumber_min": float(reference_wn[0]),
        "wavenumber_max": float(reference_wn[-1]),
        "any_nan": bool(np.isnan(X).any()),
        "any_inf": bool(np.isinf(X).any()),
        "qc": qc_summary,
    }
    write_json(dataset_dir / "dataset_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    summary = build_dataset(args.config)
    print("Dataset built")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
