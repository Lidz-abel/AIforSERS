from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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
    if normalization == "snv":
        X = snv(X_raw)
    elif normalization == "none":
        X = X_raw.copy()
    else:
        raise ValueError(f"Unsupported normalization: {normalization}")

    y = np.asarray(spectrum_labels, dtype=np.int64)
    patient_index = np.asarray(spectrum_patient_indices, dtype=np.int64)
    patient_uids_arr = np.asarray(patient_uids, dtype=object)
    spectrum_ids_arr = np.asarray(spectrum_ids, dtype=object)

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

