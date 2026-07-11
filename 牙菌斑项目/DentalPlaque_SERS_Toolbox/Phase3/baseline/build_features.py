"""Phase 3B: Build feature matrices for both strategies.

Strategy A (patient-median): collapse each patient to 1 median spectrum.
Strategy B (spectrum-level): keep individual spectra, aggregate later.

Outputs:
  - features_patient_median.npz
  - features_spectrum_level.npz
"""

from __future__ import annotations

import numpy as np

from baseline_utils import (
    build_patient_median_features,
    build_spectrum_features,
    get_patient_labels,
    get_split_masks,
    load_config,
    load_dataset,
    resolve_path,
)


def main() -> None:
    cfg = load_config()

    print("=" * 60)
    print("Phase 3B: Building Feature Matrices")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────
    print("\n[1/3] Loading dataset...")
    data = load_dataset(cfg)

    X_spectra = data["X_spectra"]
    labels = data["labels"]
    patient_index = data["patient_index"]
    patient_uids = data["patient_uids"]
    splits = data["splits"]

    patient_labels = get_patient_labels(patient_uids, labels, patient_index)
    split_masks = get_split_masks(patient_uids, splits)

    print(f"  Spectra: {X_spectra.shape}")
    print(f"  Patients: {len(patient_uids)}")
    print(f"  Train patients: {split_masks['train'].sum()}")
    print(f"  Val patients:   {split_masks['val'].sum()}")
    print(f"  Test patients:  {split_masks['test'].sum()}")

    # ── Strategy A: Patient-median ─────────────────────────
    print("\n[2/3] Building Strategy A (patient-median)...")
    X_patient, patient_ids = build_patient_median_features(
        X_spectra, patient_index, patient_uids
    )
    y_patient = patient_labels.copy()

    # Split masks map to patient index in patient_uids
    train_mask_a = split_masks["train"]
    val_mask_a = split_masks["val"]
    test_mask_a = split_masks["test"]

    print(f"  X_patient:       {X_patient.shape}")
    print(f"  y_patient:       {y_patient.shape}")
    print(f"  Train (patients): {train_mask_a.sum()} (pos={(y_patient[train_mask_a] == 1).sum()}, neg={(y_patient[train_mask_a] == 0).sum()})")
    print(f"  Val (patients):   {val_mask_a.sum()} (pos={(y_patient[val_mask_a] == 1).sum()}, neg={(y_patient[val_mask_a] == 0).sum()})")
    print(f"  Test (patients):  {test_mask_a.sum()} (pos={(y_patient[test_mask_a] == 1).sum()}, neg={(y_patient[test_mask_a] == 0).sum()})")

    out_path_a = resolve_path(cfg["paths"]["features_patient_median"])
    out_path_a.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path_a,
        X_patient=X_patient,
        y_patient=y_patient,
        patient_ids=np.array(patient_ids),
        train_mask=train_mask_a,
        val_mask=val_mask_a,
        test_mask=test_mask_a,
    )
    print(f"  Saved: {out_path_a}")

    # ── Strategy B: Spectrum-level ─────────────────────────
    print("\n[3/3] Building Strategy B (spectrum-level)...")
    X_spec, y_spec, p_idx, p_ids = build_spectrum_features(
        X_spectra, labels, patient_index, patient_uids
    )

    # Build spectrum-level split masks
    train_mask_b = np.zeros(len(X_spec), dtype=bool)
    val_mask_b = np.zeros(len(X_spec), dtype=bool)
    test_mask_b = np.zeros(len(X_spec), dtype=bool)

    for i in range(len(X_spec)):
        pid = p_idx[i]
        if train_mask_a[pid]:
            train_mask_b[i] = True
        elif val_mask_a[pid]:
            val_mask_b[i] = True
        elif test_mask_a[pid]:
            test_mask_b[i] = True

    print(f"  X_spectra:       {X_spec.shape}")
    print(f"  y_spectra:       {y_spec.shape}")
    print(f"  Train (spectra):  {train_mask_b.sum()} (pos={(y_spec[train_mask_b] == 1).sum()}, neg={(y_spec[train_mask_b] == 0).sum()})")
    print(f"  Val (spectra):    {val_mask_b.sum()} (pos={(y_spec[val_mask_b] == 1).sum()}, neg={(y_spec[val_mask_b] == 0).sum()})")
    print(f"  Test (spectra):   {test_mask_b.sum()} (pos={(y_spec[test_mask_b] == 1).sum()}, neg={(y_spec[test_mask_b] == 0).sum()})")

    out_path_b = resolve_path(cfg["paths"]["features_spectrum_level"])
    np.savez_compressed(
        out_path_b,
        X_spectra=X_spec,
        y_spectra=y_spec,
        patient_index=p_idx,
        patient_ids=np.array(p_ids),
        train_mask=train_mask_b,
        val_mask=val_mask_b,
        test_mask=test_mask_b,
    )
    print(f"  Saved: {out_path_b}")

    print("\n" + "=" * 60)
    print("Feature matrices built successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
