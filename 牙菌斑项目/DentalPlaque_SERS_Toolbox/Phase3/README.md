# Phase 3: Patient-Level AI Classification

Phase 3 builds a Python/PyTorch pipeline for SERS classification.

The data rule is strict:

1. Split train/val/test by patient.
2. Expand spectra only after patient split.
3. Train on `(single spectrum, patient label)`.
4. Evaluate by aggregating spectrum predictions back to patient level.

Current Phase 3A scope:

- Build a Python-friendly spectra dataset from raw CSV files.
- Save patient-level and spectrum-level metadata.
- Create stratified patient-level train/val/test splits.

The current builder uses raw CSV files and applies per-spectrum SNV by default.
When QC-passed spectra are exported from MATLAB or made readable in Python,
the same structure can be reused with a QC-backed source.

## Files

- `make_dataset/config.yaml` - dataset and split configuration.
- `make_dataset/build_dataset.py` - creates dataset artifacts.
- `make_dataset/make_splits.py` - creates patient-level splits.
- `make_dataset/utils.py` - shared helpers.

## Outputs

- `Results/Phase3/dataset/patient_metadata.csv`
- `Results/Phase3/dataset/spectrum_metadata.csv`
- `Results/Phase3/dataset/spectra.npz`
- `Results/Phase3/dataset/wavenumber.npy`
- `Results/Phase3/dataset/dataset_summary.json`
- `Results/Phase3/splits/split_seed42.json`

