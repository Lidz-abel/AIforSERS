# Phase 3A Dataset Card — Frozen

**Version**: 1.0
**Date**: 2026-07-10
**Status**: **FROZEN** — this dataset must not be modified. Any change requires a new versioned release.

---

## 1. Dataset Identity

| Field | Value |
|-------|-------|
| Name | `Phase3A_v1.0` |
| Source type | Raw CSV (not QC-filtered) |
| Builder | `Phase3/make_dataset/build_dataset.py` |
| Config | `Phase3/make_dataset/config.yaml` (seed=42) |

## 2. Data Source

- **Root**: `牙菌斑SERS光谱/`
- **Included groups**: `阳性+` (positive), `阴性-` (negative)
- **Excluded**: `未知` (unknown), `其它数据` (other/comparison data)
- **CSV layout**: wavenumber = column D rows 294–1025, intensity = column H rows 294–1025
- **Label definition**: patient-level clinical label. positive = 1, negative = 0. Every spectrum inherits its patient's label.

## 3. Dataset Composition

| | Patients | Spectra |
|---|---|---|
| **Total** | 52 | 1970 |
| Positive (label=1) | 31 | 1549 |
| Negative (label=0) | 21 | 421 |

- **Raman-shift range**: 395.11 – 1840.62 cm⁻¹
- **Points per spectrum**: 732
- **NaN/Inf**: none detected

## 4. Frozen Artifacts

All files live under `Results/Phase3/`.

### Dataset files (`dataset/`)

| File | SHA256 | Description |
|------|--------|-------------|
| `patient_metadata.csv` | `cb195f7781afa710024d5e4273f531106f8901a0b54ad04970094b183156fd44` | One row per patient (52 rows + header) |
| `spectrum_metadata.csv` | `ea0affcc308769d990ac41953b5021ef282b1a922a6dafb83b065dc6f3033a63` | One row per spectrum (1970 rows + header) |
| `spectra.npz` | `22903c43aabb8f21ff923a3e1de7f5c8ec3a50000cb16cfa71ad295827bb3451` | `X_spectra` (1970×732 float32, SNV), `X_raw_spectra` (1970×732, raw), `labels`, `patient_index`, `patient_uids`, `spectrum_ids` |
| `wavenumber.npy` | `2adf10f65ef9167d08ccb74d838d20df3189fdcb7c3f3b8fb63050a2f3c0f814` | Wavenumber axis (732,) |
| `dataset_summary.json` | `dd7cf87db695eaf12c376f89e486888c0e5954a17d8bd3c85de6a6e1ea918070` | Summary statistics |

### Split files (`splits/`)

| File | SHA256 | Description |
|------|--------|-------------|
| `split_seed42.json` | `dd900cc2cf141224a6789d4a28e8b5efa4c91599c619c28389dd8900152102f3` | Patient-level train/val/test assignment |

## 5. File Schemas

### `patient_metadata.csv`
| Column | Type | Description |
|--------|------|-------------|
| `patient_index` | int (0–51) | Unique patient index |
| `patient_uid` | str | Unique patient identifier (`{group}_{id}`) |
| `patient_id` | str | Original patient directory name |
| `group` | str | `阳性+` or `阴性-` |
| `label` | int | 0 = negative, 1 = positive |
| `n_spectra` | int | Number of spectra for this patient |
| `source_folder` | str | Relative path to patient folder |

### `spectrum_metadata.csv`
| Column | Type | Description |
|--------|------|-------------|
| `spectrum_index` | int (0–1969) | Global spectrum index |
| `spectrum_id` | str | Unique ID (`{patient_uid}__{idx:05d}`) |
| `patient_index` | int | FK to `patient_metadata.patient_index` |
| `patient_uid` | str | FK to `patient_metadata.patient_uid` |
| `patient_id` | str | Original patient directory name |
| `group` | str | `阳性+` or `阴性-` |
| `label` | int | 0 or 1 (inherited from patient) |
| `local_spectrum_index` | int | Index within patient folder |
| `day` | str | Sub-folder (day1, day2, ...) |
| `relative_path` | str | CSV path relative to data root |
| `file_name` | str | CSV filename |
| `file_version` | str | Instrument software version |
| `date` | str | Acquisition timestamp |
| `integration_time_sec` | str | Integration time in seconds |

### `spectra.npz` arrays
| Key | Shape | Dtype | Content |
|-----|-------|-------|---------|
| `X_spectra` | (1970, 732) | float32 | SNV-normalized spectra |
| `X_raw_spectra` | (1970, 732) | float32 | Raw intensities |
| `labels` | (1970,) | int64 | Spectrum labels |
| `patient_index` | (1970,) | int64 | Patient index mapping |
| `patient_uids` | (1970,) | object | Patient UID per spectrum |
| `spectrum_ids` | (1970,) | object | Unique spectrum ID |

### `split_seed42.json`
```json
{
  "seed": 42,
  "unit": "patient",
  "train_patients": ["阳性+_11强", ...],   // 31 patients
  "val_patients":   ["阳性+_1", ...],      // 10 patients
  "test_patients":  ["阳性+_12强", ...],   // 11 patients
  "counts": { "train": 31, "val": 10, "test": 11 },
  "label_counts": {
    "train": { "0": 13, "1": 18 },
    "val":   { "0":  4, "1":  6 },
    "test":  { "0":  4, "1":  7 }
  }
}
```

## 6. Preprocessing

Per-spectrum Standard Normal Variate (SNV), applied independently to each spectrum:

```
SNV(x) = (x - mean(x)) / std(x)
```

- No group-level or patient-level information is used during normalization.
- Raw spectra are also preserved in `X_raw_spectra` inside `spectra.npz`.
- No baseline correction was applied at this stage (raw CSV source).

## 7. Split Strategy

- **Unit**: Patient (not spectrum)
- **Method**: Stratified by label, two-stage `train_test_split` (sklearn)
- **Ratios**: train 60% / val 20% / test 20%
- **Seed**: 42
- **Leakage check**: Zero patient overlap between train/val/test (verified)

**Critical rule**: Spectra from a single patient must never appear in more than one split. MCSS / spectrum expansion happens only after this split.

## 8. Known Limitations

1. **No QC filtering**: The dataset includes all raw spectra. Spectra that would fail MATLAB QC (low SNR, saturation, too few peaks) are still present. A QC-filtered variant can be built by swapping the data source.
2. **Class imbalance**: Positive patients outnumber negative (31 vs 21), and positive spectra outnumber negative (1549 vs 421). Training should account for this (weighted loss, oversampling, or careful metric choice).
3. **Single normalization method**: Only SNV is applied. Other normalizations (e.g., area normalization) are not included.
4. **No baseline correction**: Unlike the MATLAB pipeline (airPLS), this dataset does not apply baseline correction before SNV.
5. **Limited to labeled data only**: The 17 patients in `未知` and comparison data in `其它数据` are excluded and cannot be used for supervised training.

## 9. Usage

### Load in Python
```python
import numpy as np
import pandas as pd
import json

# Metadata
patient_df = pd.read_csv("Results/Phase3/dataset/patient_metadata.csv")
spectrum_df = pd.read_csv("Results/Phase3/dataset/spectrum_metadata.csv")

# Spectra
data = np.load("Results/Phase3/dataset/spectra.npz", allow_pickle=True)
X = data["X_spectra"]          # SNV spectra, (1970, 732)
X_raw = data["X_raw_spectra"]  # raw spectra, (1970, 732)
y = data["labels"]             # (1970,)
patient_idx = data["patient_index"]  # (1970,)

# Wavenumber
wn = np.load("Results/Phase3/dataset/wavenumber.npy")

# Split
with open("Results/Phase3/splits/split_seed42.json") as f:
    split = json.load(f)
```

### Verify integrity
```bash
sha256sum -b Results/Phase3/dataset/*.csv Results/Phase3/dataset/*.npy \
             Results/Phase3/dataset/*.npz Results/Phase3/dataset/*.json \
             Results/Phase3/splits/*.json
```

## 10. Change Policy

- This dataset version (`Phase3A_v1.0`) is **frozen**.
- Any modification (new QC filter, different normalization, updated splits) must produce a **new version** (e.g., `Phase3A_v1.1`, `Phase3A_v2.0`).
- The corresponding `build_dataset.py` or `make_splits.py` run must use a new seed or updated `config.yaml`.
- Update this DATASET_CARD.md in the new version with updated hashes and change notes.
