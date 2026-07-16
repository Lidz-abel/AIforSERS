# Phase3 QC Dataset

**Date**: 2026-07-16
**Status**: MATLAB-config QC variant

## Definition

This dataset was built from the same labeled raw CSV source as `Phase3A_v1.0`,
using the Python builder:

```bash
python Phase3/make_dataset/build_dataset.py --config Phase3/make_dataset/config_qc.yaml
python Phase3/make_dataset/make_splits.py --config Phase3/make_dataset/config_qc.yaml
```

## Preprocessing

The QC spectra use the same preprocessing parameters as
`DentalPlaque_SERS_Toolbox/config.m`:

- Savitzky-Golay smoothing: order 3, window 7
- airPLS baseline correction: lambda 1000, order 2, wep 0.05, p 0.05, max_iter 50
- SNV normalization

## QC Rules

The QC thresholds match `config.m`:

- SNR minimum: 5
- Saturation value: 65535
- Saturation fraction maximum: 0.02
- Minimum peak number: 4
- Minimum mean prominence: 0.02
- find_peaks settings: prominence 0.005, distance 8, height 0.01

## Outcome

The MATLAB-config QC rules are permissive for this dataset:

| Item | Value |
|---|---:|
| Patients before QC | 52 |
| Patients after QC | 52 |
| Spectra before QC | 1970 |
| Spectra after QC | 1970 |
| Spectra removed | 0 |

Therefore, this is a preprocessing-aligned QC variant, but it is not a
reduced-size filtered dataset under the current MATLAB thresholds.

## Artifacts

- `dataset/patient_metadata.csv`
- `dataset/spectrum_metadata.csv`
- `dataset/spectrum_qc_report.csv`
- `dataset/spectra.npz`
- `dataset/wavenumber.npy`
- `dataset/dataset_summary.json`
- `splits/split_seed42.json`
