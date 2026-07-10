# Phase 3A Figure Audit Report

Date: 2026-07-10

## Data Source Verified

All Phase 3A figures are based on the labeled supervised dataset only:

- Included folders: `牙菌斑SERS光谱/阳性+`, `牙菌斑SERS光谱/阴性-`
- Excluded folders: `牙菌斑SERS光谱/未知`, `牙菌斑SERS光谱/其它数据`
- Label definition: patient label, positive = 1, negative = 0

Verified dataset values:

- Patients: 52
- Spectra: 1970
- Raman-shift points per spectrum: 732
- Raman-shift range: 395.11-1840.62 cm^-1
- Patient labels: negative 21, positive 31
- Spectrum labels: negative 421, positive 1549

## Split Verified

Split is performed at patient level before spectrum-level expansion.

- Train: 31 patients, negative 13, positive 18
- Validation: 10 patients, negative 4, positive 6
- Test: 11 patients, negative 4, positive 7
- Train/validation/test patient overlap: 0
- Unique patients across all splits: 52

Expanded spectrum counts after patient-level split:

- Train: 1197 spectra, negative 260, positive 937
- Validation: 381 spectra, negative 81, positive 300
- Test: 392 spectra, negative 80, positive 312

## Preprocessing Verified

SNV normalization is applied independently to each spectrum.

- `X_raw_spectra` shape: 1970 x 732
- `X_spectra` shape: 1970 x 732
- Maximum absolute SNV mean across spectra: 3.34e-07
- SNV standard deviation range across spectra: 0.9999998-1.0000001
- No non-finite values detected during figure-generation checks

## Figure-Level Verdict

### Figure 1: Dataset Construction Workflow

Verdict: rigorous for report use.

The figure explicitly states the data source, excluded folders, patient labels, extraction range, SNV normalization, metadata outputs, patient-level split, and leakage control rule.

### Figure 2: Dataset Composition and Patient-Level Split Audit

Verdict: rigorous for report use.

The figure separates patient-level counts from spectrum-level counts and marks the negative/positive count inside each split. This avoids treating repeated spectra as independent patients.

### Figure 3: Spectrum Extraction and SNV Preprocessing Quality Control

Verdict: rigorous for preprocessing/QC use.

The figure shows raw spectra, the same spectra after SNV, all-spectrum SNV mean/standard-deviation checks, and patient-level median spectra after SNV. Example spectra are selected deterministically by within-label median raw mean intensity, not by visual preference.

### Legacy Overview Figures

Verdict: acceptable as auxiliary overview figures, but Figures 1-3 should be preferred for formal reporting.

The old combined overview and flowchart are now logically consistent and no longer contain the earlier hard-coded patient count. However, they compress several concepts into one figure and are less explicit than Figures 1-3.

## Corrections Made During Audit

- Removed hard-coded `52 patients` from `plot_dataset_overview.py`; the value now comes from `dataset_summary.json`.
- Added a data-source/exclusion statement to the legacy flowchart.
- Re-ran all Phase 3A figure scripts successfully.

