# Phase 3 Audit Report

Date: 2026-07-10

## Scope

This audit covers:

- `Phase3/make_dataset/`
- `Phase3/baseline/`
- `Results/Phase3/dataset/`
- `Results/Phase3/splits/`
- `Results/Phase3/baseline/`
- `Figures/Phase3/`

## Summary Verdict

Phase 3A dataset construction is correct and reproducible.

Phase 3B baseline feature construction is correct, and patient-level test metrics can be treated as exploratory baseline results. However, the current uncertainty, triage, ECE, and bootstrap-CI implementations contain issues and should not be used as formal clinical triage conclusions until corrected.

## Verified Correct Components

### Phase 3A Dataset

Verified values:

- Patients: 52
- Spectra: 1970
- Raman-shift points per spectrum: 732
- Raman-shift range: 395.11-1840.62 cm^-1
- Patient labels: negative 21, positive 31
- Spectrum labels: negative 421, positive 1549
- SNV `X_spectra` shape: 1970 x 732
- Raw `X_raw_spectra` shape: 1970 x 732
- SNV maximum absolute per-spectrum mean: 3.34e-07
- SNV standard-deviation range: 0.9999998-1.0000001

Included data:

- `牙菌斑SERS光谱/阳性+`
- `牙菌斑SERS光谱/阴性-`

Excluded data:

- `牙菌斑SERS光谱/未知`
- `牙菌斑SERS光谱/其它数据`

### Patient-Level Split

Verified split:

- Train: 31 patients, negative 13, positive 18
- Validation: 10 patients, negative 4, positive 6
- Test: 11 patients, negative 4, positive 7
- Train/validation/test patient overlap: 0
- Unique split patients: 52

Expanded spectrum counts after patient-level split:

- Train: 1197 spectra, negative 260, positive 937
- Validation: 381 spectra, negative 81, positive 300
- Test: 392 spectra, negative 80, positive 312

### Phase 3B Feature Construction

`Phase3/baseline/build_features.py` was rerun successfully.

Verified feature outputs:

- `features_patient_median.npz`
  - `X_patient`: 52 x 732
  - train/val/test patients: 31/10/11
  - split overlap: 0
- `features_spectrum_level.npz`
  - `X_spectra`: 1970 x 732
  - train/val/test spectra: 1197/381/392
  - split overlap: 0

This confirms that spectrum-level training samples are expanded only after patient-level assignment.

### Phase 3A Figures

Recommended formal report figures:

- `phase3a_fig1_dataset_construction_workflow.*`
- `phase3a_fig2_dataset_split_audit.*`
- `phase3a_fig3_preprocessing_qc.*`

Verdict: acceptable for formal Phase 3A reporting.

Auxiliary figures:

- `phase3a_make_dataset_flowchart.*`
- `phase3a_dataset_overview.*`

Verdict: acceptable as overview figures, but less explicit than Figures 1-3.

## Phase 3B Exploratory Metrics

Current patient-level test metrics from `Results/Phase3/baseline/metrics.json`:

| Strategy | Model | ROC-AUC | Accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|
| patient_median | logistic_regression | 0.9286 | 0.8182 | 1.0000 | 0.5000 |
| patient_median | xgboost | 0.8571 | 0.7273 | 0.7143 | 0.7500 |
| spectrum_level | logistic_regression | 0.9643 | 0.8182 | 0.8571 | 0.7500 |
| spectrum_level | xgboost | 0.9643 | 0.9091 | 1.0000 | 0.7500 |

These metrics are patient-level and can be reported as preliminary baseline performance, with the limitation that the test set contains only 11 patients.

## Findings Requiring Correction

### F1. Uncertainty Probability Component Is Inverted

Severity: High for triage/uncertainty interpretation.

Location:

- `Phase3/baseline/baseline_utils.py`, `compute_uncertainty`

Current logic:

- `prob_conf = 1 - 2 * |P - 0.5|`
- This value is highest near P=0.5, meaning it is actually an uncertainty-like score.
- The code then computes `prob_uncertainty = 1 - prob_conf`, making confident probabilities near 0 or 1 appear more uncertain.

Impact:

- `uncertainty`, `triage_zone`, `triage_report.csv`, and `uncertainty_histogram.png` are not reliable.
- Core classification probabilities and ROC-AUC are not directly affected.

Required correction:

- Define probability uncertainty as `1 - 2 * abs(P - 0.5)` directly, or rename/rederive the component consistently.

### F2. Test Set Is Used to Normalize Test Uncertainty

Severity: High for triage/uncertainty interpretation.

Location:

- `Phase3/baseline/evaluate_baseline.py`, `evaluate_combo`

Current logic:

- `ref_stats = compute_uncertainty_components_val(y_prob, prob_variance, decision_values)` is computed on the test predictions.
- The resulting test-derived reference distribution is then used to normalize test uncertainty.

Impact:

- The validation-calibrated triage thresholds and test uncertainty scores are not on a guaranteed common scale.
- This is a form of transductive use of test distribution for uncertainty normalization.

Required correction:

- Save validation reference statistics during training.
- Reuse those validation reference statistics during test evaluation.

### F3. ECE Implementation Uses Classification Accuracy Instead of Positive Rate

Severity: Medium to high for calibration reporting.

Location:

- `Phase3/baseline/baseline_utils.py`, `expected_calibration_error`

Current logic:

- Per-bin value is computed as thresholded classification accuracy versus mean confidence.

Expected ECE logic:

- Per-bin value should compare observed positive rate `mean(y_true)` with mean predicted probability `mean(y_prob)`.

Impact:

- ECE values in `metrics.json` are not standard calibration ECE.
- Calibration curve itself is still computed with sklearn and is not affected by this specific function.

Required correction:

- Replace per-bin `acc_in_bin` with `observed_rate = mean(y_true[in_bin])`.

### F4. Bootstrap CI Skips Single-Class Resamples for All Metrics

Severity: Medium.

Location:

- `Phase3/baseline/baseline_utils.py`, `bootstrap_metric_ci`

Current logic:

- Bootstrap samples with one class are skipped before metric-specific computation.

Impact:

- This is necessary for ROC-AUC but not for accuracy, Brier score, sensitivity, or specificity.
- Confidence intervals for non-AUC metrics may be biased, especially with only 11 test patients.

Required correction:

- Make the skip behavior metric-specific.
- Only skip single-class bootstrap samples for metrics that mathematically require both classes.

### F5. Triage Threshold Semantics Are Not Clinically Final

Severity: Medium.

Location:

- `Phase3/baseline/baseline_utils.py`, `calibrate_triage_thresholds`

Current logic:

- `target_sensitivity` is implemented as positive-patient coverage within the low-uncertainty subset, not as the sensitivity of accepted predictions.

Impact:

- The phrase "target sensitivity" may be misleading.
- Triage thresholds should not be used as a clinical decision protocol.

Required correction:

- Decide whether the target is accepted-case sensitivity, positive coverage, or deferral-based sensitivity.
- Rename and implement the threshold rule accordingly.

## Documentation Issues

### D1. `Phase3/README.md` Is Out of Date

Severity: Low.

The README describes Phase 3A but does not mention the existing `Phase3/baseline/` pipeline. It should be updated after the Phase 3B fixes are completed.

### D2. Decorative Comment Characters Are Hard to Read in Some Consoles

Severity: Low.

Some box-drawing comments in baseline scripts and YAML comments may display incorrectly in PowerShell depending on code page. This does not affect execution.

## Final Recommendation

Use Phase 3A outputs as the current reliable foundation.

Before presenting Phase 3B as a formal result:

1. Fix uncertainty scoring.
2. Save validation uncertainty normalization statistics during training.
3. Use validation statistics during test evaluation.
4. Fix ECE.
5. Fix bootstrap CI behavior.
6. Regenerate `training_results.json`, `metrics.json`, `predictions.csv`, `triage_report.csv`, and baseline figures.

