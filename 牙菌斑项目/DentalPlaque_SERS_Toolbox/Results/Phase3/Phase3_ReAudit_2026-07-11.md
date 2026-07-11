# Phase 3 Re-Audit Report

Date: 2026-07-11

## Summary Verdict

Phase 3A remains correct and reproducible.

Phase 3B has improved since the previous audit:

- Validation uncertainty reference distributions are now saved during training.
- Test-time uncertainty now loads validation reference distributions instead of fitting references from the test set.
- ECE has been changed toward the standard observed-positive-rate definition.
- Bootstrap CI now passes `metric_name` and no longer skips single-class resamples for all metrics.
- Triage parameter naming was changed from `target_sensitivity` to `target_coverage`.

However, two metric issues still require correction before Phase 3B calibration and confidence intervals are used formally:

1. ECE currently excludes predictions exactly equal to 0.0 from all bins.
2. Sensitivity/specificity bootstrap intervals treat absent-class resamples as score 0.0 instead of skipping undefined resamples for that specific metric.

## Verified Correct Components

### Phase 3A Dataset

- Patients: 52
- Spectra: 1970
- Raman-shift points per spectrum: 732
- Patient labels: negative 21, positive 31
- Spectrum labels: negative 421, positive 1549
- Train/validation/test patients: 31/10/11
- Train/validation/test patient overlap: 0
- Unique split patients: 52
- SNV maximum absolute per-spectrum mean: 3.34e-07
- SNV standard-deviation range: 0.9999998-1.0000001

### Phase 3B Feature Construction

- Patient-median features: `X_patient = 52 x 732`
- Spectrum-level features: `X_spectra = 1970 x 732`
- Patient split masks have zero overlap.
- Spectrum split masks have zero overlap.
- Expanded spectrum counts: train 1197, validation 381, test 392

### Phase 3B Evaluation Re-run

`python Phase3\baseline\evaluate_baseline.py` was rerun successfully.

Generated/updated:

- `Results/Phase3/baseline/predictions.csv`
- `Results/Phase3/baseline/metrics.json`
- `Results/Phase3/baseline/triage_report.csv`
- `Results/Phase3/baseline/figures/calibration_curve.png`
- `Results/Phase3/baseline/figures/uncertainty_histogram.png`

Current patient-level test metrics:

| Strategy | Model | ROC-AUC | Accuracy | Sensitivity | Specificity | ECE |
|---|---:|---:|---:|---:|---:|---:|
| patient_median | logistic_regression | 0.9286 | 0.8182 | 1.0000 | 0.5000 | 0.0606 |
| patient_median | xgboost | 0.8571 | 0.7273 | 0.7143 | 0.7500 | 0.1946 |
| spectrum_level | logistic_regression | 0.9643 | 0.8182 | 0.8571 | 0.7500 | 0.1744 |
| spectrum_level | xgboost | 0.9643 | 0.9091 | 1.0000 | 0.7500 | 0.0863 |

These are patient-level metrics on 11 test patients.

## Previous Findings Status

### F1. Uncertainty Probability Direction

Status: functionally fixed, but comments remain confusing.

Sanity check:

- Probabilities near 0.5 now receive higher probability-only uncertainty than probabilities near 0 or 1.

Remaining cleanup:

- In `baseline_utils.py`, comments around `prob_conf` still say "high = certain" although `1 - 2*abs(P-0.5)` is highest at P=0.5. This is documentation confusion, not currently a numeric failure.

### F2. Test Set Used for Uncertainty Normalization

Status: fixed.

Evidence:

- `train_baseline.py` now saves `*_ref_stats.npz`.
- `training_results.json` includes `ref_stats_path`.
- `evaluate_baseline.py` loads `ref_stats_path` via `load_ref_stats`.

### F3. ECE Uses Classification Accuracy Instead of Positive Rate

Status: partially fixed.

Fixed:

- ECE now compares observed positive rate to mean predicted probability.

Remaining issue:

- The bin condition is currently `y_prob > lower` and `y_prob <= upper`.
- Predictions exactly equal to 0.0 are excluded from every bin.

Failing sanity check:

```text
expected_calibration_error(y_true=[1], y_prob=[0.0]) -> 0.0
expected value should be 1.0
```

Impact:

- ECE can be biased downward when calibrated models output exact 0.0 probabilities.
- Current predictions include exact 0.0 probabilities from isotonic calibration.

Required correction:

- Include the left edge in the first bin, e.g. first bin uses `>= 0.0`, or use `np.digitize`/`sklearn.calibration.calibration_curve`-compatible binning.

### F4. Bootstrap CI Single-Class Handling

Status: partially fixed.

Fixed:

- Bootstrap now receives `metric_name`.
- ROC-AUC skips single-class bootstrap samples.

Remaining issue:

- Sensitivity is undefined when a bootstrap sample contains no positive cases.
- Specificity is undefined when a bootstrap sample contains no negative cases.
- Current `sensitivity_score` and `specificity_score` return 0.0 when the denominator is absent.

Failing sanity checks:

```text
sensitivity_score(y_true=[0,0], y_pred=[0,1]) -> 0.0
specificity_score(y_true=[1,1], y_pred=[1,0]) -> 0.0
```

Impact:

- Sensitivity/specificity confidence intervals can be biased downward.

Required correction:

- For bootstrap CI:
  - skip resamples without positives when computing sensitivity CI;
  - skip resamples without negatives when computing specificity CI;
  - continue including all resamples for accuracy and Brier score.

### F5. Triage Threshold Semantics

Status: improved, but still heuristic.

Fixed:

- Config now uses `target_coverage` rather than `target_sensitivity`.
- The implementation now documents coverage of correctly classified positives.

Remaining limitation:

- On the current 11-patient test set, triage assigns nearly all patients to `confident`.
- This should be reported as an exploratory uncertainty heuristic, not as a validated clinical triage policy.

## New/Remaining Findings

### R1. ECE Exact-Zero Bin Exclusion

Severity: Medium.

Location:

- `Phase3/baseline/baseline_utils.py`, `expected_calibration_error`

Recommendation:

- Fix before reporting ECE values formally.

### R2. Sensitivity/Specificity Bootstrap Undefined-Class Handling

Severity: Medium.

Location:

- `Phase3/baseline/baseline_utils.py`, `bootstrap_metric_ci`
- `Phase3/baseline/baseline_utils.py`, `sensitivity_score`
- `Phase3/baseline/baseline_utils.py`, `specificity_score`

Recommendation:

- Fix before reporting sensitivity/specificity confidence intervals formally.

### R3. Phase 3 README Is Still Out of Date

Severity: Low.

Location:

- `Phase3/README.md`

Issue:

- It documents Phase 3A but does not describe the current `Phase3/baseline/` pipeline.

Recommendation:

- Update after the remaining metric fixes are complete.

## Final Recommendation

Phase 3A can be treated as complete.

Phase 3B is close, but before freezing baseline results:

1. Fix ECE first-bin inclusion.
2. Fix bootstrap handling for sensitivity and specificity.
3. Regenerate `metrics.json`, `predictions.csv`, `triage_report.csv`, and baseline figures.
4. Update `Phase3/README.md`.

