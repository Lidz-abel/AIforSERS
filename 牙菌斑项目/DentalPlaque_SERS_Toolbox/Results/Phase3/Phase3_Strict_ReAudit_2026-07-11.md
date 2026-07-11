# Phase 3 Strict Re-Audit Report

Date: 2026-07-11

## Verdict

Phase 3A is correct and can remain frozen.

Phase 3B is substantially improved. The previously identified ECE zero-bin issue and sensitivity/specificity absent-class return values are now fixed at the function level. The full evaluation script reruns successfully and regenerates current baseline outputs.

However, Phase 3B should still be reported as an exploratory baseline, not as a finalized clinical triage system. Two remaining issues affect interpretation rather than core patient-level classification.

## Commands Re-run

```text
python Phase3\baseline\evaluate_baseline.py
```

Generated/updated successfully:

- `Results/Phase3/baseline/predictions.csv`
- `Results/Phase3/baseline/metrics.json`
- `Results/Phase3/baseline/triage_report.csv`
- `Results/Phase3/baseline/figures/calibration_curve.png`
- `Results/Phase3/baseline/figures/uncertainty_histogram.png`

## Dataset Checks

- Patients: 52
- Spectra: 1970
- Features per spectrum: 732
- Patient labels: negative 21, positive 31
- Spectrum labels: negative 421, positive 1549
- Train/validation/test patient overlap: 0/0/0
- Unique split patients: 52
- SNV max absolute per-spectrum mean: 3.34e-07
- SNV standard-deviation range: 0.9999998-1.0000001

## Metric Sanity Checks

Passed:

```text
expected_calibration_error(y_true=[1], y_prob=[0.0]) = 1.0
expected_calibration_error(y_true=[0], y_prob=[0.0]) = 0.0
expected_calibration_error(y_true=[1], y_prob=[1.0]) = 0.0
sensitivity_score(no positive samples) = NaN
specificity_score(no negative samples) = NaN
```

Uncertainty direction sanity check:

```text
p = [0.01, 0.25, 0.50, 0.75, 0.99]
probability-only uncertainty = [0.4, 0.8, 1.0, 0.8, 0.4]
```

This confirms that probabilities near 0.5 are treated as more uncertain than probabilities near 0 or 1.

## Current Patient-Level Test Metrics

| Strategy | Model | ROC-AUC | Accuracy | Sensitivity | Specificity | ECE |
|---|---:|---:|---:|---:|---:|---:|
| patient_median | logistic_regression | 0.9286 | 0.8182 | 1.0000 | 0.5000 | 0.0606 |
| patient_median | xgboost | 0.8571 | 0.7273 | 0.7143 | 0.7500 | 0.1946 |
| spectrum_level | logistic_regression | 0.9643 | 0.8182 | 0.8571 | 0.7500 | 0.1744 |
| spectrum_level | xgboost | 0.9643 | 0.9091 | 1.0000 | 0.7500 | 0.0863 |

These are patient-level metrics on 11 test patients.

## Fixed Since Previous Re-Audit

### ECE zero-bin handling

Status: fixed.

Current binning includes `y_prob == 0.0` in the first bin and `y_prob == 1.0` in the last bin.

### Sensitivity/specificity absent-class behavior

Status: fixed at function level.

Current behavior:

- sensitivity is `NaN` when no positive samples exist;
- specificity is `NaN` when no negative samples exist;
- bootstrap CI skips `NaN` values.

### Test-set uncertainty normalization

Status: fixed.

Training saves validation reference distributions as `*_ref_stats.npz`; evaluation loads them through `ref_stats_path`.

## Remaining Findings

### R1. `triage_report.csv` writes undefined zone metrics as zeros

Severity: Medium.

Location:

- `Phase3/baseline/evaluate_baseline.py`, `zone_metrics` construction and `write_triage_report_csv`

Evidence:

- Empty zones and single-patient zones are exported as `0.0000` for accuracy, sensitivity, specificity, Brier score, and ECE.
- Example: most models have `review` and `ct_recommended` zones with `n_patients = 0`, but the CSV contains metric values `0.0000`.

Impact:

- A reader may interpret undefined metrics as true zero performance.

Required correction:

- Export undefined zone metrics as blank, `NA`, or `NaN`, not `0.0000`.
- For a non-empty single-class zone, compute only metrics that are mathematically defined and leave the others as `NA`.

### R2. XGBoost `early_stopping_rounds` is configured but not used

Severity: Low to Medium.

Location:

- `Phase3/baseline/baseline_config.yaml`
- `Phase3/baseline/baseline_utils.py`, `grid_search_xgb`

Evidence:

- Config contains `early_stopping_rounds: 20`.
- `grid_search_xgb` passes `eval_set=[(X_val, y_val)]`, but does not pass `early_stopping_rounds`.

Impact:

- The implementation does not match the configuration.
- Validation metrics are also used for XGBoost parameter selection, calibration, and triage thresholding, so validation metrics should be described as tuning/calibration metrics rather than independent validation performance.

Required correction:

- Either implement early stopping correctly for the installed XGBoost version, or remove the unused config field and update the documentation.

### R3. Uncertainty comments remain confusing

Severity: Low.

Location:

- `Phase3/baseline/baseline_utils.py`, `compute_uncertainty`

Issue:

- Comments around `prob_conf` still describe `1 - 2*abs(P - 0.5)` as confidence, but this value is highest at `P = 0.5`, so it is uncertainty-like.

Impact:

- Current numeric behavior passed sanity checks, but future maintainers could misread or accidentally reverse the logic again.

Required correction:

- Rename variables/comments to explicitly distinguish `prob_ambiguity = 1 - 2*abs(P - 0.5)` from confidence.

### R4. Phase 3 README is out of date

Severity: Low.

Location:

- `Phase3/README.md`

Issue:

- README still mainly documents Phase 3A and does not describe the current baseline pipeline.

Required correction:

- Add Phase 3B files, commands, outputs, and limitations.

## Figure Review

### Calibration curve

The calibration curve is generated correctly, but the test set has only 11 patients. The curve is visually jagged and should be presented as exploratory only.

### Uncertainty histogram

The histogram is generated correctly, but nearly all patients are assigned to `confident`. This means the current triage rule has little separation on this test set and should not be presented as a validated deferral policy.

## Final Recommendation

Do not freeze Phase 3B yet.

Before freezing:

1. Change `triage_report.csv` undefined zone metrics from `0.0000` to `NA`.
2. Resolve the unused XGBoost `early_stopping_rounds` config.
3. Clean uncertainty comments/naming.
4. Update `Phase3/README.md`.
5. Re-run `evaluate_baseline.py`.

Core patient-level baseline metrics are now usable as exploratory model comparison results, but triage-zone metrics and calibration/uncertainty figures should be interpreted cautiously.

