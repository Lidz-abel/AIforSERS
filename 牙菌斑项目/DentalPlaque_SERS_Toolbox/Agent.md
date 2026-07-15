# Agent Project Memory

Last updated: 2026-07-15

This file records the current project state, completed work, known issues, and long-term plan for the DentalPlaque SERS project. Treat it as the working memory for future agents. Do not use stale conclusions without checking the referenced result files.

## 1. Project Goal

Build a trustworthy SERS-based dental plaque/periodontal screening system.

The project direction has shifted from "train a classifier" to "build a clinically usable and statistically valid AI system":

- Predict patient-level binary disease status.
- Avoid patient-level data leakage.
- Report uncertainty and reliability, not just accuracy.
- Produce clinically interpretable confidence and routing suggestions.
- Use repeated patient-level validation rather than relying on one lucky split.

## 2. Dataset Status

Frozen dataset:

- File: `Results/Phase3/dataset/spectra.npz`
- Dataset version: Phase3A v1.0
- Task: binary classification
  - negative -> 0
  - positive -> 1
- Patients: 52
- Spectra: 1970
- Wavenumber points per spectrum: 732
- Patient-level distribution:
  - positive patients: 31
  - negative patients: 21
- Spectrum-level distribution:
  - positive spectra: 1549
  - negative spectra: 421

Dataset tensor definition:

```text
D = { (x_i, y_p, p_i) }

x_i: one SERS spectrum
y_p: label inherited from the corresponding patient
p_i: patient ID / patient index
```

Training may use single spectra as instances, but validation and testing must be patient-level. A spectrum-level random split is forbidden because it leaks patient identity across train/test.

Standard seed42 patient-level split:

```text
Train: 31 patients, 1197 spectra, 18 positive / 13 negative
Val:   10 patients, 381 spectra,  6 positive /  4 negative
Test:  11 patients, 392 spectra,  7 positive /  4 negative
```

No patient-level leakage was found in the seed42 split.

Generated dataset PDF:

- `Results/Phase3/DATASET_SUMMARY_REPORT.pdf`
- Generator script: `Phase3/make_dataset/generate_dataset_summary_pdf.py`
- Contents: dataset contribution summary, construction flow, class imbalance plots, patient split diagram, leakage rule, raw/SNV spectrum examples, modeling recommendations.

Note: temporary preview PNGs named `Results/Phase3/dataset_summary_preview*.png` may exist. They were generated only for visual checks and are not part of the formal deliverable.

## 3. Phase3 Baseline Summary

Phase3 established the classical baseline and validated the patient-level dataset design.

Most stable baseline observed:

```text
Spectrum-Aggregate + Logistic Regression
```

Across 20 patient-level repeated splits, this classical baseline was more stable than the initial deep learning model.

Known Phase3C approximate repeated-split result:

```text
AUC:        0.975 +/- 0.0376
Accuracy:   0.9273 +/- 0.0617
Sensitivity:0.9786 +/- 0.0510
Specificity:0.8375 +/- 0.1635
Brier:      0.0612
ECE:        0.1100
```

Interpretation:

- This remains the current performance reference.
- Any deep learning model should be compared against this baseline.
- Deep learning is not automatically better for this dataset because the true independent sample size is only 52 patients.

## 4. Phase4A Deep Learning Prototype

Current model family:

```text
CC-SERSNet v1
```

Architecture:

- Multi-scale 1D CNN branches with kernel sizes `[3, 7, 15]`.
- Branch channels: 32 each, concatenated to 96 channels.
- 1x1 projection to 64 channels.
- 2 residual blocks.
- GroupNorm instead of BatchNorm.
- Global average pooling.
- Dropout p=0.3.
- Linear classifier 64 -> 2.
- Approx. 57,474 parameters.

Inference:

- MC Dropout.
- Patient-level probability aggregation.
- Temperature scaling.
- Reliability metrics were explored.

Phase4A seed42 result looked promising but was not stable enough to be a final conclusion. It should be treated as an exploratory prototype only.

## 5. Phase4B v1 Findings

Old result directory:

```text
Results/Phase4/stability/
```

Status:

- Completed B0-B4 x 20 splits.
- Results showed poor stability and low specificity.
- Old results contained invalid negative temperature values.

Important: the old `Results/Phase4/stability/` results are stale and must not be used as final evidence.

Main findings from old Phase4B v1:

- CNN was unstable across patient splits.
- Specificity was often very low.
- Accuracy/balanced accuracy did not exceed the Phase3 classical baseline.
- Sensitivity-oriented thresholding pushed predictions toward positive and sacrificed specificity.
- Temperature scaling implementation was invalid because temperature could become negative.

Root causes identified:

- Only 52 independent patients, despite 1970 spectra.
- Spectrum-level CE training did not directly optimize patient-level prediction.
- Validation set has only 10 patients, making threshold tuning high variance.
- The model tended to over-predict positive.
- Preprocessing/QC may be insufficient for stable CNN learning.

## 6. Phase4B v2 Current Status

New result directory:

```text
Results/Phase4/stability_v2/
```

Major fixes already applied:

- Temperature scaling now uses `T = exp(log_T)` and clamps the effective range, so `T > 0`.
- Result directory changed from `stability` to `stability_v2` to avoid mixing old invalid results.
- Experiment matrix changed from sensitivity-oriented to accuracy-oriented.
- Added threshold strategies:
  - fixed 0.5
  - max balanced accuracy
  - max accuracy
  - max Youden
  - max balanced accuracy with specificity constraint
- `threshold_spec_constraint` is now supported and passed to `optimize_threshold`.
- `phase4b_run.py` summary CSV nested metric bug was fixed.
- `phase4b_analyze.py` report text was fixed and now targets `stability_v2`.

Current smoke-test status observed on 2026-07-15:

```text
B0: 6 split files
B1: 1 split file
B2: 1 split file
B3: 1 split file
B4: 1 split file
B5: 1 split file
B6: 1 split file
B7: 1 split file
```

Smoke-test split_00 results:

```text
B0: AUC=0.821, Acc=0.636, BalAcc=0.500, Sens=1.000, Spec=0.000, T=0.913, threshold=0.500
B1: AUC=1.000, Acc=0.909, BalAcc=0.875, Sens=1.000, Spec=0.750, T=0.050, threshold=0.440
B2: AUC=1.000, Acc=0.909, BalAcc=0.875, Sens=1.000, Spec=0.750, T=0.050, threshold=0.440
B3: AUC=0.821, Acc=0.636, BalAcc=0.661, Sens=0.571, Spec=0.750, T=0.913, threshold=0.580
B4: AUC=1.000, Acc=0.909, BalAcc=0.875, Sens=1.000, Spec=0.750, T=0.050, threshold=0.440
B5: AUC=1.000, Acc=0.909, BalAcc=0.875, Sens=1.000, Spec=0.750, T=0.050, threshold=0.280
B6: AUC=1.000, Acc=0.909, BalAcc=0.875, Sens=1.000, Spec=0.750, T=0.050, threshold=0.230
B7: AUC=0.964, Acc=0.909, BalAcc=0.875, Sens=1.000, Spec=0.750, T=0.050, threshold=0.420
```

Interpretation of smoke test:

- The positive-temperature fix worked.
- Accuracy-oriented thresholds improved split_00 compared with B0.
- Several experiments hit the lower temperature clamp `T=0.050`; calibration remains aggressive due to the tiny validation set.
- Smoke-test success does not prove full stability. Full repeated validation is still required.

## 7. Current Phase4B v2 Experiment Matrix

Current intended experiments:

```text
B0: patient-balanced CE, mean aggregation, fixed threshold=0.5, selection=val AUC
B1: patient-balanced CE, mean aggregation, threshold=max balanced accuracy, selection=val balanced accuracy
B2: patient-balanced CE, mean aggregation, threshold=max accuracy, selection=val accuracy
B3: patient-balanced CE, mean aggregation, threshold=max Youden, selection=val AUC
B4: patient-balanced CE, mean aggregation, threshold=max balanced accuracy with specificity >= 0.50, selection=val balanced accuracy
B5: patient-balanced CE, median aggregation, threshold=max balanced accuracy, selection=val balanced accuracy
B6: patient+class-balanced CE, mean aggregation, threshold=max balanced accuracy, selection=val balanced accuracy
B7: patient-balanced CE + label smoothing, mean aggregation, threshold=max balanced accuracy, selection=val balanced accuracy
```

Primary metrics:

- AUC
- Accuracy
- Balanced Accuracy
- Sensitivity
- Specificity
- Brier score
- ECE

For accuracy-focused work, do not judge by sensitivity alone. Specificity and balanced accuracy are critical.

## 8. Immediate Next Steps

1. Complete full Phase4B v2 run:

```powershell
python Phase4/stability/phase4b_run.py --resume
```

2. Generate final analysis:

```powershell
python Phase4/stability/phase4b_analyze.py
```

3. Verify after full run:

```text
Each of B0-B7 should have 20 split JSON files.
All temperatures should be > 0.
phase4b_summary.csv should contain non-NA AUC/Accuracy/BalAcc values.
PHASE4B_REPORT.md should be readable Chinese/English, not mojibake.
No old Results/Phase4/stability results should be mixed into stability_v2.
```

4. Compare Phase4B v2 against Phase3C classical baseline. If deep learning remains worse, explicitly state that the current CNN is not the final model.

## 9. Long-Term Plan

### Stage A: Freeze Data and Baselines

- Keep Phase3A dataset frozen.
- Maintain patient-level splits only.
- Keep Phase3C classical baseline as the reference.
- Build all future comparisons against repeated patient-level validation.

### Stage B: Finish Phase4B v2

- Complete B0-B7 x 20 repeated splits.
- Identify the best threshold/aggregation/loss strategy for accuracy and balanced accuracy.
- Decide whether CC-SERSNet v1 is worth retaining as a baseline only.

### Stage C: Move Beyond Spectrum-Level CNN

If Phase4B v2 remains unstable, move to patient-level learning:

- Multiple Instance Learning (MIL).
- Attention pooling over spectra from the same patient.
- Patient-level loss rather than only spectrum-level CE.
- Patient-level calibration and thresholding.

Recommended direction:

```text
Patient -> bag of spectra -> spectral encoder -> attention/MIL pooling -> patient logits -> patient-level loss
```

This better matches the clinical decision unit.

### Stage D: Improve Preprocessing and QC

Explore whether performance improves with:

- Baseline correction.
- Spike removal.
- SNR / saturation / peak-count QC.
- Peak-region features.
- Hybrid raw + engineered spectral features.

This should be versioned as a new dataset variant, not silently modifying Phase3A.

### Stage E: Reliability and Clinical Confidence

After classification stabilizes:

- Keep MC Dropout for model uncertainty.
- Use predictive entropy, expected entropy, mutual information, margin, patient agreement.
- Keep clinical confidence rule-based first.
- Do not overclaim clinical validity without external validation.

### Stage F: Paper/Report Framing

The strongest scientific story is not "CNN beats everything" unless the data supports it.

Better framing:

```text
We construct a leakage-safe patient-level SERS dataset and show that repeated patient-level validation is essential. Classical aggregate baselines are strong under small-patient regimes. We then explore trustworthy deep learning and identify that patient-level modeling is required for clinically meaningful deployment.
```

## 10. Non-Negotiable Rules

- Never use spectrum-level random train/test split.
- Never tune thresholds on test data.
- Never report only one lucky seed as final evidence.
- Never mix `Results/Phase4/stability/` old results with `Results/Phase4/stability_v2/`.
- Always report patient-level metrics.
- Always include specificity and balanced accuracy when optimizing accuracy.
- Treat clinical confidence as exploratory unless clinically validated.
