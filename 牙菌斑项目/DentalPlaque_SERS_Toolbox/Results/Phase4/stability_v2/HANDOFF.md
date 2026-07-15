# Phase4B v2 — Handoff Document

**Date**: 2026-07-15
**Status**: B0-B3 complete (20 splits each), B4 2/20, B5-B7 not started

## Quick Start (GPU machine)

```bash
# 1. Copy entire project to new machine
# 2. cd to toolbox root
cd DentalPlaque_SERS_Toolbox

# 3. Verify GPU is detected
python -c "import torch; print(torch.cuda.is_available())"
# Must print: True

# 4. Install deps if needed
pip install torch numpy scipy scikit-learn pyyaml matplotlib

# 5. Complete remaining experiments (B4 then B5-B7)
python Phase4/stability/phase4b_run.py --exp B4 --resume
python Phase4/stability/phase4b_run.py --exp B5 --resume --start_seed 1
python Phase4/stability/phase4b_run.py --exp B6 --resume --start_seed 1
python Phase4/stability/phase4b_run.py --exp B7 --resume --start_seed 1

# 6. Generate final analysis
python Phase4/stability/phase4b_analyze.py
```

## Current State

| Experiment | Splits Done | Remaining | Notes |
|-----------|------------|-----------|-------|
| B0 | 20/20 | 0 | Complete |
| B1 | 20/20 | 0 | Complete |
| B2 | 20/20 | 0 | Complete |
| B3 | 20/20 | 0 | 1 split may have timed out (split_07) |
| B4 | 2/20 | 18 | Only split_00 done |
| B5 | 1/20 | 19 | Only split_00 (smoke test) |
| B6 | 1/20 | 19 | Only split_00 (smoke test) |
| B7 | 1/20 | 19 | Only split_00 (smoke test) |

Results are in `Results/Phase4/stability_v2/splits/{EXP}/split_XX.json`.

## Experiment Matrix (from phase4b_config.yaml)

| ID | Loss | Agg | Threshold Strategy | Selection | Constraint |
|----|------|-----|-------------------|-----------|------------|
| B0 | patient_balanced_ce | mean | fixed_0.5 | val_auc | none |
| B1 | patient_balanced_ce | mean | max_balanced_accuracy | val_bal_acc | none |
| B2 | patient_balanced_ce | mean | max_accuracy | val_accuracy | none |
| B3 | patient_balanced_ce | mean | max_youden | val_auc | none |
| B4 | patient_balanced_ce | mean | max_balanced_accuracy | val_bal_acc | spec≥0.50 |
| B5 | patient_balanced_ce | median | max_balanced_accuracy | val_bal_acc | none |
| B6 | patient+class_balanced_ce | mean | max_balanced_accuracy | val_bal_acc | none |
| B7 | patient_balanced_ce + LS(0.1) | mean | max_balanced_accuracy | val_bal_acc | none |

## Key Fixes in v2

1. **Temperature scaling**: T = exp(log_T), guaranteed > 0, clamped to [0.05, 10]
2. **No sensitivity ≥ 0.90 constraint** on any experiment (was the main bug in v1)
3. **Accuracy-oriented thresholds**: max_balanced_accuracy, max_accuracy, max_youden
4. **Seed separation**: split_seed controls patient split, model_seed = split_seed + 1000
5. **Results in stability_v2/**: no mixing with old negative-T results

## Preliminary Results (B0-B2, 20 splits)

| Exp | AUC | BalAcc | Spec |
|-----|-----|--------|------|
| B0 | 0.845±0.155 | 0.704±0.172 | 0.537±0.431 |
| B1 | 0.873±0.186 | 0.754±0.177 | 0.700±0.288 |
| B2 | 0.855±0.194 | 0.747±0.172 | 0.637±0.339 |

B1 (max_balanced_accuracy) currently leads on BalAcc and Specificity.

## Files You Need

```
DentalPlaque_SERS_Toolbox/
├── Phase4/
│   ├── stability/
│   │   ├── phase4b_run.py          ← orchestrator
│   │   ├── phase4b_train.py        ← single split trainer
│   │   ├── phase4b_analyze.py      ← results analysis + report
│   │   ├── phase4b_config.yaml     ← experiment definitions
│   │   └── phase4b_utils.py        ← threshold, aggregation, split
│   └── deep_learning/
│       ├── reliability.py          ← TemperatureScaling (fixed)
│       ├── models.py               ← MultiScaleEncoder
│       ├── datasets.py             ← SpectrumDataset
│       └── train.py                ← (Phase4A, not used for 4B)
├── Phase3/
│   └── baseline/
│       └── baseline_utils.py       ← binomial_ci, ECE, write_json
└── Results/
    └── Phase4/
        └── stability_v2/
            ├── splits/             ← per-split JSON results
            └── HANDOFF.md          ← this file
```

## Expected GPU Runtime

With GPU, each split should take ~30-90 seconds (vs ~5-7 min on CPU).
B4 (18) + B5 (19) + B6 (19) + B7 (19) = 75 remaining splits ≈ **1-2 hours on GPU**.
