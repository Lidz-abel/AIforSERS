# Phase 4C MCSS-MIL

This directory contains the first implementation of MCSS-based deep learning.
It intentionally does not define an experiment matrix.

Core rule:

- patient-level split first
- MCSS bags are sampled only within each split
- validation/test use fixed bags
- threshold and temperature are fitted only on validation patients
- test patients are never used for training, model selection, thresholding, or calibration

Default single-split command:

```bash
python Phase4/mcss/mcss_train.py --split_seed 42
```

Main files:

- `phase4c_config.yaml`: one default MCSS-GatedAttention-MIL setup
- `mcss_dataset.py`: MCSS bag sampling after patient-level split
- `mcss_models.py`: multi-scale spectral encoder plus gated attention MIL
- `mcss_train.py`: train and evaluate one patient-level split
