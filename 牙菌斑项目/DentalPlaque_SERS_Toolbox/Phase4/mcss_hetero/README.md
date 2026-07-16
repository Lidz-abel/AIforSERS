# Phase 4D grouped MCSS heteroscedastic MIL

This implementation keeps the outer test patients locked while five OOF models
select the training duration, temperature, and accuracy threshold. The final
model is trained on all development patients and evaluates the locked test set
once.

The model returns a binary disease logit and a bounded log variance. Final
patient output separates MCSS sampling variability, heteroscedastic aleatoric
variability, and MC Dropout epistemic variability.

GPU workers:

```bash
CUDA_VISIBLE_DEVICES=6 python Phase4/mcss_hetero/phase4d_worker.py --worker gpu6
CUDA_VISIBLE_DEVICES=7 python Phase4/mcss_hetero/phase4d_worker.py --worker gpu7
```
