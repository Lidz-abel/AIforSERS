# Phase 4B: Stability Validation & Ablation Study

**日期**: 2026-07-15
**状态**: 探索性内部验证 (Phase4B v2 — accuracy-oriented, no forced sensitivity)

## 1. 概述

Phase4B v2 在 20 次 repeated patient-level split 上验证 CC-SERSNet v1 的稳定性。
v2 重新设计实验矩阵，移除 sensitivity ≥ 0.90 强制约束，
改为以 accuracy/balanced accuracy 为导向的阈值策略和模型选择标准。

## 2. 实验矩阵

| ID | Loss | Aggregation | Threshold | Selection | Constraint |
|----|------|------------|-----------|-----------|------------|
| B0 | Patient-balanced CE | mean | 0.5 (fixed) | val AUC | none |
| B1 | Patient-balanced CE | mean | max BalAcc | val BalAcc | none |
| B2 | Patient-balanced CE | mean | max Accuracy | val Accuracy | none |
| B3 | Patient-balanced CE | mean | max Youden | val AUC | none |
| B4 | Patient-balanced CE | mean | max BalAcc | val BalAcc | spec≥0.5 |
| B5 | Patient-balanced CE | median | max BalAcc | val BalAcc | none |
| B6 | Patient-balanced CE + class-balanced | mean | max BalAcc | val BalAcc | none |
| B7 | Patient-balanced CE + LS(0.1) | mean | max BalAcc | val BalAcc | none |

## 3. 总体结果 (mean ± std across 20 splits)

| Experiment | AUC | Bal Acc | Accuracy | Sensitivity | Specificity | Brier | ECE |
|-----------|-----|---------|----------|-------------|-------------|-------|-----|
| B0 | 0.845 ± 0.155 | 0.704 ± 0.172 | 0.750 ± 0.128 | 0.871 ± 0.207 | 0.537 ± 0.431 | 0.170 ± 0.069 | 0.174 ± 0.101 |
| B1 | 0.873 ± 0.186 | 0.754 ± 0.177 | 0.768 ± 0.173 | 0.807 ± 0.228 | 0.700 ± 0.288 | 0.148 ± 0.089 | 0.214 ± 0.078 |
| B2 | 0.855 ± 0.194 | 0.747 ± 0.172 | 0.777 ± 0.146 | 0.857 ± 0.173 | 0.637 ± 0.339 | 0.149 ± 0.081 | 0.205 ± 0.069 |
| B3 | 0.847 ± 0.120 | 0.694 ± 0.166 | 0.727 ± 0.166 | 0.816 ± 0.282 | 0.571 ± 0.345 | 0.186 ± 0.077 | 0.180 ± 0.096 |
| B4 | 1.000 ± nan | 0.875 ± nan | 0.909 ± nan | 1.000 ± nan | 0.750 ± nan | 0.050 ± nan | 0.126 ± nan |
| B5 | 1.000 ± nan | 0.875 ± nan | 0.909 ± nan | 1.000 ± nan | 0.750 ± nan | 0.052 ± nan | 0.084 ± nan |
| B6 | 1.000 ± nan | 0.875 ± nan | 0.909 ± nan | 1.000 ± nan | 0.750 ± nan | 0.026 ± nan | 0.110 ± nan |
| B7 | 0.964 ± nan | 0.875 ± nan | 0.909 ± nan | 1.000 ± nan | 0.750 ± nan | 0.079 ± nan | 0.090 ± nan |

## 4. Win Counts (best per split)

| Experiment | AUC | Bal Acc | Sensitivity | Specificity | Total |
|-----------|-----|---------|-------------|-------------|-------|
| B0 | 10 | 9 | 15 | 11 | 45 |
| B1 | 9 | 9 | 3 | 9 | 30 |
| B2 | 1 | 2 | 1 | 0 | 4 |
| B3 | 0 | 0 | 1 | 0 | 1 |
| B4 | 0 | 0 | 0 | 0 | 0 |
| B5 | 0 | 0 | 0 | 0 | 0 |
| B6 | 0 | 0 | 0 | 0 | 0 |
| B7 | 0 | 0 | 0 | 0 | 0 |

## 5. Paired Differences vs B0

| Exp | Δ AUC | Δ Bal Acc | Δ Sens | Δ Spec | Δ Brier | Δ ECE |
|-----|-------|-----------|--------|--------|---------|-------|
| B1 | +0.0286 | +0.0491 | -0.0643 | +0.1625 | -0.0218 | +0.0405 |
| B2 | +0.0107 | +0.0429 | -0.0143 | +0.1000 | -0.0208 | +0.0312 |
| B3 | +0.0000 | +0.0357 | +0.0000 | +0.0714 | +0.0000 | +0.0000 |

## 6. 结论

Phase4B 通过 20 次 repeated split 验证了 CC-SERSNet v1 的稳定性，
并比较了不同损失函数、聚合方法、阈值策略和模型选择标准对患者级性能的影响。

**This is an exploratory internal validation. No external clinical validation yet.**