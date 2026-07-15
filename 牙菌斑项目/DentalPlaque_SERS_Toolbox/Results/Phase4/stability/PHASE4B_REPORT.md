# Phase 4B: Stability Validation & Ablation Study

**日期**: 2026-07-13
**状态**: 探索性内部验证

## 1. 概述

Phase4B 在 20 次 repeated patient-level split 上验证 CC-SERSNet v1 的稳定性，
并对损失函数、患者聚合方法、决策阈值和模型选择标准进行系统消融实验。

## 2. 实验矩阵

| ID | Loss | Aggregation | Threshold | Selection |
|----|------|------------|-----------|-----------|
| B0 | Patient-balanced CE | mean | 0.5 (fixed) | val AUC |
| B1 | Patient-balanced CE | mean | max BalAcc (sens≥0.90) | val BalAcc (sens≥0.90) |
| B2 | Patient-balanced CE | median | max BalAcc (sens≥0.90) | val BalAcc (sens≥0.90) |
| B3 | Patient-balanced CE + class-balanced | mean | max BalAcc (sens≥0.90) | val BalAcc (sens≥0.90) |
| B4 | Patient-balanced CE + LS(0.1) | mean | max BalAcc (sens≥0.90) | val BalAcc (sens≥0.90) |

## 3. 总体结果 (mean ± std across 20 splits)

| Experiment | AUC | Bal Acc | Accuracy | Sensitivity | Specificity | Brier | ECE |
|-----------|-----|---------|----------|-------------|-------------|-------|-----|
| B0 | 0.795 ± 0.233 | 0.662 ± 0.201 | 0.709 ± 0.158 | 0.836 ± 0.224 | 0.487 ± 0.462 | 0.186 ± 0.066 | 0.191 ± 0.119 |
| B1 | 0.791 ± 0.234 | 0.671 ± 0.192 | 0.745 ± 0.143 | 0.943 ± 0.097 | 0.400 ± 0.392 | 0.179 ± 0.083 | 0.219 ± 0.137 |
| B2 | 0.677 ± 0.340 | 0.615 ± 0.167 | 0.705 ± 0.114 | 0.943 ± 0.117 | 0.287 ± 0.391 | 0.193 ± 0.071 | 0.205 ± 0.144 |
| B3 | 0.688 ± 0.313 | 0.641 ± 0.186 | 0.714 ± 0.145 | 0.907 ± 0.149 | 0.375 ± 0.385 | 0.193 ± 0.064 | 0.231 ± 0.102 |
| B4 | 0.723 ± 0.295 | 0.612 ± 0.166 | 0.700 ± 0.118 | 0.936 ± 0.098 | 0.287 ± 0.365 | 0.196 ± 0.063 | 0.222 ± 0.112 |

## 4. Win Counts (best per split)

| Experiment | AUC | Bal Acc | Sensitivity | Specificity | Total |
|-----------|-----|---------|-------------|-------------|-------|
| B0 | 11 | 11 | 11 | 11 | 44 |
| B1 | 5 | 5 | 6 | 6 | 22 |
| B2 | 1 | 0 | 2 | 0 | 3 |
| B3 | 2 | 4 | 0 | 3 | 9 |
| B4 | 1 | 0 | 1 | 0 | 2 |

## 5. Paired Differences vs B0

| Exp | Δ AUC | Δ Bal Acc | Δ Sens | Δ Spec | Δ Brier | Δ ECE |
|-----|-------|-----------|--------|--------|---------|-------|
| B1 | -0.0036 | +0.0098 | +0.1071 | -0.0875 | -0.0069 | +0.0279 |
| B2 | -0.1179 | -0.0464 | +0.1071 | -0.2000 | +0.0071 | +0.0140 |
| B3 | -0.1071 | -0.0205 | +0.0714 | -0.1125 | +0.0075 | +0.0406 |
| B4 | -0.0714 | -0.0500 | +0.1000 | -0.2000 | +0.0102 | +0.0314 |

## 6. 结论

Phase4B 通过 20 次 repeated split 验证了 CC-SERSNet v1 的稳定性，
并比较了不同损失函数、聚合方法、阈值策略和模型选择标准对患者级性能的影响。

**This is an exploratory internal validation. No external clinical validation yet.**