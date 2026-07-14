# Phase 4A: CC-SERSNet v1.1 — MC Dropout Deep Learning Baseline

**版本**: 1.1
**日期**: 2026-07-13
**状态**: 探索性内部验证（非临床外部验证）

**v1.1 变更** (2026-07-13):
- 修复可复现性: 设置 torch/numpy/random/DataLoader 随机种子
- Patient-balanced training: 每光谱权重 = 1/该患者光谱数，消除患者间不平衡
- 校准: Temperature 在 MC posterior 均值 logits 上拟合（与测试时一致）
- 标记 v2 (全批量 SGD) 为失败实验
- 临床置信度添加探索性警告
- 术语修正: "aleatoric" → "Data Ambiguity / Intrinsic Predictive Ambiguity proxy"

---

## 1. 概述

CC-SERSNet v1 是 Phase 4 的第一个深度学习模型，采用 Multi-scale 1D CNN + MC Dropout 架构。目标是提供带不确定性量化的深度学习基线，输出患者级预测概率、可靠性指标和临床决策建议。

## 2. 数据

与 Phase 3B 完全一致：
- **数据集**: Phase3A_v1.0 (FROZEN)
- **患者**: 52 (阳性=31, 阴性=21)
- **光谱**: 1970 (阳性=1549, 阴性=421)
- **波数点**: 732 (395.11–1840.62 cm⁻¹)
- **预处理**: Per-spectrum SNV
- **划分**: seed=42, stratified patient-level 60/20/20

## 3. 患者级数据泄露检查

```
train ∩ val  = 0  ✓
train ∩ test = 0  ✓
val ∩ test   = 0  ✓
```

训练单位为单条光谱（标签继承自患者），评估单位为患者。无患者级数据泄露。

## 4. 模型架构

```
Input [B, 1, 732]
  ├── Conv1D(k=3, out=32) ─┐
  ├── Conv1D(k=7, out=32) ─┼── Concat [B, 96, 732]
  └── Conv1D(k=15, out=32) ─┘
      ↓
  Conv1D(k=1) → [B, 64, 732]  (projection)
      ↓
  ResidualBlock(k=3) × 2
      ↓
  Global Average Pooling → [B, 64]
      ↓
  Dropout(p=0.3)
      ↓
  Linear → logits [B, 2]
```

**关键设计选择**:
- **GroupNorm** 替代 BatchNorm：避免 MC Dropout 推理时 BatchNorm 统计混乱
- **Multi-scale kernels** [3, 7, 15]：捕获不同宽度的拉曼特征峰
- **57,474 个参数**：轻量模型，适合 52 患者的小样本

## 5. 训练配置

| 参数 | 值 |
|------|-----|
| Loss | CrossEntropyLoss (patient-balanced weights: 每光谱权重=1/该患者光谱数，缓解患者间光谱数不均，非完全类别平衡) |
| Optimizer | AdamW (lr=0.001, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=10) |
| Batch size | 64 |
| Early stopping | patience=20 (monitor val_patient_AUC) |
| 最佳 epoch | 23 (val patient AUC=1.000) |
| 随机种子 | torch/numpy/random/DataLoader 均固定为 42 |

## 6. MC Dropout 推理

- **T = 50** 次前向传播（dropout 保持启用）
- 每条光谱得到 50 组 softmax 后验概率
- 不确定性分解：
  - **Predictive Entropy** H(E[p]): 总预测不确定性
  - **Expected Entropy** E[H(p)]: 数据固有模糊性 / 内在预测模糊性代理指标 (Data Ambiguity / Intrinsic Predictive Ambiguity proxy)
  - **Mutual Information** H(E[p]) - E[H(p)]: 模型不确定性（epistemic）

## 7. 患者级聚合

每名患者的光谱级结果聚合为：
- `prob_positive`: 所有光谱概率的均值
- `patient_agreement`: 光谱预测与患者级预测一致的占比
- `entropy_mean`, `mi_mean`, `margin_mean`: 可靠性指标均值

## 8. 校准

- **方法**: Temperature Scaling（在 validation MC posterior 患者级均值 logits 上拟合，更接近但非完全一致于测试管线——因 softmax 与 mean 不可交换）
- **温度 T**: 0.6028
- **注**: T < 1 表明 MC posterior 均值 logits 方差较大（模型欠自信），temperature 将其锐化以改善校准。这与 v1.0 中确定性 logits 的 T=1.31（过自信）方向相反，体现了 MC 推理对不确定性估计的影响。

## 9. 临床置信度（Rule-based）

**WARNING: 探索性指标，非经验证的临床阈值。** 权重和阈值为预设启发式值，未在独立数据或临床结局上进行校准。不可用于实际临床决策。

四个归一化组件的加权和 ∈ [0, 1]：

| 组件 | 权重 | 含义 |
|------|------|------|
| Probability margin | 0.30 | \|2P-1\|, 远离 0.5 = 高置信 |
| Normalized entropy | 0.25 | 1 - H/log(2), 低熵 = 高置信 |
| Normalized MI | 0.20 | 1 - MI/MI_max, 低模型不确定 = 高置信 |
| Patient agreement | 0.25 | 光谱间一致比例 |

**阈值**:
- High: ≥ 0.75 → "Report"
- Medium: 0.50–0.75 → "Doctor Review"
- Low: < 0.50 或 MI > 0.3 → "Further Examination"

## 10. Test Set 结果（11 名患者）

Bootstrap 95% CI 在括号中。

| 指标 | Phase4A (单次 seed=42) | Phase3C Spect+LR (20 splits mean ± std) |
|------|----------------------|----------------------------------------|
| **ROC-AUC** | 0.964 [0.80, 1.00] | 0.975 ± 0.038 |
| **Accuracy** | 0.909 [0.73, 1.00] | 0.927 ± 0.062 |
| **Sensitivity** | **1.000** [0.646, 1.000] — 7/7 | 0.979 ± 0.051 |
| **Specificity** | 0.750 [0.301, 0.954] — 3/4 | 0.838 ± 0.164 |
| **Brier** | 0.082 | 0.061 ± 0.037 |
| **ECE** | 0.181 | 0.110 ± 0.049 |

**注**：Phase4A 为单次 seed=42 划分，Phase3C 为 20 次划分均值。两者不能直接比较——Phase4A 需经过 Phase4B 稳定性验证后才能公平对比。v1.1 sensitivity=1.000 优于 Phase3C 均值但 specificity=0.750 略低，整体 AUC 接近（0.964 vs 0.975）。

### 临床分流结果

| 分流建议 | 患者数 |
|---------|--------|
| Report（可直接报告） | 7 |
| Doctor Review（医生复核） | 3 |
| Further Examination（进一步检查） | 1 |

**注**: 唯一错误（阴性-_21, false positive, P+=0.716）落入 "Doctor Review" 而非 "Further Examination"。临床置信度评分未将此误诊标记为低置信——分流规则仍是探索性的，不可依赖其可靠性分离。

## 11. 与 Phase3B 对比

| 维度 | Phase3C (Spect+LR, 20 splits) | Phase4A v1.1 (CC-SERSNet, 单次) |
|------|-------------------|---------------------|
| 模型类型 | 线性 | 深度学习 (CNN) |
| AUC | 0.975 ± 0.038 | 0.964 (单次) |
| Sensitivity | 0.979 ± 0.051 | 1.000 (单次) |
| Specificity | 0.838 ± 0.164 | 0.750 (单次) |
| 不确定性量化 | 无 | MC Dropout |
| 临床分流 | 3-zone triage | 3-level recommendation |
| 可靠性指标 | 基础 | Entropy/MI/Margin/Agreement |
| 校准 | Isotonic | Temperature Scaling (MC posterior) |
| 参数量 | 732 | 57,474 |
| 稳定性验证 | Phase3C (20 splits) | 待做 (Phase4B) |
| 训练平衡 | class_weight='balanced' | patient-balanced sample weights |

## 12. 已知局限

1. **单次 split**: 未做 Phase3C 式的多次划分稳定性验证（Phase4B 待做）
2. **样本量**: 仅 52 患者（11 测试），CI 宽
3. **类别不平衡**: 训练集光谱正负比约 3.6:1，已通过 patient-balanced weighting 缓解（v1.1 修复）
4. **校准**: Temperature=0.60 < 1（MC posterior 欠自信，需锐化），与 v1.0 确定性校准的 T=1.31 方向相反，体现了 MC 推理对不确定性估计的影响
5. **GroupNorm 替代 BatchNorm**: 可能不如 BatchNorm 在训练时的效果
6. **临床阈值**: 权重(0.30/0.25/0.20/0.25)和阈值(0.75/0.50)为预设启发式值，未在独立数据上验证，不可用于实际临床决策
7. **标签**: 仍为 SERS 筛查标签，非独立临床金标准
8. **无 QC 过滤**: 与 Phase3 一致
9. **v2 实验失败**: 全批量 SGD + weight_decay=0.01 未学习（val AUC=0.792，全部预测阳性），已放弃该方向

## 13. 文件清单

| 文件 | 说明 |
|------|------|
| `config.yaml` | 所有配置参数 |
| `datasets.py` | 数据加载、split 审计 |
| `models.py` | CC-SERSNet-v1 架构 |
| `reliability.py` | 可靠性指标、聚合、校准、临床决策 |
| `train.py` | 训练 + 评估主脚本 |
| `model.pt` | 训练好的模型权重 |
| `training_log.csv` | 逐 epoch 指标 |
| `predictions.csv` | 患者级预测 |
| `reliability_patient.csv` | 患者级可靠性指标 |
| `metrics.json` | Bootstrap CI 指标 |
| `figures/` | 训练曲线、校准曲线、置信度分布 |

## 14. 结论

CC-SERSNet v1.1 在 seed=42 的单次划分上达到 test AUC=0.964、sensitivity=1.000 (7/7)、specificity=0.750 (3/4)。与 Phase3C 中 Spect+LR 的 20 次划分均值（AUC=0.975, sens=0.979, spec=0.838）相比：AUC 接近，sensitivity 更高，specificity 略低。Patient-balanced training 使模型偏向更高的 sensitivity——该 trade-off 与筛查任务中优先降低漏诊的目标方向一致，但尚未经过临床验证。

v1.1 关键改进:
- **可复现性**: 固定所有随机种子，结果可精确复现
- **Patient-balanced training**: 每患者等权重贡献，缓解光谱数不均衡
- **MC校准一致性**: Temperature 在 MC posterior 上拟合，与测试管线一致

当前结论：对于 52 患者的小样本，深度学习可以达到与简单线性模型相近的性能，同时提供更丰富的不确定性量化。下一步应进行 Phase4B 稳定性验证（多次划分/LOPO-CV）。

**This is an exploratory internal validation. No external clinical validation yet.**
