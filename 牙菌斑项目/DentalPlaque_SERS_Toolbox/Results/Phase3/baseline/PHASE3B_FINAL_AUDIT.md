# Phase3B Final Audit

**审计日期**: 2026-07-12
**审计状态**: ✅ 通过

---

## 1. 数据来源

| 项目 | 值 |
|------|-----|
| 数据集 | Phase3A_v1.0 (FROZEN) |
| 数据路径 | `Results/Phase3/dataset/spectra.npz` |
| 划分文件 | `Results/Phase3/splits/split_seed42.json` |
| 患者数 | 52 (阳性=31, 阴性=21) |
| 光谱数 | 1970 (阳性=1549, 阴性=421) |
| 波数点 | 732 (395.11–1840.62 cm⁻¹) |
| 预处理 | Per-spectrum SNV |
| 随机种子 | 42 |

## 2. 患者级数据泄露检查

```
train ∩ val  = 0  ✅
train ∩ test = 0  ✅
val ∩ test   = 0  ✅
```

无患者级数据泄露。Spectrum 展开在 patient split 之后执行。

## 3. 划分统计

| Split | 患者数 | 阳性 | 阴性 | 光谱数 |
|-------|--------|------|------|--------|
| Train | 31 | 18 | 13 | 1197 |
| Val | 10 | 6 | 4 | 381 |
| Test | 11 | 7 | 4 | 392 |

## 4. 模型配置

### Logistic Regression
- L2 正则化, `class_weight='balanced'`, `max_iter=5000`
- 超参数搜索: `C ∈ {0.001, 0.01, 0.1, 1.0, 10.0, 100.0}`
- CV: 5-fold GroupKFold
- 校准: Isotonic Regression

### XGBoost
- 超参数搜索: `n_est ∈ {50,100,200}`, `depth ∈ {2,3,4,5}`, `lr ∈ {0.01,0.05,0.1}`, `subsample ∈ {0.7,0.8,1.0}`, `colsample ∈ {0.7,0.8,1.0}`
- `scale_pos_weight = n_neg/n_pos` (自动)
- `early_stopping_rounds = 20` ✅ 已实现（通过 XGBClassifier 构造函数传入）
- 校准: Platt Scaling (manual implementation, sklearn 1.8 compatible)

## 5. Test Set 指标（11 名患者）

Bootstrap 95% CI (2000 次患者级重采样) 在括号中。

| 策略 | 模型 | ROC-AUC | Accuracy | Sens | Spec | Brier | ECE |
|------|------|---------|----------|------|------|-------|-----|
| Patient-Median | LR | 0.929 [0.78,1.00] | 0.818 | 1.000 | 0.500 | 0.101 | 0.061 |
| Patient-Median | XGBoost | 0.929 [0.69,1.00] | 0.727 | 0.714 | 0.750 | 0.147 | 0.190 |
| Spectrum-Aggregate | LR | 0.964 [0.80,1.00] | 0.818 | 0.857 | 0.750 | 0.103 | 0.174 |
| Spectrum-Aggregate | **XGBoost** | **0.964** [0.80,1.00] | **0.909** | **1.000** | **0.750** | **0.069** | **0.086** |

**最佳模型**: Spectrum-Aggregate + XGBoost

## 6. 最佳模型误诊分析

| 患者 | 真实 | P(阳性) | 预测 | 正确? | 分诊 |
|------|------|---------|------|-------|------|
| 阴性-_11 | 0 | 0.599 | 阳性 | ❌ | CT-推荐 |
| 阴性-_21 | 0 | 0.319 | 阴性 | ✅ | Confident |

- **仅 1 个错误**: 阴性-_11（假阳性）
- **阴性-_21 分类正确**（P=0.319，预测为阴性）

## 7. 已知限制

1. 仅 52 名患者（11 名测试），CI 较宽
2. Validation set 被多重使用（超参数选择、校准、uncertainty、triage）
3. 标签为 SERS 筛查标签，非独立临床金标准
4. 无 QC 过滤
5. 单中心、单仪器
6. 未做基线校正（与 MATLAB 流程不同）

## 8. 复现命令

```bash
cd Phase3/baseline
python build_features.py    # → features_patient_median.npz + features_spectrum_level.npz
python train_baseline.py    # → training_results.json + triage_thresholds.json + models/*.joblib
python evaluate_baseline.py # → predictions.csv + metrics.json + triage_report.csv + figures/
```

## 9. 审计结论

Phase3B 基线结果可复现（从零重建输出与报告一致）。无患者级数据泄露。当前结果应视为探索性 baseline，非临床可推广结论。下一步建议进入 Phase3C 稳定性验证（repeated splits / grouped CV）。
