# Phase 3C：稳定性验证报告

**日期**: 2026-07-12
**状态**: 完成

---

## 1. 目的

Phase3B 在单个 seed=42 的 split 上得到 spectrum_level__xgboost 最优。Phase3C 通过 20 次随机患者级划分，检验该结论是否稳定，还是单次 split 的偶然结果。

## 2. 方法

- **划分方式**: 每次使用不同随机种子，执行患者级 stratified 60/20/20 划分
- **超参数**: 固定为 Phase3B 最佳参数（测试的是**数据划分稳定性**，非超参数稳定性）
- **重复次数**: 20 次（种子 42, 52, 62, ..., 232）
- **评估**: 每次在 test set 上计算患者级指标，汇总 mean ± std

## 3. 稳定性结果（20 次随机划分）

| 组合 | ROC-AUC | Accuracy | Sensitivity | Specificity | Brier ↓ | ECE ↓ |
|------|---------|----------|-------------|-------------|---------|-------|
| Median+LR | 0.902 ± 0.073 | 0.845 ± 0.122 | 0.850 ± 0.189 | 0.838 ± 0.182 | 0.104 ± 0.058 | 0.124 ± 0.083 |
| Median+XGB | 0.868 ± 0.148 | 0.764 ± 0.133 | 0.814 ± 0.212 | 0.675 ± 0.211 | 0.178 ± 0.091 | 0.224 ± 0.099 |
| **Spect+LR** | **0.975 ± 0.038** | **0.927 ± 0.062** | **0.979 ± 0.051** | **0.838 ± 0.164** | **0.061 ± 0.037** | **0.110 ± 0.049** |
| Spect+XGB | 0.954 ± 0.088 | 0.882 ± 0.112 | 0.964 ± 0.077 | 0.738 ± 0.256 | 0.087 ± 0.051 | 0.161 ± 0.045 |

**粗体** = 每列最佳 mean。

## 4. 最佳模型出现频率（按 ROC-AUC）

| 组合 | 获胜次数 | 占比 |
|------|---------|------|
| **Spectrum-Aggregate + Logistic Regression** | **8/20** | **40%** |
| Patient-Median + Logistic Regression | 6/20 | 30% |
| Patient-Median + XGBoost | 3/20 | 15% |
| Spectrum-Aggregate + XGBoost | 3/20 | 15% |

## 5. 关键发现

### 5.1 Phase3B 的 seed=42 结果存在明显的 split 依赖

- Phase3B 中 Spectrum+XGBoost 以 AUC=0.964 排名第一
- 但在 20 次随机划分中，Spectrum+XGBoost **仅赢了 3/20 次（15%）**
- 因此，seed=42 的单次结论不能作为“XGBoost 稳定最优”的证据

### 5.2 当前 20 次划分中最稳定的模型是 Spectrum-Aggregate + Logistic Regression

| 指标 | Spect+LR | Spect+XGB | 差值 |
|------|----------|-----------|------|
| Mean AUC | **0.975** | 0.954 | +0.021 |
| AUC Std | **0.038** | 0.088 | — (LR 波动不到 XGB 的一半) |
| Mean Accuracy | **0.927** | 0.882 | +0.045 |
| Mean Sensitivity | **0.979** | 0.964 | +0.015 |
| Specificity Std | **0.164** | 0.256 | — (LR 的特异度更稳定) |

Spect+LR 在主要指标均值上优于 Spect+XGB，且多数指标的标准差更低（更稳定）。但这仍是基于 20 次内部随机划分的结果，不能替代独立外部验证。

### 5.3 Spectrum-Aggregate 策略在均值上优于 Patient-Median

无论用哪个模型，Spectrum-Aggregate 策略的 mean AUC 都高于 Patient-Median：
- Spect+LR (0.975) vs Median+LR (0.902): +0.073
- Spect+XGB (0.954) vs Median+XGB (0.868): +0.086

逐 seed 配对比较显示，Spectrum-Aggregate 并非每一次都优于 Patient-Median：LR 下为 12 胜 / 7 平 / 1 负，XGBoost 下为 11 胜 / 3 平 / 6 负。因此更严谨的结论是：**患者内多条光谱的聚合信息可能提供额外诊断信息，但仍需在更大样本和外部测试集中确认**。

### 5.4 Sensitivity 表现

Spect+LR 的平均灵敏度为 0.979 ± 0.051，说明在当前 20 次内部随机划分中阳性患者识别较稳定。由于每次测试集仅约 11 名患者，这一结果仍应视为探索性筛查性能，而不是临床“低漏诊率”的证明。

## 6. 对 Phase3B 报告的修正

| Phase3B 原结论 | Phase3C 修正 |
|----------------|-------------|
| "最佳模型是 Spectrum+XGBoost" | 在 seed=42 是。但 20 次划分中 Spect+LR 更稳定（8/20 胜） |
| "XGBoost 在准确率上略优" | 20 次划分 mean：Spect+LR 0.927 > Spect+XGB 0.882 |
| LR "校准质量需要更大外部测试集确认" | 仍然正确，但 Spect+LR 的 ECE 也更低更稳定（0.110 vs 0.161） |

## 7. 最终结论

1. **Spectrum-Aggregate 策略在平均表现上优于 Patient-Median**，但并非 20/20 次划分都逐次获胜
2. **Spectrum-Aggregate + Logistic Regression 是当前 20 次内部划分中最稳定、平均表现最好的 baseline**
3. Phase3B 中 Spectrum+XGBoost 的胜利具有明显 split 依赖，不能作为稳定最优模型的结论
4. 对于 52 名患者的小样本，**简单线性模型（LR）比非线性模型（XGBoost）更稳定**——这符合"小样本优先简单模型"的统计原则
5. 后续 Phase 3 工作应以 Spectrum-Aggregate + Logistic Regression 作为主要 baseline，而非 XGBoost

## 8. 输出文件

| 文件 | 说明 |
|------|------|
| `stability_results.json` | 每次划分的完整指标 |
| `stability_summary.csv` | 汇总表 |
| `figures/stability_boxplot.png` | 箱线图 |
| `stability_config.yaml` | 配置 |
| `stability_analysis.py` | 分析脚本 |
