# Phase3C Validation Audit

**日期**: 2026-07-12
**状态**: 通过，已修正报告中过强表述

## 1. 执行验证

已重新运行：

```bash
python Phase3/baseline/stability_analysis.py
```

输出文件已重新生成：

- `Results/Phase3/stability/stability_results.json`
- `Results/Phase3/stability/stability_summary.csv`
- `Results/Phase3/stability/figures/stability_boxplot.png`

## 2. 数据划分验证

20 次随机划分均为患者级 stratified 60/20/20 split：

| 项目 | 结果 |
|------|------|
| 总患者数 | 52 |
| 阳性 / 阴性 | 31 / 21 |
| Train | 31 patients, 18 positive, 13 negative |
| Val | 10 patients, 6 positive, 4 negative |
| Test | 11 patients, 7 positive, 4 negative |
| Train-Val overlap | 0 |
| Train-Test overlap | 0 |
| Val-Test overlap | 0 |
| 覆盖患者数 | 52 / 52 |

结论：未发现患者级数据泄露。Spectrum-level 样本由 patient split 派生，未跨患者集合混入。

## 3. 稳定性结果

| 模型组合 | ROC-AUC | Accuracy | Sensitivity | Specificity | Brier ↓ | ECE ↓ |
|---------|---------|----------|-------------|-------------|---------|-------|
| Patient-Median + LR | 0.9018 ± 0.0731 | 0.8455 ± 0.1223 | 0.8500 ± 0.1888 | 0.8375 ± 0.1816 | 0.1036 ± 0.0583 | 0.1240 ± 0.0826 |
| Patient-Median + XGBoost | 0.8679 ± 0.1477 | 0.7636 ± 0.1330 | 0.8143 ± 0.2124 | 0.6750 ± 0.2107 | 0.1779 ± 0.0907 | 0.2238 ± 0.0992 |
| Spectrum-Aggregate + LR | 0.9750 ± 0.0376 | 0.9273 ± 0.0617 | 0.9786 ± 0.0510 | 0.8375 ± 0.1635 | 0.0612 ± 0.0374 | 0.1100 ± 0.0490 |
| Spectrum-Aggregate + XGBoost | 0.9536 ± 0.0883 | 0.8818 ± 0.1117 | 0.9643 ± 0.0766 | 0.7375 ± 0.2559 | 0.0865 ± 0.0511 | 0.1605 ± 0.0445 |

按 ROC-AUC 逐 split 排名的获胜次数：

| 模型组合 | 获胜次数 |
|---------|----------|
| Spectrum-Aggregate + LR | 8 / 20 |
| Patient-Median + LR | 6 / 20 |
| Patient-Median + XGBoost | 3 / 20 |
| Spectrum-Aggregate + XGBoost | 3 / 20 |

## 4. 逐 seed 配对检查

Spectrum-Aggregate 相比 Patient-Median 的逐 seed AUC 比较：

- Logistic Regression: 12 胜 / 7 平 / 1 负
- XGBoost: 11 胜 / 3 平 / 6 负

因此，不能写成“20/20 次划分均优于 Patient-Median”。更严谨的表述是：Spectrum-Aggregate 在平均 AUC 上优于 Patient-Median，但逐次划分并非全部获胜。

Spectrum-Aggregate + LR 相比 Spectrum-Aggregate + XGBoost 的逐 seed AUC 比较：

- LR: 4 胜 / 12 平 / 4 负

因此，LR 的优势主要来自更高均值、更低波动和更好的 Accuracy/Brier/ECE，而不是每个 seed 都击败 XGBoost。

## 5. 已修正内容

已修改 `Results/Phase3/stability/PHASE3C_REPORT.md`：

- 将“20/20 次划分均成立”改为“平均表现上优于”
- 将“确实是偶然”改为“存在明显 split 依赖”
- 将“几乎不漏诊”改为“当前内部随机划分中阳性识别较稳定”
- 明确当前结论不能替代独立外部验证

已修改 `Phase3/baseline/stability_analysis.py`：

- 修复控制台输出中 `{n_repeats}` 未格式化的问题

## 6. 审计结论

Phase3C 验证通过。当前最稳妥的结论是：

> 在 20 次患者级随机划分中，Spectrum-Aggregate + Logistic Regression 具有最高平均 ROC-AUC、Accuracy、Sensitivity，并且整体波动小于 Spectrum-Aggregate + XGBoost。Phase3B 中 Spectrum-Aggregate + XGBoost 在 seed=42 上最优，但该结论具有明显 split 依赖。后续应将 Spectrum-Aggregate + Logistic Regression 作为主要 baseline，同时保留 XGBoost 作为对照模型。

