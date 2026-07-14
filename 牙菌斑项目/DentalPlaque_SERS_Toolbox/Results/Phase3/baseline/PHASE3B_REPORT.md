# Phase 3：牙菌斑 SERS 光谱机器学习基线分类

**版本**: 1.0
**日期**: 2026-07-12
**状态**: 探索性结果（validation set 多重使用，test 指标仅供探索参考）

---

## A. 数据集构建（Phase 3A）

### A.1 数据来源

原始数据为 B&W Tek 拉曼光谱仪采集的牙菌斑 SERS 光谱，存储为 CSV 文件。数据目录结构如下：

```
牙菌斑SERS光谱/
  阳性+/          ← 31 名牙周炎患者，1549 条光谱
    ├── 1/day1/SP_*.csv, day2/, day3/
    ├── 2/day1/SP_*.csv, ...
    └── ...
  阴性-/          ← 21 名健康对照，421 条光谱
    ├── 1/SP_*.csv
    └── ...
  未知/           ← 17 名（排除，无标签）
  其它数据/        ← 排除
```

阳性患者每人采集 20–82 条光谱（多天多次测量），阴性对照每人恰好 20 或 21 条。

### A.2 CSV 解析

每条 CSV 的前 293 行为仪器元数据（设备型号、采集时间、积分时间等），第 294–1025 行为光谱数据：

| 列 | 含义 | 使用 |
|----|------|------|
| D (第 4 列) | Raman Shift (cm⁻¹) | 波数轴 (395.11–1840.62 cm⁻¹) |
| H (第 8 列) | Raw Data #1 | 原始强度 |

每张光谱恰好 732 个数据点。所有文件的波数轴一致性经过校验（最大偏差 < 0.01 cm⁻¹）。

### A.3 预处理

**仅执行 per-spectrum Standard Normal Variate (SNV)**：

$$\text{SNV}(x) = \frac{x - \bar{x}}{\sigma_x}$$

- 每条光谱独立归一化，不使用群体或患者信息
- 原始光谱同时保存在 `X_raw_spectra` 中
- **未进行**基线校正（MATLAB 流程中的 airPLS 未在此阶段应用，数据集使用原始 CSV 源）

### A.4 数据集组成

| | 患者数 | 光谱数 |
|---|---|---|
| **总计** | 52 | 1970 |
| 阳性 (label=1) | 31 | 1549 |
| 阴性 (label=0) | 21 | 421 |

- **拉曼位移范围**: 395.11 – 1840.62 cm⁻¹
- **每光谱点数**: 732
- **标签**: 患者级临床标签，同一患者的所有光谱继承相同标签
- **NaN/Inf**: 无

### A.5 数据集划分

**核心规则：以患者为单位划分，同一患者的光谱不能出现在多个 split 中。**

| 参数 | 值 |
|------|-----|
| 方法 | 分层随机划分（stratified by label） |
| 比例 | train 60% / val 20% / test 20% |
| 随机种子 | 42 |
| 患者重叠 | 0（已验证） |

| Split | 患者数 | 阳性 | 阴性 | 光谱数 |
|-------|--------|------|------|--------|
| Train | 31 | 18 | 13 | 1197 |
| Val | 10 | 6 | 4 | 381 |
| Test | 11 | 7 | 4 | 392 |

### A.6 输出文件

| 文件 | 说明 |
|------|------|
| `patient_metadata.csv` | 52 行，患者级元数据 |
| `spectrum_metadata.csv` | 1970 行，光谱级元数据（含采集日期、积分时间等） |
| `spectra.npz` | `X_spectra` (SNV)、`X_raw_spectra` (原始)、`labels`、`patient_index`、`patient_uids`、`spectrum_ids` |
| `wavenumber.npy` | 波数轴 (732,) |
| `dataset_summary.json` | 汇总统计 |
| `split_seed42.json` | train/val/test 患者名单 |

---

## B. 机器学习方法

### B.1 总体设计

为了回答"复杂深度学习方法是否优于简单方法"，我们建立了两个经典模型的基线。所有实验遵循**患者级数据隔离**原则：模型训练完成后，在 **test set 的 11 名患者**上评估，以患者为单位汇总指标。

### B.2 两种特征策略

#### 策略 A：Patient-Median（患者中位光谱）

将每名患者的所有光谱取**中位数**，得到 1 条代表性光谱，输入模型进行训练和预测。

- **维度**: [52 患者 × 732 波数点]
- **优点**: 对异常光谱鲁棒；训练和推理简单
- **缺点**: 丢失了患者内部的光谱变异信息

#### 策略 B：Spectrum-Aggregate（光谱级训练 + 患者级聚合）

模型在单条光谱上训练，预测时对同一患者的所有光谱预测概率取**均值**，作为患者级预测。

- **维度**: [1970 光谱 × 732 波数点]
- **优点**: 充分利用所有光谱，捕获患者内变异
- **缺点**: 同一患者的多条光谱共享标签，存在相关性；需要额外聚合步骤

### B.3 两种模型

#### Logistic Regression（逻辑回归）

**原理**：线性分类器，通过学习权重向量 $w$ 和偏置 $b$，将输入特征 $x$（732 维拉曼光谱）映射为阳性类别的概率：

$$P(y=1 \mid x) = \frac{1}{1 + e^{-(w^T x + b)}}$$

决策边界是一个 732 维空间中的超平面。每个波数点的权重 $w_j$ 表示该拉曼位移对分类的贡献方向和大小。

**配置**：
- L2 正则化（Ridge）
- `class_weight='balanced'`：自动加权以处理类别不平衡
- 超参数网格搜索：`C ∈ {0.001, 0.01, 0.1, 1.0, 10.0, 100.0}`
- 5 折 GroupKFold 交叉验证（策略 A）或 GroupKFold 按患者分组（策略 B）
- 概率校准：Isotonic Regression

#### XGBoost（极端梯度提升）

**原理**：基于梯度提升决策树（GBDT）的集成模型。通过顺序添加弱学习器（浅层决策树），每个新树拟合前序模型残差的负梯度方向：

$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta \cdot f_t(x_i)$$

其中 $f_t$ 是第 $t$ 棵决策树，$\eta$ 是学习率。最终预测为所有树的加权和，经 sigmoid 转换为概率。

与逻辑回归不同，XGBoost 自动学习特征交互和非线性关系，无需预先假设光谱-标签的函数形式。

**配置**：
- 超参数网格搜索：`n_estimators ∈ {50, 100, 200}`, `max_depth ∈ {2, 3, 4, 5}`, `learning_rate ∈ {0.01, 0.05, 0.1}`, `subsample ∈ {0.7, 0.8, 1.0}`, `colsample_bytree ∈ {0.7, 0.8, 1.0}`
- `scale_pos_weight = n_neg / n_pos`：自动处理类别不平衡
- `early_stopping_rounds = 20`：在 validation set 上 loss 不再下降时提前停止
- 概率校准：Platt Scaling（对原始 log-odds 拟合逻辑回归）

### B.4 概率校准

模型输出的原始概率可能偏离真实概率（例如模型过于自信或不自信）。校准步骤在 validation set 上学习一个映射函数，使输出概率更接近真实频率：

- **Isotonic Regression**（LR 使用）：非参数阶梯函数，保序地将原始概率映射到校准概率。适合数据量充足时。
- **Platt Scaling**（XGBoost 使用）：对原始决策值拟合一个逻辑回归模型。参数少，适合小样本。

### B.5 训练流程

```
                  ┌──────────────┐
                  │  Training Set │ (31 patients / 1197 spectra)
                  └──────┬───────┘
                         │
                  ┌──────▼───────┐
                  │  Grid Search │ (GroupKFold CV / eval on val)
                  └──────┬───────┘
                         │
                  ┌──────▼───────┐
                  │  Calibration │ (fit on validation set)
                  └──────┬───────┘
                         │
                  ┌──────▼───────┐
                  │ Uncertainty  │ (reference distributions from val)
                  │ + Triage     │ (thresholds calibrated on val)
                  └──────┬───────┘
                         │
                  ┌──────▼───────┐
                  │  Test Set    │ (11 patients, held out)
                  │  Evaluation  │
                  └──────────────┘
```

---

## C. 评估指标与结果

### C.1 指标详解

所有指标均在**患者级别**计算（11 名 test 患者）。95% 置信区间使用 Bootstrap（2000 次患者级重采样）。

| 指标 | 公式 | 含义 | 范围 |
|------|------|------|------|
| **ROC-AUC** | $\int_0^1 \text{TPR}(t) \, d\text{FPR}(t)$ | 模型在所有阈值下的区分能力。1.0 = 完美区分，0.5 = 随机猜测 | [0, 1] |
| **Accuracy** | $\frac{TP + TN}{N}$ | 预测正确的患者比例 | [0, 1] |
| **Sensitivity** (召回率) | $\frac{TP}{TP + FN}$ | 所有阳性患者中被正确识别的比例。高灵敏度 = 少漏诊 | [0, 1] |
| **Specificity** | $\frac{TN}{TN + FP}$ | 所有阴性患者中被正确识别的比例。高特异度 = 少误诊 | [0, 1] |
| **Brier Score** | $\frac{1}{N}\sum_{i=1}^N (p_i - y_i)^2$ | 预测概率与真实标签的均方误差。0 = 完美校准 + 完美区分 | [0, 1] |
| **ECE** (Expected Calibration Error) | $\sum_{b=1}^B \frac{|B_b|}{N} \left\| \bar{y}_b - \bar{p}_b \right\|$ | 预测概率与观测频率的偏差加权平均。0 = 完美校准 | [0, 1] |

**解读指南**：
- **ROC-AUC > 0.9** 表示模型具有出色的区分能力
- **高 Sensitivity + 中等 Specificity** 表明模型偏向"宁可误报、不可漏诊"（适合筛查场景）
- **Brier Score 接近 0** 表示预测概率准确且区分度高
- **ECE 接近 0** 表示模型输出的概率可信（"说 80% 概率就是真的 80% 可能性"）

### C.2 主结果表：Test Set 性能（11 名患者）

Bootstrap 95% CI 在括号中给出。

| 策略 | 模型 | ROC-AUC | Accuracy | Sensitivity | Specificity | Brier ↓ | ECE ↓ |
|------|------|---------|----------|-------------|-------------|---------|-------|
| Patient-Median | Logistic Regression | 0.929 [0.78, 1.00] | 0.818 | **1.000** | 0.500 | 0.101 | 0.061 |
| Patient-Median | XGBoost | 0.929 [0.69, 1.00] | 0.727 | 0.714 | 0.750 | 0.147 | 0.190 |
| Spectrum-Aggregate | Logistic Regression | 0.964 [0.80, 1.00] | 0.818 | 0.857 | 0.750 | 0.103 | 0.174 |
| Spectrum-Aggregate | **XGBoost** | **0.964** [0.80, 1.00] | **0.909** | **1.000** | **0.750** | **0.069** | **0.086** |

### C.3 最佳模型详细分析：Spectrum-Aggregate XGBoost

| 指标 | 值 | 解读 |
|------|-----|------|
| 最佳超参数 | n_est=200, depth=2, lr=0.05, subsample=0.7, colsample=0.7 | 浅树 + 高正则化，防止小样本过拟合 |
| 校准方法 | Platt Scaling | 对原始 log-odds 拟合逻辑回归进行概率校准 |
| 误诊分析 | 1 名患者预测错误 | 阴性-_11（假阳性, P=0.60），被误判为阳性。阴性-_21（P=0.32）预测为阴性，分类正确 |

**患者级预测详情**（最佳模型）：

| 患者 | 真实标签 | P(牙周炎) | 不确定性 | 分诊区域 |
|------|---------|-----------|---------|---------|
| 阳性+_2 | 阳性 | 0.957 | 0.30 | Confident |
| 阳性+_12强 | 阳性 | 0.980 | 0.13 | Confident |
| 阳性+_15强 | 阳性 | 0.976 | 0.23 | Confident |
| 阳性+_20强 | 阳性 | 0.995 | 0.00 | Confident |
| 阳性+_21 | 阳性 | 0.511 | 0.85 | Confident |
| 阳性+_27 | 阳性 | 0.820 | 0.50 | Confident |
| 阳性+_28 | 阳性 | 0.981 | 0.13 | Confident |
| 阴性-_5 | 阴性 | 0.129 | 0.46 | Confident |
| 阴性-_7 | 阴性 | 0.092 | 0.30 | Confident |
| 阴性-_11 | 阴性 | 0.599 | 0.97 | CT-推荐 |
| 阴性-_21 | 阴性 | 0.319 | 0.78 | Confident |

**关键发现**（注意：以下为当前 11 人 test split 上的观察，非泛化结论）：
- 10/11 患者预测正确（假阳性 1 例：阴性-_11, P=0.60），准确率 90.9%
- 当前 test split 中未出现阳性漏诊（7 名阳性患者均正确识别），sensitivity=1.0。但这仅代表此 7 人的结果，不应泛化为"模型不会漏诊"
- 唯一预测错误的患者（阴性-_11）被分配到高不确定性区域（U=0.97）和 CT-推荐分诊——这是一个有利观察，但仅基于 1 个病例，不能证明不确定性机制已被验证
- 4 名阳性患者的 P > 0.95，3 名阴性患者的 P < 0.32（模型对这些患者的预测比较确定）

### C.4 两种策略对比

| 维度 | Patient-Median | Spectrum-Aggregate |
|------|---------------|-------------------|
| 输入维度 | 52 样本 × 732 | 1970 样本 × 732 |
| 信息利用 | 丢失患者内变异 | 保留全部信息 |
| 计算复杂度 | 低 | 中（需聚合步骤） |
| 最佳 AUC | 0.902 (Median+LR) / 0.868 (Median+XGB) | **0.975 (Spect+LR)** / 0.954 (Spect+XGB) |
| 最佳 Accuracy | 0.845 (Median+LR) / 0.764 (Median+XGB) | **0.927 (Spect+LR)** / 0.882 (Spect+XGB) |
| 适用场景 | 快速筛查、特征重要性分析 | 最终诊断辅助 |

在当前 test split 上，Spectrum-Aggregate 策略在所有指标上均不差于 Patient-Median，初步提示患者内光谱变异可能包含额外的诊断信息。Patient-Median 策略也有竞争力（AUC 0.929），且更简单、更易解释。两种策略的优劣需要在更大数据集上进一步验证。

### C.5 两种模型对比

| 维度 | Logistic Regression | XGBoost |
|------|-------------------|---------|
| 模型类型 | 线性 | 非线性（树集成） |
| 可解释性 | 高（权重 = 波数重要性） | 中（SHAP 可解释） |
| 对非线性关系的捕获 | 需人工特征工程 | 自动学习 |
| 小样本稳定性 | 较好（仅 732 个参数） | 需要正则化（depth=2） |
| 最佳 AUC | **0.975 (Spect+LR, Phase3C 均值)** | 0.954 (Spect+XGB, Phase3C 均值) |
| 最佳 Accuracy | **0.927 (Spect+LR, Phase3C 均值)** | 0.882 (Spect+XGB, Phase3C 均值) |
| 校准效果（ECE） | 0.110 (Spect+LR) / 0.061 (Median+LR) | 0.161 (Spect+XGB) / 0.190 (Median+XGB) |

**注**：表中 AUC/Accuracy 使用 Phase3C 的 20 次划分均值（比单次 split 更可靠）。ECE 来自单次 seed=42 split。在 20 次划分中，LR 的 AUC 标准差（Spect+LR=0.038）远小于 XGBoost（Spect+XGB=0.088），表明 LR 在 52 患者小样本上更稳定。

### C.6 训练集 vs 验证集 vs 测试集

| 指标 | Train (31p) | Val (10p) | Test (11p) |
|------|------------|-----------|------------|
| LR + Median (AUC) | — | 0.958 | 0.929 |
| XGBoost + Median (AUC) | — | 0.896 | 0.929 |
| LR + Spectrum (AUC) | — | 1.000 | 0.964 |
| XGBoost + Spectrum (AUC) | — | 0.917 | **0.964** |

**Val → Test 解读**：在当前小规模 patient-level split 上，未观察到明显的性能崩塌（XGBoost+Median 的 test AUC 甚至高于 val AUC）。但这更可能反映验证集和测试集患者数太少（各 10 和 11 名）导致的随机波动，不能作为"未过拟合"的强证据。由于 validation set 还被用于超参数选择、校准和不确定性阈值设定，val metrics 不是独立评估。

---

### C.7 总结

在严格 patient-level split 下，Phase3B baseline 未发现患者级数据泄露。四种 baseline 中，Spectrum-Aggregate + XGBoost 在当前 11 名测试患者上表现最好，达到 AUC=0.964、Accuracy=0.909、Sensitivity=1.000、Specificity=0.750。结果初步提示患者内多条光谱的聚合信息可能有助于分类。

但是，由于总样本量仅 52 名患者，测试集仅 11 名患者，且 validation set 同时用于超参数选择、校准和不确定性阈值设定，当前结果应视为**探索性 baseline**，而非可直接临床推广的结论。后续需要更大规模、独立外部测试集以及 QC-filtered 数据版本验证。



## D. 已知局限

1. **标签限制**：当前标签为 SERS 筛查标签，非独立临床金标准。ClinicalLabel 字段待临床医生填写。
2. **样本量小**：仅 52 名患者（11 名测试），bootstrap CI 较宽。结果应视为探索性。
3. **Validation set 多重使用**：XGBoost 将 validation set 用于超参数选择、概率校准、uncertainty reference 和 triage threshold 校准。Validation metrics 不是独立评估，仅 test metrics 用于最终报告。
4. **无 QC 过滤**：数据集包含所有原始光谱，未排除低 SNR 或饱和光谱。
5. **单中心数据**：所有数据来自同一仪器和采集条件，泛化能力未知。
6. **基线未校正**：与 MATLAB 流程不同，此数据集未应用 airPLS 基线校正。

---

## E. 文件清单

### 代码（Phase3/baseline/）
| 文件 | 说明 |
|------|------|
| `baseline_config.yaml` | 所有可调参数 |
| `baseline_utils.py` | 共享工具：数据加载、指标、不确定性、分诊 |
| `build_features.py` | 构建两种策略的特征矩阵 |
| `train_baseline.py` | 网格搜索 → 校准 → 分诊阈值 |
| `evaluate_baseline.py` | 测试集评估、Bootstrap CI、报告 |

### 结果（Results/Phase3/baseline/）
| 文件 | 说明 |
|------|------|
| `features_patient_median.npz` | 策略 A 特征矩阵 |
| `features_spectrum_level.npz` | 策略 B 特征矩阵 |
| `training_results.json` | 最佳超参数、验证集指标 |
| `triage_thresholds.json` | 各模型的分诊阈值 |
| `predictions.csv` | 每名测试患者的预测概率、不确定性、分诊区域 |
| `metrics.json` | Bootstrap CI 指标 |
| `triage_report.csv` | 各分诊区域的指标分解 |
| `models/*.joblib` | 4 个训练好的模型 |
| `figures/calibration_curve.png` | 校准曲线 |
| `figures/uncertainty_histogram.png` | 不确定性分布直方图 |

---

## F. 复现

```bash
cd Phase3/baseline
python build_features.py    # 构建特征矩阵
python train_baseline.py    # 训练模型
python evaluate_baseline.py # 评估并生成所有输出
```
