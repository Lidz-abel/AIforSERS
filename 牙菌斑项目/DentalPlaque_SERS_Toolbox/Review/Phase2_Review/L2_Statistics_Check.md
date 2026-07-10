# L2: Statistical Validation

## Group Sizes (Kruskal-Wallis requirement: n ≥ 5)
- 阳性+: n=31
- 阴性-: n=21
- Adequate for Kruskal-Wallis: **YES**

## PCA Variance Explained
| PC | Variance % | Cumulative % |
|----|-----------|-------------|
| 1 | 41.31 | 41.31 |
| 2 | 23.79 | 65.11 |
| 3 | 9.29 | 74.40 |
| 4 | 6.41 | 80.81 |
| 5 | 5.07 | 85.88 |
- PC1+PC2 = 65.1%

## Pearson Correlation (Patient-level)
- Matrix: 52 × 52 (patients × patients)
- Diagonal min: 1.000000 (expect 1.0000)
- 阳性+: within r=0.8477 ± 0.2037, between r=0.7386 ± 0.2925
- 阴性-: within r=0.6964 ± 0.3056, between r=0.7386 ± 0.2925

## Statistical Methods Checklist
| Method | Correct? | Notes |
|--------|----------|-------|
| Mean ± SEM | YES | SEM = SD/sqrt(n_patients) |
| Kruskal-Wallis | YES | Non-parametric, patient-level |
| FDR (Benjamini-Hochberg) | YES | mafdr() called |
| Cohen d | YES | Pooled SD, correct formula |
| Pearson r | YES | Patient-patient correlation |
| PCA | YES | Centered, patient-level |
| 95% CI Ellipse | YES | chi2inv(0.95, 2) |
| Hierarchical Clustering | YES | (1-R) distance, average linkage |

### VERDICT
**PASS** — All statistical methods appropriate and correctly applied.
