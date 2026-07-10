# L7: PCA Review

## Input Verification
- PCA input: **52 patients × 732 wavenumber** (NOT all 1970 spectra)
- PASS: Patient-level.

## Variance Distribution
| PC | Var% | Cum% |
|----|------|------|
| 1 | 41.3 | 41.3 |
| 2 | 23.8 | 65.1 |
| 3 | 9.3 | 74.4 |
| 4 | 6.4 | 80.8 |
| 5 | 5.1 | 85.9 |

## Outlier Detection (Mahalanobis D² on PC1-2)
- Outlier patients (p < 0.05): **4**
  - Patient 9 (阴性-): D² = 13.2 (threshold = 6.0)
  - Patient 12 (阴性-): D² = 11.0 (threshold = 6.0)
  - Patient 10 (阴性-): D² = 11.0 (threshold = 6.0)
  - Patient 28 (阳性+): D² = 8.8 (threshold = 6.0)

## Confidence Ellipses
- 95% confidence ellipses: **correctly computed** (chi2inv(0.95, 2), eigenvalue-scaled).
- Overlap between groups: **present** — expected for biological data.

### VERDICT
**PASS** — PCA correctly implemented at patient level.
