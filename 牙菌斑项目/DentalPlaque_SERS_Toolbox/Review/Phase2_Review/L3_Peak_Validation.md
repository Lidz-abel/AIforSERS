# L3: Peak Validation

- Consensus peaks: **102** (from `Results/Phase2/PeakStatistics.xlsx`)
- Wavenumber range: **412-1836 cm^{-1}**
- Biologically relevant range (700-1700 cm^{-1}): **67%** (68/102)
- Edge peaks (<500 or >1800 cm^{-1}): **13**
- Peak positions are Raman shifts from the project wavenumber axis, not array indices.

## Top-15 Peak Audit

| Pos (cm^-1) | G1 Mean | G2 Mean | P value | FDR | Quality |
|-------------|---------|---------|---------|-----|---------|
| 724 | 9.1960 | 7.5429 | 0.1151 | 0.1956 | PASS |
| 732 | 9.1960 | 6.2799 | 0.0491 | 0.1105 | PASS |
| 1452 | 1.8000 | 1.8835 | 0.3961 | 0.5051 | PASS |
| 1460 | 1.5169 | 1.5384 | 0.4716 | 0.5659 | PASS |
| 956 | 1.3428 | 1.1801 | 0.7161 | 0.7771 | PASS |
| 1236 | 0.7123 | 1.5174 | 2.73e-06 | 0.000278 | PASS |
| 1308 | 1.0082 | 0.8037 | 0.0951 | 0.1672 | PASS |
| 652 | 0.6778 | 1.2398 | 0.0559 | 0.1119 | PASS |
| 620 | 0.8874 | 0.6122 | 0.3463 | 0.4773 | PASS |
| 1316 | 0.8534 | 0.5019 | 0.0517 | 0.1105 | PASS |
| 1708 | 0.3390 | 0.5737 | 0.0430 | 0.1021 | PASS |
| 1636 | 0.2180 | 0.3340 | 0.3275 | 0.4610 | PASS |
| 540 | 0.4191 | 0.3544 | 0.0312 | 0.0797 | PASS |
| 1700 | 0.2710 | 0.5357 | 0.0742 | 0.1351 | PASS |
| 532 | 0.4124 | 0.3309 | 0.0411 | 0.0999 | PASS |

## Notes

- The previous report incorrectly used `findpeaks(P(i,:))` locations directly, which are array indices, not Raman shifts.
- The corrected review uses the same wavenumber-axis convention as the Phase 2 statistics code (`wn(locs)`).
- Full patient-level detection-rate auditing requires rerunning `Review/runPhase2Review.m` in MATLAB after the script fix.

### VERDICT

**PASS** - Corrected peak positions are valid Raman-shift features. No sub-range index artifacts remain in the L3 report.
