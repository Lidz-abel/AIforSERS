# L1: Patient-Level Verification

| Metric | Value |
|--------|-------|
| Independent samples (patients) | 52 |
| Wavenumber points | 732 |
| WN range | 395.1 - 1840.6 cm^{-1} |
| Group 阳性+ | 31 patients, 1549 spectra |
| Group 阴性- | 21 patients, 421 spectra |

**All Phase-2 functions call extractPatientData() → [Npatients × NPoints].**

| Function | Input Size | Meaning |
|----------|-----------|--------|
| PCA | [52 × 732] | nPatients × nWavenumber |
| Correlation | [52 × 52] | nPatients × nPatients |
| Heatmap | 52 rows | nPatients |
| Effect Size | per-group n=31, 21 | patients |

### VERDICT
**PASS** — One Patient = One Independent Sample.
No pseudoreplication. All statistics are patient-level.
