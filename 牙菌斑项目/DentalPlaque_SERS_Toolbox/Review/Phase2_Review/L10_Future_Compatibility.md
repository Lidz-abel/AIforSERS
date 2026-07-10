# L10: Future Compatibility (Phase 3–5)

## MCSS (Phase 3)
- Patient mean spectra → directly usable as feature vectors.
- QC-passed spectra per patient → available for MCSS bag generation.
- **Compatible: YES**

## BNN (Phase 4)
- Patient-level labels match BNN input structure.
- Variability scores → uncertainty prior.
- **Compatible: YES**

## SHAP (Phase 5)
- Peak positions from M3 → SHAP peak annotation.
- Effect size ranking → prioritise SHAP-important peaks.
- CV spectra → SHAP stability analysis.
- **Compatible: YES**

## Interface Checklist
| Data | Format | Phase-3 Ready | Phase-5 Ready |
|------|--------|-------------|-------------|
| Patient Mean Spectra | [NPat × 732] | YES | YES |
| Peak Statistics | Excel table | YES | YES |
| Effect Size | Excel table | — | YES |
| PCA Scores | [NPat × K] | YES | YES |
| Variability Scores | [NPat × 1] | — | YES |
| Correlation Matrix | [NPat × NPat] | — | YES |

### VERDICT
**PASS** — All Phase-2 outputs are ready for downstream consumption.
