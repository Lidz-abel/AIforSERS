# DentalPlaque_SERS_Toolbox

## What This Is
A MATLAB toolbox for **SERS spectral analysis of dental plaque** to diagnose periodontitis.
Target journals: Nature Communications / ACS Sensors / Analytical Chemistry.

## Quick Start (for a new Claude session)
```
cd <项目根目录>\DentalPlaque_SERS_Toolbox
claude
```
Then say: "回到牙菌斑SERS工程，继续Phase 3开发"

Or run directly:
```
matlab -batch "cd('<项目根目录>\DentalPlaque_SERS_Toolbox'); addpath(genpath(pwd)); main;"
```

## Project Status
| Phase | Name | Status |
|-------|------|--------|
| 1 | Preprocessing + QC + 52 Dashboards | ✅ Complete |
| 2 | Biological Spectral Characterization (8 modules) | ✅ Complete |
| 2.5 | Scientific Review | ✅ Complete; L3 peak validation corrected, rerun in MATLAB when available |
| 3 | MCSS + Deep Learning | 🔜 Next |
| 4 | SHAP Biological Interpretation | 📅 Planned |
| 5 | Publication | 📅 Planned |

## Directory Structure
```
DentalPlaque_SERS_Toolbox/
├── main.m                    ← Entry point: main / main('read') / main('phase2') etc.
├── config.m                  ← ALL parameters live here. No hard-coding anywhere.
├── Functions/
│   ├── IO/                   → readDatabase.m
│   ├── Preprocessing/        → SG smooth, airPLS, SNV
│   ├── QC/                   → technicalQC, structuralQC, runQC
│   ├── Plot/                 → Dashboard, peak detection, export
│   ├── Statistics/           → M1-M8 biological characterization
│   ├── DeepLearning/         → MCSS (skeleton)
│   └── Utils/                → Peak assignment database
├── Results/                  → .mat files + Excel tables
├── Figures/                  → Per-patient dashboards + Phase2 figures
├── Report/
└── Review/                   → Phase 2.5 scientific audit
```

## Data
```
<项目根目录>\牙菌斑SERS光谱\
  阳性+\   → 31 patients, 1549 spectra
  阴性-\   → 21 patients, 421 spectra
CSV: D294:D1025 (wavenumber), H294:H1025 (intensity)
Range: 395.11 – 1840.62 cm⁻¹, 732 points
```

## Critical Design Rules (DO NOT VIOLATE)
1. **One Patient = One Independent Sample** — never pool spectra for statistics
2. All magic numbers in config.m
3. Each function ≤200 lines, single responsibility, full `help` comment
4. MCSS is a *training strategy*, not data augmentation
5. MCSS ONLY on training set AFTER patient-level split

## MATLAB
- Version: R2025b Update 1
- Path: use the local MATLAB executable on the current machine
- Toolboxes: Signal Processing, Statistics, ML, DL, Parallel, Image Processing

## Plot Style
- Nature single-column (19 cm wide)
- Arial, 600 dpi, white background, LineWidth 1.8
- Export: PNG + PDF + SVG

## Key Results (Phase 2)
- PCA: PC1+PC2 = 65.1% variance
- Top Cohen's d peaks: 1.22–1.57 (large effect)
- 20 significant Raman variability bands (FDR < 0.05, |Cliff's delta| > 0.5)
- Variability dysbiosis hypothesis: preliminarily supported (p = 0.0536)
- Phase 2.5 Review: L3 peak validation now uses Raman shifts (`wn(locs)`), not array indices.

## For Team Members
### Prerequisites
1. MATLAB R2025b with required toolboxes
2. Python 3.x with openpyxl (for Excel generation)
3. Data folder at `<项目根目录>\牙菌斑SERS光谱\` (auto-detected by `config.m`)

### First Run
```matlab
addpath(genpath(pwd))  % run from DentalPlaque_SERS_Toolbox
main              % runs read → preprocess → qc → dashboard → phase2
```

### Adding New Patients
1. Place CSV files in `<项目根目录>\牙菌斑SERS光谱\<Group>\<PatientID>\`
2. CSV must have D294:D1025 (wavenumber) and H294:H1025 (intensity)
3. Run `main('read')` to rebuild Database
4. Run `main` for full pipeline

### Adding New Raman Peak Assignments
Edit `Functions/Utils/peakAssignments.m` — append to the struct array.

### Changing Parameters
Edit `config.m` — ALL tunable values are there. Never edit thresholds inside functions.
