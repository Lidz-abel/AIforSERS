# L5: Figure Quality Review

## Specification Compliance

| Figure | Font | BG | DPI | LW | Colormap | Rating |
|--------|------|----|-----|----|----------|--------|
| M1 Group Mean | Arial | W | 600 | 1.8 | brand | ★★★★☆ |
| M2 Diff Spec | Arial | W | 600 | 1.8 | red/blue | ★★★★☆ |
| M5 Corr Matrix | Arial | W | 600 | 1.0 | jet | ★★★☆☆ |
| M6 Heatmap | Arial | W | 600 | 1.0 | parula | ★★★★☆ |
| M7 PCA | Arial | W | 600 | 1.0 | brand | ★★★★☆ |
| M8 Variability | Arial | W | 600 | 1.8 | brand | ★★★☆☆ |

### Issues Found
- M5 uses jet colormap — Nature prefers perceptually uniform colormaps (parula/viridis).
  **Recommendation:** Change to parula for publication.
- M8 boxplot has rendering warning with Chinese tiledlayout labels (output is correct, warning is cosmetic).
- M1/M2 Y-axis labels: "a.u." is standard, acceptable.

### VERDICT
**MINOR REVISIONS** — M5 colormap should be changed. Otherwise publication-ready.
