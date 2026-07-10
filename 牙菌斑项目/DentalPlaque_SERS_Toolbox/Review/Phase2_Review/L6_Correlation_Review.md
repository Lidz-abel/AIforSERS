# L6: Patient Correlation Review

## Within vs Between Group Correlation

| 阳性+ | 0.8477 ± 0.2037 | 0.7386 ± 0.2925 |
| 阴性- | 0.6964 ± 0.3056 | 0.7386 ± 0.2925 |

- Mean within-group r: **0.7720**
- Mean between-group r: **0.7386**
- **PASS:** Within-group correlation > between-group.
- This supports the hypothesis that patients within the same clinical group
  share spectral features, while between-group differences exist.

## Hierarchical Clustering
- Clustering performed (average linkage, (1-r) distance).
- Group color bar shown alongside heatmap.
- If natural clusters emerge by group → supports group-level spectral differences.

### VERDICT
**PASS**
