# LiDAR range-only update A/B under detection dropout (ep25000+riskcap, seed 47, 205 bars)

| cell | episodes | capture | crash | bar contacts | OOB | fused visible | closest mean (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| analytic_clean | 2050 | 80.54% | 17.17% | 337 | 10 | 21.21% | 0.806 |
| clean_range_only | 2050 | 80.00% | 16.59% | 324 | 12 | 20.02% | 0.984 |
| dropout_0p3_no_assoc | 2049 | 71.25% | 25.96% | 511 | 13 | 14.59% | 0.948 |
| dropout_0p3_range_only | 2049 | 69.30% | 28.06% | 539 | 25 | 21.22% | 0.977 |
| dropout_0p3_raw | 2049 | 67.84% | 29.33% | 559 | 23 | 21.38% | 0.972 |

## verdict: INCONCLUSIVE

- capture vs dropout_raw: +1.46 pp = 11.5% of the dropout loss
- bar contacts vs dropout_raw: -20 = 9.0% of the excess removed
- clean regression from the fix: -0.54 pp

Inter-arm SE is about 1.3 pp at this episode count, so SUPPORTED needs >= 4 pp.
