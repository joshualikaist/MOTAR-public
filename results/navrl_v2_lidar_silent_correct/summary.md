# LiDAR silent-correct A/B under detection dropout (ep25000+riskcap, seed 47, 205 bars)

| cell | episodes | capture | crash | bar contacts | OOB | fused visible | closest mean (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| analytic_clean | 2050 | 80.54% | 17.17% | 337 | 10 | 21.21% | 0.806 |
| clean_silent | 2049 | 81.11% | 16.06% | 320 | 5 | 18.60% | 0.932 |
| dropout_0p3_no_assoc | 2049 | 71.25% | 25.96% | 511 | 13 | 14.59% | 0.948 |
| dropout_0p3_raw | 2049 | 67.84% | 29.33% | 559 | 23 | 21.38% | 0.972 |
| dropout_0p3_silent | 2050 | 70.20% | 26.78% | 523 | 16 | 14.46% | 1.011 |

## verdict: INCONCLUSIVE

- capture vs dropout_raw: +2.36 pp = 18.6% of the dropout loss
- bar contacts vs dropout_raw: -36 = 16.2% of the excess removed
- clean regression from the fix: +0.58 pp

Inter-arm SE is about 1.3 pp at this episode count, so SUPPORTED needs >= 4 pp.
