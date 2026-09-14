# LiDAR fallback A/B under detection dropout (ep25000+riskcap, seed47, 205 bars)

| cell | episodes | capture | crash | bar contacts | OOB | fused visible | closest mean (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| analytic_clean | 2050 | 80.54% | 17.17% | 337 | 10 | 21.21% | 0.806 |
| clean_no_assoc | 2051 | 80.35% | 17.50% | 346 | 6 | 18.44% | 0.786 |
| dropout_0p3_no_assoc | 2049 | 71.25% | 25.96% | 511 | 13 | 14.59% | 0.948 |
| dropout_0p3_raw | 2049 | 67.84% | 29.33% | 559 | 23 | 21.38% | 0.972 |

## verdict: INCONCLUSIVE

- capture vs dropout_raw: +3.42 pp = 26.9% of the dropout loss
- bar contacts vs dropout_raw: -48 = 21.6% of the excess removed
- clean regression from the fix: -0.19 pp

Inter-arm SE is about 1.3 pp at this episode count, so SUPPORTED needs >= 4 pp.
