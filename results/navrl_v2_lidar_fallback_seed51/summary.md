# LiDAR fallback A/B under detection dropout (ep25000+riskcap, seed 51, 205 bars)

| cell | episodes | capture | crash | bar contacts | OOB | fused visible | closest mean (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| analytic_clean | 2050 | 81.12% | 16.68% | 335 | 0 | 21.29% | 0.852 |
| clean_no_assoc | 2050 | 80.10% | 17.12% | 333 | 3 | 17.41% | 0.913 |
| dropout_0p3_no_assoc | 2050 | 71.95% | 25.46% | 497 | 15 | 14.22% | 0.917 |
| dropout_0p3_raw | 2049 | 68.86% | 28.16% | 538 | 30 | 21.22% | 0.994 |

## verdict: INCONCLUSIVE

- capture vs dropout_raw: +3.09 pp = 25.2% of the dropout loss
- bar contacts vs dropout_raw: -41 = 20.2% of the excess removed
- clean regression from the fix: -1.02 pp

Inter-arm SE is about 1.3 pp at this episode count, so SUPPORTED needs >= 4 pp.
