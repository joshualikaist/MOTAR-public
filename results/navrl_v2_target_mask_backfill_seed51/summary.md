# phantom-target obstacle under detection dropout (ep25000+riskcap, seed 51, 205 bars)

| cell | episodes | capture | crash | bar contacts | OOB | fused visible | closest mean (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| analytic_clean | 2050 | 81.12% | 16.68% | 335 | 0 | 21.29% | 0.852 |
| clean_backfill | 2050 | 78.73% | 18.63% | 372 | 2 | 20.66% | 0.876 |
| dropout_0p3_backfill | 2050 | 70.63% | 26.49% | 518 | 16 | 20.94% | 1.011 |
| dropout_0p3_raw | 2049 | 68.86% | 28.16% | 538 | 30 | 21.22% | 0.994 |

## verdict: INCONCLUSIVE

- capture vs dropout_raw: +1.77 pp = 14.4% of the dropout loss
- bar contacts vs dropout_raw: -20 = 9.9% of the excess removed
- clean regression from the fix: -2.39 pp

Inter-arm SE is about 1.3 pp at this episode count, so SUPPORTED needs >= 4 pp.
