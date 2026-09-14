# phantom-target obstacle under detection dropout (ep25000+riskcap, seed47, 205 bars)

| cell | episodes | capture | crash | bar contacts | OOB | fused visible | closest mean (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| analytic_clean | 2050 | 80.54% | 17.17% | 337 | 10 | 21.21% | 0.806 |
| clean_backfill | 2049 | 79.11% | 18.55% | 365 | 5 | 21.03% | 0.832 |
| dropout_0p3_backfill | 2049 | 70.38% | 26.65% | 513 | 21 | 21.00% | 1.020 |
| dropout_0p3_raw | 2049 | 67.84% | 29.33% | 559 | 23 | 21.38% | 0.972 |

## verdict: INCONCLUSIVE

- capture vs dropout_raw: +2.54 pp = 20.0% of the dropout loss
- bar contacts vs dropout_raw: -46 = 20.7% of the excess removed
- clean regression from the fix: -1.42 pp

Inter-arm SE is about 1.3 pp at this episode count, so SUPPORTED needs >= 4 pp.
