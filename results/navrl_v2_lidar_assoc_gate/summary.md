# LiDAR association gate A/B under detection dropout (ep25000+riskcap, seed 47, 205 bars)

| cell | episodes | capture | crash | bar contacts | OOB | fused visible | closest mean (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| analytic_clean | 2050 | 80.54% | 17.17% | 337 | 10 | 21.21% | 0.806 |
| clean_ro_gate035 | 2049 | 80.48% | 16.59% | 337 | 2 | 19.46% | 0.914 |
| dropout_0p3_no_assoc | 2049 | 71.25% | 25.96% | 511 | 13 | 14.59% | 0.948 |
| dropout_0p3_raw | 2049 | 67.84% | 29.33% | 559 | 23 | 21.38% | 0.972 |
| dropout_0p3_ro_gate035 | 2049 | 69.55% | 28.01% | 546 | 20 | 21.30% | 0.902 |
| dropout_0p3_ro_gate065 | 2049 | 69.35% | 27.48% | 525 | 27 | 20.96% | 1.018 |

## verdict: INCONCLUSIVE

- capture vs dropout_raw: +1.71 pp = 13.5% of the dropout loss
- bar contacts vs dropout_raw: -13 = 5.9% of the excess removed
- clean regression from the fix: -0.06 pp

Inter-arm SE is about 1.3 pp at this episode count, so SUPPORTED needs >= 4 pp.
