# latency P3 ego-motion arms (ep25000+riskcap, seed47, 205 bars)

| cell | episodes | capture | crash | timeout | bar contacts | OOB | closest mean (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| latency_0p1s_p3 | 2049 | 78.04% | 19.67% | 2.29% | 390 | 4 | 0.846 |
| latency_0p1s_p3_p0 | 2049 | 76.57% | 20.79% | 2.64% | 406 | 8 | 0.953 |
| latency_0p1s_p3_p0_predict | 2049 | 78.43% | 19.23% | 2.34% | 370 | 7 | 0.802 |
| latency_0p1s_raw | 2049 | 37.82% | 58.22% | 3.95% | 931 | 251 | 1.611 |
| analytic_clean | 2050 | 80.54% | 17.17% | 2.29% | 337 | 10 | 0.806 |

## GO gate (capture >= 65% AND crash >= 10 pp below latency_0p1s_raw)

- latency_0p1s_p3: GO (capture vs raw +40.21 pp = 94.2% of the latency loss, bar contacts vs raw -541 = 91.1% of the excess)
- latency_0p1s_p3_p0: GO (capture vs raw +38.75 pp = 90.7% of the latency loss, bar contacts vs raw -525 = 88.4% of the excess)
- latency_0p1s_p3_p0_predict: GO (capture vs raw +40.61 pp = 95.1% of the latency loss, bar contacts vs raw -561 = 94.4% of the excess)
