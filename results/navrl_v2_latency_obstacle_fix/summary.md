# latency P2 obstacle-map arms (ep25000+riskcap, seed47, 205 bars)

| cell | episodes | capture | crash | timeout | bar contacts | OOB | closest mean (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| latency_0p1s_map_predict | 2050 | 39.66% | 56.29% | 4.05% | 867 | 278 | 1.329 |
| latency_0p1s_map_skip | 2050 | 38.83% | 56.83% | 4.34% | 895 | 252 | 1.579 |
| latency_0p1s_p0_predict | 2050 | 36.88% | 58.44% | 4.68% | 918 | 267 | 1.812 |
| latency_0p1s_raw | 2049 | 37.82% | 58.22% | 3.95% | 931 | 251 | 1.611 |
| analytic_clean | 2050 | 80.54% | 17.17% | 2.29% | 337 | 10 | 0.806 |

## GO gate (capture >= 65% AND crash >= 10 pp below latency_0p1s_raw)

- latency_0p1s_map_predict: NO-GO (capture vs raw +1.84 pp, vs clean -40.88 pp, bar contacts vs raw -64)
- latency_0p1s_map_skip: NO-GO (capture vs raw +1.01 pp, vs clean -41.71 pp, bar contacts vs raw -36)
- latency_0p1s_p0_predict: NO-GO (capture vs raw -0.95 pp, vs clean -43.66 pp, bar contacts vs raw -13)
