# latency compensation arms (ep25000+riskcap, seed47, 205 bars)

| cell | episodes | capture | crash | timeout | bar-contact share |
|---|---:|---:|---:|---:|---:|
| analytic_clean | 2050 | 80.54% | 17.17% | 2.29% | 95.7% |
| latency_0p1s_p0 | 2049 | 37.73% | 57.74% | 4.54% | 76.1% |
| latency_0p1s_p0p1 | 2050 | 28.98% | 66.54% | 4.49% | 78.9% |
| latency_0p1s_raw | 2049 | 37.82% | 58.22% | 3.95% | 78.0% |

## GO gate (capture >= 65% AND crash >= 10 pp down vs latency_0p1s_raw)

- latency_0p1s_p0: NO-GO (vs clean -42.81 pp, vs raw -0.10 pp)
- latency_0p1s_p0p1: NO-GO (vs clean -51.56 pp, vs raw -8.85 pp)
