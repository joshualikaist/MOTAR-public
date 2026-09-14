# latency budget under the corrected (timestamp-aware) model

All latency cells run with NAVRL_LATENCY_EGO_MOTION_FIX=1, i.e. the delayed detection
is lifted to world with the pose it was taken at. For scale, 0.1 s WITHOUT that fix --
the number R3 reported -- was capture 37.82% / crash 58.22%.

| cell | episodes | capture | crash | timeout | bar contacts | OOB | closest mean (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| analytic_clean | 2050 | 80.54% | 17.17% | 2.29% | 337 | 10 | 0.806 |
| latency_0p1s_p3 | 2049 | 78.04% | 19.67% | 2.29% | 390 | 4 | 0.846 |
| latency_0p2s_p3 | 2049 | 76.62% | 20.40% | 2.98% | 397 | 10 | 0.940 |
| latency_0p3s_p3 | 2049 | 72.57% | 24.79% | 2.64% | 494 | 7 | 0.940 |
| latency_0p5s_p3 | 2049 | 64.76% | 32.41% | 2.83% | 648 | 7 | 1.054 |

## cost vs clean (within 5 pp = still flyable)

- latency_0p1s_p3: capture -2.50 pp, crash +2.50 pp, bar contacts +53 -> OK
- latency_0p2s_p3: capture -3.91 pp, crash +3.23 pp, bar contacts +60 -> OK
- latency_0p3s_p3: capture -7.96 pp, crash +7.62 pp, bar contacts +157 -> DEGRADED
- latency_0p5s_p3: capture -15.77 pp, crash +15.24 pp, bar contacts +311 -> DEGRADED
