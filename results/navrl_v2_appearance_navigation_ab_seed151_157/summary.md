# appearance-envelope navigation A/B (ep25000+riskcap, 205 bars, seeds 151/157)

| cell | capture | crash | timeout | n |
|---|---:|---:|---:|---:|
| seed151 envelope analytic | 51.05% | 26.45% | 22.50% | 2049 |
| seed151 envelope learned_v7 | 67.32% | 29.71% | 2.98% | 2050 |
| seed151 nominal analytic | 81.06% | 16.45% | 2.49% | 2049 |
| seed151 nominal learned_v7 | 75.22% | 21.71% | 3.07% | 2050 |
| seed157 envelope analytic | 48.71% | 27.43% | 23.87% | 2049 |
| seed157 envelope learned_v7 | 65.93% | 31.48% | 2.59% | 2049 |
| seed157 nominal analytic | 80.77% | 16.59% | 2.64% | 2049 |
| seed157 nominal learned_v7 | 76.57% | 21.47% | 1.95% | 2049 |

- **E1 (NI, nominal)**: learned−analytic -5.021 pp, CI [-6.799, -3.243] → **FAIL** (margin −2.0 pp)
- **E2 (envelope cost)**: envelope+learned vs nominal+analytic -14.292 pp, CI [-16.171, -12.412]
- **E3 (bootstrap collapse)**: envelope+analytic vs nominal+analytic -31.040 pp, CI [-32.987, -29.092]
