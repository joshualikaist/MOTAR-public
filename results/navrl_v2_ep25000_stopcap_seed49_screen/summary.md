# NavRL v2 ep25000 seed49 stopcap screen

Prereg: `docs/prereg_2026-09-02_speed_governor_stopcap_screen.md`

| condition | n | capture | crash | timeout | bar contact | intervention | executed m/s | Δcapture | Δcrash |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 2,049 | 73.16% | 25.18% | 1.66% | 24.30% | 0.00% | 2.940 | +0.00% | +0.00% |
| fixed2p0 | 2,049 | 81.31% | 14.06% | 4.64% | 13.42% | 95.67% | 1.978 | +8.15% | -11.13% |
| riskcap | 2,050 | 81.71% | 15.95% | 2.34% | 15.37% | 25.66% | 2.741 | +8.55% | -9.23% |
| stopcap | 2,051 | 69.19% | 21.31% | 9.51% | 20.38% | 36.85% | 1.963 | -3.97% | -3.88% |
| ttc | 2,051 | 74.70% | 4.24% | 21.06% | 3.85% | 55.94% | 1.519 | +1.54% | -20.94% |

| condition | near-stop | requested m/s | scale | unsafe before | unsafe after | contact actual/executed m/s | contact clearance | contact step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 0.00% | 2.940 | 1.000 | 11.06% | 11.06% | 1.473/3.044 | 0.894 m | 99.5 |
| fixed2p0 | 0.00% | 3.036 | 0.669 | 11.60% | 8.34% | 1.050/1.972 | 1.148 m | 137.3 |
| riskcap | 0.00% | 2.953 | 0.933 | 12.18% | 8.74% | 1.122/2.024 | 0.981 m | 97.4 |
| stopcap | 30.12% | 3.059 | 0.669 | 36.87% | 2.62% | 0.813/0.302 | 0.644 m | 121.3 |
| ttc | 42.24% | 3.112 | 0.515 | 48.12% | 0.00% | 0.555/0.255 | 0.657 m | 137.2 |

## Preregistered verdicts

- M1 machinery: **IMPLEMENTATION_VOID** (stopcap unsafe-after 2.62%, limit 1.00%)
- Q1 release mechanism: **MECHANISM_UNSUPPORTED** (riskcap−fixed2p0 capture +0.40%, CI95 [-1.98%, +2.78%])
- Q2 stopcap adoption: **NOT_JUDGED_M1_VOID** (crash +5.36% CI95 [+2.98%, +7.73%], timeout 9.51%, capture -12.52%; gates {'crash_improvement': False, 'liveness': False, 'capture_cost': False})
- Q3 filter dependence: **FILTER_DEPENDENT** (off−riskcap crash +9.23%)

ttc arm is reference-only by prereg (no gate).
