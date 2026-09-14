# NavRL v2 ep24000 speed-governor screen

| condition | n | capture | crash | timeout | bar contact | intervention | executed m/s | Δcapture | Δcrash | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| off | 2,050 | 72.44% | 25.07% | 2.49% | 24.29% | 0.00% | 2.965 | +0.00% | +0.00% | FAIL |
| fixed2p0 | 2,049 | 78.53% | 16.06% | 5.42% | 15.18% | 97.00% | 1.985 | +6.09% | -9.02% | FAIL |
| fixed1p5 | 2,051 | 74.65% | 16.82% | 8.53% | 15.89% | 99.01% | 1.496 | +2.21% | -8.25% | FAIL |
| clearance | 2,049 | 69.11% | 14.30% | 16.59% | 13.08% | 46.55% | 1.699 | -3.33% | -10.77% | FAIL |
| ttc | 2,049 | 69.59% | 6.83% | 23.57% | 5.47% | 57.22% | 1.520 | -2.84% | -18.24% | FAIL |

| condition | near-stop | requested m/s | scale | unsafe before | unsafe after | contact actual/executed m/s | contact clearance | contact step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| off | 0.00% | 2.965 | 1.000 | 12.23% | 12.23% | 1.483/3.028 | 0.770 m | 96.1 |
| fixed2p0 | 0.00% | 3.051 | 0.666 | 13.13% | 9.44% | 1.154/1.970 | 0.478 m | 121.2 |
| fixed1p5 | 0.00% | 3.101 | 0.494 | 15.09% | 9.36% | 0.937/1.485 | 0.629 m | 163.3 |
| clearance | 38.60% | 3.107 | 0.572 | 44.99% | 0.00% | 0.577/0.278 | 0.608 m | 146.1 |
| ttc | 42.04% | 3.123 | 0.509 | 48.64% | 0.00% | 0.569/0.351 | 1.065 m | 154.4 |

| condition | capture delta 95% CI | crash delta 95% CI | capture/crash/timeout mean step |
|---|---:|---:|---:|
| off | [-2.74%, +2.74%] | [-2.65%, +2.65%] | 119.2/94.5/601.0 |
| fixed2p0 | [+3.46%, +8.71%] | [-11.48%, -6.56%] | 167.0/119.7/601.0 |
| fixed1p5 | [-0.49%, +4.91%] | [-10.73%, -5.77%] | 213.3/159.7/601.0 |
| clearance | [-6.11%, -0.55%] | [-13.19%, -8.36%] | 172.7/140.8/601.0 |
| ttc | [-5.62%, -0.07%] | [-20.41%, -16.07%] | 189.5/147.7/601.0 |

Adaptive GO: **NO**; selected: **none**.

Gate fixed before evaluation: crash delta <= -3.0 pp, capture delta >= -1.0 pp, timeout <= 5%; only adaptive arms authorize training.
