# NavRL v2 detector/perception robustness (R3 screen)

Policy SHA-256: `f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40`
Learned detector SHA-256: `15cb90e90615e071c938379acc6c445d3c56f73303e9b6ce915ca7fd695cf492`

Held-out seed47 · 205 bars · U[0.3,1.5] m/s · deterministic · riskcap.

| cell | episodes | capture | crash | timeout | bar contact | mean speed |
|---|---:|---:|---:|---:|---:|---:|
| analytic_clean | 2050 | 80.54% | 17.17% | 2.29% | 95.74% | 2.327 m/s |
| dropout_0p3 | 2049 | 67.84% | 29.33% | 2.83% | 93.01% | 2.305 m/s |
| latency_0p1s | 2049 | 37.82% | 58.22% | 3.95% | 78.04% | 2.332 m/s |
| latency_0p2s | 2049 | 18.50% | 76.48% | 5.03% | 76.32% | 2.322 m/s |
| range_error_0p15m | 2050 | 80.54% | 17.46% | 2.00% | 94.97% | 2.325 m/s |
| range_error_0p30m | 2049 | 80.62% | 16.89% | 2.49% | 95.95% | 2.320 m/s |
| learned_clean | 2049 | 66.62% | 24.94% | 8.44% | 96.87% | 2.279 m/s |

## Delta vs analytic_clean

- **dropout_0p3**: capture -12.70 pp (95% CI -15.35..-10.05), crash +12.16 pp (95% CI +9.60..+14.72), timeout +0.54 pp (95% CI -0.43..+1.51)
- **latency_0p1s**: capture -42.71 pp (95% CI -45.42..-40.00), crash +41.05 pp (95% CI +38.36..+43.74), timeout +1.66 pp (95% CI +0.60..+2.72)
- **latency_0p2s**: capture -62.04 pp (95% CI -64.44..-59.64), crash +59.31 pp (95% CI +56.85..+61.76), timeout +2.73 pp (95% CI +1.59..+3.88)
- **range_error_0p15m**: capture +0.00 pp (95% CI -2.42..+2.42), crash +0.29 pp (95% CI -2.02..+2.61), timeout -0.29 pp (95% CI -1.18..+0.59)
- **range_error_0p30m**: capture +0.09 pp (95% CI -2.33..+2.51), crash -0.28 pp (95% CI -2.59..+2.02), timeout +0.20 pp (95% CI -0.74..+1.13)
- **learned_clean**: capture -13.92 pp (95% CI -16.58..-11.25), crash +7.77 pp (95% CI +5.28..+10.25), timeout +6.15 pp (95% CI +4.78..+7.52)
- **learned_vs_analytic**: capture -13.92 pp (95% CI -16.58..-11.25), crash +7.77 pp (95% CI +5.28..+10.25), timeout +6.15 pp (95% CI +4.78..+7.52)
