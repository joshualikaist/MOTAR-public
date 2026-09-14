# NavRL v2 riskcap post-adaptation validation

## Uniform seed45

| policy | n | capture | crash | timeout | bar contact | intervention | executed m/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform_off | 2,049 | 70.03% | 27.87% | 2.10% | 27.09% | 0.00% | 2.961 |
| uniform_source_riskcap | 2,050 | 78.20% | 17.80% | 4.00% | 17.02% | 27.78% | 2.754 |
| uniform_trained_riskcap | 2,049 | 81.94% | 15.67% | 2.39% | 15.13% | 26.30% | 2.745 |

Mechanism replication: **PASS**.

- capture: +8.16 pp (95% CI +5.49..+10.83)
- crash: -10.06 pp (95% CI -12.61..-7.51)
- timeout: +1.90 pp (95% CI +0.85..+2.95)

Adaptation: non-inferior **True**, useful **True**, decision **PASS**.

- capture: +3.75 pp (95% CI +1.30..+6.19)
- crash: -2.14 pp (95% CI -4.42..+0.15)
- timeout: -1.61 pp (95% CI -2.68..-0.53)
- intervention: -1.48 pp

Winner: **trained riskcap**.

## Fixed-speed seed46

| speed | off capture/crash | winner capture/crash | Δcapture (95% CI) | Δcrash (95% CI) | direction |
|---:|---:|---:|---:|---:|---|
| 0.3 | 71.90%/24.88% | 81.84%/15.18% | +9.94 pp (95% CI +7.38..+12.51) | -9.70 pp (95% CI -12.13..-7.27) | PASS |
| 0.9 | 71.89%/25.77% | 80.77%/16.59% | +8.88 pp (95% CI +6.29..+11.47) | -9.18 pp (95% CI -11.66..-6.69) | PASS |
| 1.5 | 67.98%/30.31% | 75.51%/22.29% | +7.53 pp (95% CI +4.78..+10.27) | -8.01 pp (95% CI -10.70..-5.33) | PASS |

Final generalization: **PASS**.
