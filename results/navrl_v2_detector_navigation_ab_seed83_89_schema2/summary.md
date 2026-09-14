# Detector navigation A/B — schema-v2

Frozen ep25000+riskcap, 205 bars, deterministic exact-600; learned threshold 0.55.

| seed | analytic capture/crash/timeout | learned capture/crash/timeout | capture delta |
|---:|---:|---:|---:|
| 83 | 80.39/16.54/3.07% | 79.86/16.92/3.22% | -0.53 pp |
| 89 | 80.59/17.07/2.34% | 80.97/16.11/2.93% | +0.38 pp |

## Primary non-inferiority result

- pooled analytic capture: **80.49%** (n=4100)
- pooled learned capture: **80.41%** (n=4100)
- learned−analytic: **-0.07 pp**, 95% CI **[-1.79, +1.64] pp**
- preregistered margin −2.0 pp: **PASS**
