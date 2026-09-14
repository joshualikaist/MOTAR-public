# Speed × density interaction — schema-v2 primary test

Seeds 59/61; deterministic ep25000+riskcap; exact 600 actions; shared source `3303599b48b5…`.

## Seed 59

| bars | 0.3 m/s | 1.5 m/s | fast−slow |
|---:|---:|---:|---:|
| 130 | 87.96% | 87.60% | -0.36 pp |
| 160 | 86.29% | 84.83% | -1.46 pp |
| 190 | 83.85% | 79.85% | -4.00 pp |
| 205 | 81.32% | 75.38% | -5.94 pp |

## Seed 61

| bars | 0.3 m/s | 1.5 m/s | fast−slow |
|---:|---:|---:|---:|
| 130 | 88.92% | 87.99% | -0.93 pp |
| 160 | 86.54% | 83.99% | -2.54 pp |
| 190 | 83.80% | 79.94% | -3.86 pp |
| 205 | 82.24% | 76.43% | -5.81 pp |

## Preregistered primary test

Aggregate-binomial logistic model: `capture ~ seed + density + fast + density:fast`.
Interaction LR chi-square(1) = **12.7603**, p = **0.000354046**.
Decision at alpha=0.05: **interaction detected**.

All four densities are within the trained support. No OOD 220-bar cell enters the primary test.
