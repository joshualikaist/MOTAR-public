# Corrected non-overlap physical route-off held-out result

Status: **COMPLETE_VALID_WITH_METADATA_ERRATUM**

| bars | n | capture (Wilson 95%) | crash | timeout | bar contact |
|---:|---:|---:|---:|---:|---:|
| 70 | 2049 | 83.70% [82.04, 85.24] | 15.91% | 0.39% | 15.67% |
| 85 | 2051 | 80.94% [79.18, 82.58] | 18.82% | 0.24% | 18.43% |
| 100 | 2049 | 77.75% [75.89, 79.49] | 21.86% | 0.39% | 21.67% |
| 115 | 2049 | 73.45% [71.50, 75.32] | 26.35% | 0.20% | 26.06% |
| 130 | 2050 | 69.17% [67.14, 71.13] | 30.44% | 0.39% | 30.24% |
| 145 | 2049 | 65.54% [63.46, 67.57] | 34.16% | 0.29% | 34.16% |

70→145 capture change: **-18.16 pp**; mean **-3.63 pp / 15 bars**.
Timeout stays below 0.4%; the density loss is almost entirely bar contact, not timeout.

## Metadata erratum

Raw `v2_evaluation_contract.target_speed_max_mps` says 1.5 m/s because of a redundant serializer constant. The measured condition, log, runtime validator and speed strata all prove `U[0.3,1.25] m/s`. Raw artifacts were not edited; the serializer is fixed for future runs.

## Claim boundary

one incomplete route-off seed-911 policy at trained densities 70-145; no 205/routed/hardware/sim-to-real claim.
