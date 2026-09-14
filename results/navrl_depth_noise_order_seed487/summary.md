# Depth noise model order (Phase 1, seed 487, 0 distractors)

| cell | capture | crash | timeout | episodes |
|---|---:|---:|---:|---:|
| linear | 68.23% | 21.91% | 9.86% | 2,049 |
| stereo_d455 | 68.57% | 21.23% | 10.20% | 2,049 |
| stereo_d435 | 70.77% | 20.06% | 9.18% | 2,049 |

**verdict_rq: VARIANCE_INSENSITIVE**

- primary (linear - stereo_d455): **-0.34 pp** 95% CI [-3.19, +2.51]
- stress (linear - stereo_d435): -2.54 pp 95% CI [-5.36, +0.28]
- prediction 1 (capture monotone decreasing): VIOLATED

Gates (frozen before measurement): MODEL_ORDER_MATTERS if delta >= 10 pp and CI excludes 0;
PARTIAL if CI excludes 0 but delta < 10 pp; VARIANCE_INSENSITIVE if CI upper < 5 pp;
otherwise INCONCLUSIVE.
