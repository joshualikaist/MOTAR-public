# ep24000 causal checks 1--3

- Checkpoint SHA-256: `82f7978b42d9d9e95adcc638a40ae85fb3736fd897ae39cb8aa8333be39cf23f`
- Training performed: **no** (frozen-weight inference only)
- Preregistered plan: `RESEARCH_PLAN.md §8.8`

## Mirror audit (seed 42, 205 bars)

| arm | episodes | capture | crash | timeout | bar contact / episodes |
|---|---:|---:|---:|---:|---:|
| original pi | 4096 | 70.97% | 25.71% | 3.32% | 25.00% |
| conjugate M pi M | 4096 | 70.17% | 26.56% | 3.27% | 25.51% |

Capture difference (conjugate-original): **-0.81 pp** (95% CI -2.78 to +1.17 pp).
Crash difference: **+0.85 pp** (95% CI -1.05 to +2.76 pp).
Preregistered aggregate outcome-chirality verdict: **NO**.

Exact action-pair MAE [x,y,z,yaw]: `[0.926086, 1.235224, 0.416258, 1.002006]`; lateral sign mismatch: **73.08%** over 537168 comparable samples.

Outcome arms share a seed but are not episode-paired after asynchronous resets; only the action comparison is exactly paired.

Initial target-bearing outcome: negative-y 1400/1967 (71.17%) versus positive-y 1425/2008 (70.97%); positive-negative **-0.21 pp** (95% CI -3.03 to +2.61 pp).

## Independent seed replication (205 bars, original pi)

| seed | episodes | capture | crash | timeout | bar contact / episodes |
|---:|---:|---:|---:|---:|---:|
| 42 | 2050 | 72.44% | 25.07% | 2.49% | 24.29% |
| 43 | 2049 | 72.77% | 24.74% | 2.49% | 23.87% |

Capture difference (seed43-seed42): **+0.33 pp** (95% CI -2.40 to +3.06 pp).
Practical replication (<=3 pp and Wilson intervals overlap): **PASS**.
