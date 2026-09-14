# NavRL v2 ep19100 versus ep24000 forgetting evaluation

Frozen policies, 205 bars, seed 42, deterministic/original inference, 2,049 requested episodes per independent cell.

| condition | epoch | capture | crash | timeout | bar contact | lateral edge98 |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 19100 | 67.79% | 31.87% | 0.34% | 31.67% | 4.48% |
| uniform | 24000 | 72.44% | 25.07% | 2.49% | 24.29% | 3.37% |
| fast1p5 | 19100 | 64.10% | 35.61% | 0.29% | 34.93% | 6.16% |
| fast1p5 | 24000 | 67.35% | 30.75% | 1.90% | 29.97% | 4.92% |

- uniform: ep24000−ep19100 capture **+4.65%** (95% CI +1.85%..+7.45%); **improvement**.
- fast1p5: ep24000−ep19100 capture **+3.25%** (95% CI +0.35%..+6.16%); **improvement**.

Material forgetting detected: **NO**.

This is inference-only. Async resets make cells independent rather than episode-paired.
