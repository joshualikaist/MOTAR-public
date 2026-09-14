# Active-search mapped-geofence A/B (seed 367)

| arm | capture | crash | timeout | OOB | never-acq OOB/all | non-OOB crash |
|---|---:|---:|---:|---:|---:|---:|
| control | 39.04% | 22.21% | 38.75% | 21.96% | 21.28% | 0.24% |
| geofence | 85.75% | 7.91% | 6.34% | 7.47% | 7.32% | 0.44% |
| geofence_masked | 39.78% | 5.22% | 55.00% | 4.93% | 4.88% | 0.29% |

결론: **PASS_MECHANISM_UNRESOLVED**

- primary gain: 13.96% (gate >= 3.00%)
- non-OOB crash rise: 0.20% (guard <= 2.00%)
- masked loss: -2.44%; mechanism pass=False

P2 STRICT FAIL, D1 FAIL, P3 BLOCKED는 바뀌지 않는다.
