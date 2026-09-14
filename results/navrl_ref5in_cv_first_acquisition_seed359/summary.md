# ref5in first-acquisition / never-acquired telemetry (seed 359)

정책·보상·센서 range·아레나·horizon을 바꾸지 않은 동결 replay 진단이다. P3를 해제하지 않고
P2/D1 판정도 바꾸지 않는다.

| heading | outcome | episodes | never acq. | first-visible mean | median | vis→hid /ep |
|---|---|---:|---:|---:|---:|---:|
| toward | capture | 1,932 | 0.00% | 82.1 | 72 | 2.068 |
| toward | crash | 115 | 99.13% | 77.0 | 77 | 0.026 |
| toward | timeout | 2 | 100.00% | — | — | 0.000 |
| away | capture | 762 | 0.00% | 318.8 | 312 | 2.142 |
| away | crash | 189 | 93.12% | 360.8 | 381 | 0.249 |
| away | timeout | 1,098 | 87.52% | 565.2 | 569 | 0.173 |

사전 판정: `initial_acquisition_range_contract_channel_supported`

- primary: away timeout−capture never-acquired **+87.52 pp** (임계 30 pp) → 통과
- secondary: away timeout−capture first-visible **+246.4 step** (임계 100 step) → 해당 없음 (primary 통과 시 미적용)
