# ref5in D1 held-out decision — FAIL

D1은 P2를 재채점하지 않으며 P3를 자동 해제하지 않는다.

| gate | value | limit | pass |
|---|---:|---:|:---:|
| global crash | 18.62% | ≤27% | True |
| q3 crash | 19.96% | ≤30% | True |
| q3/CV timeout | 15.98% | ≤12% | False |

## D0 대비 기술적 변화 (서로 다른 평가 seed)

| stratum | capture | crash | timeout |
|---|---:|---:|---:|
| global | +8.80 pp [+7.44, +10.16] | -7.36 pp [-8.63, -6.09] | -1.44 pp [-2.12, -0.76] |
| q3 | +14.28 pp [+11.37, +17.19] | -9.13 pp [-11.74, -6.52] | -5.15 pp [-7.16, -3.14] |
| q3/CV | +15.19 pp [+10.94, +19.44] | -9.02 pp [-12.74, -5.31] | -6.17 pp [-9.57, -2.77] |

D1은 유의한 방향의 개선을 만들었지만 q3/CV timeout 절대 gate를 통과하지 못했다. 비교에는 adaptation과 q3 학습분포 변경이 함께 들어 있으므로 둘의 효과를 분리하지 않는다.

Checkpoint SHA-256: `197ea26999d6bb9cf23c4e5a55acbe945f89985e2384687d60ab1dbae66a278e`
