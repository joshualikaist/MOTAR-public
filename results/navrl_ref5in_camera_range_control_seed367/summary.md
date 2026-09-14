# ref5in 초기 관측가능성 인과 대조 (seed 367)

target camera range **한 값만** 바꾼 2-arm 대조다. 정책·보상·아레나·horizon·heading 불변.
P2/D1 재판정 없음, P3 해제 없음.

| arm | camera range | capture | crash | timeout | never-acq (timeout) |
|---|---:|---:|---:|---:|---:|
| camera_20m | 20 m | 36.39% | 7.80% | 55.80% | 89.07% |
| camera_28m | 28 m | 74.96% | 6.88% | 18.16% | 79.57% |

사전 판정: `initial_unobservability_dominant_cause_supported`

- primary: timeout 감소 **+37.65 pp** (임계 20 pp) → 통과
- guard: crash 증가 **-0.92 pp** (한계 10 pp) → 이동 없음

**한계(사전 명시):** 정책은 20 m로 학습됐다. 28 m arm은 학습 범위 밖이므로 timeout이
줄지 않아도 "비관측이 원인이 아니다"로 읽을 수 없다 — "이 정책이 장거리 검출을
활용하지 못한다"와 구분되지 않는다. 감소가 관측될 때만 단방향으로 강한 증거다.
