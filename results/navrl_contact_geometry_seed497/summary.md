# Contact-corridor forensics (seed 491, 0 distractors, 70 bars)

| arm | crash | contacts | vertical_out | behind | lateral | no_return | **in_corridor** |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 21.82% | 314 | 0.0% | 2.5% | 54.5% | 20.7% | **22.3%** |
| riskcap | 17.86% | 241 | 0.0% | 5.4% | 60.2% | 16.2% | **18.3%** |

## 가설 대조

| arm | C: 실제속도 기준 in_corridor | C: 재분류 수 | 평균 명령-실제 각차 | D: 회랑 내 막대 2개+ | 평균 막대 수 |
|---|---:|---:|---:|---:|---:|
| off | 14.0% | -26 | 29.5° | 12.1% | 0.74 |
| riskcap | 6.2% | -29 | 35.8° | 12.0% | 0.69 |

## A1 재현 판정 (seed 491 CI 게이트, 결과 이전 동결)

| arm | lateral+no_return | 허용 구간 | |
|---|---:|---|---|
| off | 75.2% | [72.5%, 81.9%] | 통과 |
| riskcap | 76.3% | [72.8%, 83.4%] | 통과 |

**verdict_replication: REPLICATED**

