# ref5in post-P2 outcome diagnostic v2

동일 seed·checkpoint의 deterministic replay로 speed bin을 실제 [0.3,1.5] 지원범위에 맞췄다.
전역 outcome과 distance/pattern/bearing count가 v1과 byte-level JSON 값으로 동일해야 유효하다.

| speed bin | episodes | capture | crash | timeout |
|---|---:|---:|---:|---:|
| q0 | 2,013 | 66.57% | 24.94% | 8.49% |
| q1 | 2,032 | 68.31% | 24.85% | 6.84% |
| q2 | 2,053 | 68.14% | 26.21% | 5.65% |
| q3 | 2,096 | 69.04% | 27.86% | 3.10% |

## Distance × pattern

- q0/cv: n=892, capture 78.14%, crash 21.86%, timeout 0.00%
- q0/waypoint: n=844, capture 77.96%, crash 21.92%, timeout 0.12%
- q1/cv: n=1,058, capture 72.02%, crash 25.99%, timeout 1.98%
- q1/waypoint: n=1,098, capture 76.23%, crash 23.13%, timeout 0.64%
- q2/cv: n=1,120, capture 61.96%, crash 26.96%, timeout 11.07%
- q2/waypoint: n=1,171, capture 68.40%, crash 28.44%, timeout 3.16%
- q3/cv: n=993, capture 48.84%, crash 29.00%, timeout 22.16%
- q3/waypoint: n=1,018, capture 62.87%, crash 29.17%, timeout 7.96%

이 평가는 계측 정정이며 P2 판정이나 P3 차단 상태를 바꾸지 않는다.
