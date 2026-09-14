# Seed 367 OOB exit forensics — post-run analysis

이 문서는 동결 정책의 2-arm 평가가 끝난 뒤 작성한 분석이다. 원시 수치와 계약은
`summary.json`, 사람이 읽는 기존 camera-range 판정은 `summary.md`에 있다. 이 분석은 P2/D1/P3
판정을 바꾸지 않으며 새 PPO 결과를 주장하지 않는다.

## 결과

| arm | episodes | capture | crash | timeout | OOB | OOB 중 never acquired |
|---|---:|---:|---:|---:|---:|---:|
| camera 20 m | 2,050 | 36.39% | 7.80% | 55.80% | 158 | 152 / 158 = **96.20%** |
| camera 28 m | 2,049 | 74.96% | 6.88% | 18.16% | 138 | 120 / 138 = **86.96%** |

20 m의 never-acquired OOB는 exit 순간 평균 속도 **1.359 m/s**, arena 중심에서 바깥쪽 성분
**+1.002 m/s**, 실제 목표 closing speed **-0.834 m/s**였다. 따라서 주 실패는 정지·수동 표류가
아니라, 표적을 한 번도 보지 못한 정책이 목표에서 멀어지는 방향으로 능동 비행하다 경계를 넘는
것이다. exit step 중앙값은 84 step, 즉 RL step 0.1 s 계약에서 약 **8.4 s**다.

28 m에서도 never-acquired OOB는 남았고, 해당 집단은 바깥쪽 **+1.260 m/s**, 목표 closing
**-1.159 m/s**였다. 반면 acquired OOB 18건은 목표 closing **+0.371 m/s**와 바깥쪽
**+0.619 m/s**가 동시에 양수다. 이는 소수의 별도 채널, 즉 표적을 쫓는 방향 자체가 arena 밖인
경우가 있음을 뜻한다. arm 간 운동학 평균 차이는 서로 다른 생존·취득 cohort가 섞이므로 camera
range의 인과 효과 크기로 해석하지 않는다.

## 관측 계약 감사

현재 perception actor는 898-D structured observation만 받는다:

- 현재 36x4 LiDAR static scan
- obstacle/robot/target 각각 5-step history
- robot history의 body velocity, yaw rate, previous action, height, validity
- detector/tracker가 만든 target history

actor에는 **world XY, arena side, 네 경계까지 거리, geofence, episode progress가 없다**.
`_arena_xy_norm`은 actor가 아니라 asymmetric critic의 GT target-distance 정규화에만 쓰인다.
네트워크의 `is_rnn()`은 false이며, 5-step history는 0.1 s RL step 기준 약 0.5 s뿐이다. arena
경계는 물리 wall/LiDAR return도 아니고 숫자 OOB termination이므로, blind phase actor는 현재
위치에서 어느 쪽 경계가 가까운지 직접 알 수 없다.

따라서 “episode를 더 길게 하면 결국 찾는다”는 처방은 현재 계약에서는 성립하지 않는다. 실제
never-acquired OOB는 600-step timeout보다 훨씬 이른 median 84 step에 종료된다. 속도 상향도 이미
밖으로 능동 비행하는 속도를 키우고 정지거리·선회반경을 늘리므로 우선 처방이 아니다.

## 다음 단일 변경축

다음 PPO lineage의 첫 A/B는 **boundary observability만** 바꾼다. camera range, target range,
episode length, speed/tilt, reward, obstacle representation은 고정한다.

권장 최소 actor feature는 VIO/GPS/known-map geofence가 있다는 실기 계약을 명시한 body-frame
boundary range 4개다. 각 feature는 현재 position/yaw와 env bounds에서 계산하며 sensor noise와
dropout 계약을 함께 둔다. 이것은 simulator-only oracle을 몰래 넣는 것이 아니라 실기에서 제공할
localization/geofence 센서를 선언하는 새 task contract다. 그런 센서를 허용하지 않는 연구 질문이면
대안은 egomotion-integrated recurrent coverage belief이며, 이는 architecture와 memory를 동시에
바꾸므로 첫 실험으로 쓰지 않는다.

사전 gate 제안:

1. 20 m / 1 bar / seed 367 조건에서 never-acquired OOB share와 OOB rate를 primary로 측정한다.
2. camera range 28 m는 positive control로만 유지하고 학습 입력 범위로 채택하지 않는다.
3. capture 개선만으로 통과시키지 않는다. blind-search OOB가 줄고 bar-contact/crash가 악화되지
   않아야 한다.
4. 한 run에서 horizon·speed·reward를 함께 바꾸지 않는다.
5. observation schema 변경이므로 frozen 898-D checkpoint warm-start 결과를 정식 비교로 쓰지 않고
   fresh policy를 사용한다.

## 현재 결론

camera 20 m 자체가 유일한 결함은 아니다. 사용자가 원하는 과제는 target이 sensor 밖에 있어도
탐색해 취득하는 것이므로, 핵심 누락은 blind phase의 **search state + boundary observability**다.
이번 동결-policy 계측은 그 설계를 정당화하지만 새 설계의 성능을 증명하지는 않는다.
