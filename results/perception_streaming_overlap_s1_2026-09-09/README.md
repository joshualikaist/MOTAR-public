# S1 optical-flow overlap

상태: **COMPLETE — OUTPUT PRESERVED, SELECTED.** NPS test는 열지 않았다.

`PerceptionPipeline`에서 후보와 무관한 Farnebäck flow를 CPU worker에 제출한 뒤 GPU detector를
실행한다. detector가 끝나면 flow를 회수해 후보 의존 GMC와 5×12 motion feature를 계산한다.
기본 경로가 overlap이며 `--serial-flow`가 같은 코드의 기준선을 실행한다.

## 전체 validation 보존 검증

같은 2,296 validation JPEG를 `detector_runs/venv`에서 직렬·겹침 각각 3회 실행했다.
여섯 실행 모두 아래 조건을 통과했다.

- candidate dictionary exact mismatch: **0 / 2,296**
- P7c predicted rank mismatch: **0 / 2,296**
- cached 5×12 float32 motion feature byte mismatch: **0 / 2,296**
- motion bytes SHA-256: `33c1af2909df…4e4f9d` (6회 동일)
- selected/hit/no-lock/false-lock/center error/loss/reacquisition: cached baseline과 exact match
- test 사용: **false**

따라서 이 validation 입력과 기록된 runtime에서 겹침 구현은 출력을 비트 단위로 보존한다.

## RTX 3070 latency

GPU 학습 없이 `serial, overlap, overlap, serial, serial, overlap` 순서로 교차 측정했다.
아래 단계 값은 세 실행의 6,888 frame timings를 합친 mean/P50/P95이고 ms/frame이다.

| 경로 | 단계 | Mean | P50 | P95 |
|---|---|---:|---:|---:|
| 직렬 | detector | 27.28 | 25.23 | 31.77 |
| 직렬 | motion | 32.87 | 35.67 | 39.28 |
| 직렬 | selector | 2.41 | 2.33 | 2.75 |
| 직렬 | decode 포함 전체 | **67.91** | **67.89** | **70.64** |
| 겹침 | detector | 27.18 | 24.39 | 32.51 |
| 겹침 | detector 뒤 대기 + GMC | 9.31 | 15.38 | 17.67 |
| 겹침 | selector | 2.33 | 2.31 | 2.70 |
| 겹침 | decode 포함 전체 | **44.49** | **45.37** | **47.48** |

실행별 FPS는 직렬 `14.7073, 14.6981, 14.7688`, 겹침
`22.2797, 22.5830, 22.5691`이다. 3회 평균은 **14.7247 → 22.4773 FPS**로
**52.65 % 증가**, mean latency는 **34.49 % 감소**했다. 데스크톱과 기존 RustDesk 프로세스는
기록했고 다른 학습·평가 compute job은 없었다.

[summary.json](summary.json)에 runtime, 지표, 실행별 FPS, pooled latency, 원본 report/timing/receipt
SHA-256을 보존했다. 원본 실행 자료는 저장소 밖
`detector_runs/results/perception_streaming_overlap_s1_2026-09-09/`에 있다.
clean-tree 전체 Python suite는 **1,231 tests OK (4 skipped)**다.

이 값에는 저장된 JPEG decode가 포함되며 camera transport, ROS, queue, PPO/control은 포함하지 않는다.
실제 카메라와 탑재 장치의 처리율을 뜻하지 않는다. 다음 perception 단계는 P8 오차 모델이다.
