# MOTAR perception 최종 구현 계획

작성: 2026-09-07  
상태: **IMPLEMENTATION AUTHORITY**  
대체 대상: 시뮬레이션 색 detector 개선과 SAM-in-sim 형상 detector 경로

## 결정

현행 `red detector`, learned v7 CNN, single-centroid 경로는 측정된 historical baseline으로만 보존한다.
동색 distractor 5개 조건에서 v7 false target lock rate(FTLR)는 90.27%였고, 현재 simulator에는 실제
quadrotor 외형을 학습할 영상 자산이 없다. 따라서 이 detector를 개선하거나 SAM을 simulator 영상에
연결하는 작업은 수행하지 않는다.

최종 인지 경로는 다음과 같다.

```text
NPS detector
  → Det-Fly zero-shot + target pixel-size별 recall
  → K=5 UAV candidates [u,v,w,h,confidence,appearance_64D]
  → association 비교: CNN only / CNN+KF / CNN+GRU / CNN+Temporal Transformer
  → 실사 오차 측정: miss, FP, bearing, ID switch, reacquisition, latency, range uncertainty
  → 측정 오차 분포를 simulator에 주입
  → PPO 재적응 및 held-out closed-loop 평가
```

## 구현 단계와 완료 조건

| ID | 구현 | 고정 조건 | 산출물·완료 조건 |
|---|---|---|---|
| P1 | 문서와 사이트 상태 정리 | 색 detector=`CURRENT BASELINE — KNOWN FAILURE`; SAM=`ARCHIVED` | 메인 architecture가 이 문서의 경로를 표시 |
| P2 | NPS baseline 동결 | 현재 YOLO checkpoint와 NPS split 보존 | checkpoint hash, split manifest, P/R/mAP50/mAP50-95 기록 |
| P3 | Det-Fly zero-shot **COMPLETE** | NPS checkpoint 무변경; 재튜닝 전 평가 | 13,271장 두 arm 전수; native AP30 0.0035, scale-matched AP30 0.2382; raw predictions+receipt |
| P3-F | 합동 detector **COMPLETE** | Det-Fly `020` test 봉인 후 validation 선택 | native AP30 0.7389; checkpoint SHA 동결; threshold test tuning 없음 |
| P4 | multi-candidate 출력 **COMPLETE** | Top-K=5; 후보별 `[u,v,w,h,c,e64]` | NPS val/test timestamp·schema·semantic·hash verification PASS |
| P5 | CNN+KF **COMPLETE — REJECTED** | P4 detector 동결 | test hit 0.4655 vs CNN 0.6145; proxy false lock 0.4843 vs 0.2450 |
| P6 | CNN+GRU **COMPLETE — NOT SELECTED** | history 8; hidden 128 | validation utility 0.6315 |
| P7 | CNN+Temporal Transformer **COMPLETE — SELECTED** | history 16; d=128; 4 heads; 2 layers | validation utility 0.6725 vs CNN-only 0.6655; test 미실행 |
| P7b/P7c | 후보 보존 + motion/GMC **COMPLETE** | corrected P7c v2 선택; 이전 P7 선택 갱신 | validation utility 0.68554; [결과](../../results/perception_p7bc_2026-09-08/README.md); test 미실행 |
| P7e/streaming | crop verifier **NOT SELECTED**; streaming+overlap **COMPLETE** | P7c 유지; test 미실행 | RGB parity PASS; 14.72→22.48 FPS; [S1 결과](../../results/perception_streaming_overlap_s1_2026-09-09/README.md) |
| P7d | learned appearance/ReID 준비 | track-ID 데이터 필요 | [착수 경계](perception_p7d_identity_boundary_2026-09-08.md) |
| P8 | **조건부 pixel/시간 오차 측정 COMPLETE** | 선택된 detector/association 동결 | 단일-GT 1,310프레임; sparse bins 명시; [결과](../../results/perception_p8_2026-09-09/README.md); degree/range/identity는 자료 대기 |
| P9 | simulator 오류 주입 | renderer로 UAV detector를 재학습하지 않음 | 실사 분포와 주입 분포의 적합도 및 seed 고정 재현성 |
| P10 | PPO 재적응·평가 | perception arm 외 reward·arena·target motion·filter 계약 고정 | held-out capture/crash/timeout과 95% CI, training/evaluation seed 분리 |

P3의 zero-shot 결과가 낮다는 이유만으로 임의 threshold에서 실패를 선언하지 않는다. 전체와 크기별
성능, 오류 유형을 먼저 보고한다. NPS-only 모델이 Det-Fly에서 실용적인 후보 생성을 하지 못하면 두
dataset을 합쳐 재학습하되, Det-Fly test split은 모델 선택과 threshold 조정에서 봉인한다.

P3 결과, native 4K와 ÷4 scale-matched의 IoU 0.3 AP는 각각 `0.0035`, `0.2382`였다.
사전 등록 규칙으로 **SIZE-DOMINANT ZERO-SHOT FAILURE**로 판정했다. 배경 4종 mapping은
배포 metadata에 없어 임의 생성하지 않았고 source group 010/020으로 대체했다. 상세
계약·지표·hash는 `perception_p3_detfly_execution_2026-09-07.md`가 정본이다.

P3 후속 split과 통합 dataset도 완료했다. Det-Fly `020` 6,913장을 source-group
단위 sealed test로 분리했고, `010`은 가장 큰 자연 ID 단절 `4333→4752`에서
train 4,200 / val 2,158장으로 나뉘었다. NPS 기존 clip split과 Det-Fly native 640 crop을
결합한 train/val은 29,824 / 9,307 samples이며 test는 학습 YAML에 없다. 상세 규칙과
receipt는 `perception_joint_dataset_v1_2026-09-07.md`가 정본이다.

P3 후속 합동 학습, P4, P5는 2026-09-08에 완료했다. validation-selected 합동 detector는 sealed
Det-Fly native 4K에서 AP30 `0.7389`를 기록했다. P4 Top-K=5 후보 계약은 NPS val/test에서 검증됐다.
P5 KF v1은 NPS test의 frame hit와 proxy false lock 모두 CNN-only보다 나빠 **REJECTED**했다. 실제
ID switch/FTLR와 degree bearing error는 원 track identity와 intrinsic 부재로 보류한다. 상세 계약과
hash는 `perception_joint_detector_training_2026-09-08.md`,
`perception_p4_p5_execution_2026-09-08.md` 및
`../../results/perception_joint_detector_p4_p5_2026-09-08/README.md`가 정본이다.

P6/P7도 2026-09-08에 validation-only로 완료했다. 같은 frozen P4 후보에서 GRU T=8과 Transformer
T=16을 학습했고, 사전 고정 utility로 Transformer를 선택했다(`0.6725`; CNN-only `0.6655`). 이는
validation 7 clips의 model selection이며 test 성능 주장이 아니다. 정본은
`perception_p6_p7_execution_2026-09-08.md`와
`../../results/perception_temporal_p6_p7_2026-09-08/README.md`다.

CNN, GRU, Transformer를 직렬로 쌓지 않는다. 동일한 frozen CNN detector 위에서 `KF`, `GRU`,
`Temporal Transformer`를 서로 대체하는 association arm으로 비교한다. 모델 선택은 validation clip에서
하고, 최종 수치는 episode 또는 source-video 단위로 분리한 test set에서 한 번 보고한다.

## 센서와 정책 인터페이스

12–28 m에서는 카메라가 UAV detection, bearing, temporal motion, approximate range와 uncertainty를
담당한다. LiDAR target range를 사용하지 않는다. 12 m 미만에서는 camera와 LiDAR 또는 stereo를
결합해 range를 보정한다. LiDAR obstacle 입력과 arc-clearance filter 경로는 semantic target 추정과
독립적으로 유지한다.

PPO target-history 입력의 목표 schema는 다음과 같다.

```text
[bearing, range_hat, bearing_rate, range_rate, confidence, covariance, age]
```

단위, 좌표계, timestamp, covariance 차원, missing-state 표현을 versioned schema에 명시한다. 이 schema
변경은 기존 898-D checkpoint와 호환되지 않으므로 P10은 새 perception lineage로 기록한다.

## 평가 계약

Perception 지표는 precision, recall, FTLR, ID switch count/rate, reacquisition time, bearing error,
latency이며, target box의 pixel-size bin과 source dataset별로 분해한다. 가능하면 calibration error와
range error/uncertainty coverage도 보고한다. frame 수와 함께 독립 video/episode 수를 병기하고,
confidence interval의 반복 단위는 frame이 아니라 source video 또는 episode로 둔다.

Closed-loop 지표는 capture, crash, timeout의 count와 rate다. baseline은 같은 PPO·같은 arena에서
오차 주입이 없는 arm 또는 이전 단계의 frozen perception arm이며, 비교마다 차이(percentage points),
95% CI, training seed 수, evaluation seed 수, cell당 episode 수를 기록한다.

## Track A와 통합 경계

현재 안전 연구는 **arc-clearance speed filter의 효과와 한계**로 마무리한다. 기존 136-cell 결과를
새 perception 구현의 근거로 재해석하지 않으며, arc 구현도 이 단계에서 변경하지 않는다. 먼저 남은
training-seed R-C 반복을 끝낸다. P10이 완료된 뒤에만
`realistic perception + PPO + arc-clearance filter` 통합 실험을 새 사전등록과 새 결과 계보로 연다.

## 현실적인 작업량

| 묶음 | GPU 추정 | 사람 작업 추정 | 추정 근거·불확실성 |
|---|---:|---:|---|
| P1–P3: 정리·Det-Fly 변환·zero-shot | 1–3 h | 6–10 h | 다운로드 시간 제외; annotation 형식 확인 전 |
| P4–P7: 후보 schema와 세 association arm | 8–20 h | 24–48 h | 데이터 loader·평가 harness 재사용 정도에 좌우 |
| P8: 오류 측정·calibration | 2–6 h | 12–24 h | range GT 확보 여부가 가장 큰 변수 |
| P9: 오류 주입기와 분포 검증 | 1–3 h | 8–16 h | versioned schema와 temporal correlation 포함 |
| P10: PPO 재적응·held-out 평가 | 5–12 h | 8–16 h | 1,000-epoch 1회 실측 0.81 h를 기준으로 seed 수에 비례 |

데이터가 로컬에 있고 range GT 대안을 확정한 뒤의 총량은 **GPU 17–44시간, 사람 작업 58–114시간**이다.
한 사람이 하루 5시간 집중 작업을 수행하면 약 **12–23 근무일**이다. 데이터 다운로드, 라이선스 회신,
새 라벨 정정, 실패한 모델 재탐색 시간은 포함하지 않는다. 각 단계의 실제 시간을 receipt에 기록해 다음
단계 추정을 갱신한다.
