# Streaming 재현성·병목·P8-lite 입력 계약 (2026-09-09)

인수인계 [`perception_streaming_handoff_2026-09-08.md`](perception_streaming_handoff_2026-09-08.md)의
세 항목에 대한 답이다. **재학습·threshold 변경·test 열람·원본 cache 덮어쓰기는 하지 않았다.**

---

## 1. 재현성 — 원인 확정, frozen output 재현 성공

### 원인: 캐시를 만든 환경과 재실행 환경이 다르다

같은 프레임을 두 환경에서 같은 producer 스크립트·같은 가중치로 돌려 비교했다.

| 환경 | Python / torch / CUDA / cuDNN | 60프레임 완전일치 | confidence 최대 상대차 |
|---|---|---:|---:|
| **`detector_runs/venv`** | 3.10 / 2.10.0+cu128 / 12.8 / 9.10.2 | **60 / 60** | **0.000e+00** |
| `datasets/detenv` | 3.8 / 2.4.1+cu121 / 12.1 / 9.1.0 | 0 / 60 | **0.45** |

`venv`가 historical candidate cache를 **비트 단위로 재현**한다. 첫 프레임 첫 후보에서
`confidence 0.3737453520298004`, `u_px 628.6032104492188`가 소수점 끝자리까지 일치한다.
`detenv`는 같은 프레임에서 `0.3739687204360962` / `628.6038818359375`를 낸다.

**"5프레임 선택 불일치"는 증상이지 원인이 아니다.** 원인은 cache가 `venv`로 생성됐는데 RGB 재실행이
`detenv`로 돌았다는 것이다. 인수인계 문서의 "RGB 실행은 `datasets/detenv/bin/python`"이 그 지점이며,
`detenv`를 고른 이유(pandas 부재)는 `venv`에는 해당하지 않는다 — `venv`에 pandas 2.3.3이 있다.

TF32·benchmark·deterministic 플래그로 재현되지 않은 것도 이것으로 설명된다. 두 환경은 cuDNN 9.1과
9.10.2로 **컨볼루션 커널 자체가 다르므로**, 한 환경 안의 플래그로는 메울 수 없다. 첫 프레임의 차이가
6e-4로 작아 보였던 것은 우연이며, 60프레임 전체에서는 최대 45 %까지 벌어진다.

### 검증: 전체 2,296프레임 RGB 감사를 `venv`로 재실행

새 출력 경로 `detector_runs/results/nps_detfly_joint_final/stream_rgb_venv_v1.json`
(기존 artifact 무변경). 결과는 **strict parity PASS**다.

| 항목 | 값 |
|---|---|
| `rank_mismatches` | **0** |
| `rank_parity_pass` | **True** |
| `candidate_frame_mismatches` | **0** |
| `first_candidate_mismatch` | None |

지표도 cached replay와 전부 일치한다.

| 지표 | cached replay | RGB (detenv) | **RGB (venv)** |
|---|---:|---:|---:|
| selected_frames | 2122 | 2121 | **2122** |
| no_lock_frames | 174 | 175 | **174** |
| proxy_false_lock_frames | 274 | 273 | **274** |
| matched_center_error_px_mean | 1.941796 | 1.940528 | **1.941796** |
| loss_events_after_first_hit | 106 | 107 | **106** |

### 부수 발견 — receipt에 실행환경이 없다

`nps_val_candidates.jsonl.gz.receipt.json`은 `device_name`, `fp16`, `yolov5_revision`,
`weights_sha256`는 기록하지만 **Python·torch·CUDA·cuDNN 버전을 기록하지 않는다.** 이번 혼선의 근본
원인이 여기 있다. 재현성 receipt에 인터프리터와 가속 스택 버전을 추가할 것을 제안한다(코드 변경은
사용자 승인 대상으로 남긴다).

조사 중 처음에 NPS 단독 `best.pt`(`ccd65dc3…`)로 실행했으나 도구의 SHA 게이트가 즉시 거부했다.
cache가 쓴 detector는 `nps_detfly_joint/yolov5s_ms_b8_e30_s0`(`aab12f39…`)다. 게이트가 설계대로 작동했다.

### 권고

RGB 실행 환경을 **`detector_runs/venv`로 고정**한다. 새 baseline·cache 버전 채택은 필요 없다 —
기존 frozen output이 그대로 재현되기 때문이다.

---

## 2. 병목 — GMC가 아니라 광류다

motion 단계를 실제 validation 프레임 119개에서 분해했다(1280×960 → flow 640×480, scale 0.5).

| 단계 | mean (ms) | p95 | motion 내 비중 |
|---|---:|---:|---:|
| **Farnebäck flow** | **36.58** | 37.40 | **97.7 %** |
| candidate features | 0.48 | 0.36 | 1.3 % |
| mask + grid 샘플링 | 0.22 | 0.24 | 0.6 % |
| RANSAC GMC | 0.17 | 0.19 | 0.4 % |

**GMC는 병목이 아니다.** RANSAC은 0.17 ms로 motion의 0.4 %다. 최적화 대상은 광류 하나다.

OpenCV 스레드 수는 지렛대가 아니다 — 1개 36.9 ms, 24개 36.0 ms이고 출력은 스레드 수와 무관하게
비트 동일이다. 이 빌드에서 Farnebäck은 사실상 단일 스레드로 동작한다.

### 제안 A — 검출기와 광류를 겹친다 (출력 보존, 실측 완료)

`estimate_backward_flow_and_gmc`에서 **flow는 `previous_gray`와 `gray`만 필요하고 candidates에
의존하지 않는다.** candidates는 그 뒤 `candidate_mask`(0.22 ms)부터 쓰인다. 따라서 CPU 광류를
GPU 검출기와 동시에 돌릴 수 있다.

실측 프로토타입(GPU 워크로드 + 광류, 8회):

| | 시간 |
|---|---:|
| 순차 | 112.3 ms |
| **겹침** | **74.7 ms** |
| 절감 | 37.6 ms (33 %) |
| **flow 출력 비트 동일** | **True** |

광류 비용이 통째로 숨는다. OpenCV가 GIL을 놓으므로 Python 스레드로 충분하다.

**예상 효과**(현재 pipeline 실측 detector 26.37 / motion 33.40 / selector 2.35 / decode 5.15 ms 기준):

| | 현재 | 겹침 적용 |
|---|---:|---:|
| frame 처리 | 62.13 ms | ≈ 35.8 ms |
| decode 포함 | 67.28 ms | ≈ 40.9 ms |
| **FPS** | **14.86** | **≈ 24** |

출력은 같은 함수·같은 입력·각 함수 내 같은 연산 순서이므로 **비트 동일**이다. 구현은
`estimate_backward_flow_and_gmc`를 `compute_flow(prev, cur)`와 `estimate_gmc(flow, candidates, …)`로
쪼개고 앞을 worker에 올리는 것이다. **구현 전 사용자 승인 대상**이며, 적용 시 전후 latency
mean/P50/P95와 ranks·hit·false-lock·no-lock·reacquisition을 함께 보고한다.

### 별도 계약이 필요한 것 (지금 하지 않음)

flow 해상도·빈도 변경, FP16, 다른 광류 알고리즘은 모델 입력 분포를 바꾼다. 인수인계 지침대로
별도 실험 계약으로 분리한다. 참고로 flow 해상도를 320으로 낮추면 이론상 4배지만 motion feature가
바뀌므로 selector 재평가 없이는 채택할 수 없다.

---

## 3. P8-lite / P9 입력 계약 초안 — 실기 정보 없이 만들 수 있는 것만

**PPO 연결은 하지 않았다.** 아래는 schema와 error model 설계이며 별도 승인 전에는 연결하지 않는다.

### 측정된 것 (2,296 validation frames, `stream_rgb_venv_v1.json`)

| 양 | 값 |
|---|---|
| pixel-center error | mean 1.942 px, median 1.034 px |
| matched IoU | mean 0.657 |
| frame hit rate | 0.8049 (1,848 / 2,296) |
| no-lock | 174 frames (7.58 %) |
| proxy false-lock | 274 / 2,122 selected (12.91 %) |
| loss events (첫 hit 이후) | 106 |
| reacquisition | 102 회복 / 4 우측중도절단, mean 0.470 s, max 19.933 s |
| latency (decode 포함) | mean 67.28 ms, p50 67.18, p95 70.32 |

### 만들지 않는 것

- **degree 단위 bearing** — camera intrinsics(fx, fy, cx, cy, distortion)가 없다. 픽셀 오차만 보고한다.
- **metric range** — 거리 GT가 없다. box 크기→거리 환산은 기체 실치수와 보정이 있어야 하고, 있어도
  approximate range로만 쓴다.
- **실제 ID-switch** — NPS는 한 영상에 UAV가 하나뿐이라 다중 후보 식별 전환을 측정할 수 없다.
  현재 보고하는 것은 **track loss / reacquisition**이며 identity tracking이 아니다.

### 제안하는 error model (sequence-aware, pixel 단위)

정책이 필요로 하는 것은 프레임별 평균이 아니라 **시간 상관을 가진 열화**다. 다음 네 성분으로 둔다.

1. **center offset** — matched 프레임에서 픽셀 오차. mean 1.94 / median 1.03 px는 heavy tail을 뜻하므로
   정규분포가 아니라 **측정 분위수에서 직접 샘플링**한다.
2. **miss burst** — no-lock은 균등 분포가 아니라 뭉친다. loss event 106회와 reacquisition 시간
   (mean 0.470 s, max 19.933 s)에서 **2-state Markov(lock ↔ no-lock)**의 전이확률을 적합한다.
3. **false-lock burst** — 12.91 %가 잘못된 대상에 걸린다. 이것이 정책에 가장 위험한 성분이며,
   단발이 아니라 지속되는지를 시퀀스에서 측정해 같은 방식으로 모델링한다.
4. **latency** — p50 67 ms, p95 70 ms. 관측이 한 프레임 늦게 도착하는 효과로 주입한다.

### downstream 입력 schema (P8-lite)

기존 [`perception_candidate_interface_v1`](../specs/perception_candidate_interface_v1.md)의 Top-K
후보 계약을 그대로 쓰되, 정책에 넘기는 것은 **선택된 1개 + 상태 플래그**로 한다.

| 필드 | 형 | 의미 |
|---|---|---|
| `u_px`, `v_px` | float | 선택 후보 중심. **픽셀 단위**, degree 아님 |
| `width_px`, `height_px` | float | 선택 후보 크기 |
| `confidence` | float | detector confidence |
| `lock_state` | enum | `LOCK` / `NO_LOCK` / `STALE` |
| `age_frames` | int | 마지막 갱신 이후 경과 프레임 |
| `latency_ms` | float | capture → 결과 |

### stale / no-lock / reset 처리 (P9 연결 전 명시 필요)

- **NO_LOCK**: 관측 없음. 정책 입력은 마지막 값을 유지하지 않고 **명시적 무효 플래그**로 넘긴다.
  값을 0으로 채우면 "정면 근거리"와 구분되지 않는다.
- **STALE**: `age_frames > N`이면 값은 있으나 신뢰할 수 없다. N은 reacquisition 분포에서 정하되
  **사전등록**한다.
- **reset**: sequence 전환 시 selector 내부 상태를 비운다. 현재 `StreamingSelector.reset()`이 그
  역할을 하며, PPO 에피소드 경계와 어떻게 맞물릴지는 P9 설계에서 정한다.

**이 세 처리는 PPO 연결 전에 별도 승인을 받는다.**

---

## 주장하지 않는 것

30 FPS, 실기 탑재, identity tracking, ROS/PPO 연결 완료 — 어느 것도 현재 결과로 주장하지 않는다.
겹침 제안의 24 FPS는 **실측 프로토타입에 기반한 추정**이며 적용·재측정 전에는 확정값이 아니다.
latency는 workstation 직렬 batch-one 측정이고 camera transport를 포함하지 않는다.
