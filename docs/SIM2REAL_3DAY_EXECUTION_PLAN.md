# MOTAR sim-to-real 72시간 실행·계측 계약

> ## ⏸ 보류 (2026-09-05) — 실행하지 않는다
>
> 이 문서는 2026-08-26 기준 **72시간 실행 계약**이고 그 창은 이미 지났다. 더 중요하게,
> **지금 sim-to-real을 할 수 있는 상황이 아니다** — 실제 기체가 미조립이고 센서 로그도
> 비행 데이터도 없다. 여기 적힌 일정·게이트는 실기가 준비된 뒤에 **날짜와 전제를 다시 세워**
> 재사전등록해야 한다.
>
> 현재 연구 전선은 `docs/plans/diagnostic_paper_plan_2026-09-05.md`(진단 논문, 재학습 없음)이며
> 판정 authority는 `VERIFICATION.md`다. 이 문서를 "다음 권한"으로 인용하지 말 것.


> 기준 시각: **2026-08-26 (Asia/Seoul)**
> 상태: **실기 비행 0회 · 센서/BOM 실측 계약 미완료 · Track A/B GPU 작업 미승인**
> 이 문서가 앞으로 3일의 **단일 실행 목록**이다. 장기 연구 질문은
> [`RESEARCH_PLAN.md`](../RESEARCH_PLAN.md), gate 판정은 [`VERIFICATION.md`](../VERIFICATION.md),
> 날짜별 사실 기록은 [`WORKLOG.md`](../WORKLOG.md)를 따른다.

## Authority split — Track A / Track B

- **Track A (이 문서의 실행 범위):** P2 `STRICT FAIL`, D1 `FAIL`, P3 `BLOCKED`; detection Stage 1
  `RANGE_INCONCLUSIVE_AT_THIS_BUDGET`, Stage 2 미승인이다. 유일한 다음 authority는 exact hardware
  BOM/calibration, 아래 210개 독립 sensor trial, real-log profile과 offline replay다.
- **Track B (종료):** recovery-v2 lower-1.25는
  `PASS_32_CELL_INTEGRITY / FAIL_ROUTE_MECHANISM`이다(7/32, 모두 off; recovery 0/16; 70-bar
  plan `93.60%`, fallback `47.87%`, 70×0.6 goals/env `0.21875`, `NO_CONNECTOR` occupancy
  `63.06%`). 후속 no-anchor probe는 primary `n=1`, identity disagreement `0`으로
  `INCONCLUSIVE`다. 추가 GPU/PPO/retune/1.5/env-count/32-cell rerun authority는 없다.

실제 hardware가 없고 real log도 없으면 이 계획에서 실행할 GPU 작업은 **없다**. 합성
software-only preflight를 반복하거나 Track B를 다시 여는 것으로 실측 조건을 대체하지 않는다.

## 0. 출발점과 72시간의 목표

검출 거리 Stage 1은 고해상도 검출 조건에서 `20 m`와 `28 m` clip을 각각 1,000 epoch 적응시킨 뒤
평가했다. 두 arm 모두 2,049 episode이고 provenance/quality gate **17/17 PASS**다.

| arm | pooled never-acquired | capture | crash | timeout |
|---|---:|---:|---:|---:|
| clip 20 m | 8.443% | 82.235% | 15.666% | 2.099% |
| clip 28 m | 3.172% | 88.677% | 11.274% | 0.049% |
| Δ (28−20) | **−5.271 pp** | +6.442 pp | −4.392 pp | −2.050 pp |

사전등록 primary는 never-acquired `≤ −15 pp`였으므로 공식 판정은
`RANGE_INCONCLUSIVE_AT_THIS_BUDGET`, Stage 2 권한은 **없다**. capture 개선은 유용한 부수 관측이지만
판정 지표가 아니며, 두 arm 모두 실기에서 아직 얻을 수 없는 **정확한 analytic range**를 쓴다.

72시간의 목표는 성공률을 한 번 더 올리는 것이 아니다.

1. 실제로 제작·탑재할 기체와 센서의 **측정 가능한 계약**을 닫는다.
2. `12–28 m`에서 camera가 줄 수 있는 bearing과 LiDAR/stereo가 줄 수 있는 range를 분리한다.
3. 실제 측정 분포로 다음 학습의 noise/latency/dropout을 사전 고정한다.
4. 데이터가 부족하면 학습을 시작하지 않는 **go/no-go gate**를 만든다.

## 1. 72시간 동안 바꾸지 않는 것

- P2 `STRICT FAIL`, D1 `FAIL`, P3 `BLOCKED` 판정은 유지한다.
- 검출 거리 Stage 2를 실행하지 않는다. 결과를 본 뒤 `−15 pp` 임계를 완화하지 않는다.
- reward, horizon, 기체, observation schema, governor를 한 run에서 함께 바꾸지 않는다.
- `28 m`를 “실기 검출 거리”나 “채택된 센서 사양”으로 부르지 않는다.
- 실제 sensor log 없이 iid Gaussian noise를 임의로 정해 장기학습하지 않는다.
- frame 수를 독립 표본처럼 세지 않는다. 신뢰구간은 **독립 trial 단위**로 계산한다.

## 2. Day 1 — 하드웨어 계약 동결과 원자료 취득

### 오전: exact BOM·좌표계·시간 계약

사용자가 먼저 채울 값이다. `미정`을 숫자로 추정해 채우지 않는다.

| 묶음 | 반드시 기록할 값 | 완료 기준 |
|---|---|---|
| 기체 | frame/prop/motor/ESC/battery 모델, 실측 AUW, CG, 축별 관성 또는 식별 계획 | 저울 사진/측정 파일과 단위 포함 |
| 추력 | 배터리 전압, prop, motor, ESC에서의 모터별 최대추력·시상수·deadband | thrust-stand 원자료 또는 `미측정` 표시 |
| 카메라 | 정확한 모델/serial/firmware, RGB/depth 해상도·FPS·HFOV, exposure/gain, stereo baseline, intrinsics | calibration YAML과 SHA-256 |
| LiDAR | 모델/serial/firmware, range/FOV/scan rate, return mode, min/max range | 설정 export와 SHA-256 |
| 상태추정 | FCU/IMU/VIO/LIO 모델·rate·frame convention | ENU/NED, body 축, quaternion 순서 명시 |
| 외부파라미터 | body↔camera↔LiDAR↔IMU의 translation/rotation | 변환 방향과 단위가 있는 extrinsics YAML |
| 시간 | monotonic clock 기준, HW/PTP/ROS sync 방식, topic별 timestamp 의미 | clock skew 분포를 계산할 수 있어야 함 |
| 연산 | compute/OS/CUDA/TensorRT/driver, power mode, thermal limit | 버전 dump와 전력모드 기록 |

기존 [`navrl_hardware_identification_manifest.yaml`](navrl_hardware_identification_manifest.yaml)은
식별값 입력 양식이다. `navrl_ref5in_quad`의 1.20 kg은 설계점일 뿐 실제 AUW가 아니다.

실기 미조립과 frame 미보유를 확인한 뒤, 현재 220 mm/5-inch simulation 형상을 가장 적게 바꾸는
**우선 설계 기준**으로 iFlight AOS 5 V5.1(공식 228 mm, 165±5 g)을 선정했다. 이는 구매 확정이
아니며 [`navrl_frame_selection_2026-08-23.md`](navrl_frame_selection_2026-08-23.md)의 payload CAD,
질량·CG, prop/sensor clearance, thrust-curve gate를 모두 통과해야 exact BOM으로 승격한다. gate 전에는
URDF 질량/충돌 형상이나 frozen checkpoint contract를 바꾸지 않는다.

### 오후: 센서 거리·가림 원자료

권장 최소 matrix:

- 거리: **5, 8, 12, 16, 20, 24, 28 m**.
- 조명: **일반 / 저조도 / 역광**.
- 표적 운동: **정지 / 횡이동**. 횡속도는 명령값이 아니라 ground-truth 실측값을 기록한다.
- 반복: cell마다 **독립 5회**, trial당 **10 s 이상**.
- 총 7×3×2×5 = **210 trials**, 순수 녹화 약 35분. 설치·거리 확인 포함 3–5시간을 잡는다.

각 trial은 다음을 같은 clock으로 보존한다.

- 원본 RGB/depth, camera info, LiDAR point/range, IMU, odometry/VIO.
- detector box/mask/confidence, tracker output이 있으면 별도 topic.
- sensor stamp, host receive stamp, policy input stamp, command publish stamp.
- 해상도, FPS, exposure/gain, CPU/GPU load·temperature·power.
- 표적 실제 거리/방위 ground truth. 레이저 거리계·측량·mocap·검증된 tag 중 방법을 명시한다.

### Day 1 품질 gate

하나라도 실패하면 Day 2에서 성능 숫자를 만들지 않고 취득을 보완한다.

- trial manifest의 파일 수와 SHA-256이 원자료와 일치한다.
- 모든 trial에 calibration ID, 조건, 반복 ID, 시작/종료 시각이 있다.
- timestamp 역행 0건; drop과 clock skew를 계산할 수 있다.
- 거리 ground truth가 없으면 **range bias/RMSE 주장을 금지**한다.
- 동일 영상을 여러 조각으로 잘라 독립 반복 수를 부풀리지 않는다.
- 실험 중 바뀐 exposure/FPS/firmware가 있으면 별도 condition으로 분리한다.

## 3. Day 2 — 실제 sensor profile과 two-zone 경계 산출

### 오전: trial 단위 통계

거리·조명·운동 cell마다 아래 값을 산출한다.

| 계층 | 기록할 값 |
|---|---|
| 검출 | precision/recall/F1, confidence p10/p50/p90, false positive/min |
| bearing | azimuth/elevation bias, MAE, p50/p90/p95 error (deg) |
| range | valid fraction, bias, RMSE, absolute error p50/p90/p95 (m) |
| temporal | first-acquisition, dropout burst p50/p95/max, reacquisition p50/p95 |
| 시간 | sensor→host, host→detector, detector→policy, total latency p50/p95/p99 |
| 처리량 | achieved FPS, missed deadline, compute utilization, power, temperature/throttle |
| 동기 | camera–LiDAR–ego-state skew p50/p95/p99, timestamp offset/drift |

- 신뢰구간 bootstrap resampling 단위는 frame이 아니라 **trial**이다.
- 평균만 보고하지 않고 p90/p95와 최악 trial을 함께 남긴다.
- train/tuning/eval split은 trial ID로 분리하고, 같은 연속 녹화의 frame이 양쪽에 섞이지 않게 한다.

### 오후: two-zone observation 계약

현재 시뮬레이터의 `28 m exact relative position`을 그대로 실기에 옮기지 않는다. 측정 결과로 경계를
정하되, 초기 후보는 다음과 같다.

```text
far zone   (LiDAR/stereo range 불신뢰): camera bearing + confidence + age + covariance
near zone  (range 품질 gate 통과):      camera bearing + LiDAR/stereo range fusion
tracker: ego-motion 보상 + timestamp-aligned state + uncertainty propagation
actor:   동일 target token에 range_valid/range_sigma를 명시
```

near/far 경계는 LiDAR 명목 12 m를 자동 채택하지 않는다. 거리 bin에서 다음 중 먼저 실패하는 지점을
경계 후보로 둔다.

- range valid fraction `< 0.80`,
- p90 range error가 사전 정의한 추적/제동 허용오차 초과,
- camera–LiDAR time skew 또는 end-to-end p95 latency가 control budget 초과.

target token 최소 필드 후보:

`azimuth, elevation, bearing_rate, confidence, range, range_valid, range_sigma, measurement_age,
track_covariance, visibility_state`.

필드를 바꾸면 observation schema가 달라지므로 기존 checkpoint 성능과 직접 비교하지 않는다.

### Day 2 산출물·gate

- `hardware_manifest`: 실제 BOM/serial/firmware/calibration SHA가 닫힘.
- `dataset_manifest`: trial 조건·split·파일 SHA가 닫힘.
- `sensor_profile`: 위 통계와 trial-level CI가 있음.
- `two_zone_contract`: 경계 선택 근거와 token field/scale/mask가 있음.
- latency p95와 clock skew p95가 없으면 PPO/실기 flight **NO-GO**.
- range ground truth 또는 extrinsics가 없으면 range fusion 학습 **NO-GO**.

## 4. Day 3 — replay 검증과 다음 학습 사전등록

### 오전: offline tracker replay

새 PPO를 돌리기 전에 held-out real log에서 다음을 검증한다.

- ego-motion 보상 전/후 bearing error와 reacquisition 비교.
- tracker range/bearing innovation과 covariance calibration.
- occlusion/dropout burst 중 predicted state가 발산하는 시간.
- late/out-of-order measurement가 들어와도 timestamp 기준으로 처리되는지.
- far-zone에서 `range_valid=0`인데 0 m로 해석되지 않는지.
- near↔far 전환 때 token이 불연속적으로 튀지 않는지.

### 오후: simulator replay/smoke와 학습 권한 판단

1. 실제 sensor profile을 simulator perturbation 파일로 **고정**한다.
2. schema가 그대로면 frozen-policy 평가 전용 진단만 한다.
3. schema가 바뀌면 unit/integration smoke까지만 하고 frozen checkpoint의 성능 비교는 하지 않는다.
4. 다음 fresh training은 별도 preregistration에 아래 계약이 모두 적힌 뒤에만 시작한다.

#### Day 3 GO 조건

- BOM/calibration/dataset/source manifest가 모두 hash로 묶였다.
- held-out real replay에서 bearing·range·latency·dropout 통계가 재산출된다.
- observation field 순서, scale, validity mask, history timing이 문서와 테스트에서 일치한다.
- 다음 학습의 한 개 조작축, seeds, 예산, primary metric, threshold, guard가 결과 전에 동결됐다.

#### NO-GO 조건

- 실제 질량/추력/CG 또는 sensor timestamp 의미가 미정이다.
- exact range를 far zone에도 계속 제공한다.
- detector/tracker와 policy가 다른 시각의 ego-state를 결합한다.
- reward/horizon/sensor/airframe/schema를 동시에 바꾸려 한다.
- Stage 1 음성 판정을 이유로 더 긴 Stage 2부터 돌리려 한다.

## 5. 다음 학습 전에 반드시 남길 숫자

### 실행 전 — manifest에 고정

| 범주 | 필수 항목 |
|---|---|
| 코드 | git commit, dirty 여부, source manifest/launcher/prereg SHA-256 |
| 기체 | URDF/config SHA, AUW, CG, inertia, motor/prop/ESC/battery, thrust curve, motor time constant |
| 센서 | serial/firmware, resolution/FPS/FOV/baseline, intrinsics/extrinsics SHA, exposure, scan rate |
| 오차 | bearing/range bin별 bias·p90, latency p50/p95/p99, skew p95, dropout burst p95 |
| 관측 | 모든 field의 순서·단위·scale·clip·valid mask, history 길이/샘플 간격 |
| 환경 | arena/bar geometry, bars, spawn/goal/target-speed 분포, target dynamics, physics/control dt |
| 정책 | action 의미/한계, reward 모든 계수, gamma/GAE, LR, clip, entropy, action std, optimizer |
| 학습 | seed, env 수, horizon, minibatch, epoch/frame budget, checkpoint cadence, curriculum/dwell |
| 판정 | primary/secondary metric, threshold/CI 방법, failure guard, 중단 조건, 허용된 다음 단계 |
| 장비 | GPU/driver/CUDA, power mode, simulator FPS 목표, VRAM/temperature limit |

### 실행 중 — epoch 또는 고정 evidence window마다

- epoch, environment frame, task step, wall-clock, simulator FPS.
- capture/crash/timeout **count와 rate**, crash cause, never-acquired, first-acquisition p50/p90.
- target visible/hidden fraction, detector recall/confidence, track age/dropout burst.
- training-only GT 기준 bearing/range error와 covariance calibration.
- requested/actual/executed speed, action edge/OOB, action delta, heading-rate/curvature proxy.
- directional clearance, stopping distance/margin, contact 직전 1초의 속도·margin.
- PPO mean/max KL, LR, entropy/std, actor/critic loss finite, rollback total/streak.
- GPU utilization/VRAM/temperature, CPU load, missed control deadlines.

### 실행 후 — 결과를 말하기 전

- `best`가 아니라 **terminal/last checkpoint** SHA와 정상 종료 이유.
- held-out seed, 요청/실제 episode 수, count 합계, Wilson/paired CI.
- 거리·속도·가림·조명·range-valid strata별 결과.
- result↔receipt↔checkpoint snapshot↔source manifest hash 연결.
- training과 evaluation의 robot/sensor/observation/reward 계약 byte 비교.
- 사후 threshold/parameter tuning이 없었다는 기록과 VOID 목록.
- 실기/HIL이면 clock skew, deadline miss, power/thermal, pilot intervention 원자료.

## 6. 72시간 뒤 의사결정

| 결과 | 다음 행동 |
|---|---|
| hardware+sensor 계약 PASS | two-zone tracker/schema를 한 축으로 구현하고 fresh PPO 사전등록 |
| 센서 profile만 PASS, 기체 계약 미완료 | offline tracker/HIL만 진행; 비행·fresh dynamics training 보류 |
| far bearing은 유효, far range는 불신뢰 | bearing-only search + near range fusion으로 진행 |
| 20 m 이후 bearing recall도 낮음 | detector/optics/data 문제부터 해결; policy/horizon으로 덮지 않음 |
| latency/skew가 control budget 초과 | timestamp-aligned tracker와 compute pipeline부터 수정 |
| 원자료 품질 gate 실패 | 재취득; simulator noise 값을 추정으로 만들지 않음 |

72시간 종료 보고에는 “성능이 몇 % 올랐는가”보다 **어떤 실기 숫자가 측정됐고, 무엇이 아직
가정이며, 그 결과 다음 학습의 조작축이 정확히 하나로 닫혔는가**를 먼저 쓴다.

## 7. 완료된 software-only preflight (현행 실행 authority 아님)

부품·rosbag이 없을 때 아래 합성 구조 검사는 이미 완료돼 `SYNTHETIC_ONLY`를 기록했다. 이 명령은
재실행 권한이나 실측 증거가 아니다. 실제 로그가 들어오면 §8의 같은 fail-closed 판정 경로를
사용하며, 그 전에는 noise를 추정하거나 GPU 작업을 시작하지 않는다.

```bash
# 합성 입력은 구조 테스트일 뿐이며 실기 증거가 아니다.
/home/fair/miniconda3/envs/aerialgym/bin/python tools/navrl_sim2real_telemetry.py \
  fixture --output /tmp/navrl_telemetry_fixture.jsonl --groups 5
/home/fair/miniconda3/envs/aerialgym/bin/python tools/navrl_sim2real_telemetry.py \
  validate /tmp/navrl_telemetry_fixture.jsonl \
  --report /tmp/navrl_telemetry_report.json \
  --max-sync-skew-ms 20 --max-sensor-to-host-latency-ms 500
```

JSONL event에는 topic/sequence/source timestamp/host receive timestamp/frame edge를 기록하고,
camera·LiDAR·ego-state에만 `sync_group`을 부여한다. 검증기는 다음을 fail-closed로 처리한다.

- manifest/schema 누락, unknown topic, frame edge 불일치;
- source/host timestamp 역행, sequence 중복·gap, 음수 latency;
- sensor source→host latency와 synchronized sensor skew gate 초과;
- policy-input→command latency와 topic별 p50/p95/p99/max 통계.

보고서의 `claim_status=SYNTHETIC_ONLY`는 합성 fixture를 실측으로 승격하지 못하게 하는 표식이다.
실제 rosbag/CSV를 변환한 JSONL에는 `source_kind=real_log`와 run/calibration manifest를 넣고,
그때만 `MEASURED_CANDIDATE`로 바뀐다. 이 단계는 policy/reward/observation/checkpoint를
변경하지 않으며, 5개 CPU 단위 테스트가 시간·frame·manifest 오류를 재현한다.

## 8. 로그가 들어왔을 때의 단일 실행 경로

실제 장비가 아직 없으므로 아래 명령은 준비만 해 두며, 저장소의 simulation CSV를 실기 로그로
대체해 실행하지 않는다. CSV의 frame/time/frame-edge를 사람이 보정해 주지 않고 원 recorder의 값을
그대로 넣어야 한다.

### 8.1 transport telemetry 변환 → 원본 계약 검증

```bash
PY=/home/fair/miniconda3/envs/aerialgym/bin/python
$PY tools/navrl_sim2real_ingest.py \
  /path/to/real_events.csv /path/to/real_events.jsonl \
  --run-id trial_batch_001 --source-kind real_log \
  --metadata-json /path/to/run_manifest.json
$PY tools/navrl_sim2real_telemetry.py validate \
  /path/to/real_events.jsonl \
  --report /path/to/telemetry_report.json \
  --max-sync-skew-ms 20 --max-sensor-to-host-latency-ms 50
```

`navrl_sim2real_ingest.py`는 `topic`, `seq`, source/host timestamp, frame edge를 필수로 요구하고
원본 SHA-256을 manifest에 기록한다. 필수 열 누락·숫자 파싱 실패·빈 입력은 종료 코드 2로
중단한다. unknown frame을 추정하지 않으며, `source_kind=real_log`를 명시하지 않으면 실측
후보로 표시하지 않는다.

### 8.2 trial 단위 sensor profile

ground-truth와 detector join이 끝난 별도 measurement CSV에 대해 다음을 실행한다.

```bash
$PY tools/navrl_sensor_profile.py \
  /path/to/heldout_measurements.csv /path/to/sensor_profile.json \
  --run-id heldout_001 --source-kind real_log
```

필수 열은 `trial_id`, 거리/조명/운동 조건, target-present/detected/range-valid, GT·추정
bearing/range, confidence, source/host timestamp다. 출력은 frame 수와 함께 trial 수, cell별
trial-macro recall/range-valid fraction/latency를 남긴다. frame을 독립 반복으로 세지 않으며,
threshold를 자동 적용하거나 simulator noise를 선택하지 않는다.

### 8.3 two-zone replay 계약

sensor profile 검토 후 사람이 근거와 함께 만든 JSON contract만 입력한다. 경계는 이 도구가
결정하지 않는다.

```json
{
  "schema_version": 1,
  "source_kind": "real_log",
  "near_boundary_m": 12.0,
  "far_range_policy": "invalid"
}
```

```bash
$PY tools/navrl_two_zone_replay.py \
  /path/to/heldout_target_tokens.jsonl /path/to/two_zone_contract.json \
  --report /path/to/two_zone_replay_report.json
```

far zone에서 `range_valid=true`이거나 invalid range를 0 m로 채우면 fail-closed한다. near zone도
dropout 자체는 허용하지만 그때 `range_valid=false`, `range_m=null`, `range_sigma_m=null`이어야
한다. PASS는 관측 계약의 구조·시간 순서가 맞다는 뜻일 뿐, 정책 성능이나 sim-to-real 성공을
주장하지 않는다.
