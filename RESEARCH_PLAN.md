# RESEARCH PLAN — 카메라·LiDAR + 시뮬레이터 ego-state UAV 요격: 밀도 × 표적속도

이 문서가 연구 charter다(가설·설계·계획). **현재 상태와 최신 수치는 여기 적지 않는다** —
진행 기록은 `WORKLOG.md`(맨 아래가 최신), 검증 gate는 `VERIFICATION.md`, 라이브 지표는 `docs/status/`를 본다.
실무(머신 셋업·GPU·이관)는 `OPERATIONS.md`, 과거 진단 도구와 측정된 음성 결과는 `CRASH_TUNING_LOG.md`(archival, 2026-08-05 정지).
현재의 하드웨어·학습 파라미터·low/high-level 자료구조는 중복 표기를 피하기 위해
[`docs/MOTAR_SYSTEM_SPEC_2026-08-24.md`](docs/MOTAR_SYSTEM_SPEC_2026-08-24.md)에 고정한다.

> **2026-09-01 authority pointer:** Track A/Track B/corrected non-overlap의 실행 결과와 허용된 다음 단계는
> [`VERIFICATION.md`](VERIFICATION.md)가 규정한다. Track A의 유일한 다음 계약은
> [`docs/SIM2REAL_3DAY_EXECUTION_PLAN.md`](docs/SIM2REAL_3DAY_EXECUTION_PLAN.md)의
> hardware/real-log 계측이며, Track B는 recovery-v2 FAIL과 no-anchor `INCONCLUSIVE` 뒤 종료됐다.
> corrected non-overlap routed 계보는 `FAIL_ROUTE_MECHANISM`이고 routed PPO/장기학습 권한이 없다.
> 다만 2026-09-01 사용자 우선순위에 따라 route-off physical PASS envelope만 사용하는 별도
> 70-bar/500-epoch learning-viability baseline 1회를 사전등록했다. 이는 overlap 수정의 fresh 학습
> 가능성을 먼저 분리하는 진단이며 routed method 성공으로 합치지 않는다. 이 charter의 후보 실험
> 설명은 그 자체로 추가 실행 권한을 만들지 않는다.

### 2026-09-05 A7 방법 — 속도 상한 법칙 × clearance 기하

`riskcap`→`dwa_arc`는 법칙과 기하를 함께 바꾼다. 기하의 독립 기여는
`riskcap`/`riskcap_arc`와 `stopcap`/`dwa_arc`의 2×2로 분리하고, 두 번째 시드에서 재현을
검사한다. [A7 사전등록](docs/prereg_2026-09-05_a7_arc_attribution.md)을 따른다.
205-bar 비교는 체크포인트·기체·초기 목표 거리도 달라 조건 전이 검사로 해석하며,
밀도 하나의 인과효과를 주장하지 않는다. 구현 감사에서 발견한 yaw=0 원호 극한 결함과
그 수정에 필요한 P1 동일 커밋 재측정 범위는 `VERIFICATION.md`에서 결정한다.

### 2026-08-25 역사적 후보 계보 — physical target global route

기존 계보의 표적 동작은 보존한다. 별도 후보
`physx_ref5in_6dof_global_astar_aabb_v1`만 actual bar AABB와 전 자세 target support를
inflation한 deterministic A* 경로를 physical 6-DoF 표적의 velocity reference로 사용한다.
이는 pursuer planner나 actor 입력이 아니라 **시뮬레이션 표적 동역학 생성기**다. 300 bars에서
configuration space가 분절되므로 목적지는 현재 connected component 안에서만 고르며,
arena-wide roaming은 주장하지 않는다. 당시 CPU engineering gate는 통과했지만 PhysX smoke 전까지
PPO 학습 권한은 없었다. 이후 실행 결과와 현행 차단은 `VERIFICATION.md`를 따른다. 고정 계약과 gate는
[`docs/preregistration_physical_target_global_route_2026-08-25.md`](docs/preregistration_physical_target_global_route_2026-08-25.md)를 따른다.

### 2026-08-31 corrected non-overlap authority addendum

중첩 없는 `footprint_clearance` 환경은 historical checkpoint와 분포가 달라 fresh PPO가 필요하다.
그러나 이를 곧바로 학습 권한으로 해석하지 않는다. seed 829의 독립 route/physical gate r2가 32셀
무결성은 통과했지만 route mechanism을 통과하지 못했으므로, 500-epoch smoke와 70→205 본학습은
0 epoch 상태로 차단한다. 다음 software 후보는 target route/controller가 braking-aware safe-state를
보존하도록 설계한 뒤, 결과를 보기 전 새 gate를 사전등록하는 것이다. 수치·판정·claim boundary는
`docs/corrected_nonoverlap_route_gate_r2_result_2026-08-31.md`와 `VERIFICATION.md`를 따른다.

### 2026-09-01 braking-aware route v3 method addendum

방법 후보를 `global_astar_braking_v3`로 고정한다. v1(`global_astar_v1`)과 recovery-v2는 alias하거나
변이하지 않는다. 개입은 공유 exact closed-AABB soft envelope, full-horizon 수락(longest-safe-prefix
실행 금지), canonical 1.5 m/s monotone p95 ceiling lookup의 terminal zero-command 정지 증명,
tangent/cross-track 추종과 stop-turn-go, 그리고 reset/goal-completion/runtime-replan을 분리한
계측으로 한정한다. reward·observation·termination·PPO·물리 동역학 파라미터는 바꾸지 않는다.
70/115/160/205가 현재 계보이고 300 bars는 disconnected stress다. 이 절은 GPU나 PPO 권한을
만들지 않는다. 사전등록은
[`docs/preregistration_braking_aware_route_v3_2026-09-01.md`](docs/preregistration_braking_aware_route_v3_2026-09-01.md)다.

### 2026-09-01 lower-contract v3 method addendum

canonical 1.5 m/s warmup 게이트가 동일 숫자로 두 번 NO-GO이므로, 같은 P-only 컨트롤러와 같은
0.05/5 s 계약을 유지한 채 인증 가능 상한만 1.25 m/s로 내리는 별도 방법을 추가한다. 개입 바이트
(`global_astar_braking_v3`)는 그대로이고, 선택되는 것은
`NAVRL_TARGET_BRAKING_CONTRACT_VARIANT=baseline_1p25`와 그 variant의 raw braking receipt뿐이다.
0.05 완화나 PID/게인 변경은 이 방법의 일부가 아니다. 이 절은 GPU나 PPO 권한을 만들지 않는다.
사전등록은
[`docs/preregistration_braking_aware_route_v3_lower1p25_2026-09-01.md`](docs/preregistration_braking_aware_route_v3_lower1p25_2026-09-01.md)다.

### 2026-09-01 matched-spawn amendment

lower-1.25 8-cell은 표적 초기 자세 matched-arm drift로 VOID됐다. 다음 방법 후보는 v3가
spawn/waypoint를 `off`와 같은 상자에서 뽑고, shared envelope는 plan/watchdog에만 쓰는
것이다. pose 해시 매칭을 VOID 이후에 끄는 것은 방법이 아니다. 이 절은 GPU나 PPO 권한을
만들지 않는다. 사전등록은
[`docs/preregistration_braking_aware_route_v3_lower1p25_matched_spawn_2026-09-01.md`](docs/preregistration_braking_aware_route_v3_lower1p25_matched_spawn_2026-09-01.md)다.

CPU forensic 뒤 고정된 execution addendum는 방법을 바꾸지 않고 새 braking receipt 1회와
seed-829 8-cell pilot 1회만 허용한다. 이 예외는 confirmatory·PPO·threshold/gain retune 권한을
만들지 않는다. 문서는
[`docs/preregistration_braking_aware_route_v3_lower1p25_matched_spawn_gpu_authority_2026-09-01.md`](docs/preregistration_braking_aware_route_v3_lower1p25_matched_spawn_gpu_authority_2026-09-01.md)다.
이 one-shot 실행은 `PASS_8_CELL_INTEGRITY / FAIL_BLOCKS_CONFIRMATORY`로 종료돼 소비됐다.
따라서 이 v3 방법은 닫혔고 새 controller 방법은 별도 사전등록이 필요하다.

## 1. 연구 질문과 기여

## 1. 연구 질문

| RQ | 질문 |
|---|---|
| RQ1 | 장애물 밀도와 표적 속도가 원시 센서 기반 표적 탐지 및 접근 성능에 어떻게 상호작용하는가? |
| RQ2 | 가림 지속시간이 표적 위치추정과 재탐지, 최종 capture에 어떤 영향을 주는가? |
| RQ3 | NavRL++식 temporal Transformer와 perception-failure fine-tuning이 표적 가림과 제어 진동을 얼마나 줄이는가? |
| RQ4 | 위치 불확실도를 인지하는 정책이 가림 중 충돌과 잘못된 추격을 줄이는가? |

## 2. 기대 기여

1. NavRL++의 structured temporal policy를 직접 관측한 camera–LiDAR target track으로 확장하는
   17-token Transformer architecture.
2. perception error가 navigation failure로 전파되는 과정을 density × target speed × occlusion 축에서 분석.
3. camera-only, LiDAR-only, tracking/no-tracking, CNN, LSTM, Transformer, PF/no-PF의 통제된 비교.
4. RTX 3070 8GB에서 재현 가능한 simulator/data/training/evaluation pipeline.

## 2. 정보 방화벽 — 무엇이 actor에 들어갈 수 있는가

## 고정 정보 경계

| 데이터 | Perception 입력 | Actor 직접 입력 | Label/Reward/Critic/Eval |
|---|---:|---:|---:|
| RGB-D camera | 예 | 아니오 | — |
| LiDAR range/points | 예 | 아니오 | — |
| structured obstacle/target history | 출력 | **예** | — |
| proprioception | tracker 보조 | **예** | — |
| target semantic mask/id | **아니오** | **아니오** | label/eval만 |
| GT target position/velocity | **아니오** | **아니오** | label/reward/critic/eval |
| GT visibility/occlusion | **아니오** | **아니오** | label/eval만 |

## 3. 사전등록 가설

## 연구 가설

- H1: camera와 LiDAR의 결합은 어느 한 센서만 사용할 때보다 표적 위치 RMSE와 capture를 개선한다.
- H2: temporal memory는 완전 가림 뒤 재탐지 시간과 track survival을 개선한다.
- H3: NavRL++식 Transformer+perception-failure fine-tuning은 single-step CNN보다 가림 견고성과 제어
  평활도가 높고, LSTM보다 accuracy–smoothness–latency Pareto 우위를 보인다.
- H4: uncertainty-aware policy는 마지막 추정 위치로 돌진하는 정책보다 충돌률이 낮다.
- H5: Transformer의 이점은 저밀도·항상-visible 조건보다 고밀도·긴 가림 조건에서 커진다.

논문 주장 한 문장:

## 논문 핵심 주장 후보

> Extending NavRL++ temporal reasoning with directly observed camera–LiDAR target tracks and
> perception-failure-aware fine-tuning enables a UAV to pursue a temporarily occluded target without
> privileged target-state input while maintaining collision avoidance and smooth control.

---

## 4. 방법 — NavRL++식 인지 + Transformer 정책

### 4.1 원논문 ablation 근거

### 2.2 무엇이 실제로 좋아졌는가

NavRL++의 Transformer가 입증한 직접적인 장점은 **표적 detection AP 향상**이 아니라 **짧은 관측 이력에
의한 시간 추론과 제어 평활화**다. 논문의 intra-simulator ablation에서 improved-training CNN 정책
`NavRL-IT`의 전체 success/control effort는 92.85%/0.093 m/s²이고, Transformer를 넣은
`NavRL-IT-T`는 90.54%/0.043 m/s²이다. 즉 Transformer 단독은 success를 자동으로 올린 것이 아니라
control effort를 절반 이하로 줄였다. Transformer와 perturbation-aware fine-tuning을 함께 적용한 최종
`NavRL++`가 94.08%/0.048 m/s²로 가장 좋은 종합 결과를 냈다.

### 4.2 17-token 사양

### 3.3 NavRL++-Target token 설계

기본 sequence는 17 tokens로 고정한다.

- `[CLS]`: 1 token.
- Static obstacle: 현재 local ray-distance map 1 token.
- Dynamic obstacle history: 5 tokens, 각 시점에 최대 N개 obstacle의 position/velocity/radius/confidence.
- Robot history: 5 tokens, 각 시점의 velocity, yaw rate, altitude, previous action.
- Target history: 5 tokens, 각 시점의
  `[existence, relative position, relative velocity, size, camera confidence, LiDAR confidence,
  covariance, measurement age]`.

현재 corrected-v2 설정은 history 2초/0.5초, dim 64, 4 heads, 4 layers, FFN 128,
**Transformer dropout 0.0**이다. 센서 detector dropout 0.3은 네트워크 dropout과 다른 입력 교란이다.
0.1초 관측을 모두 token으로 늘리지 않고 tracker는 10 Hz로 갱신하되 policy history는 2 Hz로 sampling한다.
`[CLS]`에서 target state, uncertainty, obstacle latent와 velocity action을 예측한다. 원시 camera/LiDAR patch
token을 직접 attention하는 기존 아이디어는 1차 구조가 아니라 후속 ablation으로 내린다.

### 3.4 정책과 안전 동작

- Actor 입력: 위 17-token sequence 또는 `[CLS]` latent만 허용.
- Actor 금지: GT target position/velocity/visibility와 target semantic channel.
- Critic privileged state는 actor storage와 물리적으로 분리한다.
- target confidence가 낮거나 measurement age가 길면 pursuit 속도를 낮추고, 안전 회피와 search yaw를 수행.
- target도 collision geometry에는 포함해 추격 중 표적/장애물과의 안전거리를 유지.


### 4.3 토큰 표현의 이중 압축 문제 (열린 설계 논점)

- 여기서 “8 obstacle tokens”는 Transformer에 들어가는 독립 토큰 8개가 아니다. 구현은
  각 시점의 8×12 obstacle proposal을 이어 붙여 MLP로 64차원 **단일 history token**으로
  압축한다. 따라서 실제 병목은 `4×72 scan → 8 surface proposals`의 중복 선택과
  `8 proposals → 1 temporal token`의 이중 압축이다.
- 원 NavRL++는 static geometry를 ray-distance CNN token으로 유지하고, 최대 5개 object
  slot은 position/velocity/radius가 추적되는 dynamic obstacle에 사용한다. 현재 arena의
  정적 막대를 velocity=0인 obstacle-history slot에 넣는 것은 원 설계와 다른 확장이며,
  suppression 폭만 조절하기 전에 이 정적/동적 역할 혼합을 ablation해야 한다.

후속 방향:

3. free-space corridor를 명시적 18번째 token으로 append하는 통제 ablation을 먼저 수행한다.
   geometry, observation schema, checkpoint warm-start를 각각 독립 gate로 검증하며, fixed-density
   평가 gate를 통과하지 못하면 같은 표현의 epoch만 늘리지 않는다.
4. hard selector가 부족하면 72개 bearing feature를 8개 latent slot으로 압축하는
   Set Transformer/Slot Attention 후보를 시험한다. PPO 보상만으로 slot collapse를
   맡기지 않고, simulator GT는 actor 입력이 아닌 training-only Hungarian auxiliary
   target으로만 사용한다. 85개 중 보이는 모든 막대를 8 slot에 매칭할 수 없으므로
   TTC/goal corridor로 정의한 top-risk subset만 matching한다.
5. 독립적인 8 obstacle token을 Transformer에 직접 넣는 ablation과 8→12 capacity 증가는
   selection이 개선된 뒤 수행한다. 둘 다 observation/network shape가 바뀌어 fresh training이
   필요하므로 첫 실험으로 쓰지 않는다.

### 4.4 PPO update 무결성 계약

Squashed-Gaussian 정책의 안정성은 “KL을 로그로 본다”가 아니라 다음 원자적 commit 계약으로
보장한다.

- rollout 당시 latent Normal의 `mu/sigma`를 immutable behavior policy로 보존한다. minibatch
  사이에 갱신되는 rl-games dataset reference를 hard gate 기준으로 쓰지 않는다.
- PPO epoch 전 actor와 asymmetric central critic 양쪽의 model parameter/buffer
  (input/value RunningMeanStd 포함), optimizer moments/step, AMP scaler를 snapshot한다.
- 마지막 optimizer step 뒤 전체 rollout을 normalization frozen 상태로 재추론한다. minibatch
  평균 KL>0.04 또는 actor/critic output, 누적 loss, model buffer, Adam moment/step, AMP scaler 중
  하나라도 NaN·Inf면 actor와 central critic을 같은 epoch 경계로 함께 복원한다.
- rollback 뒤 learning rate를 낮추며, 반복 rollback은 무한 학습하지 않고 fail-stop한다.
- rollback total/streak는 checkpoint state다. patience fail-stop도 소비한 rollout frame을 반영하고
  복원된 policy/critic/optimizer와 낮춘 LR을 durable `last_*` checkpoint로 먼저 저장한다.
- checkpoint resume는 명시적 LR override가 없으면 저장된 `current_action_learning_rate`를 복원해,
  이전 rollback의 backoff를 일반 resume에서도 조용히 되돌리지 않는다.
- pre-tanh latent margin은 횡축 하나가 아니라 모든 action axis를 보호한다. action-space entropy의
  큰 음수 급락과 edge saturation은 성공 지표가 아니라 latent saturation 경보다.
- multi-GPU/RNN은 동기화된 all-rank transaction/sequence audit가 구현되기 전에는 이 학습 경로를
  지원한다고 간주하지 않는다.
- 붕괴 복구는 seed1 고정130/100-epoch smoke와 seed42/2,049-episode held-out을 분리한다. 모든 smoke
  epoch의 KL/OOB/rollback, outcome count-rate, checkpoint/result SHA, same-shape perception/control
  계약이 결합된 증명서가 없으면 density curriculum을 재개하지 않는다. 증명서의 자기보고 숫자는
  신뢰하지 않고 실제 TensorBoard 9501–9600 window와 held-out JSON을 다시 읽어 canonical payload를
  재구성한다.
- held-out JSON은 evaluator가 만든 nonce/로그/스크립트 receipt와 실제로 재생한 checkpoint snapshot
  SHA에 결합한다. simulator의 config class, physics dt/substeps와 RL step dt도 task에서 실측한다.

General-spawn FOV curriculum은 표적의 world-direction 분포를 좁히지 않는다. 초기에는 drone yaw만
표적 쪽으로 정렬해 상대 방위를 camera 내부로 제한하고, curriculum 진행에 따라 ±180°까지 넓힌다.
포화 뒤와 held-out full-distribution 평가는 처음부터 unrestricted yaw를 사용한다.

평가기는 checkpoint의 arena·sensor·representation·moving-target·action-policy 계약을 고정하고,
조건별 episode accounting이 검증된 JSON/CSV가 없으면 성공 종료로 간주하지 않는다. 붕괴 복구
smoke는 추가로 100-epoch KL/OOB 기록과 2,049-episode held-out 결과를 checkpoint hash에 결합한
PASS 증명서가 있어야 밀도 커리큘럼으로 진입할 수 있다. 이 held-out는 main/base_sim(dt=0.01)/128
env, 목표 거리 6–28 m 전체, 최종 FOV를 강제해 saved curriculum clock이나 4GB simulator가 평가
난이도를 몰래 바꾸지 못한다. smoke/continue lineage는 epoch뿐 아니라 frame·task-step·horizon 관계도
감사된 ep9500 anchor에서 검증한다.

---

## 5. 실행 순서 (단일 번호 체계 P0–P6)

`ROADMAP`/`PHASE3_PLAN`/`PERCEPTION_TRANSFORMER_PLAN`에 세 가지 다른 번호가 붙어 있던 것을
여기로 단일화한다. 각 단계의 실제 완료 여부와 수치는 `WORKLOG.md`를 본다.

| 단계 | 내용 |
|---|---|
| P0 | GT→actor 정보 방화벽 + structured observation 스키마 |
| P1 | RGB-D/LiDAR 검출기 + held-out 검증 |
| P2 | association + 칼만 추적기 + 공분산 |
| P3 | 17-token temporal Transformer actor 연결 |
| P4 | 속도 상향 + 이동표적 요격 + PPO 안정화 |
| P5 | 충돌 저감 — 고도 제어 → look-ahead → 장애물 표현 |
| P6 | 밀도 커리큘럼 → **밀도 × 표적속도 지도** (논문 핵심 그림) |
| P7 | sim-to-real: exact BOM/calibration/time sync → real-log sensor profile → tracker replay → 측정 분포 기반 perturbation. 현재 72시간 실행 authority는 [`docs/SIM2REAL_3DAY_EXECUTION_PLAN.md`](docs/SIM2REAL_3DAY_EXECUTION_PLAN.md) |

### 5.1 완료 마일스톤 (B0–B5)

## 1. 완료 기반

| 단계 | 산출물 | 상태 |
|---|---|---|
| B0 | Aerial Gym/Isaac Gym/Warp/rl_games 환경 | 완료 |
| B1 | random bar arena, 36×4 LiDAR, capture task | 완료 |
| B2 | learned yaw obstacle navigation | 완료 |
| B3 | density sweep와 curriculum infrastructure | 완료 |
| B4 | scripted moving target와 interception reward | 완료 |
| B5 | camera/LiDAR occlusion·throughput prototype | 완료, 단 oracle semantic prototype |


### 5.2 게이트 G0–G5

## 6. 구현 게이트

- G0: actor 경로에 GT/semantic target channel 참조 0개.
- G1: camera-only와 LiDAR-only가 각각 visible target에 대해 chance보다 유의하게 높은 recall을 보임.
- G2: fused visible-target median position error ≤0.30 m at 10 m.
- G3: 1초 simultaneous sensor miss 뒤 track survival/reacquisition ≥80%.
- G4: Transformer가 single-step CNN보다 control effort와 occlusion robustness를 개선하고, LSTM 대비
  accuracy–smoothness–latency Pareto 우위를 하나 이상 보임.
- G5: held-out layout에서 GT upper bound와 capture gap을 perception/tracking/policy error로 분해.


### 5.3 중단 조건

## 중단 조건

- 어느 한 센서 detector도 visible target을 못 찾으면 Transformer를 키우지 말고 sensor/asset/data부터 수정한다.
- Transformer 평가는 success만 보지 않고 NavRL++처럼 control effort와 perturbation robustness를 함께 본다.
- 8GB에서 OOM이면 raw image를 policy에 넣지 않고 detector output을 detach/cache하고 env를 256→128로 낮춘다.
- 고정 density에서 KL 또는 latent mean이 지속 상승하면 더 학습하지 않는다. 현재 epoch transaction이
  last-known-good actor/optimizer/RMS를 자동 복원해야 하며, 연속 rollback 한도를 넘으면 해당 run을
  fail-stop하고 더 낮은 LR의 검증된 checkpoint에서만 재개한다.

---

## 6. 실험 설계

### 6.1 밀도 실현가능성 기하 감사

## 8. 고밀도와 저속에 대한 geometry audit

`tools/analyze_navrl_density_feasibility.py`는 실제 bar URDF 폭, 0.28 m 기체 footprint, 현재 random-rejection
spacing 완화를 재현한다. 50-layout probe에서 side clearance 0.2 m를 요구해도 130 bars는 100%, 150 bars는
92%가 spawn-to-goal 연결 경로를 가졌다. 즉 현재 고밀도 task는 대부분 **물리적으로 불가능해서** 실패하는
것이 아니다. 다만 130/150 bars에서는 모든 layout이 spacing 완화에 들어가고, 150 bars의 8%는 0.2 m
안전 마진 경로가 사라졌다.

표적 속도 0 m/s인 기존 held-out baseline도 고밀도에서 crash가 컸으므로 표적을 느리게 하는 것만으로는
해결되지 않는다. pursuer 자체를 느리게 하면 정지거리(가속 한계 2 m/s², latency 0.1 s 가정)가 2 m/s의
1.20 m에서 1 m/s의 0.35 m로 줄어 충돌은 감소할 가능성이 크지만, 30초 episode에서 24 m를 통과해야 해
timeout이 증가한다. 새 policy 학습 후 `eval_navrl_speed_density_grid.sh`로 target speed와 pursuer limit을
분리해 측정한다.

### 6.2 단계적 커리큘럼 레시피

## 9. 권장 curriculum과 3-D 검증 앱

거리 6000 epoch 뒤 밀도 6000 epoch를 무조건 수행하거나, 두 난이도를 처음부터 함께 선형 증가시키지 않는다.
현재 권장안은 `train_navrl_perception_staged.sh`의 다음 구조다.

- 0–4000 epoch: 25 bars로 고정하고 goal distance를 7→16 m로 확장한다.
- 4000 epoch 이후: 거리는 최종 범위로 유지하고, capture≥0.65/4096 episodes일 때만 막대를 5개씩
  25→110으로 승급한다.
- density 단계 reset의 25%는 5–10 m goal을 재생해 가까운 거리의 회피·재탐지 능력 망각을 막는다.
- 이 clean/static-target run을 통과한 뒤에만 perception dropout/noise, 마지막으로 target speed를 추가한다.

이는 “두 축을 영원히 완전히 분리”하는 방식이 아니라, 첫 기술을 얻은 뒤 다음 축을 competence-gated로
추가하고 이전 분포 일부를 계속 replay하는 staged curriculum이다. 총 10000 epoch는 상한이며, density 승급이
멈추면 epoch를 더 쓰는 대신 detector error/collision/timeout 원인을 먼저 분해한다.

여기서 `k_max=16`은 **막대 수 상한이 아니라 goal-distance curriculum의 최대 반경(m)** 이다.
학습 중 `k_max`는 capture gate를 통과할 때마다 시작값에서 16 m까지 단계적으로 증가하며, 포화 뒤에는
밀도 curriculum과 경쟁하지 않도록 고정한다. fixed-100 표현 A/B에서도 4–16 m를 고정해 오직 표현 차이만
비교한다. 16 m 밖 일반화는 arena 경계와 out-of-bounds 혼입을 피하기 위해 별도 평가로 다루며, 필요하면
arena 자체를 확장한 뒤 `k_final`을 함께 높인다.

### 6.3 평가 그리드

### Phase 4 — 핵심 실험

- density: bars {25,50,75,110,150}.
- target speed: {0,0.5,1.0,1.5,2.0} m/s; 2.0은 평가 전용.
- occlusion duration: {0,0.25,0.5,1.0,2.0}s 또는 관측된 연속 가림 bin.
- trajectory: CV/waypoint/circle, held-out layout과 appearance.
- 조건당 ≥1,000 episodes, 학습 seed ≥3.

### 6.4 모델 비교

## 6. 모델 비교

| 모델 | Raw sensors | Temporal | Cross-modal attention | 역할 |
|---|---:|---:|---:|---|
| GT state | 아니오 | — | — | upper bound only |
| camera CNN | camera | 아니오 | 아니오 | detector baseline |
| LiDAR temporal CNN | LiDAR | 예 | 아니오 | geometry/motion baseline |
| fused detector + tracker | 둘 다 | 예 | 아니오 | structured perception front-end |
| single-step CNN policy | 둘 다 | 아니오 | 아니오 | temporal ablation |
| CNN+LSTM policy | 둘 다 | 예 | 아니오 | recurrent baseline |
| NavRL++-Target | 둘 다 | 예, 2초 | 예 | proposed |

### 6.5 리스크와 완화

## 7. 리스크

| 리스크 | 대응 |
|---|---|
| sparse LiDAR로 target/obstacle 구분 불가 | 시간차 motion feature, target geometry, 72×6 ablation; camera와 fusion |
| Transformer가 작은 데이터에서 과적합 | structured 17 tokens, NavRL++ dim 64, detector pretraining, strong split |
| 8GB OOM | detector offline pretraining, structured 17 tokens, gradient accumulation, env 128 fallback |
| detector 오류 또는 PPO update가 정책을 붕괴 | perception freeze, confidence input, density-aware collapse guard, immutable-behavior KL 전수감사, model/RMS/Adam/AMP epoch rollback, 전축 latent margin |
| semantic label 누출 | 별도 label dict/dataloader, actor observation schema test, code review gate |
| sim-to-real gap | texture/light/noise/dropout randomization, calibration perturbation, real-log validation |
| virtual target / unvalidated vehicle dynamics | physical-target fresh lineage; identical sensor/contact OBB; substep contact telemetry; hardware-ID manifest; PPO blocked until 70/150/205/300 gate |

---

## 7. 구현 단위와 체크포인트 규약

## 4. 구현 단위

| 패키지/파일 | 예정 변경 |
|---|---|
| `navrl_task.py` | raw sensor/label 경계, perception output을 actor에 연결 |
| `navrl_detector.py` | oracle detector에서 raw render/learned-model adapter로 역할 변경 |
| `navrl_lidar_config.py` | semantic ID는 renderer 내부에만 유지; actor는 raw nearest range만 사용 |
| `navrl_task_config.py` | detector/tracker, 2초 history, PF, Transformer 설정 추가 |
| `motar_perception_dataset.py` | 신규 dual-sensor sequence recorder/dataset |
| `navrl_perception.py` | RGB-D appearance head + LiDAR association + Kalman tracker + structured history |
| `navrl_transformer_network.py` | NavRL++-style 17-token temporal actor–critic |
| `tests/test_navrl_perception.py` | GT-free API, RGB-D detection, LiDAR continuation, covariance 테스트 |


## 6. 체크포인트 규칙

- old GT/semantic vision checkpoint는 새 raw-perception 모델과 호환되지 않는다.
- navigation-only backbone은 transfer 후보지만 observation layer는 새로 초기화한다.
- perception, policy, optimizer, dataset/config hash를 한 checkpoint manifest에 기록한다.
- `gen_ppo.pth`만 보관하지 말고 detector validation-best와 navigation capture-best를 별도로 보관한다.

---

## 8. 검증 단계 (ref5in)

> **실행 authority는 [`VERIFICATION.md`](VERIFICATION.md).** 이 절은 charter와의 연결만 남긴다.

P0–P3 fail-closed gate, D0/D1 진단, camera-range §8.29 사전등록·판정·다음 단계는
VERIFICATION.md에 통합했다. v2 205-bar / TTC / riskcap 역사(구 §8.1–8.22)는
[`docs/archive/RESEARCH_PLAN_v2_history.md`](docs/archive/RESEARCH_PLAN_v2_history.md)에 보관한다.

2026-08-23 검출 거리 Stage 1은 `RANGE_INCONCLUSIVE_AT_THIS_BUDGET`로 실행 종료되어 Stage 2
권한이 없다. P2 `STRICT FAIL`, D1 `FAIL`, P3 `BLOCKED`도 유지된다. P7의 유일한 다음 authority는
[`docs/SIM2REAL_3DAY_EXECUTION_PLAN.md`](docs/SIM2REAL_3DAY_EXECUTION_PLAN.md)의 exact
BOM/calibration, 210 trials와 real-log replay다. hardware/log 없이 임의 noise를 정하거나 GPU/PPO를
재개하지 않는다.

검증 단계에서도 §1–3 정보 방화벽, §4 방법, §6 실험 설계의 **claim 범위**는 변하지 않는다.
P0–P3 PASS는 hardware validation을 대체하지 않는다.

### Physical-target prerequisite (fresh lineage)

Moving-target physical claims require, in order: dynamic 6-DoF actor, identical camera/LiDAR/contact
geometry and pose, a frozen vehicle evidence manifest, 0.01 s motor/control telemetry, then the
predeclared 70/150/205/300 tracking/contact/feasibility gate. A same-shape checkpoint is not
transferable across this transition. The gate authority and measured outcome live in
`docs/navrl_physical_target_audit_2026-08-21.md`; failure blocks PPO rather than lowering thresholds.

---

## 9. 참고문헌

## 10. 보조 설계 근거

- [NavRL++ (2026)](https://arxiv.org/abs/2605.15559): structured perception, 2초 history,
  Transformer policy, perception-failure fine-tuning의 직접 기준.
- [NavRL code](https://github.com/Zhefan-Xu/NavRL): 기존 NavRL training/deployment 구현 기준.
- [GAFusion, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Li_GAFusion_Adaptive_Fusing_LiDAR_and_Camera_with_Multiple_Guidance_for_CVPR_2024_paper.html): LiDAR–camera adaptive fusion.
- [ORTrack, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Learning_Occlusion-Robust_Vision_Transformers_for_Real-Time_UAV_Tracking_CVPR_2025_paper.html): UAV occlusion-robust tracking.
- [TransFusion, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Bai_TransFusion_Robust_LiDAR-Camera_Fusion_for_3D_Object_Detection_With_Transformers_CVPR_2022_paper.html): object-query 기반 camera–LiDAR fusion 비교안.

## 2026-08-25 routed physical-target gate update

The fresh physical lineage remains blocked. Attempt 1 is a preserved 0/32 `VOID_EXECUTION` caused by
the conda `ninja` PATH failure. Attempt 2 records all 32 preregistered cells and passes execution
integrity, but fails the route mechanism and therefore does not authorize PPO. Its route-off/on
strict cell map is:

| bars \ speed | 0.6 | 0.9 | 1.2 | 1.5 |
|---:|:---:|:---:|:---:|:---:|
| 70 | P / F | P / F | F / F | F / F |
| 150 | P / F | P / F | P / F | F / F |
| 205 | P / F | P / F | F / F | F / F |
| 300 | P / F | F / F | F / F | F / F |

Across all four routed 70-bar speed cells, pooled plan success is `0.1454668471` (gate ≥0.99) and
fallback is `0.359296875` (gate ≤0.01). The separately gated 70 bars × 0.6 m/s cell records
`0.25` goal completions/env (gate ≥0.5), and same-goal reselection is zero. Across route-on cells,
0.125–0.635% local invalidation expands to 32.6–85.0% fallback;
status counts are dominated by `unsafe_start`, consistent with a soft-envelope recovery deadlock.
This is a controller/planner diagnostic, not a global-connectivity proof. PPO, hardware claims, and
arena-wide 300-bar claims remain out of scope. Thresholds and evaluator code remain frozen.

Authority: [`VERIFICATION.md`](VERIFICATION.md), [`attempt 2 summary`](results/navrl_physical_target_routed_gate_seed827_attempt2/summary.md),
and [`target-motion audit`](docs/target_motion_training_environment_audit_2026-08-25.md).

## 2026-08-26 Track B 실행 종료 addendum

Recovery-v2 lower-1.25는 `PASS_32_CELL_INTEGRITY / FAIL_ROUTE_MECHANISM`으로 실행됐다. 7/32만
PASS했고 모두 route-off이며, recovery는 0/16이다. 70-bar plan success는 `93.60%`, fallback은
`47.87%`, 70×0.6 goals/env는 `0.21875`, `NO_CONNECTOR` occupancy는 `63.06%`다.

사전등록된 no-anchor follow-up도 실행·검증됐으나 primary `n=1`, identity disagreement `0`으로
`INCONCLUSIVE`다. 따라서 이 charter의 route/recovery 후보들은 추가 Track B GPU, PPO, retune,
1.5 m/s, env-count 변경 또는 32-cell rerun을 승인하지 않는다. 현행 실행 authority는 Track A의
hardware/real-log 계약뿐이다.
