# RESEARCH_PLAN §8.1–8.22 — v2/TTC/riskcap 역사 기록

> 2026-08-20 문서 통합 시 RESEARCH_PLAN에서 분리 보관. 현재 검증 authority는 [VERIFICATION.md](../../VERIFICATION.md).

## 8. v2 205-bar 종료 후 실패·한계 감사 사전계획 (2026-08-02)

이 절은 작성 당시 실행 중이던 recovery curriculum을 보고 난 뒤 설명을 끼워 맞추지 않기 위한
**사전등록 분석 계획**이다. 실제 판정은 §8.7에 동결했다.
작성 당시에는 프로세스·환경변수·checkpoint에 손대지 않고 `max_epochs=30000` 정상 종료 또는 안전
fail-stop을 관측하기로 했다. 이후 사용자 결정으로 ep24010에서 안전 중단했으며 §8.7에 결과를 기록했다.

### 8.1 분석 대상과 canonical lineage

분석 epoch 계보는 다음처럼 한 번만 잇는다. 재시작 전 run의 ep20701–20746은 ep20700 checkpoint에서
다시 학습된 중복 구간이므로 최종 통계에서 제외한다.

1. recovery smoke: ep9501–9600;
2. recovery curriculum: ep9601–20700;
3. recovery continuation: ep20701–종료 epoch.

종료 직후 다음 증거를 read-only로 동결한다.

- 마지막 `last_gen_ppo_ep_*.pth`와 SHA-256, epoch/frame/task-step, env-state 전체 계약;
- run summary/정상종료 marker/exit reason, 실행 명령, git commit과 dirty diff;
- 세 run의 TensorBoard event, `epoch_metrics.csv`, session log, promotion/hold window 원문;
- arena·placement·sensor·selector·action distribution·PPO rollback/LR 계약;
- 평가 시 실제로 재생한 checkpoint snapshot/receipt/result JSON.

`gen_ppo.pth`는 reward가 낮은 밀도에서 최고가 되는 curriculum 특성상 분석·평가 checkpoint로 사용하지
않는다. 모든 평균은 서로 다른 episode 수를 가진 epoch 비율을 무작정 평균하지 않고, 가능한 항목은
종료 episode count로 가중한다. 16,384-episode gate는 로그의 정확한 count/rate를 사용한다.

### 8.2 수치 분석 순서

| 순서 | 분석 축 | 반드시 산출할 값 |
|---|---|---|
| A | 학습 완결성 | 최종 epoch, exit reason, checkpoint/marker 일치, 유실·중복 epoch |
| B | 밀도별 학습 | bars별 dwell epoch/episode, 모든 gate window, capture/crash/timeout, 승급 소요량 |
| C | 시간 추세 | bars 고정 구간의 500/1000/2000-epoch 추세, change point, plateau CI, seed 내 자기상관 |
| D | held-out | density별 deterministic capture/crash/timeout과 Wilson 95% CI, 최소 2,049 episodes/cell |
| E | 탐색 잡음 | 같은 checkpoint의 deterministic 대 stochastic 실행 격차, action-axis edge/variance |
| F | 실패 모드 | bar contact/below/OOB/timeout 비중, 충돌 위치·시간, commanded stall, closest approach |
| G | 표현 용량 | bars in range/FOV, occupied bins, hit-token-given-FOV, associated/unique/duplicate, depth error |
| H | 대칭성 | target bearing 좌·우와 mirrored layout별 capture/action sign, signed-y 편향과 chirality |
| I | PPO 안정성 | behavior KL, entropy, explained variance, actor/critic loss, LR, rollback, RMS/Adam/AMP finite |
| J | 환경 난이도 | v2 배치 연결성, corridor 폭, goal 거리·target speed·가림 분포, 물리적 불가능 비율 |

학습 tail은 진단값이고 publication result가 아니다. 최종 checkpoint는 우선 bars
`{130,160,190,205,220}`에서 평가한다. 205에서는 target speed `{0.3,0.9,1.5}`와 최소 두 평가 seed를
추가해 밀도와 속도를 분리한다. 계산 비용이 허용되면 70/100/250도 경계점으로 확장한다. 조건을 바꾼
cell끼리는 직접 평균하지 않고 arena 면적당 밀도, goal 거리, target motion, detector mode, action mode를
함께 표기한다.

### 8.3 원인 가설과 반증 기준

| ID | 가능한 한계 | 지지 증거 | 기각/하향 조건 | 최소 다음 시험 |
|---|---|---|---|---|
| H-GATE | 0.70 online gate가 실제 정책보다 엄격하거나 noisy | held-out≥0.70인데 stochastic gate만 지속 hold | deterministic·stochastic 둘 다 <0.70 | 같은 checkpoint action-noise A/B; 새 PPO 학습 금지 |
| H-GEOM | 205+에서 통과 가능한 free-space가 자주 단절 | v2 동일 배치에서 inflated-body path 존재율 급락 | 0.2 m margin path가 대부분 존재 | 배치 seed replay/geometry audit |
| H-REP | 8-sector surface representation이 위험 기하를 버림 | FOV 내 수십 bars 대비 unique≈4, collision bar 미표현 | oracle-risk coverage가 높고 표현 A/B 무효 | two-depth/ray-distance token의 짧은 fixed-205 A/B |
| H-COMP | 8 proposal→1 temporal token 이중 압축 | selector coverage는 정상이나 downstream sensitivity 낮음 | 독립 token도 동일하고 critic/actor probe가 입력을 사용 | independent slots 또는 static ray token ablation |
| H-BIAS | 횡축 정책·좌표계가 한 방향에 치우침 | signed-y/edge98 비대칭, mirror 성능 불일치 | 좌우 CI가 겹치고 sign-equivariance 통과 | mirror-paired evaluator 및 좌우 균형 auxiliary |
| H-CTRL | 속도·yaw·latency가 좁은 corridor에 과격함 | 충돌 직전 command/edge/stopping distance가 clearance 초과 | 저속 deterministic 평가에서도 bar contact 불변 | inference-only speed/yaw grid; reward 변경 금지 |
| H-CURR | 단일 current-density 분포가 이전 기술을 망각시킴 | 저밀도 held-out도 이전 checkpoint보다 유의 하락 | 130–190 유지 또는 개선 | density replay 비율 A/B |
| H-PERC | target detection/tracking/가림이 병목 | timeout·track age·reacquisition이 density와 함께 급증 | 실패 대부분 visible 상태 bar contact | detector/tracker 조건부 실패표 |
| H-PPO | optimizer drift/critic failure가 plateau를 만든다 | KL/entropy/EV/LR/rollback change point가 성능 하락과 일치 | 안정성 지표가 정상 범위이고 fixed eval도 plateau | 더 긴 학습 금지; 표현/환경으로 이동 |

둘 이상의 레버를 동시에 바꾸지 않는다. 특히 gate threshold, action noise, obstacle representation,
speed limit, reward를 한 run에서 함께 바꾸면 원인을 식별할 수 없으므로 금지한다.

### 8.4 환경·파라미터 감사

기존 `tools/analyze_navrl_density_feasibility.py`의 24×24 m legacy 결과는 v2 한계의 근거로 쓰지 않는다.
40×40×3 m, full-width `navrl_band`, 실제 bar 크기·touch/gap/merge fallback, 0.28 m 기체 footprint를
그대로 미러링해 bars 130–300의 연결성과 corridor 폭을 재계산한다. 단순 west→east 연결뿐 아니라 실제
spawn/goal pair와 moving-target trajectory가 reachable component 안에 머무는지도 측정한다.

파라미터는 다음 네 층으로 분리한다.

- 환경: arena, bars, placement, goal 거리, target pattern/speed, episode length;
- 관측: LiDAR FOV/range/beams, detector/tracker, token selector/capacity/history;
- 제어·정책: v/yaw/tilt limit, action distribution/sigma, network/token compression;
- 학습: reward, LR/KL/rollback, horizon/minibatch, dwell/window/threshold/replay.

최종 보고에는 각 파라미터를 `고정됨 / 실측 민감도 있음 / 아직 미식별 / 과거 실험과 교란됨`으로 분류한다.
후보가 많다는 이유로 전수 grid를 돌리지 않고, 실패 로그가 직접 가리키는 층부터 작은 factorial 또는
paired A/B를 설계한다.

### 8.5 관련 연구 대조 원칙

최신 NavRL/NavRL++와 UAV obstacle avoidance/interception, free-space representation, self-paced
curriculum, bounded continuous-control exploration 연구를 원문 기준으로 다시 조사한다. 비교표에는 최소
arena 크기, static/dynamic obstacle 수와 면적당 밀도, obstacle 크기/높이, target 존재·속도, 센서/GT,
성공 반경, episode 제한, deterministic/stochastic 평가, seed/episode 수를 넣는다. 조건이 다른 논문 SR을
순위표처럼 직접 비교하지 않고, 우리 시스템에 빠진 설계 요소와 재현 가능한 ablation 후보를 찾는 데만
사용한다.

### 8.6 다음 학습 허가 게이트

다음 메인 학습은 아래가 모두 충족될 때만 실행 명령을 만든다.

1. 최종 run artifact와 held-out 결과가 동결되고 canonical lineage audit가 PASS;
2. dominant failure가 적어도 `환경/표현/제어/탐색잡음/optimizer` 중 한 층으로 좁혀짐;
3. 다음 변경이 한 가지 메커니즘만 바꾸고 반증 가능한 수치 gate를 가짐;
4. 1650 Ti에서 shape/config/8-epoch smoke와 짧은 paired 평가가 PASS;
5. warm-start 가능 여부, checkpoint migration, 실패 시 rollback 경로가 문서화됨.

판정에 따라 준비할 후보는 다음과 같다.

- gate/noise 문제: 새 학습보다 동일 checkpoint의 stochastic/deterministic sigma 평가를 먼저 수행;
- 표현 문제: 기존 corridor6을 연장하지 않고 two-depth 또는 static ray-distance token 한 개만 비교;
- 제어 문제: inference-only speed/yaw grid로 확인한 뒤에만 policy curriculum 변경;
- 물리 한계: 환경을 몰래 쉽게 만들지 않고 achievable-density boundary로 논문에 보고;
- detector/tracker 문제: navigation PPO를 더 돌리지 않고 perception gate부터 통과.

산출물은 `results/`의 machine-readable JSON/CSV, `WORKLOG.md`의 최종 수치·결정,
`CRASH_TUNING_LOG.md`의 메커니즘, `OPERATIONS.md`의 정확한 재현 명령, `README.md`와
`docs/status/`의 사용자용 요약이다.

### 8.7 핵심 감사 결과와 남은 인과 검증 (2026-08-02)

사용자 결정으로 continuation을 ep24010에서 안전 중단하고 ep24000 checkpoint를 동결했다. canonical
계보는 14,510 epoch/59.43M samples이며, 205 bars에만 4,910 epoch/20.11M samples를 썼다. 완결된
hold는 10회이고 최근 7회 gate 평균은 **68.94%**라 같은 조건의 추가 epoch는 중단한다.

held-out deterministic sweep는 130/160/190/205/220 bars에서 각각
**84.77/79.66/73.99/72.44/68.49%**였다. 205 bars 동일 checkpoint의 stochastic 평가는
**67.35%**로 deterministic보다 **5.09 pp** 낮고 crash는 5.33 pp 높았다. 따라서 H-GATE는 “오발”이
아니라 **training-time sampled policy와 deployment mean의 실질적 간극**으로 정제한다. 두 metric을
서로 대체하지 않는다.

H-GEOM은 0.2 m clearance에서 random-pair connectivity 99.83%로, H-PPO는 behavior KL/EV/LR과
rollback/OOB 감사로 주원인에서 기각했다. timeout·stationary-drone도 낮았다. 205 bars deterministic
capture는 거리 6–11.5 m의 81.42%에서 22.5–28 m의 61.41%로 떨어지고, 실패 대부분은 bar contact다.
따라서 dominant layer는 긴·고속 trajectory에서 누적되는 **위험 선택/탐색 잡음**이다. full 4×72 scan이
존재하므로 단순 “8 token만 보여서 생긴 용량 한계”라고 단정하지 않는다.

다음 **학습 후보**는 동결 ep24000에서 fixed-205 `cluster_sector` 대 `ttc_sector` A/B다. launcher에서
바꾸는 환경변수는 `NAVRL_OBSTACLE_SELECTOR` 하나이며 arm당 **4,096,000 samples**로 맞춘다. 단,
2026-08-05 사후 코드 감사에서 selector semantics가 candidate FOV도 cluster 240°→TTC 360°로 바꾸는 것이
확인됐다. 따라서 이것은 순수 ranking-only가 아니라 **ranking+candidate-FOV representation bundle** A/B다.
1650 Ti의 70-bar 선행
A/B가 capture +9.86 pp/crash -8.06 pp였으므로 고밀도 재현 가치가 가장 높다. 고밀도 TTC 채택 기준은
같은 profile baseline 대비 capture **+2.0 pp 이상**과 crash **-2.0 pp 이하**를 동시에 만족하는 것이다.
launcher는 `train_navrl_v2_ep24000_ttc_ab.sh`로 준비했지만 아직 실행하지 않았다.

이 핵심 감사 작성 당시 그 전에 수행할 다음 단계는 **mirror-paired 평가**였다. 이는 ep24000 checkpoint를 inference-only로
재생하며 optimizer step, gradient, RMS 갱신, checkpoint 저장을 하지 않는다. 원본과 좌우 반전 조건의
capture/crash, action-y 부호 및 sign-equivariance만 비교한다. 이어서 두 번째 seed, 고정 target speed
0.3/0.9/1.5 m/s, 이전 checkpoint 대비 망각 비교를 끝낸다. 이 인과 검증과 1650 Ti 실기 smoke가
통과하기 전까지 fixed-205 A/B 학습은 준비 상태로만 둔다.

현재 핵심 수치와 재현 경로는 `results/navrl_v2_ep24000_limit_audit.{md,json}`에 고정한다. 미해결 항목은
mirrored-layout 좌우 대칭성과 learned detector gate이며, 현 A/B에 섞지 않는다.

### 8.8 동결 정책 1--3번 인과검사 사전등록 (2026-08-02, 실행 전)

대상은
`runs/ppo_260802_0020_navrl_v2-recover-curriculum-continue-s1/nn/last_gen_ppo_ep_24000_rew_44.73549.pth`
(SHA-256 `82f7978b42d9d9e95adcc638a40ae85fb3736fd897ae39cb8aa8333be39cf23f`) 하나로
동결한다. 세 검사는 모두 inference-only이며 optimizer/gradient/RMS/checkpoint를 갱신하지 않는다.
조건은 205 bars, 40×40×3 m, `cluster_sector`, deterministic action, full 6--28 m goal,
mixed target motion, speed U[0.3,1.5] m/s, analytic detector다.

1. 문서·사이트 정합성: 핵심 감사 완료와 인과검사 미완료를 분리하고, TTC A/B를 이미 채택했거나
   학습 중이라고 표시하지 않는다.
2. 좌우 반사 감사: (a) 실제 rollout 관측 `o`마다 `pi(o)`와 `pi(Mo)`를 동시에 계산해
   `pi(Mo)` 대 `M pi(o)`의 deterministic action 오차를 정확한 관측 pair로 측정하고,
   (b) 원 정책 `pi`와 conjugate 정책 `M pi M`을 같은 seed 42에서 각각 4,096 episodes 실행한다.
   비동기 종료 뒤 episode 순서가 달라지므로 (b)를 episode-paired라고 과장하지 않고 common-seed
   aggregate A/B로 분석한다. capture 또는 crash 차이의 절댓값이 2.0 pp 이상이고 95% 차이 CI가
   0을 제외하면 material chirality로 판정한다. aggregate outcome이 equivalence여도 action-pair의
   lateral/yaw MAE, sign mismatch, signed mean을 별도 보고해 상쇄를 숨기지 않는다.
3. 독립 seed 재현: seed 43에서 2,049 requested episodes를 실행한다. 기존 seed 42 capture
   72.439%와의 차이 및 95% CI를 보고한다. 절댓값 3.0 pp 이내이고 두 95% CI가 겹치면 practical
   replication, 아니면 seed sensitivity로 분류한다. crash/timeout과 bar-contact 구성도 함께 비교한다.

평가 도중 checkpoint나 evaluator bytes가 바뀌면 해당 cell을 무효화한다. 결과를 본 뒤 위 margin이나
판정 규칙을 바꾸지 않는다. 1--3번 결과만으로 새 PPO 학습을 시작하지 않으며, 다음 fixed-speed/forgetting
검사와 1650 Ti 검증 전까지 TTC A/B는 준비 상태를 유지한다.

### 8.9 1--3번 실행 결과와 해석 (2026-08-02)

동결 checkpoint로만 평가했고 학습은 수행하지 않았다. seed42 205-bar mirror aggregate는 원 정책
**70.97% capture / 25.71% crash**와 conjugate `M pi M` **70.17% / 26.56%**였다
(capture 차이 -0.81 pp, 95% CI -2.78..+1.17). 사전등록한 aggregate material-chirality 규칙은
NO다. 그러나 대칭 환경분포에서는 `pi`와 `M pi M`의 평균이 정책 equivariance와 무관하게 상쇄될 수
있으므로 이것을 “정책 대칭 PASS”로 해석하지 않는다.

실제 rollout의 정확한 관측 pair 548,736개에서 `pi(Mo)`와 `M pi(o)`의 action MAE
`[x,y,z,yaw]=[0.926,1.235,0.416,1.002]`, lateral sign mismatch는 **73.08%**였다. reflection
schema를 898차원 전부 재감사했으며 signed field 누락은 없었다. checkpoint input RMS도 mirror mean
MAE 0.0317(특히 robot history 0.3243), scan variance 상대 MAE 11.9%라 정규화와 정책 모두 한 방향
방문분포를 흡수했다. 즉 H-BIAS는 **action-level supported**다.

그럼에도 초기 target bearing의 negative-y/positive-y capture는 **71.17%/70.97%**
(positive-negative -0.21 pp, 95% CI -3.03..+2.61)로 현재 대칭 arena에서 outcome penalty는 검출되지
않았다. 계측 추가 재생은 기존 4,096회 original 결과와 outcome count가 완전히 같아 비간섭도 확인했다.

독립 seed43은 2,049회에서 capture/crash/timeout **72.77/24.74/2.49%**로 seed42
**72.44/25.07/2.49%**를 재현했다. capture 차이는 +0.33 pp(95% CI -2.40..+3.06)로 사전등록
replication gate를 PASS했다. 다음은 고정속도 0.3/0.9/1.5와 이전 checkpoint 망각 비교다. 새 학습은
아직 금지하며, 이후 학습 후보에는 selector 한 변수뿐 아니라 reflection augmentation/RMS symmetry를
독립 arm으로 분리해 한 run에 섞지 않는다.

### 8.10 4번 고정 표적속도 인과검사 사전등록 (2026-08-02, 실행 전)

다음 단계는 새 학습이 아니라 §8.8과 동일한 ep24000 checkpoint
(SHA-256 `82f7978b42d9d9e95adcc638a40ae85fb3736fd897ae39cb8aa8333be39cf23f`)를 동결한
inference-only 평가다. 205 bars, seed 42, deterministic/original action, full 6--28 m goal,
`cluster_sector`, mixed motion, analytic detector를 유지하고 **target speed 하나만** 각각
0.3/0.9/1.5 m/s로 고정한다. cell당 2,049 requested episodes이며 optimizer, gradient, input RMS,
checkpoint는 갱신하지 않는다.

주 비교는 1.5 m/s minus 0.3 m/s capture다. 독립 rollout 비율차의 95% CI를 보고하고, 차이가
**-3.0 pp 이하이며 CI 상한도 0 미만**이면 material speed sensitivity로 판정한다. 0.9 m/s는
capture의 단조 감소 여부와 crash/bar-contact 전환을 기술하는 중간점이다. 결과를 본 뒤 속도,
표본수 또는 3.0 pp margin을 변경하지 않는다. 이 검사는 속도 효과만 분해하며 curriculum이나
정책을 개선하지 않는다. 완료 뒤 이전 checkpoint를 같은 205-bar 계약으로 재평가하는 망각 비교를
수행하고, 그 전에는 TTC/대칭성 학습 arm을 시작하지 않는다.

실행기는 `eval_navrl_v2_ep24000_fixed_speed.sh`, 결과 위치는
`results/navrl_v2_ep24000_fixed_speed/`로 고정한다. 기존 v2 evaluator는
`NAVRL_V2_FIXED_TARGET_SPEED`가 있을 때 task가 보고한 mode/min/max/point speed가 요청값과 모두
일치해야만 결과를 승인하며, 학습 support [0.3,1.5] m/s 밖의 값을 실행 전에 거부한다.

### 8.11 4번 고정 표적속도 실행 결과 (2026-08-02)

세 cell 모두 checkpoint/evaluator SHA, nonce, physics, seed, action/reflection mode 및 episode accounting
검증을 통과했다. capture/crash/timeout은 0.3 m/s에서 **73.26/23.04/3.71%**, 0.9 m/s에서
**72.62/25.09/2.29%**, 1.5 m/s에서 **67.35/30.75/1.90%**였다. 1.5-minus-0.3 capture는
**-5.91 pp** (95% CI **-8.70..-3.11 pp**)로 §8.10의 material speed-sensitivity gate를 통과했다.
capture는 세 점에서 단조 감소했다.

실패 변화는 timeout 증가가 아니라 충돌 증가다. bar-contact 절대 비율은
**22.11→24.26→29.97%**였고, 1.5-minus-0.3은 +7.86 pp다. 같은 구간 lateral executed-edge98은
2.41→4.92%로 늘었지만 평균 비행속도는 2.386→2.400 m/s로 거의 늘지 않았다. 평균 command-speed
norm은 이미 2.958→2.969 m/s였다. 따라서 현재 증거는 “빠른 표적 자체를 못 따라가 timeout”보다
**추가 속도 요구를 실제 속도 여유로 흡수하지 못하고 더 공격적인 경계 action/궤적으로 바꿔 막대에
충돌**하는 병목을 지지한다. 이는 속도 curriculum 하나만 더 오래 돌리는 것보다 위험-aware selector와
action/clearance 제어 축을 분리 검증해야 한다는 근거다.

다만 fixed-speed cell은 async reset 뒤 episode-by-episode paired가 아니므로 거리 bin의 비단조 세부값을
인과효과로 과해석하지 않는다. 다음 순서는 사전등록대로 동일 205-bar 계약의 이전 checkpoint 망각
비교다. 그 결과가 현재 ep24000의 성능 저하를 보이지 않으면 고밀도 TTC selector A/B로, 망각이 크면
replay/mixture curriculum 축으로 먼저 분기한다.

### 8.12 ep19100 대 ep24000 망각검사 사전등록 (2026-08-03, 실행 전)

이전 기준은 205 bars 승급 직전의 마지막 durable checkpoint ep19100
(SHA-256 `82d1eeac1798b4b465274551ec2363fb377be781c0a58e930577ddb822f55044`)이고, 비교 대상은
205 bars에서 4,900 epoch를 추가 학습한 ep24000
(SHA-256 `82f7978b42d9d9e95adcc638a40ae85fb3736fd897ae39cb8aa8333be39cf23f`)이다. 두 정책을 모두
205 bars, seed42, deterministic/original, mixed motion, full 6--28 m, analytic detector에서 평가한다.
일반 성능은 U[0.3,1.5] m/s, §8.11에서 확인한 고속 병목은 fixed 1.5 m/s로 분리하며 총 2×2 cell,
cell당 2,049 requested episodes다.

각 조건에서 ep24000-minus-ep19100 capture가 **-3.0 pp 이하이고 독립 비율차 95% CI 상한도 0 미만**이면
material forgetting, +3.0 pp 이상이고 CI 하한도 0보다 크면 improvement, 나머지는 no-material-change로
판정한다. uniform만 저하하면 일반 curriculum 망각, fast1p5만 저하하면 high-speed-specific 망각,
둘 다 저하하지 않으면 ep24000의 고속 실패를 205-stage forgetting보다 구조적 정책/제어 한계로 분류한다.
async reset으로 cell은 episode-paired가 아니며 결과를 본 뒤 checkpoint, 조건, 표본수 또는 margin을
바꾸지 않는다. 실행기는 `eval_navrl_v2_ep19100_vs_ep24000_forgetting.sh`이고 새 학습은 없다.

### 8.13 망각검사 결과와 다음 단계 승인 (2026-08-03)

2×2 evaluation은 전 cell의 checkpoint/evaluator SHA, nonce, physics, seed, speed 및 episode accounting을
통과했다. uniform에서 ep19100→ep24000 capture/crash/bar-contact는
**67.79→72.44% / 31.87→25.07% / 31.67→24.29%**였고 capture 차이는 **+4.65 pp**
(95% CI +1.85..+7.45)다. fixed 1.5 m/s에서는 **64.10→67.35% / 35.61→30.75% /
34.93→29.97%**, capture 차이 **+3.25 pp**(95% CI +0.35..+6.16)다. 두 조건 모두 사전등록
`improvement`이고 material forgetting은 NO다.

따라서 205-stage replay/mixture는 다음 변경이 아니다. §8.11의 고속 성능 저하는 205 학습 중 망각이
아니라 속도 여유가 없는 상태의 위험 선택·경계 action/충돌 병목으로 유지한다. 1650 Ti 70-bar 선행
TTC A/B는 capture **+9.864 pp**, crash **-8.056 pp**로 gate를 통과했고, main/4GB 두 profile의
fixed-205 launcher preflight도 baseline/TTC 모두 PASS했다. §8.6의 다음 학습 허가 게이트를 충족한
것으로 판정한다.

다음 실행 순서는 main RTX 3070에서 `ARM=baseline PROFILE=main`을 먼저 1,000 epoch(4.096M samples)
학습하고 held-out 205-bar 평가를 동결한 뒤, 동일 예산의 `ARM=ttc PROFILE=main`을 실행·평가하는 것이다.
TTC는 같은-profile baseline 대비 capture +2.0 pp 이상과 crash -2.0 pp 이하를 동시에 만족해야 채택한다.
reflection/RMS regularization과 learned detector는 별도 후속 arm이며 이번 selector A/B에 섞지 않는다.

### 8.14 main fixed-205 baseline 완료와 평가 동결 규칙 (2026-08-03)

main baseline run `ppo_260803_1819_navrl_v2-ep24000-205bars-main-baseline-s1`은 ep24001--25000,
정확히 1,000 epoch/4,096,000 samples를 끝내고 `max_epochs`로 정상 종료했다. 밀도는 전 구간 205,
selector는 `cluster_sector`였다. 최종 checkpoint는
`last_gen_ppo_ep_25000_rew_29.188496.pth`, SHA-256은
`169ddcddb83c9d74df5c79252274660bc9c52e32d7d5144d325698e32b1d9b08`이다.

최적화 안정성은 통과했다. PPO KL 평균/최대는 **0.00244/0.00757**, immutable behavior-KL audit 최대는
**0.01236 < 0.04**, rollback·KL-skip·4축 raw OOB는 모두 0이다. 그러나 훈련 proxy의 첫 100→마지막
100 epoch capture는 **69.46→67.49%**, crash는 **28.14→30.61%**였다. 이 값은 epoch별 종료 표본이
작고 stochastic on-policy이므로 성능 판정이 아니라 “발산은 없지만 추가 epoch의 개선 증거도 없음”으로만
해석한다. 마지막 epoch의 66.67%도 18개 종료 episode에서 나온 값이라 publication/A-B 숫자로 금지한다.

다음 단계는 이 checkpoint를 205 bars, seed42, deterministic/original, mixed target,
U[0.3,1.5] m/s에서 최소 2,049 episode 평가해 baseline 결과를 먼저 동결하는 것이다. evaluator preflight는
통과했다. 이 결과 artifact가 생기기 전에는 TTC arm을 실행하지 않는다. 이후 TTC arm도 같은 4.096M
sample budget을 받고 동일 held-out 계약으로 평가한다. 채택 규칙은 사전등록대로 TTC-minus-baseline
capture **≥+2.0 pp**와 crash **≤-2.0 pp**를 동시에 요구한다.

### 8.15 남은 장기 실행 로드맵과 역할 분담

현재 navigation/control backbone은 **P5 후반--P6 초반**이다. 고밀도 analytic-detector 환경의 한계와
주요 failure mechanism은 수치화됐지만, 논문의 최종 주제인 learned RGB-D/LiDAR detector·tracker(P1--P2),
그 인지 오차를 포함한 end-to-end 비교(G1--G5), sim-to-real(P7)은 남아 있다. 따라서 “시뮬레이터 정책만”
기준으로는 후반부지만, 전체 논문 증거 기준으로는 중간 지점이다. 아래 단계는 앞 단계 gate를 통과해야만
다음 단계로 넘어가며 한 run에서 selector, action, reward, perception을 함께 바꾸지 않는다.

| 실행 단계 | 예상 범위 | 핵심 질문과 산출물 | 진입/종료 gate | 실패 시 분기 |
|---|---|---|---|---|
| R0 · main TTC A/B | 1일 내외 | fixed-205 baseline 평가 → TTC 4.096M 학습 → 동일 held-out 평가 | capture +2 pp 및 crash -2 pp 동시 | selector 축을 음성 결과로 닫고 R2 control-risk 진단 |
| R1 · 선택자 재현/한계 지도 | 1--3일 | 채택 후보를 seed43, bars 190/205/220, speed 0.3/0.9/1.5에서 확인 | 이득 방향 재현, 고속 bar-contact 악화 없음 | 과적합이면 TTC 채택 취소; density curriculum 재개 금지 |
| R2 · control-risk 단일축 | 3--7일 | frozen policy의 inference-only 속도/가속/yaw/clearance governor screen 뒤 단 하나만 학습 A/B | crash 감소가 capture/timeout 희생보다 큼 | reward 재튜닝 대신 look-ahead/action-shield 또는 정책 구조로 이동 |
| R3 · learned perception | 2--4주 | RGB-D와 LiDAR detector, association, Kalman track; GT actor 누출 0 | G1--G3: visible recall, 10 m median error ≤0.30 m, 1 s miss 재획득 ≥80% | PPO 금지; dataset/asset/calibration부터 보강 |
| R4 · temporal fusion policy | 2--4주 | frozen detector output으로 single-step/LSTM/17-token Transformer 비교 | G4--G5, analytic-oracle 대비 gap과 occlusion bin 분해 | 모델 확대 전 track age/confidence/latency 병목 수정 |
| R5 · 최종 robustness map | 1--2주 | density×speed×occlusion, CV/waypoint/circle, seed≥3, perturbation ablation | 조건별 ≥1,000 held-out episodes; capture/crash/timeout CI 보고 | 실패 cell을 perception/control/geometry로 분해해 한 축만 재시험 |
| R6 · sim-to-real/논문화 | 하드웨어 의존 2--6주 | 실기 latency·센서 noise·calibration, safety cage flight, 표·그림·재현 package | 안전 중단 규칙과 real-log 재현, 결과/코드 SHA 동결 | 실제 센서 로그 replay로 sim gap을 먼저 좁힘 |

R0 결과에 따른 바로 다음 결정은 두 갈래뿐이다.

- **TTC PASS:** TTC를 곧바로 최종 방법으로 선언하지 않고 R1의 다른 seed/속도/밀도에서 재현한다.
- **TTC FAIL:** 205에서 같은 표현을 더 오래 학습하지 않는다. 동결 checkpoint에 governor 후보를
  inference-only로 먼저 적용해 유망한 control mechanism 하나만 고른 뒤 R2 A/B를 한다.

Codex가 맡을 수 있는 범위는 launcher/preflight, 학습·평가 실행과 감시, TensorBoard/JSON 통계와 CI,
실패 원인 분해, 단일변수 코드 변경, 테스트, 사이트·WORKLOG·논문 표 초안까지다. 사용자가 매 epoch을
보거나 TensorBoard run을 직접 골라 관리할 필요는 없다. 사용자 입력이 실제로 필요한 것은 아래 네 가지다.

1. RTX 3070/1650 Ti를 언제 몇 시간 사용할 수 있는지와 전기·시간 예산;
2. 두 머신 사이 checkpoint/result 이관 또는 원격 접근 가능 여부;
3. 실기 기체, 카메라/LiDAR 모델·intrinsic/extrinsic·onboard compute와 안전 비행 가능 조건;
4. 목표 학회/논문 마감일과 “고밀도 제어” 대 “learned perception” 중 우선순위.

이 네 정보가 없더라도 R0--R2와 synthetic detector dataset 준비까지는 진행 가능하다. 반대로 R3 이후의
최종 모델 크기와 R6 일정은 실제 센서/컴퓨트 사양 없이 확정하지 않는다.

### 8.16 main baseline held-out 결과와 TTC 이중 판정 (2026-08-04)

ep25000 baseline은 canonical evaluator의 checkpoint/snapshot/result/receipt/log SHA, seed42, 205 bars,
deterministic/original, full 6--28 m, U[0.3,1.5], main/base_sim/128 env 계약을 통과했다. 2,049 episodes에서
capture/crash/timeout은 **69.50/28.99/1.51%**, bar contact는 **27.82%**였다. capture Wilson 95% CI는
**67.47--71.45%**다.

같은 조건의 동결 ep24000 결과 72.44/25.07/2.49%와 독립 비율로 비교하면 ep25000-minus-ep24000은
capture **-2.94 pp** (95% CI **-5.72..-0.16**), crash **+3.92 pp**
(**+1.20..+6.63**), timeout **-0.97 pp** (**-1.83..-0.12**)다. bar contact는 +3.53 pp,
lateral edge98은 3.37→5.03%, vertical edge98은 0.01→3.20%로 증가했다. KL/rollback은 정상이었으므로
급성 PPO collapse가 아니라 고정 밀도에서의 **느린 control/action drift**다. KL guard는 큰 update를 막지만
held-out 성능의 완만한 악화를 대신 검출하지 못한다는 음성 결과로 보존한다.

TTC 결과를 보기 전에 판정을 두 층으로 고정한다.

1. **primary representation-effect gate(기존 사전등록 유지):** TTC capture ≥**71.50%**이면서 crash
   ≤**26.99%**. 이는 같은 4.096M-sample baseline 대비 +2/-2 pp이며 selector 효과를 판정한다.
2. **canonical replacement floor(추가 해석 규칙):** TTC capture ≥**72.44%**이면서 crash ≤**25.07%**일
   때만 ep24000 deployment artifact를 직접 교체할 후보로 부른다. primary만 통과하면 “TTC가 continued
   cluster drift를 완화했다”는 인과효과는 인정하지만 ep24000보다 좋은 최종 정책이라고 주장하지 않는다.

어느 경우든 한 seed 결과로 최종 채택하지 않고 §8.15 R1에서 seed43과 speed/density 재현을 수행한다.
TTC launcher는 main arm 실행 전에 baseline JSON뿐 아니라 receipt, snapshot, log bytes와 조건을 다시
검증하도록 하드닝한다.

### 8.17 main TTC 평가 결과: operational mode FAIL, pure-ranking 판정 불가 (2026-08-05)

TTC run `ppo_260804_0813_navrl_v2-ep24000-205bars-main-ttc-s1`은 ep24001--25000,
4,096,000 samples를 `max_epochs`로 정상 완료했다. final checkpoint SHA-256은
`14e4c72a744c9bedc2d07556e5aebdbef21a184c9f9b8239bc1a23d45e20823e`다. PPO KL 최대 0.00731,
behavior-KL audit 최대 0.01189, rollback/OOB 0으로 최적화 발산은 없다. training proxy는 첫100→끝100
capture 55.77→65.75%, crash 43.84→33.59%로 회복됐지만 판정에는 쓰지 않는다.

canonical held-out는 checkpoint/snapshot/result/receipt/log SHA와 205 bars, seed42,
deterministic/original, U[0.3,1.5], main/base_sim/128 env 계약을 통과했고 실제 2,051 episodes를 얻었다.

| arm | capture | crash | timeout | bar contact | lateral edge98 | speed |
|---|---:|---:|---:|---:|---:|---:|
| cluster baseline | 69.50% | 28.99% | 1.51% | 27.82% | 5.03% | 2.436 m/s |
| TTC mode | **70.21%** | **29.50%** | **0.29%** | **28.18%** | **7.17%** | 2.570 m/s |
| TTC−baseline | **+0.71 pp** | **+0.51 pp** | -1.22 pp | +0.36 pp | +2.14 pp | +0.134 m/s |

capture 차이 95% CI는 **-2.10..+3.52 pp**, crash는 **-2.28..+3.29 pp**로 0을 포함한다. timeout
감소만 -1.22 pp(95% CI -1.80..-0.64)로 명확하지만 crash로 전환됐고, primary +2/-2 gate와 ep24000
replacement floor를 모두 FAIL했다. ep24000 원본 대비 TTC는 capture -2.23 pp, crash **+4.42 pp**
(95% CI +1.70..+7.15)다. 따라서 **현재 deployable TTC mode는 채택하지 않고 추가 epoch도 금지**한다.

사후 provenance 감사에서 cluster arm의 effective candidate FOV는 240°, TTC arm은 의도적으로 360°임을
확인했다(`obstacle_selector_provenance`). 8 token budget에서 고밀도 360° 후보는 후방 위협을 포함하는 대신
전방 경로 표현을 희석할 수 있다. 이는 plausible mechanism이지만 이번 rollout만으로 입증하지 않는다.
따라서 결과의 정확한 해석은 다음 둘로 분리한다.

- **operational representation bundle:** FAIL — 현재 코드 그대로는 205 bars에 도움 없음;
- **TTC ranking 자체:** 판정 불가 — candidate FOV가 함께 바뀌어 pure ranking causal effect가 아님.

동일 mode를 더 학습하지 않는다. TTC 개념을 한 번 더 시험한다면 selector와 candidate FOV를 독립 설정해
`cluster240` 대 `ttc240`을 먼저 1650 Ti에서 screen해야 한다. 그렇지 않으면 §8.15의 원래 FAIL 분기대로
R2 control-risk inference-only screen으로 이동한다. 어느 쪽도 이번 결과와 동시에 구현하지 않는다.

### 8.18 R2 speed-risk governor 사전등록 (2026-08-05, 결과 확인 전)

TTC FAIL 분기와 사용자 승인에 따라 frozen canonical ep24000 `cluster_sector` 정책에서 control-risk를
먼저 분리한다. 당시 v2 episode는 설정상 600 RL step=60 s였지만 후속 감사에서 `>600` 비교 때문에
실제로 action 601에서 timeout된 것으로 확인됐다. main TTC crashdiag의 bar contact 평균은 74--92
step(7.4--9.2 s), held-out timeout은 0.29%뿐이므로 horizon 부족 가설은 기각한다. actor action은 0속도를
이미 표현할 수 있지만 reward가 closing speed 1.0 + ego-progress heuristic 1.0 + time cost -0.05인 반면 safety는
속도/제동거리와 결합하지 않는다. 따라서 600-step·reward·policy weight를 동결하고 실행 command만
sensor-derived LiDAR clearance로 감속하는 단일축 실험을 수행한다.

공통 조건은 ep24000 SHA `82f7978b42d…`, 205 bars, seed42, deterministic/original, mixed target,
U[0.3,1.5], main/base_sim/128 env, 최소 2,049 episodes다. 비교할 다섯 조건은 다음과 같다.

1. `off`: byte-identical command baseline;
2. `fixed2p0`: body-frame horizontal command norm을 2.0 m/s로만 제한;
3. `fixed1p5`: 동일 norm을 1.5 m/s로 제한;
4. `clearance`: 명령 방향의 sensor-only swept corridor에서 surface clearance가 3.0→0.45 m로 줄 때
   허용속도를 free cap→0으로 선형 감소;
5. `ttc`: 같은 clearance에서 `(clearance-0.45)/tau`로 허용속도를 제한. `tau`는 별도 braking probe의
   p95 stop time +0.2 s, 최소 0.8 s로 **held-out 전에** 고정한다.

LiDAR path corridor half-width는 0.45 m이고 target semantic return은 장애물에서 제외한다. command 방향은
바꾸지 않고 XY norm만 `scale≤1`로 축소한다. 각 result는 governor mode/parameter, intervention rate,
requested/executed speed, clearance/TTC/stopping margin, contact 직전 speed/command/step을 기록한다.

**1차 GO gate:** baseline 대비 crash **≥3.0 pp 감소**, capture 손실 **≤1.0 pp**, timeout **≤5%**를 모두
만족해야 한다. adaptive(`clearance`/`ttc`) 후보가 통과하지 않으면 고정 cap이 좋아도 adaptive governor
성공으로 바꾸어 해석하지 않는다. 둘 다 통과하면 capture가 높은 후보, 0.5 pp 이내 동률이면 crash가 낮은
후보를 선택한다. 이 point gate는 screen용이며 최종 주장은 seed43/속도축 재현과 CI 방향 일치를 요구한다.
PASS일 때만 선택된 한 mode를 ep24000에서 1,000 epoch/4.096M samples로 학습하고, 기존 fixed-205
cluster baseline과 같은 held-out 계약으로 비교한다. FAIL이면 reward·episode length를 즉석 변경하지 않고
R2를 음성 결과로 닫은 뒤 look-ahead/action-shield 구조를 별도 사전등록한다.

**2차 adaptation gate(적응형 screen 결과 확인 전 고정):** 1,000-epoch final은 같은 governor를 붙인
학습 전 ep24000보다 seed42 uniform capture/crash/timeout 각각 **-1.0/+1.0/+1.0 pp 이내**여야 한다.
즉 안전층이 이미 만든 이득을 학습이 훼손하지 않는 non-inferiority가 우선이다. 그 조건 안에서 capture가
**+1.0 pp 이상**이거나 intervention rate가 **-5.0 pp 이상**이면 adaptation 자체도 유용하다고 판정한다.
이 추가 유용성 조건을 못 넘으면 학습 전 governor를 최종 후보로 유지하고, 1,000 epoch가 불필요했다는
음성 결과로 기록한다. 최종 선택 정책은 seed43 uniform과 seed42 fixed 0.3/0.9/1.5 m/s를 평가하며,
각 축에서 canonical off ep24000 대비 capture delta가 음수가 아니고 crash delta가 음수인 방향을 요구한다.
개별 cell의 표본오차도 함께 제시하며, 이 replication이 어긋나면 일반화 성공으로 주장하지 않는다.

### 8.19 R2 결과와 R2b minimum-intervention risk-cap 사전등록 (2026-08-05)

유효한 sensor-only/수직각 교정 구현으로 다시 수행한 R2 5-cell screen은 결과를 보기 전에 정한 gate로
adaptive GO가 없었다. canonical off는
capture/crash/timeout 72.44/25.07/2.49%를 정확히 재현했다. fixed 2.0은 78.53/16.06/5.42%로
충돌 가설을 강하게 지지했지만 timeout 상한을 0.42 pp 넘겼다. fixed 1.5는 74.65/16.82/8.53%로
더 느려도 crash가 더 줄지 않고 timeout만 늘었다. clearance와 TTC는 각각
69.11/14.30/16.59%, 69.59/6.83/23.57%였다. 둘은 near-stop을 각각 38.60/42.04% step에서
일으켰다. 따라서 complete-stop형 scalar governor는
deadlock을 만든다는 음성 결과이며, 이 두 mode를 학습하지 않는다.

다음 R2b는 위 결과에서 후보를 여러 개 골라 맞추지 않고 **단 하나**만 새 holdout seed 44에서 검사한다.
이는 nominal RL action을 가능한 한 작게 수정하는 safety-filter/action-projection 원칙과, 속도·동역학을
장애물 거리와 함께 다루는 barrier 계열의 구조를 현재 코드에 가장 작게 옮긴 진단이다
([Kochdumper et al., 2022](https://arxiv.org/abs/2210.10691),
[Harms et al., 2025](https://arxiv.org/abs/2502.04101)). 정식 CBF/QP 안전보장을 주장하지 않으며,
이번 후보는 계산량이 작은 scalar ablation이다.

- `riskcap`: command-direction clearance **≤3.0 m**이면 XY norm 상한을 **2.0 m/s**로 두고,
  3.0--5.0 m에서 원래 free cap 3.535 m/s까지 선형 해제한다. 요청속도가 상한보다 낮으면 건드리지
  않으며, 0속도로 강제하지 않는다.
- 3.0 m activation은 실측 p10 감속도 2.961 m/s²에서 최대 요청 norm 3.536 m/s의
  `reaction 0.1 s + braking + margin 0.45 m = 2.92 m`를 위로 반올림한 값이다.
- 공통: frozen ep24000 SHA `82f7978b42d…`, 205 bars, seed44, deterministic/original,
  U[0.3,1.5], main/base_sim/128 env, 2,049 requested episodes. 새 `off`도 같은 실행에서 다시 측정한다.
- **GO gate:** riskcap-off crash ≤-3.0 pp, capture ≥-1.0 pp, riskcap timeout ≤5.0%, near-stop ≤5.0%.
  모두 만족할 때만 §8.18의 1,000-epoch adaptation과 2차 gate를 수행한다. FAIL이면 scalar speed
  layer를 닫고 다음은 방향까지 최소 투영하는 look-ahead/CBF-QP 계열을 별도 실험으로 설계한다.

R2b가 GO일 경우 seed44는 후보 선택에 이미 사용됐으므로 adaptation 최종 primary에 재사용하지 않는다.
학습 전 `off`/학습 전 `riskcap`/1,000-epoch `riskcap`을 모두 새 **seed45 uniform**에서 비교해
mechanism replication과 adaptation 효과를 분리한다. 여기서 §8.18의 non-inferiority/additional-utility
gate로 학습 전/후 중 winner를 고른 뒤, 새 **seed46 fixed 0.3/0.9/1.5 m/s**에서 canonical off와
winner를 각각 2,049 requested episodes로 평가한다. seed42 R2와 seed44 R2b는 개발·선택 자료이고,
seed45/46만 최종 일반화 판정에 사용한다.

### 8.20 R2 semantic-leak 감사와 재실행 규칙 (2026-08-05, 완료)

R2b 적응 run 도중 코드 감사를 수행해 speed governor가 표적 LiDAR return을 제외할 때
`segmentation_pixels == 50`을 직접 읽는 것을 발견했다. actor의 structured obstacle path는 이미 카메라
검출 bearing/range와 LiDAR range 일치(각도 ±15°, 거리 ±0.55 m)만으로 같은 연관을 수행하는데,
governor만 oracle semantic mask를 별도로 사용한 것이다. 이는 §1/§3의 actor GT-semantic 금지와
충돌하므로 이전 R2/R2b 결과와 ep24000→24334 중단 run은 **탐색 자료일 뿐 최종 증거가 아니다**.
삭제하지 않고 `tensorboard_archive/2026-08-05_invalid_semantic_governor/`에 격리한다.

수정은 새 추정기를 추가하지 않는다. perception front-end가 obstacle token을 만들 때 계산한
`last_target_like` mask를 governor가 그대로 재사용하고, governor가 활성인데 perception front-end가
없으면 fail-closed한다. bulk JSON/checkpoint/evaluation receipt에
`camera_lidar_association` provenance를 기록하며 요약기가 이를 재검증한다. `off`는 command를 건드리지
않으므로 기존 canonical 수치의 exact 재현을 요구한다. `fixed`는 clearance와 무관하므로 기존 결과가
재현되는지 확인하고, `clearance`/`ttc`/`riskcap`은 새 결과만 판정에 사용한다.

첫 corrected 재실행의 adaptive 셀 도중 두 번째 코드 감사를 수행해 Warp LiDAR tensor의 실제 수직 행
순서 `+20°→-10°`를 clearance projection이 `-10°→+20°`로 뒤집어 적용한 것도 발견했다. off/fixed
command에는 영향이 없지만 adaptive/riskcap의 거리 경계에는 영향을 주므로 해당 partial 재실행도 최종
증거에서 제외한다. 기본 projection을 실제 행 순서로 고치고 이를 비대칭 수직각 단위 테스트로 고정한
뒤 아래 순서를 처음부터 다시 수행한다.

재실행 순서와 gate는 결과를 보기 전에 다음처럼 고정한다.

1. seed42 original 5-cell R2를 2,049 requested episodes/cell로 전부 재실행한다. §8.18의 adaptive gate는
   변경하지 않는다.
2. complete-stop adaptive가 다시 FAIL하고 fixed-2.0 positive control이 유지되면, 제동 probe로 이미
   고정한 단일 `riskcap`(2.0 m/s ≤3 m, 3--5 m release)을 seed44에서 재실행한다. §8.19 GO gate를
   변경하지 않는다.
3. corrected seed44가 GO일 때만 frozen ep24000에서 정확히 1,000 epoch/4.096M samples를 다시 학습한다.
   중단된 semantic run checkpoint는 재개하지 않는다.
4. 최종 seed45/46 선택·일반화 gate는 §8.19 그대로 유지한다. corrected 결과가 FAIL하면 파라미터를
   사후조정하지 않고 R2를 닫는다.

### 8.21 R2b 최종 결과: non-stopping riskcap 채택, control 단계 동결 (2026-08-05)

위 재실행 순서를 한 번도 바꾸지 않고 완료했다. seed44 corrected screen에서 off는 2,050 episodes 기준
capture/crash/timeout **72.83/24.63/2.54%**, 단일 riskcap은 2,049 episodes에서
**79.55/17.62/2.83%**였다. 차이는 capture **+6.72 pp**(95% CI +4.12..+9.32), crash
**-7.02 pp**(-9.51..-4.53)이며 intervention 28.74%, near-stop 0%로 §8.19 GO gate를 모두 통과했다.

승인된 run `ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1`은 frozen ep24000에서
ep25000까지 정확히 1,000 epoch, 32×128×1,000=**4.096M samples**, LR 5e-6로 정상 종료했다.
일반 checkpoint는 `last_gen_ppo_ep_25000_rew_39.742134.pth`, SHA-256
`f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40`이다. 학습 proxy 전체 27,632
episodes의 capture/crash/timeout은 **77.67/18.50/3.83%**, 첫100→끝100 epoch는
78.70/18.61/2.69→77.24/18.71/4.06%로 plateau였다. PPO KL 최대 0.00906, immutable behavior-KL
최대 0.01096, rollback/OOB 0이어서 발산은 기각한다. training tail이 아니라 아래 독립 held-out으로
적응 가치를 판정했다.

새 seed45 uniform 최종 결과는 다음과 같다.

| policy | n | capture | crash | timeout | intervention |
|---|---:|---:|---:|---:|---:|
| ep24000 off | 2,049 | 70.03% | 27.87% | 2.10% | 0% |
| ep24000 + riskcap | 2,050 | 78.20% | 17.80% | 4.00% | 27.78% |
| ep25000 + riskcap | 2,049 | **81.94%** | **15.67%** | **2.39%** | 26.30% |

sensor-only mechanism 복제는 off 대비 capture **+8.16 pp**(95% CI +5.49..+10.83), crash
**-10.06 pp**(-12.61..-7.51)로 PASS했다. 적응은 source+riskcap 대비 capture **+3.75 pp**
(+1.30..+6.19), crash -2.14 pp(-4.42..+0.15), timeout -1.61 pp(-2.68..-0.53)여서 non-inferior와
additional-utility를 모두 만족했고 ep25000을 winner로 선택했다.

새 seed46 고정속도 일반화도 사전등록 방향을 3/3 통과했다.

| target speed | off capture/crash | winner capture/crash | Δcapture | Δcrash |
|---:|---:|---:|---:|---:|
| 0.3 m/s | 71.90/24.88% | 81.84/15.18% | +9.94 pp | -9.70 pp |
| 0.9 m/s | 71.89/25.77% | 80.77/16.59% | +8.88 pp | -9.18 pp |
| 1.5 m/s | 67.98/30.31% | 75.51/22.29% | +7.53 pp | -8.02 pp |

따라서 **ep25000 policy + 고정된 riskcap을 현재 navigation/control candidate로 동결**한다. 이 결과는
정식 CBF/QP 안전보장이나 learned detector 성능을 뜻하지 않는다. 특히 fixed 1.5 m/s에서도 bar contact가
20.78% 남고 lateral high-action 80% 비율 70.16%, signed-y +0.766으로 chirality가 지속된다. 또한 모든
최종 셀은 analytic detector와 한 아레나/205 bars에서 평가했으므로 sensor association이 흔들리는 learned
perception, 다른 obstacle realization, 실제 동역학 지연은 아직 미검증이다.

다음 단계는 R3 learned-detector robustness다. riskcap 파라미터와 ep25000 weight를 더 튜닝하지 않고,
동일 seed/속도 계약에서 analytic→learned detector 전환과 detection dropout/지연/거리오차를 각각 분리한다.
perception gate를 통과한 뒤에만 R4 temporal fusion으로 간다. 실패하면 navigation PPO를 연장하지 말고
camera–LiDAR association과 uncertainty-aware release를 고친다. 전체 원자료와 CI는
`results/navrl_v2_riskcap_postadapt/summary.{md,json}`을 canonical로 사용한다.

### 8.22 최종 계약 감사와 재개 gate (2026-08-10)

> **HISTORICAL / SUPERSEDED:** 이 절의 실행 순서는 2026-08-10 당시 계획을 보존한 기록이다.
> 현재 실행 authority는 바로 뒤 §8.23의 `P0 → P1a/P1b/P1c → P2 → P3` fail-closed 순서이며,
> 상태 확인은 `WORKLOG.md`, 실제 명령은 `README.md`와 `OPERATIONS.md`를 따른다.

§8.21의 policy/control 선택은 동결하되 publication 수치 계약은 재측정한다. frozen checkpoint는
legacy 601-action horizon과 누락된 rl_games `time_outs` bootstrap 아래 학습됐으며, 과거 평가 receipt는
imported runtime source를 보존하지 않았다. 현재 source는 정확히 action 600에서 종료하고 두 timeout key를
동시에 제공하며, checkpoint/detector/runtime source/Python environment를 schema-v2 receipt로 묶는다.

또한 `max_velocity=2.5`는 x/y 축별 limit(XY request norm 최대 3.54 m/s)이고 moving-target progress는
엄밀 PBRS가 아닌 ego-motion heuristic이다. actor z는 altitude PI에 덮여 직접 actuator authority가 없지만
raw z가 다음 `prev_action`에 남아 간접 state channel일 수 있으므로 단순 dead dimension으로 삭제하지 않는다.

당시 실행 순서는 `docs/NEXT_WEEK_HANDOFF_2026-08-10.md`를 기준으로 동일 미사용 seed의
governor/adaptation A/B/C 15 cells → ID speed interaction 16 cells/2 seed → detector offline dataset/loss/
calibration gate를 계획했다. 이 순서와 새 PPO 금지 지시는 §8.23으로 supersede됐으며 현재 명령이 아니다.
