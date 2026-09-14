# Codex independent review — 2026-08-10

검수 대상: `docs/review_brief_2026-08-10.md`, 커밋 `715dc76..d9ee124`.

범위는 frozen PPO의 평가·해석·대시보드 provenance와 학습 의미 계약이다. PPO 재학습은 검수 범위와
권고안에서 제외한다. 아래 통계는 저장된 aggregate count에서 다시 계산했으며, 서로 다른 arm은 episode-level
pair가 보존되지 않았으므로 독립 이항표본으로 취급했다.

## 판정 요약

| 항목 | 판정 | 핵심 이유 |
|---|---|---|
| A. P3 capture-time pose | **조건부 동의** | 구현·인덱스·수학은 맞다. 단, 실기 주장은 timestamp 동기화된 pose history라는 전제를 명시해야 한다. |
| B. 밀도 곡선 귀속 | **반박(현 서술 불충분)** | checkpoint와 governor 외에 seed 42→47, evaluator/source revision도 다르다. 현재 Δ는 method gap이 아니다. |
| C. 속도×밀도 상호작용 | **반박** | 학습범위 ≤205의 interaction LR test `p=0.337`; 강한 신호는 OOD 220이 만든다. |
| D. dropout/H4 메커니즘 | **부분 동의** | H2는 두 seed pooled로 실재하는 작은 효과지만 H4 69% 분해는 단일 seed·비유의·비가산 개입이다. |
| E. dashboard provenance | **대체로 동의** | pre-chirality 9개 분류는 맞다. post-fix legacy pilot은 유효한 archive지만 headline fallback에서는 제외해야 한다. |
| F. 종료/value bootstrap | **중대 결함** | 600-step 설정이 601 actions였고 rl_games가 요구하는 `time_outs` key가 없어 truncation bootstrap이 작동하지 않았다. |
| G. action/reward 표기 | **정정 필요** | 2.5 m/s는 축별 한계(XY norm 3.54), z output은 PI에 덮이지만 raw z가 다음 `prev_action`에 남는 간접 상태 채널, moving-target progress는 엄밀 PBRS가 아니다. |

## 0. 최종 코드 감사에서 발견한 중대 계약 결함

### 0.1 time-limit bootstrap이 실제로는 꺼져 있었다

YAML은 `value_bootstrap: True`지만 rl_games는 환경 info의 정확한 key `time_outs`만 읽는다. 기존
환경은 사람용 `timeouts`만 내보냈다. 따라서 rl_games가 time limit에 적용하는 rollout value 보정이
보태지지 않았고, critic은 time limit을 MDP terminal처럼 학습했다. timeout 비율이 낮은 셀에서도 value
bias가 rollout 전체 advantage에 전파될 수 있어 “영향이 작다”고 가정하지 않는다.

현재 source는 `timeouts`와 `time_outs`를 같은 boolean tensor로 함께 내보낸다. 전자는 기존 로깅
호환성, 후자는 rl_games 계약이다. checkpoint state에도 bootstrap key와 reward/action semantics를 기록한다.

### 0.2 episode horizon은 600이 아니라 601 actions였다

기존 종료식은 증가된 `sim_steps > episode_len_steps`였다. 설정 600이면 action 601 이후 truncation된다.
과거 결과 JSON의 `speed_governor.outcome_steps.timeout`을 확인하면 timeout summary가 모두 정확히 601이다.
현재 source는 `>=`로 고쳐 action 600에서 끝내며, evaluator는 timeout이 존재하는 모든 셀에서
count와 mean/p10/p50/p90가 600인지 실측 검증한다.

이 때문에 기존 601 결과와 새 600 결과는 같은 figure의 arm으로 섞지 않는다. frozen policy의 학습
provenance도 “601 + no time-limit bootstrap”로 공개하며 소급해서 고쳤다고 쓰지 않는다.

### 0.3 action과 reward의 실제 의미

- `max_velocity=2.5`는 vector norm이 아니라 x/y 각 축 clip이다. 수평 요청 norm은 최대
  `sqrt(2)·2.5 = 3.54 m/s`이고 저장 평가의 평균 requested speed도 약 2.96 m/s다.
- actor는 4-D `(x,y,z,yaw)`를 내지만 z command는 altitude PI가 덮어쓴다. 다만 raw z는 다음 관측의
  `prev_action`에 남아 직접 actuator가 아닌 간접 policy-state channel로 작동할 수 있다. 기존 checkpoint
  호환 때문에 이번에는 차원을 제거하지 않았다. 다음 fresh PPO의 3-D actor ablation은 이 메모리 채널을
  제거할지 다른 관측으로 대체할지까지 함께 통제해야 한다.
- progress 식은 `||drone_prev−target_new|| − 0.99||drone_new−target_new||`이다. 정적 표적에서는 PBRS
  대수와 같지만 이동 표적에서는 potential이 future target에 재고정되는 ego-motion heuristic이다.
  policy-invariance theorem을 주장하지 않는다.

## 1. 발표·논문 전에 반드시 내릴 주장

### 1.1 “속도는 밀도와 곱해져서만 유의미해진다”는 현재 자료로 성립하지 않는다

`results/navrl_v2_density_speed_map/summary.json`의 저장 count로 0.3→1.5 m/s 대비를 재계산했다.

| bars | capture Δ | 독립표본 95% CI | p-value |
|---:|---:|---:|---:|
| 130 | −0.878 pp | [−2.879, +1.122] | 0.389 |
| 160 | −1.569 pp | [−3.735, +0.598] | 0.156 |
| 190 | −1.854 pp | [−4.220, +0.512] | 0.125 |
| 205 | −3.064 pp | [−5.526, −0.602] | 0.0147 |
| 220 (OOD) | −5.989 pp | [−8.636, −3.342] | 9.2e−6 |

그러나 질문은 각 밀도에서 속도 효과가 있는지가 아니라 **속도 효과가 밀도에 따라 달라지는가**다.
aggregate-binomial logistic LR test 결과는 다음과 같다.

- 학습범위 130/160/190/205: continuous density×speed `p=0.337`; density별 categorical
  interaction omnibus `p=0.817`.
- OOD 220 포함: continuous interaction `p=0.0224`, categorical omnibus `p=0.139`.

따라서 현재 허용 가능한 문장은 다음 정도다.

> 205막대에서 1.5 m/s는 0.3 m/s보다 3.06 pp 낮았고(명목 95% CI −5.53..−0.60),
> OOD 220막대에서는 격차가 5.99 pp였다. 밀도에 따라 속도 비용이 증가한다는 상호작용은
> 학습범위에서 확인되지 않았으며 추가 seed 평가가 필요하다.

205 셀도 다섯 밀도 사후 대비 중 하나라 multiplicity 보정 뒤 확정 결과로 쓰면 안 된다. 0.7 m/s가
대부분의 밀도에서 0.3 m/s보다 높은 비단조성은 “회피 중 표적 이동”이라는 서사를 뒷받침하지 않는다.
trajectory mediation 증거가 없으므로 그 설명은 가설로만 남긴다.

v1/v2의 endpoint secant slope `2.98 vs 2.02 pp` 비교도 삭제한다. 서로 다른 arena, 배치 규칙,
정책, 밀도 범위의 비선형/절벽형 곡선에서 두 끝점을 직선으로 이은 수치는 같은 물리량이 아니다.
보존 가능한 교차버전 문장은 “각자의 측정 격자 안에서 밀도 주효과가 속도 주효과보다 컸다”뿐이다.

### 1.2 밀도 곡선 Δ에는 세 번째 confound와 provenance 공백이 있다

현재 비교는 다음이 동시에 다르다.

1. checkpoint: ep24000 → ep25000
2. governor: off → riskcap
3. held-out seed: **42 → 47**

또한 evaluator SHA가 `5e70...` → `ba6a...`로 다르다. receipt는 checkpoint와 최상위 shell
script만 해시하며, import되는 `navrl_task.py`, `navrl_perception.py`, config와 Python dependency
manifest는 고정하지 않는다. 옛 evaluator 바이트의 snapshot도 저장하지 않아 현재 checkout만으로
source-tree 동일성을 입증할 수 없다.

따라서 “현재 후보와 이전 측정의 관측 격차가 고밀도에서 더 컸다”는 **기술적 관찰**은 가능하지만,
governor/적응 효과 또는 method improvement로 귀속하면 안 된다. `WORKLOG.md`의 “riskcap 이득”,
`update_status_snapshot.py`의 “gain grows with density”, dashboard의 `ep24000_capture`를 governor
기여로 읽히게 하는 표현은 정정 대상이다.

205막대 seed45의 A/B/C는 같은 seed라 해당 한 점의 순차 분해로 유효하다.

- ep24000/off 70.034%
- ep24000/riskcap 78.195%: governor 단계 +8.161 pp
- ep25000/riskcap 81.942%: adaptation 단계 +3.747 pp

이는 205막대에서의 순차 차이이며 다른 밀도로 외삽할 수 없다. 현재 seed47 C arm을 재사용하려면
각 밀도에서 ep24000/off와 ep24000/riskcap을 **둘 다 seed47로** 재평가해야 한다(10 cells).
논문용 새 seed를 쓰려면 A/B/C 모두 같은 미사용 seed로 15 cells가 가장 깨끗하다. ep25000/off는
governor×adaptation interaction까지 묻고 싶을 때만 필요한 네 번째 arm이다.

### 1.3 `−13.9 pp learned detector`는 검출기 계열의 일반 한계가 아니다

해당 수치는 같은 PPO/seed에서 이 artifact를 바꿨을 때의 실제 성능 차이로는 유효하다. 그러나 현재
artifact는 다음 이유로 “learned detector”의 대표 시도라고 보기 어렵다.

- 모델은 4채널 입력의 **1×1 logistic pixel classifier** 한 층이다.
- 4096-frame 학습셋의 positive pixel rate는 **0.000536(0.054%)**인데 unweighted BCE를 쓴다.
- target은 매 수집 step에서 기체 정면 2–12 m에 강제 배치된다. 전체 bearing/range, target-absent,
  partial occlusion을 포괄하는 detector split이 아니다.
- held-out pixel/detection precision-recall, range-bin recall, calibration 결과가 없다.
- 학습된 depth weight가 음수라 threshold 0.55에서 순수 red target의 이론적 검출 범위가 대략
  14 m에서 잘린다. analytic bootstrap은 같은 방식으로 거리에 따라 약해지지 않는다.

그러므로 `−13.9 pp`는 “현 detector artifact가 analytic bootstrap보다 13.9 pp 낮다”로 쓰고,
“learned perception이 구조적으로 13.9 pp 병목”이라고 일반화하지 않는다. 다음 우선순위가 detector인
것은 맞지만, navigation 평가를 반복하기 전에 **offline dataset/loss/calibration gate**부터 고쳐야 한다.

## 2. 조건부로 유지 가능한 주장

### 2.1 P3 capture-time pose는 올바른 모델이다

코드 감사 결과 measurement와 pose는 하나의 `write_idx`에 저장되고 하나의 `read_idx`로 읽힌다.
buffer 미충전 시 `visible=False`가 correction을 막으며, τ=0은 buffer 경로 전에 return해 현재 pose를
그대로 쓴다. 동일 시각 pose가 renderer와 `observe()` 양쪽에 전달되는 것도 확인했다.

focused test 33개가 모두 통과했고, static world target에서 corrected residual `<1e−5`, naive error
`>0.2 m`, 2-step index와 startup gate를 검증한다.

취득 시각의 sensor-frame measurement를 그 시각의 pose로 world transform하는 것은 timestamped
robotics pipeline과 맞는다. ROS 2 tf2도 source frame을 평가할 `source_time`을 명시하는 API를
제공한다: <https://docs.ros.org/en/ros2_documentation/kilted/Tutorials/Intermediate/Tf2/Time-Travel-With-Tf2-Cpp.html>.

단, 논문 문장은 다음 전제를 붙여야 한다.

> P3 결과는 정확히 timestamp된 detection과 capture-time odometry/pose history를 이용하는 조건이다.

실기에서 pose timestamp offset, clock skew, pose interpolation error가 있으면 2.5 pp가 그대로 재현된다고
보장할 수 없다. P3가 target motion을 현재 시각으로 예측하는 것도 아니다(P0가 별도). 그러므로 R3의
`−42.7 pp`를 모델링 결함으로 supersede하는 것은 타당하지만 “latency가 보편적으로 benign”이라고 쓰지
말고 “정확한 timestamp/ego-motion correction 뒤 0.1 s perception delay의 잔차가 −2.5 pp”라고 쓴다.

테스트의 작은 공백은 실제 config class의 default-ON과 full-observation τ=0 on/off bit identity를 직접
묶은 회귀가 없다는 점이다. 코드는 산술상 no-op이므로 결론을 뒤집지는 않지만 다음 변경 때 보강할 수 있다.

### 2.2 H2 채널은 작지만 재현된다; H4의 69% 해석은 exploratory다

H2 association-off capture 차이는 다음과 같다.

- seed47: +3.416 pp, 95% CI [+0.600, +6.232], p=0.0174
- seed51: +3.088 pp, 95% CI [+0.295, +5.882], p=0.0302
- inverse-variance pooled: **+3.251 pp**, 95% CI [+1.268, +5.234], p=0.00131

따라서 “LiDAR association 경로에 작고 재현되는 효과가 있다”는 exploratory mechanism evidence는
정당하다. 다만 preregistered adoption gate 4 pp를 넘지 않았으므로 미채택 판정도 그대로 유지한다.

H4 silent-correct는 +2.357 pp, 95% CI [−0.473, +5.188], p=0.103으로 단일 seed에서 0을 포함한다.
`2.36 / 3.41 = 69%`는 **seed47에서 H2 intervention difference 중 차지한 비율**일 뿐이며 전체 dropout
loss 회수율은 18.6%다. H4와 H2는 중첩된 비선형 intervention이므로 남은 31%를 상태 보정의 독립 기여로
빼는 가산 분해는 성립하지 않는다. “합이 맞는다”는 산술 정의이지 별도 검증이 아니다.

코드상 LiDAR bearing/z가 tracker prediction에서 만들어지고, normal correction은 age/visibility를
갱신하며 silent arm은 state correction만 남기는 것은 사실이다. 따라서 올바른 표현은 다음이다.

> seed47 H4는 discrete seen/age 경로가 H2 효과의 일부일 가능성을 시사했지만, 단일-seed CI가 0을
> 포함해 메커니즘 비율은 확정되지 않았다.

이 가지는 더 파지 않고 종료해도 된다. detector의 직접 손실이 더 크고, H2는 adoption gate 미달이다.

## 3. Dashboard 판정

`_PRE_CHIRALITY_CURVES`의 9개는 저장 mtime/체크포인트가 모두 2026-07-29 sensor fix 이전이라 분류가
맞다. 삭제하지 않고 provenance와 superseded reason을 붙여 archive하는 방침도 재현성 측면에서 맞다.

`corrected_sensorfix_legacy_speed_axis`는 fix 이후라 **측정 자체가 무효는 아니다**. 그러나 v1, 25 bars,
legacy Gaussian, 500-epoch pilot이므로 현재 v2 성능의 자동 fallback 후보로는 부적절하다. 현재 v2 pack이
있을 때는 선택되지 않지만, pack 생성 실패 시 headline이 이 pilot로 조용히 내려간다. `superseded` 하나로
“깨진 조건”과 “유효하지만 비대표”를 섞지 말고, 예를 들어 `headline_eligible=false` 또는
`representative=false`를 별도로 두는 편이 낫다. current v2 pack이 없으면 pilot을 대신 그리지 말고
“current result unavailable”을 표시한다.

## 4. 남은 평가 순서 — PPO 재학습 없음

1. **평가 provenance/종료 계약 선행 수정**: 구현 완료. runtime source snapshot, git dirty state,
   Python environment와 exact-600 timeout 실측을 schema-v2 receipt에 묶고 테스트로 고정한다.
2. **밀도 A/B/C 전량 재측정**: 과거 C도 legacy 601 semantics이므로 재사용하지 않는다. 같은 미사용 seed와
   schema-v2 source로 ep24000/off, ep24000/riskcap, ep25000/riskcap을 5밀도에서 평가(15 cells).
3. **속도 interaction 확인**: 0.3/1.5 m/s × 130/160/190/205 bars × 새 seed 2개(16 cells)를 먼저 한다.
   primary test는 ≤205 aggregate density×speed interaction으로 사전등록하고 220은 OOD 보조자료로 둔다.
4. **detector offline gate**: full-FOV/range/occlusion/absent split, class-balanced 또는 focal/Dice 계열 loss,
   held-out PR/recall과 range-bin calibration을 만든 뒤 새 artifact만 지도학습한다. 이후 frozen PPO에서
   analytic vs learned를 같은 seed로 재평가한다.
5. dropout/H4 추가 A/B는 중단한다. learned detector가 안정된 뒤 dropout robustness를 다시 측정하면 된다.

## 5. 검증 기록

- `PYTHONNOUSERSITE=1 .../python tests/test_navrl_latency_compensate.py`: 33/33 PASS.
- speed-map aggregate count 재계산 및 logistic LR test 완료.
- H2/H4 count 기반 Wald CI와 pooled fixed-effect 재계산 완료.
- pre-chirality 9개와 post-fix legacy curve의 mtime/contract 및 dashboard selection 순서 확인.
- PPO policy bytes와 학습 weights는 변경하지 않았다.
- 최종 감사에서 종료 비교, rl_games timeout alias, evaluation source/environment receipt, 사이트 계약
  표기를 수정했다. frozen checkpoint의 과거 학습 의미는 소급 변경하지 않았다.
