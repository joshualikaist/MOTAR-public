# 사전등록 — TM-E0/E1/E2: target motion complexity vs tracking and close approach

작성 2026-09-17, **측정 개시 전**. 이 문서는 결과를 본 뒤 수정하지 않는다.

> **실행 상태: `PREREGISTERED / NOT_RUN`.**
> 이 문서는 준비까지다. 이번 작업에서 outcome experiment는 실행하지 않았고,
> 새 PPO 학습도 하지 않았다. 실행에는 별도의 GPU authority 승인이 필요하다
> (`docs/research_authority_2026-08-26.json`).

관련: [`docs/target_behavior_ladder_2026-09-16.md`](target_behavior_ladder_2026-09-16.md),
[`docs/status/gt_browser_v1_freeze_2026-09-17.md`](status/gt_browser_v1_freeze_2026-09-17.md).

---

## 1. Research question

> How does target motion complexity affect tracking and close-approach performance
> under an otherwise matched simulation contract?

`target behavior`만 독립변수로 두고, 나머지 계약은 전부 고정한다.

## 2. Independent variable — the only thing that changes

| Arm | `NAVRL_TARGET_BEHAVIOR_LEVEL` | Target motion |
|---|---|---|
| **TM-E0** | `e0_static` | Static target (speed 0) |
| **TM-E1** | `e1_cv` | Constant-velocity nominal reference |
| **TM-E2** | `e2_obstacle_aware` | Waypoint nominal reference + bounded GT-obstacle executor |

TM-E3/TM-E4는 `PLANNED`이고 선택 시 fail-closed다. 이 실험에 포함하지 않는다.

## 3. Matched contract — held identical across all three arms

```text
same dynamics          NAVRL_TARGET_DYNAMICS=bounded
same policy            single frozen checkpoint, SHA recorded in the receipt
same observation       actor 898-dim / critic 906-dim, cluster_sector, FOV 240 deg
same reward            unchanged; no reward modification is in scope
same controller        unchanged fixed control stack
same safety filter     unchanged riskcap configuration
same arena             40 x 40 x 3 m, navrl_band, touch 0.4 / gap 1.6
same densities         70 / 115 / 160 / 205 (300 is disconnected stress, excluded)
same episode budget    identical episode count per cell, identical seeds
same evaluation nonce  recorded per cell; expected to be the ONLY differing key
                       besides the target-behavior level itself
```

실행 시 `condition` 키 집합을 arm 간 대조해, **차이가 `target_behavior_level`과
`evaluation_nonce`뿐임을 receipt로 증명한다.** 2026-09-07 independent verification과
동일한 절차다. 그 대조가 실패하면 결과는 무효로 기록한다.

## 4. Known limits — stated before measurement

**(L1) 난이도 축이 하나가 아니다.** E0→E1→E2는 "더 어려움"의 단조 사다리가 아니다.
E2의 장애물 회피는 표적을 느리게 만들 수 있고, 그것이 추적을 *쉽게* 만들 수도 있다.
따라서 primary는 "E2가 더 어렵다"가 아니라 **"target behavior가 outcome을 움직이는가"**다.

**(L2) 단일 checkpoint.** 정책은 E0/E1/E2 중 어느 것으로도 학습되지 않았다.
historical target sampling으로 학습된 정책을 세 target 분포에서 평가하는 것이므로,
이것은 **generalization 측정이지 matched training 비교가 아니다.** E2 arm의 손실을
"E2가 본질적으로 어렵다"로 읽을 수 없다 — 분포 이동과 교락한다.

**(L3) 2-D kinematic target.** `bounded` lineage는 rigid-body attitude·모터·접촉이 없다.
`physical` lineage는 route gate가 `FAIL_ROUTE_MECHANISM`이므로 이 사전등록에 넣지 않는다.

**(L4)** 단일 arena family, 단일 seed 계보. 일반화 주장 없음.

## 5. Primary and guard outcomes

Primary는 **relative-distance minimization**과 **continuous tracking**으로 기술한다.
historical `capture radius` semantics는 historical arm의 provenance로만 남기고,
새 outcome의 정의로 재사용하지 않는다.

| Outcome | Definition | Role |
|---|---|---|
| `final_relative_distance_m` | episode 종료 시 `d(t) = \|\|p_tracker - p_target\|\|` | primary |
| `median_relative_distance_m` | episode 내 `d(t)` 중앙값 | primary |
| `time_within_approach_band_frac` | `d(t)` 가 선언된 approach band 안에 있던 step 비율 | primary |
| `tracking_error_rms_m` | `sqrt(mean(d(t)^2))`, reference `d* = 0` | primary |
| `bar_contact_rate` | 장애물 접촉 | guard |
| `timeout_rate` | 시간 초과 | guard |
| `target_infeasible_step_rate` | 표적 자체의 속도/가속/선회 위반 | **validity gate** |
| `target_obstacle_penetration_rate` | 표적 장애물 관통 | **validity gate** |

**validity gate가 0이 아니면 outcome은 해석하지 않는다.** 표적 생성기가 계약을 위반한
상태의 비교는 난이도 비교가 아니다.

## 6. Decision rule

사전에 고정한다. 결과를 본 뒤 바꾸지 않는다.

1. 세 arm 모두에서 validity gate가 0인지 먼저 확인한다. 아니면 **INVALID**로 기록하고 중단.
2. condition 대조에서 차이가 `target_behavior_level` + `evaluation_nonce`뿐인지 확인한다.
   아니면 **CONFOUNDED**로 기록하고 중단.
3. 그 다음에만 primary outcome을 비교한다. arm 간 차이는 셀별 분모로 계산한 비율과
   95% CI로 보고하며, **CI가 0을 포함하면 "차이 없음"이 아니라 "이 표본으로는 미검출"**로 쓴다.
4. 어떤 결과도 real-flight, deployment, contact-system 주장으로 확장하지 않는다.

## 7. Out of scope

```text
new PPO training
reward modification
real-flight deployment
TM-E3 reactive evader
TM-E4 learned evader / self-play
browser GT preview as evidence
```

browser GT tracking preview(`docs/status/arena_demo_planner.js`)는 설명용 시각화이며
이 실험의 증거가 아니다. GT target/obstacle state는 actor observation에 들어가지 않고,
`tests/test_browser_gt_state_not_in_policy_observation.py`가 이를 강제한다.

---

# AMENDMENT 1 — 2026-09-17, **측정 개시 전**

이 개정은 **어떤 outcome도 측정하기 전에** 이루어졌다. 실행 상태는 여전히
`PREREGISTERED / NOT_RUN`이고, GPU 작업은 한 번도 수행되지 않았다. 근거 감사는
[`docs/audits/retraining_readiness_audit_2026-09-17.md`](audits/retraining_readiness_audit_2026-09-17.md).

개정 사유는 세 가지 **결함**이 사전등록 자체에서 발견됐기 때문이다.

## A1.1 결함 1 — in-distribution reference arm이 없었다

frozen checkpoint `f702213936…`의 `env_state`는
`cfg_target_motion_model = symmetric_local_steer_v2_heading_continuity90`,
`cfg_target_pattern = mixed`를 attest한다. 즉 이 정책은 **legacy** target lineage,
**mixed**(cv/waypoint 50:50) pattern에서 학습됐다. 학습 launcher
`train_navrl_v2_search.sh:154`의 `NAVRL_TARGET_DYNAMICS:-legacy`도 이를 확인한다.

그런데 원안의 세 arm은 모두 `NAVRL_TARGET_DYNAMICS=bounded`를 고정했다. TM-E2는
`bounded|physical` + `waypoint`를 요구하고, E1은 `cv`를, E0는 speed 0을 강제한다.
**따라서 E0/E1/E2 중 어느 것도 학습 분포가 아니다.** 원안대로 실행하면 E2의 성능
저하를 "E2가 어렵다"와 "세 arm이 전부 off-distribution이다"로부터 분리할 수 없다.

**수정: arm H(historical)를 추가한다.** arm H는 학습 계약 그대로
(`NAVRL_TARGET_BEHAVIOR_LEVEL=historical`, `NAVRL_TARGET_DYNAMICS=legacy`,
`NAVRL_TARGET_PATTERN=mixed`)이며, **정책의 in-distribution reference**다.
모든 E0/E1/E2 대비는 arm H를 기준으로 보고한다.

## A1.2 결함 2 — `time_within_approach_band_frac`는 이 task에서 정의되지 않는다

`navrl_task.py:4999`의 주석은 명시적이다: **"Interception semantics (always on):
capture ends the episode."** `success_radius = 0.5 m`에 도달하면 swept capture 판정으로
**episode가 종료된다**(`navrl_task.py:6025`).

따라서 "approach band 안에 머문 시간 비율"은 구조적으로 편향된다: 잘 추적한 episode일수록
더 일찍 종료되어 표본을 적게 남긴다. band 숫자를 고르는 문제가 아니라 **metric 자체가
이 termination과 양립하지 않는다.**

**수정: `time_within_approach_band_frac`를 제거한다.** 임의의 band 숫자를 고르지 않는다.
대신 capture termination 아래에서 잘 정의되는 양으로 대체한다:

| Outcome | 정의 | 역할 |
|---|---|---|
| `min_relative_distance_m` | episode 내 최소 `d(t)` (`ep_min_goal_dist`, 종료 전 갱신) | **primary** — 모든 episode에서 정의됨 |
| `capture_rate` | `d < 0.5 m` swept 도달 비율 | **primary** — historical outcome |
| `crash_rate` / `timeout_rate` | historical outcome triple의 나머지 | guard |
| `steps_to_capture` | captured episode의 종료 step | secondary, **conditioning 명시 필수** |
| `median_relative_distance_m` | episode 내 `d(t)` 중앙값 | secondary, **길이 교락 명시 필수** |

`tracking_error_rms_m`와 `final_relative_distance_m`도 같은 이유로 **primary에서 강등**한다
(전자는 길이 교락, 후자는 non-captured episode에서만 정의되므로 선택 편향).

이것은 §6 termination 감사의 판정 **A. CLOSE_APPROACH_TASK**를 따른 것이다. historical
0.5 m termination을 유지하고, continuous-tracking 주장을 축소한다. 이번 작업에서
termination을 수정하지 않는다.

## A1.3 결함 3 — 고정되지 않은 숫자들

원안에 없던 값을 **지금** 고정한다. 근거 없이 고르지 않았다.

```text
arms                    H (historical), E0, E1, E2                     [A1.1]
densities               70 / 115 / 160 / 205                           현행 계보 밀도 knot
evaluation seeds        4101, 4102, 4103                               미사용 seed, 연속 3개
cells                   4 arms x 4 densities x 3 seeds = 48
episodes per cell       정확히 2048                                     아래 근거
episode length          600 steps                                      checkpoint attest
evaluation timeout      episode 길이로만 결정, 별도 wall-clock timeout 없음
CI method               per-cell binomial: Wilson score, 95%
                        arm 대비: seed 수준 평균의 paired difference,
                        95% BCa bootstrap, 20000 resample, seed=4100
seed aggregation        seed를 분석 단위로 취급. cell 내부는 seed별로
                        비율을 먼저 계산하고, arm 대비는 seed-paired로 본다.
                        episode를 독립 표본으로 풀링하지 않는다.
multiple comparison     primary 대비는 arm H 대비 E0/E1/E2 3개.
                        Holm-Bonferroni로 family-wise 보정. 밀도별 12개
                        세부 대비는 **탐색적**이며 보정하지 않고,
                        확증 주장에 쓰지 않는다.
```

**episodes per cell = 2048 근거.** 학습·평가 배치는 128 env이므로 2048 = 16 배치로
**정확히** 나누어떨어진다. 직전 held-out 설계는 `actual >= 2,049`를 요구해 9/9 cell이
2,049가 됐는데, 그 +1은 128-env tail이 만든 잔여물이다(WORKLOG 기록). 2048로 고정하면
그 tail 비대칭이 구조적으로 사라지고 표본 크기는 실질적으로 동일하다.

**evaluation seed 근거.** 4101–4103은 기존 receipt에서 사용된 적 없는 연속 3개다.
seed를 결과를 보고 고르는 경로를 차단하기 위해 지금 고정한다.

## A1.4 matched contract — 기계로 강제한다

원안 §3은 조건 일치를 산문으로 요구했다. 감사에서 시뮬레이터의 기존 guard가
**`logger.warning`만 내고 실행을 계속**한다는 것이 확인됐다
(`navrl_task.py`, "Loud config-drift guard (warn, never override)"). frozen-policy
matched comparison에서 이것은 부적절하다.

**수정: `tools/audits/check_eval_condition_contract.py`를 실행 전 gate로 의무화한다.**
exit 2(`CONFOUNDED`)이면 그 cell을 실행하지 않는다. 독립변수로 선언된 키
(`cfg_target_motion_model`, `cfg_target_pattern`, `cfg_target_speed_*`,
`cfg_target_behavior_level`, `cfg_target_dynamics`) **외의** 차이는 전부 confound다.

특히 checkpoint 학습 계약은 **HEAD 기본값으로 재현되지 않는다.** 14개 knob이 다르고
(`docs/audits/training_contract_vs_current_runtime_2026-09-17.json`), 그중 6개는
observation 차원·observation 정규화 분모·action scale·termination geometry를 바꾼다.
평가는 반드시 `train_navrl_v2_search.sh`의 env block을 재생해야 한다. gate가 이를 강제한다.

## A1.5 heading-validity threshold — ASSUMED로 고정

checkpoint `env_state`에 `heading_valid` 키가 **없다**. `resolve_heading_valid_speed_contract`는
이 경우를 `ASSUMED_PRE_KEY_DEFAULT`로 분류하며, 소스 주석은 키 도입 이전 계보가 inline
**1e-05 m/s** epsilon을 썼을 수 있다고 명시한다. 현행 값은 **0.10 m/s**로 1e4배 차이다.

이 threshold는 "잔류 속도를 진행 방향으로 볼 것인가"를 결정하므로 moving-target metric의
의미를 바꾼다. **따라서 이 실험의 모든 arm에서 이 값을 동일하게 유지하고, 결과 문서에
`heading_valid_speed_provenance = ASSUMED_PRE_KEY_DEFAULT`를 명시한다.** 소급해서 0.10을
checkpoint 계약으로 기록하지 않는다. arm 간에는 동일하므로 대비를 교락하지 않지만,
학습 시점 계약과 동일하다는 보장은 **없다**.

## A1.6 변경되지 않은 것

research question, 독립변수의 정체, validity gate(§5 원안), decision rule(§6 원안),
out-of-scope(§7 원안)는 그대로다. 새 PPO 학습은 여전히 금지이며 실행 상태는
`PREREGISTERED / NOT_RUN`이다.
