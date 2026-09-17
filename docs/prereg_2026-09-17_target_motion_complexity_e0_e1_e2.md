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
