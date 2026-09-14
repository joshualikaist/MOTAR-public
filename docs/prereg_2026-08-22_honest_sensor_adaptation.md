# 사전등록 — 정직한 센서에서의 적응 학습

작성: 2026-08-22, **센서 충실도 평가 결과를 보기 전, 어떤 학습도 실행하기 전**. 결과를 본 뒤
임계·seed·arm·예산·판정 규칙을 변경하지 않는다.

선행: `docs/prereg_2026-08-22_sensor_fidelity.md`(평가 전용, 동결 커밋 `e2b95f8`),
`WORKLOG.md` 2026-08-22 항목들, `VERIFICATION.md`(실행 authority).

---

## 1. 실행 조건 — 이 사전등록은 조건부다

**센서 충실도 평가가 `FIDELITY_COST_CONFIRMED`를 낼 때만 실행한다.**
`FIDELITY_NEUTRAL`이나 `INCONCLUSIVE_SENSOR_FIDELITY`이면 **실행하지 않는다** — 전자는 고칠
것이 없다는 뜻이고, 후자는 무엇을 고치는지 모른다는 뜻이다.

이 조건을 결과를 본 뒤에 완화하지 않는다. "생각보다 작지만 그래도 돌려보자"는 금지다.

## 2. 무엇을 묻는가

동결 ref5in D1 ep1900은 **존재할 수 없는 센서**로 학습됐다 — 지름 1.6 px 임계, 오차 0의 거리,
실기 대비 13배 조악한 해상도. 센서 충실도 평가는 그 정책을 정직한 센서에 놓았을 때의 **비용**을
잰다. 본 실험은 그 다음 질문을 잰다: **정직한 센서에서 다시 학습하면 그 비용을 얼마나 되찾는가.**

되찾지 못하는 부분이 곧 **과제 자체의 난이도**이며, 그것이 P2/D1 병목의 정직한 크기다.

## 3. 알려진 한계 — 판정 전에 명시

- **(L1) 거리는 여전히 공짜다.** `navrl_detector.py:130`이 정확한 해석적 교차 거리를 쓰고
  `NAVRL_RANGE_ERROR_M=0`이다. 본 실험은 그것을 고치지 않는다. 따라서 결과는 "검출은 정직해졌으나
  거리는 여전히 주어진" 중간 상태를 말한다. 거리 충실도는 별개 사전등록이다.
- **(L2) 잡동사니 배경 미모델링.** 임계 50 px²는 하늘 배경 기준이며 실기의 도심·수목(15–20 px)보다
  관대하다. 결과는 낙관 편향이다.
- **(L3) 모션 블러 미모델링.**
- **(L4) warm-start 계보.** 양 arm 모두 이미 D1 FAIL인 checkpoint에서 출발한다. 어떤 결과도
  P2/D1을 통과시키지 않으며 그렇게 해석해서도 안 된다.
- **(L5) 단일 학습 seed, 단일 예산.** 학습 곡선이나 수렴을 주장하지 않는다.

## 4. 실험 계약

| 항목 | 값 |
|---|---|
| 출발 checkpoint (양 arm 동일) | ref5in D1 ep1900, SHA `197ea26999d6bb9cf23c4e5a55acbe945f89985e2384687d60ab1dbae66a278e` |
| **학습 seed (양 arm 동일)** | **433** — 전수 검색 사용 이력 0건 |
| **평가 seed** | **449** — 전수 검색 사용 이력 0건 |
| 예산 (양 arm 동일) | **1,000 epoch / 4.096M samples**, `--max_epochs`로 종료 (riskcap 적응 선례와 동일) |
| 아레나 | 70 bars, 목표 거리 22.5–28 m |
| **조작 축** | **센서 충실도 하나** — `NAVRL_DETECT_WIDTH/HEIGHT`와 `NAVRL_DETECTOR_MIN_PIXELS`가 한 쌍으로 움직인다 |
| arm A (control) | detect 160×90, `MIN_PIXELS=2` — 부정직한 센서에서 같은 예산만큼 더 학습 |
| arm B (treatment) | detect 1920×1200, `MIN_PIXELS=50` — 정직한 센서에서 적응 |
| 고정 | `NAVRL_DETECTOR_MAX_RANGE=20.0`, RGB 160×90, airframe·reward·horizon·representation·governor·밀도·LR, appearance 전부 0, `NAVRL_REFLECTION_COEF=0`, `NAVRL_LATERAL_BIAS_COEF=0` |

**arm A가 필수인 이유**: 예산 1,000 epoch 자체의 효과와 센서 변경의 효과를 분리한다. arm A 없이
arm B만 돌리면 개선(또는 악화)을 센서 탓으로 돌릴 수 없다.

**`detector_max_range`는 20 m로 고정한다** — seed 367의 토큰 재정규화 교란을 피한다.

## 5. 게이트 — 결과를 보기 전에 확정

### Gate 0 — 학습 건전성 (판정보다 먼저)

양 arm 모두: `max_epochs` 정상 종료(중단·발산 없음), KL이 롤백을 유발하지 않음, 종단 checkpoint의
SHA가 기록됨. 하나라도 실패하면 그 arm은 **VOID**이며 판정하지 않는다.

### Gate R — 회복 (primary)

정직한 센서(detect 1920×1200 / `MIN_PIXELS=50`)에서 seed 449로 held-out 평가한 never-acquired:

- `NA_frozen` = 동결 정책 (센서 충실도 평가의 fidelity arm에서 이미 측정됨)
- `NA_A` = arm A 종단 정책
- `NA_B` = arm B 종단 정책

| 판정 | 조건 |
|---|---|
| **ADAPTATION_RECOVERS** | `NA_B ≤ NA_frozen − 10.00 pp` **그리고** `NA_B ≤ NA_A − 5.00 pp` |
| **ADAPTATION_INEFFECTIVE** | `NA_B > NA_A − 5.00 pp` (센서 arm이 예산 arm을 유의하게 못 이김) |
| **INCONCLUSIVE_ADAPTATION** | 그 외 |

임계 근거(사전 확정): 10.00 pp는 센서 충실도 평가가 "비용 확인"으로 쓰는 것과 같은 크기이므로,
"입힌 만큼 되찾았는가"를 대칭적으로 묻는다. 5.00 pp는 arm A 대비 마진이며, 예산만으로도 얼마간
좋아질 수 있으므로 그보다 명확히 나아야 센서 덕이라고 말할 수 있다.

### Gate P — 성능 (보고용, 판정 아님)

capture/crash/timeout을 양 arm·동결 정책에 대해 **원값으로 보고한다.** 판정에 쓰지 않는다 —
서로 다른 센서에서 측정된 성능은 직접 비교 가능한 양이 아니다.

## 6. 이 실험이 주는 권한과 주지 않는 권한

- **P2 STRICT FAIL / D1 FAIL / P3 BLOCKED를 변경하지 않는다.** 어떤 결과도 이들을 소급 변경하거나
  P3를 해제하지 않는다(`VERIFICATION.md` fail-closed 1·5). 본 실험은 P3가 **아니다** —
  P3는 70→205 bars, 30k epoch, seed 211이며 본 실험은 70 bars 고정 1,000 epoch다.
- `ADAPTATION_RECOVERS`가 나와도 **정책을 채택하지 않는다.** 채택은 P2 gate를 통과해야 하며 그것은
  별개 실행이다.
- 거리 충실도(L1)를 고치지 않으므로 "실기 준비됨"을 주장하지 않는다.
- 센서 충실도 평가의 판정을 소급 변경하지 않는다.

## 7. 하지 않을 것

- Gate 조건 미충족 시 실행 (§1)
- 예산 연장, 해상도·임계 스윕, 결과를 본 뒤의 변경
- `detector_max_range` 변경
- reflection/lateral-bias 보조 손실 동시 투입 (한 run 한 축)
- `aerial_gym/config/robot_config/**`·`resources/robots/**` 편집 (provenance freeze)
- dirty runtime 실행

## 8. 기록 요건

`results/navrl_ref5in_honest_sensor_adaptation_seed433/`. 양 arm의 run 폴더·종단 checkpoint SHA·
epoch/샘플 수·KL 통계·held-out 원값. 무효·실패 실행과 VOID 사유. 요약에
`p2_verdict_changed: false`, `d1_verdict_changed: false`, `p3_unlocked: false`,
`decision_authority: "none"`.
