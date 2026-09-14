# 센서 모델 충실도 (seed 421, 70 bars, eval-only)

**판정: `FIDELITY_NEUTRAL`**

| arm | detect | min_px | never-acquired | capture | crash | timeout | target_hidden |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | 160×90 | 2 | 18.89% | 70.52% | 19.81% | 9.66% | 82.12% |
| fidelity | 1920×1200 | 50 | 19.08% | 71.06% | 18.59% | 10.35% | 82.23% |

**never-acquired 차이 (fidelity − baseline): +0.20 pp** (임계 +10.00 pp / 중립대 ±3.00 pp)

| arm | outcome | 최초획득 중앙값 | p90 | never-acq |
|---|---|---:|---:|---:|
| baseline | capture | 89 | — | 0.00% |
| baseline | crash | 83 | — | 54.19% |
| baseline | timeout | 558 | — | 84.34% |
| fidelity | capture | 78 | — | 0.00% |
| fidelity | crash | 99 | — | 55.91% |
| fidelity | timeout | 553 | — | 83.96% |

- p90은 기록하지 못했다: navrl_task.py first_acquisition_payload() exports first_visible_step_mean and a lower first_visible_step_median only; the underlying per-outcome histogram (_fa_eval_outcome_first_hist) is not written to the result JSON, so no p90 is derivable from any recorded field. Recording it would require a runtime-source change, which this eval-only preregistration does not authorise.

## 방향 — arm B가 나빠지는 것은 예상된 결과다

정직해진 센서는 검출을 **더 어렵게** 만든다(면적 임계 2 → 50 px², 25배). 따라서 fidelity
arm의 never-acquired 상승은 **실패가 아니라 예상된 결과**이며, 그 크기가 곧 "지금까지의
성적 중 얼마가 존재할 수 없는 센서 덕분이었는가"의 추정치다. 이 실험의 가치는 개선이
아니라 **정직한 기준선의 확립**에 있다.

**capture/crash/timeout은 원값으로만 보고하며 판정에 쓰지 않는다.** 동결 정책은 부정직한
센서로 학습됐으므로 정직한 센서에서의 성능 저하는 정책의 결함이 아니라 계보의 결과다.
판정 함수 `classify_verdict()`는 never-acquired 차이 하나만 인자로 받으므로 구조적으로
이 값들을 볼 수 없다.

## 고정된 조건

- `detector_max_range` = 20.0 m, **양 arm 동일** — seed 367은 이 값을 바꾸며 actor 표적 토큰까지 재정규화했고 그것이 교란이었다. 본 실험은 이 변수를 아예 export하지 않는다.
- RGB 카메라 160×90 양 arm 동일, appearance 교란 전부 0, 검출 지연·거리오차 0, governor off, deterministic, reflection_mode original.
- 두 arm의 runtime 바이트 맵이 동일하고, 평가 계약에서 다른 값은 `detector_min_pixels` 단 하나다.

## provenance override (사전등록 §5-b)

- **baseline arm은 override를 쓰지 않았다** (`used: false`).
- **fidelity arm만** 좁은 단일 필드 override를 쓴다. 실행 시점에 force 없이 먼저 돌려 `returncode == 2`와 불일치 라인 집합이 정확히 `[cfg_detector_min_pixels: checkpoint=2 expected=50.0]` 하나임을 증명한 **뒤에만** `NAVRL_V2_FORCE=1`을 적용한다. 두 줄이거나 다른 필드면 중단한다.
- 담요식 force보다 **더 엄격하다**: 담요식은 다른 불일치까지 함께 가리지만, 이 절차는 불일치가 그 한 필드뿐임을 실행 시점에 증명한다. 이 검증은 `preflight`와 `run` 양쪽에서 수행되므로 검증되지 않은 override 아래에서 셀이 생성될 수 없다.

- checkpoint SHA-256 `197ea26999d6bb9c…`
- 품질 게이트: 판정 9개, 실패 0개

## 권한

이 실험은 **결정 권한이 없다**. P2 STRICT FAIL·D1 FAIL을 소급 변경하지 않으며 P3를 해제하지 않는다 (`p2_verdict_changed`/`d1_verdict_changed`/`p3_unlocked` 전부 false).

## 한계 (사전등록 §7)

- L1: 잡동사니 배경 미모델링. 렌더가 표적을 평평한 순수 빨강으로 칠하고 분할기가 색 규칙이라 픽셀 클래스가 자명하게 분리된다. 따라서 지름 8 px 임계는 하늘 배경 기준이며 실기의 도심·수목 배경(15–20 px)보다 관대하다. 본 실험 결과는 여전히 낙관 편향이다.
- L2: 모션 블러 미모델링. 5인치 기체가 고 yaw rate에서 10 ms 노출이면 12 px 표적이 여러 픽셀로 번진다. 현재 모델에 없다.
- L3: 거리 오차 0. navrl_detector.py:130이 정확한 해석적 교차 거리를 쓰고 NAVRL_RANGE_ERROR_M=0이다. 정책은 임의 거리에서 오차 0의 거리를 받는다. 실기에서는 28 m 스테레오 시차가 1.2–2.4 px로 측정 불가다. 이 실험은 그것을 고치지 않으므로, 결과는 '거리는 여전히 공짜로 주어진 상태에서의 검출 충실도'만 말한다.
- L4: 단일 정책·단일 seed·70막대 1조건. 계보나 밀도 전반으로 일반화하지 않는다.
- L5: 임계 50 px²는 문헌 중앙값이지 이 시스템에서 측정된 값이 아니다. 감이 아니라 유도된 값이지만 여전히 외부 기준의 이식이다.
