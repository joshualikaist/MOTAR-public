# 사전등록 — 검출 거리 (2단계: 스크리닝 → 확증)

작성 2026-08-22, **어떤 arm도 실행하기 전**. 결과를 본 뒤 임계·seed·예산·판정 규칙을 바꾸지 않는다.
선행: `docs/prereg_2026-08-22_sensor_fidelity.md`(FIDELITY_NEUTRAL), `docs/discipline_review_2026-08-22.md`.

## 1. 질문

seed 421은 검출 **임계**의 부정직함이 성적을 설명하지 않음을 보였다(never-acquired +0.195 pp,
`target_hidden_fraction` 양 arm 0.82 동일). 구속 조건은 **`detector_max_range = 20 m` 하드 클립**이며
목표 밴드가 22.5–28 m라 표적이 대부분 클립 밖에 있다.

이제 처음으로 **28 m를 물리적으로 정당하게 볼 수 있는 센서**가 있다 — detect 1920×1200에서 28 m
표적은 지름 10.8 px·면적 92 px²로 임계 50을 넘는다. 질문: **클립을 28 m로 열면 과제가 얼마나 쉬워지는가.**

## 2. 계산이 무엇을 예측하는가 (스킬 규칙 — 측정 전 기록)

클립 28 m면 목표 밴드 전체가 검출 범위 **안**에 들어온다. seed 367이 같은 클립 변경으로
timeout `55.80 → 18.16%`를 보였고, 그것은 sub-pixel 사건이었는데도 그랬다. 정직한 센서에서는
그 개선이 **유지되거나 더 클 것**으로 예측한다. never-acquired는 크게 떨어질 것으로 본다.

**예측이 명확히 양성이라는 사실이 이 실험을 불필요하게 만들지는 않는다** — seed 367은 동결 정책
평가였고 토큰 정규화 교란을 안고 있었다. 여기서는 양 arm이 각자 일관된 정규화로 학습하므로
그 교란이 없다. 예측이 빗나가면 그것이 진짜 발견이다.

## 3. 왜 2단계인가

1,000 epoch 적응은 **"이 클립에서 도달 가능한 최선"을 답하지 못한다.** 동결 정책은 20 m까지
아무것도 못 보는 세계에 맞춘 탐색 전략을 배웠고, 클립이 열리면 최적 전략이 질적으로 다를 수 있다.
게다가 양 arm이 같은 20 m 정책에서 출발하므로 **arm B만 뭔가를 잊어야 하고, 설계가 B에 불리하다.**

따라서 1단계는 **스크리닝**이다: 2시간으로 17시간을 쓸지 결정한다. 양성이면 불리함을 뚫은 것이라
신뢰할 수 있고, **음성이면 "이 예산에서 미결"이지 "효과 없음"이 아니다** — 이 해석을 여기 못박는다.

## 4. 1단계 계약 (스크리닝)

| | |
|---|---|
| 출발 | frozen ref5in D1 ep1900, SHA `197ea26999d6bb9cf23c4e5a55acbe945f89985e2384687d60ab1dbae66a278e` |
| 학습 seed / 평가 seed | **457** / **461** (전수검색 0건) |
| 예산 | arm당 1,000 epoch / 4.096M samples |
| **조작 축** | **`NAVRL_DETECTOR_MAX_RANGE` 하나** |
| arm A | 클립 **20.0 m** |
| arm B | 클립 **28.0 m** |
| 양 arm 고정 | detect **1920×1200**, `MIN_PIXELS=50`, RGB 160×90, 70막대, 목표 22.5–28 m, appearance 0, governor off, `NAVRL_REFLECTION_COEF=0`, `NAVRL_LATERAL_BIAS_COEF=0` |
| 평가 | 각 arm 종단 checkpoint를 **자기 arm의 클립에서** seed 461, 2,049 ep |

**정규화에 대한 설계 주석**: 클립을 바꾸면 표적 토큰 정규화(`rel_pos / max_camera_range`,
`navrl_perception.py:1574,1578`)가 함께 바뀐다. 이는 교란이 **아니다** — 실제 28 m 센서는 28로
정규화하며, 양 arm이 각자 일관된 정규화로 학습·평가하므로 내부 불일치가 없다. seed 367에서
교란이었던 이유는 20으로 학습된 정책에 28 정규화를 먹였기 때문이다.

## 5. 1단계 게이트

**Gate 0 (학습 건전성, 판정보다 먼저)**: 양 arm `max_epochs` 정상 종료, KL 롤백 없음, 종단 SHA 기록.
실패한 arm은 VOID.

**Gate S (스크리닝, primary)** — never-acquired (arm B − arm A, pp):

| 판정 | 조건 |
|---|---|
| `RANGE_HELPS` | `≤ −15.00 pp` |
| `RANGE_INCONCLUSIVE_AT_THIS_BUDGET` | 그 외 |

`−15.00 pp`는 seed 367이 본 timeout 변화(−37.65 pp)의 절반에 미치지 않는 보수적 값이며,
warm-start 불리함을 감안한 것이다. **`RANGE_HELPS`가 아니면 2단계를 실행하지 않는다.**

capture/crash/timeout은 원값 보고, 판정에 쓰지 않는다(서로 다른 센서 정의에서 측정된 값).

## 6. 2단계 계약 (확증) — 1단계가 `RANGE_HELPS`일 때만

| | |
|---|---|
| 초기화 | **fresh** (warm-start 없음) — warm-start 적응은 "도달 가능한 최선"의 증거가 아니다 |
| 학습 seed / 평가 seed | **463** / **467** (전수검색 0건) |
| 예산 | arm당 **10,000 epoch** (실측 epoch당 3.1 s → arm당 8.6 h, 합 17 h) |
| arm·고정 | 1단계와 동일 |

**Gate C (확증)**: never-acquired `≤ −15.00 pp` **그리고** capture `≥ +5.00 pp` → `RANGE_CONFIRMED`.
never-acquired만 통과하면 `RANGE_ACQUISITION_ONLY`(획득은 쉬워졌으나 요격으로 이어지지 않음).
둘 다 실패면 `RANGE_NOT_CONFIRMED`.

10,000이 수렴이 아님을 명시한다 — 예산 제약(주말)에 따른 선택이며, 결과는 "10k epoch에서의 비교"다.

## 7. 권한과 한계

- **P2 STRICT FAIL / D1 FAIL / P3 BLOCKED를 변경하지 않는다.** 2단계는 70막대 고정 10k이므로
  P3(70→205막대, 30k, seed 211)가 **아니다**.
- `RANGE_CONFIRMED`가 나와도 정책을 채택하지 않는다 — 채택은 P2 gate 통과가 필요하다.
- 거리 오차가 여전히 0이다(`NAVRL_RANGE_ERROR_M=0`, 해석적 정확값). 실기 28 m 스테레오 시차는
  1.2–2.4 px로 측정 불가다. **따라서 "실기 준비됨"을 주장하지 않는다.**
- 잡동사니 배경·모션 블러 미모델링 → 결과는 낙관 편향.
- 1단계 음성은 "효과 없음"이 아니라 "이 예산에서 미결"이다(§3).

## 8. 하지 않을 것

결과를 본 뒤 임계·seed·예산 변경 · 1단계 음성인데 2단계 실행 · 한 run 두 축 ·
`aerial_gym/config/robot_config/**`·`resources/robots/**` 편집 · dirty runtime 실행.
