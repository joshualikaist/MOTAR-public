# 검출 거리 1단계 — 스크리닝 (학습 seed 457 / 평가 seed 461, 70 bars)

**판정: `RANGE_INCONCLUSIVE_AT_THIS_BUDGET`**

| arm | 클립 | never-acquired | capture | crash | timeout | target_hidden |
|---|---:|---:|---:|---:|---:|---:|
| clip20 | 20.0 m | 8.44% | 82.24% | 15.67% | 2.10% | 76.89% |
| clip28 | 28.0 m | 3.17% | 88.68% | 11.27% | 0.05% | 56.06% |

**never-acquired 차이 (clip28 − clip20): -5.27 pp** (RANGE_HELPS 임계 -15.00 pp 이하)

| arm | outcome | 최초획득 중앙값 | never-acq |
|---|---|---:|---:|
| clip20 | capture | 72 | 0.06% |
| clip20 | crash | 68 | 42.99% |
| clip20 | timeout | 537 | 79.07% |
| clip28 | capture | 31 | 0.06% |
| clip28 | crash | 31 | 27.71% |
| clip28 | timeout | 558 | 0.00% |

## 게이트 0 — 학습 건전성

| arm | 종단 epoch | frame | exit | rollback | KL skip 로그 | 종단 SHA |
|---|---:|---:|---|---:|---:|---|
| clip20 | 2900 | 11,878,400 | `max_epochs` | 0 | 0 | `aa5e3e8131aceb8c…` |
| clip28 | 2900 | 11,878,400 | `max_epochs` | 0 | 0 | `caf51fbb5afe22f6…` |

## 이 실험이 답하는 것과 답하지 않는 것

1,000 epoch warm-start 적응은 **"이 클립에서 도달 가능한 최선"을 답하지 못한다**(사전등록 §3). 양 arm이 20 m 세계에 맞춰진 같은 정책에서 출발하므로 **clip28만 뭔가를
잊어야 하고, 설계가 clip28에 불리하다.** 따라서 양성이면 불리함을 뚫은 것이라 신뢰할 수
있고, **음성이면 "이 예산에서 미결"이지 "효과 없음"이 아니다.**

**capture/crash/timeout은 원값으로만 보고하며 판정에 쓰지 않는다** — 서로 다른 센서
정의에서 측정된 값이다. 판정 함수 `classify_verdict()`는 never-acquired 차이 하나만
인자로 받으므로 구조적으로 이 값들을 볼 수 없다.

## 고정된 조건

- 양 arm 동일: detect 1920×1200, `min_pixels=50`, RGB 160×90, 70 bars, 목표 22.5–28.0 m, appearance·지연·거리오차 0, governor off, `NAVRL_REFLECTION_COEF`/`NAVRL_LATERAL_BIAS_COEF` 미설정(=0).
- 학습 환경 차이는 실제로 측정된다: 정규 학습 런처가 만들어내는 환경을 양 arm에서 덤프해 대칭차를 계산하며, 허용되는 차이는 `NAVRL_DETECTOR_MAX_RANGE`와 run 태그·로그 경로뿐이다.
- 양 arm은 **하나의 학습 소스 영수증**(schema 1)을 공유하고, 두 번째 arm 시작 전에 원본 바이트를 재해싱해 검증한다.

## provenance override

- **어느 arm도 override를 쓰지 않는다.** 각 arm은 **자기가 학습한 클립에서** 평가되고, 체크포인트의 `cfg_detector_max_range/cfg_detect_width/cfg_detect_height`를 평가기의 요청값과 비교한다. `cfg_detector_min_pixels`도 학습·평가 모두 50이다. 실행 시점에 force 없는 preflight가 통과함을 요구하며, 거부되면 담요식 force 대신 **중단**한다.
- arm 구분자는 체크포인트와 평가 계약 양쪽에 독립적으로 남는다: `env_state.cfg_detector_max_range`와 `v2_evaluation_contract.target_camera_max_range_m`.

- warm start SHA-256 `197ea26999d6bb9c…`
- 품질 게이트: 판정 17개, 실패 0개

## 권한

이 실험은 **결정 권한이 없다**. P2 STRICT FAIL·D1 FAIL을 소급 변경하지 않으며 P3를
해제하지 않는다. `RANGE_HELPS`일 때에만 2단계를 실행할 자격이 생기며, 그것도 정책 채택
권한은 아니다 (`stage2_authorised: false`).

## 한계 (사전등록 §7)

- L1: P2 STRICT FAIL / D1 FAIL / P3 BLOCKED를 변경하지 않는다. 2단계는 70막대 고정 10k이므로 P3(70→205막대, 30k, seed 211)가 아니다.
- L2: RANGE_CONFIRMED가 나와도 정책을 채택하지 않는다 — 채택은 P2 gate 통과가 필요하다.
- L3: 거리 오차가 여전히 0이다(NAVRL_RANGE_ERROR_M=0, 해석적 정확값). 실기 28 m 스테레오 시차는 1.2–2.4 px로 측정 불가다. 따라서 '실기 준비됨'을 주장하지 않는다.
- L4: 잡동사니 배경·모션 블러 미모델링 → 결과는 낙관 편향.
- L5: 1단계 음성은 '효과 없음'이 아니라 '이 예산에서 미결'이다(사전등록 §3).
