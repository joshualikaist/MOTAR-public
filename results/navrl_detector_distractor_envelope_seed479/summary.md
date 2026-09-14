# distractor envelope — False Target Lock Rate (seed 479, 70 bars, 셀당 2049 에피소드)

**2 × 4 요인설계: 검출기(기본 / v7) × distractor 수(0/1/3/5) = 8 셀.**
판정(§5 Gate F)은 **검출기마다 따로** 그 검출기의 N=5 셀에서
내린다. 통합 판정은 없다 — 한쪽 결과는 다른 쪽에 대한 근거가 아니다.

| 검출기 | 동작점 | 판정 | N=5 FTLR | 가시 프레임 |
|---|---:|---|---:|---:|
| `default` | thr 0.55 | **`COLOR_SHORTCUT_CONFIRMED`** | 88.53% | 132,068 |
| `v7` | thr 0.7 | **`COLOR_SHORTCUT_CONFIRMED`** | 90.27% | 131,679 |

임계(양쪽 검출기 공통): ≥ 50.00% → `COLOR_SHORTCUT_CONFIRMED`, ≤ 5.00% → `DETECTOR_ROBUST_TO_DISTRACTORS`, 그 외 `INCONCLUSIVE_DISTRACTOR_ENVELOPE`. 분류 반경 0.5 m.

**두 판정은 서로 독립이다.** `default`는 5개 파라미터짜리 색 규칙이고 `v7`은 학습된
spatial CNN이다. 한쪽이 무너졌다고 다른 쪽이 무너지는 것도, 한쪽이 견뎠다고 다른 쪽이
견디는 것도 아니다. 각 판정은 자기 검출기에만 적용된다.

> ⚠️ **위 두 FTLR을 빼지 마시오 (한계 L6, 사전등록 §3-c).** 동결 정책이 각 검출기의
> 출력을 보고 비행하므로 두 검출기의 **궤적이 다르고**, 따라서 프레임 분포 —
> 거리·베어링·가림, 특히 **표적과 distractor가 동시에 보이는 빈도** — 가 다르다.
> FTLR은 바로 그 분포 위에서 정의되므로 두 값의 차이는 검출기 강건성과 궤적 분포를
> 뒤섞은 값이며 이 셀들로는 둘을 분리할 수 없다. 그래서 이 문서에도 `summary.json`에도
> **그 차이는 없다 — 계산하지 않았다.** 아래 셀별 표에서 유효한 비교는 **같은 검출기
> 행 안에서 N에 따른 변화**뿐이다. capture/crash/timeout도 같다: v7 행은 계보와도,
> `default` 행과도 비교하지 않는다.

## 셀별 분류 (사전등록 §4)

| 검출기 | N | FTLR | TARGET | DISTRACTOR | GHOST | 가시 프레임 | count 평균 | conf 평균 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `default` | 0 | — | — | — | — | — | — | — |
| `default` | 1 | 52.75% | 39,370 | 39,856 | 4,095 | 83,321 | 147.4 | 0.650 |
| `default` | 3 | 79.73% | 25,068 | 83,923 | 14,676 | 123,667 | 181.0 | 0.717 |
| `default` | 5 | 88.53% | 15,144 | 92,299 | 24,625 | 132,068 | 167.4 | 0.719 |
| `v7` | 0 | — | — | — | — | — | — | — |
| `v7` | 1 | 60.70% | 35,121 | 37,506 | 16,732 | 89,359 | 155.4 | 0.826 |
| `v7` | 3 | 83.06% | 20,416 | 77,138 | 22,999 | 120,553 | 173.2 | 0.896 |
| `v7` | 5 | 90.27% | 12,816 | 86,503 | 32,360 | 131,679 | 160.8 | 0.892 |

N=0 행이 비어 있는 것은 결측이 아니다. 분류기는 distractor가 0일 때 **호출되지 않는다**
— 회귀 셀이 손대지 않은 계보를 그대로 재는 것이 Gate 0.1의 요구사항이기 때문이다
(사전등록 §5). 따라서 `count`/`confidence`의 N 의존성은 N=1·3·5 세 점으로 읽는다.

## 보조 수치 (판정에 쓰지 않음, 사전등록 §4)

| 검출기 | N | capture | crash | timeout | never-acq | target_hidden | 측정↔표적 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `default` | 0 | 72.57% | 17.42% | 10.00% | 17.96% | 82.11% | — |
| `default` | 1 | 37.97% | 57.69% | 4.34% | 10.79% | 68.56% | 10.036 m |
| `default` | 3 | 22.82% | 76.60% | 0.59% | 6.19% | 47.25% | 15.791 m |
| `default` | 5 | 13.23% | 86.53% | 0.24% | 4.98% | 38.21% | 17.596 m |
| `v7` | 0 | 68.13% | 21.38% | 10.49% | 18.59% | 82.55% | — |
| `v7` | 1 | 36.85% | 59.64% | 3.51% | 10.40% | 67.48% | 10.599 m |
| `v7` | 3 | 20.74% | 78.23% | 1.02% | 6.69% | 48.14% | 15.964 m |
| `v7` | 5 | 12.64% | 87.07% | 0.29% | 4.39% | 38.73% | 17.386 m |

**capture/crash/timeout은 원값이며 판정에 쓰지 않는다.** 판정 함수
`classify_verdict()`는 FTLR 하나만 인자로 받으므로 구조적으로 이 값들을 볼 수 없다.
또한 이 세 값은 **한계 L5와 함께 읽어야 한다** — distractor를 자유 공간으로 읽는
코드 경로가 다섯 곳 남아 있으므로 distractor 충돌이 미귀속 contact로 기록된다.
v7 셀의 outcome은 계보와도 `default` 셀과도 비교할 수 없다 (한계 L6): 동결 정책이
**학습한 적 없는 검출기**로 날고 있으므로 궤적 자체가 다르다.

## 판정 방향 — 이것은 예상된 결과다

`COLOR_SHORTCUT_CONFIRMED`는 **사전등록이 예측한 결과이고 실험 실패가 아니다** (사전등록 §5).
`_detect_rgbd`는 임계를 넘은 이미지 전체 양성 픽셀을 연결 성분 분석 없이 하나의
무게중심으로 축약하므로, 표적과 distractor가 동시에 보이는 프레임에서는 구조적으로
반드시 실패한다. 이 실험의 값어치는 검출기 개선이 아니라 **결함의 정량화**에 있다.
반대로 `DETECTOR_ROBUST_TO_DISTRACTORS`가 나오면 그것이 놀라운 결과이며, 그 경우 축하하기 전에
distractor가 실제로 렌더·검출됐는지를 먼저 재확인한다.

## 게이트 0 — 구현 타당성 (판정보다 먼저)

| 항목 | 결과 | 근거 |
|---|---|---|
| 0.1 기본 off bit-identical | PASS | 10개 테스트 (`test_navrl_distractors.DefaultOffIsUnchanged`, `test_navrl_distractors.DefaultOffIsBitIdentical`) + 텔레메트리 게이트 정적 검사 |
| 0.2 decoupling 거부 | PASS | 5개 테스트 + 전 셀 카메라 해상도 경로 확인 |
| 0.3 N=0 계보 재현 (`default_n0`) | PASS | 허용오차 ±3.75 pp, 초과: 없음 |
| (참고) `v7_n0` | 게이트 아님 | descriptive within-detector reference; no lineage anchor exists for v7 at this condition and operating point, so nothing is gated on it |

허용오차 근거: two-sided Bonferroni (3 outcome rates) 95% band on the difference of two independent proportions at n1 = n2 = 2049, evaluated at the worst-case p = 0.5: z(0.05/3) x sqrt(0.25/2049 + 0.25/2049) = 2.39398 x 1.5620 pp = 3.739 pp, rounded up

| outcome | N=0 셀 | 계보(seed 421) | 차이 |
|---|---:|---:|---:|
| capture | 72.57% | 70.52% | +2.05 pp |
| crash | 17.42% | 19.81% | -2.39 pp |
| timeout | 10.00% | 9.66% | +0.34 pp |

계보 참조: `results/navrl_ref5in_sensor_fidelity_seed421/cells/baseline/70bars.json` (sha256 `57d0bbef819ad0f7…`, seed 421). 두 셀은 서로 다른 seed의 독립 표본이므로
동치 비교가 아니라 위 허용오차 내의 일치로 판정한다.

**`v7_n0`에는 계보 기준점이 없다.** 이 체크포인트의 기존 70막대
항법 셀은 전부 내장 segmenter로 돌았고, 유일한 v7 실행
(`results/navrl_detector_domain_shift_v7/`)은 205막대·thr 0.55의 **오프라인** 프레임
수준 스크리닝이라 조건도 동작점도 다르다. 따라서 그 셀은 **기술적(descriptive)**이며
v7 자신의 FTLR 추이를 읽는 기준으로만 쓰고 아무것도 게이트하지 않는다.

## 고정된 조건과 두 축

- 8셀 전부 동일: camera 160×90, **detect 160×90(= camera, decoupled 금지)**, `min_pixels=2`, 검출 거리 20.0 m(미설정=기본값), 70 bars, 목표 22.5–28.0 m, seed 479, deterministic/original, governor off, appearance·지연·거리오차 0.
- **축 1(검출기 고정 시)**: `NAVRL_DISTRACTOR_COUNT`만 다르다. 각 검출기의 네 환경을
  모든 쌍으로 비교해 확인한다.
- **축 2(distractor 수 고정 시)**: `NAVRL_DETECTOR_CHECKPOINT`, `NAVRL_DETECTOR_THRESHOLD`, `NAVRL_V2_ALLOW_DETECTOR_THRESHOLD_MISMATCH`만
  다르다. 세 변수는 하나의 요인 수준이다 — 아티팩트와 그 동작점, 그리고 그 동작점이
  필요로 하는 좁은 override. 각 N에서 두 환경을 비교해 확인한다.
  두 방향을 **따로** 검사한다: 한 번에 뭉뚱그리면 검출기를 따라 움직인 엉뚱한 변수가
  '검출기 축의 일부'로 변명될 수 있다.
- 실행 후에는 결과 문서로 같은 두 방향을 다시 확인한다: 검출기 안에서는 `condition`의
  `distractor_count`만, N 안에서는 evaluation contract의 `detector_threshold`/`detector_checkpoint_sha256`만 움직여야 한다.
- **담요식 `NAVRL_V2_FORCE`는 어느 셀에서도 쓰지 않는다.** `default` 셀은
  동결 체크포인트를 그것이 학습된 센서 조건에서 평가하므로 override 없이 preflight가
  통과해야 한다. `v7` 셀은 정책이 본 적 없는 동작점(thr 0.7)에서 돌므로 evaluator가 **반드시** 한 필드를 문제
  삼는다 — force 없는 실행이 `cfg_detector_threshold: checkpoint=0.55 expected=0.7` **한 줄로만** 거부되는
  것을 실행 시점에 증명한 뒤에야 좁은 `NAVRL_V2_ALLOW_DETECTOR_THRESHOLD_MISMATCH`를 적용한다.
  두 줄이거나 다른 필드면 중단한다.

## 권한

이 실험은 **결정 권한이 없다**. P2 STRICT FAIL·D1 FAIL을 소급 변경하지 않고 P3를
해제하지 않으며, **v7 offline gate의 8/8 PASS도 소급 변경하지 않는다** — 그 게이트는
distractor가 없는 조건에서 frame precision 0.99766을 측정했고 그 조건 안에서 유효하다.
본 실험은 같은 아티팩트를 같은 동작점(thr 0.700)에서 **그 조건 밖**에서 잰다. 두 결과는
모순이 아니라 서로 다른 질문이다. 결과가 어떻든 **검출기 교체를 승인하지 않는다**
(별도 사전등록 필요).

- 정책 체크포인트 SHA-256 `197ea26999d6bb9c…`
- 검출기 요인 수준:
  - `default`: 내장 `AppearanceTargetSegmenter` (아티팩트 없음, 3R − 2G − 2B − 0.9), 동작점 thr 0.55, 판정 셀 `default_n5`
  - `v7`: 아티팩트 `artifacts/navrl_target_detector_v7_confirmatory.pth` (sha256 `85c7974bcd85c627…`), 동작점 thr 0.7, 판정 셀 `v7_n5`
- 품질 게이트: 판정 15개, 위임 0개, 실패 0개

## 한계 (사전등록 §6)

- L1: distractor가 **정적**이다. 실기의 움직이는 오탐(새 등)은 범위 밖이다.
- L2: 색이 표적과 **동일**하다. 색 거리에 따른 성능 곡선은 범위 밖이다.
- L3: 카메라 해상도 160×90에서 잰다. 고해상도에서의 FTLR은 다를 수 있다.
- L4: 단일 정책·단일 seed·70막대 1조건.
- L5: distractor가 일부 코드 경로에서 자유 공간으로 남는다. 자산 배열이 [target?][distractors...][bars...]이고 _bar_offset이 distractor를 건너뛰도록 넓혀졌으므로 [_bar_offset : _bar_offset + n_bars_active]를 읽는 지점들은 distractor를 보지 못한다. FTLR을 직접 오염시키는 두 곳만 고쳤다 — 드론 스폰 clearance(navrl_task.py:~1912)와 표적 경로 planner(~5858). 고치지 않은 다섯 곳: 정적 goal 배치(~4102), recovery clearance(~6058), bar-contact probe(~3081, ~3127). 결과적으로 distractor 충돌은 미귀속 contact로 기록되고 정적 goal이 distractor 안에 놓일 수 있다. **capture/crash/timeout 원값을 해석할 때 이 사실을 반드시 함께 읽어야 한다** — 그 값들은 §4에서 이미 판정 대상이 아니라 보조 보고다.
- L6: **검출기 간 FTLR을 빼서 비교하지 않는다(§3-c).** 동결 정책이 각 검출기의 출력을 보고 비행하므로 두 검출기에서 궤적이 달라지고, 따라서 프레임 분포 — 거리·베어링·가림, 특히 **표적과 distractor가 동시에 보이는 빈도** — 가 달라진다. FTLR은 바로 그 분포 위에서 정의되므로 `FTLR_v7 − FTLR_default`는 검출기 강건성과 궤적 분포를 뒤섞은 값이고, 이 셀들로는 둘을 분리할 수 없다. 따라서 그 차이를 **계산하지도 게재하지도 않는다.** 반면 각 검출기의 FTLR은 그 검출기가 실제로 만든 프레임 위에서 계산되므로 **한 검출기 안에서 N에 따른 비교는 유효하다** — 그것이 이 실험의 1차 지표다. capture/crash/timeout도 같다: v7 셀의 값은 계보와도, default 셀과도 비교하지 않는다.
