# 사전등록 — 센서 모델 충실도 (eval-only)

작성: 2026-08-22, **어떤 평가 arm도 실행하기 전**. 결과를 본 뒤 임계·seed·arm·판정 규칙을
변경하지 않는다.

선행: `WORKLOG.md` 2026-08-22 두 항목(각해상도 사슬 도출, 하드웨어 관문),
`VERIFICATION.md`(실행 authority), `results/navrl_ref5in_camera_range_control_seed367/`(seed 367).

---

## 1. 무엇을 고치려는 것인가

P2/D1의 진단된 병목은 **장거리 CV에서의 초기 표적 미관측**이다. seed 367은 camera range
20→28 m에서 timeout `55.80 → 18.16%`(−37.65 pp)를 보였고 이는 인과 대조로 확인됐다.

그런데 1단계에서 **그 처방이 틀렸음이 드러났다.** seed 367은 광학을 바꾸지 않았다 — 두 arm의
소스 스냅샷이 md5 동일이고 소프트웨어 far-plane만 20→28 m로 풀렸다. 즉 **정보의 가치를 증명한
것이지 하드웨어 실현성이 아니다.** 그리고 시뮬레이터 자신의 기하로는 28 m 광축 검출이 불가능하다.

실제 결함은 **센서 모델이 세 방향으로 동시에 틀려 있다**는 것이다.

| # | 결함 | 현재 | 실기 |
|---|---|---|---|
| 1 | 해상도 | 160×90 @ 87° → fx 84.3 px/rad | AR0234급 1920×1200 @ 82° → fx 1104 (13.1배) |
| 2 | 검출 임계 | 면적 2 px² ≈ 지름 1.6 px | Johnson 검출 하한 2–2.5 px, CNN 신뢰 8–10 px(하늘) |
| 3 | 거리 | 해석적 정확값, 오차 **0** | 28 m 스테레오 시차 1.2–2.4 px — 사실상 측정 불가 |

**본 실험은 #1과 #2만 다룬다.** #3은 별개 사전등록이 필요하며 §8에 그 이유를 적는다.

## 2. #1과 #2를 분리하지 않는 이유

해상도만 올리면 임계가 여전히 Johnson 기준 미만이라 **실패 모드만 옮긴다** — 더 조밀한 센서에서
여전히 존재하지 않는 검출기를 모델링하게 된다. 임계만 올리면 현재 해상도에서 표적이 28 m에
0.62 px²이므로 검출이 **전면 붕괴**한다. 두 값은 물리적으로 한 쌍이며, 따라서 본 실험의 조작 축은
**"센서 모델 충실도" 하나**다. 이는 `VERIFICATION.md` fail-closed 3(한 run에서 여러 축 동시 변경
금지)을 위반하지 않는다 — 두 knob은 하나의 물리량(각해상도 대비 검출 가능성)의 두 얼굴이다.

이 논리를 결과를 본 뒤에 뒤집지 않는다. 만약 결과가 나쁘게 나와도 "해상도만 따로 볼걸"이라고
사후에 분해하지 않으며, 그런 분해가 필요하다고 판단되면 새 사전등록을 쓴다.

## 3. 임계와 해상도의 짝은 유도된 값이다

검출 임계는 **면적**이다(`navrl_perception.py:1207-1210`, `mask.sum(dim=(1,2))`). 지름 `d`와
면적의 관계는 `A = π(d/2)² = 0.785 d²`.

문헌 기준: 단일프레임 CNN이 소형 멀티로터를 **하늘 배경**에서 신뢰성 있게 검출하는 하한이
지름 8–10 px(Det-Fly, LRDDv3, AI-TOD). 잡동사니 배경은 15–20 px(Det-Fly 하늘 88.3 → 도심 62.0).

**지름 8 px = 면적 50 px²를 임계로 채택한다.** 하늘 배경 기준 하한이며, 잡동사니를 쓰지 않는
이유는 현재 렌더가 잡동사니 배경을 모델링하지 않기 때문이다(§7 한계).

그 임계에서 교전 거리 계약(목표 22.5–28 m)을 만족하려면:

| 해상도 | fx [px/rad] | 28 m 면적 | 50 px² 신뢰 거리 |
|---|---|---|---|
| 160×90 (현재) | 84.3 | 0.62 px² | 3.1 m |
| 480×300 | 253 | 5.8 px² | 9.5 m |
| 1280×800 | 675 | 41 px² | 25.3 m |
| **1920×1200** | **1013** | **92 px²** | **38.0 m** |

`fx = (W/2)/tan(87°/2)`. **1920×1200을 채택한다** — 28 m에서 92 px²로 임계의 1.8배 여유가 있고,
신뢰 거리 38 m가 교전 계약 상단 28 m를 덮는다. 1280×800은 25.3 m로 상단에 못 미친다.

## 4. 렌더 비용과 구현 — 명시적 설계 변경

실측(WORKLOG 2026-08-22): 8 GB에서 `envs × W × H ≲ 18.4 Mpx`이므로 1920×1200은 128 env에서
불가능하고 8 env로 내려가 셀당 37분(12.6배)이 된다.

그러나 같은 실측이 **광선 추적은 해상도에 대해 사실상 공짜**임을 보였다(픽셀 10배에 +0.8 ms).
비용은 전부 하류 — 합성 RGB 생성과 그 전체 해상도 이미지의 분할이다. 그 RGB는 표적을 평평한
순수 빨강으로 칠한 합성물이고 분할기가 `3R − 2G − 2B − 0.9`이므로, **appearance 교란이 0일 때
정보적으로 항등 왕복**이다.

따라서 **검출 해상도를 RGB/perception 해상도와 분리한다**(`NAVRL_DETECT_WIDTH/HEIGHT`).
이는 knob 조정이 아니라 **설계 변경**이므로 여기 명시한다. 채택 조건:

- **동등성 증명이 선행한다.** detect == camera일 때 현재 코드와 **bit-identical**이어야 한다
  (`torch.equal`). 이것이 실패하면 본 실험을 진행하지 않는다.
- detect ≠ camera이고 appearance 교란이 **0이 아니면 fail-closed로 거부**한다. 동등성이 그 조건
  아래서만 성립하기 때문이다.
- actor 관측은 모든 조합에서 **898-D**를 유지해야 한다. 아니면 중단한다.

## 5. 실험 계약

| 항목 | 값 |
|---|---|
| 정책 | frozen ref5in D1 ep1900, SHA `197ea26999d6bb9cf23c4e5a55acbe945f89985e2384687d60ab1dbae66a278e` |
| **학습 없음** | 본 실험은 **평가 전용**이다. 적응 학습은 이 사전등록의 권한 밖이다 |
| 아레나 | 70 bars |
| 목표 거리 | 22.5–28 m (체크포인트 계약 강제) |
| **평가 seed** | **421** — 전수 검색 사용 이력 0건 |
| 에피소드 | arm당 2,049 |
| action | deterministic, governor `off`, reflection_mode `original` |
| **arm A (baseline)** | `NAVRL_DETECT_*` = 160×90, `NAVRL_DETECTOR_MIN_PIXELS=2` — **현행 그대로** |
| **arm B (fidelity)** | `NAVRL_DETECT_*` = 1920×1200, `NAVRL_DETECTOR_MIN_PIXELS=50` |
| 고정 | camera(RGB) 해상도 160×90 양 arm 동일, `camera_min_target_pixels=1`, `NAVRL_DETECTOR_MAX_RANGE=20.0`(**변경하지 않는다**), appearance 전부 0, 잡음·드롭아웃·지연 전부 0 |

**`detector_max_range`를 20 m로 고정하는 것이 중요하다.** seed 367이 그것을 바꾸면서 actor의
표적 토큰까지 재정규화했고(`rel_pos / max_camera_range`, `navrl_perception.py:1574,1578`) 그것이
교란이었다. 본 실험은 range를 건드리지 않으므로 **토큰 정규화가 양 arm에서 동일**하다. 조작은
"같은 20 m 안에서 검출이 얼마나 정직해지는가"이다.

## 6. 판정 — 결과를 보기 전에 확정

### 게이트 0 — 구현 타당성 (판정보다 먼저)

§4의 세 조건(bit-identical 동등성, fail-closed, 898-D)이 **전부 통과**해야 한다. 하나라도 실패하면
판정은 **`FAIL_CLOSED_IMPLEMENTATION`**이며 센서 모델에 대한 어떤 주장도 하지 않는다.

### 1차 측정량

정직해진 센서에서 동결 정책의 **획득**과 **결과**가 어떻게 변하는가.

| 측정량 | 정의 |
|---|---|
| never-acquired 비율 | 에피소드 중 표적을 한 번도 획득하지 못한 비율 |
| 최초 획득 스텝 | 획득한 에피소드의 중앙값·p90 |
| target_hidden_fraction | 프레임 기준 |
| capture / crash / timeout | 원값 |

### 판정 규칙

| 판정 | 조건 |
|---|---|
| **FIDELITY_COST_CONFIRMED** | arm B의 never-acquired가 arm A 대비 **+10.00 pp 이상** 증가 |
| **FIDELITY_NEUTRAL** | 변화가 **±3.00 pp 이내** |
| **INCONCLUSIVE_SENSOR_FIDELITY** | 그 외 |

**방향에 주의**: 정직한 센서는 검출을 **더 어렵게** 만들 것으로 예상한다(임계가 25배 올라감).
따라서 arm B가 나빠지는 것은 **실패가 아니라 예상된 결과**이며, 그 크기가 곧 "지금까지의 성적 중
얼마가 존재할 수 없는 센서 덕분이었는가"의 추정치다. 이 실험의 가치는 개선이 아니라 **정직한
기준선의 확립**에 있다.

capture/crash/timeout은 **원값으로 보고하되 판정에 쓰지 않는다** — 동결 정책은 부정직한 센서로
학습됐으므로 정직한 센서에서의 성능 저하는 정책의 결함이 아니라 계보의 결과다.

## 7. 알려진 한계 — 판정 전에 명시

- **(L1) 잡동사니 배경 미모델링.** 렌더가 표적을 평평한 순수 빨강으로 칠하고 분할기가 색 규칙이라
  픽셀 클래스가 자명하게 분리된다. 따라서 지름 8 px 임계는 **하늘 배경 기준**이며 실기의 도심·
  수목 배경(15–20 px)보다 관대하다. 본 실험 결과는 여전히 **낙관 편향**이다.
- **(L2) 모션 블러 미모델링.** 5인치 기체가 고 yaw rate에서 10 ms 노출이면 12 px 표적이 여러
  픽셀로 번진다. 현재 모델에 없다.
- **(L3) 거리 오차 0.** `navrl_detector.py:130`이 정확한 해석적 교차 거리를 쓰고
  `NAVRL_RANGE_ERROR_M=0`이다. 정책은 임의 거리에서 오차 0의 거리를 받는다. 실기에서는
  28 m 스테레오 시차가 1.2–2.4 px로 측정 불가다. **이 실험은 그것을 고치지 않으므로, 결과는
  "거리는 여전히 공짜로 주어진 상태에서의 검출 충실도"만 말한다.**
- **(L4) 단일 정책·단일 seed·70막대 1조건.** 계보나 밀도 전반으로 일반화하지 않는다.
- **(L5) 임계 50 px²는 문헌 중앙값이지 이 시스템에서 측정된 값이 아니다.** 감이 아니라 유도된
  값이지만 여전히 외부 기준의 이식이다.

## 8. 이 실험이 주는 권한과 주지 않는 권한

- `FIDELITY_COST_CONFIRMED`가 나오면 **정직한 센서에서의 재학습 사전등록을 쓸 자격**이 생긴다.
  구현·실행 권한은 아니다.
- **P2 STRICT FAIL / D1 FAIL / P3 BLOCKED를 변경하지 않는다.** 어떤 결과도 이들을 소급 변경하거나
  P3를 해제하지 않는다.
- seed 367의 공식 판정을 **소급 변경하지 않는다.** §1의 재해석은 한계 기록이지 재판정이 아니다.
- 거리 충실도(#3)는 **별개 사전등록**이 필요하다. 그것을 함께 바꾸면 축이 둘이 되고, 무엇보다
  거리 오차 모델은 검출 임계와 달리 물리적으로 독립된 양이다.

## 9. 하지 않을 것

- 학습, 임계·해상도 스윕, 결과를 본 뒤의 임계·seed·arm 변경
- `detector_max_range` 변경 (토큰 재정규화 교란)
- appearance 교란을 켠 채로 detect ≠ camera 실행
- `aerial_gym/config/robot_config/**`·`resources/robots/**` 편집 (provenance freeze)
- dirty runtime에서의 실행

## 10. 기록 요건

`results/navrl_ref5in_sensor_fidelity_seed421/{cells/{baseline,fidelity},summary.{json,md}}`.
동등성 증명 산출물, 무효·실패 실행, VOID 사유. 요약에
`p2_verdict_changed: false`, `d1_verdict_changed: false`, `p3_unlocked: false`,
`decision_authority: "none"`.

---

## 5-b. 좁은 provenance override (2026-08-22, **어떤 arm도 실행하기 전** 기록)

§5는 "provenance override 없이 통과해야 한다"고 적었다. `preflight`에서 그것이 불가능함이
드러났다:

```
[eval_v2] REFUSING: v2 contract mismatch:
  cfg_detector_min_pixels: checkpoint=2 expected=50.0
```

`eval_navrl_v2_density_sweep.sh:675`가 `cfg_detector_min_pixels`를 v2 provenance 게이트의 `want`
집합에 넣는다. 즉 평가기는 검출 임계가 **체크포인트가 학습된 값**과 같기를 요구한다. arm A(=2)는
통과하고 arm B(=50)는 필연적으로 불일치한다.

**이 불일치는 결함이 아니라 실험의 정의다.** 본 실험은 "학습 때와 다른 임계에서 동결 정책을
평가한다"이며, 게이트는 정확히 그 차이를 잡아낸 것이다. §5의 "override 없이"는 이 게이트의 존재를
모르고 쓴 문장이며, 측정 전에 정정한다.

**채택: 좁은 단일 필드 override.** 저장소의 기존 패턴(`run_navrl_ref5in_cv_heading_near_open.py:107-120`
`verify_narrow_override`)을 그대로 쓴다.

- arm B에 대해 **force 없이** preflight를 먼저 돌려 `returncode == 2`임을 확인한다.
- 불일치 라인 집합이 **정확히** `["cfg_detector_min_pixels: checkpoint=2 expected=50.0"]` 하나임을
  요구한다. 두 개 이상이거나 다른 필드가 섞이면 **중단**한다.
- 그 증명 이후에만 `NAVRL_V2_FORCE=1`로 preflight를 재실행해 통과를 확인한다.
- arm A는 override를 쓰지 않으며, 쓰지 않았음을 요약에 기록한다.
- 사용된 단일 불일치 문자열을 `summary.json`의 `narrow_provenance_override`에 고정한다.

이는 게이트를 **느슨하게 하지 않는다.** 담요식 `NAVRL_V2_FORCE`는 다른 모든 불일치도 함께
가려주지만, 이 절차는 실행 시점에 불일치가 정확히 그 한 필드임을 증명하므로 오히려 더 엄격하다.
새 allow-flag를 런타임 소스에 추가하는 대안보다도 낫다 — 소스 변경이 없고, 검증이 실행 시점에
일어난다.

§6의 게이트·임계·판정 규칙은 **변경하지 않는다.**

## 5-c. 기록되지 않는 두 가지 (2026-08-22, 실행 전 기록)

1. **first-acquisition p90은 산출 불가.** `navrl_task.py first_acquisition_payload()`가
   `first_visible_step_mean`과 하위 중앙값만 내보내고 per-outcome 히스토그램은 결과 JSON에 쓰지
   않는다. §6이 요구한 p90은 기록된 어떤 필드에서도 유도할 수 없다. `null`과 사유
   `FIRST_ACQUISITION_P90_UNAVAILABLE`로 기록한다. 런타임 소스를 고치는 것은 이 평가 전용
   사전등록의 권한 밖이다.
2. **검출 해상도가 영수증에 증명되지 않는다.** 평가기가 `detect_width/height`를 영수증에도
   `v2_evaluation_contract`에도 기록하지 않으므로, **증명 가능한 arm 구분자는
   `detector_min_pixels` 하나뿐**이다. 요약에 `detect_resolution_not_recorded_by_evaluator: true`로
   명시한다. 두 값이 한 쌍으로 움직인다는 것은 런처 코드와 로그로만 확인되며, 이는 이 실험의
   provenance 상 약점이다.
