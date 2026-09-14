# 사전등록 — range 오차 대리 측정 (2026-09-09)

거리 GT 없이 **상대** range 오차의 크기를 재는 두 대리 측정을 실행 전에 등록한다.
목적은 range 오차 모델을 만드는 것이 **아니라**, 실사 거리 GT를 확보하는 비용을 치를 가치가
있는지를 판정하는 것이다. MIDGARD는 라이선스 미제공으로 배제됐고 대체 자료는 조사 중이다.

상위 맥락: [`perception_to_interception_plan_2026-09-09.md`](perception_to_interception_plan_2026-09-09.md) §1-A.

---

## 0. 왜 대리가 성립하는가, 그리고 어디까지만인가

크기 기반 단안 range 추정기는 `range = k / s` (s = 겉보기 크기, k = 초점거리 × 기체 실제 크기)다.
따라서 **상대** range 오차는 크기 비의 역수로 그대로 나온다.

    range_hat / range_true = s_true / s_hat

`k`가 소거되므로 **intrinsics도 기체 실제 크기도 거리 GT도 필요 없다.** 다만 이렇게 얻는 것은
*상대* 오차뿐이며 절대 미터는 얻을 수 없다.

**대리가 못 보는 것**: `s_true`가 자세 때문에 변하는 성분. 쿼드로터를 정면/측면에서 보면 같은
거리에서도 겉보기 크기가 달라지고, 크기 기반 추정기는 이를 거리 변화로 오독한다. GT 박스는 그
변화를 이미 포함하므로 대리 A로는 보이지 않는다. 대리 B가 이 항을 부분적으로 회수한다.

---

## 1. 모집단과 입력 (실행 전 동결)

P8과 **동일한 모집단**을 쓴다. 새 데이터를 열지 않는다.

| 항목 | 값 |
|---|---|
| split | NPS **validation만**. test는 열지 않는다(S4가 유일한 개봉이었다) |
| 프레임 | 단일-GT(`exactly_one_GT`) 1,310프레임. 다중-GT 986프레임 제외 |
| GT | `detector_runs/results/nps_detfly_joint_final/nps_val_sequence.jsonl`의 `ground_truth_xyxy` |
| 예측 | `.../candidate_motion_transformer_v2/validation_predictions.jsonl.gz`의 `box_xyxy` (선택된 후보) |
| 크기 정의 | `sqrt(box 면적)` — P8의 `size_definition`과 동일 |
| 크기 bin | P8 bin 경계 그대로: `[0,8) [8,16) [16,32) [32,64) [64,∞)`. bin 1·2·3만 지원 |
| 시간 간격 | 103.4 ms(공칭). 연속성 판정은 P8의 `max_segment_gap_seconds = 0.5`를 그대로 |
| 환경 | `detector_runs/venv` (규칙 0.1). receipt에 runtime fingerprint 기록 |

모든 입력 파일의 SHA-256을 receipt에 기록한다. `test_used: false`를 명시한다.

---

## 2. 대리 A — 검출기 박스 크기 오차 (하한)

**대상 프레임**: 위 모집단 중 선택기가 후보를 선택했고 `hit`(any-GT IoU ≥ 0.3)인 프레임.
선택 없음·false lock·no lock 프레임은 크기 비가 정의되지 않으므로 제외하고, 제외 수를 보고한다.

**추정량**

    ratio_A = s_gt / s_pred          (= range_hat / range_true, 크기 기반 추정기 가정 아래)
    rel_err_A = ratio_A - 1

**보고**: bin 1·2·3 각각과 전체에 대해 `n`, median, IQR, P5, P95, |rel_err| 의 P50·P95.
분포는 heavy tail이 예상되므로 평균이 아니라 **분위수**를 1차 지표로 쓴다(P8과 같은 방침).

**반복 단위**: 프레임이 아니라 **source video**(7개 clip)를 반복 단위로 하는 clip 단위 요약을
함께 낸다 — 계획서 "평가 계약"의 규칙이다. clip별 median의 분산을 함께 적는다.

---

## 3. 대리 B — 겉보기 크기의 시간 잔차 (자세 항의 상한)

**가정(명시)**: 참 거리는 0.5 s 규모에서 매끄럽게 변한다. 그 규모의 고주파 성분은 거리 변화가
아니라 **겉보기 크기 변동**(자세 + 주석 잡음)이다.

**절차**: 각 clip에서 단일-GT 프레임의 `log(s_gt)` 시계열을 만들고, 중앙값 필터로 매끄러운 성분을
뺀 잔차를 본다.

    resid = log(s_gt) - median_filter(log(s_gt), W)

**W(동결)**: 3, 5, 9 샘플(≈ 0.31 / 0.52 / 0.93 s). **주 지표는 W = 5.**
세 값 모두 보고하며, 결과를 본 뒤 다른 W를 추가하지 않는다.

창 안의 이웃 샘플 간 시간 간격이 0.5 s를 넘으면 그 창은 무효로 하고 잔차를 계산하지 않는다.
무효 창 수를 보고한다.

**보고**: `exp(resid) - 1`의 P50·P95를 bin별·clip별로. 이것이 상대 range 오차의 자세 성분에 대한
**상한**이다 — 주석 잡음이 섞여 있고, 실제로 빠르게 기동하는 표적의 참 거리 변화도 섞이기 때문이다.

---

## 4. 판정 규칙 (결과를 보기 전에 동결)

대리 A와 대리 B의 P95를 결합한 값을 `E95`로 쓴다(제곱합의 제곱근이 아니라 **둘 중 큰 값** —
두 항의 독립성을 주장할 근거가 없으므로 보수적으로 잡는다).

| E95 | 판정 | 다음 행동 |
|---|---|---|
| ≤ 15 % | `RANGE_ERROR_BOUNDED_SMALL` | 1-A를 닫는다. range 오차 주입과 새 관측 계보(I4)를 열지 않는다 |
| ≥ 30 % | `RANGE_ERROR_LARGE` | 실사 거리 GT 확보가 정당화된다. I4(uncertainty-aware 계약)도 후보로 올린다 |
| 그 사이 | `RANGE_ERROR_INCONCLUSIVE` | 구간으로만 보고하고, 자료 조사 결과에 따라 재판정 |

**비유의는 동등성이 아니다.** `BOUNDED_SMALL`은 "오차가 없다"가 아니라 "이 두 대리가 보는 범위에서
15 % 아래"라는 뜻이며, 대리가 못 보는 느린 bias는 여전히 미측정이다.

---

## 5. 이 측정으로 하지 않을 것

- **P9 모델에 range 항을 추가하지 않는다.** 이 측정은 판정용이고, 주입은 별도 사전등록 대상이다.
- **절대 미터 오차를 만들지 않는다.** intrinsics가 없다.
- **test를 열지 않는다.** validation만.
- **결과를 본 뒤 bin·W·문턱을 바꾸지 않는다.**
- 시뮬레이터에 대한 함의를 이 문서에서 확정하지 않는다 — 시뮬 표적은 반지름 0.15 m **구**라
  자세에 따른 겉보기 크기 변동이 원리적으로 없고, 시뮬 range는 크기 역산이 아니라 해석적
  ray-cast depth(`navrl_perception.py:1662`)다. 따라서 대리 B가 크게 나오더라도 그것을 그대로
  현재 시뮬에 주입하는 것은 정합적이지 않다. 그 처리는 별도 결정이다.

---

## 6. 산출물

`results/perception_range_proxy_2026-09-09/`에 `report.json`, `README.md`, `receipt.json`.
도구는 `tools/measure_range_error_proxy.py`. WORKLOG에 판정과 함께 기록한다.
