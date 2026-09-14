# ref5in P2 held-out 70-bar decision — STRICT FAIL

판정일: 2026-08-13
평가 계약: seed 313 · 70 bars · deterministic · original reflection · governor off ·
target U[0.3,1.5] m/s · goal U[6,28] m · requested 2,049 episodes

## 결과

| 지표 | count / 2,049 | 비율 | 사전등록 gate | 판정 |
|---|---:|---:|---:|---|
| capture | 1,399 | **68.28%** | ≥65% | PASS |
| crash | 536 | **26.16%** | ≤33% | PASS |
| timeout | 114 | **5.56%** | ≤5% | **FAIL** |

timeout은 허용 가능한 최대 102건보다 12건 많다. Wilson 95% CI는 capture 66.23–70.26%,
crash 24.30–28.11%, timeout **4.65–6.64%**다. 5%가 interval 안에 있어 효과 크기는
경계에 가깝지만, 사전등록 gate는 point estimate/count로 판정하므로 전체 verdict는 FAIL이다.

## 어디서 어려웠나

- 초기거리 사분위 capture는 **80.90 → 73.51 → 66.84 → 53.17%**로 크게 내려갔다.
- 표적속도 사분위 capture는 **68.79 / 70.65 / 67.67 / 66.51%**로 상대적으로 평평했다.
- 패턴은 CV 66.01%, waypoint 70.49%였다.
- crash 536건 중 bar contact 416건(77.61%), arena out-of-bounds 120건(22.39%)이었다.
- contact 순간 실제 속도 평균은 1.109 m/s, requested/executed stopping margin 평균은
  −1.098 m였다. governor-off 진단값이므로 제동 여유가 부족한 접촉이 남아 있다.
- timeout 114건은 전부 정확히 action 600에서 발생했다. legacy 601-step 회귀가 아니다.

현재 자료는 **표적 속도보다 긴 초기 거리/경로 길이가 더 강한 병목 후보**임을 보여준다. 다만
현행 distance strata에는 capture만 있고 crash/timeout breakdown이 없어 인과를 확정할 수 없다.

## 결정

- P3 seed 211 full-budget 학습: **시작 금지**
- legacy ep500 anchor: primary P2가 통과할 때만 실행하기로 했으므로 **미실행**
- 같은 seed를 반복하거나 timeout gate를 사후 6%로 완화: **금지**
- 다음 허용 작업: outcome별 distance/speed/pattern telemetry 추가 → 별도 diagnostic evaluation →
  long-range timeout인지 contact/OOB인지 분리 → 그 결과로 intervention smoke를 사전등록

기계 증거는 `attestation.json`, 원자료는 `ref5in/70bars.json`, evaluator receipt는
`ref5in/70bars.receipt.json`이다. attestation은 P1c checkpoint, runtime source map, Python
environment, evaluator, result/log/snapshot SHA를 다시 검증하며 verdict에 legacy anchor를 사용하지 않는다.
