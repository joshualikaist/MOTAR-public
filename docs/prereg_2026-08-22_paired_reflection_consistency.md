# 사전등록 — paired-reflection consistency 학습 개입 (2 arm A/B)

작성: 2026-08-22, **어떤 학습 arm도 실행하기 전**. 이 문서는 결과를 본 뒤 수정하지 않는다.
계수·seed·게이트·판정 규칙은 아래에서 확정되며, 측정 후 완화·재정의하지 않는다.

선행 근거: `docs/prereg_2026-08-21_n1_real_frame_reflection_audit.md`(N1 사전등록),
`results/navrl_ref5in_reflection_audit_seed373/summary.md`(N1 결과, 커밋 `9765020`),
`VERIFICATION.md`(실행 authority), `docs/diagnostic_synthesis_2026-08-21.md`.

---

## 1. 무엇이 확정됐고, 무엇이 아직 아닌가

N1이 확정한 것 (frozen ref5in D1 ep1900, 실제 프레임 15,488, 품질게이트 Q1–Q9 전부 통과):

| | 값 |
|---|---|
| `median(conj_err_lat)` | **1.454** (사전등록 게이트 0.30) |
| lateral sign agreement | **2.49%** (게이트 60%) |
| `mean π(o)[1]` / `mean π(Mo)[1]` | **−0.623 / −0.763** — equivariance면 부호가 반대여야 한다 |
| 맥락 의존성 | **없음.** 표본 충분한 7개 셀 전부 confirmed, median 1.42–1.52 |
| 정규화기 대칭화로 제거되는 비율 | **11.7%** — 나머지는 네트워크 가중치에 있다 |

**아직 아닌 것**: chirality가 성능을 해친다는 증거는 **없다**. N1은 outcome을 측정하지 않았고(L3),
2026-08-02 legacy 계보의 mirror outcome 대조는 capture **−0.81 pp (95% CI −2.78..+1.17)**로
차이를 **검출하지 못했다**. 본 실험은 그 공백을 메우는 것이 아니라, 개입이 (a) 결함을 실제로
제거하는가 (b) 성능에 얼마를 청구하는가를 분리해 측정한다.

## 2. 우선순위 — 이 실험이 병목 수리가 아님을 먼저 기록한다

`VERIFICATION.md`가 규정하는 P2 STRICT FAIL / D1 FAIL의 진단된 병목은 chirality가 **아니다**.
장거리 CV에서의 **초기 표적 미관측**(camera 20 m vs goal 22.5–28 m)이며, seed 367 인과 대조에서
camera 20→28 m로 timeout `55.80% → 18.16%`(**−37.65 pp**)가 확인됐다.

즉 프로젝트에는 이미 **인과가 확인된 처방(camera range)**이 있고, chirality는 **성능 근거가 없는
별개 결함**이다. 본 실험을 먼저 하는 것은 "인과가 확인된 것"보다 "메커니즘적으로 흥미로운 것"을
앞세우는 선택이며, 그 선택을 여기 명시적으로 기록한다. 이 문서는 그 선택을 정당화하지 않는다 —
**실행 여부는 별도 결정이다.**

## 3. 알려진 한계 — 판정 전에 명시

- **(L1) 손실과 지표는 다른 스케일에 있다.** `reflection_equivariance_loss`는
  `mus`(pre-tanh, 무계)에 걸리고(`early_stop_a2c_agent.py:462`), N1 지표는
  `tanh(mu_scale ⊙ mus)`(post-tanh, `[-1,1]`)에서 읽는다. `mu_scale`이 축별이고 tanh가 홀함수라
  `M(tanh(s⊙mu)) = tanh(s⊙M(mu))`이므로 **영점은 일치**하지만, 포화 영역의 불일치는 손실에서
  과대 가중된다. → 계수를 감으로 정하지 않고 §5의 측정 규칙으로 정한다.
- **(L2) 장애물 토큰은 반사 시 재배열되지 않는다.** 거리 오름차순이라 근사적으로 옳고,
  mode-probe가 그 영향을 `slot_permutation_max_abs = 0.0078`로 측정했다. 재측정하지 않는다.
- **(L3) 보조 forward는 update당 정책 forward를 1회 추가한다.** 따라서 두 arm의 **wall-clock은
  다르다.** 비교는 **샘플 수로 정합**하며 시간으로 정합하지 않는다.
- **(L4) 단일 학습 seed, 단일 계수, 단일 예산.** 계수-반응 곡선이나 계보 일반화를 주장하지 않는다.
- **(L5) warm-start 계보.** 두 arm 모두 이미 D1 FAIL인 checkpoint에서 출발한다. 어떤 결과도
  P2/D1을 통과시키지 않으며 그렇게 해석해서도 안 된다.

## 4. 실험 계약

| 항목 | 값 |
|---|---|
| 출발 checkpoint (양 arm 동일) | ref5in D1 ep1900, SHA `197ea26999d6bb9cf23c4e5a55acbe945f89985e2384687d60ab1dbae66a278e` |
| robot | `navrl_ref5in_quad` (config SHA `ebb71802…`, URDF SHA `5c160b0d…` — provenance freeze) |
| **학습 seed (양 arm 동일)** | **383** — 전수 검색 사용 이력 **0건** |
| **평가 seed** | **389** — 전수 검색 사용 이력 **0건** |
| 예산 (양 arm 동일) | **1,000 epoch / 4.096M samples**, `--max_epochs`로 종료 |
| 조작 변수 | **`NAVRL_REFLECTION_COEF` 단 하나** |
| arm A (control) | `NAVRL_REFLECTION_COEF=0` |
| arm B (treatment) | `NAVRL_REFLECTION_COEF=c` (§5에서 확정) |
| 고정 | airframe · reward · horizon · representation · governor · 밀도 · LR · `NAVRL_LATERAL_BIAS_COEF=0` |
| 평가 | 각 arm 종단 checkpoint를 `tools/run_navrl_ref5in_reflection_audit.py`로 1셀씩, seed 389 |

`NAVRL_LATERAL_BIAS_COEF`(`lateral_batch_bias_loss`, `early_stop_a2c_agent.py:421`)는 **0으로
고정한다.** 이는 두 번째 레버이며, 한 run에서 두 축을 동시에 바꾸지 않는다(`VERIFICATION.md` fail-closed 3).
설계상으로도 부적합하다 — batch 평균 `mu[:,1]`만 0으로 눌러도 관측별 chirality는 그대로 남을 수 있다
(절반은 좌, 절반은 우로 돌면 평균은 0이지만 equivariance는 아니다). N1의 primary는 평균이 아니라
관측별 `conj_err_lat`이다.

한 번의 평가 rollout이 `70bars.json`(capture/crash/timeout)과 `reflection_audit.json`(chirality)을
**동시에** 산출하므로, 메커니즘과 성능을 같은 rollout에서 읽는다. 별도 평가 seed를 쓰지 않는다.

## 5. 계수 확정 규칙 — arm 실행 전에 측정으로 정한다

계수 스윕은 금지된다(`VERIFICATION.md` fail-closed 2). 대신 **arm 실행 전** 프로파일링 1회로
단일 값을 정하고, 그 결과를 이 문서에 추가 기록한 뒤 arm을 시작한다.

절차: frozen checkpoint에서 optimizer step **없이** update 크기 minibatch 64개를 통과시켜
`|a_loss|`(PPO surrogate)와 `reflection_equivariance_loss(mu, reflected_mu)`의 **중앙값**을 기록한다.

```
c = round_to_1_sig_fig( 0.10 * median(|a_loss|) / median(symmetry_penalty) )
```

`0.10`은 보조항의 **초기 기여를 정책 손실의 10%로 맞춘다**는 뜻이며, 결과를 보기 전에 고정한다.
1 유효숫자 반올림은 "측정으로 얻은 크기"이지 "튜닝된 값"이 아님을 강제한다.

프로파일링 산출물(두 중앙값, 계수, minibatch 수)은 §5-b로 **arm 실행 전** 이 문서에 기록한다.

## 6. 게이트 — 결과를 보기 전에 확정

### Gate 0 — 설계 타당성 (판정보다 먼저)

arm A(control)가 chirality를 **유지해야** 한다: `median(conj_err_lat) ≥ 1.00` **그리고**
sign agreement `≤ 0.20`.

실패 시 판정은 **`INCONCLUSIVE_CONFOUNDED_BY_ADAPTATION`**이며 arm B에 대해 아무 주장도 하지 않는다.
근거: control이 손실 없이도 chirality를 잃는다면 원인은 1,000 epoch의 적응 자체이지 손실이 아니다.

### Gate M — 메커니즘 (primary)

arm B: `median(conj_err_lat) ≤ 0.50` **그리고** sign agreement `≥ 0.70`.

임계 근거(사전 확정): 기준선은 1.454 / 0.0249다. N1의 `CHIRALITY_ABSENT` 구역(≤0.10 / ≥0.90)은
1,000 epoch 예산에서 비현실적이므로 채택하지 않는다. `0.50`은 기준선 대비 **66% 축소**,
`0.70`은 우연 수준 0.50보다 명확히 위이며 기준선 0.025와 겹치지 않는다. 두 조건은 AND다.

### Gate P — 성능 청구서 (guard)

arm B capture `≥` arm A capture `− 2.00 pp` **그리고** arm B crash `≤` arm A crash `+ 2.00 pp`.
동일 seed 389, 동일 셀에서 읽는다.

## 7. 판정 규칙

| Gate 0 | Gate M | Gate P | 판정 |
|---|---|---|---|
| FAIL | — | — | `INCONCLUSIVE_CONFOUNDED_BY_ADAPTATION` |
| PASS | FAIL | — | `REFLECTION_CONSISTENCY_INEFFECTIVE` |
| PASS | PASS | PASS | `REFLECTION_CONSISTENCY_EFFECTIVE` |
| PASS | PASS | FAIL | `MECHANISM_PASS_PERFORMANCE_REGRESSION` |

**null 결과는 결과다.** `INEFFECTIVE`가 나오면 그대로 기록하고, 계수를 올려 재시도하지 않는다 —
그것이 곧 금지된 스윕이다. 다음 계수를 시도하려면 새 사전등록이 필요하며, 그 문서는 왜 첫 계수가
틀렸다고 판단하는지를 결과와 독립적인 근거로 제시해야 한다.

## 8. 이 실험이 뒤엎을 수 없는 것

- **P2 STRICT FAIL · D1 FAIL · P3 BLOCKED는 변경되지 않는다.** 어떤 결과도 이들을 소급 변경하거나
  P3를 해제하지 않는다(`VERIFICATION.md` fail-closed 1·5).
- 성능 개선을 주장하지 않는다. Gate P는 **비회귀 guard**이지 개선 가설이 아니다.
- 2026-07-27 Ablation B(`NAVRL_REFLECTION_COEF=0.01` 기각)는 **잘못된 mirror 연산자** 위에서 돌았고
  2026-07-29에 무효화됐다. 그 기각은 현재 연산자에 대한 증거가 **아니며**, 동시에 본 실험의
  근거로 인용할 수도 **없다**. 계수 `0.01`을 재사용하지 않는 이유이기도 하다 — §5가 독립적으로 정한다.
- geofence(PASS_MECHANISM_UNRESOLVED), mode probe(INCONCLUSIVE_POLICY_CHIRALITY),
  joint telemetry(연관만), topology(exploratory)의 공식 판정을 소급 변경하지 않는다.

## 9. 하지 않을 것

- 계수 스윕, 예산 연장, 결과를 본 뒤의 임계·seed 변경
- `NAVRL_LATERAL_BIAS_COEF` 동시 조작
- multi-candidate action head 구현
- riskcap 파라미터 탐색, 속도·틸트 상한 변경
- `aerial_gym/config/robot_config/**`·`resources/robots/**` 편집 (provenance freeze,
  `tests/test_navrl_ref5in_provenance_freeze.py`)
- dirty runtime에서의 실행 (VOID 후 clean 재실행)

## 10. 기록 요건

- `results/navrl_ref5in_reflection_consistency_seed383/{cells/{control,treatment},summary.{json,md}}`
- 학습 run 폴더·종단 checkpoint SHA·epoch/샘플 수·KL 통계
- 각 arm의 `reflection_audit.json` 전문과 `70bars.json` 원값
- 프로파일링 결과(§5-b), 무효·실패 실행, VOID 사유
- 요약에 `p2_verdict_changed: false`, `d1_verdict_changed: false`, `p3_unlocked: false`,
  `decision_authority: "none"`

## 11. 비용

arm당 1,000 epoch ≈ **52분**(`ppo_260813_1636_ref5in-d1-q3-adapt-s197` 실측). arm B는 update당
정책 forward가 1회 추가되므로 더 느리다(L3). 평가 2셀 ≈ 40분. 프로파일링 ≈ 10분.
**총 GPU 약 2.5–3시간.** GPU는 공유 자원이므로 착수 전 점유 확인이 필요하다.
