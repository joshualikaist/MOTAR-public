# 재사전등록 — detector 결합 진단, 거리구간별 bias 보정 3-arm

작성: 2026-08-14, **GPU arm 실행 전**. 결과를 본 뒤 기준·seed·arm을 변경하지 않는다.

## 질문과 선행 실패

첫 probe(`results/navrl_detector_coupling_probe_seed409/summary.md`)는 1.0× 합성 잡음이 v7 실물
손실의 82%를 재현했지만, 주입 range-error std가 0.6134 m로 v7 실측 0.7053 m보다 13.0% 작아
사전등록 품질 게이트(±10%)를 실패했다. 원인은 거리 quartile별 평균
`+0.205/+0.305/+0.224/−0.554 m`을 전역 평균 `+0.045 m`로 축약한 것이다.

이번 재실행은 그 누락만 보정한다. sigma, AR(1), dropout, detector threshold, policy, governor,
episode 수와 판정 기준은 바꾸지 않는다.

## 고정 계약

| 항목 | 값 |
|---|---|
| profile | seed 419, `results/navrl_detector_v7_error_profile_seed419/profile.json` |
| policy | frozen ep25000+riskcap, SHA `f70221393660…` |
| learned detector | v7 confirmatory, SHA `85c7974bcd85…` |
| evaluation seed | **431** (사전 검색 미사용) |
| detector-noise seed | **9431** (전용 generator) |
| arena | 40×40 m, bars_h3, navrl_band, 205 bars |
| arm별 episodes | 2,049, deterministic, exact 600 steps |
| threshold / governor | 0.55 전 arm / riskcap 전 arm |

## 주입 모델

기존 AR(1)+거리별 sigma에 아래 거리별 mean bias를 추가한다.

`0–2.083:+0.205442, 2.083–4.661:+0.305296, 4.661–8.641:+0.224064, 8.641–19.961:−0.553643 m`

거리 bin은 잡음을 넣기 전 analytic surface range로 선택한다. bias와 random error 모두 dose 1.0×다.
clean/v7 arm에는 주입 코드가 비활성화된다.

## 품질 게이트 — 판정보다 먼저

기존 게이트를 그대로 유지한다. profile의 analytic-driven range trace에 실제 주입 알고리즘을 replay해
얻은 pooled range-error std가 v7 실측 **0.705294 m의 ±10%** 안이어야 한다. 실패하면 GPU 결과가
있더라도 결합 판정을 하지 않고 `INCONCLUSIVE — noise model insufficient`로 남긴다.

quartile별 mean/std도 기록하되 판정 경계로 새로 사용하지 않는다. 이는 bin-wise bias가 실제로
연결됐는지 확인하는 구현 감사다.

## 3 arms와 판정

1. `analytic_clean`
2. `analytic_noise_1p0_binbias`
3. `learned_v7`

`Δnoise = arm2−arm1`, `Δv7 = arm3−arm1`.

- 품질 게이트 통과 후 `Δnoise`의 95% CI가 `Δv7` 점추정을 포함하면:
  **“v7-shaped output error reproduces the frozen-policy loss; policy–perception coupling
  hypothesis supported, but matched retraining is required for causal confirmation.”**
- 게이트 통과 후 `|Δnoise| < 0.5|Δv7|`이면: 결합만으로 설명 불가.
- 그 사이는 inconclusive.

“결합 확인”은 발표의 짧은 표기일 뿐, 학술적 본문에는 위의 제한 문장을 반드시 병기한다. 이
eval-only 실험은 같은 오차로 학습한 policy가 견디는지를 측정하지 않으므로 결합의 인과 확정이 아니다.
