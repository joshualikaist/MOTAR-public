# P10 — P9 오차 하 PPO 재적응과 held-out 평가 (2026-09-09)

계약: [`docs/plans/perception_p9_p10_execution_2026-09-09.md`](../../docs/plans/perception_p9_p10_execution_2026-09-09.md)
— P9 적합도 gate, test 개봉, 학습, 평가 **이전에** 등록됐다. 수치는 [`summary.json`](summary.json),
셀 receipt는 `results/navrl_grid_p10_empirical_s{541,547}/*/205bars.json`.

커밋 전 CPU 재계산: `python results/perception_p10_2026-09-09/recompute.py`.
8셀 결과 SHA·개수·모델 조건과 요약의 12개 대비를 검사하고 탐색적 상호작용을 출력한다.
원본 `summary.json`의 등록된 대비 수치는 보존했다.

## 설계

2 정책 × 2 인지 arm × 2 평가 시드 = 8 셀, 셀당 205 bars.

| 축 | 값 |
|---|---|
| 정책 | source = 동결 ep25000 (`f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40`) / adapted = 그 위에서 P9 오차를 켠 채 정확히 1,000 epoch (`db6d9f055362…`) |
| 인지 | clean / P9 empirical error (`89879d798a5b10cfb2c0de785dbcfb1754815630b0cd4f91a453de28e728eda9`) |
| 학습 시드 | 811 (평가 시드에 등장하지 않음) |
| 평가 시드 | 541, 547 |
| 그 외 | ep25000 계약 그대로 — riskcap, 205 bars 고정, LR 5e-6, `cluster_sector`, 898-D 관측 |

8셀 모두 `runtime_git_commit 1a39933`, `runtime_git_dirty false`에서 기록됐다.

## 셀 결과

| 셀 | P9 | n | capture | crash | timeout |
|---|---|---:|---:|---:|---:|
| source_clean_s541 | – | 2052 | 81.09% | 16.52% | 2.39% |
| source_clean_s547 | – | 2049 | 82.28% | 15.42% | 2.29% |
| source_p9_s541 | ✓ | 2050 | 77.22% | 19.66% | 3.12% |
| source_p9_s547 | ✓ | 2049 | 77.01% | 19.86% | 3.12% |
| adapted_clean_s541 | – | 2049 | 79.31% | 18.30% | 2.39% |
| adapted_clean_s547 | – | 2050 | 81.56% | 16.05% | 2.39% |
| adapted_p9_s541 | ✓ | 2049 | 79.01% | 18.16% | 2.83% |
| adapted_p9_s547 | ✓ | 2050 | 78.05% | 19.90% | 2.05% |

## 사전등록 추정량 (두 평가 시드 결합, 정규근사 양측 95% CI)

**주 추정량 — adapted-P9 − source-P9**

| 지표 | 차이 | 95% CI | 0 제외 |
|---|---:|---|---|
| capture | **+1.41 pp** | [−0.38, +3.21] | 아니오 |
| crash | −0.73 pp | [−2.44, +0.98] | 아니오 |
| timeout | −0.68 pp | [−1.39, +0.03] | 아니오 |

**부 추정량 — clean 유지도, adapted-clean − source-clean**

| 지표 | 차이 | 95% CI | 0 제외 |
|---|---:|---|---|
| capture | −1.25 pp | [−2.95, +0.44] | 아니오 |
| crash | +1.20 pp | [−0.41, +2.81] | 아니오 |

**참고 — P9 오차가 각 정책에 물리는 비용 (P9 arm − clean arm)**

| 정책 | capture | 95% CI |
|---|---:|---|
| source | **−4.57 pp** | [−6.32, −2.82] |
| adapted | **−1.90 pp** | [−3.65, −0.16] |

## 판정

**주 추정량은 0을 제외하지 못한다.** capture +1.41 pp [−0.38, +3.21]이므로 "재적응이 P9 오차 하
성능을 올린다"고 이 자료만으로 주장할 수 없다. 셀당 ~2,050 에피소드는 1–2 pp 효과를 가르기에
부족하고, 시드별로도 +1.79 / +1.04 pp로 일관 방향이지만 각각 0을 포함한다.

**capture의 참고 대비에서는 두 정책 모두 오차 비용의 CI가 0을 제외한다.** P9 오차는 source 정책의 capture를
−4.57 pp [−6.32, −2.82] 떨어뜨리고, 재적응 뒤에는 그 비용이 −1.90 pp [−3.65, −0.16]으로 남는다.
비용 변화는 `(adapted-P9 − adapted-clean) − (source-P9 − source-clean)`인 상호작용이며,
주 추정량과 다르다. 사후 계산값은 **+2.67 pp [+0.20, +5.14]**다. 이는 기존 집계와 같은
독립 셀 Bernoulli 정규근사를 적용한 **탐색적·다중비교 미보정** CI다. 동일 평가 seed를 쓰는
셀 사이 공분산은 추정하지 못했으며, 학습 seed도 하나이므로 확증적 회복 주장으로 승격하지 않는다.
개별 비용 CI의 겹침은 상호작용 검정이 아니다. `summary.json`의 참고 crash 대비 두 개와 source
timeout 대비도 명목 CI에서 0을 제외하므로, 전체 지표를 두고 "둘뿐"이라고 쓰지 않는다.

**따라서 P10의 결론은 다음 한 줄이다**: 실사에서 측정한 인지 오차를 시뮬레이터에 주입하면 동결
ep25000 정책의 capture가 이 평가에서 4.6 pp 떨어졌고, 1,000 epoch 재적응의 사전등록 주 효과는
+1.41 pp이나 CI가 0을 포함한다. clean 유지도 CI는 −2.95 pp까지의 손실을 포함하므로 성능 보존이나
비열등성이 입증된 것은 아니다.

## 주장하지 않는 것

- **시드 일반적 PPO 주장 아님.** 학습 시드는 811 하나다. 평가 시드 2개는 평가 분산만 다룬다.
- 사후 pass margin 없음. 위 문턱은 계약서에 등록된 것이고 결과를 본 뒤 바꾸지 않았다.
- 실기 성능 아님. P9는 NPS validation의 단일-GT 1,310프레임에서 보정한 pixel/시간 오차 모델이다.
  metric range·confidence 오차는 보정하지 않았다. P7c 최종 NPS test 평가는 이 결과 확정 뒤 S4에서
  한 번 실행됐고(utility 0.4701, `results/perception_s4_2026-09-09/`), 그 결과로 P8/P9를 재보정하지 않았다.
  NPS test는 앞서 P5에서, Det-Fly test는 P3-F에서 사용된 이력이 있으므로 완전 미열람 test로 부르지 않는다.
