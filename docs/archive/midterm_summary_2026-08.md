# MOTAR 중간발표 정리 — 센서 전용 UAV 요격 (2026-07-09 ~ 2026-08-10)

> WORKLOG.md 전체(약 6,300줄)를 시간순으로 압축한 발표 준비 자료.
> 모든 수치는 WORKLOG의 실측값이며, 뒤에 무효화·정정된 결과는 ⚠️로 표기하고 남겨두었다
> (기각된 가설도 결과다). 세부 근거는 각 날짜의 WORKLOG 항목과 `results/`에 있다.

---

## 0. 한 장 요약

**문제**: GT 표적 정보 없이 — 온보드 RGB-D 카메라 + LiDAR만으로 — 고밀도 장애물
숲에서 이동 표적을 탐색·추적·요격하는 RL 정책 (NavRL 스타일, IEEE RA-L 목표).

**방법 스택**: Isaac Gym 병렬 시뮬 + rl_games PPO / actor 898-D·critic 906-D
비대칭 구조(GT는 reward·종료판정·critic·평가에만) / 17-token Transformer
(LiDAR 72×4 @12 m, 장애물 토큰 8개 cluster_sector 선택, 카메라 87°@20 m 검출
→ CV Kalman tracker) / 밀도·거리 커리큘럼 / riskcap speed governor.

**현재 동결 상태 (발표 시점)**:
- 정책 **ep25000 + riskcap** (205 bars 40×40 m 아레나, seed47 held-out 2,049 ep)
- clean: capture **80.54% / crash 17.17%** (seed45 uniform 81.94/15.67, seed51 81.12/16.68)
- v2 밀도 곡선: 130막대 **89.31%** → 205막대 80.54% → 220막대(OOD) 77.76%
- 인지 강건성 진단: 현 **1×1 detector artifact −13.9pp** / dropout 0.3 **−12.7pp**
  ≫ latency 0.1 s **−2.5pp**(보정 후) > range error ±0.3 m **≈0pp**
- latency 관측: 정확한 timestamp/pose history 조건에서 0.1/0.2/0.3 s 손실은
  −2.5/−3.9/−8.0pp. 실기 예산은 아직 pose clock·보간 오차를 포함해 검증해야 한다.

---

## 1. 연구 타임라인 (Phase별 순차 정리)

### Phase 1 — v1 주입형 목표 항법 태스크 구축 (07-09 ~ 07-16)

**07-09 · 셋업 + 첫 학습**
- 512 env × 1500 epoch 도달률 **0%** → 거리 커리큘럼 도입 후 20~36%.
- 목표가 물리적으로 도달 불가한 거리였음을 진단 — 커리큘럼 필수 확인.

**07-10 · 전용 "빈 공간 + 막대" 환경 신설**
- 스톡 환경의 z-스폰 버그(바닥 아래 스폰) 제거. 새 환경 스모크에서 ever_reached
  71%(구 환경 30% 정체 대비). 성공 반경 1.0 → **0.5 m** 정정(NavRL 원본 사양).

**07-13 · loiter(배회) 붕괴 진단 → 리워드 재설계**
- captured **0.796 → 0.067 붕괴**. 원인 = 안전항이 개방공간에서 **+1.386/step 고정
  수입**(배회 가치 104 ≫ 캡처 가치 30). 재베이스라인 + 충돌 −20 + ego-progress shaping으로 회복.
  정적 표적에서는 PBRS 대수와 같지만 이동 표적에서는 future target으로 재고정되는 휴리스틱이므로
  policy-invariance 정리를 주장하지 않는다.
- 아레나 10×10 → 24×24 m, 막대 16 → 48개. *리워드의 전역 최적해를 실측으로 특정한 첫 사례.*

**07-14~15 · yaw 제어로 Phase 1 해결**
- 정직 baseline 0.65/0.35 → yaw 액션 추가(3→4-D) 후 **captured 0.954 / crash 0.046**
  (crash 7×↓). 원인은 리워드가 아니라 기하(corner-clip)였음을 입증.
- 격자 배치 → 랜덤 배치 전환, **격자 밀도 데이터 폐기**(교란변수).

**07-16 · v1 밀도 절벽 (RQ1)**
- captured: 25막대 0.972 / 75 0.942 / 110 0.926 / **150 0.656 (−27pt 절벽)**.
- 절벽 위치가 배치 기하의 RSA 재밍 한계(~148개)와 일치 — 밀도-성능은 선형이 아니라 절벽형.

### Phase 2 — 비전 피벗: 센서 전용 요격 (07-17 ~ 07-22)

**07-17 · ★ 방향 전환 (사용자 결정)**
- actor 관측에서 GT 표적 완전 제거. 305-D(ego 9 + 검출기 8 + LiDAR 2ch) + 비대칭 critic.
- GT 무누출 프로브 PASS. *이것이 논문의 최종 문제 설정.*

**07-18~20 · 센서 전용 첫 결과 + 평가 함정 정정**
- 110막대 from-scratch **47.3%**. "커리큘럼 실패(40.8%)"는 체크포인트 오독
  (`gen_ppo.pth` = 저밀도 best)이었음 — `last_gen`으로 재평가하니 **60.6% (+13pt) 성공**.
- 이후 프로젝트 철칙: 밀도 run 평가는 반드시 `last_gen` 체크포인트.

**07-22 · 연구 핵심 재정의**
- analytic semantic 주입은 상한 baseline으로 강등, **학습형 인지 + 17-token
  Transformer**로 계획 확정. 3D 상태 대시보드(docs/status) 구축 시작.

### Phase 3 — Perception + Transformer 캠페인 (07-23 ~ 07-28)

**07-23~24 · NaN 근본원인 수정 + 정직한 baseline**
- entropy 보너스가 log-std를 σ=13까지 폭주시켜 NaN사 → clamp(−5, 0.4)로 해결.
- adaptive LR 1e-4→1e-2 폭주가 또 다른 사인 → `lr_schedule: None`.
- 고정평가: 표적속도 0~1.5 m/s에서 capture 0.686~0.762로 평탄 — **병목은 표적속도가
  아니라 막대충돌(~25%)**.

**07-24~27 · 개입 캠페인: 0.735 → 0.889**
- 고도 PI(+6.7pt) → general spawn → **음성 결과 3연속**(tilt-comp, LiDAR 12 m
  look-ahead 모두 bar_contact 불변 = 가설 기각) → 계측(bar probe): 반경 내 막대
  15.8개 vs 토큰 5개, **충돌의 35%가 정책 입력에 없던 막대**.
- 표현 개입만 유효: 8토큰/72빔(bar_contact 12.7→9.0%) → **FOV 240°: capture 0.889,
  bar_contact 7.5%**. 궤적: 0.735→0.802→0.837→0.861→**0.889**.

**07-28 · 70~75막대 정체: bounded action**
- lateral std 1.49까지 포화(clamp 질량 50%) — unbounded Gaussian + hard clamp가 구조
  결함. squashed-Gaussian A/B **+4.02pp**. ⚠️ 이 수치는 다음 날 센서 버그 발견으로
  "깨진 관측 위 성능"으로 강등.

### Phase 4 — ★ 키랄리티(거울) 버그와 재베이스라인 (07-29 ~ 07-30)

**07-29 · 관측 좌우 결함 2건 발견**
- LiDAR 빈 방위 테이블이 실제 센서와 **거울상**(물리 프로브: 옛 가정 on-bar 일치
  13.9% vs 실제 규약 **94.8%**) + 카메라 far-plane 10 m가 12 m LiDAR에 **팬텀 벽**으로 융합.
- **기존 전 정책·결과는 재베이스라인 대상. warm-start 무효, fresh 재학습.**
- 같은 날: 표적 속도 실현 결함(고밀도에서 명령 대비 26~52%만 실현, 사실상 주차)도
  수정 → 전 밀도 100% 실현.

**07-30 · 수정 사후검증 + v1 헤드라인 그림**
- 밀도곡선 공통 6점 평균 **+11.1pp**(25막대 0.896→0.978). hit_token_given_fov 0.97대.
- ⚠️ "85막대 73.9%"는 단일 epoch 오독 → 실제 plateau 0.676.
- `cluster_sector` 토큰 선택 채택(clean windows 4연속 0.70+ 통과).
- **밀도×속도 맵(28셀)**: 밀도 축 **−78pp** vs 표적속도 축 **−4.2pp** — "밀도가 거의
  전부, 속도는 난이도 축이 아님". ⚠️ v1 태스크(24 m, 478 m²) 데이터 — v2와 비교 불가,
  v2 재측정 예정 (08-07 대시보드에 명시).

### Phase 5 — v2 search arena와 PPO 안전장치 (07-31 ~ 08-02)

**07-31 · v1 한계 확정 → 환경 v2 재설계**
- 100막대 17연속 hold(plateau ~0.56), corridor 토큰(898→946-D)도 게이트 FAIL(+1.57pp).
- **v2 "search arena"**: 40×40 m, 70→300막대, 목표 6~28 m(센서 20 m 초과 = 탐색 레짐),
  slit-불가능 배치. v1 결과는 확정 보존하되 **비교 불가**(task-version bump).
- 70막대 실측 천장 0.843 → 승급 임계를 실측 기반 0.82로 조정(추정치 0.85는 도달 불가).

**08-01 · ★ PPO actor collapse → 트랜잭션 체계**
- 145막대에서 KL 0.019→**2.69**, capture 62→0% (50 epoch 만에 붕괴). 원인 = KL guard가
  적용된 update를 되돌리지 못함 + LR 20배 + margin coef 누락.
- actor+critic 원자 트랜잭션·rollback·attestation 구축 — 강제 reject 검증에서
  **model 93/93 텐서 byte-exact 복원**. 테스트 129 PASS.

**08-02 · 205막대 도달, ep24000 동결, 인과검사**
- 승급 130→…→**205**. plateau 68.94%로 사용자 결정 중단, **ep24000 동결**.
- held-out: 130막대 84.77% / 205막대 **72.44/25.07%** / 220막대 68.49%.
- 미러 평가: outcome 좌우 차 −0.21pp(비대칭 미검출)이지만 reflected pair에서 lateral
  action sign mismatch **73.08%** — 학습된 turn chirality는 실재하나 현 분포에선 무해.
- 망각검사: ep19100→ep24000 **+4.65pp** — 망각 없음.

### Phase 6 — frozen 정책 위 통제 A/B와 control-risk (08-03 ~ 08-05)

**08-03~04 · fixed-density 연장의 함정**
- ep24001→25000 연장: KL 정상(max 0.0076)인데 held-out **−2.94pp** — 발산 없는 느린
  action drift. baseline 연장 금지 규칙.

**08-05 · TTC selector: 기각(판정 불가)**
- main TTC arm 70.21/29.50% — 사전등록 게이트 FAIL. 단 selector 전환이 FOV
  240°→360°를 함께 바꾼 confound 발견 — pure ranking 효과는 판정 불가로 기록.

**08-05 · ★ 8시간 control-risk 루프 → riskcap 동결**
- 도중 **2회 데이터 무효화**: ① speed governor가 semantic mask(GT)를 직접 소비하던
  leak 발견 → 해당 데이터 전부 격리, ② LiDAR 수직행 역순 발견 → 재실행.
- corrected: complete-stop 계열은 timeout 16~24%로 기각. non-stop **riskcap**:
  seed44 79.55/17.62%(off 대비 capture +6.7pp) GO → 1,000 epoch 적응 후 seed45
  uniform **81.94/15.67%**, seed46 고정속도 3/3 PASS.
- **ep25000+riskcap을 현재 candidate로 동결. 이후 PPO 재학습 금지.**

### Phase 7 — R3 인지 강건성: latency 대반전과 dropout 채널 사냥 (08-05 ~ 08-10)

**08-05 · R3 스크린 (⚠️ latency 셀은 이후 superseded)**
- clean 80.54/17.17 기준: dropout 0.3 **−12.7pp**, latency 0.1 s **−42.7pp** /
  0.2 s −62.0pp, range error ±0.15/0.30 m **≈0pp**, 현 1×1 detector artifact −13.9pp.
- 판정 당시: "latency가 1순위 병목" → 보정 후보 P0~P3 사전 설계.

**08-05~06 · P0/P1/P2 연쇄 기각으로 채널 좁히기**
- P0 forward predict **−0.10pp 무효** — 단, tracker lag 0.15 m를 제거하고도 무효라는
  것 자체가 "위치 lag 가설"의 실험적 기각.
- P1 LiDAR backup **−8.85pp 역효과**. P2 obstacle-map 보정 +1.84pp(구멍의 11%) NO-GO.

**08-06 · ★ P3 ego-motion 보정: 최대 반전**
- 원인: 지연된 기체-프레임 측정을 **현재 pose로** 월드 변환 → 드론 자신의 운동
  (병진 0.233 m + yaw 0.408 m)이 매 KF 보정에 주입. 표적 lag(0.15 m)보다 큰 항.
- 취득 시점 pose로 변환(실기 표준 관행) 시 **37.82 → 78.04% (+40.2pp, 손실의 94% 회수)**.
- **R3 naive-transform 판정 공식 기각**: 정확한 timestamp/pose history 조건의 잔차는
  0.1 s −2.5pp / 0.2 s −3.9 / 0.3 s −8.0 / 0.5 s −15.8pp. P3 기본 ON 승격.
  실기 예산은 clock skew·pose interpolation을 포함한 뒤 정한다.

**08-06~07 · dropout 채널 사냥 (−12.7pp의 해부)**
- dropout과 현 1×1 detector artifact는 **실패 경로가 정반대**: dropout은 표적에 접근하되 좁은
  clearance(0.65 m)에서 충돌↑, detector는 표적 근처 미도달(closest 1.89 m).
- 유령표적 backfill: +2.15pp / clean −1.91pp (2시드 재현) → **순효과 상쇄, 기각**.
- H2(LiDAR target association off): **+3.26pp (2시드 재현, 손실의 26%)** — 유일하게 견고.
- H3(공분산 정직화) +1.46pp 유의 미달, 게이트 축소 +1.71pp 기각 — "연관 창이 넓다"가
  아니라 **통과한 연관 자체가 해롭다**.
- 발견한 결함: 필터가 관측하지 않은 축의 공분산까지 축소(20스텝 blind에서 보고 σ
  0.09 m vs 실제 오차 3.27 m — "자신 있게 틀림"), LiDAR 각분해능 5°(5 m에서 0.44 m)는
  표적 추적에 부족. detector SHA 가드가 죽어 있던 무결성 결함도 발견·수정.

**08-10 · H4 플래그 검정: 탐색적 메커니즘 단서**
- range 보정은 **그대로 두고** age 리셋·visibility·confidence만 차단 → dropout에서
  **70.20% (+2.36pp)**, H2(연관 전체 차단, +3.41pp)의 **69%를 회수**. clean에서는 +0.58pp(무해).
- H4 +2.36pp의 95% CI는 **[−0.47,+5.19]**, p=0.103으로 0을 포함한다. 69%는 seed47의
  H2 차이에 대한 비율이고 전체 dropout 손실 회수율은 18.6%다. 비가산 개입이라 남은 31%를
  상태 보정의 독립 기여로 빼지 않는다.
- **논문용 메커니즘 문장**: LiDAR 연관은 트래커의 예측을 측정으로 되먹이고, 그 측정이
  `correct()`에서 age를 0으로 찍고 `observe()`에서 visible을 참으로 만든다. 정책의 target
  token은 **자기 예측에 대해 "방금 관측했다"는 증언**을 받아, 표적을 잃은 구간에서
  확신을 갖고 돌진한다.
- 채택은 보류(단일 시드, 4pp 게이트 미달)하되 **이 가지의 진단은 종료**. 개입 4회 + 통짜
  차단 1회로 LiDAR 연관 경로의 내부 구조가 정량화됐다.

**08-10 · 대시보드 전수 감사 (방법론 항목)**
- 곡선 밀도 표기가 **3.3× 과소**(v1 곡선을 v2 면적 1600 m²로 나눔 — 25막대가 1.6/100m²로
  표시, 실제 5.2). Speed 탭은 **키랄리티 수정 이전(07-27) 데이터**를 현재 성능처럼 표시 중이었음.
- 수정: 곡선마다 task_version·arena·placement_area·superseded를 데이터에 스탬프, 렌더러는
  superseded 시리즈를 절대 선택하지 않음. 무효 곡선 9개는 **삭제하지 않고 라벨 보존**.
- 아카이브에서 v2 곡선 2개 복원(재학습 0): v2 밀도 곡선(ep24000, 130~220막대),
  v2 고정속도 축(**현재 동결 정책** ep25000+riskcap, 0.3/0.9/1.5 m/s → 81.84/80.77/75.51%).

**08-10 · ★ v2 재측정 2건 (inference-only, 재학습 0)**
- **v2 밀도 곡선**(ep25000+riskcap): 130→220막대에서 **89.31 / 84.63 / 82.77 / 80.54 / 77.76%**.
  ep24000(governor off) 대비 격차가 밀도와 함께 증가(+4.54 → +9.27pp).
  ⚠️ **귀속 주의**: 이 Δ는 체크포인트(ep24000→ep25000, 1,000 epoch 적응)와 governor(off→riskcap)가
  **동시에 다른** 비교라 riskcap 단독 효과가 아니다. 205막대에서만 분리돼 있다 —
  governor 단독 **+8.17pp**, 적응 단독 **+3.74pp**. 밀도 의존성의 귀속은 **미확정**
  (ep24000+riskcap을 같은 격자에서 재면 분리 가능, 5셀). TTC selector가 FOV를 함께 바꿔
  판정 불가였던 것과 같은 종류의 confound다.
- **v2 밀도×속도 맵**(20셀): 이 격자의 주변 endpoint 대비는 밀도 **−11.4pp**, 속도
  **−2.7pp**다. v1은 arena·배치·정책·범위와 곡선 모양이 달라 endpoint 기울기를 직접
  비교하지 않는다.
- **속도×밀도 탐색 결과**: 표적이 0.3→1.5 m/s로 빨라질 때의 관측 비용은
  130막대 **−0.88pp** → 205막대 −3.06pp → 220막대 **−5.99pp**로 커진다.
  그러나 학습범위 ≤205의 interaction 검정은 LR p=0.337 / omnibus p=0.817로 **미확인**이며,
  강한 −5.99pp는 OOD 220 셀이다. 현재는 가설로만 두고 새 seed로 재검정한다.
- 정합성 확인: 205막대 고정속도 4셀 평균 80.33% ≈ uniform 실측 80.54%. 속도 축은 단조가
  아니다(0.7 m/s가 5개 밀도 중 4개에서 0.3보다 높음).

---

## 2. 발표 핵심 카드 (슬라이드 후보 15장)

1. **yaw 제어의 결정타**: crash 0.32→0.046 (7×↓) — 리워드가 아니라 기하가 원인 (07-15)
2. **loiter 붕괴의 산술**: 안전항 +1.386/step 수입이 배회를 전역 최적해로 (07-13)
3. **v1 밀도 절벽**: 110→150막대 −27pt, RSA 재밍 한계와 일치 (07-16)
4. **평가 함정의 교훈**: 체크포인트 오독 정정으로 "실패 40.8%"→"성공 60.6%" 반전 (07-20)
5. **음성 결과 3연속 후 계측**: 충돌의 35%가 정책 입력에 없던 막대 → 표현 개입만 유효 (07-24~27)
6. **★ 키랄리티 버그**: LiDAR 방위 거울상(13.9→94.8%) — 수정 후 +11.1pp, 전면 재베이스라인 (07-29)
7. **밀도×속도 맵**: v2 격자의 주변 endpoint 대비는 −11.4 vs −2.7pp. 학습범위
   density×speed interaction은 미확인(p=0.337/0.817), OOD 220에서만 큰 격차를 관측 (08-10)
8. **PPO 트랜잭션**: KL 2.69 붕괴 → 원자 복구, 강제 reject에서 93/93 byte-exact (08-01)
9. **v2 커리큘럼**: 70→205막대, frozen ep24000 held-out 72.44/25.07% (08-02)
10. **chirality의 이중성**: action sign mismatch 73%인데 outcome 차이 −0.2pp (08-02)
11. **연장의 함정**: KL 정상이어도 held-out −2.94pp 느린 drift (08-04)
12. **riskcap governor**: 2회 무효화를 이겨내고 81.94/15.67%, 동결 (08-05)
13. **★ R3 대반전**: "latency −42.7pp"는 ego-motion 아티팩트 — timestamp/pose-history
    조건에서 0.1 s 잔차 −2.5pp, 0.5 s는 여전히 −15.8pp (08-06)
14. **기각의 과학**: 그럴듯한 중간 변수 6개(P0/P1/P2/backfill/H3/게이트)를 저비용 A/B로
    연쇄 기각 — 유일한 생존자 H2 +3.26pp 2시드 (08-05~07)
14b. **탐색적 채널 단서**: 플래그 차단 +2.36pp이나 단일 seed CI가 0 포함; 69% 가산 분해는 금지 (08-10)
15. **다음 관문**: 현 1×1 detector artifact −13.9pp — offline detector gate부터 재설계 (08-06~10)

---

## 3. 현재 상태와 남은 일

**동결 자산**
| 항목 | 값 |
|---|---|
| 정책 | ep25000+riskcap (SHA `f7022139…`) |
| 계약 | 205 bars, 40×40 m, seed47, deterministic, 2,049 ep/cell |
| clean | 80.54 / 17.17 % (3개 시드 80.5~81.9) |
| 관측 | actor 898-D / critic 906-D, GT 무누출 |

**인지 강건성 최종표 (수정된 latency 모델)**
| 교란 | capture | Δ vs clean |
|---|---:|---:|
| clean | 80.54% | — |
| range error ±0.30 m | 80.62% | +0.1pp |
| latency 0.1 s (보정) | 78.04% | −2.5pp |
| latency 0.2 s (보정) | 76.62% | −3.9pp |
| latency 0.3 s (보정) | 72.57% | −8.0pp |
| dropout 0.3 | 67.84% | −12.7pp |
| **1×1 detector artifact** | **66.62%** | **−13.9pp** |

**남은 일 (발표의 "향후 계획" 슬라이드)**
1. **schema-v2 평가 재베이스라인** — 과거 결과는 601-action timeout semantics였고 import source
   manifest가 없었다. 정확한 600-action/동일 source·seed로 A/B/C를 전부 다시 잰다.
2. **detector offline gate** — 현 −13.9pp는 positive pixel 0.054%에 unweighted BCE를 쓴
   1×1 artifact의 결과다. full-FOV/range/occlusion/absent split과 PR/calibration을 먼저 만든다.
3. **density×speed interaction 재검정** — ID 4밀도×2속도×새 seed 2개, 220은 OOD 보조자료.
4. corridor B-arm(946-D) 재평가와 R4 temporal fusion은 위 세 gate가 끝날 때까지 보류.
5. 논문 작성: task-version(v1/v2)과 legacy-601/schema-v2 구분, superseded 데이터 처리 원칙 포함.

**연구 방법론적 기여 (발표에서 강조할 만한 것)**
- 모든 평가에 SHA-피닝·provenance receipt·무결성 가드 (evaluator 편집 시 자동 거부 실증됨)
- 기각된 가설의 체계적 기록 — 6개 연쇄 기각이 ego-motion 대반전(94% 회수)으로 수렴
- GT 누출에 대한 지속적 적대적 감사 (semantic-mask leak, 거울 테이블 등 실제로 2회 적발)
