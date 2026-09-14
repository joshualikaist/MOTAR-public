# MOTAR 발전 방향 — 문헌 접지 로드맵 (2026-08-12)

> 전제: 검증 1~4 종결, 검증 5A engineering smoke는 조건부 PASS(성과 주장은 불가), 검증 5B
> full-budget fresh PPO는 미실행. 이 문서는 **그 다음** 무엇을
> 발전시킬지를 (실측 격차 → 문헌 근거 → 실험 설계 → 비용) 형태로 정리한 것이다.
> 문헌 스캔: 2026-08-12, 2023–2026 중심, 검색 ~29회 + 원문 확인 12편 (2개 병렬 조사).

---

## 0. 먼저 — 우리가 이미 갖고 있는 신규성 3개

두 독립 조사가 **같은 결론**에 도달했다: 아래 세 측정은 선행 연구에 없다.

| 신규성 | 우리 수치 | 문헌 확인 결과 |
|---|---|---|
| **정책-인지 결합 비용의 직접 측정** | 동결 정책 + 검출기 교체 = nominal **−5.0 pp** | Swift(Nature 2023)도 노이즈모델 유무 ablation까지만 — 교체 비용을 pp로 측정한 선행 부재 |
| **"충돌 유발 장애물의 입력 탈락률" 측정** | 충돌 막대의 **16.8%가 8토큰 밖** (4배 과밀, rank 0) | 이 양을 직접 측정한 논문 부재 — 측정 방법론 자체가 기여 |
| **단일기체 dense-clutter 요격 문제 설정** | **80.5% @ 12.8 bars/100m²** (이동 표적) | 밀집 통과 SOTA(80~90%)는 전부 정지 표적 goal-reaching; 장애물 속 요격은 다중기체 1편(RA-L 2025)뿐 |

포지셔닝 환산: Science Robotics "Wild"의 최고 밀도 = 4 trees/100m² → **우리는 그 3.2배 밀도에서
이동 표적을 잡는다.** 논문에 bars/100m² ↔ trees/m² 환산표 필수.

---

## 1. 방향별 상세 (우선순위순)

### D1. 제동 실행가능성 안전층 — riskcap을 backup-CBF형 필터로 교체
- **실측 동기**: 접촉 시 executed stopping margin **−0.157 m** — governor가 1/3을 깎아도
  제동 불가 포켓에 진입 (검증 4③). 현 governor는 clearance/TTC 휴리스틱 = 동역학 비인지.
- **문헌**: FastBridge(2607.01200) — forward-simulated backup policy로 **제동 실행가능성을
  인증**하는 CBF 필터, 실기 수준. Composite CBF(2502.04101)로 다중 장애물 합성.
  CBF-RL(2510.14959) — 필터를 학습 중에 걸어 정책이 내재화("governor가 자른 만큼 정책이
  더 요구"하는 우리 악순환의 해법: requested 3.03 vs executed 2.02 m/s).
- **실험**: 배포형 필터 A/B (riskcap vs backup-brake 필터, frozen 정책, **eval-only**) →
  margin ≥ 0 강제 시 capture/crash 변화. 성공 기준: crash 17%→<10%, capture 손실 ≤2 pp.
- **비용**: eval-only~경량. **가장 값싼 고가치 항목.**

### D2. 장애물 표현 확장 — 토큰 K↑ / 어텐션 선택기 / full-scan 서로게이트
- **실측 동기**: 16.8% 입력 탈락 + 31.7개 경쟁/8슬롯 (검증 4②). rank 0 = 셀렉터는 옳고
  **용량이 부족**.
- **문헌**: Flying on Point Clouds(2503.00496) — full-scan 경량 표현으로 thin obstacle 유지.
  CaMeRL(2605.14810) — 시간 메모리로 슬롯/FOV 이탈 보완. Crowd-Robot attention(ICRA'19,
  정본 앵커) — top-K 하드컷 대신 학습형 어텐션.
- **실험**: (a) K=8→16 단독 A/B(최소 변경), (b) 어텐션 선택기, (c) full-scan 서로게이트 —
  순서대로. 각각 "탈락률 16.8%→?"와 crash 변화를 **분리 보고**(우리 probe가 그대로 계측기).
- **비용**: retrain (관측 폭 변경 → fresh 정책). 검증 5 이후 첫 재학습 후보.

### D3. 인지-in-the-loop 미세조정 — Swift식 경험적 노이즈 모델
- **실측 동기**: E1 **−5.0 pp** (검증 2) — 정책이 analytic 검출 통계에 결합.
- **문헌**: Swift(Nature 2023)의 전이 성공 결정 요인이 정확히 이것 — 실측 검출기의
  오차/드롭아웃 통계를 경험 모델로 피팅해 그 위에서 학습. 비대칭 AC 이론(2501.19116)이
  privileged critic 확장의 면허.
- **실험**: v7 검출기의 오차 통계(bearing/range/드롭아웃/FP)를 피팅 → frozen 정책을 그
  통계 위에서 fine-tune → E1 재측정. 성공: −5.0 → −2 pp 이내 + envelope 유지.
- **비용**: retrain(fine-tune). eval-only 선행 가능(노이즈 파라미터 sweep으로 결합 곡선).

### D4. 제어명령 전환 — velocity → CTBR(또는 가속도) [사용자 질문 직접 대응]
- **실측 동기**: −0.157 m margin의 나머지 반쪽 — velocity 추상화는 (i) 추종 지연,
  (ii) 정책이 제동 동역학을 모름을 동시에 만든다.
- **문헌**: 벤치마크 정본(ICRA 2022) — **LV는 "공격적 기동 자체가 불가능한" 구조적 상한**,
  CTBR이 agility·모델 불일치 강건성 모두 우위. Swift·OPEN(RA-L 2025, 추격 100% 포획
  + 실기 zero-shot)까지 **학습형 agile flight의 사실상 표준 = CTBR**. 요격 전용 문헌
  (2607.02472)은 점질량 근사 대비 풀 동역학 학습이 +30% — velocity ≈ 점질량이라는 방증.
  SimpleFlight(RA-L 2025)가 CTBR 전이 레시피(5개 설계요소) 제공. 중간 절충 = 가속도 명령.
- **실험**: 205 bars에서 velocity vs CTBR **통제 비교표** — dense-clutter 요격에서 이 비교는
  미발표 영역이라 **결과 부호와 무관하게 섹션 가치**.
- **비용**: 대형 (액션·보상·컨트롤러 인터페이스 재설계 + fresh 학습). 논문 2편째 후보.

### D5. 지연·타임스탬프 랜덤화 학습 — 검증 3 절벽 평탄화
- **실측 동기**: pose-stamp **+50 ms → −17.3 pp** 비대칭 절벽 (검증 3).
- **문헌**: MMDR(ICRA'22) — 모달리티별 지연 **독립** 랜덤화. MBRL random observation
  delays(2509.20869) — out-of-sequence 관측 정면 처리. Zero-shot sim2real ablation
  (2412.11764) — 지연·노이즈 모델링이 최상위 영향 요인.
- **실험**: 학습 중 검출 지연 + pose-stamp 오차를 독립 랜덤화 → premise budget 곡선 재측정.
  성공: +50 ms 비용 −17.3 → 한 자릿수.
- **비용**: retrain. **검증 5 (B)에 노브가 이미 있어 randomization 추가가 쉬움.**

### D6. 불확실성 토큰 — 공분산/confidence/NIS vs 이산 플래그
- **실측 동기**: H4 — dropout 손실의 2/3가 "봤다" **이산 플래그 거짓 증언** 채널.
  H3 — 정책이 공분산 크기를 사실상 안 읽음.
- **문헌**: PUARL(MobiCom 2024) — 인지 불확실성 조건 추격. Trust-Nav — **필터 일관성(NIS)**
  유래 trust 신호(우리 "just seen" 플래그의 원리적 대체). ACC 2025 — 간헐 관측에서 학습
  운동모델 > CV 예측.
- **실험**: target 토큰에 공분산+confidence+NIS 추가 vs 현 플래그 — 입력만 바꾼 retrain A/B.
  성공: dropout −12.7 pp 축소 + envelope timeout 감소.
- **비용**: retrain (입력 폭만 변경).

### D7. 탐색 국면 목적함수 — 확률맵 토큰 + 정보이득 보상
- **실측 동기**: 표적이 20 m 센서 지평 **밖**에서 시작하는데 명시적 탐색 전략 부재;
  속도×밀도 상호작용(−5.9 pp @205)의 절반은 "도착 전 이동" 서사.
- **문헌**: ReSPIRe/ASPIRE — search-and-track POMDP·MI 계획(비학습 베이스라인).
  Long-Term Target Search(Drones 2024) — 확률맵 입력 RL. OPEN — evader-prediction 보조헤드.
- **실험**: last-known belief/확률맵을 토큰 1개로 추가 + 미관측 시 정보이득 보상.
  성공: 원거리 시작 스트라텀 capture 상승(우리 결과 JSON의 distance strata가 그대로 계측기).
- **비용**: retrain + 입력 헤드.

### D8. 손제작 융합 제거 — 학습형 트래커/추정기 (장기)
- **실측 동기**: 손제작 파이프라인의 병리 **4종 실측** — phantom obstacle(P2), association
  순손실(H2), "봤다" 거짓 증언(H4), clock 비대칭(검증 3). 이 자체가 ablation 스토리.
- **문헌**: Transformers as Implicit State Estimators(2410.16546), SMART-TRACK(드롭아웃
  브리징), Recursive KalmanNet(공분산 일관 출력 — D6와 연결), NOVA(carve-out 없는
  표적/장애물 분리 표현).
- **실험**: 타임스탬프 원시 검출+LiDAR를 토큰으로 직접 소비하는 recurrent/Transformer
  추정기로 CV-KF+carve-out+association 대체. 성공: 병리 4종 소멸 + dropout/latency 동등 이상.
- **비용**: new architecture. **저널 확장/2편째의 중심 후보.**

### D9. (보너스, eval-only) envelope에 distractor 축 추가
- **실측 동기**: 검증 2 envelope은 외형만 다룸 — SimD3(2601.14742)의 지적: 검출기를 깨는 건
  외형이 아니라 **혼동체(새·유사 물체)** 분포. albedo-밖 FPR 7.6% 경고와 같은 계열.
- **실험**: 렌더러에 non-target red-ish 객체 추가 → v7 FPR/navigation 재측정. eval 중심.

---

## 2. 다음 주 실행 순서 제안 (GPU 예산 기준)

검증 5(Codex)가 GPU를 점유하는 동안 → **eval-only 선행 가능**: D1(필터 A/B), D3의 결합
곡선 sweep, D9(distractor). 검증 5 종료 후 retrain 슬롯 1개씩:

| 슬롯 | 항목 | 근거 |
|---|---|---|
| 지금(병렬) | D1 필터 설계 + D9 | eval-only, GPU 짧게 |
| retrain #1 | **D2 토큰 K↑** (최소 변경 arm부터) | 16.8%가 가장 큰 단일 설명 변수 |
| retrain #2 | D6 불확실성 토큰 | 입력만 변경, H4 서사 완결 |
| retrain #3 | D3+D5 (검증 5-(B) 계보 위에 노이즈모델+지연 랜덤화) | fresh 계보와 자연 결합 |
| 대형 | D4 CTBR 비교, D8 학습형 추정기 | 논문 2편째/저널 확장 |

주의: retrain #3에서 D3·D5를 같이 넣으면 효과 분리가 안 된다 — **하나씩** (TTC confound 교훈).

---

## 3. 논문 구조 제안

**1편 (RA-L, 현 결과로 완결 가능)** — "측정 주도" 서사:
1. 문제: 단일기체 dense-clutter 요격 (신규성 ③, Wild 대비 3.2× 밀도 환산표)
2. 시스템: 센서-온리 스택 + 커리큘럼 + riskcap (Gate 1 +5.8 pp)
3. **측정 I — 결합**: 인지 robustness ≠ 시스템 robustness (E1/E2/E3; 신규성 ①)
4. **측정 II — 전제 예산**: latency는 modeling 결함이었다(P3 94% 회수) + 비대칭 clock/yaw 스펙
5. **측정 III — 천장 분해**: 기하 무죄 → 표현 16.8%(신규성 ②) + 제동 여유
6. 밀도×속도 상호작용 (Gate 2, p=0.00035)
7. 한계·향후: 본 문서의 D1~D8을 future work로 계층화
- 기각된 가설들(P0/P1/H1/H3/게이트…)은 부록 표로 — "negative results as evidence"가 리뷰어 방어력.

**2편째 후보 (1편 이후)**: D4(제어명령 비교) 또는 D8(융합 제거) 중심의 methods 논문 —
1편의 측정이 2편의 동기 섹션이 되는 구조.

---

## 부록 — 조사 원본
두 서브에이전트 보고 전문은 세션 로그에 있으며, 본 문서는 그 종합이다. 핵심 인용 20편은
각 방향에 인라인 표기.
