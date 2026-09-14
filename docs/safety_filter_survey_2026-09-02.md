# Safety filter / speed governor — 문헌조사 및 현행 설계 진단

작성 2026-09-02. 목적: `speed_governor.py`의 재설계 계획 수립을 위한 근거 정리.
**본 문서는 조사·진단이며 어떤 코드 변경도 포함하지 않는다.** 결과 수치는 기존 receipt에서 재인용.

증거 등급: **[P]** 논문이 증명 · **[S]** 시뮬레이션만 · **[H]** 실기 비행

---

## 1. 현행 설계의 구조적 결함 — 세 갈래 수렴

### 1.1 실측 (기존 receipt, 신규 실행 없음)

`results/navrl_v2_ep24000_riskcap_seed44_screen/summary.json`, ep24000 SHA `82f7978b…`, seed44, 2,049~2,050 ep/cell:

| 지표 | off | riskcap |
|---|---:|---:|
| capture / crash / timeout | 72.83 / 24.63 / 2.54 % | 79.55 / 17.62 / 2.83 % |
| 정지여유 위반율 (요청) | 11.99 % | 13.69 % |
| **정지여유 위반율 (실행)** | 11.99 % | **9.87 %** |
| **접촉 직전 실행 속도** | 2.996 m/s | **2.029 m/s** |
| intervention / near-stop | 0 % / 0 % | 28.74 % / **0 %** |

거버너는 정지-불가능 스텝 13.69 % 중 **3.82 pp만 제거**하고 9.87 %를 남긴다.
접촉 직전 실행 속도 **2.029 m/s ≈ 하한 2.0**. 즉 **충돌은 하한에서 일어난다.**

### 1.2 해석 — 하한이 정지를 불가능하게 만드는 구간

거버너 자신의 파라미터(`brake=2.9609` 실측, `reaction=0.1`, `margin=0.45`)로
`stopping = v·0.1 + v²/(2·2.9609)`, `usable = clearance − 0.45`:

| clearance | cap | usable | stopping | margin |
|---:|---:|---:|---:|---:|
| 5.0 | 3.5355 | 4.55 | 2.46 | +2.09 |
| 3.0 | 2.0 | 2.55 | 0.88 | +1.67 |
| **1.33** | 2.0 | 0.88 | 0.88 | **0.00** |
| 1.0 | 2.0 | 0.55 | 0.88 | **−0.33** |

**clearance < 1.33 m에서 2.0 m/s 하한은 정지를 물리적으로 불가능하게 만든다.**
`stopping_margin`은 `speed_governor.py:229`에서 이미 계산되지만 **진단 전용이고 아무것도 gate 하지 않는다.**

### 1.3 이론 — magnitude-only의 원리적 한계

정지 장애물의 velocity obstacle은 **양의 원뿔**이다. 원점에서 반지름 r, 위치 p인 구를 향한 광선
`{v·t : t>0}`의 교차 여부는 v의 방향에만 의존한다(반각 `arcsin(r/|p|)`). 따라서 λ>0에 대해
`λv ∈ VO ⟺ v ∈ VO`이고 충돌 시각만 `t/λ`로 늘어난다.

> **양수 하한을 가진 magnitude-only 필터는 정지 장애물에 대해 forward invariance를 가질 수 없다.
> 충돌을 지연시킬 뿐이다.** crash 17.62 %는 튜닝 부족이 아니라 구조적 귀결이다.

**독립 실증**: MIT LL/Stanford, AIAA 2022 — 속도-only 회피 로직은 수평/수직 로직 대비
risk ratio **66–68배**, 경보 **3–4배**. 정면·추월 기하는 속도 변경으로 해소 불가.
<https://ar5iv.labs.arxiv.org/html/2204.14250>

λ→0을 허용한 TTC mode만 crash **6.83 %** 달성(riskcap 17.62 %). 하한이 없었기 때문이다.

### 1.4 deadlock의 실제 원인 (조사 보고 진단 정정)

문헌조사는 "`cap ∝ clearance`가 접근·측면 운동을 동일 처벌"한다고 진단하나 **우리 구현에는 해당하지 않는다**.
`directional_lidar_clearance`(`speed_governor.py:109`)는 명령 방향 주위 반폭 0.45 m 회랑 안 ray만 선택하므로
측면 명령 시 장애물은 회랑 밖 → clearance = max_range → 캡 해제. 이미 closing-rate 선택적이다.

실제 원인: **거버너는 방향을 고르지 않고 정책이 고른다.** 정책이 벽을 향해 계속 명령하면 거버너는 0까지
줄이고 정책은 여전히 벽을 향하므로 정지 상태가 지속된다. §1.3의 직접 귀결 — magnitude-only는 방향 오류를
고칠 수 없다. 따라서 해법은 (a) 안전층이 방향을 바꾸거나, (b) 정책에 탈출 gradient를 주는 것뿐이다.

**명명된 현상**: freezing robot problem, Trautman & Krause IROS 2010
<https://las.inf.ethz.ch/files/trautman10unfreezing.pdf>. 원인은 불확실성 증가이지 실제 장애물이 아니며,
해법은 임계값 완화가 아니라 더 나은 예측 모형이라고 명시한다.
CBF-QP의 undesired stable equilibria: Reis·Aguiar·Tabuada <https://arxiv.org/abs/2003.07819> **[P+S]**.
항공 RL에서의 독립 재현: Zhang et al. 2026이 저속에서 필터가 상대적으로 더 침습적이어서
**충돌이 아니라 정체**로 실패한다고 측정(62.25 % vs 67.75 % @3 m/s). 우리 16.59 %/23.57 % timeout의 재현.

---

## 2. 제동 능력에 3.3배 여유가 있다

sim 제원: 1.20 kg, 4 × 9.60 N = 38.4 N → **T/W 3.26**. 45° 틸트 수평 가속 한계 `g·tan45° = 9.81 m/s²`.
실측 유효 감속 **2.9609 m/s²** = 기체 능력의 **1/3.3**. 병목은 추력이 아니라 속도명령→자세 루프.

`a = 9.81`이면 3.5355 m/s 정지거리가 2.46 m → **0.99 m**. **캡을 낮추지 않고 안전을 얻는 경로가 존재한다.**

---

## 3. 문헌이 제시하는 대안

### 3.1 magnitude-only와 양립 가능한 것 (아키텍처 변경 최소)

| 방법 | 대상 | 보증 | 비고 |
|---|---|---|---|
| **DWA admissible velocity** <https://www.cs.cmu.edu/~dfox/abstracts/colli-ieee.abstract.html> | `v ≤ √(2·d·a_b)` | 정지 가능성 | 우리 `stopping_margin`을 v에 대해 푼 것과 동일. 항공 3D 확장 DWA-3D <https://arxiv.org/abs/2409.05421> **[H]** |
| **Passive motion safety** Bouraine·Fraichard·Salhi, Auton. Robots 2012 <https://link.springer.com/article/10.1007/s10514-011-9258-8> | Braking-ICS 회피 | **[P]** "충돌 시 로봇은 정지 상태" | **제한 FOV 전용 변형**. magnitude-only가 가질 수 있는 최강 성질 |
| **Mitsch et al. IJRR 2017** <https://arxiv.org/abs/1605.00604> | `‖p−o‖ > s²/2b + (V/b)s + ...` | **[P]** passive safety 정리 검증 | passive *orientation* safety = 불완전 센서 커버리지 변형 |
| **ACC용 CBF** Ames et al. TAC 2017 <https://arxiv.org/abs/1609.06408> | `h = d − v·τ − v²/2a`, 종방향 입력만 | **[P]** forward invariance | 방향 고정 + λ→0 가능할 때만 |
| **backup CBF / gatekeeper** T-RO 2024 <https://arxiv.org/html/2211.14361> | brake-to-hover 백업 | **[P]+[H]** | 쿼드의 자연 백업이 감속-호버 = magnitude-only. 단 0.35–0.82 m/s |

핵심: `stopping_margin ≥ 0`을 **진단에서 hard constraint로 승격**하는 것이 magnitude-only로
실제 정리를 가질 수 있는 유일한 구성이며, 동시에 死 파라미터 `hard_margin_m`을 되살린다.

### 3.2 방향 변경을 요구하는 것 (아키텍처 변경 필요)

APF(국소최소·진동으로 저자들이 폐기, Koren & Borenstein 1991), VO/ORCA(정지 장애물엔 §1.3),
collision-cone CBF, 그리고 아래 3.4의 HOCBF-QP — 제약 gradient가 명령에 직교 성분을 가지므로
광선으로 제한하면 QP가 infeasible.

### 3.3 지각 한계 기반 속도 상한 — 우리에게 없는 항

**Falanga·Kim·Scaramuzza, "How Fast Is Too Fast?", RA-L 2019** <https://rpg.ifi.uzh.ch/docs/RAL19_Falanga.pdf> **[P+S+H]**

> **v̄ = s / (τ + 2√(r/ā_lat))**

s = 감지 거리, τ = sense-to-act 총 지연, r = 로봇+장애물 반경, ā_lat = 최대 측방 가속.
제동은 `v ∝ √s`인데 회피는 `v ∝ s`(선형). 짧은 감지거리에서는 **지연이 지배**하고 긴 거리에서는 가속이 지배.
사례(r = 0.75 m, stereo s = 5 m, τ = 43 ms): ā_lat = 10/25/50 m/s²에서 **8.50 / 12.83 / 17.34 m/s**.

우리 캡 `3.5355 = 2.5·√2`에는 감지거리도 지연도 들어 있지 않다. 이 항이 논문의 신규성을 만든다.

### 3.4 미지 공간 처리 — 우리는 "미지 = 자유"로 가정 중

`speed_governor.py:139` `safe_range = torch.where(valid, lidar_m, max_range)` — 무반사 ray를 12 m 자유로 취급.
실기 LiDAR에서 위험.

| 접근 | 대표 | 비고 |
|---|---|---|
| 최악 가정 (미지=점유) | Richter·Vega-Brown·Roy ISRR 2015 <https://groups.csail.mit.edu/rrg/papers/richter_isrr15.pdf> **[H]** | 사각 코너에서 **4 → 1 m/s** 강제. deadlock 비용을 실측한 논문 |
| **이중 궤적** (주 궤적은 미지 진입 허용, known-free에 백업 상시 보유) | FASTER T-RO 2021 <https://arxiv.org/abs/2001.04420> 7.8 m/s **[H]** · **SUPER** Sci. Robotics 2025 <https://www.science.org/doi/10.1126/scirobotics.ado6187> **>20 m/s**, 실패율 35.9배 감소 **[H]** | 현 SOTA. 속도를 되찾는 기구 |
| 학습된 충돌확률 prior | Richter/Roy 동일 논문 | **+80 % 속도, 실증 안전 100 %**. OOD에서 최악 가정으로 복귀 |
| 능동 yaw 계획 | RAPTOR T-RO 2021 <https://arxiv.org/abs/2007.03465> | FOV 제한의 정답은 감속만이 아니라 **볼 곳을 계획**하는 것 |

### 3.5 탐지기 신뢰성과 안전층의 분리

우리 탐지기는 동색 디코이 존재 시 FTLR 90.27 %(v7). 따라서:

- **폐기해야 할 것**: 식별된 장애물 집합 + 추정 속도에 의존하는 모든 것 — VO/ORCA, NavRL의 VO shield를
  추적 결과로 먹이는 방식, collision-cone CBF, occlusion-aware MPC의 agent 가설.
- **영향받지 않는 것**: 원시 range/점유 기반 CBF(<https://arxiv.org/abs/2504.15850> **[H]**,
  <https://arxiv.org/html/2505.02294>), free-space + backup(gatekeeper), visibility CBF.
  디코이는 그냥 "점유 공간"이 되므로 식별 오류의 영향이 원리적으로 없다.
- **조건부**: conformal prediction 래퍼(<https://doi.org/10.1177/02783649251378151>)는 exchangeability
  하에서만 marginal coverage. 디코이가 체계적으로 등장하면 분포 이동이라 보증이 조용히 무효화된다.

> **결론: 안전층은 탐지기를 소비하면 안 된다. 원시 LiDAR 점유만 쓴다.
> 탐지기는 task 입력이지 safety 입력이 아니다.**

---

## 4. 실기 검증된 시스템 (속도 대조)

| 시스템 | 연도/venue | 필터 | 속도 | 핵심 수치 | 등급 |
|---|---|---|---|---|---|
| **SUPER** <https://www.science.org/doi/10.1126/scirobotics.ado6187> | 2025 Sci.Rob. | 이중 궤적 | **>20 m/s** | 실패율 35.9배↓, 2.5 mm 전선 회피 | H |
| **Zhang et al.** <https://arxiv.org/abs/2602.08653> | 2026 | **HOCBF-QP on RL 정책** | **7.5 m/s** 실기 | 성공 3 m/s 88.75 vs 84.50 %, **7 m/s 75 vs 52.25 %**. QP 90.6 µs. 개입률 13.06 % | S+H |
| **Saviolo et al.** <https://arxiv.org/abs/2409.11962> | 2025 | TTC→sigmoid로 NMPC 가중 조절 | 7.2 m/s | **1.2 kg, 25 cm, T/W 4:1 — 우리 기체와 동급** | H |
| FASTER <https://arxiv.org/abs/2001.04420> | 2021 T-RO | 백업 궤적 | 7.8 m/s | | H |
| **NavRL** <https://arxiv.org/abs/2409.15634> | 2025 RA-L | VO shield, **배포 전용** | **2.0 m/s** | 충돌/20회: 정적 0.95→0.65, 동적 2.70→0.85 | S+H |
| gatekeeper <https://arxiv.org/html/2211.14361> | 2024 T-RO | backup commit | 0.35–0.82 m/s | 중앙값 3.2–3.4 ms, MPC 대비 3–10배 빠름 | P+H |

**우리와 가장 가까운 것은 Zhang et al. 2026** — end-to-end RL 정책 + 깊이 기반 QP 필터, 같은 센서 클래스.
필터 이득이 **속도와 함께 커진다**(3 m/s 4 pp → 7 m/s 23 pp)는 것이 핵심 관찰이다.

HJ reachability·shielding·CPO/PPO-Lag는 소형 쿼드 2–4 m/s 실기 실증이 **없다**.

---

## 5. 학습 시 필터 적용 여부 — 우리 데이터가 이미 답을 갖고 있다

### 5.1 외부 증거

| 근거 | 발견 |
|---|---|
| **CBF-RL** <https://arxiv.org/abs/2510.14959> **[S+H]** | 필터로 학습 후 **필터 없이 평가 → 38.7 %**. 개입 페널티 추가 시 **92.7 %**. 보상만 91.9 %. "정책이 필터에 기댄다"의 유일한 정량 증거 |
| Oh·Nguyen·Hu·Fisac <https://arxiv.org/abs/2510.18082> **[P+S]** | 충분히 관대한 필터 하에서 학습하면 **동일 필터로 배포할 때** 최적 안전 정책과 동등한 수익. 학습 중 위반 0 |
| 해양 PSF <https://arxiv.org/abs/2312.01855> **[S]** | 학습 중 필터가 수렴을 늦추지 않음, 오히려 더 적은 episode |
| NavRL, Zhang et al. | 둘 다 **배포 전용**을 선택해 RA-L/강한 venue 게재 |

**증거 강도: 중간.** "필터에 기댄다"는 논문 1편이 지탱하며 항공 재현 없음.

### 5.2 우리 내부 증거 (seed45 uniform, 205 bars, `navrl_v2_riskcap_postadapt`)

off **70.03 / 27.87** → source+riskcap **78.20 / 17.80** (거버너 단독 **+8.17 pp**)
→ trained+riskcap **81.94 / 15.67** (적응 단독 **+3.74 pp**)

**학습 시 필터를 켜면 +3.74 pp를 더 얻는다는 우리 자신의 측정이 이미 있다.**

### 5.3 그런데 현 계보는 그걸 버리고 있다

- `train_navrl_v2_search.sh:115` — `_require_routed_value NAVRL_SPEED_GOVERNOR off` (학습 시 **강제 off**)
- `eval_navrl_v2_density_speed_map.sh:45` — `NAVRL_SPEED_GOVERNOR=riskcap` (평가 시 **on**)

→ ref5in 계보는 train/deploy 분포 불일치를 안고 있고 §5.2의 +3.74 pp를 포기하고 있다.

---

## 6. 검증되지 않은 것 — 계획에 반드시 들어가야 할 공백

1. **riskcap이 상수 2.0 캡을 이긴다는 증거가 없다.** `fixed 2.0`은 seed42에서 **78.53 / 16.06 / 5.42**,
   `riskcap`은 seed44에서 **79.55 / 17.62 / 2.83**. 시드가 달라 비교 불가.
   개입률 97 % vs 28.74 %로 완전히 다른데 결과 차이는 노이즈 수준.
   **"열린 방향에서 해제한다"는 riskcap의 핵심 기구가 값을 한다는 증거가 존재하지 않는다.**
2. **속도/틸트/요 상한이 한 번도 ablate 되지 않았다** (`MAX_VELOCITY=2.5`, `MAX_TILT_DEG=45`,
   `YAW_RATE_MAX=2.5/3.0`). `3.5355 = 2.5·√2`는 유도값이지 최적값이 아니다.
   리뷰어의 첫 질문은 "거버너 이득이 그냥 천천히 난 것 아닌가"이다.
3. **2×2의 빈 칸**: 거버너로 학습 → 거버너 **없이** 평가. CBF-RL의 의존성 시험. 미측정.
4. **밀도 의존성의 귀속 미확정** — Δ가 130→220 bars에서 +4.54 → +9.27 pp로 커지지만
   체크포인트와 거버너가 동시에 바뀐 confound. 분리하려면 ep24000+riskcap을 같은 밀도 격자에서 측정.
5. `near_stop_rate ≤ 5 %` 사전등록 게이트는 riskcap에서 **0.0 %**였다 — 하한 때문에 **실패할 수 없는 게이트**로
   판별력이 없었다. 향후 게이트 설계 시 주의.
6. 수직(z) 명령 무규제 (`navrl_task.py:4577`이 `command_xy`만 전달).
7. 회랑이 직선 무한 가정 — 정책이 다음 스텝에 방향을 바꾸는 것을 모델링하지 않음.

---

## 7. RA-L 리뷰어가 요구할 것

**ablation**: 단일 체크포인트 + 동일 episode seed의 2×2 (학습 시 거버너 × 배포 시 거버너),
거버너 축은 off / 현행 riskcap / 정지거리 법칙 / 지각한계 법칙 / clearance / TTC.
**기각된 두 변형(16.59 %·23.57 % timeout)을 반드시 게재한다** — "왜 뻔한 안전 법칙을 안 썼나"의 실측 답이다.

**필수**:
- 천장 ablation (`max_velocity` ∈ {1.5,2.0,2.5,3.0,3.5}, `max_tilt` ∈ {30°,45°,60°}). 템플릿:
  FLYINGTRUST <https://arxiv.org/abs/2510.26588> (36 플랫폼 × 7 장면, CI 포함)
- **모든 점에서 crash·capture·timeout 3개 동시 보고.** 충돌을 timeout으로 바꾼 필터는 도움이 안 됐다
- **속도 계층별 delta 보고** (Zhang et al.은 3 m/s 4 pp → 7 m/s 23 pp; 집계 하나로는 효과가 숨는다)
- 통계: **matched-seed paired 검정**(McNemar/paired bootstrap), Wilson 95 % CI, 다중비교 보정,
  학습 seed ≥ 5, 셀당 평가 episode ≥ 500~1000
- intervention rate + 최소거리 분포 (보수성 판정용)
- Pareto plot (`slow_dist`/`release_dist`/floor 스윕). 관례: <https://arxiv.org/abs/1908.01883>
- `t_react`는 **측정된** end-to-end 지연이어야 하고 민감도를 제시. Falanga 식에서 짧은 감지거리일수록 지배항

**우리 강점**: 사전등록 게이트는 이 분야 규범이 아니다. 논문에 명시하면 차별점이 된다.

**"not found" 두 건** (신규성 주장 가능):
(a) **RSS의 항공 이식이 존재하지 않는다.** 정지거리 법칙을 쿼드로터용 RSS 형태로 쓰면 인용 가능한 신규성.
    원본 <https://arxiv.org/abs/1708.06374>. 우리 `stopping_margin`은 `a_max,accel = 0`, `v_f = 0`인 RSS다.
    항공 유사물은 DAIDALUS/well-clear <https://nasa.github.io/daidalus/> **[P]** — 거리·시간 **연언** 트리거로,
    단일 TTC 임계보다 덜 민감하다(우리 TTC mode가 23.57 % timeout으로 터진 것과 대조).
(b) 로보틱스 평가에 사전등록 규범이 없다.

---

## 8. 조사에서 도출된 후보 법칙 (참고용, 미채택)

```
d_eff(u)   = clearance(u) − r_robot − m_map
v_brake(u) = −a_dec·t_react + sqrt((a_dec·t_react)² + 2·a_dec·d_eff(u))   # stopping_margin=0을 v로 푼 것
v_perc     = s_eff / (t_react + 2·sqrt(r / a_lat))                        # Falanga eq.9
cap(u)     = min(v_max, v_brake(u), v_perc)
```

`a_dec`는 가정하지 말고 **측정**한다(현재 2.9609, 45° 틸트 이론 한계 9.81 — §2).
`t_react`는 지각+추론+자세루프 상승시간의 실측 end-to-end.
이 법칙 하에서 `stopping_margin ≥ 0`은 **구성상 성립**하므로, 음수 값은 버그이거나 모델오차 사건이 되고
그 분포 자체가 게재 가능한 안전여유 plot이 된다.

하한을 유지하려면 안전이 아니라 **liveness 완화**로 제시하고, 탈출 조건(정지가능 후퇴 방향 존재,
또는 ≤ N 스텝)을 명시하며 그로 인한 여유 위반을 계수해야 한다.

---

## 부록: 진입점

- Hsu·Hu·Fisac, *The Safety Filter: A Unified View*, Annual Review of Control 2024 — <https://arxiv.org/pdf/2309.05837>
- Brunke et al., *Safe Learning in Robotics*, Annual Review of Control 2022 — <https://doi.org/10.1146/annurev-control-042920-020211>
- Krasowski et al., *Provably Safe RL* — <https://arxiv.org/pdf/2205.06750>
