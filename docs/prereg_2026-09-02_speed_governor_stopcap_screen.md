# 사전등록 — ep25000 speed-governor 재설계 스크린 (stopcap)

등록 2026-09-02, **결과 생성 이전에 판정 규칙 동결**. 근거: `docs/safety_filter_survey_2026-09-02.md`.

## 0. 배경 및 가설

seed44 receipt 재해석과 VO 원뿔 논증(조사 문서 §1)에 따르면 riskcap의 2.0 m/s 하한은
clearance < 1.33 m에서 정지를 물리적으로 불가능하게 만들고, 접촉 직전 실행 속도 2.029 m/s가
이를 실증한다. `stopping_margin`은 계산되지만 아무것도 gate 하지 않는다.

**가설 H1**: 정지거리 admissible 법칙(stopcap)은 하한 없이도(λ→0 허용) deadlock 없이 crash를
riskcap 아래로 내린다. 근거: stopcap은 정지를 보장하는 법칙 중 가장 덜 보수적이다 —
clearance 1.5 m에서 cap 2.22 m/s (clearance mode 1.46, ttc 1.05). 기각된 두 모드의 timeout
폭발(16.59 % / 23.57 %)은 과잉 보수성에서 왔고, 정지 가능성 자체에서 온 것이 아니다.

**가설 H2**: riskcap의 핵심 기구(열린 방향 해제)는 동일-seed에서 상수 2.0 캡 대비 이득이
불확실하다. 현재 교차-seed 수치(78.53/16.06 vs 79.55/17.62)는 판정 불가.

**가설 H3**: ep25000은 마지막 1,000 epoch을 riskcap과 함께 학습했으므로 필터 제거 시 성능이
악화된다(CBF-RL의 필터 의존성, arXiv:2510.14959의 38.7 % 붕괴와 같은 방향, 크기는 미상).

## 1. 고정 조건

| 항목 | 값 |
|---|---|
| checkpoint | frozen ep25000 `runs/ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1/nn/last_gen_ppo_ep_25000_rew_39.742134.pth` |
| checkpoint SHA-256 | `f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40` |
| seed | **49** (기존 캠페인 40–48 미사용 확인) |
| bars / action / reflection | 205 / deterministic / original |
| episodes per cell | 2,049 |
| runtime | main / base_sim / 128 envs / cluster_sector (기존 스크린 동일) |
| 실행 기기 | RTX 3070 (기존 governor 캠페인과 동일 기기; 1650 Ti 수치와 혼용 금지) |
| governor 공통 파라미터 | free 3.53553390593, half_width 0.45, margin 0.45, slow 3.0, release 5.0, ttc_s 1.2, brake 2.9608856678, reaction 0.1 — seed42/44 스크린과 바이트 동일 |

## 2. Arms (5 cells)

| tag | mode | 답하는 질문 |
|---|---|---|
| `off` | off | Q3 필터 의존성 |
| `fixed2p0` | fixed 2.0 | Q1 비교 대상 |
| `riskcap` | riskcap | 현행 기준 |
| `stopcap` | **stopcap (신규)** | Q2 채택 심사 |
| `ttc` | ttc | 참조 전용 — crash 하한 재현 여부. **게이트 없음** |

stopcap 법칙 (`speed_governor.py`에 신규 구현):

```
usable = clearance − hard_margin            # hard_margin(0.45)이 여기서 되살아난다
cap    = clamp( −a·t_r + √((a·t_r)² + 2·a·usable), 0, free_cap )   # a=brake, t_r=reaction
```

`stopping_margin_executed = usable − (v·t_r + v²/2a) ≥ 0`이 **구성상** 성립한다(스텝 간 clearance
변화·이산화 오차 제외). 하한 없음. 방향 불변(기존 원칙 유지). 신규 파라미터 없음 — brake/reaction/
margin은 기존 스크린에서 이미 동결된 값이다. **본 스크린은 어떤 파라미터도 튜닝하지 않는다.**

## 3. 판정 규칙 (결과 이전 동결)

CI는 기존 관례(두 비율 정규근사 95 %). 비교는 모두 동일 seed·동일 체크포인트·동일 기기.

### M1 — 기계 무결성 (fail-closed, 최우선)
`stopcap` 셀의 `negative_stopping_margin_executed_rate ≤ 0.01`.
위반 시 **IMPLEMENTATION_VOID** — 법칙이 여유 ≥ 0을 보장하므로 1 %를 넘는 위반은 구현 결함이다.
이 경우 Q2는 판정하지 않고 전체 셀을 증거에서 제외한다.

### Q1 — riskcap 해제 기구의 값어치
Δcap = capture(riskcap) − capture(fixed2p0):
- **MECHANISM_SUPPORTED**: Δcap CI95 하한 > 0
- **MECHANISM_UNSUPPORTED**: CI가 0 포함 (해제 기구 무실증 → 단순화 후보)
- **MECHANISM_REVERSED**: CI 상한 < 0

### Q2 — stopcap 채택 (GO 조건 3개 동시 충족)
1. crash(stopcap) − crash(riskcap) ≤ −0.03 **이고** 그 CI95 상한 < 0
2. timeout(stopcap) ≤ 0.05
3. capture(stopcap) − capture(riskcap) ≥ −0.02

- 셋 다 충족 → **GO** (stopcap을 배포 필터 후보로 채택, 후속: stopcap 동반 적응 학습)
- 1만 충족 → **SAFETY_ONLY** (crash는 내렸으나 liveness/capture 비용 — 채택 보류, 원인 분석)
- 1 미충족 → **NO_GO** (§0 진단 재검토)

임계 근거: −3 pp는 riskcap 자체 효과(−7.02 pp)의 절반이며 n≈2,050에서 관측된 CI 반폭(~2.5 pp)
초과. timeout 5 %는 기존 스크린 게이트와 동일. capture −2 pp는 안전 개선 대가로 수용 가능한
상한으로 사전 선언한다.

### Q3 — 필터 의존성 (사실 기록이지 실패가 아님)
Δcrash = crash(off) − crash(riskcap):
- **FILTER_DEPENDENT**: Δcrash ≥ +0.05 → 배포 시 필터 필수를 논문에 명시
- **FILTER_INDEPENDENT**: Δcrash ≤ +0.02
- 그 외 **INCONCLUSIVE**

### 예측 (기제 확인용, 게이트 아님)
- stopcap의 `contact_executed_speed_mps` < 2.0 (하한 소멸의 직접 흔적)
- stopcap intervention은 riskcap(28.74 %)보다 낮거나 비슷 — 법칙이 clearance ≥ ~3.2 m에서 무개입
- ttc 참조 셀: crash ≤ 10 %, timeout ≥ 15 % (seed42 재현 방향)
- `near_stop_rate` 게이트는 **설정하지 않는다** — seed44에서 하한 때문에 실패 불가능했던 게이트로
  판별력이 없음이 확인됐다. 대신 timeout이 liveness를 직접 측정한다.

## 4. 명시적 범위 밖 (이 스크린에서 하지 않는 것)

- 천장 ablation (`MAX_VELOCITY`/`MAX_TILT_DEG`/`YAW_RATE_MAX`) — 별도 캠페인
- 지각한계 항 `v_perc = s/(τ+2√(r/a_lat))` — end-to-end t_react 실측 선행 필요
- stopcap 동반 재학습 — Q2 GO 이후에만
- 방향 개입·백업 궤적 — 아키텍처 변경, 본 스크린은 magnitude-only 원칙 유지
- riskcap 파라미터 재튜닝 — 없음 (모든 파라미터 기존 동결값)

## 5. 산출물

`results/navrl_v2_ep25000_stopcap_seed49_screen/{off,fixed2p0,riskcap,stopcap,ttc}/`
+ `summary.{md,json}` (`tools/summarize_navrl_v2_stopcap_screen.py`).
summary에는 단수 `verdict` 키를 두지 않는다 — `verdict_m1`, `verdict_q1`, `verdict_q2`,
`verdict_q3` 네 키로 분리 기록한다.
