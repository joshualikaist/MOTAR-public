# 사전등록 — A8: 필터와 함께 재적응 학습 (분포 밖 문제의 직접 검사)

작성 2026-09-05 (Fable 5.1). **초안 — 사용자 승인 전 실행 금지.** 선행: A7(`prereg_2026-09-05_a7_arc_attribution.md`,
판정 `LAW_CARRIES_SAFETY` ×2 시드, 205 bars `DENSITY_LIMITED`).

## 0. 질문

A7까지의 모든 arm은 **동결 정책 위의 추론 시점 필터**다. 초안 §7은 이를 "정책이 riskcap과 함께 적응했으므로
다른 필터는 분포 밖"이라 썼는데, 이는 절반만 맞다:

| 조건군 | 정책 | 학습 시 거버너 | 함의 |
|---|---|---|---|
| 70 bars (A4·A7 P1·P2) | ref5in ep1900 | **off** (`train_navrl_v2_ref5in_d1_adapt.sh` L63 고정) | 네 필터가 **똑같이** 분포 밖 |
| 205 bars (09-02·A7 P3) | ep25000 | riskcap 1,000 epoch 동반 | riskcap만 분포 안 |

그래서 A7의 2×2는 riskcap 편향이 없지만, **필터와 함께 학습하면 순위가 바뀔 수 있다**는 한계는 그대로다.
구체적 위험 셋:

- **Q1 비용 회수.** 정지법칙의 비용(timeout 2배, 3 m 미만 체류 3배)은 정책이 캡에 잘릴 속도를 요구하기
  때문일 수 있다. 정책이 캡을 알면 요구 자체를 바꿔 비용이 줄 수 있다.
- **Q2 법칙 이득 유지.** 반대로, 정책이 필터를 알면 riskcap 아래에서도 스스로 감속을 배워 법칙 간 crash 차이가
  사라질 수 있다. 그러면 논문의 "안전은 법칙이 산다"는 추론 시점에만 성립하는 주장이 된다.
- **Q3 필터 의존.** 강한 필터와 함께 학습한 정책은 필터를 떼면 더 위험해질 수 있다(09-02 Q3 `FILTER_DEPENDENT`,
  off − riskcap +9.23 pp; CBF-RL 문헌의 38.7 % 붕괴). 실기 배치에서 필터 고장 시 후과.

## 1. 설계

**출발점** ref5in ep1900(`197ea269…`, 70 bars, 거버너 off, 목표 22.5~28 m). 각 arm에 정확히 **1,000 epoch**
(ep1900 → 2900) 추가. D1과 같은 계약: seed 197, 128 envs, LR 1.5e-5, 70 bars 고정, `expandable_segments:True`.
바뀌는 것은 **학습 중 거버너 mode 하나**다.

| 학습 arm | 학습 시 거버너 | 역할 |
|---|---|---|
| **T0** | `off` | 추가 1,000 epoch 자체의 효과를 분리하는 대조군. 모든 비교의 기준. |
| **T1** | `riskcap` | 현 채택 필터와 동반 적응 |
| **T2** | `dwa_arc` | 최선의 추론 시점 arm(정지법칙 + 원호)과 동반 적응 |

`stopcap` 동반학습은 넣지 않는다 — A7에서 dwa_arc가 같은 crash에 비용만 낮으므로 stopcap을 배우게 할 이유가 없다.
거버너 파라미터는 A7 값(fixed 2.0 · free 3.5355 · half 0.45 · margin 0.45 · slow 3 · release 5 · brake 2.0 · reaction 0.1)
을 학습·평가 모두에 바이트 동일하게 쓴다.

**평가** 70 bars · N=0 · ref5in 평가 계약(`run_navrl_distractor_envelope.evaluation_env("v7_n0")`) · **seed 521**
(미사용: 기존 42 44 49 211 367 373 383 389 421 433 449 461 479 481 487 491 497 503 509) · 2,049 ep/cell · 한 루트·한 소스 번들.

| cell | 정책 | 평가 필터 | 답하는 것 |
|---|---|---|---|
| S/off, S/riskcap, S/dwa_arc | 소스 ep1900 | off / riskcap / dwa_arc | 새 seed의 A7 구조 재현(기준선) |
| T0/off, T0/riskcap, T0/dwa_arc | T0 | off / riskcap / dwa_arc | 추가 학습만의 효과; 이후 비교의 대조군 |
| T1/riskcap, T1/off | T1 | riskcap / off | 동반 적응 이득, 필터 의존 |
| T2/dwa_arc, T2/off | T2 | dwa_arc / off | 동반 적응 이득, 필터 의존 |

10 cell. 학습 3 arm × 2.0~2.6 h(실측 9 s/epoch, `detrange-stage1` 동일 계약) + 평가 10 × 12 min ≈ **9~10 h GPU**,
순차(peak VRAM 7.2/8 GB라 병행 불가).

## 2. 판정 규칙 (결과 이전 동결)

주지표 crash, 부지표 capture·timeout·개입률·<3 m 체류. 95 % Wald CI(A5 `_delta_ci`), 임계 3 pp. 모든 차이는 pp.

**Q1 비용 회수** (같은 필터 아래, 동반 적응 vs 대조 학습)
- `C_arc = T2/dwa_arc − T0/dwa_arc`, `C_risk = T1/riskcap − T0/riskcap` (timeout·capture·crash 각각)
- `ADAPTATION_RECOVERS_COST` — C_arc의 timeout ≤ −2 pp·CI 0 제외 **이고** crash CI 0 포함 또는 음
- `ADAPTATION_COSTS_SAFETY` — C_arc의 crash ≥ +3 pp·CI 0 제외
- `NO_ADAPTATION_EFFECT` — timeout·capture·crash 모두 CI 0 포함

**Q2 법칙 이득 유지** (각자 자기 필터와 적응한 뒤)
- `L_adapt = T2/dwa_arc − T1/riskcap` (crash). A7의 동결값은 −8.68 / −7.71.
- `LAW_GAIN_PERSISTS` — ≤ −3·CI 0 제외 · `LAW_GAIN_ERASED` — CI 0 포함 · `LAW_GAIN_REVERSED` — ≥ +3·CI 0 제외

**Q3 필터 의존** (필터를 뗀 상태끼리)
- `D_arc = T2/off − T0/off`, `D_risk = T1/off − T0/off` (crash)
- `FILTER_DEPENDENT` ≥ +3·CI 0 제외 · `INTERNALIZED` ≤ −3·CI 0 제외 · `NEUTRAL` 그 외. arm별로 낸다.

**대조 건전성** `T0/x − S/x`가 세 필터에서 |Δ| < 3 이면 `CONTROL_STABLE`; 하나라도 ≥ 3·CI 0 제외면 추가 학습
자체가 유효한 것이므로 Q1~Q3의 모든 비교는 T0 기준으로만 읽고 S 기준 수치는 인용하지 않는다(이미 그렇게 설계됨).

**판정 → 초안**

| 결과 | 문장 |
|---|---|
| `ADAPTATION_RECOVERS_COST` + `LAW_GAIN_PERSISTS` | 정지법칙+원호는 동반 학습으로 비용까지 회수하며 안전 이득을 유지한다 → 채택 후보 |
| `LAW_GAIN_ERASED` | "안전은 법칙이 산다"는 추론 시점 한정. 정책이 필터를 알면 법칙 차이가 사라진다 |
| `FILTER_DEPENDENT`(T2) | 강한 필터는 정책을 의존적으로 만든다 — 실기에서 필터 고장 시 후과를 §7에 명시 |
| `INTERNALIZED` | 필터가 교사 역할을 한다 — 추론 시 필터 제거 가능성 |

## 3. 예측 (동결)

- **P-1** `ADAPTATION_RECOVERS_COST`: C_arc timeout −2~−5 pp, 개입률 −1~−2 pp, crash CI 0 포함. 근거: ep25000 적응이 timeout −1.56 pp를 만든 것과 같은 기제, 정지법칙은 캡이 커 회수 폭이 더 큼.
- **P-2** `LAW_GAIN_PERSISTS`: L_adapt −5~−8. 근거: 09-02 적응은 crash를 거의 안 바꿨다(−0.32 pp) — 정책은 timeout을 줄이지 crash를 줄이지 못했다.
- **P-3** T2 `FILTER_DEPENDENT`(+3~+8), T1 `NEUTRAL`. 근거: 정지법칙이 잘라주는 상황을 정책이 겪지 않게 되어 그 상황의 회피를 잃는다.
- **P-4** `CONTROL_STABLE`.
- P-2가 틀리면(`LAW_GAIN_ERASED`) 논문 §1-2를 "추론 시점 한정"으로 고쳐야 하며 이것이 이 실험의 최대 위험이다.

## 4. 기계 무결성

- **M1** 학습 세 arm: 같은 커밋, 깨끗한 작업트리, `train_source_receipts` 기록, 소스 ep1900 SHA 게이트, 정확히 1,000 epoch(마지막 체크포인트 epoch 2900).
- **M2** 학습·평가의 거버너 파라미터가 A7 값과 바이트 동일(`condition.speed_governor_*`; 학습은 run 로그의 계약 줄).
- **M3** 평가 10 cell이 한 루트·한 소스 번들·같은 `runtime_source_manifest_sha256`·`runtime_git_dirty=false`.
- **M4** 평가 seed 521 미사용 확인(§1 목록). S/riskcap 등 소스 cell은 새 seed라 과거값 일치 검사는 없고 **기록만** 한다.
- **M5** 학습 중 소스 무변경(A3·A4 교훈). 학습 3 arm + 평가 10 cell을 한 드라이버로 순차 실행.
- **M6** OOM: D1 선례대로 epoch 0 OOM은 VOID 보존 후 allocator 설정만으로 재시도.

## 5. 중단 조건

- 어느 학습 arm이든 trailing-100 crash가 소스 ep1900의 학습 로그 값 + 10 pp를 넘으면 그 arm 중단·VOID·보고. 다른 arm은 계속.
- `CONTROL_STABLE` 실패는 중단 사유가 아니다(설계가 T0 기준).
- 평가 cell 실패(evaluator ≠ 0)는 해당 루트 전체 VOID(부분 재개 금지).

## 6. 구현 (실행 전 전부 완료·테스트·커밋)

1. `aerial_gym/rl_training/rl_games/train_navrl_v2_ref5in_a8_readapt.sh` — `train_navrl_v2_ref5in_d1_adapt.sh` 복제.
   CKPT = ep1900(`197ea269…`) SHA 게이트, `MAX_EPOCHS=2900`, `NAVRL_SPEED_GOVERNOR=${A8_MODE}`(off|riskcap|dwa_arc만 허용),
   거버너 파라미터 9개 export, `AERIAL_RUN_TAG=v2-ref5in-a8-readapt-${A8_MODE}-s197`, 나머지 D1 계약 그대로.
2. `tools/run_navrl_a8_readaptation.py` — `run_navrl_contact_geometry.py` 구조. `evaluation_env("v7_n0")`에
   `NAVRL_SEED=521`, 거버너 파라미터, `NAVRL_CONTACT_GEOMETRY=1`, star-convex shadow 0. cell마다 체크포인트 경로를
   evaluator 인자로 넘기고 SHA를 기록. 결과 루트 `results/navrl_a8_readaptation_seed521/{S,T0,T1,T2}_{filter}/`.
   소스 동결 검사·기존 루트 거부·부분 재개 금지는 A7과 동일.
3. `tools/build_a8_readaptation_table.py` — §2를 기계 적용. `docs/a8_readaptation_table.md` + `results/navrl_a8_readaptation_summary.json`.
4. 테스트: 런처 mode 화이트리스트(3개만), 파라미터 pin, 체크포인트 SHA 거부, 평가 도구 cell 구성(10개·순서), 요약기 판정 규칙·경계.
5. 드라이버: T0 학습 → T1 → T2 → 평가 10 cell → 요약기. 학습 완료 체크포인트는 `nn/last_gen_ppo_ep_2900_*.pth`.

## 7. 범위 밖

205 bars 동반학습(별도, `DENSITY_LIMITED` 기제가 먼저) · stopcap·omni 동반학습 · 거버너 파라미터 튜닝 ·
방향 개입 · 커리큘럼 변경 · 1,000 epoch 초과.
