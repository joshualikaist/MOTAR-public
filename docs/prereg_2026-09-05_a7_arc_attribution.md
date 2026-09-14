# 사전등록 — A7: `dwa_arc` 이득의 귀속 분리 (2×2 요인 설계 + 재현 + 밀도)

작성 2026-09-05. 계획: Fable 5.1. 구현·실행 전 감사: Codex(`docs/a7_preexecution_audit_2026-09-05.md`).
**개정 1 (2026-09-05, 결과 이전)**: 감사가 찾은 `arc_clearance` yaw=0 결함 수정에 따라 P1을 같은 커밋의 4 arm으로 확장(합계 12 arm),
라벨 우선순위·M5 허용오차를 동결, A4 실제 분모 2,051로 정정. 사용자 승인 후 실행.
선행: `docs/prereg_2026-09-05_a4_geometry_baselines.md`, `docs/a5_ablation_table.md`, `docs/paper_draft_2026-09-05.md` §8-1.

## 0. 질문과, 원안을 버린 이유

A4는 `dwa_arc`(원호)가 `riskcap`(직선)보다 crash −7.70 pp라고 보고했고, 초안은 이를
"같은 법칙·같은 스캔·기하만 교체"라고 썼다. **그 문장은 틀렸다.** 코드
(`speed_governor.py:315`)와 A4 사전등록 §1 모두 `dwa_arc`가 **stopcap의 정지거리 법칙**을 쓴다고
명시한다. riskcap은 바닥 2.0 + 해제 법칙이다. 따라서 riskcap→dwa_arc는 **법칙과 기하를 동시에**
바꾼 비교이고, 기하만 바꾼 진짜 대조군은 이미 표 1에 있는 `stopcap`이다.

같은 법칙끼리 놓으면 (seed 509, A4):

| | crash | capture | timeout | 개입률 | <3 m 체류 |
|---|---:|---:|---:|---:|---:|
| `stopcap` (직선, 정지법칙) | 12.30 % | 67.94 % | 19.77 % | 19.3 % | 21.85 % |
| `dwa_arc` (원호, 정지법칙) | 11.07 % | 77.80 % | 11.12 % | 4.0 % | 6.25 % |
| 차이 | −1.2 pp (CI 반폭 ±2.0, 유의 아님) | +9.9 | −8.7 | −15.3 | −15.6 |

즉 현재 자료의 가장 단순한 해석은 **"정지법칙이 안전(−6.5 pp)을 사고, 원호는 그 법칙의 비용
(timeout·체류·과잉개입)을 걷어낸다"**이지 "원호가 더 안전하다"가 아니다. 개입을 15 pp 줄였는데
crash가 안 늘었으니, 걷어낸 개입은 선회 중 정면 멀리의 헛브레이크였다(원호 튜브는 선회반경
R에서 정면 √(0.9R) m 너머를 보지 않는다 — 결함이 아니라 정밀도다).

**원안(개입률을 4.0 %로 맞춘 약화 직선 riskcap)을 폐기하는 이유**: (i) 법칙 교란을 그대로 두고
기하 질문만 묻는다, (ii) 개입률을 맞추려면 여유 파라미터를 반복 조정해야 해 실행 수와 자유도가
늘어난다, (iii) 약화 직선이 떨구는 장애물과 원호가 떨구는 장애물이 달라 결국 답이 안 된다.

빠진 것은 **2×2 요인 설계의 한 칸**이다:

| | 직선 회랑 | 원호 튜브 |
|---|---|---|
| **riskcap 법칙** (바닥 2.0 + 해제) | `riskcap` ✔ | **`riskcap_arc` — 없음** |
| **정지거리 법칙** | `stopcap` ✔ | `dwa_arc` ✔ |

그리고 두 번째 위험: **stopcap의 부호가 실행 조건에 따라 뒤집힌다.**

| 실행 | 체크포인트 | 밀도 | brake | stopcap − riskcap crash |
|---|---|---:|---:|---:|
| 09-02 스크린 seed 49 | ep25000 | 205 bars | 2.9609 | **+5.36 pp** [+2.98, +7.73] |
| A4 seed 509 | ref5in ep1900 | 70 bars | 2.0 | **−6.47 pp** [−8.68, −4.26] |

초안 §1은 앞 줄을, 표 1은 뒷줄을 인용하며 둘을 화해시키지 않는다. 교란은 셋(체크포인트·밀도·brake)
이다. 정지법칙 위에 얹힌 원호 효과도 205 bars에서 뒤집힐 수 있으므로, 논문 최대 발견을 70 bars
한 시드에 걸어둔 셈이다.

세 질문:

- **Q1 (귀속)** 원호의 crash 기여는 법칙과 독립인가, 정지법칙 아래에서만 나타나는가?
- **Q2 (재현)** Q1의 답이 두 번째 시드에서 같은가?
- **Q3 (밀도)** 205 bars·ep25000에서 정지법칙과 원호의 효과는 어떻게 되는가?

## 1. arm

신규 mode 하나: **`riskcap_arc`** = riskcap 법칙 그대로 + clearance만 `arc_clearance`. yaw rate 0에서
`riskcap`과 정확히 일치한다(단위테스트로 고정). 나머지 arm은 기존 정의 그대로.

| arm | clearance 기하 | 상한 법칙 | 상태 |
|---|---|---|---|
| `riskcap` | 직선 회랑 ±0.45 m | 바닥 2.0 + 3~5 m 해제 | 기존 |
| `stopcap` | 직선 회랑 ±0.45 m | 정지거리 | 기존 |
| `dwa_arc` | 원호 튜브 ±0.45 m | 정지거리 | 기존 |
| **`riskcap_arc`** | 원호 튜브 ±0.45 m | 바닥 2.0 + 3~5 m 해제 | **신규** |

거버너 공통 파라미터는 **A4 값으로 통일**한다: fixed 2.0, free 3.53553390593, half_width 0.45,
margin 0.45, slow 3.0, release 5.0, **brake 2.0**, reaction 0.1. (09-02 스크린의 brake 2.9609·ttc 1.2는
쓰지 않는다. ttc_s는 이 네 mode에 무관. riskcap 법칙은 brake와 무관하므로 riskcap 수치는 brake와
상관없이 재현되어야 한다 — §5 M5.)

## 2. 세 파트

| 파트 | 질문 | seed | 밀도·체크포인트 | arm | 결과 루트 |
|---|---|---:|---|---|---|
| **P1** | Q1 | 509 | 70 bars · ref5in ep1900 | `riskcap`, `stopcap`, `dwa_arc`, `riskcap_arc` | `results/navrl_arc_attribution_seed509/` |
| **P2** | Q2 | 491 | 70 bars · ref5in ep1900 | `riskcap`, `stopcap`, `dwa_arc`, `riskcap_arc` | `results/navrl_arc_attribution_seed491/` |
| **P3** | Q3 | 49 | **205 bars · ep25000** (`f7022139…`) | `riskcap`, `stopcap`, `dwa_arc`, `riskcap_arc` | `results/navrl_arc_attribution_205bars_seed49/` |

**왜 P1이 4 arm인가 (개정 1).** 원안은 `riskcap`·`riskcap_arc` 2 arm에 A4의 `stopcap`·`dwa_arc`를
빌려 오는 것이었다. 실행 전 감사가 `arc_clearance`의 yaw=0 극한이 float32 상쇄로 직선과 어긋나는
결함(횡거리 0.46 m 광선이 튜브 안으로 들어옴; 배치 최대 0.107 m)을 찾았고 이를 수정했다. 수정된
함수는 A4 `dwa_arc`가 돌던 함수와 다르므로 옛 `dwa_arc`를 새 2×2에 빌려 쓸 수 없다. P1은 같은
커밋에서 4 arm 전부를 한 루트에 돌린다. 새 루트의 `riskcap`은 A4의 18.77 %(2,051 ep) 재현 검사
(§5 M5)로 남는다 — 판정을 막지는 않고 보고한다.

**왜 P2는 4 arm 전부인가.** seed 491 루트에는 `off`·`riskcap`만 있고 역시 옛 번들이다. 재현은
한 루트·한 번들 안에서 2×2가 완결되어야 깨끗하다. `off`는 이 질문에 불필요하다.

**왜 P3는 brake 2.0인가.** 09-02 스크린(brake 2.9609)의 `stopcap` 21.31 %는 이미 있다. 같은
seed·밀도·체크포인트에서 brake만 2.0으로 바꾼 `stopcap` 하나가 "부호 반전이 brake 탓인가"를
가른다. `riskcap`은 brake 무관이므로 15.95 %가 재현되어야 하고, 그것이 P3의 번들 영점 검사다.

에피소드 2,049/arm, 128 envs, deterministic/original, N=0 — 각 선행 실행과 동일.

## 3. 코드 변경 — 전부 GPU 착수 **전에** 끝내고 테스트·커밋한다

A3·A4 VOID의 재발 방지: **평가가 도는 동안 `aerial_gym/**`·`tools/**`·평가 셸을 건드리지 않는다.**
아래 전부를 먼저 끝내고, 테스트가 통과한 커밋 하나에서 세 파트를 모두 돌린다.

1. `aerial_gym/task/navrl_task/speed_governor.py`
   - `VALID_SPEED_GOVERNOR_MODES`에 `"riskcap_arc"` 추가.
   - `from_environ`의 riskcap 전용 검증(`release > slow`, `free ≥ fixed`)을 `mode in ("riskcap", "riskcap_arc")`로.
   - `apply_speed_governor`의 riskcap 분기를 `mode in ("riskcap", "riskcap_arc")`로. 정지법칙 분기는 손대지 않는다.
2. `aerial_gym/task/navrl_task/navrl_task.py` ~L5008: `elif mode in ("dwa_arc", "riskcap_arc")` → `arc_clearance`. yaw rate 출처(`robot_body_angvel[:, 2]`)는 dwa_arc와 동일.
3. `aerial_gym/rl_training/rl_games/eval_navrl_v2_density_sweep.sh`
   - L340·L342 화이트리스트에 `riskcap_arc` 추가.
   - **L374 riskcap 전용 검증**(`== "riskcap"`)도 `in ("riskcap", "riskcap_arc")`로. 화이트리스트와 별개인 세 번째 mode 분기다 — 빠뜨리면 검증만 건너뛰고 조용히 통과한다.
4. `tools/run_navrl_contact_geometry.py`
   - `_ALL_ARMS`에 `"riskcap_arc"` 추가.
   - `NAVRL_CG_ARMS="riskcap,riskcap_arc"` (쉼표 목록)로 arm 선택. 미지정 시 기존 동작 유지.
   - `NAVRL_CG_RESULT_ROOT`로 결과 루트 지정. 미지정 시 기존 `navrl_contact_geometry_seed{SEED}`.
   - `run_summarize`의 A1 게이트 `GATE[r["arm"]]`는 `off`/`riskcap` 이외 arm에서 KeyError — 미등록 arm은 게이트를 건너뛰도록.
5. 신규 런처 `aerial_gym/rl_training/rl_games/eval_navrl_v2_ep25000_arc_attribution.sh`
   (`eval_navrl_v2_ep25000_stopcap_screen.sh` 복제): 결과 루트 `results/navrl_arc_attribution_205bars_seed49`,
   arm 4개(`riskcap stopcap dwa_arc riskcap_arc`), 거버너 파라미터 §1의 A4 값(**brake 2.0, ttc 1.0**),
   체크포인트 SHA 게이트·덮어쓰기 거부·공유 소스 번들 유지. seed 49.
6. 신규 요약기 `tools/build_a7_arc_attribution_table.py` → `docs/a7_arc_attribution_table.md` + `results/navrl_arc_attribution_summary.json`.
   세 루트 + A4 루트 + 09-02 스크린 루트의 `*bars.json`만 읽는다(시뮬레이션 없음). §4의 양과 판정을 기계적으로 출력한다.
   CI는 `build_a5_ablation_table.py::_delta_ci`(두 비율 Wald, 95 %)를 그대로 쓴다.
7. 테스트 (`tests/test_navrl_speed_governor.py`)
   - `riskcap_arc`의 cap이 같은 clearance에서 `riskcap`과 1e-6 이내 일치 (M1).
   - `riskcap_arc`가 yaw rate 0에서 `riskcap`과 같은 clearance·같은 cap (M2).
   - `from_environ`이 `riskcap_arc`를 받고, riskcap과 같은 검증을 적용한다(release ≤ slow 거부).
   - `WhitelistsAgree`는 자동으로 새 mode를 검사한다. 추가로 셸 L374 검증 줄이 `riskcap_arc`를 포함하는지 문자열 검사 하나.
   - `python -m unittest discover -s tests`에서 기존 failures 8 / errors 8 이외 신규 실패 0.

## 4. 판정 규칙 (결과 이전 동결)

주지표 **crash**. 부지표 timeout · 개입률 · <3 m 체류 · capture. 임계 **3 pp**·95 % CI(A4와 동일).

정의 (모두 crash 차이, pp):

- `G_stop = dwa_arc − stopcap` (정지법칙 아래 기하 효과)
- `G_risk = riskcap_arc − riskcap` (riskcap 법칙 아래 기하 효과)
- `L_line = stopcap − riskcap` (직선 아래 법칙 효과)
- `L_arc = dwa_arc − riskcap_arc` (원호 아래 법칙 효과)

**P1 · Q1 (seed 509, 한 루트 4 arm)**

| 판정 | 조건 |
|---|---|
| `LAW_CARRIES_SAFETY` | L_line, L_arc 모두 ≤ −3 pp·CI 0 제외 **그리고** G_stop, G_risk 모두 CI 0 포함 |
| `ARC_CARRIES_SAFETY` | G_stop, G_risk 모두 ≤ −3 pp·CI 0 제외 |
| `INTERACTION` | G_stop·G_risk 중 하나만 CI 0 제외 |
| `ARC_HURTS_UNDER_RISKCAP` | G_risk ≥ +3 pp·CI 0 제외 |
| `INCONCLUSIVE` | 위 어느 것도 아님 |

**우선순위(동결).** 동시에 성립할 수 있는 쌍은 `ARC_HURTS_UNDER_RISKCAP`+`INTERACTION`뿐이다(다른
쌍은 조건이 서로 배타). 그 경우 더 구체적인 `ARC_HURTS_UNDER_RISKCAP`을 라벨로 하고 두 규칙이
맞았음을 함께 기록한다. 요약기는 `PRECEDENCE` 순서로 이를 기계 적용한다.
P3에서 `−3 < L_line < 0`이고 CI가 0을 제외하면 두 FLIP 라벨 모두 미성립 → `INCONCLUSIVE`로 보고하고 임계를 완화하지 않는다.

비용 축(부지표, 독립 판정): `ARC_REMOVES_COST` = stopcap→dwa_arc에서 timeout **과** 개입률이 모두
하락·CI 0 제외. `ARC_COST_NEUTRAL_UNDER_RISKCAP` = riskcap→riskcap_arc에서 timeout 변화 CI 0 포함.

**P2 · Q2 (seed 491, 한 루트 4 arm)**: P1과 같은 규칙으로 판정 라벨을 낸다.
`REPLICATED` = 라벨이 P1과 같고 L_line 부호 같음. `PARTIAL` = 라벨 다르나 L_line·G_stop 부호 같음.
`NOT_REPLICATED` = 그 외. `NOT_REPLICATED`면 §5 판정에서 A4 `GEOMETRY_MATTERS`를 초안에서 **철회**한다.

**P3 · Q3 (205 bars · ep25000 · seed 49 · brake 2.0)**

- 영점: `riskcap` crash가 09-02 스크린의 15.95 %와 일치(M5).
- `L_line^205 = stopcap − riskcap` (brake 2.0). 09-02의 +5.36 pp(brake 2.9609)와 대조:
  - `FLIP_IS_BRAKE` — L_line^205 ≤ −3 pp·CI 0 제외 (brake 2.0으로 바꾸니 205 bars에서도 정지법칙이 이긴다)
  - `FLIP_IS_DENSITY` — L_line^205 ≥ 0 이거나 CI 0 포함 (brake와 무관하게 205 bars에선 정지법칙이 못 이긴다)
- 2×2 라벨을 P1 규칙으로 205 bars에서도 낸다. P1과 다르면 **`DENSITY_LIMITED`** — 논문의 기하 주장에 조건을 명시한다.
  **해석 주의(감사 §3)**: P3는 밀도 외에 체크포인트(ep25000)·기체(`navrl_quad`)·초기 목표 거리(6~28 m)·시드도 다르다.
  라벨 이름과 달리 이는 **실행 조건 전이** 검사이며 밀도 단독 인과로 서술하지 않는다.
- `ARC_REMOVES_COST^205` 동일 규칙.

**판정 → 초안 문장**

| P1 라벨 | 초안 §1-3·§5의 헤드라인 |
|---|---|
| `LAW_CARRIES_SAFETY` + `ARC_REMOVES_COST` | "정지법칙이 안전을 사고 활력을 잃는다. 원호 기하는 그 활력 비용을 되돌려주되 안전을 더 주지는 않는다." DWA 인용은 "원호가 정밀하다"로 축소. |
| `ARC_CARRIES_SAFETY` | 현 초안 유지, 단 대조군을 stopcap으로 바꿔 수치 정정. "DWA가 옳았다" 유지 가능. |
| `INTERACTION` | "원호의 가치는 엄격한 법칙 아래에서만 나타난다" — 상호작용 자체가 결과. |
| `ARC_HURTS_UNDER_RISKCAP` | 원호는 만능이 아니다 — 약한 법칙에선 정면 가시거리 손실이 드러난다. |

## 5. 기계 무결성 (전부 요약기가 기계 검사)

- **M1** `riskcap_arc` cap 법칙 == `riskcap` (단위테스트).
- **M2** `riskcap_arc` clearance == `dwa_arc` clearance (같은 함수) 이고 yaw 0에서 == 직선 (단위테스트).
- **M3** 한 루트 안의 모든 arm이 같은 `checkpoint_sha256`·같은 `runtime_source_manifest_sha256`·`runtime_git_dirty == false`.
- **M4** 세 파트가 **같은 git 커밋**에서 돈다. 평가 중 소스 무변경.
- **M5 번들 영점**: 새 루트의 `riskcap`이 선행 값과 **소수 둘째 자리까지 일치** — seed 509: 18.77 % (385/2,051 ep), seed 49: 15.95 % (327/2,050 ep). 정확 일치 `PASS`, |Δ| ≤ 0.5 pp `PASS_INEXACT`(교차 루트 비교 허용하되 라벨에 "(M5 inexact)" 표기), |Δ| > 0.5 pp `FAIL`(§7). P1은 4 arm 한 루트이므로 M5는 보고용이고, M5가 판정을 막는 곳은 P3의 brake 대조(09-02 스크린과의 교차 루트)뿐이다.
- **M6** 거버너 파라미터가 세 루트 모두 §1의 A4 값과 바이트 동일(`condition.speed_governor_*`).

## 6. 예측 (결과 이전 동결 — A4 예측 2가 틀렸으므로 이번엔 더 명시적으로)

- **P1-1** `G_risk`는 ±2 pp 안, CI 0 포함. `riskcap_arc` 개입률 3~5 % (riskcap 5.8 %보다 낮음). 라벨 `LAW_CARRIES_SAFETY` + `ARC_REMOVES_COST`.
  근거: riskcap은 원래 개입이 적어 원호가 걷어낼 헛브레이크가 거의 없다.
- **P1-2** M5 통과(결정적 평가라면 정확 일치).
- **P2** `REPLICATED`.
- **P3** `FLIP_IS_DENSITY`: 205 bars에서 brake 2.0 `stopcap`은 timeout이 riskcap의 2배 이상이고 crash는 riskcap보다 3 pp 이상 낮지 않다.
  `dwa_arc`는 timeout을 riskcap ±3 pp로 되돌리지만 crash 이득은 3 pp 미만. 즉 **`DENSITY_LIMITED`**.
  근거: 09-02 스크린의 stopcap은 A4보다 *약한* 필터(brake 2.96 → 정지거리 짧음)였는데도 crash가 올랐다. 필터 강도가 아니라 밀도가 체류를 만든다는 쪽이 일관된다.
  이 예측이 맞으면 논문의 기하 주장은 "저밀도에서"로 한정되고 §8의 밀도 축 확장이 필수가 된다.

## 7. 중단·대응 조건

- **M5 `FAIL`(|Δ| > 0.5 pp)**: 해당 교차 루트 비교(P3 brake 대조) 금지, `BLOCKED`로 보고. 2×2 판정은 루트 내부 비교라 영향 없음. 원인(비결정성 또는 계측 추가가 RNG 소비를 바꿈)을 WORKLOG에 기록.
- **M3/M4 위반**: 해당 루트 전체 VOID (A3·A4와 동일 처리). 부분 재개 금지.
- **P2 `NOT_REPLICATED`**: P3 진행은 하되, 초안에서 `GEOMETRY_MATTERS`를 철회하고 두 시드 불일치를 결과로 보고.
- 어느 라벨도 "실패"가 아니다. `INCONCLUSIVE`만 추가 시드(497)를 요구한다.

## 8. 실행 순서·명령·시간

전부 `src/aerial_gym_simulator/`에서, `aerialgym` env, `PYTHONNOUSERSITE=1`.

```bash
# 0) 코드 변경 → 테스트 → 커밋. 이후 소스 동결.
python -m unittest discover -s tests 2>&1 | tail -3
git add -A && git commit -m "A7: riskcap_arc completes the law x geometry factorial"

# P1  (seed 509, 4 arm — 개정 1)
NAVRL_CG_SEED=509 NAVRL_CG_RESULT_ROOT=results/navrl_arc_attribution_seed509 \
NAVRL_CG_ARMS=riskcap,stopcap,dwa_arc,riskcap_arc python tools/run_navrl_contact_geometry.py evaluate

# P2  (seed 491, 4 arm)
NAVRL_CG_SEED=491 NAVRL_CG_RESULT_ROOT=results/navrl_arc_attribution_seed491 \
NAVRL_CG_ARMS=riskcap,stopcap,dwa_arc,riskcap_arc python tools/run_navrl_contact_geometry.py evaluate

# P3  (205 bars, ep25000, seed 49, 4 arm)
bash aerial_gym/rl_training/rl_games/eval_navrl_v2_ep25000_arc_attribution.sh

# 표·판정
python tools/build_a7_arc_attribution_table.py
```

세 파트는 같은 커밋에서 순차 실행한다(런처가 커밋·작업트리 동결을 arm마다 검사). P1이 끝나면 `--check-p1`로 M3·M6·M5를 보고한다.

시간(실측 근거): A4 70 bars 6 arm ≈ 3 h → **약 30 min/arm**(로그 내부 시간은 6~10 min, 나머지는
시뮬레이터 기동·포렌식 계측). 09-02 205 bars 스크린은 로그 내부 4~6 min/cell.
P1 4 arm ≈ 2 h · P2 4 arm ≈ 2 h · P3 4 arm ≈ 0.5~1.5 h → **합계 약 4.5~5.5 h**(12 arm), 순차 1 GPU(RTX 3070).

## 9. 결과와 무관하게 초안에서 고칠 것

1. §1-3 "같은 법칙, 같은 스캔, 기하만 교체" → 대조군을 `stopcap`으로 바꾸고 수치 정정(−1.2 pp, 비유의).
2. §1-1의 stopcap 21.31 %(205 bars·ep25000·brake 2.96)와 표 1의 12.30 %(70 bars·ep1900·brake 2.0)가 다른 실행임을 각주로 명시. P3 결과로 화해.
3. §5 "`dwa_arc`는 보수성으로 이득을 사지 않았다" → "정지법칙의 비용을 걷어냈다"로.
4. §8-1 "튜브 반폭 고정한 채 원호만 토글" → 본 사전등록으로 교체.
5. `tools/build_a5_ablation_table.py` L115~116·L124 문구 동일 수정, 표 5(2×2)는 A7 요약기가 생성.

## 10. 범위 밖

개입률 매칭 arm(§0 사유로 폐기) · 재적응 학습 · 튜브 반폭·brake 스윕 · 개입 시점 yaw rate 분포
로깅(유용하지만 계측 추가 = 소스 변경이므로 이번 실행에 섞지 않는다; A7 종료 후 별도) ·
star-convex 실켜기 A/B.
