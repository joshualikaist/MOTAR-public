# 사전등록 — N1 real-frame reflection audit (eval-only)

작성: 2026-08-21, **관측 수집 및 측정 개시 전**. 이 문서는 결과를 본 뒤 수정하지 않는다.
게이트·seed·통계량·판정 규칙은 아래에서 확정되며, 측정 후 완화·재정의하지 않는다.

관련 문서: `docs/diagnostic_synthesis_2026-08-21.md` (N1 지시), `VERIFICATION.md`,
`WORKLOG.md` 2026-08-02 항목(legacy 계보 chirality 측정), `codex/mode-probe` @ `db0c9a0`.

---

## 1. 가르려는 것

동결 정책이 좌우 반사된 세계에서 반사된 행동을 내는가(equivariance), 아니면 학습된
turn-direction chirality를 갖는가.

이미 아는 것, 그리고 **이 실험이 다시 측정하지 않는 것**:

| 사실 | 계보 | 출처 |
|---|---|---|
| lateral action MAE 1.235, sign mismatch 73.08% (548,736 obs) | **legacy `navrl_quad` ep24000** | WORKLOG 2026-08-02 |
| symmetric fixture reflection max-abs 1.8332, slot-order 0.0078 | ref5in D1 ep1900, **synthetic fixture 1개** | `codex/mode-probe` db0c9a0 |
| 대칭 아레나에서 outcome 비대칭은 검출 안 됨 (capture −0.81 pp, 95% CI −2.78..+1.17) | legacy ep24000 | WORKLOG 2026-08-02 |

따라서 N1이 새로 답하는 것은 정확히 세 가지다.

- (Q-A) **frozen ref5in 계보**에서 chirality가 **실제 프레임**으로 재현되는가 (지금까지 fixture 1개뿐)
- (Q-B) chirality가 **맥락 의존적인가** — target visible/hidden, front blocked/clear, 그리고
  episode outcome(capture/crash/timeout)별로 크기가 달라지는가
- (Q-C) 분포의 **꼬리**가 어떤가 — median뿐 아니라 p90/p95/p99. 평균 하나로는
  "가끔 크게 틀린다"와 "항상 조금 틀린다"를 구분할 수 없다

## 2. 이 방법의 알려진 한계 — 판정 전에 명시

- **(L1) 관측 반사는 세계 반사의 근사다.** 장애물 토큰은 부호만 뒤집고 **순서를 재배열하지 않는다**
  (`ppo_update_safety.py:398-403`). 토큰이 거리 오름차순이라 반사 불변이지만 정확한 동거리 tie에서는
  깨진다. → mode-probe가 이 구멍을 직접 측정해 `slot_permutation_max_abs = 0.0078`로 무시 가능함을
  보였다. 본 실험은 이 값을 재측정하지 않고 인용한다.
- **(L2) 정규화기(running_mean_std)는 좌우 대칭이 아니다.** 플레이어는 `preproc(M o)` 순서로
  적용하며(`navrl_players.py:155`), 이는 물리적으로 옳다(반사된 세계가 만드는 원시 관측이 동일한
  고정 정규화기를 통과). 그러나 측정된 chirality의 일부는 **네트워크가 아니라 정규화기 통계의
  비대칭**에서 올 수 있다. → §6의 보조 측정 S1로 분해하되, **판정에는 쓰지 않는다**.
- **(L3) 이 실험은 outcome을 측정하지 않는다.** chirality가 있어도 대칭 아레나에서는 성능 손실이
  없을 수 있다(2026-08-02에 실제로 그랬다). 따라서 어떤 결과도 "chirality가 성능을 해친다"는
  주장의 근거가 될 수 없다.
- **(L4) 단일 checkpoint·단일 조건.** 70막대 1셀, seed 1개. 계보 전반이나 밀도 전반으로
  일반화하지 않는다.

## 3. 실험 계약

| 항목 | 값 |
|---|---|
| 정책 | ref5in D1 ep1900, `runs/ppo_260813_1636_navrl_v2-ref5in-d1-q3-adapt-s197/nn/last_gen_ppo_ep_1900_rew_182.11377.pth` |
| checkpoint SHA-256 | `197ea26999d6bb9cf23c4e5a55acbe945f89985e2384687d60ab1dbae66a278e` |
| robot | `navrl_ref5in_quad` (checkpoint metadata에서 evaluator가 도출) |
| 아레나 | 70 bars (mode-probe host cell과 동일 조건) |
| 평가 seed | **373** — 전수 검색 사용 이력 **0건** |
| 요청 에피소드 | 512 (128-env 배치 4 round) |
| action 선택 | deterministic |
| speed governor | `off` (P2 canonical, ref5in 계보 기본값) |
| reflection mode (rollout) | `original` — rollout 중에는 반사 forward를 **하지 않는다** |
| 관측 샘플링 | 결정론적 stride. `NAVRL_OBS_DUMP_STRIDE=37`, 전 env 동시. RNG 미사용 |
| 예상 프레임 수 | `128 × floor(2400/37) = 8,192` |
| **최소 프레임 수 (게이트)** | **4,096** |
| worktree / branch | `.codex_worktrees/navrl_reflection_audit` / `codex/reflection-audit` |
| 영수증 스키마 | eval schema_version 2, 공유 immutable source bundle, runtime-clean manifest |

**환경 변수는 `P2.canonical_env`의 닫힌 집합을 그대로 쓰고**, 아래만 추가·변경한다.
`NAVRL_SEED=373`, `NAVRL_V2_DENSITIES=70`, `NAVRL_V2_RESULT_DIR`, `NAVRL_V2_SHARED_SOURCE_BUNDLE`,
`NAVRL_OBS_DUMP*`, `NAVRL_REQUIRE_SOURCE_ROOT`, 그리고 `PYTHONPATH=<worktree>` 재주입.
target pattern·CV heading 개입은 **하지 않는다** → generic evaluator의 provenance override가
불필요하며, 사용하지 않는다.

## 4. 절차 — rollout과 반사를 시간적으로 분리한다

1. **rollout**: 표준 평가 1셀. 정책 행동에 어떤 개입도 없다. 관측 벡터와 맥락 라벨,
   에피소드별 outcome만 디스크에 덤프한다(`NAVRL_OBS_DUMP`).
2. **offline**: rollout 종료 후, 저장된 npz를 읽어 동결 checkpoint를 다시 적재하고
   원본과 반사본을 각각 forward한다. **시뮬레이터를 실행하지 않으며, 어떤 probe action도
   시뮬레이터에 투입되지 않는다.** 이 분리 자체가 "side-forward only" 조건의 증명이다.
3. 저장된 npz의 sha256을 영수증과 요약에 고정한다. 제3자가 동일 npz로 전 수치를 재계산할 수 있다.

맥락 라벨은 **새로 정의하지 않고** `navrl_task.py:3281 _record_action_diagnostics`가 관측 소비
시점에 이미 계산하는 마스크를 그대로 기록한다: `front_blocked`/`front_clear`(depth scan, |bearing| ≤ 30°,
4.0 m 임계), `_visible_now`(target visible/hidden), `valid_y_now`. outcome은 기존 종료 귀속
마스크(`d_contact`/`d_oob`/capture/timeout)에 바인딩하며 새 귀속 논리를 만들지 않는다.

## 5. 품질 게이트 — 판정보다 먼저, 실패 시 fail-closed

하나라도 실패하면 판정은 **`FAIL_CLOSED_TRANSFORM_QUALITY`**이며, 정책에 대한 어떤 주장도 하지 않는다.

| # | 게이트 | 임계 |
|---|---|---|
| **Q1** | involution — 수집된 전 프레임에서 `max abs(M(M(x)) − x)` | `== 0.0` (부호 있는 순열이므로 float32에서 정확해야 함) |
| **Q2** | isometry — `max abs(‖M(x)‖₂ − ‖x‖₂)` | `≤ 1e-3` |
| **Q3** | schema 고정 — mirror가 쓰는 `HBEAMS/VBEAMS/MAX_OBSTACLES`를 **checkpoint metadata에서** 읽고, 산출된 `structured_obs_dim`이 수집된 obs 폭과 일치 | `== 898` |
| **Q4** | index-set 정확성 (byte-level, 합성 난수 텐서) — 부호 반전 인덱스 집합과 순열 인덱스 집합이 사전등록 집합과 **정확히 일치**, 그 외 인덱스는 불변 | 완전 일치 |
| **Q5** | scan 순열 — `i → (−i) mod 72`, 고정점이 정확히 `{0, 36}` | 완전 일치 |
| **Q6** | import origin — `aerial_gym` 해석 경로가 worktree ROOT 하위이고 그 sha256이 source manifest에 존재 | 강제(fail-closed) |
| **Q7** | checkpoint SHA 일치, runtime-clean manifest, schema_version 2 | 완전 일치 |
| **Q8** | 결정론 — offline forward를 2회 실행해 행동 텐서가 **bitwise 동일** | `torch.equal` |
| **Q9** | 표본 수 | 유효 프레임 `≥ 4,096` |

Q4의 사전등록 인덱스 집합(898-D, HBEAMS=72/VBEAMS=4/MAX_OBSTACLES=8):

- 순열: `[0:288]`, ring `v∈{0,1,2,3}`마다 `288`-블록 내 `v*72 + h`, `h → (−h) mod 72`
- 부호 반전 (obstacle `[288:768]`, `(hist=5, slot=8, dim=12)`): field 1, 4
- 부호 반전 (robot `[768:818]`, `(hist=5, dim=10)`): field 1, 3, 5, 7
- 부호 반전 (target `[818:898]`, `(hist=5, dim=16)`): field 1, 4
- 그 외 전 인덱스: 불변

## 6. 측정량

행동은 rescale·clamp **이전**의 정규화된 정책 출력(`deterministic_actions`, 사실상 `[-1,1]`)에서
읽는다 — 기존 `navrl_players._record_reflection_pair`와 동일 단위이므로 1.235/73.08%/1.8332와
직접 비교된다.

`e = π(preproc(M o)) − M·π(preproc(o))`, lateral 채널은 index 1.

| 이름 | 정의 |
|---|---|
| **conj_err_lat** (primary) | `abs(e[1]) = abs(π(Mo)[1] + π(o)[1])` |
| conj_err_yaw | `abs(e[3])` |
| conj_err_x, conj_err_z | `abs(e[0])`, `abs(e[2])` |
| **lateral sign agreement** | `sign(π(Mo)[1]) == −sign(π(o)[1])` 인 비율. **비교 가능 행만**: 양쪽 모두 `abs(·) ≥ 0.05` (기존 코드와 동일 임계) |
| **signed lateral bias** | `b = (mean π(o)[1] + mean π(Mo)[1]) / 2`. 완전 equivariance면 `b = 0`; 고정된 body-side 선호면 `b ≠ 0` |

각 측정량에 대해 **median, p90, p95, p99, mean, max, n**을 보고한다.

**분할(context)** — 전체 + 아래 각각, 그리고 교차하지 않는다(다중 비교를 늘리지 않기 위해 1차 분할만):

1. target visible / hidden
2. front blocked / front clear / unknown
3. outcome: capture / crash_bar / crash_oob / crash_other / timeout

**컨텍스트 셀 최소 표본 = 256 비교가능 행.** 미달 셀은 수치를 보고하되 **판정을 부여하지 않고**
`insufficient_sample: true`로 표시한다.

**보조 측정 S1 (비게이트, exploratory)** — 정규화기 비대칭 분해. 반사 짝 인덱스에 대해
running_mean_std의 평균·분산을 대칭화한 정규화기로 동일 계산을 반복한다. `e_sym ≈ e_raw`이면
chirality는 네트워크에 있고, `e_sym ≪ e_raw`이면 정규화기 통계가 chirality를 운반한다.
**이 값은 판정에 쓰지 않으며**, reflection intervention 설계 시 참고 자료로만 남긴다.

## 7. 판정 규칙 — 결과를 보기 전에 확정

품질 게이트 Q1–Q9가 **전부 PASS**한 경우에만 아래를 적용한다.

전체(all frames, 비교가능 행) 기준:

| 판정 | 조건 (AND) |
|---|---|
| **CHIRALITY_CONFIRMED_REAL_FRAME** | `median(conj_err_lat) ≥ 0.30` **그리고** `lateral sign agreement ≤ 0.60` |
| **CHIRALITY_ABSENT** | `median(conj_err_lat) ≤ 0.10` **그리고** `lateral sign agreement ≥ 0.90` |
| **INCONCLUSIVE_REAL_FRAME** | 그 외 전부 |

임계 근거(사전 확정): 행동 범위는 `[-1,1]`, 폭 2.0이다. `0.10`은 전 범위의 5%이며 코드가 이미
"의미 있는 부호"의 하한으로 쓰는 `0.05`와 같은 자릿수다. `0.30`은 그 3배로, 두 구역이 겹치거나
동시에 만족될 수 없도록 분리했다. sign agreement `0.60`은 우연 수준 `0.50`보다 위, `0.90`은
완전 equivariance `1.00` 바로 아래로 잡았다.

**context별 판정은 동일 임계를 쓰되, 표본 ≥ 256인 셀에만 부여한다.**

## 8. 이 실험이 부여하는 권한과 부여하지 않는 권한

`CHIRALITY_CONFIRMED_REAL_FRAME`이 나온 경우에만 **다음 단계로 reflection augmentation /
consistency loss의 사전등록을 작성할 자격**이 생긴다. 그 경우에도:

- 이 실험은 그 개입을 **구현하거나 실행할 권한을 주지 않는다**. 다음 사전등록이 필요하다.
- outcome 개선을 주장하지 않는다 (L3).
- 2026-07-27 Ablation B(`NAVRL_REFLECTION_COEF=0.01`)는 **잘못된 mirror 연산자** 위에서
  돌았기 때문에 기각되었다(WORKLOG 2026-07-29). 그 기각은 새 연산자에 대한 증거가 아니며,
  동시에 새 실험의 근거로 인용할 수도 없다.

`INCONCLUSIVE` 또는 `CHIRALITY_ABSENT`이면 reflection intervention을 **설계하지 않는다**.

## 9. 하지 않을 것

- 관측·보상·종료·환경·riskcap·정책 행동 변경
- probe action의 시뮬레이터 실행
- multi-candidate action head 또는 reflection loss 구현
- riskcap 파라미터 탐색, 속도 상한 변경
- P2 STRICT FAIL / D1 FAIL / P3 BLOCKED 판정의 소급 변경 — 이 실험은 셋 중 무엇도 건드리지 않는다
- 기존 결과(geofence, mode probe, joint telemetry, topology)의 공식 판정 소급 변경
- 결과를 본 뒤 임계·seed·통계량·컨텍스트 분할 변경
- dirty primary worktree 또는 physical-target WIP에 merge

## 10. 기록 요건

- `results/navrl_ref5in_reflection_audit_seed373/` — cells, source_bundle, receipt, summary.{json,md}
- 프레임 npz는 `results/` 하위에 두되 **git에 커밋하지 않고**, sha256을 요약에 고정한다
- 무효·실패 실행도 `WORKLOG.md`에 기록한다 (VOID 사유 포함)
- 요약에 `p2_verdict_changed: false`, `d1_verdict_changed: false`, `p3_unlocked: false`,
  `decision_authority: "none"`를 명시한다

---

## 3-b. 샘플링 계획 수정 (2026-08-21, **관측 수집 개시 전** 기록)

§3의 고정 stride 37은 결함이 있다. 프레임 수 `128 × floor(총_호출/37)`는 **총 호출 수를 미리
알아야** 하는데, 그 값은 에피소드 조기 종료율에 따라 달라진다. 조기 종료가 많으면 총 호출이
줄어 프레임이 게이트 최소치 4,096 아래로 떨어질 수 있고, 반대로 많으면 상한에서 잘려
**rollout 앞부분에 치우친 표본**이 된다. 둘 다 사전등록 위반이다.

측정을 시작하기 전에 다음으로 대체한다.

- `NAVRL_OBS_DUMP_STRIDE = 1`, `NAVRL_OBS_DUMP_MAX = 16384` (행 기준), 요청 에피소드 **1,024**
- 수집은 **스트리밍 데시메이션**으로 한다: 매 `stride_eff`번째 호출을 보관하고, 보관 행 수가
  MAX를 넘으려 하면 보관된 호출을 하나 걸러 하나씩 버린 뒤 `stride_eff`를 2배로 한다.
- 결과는 정확히 "rollout 시작부터 매 `stride_eff`번째 호출"이 되어 **전 구간에 균일**하며,
  최종 행 수는 `MAX/2` 이상 `MAX` 이하, 즉 **8,192–16,384행**으로 총 호출 수와 무관하게
  게이트 최소치를 항상 넘는다. RNG는 쓰지 않는다.
- 최종 `stride_eff`와 데시메이션 횟수를 npz와 요약에 기록한다.

§3의 나머지(정책·SHA·아레나·seed 373·governor·deterministic·reflection_mode)와 §5–§9의
게이트·임계·판정 규칙은 **변경하지 않는다**.

참고: 기존 seed 367 셀의 `condition.episode_len_steps = 600`, `condition.num_envs = 128`이다.

## 3-c. import origin 실증 (2026-08-21, **관측 수집 개시 전** 기록)

§5 Q6의 근거가 되는 실패를 실측했다. `find_spec("aerial_gym").origin`:

| cwd | PYTHONPATH | 해석 결과 |
|---|---|---|
| `<worktree>` | 없음 | `<worktree>/aerial_gym/__init__.py` |
| `<worktree>` | `<worktree>` | `<worktree>/aerial_gym/__init__.py` |
| **`<worktree>/aerial_gym/rl_training/rl_games`** | **없음** | **`src/aerial_gym_simulator/aerial_gym/__init__.py`** |
| `<worktree>/aerial_gym/rl_training/rl_games` | `<worktree>` | `<worktree>/aerial_gym/__init__.py` |

세 번째 행이 `play_navrl.sh:19`가 `cd`하는 디렉터리이므로, worktree에서 시작한 평가는 기본적으로
PRIMARY 소스를 실행한다. 따라서 본 실험은 `PYTHONPATH=<worktree>`를 재주입하고(§3), 그것과
독립적으로 런타임에서 `aerial_gym`의 실제 출처를 fail-closed로 검증한다(Q6).

이 사실은 기존 결과의 공식 판정을 **소급 변경하지 않는다**. 실행된 파일 바이트가 두 트리에서
동일했다면 수치는 유효하며, 동일성 여부는 별도 확인 작업이다. 본 실험은 그 확인을 수행하지 않는다.

## 3-d. 목표거리 고정 (2026-08-21, **관측 수집 개시 전** 기록)

§3이 목표거리를 고정하지 않은 것은 누락이다. `preflight`에서 드러났다:
`cfg_general_goal_dist_min: checkpoint=22.5 expected=6.0`. frozen ref5in D1 ep1900은 22.5–28 m로
학습됐고, generic evaluator의 계약 검사는 이 값이 어긋나면 거부한다.

`NAVRL_V2_GOAL_DIST_MIN = 22.5`, `NAVRL_V2_GOAL_DIST_MAX = 28`로 고정한다.

이는 자유 선택이 아니라 체크포인트 계약이 강제하는 유일한 값이며, 같은 체크포인트·같은 70막대
host cell을 쓴 `codex/mode-probe`의 `run_navrl_ref5in_mode_probe.py:105-106`과 동일하다.
§5–§9의 게이트·임계·판정 규칙은 변경하지 않는다.
