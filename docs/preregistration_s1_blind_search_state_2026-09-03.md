# Preregistration + 구현 계획 — S1 explicit blind-search state

Status: **IMPLEMENTED FOR REVIEW — 코드·CPU 검증만 완료, GPU 실행 권한은 열지 않음.**
Prepared: 2026-09-03
Source brief: `docs/plans/target_search_and_adversarial_evader.md` (8절 체크리스트)
Target path: `docs/preregistration_s1_blind_search_state_2026-09-03.md`

---

## 0. 체크리스트 이행 요약 (8절 1–10)

| # | 항목 | 상태 |
|---|---|---|
| 1 | brief + `navrl_perception.py` / `navrl_task_config.py` / `navrl_task.py` 읽음 | 완료 (§1) |
| 2 | seed-313 held-out `70bars.json` / `145bars.json` 읽고 claim boundary 유지 | 완료 (§1) |
| 3 | 첫 실험축 하나 선택 | **S1 explicit blind-search state** (§2) |
| 4 | 사전등록: treatment/control, seed, checkpoint 정책, metric, gate, abort | §3–§6 |
| 5 | 새 actor-visible / critic-only 필드 전부 열거 | §4 |
| 6 | evader 계약 (onboard vs privileged) | **해당 없음** — S1은 Tier-0 scripted target 고정. 계약 A/B는 E1 이후 (§2.3) |
| 7 | 단위 테스트·schema fail-closed | §7 |
| 8 | PPO 캠페인 전 deterministic smoke | §8 |
| 9 | raw artifact 불변, receipt/source manifest | §9 |
| 10 | README/status 갱신은 held-out gate 완료 후 | §9 |

---

## 1. 동결 진단 (읽은 것만 기록)

### 1.1 held-out seed 313, checkpoint `last_gen_ppo_ep_21750` (SHA `541b36bd…`)

| bars | n | capture | crash | never-acq (all) | never-acq → crash | never-acq → timeout | never-acq → OOB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 70 | 2049 | 83.70% | 15.91% | 175 (8.54%) | 169 | 6 | 5 (OOB는 crash 집계 내) |
| 145 | 2049 | 65.54% | 34.16% | 406 (19.81%) | 400 | 6 | 0 |

- crash 중 never-acquired 비율: 70 bars **169/326 = 51.8%**, 145 bars **400/700 = 57.1%**.
- captured episode의 first-visible step median: 23 (70) / 31 (145). crash episode 중 acquired인 것은 median 25 / 40.
- target_hidden 상태에서의 action fraction 70.1% / 78.8%; 그 구간 평균 실제 속도 2.84 / 2.62 m/s.
- 2026-08-21 geofence A/B (seed 367, 1-bar, away-CV)는 OOB 채널을 봤다. **현재 route-off held-out에서는 never-acquired 실패의 96.6%(70) / 98.5%(145)가 bar contact이지 OOB가 아니다.** 따라서 그 A/B의 "OOB 감소" 가설은 이 checkpoint로 이전되지 않으며, 본 실험의 primary는 OOB가 아니라 **never-acquired 전체**다.

### 1.2 관측 계약 (코드 확인)

- actor 898-D = static 288 + obstacle history 5×8×… + robot history 5×10 + target history 5×16. critic 906-D (= 898 + GT 8).
- `_target_features`: tracker inactive면 16-D target token 전체에 0을 곱한다 (`features * active`). 즉 first acquisition 전과 track 만료 후 actor는 표적 방향 belief를 전혀 받지 않는다.
- `GEOFENCE_ACTOR` 토큰(8-D: F/L/B/R 정규화 거리 + validity 4)은 이미 구현·테스트돼 있고 schema 끝에 append된다. `GEOFENCE_FORCE_INVALID`는 평가 전용 masked ablation. `NUM_TOKENS = 17 + corridor + geofence`.
- Transformer `is_rnn() == False`; history stride 0.5 s × 5 = 약 2 s 창.
- 카메라 hFOV 87°, vFOV 58°, `target_camera_max_range_m = 20`, LiDAR 12 m, arena 40×40 m, 600 step, 128 envs, governor off, route off, `U[0.3,1.25]`.

### 1.3 이 사전등록의 claim boundary

- 비교 대상은 **같은 예산의 fresh arm끼리**다. 22k epoch 학습된 seed-911 checkpoint와 3k epoch fresh arm의 capture를 직접 비교하지 않는다.
- 70 bars 학습 → 70 bars held-out이 primary. 145는 fresh arm에게 OOD이므로 secondary/탐색적.
- 205/routed/hardware/sim-to-real/adversarial evader 주장 없음.

---

## 2. 실험축 선택과 근거

### 2.1 선택: S1 explicit blind-search state

근거는 brief와 동일하되 한 가지를 추가한다: never-acquired 실패가 OOB가 아니라 **blind 상태 bar contact**로 나타나므로, "표적 방향/미탐색 영역 belief가 있으면 blind 구간 비행이 목적을 가진 탐색이 되어 무의미한 고속 blind 비행과 그에 따른 충돌이 줄어드는가"가 검증 가능한 인과 질문이 된다. 이 질문은 scripted target으로 충분히 검증되며 evader 축을 섞을 필요가 없다.

### 2.2 `fixed-density PPO 연장 금지` 재검 조건 충족 여부

CLAUDE.md의 금지 ①은 "실패를 epoch 추가로 덮는 것"을 막기 위한 것이고 재검 조건은 *"밀도가 아닌 축에서 병목이 특정되고 그 처방이 사전등록됐을 때"*다. 본 실험은 (a) 기존 run의 연장이 아니라 schema가 다른 fresh arm이고, (b) 병목이 밀도가 아닌 acquisition 축으로 특정됐으며, (c) 처방이 이 문서로 사전등록된다. 따라서 조건을 충족한다고 판단하나, **최종 승인은 사용자 몫**이다.

### 2.3 하지 않는 것

- 표적 행동 변경 없음 (E1은 S1 판정 후 별도 사전등록).
- camera range·speed envelope·reward·governor·obstacle token·density schedule·max velocity/tilt 변경 없음.
- recurrent policy 없음 (schema는 바뀌지만 architecture는 4-layer Transformer 유지; 새 토큰 1개 append).

---

## 3. Arm 설계 (nested, 한 번에 하나씩 쌓임)

새 env var `NAVRL_SEARCH_STATE ∈ {off, geofence, coverage, belief}`. `geofence` 이상은 `NAVRL_GEOFENCE_ACTOR=1`을 자동으로 함의한다 (둘이 불일치하면 fail-closed).

| arm | `NAVRL_SEARCH_STATE` | actor 추가 차원 | actor dim | 새 토큰 |
|---|---|---:|---:|---|
| A0 control | `off` | 0 | 898 | 0 (historical 17-token, byte-identical) |
| A1 geofence | `geofence` | +8 | 906 | geofence 1 (기존 코드 그대로) |
| A2 coverage | `coverage` | +8 +4 +24 | 934 | geofence 1 + search 1 |
| A3 belief | `belief` | +8 +4 +24 +25 | 959 | geofence 1 + search 1 (belief는 같은 search 토큰에 concat) |

세 가지 추가 feature 정의 (모두 body-frame, 12개 30° 방위 sector, GT 미사용 — §4 참조):

**mode (4-D, A2·A3)**
`[never_acquired, tracked, stale]` one-hot(tracker inactive & 최초 acquisition 이력 없음 / active / inactive & 이력 있음) + `blind_time / 60 s` clamp(0,1). 이 4-D가 정확히 `SEARCH / TRACK / REACQUIRE` 상태다.

**coverage (24-D, A2·A3)**
40×40 m arena를 2 m 셀 20×20 world-frame 격자로 두고, 매 RL step 카메라 frustum(hFOV 87°, 20 m) 안이면서 depth 이미지 기준 가려지지 않은 셀을 `viewed=1`로 마킹한다 (occlusion 판정: 해당 셀 방향 depth 픽셀 최소값 > 셀 거리이면 unoccluded — 센서 유도, GT 아님). 12 sector마다 `[해당 sector 내 unviewed 셀 질량 / 전체 셀 수, 가장 가까운 unviewed 셀 거리 / arena 대각선]`. 이는 "아직 안 본 곳이 어느 방향에 얼마나 있는가"이며 frontier의 최소 형태다. world 격자는 geofence와 같은 VIO/GPS pose 가정을 공유한다 (A1이 이미 그 가정을 깐다).

**belief (25-D, A3)**
같은 격자 위 표적 위치 확률 `b`. 초기 uniform; 매 step (i) viewed 셀 확률 ×(1−p_det) 후 정규화 (p_det = detector 조건 상수 0.9, 선언값), (ii) 표적 속도 prior 1.25 m/s에 맞춘 등방 diffusion(σ = 1.25 m/s × 0.1 s), (iii) tracker active면 KF posterior Gaussian으로 belief를 덮어쓰고, active→inactive 전이 시 마지막 KF state/cov를 Gaussian으로 재주입. 출력: 12 sector belief 질량 + 정규화 entropy 1. coverage와 belief를 둘 다 넣는 이유는 A2⊂A3 nesting을 유지하기 위해서다.

**A1과 A2 사이에 mode+coverage 두 가지가 동시에 들어가는 점**은 알고 시작한다. A1은 2026-08-21 토큰을 문자 그대로 재사용해 과거 결과와 대조 가능하게 두고, 4-arm 예산에서 mode-only arm은 넣지 않는다. mechanism 귀속은 §5.4 masked ablation으로 한다.

---

## 4. 필드 열거 (actor-visible / critic-only / 금지)

| 필드 | actor | critic | 출처 | GT? |
|---|:-:|:-:|---|:-:|
| geofence 8 | ✓ | ✓ | drone pose + 선언된 arena bounds | 아니오 (VIO/GPS 가정) |
| mode 4 | ✓ | ✓ | tracker `active`/`age` + episode 내 acquisition 이력 flag | 아니오 |
| coverage 24 | ✓ | ✓ | drone pose, 카메라 frustum 상수, depth 이미지 | 아니오 |
| belief 25 | ✓ | ✓ | coverage 격자, KF state/cov, 선언 상수(p_det, v_prior) | 아니오 |
| 기존 GT 8 | ✗ | ✓ | 변경 없음 | 예 (critic-only, 기존과 동일) |

**actor에 절대 들어가지 않는 것**: target GT position/bearing/visibility, semantic id/mask, 표적 pattern/speed 샘플값. belief 갱신 함수는 표적 상태 텐서를 **인자로 받지 않는다** (§7 leak test).

평가/진단 전용 (env_state / JSON에만): `blind_steps_before_first_visible`, `blind_phase_mean_speed_mps`, `blind_phase_bar_clearance_mean_m`, `coverage_fraction_at_first_visible`, `belief_entropy_at_first_visible`, `search_state_masked` flag.

---

## 5. 동결 실행 tuple

### 5.1 학습 (4 arm 동일)

| item | value |
|---|---|
| launcher | 신규 `train_navrl_s1_search_state.sh` (fresh-only, checkpoint 거부, dirty runtime 거부, `NAVRL_SEARCH_STATE`만 arm별로 다름) |
| seed | **919** (911/907/313/367 금지) — 4 arm 모두 같은 seed |
| epochs | **3000** fixed (25 epoch × 128 env × 6 RL step ~ 이전 run 기준 ~900 epoch/h 추정 → arm당 3–4 h; 추정치이며 §8 smoke에서 실측) |
| density | **70 bars 고정**, `NAVRL_DENSITY_CURRICULUM=0` |
| robot/target | `navrl_ref5in_v2_quad`; `physx_ref5in_6dof_motor_wrench_v2_same_substep`; route off; mixed CV/waypoint; `U[0.3,1.25]`, ramp 1 epoch |
| arena/placement | 40×40×3; `footprint_clearance`; surface 0.45 m; fallback/merge 0 |
| perception/governor | canonical v2 cluster-sector; governor off; corridor 0; latency/noise 기존 기본값 |
| lr / save | 1.5e-5 / 250 |
| geofence noise/dropout | 0 / 0 (첫 인과 A/B; robustness는 이후 별도) |
| 실행 순서 | A0 → A1 → A2 → A3, **직렬**, 3070 단독 (1650 Ti 학습 결과 혼용 금지) |
| 평가 checkpoint | 각 arm `last_gen_ppo_ep_3000_*.pth` (`gen_ppo.pth` 금지) |

### 5.2 held-out 평가

| item | value |
|---|---|
| seed | **331** (학습 919와 독립) |
| cell | 70 bars (primary), 145 bars (secondary, OOD 명시) |
| episodes | 2049 완료 episode / cell / arm, 128 env, deterministic |
| contract flag | `NAVRL_V2_EVAL_CONTRACT=corrected_nonoverlap_physical_off` + `NAVRL_SEARCH_STATE=<arm>` 기록 |
| masked ablation | 통과 arm에 한해 같은 seed 331, `NAVRL_SEARCH_STATE_FORCE_INVALID=1` (geofence range=1/valid=0, mode=0, coverage=0, belief=uniform 12×1/12 + entropy 1) |

### 5.3 Primary / Guard / Secondary

**Primary (70 bars):** never-acquired rate = (never-acq crash + never-acq timeout + never-acq OOB) / 2049.
PASS 조건 (arm k vs A0): 절대 감소 **≥ 2.0 pp** 이고 two-proportion 95% CI가 0을 제외. (n=2049, base ≈ 8.5% 가정 시 차이 SE ≈ 0.87 pp; 2.0 pp ≈ 2.3σ.)

**Guard (모두 충족해야 PASS):**
1. acquired episode 내 crash 비율 `crash_given_acquired`가 A0 대비 **+2.0 pp 초과 악화 금지** (탐색 개선이 후속 항법을 해치지 않음).
2. timeout rate A0 + 1.0 pp 이하.
3. first-visible step median (captured)이 A0보다 **증가하지 않음** — "조심스럽게 느리게 날아서" 얻은 감소를 탐색 개선으로 읽지 않기 위한 방향성 guard.
4. NaN/KL 안전 가드 미발동, episode 회계 누락/중복 0, receipt·source manifest 검증 PASS.

**Secondary (보고만, gate 아님):** capture/crash/timeout/OOB Wilson CI, first-acquisition time 분포(p10/p50/p90), visible fraction, visible↔hidden 전이 수, blind-phase 평균 속도·clearance, coverage/entropy at first visible, 145-bar 동일 표.

### 5.4 Mechanism 판정

Primary PASS인 arm에 masked ablation을 적용해 primary 이득의 **≥ 50%**를 잃으면 `PASS_MECHANISM`. 그렇지 않으면 outcome이 좋아도 `PASS_MECHANISM_UNRESOLVED` (2026-08-21과 같은 라벨). nested 구조에서 A3가 A2 대비 추가로 ≥1.0 pp 개선하지 못하면 "belief 추가는 이 예산에서 기여 없음"으로 기록하고 coverage 수준을 채택 후보로 둔다.

### 5.5 예상 결과 사전 기록

- H1: A1 ≥ A0 (OOB 채널이 작으므로 소폭). H2: A2 > A1 (미탐색 방향 정보가 blind 비행에 목적을 줌). H3: A3 ≥ A2 (stale-track 재주입은 145에서 더 유효할 것 — 145는 secondary).
- 모든 arm이 primary FAIL이면: "3000 epoch feed-forward 창에서 search state 토큰은 never-acquired를 줄이지 못함"이 결과이며, 다음 escalation(recurrent belief, 예산 증가)은 **새 사전등록**으로만 연다. FAIL을 epoch 추가로 덮지 않는다 (금지 ①의 취지 유지).

### 5.6 Abort / VOID

- `VOID_EXECUTION`: seed, epochs, density, speed cap, placement, route mode, checkpoint 규칙, arm별 env var, import origin 중 하나라도 이 문서와 다르면.
- ABORT (run 즉시 종료, 사유 WORKLOG): schema drift RuntimeError, NaN/KL guard, GPU OOM, 실측 throughput이 smoke 실측의 50% 미만, 다른 학습 프로세스 동시 존재.
- 도중에 arm 순서·hyperparameter를 바꾸면 전체 캠페인 VOID.

---

## 6. 구현 계획 (파일 단위, 학습 전 전부 완료)

### 6.1 perception — `aerial_gym/task/navrl_task/navrl_perception.py`
1. `SEARCH_STATE = os.environ.get("NAVRL_SEARCH_STATE","off")` 파싱, 허용값 4개 외 ValueError. `geofence|coverage|belief`이면 `GEOFENCE_ACTOR`가 1이어야 함 (아니면 ValueError — fail-closed).
2. `SEARCH_MODE_DIM=4`, `SEARCH_COVERAGE_DIM=24`, `SEARCH_BELIEF_DIM=25`, `SEARCH_DIM`을 arm별로 계산해 `STRUCTURED_OBS_DIM`에 더함 (GEOFENCE 뒤, 즉 schema 맨 끝).
3. 신규 모듈 `navrl_search_state.py`:
   - `SearchGrid(num_envs, arena_bounds, cell_m=2.0, device)` — `viewed` (bool 20×20), `belief` (float 20×20), `acquired_ever` (bool), `blind_steps` (int).
   - `update(drone_pos_w, quat, depth, tracker_state, tracker_cov, tracker_active)` — GT 텐서 인자 없음.
   - `features(drone_pos_w, quat) -> [N, SEARCH_DIM]`.
   - `reset(env_ids)`.
   - 모두 batched torch, 20×20×128은 무시할 만한 비용.
4. `_target_features` 뒤에서 `self.search.update(...)` 호출, `obs_parts.append(search_features)`. `NAVRL_SEARCH_STATE_FORCE_INVALID`는 평가 전용 마스크 (기존 `GEOFENCE_FORCE_INVALID`와 같은 패턴).
5. diagnostics dict에 §4의 진단 필드 추가.

### 6.2 network — `navrl_transformer_network.py`
- `NUM_TOKENS += 1 if SEARCH_DIM > 0`. `search_proj = Linear(SEARCH_DIM,128)-ELU-Linear(128,EMBED_DIM)`. 파싱 offset에 search 슬라이스 추가, `offset != STRUCTURED_OBS_DIM`이면 기존처럼 RuntimeError.

### 6.3 task/provenance — `navrl_task.py`
- `representation`/`env_state`에 `cfg_search_state`, `cfg_search_state_force_invalid` 기록 (기존 `cfg_geofence_*` 옆).
- reset 시 `search.reset(env_ids)`.
- 평가 telemetry에 §4 진단 필드 집계 (never-acquired 분해는 기존 `first_acquisition` 구조 재사용).

### 6.4 launcher
- `train_navrl_s1_search_state.sh`: `train_navrl_corrected_nonoverlap_physical_smoke.sh` 복제 후 `MAX_EPOCHS=3000`, `SEED=919`, `NAVRL_SEARCH_STATE` 필수 인자 1개만 허용(`off|geofence|coverage|belief`), `AERIAL_RUN_TAG=s1-search-<arm>-s919`, 나머지 export 동일. 기존 CLI-거부·checkpoint-거부·dirty-거부 로직 유지.
- `eval_navrl_s1_search_state_heldout.sh`: `eval_navrl_corrected_nonoverlap_physical_off_heldout.sh` 복제, seed 331, densities `70 145`, arm env var 및 force-invalid 옵션, 결과 루트 `results/navrl_s1_search_state_seed331/<arm>[_masked]/`.

### 6.5 summary/analysis
- `tools/summarize_s1_search_state.py`: 4 arm × 2 cell(+masked) JSON → primary/guard 표, two-proportion CI, PASS/FAIL 라벨, `summary.{md,json}` + SHA.

---

## 7. 테스트 (학습 전 전부 PASS 필요, CPU)

| test | 내용 |
|---|---|
| `test_navrl_search_state_schema.py` | 4 arm 각각 `STRUCTURED_OBS_DIM` = 898/906/934/959, `NUM_TOKENS` = 17/18/19/19; `coverage`+`GEOFENCE_ACTOR=0` → ValueError; 잘못된 값 → ValueError |
| `test_navrl_search_state_leak.py` | `SearchGrid.update` 시그니처에 target 인자 없음(inspect); 표적 위치를 임의로 바꿔도 detection 입력이 같으면 features bit-identical |
| `test_navrl_search_state_grid.py` | coverage: frustum 안·unoccluded 셀만 viewed; depth로 가린 셀은 미마킹; sector 질량 합 = 전체 unviewed 비율. belief: 질량 보존(정규화 후 합 1), 관측 셀 확률 감소, diffusion 대칭, active→inactive 재주입 위치 = 마지막 KF state, entropy ∈ [0,1] |
| `test_navrl_search_state_mode.py` | never_acquired→tracked→stale 전이 순서, reset 후 never_acquired 복귀, blind_time 증가/클램프 |
| `test_navrl_search_state_mask.py` | FORCE_INVALID 시 geofence=1/0, mode=0, coverage=0, belief=uniform, entropy=1 |
| `test_navrl_s1_launcher_contract.py` | 인자 1개 필수, checkpoint/CKPT 거부, seed 919, epochs 3000, density 70 fixed, 4 arm의 export 값 diff가 `NAVRL_SEARCH_STATE`·RUN_TAG·log 경로뿐임 |
| `test_navrl_s1_evaluator_contract.py` | seed 331, densities {70,145}, `cfg_search_state` 기록, masked flag 기록 |
| 기존 `test_navrl_perception.py`, `test_navrl_obs_dump.py`, `test_navrl_training_semantics.py` | `off`에서 회귀 없음 (898-D byte-identical) |

전체 suite 재실행 후 새 실패 0건이어야 한다 (기존 4 historical-artifact failure는 예외로 기록).

---

## 8. Smoke (GPU, PPO 캠페인 전)

1. `off`와 `belief` 두 arm만 **100 epoch** smoke (`MAX_EPOCHS=100`, seed 990, 다른 값 동일). 목적: schema 통과, throughput 실측(epoch/h → §5.1 예산 확정), obs dump에서 search 토큰이 0/uniform이 아닌 값을 실제로 갖는지, GT-leak 회귀 없음.
2. smoke 결과는 성능 판단에 쓰지 않는다. WORKLOG에 throughput과 dump 검사만 기록.
3. smoke PASS 후에만 §5.1 캠페인. 실측 throughput이 예산 대비 arm당 6 h를 넘으면 epochs를 줄이지 말고 **사용자에게 보고 후 재승인**.

---

## 9. Artifact / 공개 규칙

- 학습·평가 raw(`runs/`, JSON, receipt, log, source manifest, checkpoint SHA)는 불변. summary만 생성.
- 결과 판정 라벨: `PASS_MECHANISM` / `PASS_MECHANISM_UNRESOLVED` / `FAIL` / `VOID_EXECUTION` 중 하나.
- README·`docs/status`·발표자료 갱신은 held-out summary 봉인 후. FAIL도 WORKLOG·VERIFICATION에 동일 형식으로 기록.
- 이 문서는 GPU 실행 전 commit하고 SHA-256을 WORKLOG에 남긴다. 이후 수정은 amendment 절로만 추가.

---

## 10. 사용자 확인 필요 (실행 전)

1. 금지 ① 재검 조건 충족(§2.2)에 동의하는가 — GPU authority는 현재 closed다.
2. 4 arm × 3000 epoch(추정 12–16 h) + 평가(약 10 cell, 각 ~30 min) 예산 승인.
3. 학습 밀도 70 고정에 동의하는가 (대안: 70/145 두 밀도 교대 — 그러나 밀도를 인과축에서 제외한다는 brief 원칙에 어긋남).
4. A1과 A2 사이에 mode+coverage가 동시에 들어가는 설계(§3)에 동의하는가, 아니면 5-arm으로 mode-only를 추가할 것인가.
5. coverage/belief가 geofence와 같은 VIO/GPS pose 가정을 쓰는 것을 논문 claim boundary에 그대로 쓸 것인가 ("mapped-boundary + onboard sensing", localization-free 아님).

---

## 구현 전 명세 정합성 보정 (2026-09-03, GPU 실행 전)

§3의 표는 belief를 25-D로 고정했지만 바로 아래 문장은 12 sector mass + entropy만 열거해
13-D만 정의했다. 959-D actor 계약과 §7 schema test를 유지하기 위해 누락된 12-D를 각
sector의 정규화 거리 1차 모멘트 `sum(b(cell) * distance/arena_diagonal)`로 명시한다. 따라서
belief 순서는 `[sector mass 12, sector radial moment 12, normalized entropy 1]`이다. masked
sentinel은 `[1/12 × 12, 0 × 12, 1]`이다. 이는 동일 belief와 actor-visible pose만 사용하며
GT 입력을 추가하지 않는다.
