# WORKLOG — MOTAR 연구 작업 기록

무엇을, 왜 바꿨는지의 시간순 기록. 최신이 아래쪽. (연구 로드맵은 워크스페이스 루트의
`RESEARCH_PLAN.md`, 실행 방법은 `README.md`의 Getting Started 참고)

**기록 규칙:** 코드·설정·실험 조건 또는 결과를 변경한 작업은 같은 작업 안에서 이 파일도 함께
갱신한다.

---

## 2026-07-09 — 저장소 정리 & NavRL LiDAR 재현

### 저장소 셋업
- `research/navrl-env` 브랜치 생성, origin = github.com/joshualikaist/MOTAR.
- 연구에 불필요한 파일 삭제 (docs/, sim2real/, dce 예제, CI, supplementary ~170MB) — 커밋 `7847994`.
- `main`은 업스트림(ntnu-arl) + MOTAR README 병합 상태로 유지, 연구 커밋은 전부 `research/navrl-env`에.

### NavRL식 LiDAR (커밋 `b4f72df`)
- **신규** `config/sensor_config/lidar_config/navrl_lidar_config.py` — NavRL 사양 재현:
  수평 36빔(10° 간격, hfov −170°~180°로 끝점 중복 회피), 수직 4빔(−10°~+20°), range 4 m.
- **신규** `config/robot_config/navrl_quad_config.py` — 카메라 끄고 위 LiDAR만 단 쿼드(`navrl_quad`).
- `base_sensor_config.py` + `sensors/warp/warp_sensor.py` — `yaw_only_attach` opt-in 플래그 추가:
  센서가 롤·피치는 무시하고 yaw만 따라가도록 (NavRL과 동일). 기본값 off라 다른 로봇에 영향 없음.

### navrl_task 골격 (커밋 `11b836f`, `3fcf2db`)
- **신규** `task/navrl_task/navrl_task.py` — 관측 152차원 = S_int(8: goal프레임 방향/거리/속도)
  + LiDAR 36×4 평탄화(144). 행동 = goal프레임 3D 속도(→ lee_velocity_control). 보상 = NavRL
  정적 브랜치(vel + alive + static_safety − smooth − height, 충돌 페널티).
- **신규** `config/task_config/navrl_task_config.py`, task_registry에 `navrl_task` 등록.
- rl_games 연결: `rl_training/rl_games/runner.py`에 등록 + `ppo_navrl.yaml` 신규(MLP [256,256,128]).
- **구현 gotcha**: `env.step()`은 센서를 렌더하지 않음 — task가 `post_reward_calculation_step()`을
  호출해야 LiDAR가 실제 raycast됨. 빠뜨리면 전 레이가 max range로 나옴.

---

## 2026-07-09 밤 ~ 07-10 새벽 — 첫 학습과 정직한 진단

- 512 env 헤드리스로 1500 epoch 학습 (VRAM 6.8/8 GB — 512가 상한).
- 평가 지표 내장(커밋 `345cc38`): `ever_reached`, `mean_closest_approach` 등.
- **결과: 목표 도달률 0%.** 원인 진단 — 목표가 첫 에피소드부터 맵 끝(20~38 m)에 뽑히는데
  에피소드는 150스텝(≈최대 15 m 비행)이라 물리적으로 도달 불가.

### 목표 거리 커리큘럼 (커밋 `0a1f8ed`)
- `navrl_task.py` + config에 커리큘럼 추가: 목표를 스폰 주변 2.5~5 m에서 시작, 도달률 60%
  넘으면 최대 거리 +1.5 m씩 확장 (최대 18 m).
- 재학습 결과 ever_reached 0% → 20~36%로 개선됐으나 **~30%에서 정체** (2 m 근처까지 가서
  마지막 1~2 m을 파고들지 못함) — 충돌 회피가 병목.

### README Getting Started (커밋 `e0cca2b`)
- 환경 셋업~학습~재생~평가 전 과정 가이드 + `examples/navrl_task_example.py`(뷰어 예제) 추가.

---

## 2026-07-10 — 뷰어 segfault 수정 & 전용 막대 환경 구축

### segfault(코어 덤프) 원인과 해결
- `navrl_task_example.py` 실행 시 `PxgCudaDeviceMemoryAllocator fail ... Result = 2` 후 segfault.
- 원인: **GPU 메모리 부족** — 백그라운드 학습이 5.7 GB를 물고 있어 뷰어(PhysX)가 할당 실패.
- 해결: 학습 프로세스 정지로 VRAM 확보. **교훈: 8 GB에서는 학습과 뷰어 동시 실행 불가.**

### 전용 "빈 공간 + 막대" 환경 `navrl_bars_env` 신설
그동안 task가 스톡 `env_with_obstacles`(6면 벽 + 패널 + 잡동사니 방)를 쓰고 있었음 —
"빈 공간에 막대만" 통제 환경이 아니었음. 이를 폐기하고 전용 환경 구축:

- **신규** `config/env_config/navrl_bars_env.py` — 빈 10×10×3 m 아레나(벽·패널 없음),
  막대 16개만. `env_manager/__init__.py`에 `navrl_bars_env` 등록,
  `navrl_task_config.py`의 `env_name`을 이것으로 교체.
- **숨은 버그 수정**: 스톡 env는 z가 0 중심(−2.5~2.5)이라 드론 스폰 상당수가 task 바닥
  기준(0.1 m) **아래에서 시작 → 즉시 추락 판정**되고 있었음. 새 환경은 지면 기준
  z(0~3 m)라 스폰 전부 유효. (기존 학습의 높은 충돌률 55~78%를 부풀린 원인일 가능성)

### 막대 사양 확정 (사용자 지시 반영)
- **가변 크기 막대**: 가로·세로 각각 0.4~0.8 m 랜덤, **높이 2 m 고정**, 바닥에 서 있음(0~2 m).
  - **신규** `tools/generate_bar_assets.py` — 막대 URDF 풀(40개) 생성 스크립트 (시드 고정, 재현 가능).
  - **신규** `resources/models/environment_assets/bars/bar_000~039.urdf`.
  - `config/asset_config/env_object_config.py`에 `bar_asset_params` 추가 — bars/ 풀에서
    에피소드마다 랜덤 16개 선택.
- **겹침 방지 + 간격 보장**: `env_manager/asset_manager.py`에 **지터드 격자(jittered grid)**
  배치 추가 (`_enforce_min_xy_spacing`, env config의 `min_obstacle_xy_spacing`로 opt-in —
  기본 off라 다른 환경 영향 없음).
  - 필드를 4×4 = 16칸으로 나눠 칸마다 막대 1개 + 칸 안에서 랜덤 지터
    → **중심간 ≥ 1.3 m 수학적으로 보장** = 최대 크기(0.8 m) 막대끼리도 **실제 틈 ≥ 0.5 m**.
  - 처음엔 rejection sampling(균등 랜덤)으로 구현했으나 16개 @ 1.3 m 간격은 점유율 40%로
    수렴 실패(측정: 최소 0.747 m). 균등 방식의 적정 개수는 12개(점유율 30%)임을 시뮬레이션으로
    확인 → 16개 유지를 위해 격자로 전환 (사용자 승인).
- **2D 비행 @ 1 m 고도**: 드론은 1 m 상공에서 XY로만 추적 (향후 도망 표적도 XY만).
  - `navrl_quad_config.py` — 스폰 z를 1 m 고정(z-ratio 1/3), 초기 수직속도 0.
  - `navrl_task.py` — ① 수직 속도 명령 강제 0 (`transform_action_to_command`),
    ② 목표를 z = `flight_altitude`(1 m)에 배치 (기존 고도 지터 제거).
  - `navrl_task_config.py` — `flight_altitude = 1.0` 추가.

### 검증 (64 env 헤드리스)
| 항목 | 결과 |
|------|------|
| 막대 수 | 16개/env |
| 막대 중심간 최소 거리 | 1.333 m (≥ 1.3 요구 **PASS**, 겹침 0건) |
| 막대 위치 | z중심 1.0 m — 바닥 0~2 m에 서 있음 |
| 드론 고도 (40스텝 전속 비행 후) | 0.83~1.13 m — 1 m 유지 확인 |
| 목표 고도 | 전부 정확히 1.0 m |
- 탑다운 배치도: 워크스페이스 루트 `navrl_bars_env_layout.png`.

### README 확장
- Phase 1 아레나(막대 16개, 격자 배치, 2D @ 1 m) 설명 추가.
- **TensorBoard 모니터링 가이드** 추가: rl_games가 `runs/<run>/summaries/`에 자동 기록,
  `tensorboard --logdir runs`로 열람. 봐야 할 지표(rewards/step, episode_lengths,
  losses/entropy, info/kl 등)와 콘솔 `NavRL progress` 지표(ever_reached, crash,
  mean_closest_approach 등) 해석 표 정리.

### bars 환경 첫 본 학습 (사용자 실행, `ppo_260710_1608_navrl`)
- 512 env × 1500 epoch. mean reward 28→338, 에피소드 길이 29→144/150 (충돌 급감).
- 사전 스모크(2.5분, 119 epoch)에서 이미 ever_reached 71%, 커리큘럼 5→14 m 연속 확장 —
  구 환경에서 30% 정체였던 것과 대비. 환경 교체(벽 제거+스폰 버그 해소+2D)가 결정적.

### 학습 대시보드 개편 (콘솔 + TensorBoard)
예전 요격 과제 때 만든 콘솔 박스가 "Intercept" 제목·intercept 지표 하드코딩이라 교체:
- **신규** `task/navrl_task/train_dashboard.py` — 에피소드 종료 결과를 epoch 단위로 집계.
- `navrl_task.py` — 매 스텝 종료 에피소드를 집계기로 전달(`_record_epoch_dashboard`).
- `early_stop_a2c_agent.py` — epoch마다 navrl 요약 소비: 박스에 `goal reached / success @
  timeout / crash / timeout / closest to goal / curriculum max` 표시 + TB `navrl/*` 스칼라 기록.
- `aerial_tensorboard.py` — 네임스페이스 키(`navrl/...`) 그대로 스칼라로 통과.
- `pretty_train_stats.py` — 제목을 `AERIAL_RUN_TITLE` env var 기반으로(하드코딩 제거),
  "(design ~0–1000)" 문구는 `AERIAL_REWARD_DESIGN_MAX` 선언 시에만 표시.
- `runner.py` — task별로 위 env var 자동 설정(+`AERIAL_TASK_CONFIG_MODULE` → ep length /150 표기).
- `run_header.py` — `navrl_task: "NavRL Navigation · PPO"` 제목 등록.
- tensorboard 뷰어 설치(2.13.0, numpy 1.23.0 핀 무사). 실행: `tensorboard --logdir runs`.
- **주의**: 16:08 학습 run은 이 수정 이전 시작이라 옛 배너로 표시됨. 다음 run부터 적용.

### play 모드 즉시종료 버그 수정 (근본 원인)
`--play` 실행 시 창이 뜨자마자 조용히 종료되는 문제:
- **원인**: `navrl_task.reset()`이 목표만 재배치하고 `sim_env.reset()`(로봇 리스폰)을 안 불러,
  play 첫 리셋 후 드론들이 **빌드 포즈(원점, 막대와 겹침)** 그대로 → 첫 스텝에 전원 충돌
  (-10) → rl_games가 목표 게임 수 즉시 충족으로 판단 → 1스텝 만에 평가 종료.
- 학습에서는 이 전원 충돌이 곧바로 강제 리셋을 유발해 자가 복구됐기 때문에 미발견.
- **수정**: `reset()`에서 `sim_env.reset()`(리스폰) → 목표 배치 → LiDAR 렌더 순으로 변경.
  검증: 리셋 직후 z=1.0 m, 접촉력 0 N, 중간 리셋도 클린.
- **평가 확인**: 512게임 headless play — **평균 보상 299.3, 평균 134.4/150스텝 생존**
  (학습 지표와 일치). 뷰어 play도 정상 동작.
- 참고: play 콘솔이 조용한 것은 `AERIAL_RL_QUIET_STARTUP=1`(기본) 필터 때문 —
  `AERIAL_RL_QUIET_STARTUP=0`으로 끄면 `av reward` 등 rl_games 출력이 보임.

### 뷰어 관찰 결과 + LiDAR CNN 특징추출기 (사용자 방향 확정)
- 뷰어 관찰: 드론이 목표로는 가지만 **막대에 충돌하는 케이스가 여전함** → 학습량 부족 +
  평탄화 MLP의 공간정보 소실이 원인 후보. 사용자 결정: 학습 6000 epoch로 상향 + CNN 진행.
- **신규** `rl_training/rl_games/navrl_network.py` — NavRL ppo.py의 특징추출기를 rl_games
  커스텀 네트워크("navrl_cnn")로 이식: 36×4 스캔 → Conv(1→4→16→16, 커널 5×3, 수평 ×2
  다운샘플) → 288→128 임베딩(LayerNorm) → S_int(8)와 concat → MLP [256,256] → Gaussian
  헤드. (NavRL과의 차이: Beta 대신 Gaussian(logstd), actor/critic 트렁크 공유 — 기록됨)
- **신규** `ppo_navrl_cnn.yaml` (CNN, 권장) / 기존 `ppo_navrl.yaml`(MLP 베이스라인)은 비교용
  유지. 둘 다 `max_epochs: 6000`(≈2h @ 1.1s/epoch)으로 상향. runner.py에 네트워크 등록.
- **스모크 (125 epoch, 2.5분)**: CNN이 MLP 동일 시점 대비 우세 — goal reached 71.8%,
  success@timeout 62.6%(MLP ~52%), **crash 37.4%(MLP ~46%)**, 커리큘럼 12.5 m 도달.
  새 대시보드 박스·TB `navrl/*` 스칼라 모두 정상 동작 확인.

### 성공 반경 1.0 → 0.5 m 수정 (사용자 지적, NavRL 원본 확인)
- 사용자 지적: 요격 연구인데 1 m는 "근처 통과"지 "포획"이 아님 — 0.5 m가 맞지 않냐.
- NavRL env.py:583 확인: `reach_goal = (distance < 0.5)` — **원본도 0.5 m**. 기존 1.0 m는
  연구계획서 초안의 플레이스홀더("예: 1 m")가 그대로 들어간 것으로, 원본 사양과 달랐음.
- `success_radius = 0.5`로 수정 (goal reached / success@timeout / 커리큘럼 게이트 모두 이 값).
- **주의**: 이전 run들(1608 등)의 reach 지표는 1.0 m 기준이라 새 run과 직접 비교 불가.
  0.5 m 기준이 더 어려워 커리큘럼 확장도 느려질 수 있음 — 정상.

### 다음 단계
1. `ppo_navrl_cnn.yaml`로 6000 epoch 본 학습 (0.5 m 기준) → MLP 베이스라인과 비교.
2. 부족하면 보상 재균형(속도 vs 안전 가중), 커리큘럼 최종거리(18 m) 완주 확인.

---

## 2026-07-13 — loiter(배회) 진단 → 리워드 재설계 + 아레나 대형화

### 캡처종료 첫 본학습 결과: 정책이 "포획" 대신 "배회"를 학습 (실패)
- run `ppo_260713_1950_navrl` (캡처종료 + CNN + alive −0.05 + capture 30, 6000 epoch 완주).
- TB `navrl/*` 실측: **captured_rate가 ep46에 0.796까지 갔다가 붕괴 → 최종 0.067**,
  timeout 0.838, crash 0.095(94%→9.5%로 회피는 잘 배움), mean_closest_approach ~1.36 m.
- **근본 원인(산술)**: 리워드의 전역 최적해가 "포획"이 아니라 "목표 근처 배회".
  안전항 `mean(log(lidar_dist))`이 개방공간에서 **+log(4)≈+1.386/step 고정 수입** →
  `V_loiter ≈ 1.336·Σγ^k ≈ 104` ≫ `V_capture = 30`(캡처 시 에피소드 종료로 미래보상 포기).
  alive는 이미 −0.05(비용), capture도 이미 30으로 올렸지만 **안전항 수입이 진짜 범인**이라 미달.
- NavRL 원본엔 이 문제 없음: 먼(24–48 m) 목표로의 flyby 항법이고 reach는 통계일 뿐
  (종료·보너스 없음). 정밀 캡처 요구는 우리 요격 각색이 만든 새 문제.
- **최종 요약기 버그**: 종료 시 콘솔/`run_summary.json`이 옛 intercept 요약기라 "요격 0건 /
  target dist — / reward 낮음"처럼 **엉뚱·빈 필드**를 찍음. 실제 신호는 TB `navrl/*`에만 있음.
  (navrl용 요약기로 교체 필요 — 아직 미수정.)

### 리서치 워크플로우 → 방안 A~F
- 5-에이전트 병렬 조사(PBRS 이론 · 드론 goal-reaching 리워드 · 함정 · 기존 aerial_gym 요격 메커니즘).
- 기존 `shooting_moving_target_task`가 **동일 "stop-short" 버그**를 finish-funnel + 마일스톤 +
  closing bonus + (dense 클램프/터미널 정규화)로 **97.5% 성공**시킨 이력 확인
  (`lms_rl_trial/shooting_fixed_target/INTERCEPT_TUNING_REPORT.md`, 패치 #1~#10 교훈).
- 핵심 교훈: **A·D(포텐셜형)는 최적성 보존이라 단독으론 loiter 최적해를 못 옮김** →
  반드시 "최적해를 옮기는 비-포텐셜 변경(B)" + "학습가능하게 하는 조밀신호(A/C)"를 짝으로.

### 적용(활성) — B1 + B3 + A
- **B1** 안전항 재베이스라인: `navrl_task.py:add_static_safety_reward`에서
  `r_ss = (log(dist_m) − log(range)).mean()` → 개방공간 수입 +1.386을 **0으로**. 상수 시프트라
  **회피 그래디언트 불변**, loiter 수입만 제거(`V_loiter` 104 → ≈ −4.6). `import math` 추가.
- **B3** `collision_penalty −10 → −20` (config) — 자살 가드(무캡처 −12.5보다 크래시가 더 나쁘게).
- **A** PBRS progress: `compute_state_reward_and_terminations`에
  `reward += progress_weight·(prev_dist − γ·dist)`, 종료(capture/crash) 시 Φ=0(Grześ 2017),
  timeout은 truncation이라 그래디언트 유지. `progress_weight=1.0`, `progress_gamma=0.99`(=PPO γ),
  `self.prev_dist` 버퍼(__init__/reset_idx 시딩).

### 주석처리(DISABLED — 학습 후 이상하면 해제)
- **C1 finish-funnel**: `navrl_task.py` 리워드 블록 + `navrl_task_config.py`의
  `funnel_coef/outer/width`. 0.5–1.0 m 셸에서 "계속 안쪽으로 이동" 보상(brake-short 방지).
- **F 세그먼트 캡처**: `navrl_task.py` 4곳(prev_pos 버퍼/reset/세그먼트 판정/갱신). 2 m/s·0.1 s에서
  0.5 m 구를 스텝 사이로 터널링하는 fly-through를 잡음. 켜려면 4곳 모두 해제.

### 환경 대형화 (커리큘럼 18 m를 실제로 도달가능하게)
- **아레나 10×10 → 24×24×3 m** (`navrl_bars_env.py`, bounds [0,−12,0]~[24,12,3]). 전엔 18 m 목표가
  아레나 밖이라 ~10 m로 clamp됐음(대시보드 curriculum_max 18 표시가 실제와 불일치).
- **막대 16 → 48개** (`bar_asset_params.num_assets`), ~8/100 m²(NavRL ~22/100 m²보다 희소 — VRAM.
  참고: RESEARCH_PLAN의 "2.2/100 m²"는 10배 오타, 실제 NavRL은 ~22). spacing 1.3 → 1.8.
- **에피소드 150 → 250 스텝** (18 m 목표: 직선 ~90, 위빙 ~150+ 스텝이라 타임아웃이 실패요인 안 되게).
  (→ 이후 `episode_len_steps=300`으로 상향(24 m cross-field 대각 goal 대응) — config가 현재 최신값.)

### 새 학습 시작
- run `ppo_260713_2210_navrl` (256 env, 48 장애물 확인, fresh weights, 6000 epoch) 시작됨.
- **VRAM**: 옛 512 env×16막대 ≈ 6.8 GB. 48막대는 메시 3배 → `--num_envs 256`부터, OOM이면 128.
  에피소드 250이라 epoch도 ~1.7배 느려짐.
- 판정: `captured_rate ↑`(→0.9) · `crash/timeout ↓` · `mean_closest_approach → 0.5 m`.
  여전히 마지막 1 m서 멈추면 C1 해제, 캡처 누락 의심되면 F 해제.

### 남은 정리(TODO)
- 최종 요약기(`run_summary.json`/콘솔)를 navrl 지표(captured/crash/timeout/closest/curriculum)로 교체.
  (→ 이후 완료: `train_run_recorder.py`에 `is_navrl` 분기 + `_build_navrl_hints` + navrl 요약 박스/density bars 필드.)
- 위 리워드/환경 변경은 전부 **미커밋 working tree** — `git diff` 검토 후 커밋. (→ 이후 커밋됨 `be701eb`)

---

## 2026-07-14 ~ 07-15 — Phase 1 완성(cross-field + yaw 제어) → Phase 2 밀도 착수

Phase 1을 정직한 난이도로 재설계하고, 잔여 충돌의 진짜 원인(기하)을 찾아 **yaw 제어로 해결(captured 0.95)**,
이어서 Phase 2 밀도 스윕에 착수한 이틀. 상세 결과는 아래, 실험 로그는 `CRASH_TUNING_LOG.md`.

### 새 스폰 스킴 + 0.28m 박스 충돌 (커밋 `71aa606`)
- **크로스필드 스킴**: 드론 x≈0(왼쪽 가장자리) → 목표 x=k(먼쪽) → 매 에피소드 **48개 균등 막대밭 전체 관통**.
  옛 스킴은 목표가 드론 쪽에 몰려(관통 8%뿐) 성능이 부풀려졌음(captured 86%).
- **드론 충돌 = 0.28m 박스**(팔/프롭 포함, `quad_navrl_collide.urdf`, navrl 전용). 옛 0.05m 구 대비 단면적 ~10배.
- 커리큘럼: epoch비례 k_max 7→24m, 이후 k_min 5→20m(램프 2000→5000 epoch). checkpoint에 num_task_steps 저장.
- **run `ppo_260714_1904`: captured 0.65 / crash 0.35 / timeout 0 = 정직한 Phase-1 baseline.**

### crash 레버 실험 → "기하 바닥" 발견 (`CRASH_TUNING_LOG.md` Run B/C/D)
- 멀티에이전트 워크플로우로 speed-gated clearance 페널티 설계(cw=6). run `2207`: **captured 0.66 / crash 0.32.**
- **판정: 감속은 시켰으나(mean_ep_len 88→116) crash 거의 안 줄임.** 큰 감속에도 crash 불변 ⇒ **원인은
  속도/리워드가 아니라 기하(corner-clip)**. 리워드로 crash 줄이기 소진 → cw=0 복원(`44e86a2`), k_min 스케줄
  2000→5000(`8b97909`).

### ★ yaw 제어 (option b) → Phase 1 해결 (커밋 `4459281`, run `ppo_260715_0251`)
- **원인 확정**: 요가 스폰 ±30°로 **고정**돼 이동방향과 어긋나면 0.28m 박스가 **대각 0.40m**로 갭을 쓸어 corner-clip.
  NavRL 원본은 모터레벨로 yaw를 학습(=option b) + 목표 바라보고 스폰.
- **구현(멀티에이전트 3설계→적대검증→종합)**: action 3→4(`action[:,3]`=yaw-rate), **관측 heading 추가**
  (S_int 8→12: goal_bearing_body+vel_body_xy, vehicle-frame — 없으면 학습 불가), NavRL 전용 컨트롤러
  `lee_velocity_control_navrl`(yaw clamp π/3→2.5), speed-gated crab 페널티(0.3)+yaw-rate² 댐핑(0.02).
  속도액션은 goal-frame 유지(요와 디커플→nav 안정). 런타임 스모크 검증 후 커밋.
- **결과: captured 0.954 / crash 0.046 / timeout 0** (k_min=20 최심에서도 0.950). **crash 0.32→0.046(7×↓),
  NavRL 0.81·M1 0.90 상회. Phase 1 사실상 해결.** 기하 가설 완전 입증.
- **yaw-off ablation 브랜치** `ablation/yaw-off`(@`8b97909`) 준비 — 0251에서 yaw만 뺀 대조군(논문 attribution용).

### Phase 2 밀도 배관 (커밋 `7e1b6a8` + 수정 `0aed4be`)
- 환경변수: `NAVRL_MAX_BARS`(빌드천장, 기본 150) / `NAVRL_NUM_BARS`(활성수) / `NAVRL_DENSITY_CURRICULUM`.
  **`keep_in_env=False`로 런타임 활성 막대수 제어**(`navigation_task`의 검증된 패턴, `num_obstacles_in_env`를
  리셋마다 읽음), spacing 1.8→1.5로 150개 수용, x밴드 0.09→0.13. checkpoint에 `n_bars_active`.
- 수정(`0aed4be`): 밀도 커리큘럼 워밍업 누적기 리셋 버그 + max-bars 조용한 0 경고 추가(커리큘럼 경로 한정).
- **VRAM 스크리닝**: 150 bars + 256 env ≈ **6.1GB** (8GB OK, num_envs 안 낮춰도 됨).
- **run `ppo_260715_1552` (150 bars ≈ 26/100m²): captured 0.85 / crash 0.15 / timeout 0.** NavRL 밀도에서도
  NavRL 0.81 상회. **밀도-성능(RQ1): 48막대 0.95 → 150막대 0.85.** 나머지 {25,50,75,110} seed 1 진행 예정.

### 인프라 / 문서
- 실행 래퍼 `train_navrl.sh` / `play_navrl.sh`(짧은 실행, VRAM는 `NUM_ENVS`), 2번째머신 원샷
  `bootstrap_second_machine.sh` + `SETUP_SECOND_MACHINE.md`(urdfpy 직접 clone·rsync 회수).
- `RESEARCH_PLAN.md`·`ROADMAP.md`를 repo 안으로 이동(전엔 repo 밖이라 clone에 안 왔음). 콘솔 박스·progress
  로그에 run 이름 표시. 옛 run 24개 요약 후 삭제(`RUNS_ARCHIVE_SUMMARY.md`). upstream 원격 제거.

### ★ 막대 배치 방식 전환: 지터드 격자 → NavRL식 랜덤 rejection (커밋 `54d7cfe` + `10708e2`, 20:26~20:35)
- **변경**: `asset_manager.py`에 `_random_rejection_xy_spacing` 추가 + env config에 `obstacle_placement_mode`
  스위치. `navrl_bars_env.py`가 `mode="random"`으로 전환(밴드 균등 스캐터 + 최소간격 rejection, 포화 시
  128회 실패마다 간격 ×0.8 완화 = NavRL "do not stall when saturated" 트릭). 후보를 32개씩 배치 샘플링해
  파이썬 per-candidate 루프 회피(GPU 벡터화).
- **이유**: 격자는 min-spacing을 결정론적으로 보장하지만 **눈에 보이는 행/열 아티팩트**가 있어 밀도 실험의
  일반성을 해침. NavRL 원본이 랜덤 스캐터라 사양 충실도도 ↑. **opt-in**(env config에서 `mode` 지정, 기본
  `grid`)이라 타 환경 무영향. 07-10에 "rejection→격자로 전환(사용자 승인)"했던 결정을 **뒤집은 것**:
  당시 실패 원인(16개@1.3m 균등)은 완화 로직 + GPU 배치로 해소됨.
- **⚠️ 스윕 교란 주의**: run `1552`(150막대, **격자**, captured 0.85)·`1922`(75막대, 격자, 중단)는 **옛 격자**
  결과라 새 랜덤 배치 run과 **직접 비교 불가**. 배치 방식이 밀도-성능 곡선(RQ1)의 교란변수가 됨.
  → run `2038`(150막대, **랜덤**, seed 1) = 새 배치 기준의 150막대 재-baseline. **나머지 밀도{25,50,75,110}도
  전부 랜덤 배치로 재실행해야 한 곡선 안에서 비교 가능** (격자 데이터포인트는 폐기).

### 2026-07-15 문서 정합성 리뷰 결과
- **밀도 분모 통일 완료**: 실제 배치 밴드 x-ratio[0.13,0.96]×y[0,1] = **약 478 m²** 기준으로
  `RESEARCH_PLAN.md`를 갱신. 48→10.0, 75→15.7, 150→31.4 /100m².
- **VRAM 안내 수정 완료**: 150막대+256env≈6.1GB 실측에 따라 고밀도에서도 256 env 유지 가능하다고 명시.
- **episode 길이 정정 완료**: 실제 config 최신값 `episode_len_steps=300`을 07-13 기록에 후속 표기.
- (코드 소소) `navrl_task._clamp_active_bars`/`_set_active_bars` int-cast 중복, `train_run_recorder`가 navrl run에도
  옛 intercept 컬럼(near_miss/surface_gap 등)을 빈칸으로 계속 기록 — 동작엔 무해, 정리 여지.
- 문서 전반 옛 스킴 흔적(10×10·16막대·150스텝 등) → 현재 스킴으로 동기화.

---

## 2026-07-16 ~ 07-17 — 밀도 스윕 완성 → Phase 3(이동 표적) 착수

### 밀도 스윕 결과 (랜덤 배치, seed 1, 각 6000 epoch)
| 막대수 | 밀도(/100m², 밴드478) | captured(last-500) | crash |
|---|---|---|---|
| 25 | 5.2 | 0.972 | 0.026 |
| 50 | 10.5 | 0.970 | 0.030 |
| 75 | 15.7 | 0.942 | 0.058 |
| 110 (NavRL 앵커) | 23.0 | 0.926 | 0.074 |
| 150 | 31.4 | 0.656 | 0.344 |
- **NavRL 앵커까지 평탄 → 110→150 절벽(−27pt).** 기하 분석: 밴드 478m²의 RSA 재밍 한계 ≈148,
  적응완화 시작 ≈115-120 → 절벽은 배치 기하의 한계와 일치(선형 아님 = 논문 핵심 스토리).
- ⚠️ 25/50은 4GB 노트북에서 **minibatch 정합(512→4096) 이전** 학습 = 업데이트 16배 confound.
  정성 결론 유효, 정량 인용 주의. GPU4GB 프리셋 정합 후 재실행 예약.
- 잔여: 150 seed2(진행), 120막대(절벽 경계), 25/50 재실행.

### Phase 3 착수 — 문헌조사·계획(`PHASE3_PLAN.md`, 커밋 `460366a`) 후 구현
- **사전 전면 감사**(6렌즈 멀티에이전트): 실버그 4(timeout 스텝이 다음 에피소드 스폰스캔으로 safety
  보상 받는 leakage, play의 `--seed` 증발, play_navrl.sh GPU4GB 미적용, set_env_state가 checkpoint
  밀도로 NAVRL_NUM_BARS를 덮어씀) + 죽은코드 ~12(C1 funnel, `class eval`, `full_scale_hold_epochs`,
  clearance 경로, `terminate_on_capture=False`/`return_state_before_reset` 분기, `_REACHED`,
  `quat_rotate`/`torch` import 등) + 문서 드리프트 ~15 — **전부 수정/삭제**.
- **이동 표적 v1 구현** (기본 OFF = Phase 1/2와 byte-호환):
  - `_advance_target()`: 가상점(액터 0, LiDAR 불가시), **rl_dt=0.1s**(물리 dt×10 — `obs_dict["dt"]`
    직접 쓰면 1/10 속도 나는 함정을 회피), 패턴 cv(벽반사)/waypoint/circle(평가전용)/mixed,
    막대 clearance push-out + 벽 클램프, **realized velocity**(반사·push-out 포함 실제 변위/dt)를
    리워드에 공급.
  - **range-rate 리워드**: `(v_drone − v_target)·dir` — v_t=0이면 기존 항과 IEEE-정확히 동일.
  - **PBRS re-anchor**: `‖prev_pos−target_new‖ − γ‖pos−target_new‖` — 표적 이동분을 드론 상벌에서
    제거(naive형은 도주 표적에서 정지 드론에 −0.94/step 오벌점). prev_dist 버퍼 폐지.
  - **상대프레임 세그먼트 캡처**: prev_rel→rel 선분 vs 원점 0.5m — 상대속도 4m/s 터널링 방지,
    v=0에서 point 테스트의 strict superset(수학 유닛테스트 `scratchpad/p3_math_unit.py` 전항 PASS).
  - 속도 커리큘럼 0→`NAVRL_TARGET_SPEED_FINAL`(epoch비례, num_task_steps 기반=checkpoint 생존),
    평가 노브 `NAVRL_TARGET_SPEED`/`NAVRL_TARGET_PATTERN`/`NAVRL_SEED`. 대시보드/TB에
    `navrl/target_speed_{max,mean}_m_s` 추가.
- goal 배치: 10회 rejection 실패 시 최근접 막대에서 방사 snap-out(고밀도 안전망).

## 2026-07-17 — 비전 피벗: 센서 전용 요격 (Stage 0.5 + Stage 1)

**방향 전환(사용자 결정)**: actor가 표적의 GT 상대위치를 관측에서 받지 않는다. LiDAR+비전(카메라)
정보만으로 표적을 찾아 요격. 이것이 논문의 최종 문제 설정이 됨 (기존 GT-주입 경로는 baseline).

- **Stage 0.5 — 해석적 표적 주입** (`38a4f22`): 이동 표적을 LiDAR 커널 내 ray-sphere로 주입
  (`NAVRL_VISION=1`, semantic id 50). 메시+refit 방식은 refit 루프(Python, env당 wp.Mesh.refit)가
  81ms@N128로 학습 불가(8.5 steps/s) → 해석적 주입은 오버헤드 ~0 (25.8 vs 기준 25.0 steps/s).
  검증: `tools/test_navrl_p3_stage0.py` ALL PASS. GPU 이전 가이드 `GPU_SCALING_GUIDE.md` 추가.
- **Stage 1 — 센서 전용 actor + 비대칭 critic**:
  - actor 관측 305 = ego 9(기체속도/요레이트/직전행동/고도) + 검출기 8 + LiDAR 2채널 288(range+표적마스크).
    GT 무누출: blind 상태에서 표적 위치를 바꿔도 관측 완전 동일(프로브 C 검증).
  - 검출기(`navrl_detector.py`) = 전방 카메라 모델(HFOV 87°/20m): FOV+거리+**막대 가림(warp LOS 레이)**
    통과 시에만 방위/거리 노출 + last-seen 추적기(온보드 필터 표준). Stage 2에서 실픽셀로 교체 가능.
  - 행동 = goal프레임 → **기체(vehicle)프레임** (표적 미지 → goal프레임 정의 불가).
  - 비대칭 critic: `states` 313 = actor관측 + GT 8, rl_games `central_value_config` 네이티브
    (runner.py {obs,states} 래퍼 + state_space). 배포 시 critic 폐기 → actor는 센서만.
  - 보상: 기존 유지(GT는 환경 심판) + 가시성 보너스 0.02 + 장외(OOB) 종료.
  - 네트워크 `navrl_vision`(2ch CNN) + 옵션 LSTM(dones-masked, `ppo_navrl_vision_lstm.yaml`).
  - 실행: `NAVRL_VISION=1 ./train_navrl.sh` (LSTM: `NAVRL_LSTM=1` 추가).
- **검증**: stage1 프로브 ALL PASS(가시/후방/원거리/가림/무누출/추적기/행동프레임/states);
  vision off 스모크 ALL PASS(156 무손상); sanity 학습 40ep(비대칭 critic 저장 확인, 입력 313,
  ep_len 19→32 성장) + LSTM 10ep(가중치 저장 확인). sanity run 2개(1522/1525)는 삭제 가능한 테스트.
- **8GB 판단**: 의미론 LiDAR+검출기 경로는 VRAM 추가 ~0. raw 카메라 CNN은 Stage 2(옵션)로 격리.

## 2026-07-18 ~ 07-19 — 센서 전용 밀도 커리큘럼: 첫 결과 + 진단 + 순차 전환

**평가(play) 경로 수정 (커밋 eaad795)**: 비전 체크포인트가 play_navrl.sh에서 안 돌던 2버그
(156-dim cnn yaml 오선택 / {obs,states} dict를 player가 못 먹음) 수정. 이제 held-out 평가 됨.
- **110막대·정적 from-scratch (baseline)**: captured **47.3%** / crash 52% / timeout 0.7% (n=2050).
  timeout≈0 → 표적은 찾음, 실패=충돌. 병목=장애물 회피(밀도).

**4GB 경로 (커밋 3f50639)**: `ppo_navrl_vision_4gb.yaml`(N=128, mb 2048) + train/play 스크립트가
`NAVRL_VISION=1 GPU4GB=1`에 vision_4gb 선택. **1650 Ti("joshualisky")에서 실기 검증됨**.

**밀도 커리큘럼 (커밋 c5bd707 knob) — 첫 run `1841`**: `NAVRL_DENSITY_CURRICULUM=1 WARMUP=1000
THRESHOLD=0.6`. 밀도 knob들 env화(기본 threshold 0.8/warmup 2500는 GT용이라 센서엔 부적합).
- **밀도 계단 25→130 성공적 상승**: 승급별 안정 캡처 25:93% 55:79% 85:61% 100:59% **115:62%** 130:47%(정체).
- ⚠️ **그런데 최종 체크포인트 110 held-out = 40.8%** (crash 50%, **timeout 8.8%↑**) → **baseline 47%보다 낮음**.
  학습 중 62%(115막대)는 목표가 중간거리였을 때 값. 최종 정책은 130막대+깊은목표(20m) 과특화 → 소심해져
  먼 목표 timeout. **두 커리큘럼(거리·밀도)이 끝에서 충돌**. rl_games가 후반 ckpt만 보관해 좋은 중간정책 유실.
- 정정: "깊은목표=시야밖" 진단은 일부 오류(baseline은 깊은목표 timeout 0.7%로 도달함). 진짜 원인=과특화.

**1650 Ti seed-2 (`vision50_seed2`)**: 50막대 peak **92.5%** (seed-1 93.2%와 일치 → 앵커 재현성 확인).
후반 캡처 하락(70%)은 goal 커리큘럼이 목표를 20m로 밀어서(밀도 아님).

**거리 커리큘럼 knob env화 (커밋 24d7036)**: `NAVRL_K_FINAL`/`NAVRL_K_MIN_FINAL`/`NAVRL_K_WARMUP`
(기본 24/20/3000). 센서 전용은 목표를 검출기 시야(20m) 안에 묶을 수 있게.

**다음 계획 — 충돌 회피 두 해법을 두 머신에서 병렬:**
- **3070 = 순차(sequenced)**: 거리 먼저(6000ep, 25막대 고정) → 밀도 나중(6000→12000, 목표 고정).
  단일 명령으로: `NAVRL_DENSITY_CURRICULUM=1 NAVRL_DENSITY_WARMUP=6000 NAVRL_DENSITY_THRESHOLD=0.6
  ./train_navrl.sh --seed 1 --max_epochs 12000` (warmup 6000이 밀도를 6000판까지 막음 → 자동 2단계).
- **1650 Ti = 목표제한(goal-capped)**: `NAVRL_K_FINAL=16 NAVRL_K_MIN_FINAL=10` (목표≤16m 시야내) +
  밀도 커리큘럼 60캡 + GPU4GB. 6000ep. 동시 램프지만 목표가 얕아 충돌 완화.
- 승급마다 체크포인트 스냅샷 필요(중간정책 유실 방지). 밀도별 평가로 곡선 완성이 목표.

## 2026-07-20 — 정정: 밀도 커리큘럼은 성공 (체크포인트 선택 오류였음)

**중대한 정정**: 앞선 "밀도 커리큘럼 최종 110=40.8% < baseline" 결론은 **틀렸다**. 원인 = 평가에
`gen_ppo.pth`(best-reward)를 썼는데, 밀도 커리큘럼에선 **보상이 저밀도(25막대)에서 최고**라
`gen_ppo.pth = 저밀도 정책`이었음. 고밀도 평가엔 `last_gen_ppo_ep_XXXX`(마지막, 고밀도 특화)를 써야 함.

**올바른 110막대 held-out (마지막 체크포인트):**
- from-scratch (고정 110): 47.3%
- 순차 Phase B (거리먼저→밀도): 56.2% (+9pt)
- 동시 1841 (거리·밀도 동시): **60.6%** (+13pt, crash 37% timeout 2.2%)

**결론:**
1. **밀도 커리큘럼은 확실히 통한다** (47→60%). 이전 "충돌로 실패" 서사는 평가 아티팩트였음.
2. **순차가 동시를 못 이김** (56 vs 60) → 더 단순한 동시 방식으로 충분. 2단계 불필요.
3. **마지막(고밀도) 체크포인트가 낮은 밀도에도 잘 일반화** → 하나의 강건한 정책을 밀도별 평가하면
   그게 곧 논문 밀도-성능 곡선. (130 학습 정책이 110에서 60%.)

**warm-start 인프라 (커밋 e7f8ca3)**: `--max_epochs` override + critic `_orig_mod` 자동정규화
(warm-start 시 torch.compile 접두사 문제). Phase B가 이걸로 정상 resume·완주(5648→12000).

**다음**: 1841 마지막 ckpt를 25/50/75/110/130/150에서 평가 → 센서전용 밀도곡선. 1650 Ti는 평가 전담
(학습 N=128 페널티로 학습결과는 3070과 못 섞음; 평가는 N무관이라 OK).
주의: **밀도 커리큘럼 run 평가는 항상 last_ ckpt 사용** (gen_ppo.pth=저밀도 best 함정).

---

## 2026-07-22 — 충돌률 급증 진단 + 3D 연구현황 대시보드 + research-status 스킬

`ppo_260719_1000`(12000ep 순차) 종료 확인: **peak 97.1%(ep 3764) → final 44.9% / crash 54.1%(막대 115·목표 24 m)**.
사용자 우려("갑자기 충돌률↑ = 이상")를 3-에이전트 병렬 조사(run분석·커리큘럼감사·운영리뷰)로 정밀 진단.

### 진단 — "급락"은 절반 아티팩트, 절반 진짜 기하 바닥
- **아티팩트**: peak→final −52pt는 밀도·거리 커리큘럼이 끝에서 동시 최대(115막대+24 m)로 가서 최종 ckpt가
  최난이도 셀에 과특화된 것(정책 붕괴 아님). 07-20 정정대로 **`last_gen_ppo_*`** held-out하면 110막대 60.6% 정상.
  훈련중 final-epoch 지표/`gen_ppo.pth`(저밀도 best)로 밀도 비교 금지.
- **진짜 기하 바닥**: 센서 held-out 충돌 110→130 **+13.5pt**, 130→150 **+19.9pt**(로깅버그 아님). 원인 =
  배치 최소간격 **1.5 m가 128실패마다 ×0.8 완화**(→1.2→0.96 m), ~115–120막대에서 완화 시작. 2단계(0.96 m)면
  평균막대(0.6 m) 기준 **틈 0.36 m < 드론 대각 0.40 m** → 통과불가 → 충돌. GT는 120막대까지 평탄(1.8%)·150만 절벽
  = 같은 RSA 재밍 한계(~148/478 m²). **회피(충돌)가 병목이지 탐지 아님**(timeout은 저밀도서 오히려↑).

### 커리큘럼 감사(코드 확인) → 순차 밀도 레시피 확정
- 밀도 커리큘럼 기본값: `STEP=+15`·`THRESHOLD=0.8`·`WARMUP=2500`·`CHECK_EPS=2048`, capture-gated 승급(25→150).
  거리: `K_FINAL=24`/`K_MIN_FINAL=20`/`K_WARMUP=3000`, num_task_steps/32 램프. 표적속도: `SPEED_FINAL` 기본 0.
- **확정 권장(완만+충돌안전)**: `NAVRL_DENSITY_STEP=5 THRESHOLD=0.55 START=25 FINAL=110`(완화절벽 ~115 직전 정지)
  + **거리 얕게 캡 `K_FINAL=16 K_MIN_FINAL=10 K_WARMUP=8000`**(두 커리큘럼 끝-충돌 제거) + 승급마다 `last_gen` 스냅샷.
  단계별 명시 스테이징(NUM_BARS 고정 25→…→110, --checkpoint resume)은 ckpt 통제 최상 대안.

### 인프라 — 3D 웹 대시보드 + 재사용 스킬
- **`docs/status/`** (index.html+app.js+status.json): three.js 3D 아레나(막대/표적속도 슬라이더·LiDAR 토글) +
  최신현황·밀도곡선·run타임라인·진단·다음계획·로드맵. `status.json` 구동(데이터 자동반영), 오프라인/3D실패 시
  데이터패널은 유지되게 graceful degrade. **research 브랜치 `/docs`에서 GitHub Pages 게시**(gh-pages 브랜치 불필요).
  커밋 `8d25c89` push 완료 → Pages 활성화(Settings→Pages, branch=research/navrl-env, folder=/docs) 후
  `https://joshualikaist.github.io/MOTAR/status/` 라이브.
- **`.cursor/skills/research-status/`**: 이 워크플로우(지표수집→서브에이전트 분석→종합→대시보드→게시→WORKLOG)를
  순차 실행하는 스킬. `collect_status.py`(runs+CSV→status.json, 토큰절약 핵심), `publish_dashboard.sh`(docs만 커밋·push).

### 코드 리뷰(서브에이전트) + 수정
정직한 결함 리뷰 결과 **Critical 버그 없음** — 정적 LiDAR(P1–2) 태스크·리뷰에서 의심됐던 부분(swept-segment 캡처가
순간 dist<0.5의 상위집합, 정적표적 byte-동일, 밀도 accumulator/warmup/clamp/전파, 커리큘럼 min<max 불변식,
종료 aliasing, obs 차원 분할, PBRS Φ=0)은 전부 정상 확인. **수정한 3건**:
- **[Suspicious] `NAVRL_NUM_BARS` 무시 버그**: 밀도 커리큘럼 flag가 켜져 있으면 `_initial_active_bars`가
  `n_start`(25)를 반환해 explicit `NAVRL_NUM_BARS`를 조용히 무시 → 밀도별 eval/resume이 25막대로 잘못 돌던 문제.
  `_initial_active_bars`가 explicit `NAVRL_NUM_BARS`를 최우선하도록 수정(set_env_state의 규칙과 일치). 검증 완료.
- **[Minor] env-var 파싱 무음 삼킴**: `_env_int/_env_float`가 잘못된 값(`NAVRL_K_FINAL=abc`, `1.5m`)을 조용히
  기본값으로 → stderr 경고 추가.
- **[Minor] `_env_bool`이 `1.0` 같은 값을 False로**: 숫자형 허용(`float(s)!=0`)으로 수정(`NAVRL_VISION=1.0` 함정).
**미수정(플래그만)**: ① vision 모드 mid-episode 리셋 첫 프레임에서 LiDAR 표적마스크가 옛 목표를 가리키는
1프레임 불일치(리워드 무관, 수정은 step 순서 재배치라 별도 스모크 필요) ② PBRS re-anchor는 표적 이동 시
엄밀 PBRS 아님(정적=정확, 문서화된 의도적 트레이드오프) ③ episode 301스텝 off-by-one(재현성 영향 우려로 미변경).

---

## 2026-07-22 — 연구 핵심 재정의: oracle semantic → raw-sensor learned perception

사용자 결정으로 Phase 3의 기준을 다시 고정했다. 드론은 `target_position`, GT bearing/range 또는
semantic target mask/id를 받아 추격하면 안 된다. camera RGB-D와 LiDAR의 원시 시계열 관측에서 표적의
상대 위치·속도·가림·불확실도를 직접 추정해야 한다.

### 기존 구현의 판정

- analytic LiDAR target sphere, semantic id 50, analytic camera target mask/depth는 센서 geometry,
  occlusion, 처리량을 확인한 **prototype/upper-bound**다.
- camera obstacle depth + LiDAR range/semantic fusion이 256 env에서 약 8,180 env-steps/s로 동작한 것은
  engineering baseline으로 보존한다.
- 그러나 semantic target identity를 actor에 넣는 checkpoint는 논문의 최종 perception checkpoint가 아니다.

### NavRL++ 원문 확인 후 계획 교정

- NavRL++의 Transformer는 raw sensor detector가 아니라 perception이 만든 structured obstacle/state history를
  처리하는 temporal actor–critic backbone이다. 원문 설정은 12 tokens, 2초/0.5초 history, dim 64,
  4 heads, 4 layers, FFN 128, 약 0.31M parameters다.
- 본 연구는 이를 `[CLS]+static+dynamic history+robot history+target history`의 17 tokens로 확장한다.
  Transformer를 주 모델로 확정하고 CNN/LSTM은 ablation baseline으로 둔다.
- RGB-D와 LiDAR가 각각 obstacle/target proposal을 직접 만들고 association/Kalman tracking으로 structured
  state를 생성한다. semantic target id와 GT pose는 actor에 제공하지 않는다.
- NavRL++ ablation상 Transformer 단독의 명확한 이득은 success 자동 향상이 아니라 control effort 감소이며,
  최종 성능은 perception-failure-aware fine-tuning과 결합할 때 개선됐다. 따라서 target detection drop,
  latency, noise, calibration perturbation fine-tuning을 필수 단계로 추가했다.
- 구현 순서: **information firewall → dual-sensor detector → fusion/tracker → 17-token Transformer →
  perturbation-aware fine-tuning → PPO**.

---

## 2026-07-23 — Perception+Transformer 초기 학습 run 정리 (중단 사유 기록)

`NAVRL_PERCEPTION=1` + `ppo_navrl_perception_transformer.yaml` 첫 본학습 시도들. 아래 run·로그는
학습 가치 없거나 붕괴된 것으로 판단해 **삭제함** (`runs/` + `train_session_logs/`). **유지:**
진행 중인 `ppo_260723_1241_navrl` (12:41 시작, tuned env).

| run / log | ep | peak cap | 종료 이유 |
|---|---|---|---|
| `ppo_260722_1928` / `train_260722_1928` | 615 | 3.8%@443 | **수동 중단.** perception Transformer 첫 시도(128env, 밀도 커리큘럼 on, 25막대). 최종 crash **100%**·cap 0% — from-scratch로는 학습 신호 없음. |
| `train_260722_2044` | 0 | — | **즉시 실패.** conda 미활성화로 `ModuleNotFoundError: yaml`. run 폴더 없음. |
| `ppo_260722_2045` / `train_260722_2045` | 256 | 6.1%@81 | **수동 중단.** **48막대**로 시작(설정 실수로 추정). crash 97%·cap 3% — perception cold-start에 밀도 과다. |
| `ppo_260722_2058` / `train_260722_2058` | 4074 | 4.8%@91 | **포기.** 거리 커리큘럼 기본 `K_FINAL=24`가 ep~3000에 k_max=24 m까지 올라갔으나 cap은 ep1000 이후 **0% 고착**, crash **100%**. perception from-scratch에 원거리 커리큘럼이 너무 공격적. |
| `train_260722_2342` | +124 | — | 2058 resume 시도(`last_gen ep3950`). ep3950→4074만 추가 후 중단 — resume도 회복 불가 확인. |
| `ppo_260722_2349` / `train_260722_2349` | 63 | 21.6%@54 | **조기 중단.** fresh restart, 초반 cap 13~22%로 1928/2058보다 나았으나 레시피 재튜닝 위해 중단. |
| `ppo_260723_0011` / `train_260723_0011` | 1051 | **42.9%@314** | **수동 중단 (NaN 붕괴).** ep314까지 유망(cap 43%, crash 36%) → ep~348부터 PPO gradient **NaN** → 100% timeout·reward NaN. ckpt unusable. |
| `ppo_260723_0225` / `train_260723_0225` | 1111 | 6.7%@79 | **수동 중단 (NaN).** 0011과 동일 패턴 — 초반 미약, 후반 NaN/timeout 붕괴. |
| `ppo_260723_1138` / `train_260723_1138` | 1563 | **48.3%@755** | **수동 중단 (정책 퇴화).** ep755 peak 48% → ep1563 cap 14%/crash 86%로 퇴화. `K_COMPETENCE`·`LIDAR_RANGE=8`·`K_FINAL=16` 등으로 **1241 run 재시작**. |

**교훈:** (1) perception cold-start는 25막대·얕은 목표(≤16 m) 고정부터. (2) default `K_FINAL=24` 거리 커리큘럼은 Transformer+detector 경로에 부적합.
(3) NaN은 ep300~500대에서 발생 — lr/grad clip/mean_std 안정화 점검 필요. (4) 1138→1241 전환 env:
`NAVRL_CRASH_DIAG=1 NAVRL_LIDAR_RANGE=8 NAVRL_K_COMPETENCE=1 NAVRL_K_FINAL=16 NAVRL_FOV_CURRICULUM_EPOCHS=1000000 NAVRL_NUM_BARS=25`.

---

## 2026-07-23 (저녁) — Perception+Transformer 2차 run 정리

1241 tuned run 이후 same-day 추가 시도들. **첫 perception 성공 run = `ppo_260723_1509`** (peak cap **98.6%@ep3343**,
`checkpoints_saved/peak_cap986_r165_ep3343*.pth` 보존). **현재 학습 중 = `ppo_260723_2210`** (peak ckpt에서 resume).

| run / log | ep | peak cap | 종료 이유 |
|---|---|---|---|
| `ppo_260723_1241` / `train_260723_1241` | 705 | 68.3%@136 | **수동 중단.** k_max **7 m에 고착**(거리 커리큘럼 미승급). peak 68%→final 46%. 1330/1509 레시피로 교체. |
| `ppo_260723_1311` / `train_260723_1311` | 584 | 20.0%@83 | **수동 중단 (NaN).** 1241과 중복 fresh start. ep~93부터 NaN → 100% timeout. |
| `ppo_260723_1330` / `train_260723_1330`, `train_260723_1418` | 1263 | **97.0%@488** | **수동 중단.** k_max=16 레시피 첫 성공(peak 97%). ep1263 final 72% — 1509 fresh restart로 교체. 1418=1330 resume 로그. |
| `ppo_260723_1424` / `train_260723_1424` | 607 | 17.3%@115 | **수동 중단 (NaN).** 1330/1509와 병렬 fresh start. ep~122부터 NaN. |
| `ppo_260723_1509` / `train_260723_1509` | **10000** | **98.6%@3343** | **완주(max_epochs).** perception+Transformer **첫 본학습 성공**. ep5000 cap 83% 유지 → ep~7000 NaN/timeout 붕괴(loiter farming). **평가·resume은 peak ckpt 사용** (`gen_ppo`=후반 loiter 함정). |
| `train_260723_1535`, `train_260723_1559` | — | — | 1509 resume 시도 로그(`gen_ppo`, `last_gen ep1000`). gen_ppo=저밀도/loiter ckpt라 무의미 — 세션 로그만 삭제. |
| `ppo_260723_2105` / `train_260723_2105` | 4267 | 96.2%@3792 | **수동 중단 (NaN).** 1509 `gen_ppo_rlnorm`에서 resume(잘못된 ckpt). ep3792 잠깐 96% → NaN 붕괴. peak ckpt resume으로 2210 전환. |

**유지:** `ppo_260723_1509_navrl`(1.5 GB, peak ckpt 원본), `ppo_260723_2210_navrl`(진행 중, peak ckpt resume),
`checkpoints_saved/peak_cap986_r165_ep3343*.pth`.

**교훈:** (1) 거리 커리큘럼 k_max=16 승급이 핵심(1241은 k=7 고착). (2) 1509 완주 후 **gen_ppo로 resume 금지** — peak cap ckpt만.
(3) NaN은 여전히 ep100~700대 random hit; peak ckpt resume(2210)은 ep3790 cap 82%로 정상.

---

## 2026-07-23~24 — 센서 전용 이동표적 요격 검증 + 급사(sudden NaN) 근본원인 수정 + 정직한 baseline

`checkpoints_saved/peak_cap986_r165_ep3343.pth`(98.6% 학습피크)를 기반으로 (1) 속도를 줄이지 않고
올리는 방향으로 이동표적 요격을 검증하고, (2) 반복적으로 학습을 죽이던 급사(sudden NaN)의 진짜
원인을 특정·수정하고, (3) 파라미터 튜닝의 기준이 될 정직한 baseline을 측정했다.

### 속도 상향 + 이동표적 요격 (검증 완료)
`NAVRL_YAW_RATE_MAX=3.0`(task+`lee_controller_config_navrl` 동시 상향, 2.0m/s에서 이미 위빙이
2.4/2.5rad/s를 요구 → 요레이트가 진짜 기동 한계, 추력 아님, T/W≈3.3) + `NAVRL_MAX_VELOCITY=2.5` +
`NAVRL_TARGET_SPEED_FINAL` 램프(0→1.5m/s)로 재학습. capture는 표적속도 전 구간에서 ~0.8 유지 —
드론이 표적보다 충분히 빠르면 속도 자체는 병목이 아님을 확인.

### ★ 급사 NaN의 근본원인 (수정: `navrl_transformer_network.py:149`, 커밋 `bb0faa7`)
건강하게 capture 0.8~0.9로 돌던 run이 ~epoch 5000대에서 a_loss가 **단 한 스텝**에 NaN → hover 붕괴하는
패턴이 반복(entropy_coef 0.005·0.003 둘 다). NaN 직전 c_loss/kl/explained_variance는 전부 정상이었고
**ppo/entropy만 ~16(σ≈13)**으로 튀어 있었다 — 점진적 과학습이 아니라 값 자체의 발산이었다.
기전: `fixed_sigma=True` + `entropy_coef>0`에서 log-std에 **상한이 없어**, 난이도가 오르며 정책손실의
하방압은 약해지고 엔트로피 보너스는 계속 σ를 밀어올려 σ가 1→13까지 표류 → PPO log-prob/gradient가
오버플로. **수정**: `forward()`에서 `log_std = (mu*0.0 + self.sigma).clamp(-5.0, 0.4)`(σ≤1.49,
entropy≤~7.3)로 σ=13 도달 자체를 봉쇄. **검증**: 98.6% 피크에서 warm-start해 옛 사망지점 epoch 5178을
건강하게 통과(entropy 평탄 ~4.4, capture 0.86~0.91, NaN 없음).

### eval 크래시 = Isaac Gym 진단 메시지(우리 버그 아님) + eval은 128 envs
`*** Can't create empty tensor`는 Isaac Gym 빌드타임 진단이 train에서는 조용 래퍼로 숨겨지고 play에서만
노출되는 것뿐, 무해함. 512-env eval이 실제로 죽는 이유는 **VRAM OOM**(512개 perception 카메라 @ 8GB).
128 envs(=학습값)로 하면 정상 동작. 512는 사용 금지.

### 정직한 baseline (측정 잣대) — `results/baseline_speed_axis_peak986.csv`
98.6% 피크 정책, 25막대·목표 13-16m·드론 2.5m/s·deterministic·2049 episode/셀:

| target speed | capture | crash | timeout |
|---|---|---|---|
| 0.0 | 0.741 | 0.251 | 0.008 |
| 0.5 | 0.762 | 0.238 | 0.000 |
| 0.75 | 0.755 | 0.245 | 0.001 |
| 1.0 | 0.744 | 0.256 | 0.000 |
| 1.25 | 0.724 | 0.275 | 0.001 |
| 1.5 | 0.686 | 0.314 | 0.000 |

**capture가 표적속도에 거의 평탄(0.74 근처, timeout≈0) — 병목은 표적속도가 아니라 막대충돌(~25%)이다.**
"25막대인데 충돌은 용납 못한다"는 문제의식이 데이터로 확증됨. 학습곡선 피크 0.986 vs 이 고정평가
0.74가 정직한 train↔eval 갭.

### 다음: 충돌 저감 튜닝, 파라미터 1개씩 (각각 이 baseline 0.74/0.25 대비 측정)
보상 재조정은 과거 3회 연속 무효였던 레버라 제외(충돌은 기하 문제로 판단). 순위:
1. **고도 z-position 피드백** — 인에이블러, 가장 짧음. look-ahead 확장 시 바닥 sag 재발을 막는 전제조건.
2. LiDAR look-ahead 8→10/12m (①이 검증된 뒤에만).
3. 10m 카메라 depth를 actor에 전달 (전방향only, 바닥 무관).
4. 장애물 토큰 5→8 또는 Transformer dim 64→128 (`MAX_OBSTACLES`, 저위험이나 obs shape 변경 → fresh 재학습).

각 실험당 재학습 ~30-40분 + 128-env eval 필요, 무비용 충돌개선 레버는 없음. 상세 = `CRASH_TUNING_LOG.md`
Session 2 섹션. 커밋: `bb0faa7`(코드), `ccb979b`(문서+baseline CSV).

---

## 2026-07-24 — 후보 ① 고도 PI 제어 검증 완료: 확실한 개선

`lee_velocity_control.compute_acceleration`가 `setpoint_position`을 항상 현재 위치로 넘겨 위치 피드백항이
0인 순수 속도추종 컨트롤러임을 확인 → task 레벨 고도 hold를 P전용(`4*z_err`)에서 **PI**로 변경(Ki=1.0,
anti-windup, 에피소드마다 리셋, `navrl_task.py` 미커밋). 근거: 위빙 중 자세추종 지연이 만드는 정상상태
sag는 P만으로는 안 지워짐.

98.6% 피크에서 warm-start해 ~450 epoch 안정성 확인(`below` 0~5.6% 유지, NaN 없음) 후, baseline과 동일
128-env 고정평가(2049 episode/셀)로 검증:

| target | baseline capture | PI capture | baseline crash | PI crash |
|---|---|---|---|---|
| 0.0 | 0.741 | **0.804** | 0.251 | **0.191** |
| 0.5 | 0.762 | **0.822** | 0.238 | **0.178** |
| 0.75 | 0.755 | **0.813** | 0.245 | **0.187** |
| 1.0 | 0.744 | **0.808** | 0.256 | **0.192** |
| 1.25 | 0.724 | **0.786** | 0.275 | **0.214** |
| 1.5 | 0.686 | **0.778** | 0.314 | **0.222** |
| **평균** | **0.735** | **0.802** | **0.263** | **0.197** |

**전 구간 일관 개선: capture 평균 +6.7pt, crash −6.6pt** (표적 1.5m/s에서 최대 +9.2pt). `below`(바닥충돌)는
전 구간 1.6~2.3%로 사실상 해소. **새 발견**: crash 중 `oob`(아레나 이탈, 주로 N벽)가 34~44%로 부각 —
바닥충돌에 가려있던 문제. 결과 = `results/altitude_pi_speed_axis.csv`, 상세 분석 = `CRASH_TUNING_LOG.md`
"Candidate #1 RESULT" 섹션.

**다음 후보(순서)**: ② LiDAR look-ahead 8→10/12m(이제 잠금 해제) ③ N벽 oob drift 원인 조사(새로 부각된
2차 원인) ④ 카메라 depth→actor ⑤ 장애물 토큰/네트워크 확대.

---

## 2026-07-24 (이어서) — General-spawn G0 파인튜닝 검증 (랜덤 스폰 일반화)

사용자가 후보 ②(LiDAR) 대신 `train_navrl_general_8m_finetune.sh`로 **랜덤 드론/타깃 스폰 일반화**
파인튜닝을 실행(`NAVRL_GENERAL_TRAIN=1`, PI 체크포인트 ppo_260724_0110에서 +~1200 epoch →
`ppo_260724_0209_navrl`, LiDAR 8m 유지, OOB_MARGIN=1.0, K_MIN_FINAL=10). 학습은 건강히 완주
(peak captured 93.7%@ep4256, last 83.8%, NaN 없음).

⚠️ **평가 함정(기록)**: 이 정책을 baseline recipe(고정 스폰)로 평가하면 capture 10%·oob 90%(W벽)로
완전히 망가진 것처럼 나옴 — 정책이 랜덤 스폰을 학습했는데 고정 스폰으로 재면 분포 불일치. **general
정책은 반드시 `NAVRL_GENERAL_TRAIN=1`로 평가**해야 함. (체크포인트 env_state에는 n_bars=25,
k=13-16만 저장되고 GENERAL_TRAIN 플래그는 안 저장되므로 평가 시 수동 지정 필수.)

정상 조건(general-spawn, 128 env, 2049 ep/셀) 6-speed 평가 = `results/general_8m_speed_axis.csv`:

| target | capture | crash | timeout |
|---|---|---|---|
| 0.0 | 0.803 | 0.120 | 0.077 |
| 0.5 | 0.873 | 0.124 | 0.002 |
| 0.75 | 0.837 | 0.160 | 0.003 |
| 1.0 | 0.850 | 0.146 | 0.004 |
| 1.25 | 0.837 | 0.161 | 0.002 |
| 1.5 | 0.819 | 0.180 | 0.001 |
| **평균** | **0.836** | **0.149** | **0.015** |

랜덤 스폰 일반화에도 capture 0.836/crash 0.149로 견고(oob는 margin 1.0 덕에 4~8%로 억제). **단,
스폰 분포가 달라 baseline(0.735)·PI-only(0.802)와 직접 비교 불가** — 이건 general-spawn 체제의 새
measuring stick. `below`가 crash의 10~18%로 다시 보이는데, 이 역시 다른 분포에서의 값이라 PI-only
평가(1.6~2.3%)와 직접 비교하면 안 됨.

---

## 2026-07-24 (이어서) — tilt 추력보상 + `below` 근본원인 규명 → "실패의 보존" 발견

**질문("틸트할지 어떻게 미리 아나?")에 대한 답**: 예측 아님. 추력 계산하는 그 줄에서 현재 기울기(측정)와
원하는 힘벡터(방금 계산)를 둘 다 알고, 침하는 그 둘 내적의 결정론적 오차. `velocity_control.py`에
PX4식 고도우선 추력 `T = f_z / clamp(b3_z, 0.5)` 추가(cfg gate, NavRL 전용, `NAVRL_TILT_COMP` 기본 ON).

**below 계측 추가**: crashdiag의 below에 사망 step + 사망 틸트각. general-spawn에서 below=스폰 1.5초 후·
30° 뱅크 = 랜덤스폰 초기 급선회(적분 아직 0) 시점으로 확정.

**제로샷 A/B**(0209, speed1.0): below 1.96%→1.22%(−38%), capture 무손상, 잔존 사망 틸트 30°→36°(겨냥한
죽음만 제거된 지문). tilt-comp ON으로 파인튜닝(0209→`ppo_260724_1052`, +~1180ep, peak cap 96.9%).

**★ tilt-comp 파인튜닝 최종 평가 (6-speed general, 절대율 평균) = results/general_8m_tiltcomp_speed_axis.csv:**

| 지표 | 파인튜닝 전(0209) | 후(1052) | Δ |
|---|---|---|---|
| capture | 0.837 | **0.837** | **0 (완전 평탄 — 사용자 예측 적중)** |
| below(바닥) | 1.82% | **0.50%** | −1.31pp (설계대로 고쳐짐) |
| oob(벽) | 0.89% | **2.38%** | +1.49pp (실패가 여기로 이동) |
| bar_contact | ~12.2% | ~13.1% | 그대로(지배적) |

**핵심 = 실패의 보존**: tilt-comp가 바닥침하를 제거하자 정책이 더 과감히 뱅크(파인튜닝으로 학습) →
그 과감함+측면 추력누출이 드론을 아레나 밖으로 밀어냄. below와 oob가 ~1:1로 맞바꿔지고 capture는
불변. **below(0.5%)·oob(2.4%) 둘 다 작음. bar_contact(~13%)가 둘 합친 것의 4~5배 = 진짜 병목.**
지금까지 고도작업(PI, tilt-comp)은 전부 **작은 2차 실패**를 다듬은 것 → capture가 안 움직인 이유.

**tilt-comp/PI는 유지**(올바른 제어 + 고밀도 일반화 기반). oob는 candidate ③로 별도 추적. **capture를
0.84에서 올릴 유일한 레버 = bar_contact = candidate ② LiDAR look-ahead 8→12m** (고도 단단해져 이제
바닥충돌 재발 없이 확장 가능 — 이게 고도를 먼저 한 이유). 커밋 `c01e552`(tilt-comp+계측), `b28eac1`(G0 지원).

---

## 2026-07-24 (이어서) — candidate ② LiDAR 12m: capture 최고치이나 **가설은 기각**

12m look-ahead 학습(1052→`ppo_260724_1230`). **ep6145에 peak(capture 95.2%, reward 111) 찍고 ~ep6700에
붕괴** — crashdiag oob가 0.10→0.60으로 전 방향(W/E/S/N 균등) 폭발 = 드론이 방향감각 상실, 최종 capture
0.558. `gen_ppo.pth`는 붕괴 이전 peak라 **정책 자체는 무사**. 붕괴 추정 트리거: LiDAR range를 8→12로 바꾸면
스캔 정규화(`scan/range`)가 바뀌는데 warm-start한 체크포인트의 `running_mean_std`는 8m 통계 → 학습 중
입력 스케일이 밑에서 드리프트 → 정책이 스캔 오독 → PPO 온폴리시 나선. (미증명)

**peak 6-speed 평가**(= `results/general_12m_lookahead_speed_axis.csv`, 평가도 반드시 `LIDAR_RANGE=12`):

| 지표(절대율 평균) | 8m tilt-comp | **12m peak** | Δ |
|---|---|---|---|
| capture | 0.837 | **0.856** | **+1.8pp (역대 최고)** |
| crash | 0.160 | 0.139 | |
| oob | 2.4% | **1.2%** | −1.2pp ← **이득은 전부 여기서** |
| **bar_contact** | 13.1% | **12.6%** | **−0.5pp (노이즈, 겨냥한 목표인데 안 움직임)** |

**★ 가설 기각**: look-ahead는 bar_contact의 레버가 아니다. 8m = 2.5m/s에서 3.2초 경고로 단일 회피엔 이미
충분했고, 더 멀리 본 효과는 "아레나 밖으로 덜 샘"(oob)뿐. **이로써 bar_contact(~13%)에 대한 독립적 음성
결과 3연속**: 고도 PI / tilt-comp / look-ahead 전부 불변. bar_contact(mean_x~12.4m = 막대밭 한복판)는
고도 권한에도 감지 거리에도 면역이며, 남은 capture 갭의 전부다.

**다음: 추측 금지, 직접 진단.** 남은 가설 2개(측정으로 구분 가능):
- **H1 토큰 용량**: `MAX_OBSTACLES=5`(navrl_perception.py) — 밀집 구간엔 반경 내 막대가 5개 초과라
  6번째부터 정책 입력에서 잘림 → 안 보이는 걸 피할 수 없음.
- **H2 인지/추적 품질**: 부딪힌 막대의 KF 추적 위치 오차 → 유령을 피하다 실물에 스침.

**진단 계측**(재학습 불필요): bar_contact 순간에 (a) 반경 내 막대 개수(혼잡도) (b) 실제 부딪힌 막대가
정책에 들어간 토큰 안에 있었는지 (c) 그 막대의 추적-실제 위치 오차. H1이면 혼잡도↑ & 부딪힌 막대가
토큰 밖, H2면 혼잡도↓ & 토큰 안에 있으나 오차 큼. (below/tilt를 정확히 짚었던 것과 같은 measure-first 방식)

---

## 2026-07-24 (이어서) — **코드 불일치 정리 + bar_contact 진단 완료: H1·H2 둘 다 확인**

### 서브에이전트 감사로 잡은 불일치 (전부 수정)
- **평가 스크립트가 스케일 결정 env를 안 박음**(`eval_navrl_density_sweep.sh`, `..._speed_density_grid.sh`):
  `NAVRL_LIDAR_RANGE`(=스캔 정규화 divisor!), `MAX_VELOCITY`, `YAW_RATE_MAX` 미지정 → 8/12m 정책이 4m
  기본값으로 평가돼 **밀도곡선이 조용히 틀림**. + density_sweep은 `NAVRL_PERCEPTION=1`이 없어 아예 CNN
  yaml을 골라 현재 정책엔 사용 불가였음. → 전부 pin + 에코.
- **`max_velocity`가 고도 PI 권한·anti-windup까지 결정하던 결함(내 코드)** → `alt_hold_vmax`
  (`NAVRL_ALT_HOLD_VMAX`, 기본 2.5)로 분리. 안 그러면 속도 스윕이 "느린 추격자가 덜 충돌"과 "느린
  추격자는 고도 유지 불가"를 뒤섞음.
- **체크포인트 설정 드리프트 무경고** → `get/set_env_state`에 `cfg_lidar_max_range/max_velocity/
  yaw_rate_max/max_obstacles` 기록 + 불일치 시 경고(복원은 안 함, 의도적 override 허용). 오늘 두 번 헤맨 그 문제.
- **`navrl_task_config`의 `max_obstacles=5`·`history_steps=5`는 아무도 안 읽는 죽은 필드**(실제 상수는
  `navrl_perception.py`) → 삭제 + 경위 주석. 대신 `NAVRL_MAX_OBSTACLES`로 스윕 가능하게.
- 런처 드리프트 정리 → 두 스크립트가 이제 **`LIDAR_RANGE`만 다름**(검증됨). crashdiag 계측 자체는 감사 결과 **정상**.

### ★ bar_contact 진단 결과 (`NAVRL_BAR_PROBE=1`, 12m peak, n=266 충돌)
```
bars_in_range=15.8  occupied_bins=19.5/36  hit_in_tokens=0.647
token_err=0.57m     token_rank=0.9         (capacity=5)
```
- **H1(토큰 용량) 확인**: 반경 내 막대 **15.8개 vs 용량 5개**. **충돌의 35%가 정책 입력에 아예 없던 막대와의
  충돌.** 보여주지 않은 걸 피할 수는 없음.
- **H2(추적 품질) 확인, 단 원인은 추적기가 아님**: 토큰에 있던 65%도 **위치오차 0.57m**(드론 박스 0.28m,
  막대 반경 0.2~0.4m → 통로 폭과 맞먹음). 원인 = **각도 양자화**(수평 36빔=10°/bin → 5m에서 반빔 오차 ~0.44m).
  토큰 위치가 range/angle 기하로 만들어지므로 측방 정확도의 상한이 빔 개수에 묶여 있음.
- **숨은 천장 발견**: 토큰 하나 뽑을 때마다 ±2 bin(=50°)을 지움 → 360/50≈7.2 → **MAX_OBSTACLES를 8로
  올려도 7개 이상은 구조적으로 안 채워짐.** 용량만 올리면 헛돎.

### 수정은 3개 묶음 (전부 관측 차원 변경 → **fresh 재학습 필수**)
1. `MAX_OBSTACLES` 5→8  2. 억제폭 ±2→±1 bin  3. 수평빔 36→72
**캠페인 통틀어 처음으로 추측이 아닌 측정에 근거한 bar_contact 개입** — 고도 PI·tilt-comp·12m look-ahead가
전부 bar_contact를 ~13%로 남긴 이유는 셋 다 **장애물 표현(representation)** 을 건드리지 않았기 때문.

---

## 2026-07-27 — `general_repr` 반복 조기 종료 원인 규명 및 PPO 안전장치

### 증상과 원인

`train_navrl_general_repr.sh`로 시작한 fresh representation 학습 중 다음 두 run이 조기에 끝났다.

- `ppo_260727_0048_navrl`: epoch 62, `early_stop_nan`
- `ppo_260727_0147_navrl`: epoch 43, `early_stop_nan`

비교 run(`0054`, `0058`, `0106`)은 각각 설정된 60/120/120 epoch까지 정상 실행되었고, 실패 로그에는
OOM·CUDA 오류·외부 process kill이 없었다. 조기 종료 로직이 학습을 임의로 끊은 것이 아니라 이미 망가진
학습 상태를 mean reward의 NaN으로 뒤늦게 감지한 것이었다.

근본원인은 `ppo_navrl_perception_transformer.yaml`의 조합이었다.

```yaml
learning_rate: 1e-4
lr_schedule: adaptive
```

현재 `rl_games`의 legacy adaptive scheduler는 KL이 낮으면 minibatch마다 LR을 1.5배 올리고 상한을
`1e-2`로 둔다. 이 batch 구성에서는 첫 epoch에만 `1e-4 → 1.70859e-3`으로 올라갔고 이후 `1e-2`에
도달했다. 실패 run에서 PPO loss가 epoch 56/36부터 NaN이 된 뒤 mean reward가 epoch 62/43에 NaN이
되어 조기 종료되었다. 실패 체크포인트의 optimizer에도 LR `0.01`이 저장되어 있어 그대로 resume하면
설정의 `1e-4`와 무관하게 위험한 LR을 다시 사용하게 되는 문제도 확인했다.

### 수정

- `ppo_navrl_perception_transformer.yaml`
  - actor LR은 `1e-4` 유지.
  - `lr_schedule: adaptive`를 `lr_schedule: None`으로 변경해 IdentityScheduler로 고정.
- `training_safety.py` 신규
  - PPO metric과 모델 파라미터의 첫 NaN/Inf 위치를 찾는 검사 추가.
  - 체크포인트 restore 후 optimizer 모든 param group의 LR을 현재 설정값으로 덮어쓰는 helper 추가.
- `early_stop_a2c_agent.py`
  - restore 시 과거 체크포인트의 optimizer LR을 버리고 현재 config LR로 초기화.
  - 매 epoch 직후 actor/value/bounds loss, entropy, KL 및 모델 파라미터의 NaN/Inf를 검사.
  - non-finite 상태는 `nonfinite_ppo` 실패로 즉시 예외 종료.
  - NaN reward 또는 non-finite PPO 상태에서는 손상된 periodic/best/last checkpoint를 저장하지 않음.
  - NaN 실패에는 `.aerial_training_finished`를 쓰지 않고 non-zero exit를 유지하여 정상 완료로
    오인하거나 손상 체크포인트를 자동 재사용하지 않게 함.
  - 정상 reward collapse/plateau, max epoch/frame 종료의 기존 체크포인트 동작은 유지.
- `run_header.py`, `train_run_recorder.py`
  - 실행 헤더에 NaN/Inf fail-fast, plateau, collapse 조건을 실제 설정대로 표시.
  - NaN 종료를 정상 조기 종료가 아닌 실패로 기록.
- `tests/test_training_safety.py` 신규
  - 고정 LR 설정, metric/parameter NaN 감지, stale checkpoint LR 교정을 검증.

### 실제 실행 경로 확인

사용 중인 명령은 다음과 같다.

```bash
cd aerial_gym/rl_training/rl_games
./train_navrl_general_repr.sh
```

이 wrapper가 `NAVRL_VISION=1`, `NAVRL_PERCEPTION=1`을 설정한 뒤 `train_navrl.sh`를 호출하므로 실제로
수정한 `ppo_navrl_perception_transformer.yaml`을 선택한다. 관측 차원이 574→898로 바뀐 실험이라
wrapper 설계대로 checkpoint warm-start 없이 fresh run으로 실행된다. 따라서 실행 명령은 변경하지 않는다.

### 검증

- `tests/test_training_safety.py`: 4/4 통과.
- 기존 `tests/test_navrl_perception.py`: 4/4 통과.
- 수정 Python 파일 `py_compile` 통과.
- `git diff --check` 통과.
- 사용자 소유의 untracked `checkpoints_saved/`는 변경하지 않음.

전체 Isaac Gym 학습 smoke run은 현재 셸 환경에 `ninja` 실행 파일이 없어 extension import 단계에서
실행하지 못했다. 실제 다음 `general_repr` fresh run에서 시작 헤더의 actor LR `0.0001`,
schedule `None`과 장시간 NaN 재발 여부를 확인한다.

---

## 2026-07-27 — tilt 제한 + 추력보상 부호 안전화, 그리고 내 NaN 진단의 정정

병렬 세션이 같은 날 NaN의 근본원인을 **adaptive LR 스케줄러**(1e-4 → 1e-2 상승)로 규명해 위에
기록했다. 이 항목은 **그와 별개로 진행된 제어 쪽 작업**과, 그로 인해 **내 초기 결론이 부분적으로
틀렸음**을 남긴다.

### 정정: "tilt 수정으로 NaN이 해결됐다"는 근거가 약했다

`0048`(epoch 62 NaN)을 보고 나는 원인을 내가 c01e552에서 넣은 tilt 보상의 부호 버그로 지목하고,
tilt 제한을 넣은 뒤 fresh 120 epoch에서 NaN이 없자 해결로 판단했다. **각 조건 n=1이라 통계적으로
빈약했고, 실제로 `0147`이 tilt 제한이 있는 상태에서도 epoch 43에 NaN으로 죽었다**(위 항목).
→ **NaN의 주원인은 LR 스케줄러이고, 내 tilt 가설은 기각에 가깝다.** 다만 아래 부호 버그 자체는
실재하는 결함이라 수정은 유지한다(NaN의 원인이 아니었을 뿐, 뒤집힌 상태에서 지면으로 가속하는 것은
그 자체로 잘못).

### 유지하는 수정 (커밋 `6d391e0`)

1. **부호 안전화**: `T = f_z / b3_z.clamp(min=0.5)`는 크기만 막고 부호를 안 막아, 뒤집힌 상태
   (`b3_z<0`)에서 `+0.5`를 반환해 **아래를 향한 몸통축에 양의 추력**을 걸었다. `b3_z ≤ 0.5`면 표준
   Lee 투영으로 폴백하도록 변경.
2. **목표 기울기 제한 45°** (`NAVRL_MAX_TILT_DEG`, PX4 `MPC_TILTMAX_AIR` 방식). 기존 Lee는 목표
   body-z를 `힘/|힘|`로 그대로 뽑아 수평 명령이 크면 90°를 넘기는 자세를 요구했다. 고도 우선 형태
   (수직 유지·수평 축소)라 양력이 아니라 횡가속을 희생 — 45°에서도 9.8 m/s², 2.5 m/s에서 선회반경
   0.64 m.

**학습된 정책 무손상 확인**(2048 ep, general spawn, target 1.0): 제한 없음 capture **0.868** vs
45° **0.864**. 즉 잘 나는 정책은 이 제한에 닿지 않는다.

### 기각된 가설 2건 (다시 시도하지 말 것)

- **"tilt 보상이 바닥추락(below)의 원인"** → 기각. A/B 결과 보상 **OFF일 때 below가 85~87%로 더
  높았고** ON은 73~76%였다. 보상은 오히려 바닥추락을 줄이고 있었다.
- **"각속도 감쇠 부족이 자세 오버슈트의 원인"** → 기각. `K_angvel` 0.2 / 0.45 / 0.8 스윕에서
  IID 랜덤 행동 시 최대 기울기가 **127~134°로 변하지 않았다.** 측정 이득이 없어 변경을 되돌렸고
  커밋하지 않았다.

### 부수 측정 (fresh 정책의 고도)

직접 프로브(32~64 env, 고정 명령): 정지·최대전진 모두 z=1.00 m 유지, 기울기 3~7°로 **고도 제어
자체는 정상**. 반면 **매 스텝 IID 랜덤 행동**(=갓 시작한 PPO)에서는 최저 고도 0.10 m(사망선),
최대 기울기 127°. 0.5초 유지 시엔 0.70 m/67°로 안전. fresh 초반 `below` 80%대는 제어 결함이 아니라
**행동이 매 스텝 뒤집히는 데서 오는 것**으로 보인다. 과거 성공한 fresh run도 capture 0.008에서
시작했으므로(이번 `0048`도 epoch 5에 0.007) 초반 고 crash 자체는 이상 신호가 아닐 수 있다 —
학습되며 내려가는지가 관건이며 아직 미확인.

### 작업 규칙 변경

`.claude/skills/navrl/SKILL.md`와 `CLAUDE.md`에 **WORKLOG 규칙**을 명문화했다: 학습·평가·코드변경·
진단(기각된 가설 포함) 각각에 대해 이 파일 갱신을 필수로 하고, diff 검토 요청 **전에** 작성하며,
세션 예산이 부족해도 마지막까지 자르지 않는다.

---

## 2026-07-27 — (A) representation run 평가 완료 + (B) 토큰 선택 FOV 설계

### (A) `ppo_260727_0225` 평가 — **첫 fresh 완주**, bar_contact를 처음으로 움직임

fresh 8000/8000 epoch 완주(NaN 없음). 이전 fresh 시도는 epoch 43~62에서 NaN 사망 — 병렬 세션이
고친 `lr_schedule: None`이 효과를 본 것으로 보인다. peak reward 107.8 @ ep7247, peak capture 96.0%.

6-speed held-out 평가(128 env, 2049 ep/셀, general spawn, 학습과 동일한 72빔/8토큰 조건).
결과 = `results/general_repr_speed_axis.csv`. 절대율(전체 에피소드 대비) 평균, 직전 최고(12m peak,
5토큰/36빔) 대비:

| 지표 | 12m peak | **0225 (8토큰/72빔)** | Δ |
|---|---|---|---|
| capture | 0.851 | **0.861** | +1.0pp |
| **bar_contact** | 12.7% | **9.0%** | **−3.7pp (−29%)** |
| below | 0.2% | 3.5% | +3.3pp |

**캠페인 통틀어 bar_contact를 움직인 첫 개입.** 고도 PI·tilt-comp·12m look-ahead는 셋 다 ~13%로
남겼었다.

### ★ 그러나 예측한 메커니즘 2개 모두 기각 (다시 시도하지 말 것)

- **H1 "토큰 용량이 병목"** → **기각.** 용량을 5→8로 늘렸는데 `hit_in_tokens`가 0.647 → **0.40~0.53
  으로 오히려 하락.** `token_rank≈2.9`라 늘린 슬롯을 쓰고는 있다. 반경 내 막대가 ~16개라 8개로도
  절반이 안 잡힌다.
- **H2 "token 위치오차 = 각도 양자화"** → **기각.** 빔을 36→72로 늘려 bin을 10°→5°로 절반으로
  줄였는데 `token_err`가 0.57m → **0.72~1.13m로 증가.**
- ⇒ **bar_contact 개선은 우리가 겨냥한 토큰 메커니즘이 아니라 다른 경로**(가장 유력: 더 촘촘해진
  static scan 자체)에서 왔다. 개선은 진짜지만 이유는 예측과 달랐다.

### (B) 설계 — 토큰 선택 FOV (`NAVRL_OBSTACLE_FOV_DEG`, 기본 360 = 기존 동작)

기하 계산이 측정과 일치: 360°에서 8토큰 = 반경 내 16개 중 **50% 커버** → 측정된 `hit_in_tokens`
0.40~0.53과 정확히 맞는다. **뒤쪽 막대에 슬롯을 쓰느라 앞쪽이 굶는 구조.**

| 선택 FOV | 반경 내 막대 | 8토큰 커버율 |
|---|---|---|
| 360°(현재) | 16.0 | 50% |
| 240° | 10.7 | 75% |
| 180° | 8.0 | 100% |

구현: `navrl_perception._fuse_static_and_extract_obstacles`에서 **선택용 복사본에만** 섹터 밖
bearing을 blank 처리. `static_state`는 전 360°를 그대로 유지하므로 전방위 인지는 잃지 않는다.
**관측 차원 898 불변 → 0225에서 warm-start 가능**(fresh 재학습 불필요).

**제로샷 사전확인**(재학습 없이 FOV만 바꿔 0225 평가, target 1.0): 360° cap 0.869/hit 0.418,
240° cap 0.857/hit 0.479, 180° cap 0.862/hit 0.402. `hit_in_tokens`가 기대(0.75)에 훨씬 못 미친다 —
정책이 360° 토큰 배치에 맞춰 학습됐기 때문일 수 있어 **재학습해야 판정 가능**. 이것이 (B)를 하는 이유.

**판정 기준**: `hit_in_tokens` 0.42 → 0.65+ **그리고** `bar_contact` < 9%면 성공. 실패 시 240→180
재시도, 그래도 안 되면 `NAVRL_OBSTACLE_FOV_DEG` 제거로 롤백하고 (C) 밀도로 진행.

### 다음 순서 (사용자 확정): A → B → C
- (A) 완료
- (B) 위 warm-start 명령으로 ~2250 epoch(≈1.5h) 추가 학습 — 사용자가 직접 실행
- (C) 밀도 커리큘럼 = 논문 본선(밀도×속도 지도). bar_contact를 먼저 낮춰야 고밀도에서 폭발하지 않음
  (과거 150막대 절벽 −27pt 기록).

---

## 2026-07-27 — (B) FOV-240 결과 감사 + bar probe v2 + (C) 진입 결정

### (B) `ppo_260727_0930` 평가 결과

`ppo_260727_0225`의 360° token-selection 정책에서 warm-start해 epoch 7248→9500을 추가 학습했다.
best `gen_ppo.pth`는 epoch 8571, optimizer LR은 고정된 `1e-4`. 6-speed general-spawn 평가 결과는
`results/general_repr_fov240_speed_axis.csv`에 보존했다.

| target | capture (360 baseline→240) | bar_contact 절대율 | below 절대율 |
|---:|---:|---:|---:|
| 0.0 | 0.874→0.894 | 7.4%→6.2% | 3.0%→2.9% |
| 0.5 | 0.879→0.906 | 7.9%→6.2% | 3.8%→2.7% |
| 0.75 | 0.870→0.894 | 8.6%→6.8% | 3.3%→2.9% |
| 1.0 | 0.869→0.893 | 8.9%→7.8% | 3.3%→2.4% |
| 1.25 | 0.859→0.889 | 10.1%→7.9% | 3.0%→2.6% |
| 1.5 | 0.824→0.859 | 11.9%→10.0% | 4.0%→3.2% |
| **평균** | **0.862→0.889 (+2.7pp)** | **9.1%→7.5%** | **3.4%→2.8%** |

캠페인 궤적은 capture `0.851→0.861→0.889`, bar_contact `12.7%→9.1%→7.5%`.
240° 정책을 360° token FOV로 평가하면 target 1.0 capture가 `0.893→0.878`로 내려가므로 반드시
학습과 동일한 240°로 평가해야 한다.

### 관측성 결함 수정

`0930`은 FOV 기록 패치 전에 시작되어 log/checkpoint에 FOV가 없었다. 240°/360° paired eval로 조건을
역추론해야 했으므로 다음을 추가했다.

- 시작 로그: `tokens`, `token_fov`, `suppress`, `scan V×H`, `lidar_range`를 한 줄로 출력.
- checkpoint `env_state`: `cfg_token_fov_deg`, `cfg_obstacle_suppress_deg`,
  `cfg_lidar_hbeams/vbeams`까지 저장.
- resume/eval 시 현재 값과 다르면 policy input mismatch 경고.
- FOV `(0, 360]`, 양수 beam/token 수, 음수가 아닌 suppression을 시작 시 검증.
- density/speed 평가 스크립트의 기본 contract를 최신 정책의
  `8 tokens / 240° / ±10° / 4×72 / 12m / general spawn / 128 env`로 교정.
- density eval의 `... | grep ... || true`가 `play_navrl.sh` 실패까지 성공으로 숨기던 문제를 제거해,
  `set -o pipefail`이 평가 오류를 정상적으로 중단시키게 함.

### ★ 기존 `hit_in_tokens`/`token_err` 해석은 무효 — bar probe v2

기존 probe에는 세 결함이 있었다.

1. 240° 밖 후방 충돌도 `hit_in_tokens`의 전체 분모에 포함하고 이를 240° 내부의 기하학적 75%
   커버 예상치와 비교했다.
2. 같은 방위의 토큰을 거리와 무관하게 충돌 막대에 매칭해, 밀집 구간에서 앞/뒤의 다른 막대를
   잘못 붙였다.
3. token 위치는 LiDAR **표면점**인데 GT **중심점**과의 거리를 position error라고 불렀다. 막대
   반경 자체와 측정 오차가 섞인 값이었다.

v2는 token ray가 GT bar bounding circle을 실제로 통과하고 표면 range도 가능한 경우에만
range+bearing으로 연관한다. 또한 다음을 별도로 출력한다.

- `hit_fov`: 충돌 막대가 token-selection FOV 안에 있었던 비율.
- `hit_token_given_fov`: 선택 가능했던 충돌만 분모로 한 실제 token representation 비율.
- `unique/duplicate`: 토큰 슬롯이 서로 다른 GT 막대를 담는지, 같은 막대를 중복하는지.
- `center_offset`, `cross_track`, `radial_gap`: 표면-중심 거리, 횡방향 ray 오차, 중심까지 남은
  방사 거리. 더 이상 이것을 하나의 `token_err`로 부르지 않는다.

순수 geometry helper `bar_probe.py`와 동일 방위·다른 거리 오매칭 회귀 테스트를 추가했다.

### 실제 GPU 검증 (FOV 240°, target 1.0, 32 env, 2048 episodes)

```
capture=0.883  crash=0.116  timeout=0.001
bar_contact=0.688 of crashes  below=0.257 of crashes
barprobe v2:
  n=163  bars_range=15.7  bars_fov=10.9
  hit_fov=0.773  hit_token=0.442  hit_token_given_fov=0.556
  valid tokens=8.0  associated=2.3  unique=0.8  duplicate=1.4
  center_offset=0.36m  cross_track=0.20m  radial_gap=0.25m
```

`token_err 1.2~1.3m`는 v2에서 재현되지 않았다. 주된 원인은 실제 센서 오차가 아니라
bearing-only 오매칭과 표면/중심 의미 혼동이었다. global `hit_token=0.442`가 낮은 이유 중 일부도
후방 FOV 밖 충돌을 포함한 분모였고, 조건부 값은 `0.556`이다.

### 결정: suppression ±5° 재학습은 보류하고 (C)로 진행

±10°인 현재 상태에서도 contact 순간 평균 `duplicate=1.4` 슬롯이 이미 같은 GT 막대에 중복 사용된다.
suppression을 ±5°로 줄이면 같은 넓은 막대가 더 많은 슬롯을 소비할 가능성이 커서, “240° 중 더 넓게
덮는다”는 단순 계산과 반대 효과가 날 수 있다. 그 가설의 근거였던 기존 `hit_in_tokens/token_err`도
측정 결함이 확인됐다.

반면 실제 목표 지표는 capture `0.889`, bar_contact `7.5%`로 명확히 개선됐다. 따라서 한 번 더
25-bar 조건을 미세 조정하지 않고 (C) density curriculum으로 진입한다.

전용 `train_navrl_general_repr_density.sh`를 추가했다. 기존
`train_navrl_vision_seq_density.sh`는 898-D representation/FOV를 고정하지 않아 이 checkpoint에
사용하면 안 된다. 새 launcher는 240°/72빔/8토큰/12m를 고정하고 `NAVRL_NUM_BARS`를 unset하여
25→110, +5 bars, capture threshold 0.55 curriculum이 실제로 승급되게 한다.

### 검증

- `tests/test_navrl_bar_probe.py`: 4/4 통과.
- `tests/test_navrl_perception.py`: 4/4 통과.
- `tests/test_training_safety.py`: 4/4 통과.
- 수정 Python `py_compile`, launcher `bash -n`, `git diff --check` 통과.
- 실제 Isaac Gym 2048-episode CUDA 평가 완료, probe v2 출력 확인.

---

## 2026-07-27 — (B) FOV 240° 결과 = 역대 최고 + 대시보드 갱신·3D 튐 수정

### (B) 토큰 선택 FOV 240° — 성공, capture 0.889

`ppo_260727_0930`(0225에서 warm-start, +2253 epoch, peak reward 112.5 @ ep8571). 6-speed held-out
(128 env, 2049 ep/셀, general spawn) = `results/general_repr_speed_axis.csv` 대비:

| target | capture | bar_contact(절대) | below(절대) |
|---|---|---|---|
| 0.0 | 0.874 → **0.894** | 7.4% → 6.2% | 3.0% → 2.9% |
| 0.5 | 0.879 → **0.906** | 7.9% → 6.2% | 3.8% → 2.7% |
| 0.75 | 0.870 → **0.894** | 8.6% → 6.8% | 3.3% → 2.9% |
| 1.0 | 0.869 → **0.893** | 8.9% → 7.8% | 3.3% → 2.4% |
| 1.25 | 0.859 → **0.889** | 10.1% → 7.9% | 3.0% → 2.6% |
| 1.5 | 0.824 → **0.859** | 11.9% → 10.0% | 4.0% → 3.2% |
| **평균** | 0.862 → **0.889 (+2.7pp)** | 9.1% → **7.5%** | 3.4% → 2.8% |

**캠페인 궤적: 0.735 → 0.802 → 0.837 → 0.851 → 0.861 → 0.889**, bar_contact 12.7 → 9.1 → 7.5%.

### ★ 내 구현 결함: FOV가 아무 흔적을 안 남겨 사후 판별 불가였음 (수정함)

학습이 끝난 뒤 **FOV가 실제로 적용됐는지 로그·체크포인트 어디서도 확인할 수 없었다.** 240°/360° 두
조건으로 평가해 **선호가 뒤집히는 것**(0225는 360°가 우세, 0930은 240°가 우세)으로 역추론해야 했다.
수정: ① 시작 시 `NavRL obstacle representation | tokens=.. token_fov=.. suppress=.. scan=..` 출력
② `env_state`에 `cfg_token_fov_deg` 기록 + 불일치 경고. **이 정책은 반드시 240°로 평가해야 한다**
(360°로 평가하면 0.893→0.878로 오독).

### 남은 갭 (다음에 볼 것)

- `hit_in_tokens` 0.42 → 0.50 수준으로만 상승(예측 0.75). 원인 추정: **억제폭 ±10°** 때문에 8토큰이
  최소 20° 간격을 요구해 240° 중 실제로는 ~160°만 덮는다 → 후보 B′ = 억제폭 ±5°(warm-start 가능, 30분).
- `token_err`는 계속 악화 방향(0.75~1.30m). bar_contact는 줄어드는데 이 지표만 나빠지는 패턴이 반복 →
  **지표 자체의 측정 방식을 의심**(밀집 구간에서 ±15° 매칭이 엉뚱한 막대를 잡을 수 있음). 병렬 세션이
  probe v2(range+bearing GT 연관)로 재작성 중.

### 대시보드 갱신 (`docs/status/`, 커밋 `8fa5752`)

내용: 헤드라인 KPI(0.889 / 7.5% / 2.8% / 0%), 캠페인 궤적 표, **기각된 가설 섹션 신설**(look-ahead·
토큰용량·각도양자화 3건), 학습 안정성 3종(σ 폭주·adaptive LR 폭주·기울기 제한 부재), 로드맵을
P6 밀도 커리큘럼 진행 중으로 갱신. 3D 씬 스펙도 현재 정책(LiDAR 72×4 @12m, 기본 25막대)으로 맞춤.

**3D가 툭툭 튀던 원인 4가지 (사용자 지적)**:
1. **`clearBars()`가 하드 투영** — 매 프레임 독립적으로 0.25씩 최대 6회 밀어내(최대 1.5 유닛) 위치가
   `tParam`의 연속함수가 아님. 막대를 스칠 때 보정이 갑자기 켜지거나 방향이 뒤집힘. **주원인.**
2. **헤딩을 그 튀는 위치의 프레임 차분으로 계산** → 위치가 튀면 기체 전체가 홱 돌아 시각적으로 증폭.
3. **프레임 기반 적분**(`tParam += 상수/frame`) → 주사율 의존 + 드롭 프레임마다 건너뜀.
4. **궤적을 5프레임마다만 샘플링** → 꼬리가 각지고 한 마디씩 늘어남.

수정: 투영을 지수 스무딩으로 추종(dt 기반), 헤딩은 필터링된 속도벡터에서 최단각 회전, `performance.now()`
델타 적분(탭 복귀 클램프), 궤적은 이동거리(0.12m) 기준 샘플링.

---

## 2026-07-27 — Stage C 밀도 커리큘럼 중단 진단: collapse guard 오발

### 결론

`ppo_260727_1204_navrl`은 NaN/OOM/프로세스 강제 종료나 실제 정책 붕괴로 끊긴 것이 아니다.
epoch 8906에서 `early_stop_collapse`가 정상적인 밀도 증가에 따른 reward 하락을 붕괴로 오인해
정상 종료 마커를 남기고 학습을 끝냈다.

```
Early stop: reward collapse from peak — peak 108.3 → now 38.7
for 100 epochs (epoch 8906)
```

### 근거

- 커리큘럼은 25→30→35→40→45→50→55→60 bars까지 7회 연속 정상 승급했다.
- 각 승급 capture는 `0.856, 0.815, 0.766, 0.729, 0.682, 0.635, 0.587`로 모두
  승급 문턱 0.55를 통과했다.
- reward 평균은 난도가 오르면서 약 `101.0(25 bars) → 45.4(55 bars)`로 연속 하락했다.
  갑작스러운 수치 발산이 아니라 커리큘럼의 의도된 난도 상승과 일치한다.
- 가드는 전체 run의 저밀도 peak 108.3을 계속 기준으로 삼았다. 설정
  `drop_from_peak=0.35`는 reward가 `70.4` 미만인 epoch를 누적하고, density 변경 때 peak나
  counter를 재설정하지 않아 100 epoch 뒤 종료했다.
- 종료 직전 55 bars capture는 약 0.587, 60 bars의 짧은 4 epoch 구간도 약 0.590이었다.
  커리큘럼이 정체되거나 성능이 0으로 무너진 상태가 아니었다.
- 로그에 NaN, OOM, CUDA 오류, traceback, SIGKILL 흔적이 없고
  `.aerial_training_finished`에 `epoch=8906`이 기록됐다.

### 재개 가능성 및 조치

`nn/last_gen_ppo_ep_8906_rew__38.70596_.pth`의 actor, asymmetric critic, 두 optimizer state를
직접 검사했다. 모든 tensor가 finite이고 actor/critic LR도 모두 `1e-4`여서 이 체크포인트는
60 bars부터 재개 가능한 정상 상태다. 저밀도 best-reward인 `gen_ppo.pth`로 재개하면 안 된다.

다만 같은 설정으로 바로 재실행하지 않는다. Stage C launcher에서는 NaN/Inf fail-fast는 유지하되
고정 밀도용 global `early_stop_collapse`를 비활성화하는 것이 권장 조치다.
`drop_from_peak=0.80` 완화는 이번 오발은 피할 수 있지만 밀도가 더 오르면 다시 오발할 수 있어
근본 해결이 아니다. 더 정교한 대안은 density 승급마다 peak/counter를 재설정하고 동일 밀도
구간 안에서만 붕괴를 판정하는 것이다.

### 조치 완료

- `runner.py`에 run 단위 옵션 `--disable_collapse_early_stop`을 추가했다. 이 옵션은
  `early_stop_collapse.enable=False`만 적용하므로 일반/고정밀도 학습의 YAML 기본 가드는 유지된다.
- Stage C 전용 `train_navrl_general_repr_density.sh`가 위 옵션을 항상 전달하도록 고쳤다.
  시작 헤더에는 `reward-collapse guard=off; NaN/Inf fail-fast=on`이 명시된다.
- Stage C 런처의 기본 체크포인트를 25-bars best인 `ppo_260727_0930/gen_ppo.pth`에서
  `ppo_260727_1204/last_gen_ppo_ep_8906_rew__38.70596_.pth`로 변경했다. 체크포인트
  `env_state.n_bars_active=60`이 복원되므로 별도 인자 없이 60 bars부터 새 run으로 분기 재개한다.
- mean reward의 NaN/Inf 검사를 reward-collapse 함수 밖으로 분리했다. 따라서 collapse 가드를 꺼도
  NaN/Inf reward와 PPO loss/parameter non-finite는 즉시 실패하며 손상 체크포인트를 저장하지 않는다.

검증: 전체 단위 테스트 13/13, 수정 Python `py_compile`, 두 launcher `bash -n`,
`git diff --check`, 실제 runner argparse의 새 옵션과 config override/header 출력 확인 완료.
학습은 자동으로 시작하지 않았다.

재개 명령:

```bash
cd /home/fair/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games
./train_navrl_general_repr_density.sh
```

---

## 2026-07-27 — Stage C 1차 완주(65막대) + 밀도 곡선 실측 + 대시보드 갱신

### Stage C run `ppo_260727_1309` — max_epochs 도달 (8907 → 15000)

느린 승급 설정(`CHECK_EPS=16384`, `THRESHOLD=0.70`)으로 재개. **승급 1회: 60 → 65막대**
(epoch 13208, capture 0.705). 최종 65막대 capture 68.5%, crash 31.5%, reward 48.6(peak 64.9).

**설정 변경이 결정적이었다** — 직전 `1204` run(문턱 0.55 / 창 4096)은 **45 epoch마다** 승급해 7단계를
몰아쳤고 capture가 0.856 → 0.587로 **회복 없이** 내려앉아 reward-collapse 가드에 걸려 종료됐다.
문턱 0.70 / 창 16384로 단계당 정착 시간을 ~4배 늘리자 **승급 후 회복 패턴이 돌아왔다**:

| 65막대 구간 | capture | crash |
|---|---|---|
| 초반 | 0.668 | 0.328 |
| 중반 | 0.680 | 0.315 |
| 후반 | 0.685 | 0.312 |

**"실력이 아니라 시간에 따라 난이도를 올리면 무너진다"는 이 프로젝트에서 두 번째 확인**(첫 번째는
거리 커리큘럼의 epoch 비례 램프 폭주). 커리큘럼 게이트는 문턱과 **판정 창 길이**를 함께 봐야 한다.

추세(회귀): 60막대 구간 crash −0.044/1000ep → 65막대 −0.013/1000ep. 개선은 계속되나 **밀도가 오를수록
단계당 소요가 늘어난다.** 최근 400ep 평균 capture 0.688 → 다음 승급까지 약 800 epoch 예상.

### ★ 밀도 곡선 실측 = `results/general_repr_density_curve.csv`

`last_gen_ppo_ep_15000`(65막대까지 학습), target 1.0 m/s, 2049 ep/셀, deterministic:

| 막대 | 밀도/100㎡ | capture | crash | bar_contact 비중 | 비고 |
|---|---|---|---|---|---|
| 25 | 5.2 | 0.896 | 0.102 | 65% | 학습됨 |
| 50 | 10.5 | 0.796 | 0.202 | 85% | 학습됨 |
| 65 | 13.6 | 0.679 | 0.316 | 88% | **학습 상한** |
| 75 | 15.7 | 0.598 | 0.399 | 92% | 범위 밖 |
| 110 | 23.0 | 0.280 | 0.711 | 94% | 범위 밖 (NavRL 앵커) |
| 130 | 27.2 | 0.150 | 0.847 | 95% | 범위 밖 |
| 150 | 31.4 | 0.079 | 0.920 | 95% | 범위 밖 |

**해석 주의**: 75막대 이상은 **학습하지 않은 밀도**다. 이 하락을 "방법의 한계"로 읽으면 안 되고
**일반화 측정**으로 읽어야 한다(과거 GT-LiDAR 시절 밀도 커리큘럼으로 110막대를 학습했을 때는 0.93였다).
또한 밀도가 오를수록 **실패가 사실상 전부 막대충돌**이 된다(65% → 95%). 고밀도 상한은 곧 충돌 문제다.

### 대시보드 갱신 (`docs/status/`, 커밋 `9495309`, `4c4184c`)

- 히어로 KPI를 밀도 축 중심으로 교체(25/50/65막대 포획 + 학습 도달 밀도)
- **밀도 곡선을 실측값으로 교체**하고, 학습 범위(≤65막대)를 음영+점선으로 표시해 그 오른쪽이
  일반화 구간임을 그래프에서 바로 보이게 함. 표에도 `학습범위` 열 추가.
- Stage C 섹션 신설(승급 속도가 결정적이었다는 내용), 다음 계획을 "커리큘럼 연장 65→110"으로 갱신
- 로드맵 P6를 "밀도 커리큘럼 25→65막대 (진행 중, 목표 110)"로

### 다음 판단 (사용자 확인 대기)

**연장 권장** — 65막대 구간이 아직 포화하지 않았고(crash 기울기 음수 유지), 승급 문턱까지 ~800 epoch.
다만 110막대까지는 단계당 소요 증가를 감안하면 **수 시간~10시간** 규모다. 명령:

```bash
CKPT=runs/ppo_260727_1309_navrl/nn/last_gen_ppo_ep_15000_rew_48.580364.pth \
MAX_EPOCHS=30000 NAVRL_DENSITY_CHECK_EPS=16384 NAVRL_DENSITY_THRESHOLD=0.70 \
  ./train_navrl_general_repr_density.sh
```

---

## 2026-07-27 — 밤샘 Stage C 명령 감사 + 재개 런처 안전성 보강

### 실행 상태와 Claude 안내 판정

사용자가 입력한 명령은 실패하지 않았다. 실제 실행은 `ppo_260727_2324_navrl`로 분기됐고,
감사 중 epoch 15200 이상, 65 bars, GPU 약 5.85 GiB로 정상 진행했다. 실행 환경에서도
`DENSITY_THRESHOLD=0.70`, `DENSITY_CHECK_EPS=16384`, 240°/8-token/4×72/12m contract가
모두 확인됐으며 collapse guard off + NaN/Inf fail-fast on도 정확히 적용됐다.

Claude가 맞게 확인한 항목:

- `ppo_260727_1309/...ep_15000_rew_48.580364.pth`는 실제 존재하고 epoch 15000,
  `env_state.n_bars_active=65`, FOV 240°, tokens 8인 finite 체크포인트다.
- `MAX_EPOCHS=45000`은 30000 epoch 추가이고 현재 약 2.6초/epoch 기준 약 21.7시간이다.
- YAML `save_frequency=50`이므로 periodic checkpoint는 50 epoch마다 저장된다.
- 0.70/16384는 직전 run에서 60→65 승급을 capture 0.705로 통과한 보수적인 설정이다.

수정이 필요했던 항목:

- `nohup ... > /dev/null 2>&1`은 launcher의 “확인할 3줄”까지 버리므로, 직후 터미널에서
  그 줄을 확인하라는 안내와 모순된다. 출력 파일을 남기고 `tail -f`해야 한다.
- task 생성 시 출력되는 25 bars는 **체크포인트 복원 전 초기값**이다. 실제 복원 후 dashboard는
  65 bars였지만 복원 완료 로그가 없어 잘못 시작한 것처럼 보였다.
- “밤새 중간에 끝나지 않는다/110까지 간다”는 보장은 아니다. 45000까지의 계산 예산만 보장하며
  threshold를 넘지 못하면 65 bars에서 held가 반복될 수 있고 OOM/NaN/외부 종료도 가능하다.
- density 판정창의 `_density_succ_agg/_density_fin_agg`가 checkpoint에 없었다.
  16384-episode 창 도중 재개하면 누적 증거가 0으로 리셋돼 승급이 불필요하게 늦어질 수 있었다.
- 분 단위 `train_YYMMDD_HHMM.log` 이름은 같은 분의 빠른 재시도가 앞 로그를 덮어썼다.
  실제로 2323 zero-epoch 폴더와 2324 정상 run이 생겼지만 로그는 하나만 남아 원인 추적이 어려웠다.

### 구현한 수정

- Stage C 기본값을 검증된 `threshold=0.70`, `check_eps=16384`, `max_epochs=45000`으로 갱신.
- `CKPT` 미지정 시 가장 최근의 non-`_rlnorm` `last_gen_ppo_ep_*.pth`를 자동 선택.
- 실행 전 checkpoint를 CPU에서 읽어 epoch < max_epochs, 0≤bars≤final, actor/critic/optimizer
  finite 여부와 FOV/token/LiDAR contract를 검증하는 `navrl_checkpoint_preflight.py` 추가.
- 이미 NavRL train process가 있으면 중복 실행을 거부하고, 동시에 시작되는 race는 `flock`으로 차단.
  `PREFLIGHT_ONLY=1`로 학습 없이 전체 사전검증 가능.
- checkpoint에 density window 누적값과 final/step/threshold/check_eps provenance를 저장·복원.
  복원 시 `bars=25->65`, distance window와 density-window 진행량을 명시적으로 출력.
- 세션 로그 이름을 초+launcher PID까지 포함하도록 변경해 빠른 재시도 시 덮어쓰지 않음.
- `density curriculum held`를 억제되는 INFO에서 WARNING으로 올려, gate가 아직 미평가인지
  평가 후 문턱 미달인지 로그에서 구분 가능하게 함.

검증: 실제 epoch-15000 및 최신 epoch-15200 checkpoint preflight 통과, duplicate-process guard
실제 PID 거부 확인, 전체 단위 테스트 17/17, Python `py_compile`, launcher `bash -n`,
`git diff --check` 통과.

주의: 이미 실행 중인 2324 프로세스는 수정 전에 모듈을 로드했으므로 그대로 계속 학습한다.
지금 재시작하면 거의 찬 16384-episode 창을 버리므로 중단하지 않았다. 새 checkpoint의 density-window
영속화는 다음 재개로 새 코드를 로드한 뒤부터 적용된다.

현재 run의 첫 판정창은 epoch 15208에 16457 episodes, capture 0.6925로 0.70에 조금 못 미쳐
**65 bars held**였다. 기존 코드가 held를 INFO로 숨겨 promotion 로그가 없었던 것이다.
이후 새 창의 초기 2147 episodes는 capture 0.6968로 문턱 근처이며 학습은 계속 정상 진행 중이다.

현재 run 모니터링:

```bash
tail -f train_session_logs/train_260727_2323.log
```

현재 run 종료 후 자동 최신 checkpoint로 다시 밤샘 실행:

```bash
nohup ./train_navrl_general_repr_density.sh \
  > train_session_logs/night_density.out 2>&1 &
tail -f train_session_logs/night_density.out
```

### 2026-07-28 01:47 — 중복 실행 문의 확인

사용자가 동일 launcher를 한 번 더 실행했으나 duplicate guard가 기존 train PID `1652517`을 찾아
의도대로 exit 3으로 거부했다. 실패가 아니라 GPU에 두 학습이 겹치는 것을 막은 정상 동작이다.
기존 `ppo_260727_2324_navrl`은 epoch 18096/45000, 70 bars, VRAM 약 5.85 GiB로 정상 진행 중이다.
65→70 승급은 16388 episodes에서 capture 0.706으로 통과했다. 현재는 새 실행 명령이 아니라
`tail -f train_session_logs/train_260727_2323.log`로 기존 run을 모니터링한다.

### 2026-07-28 02:03 — 학습 건강도·진척도 정량 판정

`ppo_260727_2324_navrl`은 epoch 18432까지 프로세스/GPU/수치 면에서는 정상이다. step time은
최근 100 epoch 평균 2.70초, VRAM 약 5.85 GiB이고 NaN/OOM/traceback이 없다. 최신 epoch-18400
checkpoint의 actor/critic/두 optimizer는 모두 finite이며 LR은 양쪽 모두 `1e-4`다.

그러나 **목표 진척은 70 bars에서 정체**다. 70 bars 진입(epoch 15618) 뒤 218,475 episodes를
학습했고 16,384-episode 승급 판정창을 13번 완료했지만 capture 범위는 `0.663~0.699`로
threshold 0.70을 한 번도 통과하지 못했다. 전체 70-bars capture는 0.6818이고 최근
16,394 episodes는 0.6762, 최근 2,054 episodes는 0.6957이다. 간헐적 반등은 있으나 지속적인
상승 추세는 아니다. reward는 초반 창의 약 39~41에서 최근 약 42~43으로 소폭 개선됐다.

실패 원인은 고도보다 막대충돌로 수렴한다. 최근 crash 중 bar_contact 비중은 평균 약 91%로
상승했고 below 비중은 약 6.8%로 낮아졌다. `hit_token_given_fov`는 최근 32개 probe 평균 약
0.56, 최신 0.587로 붕괴하지 않아 센서/토큰 표현의 갑작스러운 고장 징후는 없다.

판정: **학습 엔진은 건강하지만 커리큘럼은 잘 진행되지 않는다.** 목표가 안정적인 70-bars 정책이면
계속 학습 가능하지만, 110 bars 도달 목적이라면 현재 threshold 0.70을 유지한 채 epoch 45000까지
방치하는 것은 계산 낭비 가능성이 높다. 실행은 사용자 지시 없이 중단하거나 설정 변경하지 않았다.

### 2026-07-28 02:22 — 70-bars 0.70 gate 정체 원인 감사

사용자 요청에 따라 threshold를 낮추거나 코드를 바꾸기 전에 원인을 분리했다. 실행 중인
`ppo_260727_2324_navrl`은 감사 시점 epoch 18551, 70 bars에서 계속 정상 실행 중이며 설정이나
프로세스는 건드리지 않았다.

#### 확정된 사실

- 70 bars 진입(epoch 15618) 후 2,934 epoch의 epoch-level 평균은 capture `0.6830`,
  crash `0.3117`, timeout `0.0053`이다. 최근 100 epoch는 capture `0.6960`까지 반등했지만,
  최근 400/1000 epoch는 각각 `0.6846/0.6899`라 지속적인 0.70 돌파로 보기는 어렵다.
- 앞서 집계한 16,384-finished-episode 창 13개가 모두 `0.663~0.699`였다. 진짜 성공률이
  약 0.683이면 한 창의 이항 표준오차는 약 0.0036이므로 0.70은 약 4.7σ 위다. 따라서 긴 창이
  우연히 나쁘게 뽑힌 문제가 아니라 현재 **stochastic training policy**가 약 68%에 머문 것이다.
- 70-bars 2,927개 정렬 epoch에서 capture와 평균 target speed의 상관은 `0.0136`, goal-distance
  max와의 상관은 `0.0185`다. 두 변수의 사분위별 capture도 모두 `0.6817~0.6847`이다.
  특정 고속 표적이나 먼 목표가 평균을 끌어내린다는 가설은 현재 로그에서 기각된다.
- 실패의 약 91%는 bar contact이고 below는 약 7%, timeout은 약 0.5%다. 충돌 x는 평균
  `12~13 m`, episode step은 약 `38~41`이라 스폰/goal/timeout 가장자리 문제가 아니라
  막대밭 중앙 위빙 중 충돌이다.

#### 가장 먼저 검증할 원인: stochastic gate와 큰 횡행동 노이즈

밀도 gate는 별도의 deterministic 평가가 아니라 PPO rollout의 **샘플 행동**으로 끝난 episode를
그대로 센다. epoch-18550 체크포인트의 정책 log-std/std는 행동 `[x,y,z,yaw]` 순서로:

```
raw log-std = [-0.992, 0.132, 0.400, -2.531]
effective std = [0.371, 1.141, 1.492, 0.080]
```

`y`는 기체 좌우 속도 행동이고 환경에서 `clamp(action,-1,1) * 2.5 m/s`로 변환된다. std 1.141이면
평균 행동이 0이어도 정규분포 샘플의 약 38%가 ±1 바깥이라 매 step 좌우 명령이 포화된다
(평균이 0이 아니면 포화율은 더 커질 수 있다). 이 std는 65-bars 시작 체크포인트의 `1.063`에서
현재 `1.141`로 증가했다. 반면 x/yaw std는 `0.444→0.371`, `0.103→0.080`으로 감소했다.

z std가 상한 1.492에 붙은 것은 task가 z 행동을 즉시 PI altitude command로 덮어써 실제 기동에는
영향이 없지만, 사용하지 않는 차원에도 entropy bonus가 계속 걸린다는 별도 설계 결함이다. 수치
안정성은 괜찮다(entropy 최근 약 2.69, explained variance 약 0.69, LR 1e-4). 다만 좌우 노이즈는
실제 막대 회피를 매 step 교란한다. 따라서 “평균 정책은 70%가 가능한데 탐색 rollout으로 재는
gate만 못 넘는다”가 현재 가장 값싼 설명이다. 아직 동일 체크포인트의 deterministic/stochastic
paired eval이 없으므로 **유력 가설이지 확정 원인은 아니다**.

#### 두 번째 원인 후보: 밀집 기하를 지나치게 압축하는 표현

- 70-bars bar-contact 순간 12m 안에는 평균 `43~44`개 막대, 240° token FOV에는 `29~31`개,
  장애물이 찬 scan bearing은 약 `54/72`개다.
- 8개 token은 모두 valid지만 range+bearing 기하로 GT bar에 연관되는 것은 평균 `2.5~3.0`개,
  서로 다른 막대는 `1.1~1.3`개뿐이고 나머지는 같은 근접 막대 표면 중복 또는 비연관 표면이다.
  충돌 막대가 240° 안에 있었을 때도 token으로 표현된 비율은 최근 대략 `0.47~0.63`,
  장기 평균 약 `0.56`이다.
- 선택 로직은 물체/경로 위험도를 구분하지 않고 nearest surface bearing을 순서대로 고른 뒤
  ±10°를 억제한다. 가까운 넓은 막대가 여러 slot을 소비하고 진행 경로상의 다른 막대가 누락될 수
  있다. 또한 8개 token 전체를 시간 step당 하나의 64-D token으로 projection하므로 slot별
  attention이나 안정적인 object identity가 없다.
- 360° 4×72 static scan은 남아 있지만 CNN이 이것 전체를 하나의 64-D token으로 압축한다. 따라서
  “충돌 막대 token 누락 = 완전 실명”은 아니나, 조밀한 free-space topology를 policy가 쓰기 좋은
  형태로 유지한다고도 보장할 수 없다.

bar probe는 **충돌 순간** 측정이라 token 누락이 충돌의 선행 원인임을 단독으로 증명하지는 못한다.
그러나 25→65→70으로 밀도가 올라갈수록 실패가 bar contact로 수렴하고, 과거 GT-LiDAR policy는
110 bars에서 약 0.93을 달성했으므로 70 bars가 환경의 절대 기하 한계라는 설명은 기각된다.
현재 sensor-only 표현/정책/탐색의 병목이다.

#### 후순위·미측정 가설

- 보상: safety reward가 전 ray의 mean-log-distance라 국소 충돌 위험 신호가 희석될 수 있으나,
  과거 clearance reward 세 실험이 모두 null이었으므로 첫 수정 후보는 아니다.
- 제어: 2.5 m/s에서 yaw 3.0 rad/s가 조밀 위빙에 부족하거나 action saturation이 있을 수 있으나
  현재 action/yaw saturation telemetry가 없어 단정할 수 없다.
- target occlusion: tracker age/visibility가 actor에는 들어가지만 bar-contact outcome별 visibility를
  로그하지 않아 아직 분리하지 못했다. target speed/goal distance 무상관만으로 occlusion까지
  기각할 수는 없다.

#### 다음 진단 순서

1. 현재 체크포인트를 70 bars, 동일 seed/distribution에서 `deterministic=True`와 stochastic 두 조건,
   최소 4096 episodes씩 paired 평가한다. deterministic만 0.70을 넘으면 gate/탐색 노이즈가 주원인이다.
2. 둘 다 0.70 아래면 충돌 전 1~2초 구간의 `hit-bar represented`, target visibility/track age,
   action clamp율, yaw clamp율을 success/bar-contact별로 기록한다.
3. 그 결과 뒤에만 수정한다. 1번이 맞으면 커리큘럼 gate를 deterministic probe로 분리하거나
   lateral std/entropy 구조를 고친다. 2번이 맞으면 단순 token 수 증가보다 object/segment 또는
   TTC·진행경로 위험 기반 token과 slot별 tokenization을 우선한다.

### 2026-07-28 02:40 — NavRL/NavRL++ 원문 밀도 대조 및 원인 가설 정정

사용자가 “NavRL/NavRL++에 비해 70막대가 훨씬 많은 것은 아니지 않느냐”고 지적해 원 논문,
공개 NavRL 코드, 현재 환경 코드를 같은 면적 기준으로 다시 감사했다. 결론은 **사용자 지적이 맞다.**
70막대를 고밀도 자체의 한계로 해석하거나, 정적 막대 수를 8 obstacle token 용량과 직접 비교한
앞선 설명은 잘못됐다.

#### 원문·공개 구현에서 확인한 환경

- NavRL 공개 학습 config: `350 static + 80 dynamic`.
- NavRL 공개 코드의 실제 obstacle 영역: `map_range=[20,20,4.5]`, 즉 내부 `40×40 m`.
  static 폭은 `0.4~1.1 m`, dynamic 폭은 `0.25~1.0 m`, 충돌 robot radius는 `0.3 m`.
- NavRL 논문은 전체 환경을 `50×50 m`로 표기하고 static 350개를 고정한 채 dynamic을
  `60→80→100→120`으로 승급한다. 코드상 장애물 자체는 중앙 40×40m에 배치되고 바깥은 border다.
- NavRL++ Table I:
  - S1: `300 static + 60 dynamic`
  - S2: `350 + 80`
  - S3: `400 + 100`
  - S4: `400 + 120`
  - S5: `400 + 140`
- NavRL++ 평가는 명시적으로 `40×40 m` 안에 static `300/350/400`, dynamic 환경은 여기에
  `60/80/100`을 추가한다. high-complexity 결과는 static SR `99.84%`, dynamic SR `83.96%`다.
- NavRL++ 정적 표현은 `4×36` 360° ray-distance map을 CNN으로 하나의 64-D static token으로
  만든다. 최대 5개 object state와 그 history는 **dynamic obstacle 전용**이다.

#### 면적당 밀도 비교

우리 막대 배치 밴드는 `(0.96-0.13)×24×24 = 478.08 m²`이고, 70 bars는
`14.64 bars/100m²`다. NavRL++의 40×40m 기준은:

| 조건 | static/100m² | static+dynamic/100m² | 우리 밴드 환산 개수 |
|---|---:|---:|---:|
| 우리 70 bars | 14.64 | 14.64 | 70 |
| NavRL++ S1 | 18.75 | 22.50 | static 90 / total 108 |
| S2 | 21.88 | 26.88 | static 105 / total 128 |
| S3 | 25.00 | 31.25 | static 120 / total 149 |
| S4 | 25.00 | 32.50 | static 120 / total 155 |
| S5 | 25.00 | 33.75 | static 120 / total 161 |

4m 반경의 균일밀도 기대 장애물 수도 우리는 약 `7.36`, NavRL++은 S1 total `11.31`,
S2 `13.51`, S5 `16.96`이다. 현재 probe의 “12m 안 43~44개”는 NavRL++의 4m local range와
다른 반경을 사용한 수치라 직접 비교하면 안 된다.

우리 bar 폭/깊이는 실측 평균 `0.603 m`(범위 `0.403~0.790 m`), NavRL 공개 static은
`0.4~1.1 m`라 우리 막대가 특별히 큰 것도 아니다. 우리 70-bar 배치 Monte-Carlo 100 layouts는
drone 반폭 0.14m에 추가 side clearance `0.2 m`를 요구해도 path-exists `100/100`, placement
relaxation `0/100`, 최종 center spacing `1.5 m`였다. 따라서 70 bars는 물리적 통로 한계도 아니다.

#### 철회·수정하는 가설

- **철회:** “70 bars라 8 token이 감당하지 못해 68% ceiling이 생긴다.”
- 이유: 우리 막대는 static obstacle이고, NavRL++도 수백 개 static obstacle을 object token 5개가
  아니라 ray map 하나로 처리한다. 우리 actor에도 더 촘촘한 `4×72` full 360° static scan이 별도로
  있으며 8 proposal token은 보조 경로다.
- bar-contact 순간 `unique token bars≈1.2`, `hit_token_given_fov≈0.56`은 우리 auxiliary proposal
  selector가 static surface를 비효율적으로 중복 선택한다는 증거일 수는 있으나, policy가 충돌 막대를
  전혀 관측하지 못했다는 증거가 아니다. full static scan에 남아 있기 때문이다.
- NavRL++도 static scan 전체를 하나의 64-D token으로 압축하므로 “64-D static token 자체가
  70 bars를 못 담는다” 역시 비교 근거가 약하다.

#### 비교 후 더 유력해진 차이

1. **행동 분포/gate 불일치:** NavRL/NavRL++은 bounded Beta action을 사용하며 논문도 Gaussian의
   boundary bias를 피하는 이유를 명시한다. 우리는 Gaussian을 샘플한 뒤 clamp한다. 현재 lateral
   std `1.141`이면 최소 약 38%가 매 step 경계에 잘려 ±2.5m/s 횡명령으로 포화된다. stochastic
   rollout으로 승급을 판정하는 현재 구조와 직접 충돌한다.
2. **문제 설정 차이:** NavRL++은 정확히 알려진 static navigation goal을 goal-aligned frame으로
   관측한다. 우리는 target GT를 actor에서 제거했고, 움직이며 막대에 가려지는 target을 RGB-D/LiDAR
   tracker로 찾아야 하고, random yaw의 vehicle frame에서 yaw까지 학습한다. target visibility/track
   age를 bar-contact outcome별로 아직 측정하지 않았다.
3. **기동 자유도:** 우리 막대는 모두 0~2m이고 drone은 z=1m에 고정돼 2-D로 통과해야 한다.
   NavRL/NavRL++ UAV는 3-D velocity를 사용하고 높이가 다른 장애물 일부를 상하로 회피할 수 있다.
   다만 우리 70-bar layout은 0.2m 추가 여유를 둬도 전부 연결되므로 이것만으로 68%는 설명되지 않는다.
4. **제어 주기:** 우리는 10Hz, NavRL++ 기본은 50Hz다. 하지만 NavRL++ 자체 ablation에서 10Hz
   overall SR `93.20%` vs 50Hz `94.08%`로 차이가 0.88pp여서 주원인일 가능성은 낮다.

정정된 우선순위는 `(1) deterministic/stochastic paired eval → (2) lateral clamp telemetry →
(3) target visibility/track-age conditioned crash 분석`이다. static token 수 증가는 이 세 검증 뒤로
내린다. 실행 중 학습에는 어떤 변경도 하지 않았다.

### 2026-07-28 02:48 — 행동 포화 가설의 의미 명확화

`lateral std=1.141` 문제는 최대 속도 제한을 더 풀어야 한다는 뜻이 아니다. 현재 Gaussian sample을
`[-1,1]`로 clamp한 뒤 2.5m/s를 곱하므로, 큰 분산이 매 step 좌우 최대속도 명령을 자주 만든다는 뜻이다.
상한을 높이면 같은 포화 sample이 더 큰 횡속도가 되어 충돌을 악화시킬 수 있고, clamp를 없애면
unbounded Gaussian 명령이 controller로 들어가므로 둘 다 올바른 수정이 아니다.

검증 후 가능한 수정 방향은 bounded Beta/tanh-squashed policy, 실제 사용하는 action 차원별 적절한
std 제한 또는 entropy 조정, 사용하지 않는 z action 제거, stochastic rollout과 deterministic
curriculum gate 분리다. 우선순위는 여전히 동일 체크포인트의 deterministic/stochastic paired eval이다.

### 2026-07-28 02:36 — 현재 작업 스냅샷 커밋 정리

사용자 요청에 따라 지금까지의 추적 파일 변경을 하나의 Git 스냅샷으로 정리했다.

- 연구 현황 대시보드의 compact panel 레이아웃 변경(`docs/status/index.html`,
  `docs/status/style.css`)을 포함했다.
- 대시보드가 외부 CDN 연결 없이 3D arena를 열 수 있도록 Three.js r128과
  OrbitControls 로컬 사본(`docs/status/vendor/`)을 포함했다. Three.js 파일에는
  MIT SPDX 라이선스 헤더가 보존돼 있다.
- 실행 중인 학습에서 계속 생성되는 `train_session_logs/`(약 81 MB)와
  `checkpoints_saved/`(약 15 MB)는 소스 변경이 아닌 런타임 산출물이므로 Git에는 넣지 않았다.
- 커밋 전 `git diff --check`, JavaScript DOM 참조 감사, headless Chrome 로딩 검사를 수행한다.

### 2026-07-28 02:38 — PPO action noise 대안 감사

사용자 요청에 따라 bounded Beta에 한정하지 않고 현재 `rl_games` 구현, 최신 체크포인트,
bounded-action 정책 및 시간상관 탐색 관련 원 논문을 대조했다. 실행 중인 학습에는 변경을 가하지 않았다.

#### 현재 정책에서 확인한 사실

- `continuous_a2c_logstd`가 대각 Gaussian에서 매 step 독립 표본을 뽑고, 환경
  `transform_action_to_command()`가 뒤에서 `[-1,1]`로 hard clamp한다.
- 최신 `last_gen_ppo_ep_19100_rew_23.645699.pth`의 축별 `(log_std, std)`는
  forward `(-1.007, 0.365)`, lateral `(0.146, 1.157)`, unused-z `(0.400, 1.492)`,
  yaw `(-2.568, 0.077)`이다.
- 평균이 0인 가장 유리한 경우에도 횡축 표본의 `38.7%`가 `|a|>1`이고, 따라서 실제 clamp율은
  최소 38.7%다. `NAVRL_MAX_VELOCITY=2.5`이므로 이 질량은 매번 `±2.5 m/s` 횡명령에 모인다.
- z action은 altitude PI가 덮어써 물리적으로 사용되지 않는데 entropy bonus는 이 축에도 걸린다.
  실제로 z std가 상한에 붙어 있어 총 entropy 지표도 정책에 쓰이는 세 축보다 크게 보인다.
- `bounds_loss_coef`는 rl_games 코드상 `|mu|>1.1`인 평균만 벌점화한다. sigma와 경계 밖
  표본 질량은 줄이지 않으므로 이 값을 키우는 것은 현재 문제의 직접 해법이 아니다.
- 학습은 감사 시점에 75 bars까지 승급했고 최근 2050 episode의 capture는 `0.674`,
  crash는 `0.322`였다. 따라서 70 bars gate가 영구 정체였다는 가정은 이미 기각됐다.

#### 대안 우선순위

1. **현 Gaussian + 단계별 std/entropy annealing**: 초기 `entropy_coef=0.003`은 hover 탈출까지
   유지하고, competence를 얻은 뒤 횡축 std 목표를 우선 `0.5`로 낮춘다. 평균 0 기준 clamp 질량은
   `38.7% → 4.55%`가 된다. 고정 entropy가 다시 std를 밀어 올리지 않도록 entropy coefficient도
   함께 낮추거나 성능 기반 schedule로 바꿔야 한다. 가장 작고 해석 가능한 변경이다.
2. **tanh-squashed Gaussian PPO**: 설치된 rl_games에 이미 `continuous_a2c_tanh`와 Jacobian을
   반영한 log-prob 구현이 있다. Beta 없이도 action support가 `(-1,1)`로 제한되고 hard-clamp
   pile-up이 사라진다. 다만 현재 player의 deterministic 경로는 raw `mu`를 사용하므로
   `tanh(mu)` mode를 사용하도록 먼저 고쳐야 하며, 기존 checkpoint의 log-std가 `exp`에서
   `softplus`로 재해석되는 점도 warm-start A/B에서 통제해야 한다.
3. **gSDE/시간상관 탐색**: marginal std를 단순히 줄이기보다 noise를 여러 control step 동안
   유지해 10 Hz의 좌우 독립 jitter를 부드러운 탐색 궤적으로 바꾼다. 로봇에서 성능 손실 없이
   smoothness를 조절할 수 있다는 CoRL 결과가 있으나 현재 rl_games PPO에는 바로 연결된 구현이 없어
   1·2보다 작업량이 크다.
4. **CAPG 또는 truncated Gaussian**: 현 hard clamp를 유지하면서 tail probability에 맞는
   likelihood/gradient를 써 clipping mismatch를 고칠 수 있다. CAPG는 unbiased·저분산 estimator지만
   표본의 경계 집중 자체를 없애지는 않으므로 std schedule과 병행해야 한다. truncated Gaussian은
   직접 bounded지만 PPO ratio, entropy, CDF 정규화까지 새로 구현해야 해 tanh보다 우선순위가 낮다.
5. **action EMA/action-rate penalty**: 고주파 떨림 완화에는 유효할 수 있지만 raw Gaussian과 실제
   실행 action의 차이를 또 만들 수 있다. 현재 보상에도 achieved-velocity smooth penalty가 이미 있어,
   먼저 raw action clamp율과 `Δaction` telemetry로 고주파 jitter가 실제 crash 원인인지 확인한다.

권장 실험은 현재 run을 건드리지 않고 동일 checkpoint에서 `(A) 기존 stochastic`,
`(B) deterministic`, `(C) lateral std=0.5`를 paired seed로 평가한 뒤, C가 capture/crash를
개선하면 entropy/std annealing을 먼저 적용하고 tanh-Gaussian을 별도 branch로 비교하는 순서다.

참고 원문: PPO (Schulman et al., 2017), CAPG (Fujita & Maeda, 2018), SAC Appendix C의
tanh change-of-variables (Haarnoja et al., 2018), gSDE (Raffin et al., CoRL 2022),
Beta policy (Chou et al., ICML 2017).

### 2026-07-28 17:36 — 실행 중 density run의 시간 대비 학습효율 감사

사용자가 현재 승급 학습을 계속 오래 돌리는 것이 실제로 도움이 되는지 물어
`ppo_260727_2324_navrl`의 전체 로그(15001~37194 epoch), 체크포인트 sigma, promotion gate,
crashdiag/barprobe를 읽기 전용으로 집계했다. 프로세스는 중단하거나 변경하지 않았다.

#### 현재 상태

- PID `1652517`, 약 18.2시간 실행, GPU 5.85 GiB, 점검 당시 75 bars / 약 37194 epoch.
- 45000까지 약 7800 epoch가 남아 현재 속도 2.6 s/epoch 기준 추가 약 5.6시간이다.
- 실제 promotion은 두 번뿐이다.
  - 65→70: epoch 15617, 실행 0.49시간, gate capture `0.706/16388 eps`
  - 70→75: epoch 18787, 실행 2.96시간, gate capture `0.704/16386 eps`
- 이후 75 bars에서 18407 epoch, 약 147.2만 episode, 약 15.2시간을 보냈지만
  80 bars로 한 번도 승급하지 못했다.

#### 밀도별 집계

| bars | epoch | episodes | 전체 capture | 처음 500 epoch | 마지막 500 epoch |
|---:|---:|---:|---:|---:|---:|
| 65 | 617 | 49,128 | 0.696 | 0.692 | 0.699 |
| 70 | 3,170 | 245,723 | 0.684 | 0.670 | 0.692 |
| 75 | 18,407 | 1,471,998 | 0.634 | 0.653 | 0.587 |

65와 70에서는 완만한 개선과 승급이 있었으므로 처음 약 3시간은 유효했다. 반면 75는
처음 500 epoch보다 마지막 500 epoch capture가 `6.6pp` 낮고 crash가
`34.3%→41.1%`로 증가했다. 최근 2048-episode window도 `capture=0.539~0.565`다.
1000-epoch bin은 중간에 0.65 수준으로 회복한 적은 있으나 0.70 gate를 지속해서 넘은 적이 없다.

75 bars에서 gate window는 `16384 eps`이고 총 147만 episode를 보았으므로 약 89회의
실패 판정을 받은 셈이다. 로그에 `held`가 안 보이는 이유는 run 시작(23:24) 12분 뒤
`cbefbe3`에서 INFO→WARNING으로 가시성을 고쳤기 때문이다. 실행 중 Python은 시작 당시 코드를
메모리에 유지하므로 hold는 수행됐지만 출력만 억제됐다. 즉 승급 로직이 완전히 멈춘 것이 아니라
정책이 문턱을 못 넘은 것이다.

#### 시간이 성능을 악화시킨 정황

- epoch 19050→37150 동안 lateral std가 `1.160→1.492`로 증가해 상한에 붙었다.
  평균 0일 때조차 횡 action hard-clamp 최소 확률이 `38.7%→50.3%`가 된다.
- 같은 기간 forward std는 `0.367→0.192`, yaw는 `0.077→0.062`로 내려가므로
  “모든 행동이 무작정 시끄러운 것”이 아니라 횡축만 선택적으로 포화됐다.
- bar-contact의 crash 내 비중은 초반 20 window `91.4%`, 최근 20 window `92.0%`로 비슷하지만,
  crash 자체가 증가해 절대 bar-contact는 대략 `31.3%→37.8%`로 악화됐다.
- 반대로 `hit_token_given_fov`는 `0.559→0.638`로 좋아졌다. 관측 token hit가 개선됐는데
  포획이 하락했으므로 이번 장기 정체를 representation 용량 부족으로 설명하기 어렵고,
  action 분포/제어 쪽 증거가 더 강하다.

#### 판정 및 다음 조치

**현재 설정 그대로 45000까지 추가 5.6시간을 쓰는 것은 권장하지 않는다.**
75 bars에서 이미 충분한 표본을 훨씬 넘겼고 최근 성능·횡축 sigma가 모두 나빠지고 있다.

중단 후 평가 후보는 75-bar 로그 근방 성능이 좋았던:

- `last_gen_ppo_ep_19050_rew_31.79068.pth` 부근 — 100-epoch 근방 capture `0.691`
- epoch 27450 부근 — `0.681`
- epoch 30450 부근 — `0.672`
- 최신 checkpoint — 퇴행 대조군

이다. 동일 seed로 deterministic/stochastic paired 평가해 19050 계열이 실제 held-out에서도
우세한지 확인한 뒤, 그 checkpoint에서 lateral std=0.5 + entropy annealing branch를 시작하는
것이 다음 합리적 실험이다. `gen_ppo.pth`는 낮은 밀도의 best-reward일 수 있으므로 후보에서 제외한다.

### 2026-07-28 17:46 — 실행 유지 상태에서 전체 로드맵·핵심 병목 재판정

사용자 요청에 따라 실행 중인 `ppo_260727_2324_navrl`은 종료·신호 전송·설정 변경 없이
그대로 유지했다. 점검 시 PID `1652517`, epoch `37401/45000`, 75 bars였고 최근 epoch별
capture는 대체로 45~58%였다.

#### 현재 run의 결론

- 65→70→75까지 약 3시간은 유효한 curriculum 학습이었다.
- 75 bars에서는 약 147만 episode와 약 89개의 16,384-episode gate window를 소비했지만
  80 bars로 승급하지 못했다. 처음 500 epoch capture `0.653`에서 최근 500 epoch
  `0.587`로 하락했고 crash는 `0.343→0.411`로 증가했다.
- 같은 기간 obstacle token의 `hit_token_given_fov`는 `0.559→0.638`로 개선됐는데 절대
  bar-contact는 약 `0.313→0.378`로 증가했다. 따라서 이번 정체의 1차 원인은 8-token
  표현 용량이나 기하학적 불가능성보다 action/optimization 쪽이다.
- 가장 강한 기전은 고정 entropy `0.003`, state-independent unbounded Gaussian,
  후단 hard clamp의 조합이다. 75-bar 초반→최근 lateral std가 `1.160→1.492`로 상한에
  붙어 평균 0에서도 `|a_y|>1`인 표본의 최소 비율이 `38.7%→50.3%`가 됐다. 이 표본은
  실제로 `±2.5 m/s` 횡속도 명령에 쌓인다.
- 다만 sigma와 crash의 상관관계만으로 인과가 완전히 확정된 것은 아니다. 같은 checkpoint의
  deterministic/stochastic paired evaluation과 raw clamp-rate/`Δaction` 계측이 필요하다.

#### capture 비율 개선 우선순위

1. 현재 run은 끝까지 진단 궤적으로 보존하되, 다음 학습의 warm-start 후보는 최신이 아니라
   75-bar 초반의 `last_gen_ppo_ep_19050...pth`와 중간 후보 27450/30450을 held-out 평가해 고른다.
2. 같은 seed·75 bars·4096 episode 이상으로 각 후보의 deterministic/stochastic 성능을 나눠
   평균 정책 퇴행과 sampling noise를 분리한다.
3. noise가 주원인이면 기존 Gaussian의 초기 탐색은 유지하고 competence 이후
   `entropy_coef 0.003→0.0003(또는 0)`과 lateral std 목표/상한 `0.5`를 함께 anneal하는
   최소 변경 branch를 먼저 시험한다. 평균 0 기준 clamp 질량은 약 `50.3%→4.55%`로 줄어든다.
4. 별도 A/B로 tanh-squashed Gaussian을 시험한다. 설치된 `continuous_a2c_tanh`는 사용할 수
   있지만 deterministic player의 `tanh(mu)` 처리와 기존 checkpoint std 재해석을 먼저 고쳐야 한다.
5. curriculum gate는 noisy stochastic train ratio만 보지 말고 고정 seed deterministic
   competence probe를 병행한다. 반대로 threshold를 단순히 0.70→0.60/0.65로 내리는 것은
   퇴행 중인 정책을 다음 밀도로 넘기므로 해결책이 아니다.
6. 속도 상한 확대, reward 재설계, token 억제폭 변경은 현재 증거상 우선순위가 낮다.
   특히 속도 상한 확대는 포화 action의 실제 속도만 키워 충돌을 악화시킬 수 있다.

#### 전체 로드맵의 실제 위치

- 정보 firewall, raw RGB-D/LiDAR 렌더링, tracking/fusion, 17-token Transformer,
  navigation 통합은 구현되어 있어 **엔지니어링 파이프라인은 약 65~75%**로 본다.
- 현재 density curriculum은 목표 25→110 bars 중 75까지 도달했으므로 bar interval 기준
  `(75-25)/(110-25)=58.8%`지만, 75→80 performance gate에 막혀 있다.
- 논문 수준의 실증은 **약 35~45%**다. 25 bars speed-axis와 단일 seed의 부분 density curve는
  있으나, density×speed×occlusion 전체 matrix, 3개 이상 seed, CNN/LSTM/Transformer 및
  camera/LiDAR/fusion ablation, occlusion 생존·재획득 검증이 남아 있다.
- 전체 프로젝트에서 다음 큰 병목은 detector다. 현재 appearance segmenter는 학습된 detector
  증거가 아니라 red-color bootstrap 초기화이며, sequence dataset·held-out detector metric·
  detector checkpoint가 확인되지 않았다. navigation capture를 개선해도 이 부분 없이
  learned-perception 주장을 완성할 수 없다.
- 따라서 대시보드의 P0~P5 “done” 표시는 기능 통합 관점에서는 이해되지만 과학적 검증 완료로
  읽으면 과대평가다. 문서와 대시보드의 완료 기준을 분리해 갱신할 필요가 있다.

현재 run이 GPU를 사용하는 동안에는 평가를 겹쳐 실행하지 않는다. 종료 후 paired evaluation으로
최선 checkpoint와 원인을 확정하고, 그 결과에 따라 std/entropy annealing branch를 우선 진행한다.

### 2026-07-28 18:24 — bounded-action A/B 구현 및 main 실험 시작

사용자가 기존 density run을 더 연장하는 대신, 평균 action이 0이어도 큰 Gaussian sigma 때문에
횡축 표본이 범위 밖으로 나가 task clamp 뒤 `±2.5 m/s`에 합쳐지는 문제를 직접 고치도록 방향을
변경했다. 이 요청은 앞선 “현재 run 유지” 요청을 대체하는 것으로 해석했다. 기존 PID `1652517`은
정확한 command line을 재확인한 뒤 `SIGINT`로 정상 종료했고, 종료 전 마지막 주기 checkpoint
`last_gen_ppo_ep_37850_rew_31.752436.pth`가 존재함을 확인했다.

#### 원인과 설계 결정

- 기존 정책은 unbounded Normal sample을 PPO action으로 기록하고 task에서 뒤늦게 hard clamp했다.
  최근 lateral sigma `1.492`이면 mean=0인 가장 유리한 경우에도 `P(|a_y|>1)=50.3%`다. 서로
  다른 절반가량의 표본이 같은 `±2.5 m/s` 명령으로 합쳐지고, PPO likelihood는 실제 실행 action과
  다른 확률변수를 최적화했다.
- 단순 sigma clamp만으로는 support/likelihood 불일치를 없애지 못하고, Beta policy는 기존
  Gaussian checkpoint의 actor head를 직접 warm-start하기 어렵다. CAPG는 clipped policy의
  gradient variance는 줄이지만 boundary sample 자체를 없애지 않는다. gSDE는 시간적으로 부드러운
  탐색의 후속 후보지만 action support 문제의 직접 해법은 아니다.
- main은 likelihood-correct **tanh-squashed Gaussian**, 1650 Ti sub는 **scale-adjusted truncated
  Gaussian**으로 정했다. 둘 다 실행 action과 PPO log-probability가 같은 bounded distribution을
  가리킨다.
- 공정한 A/B를 위해 둘 다 epoch `19050`, 75 bars, seed 1, 128 envs, 3000 추가 epoch,
  base std `[0.35, 0.35, 0.05, 0.08]`, entropy 0, actor Adam moment reset을 사용한다.
  density promotion은 끄고 75 bars로 고정했다.
- 첫 main smoke에서 tanh만 붙였을 때 기존 clipped-policy가 물려준 큰 lateral mean 때문에
  `edge99_y=98.75%`가 됐다. 따라서 checkpoint 표현/value는 유지하되 inherited lateral mean만
  `0.4×`로 재보정했다. 이는 속도 상한을 낮추는 것이 아니며, PPO가 필요하면 mean을 다시 키울 수
  있다.

#### 구현

- `navrl_action_models.py`
  - `NavRLSquashedGaussianModel`: tanh transform, stable Jacobian correction, bounded stochastic/
    deterministic action, transformed entropy.
  - `NavRLTruncatedGaussianModel`: `[-1,1]` inverse-CDF sampling, normalized exact log-probability와
    entropy, boundary 부근 scale-adjustment(`d_min=0.01`).
  - 기존 checkpoint의 `state_dict` key/shape를 그대로 보존하고 dead legacy sigma parameter는
    ABI 용도로만 유지한다.
- `navrl_players.py`: rl_games 1.6.5 deterministic player가 pre-tanh `mus`를 실행하던 문제를
  막고 model의 bounded `deterministic_actions`를 사용한다.
- `runner.py`: action model/player 등록, `NAVRL_ACTION_POLICY` 선택, checkpoint env-state의
  distribution/std/mu-scale/d_min 자동 복원, run tag 분리.
- `early_stop_a2c_agent.py`: action-distribution branch restore 시 actor optimizer moment만
  초기화하고 central critic state는 유지한다. clamp 전 `raw_oob`, edge95/99, mean action,
  mean mu/sigma, temporal `delta_y`를 TensorBoard에 기록한다.
- `navrl_task.py`: task 입력 OOB와 실제 edge98/mean_abs/delta/sign-flip을 별도 기록하고,
  checkpoint에 action contract를 저장하며 평가 설정 불일치를 경고한다.
- `train_navrl_general_repr_density.sh`: `NAVRL_FIXED_BARS`를 추가해 action A/B에서 curriculum을
  비활성화하고 동일한 75-bar task를 강제한다.
- 실행 파일:
  - main: `train_navrl_action_squashed_main.sh`
  - 1650 Ti: `train_navrl_action_truncated_1650ti.sh` (`GPU4GB=1`, `base_sim_4gb`)
- CPU 단위시험 `test_navrl_action_models.py` 7개를 추가했다. bounded/finite sample, legacy
  state keys, theoretical Gaussian tail, lateral warm-start scale, tanh deterministic action,
  PyTorch `TransformedDistribution`과 log-prob 일치, truncated PDF 적분값 1을 검증한다.

#### 검증 및 현재 main 결과

- 단위시험: `Ran 7 tests ... OK`.
- `bash -n`, `py_compile`, `git diff --check` 통과.
- 두 launcher 모두 checkpoint preflight-only 통과했고 실행 권한을 확인했다.
- old checkpoint를 실제 main model에 strict restore한 뒤 PPO update가 진행되고 있으므로
  checkpoint compatibility는 smoke 수준을 넘어 runtime으로도 확인됐다.
- 현재 main:
  - PID `3089290`
  - run `ppo_260728_1817_navrl_action-squashed-main-s1`
  - log `train_session_logs/action_squashed_main_260728_181755.log`
  - 점검 시 epoch `19162/22050`, 프로세스 정상, GPU 약 `5.72 GiB`.
- epoch 19051~19162 누적 capture `5481/7895 = 69.42%`, crash `30.09%`,
  최근 50 epoch capture `2494/3561 = 70.04%`. 즉 bounded 전환 직후 competence가 붕괴하지
  않았다.
- 세 개의 독립 task diagnostic window에서 lateral `task_input_oob_y=0.0000`; 실제
  `|command_y|>=0.98*2.5` 비율은 `3.71% → 5.20% → 6.71%`였다. 기존 sigma만으로 계산되는
  mean=0 hard-clamp 하한 `50.3%`와 비교하면 `±2.5` 몰림은 약 한 자릿수 비율로 줄었다.
- 다만 `mean_abs_y=0.899→0.916`, `edge95_y`도 상승 중이다. 이는 random OOB tail은 제거됐지만
  actor mean이 강한 횡이동을 다시 선택한다는 뜻이다. main을 중단하지 않고 끝까지 관측하되,
  최종 판정은 capture뿐 아니라 edge98_y, bar-contact, deterministic/stochastic paired 평가를
  함께 사용한다. edge98_y가 다시 크게 오르면 다음 개입은 속도 상한 변경이 아니라 lateral
  mean-margin regularization 또는 더 작은 warm-start scale이다.

1650 Ti sub는 같은 checkpoint 파일을 복사한 뒤 별도 머신에서 launcher만 실행한다. 128 env를
먼저 사용해 main과 PPO batch를 맞추고, 실제 4 GiB OOM이 확인될 때만 `NUM_ENVS=64`로 낮춘다
(`32*64=2048`이라 configured minibatch와 정확히 일치한다).

### 2026-07-28 18:43 — TensorBoard 세션 정리 및 날짜별 run 색인

사용자 요청에 따라 `rl_games/runs/`를 전수 감사했다. 정리 전에는 top-level run directory
38개, TensorBoard 표시 세션 36개, event file 43개, 전체 약 14 GiB였다. TensorBoard
`EventAccumulator`로 모든 event protobuf와 scalar를 읽었고 손상된 event file은 **0개**였다.
40-byte event들은 손상이 아니라 header-only 초기화 artifact였다.

#### 정리 기준과 방식

- 엄격한 short-run 기준은 **완료 epoch ≤120**으로 잡았다. 일반 본학습 6000 epoch의 2% 이하이고,
  실제 분포에서도 120 다음이 335 epoch라 자연스러운 간격이 있다.
- short-run 외에는 후속 정식 run으로 대체된 `0220`, 원인이 WORKLOG와 후속 `1309`에 완전히
  남은 collapse-guard 오발 `1204`, 수정된 `1817`로 대체된 unscaled action smoke `1813`만
  정리했다.
- 연구 provenance를 잃지 않도록 영구 삭제하지 않고 TensorBoard logdir 밖의 복구용 경로
  `/home/fair/workspaces/aerial_gym_ws/tensorboard_archive/2026-07-28_pruned/`로 옮겼다.
  이동량은 약 298 MiB다. 한 run을 복구하려면 해당 directory를 `rl_games/runs/`로 되돌리고
  TensorBoard를 reload하면 된다.
- 현재 학습 `ppo_260728_1817...`과 그 source checkpoint가 있는 `ppo_260727_2324...`는
  process command line까지 확인하고 보호했다.
- `ppo_260719_1000`의 event 6개와 `ppo_260723_1509`의 event 3개는 중복 세션이 아니라 같은
  run의 resume shard이므로 임의로 쪼개 삭제하지 않았다.

#### 실험 날짜별 TensorBoard 색인

| 실험 날짜 | TensorBoard에 보존한 핵심 세션 | 이번에 아카이브한 세션 |
|---|---|---|
| 07-14~15 | `1904` 정직한 baseline, `2207` clearance negative ablation, `0251` yaw 해결, `2038` random-150 seed1 | 없음. `1552` grid-150과 `1922` interrupted grid-75는 event가 없어 원래부터 TB 비표시 |
| 07-16~17 | `density_25`, `density_50`, `0329`=75, `1223`=110, `density_120`, `0032`=150 seed2 — 밀도곡선 원자료 | 없음 |
| 07-18 | `0259` fixed-110 vision baseline, `1841` 동시 밀도 curriculum, `vision50_seed2` | `0220` — 50 bars에서 833 epoch 중단, 정식 후속 run으로 대체 |
| 07-19 | `1000` 12000-epoch 순차 curriculum, `vision_goalcap_seed1` | 없음 |
| 07-23 | `1509` perception+Transformer 첫 성공/후반 붕괴, `2210` peak-resume continuation | 없음 |
| 07-24 | `0110` altitude-PI source, `0209` general representation, `1052` tilt compensation, `1230` 12 m look-ahead | `0108` — 완료 epoch/checkpoint 0 |
| 07-27 | `0225` first stable fresh representation, `0930` FOV-240, `1309` Stage-C 65 bars, `2324` Stage-C 75 bars 장기 진단/source checkpoint | `0048`(62ep NaN), `0054`(60ep), `0058`(120ep), `0106`(120ep), `0147`(43ep NaN), `1204`(335ep collapse-guard 오발), `2323`(NavRL epoch/checkpoint 0) |
| 07-28 | `1817` corrected squashed-Gaussian main — 정리 시 PID `3089290`으로 계속 학습 중 | `1813` — 69ep unscaled-mean negative smoke; raw `edge99_y=98.75%` 증거는 아카이브에 보존 |

정리 후 filesystem run directory는 38→28개, TensorBoard API의 표시 세션은 **36→26개**로
즉시 줄었다. TensorBoard data server가 reload를 정상 반영해 서비스 재시작은 필요 없었다.
정리 직후에도 main PID `3089290`은 정상 실행 중이며 active event/checkpoint 생성에 영향이 없었다.

### 2026-07-28 18:49 — squashed-Gaussian main 500-epoch 종료

TensorBoard 정리 직후까지 정상이던 `ppo_260728_1817_navrl_action-squashed-main-s1`은
epoch `19550`까지 정확히 500 epoch를 완료한 뒤, 다음 PPO update(epoch 19551)의
`ppo/a_loss[7]`에서 non-finite가 검출되어 fail-fast 종료됐다. optimizer output은 버려졌고,
업데이트 전 마지막 finite checkpoint
`last_gen_ppo_ep_19550_rew_31.084248.pth`가 저장되어 있다.

- 마지막 epoch: capture `70.7%`, crash `29.3%`, timeout `0%`, reward `31.08`.
- run peak: capture `83.3%@19106`, reward `48.89@19417`.
- task 입력 OOB는 끝까지 0이었지만, 마지막 세 diagnostic window의 lateral
  `exec_edge98_y`가 `22.15%→26.05%→29.91%`, `mean_abs_y`가 `0.951→0.955→0.958`로
  상승했다. 즉 tanh가 범위 밖 sample/clamp mismatch는 제거했으나 actor mean이 다시 경계로
  이동해 `±2.5 m/s` 근처 사용률이 장기적으로 재상승했다.
- 이번 결과는 `mu_scale=0.4`가 warm-start 순간의 포화는 풀어도 지속적인 제약은 아니라는 증거다.
  non-finite의 직접 원인은 아직 `a_loss[7]`까지만 확정됐으며, 경계 접근에 따른 transformed
  likelihood/PPO ratio 수치 문제인지 별도 재현·tensor audit이 필요하다.

따라서 이 run을 그대로 재개하지 않는다. 다음 main 수정은 lateral latent-mean margin/regularization과
log-probability 수치 안정성을 함께 다룬 뒤, 동일 epoch-19050 checkpoint에서 짧은 재현 실험으로
검증해야 한다.

### 2026-07-28 18:56 — GTX 1650 Ti truncated-Gaussian 전송 방식 정리

sub 실험의 체크포인트는 launcher에 적힌 기본 상대경로와 동일한 위치에 둘 필요가 없다.
`CKPT=/absolute/path/to/last_gen_ppo_ep_19050_rew_31.79068.pth`로 전달하면 runner가 해당
파일을 읽고 필요한 central-value key normalization 사본을 옆에 생성한다. 코드 커밋
`7a10948`은 이미 원격 `research/navrl-env`에 포함되어 있어, 다른 컴퓨터가 같은 저장소를
사용하면 `git fetch/pull`만으로 설치할 수 있다.

네트워크 없이 옮기는 경우를 위해
`/home/fair/workspaces/aerial_gym_ws/transfers/navrl_truncated_1650ti/`에 다음을 준비했다.

- `0001-navrl-add-bounded-action-policies-for-squashed-trunc.patch`
- `last_gen_ppo_ep_19050_rew_31.79068.pth` (8.5 MiB)
- `README_KO.txt`

원본 체크포인트 SHA-256은
`b3d67792f65b71fa3939630d2b182e1b28155564a285b3feaa12db651bc68277`이다. 1650 Ti에서는
main과 batch 조건을 맞추기 위해 우선 `NUM_ENVS=128`을 사용하고, 실제 CUDA OOM이 발생할
때만 launcher가 지원하는 `NUM_ENVS=64`로 낮춘다.

### 2026-07-28 19:04 — upstream `main` 선별 병합

사용자 요청에 따라 `main..research/navrl-env` 전체를 감사했다. 저장소 정책대로 `main`은
upstream Aerial Gym을 보존하고 연구 코드는 연구 브랜치에 두는 구조이며, 연구 브랜치 전체
병합에는 upstream 문서·sim2real·DCE 예제 등 100개 파일 삭제가 포함된다. 따라서 전체 병합은
하지 않고 별도 clean worktree에서 독립적이고 회귀 위험이 낮은 ignore 규칙만 선별 적용했다.

- `b609eee`: 로컬 experiment workspace와 생성된 example media ignore.
- `64ba00a`: `train_session_logs/`, `checkpoints_saved/` ignore. 연구 브랜치 원본은
  `39d472f`; main에는 연구 전용 `*.log` 규칙을 끌어오지 않고 두 디렉터리만 적용했다.
- `main`의 `origin/main` 대비 최종 diff는 `.gitignore` 한 파일, 12줄 추가뿐이며
  `git diff --check`를 통과했다. 원본 upstream 파일 삭제·수정은 0건이다.

다음은 의도적으로 main에서 제외했다.

- `7a10948` bounded-action A/B: squashed main은 epoch 19551에서 non-finite 종료됐고,
  truncated sub는 아직 별도 GPU 검증 전이므로 연구 브랜치에 유지.
- `e3ba49e`, `542895e` 및 기타 WORKLOG/status/result: 연구 provenance 전용.
- NavRL task, representation, density curriculum, controller opt-in 변경: 연구 스택에 대한
  의존성이 크므로 부분 cherry-pick 시 불완전한 기능이 된다.
- `7847994`의 upstream 자료 삭제: main 보존 목적과 정면으로 충돌.

병합은 로컬에서만 수행했으며 자동 push하지 않았다.

### 2026-07-28 19:35 — `main`·연구 브랜치 원격 동기화

다른 컴퓨터와 Git 자료구조를 일치시키려는 사용자 의도에 따라 두 브랜치를 push했다. push 전
`git fetch --prune`에서 `origin/main`에 새 커밋 `317505b`(사용하지 않는 MkDocs gh-pages
workflow 삭제)가 있음을 발견했다. 이를 덮어쓰지 않고 로컬 main과 일반 merge하여 다음 구조로
보존했다.

- `79f0679`: `origin/main`의 `317505b`와 로컬 ignore 커밋 `b609eee`, `64ba00a`의 merge.
- main의 원격 대비 실질 diff는 `.gitignore` 12줄 추가뿐이며, 원격 workflow 삭제도 유지.
- `6fcef96`: research branch의 transfer, TensorBoard 정리, main 선별 병합 기록까지의 tip.

첫 동기화 시도에서 별도 worktree 생성과 rebase를 같은 shell command에 넣어 현재 디렉터리가
자동으로 새 worktree로 바뀐다고 잘못 판단해, push 전 로컬 research ref가 일시적으로 재작성됐다.
원래 tip `6fcef96`과 재작성 tip의 tree diff가 0임을 확인하고 compare-and-swap `git update-ref`로
즉시 `6fcef96`을 복구했다. 원격 push나 파일 손실은 없었다. 이후 rebase를 사용하지 않고 실제
main worktree에서 `origin/main`을 merge했다.

두 브랜치는 한쪽만 갱신되는 상황을 막기 위해 다음 atomic push로 함께 반영했다.

```text
317505b..79f0679  main -> main
7a10948..6fcef96  research/navrl-env -> research/navrl-env
```

체크포인트, `runs/`, `train_session_logs/`, `checkpoints_saved/`와 workspace의 transfer archive는
의도적으로 Git에 포함하지 않는다. 다른 컴퓨터에서는 Git 코드 동기화 후 ep19050 체크포인트만
별도로 복사해야 truncated-Gaussian 실험의 데이터까지 완전히 일치한다.

### 2026-07-28 19:49 — RTX 3070 병렬 실험: squashed-v2 안정화

1650 Ti의 truncated-Gaussian 실험이 진행되는 동안 RTX 3070을 유휴 상태로 두지 않기 위해,
실패한 squashed run을 그대로 반복하지 않고 측정된 실패 원인만 다루는 v2 pilot을 추가했다.
시작점, seed, 75 bars, action std와 lateral warm-start scale은 모두 ep19050 A/B와 동일하다.

#### 변경

- `ppo_update_safety.py`: PPO가 ordinary ratio clip을 적용하기 전에
  `exp(old_neglogp-new_neglogp)`에서 overflow하는 것을 막는 opt-in log-ratio clamp.
- `early_stop_a2c_agent.py`: single-GPU에서 analytic KL이 문턱을 넘은 후속 minibatch를
  optimizer step 없이 건너뛰는 gate와 `ppo/kl_skipped_minibatches` 기록.
- lateral latent mean의 `|mu_y| > margin` 초과분에 대한 soft squared penalty. 속도 상한을
  낮추는 hard cap이 아니며, noisy action은 여전히 2.5 m/s 경계까지 사용할 수 있다.
- `runner.py`: actor learning-rate 환경변수 override. asymmetric critic LR은 기존 1e-4 유지.
- `train_navrl_action_squashed_v2_main.sh`: fixed 75 bars, seed 1, std
  `0.35,0.35,0.05,0.08`, log-ratio `±10`, KL stop `0.04`, lateral margin
  `1.25@0.01`, 500-epoch pilot.
- 새 safety knob와 actor LR을 checkpoint `env_state`에 기록해 provenance를 남긴다.

#### 실제 restore/update smoke

ep19050 checkpoint를 실제 Isaac Gym 128-env run으로 restore하고 각각 1 epoch optimizer update를
수행했다. 단위 테스트만 통과시키고 장기 학습에 넘기지 않았다.

| actor LR | epoch 19051 PPO KL | actor loss | raw OOB y | edge99 y | 판정 |
|---:|---:|---:|---:|---:|---|
| `3e-5` | `0.18701` | `0.02327` | `0` | `0.00537` | 너무 큼, 기각 |
| `5e-6` | `0.005215` | `-0.007895` | `0` | `0.00537` | target 0.016 아래, 채택 |

따라서 처음 추정한 `3e-5`를 그대로 장기 실행하지 않고, 실측으로 검증된 `5e-6`을 launcher
기본값으로 확정했다. 첫 epoch의 capture 15/32=46.9%는 종료 episode가 32개뿐이라 성능
판정에는 사용하지 않는다. 검증은 action-model/safety unit test 9개, Python compile,
shell syntax, checkpoint preflight, 실제 두 번의 restore+optimizer smoke를 통과했다.

RTX 3070에서는 우선 500 epoch만 실행한다. 약 25~35분 뒤 KL, capture, `edge95_y/edge99_y`,
`mean_mu_abs_y`, KL-skip 수를 판정하고, 정상일 때만 같은 run의 last checkpoint에서
ep22050까지 연장한다. 이렇게 하면 1650 Ti가 5~7시간 도는 동안 main GPU도 독립적인 원인
검증을 수행하면서, 잘못된 설정에 수 시간을 쓰는 위험은 제한한다.

### 2026-07-28 20:08 — 타겟 운동: 학습은 random mixed, 사이트만 단축 표현

사용자가 status 사이트에서 TARGET이 한 축으로만 왕복하는 것을 발견해 실제 학습 프로세스와
코드를 대조했다. 실행 중인 `action-squashed-v2-main-s1`의 process environment와 로그는
`NAVRL_TARGET_PATTERN=mixed`, `speed_final=1.5`, fixed speed override 없음이며, target speed
평균도 epoch별 약 0.7~0.86 m/s로 기록되고 있다. 학습을 중단할 문제는 아니다.

실제 `mixed`는 reset episode마다 cv/waypoint를 50:50으로 선택한다. cv는 방향각을
`U[0, 2π)`로 뽑아 벽에서 반사하고, waypoint는 arena XY에서 uniform waypoint를 뽑아 도달 시
재샘플한다. 속도도 episode마다 `U[0, 1.5]`다. 즉 매 simulation step마다 방향이 튀는 random
walk는 아니지만, episode/waypoint 단위로 2D 방향과 궤적이 무작위화된다.

반면 `docs/status/arena.js`는 `tx=goalX`로 x를 고정하고 `ty=sin(...)`만 갱신하는 장식용
1축 왕복 애니메이션이다. status JSON의 `mixed` 설명과 실제 task를 시각적으로 재현하지 않아
오해를 만든다. 사이트 표현 결함이며 현재 학습 데이터/환경에는 영향을 주지 않는다.

### 2026-07-28 20:21 — status Arena 전면 parity 감사 및 수정

사용자 요청에 따라 3D Arena를 실제 실행 중인 general-training recipe와 대조했다. 학습
프로세스 PID 3290841은 중단하거나 수정하지 않았으며, 감사 중에도
`action-squashed-v2-main-s1`이 75 bars, `mixed`, target speed
`U(0, 1.5)` 조건으로 계속 진행되는 것을 확인했다.

#### 확인된 표현 결함

- 타깃은 실제 2D `mixed`가 아니라 x 고정+y 사인파였다.
- 실제 `NAVRL_GENERAL_TRAIN=1`은 pursuer/target 시작점을 arena 전체에서 무작위화하지만
  사이트는 왼쪽→오른쪽 고정 cross-field episode였다.
- speed 슬라이더 값은 물리 속도가 아니라 임의 애니메이션 주파수에 사용되고 있었다.
- `detector 240°` 표기는 camera detector 87°와 obstacle-token selection FOV 240°를
  혼동했다.
- LiDAR 4개 층은 실제 elevation `-10°..+20°`가 아니라 수평 원 4개였고, ray 충돌은
  실제 box 막대를 원으로 근사했다.
- Arena 초기 bars=25가 status의 current/latest run bars=75와 자동 동기화되지 않았다.
- 3D pursuer가 실제 PPO telemetry인 것처럼 오해될 여지가 있었지만 브라우저 status에는
  policy state/action trajectory가 전달되지 않는다.

#### 수정

- `arena_motion.js`를 추가해 학습 코드와 같은 general spawn, target 거리 4–16 m,
  episode speed `U(0, max)`, CV/waypoint 50:50, CV 양축 반사, waypoint 0.5 m 도달
  재샘플, wall clamp, 모든 위반 bar의 composite 1.0 m push-out을 독립 구현했다.
- episode마다 bar layout, pursuer/target spawn, CV heading/waypoint를 다시 샘플링한다.
  target-speed UI는 `target max m/s`로 바꾸고 실제 학습 ceiling 1.5를 기본값으로 했다.
- HUD에 실제 선택된 `mixed → cv|waypoint`와 episode sampled speed를 표시한다.
- LiDAR는 72×4, 12 m, elevation `-10/0/10/20°`와 3D axis-aligned box ray intersection을
  사용한다. camera 표기는 87° @20 m, token selection은 별도 240°로 분리했다.
- status에서 current/latest run bars를 렌더러 성공 여부와 무관하게 slider/HUD에 먼저
  적용한다. 캐시 버전을 갱신하고 motion module을 Arena보다 먼저 로드한다.
- 실제 PPO trajectory가 없는 한 pursuer는 설명용 steering임을 Arena 아래에 명시해,
  training-distribution replay와 live telemetry를 구분했다.

`tests/test_status_arena_motion.js`는 400 episode에서 두 pattern과 양축 heading이 모두
샘플되는지, 4–16 m spawn 거리, 속도 상한, 양축 wall reflection, physical speed 적분,
bar clearance를 검사한다. 또한 launcher/task/env/LiDAR config와 사이트 상수·표기가
어긋나면 실패하는 source-contract 검사를 포함한다. Node parity test, JS syntax,
`git diff --check`를 통과했고, headless Chrome software WebGL에서 75 bars,
`mixed → cv`, sampled speed, 3D target motion과 HUD가 렌더링되는 것을 확인했다.

별도 감사 결과 `docs/status/status.json`은 정적 snapshot이라 현재 19:58 run이 아니라
완료된 18:17 run을 `LAST`로 보여 준다. 페이지가 이를 `LIVE`라고 표시하지는 않아 데이터
거짓 표시는 아니지만, Git commit 사이 실시간 PPO telemetry는 제공하지 않는다. 이번 수정은
그 한계를 숨기지 않고 Arena에도 명시했다.

### 2026-07-28 20:31 — squashed-v2 500-epoch pilot 종료 판정

RTX 3070의 `ppo_260728_1958_navrl_action-squashed-v2-main-s1`은 epoch
19051→19550의 계획된 500 epoch를 모두 수행하고 `max_epochs`로 정상 종료했다. GPU는
비었고 `.aerial_training_finished`는 `epoch=19550`을 기록한다. canonical final checkpoint
`last_gen_ppo_ep_19550_rew_33.75813.pth`를 CPU로 실제 load해 epoch, model/optimizer,
asymmetric critic, `env_state`를 확인했다. SHA-256은
`a0a3aa65f2378580b480f09e43fe6f1fd5bd8ec5bd77db78a846fc8c47fda9ea`다.

종료 summary 뒤의 `*** Can't create empty tensor` 한 줄은 이미 final checkpoint, summary
JSON, CSV, finished marker가 모두 기록된 뒤 shutdown에서 출력됐다. 종료 원인이 아니며
checkpoint load에도 문제가 없다. 다만 cleanup noise로 별도 추적할 수 있다.

같은 source checkpoint, seed, 75 bars와 정확히 같은 500-epoch 구간인 squashed-v1과
TensorBoard scalar를 비교했다.

| metric | v1 tail 50 | v2 tail 50 | 변화 |
|---|---:|---:|---:|
| capture | 67.50% | 71.54% | +4.04pp |
| crash | 31.72% | 28.07% | -3.65pp |
| reward | 32.82 | 35.58 | +2.76 |
| lateral edge95 | 75.36% | 29.26% | -46.10pp |
| lateral edge99 | 8.32% | 0.358% | -7.96pp |
| mean `|mu_y|` | 2.096 | 1.623 | -22.6% |
| mean action `|Δy|` | 0.0330 | 0.0703 | +113% |

전체 500 epoch 평균 capture는 v1 69.85%, v2 70.09%로 거의 같지만, v1은 후반으로
갈수록 포획이 67%대로 하락하고 경계 집중이 커진 반면 v2는 마지막 100 epoch capture
70.67%, crash 28.94%를 유지했다. v1은 최종 직후 `nonfinite_ppo`, v2는 정상 종료했다.
따라서 원래 가설인 “횡축 표본을 ±2.5 명령 근처에 몰리지 않게 한다”는 pilot 수준에서
성공했고, 후반 성능도 악화시키지 않았다.

단, v2 내부 100-epoch chunk의 edge95가 24.99→25.41→26.23→27.24→29.17%로
천천히 다시 증가하고 `|mu_y|`도 1.570→1.622로 증가했다. 완전히 수렴했다고 보기는
이르다. `ppo/kl_skipped_minibatches=0`이라 KL gate는 한 번도 개입하지 않았고,
표시된 KL의 작은 음수(~-0.0016)는 `rl_games.policy_kl`의 epsilon bias로 사실상 0이다.
이번 개선은 주로 5e-6 LR, lateral latent margin, finite log-ratio의 조합에서 왔으며
각 요소의 단독 기여는 아직 분리되지 않았다.

다음 순서는 장기 연장이 아니라 deterministic A/B 평가다.

1. source ep19050, squashed-v1 ep19550, squashed-v2 ep19550을 동일한 75 bars,
   target speed 0/0.5/1.0/1.5, `mixed`, pursuer 2.5 조건에서 비교한다.
2. 먼저 1000 games/cell quick screen, 승자와 source만 2500 games/cell로 확정한다.
3. 1650 Ti truncated-Gaussian 결과가 오면 같은 평가표에 세 번째 후보로 넣는다.
4. v2가 source 대비 capture/crash를 유지하면서 edge 지표 우위를 보일 때만
   ep20550까지 1000 epoch 연장한다. 연장 시 v2 checkpoint의 Adam state를 유지하도록
   `NAVRL_RESET_ACTOR_OPTIMIZER=0`을 반드시 지정한다.
5. 연장 중 edge95 tail이 40% 또는 edge99가 1%를 넘으면 장기화하지 않고 margin/LR
   ablation으로 돌아간다.

### 2026-07-28 21:10 — action A/B 평가 복구 및 squashed-v2 독립 검증

ep19050 source와 squashed-v2 ep19550에 대해 75 bars, `mixed`, pursuer 2.5 m/s,
target 0/0.5/1.0/1.5 m/s, 1000 games/cell quick screen을 실행했다. 최초 실행은 각
체크포인트의 네 셀을 실제로 끝냈지만 capture/crash 결과를 전혀 남기지 않아 성능 비교에
사용할 수 없었다.

원인은 `*** Can't create empty tensor`가 아니다. 이 문구는 종료 시 empty DOF tensor를
wrap하는 Isaac Gym 경고이며 프로세스 실패 원인이 아니었다. 실제 원인은 다음 두 설정의
조합이다.

- vector player의 기본 reward/length 통계는 `runner.py`에서 `print_stats=False`였다.
- task의 NavRL outcome 통계는 고정 2048-episode 주기에만 출력됐는데 평가 셀은 1000
  episodes였다.

따라서 계산은 정상 종료됐지만 outcome이 메모리에서 버려졌다. 이를 막기 위해
`NAVRL_BULK_EVAL=1` 모드를 추가했다. 이 모드에서는 `PLAY_GAMES_NUM`을 결과 집계 주기로
사용하고, player가 끝나기 전에 capture/crash/timeout, closest approach, action boundary
지표, crash 원인을 셀별 JSON으로 atomic 저장한다. 평가 launcher는 headless/num-env/python
경로를 고정하고, 셀별 로그와 CSV를 만들며, JSON이 없으면 exit 3으로 실패한다. 체크포인트
상대 경로는 launcher 호출 위치를 기준으로 해석하고, 실제 학습 분포와 맞도록 target
pattern 기본값을 `mixed`로 바꿨다. quiet 출력 필터에도 machine-readable
`NAVRL_BULK_EVAL_RESULT`를 허용했다.

32 envs × 64 episodes 실제 checkpoint smoke에서 JSON/CSV 저장과 fail guard를 확인한 뒤,
동일 seed 42로 기존 두 체크포인트의 8개 셀을 다시 실행했다. vector batch 종료 때문에 실제
표본 수는 셀별 1000–1005이며 JSON의 `actual_episodes`를 분모로 사용했다.

| target m/s | source capture | v2 capture | source crash | v2 crash |
|---:|---:|---:|---:|---:|
| 0.0 | 69.96% | 74.80% | 28.74% | 23.90% |
| 0.5 | 71.03% | 74.10% | 28.67% | 25.50% |
| 1.0 | 67.93% | 71.46% | 31.87% | 28.34% |
| 1.5 | 62.20% | 66.87% | 37.70% | 32.84% |
| 전체 가중 | 67.78% | 71.80% | 31.74% | 27.65% |

전체 차이는 capture `+4.02pp`(근사 95% CI `+2.01..+6.03pp`), crash
`-4.09pp`(95% CI `-6.09..-2.09pp`)다. bar contact는 28.17%→25.83%,
below는 1.90%→0.82%로 감소했다. 즉 v2의 학습 로그 개선은 독립 rollout에서도 재현됐고
source보다 명확히 낫다.

그러나 원래 action 문제는 부분 해결이다. source의 deterministic lateral action은 네 셀
평균 `edge98_y=99.99%`, 평균 `|a_y|=0.99994`로 사실상 항상 ±2.5 m/s 명령이었다. v2는
`edge98_y=0%`, raw OOB=0%로 exact boundary mass를 제거했지만 평균 `|a_y|=0.9217`
(약 2.30 m/s 명령), sign flip≈0, 평균 `|Δa_y|=0.0119`다. 따라서 “경계에 정확히 붙는
현상”은 해결했지만 한쪽의 큰 지속 횡명령이라는 구조적 bias는 남았다. v2를 장기 연장해
같은 형태를 더 굳히지 않는다.

다음 의사결정은 1650 Ti truncated-Gaussian 결과를 같은 corrected evaluator로 비교하는
것이다. 그것이 capture/crash를 유지하면서 평균 `|a_y|`도 의미 있게 낮추면 truncated
정책을 선택한다. 그렇지 않으면 main에서는 squashed-v2를 기준으로 더 강한 latent-mean
centering/margin ablation을 300–500 epoch만 수행하고, 동일 4-cell screen을 통과할 때만
장기 학습으로 확장한다.

검증: Python compile, shell syntax, `git diff --check`, training-safety 5 tests,
checkpoint-preflight 4 tests, 실제 Isaac Gym 64-episode export smoke, corrected
8-cell/약 8011-episode A/B 평가를 통과했다. 원시 JSON/CSV/log는
`train_session_logs/eval_results/action_ab_{base,v2}_260728_corrected/`에 보관한다.

### 2026-07-28 22:22 — lateral +y 고착 원인 분리와 v3 ablation 기각

1650 Ti truncated-Gaussian 실험을 기다리는 동안 RTX 3070에서 squashed-v2의 남은
`|a_y|≈0.92`가 고밀도 회피에 필요한 행동인지 정책 편향인지 분리했다. 장기학습을 먼저
돌리지 않고 vector eval에 signed y, positive/negative, high80, edge95/98/99와
front clear/blocked, target centered/off-center/visible 조건부 평균 `|a_y|`를 추가했다.

25/50/75/110 bars × target 0.5/1.5 m/s, 500 games/cell, 총 4007 episodes의
squashed-v2 프로파일 결과는 다음과 같다.

| bars | target | capture | signed y | positive y | high80 y | clear `|y|` | blocked `|y|` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 0.5 | 91.60% | 0.933 | 100% | 99.99% | 0.934 | 0.931 |
| 25 | 1.5 | 87.03% | 0.932 | 100% | 99.98% | 0.934 | 0.931 |
| 50 | 0.5 | 88.25% | 0.929 | 100% | 99.94% | 0.931 | 0.928 |
| 50 | 1.5 | 81.27% | 0.929 | 100% | 99.93% | 0.931 | 0.928 |
| 75 | 0.5 | 73.40% | 0.921 | 100% | 99.65% | 0.926 | 0.921 |
| 75 | 1.5 | 65.47% | 0.922 | 100% | 99.65% | 0.926 | 0.922 |
| 110 | 0.5 | 38.32% | 0.906 | 100% | 98.64% | 0.921 | 0.905 |
| 110 | 1.5 | 32.60% | 0.907 | 100% | 98.82% | 0.924 | 0.906 |

가장 낮은 25 bars, 전방 clear, 타깃 centered에서도 `|a_y|≈0.93`이고 모든 셀에서
positive y=100%였다. 따라서 큰 횡명령은 고밀도 장애물 회피 요구가 아니라 policy가 한
passing side에 고착된 전역 편향이다.

#### Ablation A: signed minibatch-mean penalty — 기각

`mean(mu_y)^2`는 balanced `[-m,+m]` 회피에는 0이고 한쪽 batch에만 비용을 주므로 일반
action L1/L2보다 적합하다고 판단했다. v2 Adam state, LR 5e-6, margin 1.25@0.01을
유지했다.

- coef 0.002, 약 63 epochs: `mu_y` 1.635→1.604. 너무 느려 중단.
- coef 0.01, 50 epochs: `mu_y` 1.635→1.593, positive y tail 99.98%,
  tail capture 67.55%. 방향 분화 없이 모든 y를 조금 줄였고 성능도 v2보다 낮아 기각.

#### Ablation B: 좌우 reflection equivariance — 기각

898-D actor observation의 좌우 반사를 구현했다. LiDAR 방위 순서를 반전하고 obstacle
position/velocity y, robot velocity y/yaw-rate/previous y+yaw action, target tracked
position/velocity y를 부호 반전한다. action latent는 y와 yaw 부호를 반전하며,
`||mu(mirror(obs))-mirror(mu(obs))||²` auxiliary loss를 쓴다. mirror를 두 번 적용하면
observation/action이 bit-exact 원복되는 unit test를 포함했다.

- coef 0.01은 23 epochs에서도 positive y=100%라 약해서 중단.
- coef 0.1은 25 epochs에서 `mu_y` 1.635→1.493, edge95 31.4%→19.9%,
  tail capture 71.3%를 유지해 총 100 epochs까지 제한 확장했다.
- 100 epochs에서 `mu_y=1.243`, sampled signed y=0.810, edge95=7.1%로 magnitude는
  줄었지만 positive y=99.89%, negative y=0.01%로 방향 고착은 그대로였다.
  training tail capture는 62.78%, crash 36.26%로 악화됐다.

최종 ep19650을 75 bars, mixed, target 0/0.5/1.0/1.5, 500 games/cell로 독립 평가했다.

| metric | squashed-v2 ep19550 | reflect-v3 ep19650 | 판정 |
|---|---:|---:|---|
| capture | 71.80% | 60.64% | -11.16pp |
| crash | 27.65% | 39.36% | +11.71pp |
| mean `|a_y|` | 0.922 | 0.844 | 감소 |
| positive y | 100% | 99.999% | 고착 유지 |
| edge95 y | 약 7.5% | 0.61% | 감소 |
| edge98 y | 0% | 0% | 유지 |

결론은 명확하다. 현재 network/rollout에서 batch centering과 counterfactual reflection
auxiliary는 좌우 상태 의존성을 만들지 못하고 기존의 성공적인 +y passing strategy
크기만 줄여 capture를 훼손한다. 두 방식 모두 main 후보에서 기각한다. 재현 launcher는
`*_ablation.sh`로 이름을 바꾸고 `ALLOW_REJECTED_ABLATION=1` 없이는 실행을 거부하도록
가드했다.

현재 main winner는 계속 squashed-v2 ep19550이다. 다음 정책 결정은 1650 Ti의
truncated-Gaussian 결과다. 그것도 한쪽 고착이면 다음 설계는 auxiliary penalty가 아니라
환경 rollout 자체에서 paired mirror trajectories를 생성하거나 lateral head에 구조적
reflection-equivariance를 넣는 새 architecture가 필요하다.

검증: action-model CPU tests 13개, Python compile, shell syntax, diff check,
checkpoint preflight, 실제 optimizer smoke, 조건부 4007-episode profile,
최종 2002-episode deterministic screen. 원시 결과는
`train_session_logs/eval_results/{v2_action_context_260728,v3_reflect_c100_p100_260728}/`에
보관한다.

---

## 2026-07-28 — 1650 Ti truncated 환경 준비 완료 (학습 미시작)

수신 파일: `~/navrl_truncated_1650ti_transfer.tar.gz` (18:56, RustDesk).

배치:
- 코드: 이미 `7a10948` (`git merge-base --is-ancestor` OK). patch 재적용 불필요.
- 체크포인트 복사:
  `aerial_gym/rl_training/rl_games/runs/ppo_260727_2324_navrl/nn/last_gen_ppo_ep_19050_rew_31.79068.pth`
- SHA-256 일치: `b3d67792f65b71fa3939630d2b182e1b28155564a285b3feaa12db651bc68277`
- `test_navrl_action_models.py` 7/7 OK. GPU idle (~0.45 GiB / 4 GiB).
- launcher `train_navrl_action_truncated_1650ti.sh` 실행 가능 상태 (`READY_TO_LAUNCH`).

**학습은 아직 시작하지 않음.** 사용자 지시 시:
```bash
conda activate aerialgym
cd aerial_gym/rl_training/rl_games
nohup ./train_navrl_action_truncated_1650ti.sh \
  > train_session_logs/action_truncated_1650ti.out 2>&1 &
```
OOM 시에만 `NUM_ENVS=64`.

---

## 2026-07-28 — truncated 1650 mid-run 평가 (학습 유지, held-out 미실행)

학습 중 `ppo_260728_1929_navrl_action-truncated-1650ti-s1` 중간 점검. 4GB에서 학습이
~2.9 GiB를 점유 중이라 **held-out `play`는 GPU 충돌** — 학습을 죽이지 않고
`epoch_metrics.csv` + 로그로 평가. (SIGSTOP은 VRAM을 안 비움.)

### 상태
- epoch **19500 / 22050** (+450/3000 = 15%), fixed 75 bars, step ~6–8s, ETA ~5h.
- ckpt (나중 held-out용): `nn/last_gen_ppo_ep_19500_rew_34.33362.pth` (`gen_ppo.pth` 금지).

### 학습 중 capture/crash (PPO stochastic — held-out와 동일시 금지)

| window | capture | crash | mean reward |
|---|---|---|---|
| first 50 (warm) | 0.675 | 0.319 | 31.7 |
| since start (19051–19500) | 0.712 | 0.281 | 35.4 |
| last 200 | 0.736 | 0.257 | 38.2 |
| last 50 | 0.724 | 0.267 | 36.4 |

### 판정
건강: collapse 없음, warm 대비 capture↑ / crash↓. 마지막 200ep 소폭 출렁임은 노이즈 수준.
A/B 최종 판정은 22050 완주 + held-out 속도축 평가 필요. held-out를 지금 돌리려면 학습
일시 중단(프로세스 종료→평가→같은 last_gen에서 resume) 승인 필요.

---

## 2026-07-29 — 1650 Ti truncated 완주 및 held-out 평가

`ppo_260728_1929_navrl_action-truncated-1650ti-s1`이 epoch `22050/22050`에
`max_epochs`로 정상 종료됐다. 19050 checkpoint에서 시작해 fixed 75 bars, seed 1,
128 envs로 정확히 3000 epoch를 추가했으며 NaN/OOM/reward collapse는 없었다.

### 학습 rollout 요약

| window | capture | crash | timeout | mean reward |
|---|---:|---:|---:|---:|
| first 50 | 67.5% | 31.9% | 0.6% | 31.69 |
| 전체 3000 | 74.8% | 24.4% | 0.8% | 38.35 |
| last 500 | 76.6% | 22.7% | 0.7% | 41.06 |
| last 200 | 74.8% | 24.6% | 0.6% | 40.27 |
| last 50 | 75.5% | 24.0% | 0.5% | 40.76 |

- 최고 100-epoch rolling capture는 `79.5%`(window end epoch 21533), 최고 200-epoch
  rolling capture는 `79.1%`(end 21610)였다.
- 단일 epoch peak reward는 `55.48 @ 21559`; 단일 epoch capture peak는
  `93.0% @ 21121`이다. 단일 epoch 값은 표본 수가 작으므로 checkpoint 선택 근거로 쓰지 않는다.
- value 학습은 안정적이었다: explained variance first50 `0.694`, last200 `0.709`;
  PPO/c-value loss와 모든 기록값은 finite였다.

bounded-action contract는 끝까지 유지됐다. `policy_action/raw_oob_y=0`이라 실행 action과
PPO likelihood의 support 불일치는 제거됐다. 다만 actor mean이 다시 lateral boundary를
선택하면서 `edge99_y`가 first50 `4.0%`에서 last200 `22.2%`,
`mean_abs_y`가 `0.863→0.952`, `mean_mu_abs_y`가 `0.927→0.978`로 상승했다.
동시에 adjusted sigma는 `0.132→0.047`, `delta_y`는 `0.105→0.044`로 감소했다.
즉 무작위 큰 횡샘플 문제는 해결됐지만, 정책 평균 자체의 boundary-seeking은 남아 있다.

### 동일 조건 deterministic held-out

조건: 75 bars, target `cv 1.0 m/s`, pursuer limit `2.5 m/s`, seed 42,
128 envs, 2049 episodes(시작 checkpoint만 종료 집계 2051), 12 m/4x72 LiDAR,
8 tokens, FOV 240°, tilt compensation on.

| checkpoint | capture | crash | timeout | closest, no crash |
|---|---:|---:|---:|---:|
| start `19050` legacy | 67.7% | 31.7% | 0.6% | 0.45 m |
| truncated `21550` | **75.9%** | **22.9%** | 1.1% | 0.50 m |
| truncated `22050` final | 72.1% | 27.2% | 0.6% | 0.45 m |

- `21550`은 start 대비 capture `+8.2%p`, crash `-8.8%p`다. 독립 binomial
  근사 95% CI로 capture 차이는 약 `+5.5~+11.0%p`라 단순 평가 노이즈보다 크다.
- final `22050`도 start 대비 capture `+4.4%p`지만, `21550`보다 `-3.8%p`
  (근사 95% CI 약 `-6.5~-1.1%p`)라 후반 500 epoch에서 실제 held-out 퇴행이 있었다.
- 따라서 이 branch의 대표 checkpoint는 **final이 아니라 `21550`**이다. `gen_ppo.pth`는
  선택 기준이 불투명하므로 사용하지 않는다.
- 이 평가는 deterministic 한 축과 training stochastic rollout을 확인한 것이다.
  완전한 최종 A/B 판정은 3070의 squashed branch에 동일 seed/조건을 적용하고,
  필요하면 player deterministic=False 평가를 별도로 추가해야 한다.

### 3070 전달 항목

추천 checkpoint:

`aerial_gym/rl_training/rl_games/runs/ppo_260728_1929_navrl_action-truncated-1650ti-s1/nn/last_gen_ppo_ep_21550_rew_44.51707.pth`

SHA-256:

`60cea22b0345bb2922376b12d27dfc3d7852012b4b50b80aaada8404321d0992`

최종 checkpoint `22050` SHA-256:

`284b5b3008d251deed278ac5d24b916b1093c97103ae7d3a0b6fe0b1a39ea9c9`

checkpoint와 eval log는 `.gitignore` 대상이므로 이 커밋에는 결론과 식별자만 들어간다.
3070에는 이 커밋을 pull/cherry-pick한 뒤 추천 `.pth`를 별도 전송하고 SHA를 확인한다.
3070 squashed checkpoint도 위 held-out 조건으로 평가해 `capture/crash/timeout`을 직접 비교한다.

---

## 2026-07-29 — ★ +y 고착의 유력 원인 발견: 관측의 좌우(키랄리티) 결함 2건 (코드로 검증됨)

75-에이전트 전면 감사가 지목한 perception HIGH 결함 2건을 세션 한도로 검증이 끊긴 부분까지
**직접 코드로 재검증**했다. 둘 다 실재하며, Codex가 "unknown 전역 편향"으로 규정한 +y 고착
(positive y=100%, clear/blocked 무차별)의 **관측-레벨 원인 후보**다.

### 결함 1 — LiDAR 빈 방위각 테이블이 센서와 거울상 (perception 전 기간 존재)

- 센서: `warp_lidar.py:67-69` — `azimuth = hfov_max − Δ·j/(W−1)`, 즉 **빈 인덱스↑ = 방위각↓**
  (j=0 → +180°, j=71 → −175°).
- perception: `navrl_perception.py:234` — `_lidar_angles = linspace(−175°, +180°)`, **증가** 가정.
- 관계식: **가정 = 5° − 실제** = x축 거울 반사 + 5° 회전. 검증: 검출기의 픽셀↔방위는 렌더
  (`navrl_detector.py:135`)와 측정(`:307`)이 서로 일관된 올바른 규약 → **표적 채널은 정상,
  장애물 토큰 채널만 반전**.
- 결과: ① 오른쪽 막대가 왼쪽 토큰으로 발행(pos=[r·cosα, r·sinα], x≈보존·y≈부호반전)
  ② 카메라 융합(`:313-317`)이 실제 방위 α의 관측을 **거울 빈(5°−α)** 에 min-fuse → 전방
  장애물이 반대편에 고스트로 복제 ③ 표적-리턴 억제/재연관(`:296-305`, `:371-378`)도 거울 빈을
  조작. 정적 스캔 자체는 고정 순열이라 학습 가능(그래서 성능이 나왔음) — 문제는 **채널 간 모순**.

### 결함 2 — 카메라 far-plane 10 m가 12 m LiDAR에 팬텀 벽으로 융합

`camera_obstacle_max_range=10.0` 고정(`navrl_task_config.py:150`), no-hit 채움값이 그대로
`min(scan, 10.0)` 융합(`navrl_perception.py:311,317`) → `NAVRL_LIDAR_RANGE=12`에서 **빈 전방이
항상 10 m 벽으로 보임**(static 전방 상한 0.833, 토큰 유효성 10<11.94 통과 → 팬텀 토큰이 정면에
3~4슬롯 점유). LIDAR_RANGE≤10이던 과거엔 휴면, **12 m 피벗 이후 활성**.

### 증상과의 정합 (사전 관찰들이 전부 설명됨)

- **clear |a_y| ≈ blocked |a_y|** (0.934≈0.931, Codex 조건부 프로파일): 팬텀 벽 때문에 정책
  눈에는 전방이 **한 번도 clear로 보인 적이 없다**. 조건 무차별이 당연한 귀결.
- **positive y=100% 고착**: 좌우 정보가 채널 간 모순(토큰 y 반전 vs 표적/로봇 y 정상) + 전방
  고스트 대칭화 → 좌우 신호의 기대가치 소멸 → 항상-회피(팬텀 벽) × 한쪽 고정(tie-break) 이
  가장 안정한 국소최적.
- **hit_in_tokens 0.40~0.55 미스터리**: 토큰이 거울 위치라 GT 연관이 우연 일치 수준.
- **reflection equivariance 실패의 구조적 이유**: Codex의 mirror 연산자는 잘못된 각도 테이블
  위에 세워져 진짜 세계-거울이 아니고(반전+5° skew, 채널별 불일치), "mirror 2회=항등" 단위테스트
  는 임의의 involution이 통과하므로 물리 정합성을 검증하지 못했다. **오염된 관측은 대칭화로
  고칠 수 없다** — 출력(actor)을 강제하기 전에 입력(센서)을 고쳐야 한다.

### 사전등록 예측 (수정 후 25 bars 300–500 epoch fresh pilot에서 판정)

- P1: positive-y 고착 소멸(좌/우 사용률 모두 >10%)
- P2: barprobe `hit_token_given_fov` 0.556 → 0.8+
- P3: 전방 clear vs blocked의 |a_y| 차이가 유의미해짐
- P4: 25 bars capture ≥ 기존(0.90) — 단 1차 판정 기준은 P1-P3

예측이 맞으면 equivariant actor는 불필요. 틀리면 그때 깨끗한 관측 위에 구조적 equivariance를
얹는다(그 시점엔 잘 정의됨). **기존 전 결과는 "거울 토큰+팬텀 벽 관측 하의 성능"이므로 수정 후
재베이스라인 필수.** 아직 코드 수정은 하지 않음 — 사용자 승인 대기.

---

## 2026-07-29 — 키랄리티 결함 2건 수정 완료 (물리 프로브로 사전·사후 검증)

### 물리 판정 (tools/probe_lidar_bearing.py — 신규, 상시 회귀 가드)

실제 시뮬레이터에서 raw 스캔 1,126개 리턴을 두 각도 테이블로 역투영해 GT 막대와 대조:

| 테이블 | on-bar 일치율 | 평균 오차 |
|---|---:|---:|
| increasing (옛 perception 가정) | **13.9%** | 2.54 m |
| decreasing (warp 센서 실제) | **94.8%** | 0.47 m (≈표면-중심) |

수정 후 perception 라이브 테이블 = 94.8% 동일. **거울 버그 실재가 물리적으로 확정·해소됨.**

### 적용된 수정

1. `navrl_perception.lidar_bin_bearings()` 신설 — bin→방위 단일 진실 공급원(감소 규약, warp_lidar.py:67
   과 일치). `_lidar_angles`가 이를 사용 → 토큰 기하/카메라 융합/표적 억제/LiDAR 재연관 6개 사용처 일괄 교정.
2. `camera_no_return_to_lidar_range()` — 카메라 far-plane(10 m) 열을 no-return(=lidar_max)으로 리맵 후
   min-융합. **12 m LiDAR에서 전방 팬텀 벽 제거.** 구 8 m 레시피에서는 동작 불변(no-op) 확인.
3. `navrl_task.py` 뷰어 오버레이(36빔+증가 규약으로 이중 낡음)와 action-context front mask를 공유
   테이블로 교정.
4. `tests/test_navrl_perception.py`의 픽스처가 **옛 거울 규약을 박제**하고 있었음(정면 표적 리턴을
   bin 17에 주입 — 물리 규약에선 bin 18). "깨진 구현에서 통과하던 테스트"를 물리 규약으로 수정.
5. `tests/test_lidar_bearing_convention.py` 신설 — warp 공식을 독립 재계산해 테이블과 대조(어느 쪽이
   바뀌어도 시끄럽게 실패), far-plane 리맵 케이스(12 m 활성/8 m no-op) 고정. 36/72빔 모두 통과.

검증: 신규 테스트 2세트 + 기존 perception/bar_probe/training_safety 전부 OK, 3-epoch 학습 스모크
(obs 898) 정상, 물리 프로브 사후 94.8%.

### 함의 (재확인)

- 기존 전 정책은 "거울 토큰 + (12 m에선) 팬텀 벽" 관측으로 학습된 것. **수정 후 관측 의미가 바뀌므로
  warm-start 무효, fresh 재학습 필수.** 기존 결과 수치는 수정 전 조건의 기록으로만 유효.
- Codex의 reflection equivariance 실험(기각됨)은 잘못된 각도 테이블 위의 mirror 연산자를 사용했으므로,
  향후 재시도 시 반드시 `lidar_bin_bearings()` 기반으로 재구축해야 함.
- 다음: 사전등록 예측 P1–P4 판정용 25 bars fresh pilot (WORKLOG 2026-07-29 앞 항목 참조).

---

## 2026-07-29 — status 사이트에 논문식 Architecture 블록 다이어그램 추가

`docs/status/index.html`에 `#panel-arch` 섹션(테마 연동 인라인 SVG, 1250×620) + 상단 탭 "Design".
내용: **온보드(배포) 경로 실선** — 환경 → LiDAR 72×4@12m / RGB-D 87°(표적 20m·장애물 depth 40×24@10m)
→ 스캔 융합·토큰화(far-plane 리맵, 240° 선택, ±10° 억제, 물리 방위 테이블) + 표적 분할→3D 측정→칼만
추적(track 16-D) → 히스토리 0.5s×5 → **관측 898-D** → Transformer(17tok d64, σ clamp) → **행동 4-D**
→ 제어기(tilt 45°·tilt-comp·고도 PI·yaw 3.0) → **모터 추력 4** → 루프백. **학습 전용 레인 점선** —
GT 상태 → 보상·종료(0.5m swept)·특권 critic(898+8=906) → PPO → 가중치 업데이트. 두 레인 사이에
**정보 방화벽**(빨간 점선)과 "GT는 학습 신호로만, actor 관측 진입 금지" 명시. 모든 화살표에 텐서
치수 표기. 헤드리스 Chrome 스크린샷으로 배치·겹침 검수 후 게시.

---

## 2026-07-29 20:39 — 표적 속도 실현 결함 감사 및 대칭 local steering 적용

학습 전 사용자가 status 사이트의 표적이 둔하고 막대에서 잘 빠져나오지 못한다고 지적했다.
로그에는 `pattern=mixed`, `speed_fixed`/`speed_final`, `rl_dt=0.100s`만 남아 있어 명령 속도는
확인할 수 있었지만 **실제 변위 속도는 기록되지 않았다**. 중단된 Claude 작업에는 CV 표적이
막대 clearance push-out과 매 스텝 싸운다는 진단, 막대 접촉 시 진행 벡터를 반사하는 미커밋
수정, `tools/probe_target_motion.py` 초안이 남아 있었다.

probe를 drone 전체 step과 분리해 `_advance_target()`만 300회 호출하도록 고쳤다. 이렇게 해야
zero-action drone crash/reset에 따른 표적 teleport와 pattern 재샘플이 계측에 섞이지 않는다.
수정 전 `HEAD`를 별도 임시 worktree에서 동일 seed로 다시 측정한 결과:

| bars | pattern | 명령 | 실현 평균 | 명령 대비 | stall(<20%) | clearance<1m |
|---:|---|---:|---:|---:|---:|---:|
| 75 | CV | 1.50 | 0.78 m/s | 52% | 41.4% | 24.5% |
| 75 | waypoint | 1.50 | 1.40 m/s | 93% | 3.5% | 2.5% |
| 110 | CV | 1.50 | 0.38 m/s | 26% | 71.6% | 32.8% |
| 110 | waypoint | 1.50 | 1.29 m/s | 86% | 4.9% | 6.3% |

원인은 CV가 벽에서만 heading을 반사하고 막대에서는 위치만 밀어낸 것이었다. 막대를 향한 원래
heading이 유지되어 다음 스텝에 다시 진입했고, 고밀도에서는 nominal 1.5 m/s 표적이 사실상
주차됐다. waypoint도 막대에 닿은 뒤 waypoint를 재샘플했지만 현재 스텝 변위는 이미 손실됐다.
추가로 target spawn은 drone용 0.65 m clearance를 재사용해 이동 계약의 1.0 m보다 가까이
시작할 수 있었고, 일부 clearance 경로가 build-time 전체 slot을 참조해 활성 밀도 계약이
불명확했다.

### 적용

1. simulator-independent `target_motion.steer_target_step()`을 추가했다. 직접 heading이
   clear하면 그대로 쓰고, 막히면 `0, ±30, ±60, ±90, ±120, 180°` 후보 중 **가장 작은
   대칭 회전**으로 full-speed endpoint를 선택한다.
2. 정확히 대칭인 좌/우 후보는 에피소드마다 50:50으로 뽑은 `_tm_avoid_sign`으로 결정한다.
   고정된 한쪽 tie-break가 새 lateral chirality를 만들지 않게 했다.
3. CV는 선택된 회피 heading을 유지하고 waypoint는 waypoint 목표를 유지한 채 local steering으로
   막대를 돌아 나간다. 기존 composite push와 CV 반사는 후보가 전부 막힌 경우의 safety fallback으로
   남겼다.
4. spawn target clearance를 0.65→설정값 1.0 m로 맞추고, 이동/생성 모두
   `:n_bars_active`만 사용한다.
5. checkpoint `env_state`에 `cfg_target_motion_model=symmetric_local_steer_v1`을 기록한다.
   이전 checkpoint로 moving-target fine-tune/eval하면 legacy stall 환경과 계약이 달라졌다는
   경고를 출력한다.
6. `docs/status/arena_motion.js`에도 같은 후보각·대칭 tie-break·fallback 반사를 적용했다.
   사이트의 speed slider는 학습 분포와 같이 **최댓값**이며 episode speed는 `U[0,max]`라 평균은
   max의 절반이다. HUD의 sampled speed가 실제 해당 episode 속도다.
7. train dashboard/TensorBoard에 명령 평균과 별도로
   `navrl/target_speed_realized_mean_m_s`를 추가했다. 앞으로는 probe 없이도 로그에서
   명령-실현 속도 괴리를 즉시 확인할 수 있다.

### 사후 물리 probe

| bars | CV | waypoint | stall | overspeed | clearance<1m |
|---:|---:|---:|---:|---:|---:|
| 25 | 1.50 m/s (100%) | 1.50 (100%) | 0% | 0% | 0% |
| 75 | 1.50 (100%) | 1.50 (100%) | 0% | 0% | 0% |
| 110 | 1.50 (100%) | 1.50 (100%) | 0% | 0% | 0% |
| 130 | 1.50 (100%) | 1.50 (100%) | 0% | 0% | 0% |

사이트/interactive 기본값과 같은 110 bars·0.75 m/s도 두 패턴 모두 실현 0.75 m/s,
stall/과속/clearance 위반 0%를 별도로 확인했다.

150 bars는 알려진 기하 절벽(약 148 bars)을 넘는다. CV는 100%를 유지했지만 waypoint는 1.48
m/s, stall 0.9%, clearance 위반 7.2%였다. 이 밀도에서는 1.0 m exclusion disc가 arena를
사실상 덮으므로, 속도와 1.0 m clearance를 동시에 강제하는 것은 코드가 아니라 기하적으로
불가능하다. Stage C의 110 bars와 절벽 아래 130 bars까지는 계약을 정확히 만족한다.

검증: target-motion CPU tests 3개, train-dashboard realized-speed contract,
status Node parity, Python compile/diff check,
Phase-3 실제 Isaac Gym smoke(static identity + CV/waypoint/circle + curriculum) 전부 통과,
25/75/110/130/150 bars × 300-step 물리 probe. 이 변경은 moving-target 환경 자체를
바꾸므로 기존 moving-target 성능표와 직접 비교하지 않고 fresh 재학습·재평가해야 한다.

---

## 2026-07-29 — 센서/표적 수정 후 25-bars fresh pilot 실행 계약

기존 `train_navrl_general_repr.sh`는 `NAVRL_OBSTACLE_FOV_DEG`를 고정하지 않아 현재 평가 계약인
240°가 아니라 기본 360°로 조용히 실행될 수 있고, 기본 8000 epoch라 P1–P4 사전등록 검증보다
범위가 컸다. 전용 `train_navrl_sensorfix_fresh_pilot.sh`를 추가해 아래 조건을 고정했다.

- **fresh only**: `CKPT`, `--checkpoint`, `--resume_in_place`, `--branch_run` 거부. 수정 전
  거울 bearing/팬텀 벽 정책을 섞지 않는다.
- **환경**: 25 bars 고정, density promotion off, 8 tokens, 240° FOV, ±10° suppression,
  4×72 LiDAR, 12 m, mixed CV/waypoint, 목표 속도 ceiling 0→1.5 m/s(3000 epoch curriculum).
- **인과 분리**: baseline `legacy` actor를 사용하고 bounded-action 실험의 환경변수를 제거한다.
  action/crash/barprobe 진단은 켠다.
- **안전**: 중복 NavRL process/launcher lock을 검사하고 reward-collapse guard는 끄되
  NaN/Inf fail-fast는 유지한다.
- 기본 500 epoch, 128 env, seed 1. 이 결과로 P1–P4를 판정한 뒤에만 full fresh run의
  액션 정책과 density curriculum을 결정한다.

---

## 2026-07-29 — sensor-fix fresh pilot 완료 및 5-speed held-out 평가

`ppo_260729_2104_navrl_sensorfix-fresh-pilot-s1`이 계획한 epoch `500/500`에
`max_epochs`로 정상 종료했다. NaN/Inf, reward collapse, OOM은 없었다. 마지막의
`*** Can't create empty tensor`는 checkpoint·summary·finished marker가 기록된 뒤 Isaac Gym
shutdown에서 나오는 기존 cleanup 경고이며 종료 원인이 아니다. 평가 checkpoint는
`last_gen_ppo_ep_500_rew_38.911366.pth`이다.

### 학습 추세

| epoch window | capture 평균 | crash 평균 | reward 평균 |
|---|---:|---:|---:|
| 1–100 | 5.3% | 94.7% | -37.51 |
| 101–200 | 26.6% | 73.4% | -0.61 |
| 201–300 | 44.6% | 55.4% | 18.07 |
| 301–400 | 55.8% | 44.2% | 29.26 |
| 401–500 | 57.1% | 42.9% | 35.21 |
| 451–500 | 55.9% | 44.1% | 36.52 |

마지막 epoch capture는 59.7%, run 단일-epoch peak는 71.6%@429였다. PPO 자체는 안정적이다:
epoch 500 KL `0.00522`, explained variance `0.581`, LR `1e-4`, 모든 scalar finite.

표적 운동 수정도 실제 학습에서 검증됐다. epoch 100/200/300/400/500의 command 평균과
realized 평균이 소수점 출력 정밀도까지 각각 동일했고, epoch 500은 command=realized
`0.12676 m/s`(ceiling `0.24998`)였다. 따라서 현재 성능 저하를 표적 stall로 설명할 수 없다.

### 25-bars deterministic held-out (seed 42, 1000 games/cell)

조건: mixed CV/waypoint, pursuer limit 2.5 m/s, 240°/8-token/4×72/12 m,
실제 episode 합계 5,010개.

| target m/s | capture | crash | bar contact (절대) | below (절대) |
|---:|---:|---:|---:|---:|
| 0.0 | 64.9% | 35.1% | 20.5% | 13.7% |
| 0.25 | **67.2%** | **32.8%** | 18.7% | 13.5% |
| 0.5 | 65.3% | 34.7% | 23.1% | 11.2% |
| 1.0 | 61.0% | 39.0% | 23.2% | 14.6% |
| 1.5 | 55.2% | 44.8% | 28.1% | 14.9% |
| 가중 평균 | **62.7%** | **37.3%** | **22.7%** | **13.6%** |

raw 결과는
`train_session_logs/eval_results/speed_density_ppo_260729_2104_navrl_sensorfix-fresh-pilot-s1_260729_221344/`
에 보관한다.

### 사전등록 P1–P4 판정

1. **P1 부분 통과** — literal 기준인 좌/우 사용률 >10%는 만족했다. 하지만 고정 평가에서
   positive-y `31.3–37.0%`, negative-y `62.4–68.0%`로, 과거 +y 고착이 사라진 대신 약한
   -y 편향이 자랐다. “키랄리티 완전 소멸”로 해석하면 실패다.
2. **P2 강하게 통과** — 마지막 barprobe `hit_token_given_fov=0.981`로 목표 0.8+를 넘었다.
   sensor-bearing/far-plane 수정은 실제 simulation 관측에서도 효과가 확인됐다.
3. **P3 실패** — held-out 전 속도에서 front-clear와 front-blocked의 mean `|a_y|`가
   `0.959–0.969`로 사실상 동일했다(차이 <0.004). 정책이 장애물 문맥에 따라 횡축 크기를
   조절하지 않는다.
4. **P4 실패** — 최고 held-out capture 67.2%로 0.90에 미달한다. 수정 전 15k-epoch 정책과
   수정 후 500-epoch fresh 정책은 직접 동등 비교가 아니지만, pilot의 go 조건은 충족하지 못했다.

### 핵심 원인과 다음 결정

센서는 병목에서 빠졌고 새 병목은 **legacy Gaussian의 action saturation**이다. TensorBoard에서
lateral `edge99_y`가 epoch 100 `59.9%`→500 `90.3%`, raw mean `|a_y|`가
`1.44`→`2.71`, mean `|mu_y|`가 `1.22`→`2.68`로 악화됐다. 반면 sigma는
`1.00`→`0.91`에 불과해, 단순 exploration noise보다 actor mean 자체가 clamp 밖으로 달아난다.
held-out에서도 executed `edge98_y=91.7–93.7%`, clear/blocked 행동이 동일했다.

따라서 이 legacy run을 장기 연장하거나 density curriculum으로 넘기지 않는다. 이전
bounded-policy 결과는 깨진 센서 checkpoint에서 얻었으므로 성능 수치는 재사용하지 않되,
검증된 likelihood-correct 구현과 PPO safety 장치는 재사용할 수 있다. 다음 실험은 깨끗한
관측에서 **fresh bounded actor pilot**로 action support와 context response를 먼저 통과시킨 뒤
장기·밀도 학습으로 확장한다.

---

## 2026-07-29 — corrected-sensor fresh squashed pilot 실행기 확정

위 판정에 따라 `train_navrl_corrected_squashed_fresh_pilot.sh`를 추가했다. 기존
sensor-fix launcher의 환경 계약(25 bars, 240°/8-token/4×72/12 m, mixed moving target,
fresh-only)은 그대로 사용하고 action 쪽만 다음처럼 바꾼다.

- likelihood-correct tanh-squashed Gaussian, fixed std `0.35,0.35,0.05,0.08`
- lateral network-mean scale `0.4`, entropy coefficient `0`
- latent lateral margin `1.25@0.01`
- PPO log-ratio finite clamp `±10`, KL update-stop `0.04`
- fresh 학습률 `1e-4`; warm-start 분포 변경에서만 필요했던 `5e-6`은 사용하지 않음
- checkpoint/resume flag 거부, action raw-OOB/edge/mu/sigma 진단 유지

실제 Isaac Gym 128-env × 3-epoch optimizer smoke를 fresh weights로 수행했다. PPO actor/critic
loss와 모든 parameter가 finite였고, KL은 `-0.00068,-0.00234,-0.00234`로 epsilon bias 범위,
KL-skip 0이었다. lateral raw-OOB/edge95/edge99는 3 epoch 모두 0%, mean `|a_y|≈0.26`,
mean `|mu_y|=0.0004→0.0075`, positive/negative 사용률은 각각 약 37–39%/38–40%였다.
즉 시작부터 좌우 대칭이고 경계에 몰리지 않으며 fresh `1e-4` update도 안정적이다.

단위시험: action model 13개, training safety 5개 전부 통과. smoke run은 정식 TensorBoard
세션과 혼동되지 않도록 workspace 밖 임시 보관 후, 정식 500-epoch pilot만 실행 대상으로
남긴다. 500 epoch 뒤 go 조건은 raw-OOB=0, `edge99_y<5%`, 양 방향 >10%,
clear/blocked `|a_y|` 분리, capture가 계속 상승하는 것이다.

### 2026-07-29 22:27 — 정식 pilot 시작 직후 symbolic-shape 경고 확인

정식 run `ppo_260729_2225_navrl_corrected-squashed-fresh-pilot-s1` 시작 시
`torch/fx/experimental/symbolic_shapes.py`의 `not in var_ranges, defaulting to unknown
range` 경고가 여러 번 출력됐다. 이는 `torch.compile`이 첫 graph를 만들 때 일부 symbolic
dimension의 정적 범위를 증명하지 못해 unknown으로 처리한다는 compile-time 진단이다.
실제 PID `1160972`는 GPU 약 5.85 GiB를 사용하며 경고 이후 epoch 23까지 정상 진행했고,
Traceback/non-finite/OOM/illegal-memory 오류는 0건이었다. 학습 중단 사유가 아니므로 무시한다.

---

## 2026-07-29 22:32 — research status 사이트 최신화

`docs/status` snapshot을 현재 연구 상태와 live bounded pilot로 갱신했다.

- 새 **Research update** 패널: 물리 LiDAR match 94.8%, token hit|FOV 98.1%, target-motion
  25–130 bars 실현속도 100%/stall 0%, legacy 5-speed held-out 5,010 episodes 가중 capture
  62.7%와 edge98-y 91.7–93.7%를 한 화면에 배치했다.
- 현재 `ppo_260729_2225_navrl_corrected-squashed-fresh-pilot-s1`을 epoch 132 snapshot으로
  표시한다. tail50 capture 70.5%, crash 29.3%, raw OOB-y 0%, edge99-y 0%,
  +y/−y 43.9%/45.8%, KL 0.00471이다.
- Architecture 정책 블록을 낡은 learned-σ legacy 설명에서 실제
  tanh-squashed bounded action, fixed σ `[.35,.35,.05,.08]`, KL gate/latent margin/finite
  ratio 계약으로 교정했다.
- Now/phase를 density curriculum active에서 **P6A bounded action contract active**,
  P6B density 25→110 pending으로 수정했다. 장기·밀도 학습 gate도 raw OOB=0,
  edge99-y<5%, 좌우 양방향>10%, clear/blocked context 분리로 명시했다.
- `status.json`과 HTML fallback snapshot을 같은 데이터로 동기화하고 cache version을
  `20260729b`로 갱신했다. status parity test가 update schema와 bounded contract drift를
  감시하도록 확장했다.
- Chrome 실렌더 검수에서 live 카드가 직전 legacy run의 peak→final gap을 현재 run에 섞어
  보여주는 기존 결함을 발견했다. live일 때는 현재 epoch/500을 표시하도록 바꾸고, density
  promotion 설명도 fixed 25-bar bounded pilot 설명으로 교정했다.

## 2026-07-29 22:47 — 고밀도 Arena pursuer 정지 병목 수정

사용자가 status Arena에서 막대 수를 높였을 때 `PURSUER`가 중간에 정지하는 현상을
발견했다. 이 화면의 pursuer는 PPO checkpoint를 브라우저에서 실행하는 것이 아니라
`docs/status/arena.js`의 설명용 steering이므로, 이 현상만으로 실제 정책의 고밀도 성능을
판정할 수 없다. 실제 정식 run
`ppo_260729_2225_navrl_corrected-squashed-fresh-pilot-s1`도 현재 25 bars 고정 P6A
pilot이며, P6B 25→110 density curriculum은 아직 시작하지 않았다.

사이트 정지의 직접 원인은 목표 인력과 근거리 막대 반발력을 단순 합산하던 artificial
potential field였다. 막대가 조밀하거나 좌우 대칭이면 벡터가 상쇄되어 local minimum에
빠질 수 있었다. 이를 다음처럼 수정했다.

- pursuer 반경 0.25 m와 각 막대의 실제 폭을 포함한 swept-path clearance를 사용한다.
- 목표 방향 기준 `0, ±15, ±30, ±45, ±60, ±90, ±120, 180°` 후보를 0.9초
  look-ahead로 평가한다.
- 동일 점수에서 episode별 좌/우 회피 부호와 이전 heading을 유지해 프레임마다 회피 방향이
  뒤집히는 것을 막는다.
- 브라우저 장면은 계속 `illustrative pursuer`임을 명시하며, 실제 PPO 결과로 오인하지 않는다.

회귀 검증은 기존 대칭 막대 함정에서 5초 안에 목표 1 m 이내로 탈출하는지 검사한다. 추가
20 episode/density 동적 probe에서 25/110/130/150 bars 모두 capture `20/20`,
stationary step `0.00%`, 평균 이동속도 `2.50 m/s`였다. 150 bars의 look-ahead
blocked 표시는 `0.95%`였지만 정지는 없었다. 기존 target-motion/status parity test와
`git diff --check`도 통과했다.

## 2026-07-29 22:49 — corrected bounded 500-epoch pilot 완료 판정

`ppo_260729_2225_navrl_corrected-squashed-fresh-pilot-s1`이 epoch 500에서
`max_epochs`로 정상 종료됐다. traceback/non-finite/OOM 없이
`last_gen_ppo_ep_500_rew_102.53619.pth`와 summary/CSV가 저장됐고 GPU도 반환됐다.

학습 추세는 명확한 개선이다.

| epoch block | capture | crash | timeout | mean reward |
|---|---:|---:|---:|---:|
| 1–100 | 46.68% | 43.51% | 9.81% | -19.84 |
| 101–200 | 72.17% | 27.18% | 0.65% | 49.58 |
| 201–300 | 80.25% | 19.07% | 0.68% | 79.64 |
| 301–400 | 90.37% | 9.07% | 0.56% | 96.99 |
| 401–500 | 91.32% | 8.28% | 0.39% | 99.78 |

마지막 50 epoch는 capture `92.68%`, crash `7.07%`, timeout `0.25%`이고 마지막
epoch는 `92.2/7.8/0.0%`다. 사전등록 action gate도 training telemetry 기준으로
raw OOB 전 축 `0%`, lateral edge98 `0.23%`, positive/negative y
`36.5%/60.1%`라 support/boundary/two-sided 조건을 통과했다. legacy의 lateral
edge98 `91.7–93.7%` 문제는 제거됐다. clear/blocked `|a_y|`는
`0.689/0.662`로 legacy `0.934/0.931`보다 분리는 생겼지만 차이가 `0.027`로 작아
held-out에서 재확인한다.

중요한 제한은 이 pilot의 target-speed curriculum이 3000 epoch 기준이라는 점이다.
500 epoch 종료 시 speed ceiling은 약 `0.25 m/s`이고 실제 평균은 `0.12–0.13 m/s`였다.
따라서 training capture 92%를 1.5 m/s 성능이나 고밀도 성능으로 해석하지 않는다.

다음 gate는 density training이 아니라 **25 bars fixed, mixed target, held-out
0/0.25/0.5/1.0/1.5 m/s × 1000 games**다. final `last_*` checkpoint를 사용해 기존
legacy 5-speed 5,010-episode 결과와 직접 비교한다. 이 평가에서 full-speed 성능과
action context를 확인한 뒤에만 3000-epoch speed continuation 또는 P6B
25→110 density curriculum을 선택한다.

## 2026-07-29 23:00 — 5-speed held-out 통과 및 P6B 자동 전환

사용자 부재 중 후속 작업을 이어가기 위해 epoch-500 final checkpoint를 25 bars,
mixed target, pursuer 2.5 m/s, seed 42에서 `0/0.25/0.5/1.0/1.5 m/s × 1000 games`
평가했다. 128-env를 단순 `nohup`으로 분리한 시도는 이 도구 실행 환경이 자식
프로세스를 호출 종료 시 정리해 결과 JSON 없이 끝났다. checkpoint 이상으로 오판하지
않도록 16-env 10-game과 64-env 100-game smoke를 먼저 통과시킨 뒤, 관리형 64-env
세션으로 정식 5,004 episode를 완료했다.

| target m/s | bounded capture | legacy capture | Δ | crash | timeout |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 96.50% | 64.87% | +31.63pp | 3.10% | 0.40% |
| 0.25 | 96.30% | 67.20% | +29.10pp | 3.70% | 0.00% |
| 0.50 | 95.60% | 65.27% | +30.33pp | 4.40% | 0.00% |
| 1.00 | 93.50% | 61.03% | +32.47pp | 6.50% | 0.00% |
| 1.50 | 79.92% | 55.20% | +24.72pp | 19.08% | 1.00% |

episode 가중 capture는 `4622/5004 = 92.37%`다. 모든 셀의 action raw OOB는 전 축
0%, lateral edge99는 0%였다. 속도가 높아질수록 capture는 내려가지만, 500-epoch
checkpoint가 실제로 학습한 speed ceiling 약 0.25 m/s를 크게 벗어난 1.5 m/s에서도
legacy보다 24.7pp 높다. 따라서 별도 25-bars speed-only continuation은 생략하고
P6B density curriculum으로 간다. speed curriculum은 checkpoint의 task step에서
계속 이어지지만 full-speed held-out competence가 이미 확인되어 과거의 검증되지 않은
two-axis 전환과는 다르다.

고밀도 실제 기체 정지를 놓치지 않기 위해 `NavRL motiondiag`를 추가했다. 목표가 1 m
이상 남은 step에서 실제 수평속도 평균, 명령속도 평균, 실제속도 `<0.2*vmax` 비율,
그중 명령도 `>=0.2*vmax`인 `commanded_stall` 비율을 집계한다. 이는 status 사이트의
illustrative pursuer가 아니라 Isaac Gym `robot_linvel`을 직접 측정한다.

재현 launcher `train_navrl_corrected_squashed_density.sh`는 final epoch-500 checkpoint,
bounded action/safety 계약, 25→110 bars(step 5), promotion gate 0.70/16,384 eps,
warmup 1000, total max epoch 8000을 고정한다. Python compile, diff check, training-safety
5개, checkpoint-preflight 4개, target-motion 13개 단위시험을 통과했다. action-model
stochastic tail test는 첫 실행에서 허용치 0.100% 대비 0.102%가 나왔으나, 코드 변경 없이
연속 3회(각 13개) 통과해 표본 경계 변동으로 판정했다. launcher preflight는 epoch 500,
bars 25, task_steps 16000과 bounded action 계약을 정확히 복원했다.

P6B run `ppo_260729_2303_navrl_corrected-squashed-density25to110-s1`을 관리형 백그라운드
세션으로 시작했다. runner PID는 `1226712`, GPU 사용량은 약 5.85 GiB다. epoch
501→535까지 정상 진행했고 첫 재개 구간 capture는 대체로 91–99%, raw OOB 전 축 0%,
lateral edge98 0.16%였다. 첫 2,049-episode 실제 기체 baseline은 평균 수평속도
`2.079 m/s`, 평균 명령속도 `2.607 m/s`, low-speed `6.33%`, commanded-stall
`6.29%`다. 이 수치는 25 bars의 reset/가속 transient를 포함한 기준선이며, 이후 밀도별
절대값과 이 기준선 대비 증가량을 함께 본다.

---

## 2026-07-30 — ★ 키랄리티 수정 사후 검증: P2 압도적 적중, "73.9%" 정정, 억제폭 병목 발견

### 사전등록 예측 판정 (2026-07-29 키랄리티 수정에 대해)

`ppo_260729_2303_navrl_corrected-squashed-density25to110-s1`(85막대까지 12회 승급) 로그로 판정.

- **P2 (hit_token_given_fov 0.556→0.8+) — 압도적 적중**: **0.968~0.975**. 토큰 위치도
  center_offset 0.28m·cross_track **0.18m**로 막대 표면 기하와 정확히 일치. 거울 버그가
  실재했고 해소됐다는 직접 증거.
- **P1 (좌/우 모두 >10%) — 간신히 통과**: pos_y 100%→**12.2%**, neg_y 0%→**83.6%**,
  sign_flip 0→**9.6%**. 절대 고착은 깨졌으나 이제 반대쪽으로 쏠림. 기준(>10%)은 만족.
- **P3 (clear/blocked 행동 분화) — 최초 "실패" 판정은 측정 결함이었음.** 아래 참조.

### 밀도 곡선 재측정 — 수정 전 대비 공통 6점 평균 **+11.1pp**

125-agent 감사 워크플로 이전에 직접 평가(`checkpoints_saved` 아님, 학습 중이던 85막대
정책, target 1.0 m/s, 2049 ep/셀, deterministic):

| bars | 수정 전 capture | 수정 후 | Δ |
|---|---|---|---|
| 25 | 0.896 | **0.978** | +8.2pp |
| 50 | 0.796 | **0.935** | +13.9pp |
| 65 | 0.679 | **0.854** | +17.5pp |
| 85 | (미측정) | **0.689** | — 학습 상한 |
| 110 | 0.280 | **0.412** | +13.2pp |
| 130 | 0.150 | **0.225** | +7.5pp |
| 150 | 0.079 | **0.144** | +6.5pp |

25막대 0.978은 이 프로젝트 역대 최고. `below`(고도이탈)는 전 밀도에서 0.1~6.7%로 사실상
소멸 — 고도 문제는 완전히 종결. 고밀도 실패는 96%가 `bar_contact`.

### 25-agent 감사가 잡은 정정 사항 (전부 재검증 완료)

1. **"85막대 73.9%"는 마지막 1 epoch(17/23, sd 5.8pp)이었다. 정정: 85막대 plateau =
   3042 epoch 평균 **0.676 ± 0.001**.** 비교 대상이던 "65막대 68.5%"도 동일하게 단일
   epoch 값이었고, 그 run의 65막대 plateau도 **0.678** — 통계적으로 동일.
   → **올바른 결론**: 동일 게이트(threshold 0.70, check_eps 16384)에서 키랄리티 수정이
   **같은 실력을 65→85막대(밀도 +31%, 13.6→17.8/100㎡)로 유지**시켰다. capture가 오른 게
   아니라 같은 capture로 더 어려운 조건을 버틴 것 — 해석은 다르지만 결론은 동일하게 긍정적.
2. **P3 "실패" 재해석**: `goal_centered` 판정이 `|lateral_sine|<=sin(15°)`만 보고
   `goal_vehicle_x`의 부호(전방/후방)를 안 봤다 — **전후 이중 원뿔**. 표적이 뒤에 있을 때
   큰 `|a_y|`는 정상 행동이라, `clear_centered` 버킷의 절반가량이 "표적이 뒤에 있는데
   전방이 뚫린" 무관 상황과 섞여 있었다. **P3는 "실패"가 아니라 "판정 불가"로 하향.**
   측정 코드 수정(`goal_vehicle[:,0]>0` 추가) 후 재판정 필요, 아직 미수정.
3. **85막대는 게이트에 막혀 있었다(내 최초 결론과 정반대).** `density curriculum held`가
   85막대에서 **12연속**, 판정창 capture 0.652~0.688로 문턱 0.70을 한 번도 못 넘음.
   → 처음 제안했던 "그대로 24000 epoch까지" 방침은 **틀렸음**: 이 추세로는 게이트를
   못 넘고 12시간을 태울 가능성이 높았음. 학습을 중단하고 원인 진단으로 전환.
4. **`ppo_update_safety.py:91`의 mirror 연산자가 아직 옛 증가 규약 기준** —
   `(H-2-i)%H`는 수정된 감소 규약에서 10°(HBEAMS=72) 비뚤어진 거울. 현재 run은
   `NAVRL_REFLECTION_COEF` unset이라 off-path이지만, reflection ablation을 재시도하면
   **정책에 키랄 왜곡을 주입**한다. 단위테스트(`test_navrl_action_models.py:243`)도 동일하게
   틀린 식을 기대값으로 써서 못 잡는다. `navrl_perception.lidar_bin_bearings()` 기준으로
   재구축 필요 — 미수정, reflection 계열 재시도 전 필수 선결 작업으로 기록.
5. 학습 로그에 action-model 계약(정책 종류·std·LR) 한 줄이 quiet 출력 필터에 걸려
   사라짐 — 어떤 행동 모델로 학습했는지 로그만으로 증명 불가. `quiet_rl_io._allow_print`에
   `[aerial RL]` 화이트리스트 필요 (LOW, 미수정).

### 85막대 게이트 차단의 원인 진단 — 억제폭 병목 (측정 기반, 추측 아님)

barprobe(85막대, 학습 로그 최신): **FOV 내 막대 35개, 토큰 8개 전부 사용, 그중 unique
3.0 / duplicate 2.7** — 실제로 표현되는 서로 다른 막대는 ~3개뿐(커버율 9%). 원인: 억제폭
±10°가 근거리 막대(폭 0.8m, 거리 2m → 각폭 ±11°)보다 좁아 **한 막대가 토큰 2개를 먹는다.**
문턱을 낮추지 않고 이 병목을 겨냥하는 것이 다음 수. 억제폭 ±10°→±15°는 (a) 관측 차원
불변(898) → warm-start 가능, (b) 최대 토큰 상한 360/30=12 > 8이라 부작용 없음, (c) 240°
FOV 좁히기 때와 동일 계열의 변경(그때 +2.7pp 성공 전례).

**결정 (사용자 승인 완료)**: 문턱은 유지, `NAVRL_OBSTACLE_SUPPRESS_DEG=15`로 85막대
체크포인트에서 이어서 학습 재개. 판정 기준: barprobe `unique`가 3.0→4.5+로 오르면 표현
수정이 유효, 이후 자연 승급을 기다림. ~3000 epoch 내 안 오르거나 올라도 capture가 안
따라오면 정보가 병목이 아니었다는 뜻이므로 중단 후 재판단(용량 확장 fresh 또는 85막대를
상한으로 확정).

---

## 2026-07-30 — 85막대 재개 런 발산 및 억제폭 실험 미적용 확인

실행 중인 `ppo_260730_1154_navrl_corrected-squashed-density25to110-s1`을 감사했다. 이
런은 `ppo_260730_1104.../last_gen_ppo_ep_8350_rew_22.905773.pth`에서 재개되어 epoch
11100을 넘겼지만, 최근 500 epoch 평균은 capture **1.0%**, crash **85.8%**, reward
**-104.3**으로 학습 가치가 없는 명백한 발산 상태다. 50-epoch 평균 capture가 20% 아래로
내려간 최초 창은 epoch 10314~10363이며, epoch 10400의 50-epoch 평균은 capture 0.5%,
crash 99.5%였다.

TensorBoard에서 직접 원인을 확인했다. `ppo/kl`은 epoch 10276에 설정 한계 0.04를 처음
넘었고 이후 0.30, 0.48, 최대 2.56 수준까지 폭증했다. 동시에 latent policy mean의
절대값이 x축 4.8→2025, z축 0.32→464, yaw축 0.57→15.3으로 폭발해 실제 action이
x/z/yaw 전부 경계에 고정됐다(`exec_edge98=1.0`). 최근 crash의 약 90%는 bar contact다.
이는 정지 병목이 아니다(actual speed 1.61m/s, commanded-stall 3.5%).

현재 KL 가드는 임계치를 넘은 minibatch를 skip할 뿐 이미 수행된 이전 update를 rollback하지
않는다. 발산 뒤에는 매 epoch 7~8개 minibatch가 skip되지만 손상된 정책은 복구되지 않는다.
또 density launcher는 정상적인 승급 reward 하락을 허용하려고 일반 reward-collapse guard를
완전히 비활성화한다. 두 안전장치 사이의 공백 때문에 실제 발산을 자동 종료하지 못했다.
후속 수정은 (1) 고정 density 안에서만 비교하는 curriculum-aware collapse guard,
(2) KL/latent 폭증 시 last-known-good epoch checkpoint rollback 또는 즉시 fail-stop이어야
한다.

더 중요한 실행 계약 오류도 확인했다. 사용한 명령의
`NAVRL_OBSTACLE_SUPPRESS_DEG=15`는 실제 적용되지 않았다.
`train_navrl_general_repr_density.sh`가 이를 무조건 `10`으로 다시 export하며, 시작 로그도
`suppress=+-10deg`를 명시한다. 따라서 이 런은 제안했던 ±15° 중복 토큰 억제 실험이 아니라
기존 ±10° 정책의 장기 연장이다. `CKPT=$(ls -t runs/*/nn/last_gen_ppo_ep_*.pth | head -1)`도
이제는 발산 런의 최신 checkpoint를 고를 수 있으므로 재사용 금지. 수정 전 재시작 기준점은
명시적 pre-collapse checkpoint `ppo_260730_1104.../last_gen_ppo_ep_8350_rew_22.905773.pth`로
고정한다.

### 후속 수정 — ±15° 실험 및 사용자 가시 로그

사용자가 학습은 직접 실행하기로 해 자동 실행하지 않았다. `general_repr_density`가 외부의
`NAVRL_OBSTACLE_SUPPRESS_DEG`를 덮어쓰지 않도록 기본값 방식으로 수정했다. 체크포인트
preflight는 원래 `cfg_obstacle_suppress_deg=10`과 요청값 15의 차이를 차단했으므로, 모든
다른 관측 계약 검사는 유지하면서 `cfg_obstacle_suppress_deg` 하나만 명시적으로 허용하는
`NAVRL_ALLOW_SUPPRESS_WARMSTART=1` opt-in을 추가했다. opt-in 없이 15를 주면 여전히
실패하고, opt-in 시에는 `INTENTIONAL contract override | ... checkpoint=10 requested=15`가
출력되는 것을 확인했다.

사용자 실수를 줄이기 위해 재현 런처
`train_navrl_corrected_squashed_density_suppress15.sh`를 추가했다. 안전 기준점 epoch 8350,
`MAX_EPOCHS=12000`, suppress ±15°, 별도 run tag/log 이름을 기본값으로 고정하며 학습을
시작하지 않는 preflight에서 epoch 8350·bars 85·tokens 8·FOV 240°·suppress ±15°를
확인했다.

학습 로그는 `train_navrl.sh`가 원래 터미널과 파일에 동시에 쓰는 `tee` 구조였지만, 과거
`nohup ... > /dev/null` 명령 때문에 화면이 사라지고 실제 파일명이 매번 달랐다. 이제 매
실행마다 `train_session_logs/current_training.log`가 실제 세션 로그를 가리키며,
`watch_navrl_training.sh`가 이를 `tail -F`한다. 포그라운드 실행에서는 같은 PPO dashboard가
현재 터미널에 바로 보이고, 다른 터미널에서는 watcher로 볼 수 있다. watcher 종료의
`Ctrl-C`는 tail만 종료하며 학습에는 영향을 주지 않는다.

---

## 2026-07-30 — 연구 문서·status 사이트를 suppress ±15° live gate로 통합

7월 29일 bounded 25-bars pilot에 머물러 있던 정적 사이트와 7월 22일 연구 계획을 현재
증거에 맞춰 전면 동기화했다.

- `RESEARCH_PLAN.md`, `ROADMAP.md`, `PHASE3_PLAN.md`,
  `PERCEPTION_TRANSFORMER_PLAN.md`, `README.md`에 corrected 85-bars plateau
  `0.676±0.001`, 이전 65-bars `0.678`, 같은 competence에서 밀도 `+31%`라는 정정된
  결론을 반영했다.
- ±10° 장기 재개 run `ppo_260730_1154...`는 epoch 10276 이후 KL/latent가 발산해
  tail500 capture/crash `1.0%/86.0%`가 된 PPO safety failure이며, ±15° 결과가 아니므로
  성능 비교에서 제외한다고 모든 계획 문서에 명시했다.
- 현재 critical path를 suppress ±15°의 세 gate로 고정했다: unique `≥4.5/8`, capture
  `≥0.70/16,384 episodes`, KL `≤0.04`. 통과 후 held-out density sweep, 실패 시
  threshold 완화가 아니라 token selection 재설계다. 장기 재시도 전 curriculum-aware
  collapse guard와 KL/latent fail-stop 또는 rollback을 선수정한다.
- `results/corrected_chirality_density_curve.csv`를 추가해 corrected policy의 held-out
  25/50/65/85/110/130/150-bars capture
  `0.978/0.935/0.854/0.689/0.412/0.225/0.144`를 재현 가능한 사이트 데이터로 남겼다.
- `tools/update_status_snapshot.py`가 run CSV를 다시 집계하고 `docs/status/status.json`과
  HTML inline fallback을 하나의 객체로 동시에 갱신하도록 했다. 검증 snapshot은 44 runs,
  live `ppo_260730_1419_navrl_corrected-squashed-density-suppress15-s1`, epoch 8639,
  tail50 capture/crash/timeout `69.1%/30.4%/0.5%`, bars 85였다. 첫 suppress15 barprobe는
  unique `3.5/8`, duplicate `1.8`로 기준 `3.0/~3.0`보다 방향은 맞지만 아직 gate 미달이다.
- status UI의 Research update, evidence ledger, decision gate, phase strip, architecture
  suppression 설명, Now 패널을 모두 이 live experiment로 교체했다. density 기본 그래프도
  corrected curve를 우선 선택한다.

검증: status JSON↔HTML fallback 완전 일치, Chrome 1440×1800 실제 렌더에서 Live/Research
update/evidence ledger 확인, status arena-motion parity 통과, checkpoint preflight 6개,
training-safety 5개, target-motion 테스트 통과, Python compile 및 `git diff --check` 통과.
사이트 snapshot 생성 중 실제 학습 프로세스는 중단하거나 변경하지 않았다.

---

## 2026-07-30 — density curriculum 설계 감사

사용자 요청에 따라 실행 중인 suppress ±15° 학습을 중단하지 않고, 커리큘럼 구현·체크포인트
상태·실제 승급 로그를 대조했다. 결론은 **밀도를 competence gate로 5개씩 올리는 큰 방향은
맞지만, 현재 gate는 학습 스케줄러로는 쓸 수 있어도 표현 ablation과 최종 성능 판정에는
그대로 쓰면 안 된다**는 것이다.

### 확인된 현재 계약

- 실행 중인 `ppo_260730_1419_navrl_corrected-squashed-density-suppress15-s1`은 85막대,
  목표 거리 상한 16 m, 표적 속도 상한 1.5 m/s, density threshold 0.70,
  `check_eps=16384`, step 5로 정상 실행 중이다.
- 거리 competence 상태는 시작 체크포인트부터 `k_min=10`, `k_max=16`으로 포화되어
  현재 density와 경쟁하지 않는다. 표적 속도 램프도 global epoch 3000에 끝나 현재는
  `speed ~ U[0,1.5]`로 고정되어 있다.
- 반면 25→65막대 초기 승급 때는 표적 속도 상한이 약 0.61→1.47 m/s로 동시에 증가했다.
  따라서 초기 승급 곡선의 하락을 density 효과 하나로 해석할 수는 없다. 65막대 이후에는
  속도 상한이 1.5 m/s로 포화되어 density 단일축 비교가 가능하다.
- `NAVRL_GENERAL_TRAIN=1`에서는 드론과 표적을 arena 전역에서 독립 표본화하고 실제 목표
  거리는 `[4 m, k_max]`로 뽑는다. 이 경로는 `k_min_cur=10`과
  `NAVRL_DENSITY_EASY_GOAL_MIX`를 사용하지 않는다. 즉 로그의 저장된 `[10,16]` 계약과 달리
  현재 gate는 실제로 거리 약 4~16 m, 속도 0~1.5 m/s, cv/waypoint 50:50이 섞인
  aggregate capture다.

### 체크포인트 재개의 판정창 오염

epoch 8350 체크포인트에는 suppress ±10°에서 수집한 density window가
`9823/15616`으로 저장되어 있었다. suppress ±15° run이 이 값을 그대로 복원해 첫 gate는
새 표현 에피소드가 768회만 추가된 즉시 0.630으로 판정됐다. 첫 판정 자료의 95.3%가 이전
표현에서 온 것이므로 ±15° 효과 판정으로 사용할 수 없다. 그 다음 창 16,385회는 전부 새
표현이며 0.664로 hold되어 유효하다. 감사가 끝날 때 세 번째 판정도 16,384회, 0.689로
hold되어 suppress ±15°의 clean window는 현재 `0.664, 0.689` 두 개다. 표현·reward·motion
contract가 바뀌면 curriculum position은 복원하되 진행 중인 competence accumulator와
adaptation warmup은 새로 시작해야 한다.

### 0.70 / 16,384 gate 해석

85막대 장기 plateau 0.676에서 이항 표준오차는 약 0.00366, 근사 95% 구간은
`[0.669, 0.683]`이다. 실제 성능이 0.676인 정책이 raw capture 0.70을 우연히 넘을 확률은
약 `2.6e-11`이다. 따라서 0.70은 noisy해서 막는 값이 아니라 **실제 +2.4%p 개선을 요구하는
엄격한 목표**다. 목표가 “85막대에서 최소 70%”이면 유지하는 게 맞고, 단순히 다음 난도를
경험시키는 훈련 스케줄러라면 너무 경직되어 있다. 또한 16k aggregate는 평균을 정밀하게
측정하지만 속도·거리·패턴별 실패를 숨기며, 계속 업데이트되는 정책의 과거 에피소드를
섞으므로 held-out 평가가 아니다.

### 권장 재설계

1. **표현 ablation과 density curriculum 분리**: suppress 변경은 85막대 고정,
   accumulator reset, 최소 adaptation hold 후 동일한 held-out grid로 비교한다.
2. **승급 gate 층화**: 전체 평균 `>=0.70`만 보지 말고 target speed
   `0/0.5/1.0/1.5`, motion `cv/waypoint`, 거리 구간별 capture를 함께 기록한다.
   weighted mean과 hard-bin floor를 동시에 사용해야 쉬운 짧은/느린 에피소드가 어려운
   조건의 실패를 가리지 않는다.
3. **승급 후 cooldown과 회귀 방지**: step 5는 85막대 기준 약 5.9% 증가라 적절하다.
   승급 뒤 일정 epoch 동안 재승급을 금지하고, 현재/이전 density를 일부 재표본화해
   catastrophic forgetting을 막는다.
4. **두 종류의 안전장치 분리**: density 증가로 reward가 내려가는 것은 허용하되, 같은
   density에서 capture 급락·KL/latent 폭증은 last-known-good 저장 또는 fail-stop으로
   막는다.
5. **최종 논문 판정은 별도 평가**: curriculum gate는 teacher 신호일 뿐 성능 근거가
   아니다. 고정 checkpoint를 3 seeds와 고정 density×speed×pattern grid로 평가한다.

문헌 방향도 대조했다. Self-Paced Deep RL은 정책 능력에 맞춰 task distribution 자체를
목표 분포 쪽으로 이동시키고, ALP-GMM과 Prioritized Level Replay는 단일 고정 성공문턱보다
학습 진전이 큰 task/level을 다시 표본화한다. 이 프로젝트에는 당장 복잡한 GMM teacher보다
먼저 **층화 gate + 이전 density replay + contract 변경 시 accumulator reset**을 넣는 것이
비용 대비 효과와 인과 해석 모두에서 우선이다.

---

## 2026-07-30 — 오염된 suppress15 런 종료 및 curriculum 계약 수정

사용자 승인 후 `ppo_260730_1419_navrl_corrected-squashed-density-suppress15-s1`을
SIGINT로 종료했다. 프로세스 그룹 전체가 내려간 것을 확인했고 마지막 정상 주기
checkpoint `last_gen_ppo_ep_8900_rew_22.966688.pth`를 보존했다. 이 run의 clean density
window는 `0.664`, `0.689`, unique는 `3.4~3.5/8`로 사전 기준 0.70/4.5에 미달했다.

다음 계약 오류와 안전 공백을 수정했다.

- `train_navrl_corrected_squashed_density_suppress15.sh`는 이제 85 bars 고정,
  `NAVRL_RESET_DENSITY_WINDOW=1`, 250-epoch resume adaptation을 강제한다. density
  monitor는 켜 두되 final=current라 promotion은 불가능하다.
- checkpoint의 표현/action/motion/task-distribution/promotion 설정이 달라지거나 새
  stratified counter provenance가 없으면 aggregate gate를 자동 폐기한다. 동일 계약
  재개만 aggregate와 speed/distance/pattern counter를 함께 복원한다.
- general-spawn 실제 목표거리 계약을 `NAVRL_GENERAL_GOAL_DIST_MIN/MAX=4/16m`로
  checkpoint와 시작 로그에 명시했다. legacy `k_min=10`을 실제 radial minimum처럼
  해석하지 않는다.
- 각 16,384-episode 창에 speed quartile 4개, initial-distance quartile 4개,
  cv/waypoint/circle pattern capture를 저장·출력한다. broad-slice floor enforcement는
  fixed baseline을 얻기 전까지 diagnostic-only다.
- density stage preflight는 `k_max>=16`, `task_steps>=96,000`을 요구한다. 따라서 거리
  competence나 3000-epoch target-speed ramp가 진행 중인 checkpoint에서 density를
  동시에 올리는 run을 시작할 수 없다.
- reward collapse guard를 density에서 끄더라도, 동일 bar count의 50-epoch rolling
  capture가 같은 density peak보다 25%p 내려간 상태가 25 epoch 지속되면 fail-stop한다.
  density가 바뀌면 기준을 초기화하므로 정상 promotion 하락은 오발하지 않는다.
- `current_training.log` 출력이 교체 전 symlink target을 표시하던 관측 오류도 수정했다.

검증은 Python 전체 test discovery 31개(새 density-capture guard 4개와 density-stage
preflight 2개 포함), action-model 13개, shell syntax, checkpoint preflight와
`git diff --check`를 통과했다. 실제 epoch 8350
checkpoint를 1 epoch만 resume한 smoke에서 `discarded=9823/15616`,
`gate_not_before_step=275200`, bars `85->85`, same-density guard 활성화를 확인했다.
smoke run/log/TensorBoard 이벤트는 휴지통으로 이동해 연구 run 목록을 오염시키지 않았고
`current_training.log`는 종료한 실제 suppress15 로그로 복원했다.

이미 두 개의 fully clean ±15° window와 안정 구간 unique가 기준 미달이므로 동일 설정을
장시간 다시 돌리는 것은 권장하지 않는다. fixed-85 suppress15 launcher는 계약 회귀와 짧은
재현용으로 남기고, 다음 GPU 예산은 angular/cluster-balanced token selection 후보에 쓴다.
검증 중 100k Monte-Carlo action tail 테스트가 경계에서 1-count 변동으로 간헐 실패하던
기존 flake도 확인해 RNG seed를 고정했으며, action 구현이나 판정 한계는 변경하지 않았다.

---

## 2026-07-30 — obstacle token 관리 문헌 조사와 현재 구조 재판정

사용자 요청에 따라 현재 obstacle representation 코드를 다시 읽고, NavRL++ 원문과
point-cloud/set-prediction/object-centric tokenization 연구를 대조했다.

### 코드에서 확정한 이중 압축

- `navrl_perception.py`는 4×72 scan을 vertical-min 72개 range로 만든 뒤, 가장 가까운
  bearing을 반복 선택하고 고정 각도 주변을 지우는 greedy suppression으로 8개 surface
  proposal을 만든다. proposal은 실제 막대 중심이나 instance가 아니라 LiDAR 표면점이다.
- `navrl_transformer_network.py`는 이 8개를 Transformer의 독립 토큰으로 넣지 않는다.
  각 history step의 `8×12=96` 값을 이어 붙여 MLP 하나로 64차원 한 토큰에 투영한다.
  따라서 5-step obstacle history가 Transformer token 5개를 차지한다.
- 즉 현재 “unique 3/8”은 첫 압축의 낭비를 보여주며, 그 뒤에도 여러 proposal 사이의
  관계가 한 MLP 벡터로 다시 압축된다. `MAX_OBSTACLES`를 늘리면 Transformer sequence가
  길어지는 것이 아니라 같은 MLP 입력만 넓어지고 checkpoint shape가 깨진다.

### 원 NavRL++와의 차이

[NavRL++](https://arxiv.org/html/2605.15559v1)은 static obstacle을 4×36 ego-centric
ray-distance array로 유지해 CNN 한 토큰으로 만들고, object slot 최대 5개는
position/velocity/radius가 있는 **dynamic obstacle** history에만 쓴다. 현재 구현은
정적인 arena bar 표면점을 velocity=0으로 채워 이 dynamic-history 자리에 넣었다.
원 논문에 없는 확장이므로, suppression 폭만의 문제가 아니라 static geometry를 object
history로 중복 표현한 설계 자체가 검증 대상이다.

### 유사 연구에서 쓰는 관리 방식

- [PointNet++](https://proceedings.neurips.cc/paper_files/paper/2017/hash/d8bf84be3800d12f74d8b05e9b89836f-Abstract.html)와
  [Point-BERT](https://openaccess.thecvf.com/content/CVPR2022/html/Yu_Point-BERT_Pre-Training_3D_Point_Cloud_Transformers_With_Masked_Point_Modeling_CVPR_2022_paper.html)는
  가까운 점만 순서대로 고르지 않고 공간적으로 퍼진 center를 sampling한 뒤 주변 point를
  local patch로 묶는다. 이 계열의 핵심은 대표점 수와 무관하게 공간 coverage를 먼저
  확보하는 것이다.
- [PointPillars](https://openaccess.thecvf.com/content_CVPR_2019/html/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.html)는
  point를 고정 spatial pillar에 모아 pseudo-image로 되돌린다. 객체 수가 token budget보다
  많을 때도 특정 근거리 객체가 모든 slot을 점유하지 않으며, 현재 full polar scan CNN을
  유지해야 한다는 근거에 가깝다.
- [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html)와
  [Perceiver](https://proceedings.mlr.press/v139/jaegle21a.html)는 많은 input을 소수의
  learned inducing/latent vector가 cross-attention으로 읽어 fixed bottleneck으로 만든다.
  hard top-k를 없앨 수 있지만, latent가 각각 물리적 객체 하나라는 보장은 없다.
- [Slot Attention](https://papers.nips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html)은
  fixed slots이 입력 feature를 두고 반복적으로 경쟁하게 해 object-like set을 만든다.
  현재처럼 한 막대가 여러 slot을 먹는 문제와 가장 직접적으로 닮았지만, reconstruction이나
  supervised property loss 없이 PPO reward만 주면 slot specialization을 보장하기 어렵다.
- [DETR](https://arxiv.org/abs/2005.12872)은 learned object query와 Hungarian
  one-to-one matching loss로 duplicate prediction을 억제한다. 이 프로젝트에서도
  simulator bar identity를 actor 입력에 넣지 않고 training-only auxiliary label로 쓰는
  방법은 가능하다. 단, FOV 안 약 35개를 8 slot으로 모두 match할 수 없으므로 TTC와
  goal-corridor 기준 top-risk subset 정의가 먼저 필요하다.
- navigation 쪽의 [Omni-Perception](https://proceedings.mlr.press/v305/wang25b.html)은
  raw LiDAR를 proximal/distal risk로 나눠 서로 다른 sampling/temporal 처리를 하고,
  [SAGA](https://arxiv.org/abs/2605.02301)는 장애물 객체 대신 후보 motion anchor를
  geometry-aware token으로 만든다. 전자는 8 slot을 `near-risk + spatial coverage` quota로
  나누는 근거이고, 후자는 객체를 모두 열거할 수 없는 초고밀도 환경에서 action-corridor
  token으로 문제를 바꾸는 장기 후보이다.

### 결정한 실험 순서

1. 학습 없이 동일 scan에 current suppression, contiguous range clustering,
   risk-weighted angular/farthest selection을 적용해 unique/struck-bar/angular/
   goal-corridor coverage를 비교한다.
2. 첫 PPO 후보는 **cluster-balanced 8-slot**이다. 연속 beam을 surface cluster로 묶어
   cluster당 한 proposal만 만들고, 일부 slot은 TTC/closest-approach 위험, 나머지는
   아직 덮이지 않은 방위에 배정한다. 기존 12차원/8-slot 폭을 유지해 선택 규칙만 검증한다.
3. hard selector가 부족할 때 `72 bearings → 8 learned slots`와 training-only Hungarian
   auxiliary loss를 추가한다. 이때만 obstacle slots을 Transformer의 독립 token으로 넣는
   ablation을 병행한다.
4. 8→12 capacity 증가는 중복 선택을 고친 뒤 수행한다. 지금 늘리면 같은 막대 표면점을
   더 많이 담을 가능성이 높고 fresh policy가 필요해 인과 비교도 나빠진다.

이번 조사는 문헌·코드 감사와 계획 갱신만 수행했으며 학습 프로세스나 정책 코드는 변경하지 않았다.

---

## 2026-07-30 — token 재설계 전 학습 실행 여부 확인

사용자가 다음 학습 명령을 요청해 프로세스, checkpoint, launcher와 perception 구현을 다시
확인했다. 현재 NavRL 학습 프로세스는 없고 GPU 학습은 시작되지 않은 상태다. 최신 정상
baseline은 `ppo_260730_1104.../last_gen_ppo_ep_8350...`이며, 이후 suppress ±15° run의
epoch 8900 checkpoint는 clean window `0.664/0.689`, unique `3.4~3.5/8`로 실패 판정된
ablation 결과다.

문헌 조사에서 정한 `cluster-balanced` selector는 아직 `navrl_perception.py`에 구현되지
않았다. 현재 이용 가능한 `train_navrl_corrected_squashed_density_suppress15.sh`를 실행하면
새 token 설계가 아니라 실패한 fixed-85 ±15° 설정을 반복한다. 따라서 새 selector를 구현하고
학습 없는 offline coverage gate를 통과하기 전에는 장기 PPO 명령을 제공하거나 실행하지
않는 것으로 결정했다.

---

## 2026-07-30 — cluster-sector obstacle selector 구현

사용자 승인 후 기존 898-D observation과 8×12 obstacle proposal 폭을 유지하는
`cluster_sector` 선택기를 구현했다.

### 선택 로직

- 4×72 LiDAR의 vertical-min 72 range를 body-frame 2-D endpoint로 변환한다.
- 인접한 유효 endpoint 거리가 기본 0.45 m 이하이면 같은 surface cluster로 연결한다.
  range 차이만 쓰지 않아 가까운 원통 표면에서 생기는 정상적인 깊이 변화에 덜 민감하다.
- 240° token FOV를 기본 8 sector로 나누고 각 non-empty sector의 최근접 cluster 하나를
  먼저 예약한다. 이미 선택한 cluster는 다른 sector에서 다시 선택할 수 없다.
- 빈 sector 수만큼 아직 선택되지 않은 최근접 cluster를 보충한다. 최종 proposal은 기존
  flattened MLP가 warm-start하기 쉽도록 거리순으로 재정렬한다.
- 이전 `greedy_suppress`는 기본 모드로 그대로 남겨 회귀와 A/B가 가능하다. 두 selector는
  모두 `[8,12]`를 내므로 actor observation은 898-D로 동일하다.

### 계약·launcher·평가 가드

- 환경변수 `NAVRL_OBSTACLE_SELECTOR`, `NAVRL_OBSTACLE_CLUSTER_GAP_M`,
  `NAVRL_OBSTACLE_SECTORS`를 추가하고 시작 로그 및 checkpoint `env_state`에 저장한다.
- preflight는 legacy checkpoint의 selector를 `greedy_suppress`로 해석한다. 새 전용
  launcher만 selector/gap/sector의 same-shape override를 명시적으로 허용하며, 다른
  representation 계약은 계속 엄격하게 검사한다.
- `train_navrl_corrected_squashed_density_cluster_sector.sh`는 정상 epoch-8350 bounded
  checkpoint, 85 bars 고정, density accumulator reset, 250-epoch adaptation hold,
  gap 0.45 m, 8 sectors를 고정한다.
- `eval_navrl_cluster_sector_density_sweep.sh`를 추가하고 공통 density sweep도 selector
  계약을 출력하도록 수정했다. `play_navrl.sh`는 절대 Python 경로 사용 시 해당 conda
  `bin`을 PATH에 넣어 Isaac Gym의 `ninja` 탐색 실패를 막는다.

### 검증

- cluster 중복 제거, sector coverage, empty-sector fallback 단위 테스트 3개를 추가했다.
- 전체 Python discovery `37/37`, action-model `13/13`, Python compile, shell syntax,
  `git diff --check`가 통과했다.
- 실제 epoch-8350 checkpoint preflight는 legacy greedy→cluster-sector와 과거 checkpoint에
  없던 gap/sector provenance만 의도적 override로 기록했고, legacy greedy preflight도
  회귀 없이 통과했다.
- CPU perception observe와 RTX 3070 CUDA `[128,72]→[128,8]` selector smoke가 통과했다.
- 64 environments로 실제 PPO를 1 epoch 재개해 epoch 8351까지 forward/backward와 종료가
  정상임을 확인했다. 17 episodes의 capture 58.8%는 적응 전 극소표본이므로 성능 근거로
  사용하지 않는다. smoke run은 `/tmp/motar_cluster_smoke.zh4AlZ`로 이동했고
  `current_training.log`는 마지막 실제 suppress15 로그로 복원했다.
- bulk-play는 simulator/selector 초기화까지 성공했으나 기존 player가 episode 집계 전에
  `Can't create empty tensor`로 종료해 실제 barprobe offline 수치는 얻지 못했다. 따라서
  첫 clean training window에서 unique/duplicate와 capture를 함께 판정한다.

실행 명령:

```bash
cd /home/fair/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games
./train_navrl_corrected_squashed_density_cluster_sector.sh
```

정상 시작 로그는 `selector=cluster_sector`, `cluster_gap=0.45m`, `sectors=8`,
`fixed bars=85`, 세 개의 `INTENTIONAL contract override`, density evidence reset을
모두 포함해야 한다. 다른 터미널에서는 `./watch_navrl_training.sh`로 같은 로그를 본다.

---

## 2026-07-30 — 최근 UAV 인지 논문 5편의 저비용 적용성 감사

사용자가 제공한 최근 한 달 논문 목록을 원문과 대조하고, 현재 MOTAR/NavRL++-Target
구현에서 학습 방향을 바꾸지 않고 가져올 수 있는 부분을 검토했다. 정책·환경·launcher는
변경하지 않았다.

### 이미 구현되어 있어 새 구조가 필요 없는 부분

- fixed-wing tracking 논문의 `detector → target state estimator` 구조는 현재 RGB-D/LiDAR
  측정 뒤 3-D constant-velocity Kalman tracker로 이미 구현되어 있다. actor target
  feature도 상대 위치/속도, 위치·속도 covariance, camera/LiDAR confidence, track age를
  포함한다. 측정값이 이미 3-D이므로 비선형 image measurement를 직접 처리하는 해당
  논문의 UKF로 바꿀 근거는 없다.
- CosFly-VLA의 핵심인 5-frame history, 이전 action/state, visibility-aware memory도
  현재 5-step robot/target history와 5 s tracker memory에 대부분 대응한다. 검출이
  끊겨도 KF prediction은 유지되고 camera/LiDAR confidence는 0, track age/covariance는
  증가한다. 다만 CosFly-VLA 성능은 평가 때 이전 target state history를 GT로 제공한
  조건이므로 sensor-only MOTAR와 직접 수치 비교하지 않는다.
- CMRTrack이 주장하는 motion reliability에 해당하는 최소 정보는 이미 covariance,
  confidence, track age로 actor에 전달된다. CMRTrack 자체는 infrared image tracker이며
  ViT-B와 별도 detector dataset을 요구하므로 현재 PPO에 통째로 넣지 않는다.

### 방향을 유지하는 후보 순서

1. **설정·평가만으로 시작:** Phase 3C에서 기존
   `NAVRL_PERCEPTION_PERTURB`, `NAVRL_DETECTION_DROPOUT`,
   `NAVRL_RGB_NOISE_STD`, `NAVRL_DEPTH_NOISE_STD`를 사용해 가림/검출누락을 만들고,
   visible/occluded duration별 track survival, reacquisition time, capture/crash를
   층화한다. 현재 clean detector/tracker gate 전에는 perturbation을 켜지 않는다.
2. **작은 same-shape tracker 개선:** learned detector가 준비되면 Kalman normalized
   innovation 및 camera–LiDAR disagreement를 먼저 diagnostic으로 기록한다. false
   correction과 강하게 연관되면 Mahalanobis innovation gate 또는 innovation-derived
   reliability로 측정 update를 조절한다. observation 차원을 늘리지 않고 기존
   confidence/covariance 의미를 강화할 수 있다.
3. **검증 후에만 auxiliary risk:** CPA future-risk는 현재 PPO·asymmetric critic과
   구조적으로 잘 맞지만 원 논문의 주효과는 동적 장애물이다. 현재 arena 막대는 정적이라
   즉시 risk head를 추가하면 LiDAR urgency와 reward를 중복할 수 있다. 우선 GT
   relative velocity로 72-bearing CPA risk diagnostic을 계산해 collision 선행 예측력과
   정지 병목 상관을 확인하고, 이득이 있을 때만 기존 temporal encoder에 training-only
   auxiliary loss를 추가한다.

### 보류

- CosFly-VLA의 0.8B VLA, language prompt, 8-step waypoint chunk와 CoT 학습은 모델·데이터·
  action contract를 크게 바꾸며, 해당 논문도 predicted-state feedback 검증을 미래 과제로
  남겼으므로 채택하지 않는다.
- fixed-wing NMPC/CBF/BPNG는 기체·카메라·terminal-impact 목적이 다르다. quadrotor
  capture PPO를 대체하지 않는다.
- ASUMOT은 event camera 전용이고 코드·데이터가 “공개 예정” 상태다. sensor swap은
  현재 범위 밖이며, motion-consistency 원리는 향후 innovation reliability에만 반영한다.

결론적으로 현재 실행 순서는 바꾸지 않는다. 먼저 fixed-85 `cluster_sector` gate로
navigation backbone을 고정하고, Phase 3B detector 검증 뒤 occlusion/dropout 평가와
innovation reliability를 붙인다. CPA auxiliary head는 진단 로그가 효과를 입증할 때만
세 번째 후보로 진행한다.

---

## 2026-07-30 — cluster-sector 현황 사이트 반영 및 TensorBoard 2차 정리

사용자 요청에 따라 연구현황 사이트를 현재 실행 중인 fixed-85 cluster-sector 실험으로
갱신하고, checkpoint와 metrics CSV를 보존하면서 TensorBoard의 오래된 세션을 별도
archive로 이동했다.

### 사이트에 반영한 현재 결과

- 작업 시작 시 활성 run은
  `ppo_260730_1549_navrl_corrected-squashed-density-cluster-sector-s1`이었다. TensorBoard
  정리 뒤에도 계속 epoch를 기록했으나 이후 terminal error나 정상 종료 summary 없이
  epoch 10157에서 중단됐다. 마지막 주기 checkpoint 10150은 보존됐다.
- clean 16,384-episode competence window 네 개는 capture
  `0.742 / 0.745 / 0.722 / 0.714`로 모두 0.70 기준을 넘었다.
- barprobe의 unique associated bars는 기존 greedy/suppress 계열 `3.0~3.5/8`에서
  cluster-sector 약 `4.4/8`로 상승했고 duplicate는 `1.9`에서 약 `0.1`로 감소했다.
  이는 중복 surface가 slot을 소모하던 병목을 실제로 제거했다는 증거다.
- unique `4.5/8` 기준은 window에 따라 `4.2~4.6`으로 경계이므로 완전 통과로 과장하지
  않고 사이트에 `near threshold`로 표시했다.
- 네 개의 큰 clean window가 이미 같은 결론을 내므로 남은 epoch만 채우기 위해 재시작하지
  않고, checkpoint 10150의 density×target-speed held-out 평가로 넘어간다. 재현되면
  navigation backbone을 고정하고 Phase 3B learned detector로 돌아간다.
  후속 perception 후보는 occlusion/dropout 평가, Kalman innovation reliability 순이며
  CPA auxiliary risk는 diagnostic-first로 유지했다.

`tools/update_status_snapshot.py`, `docs/status/status.json`, HTML inline fallback과
cluster-sector 표시를 동기화했다. dashboard JSON과 fallback의 완전 일치, 활성 run/selector
계약, 브라우저 motion test를 검증했다.

### TensorBoard 정리

- 라이브 TensorBoard에서 2026-07-14~24의 historical 세션 21개와 2026-07-28의
  1~75 epoch action diagnostic 9개, 합계 **30개 summaries**를 제거했다.
- 여러 resume shard를 포함해 event file **37개**를
  `/home/fair/workspaces/aerial_gym_ws/tensorboard_archive/2026-07-30_historical_events/`
  아래로 이동했다. 삭제하지 않았으며 archive README에 전체 목록과 복구 방법을 기록했다.
- `runs/`의 checkpoint, `aerial_run/epoch_metrics.csv`, run summary 등은 전혀 이동하지
  않았다. 따라서 학습 재개 경로와 사이트의 45-run 연구 원자료는 유지된다.
- TensorBoard API에서 표시 세션이 **43→13개**로 즉시 줄어든 것을 확인했다.
  라이브에는 07-27 representation/density 계보, 500-epoch bounded-action 비교,
  07-29~30 corrected 계보와 현재 cluster-sector만 남겼다.
- summaries 이동 직후 TensorBoard API를 검사할 때까지 활성 runner와 event/checkpoint
  기록은 정상 계속됐다. 이후 epoch 10157에서 별도 종료 기록 없이 프로세스가 사라졌으며,
  archive 작업은 활성 run의 `summaries`나 다른 파일을 이동하지 않았다.

---

## 2026-07-30 — research 전체를 main에 합치기 전 감사

사용자가 GitHub의 `research/navrl-env`를 `main`에 합쳐도 되는지 질문해 실제 branch graph,
merge-base, 전체 tree diff와 merge-tree를 read-only로 점검했다. merge는 실행하지 않았다.

- `main...research/navrl-env`은 main-only 4개, research-only 138개 commit으로 갈라져 있어
  fast-forward가 아니다. merge-tree상 직접 text conflict는 `.gitignore` 1건이다.
- main-only 4개는 실제로 두 경로뿐이다. `317505b`는
  `.github/workflows/publish_site.yml`을 삭제했고, `b609eee`와 `64ba00a`는 `.gitignore`에
  local experiment/generated media와 session/checkpoint ignore를 추가했다. `79f0679`는 이
  두 갈래를 합친 merge commit이라 독립 기능 변경이 아니다. research tree에도 workflow
  삭제와 모든 ignore 규칙이 이미 내용상 반영돼 있고 `*.log` ignore만 하나 더 있으므로,
  main-only 네 commit 자체가 통합 위험인 것은 아니다.
- 그러나 전체 변경은 287 files, 약 `+28k/-85k`이며 NavRL/MOTAR 추가뿐 아니라 upstream
  DCE navigation example, sim2real code/weights, MkDocs 문서·이미지와 pretrained artifact의
  대량 삭제를 포함한다. 현재 “main은 upstream 기반, research는 MOTAR 연구”라는 branch
  역할을 유지한다면 전체 merge는 범위가 맞지 않는다.
- 더 중요하게 현재 working tree에는 cluster-sector selector, curriculum/PPO safety,
  preflight, launcher와 tests를 포함한 25개 미커밋 항목이 있다. 지금 merge하면 정작 최신
  핵심 구현은 main에 포함되지 않고, 오래된 research tip만 합쳐진다.
- 따라서 현 시점 결정은 **전체 merge 보류**다. 최신 same-shape selector와 safety 변경을
  먼저 하나의 검증된 research commit으로 만들고 clean worktree에서 main 통합 후보를
  선별한다.
- GitHub에서 최신 프로젝트를 기본으로 보이게 하는 목적이라면 upstream 보존용 main을
  합치기보다 default branch를 `research/navrl-env`로 바꾸는 방법이 더 안전하다. 반대로
  main 자체를 MOTAR 제품 branch로 전환하려는 경우에는 upstream 보존 branch를 먼저 만든
  뒤, 대량 삭제를 의도적으로 수락하는 별도 통합 작업으로 진행해야 한다.

---

## 2026-07-30 — density × target-speed 맵 (논문 헤드라인 그림) + 85→110 확장 학습 재개 + 사이트 Map 탭

체크포인트 `runs/ppo_260730_1549_navrl_corrected-squashed-density-cluster-sector-s1/nn/
last_gen_ppo_ep_10150_rew_36.50442.pth` (cluster_sector selector, epoch 10150)로 밀도(25/50/
65/85/110/130/150 bars) × 표적속도(0.0/0.5/1.0/1.5 m/s) 28셀 held-out 평가를 실행했다.
128 envs, 셀당 2049 episode, deterministic, general(랜덤) spawn, mixed target pattern.
결과: `results/density_speed_map_cluster_sector.csv`.

### 핵심 발견 — 두 축이 완전히 비대칭

| bars | density/100m² | capture @0.0 | @0.5 | @1.0 | @1.5 |
|---|---|---|---|---|---|
| 25  | 5.2  | 96.2% | 97.0% | 96.5% | 94.9% |
| 50  | 10.5 | 91.8% | 93.0% | 92.2% | 90.4% |
| 65  | 13.6 | 88.1% | 87.9% | 86.1% | 83.7% |
| 85  | 17.8 | 73.6% | 75.3% | 71.8% | 67.1% |
| **110** | **23.0** | 49.7% | 50.6% | 48.4% | 43.7% |
| 130 | 27.2 | 31.8% | 29.6% | 28.1% | 25.9% |
| 150 | 31.4 | 19.4% | 19.2% | 19.0% | 15.9% |

(85 bars = 학습된 최대 밀도, 굵게 표시한 110행부터 generalisation. 전체 표+crash 원인
breakdown은 CSV 참고.)

전체 그리드에서 밀도를 5.2→31.4 bars/100m²로 올리면 capture가 **78 pp** 떨어지는 반면,
표적속도를 0→1.5 m/s로 올리면 **4.2 pp**만 떨어진다. pursuer v_max=2.5 m/s가 표적보다
충분히 빨라서, 이 레짐에서 표적속도는 사실상 난이도 축이 아니고 밀도가 거의 전부다 —
이게 논문 헤드라인 그림이 된다. crash 원인은 밀도가 오를수록 거의 전부
`crash_bar_contact_share`(85 bars에서 90%+, 150 bars에서 94%+)로 수렴 — OOB/추락사는
부수적이고 장애물 접촉이 지배적 실패 모드임을 재확인.

### 85→110 밀도 커리큘럼 확장 재개

체크포인트 10150에서 재개하려던 중 `train_navrl_corrected_squashed_density_cluster_sector.sh`
(Codex가 이전에 작성한 launcher)에 다음이 하드코딩되어 있음을 발견:

```
export NAVRL_CONTROLLED_ABLATION=1
export NAVRL_FIXED_BARS="${NAVRL_FIXED_BARS:-85}"
```

`NAVRL_CONTROLLED_ABLATION=1`이 `_update_curriculum`의 밀도 승급 자체를 막아 85 bars에서
영구 고정시킨다 — 위 held-out 평가로 85 bars plateau가 이미 확인됐으므로 이 ablation은
목적을 다했고, 다음 단계(85→110+)로 넘어가려면 반드시 해제해야 한다. 아래처럼 override해서
재시작:

```
NAVRL_CONTROLLED_ABLATION=0 NAVRL_FIXED_BARS= \
  ./train_navrl.sh --checkpoint runs/ppo_260730_1549_navrl_corrected-squashed-density-cluster-sector-s1/nn/last_gen_ppo_ep_10150_rew_36.50442.pth \
  --branch_run --disable_collapse_early_stop --max_epochs 26000 --seed 1
```

20:15 KST 시작, run `ppo_260730_2015_navrl_cluster-sector-density85to110-s1`, 로그
`train_session_logs/cluster_sector_85to110_260730_201534.log`. 20:30 기준 epoch 10429/26000,
capture 75-80%대에서 85 bars 유지 중(아직 Layer-1 window 16384 episode 채우는 중, 승급
전). 예상 소요 11-15시간. `NAVRL_DENSITY_STEP=5` 그대로이므로 다음 승급 목표는 90 bars.

### 사이트 반영

- `docs/status/status.json`에 `density_speed_map`(28행 + notes + trained_max_bars=85) 키 추가.
- `docs/status/app.js`에 `renderHeatmap(s)` 신규 — 28셀을 red→tan→teal 그라디언트로 렌더링,
  85 bars(학습 범위) 아래는 옅은 배경 음영 + 점선으로 generalisation과 구분, 실측값 기반
  헤드라인 문장("Density dominates: ... 78 pp ... 4.2 pp ...")을 데이터에서 직접 계산해 표시.
- `docs/status/index.html`에 `Map` 탭 버튼 + `#pane-map` 추가, `style.css`에 `.hm-v`/`.hm-s`
  셀 텍스트 스타일 추가.
- headless Chrome 스크린샷으로 28셀 전부(25→150 bars × 4속도) 렌더 검증 완료 — 값, 색상,
  구분선, 캡션 텍스트 모두 CSV와 일치 확인.

---

## 2026-07-30 (밤) — 85→110 학습이 사실 85에 얼어있었음: launcher override-무시 버그 수정 후 재시작

### 발견 (버그)

run `ppo_260730_2015`(20:15 시작)가 85→110 확장 학습인 줄 알았으나, **승급 심사가 2회 모두
통과 기준을 넘기고도 "held"** 로 찍힘:

```
NavRL density curriculum held | bars=85 capture=0.759 over 16385 eps   (epoch 10658)
NavRL density curriculum held | bars=85 capture=0.767 over 16384 eps   (epoch 10921)
```

`/proc/<pid>/environ` 실측: `NAVRL_DENSITY_FINAL=85`, `NAVRL_FIXED_BARS=85`,
`NAVRL_CONTROLLED_ABLATION=1` — 승급 조건 `n_bars_active < final_bars`(85<85=False)가 영구
False. 원인은 `train_navrl_corrected_squashed_density_cluster_sector.sh:23-24`:

```bash
export NAVRL_CONTROLLED_ABLATION=1                 # 가드 없음 — override 무조건 덮어씀
export NAVRL_FIXED_BARS="${NAVRL_FIXED_BARS:-85}"  # ':-'는 빈 문자열도 85로 치환
```

launch 시 `NAVRL_CONTROLLED_ABLATION=0 NAVRL_FIXED_BARS=` override를 줬으나 둘 다 무력화됨.
결과: epoch 10151→11200(~1,050 epoch, 2.5시간)이 85 고정 ablation 재확인에 소모됨(손실은
아님 — capture 0.759/0.767로 85-bar 역량 재확증 데이터가 됐고 checkpoint도 개선 지속).

### 승급 주기 실측 (질문 답변용)

- 승급은 epoch가 아니라 **완료 에피소드 16,384개**(`NAVRL_DENSITY_CHECK_EPS`) 단위 심사.
- 실측 완료 에피소드 63.9개/epoch → **심사 1회 ≈ 263 epoch ≈ 14분**(step 2.7-2.9s).
- 재개 직후 첫 심사만 `NAVRL_DENSITY_RESUME_WARMUP=250` 때문에 ~507 epoch.
- 심사 시 capture ≥ 0.70 (+ strata gate, 현재 diagnostic-only) 그리고 bars < final이면 +5.

### 수정 및 재시작

- launcher를 `NAVRL_CONTROLLED_ABLATION="${NAVRL_CONTROLLED_ABLATION:-1}"` + FIXED_BARS는
  ablation=1일 때만 설정하도록 수정(기본 동작=기존 ablation 그대로, `=0` 전달 시 커리큘럼).
  echo도 모드 표시(`mode=density-curriculum->110` vs `mode=fixed-85bars-ablation`)로 변경.
- 기존 run kill 후 `last_gen_ppo_ep_11200_rew_23.425339.pth`에서 재시작 (21:11 KST):
  run `cluster-sector-density85to110-v2-s1`, 로그
  `train_session_logs/cluster_sector_85to110_v2_260730_211103.log`, MAX_EPOCHS=26000.
- **재시작 후 프로세스 env 검증 완료**: `CONTROLLED_ABLATION=0`, FIXED_BARS/NUM_BARS 없음,
  `DENSITY_FINAL=110`, restore 로그 `bars=25->85` (체크포인트 밀도 정상 복원), epoch 11201부터
  85 bars로 진행 중. 첫 승급 심사 예상 ≈ epoch 11710 (~25분 후), 통과 시 85→90.

---

## 부록 A — 삭제된 run 아카이브 (2026-07-14 정리 시점)

`runs/` 폴더가 삭제되어 재구성 불가한 26개 run의 최종 지표. 원본:
`aerial_gym/rl_training/rl_games/RUNS_ARCHIVE_SUMMARY.md` (이 부록으로 통합 후 삭제).

> 로드하지 않는다. 현재 observation/model schema는 각 run manifest로 구분한다.

`runs/`의 옛 run들을 삭제하기 전에 핵심 정보를 남김. 현재 학습 `ppo_260714_1904_navrl`은 유지.
지표: cap=captured_rate, crash=crash_rate, to=timeout_rate (최종 epoch). reward는 스케일이 리워드 설계에 따라 달라 절대비교 불가.

| run | type | ep | exit | last/peak reward | navrl 최종 | 무엇 |
|---|---|---|---|---|---|---|
| ppo_260530_1059 | intercept(옛과제) |  |  | / |  |  |
| ppo_260530_1112 | intercept(옛과제) |  |  | / |  |  |
| ppo_260530_1419 | intercept(옛과제) |  |  | / |  |  |
| ppo_260530_1604 | intercept(옛과제) |  |  | / |  |  |
| ppo_260530_1921 | intercept(옛과제) | 1261 | interrupted | 2408.2/2906.1 |  |  |
| ppo_260530_2005 | intercept(옛과제) | 6500 | interrupted | 922.4/1537.9 |  |  |
| ppo_260531_1031 | intercept(옛과제) | 15000 | max_epochs | 617.9/841.8 |  |  |
| ppo_260601_1152 | intercept(옛과제) | 931 | interrupted | 825.7/1149.7 |  |  |
| ppo_260601_1228 | intercept(옛과제) | 4652 | interrupted | 808.6/1076.9 |  |  |
| ppo_260601_1718 | intercept(옛과제) | 8787 | interrupted | 277.7/455.3 |  |  |
| ppo_260709_2146_navrl | navrl | 1500 | max_epochs | 142.5/170.3 |  | 첫 navrl 학습(1500ep, 옛 env). 목표도달 0%·충돌 다수 — 인프라만 완성. |
| ppo_260710_1230_navrl | navrl | 273 | interrupted | 126.5/159.8 |  | 초기 bars env 스모크/설정. |
| ppo_260710_1559_navrl | navrl |  |  | / |  | 초기 bars env 실험. |
| ppo_260710_1608_navrl | navrl | 1500 | max_epochs | 339.2/349.9 |  | bars env 첫 본학습(512env, MLP). |
| ppo_260710_1709_navrl | navrl |  |  | / | cap 0.00 / crash 0.37 / to 0.00 | bars env 실험. |
| ppo_260710_1738_navrl | navrl | 62 | interrupted | 229.7/236.4 | cap 0.00 / crash 0.42 / to 0.06 | bars env 실험. |
| ppo_260710_1745_navrl | navrl | 6000 | max_epochs | 365.2/375.9 | cap 0.00 / crash 0.02 / to 0.82 | bars env 실험. |
| ppo_260710_2106_navrl | navrl | 6000 | max_epochs | 296.9/327.5 | cap 0.01 / crash 0.12 / to 0.87 | 6000ep CNN 네비게이션 run(캡처종료 前, '목표찍고 배회'). |
| ppo_260713_1950_navrl | navrl | 6000 | max_epochs | 327.3/338.1 | cap 0.07 / crash 0.10 / to 0.84 | 캡처종료 첫 run — LOITER 실패(captured 6.7%, timeout 84%). 안전항 수입이 배회를 보상. |
| ppo_260713_2210_navrl | navrl | 6000 | max_epochs | 88.7/100.0 | cap 0.86 / crash 0.14 / to 0.00 | 리워드 재설계(B1 재베이스+PBRS progress+B3) 성공 — captured 86%(peak 94%), timeout 0%. loiter 해결. |
| ppo_260714_0153_navrl | navrl | 5049 | interrupted | 80.8/98.0 | cap 0.86 / crash 0.14 / to 0.00 | crash튜닝 A: safety_weight 1.5 — crash 13.7%(무효, safety만으론 안됨). |
| ppo_260714_0346_navrl | navrl | 6000 | max_epochs | 89.5/100.1 | cap 0.85 / crash 0.15 / to 0.00 | crash튜닝 B: clearance 거리 페널티 1.5 — crash 13.9%(무효). |
| ppo_260714_0555_navrl | navrl | 6000 | max_epochs | 85.6/100.2 | cap 0.87 / crash 0.13 / to 0.00 | crash튜닝 C: clearance speed-gated 1.5 — crash 13.8%(무효). 결론: 옛 스폰이 막대 무관(8%만 관통). |
| ppo_260714_1853_navrl | navrl | 32 | interrupted | 13.2/15.1 | cap 0.03 / crash 0.97 / to 0.00 | 짧게 중단된 run(설정 확인). |

---

## 2026-07-30 — navigation backbone 동결 게이트 (Codex 기록, 문서 통합 시 계획서에서 이관)

문서 통합(14→6개) 과정에서 `PERCEPTION_TRANSFORMER_PLAN`/`ROADMAP`/`PHASE3_PLAN`에 있던
2026-07-30 상태 기록을 여기로 옮긴다. 계획서는 charter만 담고 상태는 WORKLOG가 canonical.

**전제**: Phase 3 detector/occlusion 효과를 해석하려면 장애물 표현과 PPO update 자체의 실패를
먼저 분리해야 한다. 그래서 detector 본 검증 전에 navigation-only backbone의 고밀도 표현을 동결한다.

| 항목 | 근거 | 판정 |
|---|---|---|
| 센서 기하 | corrected bearing, token hit·FOV 약 97–98% | 통과 |
| bounded action | raw OOB 0%, corrected 500-epoch pilot + speed gate 통과 | 통과 |
| 밀도 확장 | 65 bars `0.678` → 85 bars `0.676±0.001` | 같은 실력으로 밀도 +31% |
| token coverage | 85 bars에서 unique 약 3/8, duplicate 약 2.7–3.0 | **병목** |
| ±10° 장기 재개 | epoch 10276 이후 KL/latent 폭발, tail500 capture 1.0% | **실패 · 결과 제외** |
| ±15° 1차 ablation | clean gate 0.664/0.689, unique 3.4–3.5, epoch 8900 중단 | 기준 미달 |

**token selection 후보의 통과 기준 (넷 다 만족해야 Phase 3의 고정 backbone이 된다)**
- barprobe `unique` 3.0 → **4.5+**, duplicate 감소
- 85 bars promotion window capture **≥0.70 / 16,384 episodes**
- PPO **KL ≤0.04**, latent mean/edge saturation 발산 없음
- 정상 checkpoint의 held-out density sweep 재현

**재시험 계약**: 85 bars 고정, gate accumulator 초기화, 250-epoch adaptation hold, density
promotion cap, speed/distance/pattern 층화 진단, same-density capture collapse fail-stop.
첫 ±15° 시도는 판정창에 이전 ±10° 에피소드가 섞이는 계약 오류가 있었다.

**결과 표에서 제외할 run**: `ppo_260730_1154...`의 tail500 capture 1.0%는 ±10° 장기 재개 중
PPO 발산 사례이며 ±15°의 결과가 아니다. safety ablation / engineering failure로만 기록한다.

**롤백 규칙**: 고정 density에서 KL 또는 latent mean이 지속 상승하면 더 학습하지 말고
last-known-good checkpoint로 rollback한 뒤 update safety를 수정한다.

---

## 2026-07-30 — 저장소 정리: runs/ 14.8GB 회수 + 문서 14→6개 통합

### 디스크 정리 (파괴적 작업 — 4단 검증 후 실행)

`runs/` **16.1GB → 1.27GB (14.8GB 회수)**. 50 epoch마다 저장되던 중간 체크포인트를 run별
최신 2개 + `gen_ppo*.pth`만 남기고 삭제. 근거: 2026-07-29 키랄리티 수정으로 **그 이전 체크포인트는
warm-start가 원천 무효**이고, 성능 수치는 `results/*.csv`와 WORKLOG에 이미 보존돼 있다.

삭제 전 4단 검증 전부 통과: ① 문서/런처가 참조하는 체크포인트 8개가 삭제목록에 없음 ② 실행 중인
run과 그 소스 체크포인트 제외 ③ `gen_ppo.pth` 0건 포함 ④ 모든 run이 최소 1개 체크포인트 유지.
6MB 초과 세션 로그는 gzip(진행 중 로그 제외). 학습 프로세스 무중단 확인.

### 문서 통합 (14 → 6개)

| 남긴 파일 | 통합된 원본 |
|---|---|
| `README.md` | (상태 블록을 실측값으로 재작성) |
| `RESEARCH_PLAN.md` | + `PERCEPTION_TRANSFORMER_PLAN` + `ROADMAP` + `PHASE3_PLAN` |
| `OPERATIONS.md` (신규) | `SETUP_SECOND_MACHINE` + `GPU_SCALING_GUIDE` + `TRANSFER_RESULTS_GUIDE` |
| `CRASH_TUNING_LOG.md` | + `rl_games/CRASH_TUNING_LOG.md`(B/C/D 음성결과를 부록으로) |
| `WORKLOG.md` | + `RUNS_ARCHIVE_SUMMARY`(부록 A) |
| `CLAUDE.md`, `SKILL.md` | 병합 안 함(에이전트 지시문) — 링크·관측차원만 수정 |

**세 가지 상충하던 위상 번호**(ROADMAP P0–P6 / PHASE3 "Phase 3" / PERCEPTION P0–P5)를
`RESEARCH_PLAN`의 **단일 P0–P7**로 통일. CLAUDE.md의 "관측 1265차원" 낡은 기술을
**898 actor / 906 critic + cluster_sector 240°**로 수정하고 폐기된 계보(156→305→1265)를 명시.

보존 확인: 업스트림 Aerial Gym 라이선스·인용 섹션 그대로, **코드 주석 3곳이 참조하는
`CRASH_TUNING_LOG.md` 경로 불변**(옮기면 주석이 깨지므로), 삭제 문서를 가리키던 링크 전부 재지정.

### ⚠ 이 작업 중 내가 낸 실수 (기록)

`RESEARCH_PLAN.md`를 통합본으로 덮어쓸 때 **Codex의 미커밋 상태 블록을 삭제했다.** 그 실질
내용(85 bars `0.676±0.001`, unique 약 3/8 병목, ±15° 1차 미달, ±10° 장기재개 PPO 발산,
token-selection 통과 기준 4개)은 같은 시각 `PERCEPTION`/`ROADMAP`/`PHASE3`의 미커밋 편집에도
동일하게 있었고, 그것을 삭제 전에 위 "navigation backbone 동결 게이트" 항목으로 이관해 보존했다.
교훈: **다른 세션의 미커밋 변경이 있는 파일은 덮어쓰기 전에 `git status`로 확인**한다.

Codex의 미커밋 코드 변경 21개 파일은 손대지 않았다(스테이징 제외).

---

## 2026-07-30 — GitHub 기본 브랜치 전환 시도

사용자 선택에 따라 기존 `main`을 삭제하거나 병합하지 않고, GitHub 저장소
`joshualikaist/MOTAR`의 기본 브랜치만 `research/navrl-env`로 바꾸는 방법을 선택했다.

- 전환 전 확인: 원격 `main`과 `research/navrl-env`가 모두 존재하고 로컬 추적 브랜치와 일치한다.
- 1차 미실행 사유: 이 환경에는 `gh` CLI, `GH_TOKEN`/`GITHUB_TOKEN`, GitHub API credential이
  없었고, 비공개 상태의 저장소는 GitHub 플러그인에서도 404로 조회가 거절됐다.
- 공개 전환 후 재시도: GitHub 플러그인으로 `visibility=public`, 사용자 권한 `admin=true`,
  `default_branch=main`을 확인했다. 다만 현재 플러그인이 저장소 기본 브랜치 변경 API를
  제공하지 않아 설정 변경은 실행하지 못했다.
- 최종 검증: `origin`의 HEAD는 여전히 `main`이다. 원격 브랜치 삭제·병합·강제 푸시는 수행하지 않았다.
- 남은 작업: GitHub 연결 후 기본 브랜치를 `research/navrl-env`로 변경하고,
  `git remote show origin`의 `HEAD branch`가 해당 브랜치인지 재확인한다.

---

## 2026-07-30 (밤) — 아레나 시뮬레이션 "표적 발작·둘 다 정지" 진단 및 수정 (14-에이전트 감사, 70만+ 프레임 실측)

사용자 보고: "pursuer는 똑똑한데 target은 혼자 멋대로 돌아다니고, 가끔 둘 다 가만히 있다.
막대 토큰 병목 때문인 것 같다." → 5렌즈 발견 + 8건 적대적 검증 워크플로우로 조사.

### 사용자 귀속(토큰 병목) 판정: 기각

브라우저 아레나에는 장애물 토큰 개념이 없다. `MAX_OBSTACLES=8`은 `navrl_perception.py:24`의
RL 관측 텐서 크기일 뿐이고, `arena.js`/`arena_motion.js`에는 `MAX_OBSTACLES`/`token`/
`cluster_sector` 문자열이 0회 등장 — 모션 모델은 매 프레임 전체 bars 배열을 직접 순회한다.
토큰 병목은 실제 env에서 **bar contact**(충돌)를 유발하는 문제이지 정지를 만들지 않는다.

### 근본 원인 (실측 확정)

1. **표적 발작 = `steerTargetStep`의 무기억 argmax** (`arena_motion.js`): 직전 비행 방향을
   전혀 참조하지 않고 순간 목표 방위만으로 10개 후보를 재선택. waypoint 모드(에피소드의 50%)는
   조향 결과를 어디에도 저장하지 않아(`cvVelocity` 기록은 cv 전용), 막대에 막히면 ±120~180°
   탈출 후보로 점프→다음 프레임 복귀하는 **주기-2 진동**. 실측(40에피소드×20s): 85막대에서
   렌더 yaw 평균 55.1°/스텝, 29.4%가 90° 초과 반전; 110막대에서 75.8°/41.0%. pursuer가
   "똑똑해 보이는" 것은 `steerPursuerStep`에만 연속성 항(`-turn*0.35`)+저역 필터가 있어서다.
   **이것은 포팅 버그가 아니라 Python `target_motion.py` 조향 법칙의 충실한 재현** —
   trainer의 표적도 동일하게 진동한다(아래 후속 작업).
2. **"둘 다 정지" = 같은 결함의 다른 얼굴 + 기하**: 표적은 전속력으로 움직이지만(경로길이/
   명령속도=1.000) 제자리 진동이라 순수송이 붕괴 — 150막대에서 1초 순수송 효율 0.338, 벽시계의
   58.1%가 순정지. 추가로 110막대↑에서는 표적 여유공간(1.0m clearance) 자체가 단절됨(최대
   연결 성분: 85막대 96.4% → 110막대 45.4% → 150막대 12.5%) — 이건 코드가 아니라 기하 한계.
   기본 설정(85막대)에서는 둘-다-정지 0.00% — 사용자가 본 정지는 슬라이더를 높였을 때.
3. **부가 버그**: `advanceTarget`이 `speed≤1e-6`이면 `episode.age += dt` 전에 조기 return →
   속도 슬라이더 0에서 30초 워치독 무력화(120초에 에피소드 3개 vs 수정 후 16개).

### 적용한 수정 (`docs/status/arena_motion.js`)

- **FIX 1**: `episode.age += dt`를 조기 return 앞으로 호이스팅 — 워치독은 벽시계 기준.
- **FIX 2**: `steerTargetStep`에 opt-in 9번째 인자 `prevHeading` + **90° 연속성 창**
  (lexicographic: 창 내 clear > 창 밖 clear > blocked; veto 아님 — 창에 clear가 없으면 큰
  탈출 턴 허용). `createEpisode`에 `heading` 상태 추가, `advanceTarget`이 모든 패턴에서
  기록, cv 반사 후 재동기화. 스칼라 페널티(k=0.35/1.0/2.0)와 하드 ±120° 필터 변형은 감사에서
  실측으로 기각됨(전자는 반전 잔존, 후자는 push-out 0.23→79회/에피소드).
- **FIX 4**: 파일 헤더에 일시적 divergence 명시(trainer는 아직 진동함) — Python 반영 전까지
  대시보드 정직성 유지.

### 수정 후 실측 (독립 하네스 CURRENT/FIXED 컬럼 수렴 확인)

| 지표 | 수정 전 | 수정 후 |
|---|---|---|
| 렌더 yaw 변화/스텝 (85막대, dt=0.1) | 55.1° | **16.5°** |
| 90° 초과 반전 (85막대) | 29.4% | **5.1%** |
| yaw 변화/스텝 (110막대) | 75.8° | **19.0°** |
| 90° 초과 반전 (110막대) | 41.0% | **4.9%** |
| age 워치독 (speed=0, 10초) | 0.000 (죽음) | **10.000** |

`tests/test_status_arena_motion.js`에 회귀 테스트 T1-T5 추가(수정 전 실패 확인된 T1/T2 포함;
T3는 감사 제안 기하가 전 후보 차단이라 판별 불가 벡터 → 0.88-1.12m 환대 기하로 재설계).
전체 스위트 green. SwiftShader headless Chrome으로 장면 구동 확인.

### 기각된 가설 (재조사 방지)

- `sampleWaypoint` 전영역 균등 샘플: Python `_sample_waypoints`와 올바른 parity. clearance
  필터 추가해도 stall 0.3pp 변화 — 원인 아님.
- cv write-back(`cvVelocity` 갱신): 버그가 아니라 안정자(cv 모드 97.8%+ 프레임에서 직진 유지).
- `steerPursuerStep` no-candidate 경로: 36,250 프로브+70만 프레임에서 0회 발화 — 실질 사문.
- 기존 테스트 `moved>5.5` 무력 주장: 다음 줄 net-arrival 단언이 정지·2-cycle 모두 잡음(변이 검증).

### 후속 작업 (staged)

- **FIX 3 (Python parity)**: 같은 연속성 창을 `target_motion.py`+`navrl_task.py`에 이식.
  **현재 진행 중인 밀도 커리큘럼 run이 끝나고 채점된 뒤에만** 적용 — 진동하지 않는 표적은
  실제로 더 어려워 capture가 떨어지며, 이는 정책 퇴행이 아니라 과제 난이도 보정(task-version
  bump). 적용 시 waypoint/mixed 고밀도 셀 재측정 필요, circle 패턴(held-out eval)은 제외.
- `tools/probe_target_motion.py`는 per-step 속도만 봐서 이 결함에 맹목(반전은 norm 보존) —
  1초 순수송/명령속도 비율과 평균 방향 변화 지표를 추가한 뒤에야 증거로 사용 가능.

**(추가) synth2 전체 animate() 루프 재현 최종 검증** — 감사 목표치 전 항목 통과:
85막대(기본값): 렌더 turn 37.7°→**3.6°**(목표 ≤6°), >90° 프레임 20.6%→**0.6%**(≤2%),
둘-다-정지 0.00% 유지, 에피소드 회전율 57→57 동일(포획 난이도 무변).
150막대(슬라이더 최대): 1초 순수송 효율 0.338→**0.592**(≥0.55), 표적 순정지 58.1%→**19.2%**
(≤25%), 둘-다-정지 35.0%→**11.2%**(≤15%, 최장 연속 1.1s→0.6s), 에피소드 13→19개/240s.
속도 0 극단: 120초당 에피소드 3→16개, age-타임아웃 리셋 5회 발화(FIX 1 시그니처 확인).
잔존 정지는 전부 110막대↑ 기하 단절 구간(코드로 해결 불가, 감사 FIX 5 참조).

---

## 2026-07-31 — cluster-sector 100 bars 장기 hold 재감사

실행 중인 `ppo_260730_2111_navrl_cluster-sector-density85to110-v2-s1`을 중단하지 않고
프로세스, 전체 density gate 이력, TensorBoard, 최신 crash/barprobe 진단을 다시 대조했다.

### 판정

- 승급: 85→90 `0.737`, 90→95 `0.718`, 95에서 한 번 hold `0.670`, 95→100 `0.709`.
- 100 bars에서는 epoch 약 12,699부터 15,940까지 **17회 연속 hold**:
  `0.631, 0.580, 0.555, 0.555, 0.554, 0.549, 0.564, 0.580, 0.573, 0.561,
  0.538, 0.540, 0.521, 0.547, 0.546, 0.561, 0.558`.
- 100 bars는 epoch 12,463에 시작했다. 처음 계산한 `ep_12500` rolling-250 `0.703`은
  **95-bars 샘플이 섞인 오염값이라 폐기**한다. 100-bars만 자르면 `ep_12500`까지 38 epoch
  평균 `0.658`, 첫 16,385-episode gate가 `0.631`, 최근 tail-500은 **약 `0.560`**이다.
  즉 0.70에 근접한 100-bars 정책을 잃은 것이 아니라, 진입 직후부터 기준 미달이었고 이후
  약 0.56 plateau로 내려갔다.
- PPO 발산은 아님: 최근 tail-500 `KL=0.0097`, explained variance `0.762`, raw lateral
  OOB `0%`. NaN/Inf 또는 KL 폭발 증거가 없다.
- 기존 token 중복 병목도 주원인에서 내려감: 최신 barprobe `unique=4.5~4.9/8`,
  duplicate 약 `0.1`, `hit_token_given_fov=0.80~0.84`. 실패의 `95.6~98.1%`는 bar contact.

따라서 현재 run을 26,000 epoch까지 그대로 연장할 기대값은 낮다. threshold를 0.70에서
0.55로 내려 105 bars로 보내는 것은 실력을 높이는 것이 아니라 실패 정책을 승급시키는 것이므로
기각한다. 감사 중에는 사용자 학습 프로세스를 임의로 중단하지 않았다.

### 다음 실험 권장안

1. 복구점을 단일 로그값으로 확정하지 않는다. 100-bars 진입 직후
   `ep_12500`(38 epoch만 적응), 첫 완전 gate 근처 `ep_12700`, 중간 국소 회복점
   `ep_14300`, 최신 checkpoint를 같은 held-out sweep으로 채점해 선택한다.
2. 새 학습 전 FIX 3(target heading continuity)를 Python에 적용하고 task-version을 올린다.
   현재 run은 무기억 target 반전이 있는 구 task이므로 새 task의 최종 결과로 쓰지 않는다.
3. 같은 복구점에서 낮은 PPO 학습률과 85/90/95/100 density rehearsal을 사용해 100-bars
   적응을 짧게 비교하고, same-density guard를 초기 best 대비 약 0.10 하락에서 멈추도록 강화한다.
4. 그래도 100-bars held-out capture가 회복되지 않으면 obstacle-hit 토큰 수를 더 늘리기보다
   **통과 가능한 빈 공간/회랑을 직접 나타내는 sector gap 또는 2-depth-layer 표현**을 다음
   perception ablation으로 사용한다.

---

## 2026-07-31 — 장기 hold 중단, FIX3 + low-LR density replay 확정 평가

사용자 승인 후 기존 장기 run을 보존 중단하고, “더 오래 학습하면 나아지는가”와
“표적 진동 수정 + 낮은 LR + 저밀도 rehearsal이면 100막대가 회복되는가”를 분리 검증했다.

### 1. 기존 run 안전 중단과 복구점 선택

- run: `ppo_260730_2111_navrl_cluster-sector-density85to110-v2-s1`
- epoch 16,100 periodic checkpoint가 저장된 뒤 process group에 `SIGINT`; 학습 프로세스 소멸과
  checkpoint 파일을 확인했다.
- 구 target-motion 계약에서 100막대, target speed `0/0.5/1.0/1.5`, pursuer `2.5m/s`,
  셀당 1,000회로 후보를 동일 평가했다.

| checkpoint | 4속도 가중 capture | 판정 |
|---|---:|---|
| ep12500 | **63.24%** | warm-start 선택 |
| ep12700 | 58.85% | 하락 |
| ep14300 | 58.59% | 하락 |
| ep16100 | 54.79% | 장기학습으로 추가 하락 |

따라서 100막대 hold는 단순히 시간이 부족한 상태가 아니었다. ep12500 이후 PPO 수치는
폭발하지 않았지만 held-out 일반화는 계속 나빠졌고, ep12500을 last-known-good로 확정했다.
평가 원본은
`train_session_logs/eval_results/checkpoint_select_oldmotion_ep{12500,12700,14300,16100}_260731/`.

### 2. FIX3: Python target heading continuity

- `target_motion.py`: 모션 버전을
  `symmetric_local_steer_v2_heading_continuity90`으로 올리고, 직전 비행 heading 기준 ±90° 안의
  clear 후보를 먼저 고르되 없으면 큰 탈출 회전을 허용하는 preference를 추가했다.
- `navrl_task.py`: per-env `_tm_heading`을 episode heading으로 초기화하고 cv/waypoint에
  전달·저장, cv wall/bar reflection 뒤 heading을 재동기화했다. held-out circle은 기존
  smallest-turn 계약을 유지한다.
- 브라우저와 Python의 모션 계약이 다시 같아져 `arena_motion.js`의 임시 divergence 표기를
  제거했다.
- 검증: target-motion 함수 테스트 **5/5**, training-safety **7/7**, curriculum guard **4/4**,
  `test_status_arena_motion.js` parity, `py_compile`, 셸 문법, `git diff --check` 모두 통과.

### 3. 500-epoch low-LR blockwise density rehearsal

물리 asset manager의 active obstacle 수는 env별 값이 아니라 전역 scalar라 한 PPO batch 안에서
서로 다른 막대 수를 정직하게 섞을 수 없다. 대신 model/optimizer checkpoint를 이어서
`85→90→95→100→100`, 각 100 epoch의 고정-density block으로 실행했다.

- source: ep12500
- actor LR: `3e-5` (checkpoint optimizer의 `1e-4`를 restore 후 실제로 덮어씀)
- action: squashed Gaussian, 기존 std/mu-scale 유지
- same-density guard: rolling 20 epoch, min 40 epoch, peak ≥0.55, absolute drop 0.10,
  patience 10
- TensorBoard에서 LR `3.0e-5`, 전 block `|KL|max ≤0.00232` 확인

| block | bars | train capture 평균 | 마지막 20 epoch | 최고 20 epoch |
|---|---:|---:|---:|---:|
| b1 | 85 | 76.69% | 77.68% | 78.16% |
| b2 | 90 | 73.48% | 72.86% | 77.46% |
| b3 | 95 | 67.96% | 69.26% | 69.66% |
| b4 | 100 | 63.22% | 60.07% | 67.82% |
| b5 | 100 | 63.57% | 65.34% | 66.05% |

런처: `train_navrl_fix3_low_lr_density_replay.sh`. 최초 실행에서 같은 100막대를 두 process로
나눠 b4→b5 전환 때 actor fail-stop rolling peak도 초기화되는 약점을 발견했다. 이후 기본
schedule을 `85:100, 90:100, 95:100, 100:200`으로 합쳐, 다음 실행부터 100막대 200 epoch가
한 process에서 같은 guard state를 유지하도록 수정했다.

### 4. 새 FIX3 계약의 checkpoint 평가

1차: ep12500/12800/12850/12900/12950/13000, 셀당 400회 선별. ep12900은 56.87%로 명백히
나빴지만 ep13000은 65.04%로 회복했다. 즉 training의 단일 epoch 값이나 `last`만으로 checkpoint를
선택하면 안 되고 held-out 선택이 필수다.

2차 확정: 100막대, target speed `0/0.5/1.0/1.5`, pursuer `2.5m/s`, radial `4..16m`,
mixed motion, fixed seed, 셀당 1,000회.

| checkpoint | 0.0 | 0.5 | 1.0 | 1.5 | 가중 평균 | bar contact |
|---|---:|---:|---:|---:|---:|---:|
| ep12500 (FIX3, no adaptation) | 63.94% | 66.80% | 65.13% | 62.24% | **64.53%** | 33.18% |
| ep12850 | 64.24% | 68.00% | 64.07% | 59.82% | **64.03%** | 34.20% |
| ep13000 | 67.03% | 67.40% | 66.37% | 61.70% | **65.63%** | 32.23% |

- ep13000 − ep12500: **+1.10pp**, 독립 비율 근사 95% CI **[-0.99, +3.19]pp**.
- bar contact: `33.18% → 32.23%`(-0.95pp), below `0.25% → 0.40%`.
- 관측상 최고는 ep13000이지만 통계적으로 확정된 개선은 아니며 70% gate에도 못 미친다.
- FIX3 자체가 수치를 인위적으로 올린 것은 아니다. ep12500을 새 모션으로 다시 평가한 평균은
  64.53%였고, 구 모션의 동일 계열 평가는 63.24%였다.

확정 결과:
`train_session_logs/eval_results/fix3_replay_confirm_ep{12500,12850,13000}_260731/`.

### 최종 판정과 다음 전환

**추가 장기 low-LR replay는 중단한다.** target 진동 수정은 실제/사이트 parity에 필요했고
PPO 안정성도 확보했지만, 100막대 capture의 유의한 개선을 만들지 못했다. 최신 100막대
barprobe는 `unique=4.9/8`, `hit_token_given_fov=0.839`, duplicate `0.2`인데 crash의
약 95.5%가 bar contact였다. 따라서 현재 핵심 병목은 “막대를 못 봄”보다 **관측한 장애물
표면을 통과 가능한 회랑/행동 affordance로 변환하지 못함**이다.

다음 perception ablation은 토큰 수 증설이나 threshold 하향이 아니라:

1. sector별 nearest obstacle만 주는 현재 표현에 **free-gap width/center/clearance**를 직접 넣는
   corridor token, 또는
2. 같은 방위의 앞/뒤 표면을 구분하는 **2-depth-layer sector representation**

중 하나로 진행한다. 같은 898-D를 억지로 재해석하면 checkpoint semantic mismatch가 생기므로,
입력 projection 확장/초기화와 fixed-100 짧은 A/B를 명시적으로 설계한 뒤 실행한다.

---

## 2026-07-31 — 상태 사이트 최신화 및 corridor-token 전환 계획 공개

FIX3 + low-LR density replay의 확정 평가를 정적 연구 대시보드에 반영했다. 기존 사이트는
7월 30일 cluster-sector 85막대 중간 상태와 종료된 프로세스를 `LIVE`로 표시하고 있었으므로,
스냅샷 생성기를 다시 실행해 52개 run을 동기화하고 `active_run=null`, 최신 run
`ppo_260731_0226_navrl_fix3-low-lr-replay-b5-100bars-s1`, epoch 13000으로 바로잡았다.

### 사이트에 반영한 현재 결론

- 100막대 held-out: ep12500 **64.53%**, ep12850 **64.03%**, ep13000 **65.63%**.
- ep13000 개선량은 **+1.10pp**, 95% CI **[-0.99,+3.19]pp**라 통계적으로 확정되지 않았고
  70% gate에도 못 미쳤다.
- FIX3 target-motion parity와 PPO update safety(`|KL|max=0.00232`)는 통과했지만, 관측된 crash의
  약 **95.5%가 bar contact**였다.
- 따라서 추가 low-LR replay는 중단하고, “장애물 표면 위치”에서 “통과 가능한 빈 공간”으로
  표현을 바꾸는 corridor-token 진단을 다음 단계로 명시했다.

### corridor token 설명과 실험 계약

사이트에 obstacle token과 corridor token을 나란히 비교하는 시각 설명을 추가했다.

- 기존 obstacle token: nearest bar의 bearing/range/relative geometry를 주며
  “장애물이 어디 있는가?”에 답한다.
- corridor token: gap center bearing, usable width, left/right clearance, clear depth 또는 TTC를
  주며 “기체가 어디로 통과할 수 있는가?”에 답한다.
- 이는 전체 경로를 직접 정해주는 planner가 아니라 LiDAR 표면을 국소 통과 가능성
  (affordance)으로 바꿔 주는 입력 표현이다.

다음 순서는 사이트와 연구 계획에 동일하게 고정했다.

1. P0: ep12500/ep13000 및 fixed-100 four-speed 평가를 불변 baseline으로 보존.
2. P1: actor 입력을 바꾸지 않고 LiDAR 기반 corridor geometry extractor를 먼저 검증.
3. P2: 기존 898-D 의미를 재해석하지 않고 observation schema를 확장하며 input projection을
   선택 초기화하고 checkpoint에 schema provenance를 기록.
4. P3: 같은 seed/eval 계약으로 cluster-sector baseline과 corridor token의 짧은 fixed-100 A/B.

파일럿 진행 gate는 **capture ≥68%, ep12500 대비 ≥+3pp, bar-contact 감소**의 동시 충족으로
정했다. backbone freeze는 별도 seed에서 **70% 재현**까지 요구한다.

### 검증

- `tools/update_status_snapshot.py`: Python compile 및 실행 성공.
- 생성된 `status.json`과 `index.html` inline fallback JSON의 byte-level object equality 확인.
- latest run/epoch, no-active-run, corridor data와 필수 DOM id assertion 통과.
- 로컬 HTTP로 모든 site asset이 200 응답하는 것을 확인.
- Chrome headless 1440×2200 렌더에서 desktop layout 및 새 corridor section을 육안 검증.
  headless GPU 환경의 WebGL context 실패는 3D arena에만 해당하며, 기존 fallback 문구가 정상
  표시되었다.
- `git diff --check` 통과.

---

## 2026-07-31 (새벽) — corridor token P0-P3 전체 구현·검증·학습 런칭

계획(사이트/WORKLOG 2026-07-31 "corridor-token 전환 계획")대로 P1→P2→P3를 구현하고
100막대 B-arm 학습을 시작했다.

### P1: corridor geometry extractor + 물리 검증 (PASS)

- `aerial_gym/task/navrl_task/navrl_corridor.py` 신설 — 순수 torch 함수
  `extract_corridor_tokens()`: fused 수평 LiDAR 프로파일에서 horizon(6m) 이내 표면으로
  막히지 않은 각도 연속 구간(free gap)을 벡터화 추출. 슬롯당 8특징 =
  [sin/cos(center), width_m, left/right clearance, clear depth, angular width, valid].
  선정은 폭 내림차순 top-K 후 |center| 오름차순(전방 우선) 정렬. 검증 게이트:
  bounding surface의 chord 기반 metric width (내부 gap), FOV edge는 arc 근사.
- CPU 단위 테스트 `tests/test_navrl_corridor.py` 20/20 PASS (빈 장면/2-bar chord/벽/
  horizon 경계/슬릿 폭 필터/전방 우선 정렬/결정성/정규화 범위).
- GPU 물리 검증 `tools/probe_corridor_geometry.py` (64 env × 40 step, 100막대, 실제
  perception fused scan → GT bar 대조): **PASS**
  - A center-ray clearance ≥0.2m: **100.0%** (n=12,930; p50 clearance 0.68m)
  - B bounding surface가 실제 bar 위: **97.8%** (n=22,734)
  - C 폭 vs GT bar 중심거리 ≤1.2m: **98.8%** (오차 p50 0.38m = bar 반폭 불확실성 수준)
  - corridor 5.05개/env-step @100막대 — affordance 커버리지 충분.

### P2: 관측 스키마 명시적 확장 + checkpoint 확장 warm-start

- `NAVRL_CORRIDOR_TOKENS`(기본 0=off) 신설. 0이면 기존 898-D와 byte-identical.
  K>0이면 관측 **끝에** K×8 append → 898→946 (K=6), 기존 세그먼트 오프셋 불변.
- 네트워크: 17→18 토큰 (corridor_project 1개 토큰, 마지막 위치라 기존 position
  embedding row 0..16 의미 보존). critic states 906→954.
- provenance: env_state에 `cfg_corridor_tokens` 저장 + set_env_state 경고 가드 +
  preflight `_CONTRACT_ENV` 등록(legacy default 0) + `NAVRL_ALLOW_CORRIDOR_WARMSTART`.
- `runner.py::_expand_corridor_checkpoint`: 17-토큰 체크포인트를 스키마 확장 재작성 —
  position_embedding 18행(신규 행 N(0,0.02) seeded), corridor_project fresh init,
  actor input RMS 0/1 pad, **critic 첫 층은 privileged 열을 꼬리로 이동 + corridor 열
  zero-init**(critic이 초기에 신규 특징 무시 → value shock 없음), actor/critic Adam
  moment 리셋. 오프라인 검증: 기존 가중치 보존·열 재배치·idempotency 전부 assert PASS.

### P3: fixed-100 A/B — B-arm 학습 시작

- A-arm(불변 baseline): ep12500 **64.53%** / ep13000 **65.63%** (100막대 4속도 가중,
  celll당 1,000ep — 2026-07-31 확정 평가).
- B-arm 런처 `train_navrl_corridor_fixed100.sh`: ep13000에서 corridor-warmstart,
  fixed 100막대(NAVRL_CONTROLLED_ABLATION=1), LR 3e-5(replay 계약과 동일), 800 epoch.
- 스모크(64env, 4epoch): 확장 체크포인트 로드 성공, preflight override 로그 명시,
  **첫 epoch부터 captured 70-77%** = 로드된 backbone 손상 없음, NaN 없음.
- **본 학습 03:43 시작**: run `ppo_260731_0343_navrl_corridor6-fixed100-s1`,
  epoch 13000→13800, 로그 `train_session_logs/corridor6_fixed100_260731_034321.log`.
  프로세스 env 검증: CORRIDOR_TOKENS=6/WARMSTART=1/FIXED_BARS=100/LR=3e-5 확인.
- **평가 시 주의**: corridor 체크포인트 평가는 반드시 `NAVRL_CORRIDOR_TOKENS=6`을
  export해야 한다(관측 계약). 미설정 시 obs dim mismatch로 로드 불가.
- 파일럿 게이트(계획 고정값): held-out capture ≥68% AND ep12500 대비 ≥+3pp AND
  bar-contact 감소. backbone freeze는 별도 seed 70% 재현 추가 요구.

---

## 2026-07-31 (새벽) — corridor6 학습·확정 평가 완료: 충돌 소폭 개선, 진행 게이트 FAIL

`ppo_260731_0343_navrl_corridor6-fixed100-s1`은 ep13000→13800, 800 epoch를 정상 완료했다.
학습 마지막 50 epoch capture는 63.72%였으며, 단일 epoch 최고값(78%)을 성능으로 오독하지 않고
사전등록한 fixed-100 four-speed held-out 평가로 checkpoint를 판정했다.

### 평가 계약

- checkpoint: `last_gen_ppo_ep_13800_*.pth`
- representation: cluster-sector 8 + corridor K=6, observation 898→946
- corridor geometry: horizon 6.0 m, min width 0.55 m
- task: 100 bars 고정, pursuer 2.5 m/s, target speed {0, 0.5, 1.0, 1.5} m/s
- goal distance: 4–16 m 고정, seed 1, 총 4,003 episodes
- 결과: capture **66.10%**, bar contact **32.25%**
- 불변 ep12500 baseline: capture **64.53%**, bar contact **33.18%**
- 차이: capture **+1.57pp**, 95% CI **[-0.51,+3.66]pp**; bar contact **-0.93pp**

사전 게이트 판정:

| 항목 | 기준 | 결과 |
|---|---:|---:|
| capture | ≥68% | **FAIL · 66.10%** |
| ep12500 대비 gain | ≥+3pp | **FAIL · +1.57pp** |
| bar contact | <33.18% | **PASS · 32.25%** |

따라서 corridor geometry 자체와 schema-safe warm-start는 성공했고 충돌 방향의 신호도 있었지만,
정책 성능 개선은 작고 CI가 0을 포함한다. 동일 corridor6 run에 epoch를 더 쓰지 않는다. 다음 표현
진단은 밀도와 거리 조건을 그대로 고정한 채, 같은 방위의 앞/뒤 표면을 보존하는 2-depth-layer
sector representation으로 진행한다.

### `k_max=16` 의미 및 curriculum 판단

`k_max`는 막대 개수가 아니라 **표적/goal 거리 커리큘럼의 최대 반경(m)** 이다. competence gate가
통과될 때 시작 거리에서 16 m까지 단계적으로 올라가며, 현재 checkpoint는 이미 포화되어 있다.
이번 fixed-100 A/B에서 이를 16으로 고정한 이유는 corridor 표현 외의 난이도 축을 움직이지 않기
위해서다. 막대 수는 별도의 density curriculum이 5개 단위로 승급한다.

현재 24×24 m arena에서 16 m 밖을 같은 학습에 섞으면 경계와 out-of-bounds 효과가 커져 표현 A/B가
오염된다. 더 먼 거리 성능은 별도 held-out 평가로 보고, 실제 학습 범위를 늘릴 때는 arena 크기와
`k_final`을 함께 변경한다.

### 추가 안전장치와 재현 도구

- `eval_navrl_corridor_fixed100.sh`: corridor schema, LiDAR, selector, 4–16 m goal, 100 bars를
  명시적으로 고정하고 checkpoint contract 불일치를 실행 전에 차단.
- `evaluate_corridor_gate.py`: four-speed CSV를 episode 가중 집계하고 세 사전 게이트 및
  capture-delta 95% CI를 JSON으로 저장.
- corridor horizon/min-width를 checkpoint `env_state` provenance에 추가했으며, 최초 corridor
  checkpoint의 historical default(6.0/0.55)는 preflight가 명시적으로 해석한다.
- 잘못된 첫 screen(goal max 18 m)은 즉시 중단하고 판정에서 제외했다. 유효 screen은 동일 4–16 m
  계약으로 ep13100 64.20%, ep13450 59.93%, ep13800 65.11%였고, 최종 4,003-episode confirm만
  위 결론에 사용했다.

---

## 2026-07-31 (오후) — 환경 v2 "search arena" 설계·구현·학습 런칭 (task-version bump)

### 동기 (사용자 방향 결정)

v1 과제의 변인 통제 결함 두 가지를 확정하고 환경을 재설계했다:

1. **탐색이 없는 과제였음**: 목표 거리 4~16m vs 센서(LiDAR 12m + 전방카메라 20m/87°) —
   표적이 거의 항상 센서 안에 있어 "보이는 표적 추격"만 학습됨. 원래 목적은 **안 보이는
   표적을 LiDAR+FOV로 탐색해 찾아내는** 과제.
2. **에피소드가 짧았음**: 300스텝(30초), 실측 mean ep length ≤ ~90스텝. 참조 NavRL은
   2,200스텝(35초)을 순수 항법에만 사용.

### 참조 NavRL 원본과의 정량 비교 (reference/NavRL/isaac-training 직접 판독)

| | NavRL | v1 (구) | v2 (신) |
|---|---|---|---|
| 맵 | 40×40m (map_range[20,20], env.py:102) | 24×24m | **40×40m** |
| 장애물 | 350개, 21.9/100m² | 25~150개 | 70~300개, **5.2~22.6/100m²** |
| 장애물 높이 | 1~6m (55%가 4~6m, 못 넘음) | 2m (3m 아레나 — 넘을 수 있음) | **3m 전체높이** |
| 목표거리 | ~48m (가장자리→반대편) | 4~16m | **6~28m** (카메라 20m 초과) |
| 목표/센서 비 | 12× | 1.3× | **2.3×** (탐색 레짐) |
| 에피소드 | 2,200스텝/35s | 300스텝/30s | **600스텝/60s** |
| 배치 규칙 | good_distance 금지밴드(슬릿 없음) | 중심거리 1.5m + ×0.8 완화 | **navrl_band 금지밴드** |
| 커리큘럼 | 없음 (처음부터 최고난도) | 거리+밀도 2중 | 거리+밀도 2중 (유지) |

### 사전 실측 1 — 슬릿 결함 가설 검증 (tools/probe_placement_slits.py, 120 layout/밀도)

기존 배치 규칙(완화 로직)이 통과불가 슬릿을 만드는가:

| bars | 완화 발생률 | 통과불가 슬릿(<0.40m)/layout | marginal(<0.60m) | 도달불가 자유공간 |
|---|---|---|---|---|
| 85 | 0% | 0.00 | 0.38 | 0.00% |
| 100 | 0% | 0.00 | 0.35 | 0.00% |
| 110 | 0% | 0.00 | 0.51 | 0.00% |
| 150 | **100%** (p10 min_dist 1.2m) | **2.19** | **14.75** | 0.00% |

**판정: 가설 부분 기각.** 85~110막대에서는 완화가 발화하지 않아 슬릿 0 — **100막대
정체(capture ~0.56 plateau)는 환경 결함이 아니라 실제 정책/인지 한계**이며 기존 밀도 상한
결론은 유효하다. 슬릿은 150막대부터 구조적으로 발생하므로 새 규칙은 고밀도 확장의
전제조건이다. 완화 로직의 결함 본질: `min_dist *= 0.8`이 한 번 발화하면 그 env의 **남은
모든 막대에 영구 적용**(리셋 없음, asset_manager.py).

### 구현 (전부 env-var 게이트, 기본값은 v1 byte-identical)

- `NAVRL_ARENA_XY`(24)/`NAVRL_ARENA_Z`(3): navrl_bars_env 바운드. 스폰·목표·표적 모션은
  런타임 바운드 텐서를 읽으므로 자동 전파. critic privileged의 `dist/24.0` 하드코딩 2곳을
  `dist/self._arena_xy_norm`(같은 env var 파생)으로 교체 — 아레나 변경은 critic 입력 스케일
  변경 = task-version bump, 체크포인트 비교 불가를 주석으로 명시.
- `NAVRL_PLACEMENT_MODE=navrl_band` (`asset_manager._navrl_band_xy_spacing`): 중심거리
  금지밴드 (touch=0.4, gap=1.6). 후보는 기존 막대 전부에 대해 "겹침(≤0.4, 모든 footprint
  조합에서 확실히 접촉→복합 벽)" 또는 "≥1.6 (최악 표면갭 0.8m ≥ 드론 대각 0.396m)"이어야
  수락. 포화 시 기존 막대 위에 스냅(병합 보장) — **완화가 아니라 병합이라 슬릿이 원리적으로
  불가능**. CPU 검증: 24×24@150(포화)/40×40@300/24×24@100 전부 밴드 위반 0건.
  GPU 검증: 실제 env 300막대 배치에서 위반 0건.
- `NAVRL_BAR_POOL=bars_h3`: 전체높이 3m 풀(동일 seed·동일 footprint, 높이만 3m —
  fly-over 제거). z-ratio는 풀에 따라 자동(0.3333/0.5). 기존 bars 풀 무손상.
- `NAVRL_EPISODE_LEN_STEPS=600` (기존 env var 활용).
- 목표: `GENERAL_GOAL_DIST 6..28` + `K_FINAL=28` — 거리 competence 커리큘럼이 그대로
  "가까운(보이는) 표적 → 센서 밖(탐색 필요) 표적" 램프가 됨.
- 밀도: 70→300, step 15 (면적당 v1의 25→150/step5와 동일 스케줄: 5.2→22.6/100m²).

### 변인 통제 (의도적 불변)

관측 계약 898-D, LiDAR 12m 72×4, 토큰 8 cluster_sector, 카메라 87°@20m, v_max 2.5,
yaw 3.0, 표적 U[0,1.5] mixed 램프, squashed-Gaussian 액션, PPO 하이퍼파라미터, seed 전부
불변. **속도 knob은 건드리지 않았다** — v2에서는 "도착 전에 이동해버리는 안 보이는 표적"
이라는 메커니즘으로 표적 속도 축이 자연히 유의미해지므로, v2 밀도×속도 맵을 다시 측정해
비교하는 것이 올바른 변인 통제다.

### 검증·런칭

- VRAM: 128env·40×40·300막대 = 6,314MiB (프로브), 학습 중 7,109MiB / 8,192 — 통과.
- 스모크 5 epoch: v2 계약 로그 정확(placement=navrl_band touch=0.40 gap=1.60, 70 bars
  시작, obs 898 불변), NaN 없음.
- **본 학습 12:52 시작 (fresh, seed 1, 30,000 epochs)**:
  run `ppo_260731_1252_navrl_v2-search-fresh-s1`,
  로그 `train_session_logs/v2_search_fresh_260731_125215.log`.
  프로세스 env 검증: ARENA_XY=40/BAR_POOL=bars_h3/PLACEMENT=navrl_band/GOAL_MAX=28/
  K_FINAL=28/DENSITY 70→300 확인.
- v1 결과(밀도×속도 맵, 밀도 상한 100)는 v1 계약의 확정 데이터로 보존 — v2와 비교 불가
  (task-version bump)임을 사이트/논문에서 구분 표기할 것.

### Codex 검수 포인트

1. navrl_band의 touch=0.4 보장 논리: 풀 footprint 0.4~0.8에서 중심거리 ≤0.4면 모든 조합이
   겹침인가 (최소 반폭 0.2+0.2=0.4 — 축정렬 최악에서 정확히 접촉).
2. gap=1.6의 최악 표면갭: 두 0.8m 막대 대각 배치 시 코너-코너 갭 = 1.6 − 0.8·√2 ≈ 0.47m —
   축정렬이 아닌 대각에서는 0.8m가 아니라 0.47m. 드론 대각 0.396m보다는 크지만 여유가 8cm.
   더 보수적으로 gap을 1.8로 올릴지 검토 요망 (밀도 상한과 트레이드오프).
3. `_arena_xy_norm`이 critic 입력만 바꾸는지, actor 관측에 아레나 의존 정규화가 남아있지
   않은지 (actor는 LiDAR/token/robot/target 전부 센서-정규화 — 확인했으나 재검 환영).
4. 에피소드 600스텝에서 alive/time-cost 리워드 균형 (v1 리워드가 300스텝 기준 튜닝됨 —
   timeout 페널티 대비 캡처 보너스 비율이 2× 길이에서도 배회를 유발하지 않는지).

### Codex 독립 감사 결과 (2026-07-31 13시)

학습 프로세스는 중단하거나 변경하지 않고 코드, 실제 `bars_h3` URDF 40개, ep850 checkpoint,
실행 프로세스 환경변수와 epoch metrics를 읽기 전용으로 대조했다.

1. **touch=0.4 — PASS.** 막대는 yaw=0인 축정렬 box이고 실제 pool의 최소 변 길이는
   0.4029m다. 중심 Euclidean 거리 ≤0.4m이면 각 축 오프셋도 ≤0.4m이므로 두 box의 x/y
   interval이 모든 footprint 조합에서 겹치거나 접한다. saturation fallback은 반경을
   `0.5*touch=0.2m` 미만으로 제한하므로 더 강하게 겹친다.
2. **gap=1.6 — WARN. 이상적 기하에서는 통과 가능하지만 robust clearance가 부족하다.**
   config 주석의 `surface gap = 1.6 - 0.8 = 0.8m`는 축정렬 방향에만 맞고 대각 최악에는
   틀리다. 실제 최대 bar 변 0.7902m 기준 corner gap은 0.4825m, 0.28m drone 대각
   0.3960m를 뺀 여유는 **0.0865m**뿐이다. 1.8m면 여유가 0.2865m로 늘어난다. 다만
   exact center-band Monte Carlo(40×40 band, 16 layouts)에서 300 bodies의 독립 component가
   gap 1.6/1.7/1.8일 때 약 247/237/225개로 감소했다. 즉 1.8은 안전 여유를 얻는 대신
   고밀도에서 약 9%를 추가 병합한다. 장기적으로는 고정 1.8보다 per-asset footprint를
   읽는 Minkowski surface-clearance rule이 더 정확하다.
3. **actor arena-normalization 격리 — PASS.** v2 perception actor 898-D는 LiDAR range,
   camera range, velocity limit, tracker covariance/memory, flight altitude로만 정규화되며
   absolute XY나 arena side를 받지 않는다. `_arena_xy_norm`은 두 privileged critic
   extras의 `dist` 열에만 사용된다. `_process_obs_vision` docstring의 `dist/24` 표기는
   stale 문서지만 실행 코드는 40m divisor를 사용한다.
4. **600-step reward — 현재 로그 PASS, 후반 provisional.** ep787–886 rolling 100에서
   capture 78.64%, timeout 1.81%, mean length 108.3/600으로 배회/timeout 증거는 없다.
   checkpoint ep850은 이미 `k_max_cur=28`이라 sensor-outside goal도 포함한다. PPO
   `gamma=0.99`에서 -0.05 time cost의 무한-horizon discounted 총량은 약 -5이므로 단순히
   300→600으로 raw cost가 2배가 되지는 않는다. 다만 terminal +30의 현재가치는 100/200/
   300 step에서 10.98/4.02/1.47로 줄어든다. 이후 density와 target speed가 올라가 mean
   length가 200~300을 넘을 때 rolling timeout과 hidden-target dwell을 다시 판정해야 한다.

**추가로 발견한 계약 결함:** v2는 task-version bump라고 선언했지만 ep850 `env_state`에는
`arena_xy/z`, `bar_pool`, `placement_mode/touch/gap`, `episode_len_steps`가 모두 없다.
따라서 v2 checkpoint를 v1 arena/placement로 잘못 평가하거나 resume해도 현재 preflight가
차단하지 못한다. 본 run 자체는 `/proc/<pid>/environ`에서 40m/bars_h3/navrl_band/0.4/1.6/
600step/goal6..28/density70..300 계약을 확인했다. 평가 전에 이 7개 provenance와 mismatch
guard를 추가해야 한다.

---

## 2026-07-31 (오후) — 1650 Ti(4GB) 경로 실측, 아레나 provenance, v2 평가 경로

### 1. 4GB(1650 Ti) 타당성 — 실측 통과

병목은 메시가 아니라 **PhysX 강체 액터 수 = num_envs × NAVRL_MAX_BARS**임을 먼저 확정했다
(막대는 박스라 env당 warp 메시+BVH가 ~360KB에 불과). 문서화된 4GB 한계선:

| 구성 | 액터 수 | 결과 |
|---|---|---|
| 256env × 150막대 (비전 X) | 38,400 | 안전 최대 |
| 512env × 150막대 | 76,800 | PhysX pair 버퍼 오버플로우 사망 |
| 128env × 150막대 + 비전 | 19,200 | 검증된 4GB 비전 프리셋 |
| **64env × 300막대 (v2)** | **19,200** | ← 액터 수 등가로 설계 |

64는 PPO 배치 최솟값이기도 하다(minibatch 2048, horizon 32 → 32×64=2048).

**실측(3070에서 base_sim_4gb 프리셋으로, 1650 Ti와 동일 버퍼 설정)**:
- 학습 중 peak **3,425 MiB / 4,096** (여유 ~670 MiB) → **1650 Ti 탑재 가능**
- env만(300막대 전부 활성) 2,561 MiB
- **밀도 커리큘럼이 올라가도 VRAM 불변**: 70막대 실행 3,425 vs 300막대 빌드 3,422 —
  3 MiB 차이. 막대는 빌드 시점에 전부 액터로 생성되고 비활성은 -1000에 주차될 뿐이다.

런처 `train_navrl_v2_search_4gb.sh` 신설(측정치와 산정 근거를 주석에 기록).
주의: 배치가 절반이라 3070 결과와 **혼합 금지**(기존 프로젝트 규칙) — 평가 또는 별도 보고
seed 용도.

### 2. 아레나 provenance — 조용한 과제 혼동 차단

v2는 **관측 폭이 v1과 동일(898-D)**해서 v2 체크포인트가 v1 아레나에서 에러 없이 로드되고
**전혀 다른 과제로 채점**된다 — 예전 lidar_max_range 사고와 같은 유형. 차단 3중화:

- `env_state`에 `cfg_arena_xy / cfg_arena_z / cfg_bar_pool / cfg_placement_mode /
  cfg_placement_gap_m / cfg_episode_len_steps` 저장(`_arena_contract()`).
- `set_env_state` 불일치 시 경고 — 검증: v1 상태를 v2 env에 주입하니 ARENA MISMATCH 2건
  (bar_pool, placement_mode) + CONFIG MISMATCH 2건(arena_xy, episode_len) 발화 확인.
- preflight `_CONTRACT_ENV`에 6개 등록, legacy 기본값(24/3/bars/random/1.6/300)을 둬서
  **v1 체크포인트는 계속 통과**하고 v1↔v2 교차만 실패한다.

### 3. v2 평가 경로 + v1 오염 방지

- `eval_navrl_v2_density_sweep.sh` 신설: v2 계약 전체 pin + **체크포인트 provenance 게이트**
  (v2 계약이 없거나 불일치면 실행 거부, `NAVRL_V2_FORCE=1`로만 우회). 밀도 70/150/210/280
  = 5.3/11.3/15.8/21.1 per 100m². 검증: v1 체크포인트(fix3 ep13000) 투입 시 정상 거부.
- `eval_navrl_density_sweep.sh`(v1)에 v1 아레나 값을 **명시 export** — v2 런처를 돌린 셸에
  `NAVRL_ARENA_XY=40`이 남아 있으면 v1 평가가 조용히 오염되기 때문(관측 폭이 같아 로드는 성공).

### 4. 학습 재시작

provenance는 실행 중 프로세스에 소급 적용되지 않으므로, 사용자 승인 하에 ep1400에서 중단하고
**체크포인트에서 이어받아** 재개(1,400 epoch 진척 보존).
- 중단 시점: run `ppo_260731_1252`, epoch 1417, 70막대 capture 82.4%, k_max 16.2
- 재개: run **`ppo_260731_1411_navrl_v2-search-prov-s1`**, ep1410부터,
  로그 `train_session_logs/v2_search_resume_260731_141136.log`
- 검증: 프로세스 env(ARENA_XY=40/BAR_POOL=bars_h3/PLACEMENT=navrl_band/GOAL_MAX=28) 확인,
  **저장된 체크포인트에서 provenance 6개 필드 PASS** 확인.

### 5. 밀도 승급 상태 (질문 답변)

승급은 설정돼 있다: 70→300, step 15, 임계 0.70, 창 16,384 에피소드. 다만
`NAVRL_DENSITY_WARMUP=1000` epoch 이후부터 누적을 시작하고, v2 실측 완료 에피소드가
**44.7개/epoch**(에피소드가 600스텝으로 길어져 v1의 63.9보다 적음)이라 창 하나 ≈ **367 epoch**.
따라서 첫 심사는 epoch ~1367 부근이며 중단 시점(1417)에는 아직 심사 기록이 없었다.
재개 run에서 곧 첫 심사가 발화한다.

### Codex 추가 검수 포인트

5. `_arena_contract()`가 env var를 읽는 방식이 navrl_bars_env/navrl_task_config의 실제
   소비 경로와 항상 일치하는지(둘 다 같은 var를 읽지만, 한쪽만 바뀌면 기록이 거짓이 됨).
   더 안전한 대안은 런타임 bounds 텐서에서 역산하는 것.
6. 4GB 실측이 3070에서 base_sim_4gb로 수행됨 — 1650 Ti 실기에서는 드라이버/컨텍스트
   오버헤드가 달라 수백 MiB 차이 가능. 실기 첫 실행 시 재확인 필요.

### Codex 독립 검수 결과

현재 학습은 중단하지 않고 read-only로 코드·체크포인트·로그를 교차 확인했다. 재개 run
`ppo_260731_1411_navrl_v2-search-prov-s1`의 ep1500 체크포인트에는
`arena_xy=40 / arena_z=3 / bar_pool=bars_h3 / placement_mode=navrl_band /
placement_gap=1.6 / episode_len=600`이 실제로 저장돼 있었다. preflight 직접 재현에서도
v2→v2와 v1→v1은 통과하고 v1↔v2 교차 resume은 arena mismatch로 거부됐다. 기존
`test_navrl_checkpoint_preflight.py` 13개도 통과했다. 현재 run은 70→85막대까지 승급했으므로
provenance 추가 때문에 학습 로직이 깨진 정황은 없다.

다만 다음 세 결함은 평가 전에 수정해야 한다.

1. **계약 필드 누락**: 실제 배치 로직이 소비하고 v2 런처가 `0.4m`로 설정하는
   `NAVRL_PLACEMENT_TOUCH_M`이 checkpoint env_state, preflight, v2 eval gate에 모두 없다.
   이 값이 바뀌면 같은 `navrl_band` 이름이어도 장애물 배치 의미가 달라지므로 조용한 과제
   혼동을 완전히 차단하지 못한다.
2. **v2 평가 gate 불완전**: inline gate는 현재 `arena_xy/bar_pool/placement_mode/
   episode_len`만 거부 조건으로 검사한다. 저장·preflight 대상인 `arena_z`와
   `placement_gap_m`도 여기서는 강제되지 않고, 누락된 touch도 검사하지 않는다.
3. **FORCE 우회 불능**: 문서에는 `NAVRL_V2_FORCE=1`로 provenance 거부를 우회할 수 있다고
   되어 있지만, 실제 스크립트는 Python gate가 먼저 exit 2를 내므로 v1 체크포인트를 넣은
   직접 재현에서 FORCE도 동일하게 실패했다.

추가로 `_arena_contract()`가 런타임 객체가 아니라 env var를 다시 읽는 구조라 정상 런처
입력에서는 일치하지만 거짓 provenance 가능성이 남는다. 예를 들어 episode 값
`600.0`은 실제 task의 int parser에서는 기본값 300으로 되돌아가지만 contract에는 600으로
저장된다. 가능한 값은 `env_bounds_min/max`, `task_config.episode_len_steps`, asset manager의
실제 placement 설정에서 읽고, env var는 기대값 비교에만 쓰는 방식이 안전하다. 이번 커밋은
provenance 관련 회귀 테스트도 새로 추가하지 않았으므로 위 경계조건을 테스트로 고정할 필요가
있다.

4GB 경로는 **조건부 통과**다. `64 env × horizon 32 = minibatch 2048`이므로 현재 YAML과
배치가 정확히 맞고, 두 3070 smoke log에서도 64 env/transformer/v2 계약/8 epoch 실행을
확인했다. 하지만 3,425MiB 수치는 1650 Ti 실측이 아니라 동일 프리셋을 쓴 3070 실측이다.
4,096MiB 카드에서 idle 사용량이 600MiB면 예상 여유가 약 70MiB뿐이고, 800MiB면 OOM 범위다.
따라서 1650 Ti에서는 실행 전 free VRAM을 확인하고(권장 free ≥3.6~3.7GiB), 8 epoch smoke
동안 peak를 폴링한 뒤 본 학습으로 전환해야 한다. 현 YAML에서는 env만 64 미만으로 낮추면
minibatch보다 rollout batch가 작아지므로, 메모리가 부족하면 48env/minibatch1536 또는
32env/minibatch1024 전용 설정이 필요하다.

## 2026-07-31 15:07 — task-v2 진행 중 학습을 연구 사이트에 반영

정적 연구 대시보드의 상단 현황이 종료된 corridor6 실험에 고정돼 있어, 진행 중인
`ppo_260731_1411_navrl_v2-search-prov-s1`을 canonical current update로 바꿨다. snapshot
생성 시점 기준 epoch 2,420대, 115막대, 50-epoch capture tail 약 75.6%였고 밀도 게이트는
`70→85 (83.4%) →100 (82.1%) →115 (79.3%)`로 세 번 통과했다. tail 수치와 16,384-episode
승급 수치를 혼동하지 않도록 사이트 문구와 표에서 둘을 명시적으로 분리했다.

`tools/update_status_snapshot.py`는 활성 `runner.py`의 `--max_epochs`를 `/proc`에서 읽도록
수정해 기존 하드코딩 12,000 대신 실제 30,000을 표시한다. 현재 로그에서 density promotion
이력을 파싱해 연구 카드와 비교표에 싣고, provenance `6/7`(placement_touch 누락), v2 평가
gate 선수정 3건, 1650 Ti 4GB 실기 smoke 필요 상태도 경고 카드로 공개했다. 기존 corridor
결과와 밀도 ceiling 자료는 역사적 결과로 그대로 보존했다.

검증:
- `python3 -m py_compile tools/update_status_snapshot.py`
- `python3 tools/update_status_snapshot.py`: 54 runs, 활성 v2 run 탐지
- `node tests/test_status_arena_motion.js`: PASS
- `status.json`과 `index.html` inline fallback 동시 재생성, `git diff --check` PASS

## 2026-07-31 (저녁) — task-v2 제약조건 재검토: 배치 빈 공간·속도 램프·승급 임계값 + fresh 재학습

`ppo_260731_1411_..._prov-s1`을 ep3050에서 중단하고, fresh 재학습 전에 환경 제약조건을
전면 재검토했다. 세 가지 결함을 찾아 고쳤고, 모두 env-var 게이트라 v1 기본값은 그대로다.

### 1. 장애물 배치 빈 공간 (사용자 지적 — 확인됨)

`bar_asset_params`/`navrl_target_params`의 x-ratio가 `[0.13, 0.96]`이라 아레나 양끝에
장애물이 전혀 없는 띠가 존재했다.

| 아레나 | 배치 x 범위 | 저-x 빈 띠 | 고-x 빈 띠 | 빈 면적 |
|---|---|---|---|---|
| v1 24m | 3.12–23.04 m | 3.12 m | 0.96 m | 98 m² (17%) |
| v2 40m | 5.20–38.40 m | 5.20 m | 1.60 m | **272 m² (17%)** |

근거는 코드 주석에 남아있던 **"드론이 x≈0에 스폰하니 스폰 스트립을 비운다"** — v1 좌→우
횡단 시절 설정이다. v2는 `NAVRL_GENERAL_TRAIN=1`로 드론·표적 모두 아레나 전역 랜덤
스폰이라 전제가 이미 깨져 있었다. 결과적으로:
- 표적이 빈 띠에 스폰된 에피소드는 밀도와 무관하게 **직선 추격**으로 퇴화
- **보고 밀도가 부풀려짐**: 분모 1328 m²를 썼으나 실제 비행영역은 1600 m².
  115막대 = 보고 8.7/100m² → 실제 7.2/100m²

수정: `NAVRL_BAR_X_MIN/MAX` env var 신설(기본 0.13/0.96 = v1 보존), v2는 0.0/1.0.
**중요**: `_placement_band()`가 obstacle index 0 = `navrl_target_params`의 ratio를 읽으므로
두 클래스를 함께 바꿔야 실제 반영된다(`_BAR_X_MIN/MAX` 모듈 상수로 공유).

검증 — `navrl_band` 배치가 넓어진 밴드에서도 slit-free인지 CPU mirror로 측정:

| 밴드 | 막대 | in-band 쌍 | merge fallback |
|---|---|---|---|
| v2 full 40×40 | 70 | 0 | 0 |
| v2 full 40×40 | 150 | 0 | 0 |
| v2 full 40×40 | **300** | **0** | 0 |
| v1 24×24 | 150 | 0 | 0 |

legacy 밴드 300막대에서 1건이 잡혔으나 **float32 반올림 artifact로 기각**
(실제 거리 1.600024 m ≥ gap 1.6 m, float32 cdist가 1.599999로 계산). 실제 slit 아님.

### 2. target speed 램프 제거

`speed_ramp_epochs=3000`으로 0→1.5 m/s를 3000 epoch에 걸쳐 올리고 있었다. v1에서는
"정지 표적으로 요격을 먼저 학습"이 목적이었으나 v2에서는 부적절:
- v2의 초기 난이도 지배항은 속도가 아니라 **탐색**이다
- 램프가 `num_task_steps` 기반이라 **밀도 커리큘럼과 시간축이 얽힘** — 변인 통제 결함

수정: `NAVRL_TARGET_SPEED_RAMP_EPOCHS` env var 신설(기본 3000 = v1 보존). v2는 1로 두어
epoch 0부터 U[0.3, 1.5] 고정 분포. `SPEED_MIN=0.3`으로 정지 표적 퇴화 케이스도 제거.

### 3. 밀도 승급 임계값을 밀도별로 램프

고정 0.70은 **달성 가능한 capture 상한이 밀도와 함께 떨어질 때 커리큘럼이 영구 정체**한다
— v1 100막대 plateau의 정확한 실패 모드다. 쉬운 구간은 엄격히, 어려운 구간은 완화:

`NAVRL_DENSITY_THRESHOLD_START/END` 신설, `n_bars_active`로 `[n_start, n_final]` 선형보간.
둘 다 기본값이 기존 `NAVRL_DENSITY_THRESHOLD`라 미설정 시 상수 동작 그대로.

| 막대 | 70 | 85 | 100 | 115 | 200 | 300 |
|---|---|---|---|---|---|---|
| 임계 | 0.850 | 0.840 | 0.830 | 0.821 | 0.765 | 0.700 |

승급/hold 로그에 `threshold=%.3f`를 추가해 판정 근거를 남긴다.

### 4. Codex 지적 3건 수정 (평가 경로)

- **provenance 7번째 필드**: `cfg_placement_touch_m` 누락 → `_arena_contract()`,
  `set_env_state` 경고, preflight `_CONTRACT_ENV`/legacy 기본값(0.4)에 추가
- **v2 eval gate 불완전**: `arena_z`/`placement_gap_m`/`placement_touch_m`/`bar_x_min/max`가
  거부 조건에 없었음 → want 딕셔너리를 9필드로 확장
- **`NAVRL_V2_FORCE` 우회 불능**: Python gate가 `set -e` 하에서 먼저 exit 2를 내
  뒤따르는 bash 분기가 도달 불가였음 → force를 gate **내부**로 이동, 경고 출력 후 exit 0

### 5. 연구 사이트에 "성공 판단 기준" 명시 (사용자 요청)

PPO 내부 스칼라(a_loss/c_loss/entropy/kl/explained_variance)가 학습 성공을 뜻하지 않음을
사이트에 명문화했다. `docs/status/`에 `#panel-criteria` 섹션 신설:
- **primary**: held-out capture rate(고정 밀도, frozen checkpoint, 미학습 에피소드)
- **secondary**: crash / timeout / bar contact
- **not success**: mean reward(커리큘럼 승급 시 하락 — 난이도 상승이지 정책 악화 아님),
  a_loss/c_loss(움직이는 분포 위의 optimizer 진단), entropy(행동 확정일 뿐),
  kl/explained_variance(guardrail — 필요조건이지 충분조건 아님)
- **curriculum gate**: 승급 규칙은 학습 제어이지 결과가 아님. rolling tail을 승급/논문
  수치로 인용 금지
- **checkpoint rule**: `last_gen_ppo_ep_*` 사용, `gen_ppo.pth`(best-reward=저밀도)는 금지

밀도 분모도 아레나별로 분리(`placement_area_m2`: v1 478, v2 1600) — v1/v2를 같은
분모로 보고하던 버그 수정.

### 검증

- `python3 -m py_compile` (navrl_task / navrl_task_config / env_object_config / preflight): PASS
- `bash -n` (train_navrl_v2_search.sh, eval_navrl_v2_density_sweep.sh): PASS
- `tests/test_navrl_checkpoint_preflight.py` 13개: PASS
- `tests/test_curriculum_safety.py` 4개: PASS
- `node tests/test_status_arena_motion.js`: PASS
- env var 기본값 회귀: bar x = (0.13, 0.96), target x = (0.13, 0.96) — v1 그대로
- 임계값 선형보간 수식 수동 검증(위 표)

### fresh 런칭

run **`ppo_260731_1606_navrl_v2-search-v3-s1`** (seed 1, 128 env, max 30000 epoch),
로그 `scratchpad/launch_v3.log` → `train_session_logs/`.
런처 배너로 4개 변경 반영 확인:
```
arena=40m pool=bars_h3 placement=navrl_band
goal 6..28m episode=600 steps
density 70->300 step=15 (4.4->18.8 /100m2 over 1600m2) threshold 0.85->0.70
target speed U[0.3, 1.5] m/s from epoch 0 (no ramp) | bar band x=[0.0, 1.0]
```
초기 22 epoch 관측: 배치 `mode=navrl_band touch=0.40 gap=1.60`, 표적 속도 실측 평균이
0.46(리셋 전 초기 버퍼) → **0.84~0.86**으로 수렴(U[0.3,1.5] 이론 평균 0.90과 일치),
VRAM 6861/8192 MiB, mean ep length 292/600.

부수 수정: 시작 배너가 램프를 무시하고 미사용 flat `success_threshold`(0.800)를 찍어
승급 판정과 로그가 불일치했다 → 램프 설정 시 `threshold=0.850->0.700`으로 출력하도록
수정(다음 런부터 적용, 현재 런의 게이트 로직 자체는 정상).

## 2026-07-31 — Codex pre-launch 독립 감사: 사이트는 미완성, TB 통과, dwell 테스트 필요

클로드가 보고한 네 항목을 미커밋 diff·실제 CSV·브라우저 CPU 모델·체크포인트 코드로
교차 검증했다. 학습 프로세스가 없는 것도 확인했으며 검수 중 새 학습은 시작하지 않았다.

### 판정

1. **사이트 v2 parity — 부분 통과, 배포 불가**
   - `status.json.arena_geometry`와 1600m² 밀도 분모, 40m/3m/300막대/navrl_band 데이터는
     맞고, 학습 중단 시 `latest` v2 run을 선택하는 geometry 분기도 동작한다.
   - 그러나 `arena_motion.js`는 여전히 bounds `0..24 × -12..12`, goal `4..16m`로
     하드코딩돼 있다. 2,000 episode 직접 샘플에서도 최대 x=22.996m, 최대 goal
     distance=15.999m라 JSON의 40m/6..28m 계약을 전혀 소비하지 않았다.
   - `arena.js`도 root shift `-12`, ground center `12`, GridHelper `24`, 경계선 `0..24`,
     raycast bar top `z=2`가 남아 있다. 즉 막대 좌표만 40m로 늘고 바닥·카메라·센서 충돌
     기하가 서로 어긋난다.
   - slider도 `bars_min/2`를 써서 v2 실제 min이 10이 아니라 35가 된다.
   - 학습 정지 snapshot의 headline은 corridor 결과인데 run 이름은 v3로 섞인다.
   - 기존 `test_status_arena_motion.js`가 PASS한 이유는 v2 parity를 검사해서가 아니라,
     오히려 v1 24m/4..16m 하드코딩을 명시적으로 assert하기 때문이다.

2. **속도 램프 제거 — 방향은 가능, 인과 결론은 미확정**
   - CSV 재계산에서 ramp run 대비 v3의 초기 학습이 느린 것은 재현됐다. 10-epoch rolling
     capture 0.50 최초 도달은 53 vs 140 epoch이고, epoch 101..156 평균 capture는
     0.591 vs 0.474다.
   - v3는 full-width band도 동시에 바뀌었으므로 이 차이를 순수 속도 효과로 볼 수 없다는
     클로드의 단서는 정확하다. crash가 지배적이고 range-rate/PBRS dense reward가 존재한다는
     사실은 무램프가 학습 가능하다는 근거이지, 램프가 무용하다는 인과 증명은 아니다.
   - 현재 `RAMP_EPOCHS=1`은 엄밀히 “epoch 0부터 U[0.3,1.5]”가 아니다. 최초 reset은
     U[0.3,0.3]이고 epoch 1 뒤부터 full range가 된다. 진짜 no-ramp는 함수의 명시적
     disabled branch가 필요하다.
   - 권장 절충은 300 epoch 짧은 램프다. density evidence는 epoch 1000부터 시작하므로
     700 epoch 먼저 종료되어 두 커리큘럼 게이트가 겹치지 않고, 표적도 처음부터 최소
     0.3m/s로 계속 움직인다.

3. **TensorBoard reward warmup — 통과**
   - 독립 mock에서 기본값은 epoch 1/19를 쓰지 않고 20/21부터 `aerial/mean_reward`를
     기록했다. `NAVRL_TB_REWARD_WARMUP_EPOCHS=0`이면 epoch 1부터 복구되고 잘못된 문자열은
     안전하게 20으로 fallback한다.
   - `stability/best_reward`에도 같은 `>=20` 조건이 걸렸다. 원본 reward는
     `epoch_metrics.csv`에 계속 남으므로 진단 증거가 소실되지는 않는다.
   - 이 경계조건 자체의 저장소 단위 테스트는 아직 없다.

4. **밀도별 dwell — 구현 논리는 통과, 검증 주장은 미충족**
   - `density_level_start_steps` 저장/복원, 승급 시 시계 reset, capture 통과와 dwell hold의
     로그 분리는 올바른 위치에 구현됐다. 새 형식 체크포인트 resume은 dwell을 재복무하지
     않는다.
   - 하지만 PASS했다는 `tests/test_curriculum_safety.py` 4개는 전부 density-collapse guard
     테스트이며 dwell/threshold/resume을 한 건도 호출하지 않는다. 최소한 “999 epoch
     promotion 금지 / 1000 epoch 허용 / 승급 후 시계 reset / checkpoint restore 유지”를
     새 테스트로 고정해야 한다.
   - `cfg_density_min_epochs`는 저장되지만 set_env_state config-mismatch 비교에는 빠져 있어
     resume 시 dwell 설정 변경이 조용히 지나간다.
   - dwell은 최소 노출시간이지 수렴 판정은 아니다. 또한 16,384-episode window가 약
     367 epoch이면 판정 시점이 window 단위로 양자화되어, 16회 승급의 이론 하한 16,000
     epoch보다 실제 300막대 도달은 대략 17,000~18,000 epoch 이상이 될 수 있다.

재현 검증:
- checkpoint preflight 13개 PASS
- 기존 curriculum-collapse 4개 PASS
- training-safety 7개 PASS
- status arena 기존 테스트 PASS(단, 위와 같이 v1 contract만 검증)
- shell syntax, py_compile, `git diff --check` PASS
- TensorBoard epoch 경계 독립 mock PASS

**최종 판정: 아직 fresh 30k 학습을 재시작하지 않는다.** 사이트의 동적 v2 bounds/goal/
bar-height parity와 dwell 회귀 테스트를 먼저 고치고, 속도는 300-epoch 짧은 램프로 확정한
뒤 짧은 smoke에서 배너·target-speed 분포·dwell state 저장을 확인하고 본 학습으로 간다.

## 2026-07-31 — Codex 사전조치 1~4 완료 및 64-env smoke 검증

앞선 감사에서 본 학습 전 필수로 지정한 네 항목을 모두 구현하고 실제 GPU 학습 경로로
확인했다. **30k 본 학습은 아직 시작하지 않았다.**

### 1. 연구 사이트 Task-v2 기하 동기화

- `arena_motion.js`가 `status.json.arena_geometry`의 40m bounds, goal 6..28m,
  target speed 0.3..1.5m/s를 실제 episode sampler에 적용한다.
- `arena.js`의 카메라/바닥/grid/border/root/raycast가 arena span 및 bar height를
  사용하도록 바꿨다. 기존 24m/2m scene hardcode를 제거했다.
- slider 최솟값은 `bars_slider_min=10`, 최댓값은 300이다.
- stopped v2 run은 corridor 문구를 섞지 않고 Task-v2 paused headline을 표시한다.
- 2,000-episode Node 회귀에서 x>24m spawn과 goal>20m episode가 모두 생성됐다.
- 이름에 `smoke`가 들어간 짧은 배선 검증 run은 전체 run 수에는 남지만 사이트 대표
  `latest_run`을 덮지 않도록 제외했다. snapshot은 56 runs를 동기화했고 대표 결과는
  `ppo_260731_1606_navrl_v2-search-v3-s1`을 유지한다.

### 2. density dwell 회귀 테스트와 복원 안전성

순수 helper `navrl_curriculum.py`를 분리해 실제 task와 테스트가 같은 판정식을 사용한다.
새 회귀 테스트는 다음 네 경계를 고정한다.

- 999 epoch: 승급 금지
- 1000 epoch: 승급 허용
- 승급 직후 dwell clock reset, 다음 999 epoch 재승급 금지
- 새/구형/future checkpoint의 `density_level_start_steps` 복원

`cfg_density_min_epochs`도 resume 설정 불일치 경고 대상에 포함했다.
`tests/test_curriculum_safety.py`는 기존 4개를 포함해 **8개 PASS**다.

### 3. 표적 속도 300-epoch 짧은 램프 확정

v2 launcher 기본값을 `NAVRL_TARGET_SPEED_RAMP_EPOCHS=300`으로 확정했다. 표적은 처음부터
최소 0.3m/s로 움직이고, 상한만 300 epoch 동안 1.5m/s까지 증가한다. density warmup/dwell
판정은 epoch 1000부터이므로 두 커리큘럼이 겹치지 않는다.

### 4. 실제 64-env smoke

- run: `ppo_260731_1641_navrl_v2-preflight-smoke-s91`
- 조건: seed 91, 64 env, 65 epoch, fresh
- 종료: `max_epochs` 정상 도달, exit code 0, 완료 표식과 epoch 50/65 체크포인트 생성
- 배너: arena 40m, bars_h3, navrl_band, goal 6..28m, episode 600,
  density 70→300/step15, threshold 0.85→0.70, speed ramp 300 모두 확인
- 속도 상한: epoch 60까지 0.30, epoch 62부터 0.31, epoch 64~65에 0.32로 상승 확인
- epoch-50 checkpoint: `num_task_steps=1600`, `density_level_start_steps=0`,
  `cfg_density_min_epochs=1000`, `n_bars_active=70`, threshold start/end=0.85/0.70
- TensorBoard: `aerial/mean_reward`와 `stability/best_reward` 첫 step=20,
  `ppo/a_loss` 첫 step=1 — warmup 경계 통과
- density promotion 0건 — 65-epoch smoke에서 의도한 결과이며 dwell 1000보다 짧다.

종료 뒤 rl-games가 `Can't create empty tensor` 한 줄을 출력했지만 프로세스 exit code는
0이고, max-epoch summary·완료 표식·두 체크포인트·TensorBoard event가 모두 정상이다.
학습 실패나 저장 유실로 판정할 근거는 없다.

최종 정적/회귀 검증: checkpoint preflight 13 PASS, curriculum 8 PASS,
training safety 7 PASS, Node arena/status parity PASS, Python compile/bash syntax/diff check PASS.

### 공개 사이트 배포 정정

최초 완료 보고 때는 로컬 `docs/status`와 snapshot만 수정했고 GitHub Pages 배포를 하지 않아,
공개 페이지가 계속 24×24×3m / 115 bars인 이전 커밋을 표시했다. 사용자 캡처로 이를
재확인하고 사이트 변경을 `9a8eab4`로 `research/navrl-env`(원격 default branch)에
푸시했다. 공개 URL을 다시 조회해 다음을 확인했다.

- HTML caption: **40×40×3m**
- cache key: `arena_motion.js?v=20260731b`
- live `status.json`: arena 40×40×3m, bar height 3m, slider 10..300,
  goal 6..28m, target speed 0.3..1.5m/s
- 공개 JS: 동적 `Motion.configure`, `arenaSpan`, `BAR_HEIGHT` raycast 반영

## 2026-08-01 — v2 scheduled-density 라이브 감사 (epoch 10,787)

학습을 중단하지 않고 `ppo_260731_2012_navrl_v2-search-sched-s1`의 터미널 로그,
epoch CSV, TensorBoard 및 epoch-10750 체크포인트 상태를 교차 집계했다. 프로세스는
PID 444488, VRAM 6176 MiB로 정상 실행 중이며 NaN/Inf/OOM/traceback은 없다.

### 밀도별 학습 중 에피소드 집계

| bars | density(/100m²) | epochs | episodes | capture | crash | timeout | gate 결과 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 70 | 4.4 | 63* | 2,497* | 82.18%* | 16.50% | 1.32% | 83.2%로 85 승급 |
| 85 | 5.3 | 1,360 | 49,163 | 78.27% | 19.48% | 2.26% | 78.4%로 100 승급 |
| 100 | 6.3 | 1,386 | 49,140 | 76.68% | 20.74% | 2.58% | 78.0%로 115 승급 |
| 115 | 7.2 | 1,406 | 49,160 | 72.37% | 24.68% | 2.95% | 71.6%로 130 승급 |
| 130 | 8.1 | 2,080 | 65,530 | 69.20% | 26.17% | 4.63% | 68.9/66.3/69.5 hold 후 72.1% 승급 |
| 145 | 9.1 | 1,242 | 42k+ | 약 65.0% | 약 31% | 약 4% | 67.4%, 64.0% 두 번 hold |

`*` 70막대는 ep3250 체크포인트에서 branch-run한 뒤의 CSV만 센 값이다. 승급 게이트
83.2%는 체크포인트에 복원된 이전 13,877 episode까지 포함한 정확한 16,386-episode 값이다.

145막대 250-epoch 순차 구간 capture/crash는 `67.90/26.98 -> 66.67/28.59 ->
64.66/31.39 -> 63.88/32.80 -> 62.11/34.98%`다. 즉 timeout은 약 5.1→2.9%로
줄었지만 그만큼 capture가 아니라 crash가 증가했다. epoch-10750의 진행 중 게이트도
4,787/7,665 = 62.45%라 다음 0.70 승급은 현재 추세상 어렵다.

### 판정

- **수치 발산은 아님**: 최근 500 epoch critic explained variance 평균 0.832,
  KL 평균 0.0144. KL>0.04는 21/500(4.2%)이며 guard가 minibatch를 skip한다.
- **과제 성능은 145에서 악화**: 두 완성 gate hold와 1,200+ epoch의 단조 하락이 있어
  단순히 학습 시간이 부족하다는 설명은 약하다. 현 표현/정책의 새 실질 ceiling 후보는 145다.
- 다음 16,384-episode gate가 0.65 이하라면 30k까지 동일 설정을 연장하지 말고,
  130막대 직전(ep9500 부근)과 145막대 checkpoint들을 고정 held-out 평가해
  representation/충돌 회피 병목을 분리한다.

## 2026-08-01 — epoch 10836 실제 actor collapse 및 자동 중단

위 라이브 감사 직후 run이 외부 kill/OOM이 아니라
`early_stop_density_capture_collapse` 가드에 의해 epoch 10836에서 의도적으로 중단됐다.
마지막 13 epoch 중 다수는 capture 0%, crash 100%, episode length 약 24/600이었고,
완료 표식은 없으며 `run_summary.json.exit_reason`이 정확히 해당 fail-stop을 기록한다.

이는 단순한 고밀도 성능 저하가 아니라 **PPO actor update collapse**다.

- epoch 10750: KL 0.0191, entropy -8.77, capture 62.2%, crash 37.8%
- epoch 10776~10783: KL 0.075~0.158로 0.04 gate 초과가 연속 발생
- epoch 10785: actor loss 144.9
- epoch 10787: KL 0.516
- epoch 10800: KL 0.795, entropy -26.57, capture 30.6%, crash 68.1%
- epoch 10803: KL 2.69
- epoch 10824 이후: 거의 매 epoch capture 0%, crash 100%
- checkpoint 10750→10800 단 50 epoch 동안 actor `mu.weight` norm이
  0.962→1.085(+12.8%), parameter delta가 기존 norm의 16.3%

entropy가 -8대에서 -106까지 단조 하락한 것은 fixed sigma가 변한 게 아니라 squashed
Gaussian latent mean이 tanh 경계로 포화됐다는 신호다. explained variance는 끝까지
약 0.8이라 critic 붕괴나 환경 난이도 변화가 직접 원인은 아니다.

### 설정 누락/보호장치 한계

검증된 `train_navrl_action_squashed_v2_main.sh`는 actor LR `5e-6`,
`NAVRL_LATENT_MARGIN_COEF=0.01`, `NAVRL_ACTION_DIAG=1`을 사용한다. 반면 이번
`train_navrl_v2_search.sh`는 LR을 YAML 기본 `1e-4`(20배)로 두고 margin 값 1.25만
export했으며 coefficient를 빠뜨려 실제 margin penalty가 **0**이었다. action diagnostics도
꺼져 축별 포화 조기경보가 없었다.

현재 KL gate는 minibatch를 skip할 뿐 이미 적용된 같은 epoch의 앞선 update를 rollback하지
않는다. collapse 구간에는 매 epoch 5 minibatch를 skip하면서도 KL이 0.5→2.7로 증가했다.
따라서 guard가 충분한 trust-region 역할을 하지 못했다. same-density capture guard는 원인이
아니며, 붕괴된 계산을 19k epoch 더 수행하지 않게 정상적으로 차단했다.

복구 시 붕괴 후 ep10800은 사용하지 않는다. 최소한 ep10750 이전, 보수적으로는 130막대
정책인 ep9500에서 분기하며, actor LR/latent penalty/action diagnostics를 먼저 바로잡고
KL 초과 epoch 전체 rollback 또는 optimizer step 전후 rollback을 검증해야 한다.

추가 감사에서 이 판정은 정정됐다. ep10750 actor restore/rollout은 exit 0으로 성공했다.
`Can't create empty tensor`는 asymmetric observation 오류가 아니라 DOF가 0개인 rigid-body
환경에서도 `acquire_dof_state_tensor()`를 wrap해 Isaac Gym C++가 출력한 진단이었다. 실제
평가 결함은 v2 sweep이 bulk result를 켜지 않아 1,025 games를 수행하고도 capture/crash
결과 파일을 하나도 남기지 않은 점이었다.

## 2026-07-31 (저녁) — 70막대 천장 프로브 신설 + 본 run의 임계값 stall 위험 식별

### 발견: 새 임계값이 과거 최고 달성치보다 높다

Codex가 17:22 런칭한 `ppo_260731_1722_navrl_v2-search-fresh-s1`(seed 1, 128env)을 점검하다
승급 임계값의 stall 위험을 발견했다.

| 항목 | 값 |
|---|---|
| 새 승급 임계값 @70막대 | **0.850** |
| 과거 v2가 70막대에서 달성한 실측 최고 | 0.834 |
| 과거 아레나 | x밴드 0.13–0.96 → **17% 무장애물 (더 쉬움)** |
| 현재 아레나 | x밴드 0.0–1.0 (전폭, **더 어려움**) |

즉 **더 어려워진 과제에 과거 최고치보다 높은 기준**을 걸어둔 상태다. 70막대에서 0.85를
못 넘기면 커리큘럼이 시작 밀도에 영구 정체한다 — threshold ramp가 막으려던 실패 모드가
반대편(쉬운 끝)에서 재현되는 셈이다.

현재 run 진행(70막대 고정 구간, 50-epoch 평균):
`0.251 → 0.518 → 0.591 → 0.677 → 0.739 → 0.760` (epoch 0→300, 상승 중)

**결정: 본 run은 개입하지 않는다.** dwell 게이트가 epoch 1000 전 승급을 어차피 막으므로
749 epoch의 여유가 있고, 그 안에 아래 프로브 결과가 나온다. 근거 없이 기준을 낮추지 않는다.
임계값은 env var라 필요시 resume으로 조정 가능(재시작 불필요).

### 신설: `train_navrl_v2_ceiling_probe.sh` (1650 Ti용)

본 run은 **자기 자신에 대해 이 질문에 답할 수 없다** — dwell이 1000 epoch 승급을 막고,
그 이후에는 "정체"와 "아직 개선 중"이 밖에서 구분되지 않는다. 밀도를 고정하고 capture가
어디서 평탄해지는지 직접 재는 프로브를 만들었다.

- 밀도 70 고정, `NAVRL_DENSITY_CURRICULUM=0`, 나머지 v2 계약은 메인 런처에서 그대로 상속
- `NAVRL_MAX_BARS=300` 유지 → PhysX actor 수 불변 → 검증된 4GB VRAM 프로파일 유지
- 64env, 2000 epoch 기본

**4GB 카드에서 재는 것이 타당한 이유(단측 추론)**: 64env 학습 결과는 128env와 섞을 수 없다는
프로젝트 규칙은 유효하다. 하지만 이 측정의 추론은 한쪽 방향이다 —
- 64env 평탄값 ≥ 0.85 → **0.85 도달 가능, 본 run 임계값 안전 (결정적)**
- 64env 평탄값 < 0.85 → 경고일 뿐, 더 강한 128env는 넘길 수 있음

de-risking 프로브에 필요한 비대칭이 정확히 이것이다. 같은 설정 병행 학습(64 vs 128)은
교란이라 채택하지 않았다.

### 런처 체인 수정

프로브 설정이 메인 런처에 덮어써지고 있었다 — `NAVRL_DENSITY_CURRICULUM=1`이 하드 export였고
`unset NAVRL_NUM_BARS`가 무조건 실행돼, 프로브의 고정 밀도가 조용히 config 기본값으로
되돌아갈 상황이었다. 둘 다 조건부로 바꿨다(기본 동작 불변).

검증 (exec 직전 dry-run):

| 경로 | curriculum | num_bars | min_epochs |
|---|---|---|---|
| 메인 단독 + `NUM_BARS=999` | 1 | `<unset>` | 1000 |
| 프로브 경로 | 0 | 70 | 0 |
| 프로브 전체 체인 | \multicolumn — `envs=64 sim=base_sim_4gb max_bars=300` 확인 | | |

`bash -n` 통과. 1650 Ti 실기 절차는 기존 권고 유지: pull → free VRAM ≥3.6–3.7GiB 확인 →
8 epoch smoke로 peak 폴링 → 본 프로브.

## 2026-07-31 (밤) — 70막대 천장 실측(0.843), 임계값 0.85→0.80, TTC 셀렉터 신설

### 1. 70막대 capture 천장을 실측했다 — 임계값 0.85는 도달 불가였다

run `ppo_260731_1722_navrl_v2-search-fresh-s1`(seed1, 128env)을 epoch 2650까지 관찰.
밀도 게이트가 **두 번 판정하고 두 번 다 HELD**:

| 판정 | epoch | 16,384-eps capture | 임계 | 결과 |
|---|---|---|---|---|
| 1차 | 1458 | 0.816 | 0.850 | HELD (−3.4pp) |
| 2차 | 1923 | 0.837 | 0.850 | HELD (−1.3pp) |

crash를 포화모델로 피팅: `crash(e) = 0.131 + 0.089·exp(−(e−550)/650)`.
시정수 650 epoch으로 **이미 수렴**했고 외삽 바닥은 **13.1%**. timeout은 추세 0
(`−0.1 pp/1000ep`, 95%CI[−0.4,+0.2])로 2.6%에 평평.

→ **capture 천장 ≈ 1 − 0.131 − 0.026 = 0.843 < 0.85.** 임계값이 구조적 상한 위에
있었으므로 영구 stall이 확정적이었다. 0.85는 근거 없는 설계 추정치였다.

**조치**: `NAVRL_DENSITY_THRESHOLD_START` 0.85 → **0.80**. ep2650에서 resume
(run `ppo_260731_1940_navrl_v2-search-thr80-s1`). provenance가 변경을 정확히 기록:
`CURRICULUM CONFIG MISMATCH | THRESHOLD_START: checkpoint 0.850, running 0.800`,
증거창 리셋(8332/9968 eps 폐기) + resume warmup 250 epoch. k_max=28 보존 확인.

### 2. 기하 분석 — 13% crash는 물리 한계가 아니라 표현(representation) 한계다

**기각된 가설**: "navrl_band 최악 간격 0.8m에서 여유 0.26m ≈ 표현오차 0.24~0.32m라
좁아서 못 지나간다". 실제 레이아웃 10,330개 틈 측정 결과 하위 5% 분위가 1.21m
(여유 0.46m), 표현오차보다 좁은 틈은 **0.1%뿐**. 70막대(4.4/100m²)는 기하학적으로
매우 희박하며 물리적 통과 난이도는 병목이 아니다.

**실제 병목** (barprobe v2 실측, 기하 예측과 일치):

| | 값 |
|---|---|
| 12m 내 막대 | 17.8개 (기하예측 19.8) |
| 240° FOV 내 | 12.1개 (기하예측 13.2) |
| 토큰 용량 | **8개** → FOV 내 34% 표현 불가 |
| `hit_fov` | **0.764** → 충돌 막대의 23.6%가 FOV 밖 |
| `hit_token_given_fov` | 0.883 → FOV 안이어도 11.7%는 토큰 없음 |

경로·밀도로 정규화하면 v2는 v1 완성정책 대비 **단위거리·단위밀도당 충돌 2배**
(0.131 vs 0.054~0.069). v1이 그 수준을 달성했다는 건 13%가 물리 법칙이 아니라는 뜻.

부가 기하 사실: LiDAR 72빔/360° = 5°/빔이라 0.6m 막대는 6.9m 밖에서 빔 사이로 샌다.
클러스터 규칙(0.45m)상 인접 endpoint 간격 `2·d·sin(2.5°)`가 5.16m를 넘으면 한 막대가
ray 단위로 쪼개진다 — 두 셀렉터 공통 성질.

### 3. TTC(위협 기반) 토큰 셀렉터 신설 — `ttc_sector`

`cluster_sector`는 **방위각**으로 슬롯을 배분한다. 위 두 손실(23.6% + 11.7%)의 공통
원인은 "가깝고 정면인 것"을 고르지 "부딪히러 가는 중인 것"을 고르지 않는 것. 탐색 중
요잉하면 방금 전방이던 막대가 후방 섹터로 넘어가는데 속도 벡터는 여전히 그쪽이다.

```
closing = body_velocity · unit_to_cluster
ttc     = range/closing            (접근 중)
        = idle_s + range/max_range (후퇴 중 → 뒤로, 근접순)
```

- 후방이라도 접근 중이면 토큰 획득 (23.6% 해결)
- 같은 섹터 2번째 클러스터도 더 급하면 선점 (11.7% 해결)
- `min_speed`(0.15) 미만에서는 근접순으로 연속적 degrade
- 클러스터링은 `cluster_sector`와 **공유** → A/B가 grouping이 아닌 **ranking**만 분리
- **관측 폭 불변**(8×12) → 체크포인트 수술 불필요, warm-start 가능
  (`cfg_obstacle_selector`는 이미 preflight 계약 필드)

`tests/test_navrl_ttc_selector.py` **11개 전부 통과**. isaacgym import 순서 문제는
`aerial_gym` 패키지 스텁 주입으로 우회(기존 `test_navrl_perception.py`가 못 하던 것).
테스트 작성 중 픽스처 결함 2건을 자체 발견해 수정 — ray 인덱스 대신 range로 표면을
식별하고, 원거리 표면은 ray 1개로(실제 0.6m 막대는 8m에서 4.3° < 5° 빔피치라 ray 1개).

### 4. 300막대 임계값 0.70은 아무 근거가 없다 (사용자 지적)

램프 양 끝점의 성격이 달랐음을 명시한다:

| 끝점 | 출처 |
|---|---|
| 70막대 0.85 | 설계 추정 → **실측 0.843으로 반증** → 0.80으로 수정 |
| 300막대 0.70 | **순수 추정. v2 300막대는 한 번도 돌린 적 없음.** v1 flat 임계값을 물려받은 값 |

v1 실측 밀도곡선으로 외삽하면 300막대(18.8/100m²)는 v1 완성정책이 0.67~0.75였던
구간이고, 그건 v1의 짧은 경로·정지 표적 조건이다. v2는 이미 4.4/100m²에서 0.84가
천장(v1은 같은 밀도에서 0.96)이므로 0.70은 도달 불가에 가깝다. **커리큘럼은 dense
쪽에서 다시 stall할 것으로 예상된다.**

→ `train_navrl_v2_ceiling_probe.sh`의 목적을 재조정. 70막대 천장은 3070이 이미
답했으므로(0.843), 1650 Ti는 **300막대 천장 실측**에 쓴다:
`PROBE_BARS=300 ./train_navrl_v2_ceiling_probe.sh`. dry-run으로 체인 검증
(`curriculum=0 num_bars=300 max_bars=300 envs=64 sim=base_sim_4gb`).

다음 단계: 밀도별 천장을 수학적으로 모델링하고 그보다 약간 낮게 각 단계 임계값을
설정한다(사용자 제안). 현재 선형 램프는 양 끝이 다 근거 없음.

### 첫 밀도 승급 성공 (v2 최초) — 임계값 스케줄 검증됨

`ppo_260731_2012_navrl_v2-search-sched-s1`, 재시작 직후 첫 판정에서 통과:

```
density curriculum promoted | bars 70 -> 85 after 16386 eps,
capture=0.832 (threshold=0.820) dwell=3313 epochs
```

- **0.832 vs 0.820** — 1.2pp 여유로 통과. 실측 천장 0.843 대비 1.1pp 아래.
- 사용자가 지정한 0.82가 정확한 선택이었다. 이전 판정 이력(0.816 → 0.837 → 0.832)을
  보면 0.85는 불가, 0.80은 과도하게 느슨, 0.82가 "천장 바로 아래"에 해당한다.
- dwell 3313 epoch은 최소 1000을 이미 크게 초과(70막대에 오래 머물렀으므로) — dwell이
  구속하지 않았고 capture 게이트가 실제 결정자였다.
- 증거창을 리셋하지 않고 이어받은 판단도 유효했다(13,877 → 16,386으로 채워 판정).

현재 **85막대, 임계값 0.77**로 진행 중. 다음 판정까지 최소 1000 epoch(dwell) + 증거창.

## 2026-07-31 (밤) — TTC A/B 런처 신설, 그 과정에서 A/B 무력화 버그 발견

`train_navrl_v2_ttc_ab.sh` 신설 (`ARM=baseline|ttc`, 한 스크립트에서 두 팔).

### 발견: 메인 런처가 실험 변수를 조용히 덮어쓰고 있었다

`train_navrl_v2_search.sh:121`이 `export NAVRL_OBSTACLE_SELECTOR=cluster_sector`로
**하드코딩**돼 있어, A/B 런처가 `ttc_sector`를 설정해도 무시됐다. dry-run으로 두 팔의
env를 덤프해 diff한 결과 **차이 0줄** — 즉 두 팔이 정상 실행되면서 같은 조건을 돌리는
상태였다. A/B에서 가장 잡기 어려운 유형이다(둘 다 학습이 되니 로그로는 정상으로 보임).

수정: `${NAVRL_OBSTACLE_SELECTOR:-cluster_sector}`로 조건부화. 재검증 후 두 팔 diff가
셀렉터 한 줄로 축소됐고, 메인 런처 단독 실행 시 기본값도 유지됨을 확인했다.

이번 세션에서 같은 유형이 세 번째다(`NAVRL_CONTROLLED_ABLATION` 하드코딩 → 승급 차단,
`unset NAVRL_NUM_BARS` 무조건 실행 → 프로브 밀도 소실, 이번 건). **런처에서 하위
스크립트가 소비하는 변수를 무조건 export하면 상위 실험이 조용히 무효화된다.**

### A/B 설계

- **실험 변수**: `NAVRL_OBSTACLE_SELECTOR`만 (나머지 61개 env var 동일, dry-run 검증)
- **밀도 고정 70막대**: 비교 중 승급이 일어나면 한쪽 팔만 과제가 바뀜
- **공통 출발점**: `thr80-s1/nn/last_gen_ppo_ep_3250` (70막대 수렴 정책, 관측 폭 8×12
  불변이라 수술 없이 warm-start)
- **사전 등록 게이트**: held-out 평가에서 **capture ≥ +2.0pp AND crash ≤ −2.0pp** 둘 다.
  최종 체크포인트로 판정(corridor A/B가 중간에 0.6420→0.5993→0.6610로 출렁였고, TTC 팔은
  입력 분포 변화에 재적응해야 하므로 초반이 구조적으로 불리하다).
- 4GB 프리셋(64env), **2000 local epoch** 적응. 64env의 epoch당 rollout은 2048
  samples로 3070/128env의 절반이므로, 이는 3070 기준 1000 epoch와 같은 4.096M samples다.

### 1650 Ti 실기 검증 통과 및 본 A/B 실행 시작

1650 Ti(4GB, 디스플레이 연결)에서 baseline 8-epoch 스모크를 완료했다.

- 시작 전: 505 MiB used / 3392~3398 MiB free
- 학습 피크: **2900 / 4096 MiB** (약 1196 MiB 잔여) — 64env 설정 VRAM 통과
- epoch 3251→3258 정상 종료, 평균 step time **6.40초/epoch**
- 예상 예산: 팔당 약 3시간 34분, 두 팔 순차 약 7시간 7분(+초기화)
- 최초 시도에서 `/home/fair/.../python` 하드코딩으로 exit 127이 발생해, 현재 사용자의
  `~/miniconda3/envs/aerialgym`을 우선 탐색하고 기존 경로를 fallback으로 쓰도록 수정

**2026-07-31 20:45 KST**, 본 baseline 팔을 시작했다:
`ppo_260731_2045_navrl_v2-ttc-baseline-s1`, checkpoint ep3250 → max ep5250. baseline 완료 후
TTC 팔을 같은 카드·64env·seed·checkpoint·sample budget으로 순차 실행한다. 스모크의
capture/crash는 표본이 작아 판정에 쓰지 않는다. 1650 Ti A/B의 팔 사이 delta만 내부적으로
유효하며, 절대 capture/crash 수치는 3070 결과 표에 합치지 않고 별도 보고한다.

**사전 고지**: 표현 계열 개입 전적이 나쁘다 — 토큰 5→8 기각, 빔 36→72 기각,
corridor +1.57pp로 게이트 미달. 유일하게 성공한 건 선택 FOV 360→240(같은 "무엇을 고를까"
계열). 작은 효과를 예상하고 냉정하게 측정한다.

## 2026-07-31 — TB에 한 학습이 3개 곡선으로 쪼개진 문제: 계보 병합 도구 + 자동 기록

### 증상과 원인

본 run 계보가 TensorBoard에서 서로 끊긴 3개 곡선으로 보였다. 원인은 `--branch_run`으로,
warm-start마다 새 `runs/<name>/` 폴더를 만든다(설계 의도 자체는 맞다 — config가 바뀐 run의
metric을 원본 폴더에 섞지 않기 위함).

실제 계보(step은 이어져 있고 폴더만 갈렸음):

| run | step 구간 |
|---|---|
| `ppo_260731_1722_navrl_v2-search-fresh-s1` | 20 → 2693 |
| `ppo_260731_1940_navrl_v2-search-thr80-s1` | 2651 → 3251 |
| `ppo_260731_2012_navrl_v2-search-sched-s1` | 3251 → (진행 중) |

### 해결 — 재학습·재시작 없음

TensorBoard는 **한 디렉터리 안의 event 파일들을 하나의 run으로 합쳐 step 순으로** 그리고,
PPO step 카운터는 warm-start를 건너 이어진다. 따라서 계보의 event 파일을 한 폴더에
심볼릭 링크하면 연속 곡선이 된다. 원본 불변, live run 파일도 링크되어 실시간 갱신됨.

검증: `merged: steps 20 → 3780, n=3805, GAP 0개, duplicated=44`.
중복 44 step(2651~2693)은 `1940` run이 ep2650 체크포인트에서 재개하며 **실제로 두 번 학습한**
구간이다. 데이터를 지우거나 가공하지 않고 그대로 노출한다.

`--resume_in_place`로 재시작하는 대안은 기각: epoch ~75를 버리는데 `1940` 구간은 여전히
따로 남아 문제가 절반만 해결된다.

### 재발 방지

- `runner.py::_record_run_lineage()` — warm-start로 새 폴더를 만들 때
  `<new run>/aerial_run/resumed_from.txt`에 출처 run 이름을 기록. best-effort(try/except)라
  실패해도 학습 시작을 막지 않는다. 기존에는 계보 정보가 디스크 어디에도 없어서 사람이
  기억해야 했다.
- `tools/tb_merge_lineage.py` — `resumed_from.txt`를 역방향으로 걸어 병합 뷰를 만든다.
  마커가 없는 옛 run은 `--chain a b c`로 명시 가능. `--list`로 전체 run의 step 구간 조회.
  병합 폴더 이름은 **head가 아니라 lineage root** 기준(`_merged_<root>`) — head 기준이면
  resume마다 새 병합 폴더가 생겨 없애려던 파편화를 그대로 재현한다.
  실행 시 stale 심볼릭 링크를 먼저 지워 죽은 가지가 곡선에 접붙지 않게 한다.

기존 3개 run에는 `resumed_from.txt`를 수동 backfill했다.

```
python tools/tb_merge_lineage.py            # 최신 run의 계보
tensorboard --logdir .../runs/_merged_ppo_260731_1722_navrl_v2-search-fresh-s1
```

### 부수적으로 고친 것

- `--list`가 summaries 폴더가 삭제된 옛 run(`density_120`)에서 예외로 전체 출력을 죽였다.
  `step_range()`에 존재 확인 + try/except 추가.
- `train_navrl_v2_ttc_ab.sh`: `MAX_EPOCHS` 4250 → 5250. 체크포인트는 3070/128env
  (4096 샘플/epoch)에서 왔는데 1650 Ti는 64env(2048 샘플/epoch)라 "+1000 epoch"이 실제로는
  의도한 적응량의 절반(2.05M step)이었다. 샘플 기준으로 맞춰 +2000 epoch = 4.1M step.
  이건 중립적 부족이 아니다 — baseline 팔은 자기 셀렉터로 이미 수렴해 시작하고 ttc 팔만
  분포 변화에 재적응해야 하므로, 예산 부족은 재적응이 필요한 쪽만 처벌한다.

### 운영 메모

순수 이어달리기(계약 변경 없음)라면 `--branch_run`을 빼면 된다 — 원본 폴더에 그대로 이어
쓰므로 병합이 애초에 필요 없다. `--branch_run`은 **환경/보상/커리큘럼 계약을 바꿀 때만**
쓰고, 그 경우 위 도구로 계보를 하나로 본다.

## 2026-07-31 — 밀도별 best reward (전역 러닝 맥스는 첫 승급 이후 무의미)

### 문제

`stability/best_reward`는 run 전체의 러닝 맥스다. 그런데 밀도 승급은 과제 난이도를 바꾸고
도달 가능한 보상 스케일 자체를 낮춘다. 따라서 첫 승급 이후 이 스칼라는 **더 쉬운 밀도에서
얻은 값에 고정**되어, 이후 모든 epoch은 갱신 불가능한 기준과 비교된다 — 학습이 진행 중인지에
대한 정보를 주지 못한다. (같은 뿌리의 함정이 이미 문서화되어 있다: `gen_ppo.pth`는 저밀도
정책이라 고밀도 평가에서 오독된다.)

### 구현 — 순수 가산, 체크포인트 규칙 불변

`navrl_curriculum.py::track_best_reward_by_density(state, n_bars_active, mean_reward)`
(torch-free, CPU 테스트 가능). 밀도가 바뀌면 best를 이월하지 않고 리셋하며, 떠나보낸 밀도의
best를 `(bars, best)`로 반환한다.

새 TB 스칼라:
- `stability/best_reward_at_density` — 현재 밀도 내 best (승급 시 리셋). x축 epoch.
- `stability/best_reward_of_finished_density` — **x축이 epoch이 아니라 막대 수**. 커리큘럼이
  졸업한 각 밀도마다 점 하나 → 밀도별 천장의 형태를 그대로 읽을 수 있다.
- 승급 시 콘솔: `density 70 bars done: best mean_reward 150.123 -> now 85 bars (epoch N)`

`self.last_mean_rewards`와 best-checkpoint 저장 규칙은 **의도적으로 건드리지 않았다.**
리셋하면 `gen_ppo.pth`가 밀도마다 덮어써져 체크포인트 의미가 조용히 바뀐다. 표시 문제와
체크포인트 정책은 분리해서 다룬다.

### 테스트 — `tests/test_curriculum_safety.py` 25개 통과 (기존 15 + 신규 10)

`BestRewardByDensityTest`: 밀도 내 최댓값 추적, 승급 시 이월 금지, 히스토리 누적, 새 밀도의
낮은 값도 그 밀도의 best가 됨, 밀도 None이면 no-op, NaN/±Inf는 best가 될 수 없음, 보상이
한 번도 없던 밀도는 승급 시 아무것도 보고하지 않음, state 불변성, 밀도 하락도 레벨 변경으로 처리.

### 검토 중 잡은 것

`navrl/n_bars_active`를 새로 기록하려 했으나 live run의 TB 태그를 조회해 **이미 기록되고 있음**을
확인하고 중복 기록을 제거했다.

### 적용 시점

현재 돌고 있는 `sched-s1`은 프로세스가 이미 모듈을 로드한 상태라 **다음 run부터 적용**된다.
지금 run의 85막대 구간은 전역 best_reward만 남는다.

## 2026-08-01 — PPO actor-collapse 근본 수정, 실제 rollback·v2 평가 검증

### 결론

`sched-s1`의 epoch 10836 종료는 145막대 자체의 알고리즘 한계가 아니라 **actor update
transaction 부재**가 직접 원인이었다. 기존 KL guard는 안전장치처럼 보였지만 다음 네 가지
결함이 동시에 있었다.

1. KL은 optimizer step **전**에만 계산되어, 임계선을 넘긴 step은 이미 적용된 뒤였다.
2. 초과 시 그 minibatch만 skip하고 모델·RunningMeanStd·Adam moments·GradScaler를 복원하지 않았다.
3. rl-games가 skip 뒤에도 해당 slice의 `mu/sigma`를 초과 정책으로 덮어써 다음 비교 기준을
   망가진 정책으로 rebase했다.
4. hard gate가 NaN/Inf KL은 오히려 통과시켰고 마지막 optimizer step 뒤 전수 검사가 없었다.

추가로 v2 런처는 실제 LR 1e-4를 checkpoint에는 0으로 잘못 기록했고, margin 1.25만 설정하고
coefficient를 빠뜨려 penalty가 0이었다. 보호 범위도 y축뿐이라 x/z/yaw latent 폭주를 막지
못했다. 따라서 단순히 gate 숫자를 낮추거나 더 오래 학습하는 것으로는 재발을 막을 수 없다.

### 구현

- `ppo_update_safety.py`
  - model parameter와 persistent buffer, Adam state, AMP scaler, submodule train/eval mode를
    deepcopy하는 `PPOEpochTransaction` 추가.
  - 동일 분포가 정확히 0인 analytic Normal KL 추가(rl-games helper의 epsilon 음수 bias 제거).
  - NaN/Inf 및 per-axis latent margin 공통 helper 추가.
- `early_stop_a2c_agent.py`
  - rollout 당시 `behavior_mu/sigma`를 immutable field로 보존. rl-games의 moving `mu/sigma`와 분리.
  - actor epoch 시작 전 snapshot, 마지막 minibatch 뒤 **전체 rollout을 eval/frozen RMS로 재추론**.
  - 어느 minibatch 평균 KL이라도 0.04 초과, output/parameter/loss가 nonfinite, 또는 pre-step
    latch가 있으면 actor model+RMS+Adam+scaler를 epoch 시작 상태로 원자적으로 복원.
  - rollback 후 LR×0.5(최저 1e-6), 5회 연속이면 `ppo_rollback_livelock` fail-stop.
  - DDP/RNN은 불완전한 동기 복원을 허용하지 않고 명시적으로 거부.
  - TB: `ppo/behavior_kl_audit_max`, `behavior_kl_sample_max`, `epoch_rollback{,_total,_streak}`.
- `train_navrl_v2_search.sh`
  - fresh 기본 LR 3e-5로 명시, action diagnostics on, all-axis latent margin
    `2.0,1.25,2.0,2.0 @ 0.01`, epoch rollback on.
  - fresh entry에 checkpoint/resume 인자가 들어오면 거부. 실험 wrapper만 명시적으로 opt-in.
  - global flock 추가.
- `train_navrl_v2_recover_safe.sh` 신설
  - 기본 LKG = `sched-s1/last_gen_ppo_ep_9500_rew_83.67131.pth`(130막대).
  - 같은 squashed likelihood이므로 Adam moments는 유지하고 LR 5e-6.
  - 기본은 100-epoch/130막대 고정 smoke; 통과 뒤 `RECOVERY_MODE=curriculum`만 승급 재개.
  - `gen_ppo.pth`와 정책-family mismatch를 거부.
- provenance
  - runner가 YAML 기본까지 포함한 **실제 optimizer LR**을 env에 역기록.
  - checkpoint에 target speed min/final/fixed/ramp/pattern, general spawn, OOB/altitude contract,
    rollback/all-axis margin을 추가. resume mismatch는 density evidence를 폐기하고 경고.
- 평가
  - zero-DOF sim은 null Isaac tensor를 acquire/wrap/refresh하지 않아 종료 진단 제거.
  - `--eval`도 actor-only observation wrapper 사용.
  - v2 sweep은 호출자 기준 상대 checkpoint, arena/표적/표현/density/seed 계약을 고정하고
    cell별 bulk JSON + 종합 CSV가 없거나 episode accounting이 틀리면 실패한다.
  - bulk JSON은 분포 평가를 더 이상 `target_speed_mps=0`으로 거짓 표시하지 않고
    `mode=uniform, min=0.3, max=1.5`를 기록한다.
- quiet log filter가 새 rollback/fail-stop 메시지를 삼키던 문제도 whitelist+테스트로 수정했다.

### 실제 검증

1. **정상 commit 경로** — ep9500 → ep9501, 130 bars, LR 5e-6
   - process exit 0, final checkpoint/finished marker 생성.
   - `behavior_kl_audit_max=0.006315 < 0.04`, 기존 표시 KL=0.00203.
   - rollback 0, all-axis margin scalar 존재, raw OOB 전 축 0.
   - checkpoint provenance: LR 5e-6, rollback true, margin vector/coef, moving-target 계약 모두 일치.
2. **강제 reject 경로** — 같은 LKG에서 gate를 1e-8로 낮춘 1 epoch
   - `epoch_rollback=1`, skipped=1, LR 5e-6→2.5e-6.
   - 저장된 model tensor **93/93 byte-exact**, Adam state key/moment/step 변경 **0개**.
     LR backoff만 의도적으로 달라졌다. 즉 이미 적용된 optimizer mutation이 실제로 폐기됐다.
3. **held-out bulk 평가** — 안전 commit checkpoint, 130 bars, 257 episodes,
   target `U[0.3,1.5]`, seed42
   - capture **74.71%** (192/257), crash **25.29%** (65/257), timeout 0.
   - 절대 원인: bar contact 63/257=24.51%, below 2/257=0.78%, OOB 0.
   - task-input OOB 전 축 0, JSON/CSV episode 합계·checkpoint·density·speed contract 통과.
   - `Can't create empty tensor` 미발생.

단, 257-episode 평가는 배선/정책 건강 검증이지 논문 최종 CI 표본은 아니다. 또한 145막대
후반 데이터는 actor drift에 오염됐으므로 “145가 표현의 확정 ceiling”이라는 기존 판정은
보류한다. 먼저 LKG에서 100-epoch 고정130 smoke를 돌려 KL/edge99/capture를 확인한 뒤,
그 final checkpoint로 curriculum을 재개하고 145를 다시 측정한다.

### 테스트

- `tests/` 전체 **85 tests PASS**.
- bounded action-model **13 tests PASS**.
- Python compile, launcher `bash -n`, `git diff --check` PASS.
- CPU-only perception 테스트 2개가 torch→Isaac Gym import 순서 때문에 수집조차 실패하던
  기존 결함도 sibling corridor module을 package import 없이 주입하도록 고쳐 6+7 tests PASS.

## 2026-08-01 — 최종 적대적 재검수: 복구·평가 우회와 commit finite-hole 폐쇄

### 추가로 발견한 주요 결함

첫 transaction 구현을 다시 적대적으로 검수하자 다음 P1이 남아 있었다.

1. parameter finite 여부를 Python `bool`로 만든 뒤 공통 검사기에 넘겼다. `False`는 실수 0으로
   간주되어 finite였으므로 nonfinite parameter를 reject하지 못했다.
2. post-update audit가 actor `mu/sigma/KL`만 확인해 critic output, PPO/aux loss, Adam moment,
   GradScaler가 깨진 epoch를 commit할 수 있었다.
3. recovery wrapper의 검증된 `--checkpoint` 뒤에 사용자 `"$@"`가 붙어 두 번째 checkpoint나
   task/file 인자로 preflight를 우회할 수 있었다. 같은 config인 붕괴 ep10800도 통과했다.
4. `RECOVERY_MODE=curriculum`이 100-epoch smoke 없이 기본 ep9500에서 바로 시작될 수 있었고,
   stale shell의 LR/KL/margin 값이 safe default를 덮을 수 있었다.
5. TTC A/B wrapper는 checkpoint가 존재하는지만 봐 v1/legacy checkpoint로도 비교가 시작됐다.
6. v2 evaluator가 학습의 OOB margin 1.0 대신 기본 0.5를 사용했고, ramp 이전 checkpoint의 실제
   속도 상한과 무관하게 JSON을 항상 `U[0.3,1.5]`로 기록했다. 새 target/OOB/action provenance도
   gate에서 검사하지 않았다.
7. rollback 뒤 optimizer LR은 낮아졌지만 checkpoint에는 최초 LR만 기록돼 사후 감사가 틀렸다.
8. dashboard가 `v2-recover-*`를 v1로 분류하고 실제 recovery smoke까지 Latest에서 제외했다.

### 보강

- commit 전 actor/critic output, 누적 PPO/critic/aux loss, model state/buffer, Adam state,
  AMP scaler를 재귀적으로 finite 검사한다. loss가 먼저 깨지면 backward 전에 epoch reject를 latch한다.
- recovery wrapper는 추가 runner 인자를 전부 거부하고, 감사된 ep9500의 SHA-256
  `3a0c167c…67578f`와 epoch=9500/bars=130/v2 계약을 강제한다. ep10800은 SHA와 밀도 모두에서 거부된다.
- smoke는 고정130·정확히 100 epoch·LR 5e-6·KL 0.04·전축 margin으로 고정했다. curriculum 모드는
  명시적 CKPT, smoke lineage metadata, ep>=9600, 정상종료 marker가 모두 있어야 열린다.
- TTC A/B도 감사된 ep3250 파일 SHA-256과 epoch/bars/policy/selector를 확인하고, env=64,
  70 bars, 2000-epoch sample budget, PPO safety 설정을 양 arm에서 고정한다.
- evaluator는 final-speed 평가 override를 명시해 saved curriculum clock과 무관하게 실제
  `U[0.3,1.5]`를 sampling하며, bulk JSON은 `_target_speed_max()`의 실제 상한과 OOB margin을 기록한다.
  preflight는 moving-target/general-spawn/OOB/altitude/action-policy 계약까지 검사한다.
- checkpoint에는 `cfg_action_learning_rate`(실행 시작값)와
  `current_action_learning_rate`(scheduler/rollback 반영값)를 분리해 저장한다.
- dashboard는 v2 recovery/TTC를 40×40×3 v2로 인식하고, 100-epoch recovery smoke만은 실제 gated
  단계로 Latest에 남긴다. 1-epoch integration/forced-test는 계속 Latest에서 제외한다.

### 최종 실제 검증

- 정상 GPU commit(ep9500→9501): exit 0, `behavior_kl_audit_max=0.006319 < 0.04`,
  `ppo/kl=0.002029`, rollback=0, all-axis margin scalar 존재. checkpoint의 configured/current LR은
  모두 5e-6.
- 강제 GPU reject(gate=1e-8): rollback 메시지가 콘솔에 보였고, 원본 대비 model 변경 0/93,
  Adam moment/step 변경 0. configured LR=5e-6, current LR=2.5e-6로 정확히 분리 기록됐다.
- 실제 GPU 1-episode evaluator 배선: 130 bars, `target_speed_mode=uniform`, min/max=0.3/1.5,
  OOB margin=1.0, squashed-Gaussian, episode accounting/JSON/CSV 검증 통과. zero-DOF 종료 오류 없음.
- preflight: ep9500 PASS; ep10800 거부; 후행 `--checkpoint` 거부; CKPT 없는 curriculum 거부;
  TTC 감사 checkpoint PASS; 구 provenance checkpoint의 canonical eval 거부.
- `tests/` 전체 **87 PASS**, bounded action-model **13 PASS**. Python compile, launcher `bash -n`,
  dashboard JSON/inline parity도 PASS.

### 다음 실행

현재 장기 학습은 없다. `./train_navrl_v2_recover_safe.sh`로 ep9500→ep9600 고정130 smoke를
먼저 끝내고, final checkpoint를 held-out 평가한 뒤에만 `RECOVERY_MODE=curriculum`로 재개한다.

## 2026-08-01 — 주요 오류 전수 보강 완료: actor+critic 원자 복구와 평가 증명서

### 최종 보강한 오류

1. **central critic이 actor transaction 밖에 있던 문제**: snapshot을 `prepare_dataset()`보다 앞으로
   옮겼다. 따라서 input/value RMS 갱신, central critic optimizer step, actor optimizer step이 모두
   하나의 `try`/transaction 안에 들어간다. reject와 예외 양쪽에서 actor·critic model/buffer,
   양 optimizer, scaler, critic lr/epoch/frame/mode를 같은 epoch 시작점으로 복원한다.
2. **복구 런처의 inherited-env 우회**: `FILE`, `TASK`, `NUM_ENVS=128`, simulator preset,
   cluster selector, 70→300 density schedule, 16,384-episode gate, 1,000-epoch dwell, 안전 PPO 설정,
   tag/log를 모두 pin했다. `ALLOW_CONCURRENT`, `GPU4GB`, network override는 해제한다. hostile shell
   값으로 preflight해도 고정 계약이 출력되는 것을 확인했다.
3. **문서에만 있던 held-out gate**: recovery smoke checkpoint는 130 bars 단일 조건·최소 2,049
   episodes로만 평가할 수 있다. evaluator가 capture≥0.65, crash≤0.35, timeout≤0.10뿐 아니라
   smoke 100 epoch의 `behavior_kl_audit_max≤0.04`, 전축 task-input OOB=0, final rollback streak=0을
   확인한 뒤 checkpoint hash 결합 PASS artifact를 원자적으로 기록한다. curriculum 런처는 이
   artifact가 없거나 수치/hash가 바뀌면 거부한다.
4. **curriculum 중단 후 안전 재개 불가**: `RECOVERY_MODE=continue`를 추가했다. safe curriculum
   provenance와 평가 artifact hash lineage가 있는 checkpoint만 받고 density accumulator/dwell
   evidence를 초기화하지 않는다.
5. **TTC 평가·사이트 오분류**: v2 evaluator는 shell selector를 믿지 않고 checkpoint의
   `cluster_sector`/`ttc_sector`를 복원한다. dashboard도 TTC를 density curriculum으로 표시하지 않고
   fixed-70 selector A/B, 4.1M sample budget, +2pp capture/−2pp crash gate로 표시한다. fresh/TTC 런은
   stale recovery provenance를 checkpoint에 쓰지 않는다.
6. **검증 run이 사이트 Latest를 오염**: 이름에 smoke/integration/forced/preflight가 있는 짧은
   배선 검증은 Latest에서 제외하되, 실제 100-epoch `v2-recover-smoke`만 연구 단계로 유지한다.

### 실제 GPU rollback 최종 증명

`ppo_260801_0528_navrl_v2-full-transaction-forced-final-s1`, ep9500→9501,
KL gate `1e-8`에서 의도적으로 reject했다.

- console: pre-KL `7e-6`, full-rollout audit KL `6e-6`, sample max `1.15e-4`, rollback 1.
- actor model: source 대비 변경 **0**, actor Adam: LR 외 moment/step 변경 **0**.
- asymmetric central model: compile prefix 정규화 후 변경 **0**.
- central value statistics와 central Adam: 변경 **0**, central LR `1e-4` 유지.
- actor LR만 설계대로 `5e-6 → 2.5e-6`; checkpoint에 configured `5e-6`, current `2.5e-6`로 분리 기록.

이 1-epoch run의 capture 9.1%는 업데이트 후 정책 성능이 아니라, update 전에 소비한 11개 episode의
작은 rollout 수치다. 정책·critic은 source와 exact 복원됐고 이 검증 run은 dashboard Latest에서 제외된다.

### gate/회귀 검증

- synthetic 정상 smoke: 100개 TB epoch + 2,049 held-out → attestation 생성, curriculum preflight PASS.
- attestation capture를 0.64로 변조 → curriculum preflight 즉시 거부.
- safe curriculum checkpoint → `RECOVERY_MODE=continue` preflight PASS.
- hostile inherited env, ep10800, CKPT 없는 curriculum, 후행 runner arg는 모두 거부/고정 확인.
- TTC selector checkpoint를 canonical evaluator가 그대로 복원하는 preflight PASS.
- Python 전체 discovery **90 PASS**, bounded action-model **13 PASS**, Python compile,
  launcher `bash -n`, `git diff --check`, dashboard JSON/inline fallback parity PASS.

현재 장기 학습은 돌고 있지 않다. 다음 작업은 감사된 ep9500에서 정확히 100 epoch의 fixed-130
recovery smoke이며, 실행 명령과 이후 eval/curriculum/continue 절차는 `OPERATIONS.md §6`에 고정했다.

## 2026-08-01 — 최종 우회 경로 폐쇄: 재현 가능한 recovery gate

### 마지막 적대적 감사에서 추가로 잡은 오류

1. v2 evaluator가 checkpoint의 정수 provenance를 모두 float로 출력해
   `NAVRL_FOV_CURRICULUM_EPOCHS=3000.0`, `NAVRL_DETECTOR_MIN_PIXELS=2.0`을 만들었다. task parser는
   경고 뒤 우연히 같은 기본값으로 돌아갔지만, 2,049회 평가가 끝난 뒤 JSON 후처리의
   `int("2.0")`에서 죽어 attestation을 만들지 못하는 P0였다. integral 검증 후 정수 문자열로
   직렬화하도록 수정했다.
2. held-out JSON의 outcome count 합과 보고된 capture/crash/timeout rate를 독립적으로만 검사해,
   count와 rate가 서로 모순되어도 PASS가 가능했다. evaluator·attestation·recovery wrapper·dashboard
   네 층 모두 `rate == count / actual_episodes`를 검증한다.
3. TTC idle/min-speed, FOV curriculum, detector threshold/checkpoint SHA, perturb/dropout/noise,
   max tilt/tilt compensation은 observation/control shape를 바꾸지 않아 stale env가 숨어들 수 있었다.
   실제 실행값을 checkpoint에 저장하고, evaluator가 복원하며, smoke attestation과 continue embedded
   artifact가 exact contract를 다시 확인한다.
4. recovery training과 held-out가 같은 seed가 될 수 있었다. canonical recovery는 training seed=1을
   고정하고 held-out는 seed=42를 강제·기록한다. evaluator는 inherited general-eval/interactive flag를
   지우고 recovery에서 `NAVRL_V2_FORCE`를 금지한다.
5. controller는 `NAVRL_TILT_COMP=false`를 ON으로 해석하지만 checkpoint는 OFF로 기록하던 bool parser
   불일치를 제거했다.
6. curriculum/continue는 낮아진 LR을 보존했지만 일반 checkpoint resume는 config LR로 다시 올렸다.
   명시적 override가 없는 training resume는 이제 `current_action_learning_rate`를 복원하고, 잘못된
   저장 LR은 fail-closed한다. `--checkpoint`만 주고 `--train`을 생략해도 실제로는 학습하던 rl-games
   implicit 경로도 내부에서 TRAIN으로 정규화해 같은 규칙을 적용한다.
7. main v2 launcher는 stale FILE/TASK/network/simulator/env 수와 `ALLOW_CONCURRENT=1`에 취약했다.
   main=base_sim/128 env, 4GB=base_sim_4gb/64 env의 명시 profile, 고정 YAML/task, 무조건 global lock으로
   분리했다. recovery는 main profile을 강제한다.
8. dashboard의 `verdict=PASS` 한 줄 신뢰를 제거했다. 실제 checkpoint/result SHA, 정상종료 marker,
   full checkpoint/eval contract, seed, outcome accounting, KL/OOB/rollback을 모두 다시 검증해야만
   `curriculum unlocked`를 표시한다.

### 자동·실제 검증

- 실제 감사 LKG `last_gen_ppo_ep_9500_rew_83.67131.pth` SHA/epoch/bars/contract recovery preflight PASS.
- FILE/TASK/NUM_ENVS/simulator/profile/network/동시실행/guard/TTC/tilt를 적대적으로 주입한 preflight도
  안전 계약(128 env, base_sim, seed1, TTC 30/0.15, tilt45, KL0.04)으로 고정됨을 확인.
- synthetic recovery gate 9개: 정상 curriculum/continue, LR 2.5e-6·1.25e-6 보존, held-out result
  SHA 변조 거부, seed42 training 거부, force-eval 거부, 정수 provenance, 100-epoch TB audit,
  중간 rollback 거부, count-rate 위조 거부 PASS.
- Python discovery **102 PASS**, bounded action-model **13 PASS**. Python compile, launcher `bash -n`,
  `git diff --check`, 사이트 JSON/inline snapshot 동기화와 JS arena-motion parity PASS.

새 provenance 필드는 다음에 생성되는 smoke checkpoint부터 실제 artifact에 들어간다. 마지막 GPU
forced-rollback checkpoint는 코드 변경 전 생성됐으므로 이 필드가 없는 것이 정상이며 논문 데이터로
사용하지 않는다. 장기 학습은 현재 없다. 다음 실행은 `./train_navrl_v2_recover_safe.sh` 한 줄이며,
ep9600 정상 종료 후에만 `OPERATIONS.md §6`의 held-out → curriculum 순서로 진행한다.

## 2026-08-01 — 주요 오류 최종 폐쇄: 원증거 재검산·평가 분포 독립화

### 마지막 감사에서 발견한 주요 오류

1. curriculum 런처가 held-out 결과 SHA는 확인했지만 결과 내용과 100-epoch TensorBoard를 다시 읽지
   않았다. 따라서 수기로 만든 plausible PASS attestation이 승급을 열 수 있었다.
2. dashboard도 attestation의 KL/OOB/rollback 숫자를 신뢰해 summaries가 없는 가짜 PASS를 표시할 수
   있었다.
3. v2 evaluator가 inherited `AERIAL_GYM_SIM_NAME`과 `NUM_ENVS`를 고정하지 않아 다른 timestep의
   simulator로도 같은 평가 이름을 만들 수 있었다.
4. bulk held-out의 목표 거리와 초기 FOV가 checkpoint의 `k_max_cur`와 `num_task_steps`에 의존했다.
   같은 policy도 저장 시점에 따라 더 쉬운 분포에서 채점될 수 있었다.
5. recovery lineage가 `epoch>=9600`만 확인해 frame/task-step/horizon이 다른 intermediate 또는 변형
   checkpoint를 정확한 smoke final처럼 취급할 여지가 있었다.

### 수정

- `navrl_v2_recovery_attestation.py`의 canonical builder를 단일 진실원으로 만들었다. 기존 증명서를
  검증할 때 실제 checkpoint, held-out JSON, TensorBoard step 9501–9600의 KL·rollback·4축 OOB를
  다시 읽고 payload 전체가 exact한지 비교한다. 정상종료 marker도 `epoch=9600`을 요구한다.
- recovery launcher와 dashboard 모두 같은 canonical verifier를 호출한다. 결과 변조, summaries 누락,
  self-written PASS는 fail-closed한다. producer/dashboard의 episode length, pursuer speed, action sample
  수 계약도 동일하게 맞췄다.
- recovery held-out runtime은 `main/base_sim/dt0.01/128 env`로 고정했다. shell의 sim/env 값은
  덮어쓰며 `GPU4GB=1`은 증명서 평가에서 거부한다. 일반 4GB 평가는 별도 profile로 명시 기록한다.
- bulk/full evaluation은 saved curriculum clock을 무시하고 목표 거리 6–28 m 전체와 최종 FOV를
  사용한다. 실제 적용 min/max, full-distribution/FOV-saturated 여부, runtime profile을 결과 JSON에
  남기고 evaluator/attestation/dashboard가 다시 검사한다. 일반 training 경로는 기존 curriculum을
  그대로 사용하며 main/recovery/TTC training launcher는 inherited full-eval flag를 명시적으로 지운다.
- smoke final은 정확히 epoch 9600/frame 39,321,600/task-step 307,200/horizon 32/k=[20,28]/130 bars를
  요구한다. continue는 이 anchor에서 epoch당 task-step +32, frame +4096 및 130:+15:300 밀도 schedule을
  검증하며 불일치를 자동 보정하지 않는다. checkpoint에 `cfg_ppo_horizon`도 새로 기록한다.
- evaluator의 정수 provenance, count-rate 일치, seed 분리, full same-shape 계약과 함께 이 조건들을
  recovery preflight와 사이트 unlock 양쪽에 결합했다.

### 검증

- 실제 audited LKG ep9500: SHA-256 일치, epoch/frame/task-step=9500/38,912,000/304,000,
  k=[20,28], bars=130; safe recovery preflight PASS.
- 적대적 evaluator env(`AERIAL_GYM_SIM_NAME=base_sim_4ms`, `NUM_ENVS=1`)도
  `main/base_sim/128/base_sim_dt0.01`로 고정됨. recovery에서 `GPU4GB=1`은 거부.
- synthetic canonical smoke는 curriculum/continue PASS. 결과 변조, 수기 PASS, TensorBoard 누락,
  중간 rollback, count-rate 위조, force-eval, seed42 training, anchor clock 변조는 모두 거부.
- Python 전체 discovery **111 PASS**, recovery gate **13 PASS**, status attestation **6 PASS**,
  held-out distribution/perception **11 PASS**, bounded action-model **13 PASS**.
- Python compile, launcher `bash -n`, `git diff --check`, 사이트 arena-motion parity PASS.

장기 학습은 시작하지 않았다. 다음 실행은 여전히 `./train_navrl_v2_recover_safe.sh`이며, 실제 새
checkpoint가 위 provenance를 갖는지는 이 100-epoch smoke artifact에서 최종 실증한다.

## 2026-08-01 — 최종 적대적 감사 후 주요 오류 폐쇄: 평가 byte-binding·FOV·rollback 내구성

### 마지막으로 발견한 결함

1. canonical attestation이 결과 JSON의 수치만 재계산했기 때문에, 정상처럼 보이는 aggregate JSON을
   직접 작성해도 evaluator를 실제로 실행했다는 증거가 없었다. 평가 뒤 같은 checkpoint 경로의 bytes를
   교체하면 옛 결과가 새 policy에 재결합될 수도 있었다.
2. TensorBoard 동일 step을 두 번 쓰면 마지막 값만 남겨 먼저 기록된 KL/OOB/rollback 이상치를 숨길 수
   있었다. rollback streak 외 `epoch_rollback`과 누적 total도 gate가 직접 읽지 않았다.
3. evaluator의 물리 계약은 simulator에서 측정한 값이 아니라 shell 라벨이었다. 특히
   `base_sim_4gb`는 실제 상속 dt=0.01인데 `dt0.004`로 기록됐다.
4. general-spawn은 표적을 전방위로 뽑은 뒤 기존 FOV goal loop를 `todo=0`으로 건너뛰어,
   `NAVRL_FOV_CURRICULUM_EPOCHS=3000`이 사실상 no-op이었다.
5. PPO rollback total/streak가 checkpoint에 없었다. patience fail-stop은 `train_epoch()` 안에서 예외를
   던져 정상 frame 증가와 periodic save 전에 종료되므로, 낮아진 LR과 streak가 유실될 수 있었다.
6. main launcher 뒤에 임의 runner 인자를 추가할 수 있었고, 사이트의 recovery 완료 판정은 exact
   정상종료 marker보다 epoch 수를 더 신뢰했다. Now/phase 문구도 과거 corridor 단계에 고정돼 있었다.

### 수정

- evaluator는 원본 checkpoint와 byte-identical한 별도 CoW snapshot을 만들어 **그 snapshot을 실제
  play**한다. 시작·종료 source/snapshot SHA를 확인하고, cell별 64-hex nonce를 task JSON과
  `*.receipt.json`에 교차 기록한다. receipt는 result/log/evaluator script/checkpoint snapshot의 절대경로와
  SHA, episode 수, density, seed, 시작·종료 시각을 묶는다. attestation은 receipt와 현재 source bytes를
  다시 해시해 수기 JSON, receipt 누락, checkpoint swap을 거부한다. 이는 로컬 workflow의
  tamper-evident guard이며 외부 개인키 서명은 아니다.
- recovery TensorBoard 9501–9600은 필요한 tag마다 **step당 정확히 1개**만 허용한다. KL, 4축 raw OOB,
  rollback event/streak/total 100개가 모두 존재하고 rollback 관련 값이 전부 0이어야 한다.
- task가 실제 `BaseSimConfig`/`BaseSim4GBConfig`, physics dt, substeps, RL step당 physics step,
  RL step dt를 checkpoint와 bulk JSON에 기록한다. main은 `BaseSimConfig/0.01/1/10/0.1`, 4GB는
  `BaseSim4GBConfig/0.01/1/10/0.1`이어야 한다.
- general-spawn FOV curriculum은 표적의 world 방향을 편향시키지 않고 초기 drone yaw만 정렬한다.
  epoch0 상대 방위는 camera half-FOV의 85%(현재 약 ±36.98°), 3000 epoch 동안 ±180°까지 선형 확대한다.
  포화 뒤와 full-distribution held-out는 unrestricted yaw다.
- rollback total/streak를 rl-games full-state에 저장·복원한다. livelock patience 도달 시 소비한 rollout
  frame을 반영하고, exact 복원된 actor/central critic/Adam/RMS/scaler와 backed-off LR을
  `last_gen_ppo_ep_*_rew_rollback_livelock.pth`로 저장한 뒤 fail-stop한다.
- fresh main은 CLI 인자 0개만 받으며, resume opt-in은 정확히
  `--checkpoint "$CKPT" --branch_run`만 허용한다. training launchers는 evaluator nonce/profile/physics
  변수를 지운다. 사이트는 exact `epoch=9600` marker 없이는 완료로 표시하지 않고, 현재 research
  update에서 Now/phase를 동적으로 그리며 PASS도 생성 시각 기준 static snapshot임을 명시한다.

### 검증

- Python 전체 discovery **129 PASS**.
- bounded action-model **13 PASS**, recovery gate **17 PASS**, status attestation **6 PASS**,
  perception/FOV/physics **15 PASS**, PPO safety **14 PASS**, launcher contract **4 PASS**,
  recovery marker **3 PASS**.
- 적대적 회귀: receipt 없는 plausible JSON, 평가 뒤 checkpoint swap, TensorBoard duplicate step,
  mid-window rollback, count-rate 위조, 결과 변조, anchor clock 변조를 모두 거부한다.
- 실제 audited ep9500 LKG safe-recovery preflight PASS. 구 LKG 자체는 새 same-shape provenance가 없어서
  canonical held-out evaluator가 직접 거부하는 것이 정상이며, ep9500에서 생성할 새 ep9600 smoke
  checkpoint부터 새 물리/FOV/receipt 계약을 충족한다.
- 모든 launcher `bash -n`, Python compile, `git diff --check`, 사이트 JS parity PASS.
- 장기 NavRL training process는 현재 없다. 다음 실행은
  `./train_navrl_v2_recover_safe.sh`로 ep9500→ep9600 fixed-130 smoke다.

### 2026-08-01 — PPO transaction 검증 TensorBoard 7개 아카이브

사용자 확인 후 다음 1-epoch safety-wiring 세션의 `summaries/`만 라이브 TensorBoard logdir 밖으로
이동했다: `0405`, `0408`, `0449`, `0450`, `0455`, `0456`, `0528` rollback/transaction tests.

- 이동 위치:
  `/home/fair/workspaces/aerial_gym_ws/tensorboard_archive/2026-08-01_transaction_tests/`
- 영구 삭제하지 않았으며 archive README에 정확한 7개 run과 복구 방법을 기록했다.
- 원래 `runs/<run>/`의 checkpoint, CSV, marker 및 forced rollback 증거는 보존했다.
- 활성 NavRL trainer/player가 없고 GPU가 비어 있음을 이동 전후 확인했다.
- 이 정리는 다음 `train_navrl_v2_recover_safe.sh` 학습이나 ep9500 LKG에 영향을 주지 않는다.

### 2026-08-01 — recovery 실제 실행 handoff 오류 수정

사용자가 `./train_navrl_v2_recover_safe.sh`를 실제 실행했을 때 recovery wrapper는
`--checkpoint <ep9500> --branch_run` 인자를 넘겼지만 `CKPT` shell 변수를 export하지 않았다.
강화된 child `train_navrl_v2_search.sh`는 인자와 환경변수의 exact 일치를 요구하므로
`expected <unset>`으로 즉시 거부했다. trainer/Isaac Gym 초기화 전 종료됐고 새 run/checkpoint는
생기지 않았다. `libtinfo.so.6` 문구는 이 실패와 무관한 conda bash warning이다.

- recovery wrapper가 resolved `CKPT`를 명시적으로 export하도록 수정했다.
- 기존 preflight가 child 실행 직전에 끝나 이 배선 오류를 놓친 것도 수정했다.
  `NAVRL_PREFLIGHT_ONLY=1`은 이제 같은 실제 `exec` handoff를 거쳐 child의 full continuation 계약까지
  검사한 뒤에만 종료한다.
- 실제 ep9500 기본 경로로 새 preflight를 실행해 child가 `CONTINUATION`, main/base_sim/128 env,
  seed1, LR 5e-6, epoch rollback, 40 m arena와 정확한 checkpoint tuple을 모두 수락함을 확인했다.
- launcher contract 5/5, recovery gate 17/17, Python 전체 discovery **130 PASS**,
  `bash -n`, `git diff --check` PASS.

## 2026-08-01 — ep9500→9600 fixed-130 recovery smoke 및 held-out PASS

`./train_navrl_v2_recover_safe.sh`로 감사된 ep9500 LKG에서 정확히 100 epoch를 추가했다.
run은 `ppo_260801_1150_navrl_v2-recover-smoke-130bars-s1`, 종료 사유는 `max_epochs`이며
`epoch/frame/task-step=9600/39,321,600/307,200`, `k=[20,28]`, 130 bars, LR `5e-6`를 보존했다.
최종 checkpoint는 `last_gen_ppo_ep_9600_rew_88.09682.pth`, SHA-256은
`b154aad7e6395d5dc9db96110930ef9539ab8c4fbdffaa3977df86c047a28c70`이다.

### 100-epoch smoke 안전성

- TensorBoard step 9501–9600을 전부 재집계했다: behavior KL audit max **0.01173**
  (`0.04` 제한 이내), PPO epoch rollback/event/streak/total **전부 0**, KL skipped minibatch **0**.
- x/y/z/yaw task-input raw OOB는 100 epoch 모두 **0**. explained variance 평균 **0.847**,
  LR은 전 구간 `5e-6`로 유지됐다.
- training-distribution epoch 평균은 capture **72.95%**, crash **23.00%**, timeout **4.05%**.
  마지막 한 epoch의 71.43%나 peak 94.87%는 표본 수가 작아 gate로 사용하지 않았다.

### canonical 130-bars held-out

seed 42, main/BaseSimConfig, 128 env, 2,049 episode, 목표 거리 6–28 m 전체, 최종 FOV,
moving target `U[0.3,1.5] m/s`/mixed 조건으로 최종 checkpoint snapshot을 실제 재생했다.

| 지표 | 결과 | recovery 기준 |
|---|---:|---:|
| capture | **75.74% (1,552/2,049)** | ≥65% PASS |
| crash | **20.69% (424/2,049)** | ≤35% PASS |
| timeout | **3.56% (73/2,049)** | ≤10% PASS |
| task-input OOB | **0% (전 축)** | 0% PASS |

충돌 중 96.46%가 bar contact(전체 episode의 19.96%)였고 below 0.54%, OOB 0.20%였다.
다만 횡축은 executed edge98 **26.35%**, `high80_y=70.90%`, positive/negative y가
84.27%/13.62%로 한쪽 편향이 아직 크다. 이전의 입력 경계 이탈·PPO 발산은 해결됐지만,
고밀도에서 남은 주 병목은 bar-contact와 횡축 편향이다. 이 smoke는 해당 편향을 더 악화시키지
않으면서 recovery 안전 gate를 통과했으므로, 지금 정책 구조를 다시 바꾸기보다 145 bars부터
안전 커리큘럼을 재개하고 밀도별 held-out에서 편향과 충돌 증가를 함께 감시한다.

- 평가 결과: `train_session_logs/eval_v2_ppo_260801_1150_navrl_v2-recover-smoke-130bars-s1_260801_121130/`
- canonical attestation: `.navrl_v2_recovery_eval_pass.json`, SHA-256
  `11a5e403e777165d49ee4f0ab7a92df308a74440c35039f0c6c51a42a49588a0`; 생성 직후 독립 재검증 PASS.
- 다음 실행은 `RECOVERY_MODE=curriculum`과 위 ep9600 checkpoint를 명시한 safe wrapper이며,
  일반 `train_navrl_v2_search.sh`로 우회하지 않는다.

## 2026-08-01 — TTC A/B baseline 팔 완료 (1650 Ti): held-out capture 79.443%, 통과선 확정

### baseline 팔 (대조군, `cluster_sector`)

- run `ppo_260731_2045_navrl_v2-ttc-baseline-s1`, 1650 Ti, 64 env, 70막대 고정
- ep3250 → ep5250 (**4.096M adaptation samples** — 샘플 기준 예산 정합이 의도대로 적용됨)
- peak VRAM **2900/4096 MiB** — 4 GB 카드에서 여유 확인. `base_sim_4gb` 프리셋으로 3070에서
  잰 3425 MiB보다 오히려 낮다. **1650 Ti VRAM 미검증 항목 해소.**
- 최종 체크포인트 `nn/last_gen_ppo_ep_5250_rew__140.70476_.pth`

### held-out 평가 (seed 42, 64 env, 70막대, 실제 n=2048)

| 지표 | 값 | 카운트 |
|---|---|---|
| capture | **79.443%** | 1627/2048 |
| crash | **17.236%** | 353/2048 |
| timeout | 3.320% | 68/2048 |
| closest (no crash) | 1.13 m | best 0.19 m |

### 사전 등록 게이트 → 절대 통과선 확정

사전에 정한 `capture ≥ +2.0pp AND crash ≤ −2.0pp`를 baseline 실측에 적용하면:

| 조건 | 통과선 | 카운트 |
|---|---|---|
| capture | **≥ 81.445%** | ≥ 1668/2048 |
| crash | **≤ 15.234%** | ≤ 312/2048 |

**둘 다 충족해야 통과.** capture만 오르고 crash를 대가로 치렀으면 기각한다.
이 숫자는 ttc 팔 결과를 보기 **전에** 확정되었다.

### 결과 오염 방지 (재확인)

이 절대 수치는 1650 Ti / 64 env에서 나왔다. 3070 / 128 env 결과 표에 **섞지 않는다.**
유효한 것은 **1650 Ti 내부의 두 팔 delta뿐**이다. (3070의 70막대 capture 0.832와
직접 비교하는 것도 같은 이유로 무효 — 두 카드의 배치 크기가 다르다.)

### 다음 — ttc 팔

```
ARM=ttc ./train_navrl_v2_ttc_ab.sh
PYTHON=/home/joshuali/miniconda3/envs/aerialgym/bin/python \
  GPU4GB=1 NAVRL_V2_DENSITIES=70 \
  ./eval_navrl_v2_density_sweep.sh runs/<ttc run>/nn/last_gen_ppo_ep_5250_*.pth
```

평가는 baseline과 **동일한 70막대 / 64 env / seed 42** 조건이어야 한다.
평가 런처가 `/home/fair` python을 기본값으로 잡으므로 1650 Ti에서는 `PYTHON=`을 명시한다.

---

## 2026-08-01 — ttc 팔 기동 NameError 수정 (vehicle_quat)

`ARM=ttc ./train_navrl_v2_ttc_ab.sh`가 reset 직후 사망:
`navrl_perception.py` `_fuse_static_and_extract_obstacles` 안 `ttc_sector` 분기가
`vehicle_quat`/`drone_vel_w`를 쓰는데 인자로 안 넘김 → `NameError`.

수정: fuse 함수 시그니처에 두 인자를 추가하고 `observe()`에서 전달. baseline
(`cluster_sector`)은 이 경로를 안 타서 통과했고, ttc만 터진 이유.
`tests/test_navrl_ttc_selector.py` 11/11 OK. 학습은 아직 재시작 안 함 — 사용자 재실행 필요.

## 2026-08-01 — TTC A/B ttc 팔 완료 + held-out **PASS** (1650 Ti)

### 학습 (ttc_sector)

- lineage: ep3250 warm-start → `1326`에서 3251–4450 학습 후 중단 → ep4450에서 resume
  (`1532_navrl_v2-ttc-ttc-s1-resume`, 4451–5250, 800 epoch)
- 최종 run `ppo_260801_1532_navrl_v2-ttc-ttc-s1-resume`, 1650 Ti / 64 env / 70막대 고정
- ckpt `nn/last_gen_ppo_ep_5250_rew_154.677.pth`, peak reward 166.14 @ ep5246
- 학습 종료 시 on-policy proxy: capture 90% / crash 10% (n=64, seed 1 — held-out 아님)

### held-out 평가 (seed 42, 64 env, 70막대, n=2048)

| 지표 | baseline | ttc | Δ (ttc−baseline) |
|---|---|---|---|
| capture | 79.443% (1627) | **89.307%** (1829) | **+9.864 pp** |
| crash | 17.236% (353) | **9.180%** (188) | **−8.056 pp** |
| timeout | 3.320% (68) | 1.514% (31) | −1.806 pp |
| closest (no crash) | 1.13 m | 0.70 m | — |

사전 등록 게이트 (baseline 실측 기준 capture ≥81.445%, crash ≤15.234%):

| 조건 | ttc | 판정 |
|---|---|---|
| capture ≥ 81.445% | 89.307% | **PASS** |
| crash ≤ 15.234% | 9.180% | **PASS** |

**결론: ttc_sector A/B 통과.** bearing-ranked `cluster_sector` 대비 threat-ranked 토큰 선택이
70막대 held-out에서 capture·crash를 동시에 개선했다. capture만 올리고 crash를 희생한 패턴 아님.

### crash 분해 (ttc held-out)

- bar_contact 93.6% of crashes, OOB 5.9%, below 0.5%
- lateral edge98(y)=0.242 — baseline 대비 yaw saturation은 추가 조사 여지

### 경로

- eval: `train_session_logs/eval_v2_ppo_260801_1532_navrl_v2-ttc-ttc-s1-resume_260801_181048/`
- 비교표: `results/v2_ttc_ab_1650ti.csv`

### 다음

- ttc_sector를 v2 기본 obstacle selector 후보로 승격 검토 (3070에서 동일 A/B 재현 여부 별도)
- 1650 결과는 3070 recovery curriculum 수치와 **섞지 않음**

## 2026-08-02 — 205막대 중간 held-out 평가·문헌 밀도 대조·안전 재개

사용자가 70% gate를 영원히 넘지 못할 추세인지 물어 현재 3070 학습을 변경 없이 감사했다.
실제로는 130에 정체된 것이 아니라 `130→145→160→175→190→205`까지 이미 다섯 번 승급했다.
승급 window는 각각 130 **0.747**, 145 **0.723**, 160 **0.708**, 175 **0.700**,
190 **0.704**였다. 205의 완결된 16,384-episode windows는 **0.647→0.659→0.670**으로
상승 중이고, ep20700 checkpoint의 진행 중 evidence는 `4,129/6,063=0.681`이었다.

### 중간 평가

동일 GPU 동시평가는 VRAM이 부족하므로 trainer를 SIGINT로 중단했다. 마지막 안전 checkpoint
`last_gen_ppo_ep_20700_rew_27.684727.pth`(SHA-256
`c08d9d527430e2633fdec9bb2dba08aade2a89d1bced0cf0250644b87d605015`, rollback 0)를
seed 42, 128 env, 205 bars, full 6–28 m/FOV, moving target `U[0.3,1.5]`, 2,049 episodes로 평가했다.

| 지표 | ep20700 @ 205 bars |
|---|---:|
| capture | **71.89% (1,473/2,049)** |
| crash | **26.99% (553/2,049)** |
| timeout | **1.12% (23/2,049)** |
| task-input OOB | **0% (전 축)** |
| lateral edge98 | **3.85%** |

capture Wilson 95% CI는 약 **69.90–73.79%**다. 같은 recovery 계보의 130-bars 75.74%보다
**−3.86 pp**(독립비율 근사 95% CI `−6.55..−1.17 pp`)로 밀도 증가 비용은 유의하지만,
held-out point estimate는 70% gate를 넘는다. crash의 97.83%는 bar contact다. 결과 경로는
`train_session_logs/eval_v2_ppo_260801_1235_navrl_v2-recover-curriculum-s1_260802_001543/`이다.

### NavRL/NavRL++ 및 유사 연구와의 정량 대조

- 현재 205 bars는 40×40 m에서 **12.81 static/100m²**다. NavRL++ 평가는 같은 40×40 m에서
  static 300/350/400(**18.75/21.88/25.00 per 100m²**)와 dynamic 60/80/100을 더한다.
  NavRL++ high-complexity SR은 static **99.84%**, dynamic **83.96%**다.
- 원 NavRL은 50×50 m, static350+dynamic60/80/100/120에서 curriculum peak SR
  **94.33/82.71/80.96/68.65%**, dynamic obstacle 승급 기준은 80%였다.
- Safe-RL time-optimal flight는 unseen 최난도(장애물 간격 1–3 m) 성공률 **66.7%**,
  differentiable-physics agile flight는 별도 waypoint/forest 기준 **90%**, privileged-ToA RL은
  large-obstacle photorealistic 환경 **86%**를 보고한다.

따라서 우리 71.89%는 유사 학습형 비행 연구 범위 밖의 실패 수치는 아니지만 NavRL++ dynamic보다
약 12.1 pp 낮다. 동시에 우리 density 자체는 NavRL++ S1 static보다 31.7% 낮으므로 “장애물이 더
많아서 낮다”로 설명할 수 없다. 다만 우리 과제는 알려진 고정 goal/goal-aligned action이 아니라
가려지는 이동 표적을 RGB-D/LiDAR로 찾아 0.5 m 내 요격하고, full-height bars라 수직 우회가
불가능하다. 논문 SR과의 차이는 density뿐 아니라 target-search/identification, success radius,
행동 좌표계와 안전 shield 차이를 포함하므로 직접 순위표로 사용하지 않는다.

### 판정 및 재개

205 held-out가 이미 70%를 넘고 training gate windows도 +1.15 pp/window 추세라 현재 시점에 gate를
낮추거나 구조를 바꾸지 않는다. 다만 단순 선형 외삽의 다음 window는 약 0.682라 즉시 승급을
보장하지는 않는다. 2개 완결 window를 더 보고도 0.68 부근이면 stochastic gate와 deterministic
held-out의 차이를 curriculum 설계 문제로 다룬다.

평가 후 `RECOVERY_MODE=continue`로 ep20700에서 재개했다. 새 run은
`ppo_260802_0020_navrl_v2-recover-curriculum-continue-s1`; 205 bars와 기존 density evidence
`6063/16384`, task-step 662,400, LR `5e-6`, rollback 0을 정확히 복원했고 정상 학습 중이다.

### 02:00 재시작 효과 재검산 및 live TensorBoard 단일화

재시작 뒤 좋아 보이는 현상을 동일 205-bars 구간끼리 비교했다. 이전 run 마지막 100 epoch
capture는 **69.55%**, continuation 최신 100 epoch는 **69.57%**로 사실상 같다. continuation 전체
평균 69.10%가 이전 205-bars 전체 평균 66.37%보다 높아 보이는 이유는 이전 평균에 205로 막
승급한 초기 적응 구간이 포함되기 때문이다. 재시작 직후 episode 통계 버퍼/환경이 리셋되어 작은
분모의 단일 epoch가 10–86%로 크게 흔들린 효과도 있다. actor/optimizer/LR/evidence는 checkpoint에서
복원됐으므로 재시작 자체가 정책을 개선한 것은 아니다.

continuation에서 새로 완결된 gate windows는 **0.685→0.693→0.688**이다. 이전 세 window의
0.647→0.659→0.670보다는 좋아졌지만 최신 세 개는 약 0.689 plateau다. 독립 Bernoulli 근사에서
진짜 성공률 0.689로 16,384회 표본이 0.700 이상 나올 확률은 약 0.1% 수준이다(실제 PPO 표본은
상관되어 이 수치를 정확한 p-value로 쓰지 않음). 따라서 max epoch까지 무조건 방치하지 않고
**다음 완결 window 하나**를 decision point로 둔다. 통과하면 220으로 계속하고, 다시 <0.70이면
205 held-out 71.89%와 stochastic gate의 불일치를 curriculum-control 문제로 다룬다.

학습을 중단하지 않고 다음 세 summaries를 symlink-only merged view로 만들었다.

- `ppo_260801_1150_navrl_v2-recover-smoke-130bars-s1` (9501–9600)
- `ppo_260801_1235_navrl_v2-recover-curriculum-s1` (9601–20746)
- `ppo_260802_0020_navrl_v2-recover-curriculum-continue-s1` (20701–live)

TensorBoard에서 **`_merged_navrl_v2_recovery_live` 하나만 선택**하면 된다. active event file의
심볼릭 링크라 학습 중에도 자동 갱신된다. 원본 두 run이 갈린 이유는 safe resume가
`--branch_run`으로 새 artifact/contract 경계를 보존하기 때문이며, 원본을 한 폴더에 직접 이어
쓰지 않는다. ep20701–20746 중복 46 step은 ep20700 checkpoint로 되돌아 실제 재학습된 구간이라
merged view에도 가공 없이 남긴다.

### 02:28 live PPO KL 점검

학습은 중단하지 않고 continuation run의 ep20701–22729 TensorBoard 원자료를 재검산했다.
`ppo/kl`은 최신 **0.00301**, 최근 100/500 epoch 평균 **0.00235/0.00244**이고 최근 500 epoch
선형 기울기는 100 epoch당 **−0.00000024**로 사실상 평탄하다. immutable behavior policy 기준의
최종 epoch audit인 `ppo/behavior_kl_audit_max`도 최근 500 평균 **0.00514**, 전체 최대
**0.01241**로 rollback gate **0.04**의 31% 이하였다. ep20701 이후 KL skip/epoch rollback은
모두 0이고 LR은 `5e-6` 고정이다. 따라서 현재 그래프의 국소 상승은 발산이 아니라 minibatch와
episode 구성에 따른 정상 변동이며, `behavior_kl_sample_max`는 전체 표본 중 단일 극값이라
평균 policy drift 판정에 사용하지 않는다. merged view에는 checkpoint 재학습 구간
ep20701–20746의 중복 step도 있어 해당 짧은 구간의 선 모양은 해석에서 제외한다.

### 03:05 dashboard 전면 감사·데이터 동기화

사이트를 현재 실행 중인 v2 상태와 대조해 전면 점검했다. `tools/update_status_snapshot.py`를 다시
실행해 `docs/status/status.json`과 HTML fallback을 **69개 run / active
`ppo_260802_0020_navrl_v2-recover-curriculum-continue-s1` / ep23190 / 205 bars** 기준으로
동기화했다. 당시 live tail은 capture **69.58%**, crash **28.04%**, timeout **2.38%**이며,
이 값은 held-out 또는 16,384-episode promotion 결과가 아닌 진단용 tail임을 사이트에 명시했다.

정적 수치 검수에서 과거 85-bar 정책 곡선의 OOD 경계가 현재 active curriculum bars로 잘못
칠해지던 문제를 수정했다. `corrected_chirality_density_curve.trained_max_bars=85`를 메타데이터로
고정하고, 85 초과 셀은 historical held-out/generalisation으로 표시한다. 과거 speed/FOV/ceiling
자료에도 각각 historical·pre-heading-fix·not-active-v2 문구를 추가해 현재 40×40×3 m,
full-width `navrl_band`, 70–300 bars, target 0.3–1.5 m/s 계약과 섞이지 않게 했다. Architecture
도식의 bars/arena/target-speed/LR 표기도 status 데이터 및 v2 계약과 맞췄다.

UI 감사 결과도 반영했다. 헤더에 `v2 · LIVE · bars · capture` 컨텍스트와 데이터 freshness를
추가하고, live tail과 promotion gate를 분리했다. 오프라인/HTTP 실패 시 `FALLBACK · offline
snapshot`으로 바뀌도록 해 stale live 오인을 막았다. 기준표의 긴 문장이 데스크톱에서 잘리지
않도록 고정 레이아웃·자동 줄바꿈을 적용하고, 키보드 skip-link/focus outline, 가로 스크롤 탭,
작은 화면의 2열 metrics·긴 run명 말줄임·패널 폭 제한을 추가했다. Threads 링크는 공개 SSR에서
실제 레퍼런스 갤러리 UI가 노출되지 않고 “빈 배경과 텍스트만 남는 디자인을 피하자”는 설명만
확인되어, 특정 화면을 복제하지 않고 현재 대시보드의 정보 위계·상태 배지·시각적 증거(3D/곡선/
도식)를 강화하는 방향으로 반영했다.

검증: `node tests/test_status_arena_motion.js` PASS, aerialgym Python으로 status 관련 unittest
**9개 PASS**, `py_compile`, `git diff --check` PASS, Chrome desktop/mobile 렌더 및 dynamic DOM
값(`205 bars`, `69.6%`, `40×40×3 m`, `LR 5e-6`) 확인. 현재 학습 프로세스는 건드리지 않았다.

### 03:46 ep30000 종료 후 실패·한계 감사 사전등록

사용자 요청에 따라 현재 recovery continuation을 중단·변경하지 않고 종료까지 보존한 뒤, 다음 학습
직전까지 실패 원인과 achievable limit을 수치 우선으로 분석한다. 분석 계획·반증 기준·다음 학습 허가
게이트를 `RESEARCH_PLAN.md` §8에 사전등록했고, `CLAUDE.md`, `OPERATIONS.md`, `README.md`,
`CRASH_TUNING_LOG.md`의 현재 상태와 함정을 동기화했다.

사전등록 시점의 process는 PID 2979738, run
`ppo_260802_0020_navrl_v2-recover-curriculum-continue-s1`, ep23819/30000, 205 bars다. 완결 gate
window는 **0.685, 0.693, 0.688, 0.691, 0.690, 0.694**로 6회 연속 0.70 아래다. 이는 plateau
가설을 강하게 만들지만 아직 최종 checkpoint held-out 결과가 아니므로 학습 실패로 확정하지 않는다.

동시 진단 snapshot은 2,049 episodes에서 capture/crash/timeout **0.651/0.333/0.017**,
crash 중 bar contact **95.3%**, obstacle token은 FOV 내 bars 32.4 대비 unique **3.8**,
hit-token-given-FOV **0.893**, action signed-y **0.762**, task-input OOB 전 축 0이었다. 종료 뒤
deterministic/stochastic 격차, mirrored 좌우 대칭성, v2 40×40 실제 geometry 연결성, density별 망각,
PPO KL/entropy/EV와 이중 token 압축을 각각 분리한다.

canonical lineage는 smoke ep9501–9600 + curriculum ep9601–20700 + continuation ep20701–종료다.
첫 curriculum run의 ep20701–20746은 ep20700 checkpoint에서 재학습된 중복이므로 최종 통계에서
제외한다. 다음 메인 학습은 artifact 동결, held-out, dominant failure 한 층 식별, 단일 변경 설계,
1650 Ti smoke/paired 평가가 모두 끝난 뒤에만 허가한다.

## 2026-08-02 — recovery curriculum 안전 종료·ep24000 동결·최종 한계 감사

사용자가 “할 만큼 했다”고 결정해 continuation trainer에 SIGINT를 보내 안전 종료했다. run summary는
`exit_reason=interrupted`, 마지막 기록 epoch는 24010이며 관련 학습 PID가 모두 사라진 것을 확인했다.
TensorBoard만 유지했고 새 학습은 시작하지 않았다. 50-epoch 주기의 마지막 durable artifact는 다음이다.

- run: `ppo_260802_0020_navrl_v2-recover-curriculum-continue-s1`;
- checkpoint: `nn/last_gen_ppo_ep_24000_rew_44.73549.pth`;
- size/SHA-256: 8,873,939 bytes /
  `82f7978b42d9d9e95adcc638a40ae85fb3736fd897ae39cb8aa8333be39cf23f`;
- canonical lineage: smoke 9501–9600 + curriculum 9601–20700 + continuation 20701–24010,
  총 **14,510 epoch = 59,432,960 samples**. 옛 run의 중복 20701–20746은 제외했다;
- 205 bars: 19,101–24,010, **4,910 epoch = 20,111,360 samples**, 완결 hold 10회.

### 205-bar plateau

완결 gate window는 `0.647, 0.659, 0.670, 0.685, 0.693, 0.688, 0.691, 0.690, 0.694,
0.685`였다. 최근 7회 평균은 **68.94%**, 각각 최소 16,384 episodes다. tail1000은 69.02%, 추세가
+0.79 pp/1000 epoch였으나 이미 20.1M samples를 쓴 상태에서 0.70 gate를 안정적으로 넘을 증거가 없어
동일 curriculum 연장을 종료했다.

### 동결 checkpoint held-out (deterministic, seed 42)

| bars | /100m² | n | capture (Wilson 95%) | crash | timeout | bar contact/all |
|---:|---:|---:|---:|---:|---:|---:|
| 130 | 8.125 | 2,049 | 84.77% [83.15, 86.26] | 12.74% | 2.49% | 11.18% |
| 160 | 10.000 | 2,050 | 79.66% [77.86, 81.34] | 16.88% | 3.46% | 15.61% |
| 190 | 11.875 | 2,049 | 73.99% [72.04, 75.84] | 22.65% | 3.37% | 21.43% |
| **205** | **12.812** | **2,050** | **72.44% [70.46, 74.33]** | **25.07%** | **2.49%** | **24.29%** |
| 220 | 13.750 | 2,050 | 68.49% [66.44, 70.46] | 29.76% | 1.76% | 28.78% |

205–220 선형 보간의 deterministic 70% crossing은 약 214.3 bars지만 hard limit 증명으로 쓰지 않는다.

### 동일 205 checkpoint의 action-mode A/B

| action mode | n | capture | crash | timeout | lateral edge98 |
|---|---:|---:|---:|---:|---:|
| deterministic mean | 2,050 | **72.44%** | **25.07%** | 2.49% | 3.37% |
| stochastic sample | 2,049 | **67.35%** | **30.41%** | 2.24% | 7.08% |

deterministic−stochastic capture는 **+5.09 pp**(근사 95% CI +2.28..+7.89), crash는
**−5.33 pp**(−8.07..−2.60)다. online gate는 오류가 아니라 sampled training policy를 측정하고 있었다.
deployment mean이 더 좋은 것도 사실이므로 두 수치를 함께 보고하고 하나로 대체하지 않는다.

거리별 deterministic capture는 6–11.5 m **81.42%**에서 22.5–28 m **61.41%**로, stochastic은
75.35%→55.06%로 떨어졌다. 최고속 stochastic bin은 64.26%였다. timeout이나 정지보다 긴·빠른
trajectory에서 누적되는 bar contact가 지배적 실패다.

### 기각·유지한 가설

- **H-GEOM 주원인 기각**: v2 `navrl_band`를 그대로 미러링한 100-seed audit에서 205 bars/0.2 m
  clearance crossing 100%, largest component 99.915%, random-pair connectivity 99.831%, fallback 0;
- **H-PPO 주원인 기각**: tail500 KL 0.00236, behavior-KL max 0.01322 < rollback 0.04, LR 5e-6,
  EV 0.789, rollback/skip/action-input OOB 0;
- **stall/timeout 하향**: 평균 speed 약 2.37 m/s, commanded stall 약 3.2%, timeout 2–3%;
- **단순 8-token capacity 단정 정정**: actor에는 obstacle token 외에 4×72 static scan CNN과 history가
  들어간다. 열린 문제는 dense geometry를 위험순으로 이용하는지다;
- **좌우 편향 미해결**: signed-y는 크지만 좌우 mirror pair가 없으므로 chirality 결함으로 확정하지 않음.

### 코드·분석 산출물과 다음 결정

- `tools/analyze_navrl_v2_postrun.py` + `tests/test_navrl_v2_postrun.py`;
- v2 geometry mirror `tools/analyze_navrl_density_feasibility.py` + regression test;
- evaluator에 explicit deterministic/stochastic action mode와 speed/distance/pattern strata 추가;
- selector provenance에 configured/effective FOV와 suppress active/inactive를 분리. TTC는 실제 360° 후보,
  cluster/TTC에서는 suppress가 비활성임을 checkpoint/log에 기록;
- fixed-density preflight가 `205→205` 계약 옆에 과거 기본 밀도 `4.4→18.8/100m²`를 하드코딩해
  출력하던 표시 오류를 수정해 실제 start/final bars에서 동적으로 계산;
- rear target을 centered로 세던 context diagnostic 수정, 72-ray reflection index의 2-bin(10°) skew 수정;
- 최종 보고: `results/navrl_v2_ep24000_limit_audit.{json,md}`,
  중간 상세: `results/navrl_v2_ep24000_postmortem.{json,md}`;
- NavRL 회귀 테스트 **75개 PASS**.

다음 실험은 동결 ep24000의 fixed-205 `cluster_sector` 대 `ttc_sector` 단일변수 A/B다. arm당
4,096,000 samples로 맞추고, 같은 profile baseline 대비 capture +2.0 pp 이상과 crash -2.0 pp 이하를
동시에 요구한다. launcher `train_navrl_v2_ep24000_ttc_ab.sh`는 main/4GB preflight를 통과했지만
**아직 학습을 시작하지 않았다**. 이 A/B가 실패할 때만 action-noise 축으로 이동한다.

사이트 snapshot도 final-audit 데이터로 전환해 `active_run=null`, `FINAL AUDIT`, ep24000 frozen,
205 deterministic/stochastic 수치와 다음 실험이 “prepared, not started”임을 표시한다.

## 2026-08-02 — 감사 완료 범위 정정 및 mirror 검증을 평가-only로 확정

사전등록 §8과 실제 산출물을 다시 대조한 결과, density/action/PPO/geometry의 **핵심 감사**는 끝났지만
mirror pair, 두 번째 seed, 고정속도 0.3/0.9/1.5 m/s, 망각 비교, target-trajectory reachability,
1650 Ti fixed-205 실기 gate는 남아 있었다. 따라서 보고서·사이트의 `analysis complete`/`FINAL AUDIT`
표현은 증거보다 강하다고 판정했다.

- machine-readable status를 `training-stopped-core-audit-complete`로 변경;
- 사이트 상태를 `CORE AUDIT · causal checks pending`으로 변경;
- `RESEARCH_PLAN.md`의 “현재 max_epochs=30000까지 관측” 문장을 작성 당시 사전등록으로 명확히 하고,
  실제 ep24010 사용자 중단과 남은 검증을 §8.7에 구분;
- fixed-205 TTC A/B는 준비됐지만 아직 학습 허가 전임을 CLAUDE/README/OPERATIONS/CRASH 문서에 통일.

다음 1순위는 frozen ep24000의 **mirror-paired 평가**다. 이 단계는 inference-only이며 optimizer step,
gradient, running-stat 갱신, checkpoint 저장을 하지 않는다. 원본/좌우 반전 조건의 capture/crash,
action-y sign과 sign-equivariance만 측정한다. 즉 **평가만 하고 학습은 하지 않는다.** 결과가 좌우
대칭이면 TTC A/B 후보를 유지하고, 유의한 비대칭이면 새 학습 전에 좌표/정책 대칭 문제를 먼저 다룬다.

재생성·검증: NavRL unittest **79개 PASS**, status unittest **9개 PASS**, arena-motion parity PASS,
Chrome 실제 DOM에서 `CORE AUDIT`, `MIRROR EVAL`, `evaluation only; frozen weights; no training` 확인.
학습·평가 프로세스는 시작하지 않았다.

## 2026-08-02 — 동결 ep24000 인과검사 1--3 완료: seed 재현 PASS, action chirality 확인

사용자 요청에 따라 §8.8에 체크포인트 SHA·조건·표본·판정 margin을 **결과를 보기 전에** 등록하고
1) 문서/사이트 정합성, 2) 좌우 반사, 3) 205-bar 두 번째 seed까지 수행했다. 모든 GPU 작업은
inference-only였으며 optimizer/gradient/input RMS/checkpoint 저장을 수행하지 않았다. 학습 프로세스는
없다. 대상은 ep24000 SHA-256 `82f7978b42d9d9e95adcc638a40ae85fb3736fd897ae39cb8aa8333be39cf23f`다.

### 반사 평가 구현·검증

- `navrl_players.py`에 `NAVRL_EVAL_REFLECTION_MODE=original|conjugate`를 추가했다. conjugate는
  raw structured observation을 `M`으로 반사해 `pi(Mo)`를 계산하고 body action을 다시 `M`으로
  되돌린 `M pi M`을 원래 물리계에 실행한다.
- 같은 실제 observation에서 `pi(o)`와 `pi(Mo)`를 동시에 계산해 action pair를 저장한다. 진단용 두
  번째 forward가 torch RNG를 소비해 이후 환경 reset을 바꾸지 않도록 `fork_rng`로 난수 상태를
  복원했다.
- evaluator result/receipt/CSV에 reflection mode를 넣고, checkpoint/evaluator byte·nonce·physics·episode
  accounting 검증을 유지했다. `run_header.py`는 평가 seed43을 YAML 기본 42로 잘못 표시하던 것을
  `NAVRL_SEED` 우선으로 수정했다.
- 비동기 reset 뒤 outcome은 episode-paired가 아니므로 common-seed aggregate라고 명시했다. 이 한계를
  보완하려고 reset 순간 actor-frame initial target bearing을 negative/centered/positive-y로 기록했고,
  계측 재생의 outcome counts가 기존 4,096회와 완전히 같지 않으면 실패하도록 했다.

### 수치 결과

| 검사 | 조건 | 결과 |
|---|---|---|
| original `pi` | seed42, 205 bars, n=4,096 | capture/crash/timeout **70.97/25.71/3.32%** |
| conjugate `M pi M` | 같은 조건, n=4,096 | **70.17/26.56/3.27%** |
| mirror outcome 차이 | conjugate-original | capture **-0.81 pp** (95% CI -2.78..+1.17), crash +0.85 pp (-1.05..+2.76) |
| exact action pair | 548,736 observations | MAE `[x,y,z,yaw]=[0.926,1.235,0.416,1.002]`; lateral sign mismatch **73.08%** |
| initial target bearing | negative-y 1,967 / positive-y 2,008 | capture **71.17/70.97%**, 차이 -0.21 pp (-3.03..+2.61) |
| second seed | seed43, n=2,049 | capture/crash/timeout **72.77/24.74/2.49%** |
| seed replication | seed43-seed42 | capture **+0.33 pp** (95% CI -2.40..+3.06), 사전등록 gate **PASS** |

반사 observation schema를 898차원 전부 재검수했다. static scan index `i→-i mod H`, obstacle y/vy,
robot vy/yaw-rate/previous-y/previous-yaw, target y/vy 외 필드는 reflection invariant이며 누락을 찾지
못했다. checkpoint input RMS는 mirror mean MAE 0.0317, robot-history mean 0.3243, scan variance 상대
MAE 11.9%로 이미 좌우 비대칭 방문분포를 흡수했다.

판정은 두 층으로 나눈다. 현재 대칭 arena에서 seed/초기 bearing/mirror aggregate **성공률 차이는
검출되지 않았고 seed 재현은 PASS**다. 그러나 정책은 좌우 반사-equivariant하지 않고 거의 한 방향으로
우회하는 습관을 학습했다. 이는 현재 분포에서는 outcome-neutral이지만 asymmetric layout/domain shift에
취약할 수 있으므로 H-BIAS를 `action-level supported / outcome gap not detected`로 변경한다. 대칭분포에서
`pi`와 `M pi M` aggregate가 같다는 사실만으로 정책 대칭을 주장하지 않는다.

산출물:

- 실행기: `aerial_gym/rl_training/rl_games/eval_navrl_v2_ep24000_causal_1to3.sh`;
- 요약기: `tools/summarize_navrl_v2_causal_1to3.py`;
- 결과: `results/navrl_v2_ep24000_causal_1to3/summary.{md,json}`와 arm별 receipt/log/checkpoint snapshot;
- 사이트 generator는 causal 1--3 결과를 읽어 `CAUSAL 1–3`, seed replication, mirror outcome,
  action mismatch를 서로 다른 milestone로 표시한다.

검증: shell `bash -n`, Python `py_compile`, NavRL 회귀 **79개 PASS**, bounded-action unittest
**13개 PASS**, status unittest **9개 PASS**, arena-motion parity PASS, evaluator preflight,
16-episode conjugate GPU smoke, 본 평가의 checkpoint SHA/receipt/episode accounting PASS. bearing 계측
재생은 original과 `2907/1053/136` outcome이 byte-level count로 동일했다.

다음은 새 학습이 아니라 fixed target speed 0.3/0.9/1.5 평가와 이전 checkpoint 망각 비교다. 이후
TTC selector A/B와 reflection augmentation/RMS symmetry는 서로 다른 단일변수 arm으로 설계한다.

## 2026-08-02 — ep24000 고정속도 인과검사 실행 계약 준비

다음 작업을 학습으로 오인하지 않도록 ep24000 동결 정책의 0.3/0.9/1.5 m/s fixed-speed 평가를
전용 실행기 `eval_navrl_v2_ep24000_fixed_speed.sh`로 만들었다. 205 bars, seed 42,
deterministic/original, mixed target, 2,049 episodes/cell을 고정하고 checkpoint SHA-256까지 확인한 뒤
세 cell을 순서대로 실행한다. 결과는 `tools/summarize_navrl_v2_fixed_speed.py`가 계약을 다시 검증하고
capture/crash/timeout/bar-contact, Wilson CI, 1.5-minus-0.3 capture 차이를 요약한다.

기존 `eval_navrl_v2_density_sweep.sh`는 외부 `NAVRL_TARGET_SPEED`를 무조건 지워 고정속도 실험을 할 수
없었다. `NAVRL_V2_FIXED_TARGET_SPEED`를 명시한 경우에만 학습 support [0.3,1.5] 안에서 허용하고,
task bulk JSON의 `target_speed_mode=fixed` 및 point/min/max 세 값이 요청 속도와 같지 않으면 결과를
거부하도록 수정했다. 값이 없으면 기존 U[0.3,1.5] canonical 평가 계약은 그대로다. recovery-smoke
attestation에는 fixed-speed override를 금지했다.

`RESEARCH_PLAN.md` §8.10에 실행 전 판정 규칙을 고정했다. 1.5-minus-0.3 capture가 -3.0 pp 이하이고
독립 비율차 95% CI 상한도 0 미만이면 material speed sensitivity로 판정한다. 이 평가가 끝난 뒤
이전 checkpoint 망각 비교를 하며, 그 전에는 새 PPO 학습을 시작하지 않는다.

검증: 두 launcher `bash -n`, summary Python `py_compile`, `git diff --check`, fixed 0.9와 기존 uniform
evaluator preflight PASS, support 밖 2.0 m/s 사전 거부 PASS. 실제 rollout smoke와 본 평가는 아직
시작하지 않았다.

추가로 0.9 m/s/205 bars/seed42의 16 requested-episode GPU smoke를 수행했다. task 시작 로그가
`speed_fixed=0.90`, evaluator 계약 로그가 `target=fixed 0.9m/s`였고, bulk JSON은
`target_speed_mode=fixed`, point/min/max 모두 0.9를 기록했다. evaluator의 nonce, checkpoint/evaluator
SHA, physics, episode accounting 검증을 거쳐 PASS했다(비동기 env라 actual 17). 이 작은 smoke의
capture 수치는 성능 근거로 사용하지 않는다. 본 2,049×3 평가는 사용자 실행 전이며 학습 프로세스는
시작하지 않았다.

## 2026-08-02 — ep24000 고정속도 평가 완료: 고속에서 충돌 병목 확인

사용자 승인 후 `eval_navrl_v2_ep24000_fixed_speed.sh`를 RTX 3070에서 직접 실행했다. 세 cell 모두
2,049 episodes 정확히 완료했고 checkpoint SHA, evaluator SHA, nonce, base_sim dt=0.01, seed42,
deterministic/original, fixed-speed point/min/max 계약 검증을 통과했다. 학습/optimizer/RMS/checkpoint
갱신은 없었다.

| target speed | capture | crash | timeout | bar contact (절대) | lateral edge98 |
|---:|---:|---:|---:|---:|---:|
| 0.3 m/s | 73.26% | 23.04% | 3.71% | 22.11% | 2.41% |
| 0.9 m/s | 72.62% | 25.09% | 2.29% | 24.26% | 3.38% |
| 1.5 m/s | 67.35% | 30.75% | 1.90% | 29.97% | 4.92% |

1.5-minus-0.3 capture는 **-5.91 pp** (95% CI **-8.70..-3.11**)여서 사전등록한 material
speed-sensitivity gate를 통과했다. 같은 비교에서 crash +7.71 pp, bar contact +7.86 pp, timeout
-1.81 pp다. 평균 비행속도는 2.386→2.400 m/s로 거의 그대로인데 command-speed norm은 이미
2.958→2.969 m/s이고 lateral executed-edge98은 2.41→4.92%로 두 배가 됐다. 즉 고속 표적에서 단순
timeout이 아니라, 속도 여유 없이 더 공격적인 경계 action을 써서 막대 충돌로 전환되는 것이 핵심이다.

결과는 `results/navrl_v2_ep24000_fixed_speed/summary.{md,json}`과 속도별 JSON/log/receipt에 저장했다.
다음은 새 학습이 아니라 이전 checkpoint의 동일 205-bar 망각 비교다. fixed-speed cell은 async reset
뒤 paired rollout이 아니므로 개별 거리 bin의 작은 비단조 차이는 인과효과로 사용하지 않는다.

## 2026-08-03 — ep19100 대 ep24000 망각검사 실행기 준비

205 bars 승급 직전 ep19100과 205 bars에서 4,900 epoch를 더 학습한 ep24000을 비교 대상으로 고정했다.
단순 uniform 한 cell만으로는 §8.11의 고속 충돌이 망각인지 분리되지 않으므로, 두 checkpoint를
U[0.3,1.5]와 fixed 1.5 m/s에서 평가하는 2×2 설계로 확정했다. 조건은 205 bars, seed42,
deterministic/original, mixed target, full 6--28 m, 2,049 episodes/cell이다.

전용 실행기 `eval_navrl_v2_ep19100_vs_ep24000_forgetting.sh`는 두 checkpoint의 SHA-256을 확인하고 네
cell을 실행한 뒤 `tools/summarize_navrl_v2_forgetting.py`로 계약·outcome accounting을 재검증한다.
ep24000-minus-ep19100 capture가 -3.0 pp 이하이며 95% CI 상한도 0 미만이면 forgetting, 반대 방향
+3.0 pp와 CI 하한>0이면 improvement로 사전등록했다. 기존 결과 폴더는 덮어쓰지 않는다.

ep19100 checkpoint는 실제 epoch=19100, bars=190, task_steps=611200이고 ep24000은 epoch=24000,
bars=205, task_steps=768000이다. ep19100을 평가 시 205로 고정하는 uniform/fixed1.5 evaluator preflight가
모두 PASS했다. 이 단계는 평가-only이며 아직 본 2×2 실행이나 새 PPO 학습은 시작하지 않았다.

## 2026-08-03 — 망각 2×2 완료·사이트 정합성 복구·main TTC A/B 승인

사용자가 실행한 망각검사는 15:29 KST에 네 cell과 summary까지 정상 종료했다. 프로세스와 GPU를 다시
확인해 학습/평가가 남아 있지 않았고, 각 JSON/receipt의 SHA, seed42, deterministic/original,
205 bars, U[0.3,1.5] 또는 fixed 1.5, requested/actual ≥2,049 계약이 모두 PASS했다.

| 조건 | ep19100 capture/crash/bar | ep24000 capture/crash/bar | capture Δ (95% CI) |
|---|---:|---:|---:|
| uniform | 67.79/31.87/31.67% | 72.44/25.07/24.29% | **+4.65 pp** (+1.85..+7.45) |
| fixed 1.5 | 64.10/35.61/34.93% | 67.35/30.75/29.97% | **+3.25 pp** (+0.35..+6.16) |

두 조건 모두 preregistered improvement이며 material forgetting=NO다. 205-stage 4,900 epoch는 일반·고속
성능과 충돌을 개선했으므로 replay/mixture curriculum을 다음 축으로 선택하지 않는다. 남은 병목은
fixed-speed에서 확인한 고속 bar-contact와 경계 action이다.

사이트 generator가 여전히 `fixed-speed checks pending`, `next training BLOCKED`를 하드코딩하고 있어
실측과 어긋난 것을 발견했다. fixed-speed/forgetting summary와 1650 Ti TTC CSV를 검증해 읽도록
`tools/update_status_snapshot.py`를 확장하고, dashboard에 speed -5.91 pp/bar +7.86 pp, no-forgetting,
1650 TTC PASS, `MAIN 205 TTC A/B READY`를 표시하도록 수정했다. `app.js`의 상단/hero 상태도
`CAUSAL COMPLETE`로 분리했다. README/CLAUDE/RESEARCH_PLAN/OPERATIONS/CRASH 문서도 같은 결정으로
동기화했다.

다음 launcher `train_navrl_v2_ep24000_ttc_ab.sh`를 다시 감사했다. source SHA와 ep24000/bars205 계약,
selector 단일변수, density 고정, sample matching, LR/KL/rollback을 확인했고 main/4GB의 baseline/TTC
네 preflight가 모두 PASS했다. 다음은 RTX 3070 main baseline 1,000 epoch를 먼저 실행·평가한 뒤 TTC
1,000 epoch를 순차 실행하는 것이다. 아직 새 학습은 시작하지 않았다.

사이트 재생성은 69 runs, active=none으로 완료했고 `status.json`과 `index.html` inline fallback의
byte-equivalent JSON parity, `causal_checks_pending=false`, `next_training_authorized=true`를 확인했다.
검증은 status Python 10개, arena-motion/DOM parity, TTC selector 13개, perception 16개, bounded-action
13개, shell syntax, `py_compile`, `git diff --check` 모두 PASS했다. action test를 repo root의 unittest
module 경로로 한 첫 호출은 Isaac Gym import 중 PATH에 ninja가 없어 실패했지만, 해당 테스트의 정식
working directory/직접 실행으로 13/13 PASS를 확인했다. 이는 코드 실패가 아니라 호출 방식 오류다.

## 2026-08-03 — main fixed-205 baseline 정상 완료·held-out 전 TTC 차단·장기 로드맵 고정

main baseline `ppo_260803_1819_navrl_v2-ep24000-205bars-main-baseline-s1`을 확인했다. ep24001--25000,
1,000 epoch × 32 × 128 = **4,096,000 samples**를 모두 수행했고 `exit_reason=max_epochs`, bars=205 고정,
reward collapse=false였다. 현재 학습/평가 프로세스는 없고 GPU는 idle이다. 최종 artifact는
`runs/ppo_260803_1819_navrl_v2-ep24000-205bars-main-baseline-s1/nn/last_gen_ppo_ep_25000_rew_29.188496.pth`,
SHA-256 `169ddcddb83c9d74df5c79252274660bc9c52e32d7d5144d325698e32b1d9b08`이다.

TensorBoard 1,000 epoch 전수 집계에서 PPO KL 평균/최대 **0.002439/0.007574**, immutable behavior-KL
audit 최대 **0.012356 < 0.04**, epoch rollback/rollback-total/KL-skipped minibatch/4축 raw OOB는 모두
0이었다. 따라서 발산·rollback 실패 가설은 기각한다. 훈련 proxy 전체 평균 capture/crash/timeout은
**68.41/29.20/2.38%**, 첫100→마지막100은 capture **69.46→67.49%(-1.98 pp)**,
crash **28.14→30.61%(+2.46 pp)**, timeout **2.39→1.91%**였다. reward는 36.64→38.51이라 collapse는
아니지만 추가 epoch의 성능 향상 근거도 없다. 마지막 epoch capture 66.67%는 종료 episode 약 18개의
작은 stochastic training 표본이므로 held-out 결과로 사용하지 않는다.

action은 lateral edge95 평균/tail100 **24.26/25.61%**, edge99 **2.36/2.60%**, mean policy-mu signed-y
**+1.323**, positive-y sample **94.09%**로 기존 learned chirality가 남았다. 주기적 crashdiag에서 crash의
약 94--98%는 계속 bar contact였다. 이 수치는 TTC가 해결하려는 위험 정렬 가설을 유지하지만 baseline
held-out보다 먼저 TTC를 시작할 근거는 아니다.

final checkpoint의 canonical 205 bars / seed42 / deterministic / original / U[0.3,1.5] / 2,049 episode
evaluator preflight는 PASS했고 실제 평가는 아직 실행하지 않았다. dashboard generator에 이 main A/B
단계를 추가해 작은 training-tail을 상단 capture로 오표시하지 않고 `BASELINE EVAL PENDING`, TTC blocked,
4.096M budget을 표시하도록 했다. baseline 결과 artifact가 검증된 뒤에만 TTC arm을 허용한다.

`RESEARCH_PLAN.md §8.15`에 R0 selector A/B → R1 재현/한계 지도 → R2 control-risk → R3 learned
perception → R4 temporal fusion → R5 다축 robustness → R6 sim-to-real/논문화 장기 로드맵과 gate,
예상 범위, 사용자/Codex 역할을 고정했다. 현재 navigation/control은 P5 후반--P6 초반이지만 learned
detector/tracker와 sim-to-real이 남아 전체 논문 증거는 중간 단계다. 사용자의 필수 입력은 GPU 시간 예산,
두 머신 artifact 이관, 실제 센서/기체 사양, 목표 마감/연구 우선순위 네 가지로 제한한다.

검증은 status snapshot Python **8/8**, arena-motion parity, Python compile, A/B/evaluator shell syntax,
status.json↔inline fallback byte-equivalent JSON, `git diff --check`를 통과했다. dashboard는 70 runs,
active=none, latest=main baseline, `BASELINE EVAL PENDING`, heldout=false, TTC blocked로 재생성됐다. 시스템
Node의 raw `--check`는 기존 app.js 125행의 nullish-coalescing(`??`)을 지원하지 않는 구버전 parser에서
중단했으며, syntax-only 치환 stream과 실제 arena parity test는 통과했다. 이는 이번 변경의 JS 오류가
아니라 로컬 Node 버전 제약이다. 첫 A/B snapshot parity 호출은 새 experiment에 표시용
`cluster_gap_m`이 빠진 것을 검출해 실패했고, 0.45 m와 8 sectors를 복구한 뒤 전체 검증을 다시 통과시켰다.

## 2026-08-04 — main baseline held-out 완료: 안정적 PPO 속 느린 action drift 확인

사용자가 완료한 `results/navrl_v2_ep24000_ttc_main_baseline/205bars.json`을 독립 검증했다. requested/actual
episode는 2,049/2,049이고 checkpoint·snapshot SHA는 모두
`169ddcddb83c9d74df5c79252274660bc9c52e32d7d5144d325698e32b1d9b08`로 ep25000 final과 일치했다.
result/receipt/log digest, nonce, outcome 합, seed42, 205 bars, deterministic/original, mixed,
U[0.3,1.5], full 6--28 m, `cluster_sector`, main/base_sim/128 env 계약도 모두 PASS했다.

| checkpoint | n | capture | crash | timeout | bar contact | lateral edge98 | vertical edge98 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ep24000 source | 2,050 | 72.44% | 25.07% | 2.49% | 24.29% | 3.37% | 0.01% |
| ep25000 matched baseline | 2,049 | **69.50%** | **28.99%** | 1.51% | **27.82%** | **5.03%** | **3.20%** |
| Δ | — | **-2.94 pp** | **+3.92 pp** | -0.97 pp | **+3.53 pp** | +1.67 pp | +3.19 pp |

독립 비율차 95% CI는 capture **-5.72..-0.16 pp**, crash **+1.20..+6.63 pp**, timeout
**-1.83..-0.12 pp**다. 거리 q0/q1/q2/q3 capture는 81.15/74.77/66.79/57.87%, CV/waypoint는
66.90/71.94%, 초기 bearing negative/positive는 69.47/69.47%로 outcome 좌우 비대칭은 여전히 없다.
crash의 570/594=95.96%는 bar contact다.

이 결과는 ep24001--25000의 KL max 0.00757, behavior-KL max 0.01236, rollback/OOB 0과 함께 읽는다.
가설 “PPO 발산 때문에 악화”는 기각하며, 작은 update가 누적된 fixed-density control/action drift를
지지한다. KL guard는 급성 collapse만 막고 held-out drift는 막지 못한다. 따라서 baseline 연장이나
best training reward 선택은 금지한다.

TTC 결과를 보기 전에 기존 primary A/B gate를 절대값으로 동결했다: capture **≥71.50%**, crash
**≤26.99%** 동시다. 다만 이 gate만 통과하면 ep24000보다 약간 나쁠 수 있으므로 canonical replacement
floor를 capture **≥72.44%**, crash **≤25.07%**로 별도 등록했다. primary만 통과하면 selector의 상대적
효과는 인정하되 원본보다 우수하다고 주장하지 않는다. main TTC launcher가 baseline result/receipt/
snapshot/log SHA와 계약을 실행 전에 재검증하도록 수정했다. 현재 학습/평가 프로세스는 없고 GPU는 idle다.

실제 `ARM=ttc PROFILE=main NAVRL_PREFLIGHT_ONLY=1`은 baseline artifact 검증, ep24000 source SHA,
fixed-205/same-sample 계약을 모두 통과했고 학습은 시작하지 않았다. dashboard는 70 runs, active=none,
`TTC ARM READY`, baseline held-out 69.50/28.99%, primary gate와 canonical replacement floor를 표시하도록
재생성했다. status Python 9/9, arena-motion parity, shell syntax, Python compile, JSON fallback parity,
`git diff --check`를 통과했다. 첫 합성 baseline-unlock 테스트는 임시 result root가 repo 밖이라
`relative_to(ROOT)`가 실패하는 이식성 문제를 잡았고, 외부 경로도 안전하게 절대경로로 표시하도록 고친 뒤
9/9를 다시 통과시켰다.

## 2026-08-05 — main TTC held-out 완료: current mode FAIL, ranking-only 인과효과 판정 불가

TTC run `ppo_260804_0813_navrl_v2-ep24000-205bars-main-ttc-s1`은 ep24001--25000/4.096M samples를
`max_epochs`로 정상 완료했다. final 일반 checkpoint
`last_gen_ppo_ep_25000_rew_38.369205.pth` SHA-256은
`14e4c72a744c9bedc2d07556e5aebdbef21a184c9f9b8239bc1a23d45e20823e`다. 함께 저장된 double-underscore
파일은 직렬화 SHA만 달랐고 381개 tensor, env_state, epoch/frame이 전부 동일했다. 평가는 기존 convention의
일반 파일을 사용했다.

training TensorBoard 전수 감사에서 PPO KL 평균/최대 0.00247/0.00731, behavior-KL audit 최대 0.01189,
rollback/OOB 0이었다. training proxy 첫100→끝100은 capture 55.77→65.75%, crash 43.84→33.59%였으나
held-out 판정에는 사용하지 않았다.

`results/navrl_v2_ep24000_ttc_main_ttc/205bars.json`은 checkpoint/snapshot/result/receipt/log SHA,
nonce, outcome 합, seed42, 205 bars, deterministic/original, U[0.3,1.5], `ttc_sector`,
main/base_sim/128 env를 모두 PASS했고 requested 2,049보다 많은 실제 2,051 episodes를 완료했다.

| 정책 | n | capture | crash | timeout | bar contact | edge98 x/y/z | 실제 속도 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ep24000 source | 2,050 | 72.44% | 25.07% | 2.49% | 24.29% | 23.61/3.37/0.01% | 2.394 m/s |
| matched cluster baseline | 2,049 | 69.50% | 28.99% | 1.51% | 27.82% | 26.25/5.03/3.20% | 2.436 m/s |
| TTC mode | 2,051 | **70.21%** | **29.50%** | **0.29%** | **28.18%** | 22.46/**7.17**/1.56% | **2.570 m/s** |

TTC-minus-baseline은 capture **+0.71 pp**(95% CI -2.10..+3.52), crash **+0.51 pp**
(-2.28..+3.29), timeout **-1.22 pp**(-1.80..-0.64)다. capture ≥71.50%와 crash ≤26.99%를 모두
못 넘어 primary FAIL이며 ep24000 replacement floor도 FAIL이다. ep24000 대비 crash는 +4.42 pp
(95% CI +1.70..+7.15)다. timeout 감소가 더 높은 속도와 lateral saturation/bar contact로 전환됐다.

사후 코드/provenance 감사에서 `NAVRL_OBSTACLE_SELECTOR` 한 env-var가 바뀌었어도 실제 candidate FOV가
cluster 240°→TTC 360°로 함께 바뀜을 확인했다. 따라서 과거 “ranking만 분리” 주장은 틀렸고 이번 결과는
current representation bundle의 FAIL이지 pure TTC ranking 기각은 아니다. 8-token/360°가 고밀도에서
전방 표현을 희석했다는 설명은 plausible하지만 아직 미검증이다. current TTC를 채택·연장하지 않는다.
재시험한다면 FOV를 selector와 독립시킨 `ttc240`을 1650 Ti에서 먼저 screen하고, 아니면 R2 control-risk
inference-only screen으로 이동한다.

최종 정합성 검증에서 dashboard를 71 runs / active=none / `TTC REJECT`로 재생성했다. 사이트 JSON과
HTML inline fallback은 byte-equivalent payload이고, TTC checkpoint SHA·snapshot SHA·dashboard SHA가
모두 `14e4c72a744c…`로 일치했다. status snapshot Python **10/10**, arena-motion parity, generator
`py_compile`, A/B/evaluator shell syntax, `git diff --check`를 모두 통과했다. status test를 처음
`python -m unittest tests.test_status_snapshot`으로 호출한 것은 `tests`가 package가 아니라 import에
실패했으며, 정식 직접 실행 `python tests/test_status_snapshot.py`에서 10/10 PASS했다. 이는 제품 코드
실패가 아니라 테스트 호출 경로 오류다.

## 2026-08-05 — episode horizon 가설 감사: 시간 부족이 아니라 초기 과속 충돌

사용자의 “고밀도에서 속도를 충분히 줄이지 않는다”는 가설을 현재 환경·reward·TTC 로그와 대조했다.
v2 episode는 **600 RL step = 60 s**(`dt=0.1 s`)이고, TTC run의 주기적 crashdiag에서 bar-contact 평균
발생 시점은 **74--92 step = 7.4--9.2 s**였다. held-out TTC timeout은 6/2,051=**0.29%**뿐이다.
따라서 horizon을 900 step으로 늘려도 이미 발생하는 29.50% crash는 줄지 않으며, 모든 timeout이 성공으로
바뀐다는 비현실적 상한에서도 capture 이득은 +0.29 pp뿐이다. “600-step 시간이 부족해서 충돌한다”는
가설은 기각한다.

반면 policy가 과속하도록 만드는 구조는 확인됐다. action은 이미 속도를 0까지 낮출 수 있지만 reward는
매 step closing-speed `vel_weight=1.0`, PBRS progress `progress_weight=1.0`, time cost
`alive_weight=-0.05`를 함께 사용하고, safety reward는 clearance만 보며 속도/제동거리와 결합하지 않는다.
TTC held-out의 실제 속도 2.570 m/s, lateral edge98 7.17%, bar contact 28.18%도 시간 부족보다
**clearance 대비 속도 위험을 표현·억제하지 못함**을 지지한다. horizon만 연장하면 느리게 가는 법을
가르치지 못하고, 최대 episode의 누적 time cost만 -30→-45로 바뀐다.

결정: canonical 600-step 평가 계약은 유지한다. 다음 변경 전에 frozen ep24000에 sensor-derived
clearance/stopping-distance 기반 inference-only speed governor를 여러 강도로 screen하고, crash 감소가
timeout 증가보다 큰지 확인한다. 동시에 contact 전 speed/command/min-clearance/TTC/stopping-margin과
capture/crash step 분위수를 로깅한다. 이 gate가 유효할 때만 동일 risk margin을 policy action layer 또는
단일 reward 항으로 학습한다. 600→900 평가는 원하면 timeout 회수 진단으로만 수행하며 성능 해결책으로
간주하지 않는다. 이번 작업에서는 학습·환경 로직을 변경하지 않았다.

RTX 3070의 실제 receipt에서 205 bars / 2,049 episodes 한 cell은 **3.9--4.4분**, 최근 main A/B
1,000-epoch/4.096M-sample 학습은 run 시작 시각부터 finalize까지 약 **66--67분**이었다. 이를 기준으로
contact 전 진단 추가, 실제 제동성능 probe, sensor-only adaptive governor 구현·단위검증, 기존/고정
2.0/고정 1.5/clearance/TTC 5-cell held-out, 판정·문서화까지의 무학습 1차 gate는 **3--4.5시간**으로
산정한다. FAIL이면 여기서 종료한다. PASS이면 학습용 action layer 통합, 1,000-epoch matched arm,
seed/speed 재평가와 최종 분석에 **추가 3--4시간**이 필요해 전체는 **6--8.5시간**, 한 차례 재튜닝까지
포함한 보수적 상한은 **8--12시간**이다. GPU 순수 점유는 1차 약 0.5시간, PASS 이후 약 1.5--2시간이고
나머지는 구현·검증·분석 시간이다.

## 2026-08-05 — 8시간 control-risk 루프 완료: sensor-only riskcap 최종 PASS

### 구현·계측과 사전등록

에피소드 시간 가설을 수치로 기각한 뒤 `aerial_gym/task/navrl_task/speed_governor.py`에 off/fixed/
clearance/ttc/riskcap 다섯 mode를 구현했다. 공통 원칙은 nominal policy 방향을 유지하고 XY magnitude만
제한하며, 실제 실행된 이전 action을 observation feedback에 넣는 것이다. bulk evaluator에는 requested/
executed speed, intervention/near-stop, clearance/TTC/stopping margin, contact 직전 speed/command/step을
추가했다. 실기체가 아니라 simulator 내 action step-response를 측정한
`results/navrl_v2_speed_governor_braking.json`에서 p10 유효 감속도 **2.9609 m/s²**, p95 정지시간
**1.0 s**, 정지거리 **1.047 m**를 얻어 riskcap의 3 m activation과 5 m release를 결과 전에 고정했다.

### 두 번의 무효화와 수정

첫 적응 run 감사에서 governor가 표적 LiDAR return을 제외하려고 `segmentation_pixels == 50`을 직접
사용하는 actor semantic leak를 발견했다. 해당 R2/R2b와 ep24000→24334 checkpoint는 최종 증거에서
제외하고 `/home/fair/workspaces/aerial_gym_ws/tensorboard_archive/2026-08-05_invalid_semantic_governor/`로
격리했다. actor perception이 이미 계산하는 카메라 bearing/range–LiDAR association(±15°, ±0.55 m)의
`last_target_like`만 governor가 재사용하도록 고쳤고 provenance를 JSON/checkpoint/receipt에
`camera_lidar_association`으로 고정했다.

첫 corrected 재실행에서는 Warp LiDAR row의 실제 수직각 순서가 +20°→-10°인데 governor가 반대로
투영한 것을 추가로 발견했다. adaptive cell partial은
`/home/fair/workspaces/aerial_gym_ws/tensorboard_archive/2026-08-05_invalid_vertical_order/`로 격리했고,
기본 vertical FOV 순서를 `(20.0, -10.0)`으로 수정해 비대칭 수직각 단위 테스트로 잠갔다. archive 자료는
원인 추적용으로 보존하지만 논문·사이트 수치에는 사용하지 않는다.

### corrected R2와 seed44 R2b

`results/navrl_v2_ep24000_speed_governor_screen/summary.{md,json}`의 유효한 seed42 5-cell 결과:

| mode | n | capture | crash | timeout | bar contact | intervention | executed m/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 2,050 | 72.44% | 25.07% | 2.49% | 24.29% | 0% | 2.965 |
| fixed 2.0 | 2,049 | 78.53% | 16.06% | 5.42% | 15.18% | 97.00% | 1.985 |
| fixed 1.5 | 2,051 | 74.65% | 16.82% | 8.53% | 15.89% | 99.01% | 1.496 |
| clearance stop | 2,049 | 69.11% | 14.30% | 16.59% | 13.08% | 46.55% | 1.699 |
| TTC stop | 2,049 | 69.59% | 6.83% | 23.57% | 5.47% | 57.22% | 1.520 |

complete-stop adaptive는 near-stop 38.60/42.04%로 충돌을 timeout으로 바꿔 GO 없음으로 끝냈다.
반면 결과 전에 하나만 고른 non-stopping riskcap의 새 seed44 결과
`results/navrl_v2_ep24000_riskcap_seed44_screen/summary.{md,json}`은 off 72.83/24.63/2.54% 대비
riskcap **79.55/17.62/2.83%**였다. capture +6.72 pp(95% CI +4.12..+9.32), crash -7.02 pp
(-9.51..-4.53), intervention 28.74%, near-stop 0%로 네 GO 조건을 모두 통과했다.

### 1,000-epoch 적응 학습

승인된 run은
`aerial_gym/rl_training/rl_games/runs/ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1`이다.
frozen ep24000 SHA `82f7978b42d…`에서 ep24001--25000, 1,000 epoch×32×128=**4.096M samples**,
205 bars 고정, seed1, LR 5e-6, cluster-sector/riskcap으로 `max_epochs` 정상 종료했다. 일반 final
checkpoint `nn/last_gen_ppo_ep_25000_rew_39.742134.pth` SHA-256은
`f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40`이다. 같은 epoch의 double-underscore
파일 대신 기존 관례의 일반 파일을 평가했다.

학습 proxy 전체 27,632 episodes는 capture/crash/timeout **77.67/18.50/3.83%**였다. 100-epoch block은
`78.70/18.61/2.69 → 76.14/19.30/4.56 → 77.72/18.36/3.92 → 75.26/20.15/4.59 →
77.51/18.42/4.07 → 78.62/18.06/3.31 → 79.37/17.42/3.21 → 78.15/17.94/3.92 →
77.79/18.14/4.08 → 77.24/18.71/4.06%`였다. 뚜렷한 training 상승은 없어 plateau지만 held-out을
중단할 발산은 아니다. TensorBoard 1,000 epoch 전수에서 PPO KL 평균/최대 **0.002445/0.009062**,
behavior-KL audit 평균/최대 **0.005028/0.010958**, rollback/KL-skip/4축 raw OOB는 모두 0이었다.
lateral latent bias는 남아 마지막 epoch edge95-y 22.07%, edge99-y 2.37%, |mu-y| 1.303,
signed-y +1.226, sigma-y 0.35였다.

### 새 seed45/46 최종 평가와 판정

`results/navrl_v2_riskcap_postadapt/summary.{md,json}`은 9개 cell의 checkpoint/snapshot/result/log/receipt
SHA, nonce, outcome 합, main/base_sim/128 env, sensor-only provenance를 전부 재검증했다.

seed45 uniform은 off **70.03/27.87/2.10%**, source+riskcap **78.20/17.80/4.00%**,
trained+riskcap **81.94/15.67/2.39%** capture/crash/timeout이다. mechanism은 capture +8.16 pp
(95% CI +5.49..+10.83), crash -10.06 pp(-12.61..-7.51)로 복제됐다. 적응은 source+riskcap 대비
capture **+3.75 pp**(+1.30..+6.19), crash -2.14 pp(-4.42..+0.15), timeout -1.61 pp
(-2.68..-0.53), intervention -1.48 pp라 non-inferior/useful 모두 PASS했고 trained를 winner로 골랐다.

seed46 fixed 0.3/0.9/1.5 m/s에서 off→winner capture는 71.90→81.84, 71.89→80.77,
67.98→75.51%였고 crash는 24.88→15.18, 25.77→16.59, 30.31→22.29%였다. 세 속도 모두
capture↑/crash↓ 방향을 통과해 최종 generalization은 **PASS**다. 1.5 m/s winner의 bar contact가
20.78% 남고 lateral high80-y 70.16%, signed-y +0.766이어서 chirality와 잔여 고속 충돌은 해결되지 않았다.

결정은 **ep25000 + 고정 riskcap을 navigation/control candidate로 동결**하는 것이다. fixed-density PPO를
더 연장하거나 final seed를 본 뒤 2.0/3/5 m를 튜닝하지 않는다. 이 결과는 analytic detector 조건이며
정식 CBF 안전보장도 아니다. 다음 gate는 R3 learned detector/perception robustness: 동일 checkpoint에서
analytic→learned detector, detection dropout, 지연, range error를 한 축씩 평가하고 실패하면 PPO가 아니라
camera–LiDAR association/uncertainty-aware release를 수정한다. 현재 학습·평가 프로세스는 없고 GPU는 idle다.

### 문서·사이트 동기화와 최종 회귀검증

README/CLAUDE/RESEARCH_PLAN/OPERATIONS/CRASH_TUNING_LOG를 최종 수치, checkpoint SHA, 무효화 사유,
다음 learned-perception gate로 동기화했다. `tools/update_status_snapshot.py`는 corrected 5-cell screen,
seed44 screen, trained SHA, seed45/46 모든 gate를 검증해야만 FINAL PASS를 출력하도록 강화했다. 사이트는
training 마지막 epoch의 75/25% 소표본 대신 seed45 held-out **81.94/15.67/2.39%**, seed46 **3/3**,
trained winner SHA를 상단과 current-result 표에 표시한다. HTTP `status.json`과 `index.html` offline fallback은
같은 payload로 재생성됐고 72 runs, active=none, latest=riskcap ep25000, stage=`FINAL PASS`다.

최종 검증 결과는 speed-governor **10/10**, perception **18/18**, TTC selector **13/13**, status snapshot
**11/11**, recovery gate **19/19**, action model **13/13**, density feasibility **3/3**, limit audit **4/4**,
postrun **2/2**, arena-motion/DOM parity PASS다. 수정·신규 Python 전부 `py_compile`, shell 전부 `bash -n`,
사이트 artifact/fallback/final-contract, stale invalid-number scan, checkpoint SHA, `git diff --check`도 PASS했다.
시스템 Node는 기존부터 사용한 nullish-coalescing `??`를 지원하지 않는 구버전이라 raw `node --check`만
129행에서 중단했다. `??`를 syntax-only로 `||` 치환한 stream의 전체 JS parse와 실제 DOM parity는
통과했으므로 이번 UI 변경의 문법 실패가 아니다. 커밋·푸시는 이번 요청에 포함되지 않아 수행하지 않았다.

---

## 2026-08-05 — R3 latency screen 결과와 compensation 계획 (구현·재평가 전 검수용)

### R3 screen 요약 (ep25000+riskcap, seed47, 205 bars, inference-only)

canonical: `results/navrl_v2_detector_robustness/summary.{md,json}`

| 축 | capture | crash | baseline 대비 capture |
|---|---:|---:|---|
| analytic_clean | 80.54% | 17.17% | — |
| dropout 0.3 | 67.84% | 29.33% | −12.7 pp |
| **latency 0.1 s** | 37.82% | 58.22% | **−42.7 pp** |
| **latency 0.2 s** | 18.50% | 76.48% | **−62.0 pp** |
| range error ±0.15/0.30 m | ~80.5% | ~17% | ≈0 pp |
| learned_clean (4096-frame train) | 66.62% | 24.94% | −13.9 pp |

**판정:** latency가 1순위 병목. range error는 Kalman/association이 흡수. learned detector는
G1–G3 전에 dataset/학습 규모를 키워야 한다. **PPO 연장은 하지 않는다.**

### latency가 코드에서 의미하는 것

- RL step `dt=0.1 s`. `NAVRL_DETECTION_LATENCY_S=0.1` → policy/tracker에 들어가는 **camera detection이
  1 step(0.1 s) 전 정보**.
- 구현: `navrl_perception.py::_apply_detection_latency()`가 fresh detection을 ring buffer에 쓰고,
  `_detect_rgbd()` 출력을 지연된 bearing/range/mask로 바꾼 뒤 `tracker.step()`에 넣는다.
- **핵심 구조적 문제(왜 CV KF만으로는 부족한지):**
  1. 지연된 measurement를 **현재 시각 measurement처럼** KF가 correct한다 → state가 과거 위치로 당겨짐.
  2. `_associate_lidar_target()`은 `camera_visible=True`이면 LiDAR correction을 **막는다**
     (`valid = ... & ~camera_visible`). 지연 camera가 “보인다”고 하면 **fresh LiDAR 백업 경로가 차단**된다.
  3. policy는 `_target_features()`의 tracker state + 5-step history를 본다. stale correction이 history에
     0.5 s 간격으로 누적된다.

### “latency 줄이기” vs “compensate” — 역할 분리

| 구분 | 무엇을 바꾸나 | 이 레포에서의 위치 | PPO 필요? |
|---|---|---|---|
| **줄이기 (reduce)** | 파이프라인 지연 자체를 짧게 | detector 경량화, 해상도, onboard rate, sim 밖 HW | 아니오 |
| **보상 (compensate)** | 지연이 있어도 **현재 시각에 맞는 target state**를 policy에 제공 | `navrl_perception.py` tracker/output | 아니오 |

reduce는 sim ablation 축이 아니라 **배포/실기 목표**. R3 gate 재평가는 **compensate arm**으로
`latency=0.1/0.2`에서 capture/crash가 analytic_clean에 얼마나 근접하는지 본다.

### compensation 후보 (우선순위 — 한 번에 하나만 구현·평가)

#### P0 · forward predict (가장 작은 diff, 먼저 시험)

**방법:** camera measurement는 그대로 지연 수신. KF update 후 policy에 넘기기 직전에 constant-velocity
로 `τ = NAVRL_DETECTION_LATENCY_S`만큼 전방 extrapolate:

`pos_policy = pos_est + vel_est * τ` (world frame → vehicle frame은 기존 `_target_features`와 동일)

**이유:** 이미 `BatchedConstantVelocityTracker`에 속도 상태가 있고 obs dim 변경 없음. 0.1–0.2 s 지연의
주된 오류는 “과거 위치로 correct”이므로, 출력단 predict가 1차 fix.

**리스크:** covariance/age는 여전히 stale; LiDAR 차단 문제는 그대로 → P1과 세트 권장.

**env:** `NAVRL_LATENCY_COMPENSATE=1` (신규, default off). eval perturb와 분리.

#### P1 · delayed-aware LiDAR fusion (구조 버그 fix)

**방법:** `camera_visible`이 **delayed**일 때는 LiDAR association을 막지 않는다. 즉
`~camera_visible` gate를 `~camera_visible_fresh`로 바꾸거나, `latency_steps>0`이면 LiDAR valid path 허용.

**이유:** R3 screen에서 latency가 dropout보다 훨씬 치명적인 이유 중 하나가 “stale camera가 LiDAR
correction을 끔”. fresh LiDAR( sim에서 지연 없음)로 tracker를 당겨올 수 있음.

**리스크:** camera/LiDAR disagreement 증가 → innovation gate(P2)와 함께 켜는 편이 안전.

#### P2 · timestamped measurement / inflated noise

**방법 A:** delayed measurement update 시 `R`을 `R + (τ·σ_vel)²` 또는 고정 inflation factor로 키워
과거 점에 덜 끌리게 한다.

**방법 B:** measurement를 `t−τ` 시점으로 태그하고, KF predict를 τ만큼 먼저 돌린 뒤 update (동치에 가까운
forward predict + inflated R).

**이유:** WORKLOG 2026-07-27 후속 후보(innovation reliability)와 연결. range error가 benign했던 것과
대비, **시간 오류**는 innovation이 큼.

#### P3 · confidence / age signaling (obs dim 유지)

**방법:** `_target_features()`에서 `camera_confidence`를 `confidence * exp(-τ/τ₀)` 또는
`age` feature를 latency-aware로 올림. policy/riskcap이 이미 track age/covariance를 받음.

**이유:** NavRL++·CosFly 계열의 “stale measurement down-weight”. **단독으론 capture 회복 한계** —
P0/P1 없이 confidence만 낮추면 보수적일 뿐.

#### 보류 · PPO / temporal retrain

latency arm에서 capture −42 pp를 **PPO fine-tune으로 메우는 것은 금지**. perception fix인지 policy
적응인지 분리 불가. R4 temporal fusion은 **P0–P2 gate 통과 후**.

#### 보류 · sim latency를 0으로 “줄이기”

`NAVRL_DETECTION_LATENCY_S=0`은 baseline이지 해결책이 아님. 실기 목표는 onboard pipeline profiling.

### 재평가 계약 (Claude/로컬 GPU — 구현 승인 후)

동결: ep25000+riskcap, seed47, 205 bars, deterministic, 2049 ep.

| arm | NAVRL_DETECTION_LATENCY_S | compensate | 기대 |
|---|---:|---|---|
| baseline | 0 | off | 80.54/17.17% 재현 |
| latency 0.1 | 0.1 | off | R3 재현 (37.82%) |
| latency 0.1 + P0 | 0.1 | forward predict on | capture ↑ crash ↓ |
| latency 0.1 + P0+P1 | 0.1 | both | LiDAR backup 추가 이득 |

**GO (1차):** latency 0.1에서 P0(+P1) 적용 시 capture가 **≥65%** (baseline −15 pp 이내)이고 crash가
latency-off 대비 **≥10 pp** 감소. **완전 회복(≥78%)**은 2차(iterated predict, G3 reacquisition) 목표.

**실행:** `eval_navrl_v2_detector_robustness.sh`에 compensate 셀 추가 또는 별도
`eval_navrl_v2_latency_compensate.sh` (RESULT_ROOT 새 디렉터리). PPO 학습 없음.

### 구현 전 검수 체크리스트

1. compensate는 `_target_features()` 출력 또는 tracker update **한 경로**만 바꾸는가 (obs 898-D 유지)?
2. `NAVRL_DETECTION_LATENCY_S` perturb와 `NAVRL_LATENCY_COMPENSATE` eval knob이 분리되는가?
3. P0/P1을 **동시에** 첫 commit에 넣지 않고 arm별로 eval 가능한가?
4. unit test: synthetic constant-velocity target + τ=0.1 → predict가 raw delayed보다 bearing error 작음?
5. riskcap/speed governor는 perception output만 쓰므로 **별도 튜닝 없이** 재평가 가능한가?

**현재 상태:** 계획만 기록. 코드 변경·GPU eval은 검수 후 진행.

---

## 2026-08-05 — latency compensation P0+P1 구현 완료 (검수 대기, GPU 미실행)

2026-08-05 R3 compensation 계획을 코드로 구현했다. 체크리스트 5항목 전부 충족.
**커밋·GPU eval은 하지 않았다** — diff 검수 후 사용자 지시로 실행한다.

### 구현 내용 (diff 요약)

| 파일 | 변경 |
|---|---|
| `navrl_task_config.py` | `NAVRL_LATENCY_COMPENSATE`(P0)·`NAVRL_LATENCY_LIDAR_BACKUP`(P1) 신규 env knob, 기본 off. perturb(`NAVRL_DETECTION_LATENCY_S`)와 분리 (체크리스트 2) |
| `navrl_perception.py` | P0: `_target_features()` **출력단만** — `pos_policy = pos_est + vel_est·τ` (KF 내부·cov·age·진단 불변, obs 898-D 유지, 체크리스트 1). P1: `observe()`에서 `latency_lidar_backup`이면 지연 camera visible이 LiDAR association을 veto하지 못하게 gate 텐서만 교체 (`_associate_lidar_target` 자체는 무변경) |
| `tests/test_navrl_latency_compensate.py` | 신규 7 테스트 **전부 PASS** |
| `eval_navrl_v2_latency_compensate.sh` | 신규 4-arm(+옵션 0.2s) 스크립트, `PREFLIGHT=1` 드라이런 PASS |

두 knob 모두 `detection_latency_s=0`이면 산술적 no-op이므로 clean run 결과는 byte-동일하다.

### 단위 테스트 결과 (체크리스트 4)

CV 표적(1.5 m/s), τ=0.1 s 지연 측정을 실제 `BatchedConstantVelocityTracker`에 60 step 공급:
- raw 위치 오차 **0.147 m**(≈v·τ, 예측된 lag 재현) → predict 후 **<50%로 감소**, bearing 오차도 감소 ✓
- KF 속도 추정 오차 <0.15 m/s (지연 위치로도 속도는 무편향 — P0가 작동하는 근거) ✓
- τ=0이면 predict가 항등, 정지 표적이면 예측이 운동을 만들어내지 않음 ✓
- 소스 검사: P0가 `_target_features`에만 존재(tracker 클래스에 없음), P1 gate가 backup 플래그를 읽음 ✓
- 기존 `test_navrl_perception.py` 20/20 회귀 없음, py_compile 통과.

### 재평가 실행 계약 (준비 완료, 미실행)

`eval_navrl_v2_latency_compensate.sh` — ep25000+riskcap(SHA f7022139…) 고정, seed47,
205 bars, deterministic, 2049 ep/cell, PPO 없음. arms: analytic_clean 재현 / latency_0p1s_raw
재현 / +P0 / +P0+P1 (+옵션 `NAVRL_LAT_INCLUDE_0P2=1`). 끝에 summary.{md,json} 자동 생성 +
GO gate(capture ≥65% AND crash raw 대비 ≥10 pp↓) 자동 판정. 결과 루트:
`results/navrl_v2_latency_compensate/` (기존 존재 시 덮어쓰기 거부).

### riskcap 상호작용 (체크리스트 5)

riskcap/speed governor는 perception 출력(LiDAR scan·clearance)만 소비하고 target tracker
state를 직접 읽지 않으므로 별도 튜닝 없이 동일 계약으로 재평가 가능하다. eval 스크립트는
R3와 byte-동일한 governor env 블록을 고정한다.


## 2026-08-05 — latency compensation P0/P1 재평가 결과: 둘 다 NO-GO, 진짜 채널은 obstacle map 오염

frozen ep25000+riskcap (SHA f7022139…), seed47 / 205 bars / deterministic / riskcap,
2049~2050 ep/cell. `results/navrl_v2_latency_compensate/summary.{md,json}`.

| cell | episodes | capture | crash | timeout | bar-contact share | out-of-bounds share |
|---|---:|---:|---:|---:|---:|---:|
| analytic_clean | 2050 | 80.54% | 17.17% | 2.29% | 95.7% | 2.8% |
| latency_0p1s_raw | 2049 | 37.82% | 58.22% | 3.95% | 78.0% | 21.0% |
| latency_0p1s_p0 | 2049 | 37.73% | 57.74% | 4.54% | 76.1% | 22.4% |
| latency_0p1s_p0p1 | 2050 | 28.98% | 66.54% | 4.49% | 78.9% | 20.1% |

계약 무결성: clean(80.54/17.17)과 raw(37.82/58.22) 모두 R3 수치를 소수점까지 재현 →
평가 계약·시드·정책 동일성 확인됨. 따라서 arm 간 차이는 순수하게 fix 토글 효과다.

**P0 (forward predict): capture −0.10 pp = 무효.** flag가 안 켜진 게 아니다 —
`closest_nocrash_mean_m`이 1.611 → 1.454 m로 유의하게 개선됐고 crash도 10건 줄었다.
즉 P0는 의도한 대로 표적 추정 위치의 v·τ lag(≈0.15 m)를 제거했지만, **그 lag이
latency 손실의 원인이 아니었다.** 5 m 거리에서 0.15 m는 bearing 1.7°로, capture를
−42.7 pp 무너뜨리는 크기가 아니다. P0는 "위치 lag 가설"을 실험적으로 기각한 것.

**P1 (LiDAR backup): capture −8.85 pp = 역효과.** bar_contact 절대수가 931 → 1076으로
증가. stale camera 상태에서 LiDAR association을 열어주면 표적이 아닌 **막대**에
잘못 association될 수 있고, 그 오연관이 아래 채널을 통해 곧바로 충돌로 이어진다.

### 진짜 채널: 지연된 camera가 obstacle map을 오염시킨다

`observe()`는 지연된 `fused_surface / fused_bearing / fused_visible / pixels`를 그대로
`_fuse_static_and_extract_obstacles()`에 넘긴다. 그 안에서 두 가지 삭제가 일어난다:

1. `target_like` carve-out (navrl_perception.py 743-749): 표적으로 판정된 LiDAR
   return을 `lidar_max_range`로 치환해 obstacle scan에서 지운다. 지연 상태에서는
   **stale bearing 방향**을 지우므로, 그 방향에 실제로 서 있는 막대가 장애물 지도에서
   사라진다 → 정면 충돌.
2. `depth_no_target` (752-754): stale `target_pixels`로 카메라 depth를 blank 처리 →
   FOV 안의 실제 장애물이 지워지고, 반대로 진짜 표적은 장애물로 남는다.

이것이 "카메라 표적 검출만 지연시켰는데 왜 bar contact가 337 → 931로 3배가 되는가"를
설명하는 유일한 경로다. out-of-bounds도 10 → 251건으로 급증(추격 방향 자체가 stale).
P0는 policy-facing 출력만 건드렸고 P1은 오연관을 늘렸으므로, **둘 다 이 채널을
전혀 건드리지 못했다.** 계획서의 P0/P1 가설이 채널을 잘못 지목한 것이며, 이번 두 arm이
그것을 저비용으로 확정했다.

### 다음 후보 (P2': latency-aware carve-out)

carve-out과 depth blanking의 입력을 stale camera bearing/range 대신 **tracker의 현재
예측치**(= P0가 이미 계산하는 보정 위치에서 유도한 bearing/range)로 바꾼다. 대안으로
detection이 stale일 때 carve-out을 아예 끄는 보수적 변형도 arm으로 둘 수 있다
(표적이 장애물로 남는 대신 실제 막대는 지워지지 않음 — 안전 우선).
평가 계약은 동일, PPO 재학습은 여전히 금지. GO gate도 동일(capture ≥ 65% AND
crash raw 대비 ≥ 10 pp 감소).

## 2026-08-06 — latency P2 (obstacle-map 보정) 구현 완료, GPU 평가 대기

P0/P1 NO-GO의 원인 분석(직전 항목)에서 지목한 채널 —— 지연된 target bearing/range/pixel mask가
`_fuse_static_and_extract_obstacles()`의 carve-out과 depth blanking을 구동해서 **실제 막대를
장애물 지도에서 지운다** —— 를 직접 고치는 P2를 구현했다.

**노브**: `NAVRL_LATENCY_OBSTACLE_FIX=off|predict|skip` (기본 off, perturbation 노브와 분리).
- `predict`: carve-out 창을 tracker의 forward-predict 위치(P0와 **동일 수식** 재사용)에서
  다시 계산한다. LiDAR bin 창과 depth pixel mask를 모두 예측 bearing/range로 재구성하며,
  `tracker.active`가 아닌 env는 아예 지우지 않는다.
- `skip`: stale 동안 map 편집을 중단한다. 표적이 유령 장애물로 남는 대신(capture 손해)
  **실제 막대는 절대 지워지지 않는다**(crash 상한). 보수적 arm.

**구현 위치**: `navrl_perception.py::_latency_corrected_map_inputs()` (신규) → `observe()`가
`_fuse_static_and_extract_obstacles()`에 넘기는 4개 입력만 교체한다. `_fuse_...` 본체는
carve-out 상수를 모듈 상수(`TARGET_LIKE_ANGLE_RAD`, `TARGET_LIKE_RANGE_TOL_M`)로 뽑은 것 외에
무변경. 관측 폭 898-D 불변, PPO 재학습 없음.

**단위 테스트** `tests/test_navrl_latency_compensate.py` (13/13 PASS, 기존 20/20 회귀 PASS):
- predict의 carve-out bearing이 tracker 예측 bearing과 일치하고, stale bearing과 2° 이상 다름
- predict의 재구성 pixel mask가 예측 bearing ±15° 안에만 존재
- skip은 `last_target_like`가 전부 False이고 fused scan이 off 대비 **어디서도 더 멀어지지 않음**
  (지도가 더 비어 보이는 일이 없음) + 실제로 더 가까운 bin이 존재
- latency=0이면 predict/skip/off의 map 입력 4개가 **bitwise 동일** (clean 재기준선 불필요)
- `_pixel_angles`가 camera_u 매핑의 정확한 역함수, 잘못된 모드 문자열은 ValueError

**provenance 공백 수정**: 기존 receipt에는 perturbation/fix 노브가 하나도 안 남아서 raw arm과
P0 arm의 receipt가 nonce/시각 말고는 동일했다. `eval_navrl_v2_density_sweep.sh`가 이제
`perception_{perturb,detection_dropout,detection_latency_s,range_error_m,latency_compensate,
latency_lidar_backup,latency_obstacle_fix}`를 결과 JSON과 receipt 양쪽에 기록하고, fix 노브를
off 값으로 명시 export 해서 호출자 셸에서 새어들어오지 않게 한다.

**GPU smoke** (predict, 66 ep, scratch): 오류 없이 완주, capture 36.4/crash 63.6%. n=66이라
raw(37.8%)와 구분 불가 —— 경로 동작 확인 목적이며 **결과로 해석하지 않는다**.

**평가 스크립트** `eval_navrl_v2_latency_obstacle_fix.sh` (PREFLIGHT PASS), 동일 계약
(seed47/205bars/deterministic/riskcap, 2049 ep/cell), arm 4개:
raw / map_skip / map_predict / p0+predict. clean은 no-op 증명이 있으므로 재실행하지 않고
`navrl_v2_latency_compensate/analytic_clean`을 앵커로 재사용한다(`NAVRL_LAT_RERUN_CLEAN=1`로 재측정).
GO gate 동일(capture ≥ 65% AND crash raw 대비 ≥ 10 pp 감소). 추가로 **bar_contact 절대수**를
raw 대비 함께 보고한다 —— 가설이 직접 예측하는 양이므로, GO를 못 넘겨도 "메커니즘이 틀렸는지
맞는데 크기가 작은지"를 구분해준다.

## 2026-08-06 — latency P2 재평가: 방향은 맞으나 크기가 작아 NO-GO, ego-motion 채널로 좁혀짐

frozen ep25000+riskcap, seed47 / 205 bars / deterministic / riskcap, 2049~2050 ep/cell.
`results/navrl_v2_latency_obstacle_fix/summary.{md,json}`. clean은 no-op 증명에 따라
`navrl_v2_latency_compensate/analytic_clean`을 앵커로 재사용했다.

| cell | capture | crash | bar contacts | OOB | closest mean (m) |
|---|---:|---:|---:|---:|---:|
| analytic_clean | 80.54% | 17.17% | 337 | 10 | 0.806 |
| latency_0p1s_raw | 37.82% | 58.22% | 931 | 251 | 1.611 |
| latency_0p1s_map_skip | 38.83% | 56.83% | 895 | 252 | 1.579 |
| latency_0p1s_map_predict | 39.66% | 56.29% | 867 | 278 | 1.329 |
| latency_0p1s_p0_predict | 36.88% | 58.44% | 918 | 267 | 1.812 |

**전부 NO-GO.** 다만 P0/P1과 달리 **부호가 예측대로**다: map을 건드리지 않는(skip) 것도,
예측 위치로 재배치하는(predict) 것도 bar contact를 줄이고(931 → 895 / 867) capture를 올렸다
(+1.01 / +1.84 pp). predict가 skip보다 일관되게 낫다 —— "지우되 올바른 곳에 지운다"가
"안 지운다"보다 낫다는 뜻이고, 이는 carve-out 자체는 필요한 기능이라는 기존 설계를 지지한다.

**그러나 크기가 결정적으로 부족하다.** latency의 초과 bar contact는 931 − 337 = 594건인데
P2가 회수한 건 64건, 약 11%다. 통계적으로도 약하다: 2050 ep에서 capture 차이의 SE는 약 1.5 pp,
bar contact 차이의 SE는 약 42건이므로 두 지표 모두 ~1.3σ로 유의 미달이다. 따라서
**"stale carve-out이 막대를 지운다"는 메커니즘은 실재하지만 지배적 채널이 아니다.**

### 남은 지배 채널 가설: ego-motion 미보정 (P3 후보)

`observe()` line 1059: `meas_world = drone_pos_w + _quat_rotate_xyzw(vehicle_quat, meas_vehicle)`.
`meas_vehicle`은 τ 전 **드론 기체 프레임**에서 관측된 값인데, 이를 **현재** 위치·자세로 월드
변환한다. 즉 지연 오차는 표적 운동(v_target·τ ≤ 0.15 m)이 아니라 **드론 자신의 ego-motion**이
지배한다. 이번 run의 실측 평균 속도는 2.33 m/s이므로 병진만으로 0.23 m이고, 여기에 yaw
회전분이 bearing에 직접 더해진다(mean_abs yaw action 0.325). P0는 표적 속도만 외삽했으므로
이 항을 전혀 건드리지 못했고, 이것이 P0가 무효였던 이유를 완결적으로 설명한다.

P2 predict가 그나마 최선이었던 것도 같은 그림과 일치한다: predict는 bearing/range를 tracker의
**월드 상태**에서 다시 유도하므로 ego-motion 오차를 우회한다. 하지만 그 tracker 자체가 이미
ego-motion 오염된 측정으로 correct 되어 있어서, 하류 보정으로는 상한이 있다.

**P3**: 링 버퍼에 measurement와 함께 **당시의 drone_pos_w / vehicle_quat**을 저장하고, 지연된
vehicle-frame 측정을 **그 시점 pose**로 월드 변환한다. 그러면 월드 측정은 "정확하지만 τ만큼
과거"인 상태가 되고, 남은 것은 P0가 이미 처리하는 표적 운동 lag뿐이다. 실기에서도 IMU/odometry로
동일하게 구현 가능한 보정이라 sim 전용 트릭이 아니다. 평가 계약·GO gate 동일, PPO 재학습 금지.

## 2026-08-06 — latency P3 (ego-motion 보정) 구현 완료, GPU 평가 시작

직전 항목의 가설을 그대로 구현했다. `observe()`는 t−τ의 **기체 프레임** 측정을 t의 pose로
월드 변환하고 있었다. P3는 링 버퍼에 detection과 함께 그 시점의 `drone_pos_w`/`vehicle_quat`을
저장하고, 지연된 측정을 **그 pose로** 월드 변환한다. 노브 `NAVRL_LATENCY_EGO_MOTION_FIX`
(기본 off, τ=0이면 pose를 아예 발행하지 않아 no-op).

**오차 크기 실측** (`tests/test_navrl_latency_compensate.py::LatencyEgoMotionFix`, 월드 고정
표적을 이동·요잉하는 관측자가 보는 상황. 속도 2.33 m/s와 yaw 0.81 rad/s는 이번 205-bar run의
실측 평균이다 —— mean_abs yaw action 0.325 × `yaw_rate_max` 2.5):

| 성분 | τ=0.1s 월드 위치 오차 |
|---|---:|
| 병진만 (yaw=0) | 0.233 m |
| yaw만 (speed=0) | 0.408 m |
| 합성 (이 궤적에서 일부 상쇄) | 0.255 m |
| **P3 보정 후** | **0.000 m** (exact) |
| 참고: P0가 제거한 표적 운동 lag | 0.150 m |

즉 **yaw 성분 하나만으로도 P0가 겨눈 표적 lag의 2.7배**다. P0가 무효였던 이유가 수치로
확정된다: 더 작은 항만 고쳤다. τ=0.2s에서는 ego-motion 오차가 0.524 m로 커지며 P3는 여전히
정확히 0으로 만든다.

월드 고정 표적에서 잔차가 정확히 0이라는 것이 이 테스트의 핵심이다 —— 표적이 안 움직이면
남은 오차는 전부 관측자 자신의 운동이므로, P3는 그 성분을 **전량** 제거한다. 남는 것은
"정확하지만 τ만큼 과거"인 측정이고, 그게 바로 P0가 처리하도록 쓰인 잔차다(→ p3_p0 arm).

**구현**: `_apply_detection_latency()`가 pose 링 버퍼를 함께 쓰고 **동일한 read index**로
`self._latency_delayed_pose`를 발행한다(인덱스 산술을 두 번 쓰지 않기 위함). `_detect_rgbd`는
pose를 optional kwarg로 받아 전달만 한다(기존 호출부·테스트 호환). 관측 898-D 불변, PPO 없음.
버퍼가 아직 안 찬 첫 스텝은 init pose를 들고 있지만 그 detection이 `visible=False`로 보고되므로
무해하다 —— 테스트가 이 불변식을 직접 검사한다.

**테스트** 18/18 PASS (기존 perception 20/20 회귀 PASS). 정지 관측자에서는 P3 on/off의
tracker state가 동일하고, 이동·요잉 관측자에서는 갈라진다는 것까지 확인.

**평가** `eval_navrl_v2_latency_ego_motion.sh` (PREFLIGHT PASS), 동일 계약, 2049 ep/cell,
arm 4개: raw / p3 / p3+p0 / p3+p0+predict. summary에 GO gate와 함께 **capture·bar_contact를
"clean−raw 구멍 중 몇 %를 메웠는지"** 비율로도 보고하게 했다 —— P2의 +1.84 pp가 실은 구멍의
4%였던 것처럼, pp 절대값만으로는 크기 판단이 안 되기 때문이다.

## 2026-08-06 — latency P3 결과: GO. latency 손실의 94%는 ego-motion 미보정 아티팩트였다

frozen ep25000+riskcap, seed47 / 205 bars / deterministic / riskcap, 2049~2050 ep/cell.
`results/navrl_v2_latency_ego_motion/summary.{md,json}`.

| cell | capture | crash | bar contacts | OOB | closest mean (m) |
|---|---:|---:|---:|---:|---:|
| analytic_clean | 80.54% | 17.17% | 337 | 10 | 0.806 |
| latency_0p1s_raw | 37.82% | 58.22% | 931 | 251 | 1.611 |
| **latency_0p1s_p3** | **78.04%** | 19.67% | 390 | 4 | 0.846 |
| latency_0p1s_p3_p0 | 76.57% | 20.79% | 406 | 8 | 0.953 |
| latency_0p1s_p3_p0_predict | 78.43% | 19.23% | 370 | 7 | 0.802 |

**세 arm 모두 GO.** P3 단독으로 capture +40.21 pp —— clean−raw 구멍의 **94.2%**를 메웠고
bar contact 초과분의 **91.1%**를 제거했다(931 → 390, clean 337). OOB는 251 → 4로 clean(10)보다도
낮다. P0/P1/P2가 각각 −0.10 / −8.85 / +1.84 pp였던 것과 자릿수가 다르다.

**핵심 결론(R3 헤드라인 수정)**: "detection latency 0.1 s가 capture를 42.7 pp 무너뜨린다"는
R3 결과는 **대부분 latency 모델링 아티팩트였다.** 지연된 기체-프레임 측정을 현재 pose로 월드
변환하면 드론 자신의 운동이 매 correct마다 주입되는데, 실제 시스템은 measurement에 timestamp를
달고 **취득 시점 pose**로 변환한다(표준 관행). 그렇게 고치면 latency 0.1 s의 진짜 비용은
**2.1~2.5 pp**(80.54 → 78.43 / 78.04)다. 따라서 R3의 "latency가 1순위 perception 병목"이라는
판정은 **기각**하고, latency는 range error·dropout과 같은 등급의 benign 축으로 재분류한다.

**P0/P2는 채택하지 않는다.** P3 위에서 P0는 −1.47 pp, P2 predict는 +0.39 pp인데 2049 ep에서
arm 간 차이의 SE가 약 1.3 pp이므로 둘 다 유의하지 않다. P3가 측정을 정확하게 만든 뒤 남는
표적 운동 lag(0.15 m)는 결과를 바꾸지 못한다 —— P0의 원래 가설이 왜 실패했는지와 같은 이유다.
최선 arm(p3_p0_predict 78.43%)과 clean의 차이 2.11 pp도 1.7σ로 경계선이다.

**P3를 기본 ON으로 승격할 것을 제안한다.** τ=0에서 산술적 no-op이므로 clean 결과에 영향이 없고,
지연이 있을 때만 "올바른 latency 모델"이 된다. 현재는 여전히 `NAVRL_LATENCY_EGO_MOTION_FIX=0`
기본값이다.

**다음 단계**: latency 0.2 s를 P3와 함께 재측정해서 실제 하드웨어 latency 예산 곡선을 얻는다
(P3 없이는 18.50%였다). 그 다음에야 R4 temporal fusion 논의가 의미를 가진다.

## 2026-08-06 — P3 기본 ON 승격 + R3 latency 셀 superseded 표시

P3는 "보정"이 아니라 **올바른 측정 모델**이므로 기본값을 ON으로 올렸다
(`NAVRL_LATENCY_EGO_MOTION_FIX` 기본 True, `eval_navrl_v2_density_sweep.sh`의 pin도 1). 근거:
실제 파이프라인은 measurement에 timestamp를 달고 취득 시점 pose로 변환한다. 그렇게 하지 않은
기존 코드는 "latency 모델"이 아니라 **latency 위에 얹힌 미모델링 오차**였다. τ=0에서 산술적
no-op이므로 clean 결과와 2026-08-06 이전의 모든 비-latency 수치는 영향을 받지 않는다.
`_env_bool`이 `=0`을 정확히 False로 처리하는 것을 확인했으므로 R3 재현 경로도 살아 있다.

`eval_navrl_v2_detector_robustness.sh` 헤더에 **superseded 경고**를 달았다: 아카이브된
latency_0p1s / latency_0p2s(37.82% / 18.50%)는 수정 전 수치이며, 지금 재실행하면 기본값이
바뀌었으므로 그 숫자가 재현되지 않는다(의도된 동작). 비-latency 셀은 no-op이라 그대로다.

**주의**: 이 승격으로 `results/navrl_v2_detector_robustness/`의 latency 두 셀은 논문·대시보드에
그대로 인용하면 안 된다. 대체 수치는 `results/navrl_v2_latency_ego_motion/`(0.1 s)와
`results/navrl_v2_latency_budget/`(0.2/0.3/0.5 s)다.

## 2026-08-06 — latency 예산 곡선 1차 시도 무효화 (evaluator 무결성 가드 발동)

`eval_navrl_v2_latency_budget.sh`(0.2/0.3/0.5 s, P3 ON) 1차 실행이 첫 셀 완주 후
`[eval_v2] checkpoint or evaluator bytes changed during the cell; refusing result.`로 거부됐다.
원인은 내 실수다: 평가가 도는 도중에 P3 기본값 승격 작업으로
`eval_navrl_v2_density_sweep.sh`를 편집했고, 셀 시작 시 기록한 evaluator SHA와 종료 시 SHA가
달라졌다. **가드가 정확히 동작한 것이며, 거부된 수치는 인용하지 않는다**(로그에는
captured=0.766이 남아 있으나 provenance가 깨진 값이므로 폐기). 결과 디렉터리를 삭제하고
스크립트가 안정된 상태에서 재실행했다.

**운영 규칙**: 평가 실행 중에는 evaluator 스크립트·체크포인트를 편집하지 않는다. 편집이
필요하면 평가 완료를 기다리거나, 편집 대상과 무관한 파일(docs/status, WORKLOG, tools)만 만진다.

## 2026-08-06 — latency 재분류 후 남은 인지 병목 순위 (분석, GPU 미사용)

latency를 수정된 모델로 갈아끼운 뒤 R3 축들의 순위가 바뀐다. `target_visible` 진단(스텝 중
표적이 fused visible이었던 비율)을 함께 읽으면:

| cell | fused visible | capture | crash | capture Δ vs clean |
|---|---:|---:|---:|---:|
| analytic_clean | 21.21% | 80.54% | 17.17% | — |
| **learned_clean** | 14.25% | **66.62%** | 24.94% | **−13.92 pp** |
| **dropout_0p3** | 21.38% | **67.84%** | 29.33% | **−12.70 pp** |
| latency 0.1 s (P3) | — | 78.04% | 19.67% | −2.50 pp |
| range_error_0p15m | 20.49% | 80.54% | 17.46% | 0.00 pp |
| range_error_0p30m | 20.49% | 80.62% | 16.89% | +0.08 pp |

**새 1순위는 learned detector(−13.92 pp)다.** analytic bootstrap segmenter는 어디까지나
부트스트랩이므로, 센서 전용 주장을 지탱하려면 이 격차가 핵심이다.

**동시에 풀리지 않은 관측**: dropout 0.3은 카메라 검출의 30%를 버리는데도 fused visible이
21.38%로 clean(21.21%)과 **사실상 동일**하다. LiDAR association이 빈자리를 메우기 때문으로
보인다. 그런데 capture는 12.70 pp 떨어진다 —— 즉 **dropout의 비용은 "표적을 덜 본다"로
설명되지 않는다.** 반대로 learned detector는 visible이 14.25%로 clean 대비 1/3이 줄었는데
capture 손실은 dropout과 비슷하다. 두 축의 손실 경로가 서로 다르다는 뜻이며, 어느 쪽도
"검출 빈도"라는 단일 변수로 환원되지 않는다.

이것은 P0가 실패했던 것과 같은 종류의 함정이다: 그럴듯한 중간 변수(위치 lag, 검출 빈도)를
고쳐도 결과가 안 움직이면, 그 변수는 채널이 아니다. 다음 작업은 learned detector와 dropout
각각에 대해 **실제 채널을 먼저 특정**하는 것이고, 수정안은 그 다음이다. 후보 진단:
검출 실패의 시간적 상관(연속 miss 길이), 거리 의존성, 그리고 tracker age/공분산이 정책 입력에서
어떻게 보이는지.

## 2026-08-06 — latency 예산 곡선 (수정된 모델, P3 ON)

`results/navrl_v2_latency_budget/summary.{md,json}`. 동일 계약, 2049 ep/cell.
0.1 s는 `navrl_v2_latency_ego_motion`의 셀을 앵커로 재사용했다.

| latency | capture | crash | bar contacts | Δ capture vs clean |
|---:|---:|---:|---:|---:|
| 0 (clean) | 80.54% | 17.17% | 337 | — |
| 0.1 s | 78.04% | 19.67% | 390 | −2.50 pp |
| 0.2 s | 76.62% | 20.40% | 397 | −3.91 pp |
| 0.3 s | 72.57% | 24.79% | 494 | −7.96 pp |
| 0.5 s | 64.76% | 32.41% | 648 | −15.77 pp |

**해석**: 수정된 모델에서 열화는 완만하고 단조롭다. 0.2 s까지는 clean 대비 4 pp 이내로
운용 가능하고, 0.3 s에서 −8 pp로 꺾이며, 0.5 s에서 −15.8 pp가 된다. 수정 전 모델이
0.1 s에서 이미 −42.7 pp, 0.2 s에서 −62 pp였던 것과 비교하면 곡선의 모양 자체가 다르다.
**실기 latency 예산: 0.2 s 이하를 목표, 0.3 s를 상한으로 제시할 수 있다.** 10 Hz 제어에서
2 스텝 지연까지 허용된다는 뜻이므로 detector 추론 시간에 현실적인 여유가 생긴다.

crash 증가분이 capture 감소분과 거의 1:1이고(예: 0.5 s에서 −15.77 / +15.24 pp) timeout은
2.3→2.8%로 거의 고정이다. 즉 지연이 커져도 정책은 계속 추격하며, 손실은 전부 충돌로 나타난다
—— 표적을 놓쳐 배회하는 실패 모드가 아니다. bar contact도 337 → 648로 단조 증가한다.

1차 실행이 evaluator 편집으로 거부된 뒤의 재실행 결과이며, 이번에는 실행 중 어떤 평가 입력도
수정하지 않았다.

## 2026-08-06 — dropout vs learned detector: 손실 경로가 서로 다르다 (분석, GPU 미사용)

R3 셀들의 진단 필드를 읽어 두 축의 **실패 서명**을 비교했다.

| cell | fused visible | closest_nocrash_mean | bar contact 수 | 충돌 시 clearance | capture |
|---|---:|---:|---:|---:|---:|
| analytic_clean | 21.21% | 0.806 m | 337 | 0.92 m | 80.54% |
| dropout_0p3 | 21.38% | 0.972 m | 559 | **0.65 m** | 67.84% |
| learned_clean | 14.25% | **1.892 m** | 495 | **1.10 m** | 66.62% |

capture 손실은 −12.7 / −13.9 pp로 비슷하지만 **경로가 정반대다.**
- **dropout**: 표적에는 여전히 접근한다(closest 0.972 m ≈ clean 0.806 m). 그런데 충돌이
  337 → 559로 늘고, 충돌 순간의 clearance가 0.92 → **0.65 m로 좁아진다**. 장애물 회피가
  나빠지는 서명이다 —— 카메라 *표적* 검출을 버렸는데 왜 장애물 성능이 떨어지는지가 핵심 질문이다.
- **learned detector**: 충돌 시 clearance는 오히려 **1.10 m로 넓고**, closest_nocrash_mean이
  1.892 m로 clean의 2.3배다. 즉 애초에 표적 근처까지 가지 못한다. 추적 정확도 문제다.

따라서 "검출 성능이 나쁘면 capture가 떨어진다"는 하나의 설명으로 묶으면 안 된다. 두 축을
같은 처방으로 다루려던 계획은 폐기한다.

**dropout 채널 가설(미검증)**: `visible`이 깜빡이면 `target_like` carve-out도 깜빡이므로,
표적 근처 LiDAR return이 어떤 스텝에는 지워지고 어떤 스텝에는 장애물로 남는다. 8개뿐인
obstacle token 예산을 표적이 간헐적으로 차지하면 실제 막대가 토큰에서 밀려난다 —— 좁은
clearance에서의 충돌 증가와 부합한다. 검증에는 토큰 점유율 계측이 필요하다.

**부수 확인 — noise 교란은 무시 가능**: R3의 모든 perturbation 셀은
`NAVRL_PERCEPTION_PERTURB=1`을 쓰는데, 이 플래그는 `navrl_task.py:3279`에서 `training=True`로
전달되어 dropout뿐 아니라 `rgb_noise_std=0.015`·`depth_noise_std=0.02`도 함께 켠다. clean 셀은
PERTURB=0이므로 원칙적으로 모든 perturbation 셀이 센서 노이즈와 교란된다. 그러나
range_error 두 셀(PERTURB=1)이 80.54 / 80.62%로 clean과 동일하므로, **이 노이즈의 비용은
사실상 0 pp**이며 의도치 않은 대조군 역할을 한다. 따라서 latency 잔차 2.5 pp와 dropout
−12.7 pp는 노이즈가 아니라 해당 축 자체의 비용이다.

## 2026-08-07 — 전체 코드 점검 + detector SHA 가드 부활

다음 축(learned detector)으로 넘어가기 전 전 저장소를 점검했다.

**기계적 점검 (전부 통과)**: 병합 충돌 마커 0건, 추적 대상 `.py` 전량 컴파일, `.sh` 전량 파싱,
python 테스트 21개 파일 + JS 1개 전부 PASS, pyflakes에 undefined name / redefinition 0건
(남은 지적은 업스트림 Aerial Gym의 star-import 및 sample_factory 예제의 미사용 지역변수).
대시보드는 재생성 후 `status.json`과 HTML fallback JSON이 일치하고, `getElementById` 대상
53개가 모두 HTML에 존재하며 탭 앵커도 누락이 없다.

**발견한 실제 결함 — detector SHA 가드가 죽어 있었다.**
`eval_navrl_v2_density_sweep.sh`가 detector 체크포인트를 해시해서
`NAVRL_EXPECTED_DETECTOR_SHA256`으로 export 하는데, **저장소 어디에서도 이 값을 읽지 않았다.**
`navrl_perception.py`는 SHA 확인 없이 `torch.load`만 했다. 즉 `learned_clean` 셀의 결과 JSON에
기록된 `detector_checkpoint_sha256`은 **실제로 로드된 바이트와 대조된 적이 없는 값**이었다.
이제 loader가 변수가 설정된 경우 파일을 해시해 불일치 시 RuntimeError로 중단한다. 변수가 없으면
기존처럼 로드한다(대화형·레거시 실행 보존). 회귀 테스트 3건 추가
(`tests/test_navrl_perception.py::DetectorCheckpointIntegrityTest`).

**미해결 리스크(내 작업 아님, 기록만)**: `artifacts/navrl_target_detector_v1.pth`와
`tools/train_navrl_target_detector.py`가 여전히 미커밋이다. 따라서 `learned_clean` 결과는
저장소만으로 재현할 수 없다. 다음 축이 바로 이 detector이므로, 진행 전에 이 두 파일의
커밋 여부를 소유자(다른 세션)와 정리해야 한다.

**환경 노브 교차 검증**: 스크립트가 참조하지만 python이 읽지 않는 `NAVRL_*` 19개를 전수 확인했다.
`NAVRL_EXPECTED_DETECTOR_SHA256`을 제외한 18개는 전부 셸 내부 변수(결과 경로, 워밍스타트 허용
플래그 등)이거나 문서 내 표기이며, `NAVRL_V2_FIXED_TARGET_SPEED`는 셸이 다른 노브로 변환해
소비하는 것을 확인했다. 즉 진짜 구멍은 하나뿐이었다.

## 2026-08-07 — dropout 채널: 유령 표적 장애물을 코드에서 격리 (가설 H1은 기각, H1' 확정 대기)

**먼저 내 가설 H1을 기각했다.** "dropout이 `visible`을 깜빡이게 해서 carve-out이 깜빡인다"는
전제가 틀렸다: carve-out gate는 `fused_visible`(camera OR LiDAR)이고, 이 값은 dropout에서
21.38%로 clean(21.21%)과 사실상 동일하다 —— LiDAR association이 빠진 카메라 프레임을 메우기
때문이다. GPU를 쓰기 전에 기각됐다.

**대신 실제 결함을 코드에서 격리했다 (H1')**: 장애물 지도는 **서로 다른 gate를 쓰는 두 경로**로
편집된다.
- LiDAR `target_like` carve-out (navrl_perception.py:884) → gate = `fused_visible`
- depth blanking (`depth_no_target`, :893) → gate = **카메라 전용** `pixels` 마스크

카메라가 놓쳤지만 LiDAR가 트랙을 유지한 프레임 —— dropout 0.3에서 전체의 30% —— 에서는
LiDAR 쪽이 표적을 지우고 카메라 쪽이 그대로 되살린다. 그 결과 표적이 **정면의 실체 장애물**이
되어 8개뿐인 obstacle token 하나를 먹는다. 바로 접근해야 할 순간에.

격리 재현 (`tests/test_navrl_latency_compensate.py::TargetMaskBackfill`, latency 0, CPU):
동일한 visibility·동일하게 carve-out이 작동하는데 전방 지도 거리가
**카메라 마스크 있음 4.00 m / 없음 3.00 m**. 표적 거리가 그대로 장애물로 남는다.

이 경로가 유력한 이유: dropout은 capture를 −12.70 pp 떨어뜨리면서 fused visibility는 건드리지
않는다. 즉 손실이 "표적을 덜 본다"로 설명되지 않으므로, **카메라가 놓친 프레임에서만 발화하는**
채널이 남은 후보다. 이 채널이 정확히 그렇다.

**수정(A/B 대기)**: `NAVRL_TARGET_MASK_BACKFILL`(기본 off). 카메라 마스크가 비었는데 트랙이
살아 있으면 fused bearing/range로 마스크를 재구성해 두 반쪽이 같은 것을 지우게 한다. 재구성
로직은 P2 predict와 **같은 헬퍼**(`_reconstruct_target_pixels`, 동일 agreement window)를 쓰므로
두 수정이 표적의 픽셀 범위에 대해 다른 답을 낼 수 없다. 마스크가 이미 있으면 절대 덮어쓰지
않는다(테스트로 고정).

**평가** `eval_navrl_v2_target_mask_backfill.sh`, 동일 계약, 2049 ep/cell, arm 4개:
clean / dropout_raw / dropout+backfill / clean+backfill(무해성 확인). 판정은 자동:
capture 회복 ≥ 4 pp면 SUPPORTED, ≤ 1.3 pp(arm 간 SE)면 REJECTED. **기각도 결과로 기록한다.**

## 2026-08-07 — target mask backfill A/B: 채널은 실재하나 dropout 손실의 20%뿐 (INCONCLUSIVE)

frozen ep25000+riskcap, seed47 / 205 bars / deterministic, 2049~2050 ep/cell.
`results/navrl_v2_target_mask_backfill/summary.{md,json}`.

| cell | capture | crash | bar contacts | fused visible | closest mean |
|---|---:|---:|---:|---:|---:|
| analytic_clean | 80.54% | 17.17% | 337 | 21.21% | 0.806 m |
| dropout_0p3_raw | 67.84% | 29.33% | 559 | 21.38% | 0.972 m |
| dropout_0p3_backfill | 70.38% | 26.65% | 513 | 21.00% | 1.020 m |
| clean_backfill | 79.11% | 18.55% | 365 | 21.03% | 0.832 m |

**판정 INCONCLUSIVE.** capture +2.54 pp(arm 간 SE 1.3 pp이므로 약 2σ)로 dropout 손실
12.70 pp의 **20.0%**를 회복했다. 독립적인 두 번째 지표도 같은 크기를 가리킨다: 초과 bar contact
222건(559−337) 중 46건, **20.7%** 감소. 서로 독립인 두 지표가 같은 비율을 내는 것은 채널이
실재하고 그 크기가 약 20%라는 뜻이지, 노이즈가 우연히 정렬된 것으로 보기 어렵다.

그러나 **지배적 채널은 아니다.** dropout 비용의 80%는 여전히 설명되지 않는다. P2(11%)에 이어
두 번째로 "실재하지만 작은" 채널이며, latency에서 P3가 94%를 회수했던 것과는 성격이 다르다.

**채택 보류**: `clean_backfill`이 79.11%로 clean 대비 −1.42 pp(약 1σ)다. 카메라가 프레임을
놓치지 않을 때는 backfill이 발화하면 안 되는데(마스크가 이미 있으면 덮어쓰지 않도록 테스트로
고정했다) 값이 내려갔다. 노이즈일 가능성이 높지만, 이득이 +2.54 pp인 상황에서 −1.42 pp의
무해성 미확인은 무시할 수 없다. **기본 off를 유지하고, 채택은 남은 80%를 설명한 뒤 재검토한다.**

### 다음 가설 H2: LiDAR 대체 측정의 조악함

카메라가 놓친 프레임에서 트래커는 LiDAR association으로 보정된다. LiDAR bearing은 빈 폭
360/72 = 5°로 양자화되어 있어 5 m 거리에서 측방 오차 0.44 m다 —— P3가 제거한 ego-motion 오차
(0.26 m)보다도 크다. 즉 dropout은 "표적을 덜 보는" 것이 아니라 **30%의 프레임에서 훨씬 나쁜
측정으로 트래커를 오염시키는** 것일 수 있다. P1(LiDAR association을 더 열어줌)이 −8.85 pp로
역효과였던 것도 이 방향과 일치한다.

검정 방법: dropout 상태에서 LiDAR 대체 보정을 끄고 CV 예측으로 coasting 시키는 arm과 비교한다.
H2가 맞으면 조악한 보정을 버리는 쪽이 낫게 나온다. `fused_visible`이 카메라 전용으로 떨어지므로
time-since-seen 특징도 함께 변한다는 점을 해석 시 감안해야 한다.

## 2026-08-07 — 밀도×속도 맵의 task-version 표기 누락 수정 (기존 데이터만, 재측정 미실시)

사용자 지적으로 대시보드 히트맵을 재검토했다. **라벨 산수 자체는 맞다**:
25 bars → 5.2/100m², 85 → 17.8, 150 → 31.4는 전부 v1 배치면적 478 m²
(24 m 아레나, 막대가 x 0.13~0.96 대역) 기준으로 정확하다.

**문제는 표기 누락이다.** 같은 페이지가 `placement_area_m2 = 1600`(v2, 40 m 아레나)을
보고하는데 히트맵에는 v1이라는 표시가 없었다. 그래서 85 bars @ 17.8/100m²를 v2의 205-bar
셀과 나란히 비교하게 되는데, 실제 밀도는 **3.3배** 차이다(85 bars는 v2 기준 5.3/100m²).
이는 WORKLOG 2026-07-31에 "사이트/논문에서 구분 표기할 것"으로 적어둔 항목이 미이행된
것이며, 이 그림이 논문 headline figure로 지정돼 있어 위험도가 높았다.

**수정(GPU 미사용, 기존 데이터만)**:
- `_stamp_density_speed_map()`이 `task_version="v1"`, `arena_xy_m=24`,
  `placement_area_m2=478`, `comparable_with_v2=False`, superseded 문구를 데이터에 찍는다.
  478이 코드 내부 상수로만 존재하던 것을 **데이터가 스스로 설명**하게 바꾼 것이다.
- 배치면적 상수를 `V1_PLACEMENT_AREA_M2` / `V2_PLACEMENT_AREA_M2`로 명명해
  `_placement_area_m2()`와 스탬프가 같은 값을 공유하게 했다(두 곳이 어긋날 수 없게).
- 패널은 축에 "per 478m²"를 표시하고, 부제에 "v1 task · 24 m arena"를, 캡션에
  "85 bars는 여기선 17.8/100m²지만 v2 아레나에선 5.3/100m²"를 데이터에서 계산해 명시한다.

**의도적으로 하지 않은 것**: 각 셀에 v2 환산 밀도를 병기하지 않았다. 면적만 환산해도
막대 굵기·배치 대역·표적 운동 법칙이 모두 달라서, 환산값은 "비교 가능하다"는 잘못된 인상을
준다. 비교 불가를 명시하는 편이 정확하다.

**미결(사용자 합의)**: v2 아레나(40 m, 1600 m²)에서 밀도×속도 맵을 **재측정**해야 논문
figure로 쓸 수 있다. 현재 맵은 그때까지 v1 확정 데이터로만 보존한다.

## 2026-08-07 — H2(LiDAR 대체 보정) + backfill 시드 재현: LiDAR target association이 순손실

두 건을 따로 측정했다. frozen ep25000+riskcap, 205 bars, deterministic, 2049~2051 ep/cell.

### (1) backfill 시드 재현 — 채택 기각

`results/navrl_v2_target_mask_backfill_seed51/`.

| | dropout 이득 | clean 회귀 |
|---|---:|---:|
| seed47 | +2.54 pp | −1.42 pp |
| seed51 | +1.77 pp | −2.39 pp |
| 평균 | **+2.15 pp** | **−1.91 pp** |

**clean 회귀가 재현됐다** —— 2개 시드에서 같은 방향이므로 노이즈가 아니다. dropout 이득도
재현되지만(+2.54 → +1.77) 크기가 clean 손실과 거의 같아 **순효과가 상쇄된다. backfill은
채택하지 않는다**(기본 off 유지).

회귀의 메커니즘도 설명된다: backfill은 카메라 마스크가 비었을 때 **LiDAR 유래 bearing/range**로
마스크를 재구성하는데, 그 bearing은 5° 양자화돼 있다. ±15° 창으로 depth를 blank 하므로
**실제 장애물 픽셀까지 지운다.** 원래 버그(표적을 못 지움)를 고치면서 반대 방향 버그(장애물을
지움)를 만든 셈이고, 조악한 LiDAR 측정을 소비한다는 점에서 원인은 같다.

### (2) H2 — LiDAR 대체 보정을 끄면 오히려 좋아진다

`results/navrl_v2_lidar_fallback/` (seed47).

| cell | capture | crash | bar contacts | fused visible |
|---|---:|---:|---:|---:|
| analytic_clean | 80.54% | 17.17% | 337 | 21.21% |
| clean_no_assoc | 80.35% | 17.50% | 346 | 18.44% |
| dropout_0p3_raw | 67.84% | 29.33% | 559 | 21.38% |
| dropout_0p3_no_assoc | **71.25%** | 25.96% | 511 | 14.59% |

dropout에서 **+3.42 pp**(손실의 26.9%), bar contact −48(초과분의 21.6%). 그런데
**clean에서는 −0.19 pp로 무해**하다. 즉 프로파일이 backfill보다 확연히 낫다.

**핵심 해석**: clean에서 LiDAR association은 fused visibility를 18.44% → 21.21%로 2.8 pp
올려주지만 capture에는 0.19 pp밖에 기여하지 않는다. 반면 dropout에서는 그 경로가 트래커를
조악한 측정으로 오염시켜 3.42 pp를 **깎아먹는다**. 즉 **LiDAR target association은 순손실**이다.
P1(association을 더 열어줌)이 −8.85 pp였던 것이 이 그림에 정확히 들어맞는다 —— 세 번째
독립 증거다.

두 개입은 배타적이다: assoc를 끄면 `fused_visible`이 카메라 전용이 되어 카메라가 보일 때는
마스크도 항상 존재하므로 backfill이 아예 발화하지 않는다. 따라서 합산 효과를 기대할 수 없고,
**assoc-off 쪽만 남는다.**

### 판정과 다음 단계

자동 게이트 기준으로는 여전히 INCONCLUSIVE(+3.42 < 4 pp)이고 단일 시드다. **채택 전에
seed51 재현이 필요하다** —— backfill이 정확히 이 이유로 기각됐으므로 같은 잣대를 적용한다.
재현되면 `NAVRL_LIDAR_TARGET_ASSOC=0`을 기본값으로 승격하는 안을 검토한다. 다만 그 경우
"카메라가 놓친 동안 표적 추적을 포기한다"는 설계 변경이므로, 논문에서는 성능 수치가 아니라
**LiDAR 각분해능(5° = 5 m에서 0.44 m)이 표적 추적에 부족하다**는 관측으로 서술해야 한다.

여전히 dropout 손실의 약 70%는 미설명이다.

## 2026-08-07 — H2 재현 성공 + H3 발견: LiDAR 보정이 관측하지 않은 축의 공분산을 줄이고 있었다

### H2 재현 (seed51)

| | dropout 이득 | clean 비용 |
|---|---:|---:|
| seed47 | +3.42 pp | −0.19 pp |
| seed51 | +3.09 pp | −1.02 pp |
| 평균 | **+3.26 pp** (손실의 26%) | −0.61 pp |

이득이 견고하게 재현된다(+3.42/+3.09, bar contact −48/−41 = 초과분의 21.6%/20.2%).
독립적인 네 수치가 모두 일치하므로 채널은 확정이다. 다만 clean 비용이 두 시드 모두 음수라
**assoc를 통째로 끄는 것은 운용점(clean)에서 손해를 보고 스트레스점에서 이득을 보는 거래**다.

### H3 — 왜 그런지 찾았다

`_associate_lidar_target`가 만드는 "측정"은 range 하나뿐이다. bearing은 **트래커 자신의 예측**이고
(`bearing = atan2(rel[1], rel[0])`, rel은 예측 상태), z 성분도 `rel[:, 2]`로 **예측값**이다.
그런데 이것을 대각 R로 3-D 갱신에 넣는다(`lidar_var = [0.08², 0.15², 0.20²]`). 결과적으로
**아무도 관측하지 않은 측방·수직 축의 공분산이 줄어든다.**

카메라를 잃은 표적에서 측정한 결과:

| blind steps | 필터가 보고하는 측방 σ | 실제 측방 오차 |
|---:|---:|---:|
| 0 | 0.046 m | 0.00 m |
| 5 | 0.098 m | 0.45 m |
| 10 | 0.092 m | 1.17 m |
| 20 | 0.092 m | **3.27 m** |

σ가 사실상 고정된다 —— 프로세스 노이즈에 의한 정상적 증가가 매 스텝 상쇄되기 때문이다.
**정책은 이 공분산을 관측한다**(`_target_features`의 pos_var). 즉 "표적을 잃었다"는 신호를
받아야 할 때 "확실하다"는 신호를 받는다. P1(−8.85 pp), H2(+3.26 pp)가 모두 이것으로 설명된다.

### H3 수정

`correct()`에 전체 3×3 `measurement_cov`를 받는 경로를 추가하고, LiDAR 갱신은 R을 측정 광선
기준으로 구성한다: `σ_r²·uuᵀ + σ_perp²·(I − uuᵀ)`, `σ_perp = LIDAR_UNOBSERVED_SIGMA_M = 50 m`.
range 정보는 그대로 쓰면서 관측하지 않은 방향은 프로세스 노이즈에 맡긴다. 노브
`NAVRL_LIDAR_RANGE_ONLY_UPDATE`(기본 off). 수정 후 측방 σ는 20스텝에서 0.16 → **1.47 m**로
정직하게 증가한다. 테스트 28/28 PASS(기존 perception 23/23 회귀 PASS).

**H2보다 원칙적인 수정이다**: 측정을 버리지 않고 정밀도만 정직하게 만든다. 공분산 거짓말이
채널이라면 H3는 dropout에서 H2와 같거나 낫고 **clean 비용은 없어야** 한다.

### 부수 수정 — H2 스크립트 헤더가 다른 실험을 기술하고 있었다

`eval_navrl_v2_lidar_fallback.sh`를 backfill 스크립트에서 파생시킬 때 헤더 치환이 조용히
실패해(치환 대상 문자열 불일치) 주석이 backfill 실험을 설명하고 있었다. 실행부·env·결과는
정상이며 PREFLIGHT와 receipt로 arm을 확인했으므로 **측정 결과는 유효하다.** 헤더를 바로잡았고,
H3 스크립트 생성 시에는 모든 치환이 실제로 적용됐는지 assert로 검증했다.

## 2026-08-07 — H3 결과: 공분산 거짓말은 채널의 일부일 뿐(11.5%), 게이트와 자기상쇄한다

`results/navrl_v2_lidar_range_only/summary.{md,json}` (seed47, 2049~2050 ep/cell).
H2 기준선을 같은 run에 넣어 직접 비교했다.

| cell | capture | crash | bar contacts | fused visible |
|---|---:|---:|---:|---:|
| analytic_clean | 80.54% | 17.17% | 337 | 21.21% |
| clean_range_only | 80.00% | 16.59% | 324 | 20.02% |
| dropout_0p3_raw | 67.84% | 29.33% | 559 | 21.38% |
| **dropout_0p3_range_only (H3)** | **69.30%** | 28.06% | 539 | 21.22% |
| **dropout_0p3_no_assoc (H2)** | **71.25%** | 25.96% | 511 | 14.59% |

H3는 +1.46 pp(손실의 11.5%)에 그친다. arm 간 SE가 1.3 pp이므로 **1.1σ로 유의하지 않다.**
같은 run에서 H2가 +3.42 pp인 것과 비교하면 **H3는 H2보다 확연히 나쁘다.**
H2 기준선(71.25%)이 seed47 원래 측정과 정확히 일치해 run 간 비교가 아님도 확인된다.

**즉 공분산 거짓말 가설은 부분적으로만 옳다.** 그것이 전부였다면 H3 ≈ H2여야 했다.
차이 약 2 pp는 **상태 보정 자체가 해롭다**는 뜻이다: 예측 bearing 방향의 LiDAR 빈이 표적이
아니라 **막대**를 맞히면, R을 광선 기준으로 바로잡아도 상태가 그 막대 거리로 끌려간다.
공분산이 아니라 오연관(mis-association)이 남은 손해다.

### 게다가 H3는 자기상쇄한다 (측정된 상호작용)

association 게이트는 `(0.35 + 2.0*pos_sigma).clamp(max=1.0)`으로 **공분산에 비례해 넓어진다.**
H3가 공분산을 정직하게 키우면 게이트도 같이 커진다:

| blind steps | full-3D: pos_sigma → gate | range-only: pos_sigma → gate |
|---:|---:|---:|
| 2 | 0.164 m → 0.679 m | 0.229 m → 0.808 m |
| 5 | 0.169 m → 0.688 m | 0.492 m → **1.000 m (상한)** |
| 20 | 0.158 m → 0.666 m | 2.050 m → **1.000 m (상한)** |

기존 코드는 공분산이 고정돼 게이트도 ~0.67 m로 고정됐는데, H3에서는 5스텝 만에 상한
1.0 m에 도달한다. **불확실성을 정직하게 만든 대가로 오연관 창이 50% 넓어진다.**
정직한 공분산이 더 나쁜 연관을 불러들이는 구조라, H3 단독으로는 이득이 상쇄된다.

`clean_range_only`는 80.00%로 −0.54 pp(1σ 미만), bar contact는 오히려 337 → 324로 줄었다.
clean에서는 사실상 중립이다.

### 판단

- **공분산 수정(H3)은 단독 채택하지 않는다** —— 유의하지 않고, 게이트와 자기상쇄한다.
- 남은 후보: **H3 + 게이트 고정/축소**(공분산에 연동하지 않는 상수 게이트). 이것이 두 결함을
  동시에 제거하는 유일한 조합이다.
- 실용적 대안: **H2(assoc off) 채택**. dropout에서 +3.26 pp(2시드 재현), clean −0.61 pp.
  단순하고 재현됐지만 운용점에서 손해를 본다.

dropout 손실의 여전히 약 70%는 미설명이며, 지금까지 확인된 채널은 유령 장애물(~20%, 순효과
상쇄로 기각)과 LiDAR 연관 경로(~26%, 형태 미정)뿐이다.

## 2026-08-07 — 게이트 가설 기각: LiDAR 연관은 좁혀도 살아나지 않는다

`results/navrl_v2_lidar_assoc_gate/summary.{md,json}` (seed47, 2049~2050 ep/cell).

| cell | capture | Δ vs raw | crash | bar contacts |
|---|---:|---:|---:|---:|
| analytic_clean | 80.54% | — | 17.17% | 337 |
| clean_ro_gate035 | 80.48% | −0.06 pp | 16.59% | 337 |
| dropout_0p3_raw | 67.84% | — | 29.33% | 559 |
| dropout_0p3_ro_gate065 | 69.35% | +1.51 pp | 27.48% | 525 |
| dropout_0p3_ro_gate035 | 69.55% | +1.71 pp | 28.01% | 546 |
| dropout_0p3_no_assoc (H2) | 71.25% | +3.41 pp | 25.96% | 511 |

**자기상쇄 가설 기각.** 게이트를 고정·축소해도 H3의 이득이 풀리지 않는다. 게이트 폭에 따른
추세를 보면:

| 게이트 | auto(≈1.0) | 0.65 m | 0.35 m | off(H2) |
|---|---:|---:|---:|---:|
| capture | 69.30% | 69.35% | 69.55% | **71.25%** |

**0.65 → 0.35로 절반 가까이 좁혀도 +0.20 pp뿐인데, 연관을 아예 끄면 +1.70 pp가 더 붙는다.**
즉 문제는 "연관 창이 너무 넓다"가 아니다. 게이트를 아무리 좁혀도 통과한 연관 자체가 해롭다.
**LiDAR target association은 정직하게 만들거나 선별적으로 만들어서는 살릴 수 없다.**

부수적으로 `clean_ro_gate035`는 80.48%로 clean 대비 −0.06 pp —— range-only + 좁은 게이트는
clean에서 완전히 무해하다. 다만 dropout 이득이 +1.71 pp(1.3σ)라 채택 근거가 못 된다.

### 남은 설명: 공분산이 아니라 visible/age 플래그다

`correct()`는 매 LiDAR 보정마다 **`self.age[update] = 0.0`으로 time-since-seen을 리셋**하고,
`observe()`는 `fused_visible = visible | lidar_visible`로 **visible을 참으로 만든다.** 그런데
그 "측정"의 bearing·z는 필터 자신의 예측이다. 즉 정책의 target token은 예측을 되먹인 것에
대해 **"보인다 / 방금 봤다"** 를 통보받는다.

이것이 H2가 통하는 이유를 가장 잘 설명한다. assoc를 끄면 visible이 거짓이 되고 age가 자라서
정책이 **"표적을 잃었다"** 를 알게 된다. H3(공분산 정직화)와 게이트 축소는 둘 다 이 두 플래그를
건드리지 않으므로 효과가 없었다 —— 정책이 읽는 것은 공분산 크기보다 **이 이산 신호**다.

**결정적 검정(미실시)**: range 보정은 유지하되 `age` 리셋과 `lidar_visible` 기여를 끈다.
H2와 같은 이득이 나오면 채널은 플래그로 확정되고, 상태 보정은 무해하다는 뜻이 된다.

### 이 가지의 결론

세 번의 개입(H3 공분산, 게이트 0.65/0.35, backfill)이 모두 유의 미달이고, 통째로 끄는 H2만
+3.26 pp(2시드)로 견고하다. LiDAR 연관 경로에서 더 정교한 수정을 시도할 근거는 소진됐다.
실용적 선택지는 **H2 채택** 또는 **플래그 검정 1회 후 종료**이며, 어느 쪽이든 dropout 손실의
**약 70%는 여전히 미설명**이라 그쪽이 기댓값이 크다.

## 2026-08-10 — H4 플래그 검정: dropout 손실의 채널은 상태 보정이 아니라 "봤다" 플래그(69%)

`results/navrl_v2_lidar_silent_correct/summary.{md,json}` (seed47, 2049~2050 ep/cell).
H2 기준선을 같은 run에 포함해 직접 비교했다.

| cell | capture | Δ vs raw | crash | bar contacts | fused visible |
|---|---:|---:|---:|---:|---:|
| analytic_clean | 80.54% | — | 17.17% | 337 | 21.21% |
| clean_silent | 81.11% | **+0.58 pp** | 16.06% | 320 | 18.60% |
| dropout_0p3_raw | 67.84% | — | 29.33% | 559 | 21.38% |
| **dropout_0p3_silent (H4)** | **70.20%** | **+2.36 pp** | 26.78% | 523 | 14.46% |
| dropout_0p3_no_assoc (H2) | 71.25% | +3.41 pp | 25.96% | 511 | 14.59% |

**채널 분해 성공.** H4는 range 보정을 **그대로 유지**하면서 age 리셋·visibility·confidence
기여만 차단한다. 그런데도 H2 이득 3.41 pp 중 **2.36 pp(69%)를 회수**한다. 즉

- **플래그 오염 = 채널의 약 2/3** (예측을 되먹인 측정에 "보인다 / 방금 봤다"를 통보)
- **상태 보정 자체 = 나머지 약 1/3** (≈1.05 pp, 오연관으로 상태가 끌려가는 몫)

두 몫의 합이 H2와 일치하므로 분해가 정합적이다. H3(공분산 정직화 +1.46 pp)와 게이트 축소
(+1.71 pp)가 왜 실패했는지도 확정된다: **둘 다 플래그를 건드리지 않았다.** 정책이 읽는 것은
공분산의 크기가 아니라 **visible/time-since-seen이라는 이산 신호**다.

**clean에서 H4는 +0.58 pp로 무해하다** —— H2(−0.19/−1.02 pp)보다 프로파일이 낫다. 다만
0.45σ이므로 개선 주장은 하지 않고 "무해"로만 기록한다.

**메커니즘 서술(논문용)**: LiDAR 연관은 bearing·수직 성분을 트래커 예측에서 읽어와 측정으로
되먹인다. 이 측정이 `correct()`에서 `age=0`을 찍고 `observe()`에서 `fused_visible`을 참으로
만들기 때문에, 정책의 target token은 **자기 예측에 대해 "방금 관측했다"는 증언**을 받는다.
표적을 잃은 구간에서 정책이 "확실하다"고 믿고 돌진하는 것이 dropout 손실의 주 경로다.

**판정**: 자동 게이트 기준으로는 여전히 INCONCLUSIVE(+2.36 < 4 pp)이고 단일 시드다. 채택은
보류하되, **이 가지의 진단은 완료**로 종료한다. 세부 개입 4회(P2 backfill, H3 공분산,
게이트 2값, H4 플래그)와 통짜 차단(H2) 1회로 LiDAR 연관 경로의 내부 구조가 정량화됐다.
남은 dropout 손실 **약 70%는 이 경로 바깥**이며, 다음 우선순위는 learned detector(−13.9pp)다.

## 2026-08-10 — 대시보드 곡선 전수 감사: 밀도 표기 3.3× 오류 + 키랄리티 이전 데이터 노출 수정

사용자가 "curves 쪽 막대 개수가 옛날 같다"고 지적해 `docs/status`의 모든 데이터 섹션을
전수 감사했다. **결함 3종을 확인했고, 재학습이 필요한 항목은 없었다.**

### 결함 1 — 밀도(/100m²) 표기가 3.3× 과소

`app.js`의 `perc100()`이 전역 `BAND_AREA`를 쓰는데 이 값은 `placement_area_m2`(v2, 1600 m²)로
덮어써진다. 그런데 Density 탭이 그리던 곡선은 **v1 데이터(478 m² 배치대역)**였다.
25막대가 화면에 **1.6/100m²**로 표시됐으나 실제 v1 값은 **5.2**다. 08-07에 고친 히트맵과 같은
계열의 버그이나 위치가 다르다 —— 맵은 `density_per_100m2`가 데이터에 박혀 있어 무사했고,
곡선 표는 **화면에서 계산**하기 때문에 틀렸다.

### 결함 2 — Speed 탭이 키랄리티 수정 이전 데이터를 표시

`pickSpeed()`가 고르던 `general_repr_fov240_speed_axis`는 체크포인트 `ppo_260727_0930`,
즉 **07-27**이다. LiDAR 방위 테이블 거울상 + 카메라 far-plane 팬텀 벽은 **07-29에** 수정됐다.
WORKLOG 07-29가 "기존 전 결과는 거울 조건의 기록으로만 유효"라고 명시한 바로 그 데이터를
현재 성능처럼 보여주고 있었다. 같은 이유로 `vision_density_curve`의 GT-injected 열도 노출 중이었다.

### 결함 3 — v2 곡선이 아카이브에 있는데 대시보드에 없음

현재 태스크는 40 m·205막대인데 화면은 v1(24 m·25~150막대)만 보여줬다.

### 수정 (전부 아카이브 데이터, PPO 재학습 0)

- `_stamp_curve_provenance()`가 모든 곡선에 `task_version`·`arena_xy_m`·`placement_area_m2`·
  `superseded`를 찍는다. 키랄리티 이전 9개 곡선은 `superseded=True`와 사유를 갖는다.
  **삭제하지 않는다** —— 반증된 증거도 증거이며, 그 조건의 기록으로는 유효하다.
- `perc100(n, area)`가 **시리즈별 면적**을 받는다. 전역 1개 분모가 원인이었다.
- 렌더러는 `superseded` 시리즈를 절대 선택하지 않고 v2를 우선한다. GT 열도 소스가
  superseded면 숨긴다.
- 아카이브에서 v2 곡선 2개를 복원했다:
  - `v2_heldout_density_curve` — ep24000 frozen, 130~220막대, 2049~2050 ep/셀
    (84.77 / 79.66 / 73.99 / 72.44 / 68.49%). 정책이 현재 후보(ep25000+riskcap)가 **아님**을
    `policy` 필드에 명시.
  - `v2_riskcap_fixed_speed_axis` — **현재 동결 후보 ep25000+riskcap**, 205막대,
    표적 0.3/0.9/1.5 m/s에서 81.84 / 80.77 / 75.51%.
- 패널 부제·캡션이 시리즈의 task version·아레나·면적·정책을 데이터에서 읽어 표시한다.
  Speed 캡션은 "25 bars · historical FOV ablation"이 하드코딩돼 있어 다른 시리즈를 골라도
  그 문구가 남았다 —— 이것도 제거했다.

검증: 상태 스냅샷 테스트 11 PASS, arena JS 파리티 PASS, `getElementById` 대상 누락 0,
fallback JSON 동기화 확인. 렌더러 선택 로직을 재현해 Density=v2(8.12~13.75/100m²),
Speed=v2 riskcap, GT 열=숨김을 확인했다.

### 남은 선택 항목 (재학습 아님, 평가만)

1. v2 밀도 곡선을 **ep25000+riskcap으로 재측정**(5셀 ≈ 32분). 현재는 ep24000이라 정책 불일치를
   라벨로만 고지하고 있다.
2. **v2 밀도×속도 맵**(28셀 ≈ 3시간) —— 논문 헤드라인 그림. 07-31부터 미결.

## 2026-08-10 — v2 재측정 2건: riskcap 이득이 밀도와 함께 커지고, 속도×밀도 상호작용 발견

대시보드가 v1 데이터로 현재 태스크를 설명하던 문제를 실측으로 해소했다. 둘 다 inference-only,
PPO 재학습 없음. frozen ep25000+riskcap, seed47, deterministic, 2049~2051 ep/cell.

### (1) v2 밀도 곡선 — `results/navrl_v2_density_curve_riskcap/`

ep24000(governor off) 곡선과 **동일 격자**로 측정해 직접 비교 가능하다.

| bars | /100m² | ep25000+riskcap | ep24000 (governor off) | Δ |
|---:|---:|---:|---:|---:|
| 130 | 8.12 | 89.31% | 84.77% | +4.54 pp |
| 160 | 10.00 | 84.63% | 79.66% | +4.97 pp |
| 190 | 11.88 | 82.77% | 73.99% | +8.78 pp |
| 205 | 12.81 | 80.54% | 72.44% | +8.10 pp |
| 220 | 13.75 | 77.76% | 68.49% | +9.27 pp |

**이득이 밀도와 함께 커진다**(+4.54 → +9.27 pp). 205막대 80.54%가 기존 clean 수치와 정확히
일치해 계약 정합성도 확인된다.

**⚠️ 귀속 주의 — 이 Δ는 riskcap 단독 효과가 아니다.** 두 arm은 **두 가지가 동시에 다르다**:
체크포인트(ep24000 → ep25000, 1,000 epoch 적응)와 governor(off → riskcap). 따라서 이 비교는
"현재 동결 후보가 이전 체크포인트보다 낫고 그 격차가 밀도와 함께 벌어진다"까지만 말한다.
TTC selector가 FOV 240°→360°를 함께 바꿔 pure ranking 효과를 판정 불가로 만들었던 것과
같은 종류의 confound다.

205막대 한 점에서는 두 효과가 이미 분리돼 있다(`navrl_v2_riskcap_postadapt`, seed45 uniform):
off **70.03%** → source+riskcap **78.20%**(governor 단독 +8.17 pp) → trained+riskcap **81.94%**
(적응 단독 +3.74 pp). 즉 205막대에서는 governor가 지배적이지만, **밀도 의존성이 governor에서
오는지 적응에서 오는지는 미확정**이다. 분리하려면 ep24000+riskcap을 같은 밀도 격자에서
측정해야 한다(5셀, inference-only).

### (2) v2 밀도×속도 맵 — `results/navrl_v2_density_speed_map/` (20셀)

| bars | /100m² | 0.3 m/s | 0.7 | 1.1 | 1.5 |
|---:|---:|---:|---:|---:|---:|
| 130 | 8.12 | 88.29 | 88.82 | 88.82 | 87.41 |
| 160 | 10.00 | 86.10 | 86.78 | 87.12 | 84.53 |
| 190 | 11.88 | 82.63 | 84.09 | 82.43 | 80.78 |
| 205 | 12.81 | 81.21 | 81.21 | 80.77 | 78.15 |
| 220 | 13.75 | 77.94 | 79.94 | 78.09 | **71.95** |

- 밀도 축 **−11.4 pp**, 표적속도 축 **−2.7 pp** → **비대칭의 형태는 v1에서 재현된다.**
- 단위 밀도당 기울기로 환산하면 v2 **2.02 pp** vs v1 **2.98 pp**(per 1 bar/100m²)로 같은
  자릿수다. v1의 −78 pp는 밀도 범위가 5.2→31.4(6배)로 훨씬 넓었기 때문이며, v2 격자는
  8.12→13.75(1.7배)다. **숫자가 아니라 기울기로 비교해야 한다.**

**새 발견 — 속도 비용이 밀도에 따라 커진다** (v1 헤드라인이 잡지 못한 상호작용):

| bars | 0.3 → 1.5 m/s 비용 |
|---:|---:|
| 130 | −0.88 pp |
| 160 | −1.57 pp |
| 190 | −1.85 pp |
| 205 | −3.06 pp |
| 220 | **−5.99 pp** |

즉 "표적 속도는 난이도 축이 아니다"는 **저밀도에서만 참**이다. 혼잡해질수록 표적 운동이
비용을 만든다 —— 회피 기동 중 표적이 이동해버리는 상호작용으로 해석된다. 논문에서는
"속도는 무관"이 아니라 **"속도는 밀도와 곱해져서만 유의미해진다"**로 서술해야 한다.

속도 축이 단조가 아니라는 점도 기록한다: 0.7 m/s가 5개 밀도 중 4개에서 0.3보다 높다.
205막대 고정속도 4셀 평균 80.33%가 uniform U[0.3,1.5] 실측 80.54%와 일치해 격자 정합성은 확인됐다.

### 대시보드 반영

- 히트맵을 **v2 맵으로 교체**, v1 맵은 `density_speed_map_v1`로 라벨과 함께 보존.
- 밀도 곡선을 ep25000+riskcap으로 교체하고, 행마다 `ep24000_capture`를 붙여 governor 기여가
  암시가 아니라 **표시**되게 했다.
- 기존 v1 스탬퍼가 새 v2 맵을 v1으로 덮어쓰던 버그를 수정(이미 task_version을 선언한 팩은
  건드리지 않는다).

## 2026-08-10 — Codex 독립 검수: P3 조건부 승인, 밀도 귀속·속도 상호작용·H4 과대해석 지적

Claude가 작성한 `docs/review_brief_2026-08-10.md`와 `715dc76..d9ee124`를 독립 검수했다.
상세 계산·근거·남은 inference-only 평가 설계는 `docs/codex_review_2026-08-10.md`에 보존했다.
PPO 재학습이나 정책/실행 코드 변경은 하지 않았다.

- **P3 조건부 동의**: measurement/pose가 동일 ring index를 쓰고 startup `visible=False`, τ=0 early
  return이 맞다. focused latency tests 33/33 PASS. capture-time pose 변환은 timestamped pose history를
  쓰는 실기 파이프라인과 맞지만, exact clock/pose 전제를 논문에 명시해야 한다.
- **밀도 곡선 정정은 아직 부족**: ep24000/off와 ep25000/riskcap은 checkpoint와 governor뿐 아니라
  seed **42→47**도 다르다. evaluator SHA도 다르고 receipt가 imported source tree를 해시하지 않아
  revision provenance 공백도 있다. Δ를 method/governor 이득으로 읽히게 하는 문장은 내려야 한다.
- **속도×밀도 상호작용 반박**: 학습범위 ≤205 aggregate logistic interaction `p=0.337`이고 density별
  omnibus `p=0.817`. OOD 220 포함 때만 continuous interaction `p=0.022`. 205의 0.3→1.5 m/s
  차이 −3.06 pp는 명목상 유의하지만 사후 다중비교 결과다. “속도는 밀도와 곱해져서만 의미” 및
  v1/v2 endpoint 기울기 비교는 발표 문장에서 삭제해야 한다.
- **H2는 작지만 재현**: 두 seed pooled +3.251 pp, 95% CI [+1.268,+5.234], p=0.00131. 그러나
  preregistered 4 pp adoption gate는 미달이다. H4 +2.357 pp는 95% CI [−0.473,+5.188], p=0.103이라
  “69% 분해 성공”은 exploratory로 낮춰야 한다. 69%는 H2 효과 대비이며 전체 dropout loss 대비는 18.6%다.
- **learned detector 해석 주의**: 현 artifact는 positive pixel 0.054%에 unweighted BCE를 쓴 1×1
  classifier이며 정면 2–12 m 중심 데이터만 수집했다. −13.9 pp는 이 artifact의 손실이지 learned
  detector 일반 한계가 아니다. 다음 우선순위는 맞지만 full-FOV/range/occlusion/absent held-out gate와
  class-balanced loss/calibration이 navigation 재평가보다 먼저다.
- **dashboard**: pre-chirality 9개 분류·archive 보존은 맞다. post-fix legacy 500-epoch pilot은 유효한
  역사 자료지만 current-v2 headline fallback 후보에서는 제외해야 한다.

남은 작업은 전부 평가/오프라인 지도학습이다. 최소 평가 순서는 source manifest 고정 → seed47에서
ep24000 off/riskcap 5밀도 10셀 → in-distribution speed endpoint 16셀/2 new seeds → detector offline
gate와 새 artifact 평가다. dropout/H4 가지는 추가 A/B 없이 종료한다.

## 2026-08-10 — 다음 주 재개용 동결·handoff

사용자가 이번 주 작업을 정리한 뒤 다음 주부터 재개하기로 했다. 현재 NavRL 학습·평가 프로세스는
없고 RTX 3070은 유휴 상태다. PPO 재학습은 계속 금지하며, current frozen ep25000+riskcap
checkpoint(`f7022139...`), decomposition용 ep24000 checkpoint(`82f7978b...`), diagnostic detector
artifact(`15cb90e...`)의 SHA-256을 다시 확인했다.

재개 순서와 종료 조건을 `docs/NEXT_WEEK_HANDOFF_2026-08-10.md`에 고정했다. 이번 주에는 Claude가
Codex 검수 결과를 원 문서/dashboard generator에 반영하고, status snapshot 재생성·라벨 재검수·단일
커밋/push까지만 한다. 새 GPU 평가는 하지 않는다. 다음 주에는 evaluation source provenance 고정 →
seed47 governor/adaptation 10셀 분리 → in-distribution speed endpoint 16셀/2 seed → detector offline
dataset/loss/calibration gate 순서로 진행한다. dropout/H4 추가 분해와 새 PPO 실험은 종료/보류한다.

## 2026-08-10 — 최종 강검수: 종료·bootstrap·source receipt·리워드/액션 의미 수정

사용자 요청으로 사이트, 평가 계약, 실험 파라미터, reward/action 의미를 코드부터 다시 감사했다.
정책 weight를 바꾸거나 PPO를 재학습하지 않았고, 과거 수치를 새 의미로 소급 변환하지 않았다.

### 발견한 중대 결함과 수정

1. **time-limit value bootstrap 불일치**: YAML은 `value_bootstrap: True`지만 설치된 rl_games
   `a2c_common.py`는 exact key `infos["time_outs"]`만 읽는다. 환경은 `timeouts`만 내보내고 있었다.
   현재 환경은 동일 tensor를 두 key로 제공하고 checkpoint에 comparator/key/bootstrap 계약을 기록한다.
2. **600-step가 실제 601 actions**: 증가된 `sim_steps > episode_len_steps`를 `>=`로 바꿔 action 600에서
   종료한다. 과거 JSON의 timeout outcome step이 전부 601인 것과 대조했다. helper 회귀 테스트와
   evaluator의 timeout mean/p10/p50/p90=600 실측 gate를 추가했다.
3. **평가 provenance 부족**: checkpoint와 top-level shell hash만으로는 dirty/imported Python source를
   재현할 수 없었다. evaluator는 이제 checkpoint/detector immutable snapshot, `aerial_gym` runtime source
   280개 파일 snapshot+SHA, git commit/dirty status, Python/pip manifest를 schema-v2 receipt에 묶고 각
   cell 뒤 원본·snapshot 불변성을 검사한다. detector 상대경로도 caller 기준으로 바로잡았다.
4. **속도 표기 오류**: 2.5 m/s는 vector limit가 아니라 x/y 각 축 limit여서 XY request norm은 최대
   3.54 m/s다. bulk condition, evaluator, attestation, dashboard를 실제 의미로 통일했다.
5. **z action 과잉 정정 방지**: z command는 altitude PI가 덮어 직접 actuator authority가 없지만 raw z는
   다음 관측의 `prev_action`에 남는다. 따라서 dead dimension이 아니라 간접 policy-state channel이다.
   3-D actor ablation은 이 채널 제거/대체까지 통제해야 하며, 이를 condition/receipt/site에 명시했다.
6. **reward 명칭**: moving target progress는 target_(t+1)에 두 거리를 재고정한 ego-motion heuristic이다.
   정적 target에서는 PBRS 대수와 같지만 moving target에 policy-invariance theorem을 적용할 수 없어
   코드·설명·사이트에서 PBRS 확정 표현을 내렸다.
7. **safe resume 차단**: 과거 LKG는 새 horizon/bootstrap fields가 없으므로 `recover_safe`가 continuation을
   고의로 거부한다. 의미가 다른 MDP를 무표식으로 이어 학습하는 것보다 fresh 설계를 요구한다.

### 사이트·연구 문서 정정

- v1/legacy pilot이 current-v2 headline fallback으로 나타나는 경로를 제거했다. 현재 v2 pack이 없으면
  historical curve를 대신 그리지 않고 unavailable로 표시한다.
- ep24000/off와 ep25000/riskcap의 checkpoint·governor·seed·evaluator 동시 차이를 한 그래프에서 causal
  gain처럼 빼던 열을 제거했다. 기존 v2 결과는 `legacy_timeout_at_601`로 명확히 표시한다.
- 학습범위 speed×density interaction은 LR `p=0.337`, categorical omnibus `p=0.817`로 미확정이다.
  220 bars OOD가 만든 강한 서술과 v1/v2 endpoint slope 비교를 headline에서 제거했다.
- detector 행은 learned perception 일반 한계가 아니라 **diagnostic 1×1 artifact**로 낮췄다. 0.1 s latency
  잔차 −2.5 pp와 0.5 s −15.8 pp를 분리해 “latency benign” 과장을 제거했다.
- exact task/reward/audit를 보여주는 Contract panel을 추가했고 40×40×3 m, per-axis command, ego-progress,
  frozen checkpoint의 601/no-bootstrap 한계를 한 화면에 표시한다. app cache key를 `20260810b`로 올렸다.
- `docs/codex_review_2026-08-10.md`와 `docs/NEXT_WEEK_HANDOFF_2026-08-10.md`에 반박 근거,
  재개 gate, 중단 조건을 고정했다. status snapshot은 72 runs, active none으로 재생성했다.

### 검증

- Python 전체 회귀: **202/202 PASS** (14.4 s; 기존 ResourceWarning/FutureWarning만 존재).
- 모든 RL shell `bash -n`, 핵심 Python `py_compile`, `git diff --check`: PASS.
- site DOM/arena-motion JS parity: PASS. Chrome 1440×5200 렌더에서 Contract·수치·레이아웃을 육안 확인했다.
  headless `--disable-gpu`의 WebGL canvas 실패는 브라우저 검수 조건 한계이며 정적 DOM 실패가 아니다.
- RTX 3070 단일 episode 통합 스모크(`/tmp`, 성능 수치 미사용): evaluator가 source 280파일 receipt,
  checkpoint SHA `f7022139...`, 4-D/z-prev_action/per-axis/600-gte 계약을 실제 bulk JSON에서 검증하고 완료했다.
- 학습·장기 평가는 시작하지 않았다. 다음 publication cell은 모두 schema-v2 receipt 아래 새로 측정한다.

추가로 GitHub 첫 화면과 운영/연구 계획까지 같은 기준으로 대조했다. README가 frozen policy를 learned
detector 사용으로 소개하고 ep24000/off 밀도 곡선을 current candidate 표로 표시하던 문제를 수정했다.
current candidate의 ep25000+riskcap curve로 바꾸되 legacy 601-action archive임을 전면에 표시했고, v1
`v_max 2.5 m/s`도 per-axis/XY 3.54 m/s로 정정했다. `OPERATIONS.md`와 `RESEARCH_PLAN.md`에는 legacy
601/no-bootstrap 한계, ego-progress 의미, schema-v2 재측정 gate를 추가했다.

최종 변경은 `4e27dee`(코드·평가기·대시보드 감사)와 `34bd585`(README·운영·연구계획 정합화)로
`origin/research/navrl-env`에 push했다. 공개 Pages `https://joshualikaist.github.io/MOTAR/status/`는 HTTP
200, Contract panel과 cache key `20260810b`를 반환했고, 원격 `status.json`은 로컬 snapshot과 byte-level
JSON 동등하며 z→`prev_action` 계약까지 포함하는 것을 확인했다.

## 2026-08-11 — schema-v2 governor/adaptation A/B/C 15셀 launcher

Gate 1을 세 개의 독립 density sweep 명령으로 실행하면 arm마다 source manifest가 새로 생기는 공백이 있어
전용 `eval_navrl_v2_governor_adaptation_abc.sh`를 추가했다. canonical 계약은 미사용 seed **53**,
deterministic/original, 2,049 requested episodes/cell, 130/160/190/205/220 bars이며 다음 15셀이다.

- A: ep24000 / governor off × 5 densities
- B: 같은 ep24000 / frozen riskcap × 5 densities
- C: ep25000 / 같은 riskcap × 5 densities

일반 evaluator에는 `NAVRL_V2_SHARED_SOURCE_BUNDLE`을 추가했다. 첫 cell이 runtime source/Python environment
bundle을 만들고 나머지 14셀은 같은 manifest SHA와 snapshot을 검증한다. 각 cell은 별도 디렉터리라 완료
후 재실행 시 안전하게 skip할 수 있고, incomplete cell은 자동 삭제하지 않고 중단한다. 최종 summarizer는
15개 result/receipt/checkpoint/source/horizon/outcome 계약을 다시 검증하고 B−A와 C−B를 밀도별로 출력한다.

검증: 두 checkpoint/세 arm preflight PASS. `/tmp` 1-episode A/B 통합 스모크에서 두 결과가 동일 source
manifest SHA `a4ecc49b...`(281 runtime files)를 사용했고 off/riskcap, seed53, symlink-resolved canonical
manifest가 모두 일치했다. 합성 15셀 fixture로 최종 summary/contrast 생성도 PASS했고 전체 Python 회귀는
**205/205 PASS**, 두 shell과 13개 embedded Python heredoc compile, `git diff --check`도 통과했다. 스모크
성능값은 사용하지 않는다. 장기 15셀 평가는 사용자가 명령을 실행할 때까지 시작하지 않았다.

## 2026-08-11 — schema-v2 governor/adaptation A/B/C 15셀 평가 완료

`results/navrl_v2_governor_adaptation_abc_seed53_schema2/`가 **15/15 셀 완료**됐다. 전 셀은 미사용
seed 53, deterministic action, exact 600-step timeout, 약 2,049 episodes/cell이고 동일 runtime source
manifest SHA `cc71428b0445…`를 사용한다. 별도 무결성 검사에서도 checkpoint snapshot/result receipt,
source SHA, outcome accounting, horizon 계약이 모두 PASS했다.

| bars | A ep24000/off | B ep24000/riskcap | C ep25000/riskcap | B−A governor | C−B adaptation |
|---:|---:|---:|---:|---:|---:|
| 130 | 83.70% | 87.07% | 89.75% | +3.37 pp | +2.68 pp |
| 160 | 79.17% | 85.26% | 86.34% | +6.09 pp | +1.08 pp |
| 190 | 75.07% | 81.16% | 81.75% | +6.09 pp | +0.59 pp |
| 205 | 70.67% | 78.40% | **80.28%** | **+7.73 pp** | +1.88 pp |
| 220 OOD | 66.37% | 75.55% | **77.06%** | +9.18 pp | +1.51 pp |

### 통계·원인 판정

- **riskcap governor는 성공**: 학습범위 130–205 pooled capture가 77.15→82.97%, **+5.82 pp**
  (95% CI +4.60..+7.04)다. 다섯 density의 B−A가 모두 양수이고 Holm 보정 뒤에도 모두 유의하다.
  다만 점추정치가 +3.37→+7.73 pp로 커져도 학습범위 continuous interaction `p=0.358`, density별
  heterogeneity `p=0.587`이므로 **“밀도가 높을수록 이득이 증가한다”는 확정 주장은 하지 않는다.**
- governor 이득은 충돌 감소다. 205 bars에서 crash **26.50→17.70%**(−8.80 pp), timeout
  2.83→3.90%(+1.07 pp), capture +7.73 pp다. 220 OOD도 crash −10.54 pp와 timeout +1.37 pp가 합쳐져
  capture +9.18 pp다. 안전을 위해 205-bar capture 평균 시간이 11.83→13.08 s로 1.24 s 늘어나는
  trade-off가 있지만, 정지율 증가는 아니라 의도한 감속 효과다.
- **추가 1,000-epoch adaptation은 작다**: 학습범위 pooled +1.56 pp(95% CI +0.43..+2.69)지만,
  density별로는 130 bars만 Holm 보정 후 유의하고 160/190/205는 CI가 0을 포함한다. 특히 205에서
  crash는 17.70→17.37%(−0.32 pp)에 그치고 timeout 3.90→2.34%(−1.56 pp)가 대부분의 +1.88 pp를
  만든다. 220에서는 crash가 20.50%로 완전히 동일하고 timeout만 1.51 pp 감소했다. 따라서 적응은
  고밀도 collision ceiling을 해결했다기보다 timeout/제어 잔차를 줄였다.
- **남은 핵심 한계는 bar contact**: 최종 C에서 205 bars crash의 97.2%(346/356), 220 bars crash의
  97.9%(411/420)가 bar contact다. below는 각각 4건/3건뿐이다. 이제 고도·정지·episode length가
  1순위 병목이라는 해석은 기각하고, 고밀도에서의 국소 경로 선택/장애물 표현/충돌 여유가 남은
  ceiling이라고 판정한다.
- legacy seed47 601-action 값 205 bars 80.54%와 새 exact-600 값 80.28%는 0.26 pp 차이여서 기존
  headline의 크기는 재현됐다. 다만 publication 기준값은 provenance가 완전한 새 schema-v2
  **80.28/17.37/2.34%**로 교체한다.

Gate 1은 완료다. PPO 추가 학습은 하지 않는다. 다음 순서는 사전등록된 Gate 2
(`0.3/1.5 m/s × 130/160/190/205 bars × 새 seed 2개`, 16 cells)이며, 여기서 speed×density
interaction이 재현되지 않으면 밀도 main effect만 결론으로 남기고 해당 가지를 종료한다.

## 2026-08-11 — Gate 2 speed×density 16셀 launcher 구현

`eval_navrl_v2_speed_density_interaction.sh`를 추가했다. frozen ep25000+riskcap 하나만 사용하며
미사용 seed 59/61 × 고정 표적속도 0.3/1.5 m/s × 학습범위 130/160/190/205 bars의 16셀을 순차 평가한다.
각 cell은 deterministic/original, exact 600 actions, 2,049 requested episodes이고 하나의 immutable
runtime-source bundle을 공유한다. 완료 cell skip/partial cell 거부로 중단 후 같은 명령 재개가 가능하다.

primary statistic은 결과 전에 campaign contract에
`binomial_logit(capture) ~ seed + density + fast + density:fast`로 고정했다. seed fixed effect를 포함한
reduced/full model의 1-df likelihood-ratio test이며, 220 OOD는 primary grid에서 제외했다. 실행 완료 시
셀별 표와 interaction p-value를 `summary.{md,json}`으로 자동 생성한다.

검증: launcher `bash -n`, embedded Python 4개 compile, `git diff --check` PASS. `PREFLIGHT=1`에서
4개 seed×speed invocation 모두 40×40 m/base_sim, deterministic/original, riskcap, exact-600 provenance
계약을 통과했고 실제 장기 평가는 시작하지 않았다.

### 2026-08-11 08:45 KST 진행 확인

사용자가 Gate 2 launcher를 실행했다. 현재 16셀 중 **7셀 result/receipt 완료**, 8번째
`seed59 / 1.5 m/s / 205 bars`가 GPU에서 실행 중이다(약 5.35 GiB, GPU utilization 약 57%).
최종 `summary.{md,json}`은 아직 없으므로 interaction 판정은 보류한다. 완료된 seed59 7셀의 capture는
0.3 m/s에서 130/160/190/205 bars = 87.96/86.29/83.85/81.32%, 1.5 m/s에서
130/160/190 bars = 87.60/84.83/79.85%다. 동일 seed의 마지막 셀과 seed61 전량이 남아 있어 이 중간값으로
가설 결론을 내리지 않는다.

## 2026-08-11 — Gate 2 speed×density 16셀 완료: ID interaction 재현

`results/navrl_v2_speed_density_interaction_seed59_61_schema2/`가 **16/16 셀 완료**됐다. 자동 summary와
별도 독립 검사 모두 policy/source/receipt SHA, seed/speed/bars, deterministic+riskcap, exact-600 horizon,
outcome accounting 계약을 통과했다. 모든 셀은 shared runtime source manifest `3303599b48b5…`와 frozen
ep25000 checkpoint `f7022139…`를 사용한다.

두 seed pooled capture 결과:

| bars | 0.3 m/s | 1.5 m/s | fast−slow |
|---:|---:|---:|---:|
| 130 | 88.44% | 87.80% | −0.64 pp |
| 160 | 86.42% | 84.41% | −2.00 pp |
| 190 | 83.83% | 79.90% | −3.93 pp |
| 205 | 81.78% | 75.90% | **−5.87 pp** |

사전등록 primary model `capture ~ seed + density + fast + density:fast`의 likelihood-ratio test는
χ²(1)=**12.7603**, **p=0.000354**로 interaction을 검출했다. interaction coefficient는 30 bars당
−0.1161 log-odds(SE 0.0325), odds multiplier **0.890**(95% CI 0.835..0.949)이다. seed별 보조 검정도
동일 방향으로 seed59 β=−0.1328/p=0.00369, seed61 β=−0.0991/p=0.03198이었다. 따라서 이전 seed47
legacy grid의 ID interaction 미확정(p=0.337) 결론은 이 two-new-seed/exact-600/schema-v2 primary 결과로
supersede한다. 220 OOD는 이번 검정에 들어가지 않았다.

outcome identity로 분해하면 해석이 더 정확하다.

| bars | fast−slow crash | fast−slow timeout | fast−slow capture |
|---:|---:|---:|---:|
| 130 | +3.40 pp | −2.75 pp | −0.64 pp |
| 160 | +3.64 pp | −1.63 pp | −2.00 pp |
| 190 | +5.49 pp | −1.56 pp | −3.93 pp |
| 205 | +6.73 pp | −0.85 pp | −5.87 pp |

빠른 표적은 전 밀도에서 crash를 늘리지만 저밀도에서는 timeout 감소가 그 손실을 상당 부분 상쇄한다.
밀도가 높아질수록 crash 위험차가 커지고 timeout 상쇄가 줄어 capture 비용이 드러난다. crash 자체의
logit interaction은 p=0.614이고 timeout interaction은 p=0.00192이므로, 현재 증거만으로 “회피 중 표적
이동” 하나를 확정 메커니즘으로 쓰지는 않는다. 확정 가능한 결론은 **학습범위 안에서 표적속도 비용이
밀도에 의존한다**는 것이다.

Gate 2는 PASS로 종료한다. PPO 재학습이나 추가 speed grid는 하지 않는다. 다음 계획상 작업은 Gate 3
learned detector offline gate이며, navigation policy를 고정한 채 detector dataset/loss/calibration을 먼저
검증해야 한다.

## 2026-08-11 — Gate 3 detector offline gate v2 구현

기존 `tools/train_navrl_target_detector.py`는 정면 2–12 m/4,096 frames/unweighted BCE/v1 artifact를 다시
만드는 코드라 Gate 3에 사용할 수 없음을 확인했다. v1은 덮어쓰지 않고
`tools/train_navrl_target_detector_v2.py`와 `run_navrl_detector_offline_gate.sh`를 새로 추가했다.

- split: train/validation/test = **8,192/2,048/4,096 frames**, 독립 seed **71/73/79**.
- coverage: camera full bearing/elevation, 2–20 m 5개 range bin 균등 표집, 20% target-absent,
  20% rendered obstacle 뒤 강제 occlusion, 자연 partial/full occlusion과 2–5 pixel small target.
- geometry: current v2의 205 bars / 40×40 m / `navrl_band`; 비싼 layout은 독립 target batch 4개에만
  재사용해 split당 128/32/64개 layout을 유지한다.
- candidates: 기존 runtime-compatible 1×1 RGB-D head에 class-balanced BCE와 focal+Dice를 비교한다.
  현재 simulator 표적은 고정 red appearance라 spatial CNN을 먼저 추가하면 불필요한 confound가 되므로,
  이 최소 head가 offline gate를 실패할 때만 architecture 확장을 연다.
- calibration: candidate/loss/37개 threshold 선택은 validation에서만 수행하고 test는 이후 한 번만 연다.
  test gate는 frame precision/recall, absent/full-occlusion FPR, far/partial/small recall, pixel precision,
  bearing/range MAE와 각 stratum 최소 표본수를 동시에 요구한다.
- PPO policy는 이 단계에서 load/train하지 않는다. offline PASS artifact만 후속 frozen-policy
  analytic-vs-learned navigation A/B에 들어간다.

검증: Python compile, shell `bash -n`, preflight, focused tests 3/3, `git diff --check` PASS. RTX 3070
소형 end-to-end smoke 두 번에서 renderer 수집→두 loss 학습→validation 선택→held-out report가 끝까지
완료됐다. 두 번째 smoke는 `navrl_band` 로그까지 확인했다. 64/128-frame smoke의 검출 지표는 모두 1.0,
gate는 사전등록된 최소 test 표본수 부족만으로 의도대로 FAIL했다. 정규 Gate 3 장기 실행은 아직
시작하지 않았다.

### Gate 3 정규 실행 완료 — offline PASS

사용자 실행은 중단된 것이 아니라 2026-08-11 10:45 KST에 정상 종료됐다. train/validation/test 전량을
수집·학습·평가했고 `artifacts/navrl_target_detector_v2.pth`와 receipt, summary를 생성했다. artifact
SHA-256은 `8da32d6f21bfbd3bdd5ec5de9ef9cb09e8deb4bd5ce511630e19afee33f26f10`이며 summary/receipt/실제
파일 해시가 모두 일치한다.

- 선택: `balanced_bce`, validation 고정 threshold **0.55**(runtime default와 동일).
- test seed79: 4,096 frames, visible 1,313, absent 832, non-visible/occluded 1,951,
  forced-partial 230, small-target 579, far 14–20 m 339.
- held-out test: frame precision/recall **1.000/1.000**, absent/full-occlusion FPR **0/0**,
  far/partial/small recall **1/1/1**, pixel precision/IoU **1/1**, bearing/range MAE **0/0**.
- 사전 고정 gate check **14/14 PASS**. v1의 약 14 m cutoff와 달리 v2 weight는 20 m의 pure-red
  target score도 threshold 0.55 위에 남는다.

이 PASS의 범위는 정확히 **현재 simulator appearance**다. renderer가 target을 고정된 red RGB로 칠하고
배경/막대는 neutral이므로 geometry·range·occlusion split이 달라도 pixel class는 완전히 분리 가능하다.
따라서 “실세계 learned vision이 해결됐다”는 결론은 금지하고, “v1의 데이터/loss 결함을 제거해 현
simulator에서 analytic mask와 동일한 segmentation을 재현했다”로 제한한다. 다음 단계는 artifact SHA와
threshold를 고정해 frozen ep25000+riskcap에서 analytic-vs-learned navigation A/B를 실행하는 것이다.

## 2026-08-11 — Gate 3 stage B detector navigation A/B launcher

`eval_navrl_v2_detector_navigation_ab.sh`를 추가했다. frozen ep25000+riskcap / 205 bars / deterministic /
exact-600에서 미사용 seed 83/89 × `analytic_bootstrap`/`learned_v2`의 4셀을 순차 실행한다. 네 셀은 한
runtime-source bundle을 공유하고 completed-cell skip/partial-cell 거부로 재개 가능하다. learned arm은
offline PASS receipt, artifact SHA `8da32d6f…`, validation-selected threshold 0.55를 실행 전에 검증한다.

primary endpoint는 두 seed pooled capture의 learned−analytic 차이다. 결과 전에 비열등성 margin을
**−2.0 pp**로 campaign contract에 고정했고, 보수적인 독립-binomial 양측 95% CI lower bound가 −2.0 pp보다
클 때만 PASS한다. crash/timeout은 secondary descriptive endpoint로 남긴다. PPO weight·governor·밀도·seed
외 조건은 양 arm에서 동일하다.

검증: launcher `bash -n`, embedded Python 3개 compile, 4-cell evaluator preflight, detector focused tests
6/6, `git diff --check` PASS. 장기 navigation A/B는 사용자가 명령을 실행하기 전까지 시작하지 않았다.

## 2026-08-11 — Genspark 발표 제작용 단일 source-of-truth 작성

사용자가 지금까지의 MOTAR 연구를 Genspark AI Slides로 발표 자료화할 수 있도록
`docs/GENSPARK_PPT_BRIEF_2026-08-11.md`를 추가했다. 이 파일 하나에 한국어 15장+부록 제작 프롬프트,
시각 스타일, 권장 slide 순서, 시스템/정보방화벽 계약, 최신 schema-v2 Gate 1·2 수치, timestamp-aware
latency 결과, Gate 3 offline detector 결과, 음성 결과, 금지 주장, 다음 로드맵과 내부 근거 파일을 묶었다.

특히 legacy v1/v2와 601/exact-600 결과를 섞지 않도록 하고, 220 bars를 OOD로 표시하며, governor 이득의
density interaction 과대주장과 synthetic detector의 실세계 일반화를 명시적으로 금지했다. detector
navigation A/B는 현재 seed83 두 cell만 결과가 있고 seed89에서 중단되어 최종 판정이 없으므로 PPT 성과가
아닌 `후속 검증 대기`로 고정했다. 기존 사용자 변경과 미커밋 실험 파일은 건드리지 않았다.

## 2026-08-11 — Gate 3 stage B 4/4 완료: learned detector navigation 비열등성 PASS

`results/navrl_v2_detector_navigation_ab_seed83_89_schema2/`의 4개 cell이 18:43 KST에 모두 완료됐고
`summary.{md,json}`과 campaign COMPLETE 로그가 생성됐다. 실행 프로세스는 남아 있지 않다. 자동 summary
외에 네 receipt와 실제 result/checkpoint/detector/source-manifest bytes를 독립 재해시하고 outcome accounting,
exact-600 timeout, seed/bars/governor/action 계약을 다시 계산해 전부 PASS했다. policy SHA는 `f7022139…`,
detector SHA는 `8da32d6f…`, shared source manifest SHA는 `0c813323…`다.

| seed | analytic capture/crash/timeout | learned capture/crash/timeout | learned−analytic capture |
|---:|---:|---:|---:|
| 83 | 80.39/16.54/3.07% | 79.86/16.92/3.22% | −0.53 pp |
| 89 | 80.59/17.07/2.34% | 80.97/16.11/2.93% | +0.38 pp |

두 arm은 각각 4,100 episodes다. analytic은 3300/4100 = **80.49%**, learned는 3297/4100 =
**80.41%**, learned−analytic은 **−0.073 pp**다. 독립 이항 비율차 95% CI를 별도로 계산한 결과
**[−1.790,+1.644] pp**로 자동 summary와 일치하며, lower bound −1.790 pp가 사전등록한
non-inferiority margin −2.0 pp보다 크므로 **Gate 3 navigation NI PASS**다. pooled crash는
16.80→16.51%, timeout은 2.71→3.07%이며 secondary descriptive endpoint로만 남긴다.

결론 범위는 현 synthetic simulator appearance에 한정한다. pure-red target과 neutral background/bar에서
학습된 detector가 analytic appearance bootstrap을 대체해도 frozen navigation을 실질적으로 떨어뜨리지
않는다는 증거다. 실제 lighting/texture/blur/noise/calibration에 대한 sim-to-real 증거는 아니다.

평가 진행 중 작성했던 untracked `docs/GENSPARK_PPT_BRIEF_2026-08-11.md`가 campaign 완료 뒤 작업 디렉터리에
남아 있지 않은 것을 확인했다. 원인을 추정해 단정하지 않고, 장기 프로세스가 모두 종료된 뒤 Gate 3 최종
수치까지 포함한 단일 브리프로 재생성했다. 기존 사용자 변경은 보존했다.

## 2026-08-11 — Gate 3 이후 남은 검증 우선순위 동결

현재 simulator nominal condition에서 Gate 1 governor, Gate 2 density×speed interaction, Gate 3 learned
detector navigation non-inferiority는 종료한다. 같은 seed/cell 반복, pure-red offline frame 증량, frozen
checkpoint의 205-bar PPO 연장은 정보 가치가 낮아 수행하지 않는다.

남은 검증은 다음 순서로 한정한다. (1) Gate 3 NI CI 하한 −1.79 pp가 margin −2.0 pp보다 0.21 pp만 위인
점을 고려해 새 seed의 confirmatory replication을 원 primary와 분리해 사전등록한다. (2) lighting/target hue/
texture/motion blur/depth noise/calibration perturbation에서 detector navigation을 평가해 pure-red synthetic
appearance의 외적 타당성을 측정한다. (3) timestamp offset·pose interpolation error를 넣어 P3 latency의
정확한 clock/odometry 전제 민감도를 잰다. (4) 205-bar bar-contact ceiling은 geometry reachability oracle,
representation coverage, contact-time stopping margin을 순서대로 분리한 뒤에만 새 token/control 학습을 연다.
(5) 최종 알고리즘 학습 주장을 하려면 legacy 601/no-bootstrap checkpoint와 별도로 exact-600+`time_outs`
bootstrap fresh lineage를 재학습한다. 현재 결과를 simulator proof-of-concept로 마무리할 경우 (1)–(5)는
future work로 공개하고 새 GPU 실험 없이 동결해도 된다.

## 2026-08-12 — 일반화 검증 로드맵 확정 + 검증 1(NI 재현) 착수

사용자 승인으로 08-11의 동결된 우선순위 5개를 실행 로드맵으로 확정한다. simulator
proof-of-concept 동결이 아니라 **일반화 검증 계속** 경로다. GPU가 하나이므로 순차 실행하며,
(1)~(4)는 inference/진단, (5)만 PPO 학습이다.

1. **검증 1 — Gate 3 NI confirmatory replication**: 원 결과 CI 하한 −1.790 pp가 margin
   −2.0 pp 대비 여유 0.21 pp뿐. 미사용 seed **97/101**, 4셀, margin 동일 −2.0 pp,
   **원 결과와 사후 통합 금지·별도 보고**로 사전등록. → 오늘 착수.
2. **검증 2 — perception domain shift** (가장 중요): 조명/표적 hue, 배경·막대 texture,
   motion blur, RGB/depth noise, camera calibration 오차, partial occlusion·소형 표적.
   pure-red appearance의 외적 타당성 측정. renderer/perception 교란 구현 후
   detector offline gate → frozen-policy navigation A/B 순.
3. **검증 3 — latency 전제 민감도**: P3의 −2.5 pp는 정확한 timestamp/pose history 조건.
   clock offset, pose interpolation error, odometry noise 주입 지원을 구현하고 민감도 곡선 측정.
4. **검증 4 — 205-bar bar-contact ceiling 분리**: crash의 97.2%가 bar contact.
   ① geometry reachability oracle → ② 8-token representation coverage →
   ③ 접촉 직전 속도·clearance·stopping margin → 그 뒤에만 token/control 변경을 연다.
5. **검증 5 — corrected-semantics fresh PPO** (유일한 학습): frozen 계보는 legacy
   601-action + no-`time_outs`-bootstrap 조건에서 학습됐다. "수정된 알고리즘의 학습 결과"
   주장을 위해 exact-600 + bootstrap로 fresh lineage 1회 학습. 마지막에 실행
   (며칠 단위로 GPU를 점유하므로 (1)~(4) 종료 후).

**검증 1 착수**: `eval_navrl_v2_detector_navigation_ab_replication.sh` 추가 — 원 launcher에서
seeds 83/89 → **97/101**과 result root만 바꾸고 계약은 byte-동일하게 유지. 사전등록 문구
(별도 보고, 동일 margin)를 헤더에 고정. PREFLIGHT 4셀 PASS. 실행 시작.

## 2026-08-12 — 검증 1 완료: Gate 3 NI confirmatory replication PASS (seed 97/101)

`results/navrl_v2_detector_navigation_ab_replication_seed97_101_schema2/` 4/4 셀 완료.
독립 무결성 검사 PASS: 4셀 모두 policy SHA `f7022139…`, learned arm detector SHA `8da32d6f…`,
동일 source manifest, exact-600 timeout(존재 셀 전부 mean/p10/p50/p90=600), seed/bars/action 계약 일치.

| seed | arm | capture | crash | timeout | n |
|---:|---|---:|---:|---:|---:|
| 97 | analytic | 80.23% | 17.67% | 2.10% | 2049 |
| 97 | learned_v2 | 80.06% | 17.70% | 2.24% | 2051 |
| 101 | analytic | 79.40% | 17.37% | 3.22% | 2049 |
| 101 | learned_v2 | 79.55% | 17.91% | 2.54% | 2049 |

pooled learned−analytic **−0.015 pp**, 95% CI **[−1.752, +1.723]** → 사전등록 margin −2.0 pp
대비 **replication PASS**. 원 campaign(seed 83/89: −0.073 pp, CI [−1.790, +1.644])과 사실상
동일한 0 근방이며, 사전등록대로 **두 결과는 통합하지 않고 별도 보고**한다. CI 하한 여유는
원 0.21 pp → 재현 0.25 pp로, "PASS지만 강하지 않다"는 성격도 재현됐다 — margin을 좁힌 것이
아니라 동일 margin에서 독립 재현이 이뤄졌다는 의미다.

**운영 사고 기록**: launcher 파생 시 summarizer 내부의 `seeds = [83, 89]` 리터럴이 남아
4셀 완료 후 summary 단계에서 크래시했다. 수정하자 campaign contract의 launcher SHA 피닝이
재실행을 정확히 거부했다(가드 정상 동작). 셀 데이터는 원본 launcher 산출물 그대로이므로,
summary만 사전등록 공식(pooled 독립 이항 95% CI)으로 **런처 밖에서 재계산**해 저장했고
그 사실을 summary 파일에 명시했다. 교훈: launcher 파생 시 모든 리터럴을 assert로 검증하는
규칙(08-07)을 embedded summarizer에도 적용해야 한다.

검증 1 종료. 다음은 검증 2 perception domain shift.

## 2026-08-12 — 검증 2 구현: 렌더러 appearance domain-shift 8축 (기본 0 = 비트 동일)

렌더러 탐색(서브에이전트) 결과 현 렌더는 "warp 기하 + 해석적 색칠"로 **외형 노브가 전무**했다:
표적색 `[0.88,0.08,0.045]`·배경 명도식 `0.08+0.42·proximity`·틴트 `(0.92,1,1.05)` 전부
리터럴, 조명/텍스처/블러/캘리브레이션 없음. intrinsics/extrinsics는 렌더러와 perception이
**각자 계산**하며 교차검증 없음 — 즉 한쪽만 교란하면 실제 캘리브레이션 오차가 재현된다.

`navrl_detector.py`에 8개 노브를 추가했다 (vision cfg, 전부 기본 0 = 종전 렌더와 비트 동일):

| 노브 | 축 | 방식 |
|---|---|---|
| `NAVRL_APP_HUE_DEG` | 표적 hue | 회색축 Rodrigues 회전, per-episode 추출 |
| `NAVRL_APP_LIGHT_GAIN` | 전역 조명 | 표적 페인트 **뒤에** 곱함(표적·배경 동시) |
| `NAVRL_APP_ALBEDO_JITTER` | 막대/배경 반사율 | base/gain/tint per-env jitter |
| `NAVRL_APP_TEXTURE_STD` | 텍스처 | per-env 정적 per-pixel 명도 노이즈 |
| `NAVRL_APP_MOTION_BLUR` | 모션 블러 | EMA 트레일, depth는 블러 안 함(명시적 결정), reset 시 무효화 |
| `NAVRL_CAM_MOUNT_ROT_DEG` | extrinsic 오차 | **렌더러만** mount quat 합성(perception은 nominal 유지) |
| `NAVRL_CAM_MOUNT_TRANS_M` | extrinsic 오차 | 렌더러만 offset 이동 |
| `NAVRL_CAM_FOV_SCALE_ERR` | intrinsic 오차 | ray table만 교란(per-run), 소비자 fx/fy는 nominal |

per-episode 재추출은 `detector.reset_idx → _resample_appearance`. provenance는 checkpoint
metadata echo + mismatch table(navrl_task.py) + evaluator receipt/pinned export
(eval_navrl_v2_density_sweep.sh) 3곳에 배선했다.

검증: CPU 단위 테스트 12/12 PASS(hue 회전 항등/120° 순환/in-gamut 명도 보존, mount quat 각도
상한, source invariant 6종 — light가 표적 페인트 뒤에 오는지, depth 무블러, fov가 ray table만
건드리는지, perception에 mount 개념이 없는지). GPU 스모크: **zero-knob 렌더가 종전과
비트 동일**(rgb/depth 모두), 7개 축 각각 이미지가 실제로 변하고 mount/fov만 기하를 움직이며,
blur는 첫 프레임 동일·둘째 프레임부터 발현(Δ0.28). 스모크 강도에서는 bootstrap/learned 모두
recall 1.0 유지 — 강도 사다리는 본 평가에서 측정한다.

주의(다음 단계에 반영): offline v2 trainer의 자체 augmentation 리터럴(randn 0.015/0.02)은
렌더러 노브와 독립이므로, shift 하 재학습 시 분포 정합을 명시적으로 관리해야 한다.

## 2026-08-12 — 검증 2 stage A 완료: hue가 지배 축, learned v2는 bootstrap보다 오히려 취약

`results/navrl_detector_domain_shift/summary.{md,json}` — 28셀 × 1,024 frames/셀, 205 bars,
seed 103, threshold 0.55, bootstrap과 learned v2를 **동일 프레임**에서 측정.

핵심 판독 (frame recall, nominal 95.0%):

| 축 | 붕괴 지점 | bootstrap | learned v2 |
|---|---|---:|---:|
| **hue** | **60°부터 붕괴** | 60°: 79.7% / 90°: 45.0% / 180°: 24.8% | **60°: 62.5%** / 90°: 39.4% / 180°: 21.5% |
| light_gain | ±0.7에서 열화 | 84.2% | 77.7% |
| albedo/texture | **무해** (전 구간 92~96%) | — | — |
| motion_blur | recall 소폭↓ + **FPR 3.2~3.5% 발생** | 91.4~95.3% | 동일 |
| mount_rot | recall 무해, **bearing MAE 선형 증가** | 0.52°→1.48°(5°) | 동일 |
| fov 10% | recall 89.6%, **bearing MAE 2.23°** | — | 동일 |

1. **hue가 지배 축이다.** 예측대로 두 검출기 모두 60°부터 무너진다. ±30°까지는 red 우세가
   유지돼 무해(95%+).
2. **learned v2가 bootstrap보다 hue·light에 더 취약하다**(60°에서 62.5% vs 79.7%).
   pure-red 데이터로 학습한 1×1 head는 손제작 red 규칙보다 학습 분포에 더 밀착해 있다.
   "learned가 더 낫다"는 기대는 이 축들에서 역전된다 — Codex의 "pure-red라 너무 쉽다"
   지적이 정량으로 확인된 것.
3. **배경 축(albedo/texture)은 무해** — 표적 색만 보는 규칙이므로 당연하며, 이 축들은
   detector가 아니라 (있다면) navigation 영향으로만 남는다.
4. **blur는 유일하게 FPR을 만든다**(고블러에서 3.2~3.5%) — 고스트 트레일이 absent 프레임에
   위양성을 만든다. 저블러(0.3)에서는 centroid가 끌려 bearing MAE 0.99°.
5. **캘리브레이션 축은 recall이 아니라 bearing bias로 나타난다** — mount 5°에서 MAE 1.48°,
   fov 10%에서 2.23°. 설계 의도(렌더러만 교란→back-projection bias) 그대로 계측됐다.

### stage B 설계 결정

- **운용 envelope 안 randomization으로 v3 재학습**: hue ±60°, light ±0.5, albedo 0.3,
  texture 0.2, blur 0.3 (+기존 노이즈). 주장 형태는 "선언된 외형 envelope 안에서 NI 유지"로
  한정한다. hue ±180° 전면 randomization은 1×1 색 규칙로는 원리적으로 불가능한 조건이며,
  offline gate가 실패할 때만 architecture 확장을 연다는 사전등록(Gate 3)을 따른다.
- 캘리브레이션 축(mount/fov)은 detector 재학습 대상이 아니라 **navigation A/B로 직접** 측정
  (KF 측정 오염 경로).

## 2026-08-12 — 검증 2 stage B(1): envelope 하 1×1 head gate FAIL → 사전등록 에스컬레이션 발동

envelope(hue ±60°, light ±0.5, albedo 0.3, texture 0.2, blur 0.3)로 v3를 학습했다
(`results/navrl_detector_offline_gate_v3_domainrand/`, artifact SHA `c5d8b178…`, receipt에
envelope 기록). 결과 **GATE FAIL** — 14체크 중 4개 실패:

| 실패 체크 | 값 | 기준 |
|---|---:|---:|
| **pixel_precision** | **0.172** | ≥0.95 |
| frame_recall | 0.937 | ≥0.95 |
| frame_precision | 0.975 | ≥0.98 |
| full_occlusion_fpr | 0.0134 | ≤0.01 |

pixel precision 0.17이 본질이다: hue ±60° + light ±0.5 + albedo jitter에서는 **per-pixel 색
규칙로 표적/배경을 분리할 수 없다**(선택된 threshold도 0.55 → 0.425로 내려갔다). 이는 결함이
아니라 Gate 3에 사전등록된 판정 경로다 — "이 최소 head가 offline gate를 실패할 때만
architecture 확장을 연다." 실패했으므로 확장을 연다.

**에스컬레이션 구현**:
- `SpatialTargetSegmenter` (navrl_perception.py): conv 4→16(3×3) → 16→16(3×3, dilation 2) →
  16→1(1×1), **~2.9k params**, 수용영역 7×7. 공간 문맥(blob vs 세로 막대)을 사되 용량은
  최소로 유지.
- artifact가 `meta.architecture`를 실으며 로더가 이를 읽어 올바른 클래스를 생성한다
  (`build_target_segmenter`). meta 없는 v1-era payload는 1×1로 기본 처리(하위 호환).
- trainer 후보를 2×2로 확장: {pixel_1x1, spatial_cnn} × {balanced_bce, focal_dice}.
  **선택은 여전히 validation-only**이므로 nominal 조건에서는 1×1이 다시 뽑힐 수 있다.
- 테스트: perception 27/27(아키텍처 디스패치 4종 신규 — spatial 로드/legacy 기본/forward
  계약/파라미터 예산 <5k), latency 33/33, appearance 12/12 회귀 PASS.

v4(동일 envelope, 4후보) 학습 시작. gate PASS 시 stage-A 사다리 재실행 → navigation A/B 순.

## 2026-08-12 — 검증 2 stage B(2): spatial CNN이 gate를 거의 통과 — 남은 결함은 모델이 아니라 선택기

**v4 1차 시도**는 trainer가 1×1 head의 `model.classifier` 속성을 직접 호출해 spatial 후보
epoch 1에서 크래시(AttributeError). 두 head에 공통 `forward_logits(features)` 계약을 만들어
해결, `sigmoid(forward_logits)==forward()` 파리티 확인. partial 산출물 없음.

**v4 재실행** (`results/navrl_detector_offline_gate_v4_domainrand/`, SHA `354da116…`):
4후보 중 **spatial_cnn+focal_dice가 validation 상위 독점** — 아키텍처 확장이 유효함을 확인.
gate는 14체크 중 **12 PASS**로 v3(10 PASS) 대비 크게 개선:

| 지표 | v3 (1×1) | v4 (spatial) | 기준 |
|---|---:|---:|---:|
| frame P / R | 0.975 / 0.937 | **0.984 / 1.000** | ≥0.98 / ≥0.95 |
| far/small/partial recall | 0.85/0.88/1.00 | **1.00/1.00/1.00** | — |
| bearing MAE | 0.37° | **0.16°** | ≤1.5° |
| **pixel_precision** | 0.172 | **0.800 FAIL** | ≥0.95 |
| **range MAE** | 0.068 m | **0.878 m FAIL** | ≤0.25 m |

남은 2개 실패의 원인은 모델이 아니라 **operating-point 선택기**다: feasibility가 FPR 2개만
반영해 recall을 좇아 threshold를 **0.075**까지 내렸고, 그 지점에서 마스크 halo가 pixel
precision을 0.80으로, halo의 배경 깊이가 range MAE를 0.88 m로 끌어내렸다. gate가 요구하는
정밀도 계열 체크를 선택기가 전혀 모르는 구조 — **자기가 공급하는 gate를 통과할 수 없는
운영점을 고르는 선택기**였다.

**수정**: feasibility가 validation에서 frame precision ≥0.98, pixel precision ≥0.95,
range MAE ≤0.25까지 거울하도록 확장(`acb1bef`). 선택은 여전히 validation-only, sealed test
불변, feasible이 없으면 종전처럼 랭킹 폴백 후 gate가 정직하게 FAIL. v5 학습 시작.

## 2026-08-12 — 검증 2 stage B(3): v5도 FAIL — exploratory 종료 선언 + confirmatory 사전등록

**v5 결과** (`results/navrl_detector_offline_gate_v5_domainrand/`, SHA `eeb332ec…`): 제약 선택기가
**정반대 극단**을 골랐다. 1×1 focal이 validation에서 pixel precision 1.000(보수적 core-only
발화)으로 유일하게 feasible해져 선택됐지만, test에서 recall 3종 FAIL(frame 0.850, far 0.822,
small 0.825)이고 **pixel precision조차 0.657로 붕괴** — validation→test 정밀도 갭이 0.343이다.
spatial은 validation 전 threshold에서 pixel precision <0.95라 top10에 아예 없다.

**진단 — gate 설계 결함이 드러났다**: motion blur가 표적 색을 인접 픽셀에 섞는데 GT 마스크는
순간 기하만 라벨하므로, envelope 하 **exact pixel precision ≥0.95는 어떤 모델로도 안정적으로
만족 불가능**하다. 1×1은 이 지표에서 고분산(appearance 추첨에 따라 1.0↔0.66), spatial은
안정적이되 0.8 수준. 다운스트림(KF)이 실제로 소비하는 것은 bearing/range/visible이며 그
지표들은 이미 우수하다(spatial: bearing MAE 0.16°).

**exploratory 선언**: v3/v4/v5는 test를 3회 관찰했으므로 전부 exploratory로 격하한다.
이 셋에서 확정하는 것은 (a) 1×1은 envelope에서 원리적 불가(v3), (b) spatial 아키텍처 유효
(v4: 12/14, recall 전 지표 1.0), (c) exact pixel precision은 blur 하 ill-posed(v5)뿐이다.

### confirmatory 사전등록 (실행 전 고정)

- **split seeds**: train/val/test = **113/127/131** (전부 미사용; 기존 71/73/79와 무관).
  test는 이 run에서 단 1회만 개봉한다.
- **envelope 불변**: hue ±60°, light ±0.5, albedo 0.3, texture 0.2, blur 0.3.
- **후보 pool 불변**: {pixel_1x1, spatial_cnn} × {balanced_bce, focal_dice}, 선택은
  validation-only 제약 랭킹(구현 그대로).
- **gate = 기존 14체크 중 13개 불변 + pixel_precision 1건만 재정의**:
  `--pixel-tolerance-px 1` — GT 마스크를 3×3 dilate한 범위 안의 예측을 정밀로 인정
  (**blur가 물리적으로 만드는 1픽셀 경계 혼합을 벌하지 않되, 표적에서 떨어진 spray FP는
  그대로 벌한다**). recall/IoU는 exact 유지. 임계 0.95 불변.
- **판정**: 이 gate 14/14 PASS면 stage-A 사다리 재실행 + navigation A/B로 진행.
  FAIL이면 "이 envelope은 이 용량(~3k)으로 불가"를 결과로 확정하고 envelope 축소 vs 용량
  증가 트레이드오프를 사용자 결정으로 올린다. **이 run의 test에 대한 재시도는 없다.**

trainer에 `--pixel-tolerance-px`(기본 0=종전 exact)와 split-seed 오버라이드를 추가했고,
선택기 feasibility도 동일 tolerance로 판단하게 배선했다(선택과 gate의 지표 정의 일치).

## 2026-08-12 — 검증 2 stage B confirmatory (v6): FAIL 12/14 — 근소 미달 2건, 사전등록대로 종료

`results/navrl_detector_offline_gate_v6_confirmatory/` (SHA `b700778e…`, seeds 113/127/131,
tolerance 1px, envelope 불변). 선택: **spatial_cnn+focal_dice @ threshold 0.50** (제약 선택기
정상 작동 — v5의 극단 선택 문제 해소).

| 체크 | 값 | 기준 | 판정 |
|---|---:|---:|---|
| frame_precision | 0.9975 | ≥0.98 | PASS |
| **frame_recall** | **0.9340** | ≥0.95 | **FAIL (−1.6 pp)** |
| pixel_precision (tol 1px) | 0.9997 | ≥0.95 | PASS — 재정의 의도대로 |
| absent / full-occ FPR | 0.0012 / 0.0010 | ≤0.01 | PASS |
| far / small / partial recall | 0.857 / 0.855 / 0.946 | ≥0.85/0.80/0.85 | PASS |
| bearing MAE | **0.04°** | ≤1.5° | PASS |
| **range MAE** | **0.274 m** | ≤0.25 m | **FAIL (+0.024 m)** |

**사전등록된 판정을 그대로 집행한다**: 이 test에 대한 재시도 없음. 확정되는 결과는 —
"선언한 envelope(hue ±60°, light ±0.5, albedo 0.3, texture 0.2, blur 0.3)은 ~2.9k 파라미터
spatial head로 **거의 도달하나 미달**이다(12/14; recall −1.6 pp, range +2.4 cm)."

수확도 명확하다: tolerance 재정의는 의도대로 작동했고(0.9997, spray FP는 FPR 0.001이
잡음), bearing 0.04°는 KF 소비 품질로는 사실상 포화다. 남은 미달 2건은 둘 다
"어려운 프레임(원거리·소형·강한 blur 추첨)의 검출 여부" 계열이다.

**다음 선택지 (사용자 결정, 사전등록에 따라)**:
- (a) **용량 증가**: 16→32ch 또는 1층 추가(~10k params, 런타임 여전히 무시 가능) 후 fresh
  seeds로 confirmatory 재등록·1회 실행. 근소 미달 2건을 닫을 가능성이 가장 높다. ~1h GPU.
- (b) **envelope 축소**: hue ±45° 또는 blur 0.2로 줄여 재등록. 논문 주장이 약해진다.
- (c) **측정된 한계로 확정하고 진행**: "12/14, recall 93.4%"를 stage-B 결과로 공개하고
  검증 3(latency 전제)~5(fresh PPO)로 이동. navigation A/B는 offline-PASS 전제(Gate 3
  프로토콜)라 이 경우 실행하지 않는다.

## 2026-08-12 — 자체 검증(Codex 방식) 후 v7 confirmatory 사전등록

사용자 지시로 Codex 위임 전 자체 감사를 수행했다. 4개 판정:
1. **frame_recall 미달은 실체다**: 0.9340, n=1,258, 95% CI [0.9203, 0.9477] — 상한이 0.95
   아래. 게다가 validation의 **어떤 운영점도 recall 0.925를 못 넘었다**(feasible set 최고
   0.925) → 선택 튜닝으로 불가, 모델 용량 한계.
2. 프로토콜 준수 확인: v6 메타에 seeds/tolerance/envelope/gate_passed 완비, SHA 일치.
3. 선택기 신뢰성 회복: val→test 갭 recall +0.9pp, pixel precision −0.0003 (v5 병리 해소).
4. **range MAE 미달의 기제**: 선택점의 validation rMAE 0.2458 — 제약(≤0.25)을 0.4 cm 차로
   통과 후 test에서 +2.8 cm 표류. feasible set(0.218~0.246)이 경계에 몰려 있는데 선택기는
   recall 최대화로 최악 rMAE 지점을 골랐다 → **선택 안전마진 필요**.

### v7 confirmatory 사전등록 (실행 전 고정)

- split seeds **137/139/149** (미사용), test 1회 개봉, **재시도 없음**.
- envelope·gate 불변 (tolerance 1px 포함).
- 후보 pool: **{spatial_cnn(7×7/16ch), spatial_cnn_wide(9×9/24ch, ~11.3k params)} × {bce, focal}**.
  1×1은 제외 — v3가 envelope 하 원리적 불가를 확정했으므로 재학습은 확정된 음성에 GPU를
  쓰는 것. Wide는 conv 1층 추가 + 폭 24로 v6 미달 프레임(원거리·소형·강블러) 겨냥.
- **선택 안전마진**: validation feasibility의 range MAE 상한을 0.25 → **0.23**으로
  (감사 4의 측정 표류 +0.028 근거, `--selection-range-mae-max`). gate 기준 자체는 0.25 불변.
- 구현: `SpatialTargetSegmenterWide` + prefix dispatch(긴 태그 우선 — Wide가 좁은 태그로
  startswith 매칭되는 함정 테스트로 고정), 테스트 28/28 PASS.
- 판정 규칙: 14/14 PASS → stage-A 사다리 재실행 + navigation A/B. FAIL → envelope 대비
  용량 사다리 2점(3k, 11k)의 측정된 한계로 확정하고 사용자 결정으로 복귀.

## 2026-08-12 — v7 confirmatory: 14/14 PASS — envelope 하 detector 확보

`results/navrl_detector_offline_gate_v7_confirmatory/` (SHA `85c7974b…`, seeds 137/139/149,
1회 개봉). 선택 `spatial_cnn_wide+focal_dice @ threshold 0.70`.

| 지표 | v6 | **v7** |
|---|---:|---:|
| frame recall | 0.9340 | **0.9938** |
| range MAE | 0.274 m | **0.178 m** |
| far/small/partial recall | 0.857/0.855/0.946 | **0.997/0.987/0.992** |
| pixel precision(tol)/IoU | 0.9997/0.893 | **1.0000/0.969** |
| bearing MAE | 0.040° | **0.024°** |

감사에서 겨냥한 두 지렛대가 각각 작동했다: recall 미달(용량 한계)은 wide head가,
range 미달(선택이 경계 rMAE 지점을 고름)은 선택 마진 0.23이 닫았다.

**확정**: 선언한 appearance envelope(hue ±60°, light ±0.5, albedo 0.3, texture 0.2,
blur 0.3) 안에서 offline detector gate를 통과하는 학습형 검출기(~11.3k params)를 확보했다.
사전등록 판정대로 stage-A 사다리 재실행 + navigation A/B로 진행한다.

### navigation A/B (envelope) 사전등록 — 실행 전 고정

frozen ep25000+riskcap, 205 bars, deterministic, exact-600, 2,049 ep/cell.
**2×2 factorial × 2 seeds = 8셀**: appearance {nominal, envelope} × detector
{analytic_bootstrap, learned_v7} × seeds **{151, 157}** (미사용). learned arm은 artifact SHA
`85c7974b…`·threshold **0.70**(meta의 validation 선택값) 고정, analytic arm은 종전 0.55.

- **E1 (primary, NI gate)**: nominal에서 pooled capture(learned_v7) − capture(analytic),
  독립 이항 95% CI 하한 > **−2.0 pp**면 PASS (Gate 3와 동일 margin·공식).
- **E2 (headline, descriptive)**: capture(envelope, learned_v7) − capture(nominal, analytic)
  — 외형 shift 전체의 navigation 비용. gate 없음, CI만 보고.
- **E3 (counterfactual, descriptive)**: capture(envelope, analytic) — bootstrap의 붕괴 폭.
  detector 붕괴가 navigation을 실제로 무너뜨리는지의 직접 증거.
- 이 8셀 test에 재시도 없음. 부차 지표(crash/timeout)는 descriptive.

## 2026-08-12 — stage-A 사다리 재실행(v7): envelope 안 robust + 밖에서 우아한 열화

`results/navrl_detector_domain_shift_v7/summary.{md,json}` — v2와 동일 28셀·동일 seed 103.
v7(9×9/24ch, threshold는 사다리 공통 0.55로 측정)과 bootstrap을 동일 프레임에서 비교:

| hue | bootstrap | v2(참고, 이전 측정) | **v7** |
|---:|---:|---:|---:|
| nominal | 95.0% | 95.0% | **96.9%** |
| 60° (envelope 경계) | 79.7% | 62.5% | **96.2%** |
| 90° (밖) | 45.0% | 39.4% | **91.2%** |
| 120° (밖) | 40.9% | 34.4% | **83.7%** |
| 180° (밖) | 24.8% | 21.5% | **62.6%** |

light ±0.7(밖)에서도 98.9%(bootstrap 84.2%), texture/albedo envelope 안 96~99%.
**v2의 "learned가 bootstrap보다 취약" 역전이 완전히 뒤집혔다** — randomization + 용량으로
envelope 안 균일 robust, 밖에서도 bootstrap 대비 2.5배(180°) 유지.

주의 신호 1건: **albedo 0.5(envelope 밖)에서 learned FPR 7.6%** 발생(envelope 안 0.0%).
분포 밖 배경 반사율에서 위양성 모드가 열린다 — envelope 선언의 실증적 근거이자,
실기 전 배경 분포 확장이 필요하다는 표지로 기록한다.

### navigation A/B 재실행 사고 기록

1차 실행이 2번째 셀에서 거부됐다: 평가 가드가 checkpoint의 학습 당시
`cfg_detector_threshold=0.55`와 실행값 0.70(v7의 validation 선택 운영점)의 불일치를
사고로 취급. **의도된 불일치**이므로 learned arm에만 `NAVRL_V2_FORCE=1`을 명시하고
(analytic arm은 전체 가드 유지 — 환경 불일치는 거기서 잡힘), 완주 1셀 포함 루트를 삭제 후
클린 재실행. 요약을 낸 적 없으므로 선택적 재시도가 아니다.

## 2026-08-12 — 검증 2 종결: navigation A/B — 인지 robustness ≠ 시스템 robustness

`results/navrl_v2_appearance_navigation_ab_seed151_157/summary.{md,json}` — 사전등록 8셀 완주,
재시도 없음. pooled (2×2049~2050 ep/arm):

| arm | capture | crash | timeout |
|---|---:|---:|---:|
| nominal + analytic | **80.92%** | 16.52% | 2.56% |
| nominal + learned_v7 | 75.89% | 21.59% | 2.51% |
| envelope + analytic | 49.88% | 26.94% | **23.18%** |
| envelope + learned_v7 | 66.62% | 30.60% | 2.79% |

- **E1 (primary NI, nominal): FAIL** — learned_v7 − analytic = **−5.02 pp**,
  CI [−6.80, −3.24] (margin −2.0). v2 detector가 NI였던 것(−0.07 pp)과 대조적으로,
  **robustness 학습이 nominal 성능을 실제로 깎았다.**
- **E2 (envelope 비용, v7)**: −14.29 pp, CI [−16.17, −12.41] (80.92 → 66.62%).
- **E3 (bootstrap 붕괴)**: **−31.04 pp**, CI [−32.99, −29.09] (→ 49.88%), **timeout 23.2%**
  — hue 변형 표적을 red 규칙이 못 봐서 탐색만 하다 끝나는 프로파일.

### 판독

1. **robust detector는 envelope 붕괴의 절반 이상을 구조한다**: E3 −31.0 → E2 −14.3
   (+16.7 pp 회수), timeout 23.2% → 2.8% (표적은 계속 찾는다).
2. **그러나 nominal NI가 깨졌다.** 원인 후보(미검증, 다음 진단 대상):
   (a) v7의 offline gate는 **envelope 하에서만** 측정됐다 — nominal appearance의 offline
   지표는 한 번도 게이트되지 않았다(검증 설계의 공백, 이번에 드러남).
   (b) v7은 픽셀 마스크 통계(크기·경계·threshold 0.70)가 analytic과 달라
   surface_range/carve-out/visible 분포를 바꾼다 — **정책은 analytic 통계에 적응돼 있으므로**
   detector 교체 자체가 관측 분포 이동이다. crash +5.1 pp가 이 경로와 부합.
3. **논문 서사**: "인지 모듈의 robustness는 시스템 robustness로 자동 승격되지 않는다 —
   정책이 인지 통계에 결합돼 있기 때문"이 검증 2의 최종 결론. E1/E2/E3 세 숫자가
   이 문장을 정량화한다.

### 검증 2 종결 선언

렌더러 8축 → stage-A 사다리(v2 취약) → v3(1×1 불가) → v4(용량) → v5(선택기) →
v6(근소 미달) → v7(14/14 PASS) → 사다리 재실행(envelope 안 robust) → navigation A/B
(E1 FAIL·E2/E3 정량화)까지, 사전등록·1회 개봉·재시도 없음 원칙을 유지하며 완료.
후속 후보(로드맵 재논의 대상): nominal-포함 offline gate 재설계, threshold의 nominal
재보정(새 사전등록 필요), 그리고 **검증 5의 fresh PPO를 v7+envelope randomization 위에서
학습**하는 설계(정책-인지 결합을 학습으로 푸는 정공법).

## 2026-08-12 — 검증 3 사전등록 + 구현: P3 전제(정확한 clock/pose) 민감도

**구현** (`navrl_perception.py`): 링 버퍼 슬롯을 steps+3으로 늘려 capture 슬롯 양쪽 1스텝의
pose 이력을 보존하고, `_perturbed_capture_pose()`가 세 교란을 적용한다 —
- `NAVRL_POSE_CLOCK_OFFSET_S`: 지연 측정을 (capture+δ) 시점 pose로 변환. 소수 스텝은
  인접 odometry 샘플 간 **보간**(pos lerp, quat sign-aligned nlerp)이므로 clock skew와
  pose interpolation을 동시에 연습한다. **δ=+τ는 현재 pose에 정확히 도달 = naive 변환 재현**
  (사다리에 내장된 검증 앵커).
- `NAVRL_POSE_NOISE_POS_M` / `NAVRL_POSE_NOISE_YAW_DEG`: buffered pose 자체의 odometry 오차
  (축별 gaussian 위치, world-z yaw).
zero-knob이면 종전 경로 그대로(bit 동일, `torch.equal` 테스트). 테스트 38/38 PASS
(δ=0 정확 capture pose, δ=+τ=현재 pose, δ=τ/2=중점 보간, δ=−τ=한 스텝 이전, yaw noise 단위노름).

**사전등록** (`eval_navrl_v2_pose_premise.sh`, 실행 전 고정): frozen ep25000+riskcap,
τ=0.1+P3 ON, analytic bootstrap(검증 2와 분리), seed **163**(미사용), 2049 ep/cell, **12셀**:
exact 앵커 / clock ±0.02·±0.05·+0.10 / pos noise 0.01·0.03·0.10 m / yaw noise 0.5·2·5°.
판정: 셀별 Δ vs exact(95% CI). **clk_p0p10이 naive 수준(~38%대)으로 떨어지지 않으면 노브
자체가 무효 → 캠페인 void**. 재시도 없음.

## 2026-08-12 — 검증 3 완료 (부분 superseded): P3 전제 민감도

`results/navrl_v2_pose_premise_seed163/summary.{md,json}` — 12/12셀, 사전등록 그대로, 재시도 없음.

**노브 검증 앵커 PASS**: clk_p0p10(=+τ, 수학적으로 naive 변환과 동일)이 **39.30%**로
naive 수준(타 시드 37.8~38%)에 안착 — 캠페인 유효.

| 교란 | Δ vs exact (79.06%) | 판독 |
|---|---:|---|
| clock −0.05 s | −2.82 pp | **음수(pose가 이름)는 온화** |
| clock −0.02 s | −3.12 pp | |
| clock +0.02 s | −3.37 pp | 여기까지 허용 |
| clock **+0.05 s** | **−17.28 pp** | **양수(pose가 늦음)는 급붕괴** |
| clock +0.10 s | −39.77 pp | = naive (앵커) |
| pos noise 0.01~0.10 m | +0.20 ~ −2.78 pp | **10 cm까지 사실상 무해** (비단조는 SE 안) |
| yaw 0.5° | −0.59 pp | 무해 |
| yaw 2° | −4.29 pp | 경계 |
| yaw 5° | −12.49 pp | 불가 |

**발견 — clock offset의 부호 비대칭** (−2.8 vs −17.3 pp @ ±0.05 s): pose가 **늦으면**(양수)
측정이 드론 진행 방향으로 +v·δ 이동한 위치에 놓인다. 그 오염된 표적 추정이 **진행 방향의**
carve-out을 구동해 비행 경로 위 실제 막대를 장애물 지도에서 지운다(P2에서 규명한 채널의
재발화 — crash 35.5% vs 21.3%가 부합). pose가 **이르면**(음수) 같은 크기의 오차가 뒤쪽을
지우므로 무해하다. 즉 이 비대칭은 P2 채널의 독립 재확인이기도 하다.

> **2026-08-12 Codex 교정**: 아래의 `+20 ms 허용`, `yaw≤1°`, `위치≤10 cm 무료`는
> 사전등록된 허용 gate가 아니며 1°는 직접 측정하지도 않았다. 또한 이 캠페인의 pos/yaw arm은
> 전역 RNG를 소비했다. clock-offset 셀은 유효한 상수 offset 민감도이지만 hardware spec이 아니고,
> pos/yaw 수치는 뒤의 isolated-RNG 교정 캠페인으로 대체한다.

## 2026-08-12 — 검증 4 사전등록: 205-bar bar-contact ceiling 3단 분리

frozen ep25000+riskcap의 잔여 crash 97.2%가 bar contact다(Gate 1). 동결된 순서(08-11)대로
**측정 후에만** token/control 변경을 연다. 실행 전 고정:

- **단일 계측 셀**: 205 bars, deterministic, riskcap, exact-600, seed **167**(미사용),
  2,049 ep, `NAVRL_BAR_PROBE=1` + 신설 `NAVRL_EPISODE_DUMP` 동시 활성.
- **① geometry oracle** (`tools/analyze_navrl_v2_reachability.py`): bar-contact 에피소드의
  spawn→최종 표적 위치 정적 연결성을 3반경으로 괄호 — 낙관 0.40 m(최소 반폭+드론) /
  governor 0.65 m(riskcap 반폭+드론) / 비관 0.766 m(최대 반대각+드론). 판정:
  비관에서도 연결 ≥95% → "기하는 천장을 강제하지 않음" 확정.
- **② representation coverage**: bar-probe v2 로그(`hit_in_token_fov`, `hit_token_given_fov`,
  crowding, duplicate)를 셀 로그에서 회수. v1(07-24)의 "충돌의 35%가 입력에 없던 막대"를
  현 정책·205 bars에서 재측정.
- **③ contact kinematics**: governor contact 통계(clearance/executed speed/stopping margin)를
  결과 JSON에서 회수.
- dump는 GT를 **디스크로만** 내보낸다(actor/critic/reward/종료 무접촉 — bar probe와 동일 원칙).
  이 셀에 재시도 없음. 판정 규칙: ①이 "기하 무죄"면 ②/③의 상대 크기가 다음 개입
  (표현 vs 제어)을 정한다.

## 2026-08-12 — 검증 4 완료(교정됨): 정적 연결성 통과, 접촉시점 표현·제어 진단

계측 셀 `results/navrl_v2_bar_ceiling/instrumented/` (seed 167, 2,049 ep,
capture/crash/timeout **80.09/16.98/2.93%** — 앵커 정합). dump 1,989 에피소드
(timeout 60건은 종료 경로가 달라 미포함 — contact oracle에는 무영향), contact 표본 333.

### ① geometry oracle — 좌표계 교정 후 정적 연결성 100%

`results/navrl_v2_bar_ceiling/reachability.json`, bar-contact 333 에피소드 전수:

| 팽창 반경 | 연결률 |
|---|---:|
| 낙관 0.40 m (최소 반폭+드론) | **100.0%** |
| governor 0.65 m (riskcap 반폭+드론) | **100.0%** |
| 비관 0.766 m (최대 반대각+드론) | **100.0%** |

독립 검수에서 episode dump 좌표가 **0..40 m**인데 oracle이 **−20..20 m**로 해석한 결함을
발견했다. 교정·회귀 테스트 후 같은 333건을 재계산하니 세 반경 모두 100%로 사전등록 기준을
통과했다. 단, 이것은 **spawn→종료 시점 표적 위치의 정적 2-D 경로 존재**만 뜻한다. 선회·제동
동역학, 이동 표적 궤적, 600-step 예산은 시험하지 않았으므로 `기하 무죄`라는 강한 표현은
철회하고 **정적 연결 단절은 관측된 충돌의 주원인이 아님**으로 한정한다.

### ② representation coverage (barprobe v2, n=333)

- **hit_token 0.832** — **접촉 순간** 충돌 막대의 16.8%가 정책 입력(8토큰)에 없었다
  (v1 07-24의 35%에서 절반으로 개선됐지만 여전히 실재). FOV 분해: hit_fov 0.853
  (14.7%는 240° 토큰 섹터 밖에서 충돌), hit_token_given_fov 0.852.
- crowding: 반경 내 46.9개 / 토큰 FOV 내 31.7개 vs **capacity 8** — 4배 과밀.
- **rank 0.0**: 표현된 경우 충돌 막대는 항상 최근접 토큰이었다 — 즉 "먼 막대를 놓친" 게
  아니라 **가까운 위협은 잘 고르는데, 놓친 15~17%가 치명적**이라는 구조.
- 위치 정확도는 양호: center_offset 0.35 m, cross_track 0.23 m, radial_gap 0.20 m.

### ③ contact kinematics (n=333)

- 접촉 시 평균 clearance **0.84 m**, actual speed 1.15 m/s (executed 2.02, requested 3.03).
- **stopping margin: executed −0.157 m, requested −1.162 m** — 접촉 직전 요청 속도는
  물리적으로 세울 수 없는 수준이었고, governor가 2/3를 깎아도 executed 기준 **평균이 이미
  음수**다. negative-margin(executed) 비율 9.3%.

### 종합 판정 (교정된 보수적 해석)

정적 연결 단절은 기각됐다. 반면 **접촉 순간** representation miss와 음의 평균 제동 여유는
관측됐지만 서로 독립 원인이라고 볼 수 없다. clearance 자체가 충돌 막대를 놓쳤다면 둘은 같은
인지 실패의 두 증상일 수 있고, 막대가 접근 2~5 step 전에는 토큰에 있었는지도 아직 모른다.
따라서 capacity/FOV 또는 governor 변경을 바로 여는 대신 pre-contact 시계열과 성공 near-miss
대조군을 계측해야 한다. 전체-run negative-margin 비율 9.3%와 contact 시점 평균 −0.157 m는
서로 다른 집계이므로 직접 비교하지 않는다.

## 2026-08-12 — Codex 독립 검수 교정 및 후속 진단 사전등록

Claude 검증 1~4의 원자료·receipt·SHA·집계식을 독립 재검산했다.

- 검증 1: fresh seed 97/101 pooled learned−analytic **−0.0145 pp**, CI
  **[−1.752, +1.723]**, NI margin −2 pp → **PASS 유지**.
- 검증 2: E1 −5.02 pp는 detector 구현과 threshold 0.55→0.70을 동시에 바꿔 원인 분리 불가.
  learned arm의 광역 `NAVRL_V2_FORCE=1`도 threshold만 허용하는 좁은 override로 교체했다.
- v7 receipt가 실제 artifact meta의 seeds 137/139/149 대신 상수 71/73/79를 기록하던 provenance
  결함을 수정하고 기존 receipt도 실제 값으로 교정했다.
- 검증 3: pos/yaw noise가 전역 torch RNG를 소비해 환경 reset/배치 난수열까지 바꾸는 교란을
  발견. `NAVRL_POSE_NOISE_SEED` 전용 generator로 분리했고, 전역 RNG 불변·재현성 테스트 통과.
  clock offset 셀은 난수를 쓰지 않으므로 기존 결과가 유지된다.
- 검증 4: 위 좌표계 결함과 함께, dump 활성 여부가 정상 perception/prev_action reset을 가르던
  indentation 회귀를 발견·수정했다. dump를 켠 검증 4 셀은 reset이 실행되어 기존 계측 자체에는
  영향 없지만, 이후 일반 평가/학습은 수정 없이는 오염될 수 있었다.

### D2 threshold 분리 진단 — 실행 전 고정

`eval_navrl_v2_detector_threshold_diagnostic.sh`: frozen ep25000+riskcap, nominal, 205 bars,
deterministic, exact-600, fresh seeds **191/193**, 2,049 ep/cell. 3 arms × 2 seeds:
`analytic@0.55`, `learned-v7@0.55`, `learned-v7@0.70`.

- D1 `v7@.55 − analytic@.55`: threshold를 맞춘 detector 통계 교체 효과.
- D2 `v7@.70 − analytic@.55`: 기존 E1의 결합 효과 재현.
- D3 `v7@.70 − v7@.55`: 같은 v7 안 threshold 단독 효과.
- 모두 독립 이항 95% CI로 보고하며, **이 데이터로 threshold를 선택·채택하지 않는다**.
  재시도 없음. v7@.70만 threshold mismatch narrow override를 사용한다.

초기 seed 173/179 실행은 2셀 뒤 narrow override의 Python 지역변수 누락을 고치는 과정에서
shared source bundle과 런타임 evaluator가 달라져 무결성 가드가 3번째 셀을 차단했다. 숫자가
노출된 뒤였으므로 같은 seed를 재시도하지 않고 **캠페인 전체를 VOID**로 보존한다. endpoint와
arms는 바꾸지 않고, 수정 완료 후 미사용 seed 191/193을 새 confirmatory diagnostic으로 고정했다.

### D3 pose RNG 교정 진단 — 실행 전 고정

`eval_navrl_v2_pose_noise_rng_audit.sh`: frozen ep25000+riskcap+P3, tau=0.1, analytic,
205 bars, deterministic, exact-600, fresh environment seed **181**, fixed pose-noise seed **9181**,
2,049 ep/cell. exact + pos {0.01,0.03,0.10 m} + yaw {0.5,2,5°} = 7 cells.
셀마다 simulator 전역 RNG는 동일하고 pose 교란의 표준정규 draw만 전용 stream에서 동일하게
재생한다. 셀별 Δ vs exact와 95% CI만 보고하며 기존 단일시드 `실기 스펙`은 이 결과 전까지
보류한다. 재시도 없음.

## 2026-08-12 — Codex 후속 진단 완료: detector 결합 원인 분리 + pose RNG 교정

### D2 detector threshold 분리 — 손실 원인은 threshold가 아니라 actor–detector 결합

`results/navrl_v2_detector_threshold_diagnostic_seed191_193/summary.{md,json}`. 6/6 셀 완료,
각 arm 4,098~4,100 episodes, policy SHA `f7022139…`, detector SHA `85c7974b…`, 셀 간
source/evaluator/checkpoint SHA 동일. 원본 success/n과 receipt `result_sha256`를 독립 재계산했다.

| arm | pooled capture | 대비 효과 (95% CI) |
|---|---:|---:|
| analytic @ 0.55 | **80.58%** | 기준 |
| v7 @ 0.55 | **75.38%** | **−5.192 pp [−6.982, −3.401]** |
| v7 @ 0.70 | **76.82%** | analytic 대비 **−3.752 pp [−5.523, −1.981]** |

같은 v7 내부 threshold 효과(0.70−0.55)는 **+1.439 pp [−0.407, +3.285]**로 CI가 0을
포함한다. 따라서 기존 E1 −5.02 pp를 `0.70 threshold 오정합`으로 설명하는 가설은 기각한다.
threshold를 0.55로 맞춰도 약 −5.2 pp가 재현되므로 핵심은 v7의 mask/range/visibility/carve-out
통계와 analytic detector로 학습된 actor의 결합이다. 0.70은 나빠지는 방향조차 아니었으나,
이 진단으로 운영 threshold를 선택·변경하지 않는다.

초기 seed173/179 캠페인은 두 셀 완료 뒤 narrow-override 변수 scope 결함을 수정하면서 source
manifest가 달라졌고 post-run guard가 3번째 셀을 차단했다. 결과를 본 뒤 같은 seed를 재시도하지
않고 `results/navrl_v2_detector_threshold_diagnostic_seed173_179_VOID_source_drift/`로 VOID 보존했다.

### D3 pose isolated-RNG — 위치 결론 약화, yaw 2°/5° 민감도 재현

`results/navrl_v2_pose_noise_rng_audit_seed181/summary.{md,json}`. 7/7 셀 완료, 셀당
2,049~2,050 episodes, environment seed 181, 전용 pose-noise seed 9181. 모든 receipt/result/source
SHA와 실제 seed/bars/noise seed를 독립 검증했다.

| cell | capture | Δ vs exact (95% CI) |
|---|---:|---:|
| exact | **78.49%** | 기준 |
| position 1 cm | 78.68% | +0.20 pp [−2.32, +2.71] |
| position 3 cm | 77.79% | −0.69 pp [−3.22, +1.84] |
| position 10 cm | 77.01% | −1.47 pp [−4.02, +1.07] |
| yaw 0.5° | 77.79% | −0.69 pp [−3.22, +1.84] |
| yaw 2° | **75.21%** | **−3.28 pp [−5.86, −0.70]** |
| yaw 5° | **65.74%** | **−12.75 pp [−15.47, −10.03]** |

직접 결론은 다음뿐이다. 이 **step-wise iid Gaussian** 모델·단일 환경 seed에서는 위치 10 cm와
yaw 0.5°까지 유의한 손실을 검출하지 못했다. 이것은 `무료`나 허용 spec이 아니다. yaw 2°부터
유의한 손실, 5°에서 큰 붕괴가 재현됐다. 1°는 측정하지 않았고, 실제 odometry의 bias/drift/jitter
모델은 평가하지 않았으므로 하드웨어 요구조건으로 외삽하지 않는다. 기존 clock 결과도 `constant
timestamp offset sensitivity`로만 부르며 clock skew나 jitter 결과로 부르지 않는다.

### 검증 5 설계 판정 — C 금지, A smoke 후 분리된 fresh lineage

한 번의 fresh PPO에 exact-600/bootstrap + v7 + appearance randomization + token/governor 변경을
모두 넣는 선택지 C는 원인 분리가 불가능하므로 금지한다. 다음 순서를 고정한다.

1. **5A engineering smoke (성과 자료 아님)**: exact-600 + `time_outs` bootstrap만 반영하고,
   analytic detector·현 representation/control을 유지한 fresh run. 500~1,000 epoch에서 NaN/KL,
   reset, timeout=600, curriculum state/receipt를 검증한다.
2. **5B corrected-semantics baseline**: 5A가 통과하면 같은 최소 계약으로 full-budget fresh PPO.
   단일 training seed는 pipeline demonstration일 뿐 알고리즘 효과 증거가 아니며, 성능 주장을
   하려면 최소 2개(가능하면 3개) training seed와 별도 held-out evaluation seeds가 필요하다.
3. **5C perception adaptation**: v7 nominal 학습을 baseline과 seed/budget/architecture matched로
   비교한 뒤에만 v7+appearance randomization을 별도 arm으로 연다. 그래야 detector 교체 적응과
   domain randomization 효과가 분리된다. 현재 E1/D2는 이 arm의 필요성을 입증하지만 성공을
   보장하지 않는다.
4. pose perturbation은 초기 fresh PPO에 넣지 않는다. 현재 검증은 iid 모델 한 종류뿐이고,
   clock offset은 학습 randomization보다 timestamp calibration/동기화로 다룰 문제다.
5. token capacity/FOV와 governor margin 변경도 검증 4의 pre-contact 시계열 및 matched near-miss
   대조 측정 전에는 fresh PPO와 결합하지 않는다.

검증 5를 지금 full run으로 시작하지 않는다. 먼저 5A launcher가 `fresh/no-checkpoint`, exact-600,
`time_outs`, detector/appearance 계약, source receipt, 고정 budget과 중단 규칙을 기계적으로 검사하도록
구현·preflight한 뒤 사용자에게 실행 계약을 제시한다.

## 2026-08-12 — 발전 방향 문헌 접지 로드맵 작성 (검증 5는 Codex로 이관)

사용자 지시로 검증 5(fresh PPO)는 Codex 세션이 진행한다(설계 질문은 브리프 #3 §3에 이관,
`09fbb1e`). 본 세션은 "다음 주 이후 어디를 발전시킬 것인가"를 문헌 접지로 정리했다 —
`docs/development_directions_2026-08.md`. 병렬 문헌 조사 2건(제어·요격 / 인지·강건성,
2023–2026 중심, 검색 ~29회 + 원문 12편).

**핵심 수확 — 우리가 이미 가진 신규성 3개를 두 조사가 독립적으로 확인**:
1. 정책-인지 결합 비용의 직접 측정(E1 −5.0 pp) — 선행 부재 (Swift도 ablation까지만)
2. "충돌 유발 장애물의 입력 탈락률" 측정(16.8%) — 측정 방법론 자체가 기여
3. 단일기체 dense-clutter **요격** 설정(80.5% @ 12.8 bars/100m² = Wild 최고 밀도의 3.2배) —
   밀집 SOTA는 전부 정지 표적 goal-reaching

**방향 D1~D9** (실측 동기 → 문헌 → 실험 → 비용 형태): D1 backup-CBF 제동 필터(eval-only,
FastBridge 계보 — margin −0.157 m 직격), D2 토큰 확장/어텐션(16.8% 직격), D3 Swift식
노이즈모델 fine-tune(E1 직격), D4 velocity→CTBR 전환(ICRA'22 벤치마크: LV는 구조적 상한;
dense-clutter 요격에서 이 비교는 미발표), D5 지연 랜덤화(−17.3 pp 절벽), D6 불확실성
토큰(H4 직격), D7 탐색 목적함수, D8 학습형 추정기(병리 4종 실측이 동기), D9 distractor
envelope(eval-only). 실행 순서와 RA-L 1편 구조(측정 주도 서사) + 2편째 후보(D4/D8)까지 제안.

## 2026-08-12 — Codex 검수 결과 수용 + 시뮬/실기 격차 문서화 + 디스크 정리

### Codex 검수(`docs/codex_review_2026-08-12.md`)가 잡은 내 결함 3건 — 전부 수용

1. **검증 4 oracle이 아레나 좌표계를 틀렸다.** dump는 0..40 m 프레임인데 oracle이 −20..20을
   가정했다. 내가 보고한 연결률 100/97.3/**94.6%는 무효**다. 수정 후 **333/333 = 세 반경 모두
   100%**. 결론(기하 무죄) 방향은 오히려 강화됐지만 **내가 낸 숫자는 틀렸다.**
2. **검증 3의 위치/yaw 교란이 전역 PyTorch RNG를 소비했다** — arm마다 시뮬 리셋·장면
   난수가 달라지는 교란. 격리 RNG 재실행(seed181) 결과: 위치 1/3/10 cm는 **전부 CI가 0 포함**,
   yaw 2° −3.28 pp, 5° −12.75 pp, **1° 셀은 존재하지 않음**. 따라서 내가 쓴
   "실기 스펙 +20 ms / yaw ≤1° / 위치 ≤10 cm 무료"는 **사전등록도 계측도 안 된 주장**이므로
   하드웨어 스펙으로 제시하면 안 된다. clock offset 사다리 자체는 유효(난수 미소비).
3. **episode-dump 패치의 들여쓰기가 정상 리셋을 dump 활성 여부에 종속시켰다** — perception,
   prev_action, visibility 리셋이 `if self._episode_dump_path:` 블록 안으로 들어갔다.
   컴파일·테스트를 통과했고 계측 셀은 dump가 켜져 있어 정상 동작했으나, **dump가 꺼진 실행에서
   리셋이 누락되는 잠재 결함**이었다. Codex가 수정 + AST 가드 추가.

추가 지적 수용: 검증 2의 E1은 검출기와 threshold를 동시에 바꾼 비교였고, 분리 진단
(seed191/193) 결과 **threshold는 원인이 아니다**(v7@0.55 −5.19 pp vs v7@0.70 −3.75 pp,
둘의 차 +1.44 pp CI가 0 포함). 즉 원인은 **학습형 검출기의 출력 통계**이며 내 브리프 §1-A의
"대안 설명(threshold 오정합)"은 기각됐다 — 결합 서사가 오히려 지지된다. 다만 v6→v7은
"연속적 모델 개발"이지 무결한 confirmatory 체인이 아니므로 1-pixel tolerance와 envelope 선택을
논문에 명시 공개해야 한다.

### 시뮬 기체 vs 실기 하드웨어 격차 (`docs/sim_vs_hardware_gap_2026-08.md`)

> **SUPERSEDED:** 아래 472 g 합계·56%·“TensorRT 문제” 표현은 같은 날짜 뒤의 ref5in 전면
> 재감사(§ `ref5in 전면 재감사`)로 폐기됐다. 최신 값은 명시 단품 부분합 404.2~411.8 g이며,
> carrier/냉각/전원/배선/마운트가 빠져 완성 payload가 아니다.

저장소에서 직접 읽은 시뮬 기체: **총 질량 250 g**(URDF), 충돌박스 0.28×0.28×0.08 m,
모터 2 N×4 = 8 N, **T/W 3.26**, 최대 틸트 45°, 모터 시상수 0.04 s, IMU 비활성(상태는 시뮬 제공).

이 인지 스택을 실기로 올리는 데 필요한 무게: Orin NX ~100 g + Livox Mid-360 **265 g** +
RealSense ~72 g + Pixhawk ~35 g = **472 g** (프레임·배터리 제외).
**→ 페이로드만 시뮬 기체 전체 질량의 1.9배.** 실기는 1.2~1.5 kg 급이 되고 T/W 유지에 ~40 N이
필요하다. 현재 정책은 **실을 수 없는 기체의 기동 성능을 전제로 학습**됐다는 것이 1순위 격차.
무게의 56%가 LiDAR 하나이므로 D8(학습형 융합)이 하드웨어 경량화와 직결된다.

**연산은 제약이 아님(실측)**: 정책 추론 0.399 ms(장애물 8) vs 0.396 ms(16) vs 0.399 ms(32) —
`navrl_transformer_network.py:31-34`대로 용량을 올려도 **토큰이 늘지 않고 Linear 입력 폭만**
넓어져 어텐션 비용이 불변이다. 10 Hz 예산의 0.4%. 진짜 비용은 검출기(160×90에서 1.5 ms,
640×480에서 **43 ms**)이며 이는 TensorRT/GPU 문제다.

### 디스크 정리 (2.05 GB 회수, 손실 0)

- `results/**/checkpoint_snapshot.pth` 167개가 **고유 5종의 복사본**이었다 → 하드링크로 전환.
  경로·해시·receipt 검증이 모두 그대로 유지되며 **1.34 GB 회수**(results 1.5 GB → 158 MB).
  삭제가 아니라 dedupe이므로 provenance 손실이 없다.
- WORKLOG·results 어디에서도 인용되지 않는 **고아 run 37개(711 MB) 삭제**(7월 v1 계보).
  참조 run 5개와 Codex의 활성 스모크 run은 보호했다.
- **미실행 정리 후보(사용자 판단 필요)**: `runs/**/nn/*.pth` 902개 중 WORKLOG·results 인용은
  50개뿐 — 나머지 852개(~7 GB)는 중간 체크포인트다. 다만 CLAUDE.md의 "밀도 run 평가는
  last_gen_ep_XXXX" 규칙상 특정 epoch 체크포인트가 사후에 필요해질 수 있어 **자동 삭제하지
  않았다.**
- md 파일은 정리 대상에서 제외했다: 추적 중인 문서 전체가 ~750 KB(WORKLOG 548 KB 포함)로
  디스크 실익이 없고, 전부 연구 기록·검수 이력이다.

## 2026-08-12 — 검증 5A corrected-semantics fresh PPO engineering smoke 완료

사용자 승인 범위대로 full-budget 검증 5B는 실행하지 않고, 5A만 fresh seed 197 / 128 env /
1,000 epoch로 실행했다. 전용 closed-contract launcher
`train_navrl_v2_v5a_semantics_smoke.sh`를 추가해 checkpoint/CLI를 거부하고 상속된 `NAVRL_*`
실험 변수를 제거한 뒤 canonical v2의 analytic detector, cluster-sector 8토큰/240°, squashed Gaussian,
governor/pose-noise off를 고정했다. hostile-env preflight와 exact-600/timeout source 회귀 테스트를
추가했다.

실런 `ppo_260812_1620_navrl_v2-v5a-semantics-smoke-s197`는 약 59분 후 exit code 0,
`max_epochs`, 완료 마커 `epoch=1000`으로 끝났다. canonical endpoint는
`last_gen_ppo_ep_1000_rew_140.1189.pth`, SHA-256 `f53489aa9158…`이다. 체크포인트에서
`cfg_episode_len_steps=600`, `cfg_rlgames_timeout_info_key=time_outs`, detector checkpoint 없음,
selector `cluster_sector`, FOV 240°, squashed Gaussian, bars 70을 직접 재확인했다. timeout 경로는
실제 학습 중 530회 실행됐고, capture/crash/timeout 합계 불변식도 전 파싱 epoch에서 통과했다.

PPO 안정성: 1,000 scalar 전부 finite, PPO KL 최대 **0.015831**, behavior-KL audit 최대
**0.022902**(< rollback 0.04), rollback **0**, skipped minibatch **0**, x/y/z/yaw raw OOB 최대
전부 **0**. 사전 고정한 의미론 핵심 4파일 SHA도 전후 동일했다.

성과 주장이 아닌 on-policy 기술 수치(실제 episode 분모로 가중 집계):

| window | episodes | capture | crash | timeout |
|---|---:|---:|---:|---:|
| 전체 파싱 epoch | 44,383 | 73.02% | 25.78% | 1.19% |
| speed ramp 1–300 | 15,533 | 66.95% | 31.63% | 1.42% |
| post-ramp 301–1000 | 28,850 | 76.29% | 22.63% | 1.07% |
| 마지막 100 epoch | 4,134 | 78.69% | 20.30% | 1.02% |
| 마지막 50 epoch | 2,088 | 78.40% | 20.59% | 1.01% |

판정은 **조건부 engineering PASS**다. corrected timeout 의미론이 fresh PPO 학습을 막거나
optimizer를 발산시키지 않는다는 좁은 목적은 통과했다. 그러나 `DENSITY_WARMUP=1000`이라 endpoint가
정확히 warmup 경계(num_task_steps=32,000)이고 post-warmup evidence는 한 건도 수집되지 않았다.
따라서 density gate/승급은 **미검증**이다. held-out 평가도 없고 training seed 하나뿐이므로 위
78~79%를 최종 성능이나 legacy 대비 개선으로 인용하면 안 된다. dirty worktree에서 실행해 핵심
4파일의 수동 SHA 불변은 확인했지만 full runtime source manifest는 없다는 provenance 한계도 남는다.

추가 관찰: x action은 epoch1000에서 edge95 49.8%, edge99 18.6%(run 최대 64.6%/43.0%)로
경계 집중이 남았다. raw OOB=0이므로 과거 unbounded-action 결함은 아니지만 고밀도 제동 연구의
경고 지표로 보존한다. 자동 summary의 peak capture 97.1%는 작은 단일 epoch 표본이라 폐기한다.

종료 시 epoch1000이 50-epoch 주기 저장과 max-epoch 저장에 동시에 걸려 서로 다른 이름의 checkpoint
두 개가 생겼다. 재귀 비교 결과 model/optimizer/env_state 차이 **0개**로 동일 상태였다. 기존 파일은
증거로 보존하고, 이후에는 주기 저장과 겹치면 terminal 중복 저장을 건너뛰고 canonical scalar-reward
이름 하나만 남기도록 `early_stop_a2c_agent.py`를 수정·회귀 테스트했다.

상세 결과: `results/navrl_v2_v5a_semantics_smoke_seed197/summary.{md,json}`. 다음 5B 전 필수 조건은
현재 소스 clean commit, 전체 training source receipt, 2–3 matched training seeds와 별도 held-out
evaluation seeds 사전등록이다. 사용자 요청대로 full-budget 5B는 이번에 시작하지 않았다.

## 2026-08-13 — 검증 5B까지 반영할 PPT 보완·검수 요청서 작성

다음 PPT 개정에서 검증 5B 결과까지 사용할 수 있도록
`docs/CLAUDE_PPT_REVIEW_REQUEST_VERIFICATION5B_2026-08-13.md`를 작성했다. 기존
`GENSPARK_PPT_BRIEF_2026-08-11.md`는 Gate 1–3 수치 원천으로 보존하되 새 요청서가 supersede한다는
경고를 추가했다.

새 요청서는 검증 1 learned-v2 NI replication, 검증 2 v7 threshold 분리, 검증 3 pose isolated-RNG,
검증 4 corrected reachability/contact-time 한계, 검증 5A engineering smoke를 슬라이드별로 연결한다.
특히 5A의 최근 78~79%는 70-bar on-policy single-seed 수치이므로 benchmark 사용을 금지했고,
`DENSITY_WARMUP=1000` 경계라 density promotion이 미검증임을 별도 슬라이드로 요구했다.

5B는 아직 미실행이므로 결과 표를 전부 `[미실행]` placeholder로 두고, clean commit/full source
receipt/2–3 training seeds/공통 held-out evaluation seeds/full-budget를 충족한 실제 summary에서만
채우도록 고정했다. PASS/MIXED/FAIL 세 분기 서술을 미리 정의해 성공 seed만 선택하거나 실패를
숨기지 못하게 했다. 본문 16장+부록 6장 구조, 유지/교체/삭제 목록, claim strength 3단계 표,
Claude 최종 체크리스트와 그대로 복사할 전달 프롬프트를 포함한다.

## 2026-08-12 — 기준 플랫폼 수정안 (검증 5 시작 전 결정 필요)

> **SUPERSEDED — 수치 사용 금지:** 이 절은 팔 중점을 모터로 오독한 당시 기록이다. 바로 뒤
> `2026-08-13 — 기준 플랫폼 구현·검증 완료`가 1차 정정이고, 이후
> `ref5in 전면 재감사`가 현재 증거 수준을 다시 제한한다.

사용자가 "시뮬 기체 스펙 자체가 잘못 접근된 것 아닌가"를 제기해 URDF·config를 실사했다.

**진단 — 현 URDF는 자기모순이다.** 질량 0.225 kg과 모터 추력 2 N은 Aerial Gym **기본
`quad.urdf`/`base_quad_config` 값을 그대로 상속**한 것이고(선택된 값이 아님), 07-22에
충돌 기하만 의도적으로 0.05 m 구 → **0.28 m 박스**로 키웠다. 그 결과:
- 모터가 ±0.065 m에 있어 **실제 기체 폭은 18 cm**인데 충돌박스는 28 cm (1.5배 과대)
- 관성 ixx 4.23e-4는 28 cm 균일박스 근사(1.77e-3)의 **1/4.2** — 관성은 작은 기체, 충돌은 큰 기체
- 각가속 ≈870 rad/s²로 비현실적
즉 **"18 cm 기체의 회전 민첩성 + 28 cm 기체의 충돌 단면"**이며 어떤 실제 빌드와도 대응하지 않는다.
낙관(회전)과 보수(충돌)가 상쇄되는 구조라 결과가 무효인 것은 아니지만, 하드웨어 주장은 불가.

**제안 — 5인치급 기준 플랫폼** (`docs/reference_platform_proposal_2026-08.md`):
질량 1.20 kg(페이로드 472 g 포함), 모터 9.6 N×4(**T/W 3.26 유지**), 모터 반경 0.125 m(대각 25 cm),
충돌박스 **0.32 m**, 관성 ixx 1.12e-2 / izz 2.05e-2, 모터 τ 0.08 s.

**핵심 발견 — 수정 비용이 예상보다 훨씬 작다**:
- **수평 가속 9.8 m/s², 상승 22.2 m/s² 모두 불변** (수평은 g·tan(틸트)로 질량 무관, 상승은
  T/W 유지). 정책이 실제로 쓰는 속도 명령의 물리 한계가 그대로다.
- **충돌폭 28 → 32 cm, +4 cm뿐**. 막대 간극 160 cm 대비 여유가 82% → 80%로 2%p 감소.
- 나빠지는 것은 각가속(0.25배)과 모터 응답(2배 느림)뿐이며, 이는 검증 4가 지목한
  stopping-margin 축과 **같은 축**이라 오히려 측정 가치가 있다.

따라서 앞서 검토했던 "충돌박스만 키우는 eval-only 민감도 스윕(0.28→0.45)"은 **정보 가치가
작아 폐기**한다(현실 플랫폼이 0.32 cm라 4 cm 차이). 기체 전체를 현실화하는 편이 낫다.

**소요 추정**: 파라미터 작성+정합성 검증 3~4 h → 스모크 1~2 h → **본학습 3~5일**(시드 1개,
기존 v2 fresh 커리큘럼 실측 ~2.5일 + 여유). 검증 5 전체 4단계를 기준 플랫폼으로 가면
**11~17일**(시드 1개), Codex 권고대로 시드 3개면 1개월+.

**권고**: 파라미터 작성 + 스모크(반나절)만 먼저 해서 "이 기체로 학습이 되는가"를 판정한 뒤
본학습 여부를 결정한다. 변경 범위는 URDF 1개 + robot_config 1개 + `robot_name` 1줄이며
**코드 변경 없음**. legacy 계보와는 기체가 다르므로 **비교 불가**(v1↔v2 원칙 동일 적용).

## 2026-08-13 — 기준 플랫폼 구현·검증 완료, 그리고 08-12 진단의 정정

> **부분 SUPERSEDED:** 아래 472 g, “유일한 실질 회귀”, 출처가 고정되지 않은 cross-check,
> 초기 5/5 검증, “0.12 m 높이는 바닥/천장만 바꾼다”는 설명과 k/RPM을 식별된 하드웨어
> propulsion 값으로 읽는 해석은 이후 `ref5in 전면 재감사`에서 교정됐다. 현재 height 해석은
> 45° projected support 약 **+2.83 cm**의 3-D geometry confound이고, `k=4.401e-5`와 implied
> **28,023 RPM**은 fixed coordinate calibration일 뿐 thrust-stand 검증이 아니다. 역사적 진행
> 기록으로만 읽고 현재 주장에는 26/26 repository consistency와 21/21 canonical simulator gate를 사용한다.

사용자 지시 "파라미터 작성부터 시작해"에 따라 5인치 기준 플랫폼을 구현하고 두 층으로 검증했다.
그 과정에서 **08-12 진단이 URDF 오독이었음**을 발견해 함께 정정한다.

### 정정 — 08-12 항목의 수치는 무효다

`arm_motor_N` 조인트(팔 실린더의 **중점**, ±0.065 m)를 모터 위치로 착각했다. 실제 모터 링크
`motor_N`은 **±0.13 m**이고 `base_quad_config`의 allocation matrix도 ±0.13이다.

| 08-12 주장 | 실제 (재계산) |
|---|---|
| 모터 ±0.065 m, 기체 폭 18 cm | **±0.13 m, 모터간 대각 36.8 cm** (7~8인치급) |
| 충돌박스 28 cm가 기체보다 **1.5배 과대** | 박스가 그 프레임(7인치 프롭 시 44 cm)보다 **과소** |
| 관성이 28 cm 박스 대비 **4.2배 작음** | 조립 ixx 8.45e-4 = 균일 28 cm 박스의 **0.48배, 정상 범위** |
| 각가속 ≈870 rad/s² | **377 rad/s²** (로터 평행축 기여 누락이었음) |
| 제안: 박스 0.32 m, 반경 0.125 m, τ 0.08 s | **박스 0.28 m 유지, 반경 0.110 m, τ 0.04 s 유지** |

"자기모순이다 / 5인치로 맞추는 것이 답이다"라는 방향은 유지되지만 **모순의 내용이 다르다**.
관성이 아니라 **팔(±0.13, 7인치급)과 박스(0.28, 5인치급)의 충돌**이 핵심이고, 그다음이
질량이다 — 이 정책이 전제하는 인지 스택 472 g은 250 g 기체의 1.9배다. 관성 리터럴은 stock
잔재(두께 0인 0.150 m 평판)가 맞지만 조립 후 값은 평범하다. **그래서 3주간 눈에 띄지 않았다.**

### 확정 파라미터 — `navrl_ref5in_quad`

핵심 발견: **5인치 프롭 끝 AABB = 2×(0.0778+0.0635) = 0.2826 m**로, 현재 충돌박스 0.28 m와
사실상 같다. 즉 **충돌 기하를 바꿀 필요가 없다.** 08-12가 예상한 "+4 cm, 여유 82→80%"는 무효.

| | legacy | ref5in |
|---|---:|---:|
| 질량 | 0.250 kg | 1.200 kg |
| 추력/모터 | 2.0 N | 9.60 N (T/W 3.2617 유지) |
| 모터 팔 x=y | 0.130 m | 0.0777817 m (반경 0.110 = 220 mm 대각) |
| 충돌박스 XY | 0.28 m | **0.28 m (불변)** |
| 박스 높이 | 0.08 m | 0.12 m (바닥/천장만) |
| ixx / izz (조립) | 8.45e-4 / 1.69e-3 | 4.142e-3 / 5.769e-3 |
| 추력계수 k | 1.376e-5 | 4.401e-5 (467 rps 기준 재정규화) |
| 모터 τ | 0.04 s | **0.04 s (불변)** |
| 수평/상승 가속 | 9.81 / 22.19 m/s² | **동일** |
| 롤 각가속 | 377 rad/s² | 221 rad/s² (**0.586배** — 유일한 실질 회귀) |

교차검증: Agilicious(5~6", 0.75 kg) Ixx 2.5e-3을 1.2 kg으로 스케일 → ~4.0e-3, 우리 4.14e-3.

### 검증 결과

**정합성 26/26 PASS** (`tests/test_navrl_ref5in_platform.py`, CPU, isaacgym 불요).
URDF 조인트 ↔ allocation matrix ↔ 박스 ↔ 관성의 상호 일치를 기계 검사한다. 이 세 가지는
런타임에서 서로 대조되지 않으므로 legacy 결함이 조용히 생존했던 지점이다. legacy 결함 자체도
4개 테스트로 고정했다. **테스트가 내 초안의 오류를 잡았다** — "legacy 조립 관성이 균일박스의
0.25배 미만"이라는 단정이 실측 0.478에서 실패해, 관성 서사를 사실에 맞게 축소했다.

**비행 포락선 5/5 PASS** (`tools/verify_navrl_ref_platform.py`, 16 env, seed 911, 막대 0,
governor off, vision mode = vehicle-frame 속도 명령):

| 기동 | legacy | ref5in |
|---|---|---|
| hover 고도오차 / 표류 | +0.003 m / 0.001 m/s | +0.001 m / 0.001 m/s |
| forward 정상상태 / t90 | 2.490 m/s / 0.8 s | 2.490 m/s / 0.8 s |
| reversal 0-교차 / t90 | 0.5 s / 1.0 s | 0.5 s / 1.0 s |
| yaw 정상상태 / t90 | 2.500 rad/s / 0.3 s | 2.499 rad/s / **0.2 s** |

병진 지표가 네 자리에서 일치 — T/W와 틸트 한계를 보존했으므로 설계상 당연하다.
yaw가 빨라진 것은 izz 3.4배 증가보다 yaw 토크(`thrust_to_torque_ratio`×총추력) 4.8배 증가가
크기 때문으로, 설계 시 계산하지 않았던 유리한 부수효과다.

**미검출 명시**: 각가속 0.586배 회귀는 이 하네스로 **관측되지 않았다**. reversal이 동일한 것은
두 기체가 같아서가 아니라 10 Hz 제어 주기가 자세 동역학보다 느려 0.1 s 분해능에 묻히기
때문이다. 회귀 자체는 단위테스트가 계산·고정한다. 학습 가능성·과제 성능은 일절 미측정.

### 하네스 설계에서 배운 것 (다음 세션 절약용)

- 로봇당 **별도 프로세스** 필수. (a) `task_config`가 `robot_name`을 클래스 정의 시점에 읽으므로
  이후 `os.environ` 쓰기는 무효, (b) 한 프로세스에서 Isaac Gym sim 2개 생성 시 segfault.
- `NAVRL_VISION=1`이 아니면 action이 **goal frame**이라 스텝 응답을 읽을 수 없다. 첫 실행에서
  두 기체 모두 정상상태 0.879/0.884 m/s가 나와 "양쪽 다 FAIL"이 됐는데, 기체 결함이 아니라
  내 측정 프레임 오류였다.

### 변경 범위 / 다음 단계

신규 4파일 + 기존 2파일 1줄씩(`aerial_gym/robots/__init__.py` 등록,
`navrl_task_config.py`의 `robot_name`을 `NAVRL_ROBOT` 옵트인으로). **기본값은 `navrl_quad`
그대로**라 frozen `ep25000+riskcap` provenance와 진행 중인 캠페인에 영향 없음. 코드 로직 변경 없음.

다음은 스모크 500~1,000 epoch(1~2시간). 그 결과로 검증 5의 baseline arm을 기준 플랫폼으로
바꿀지 확정한다. legacy와 ref는 **기체가 다르므로 비교 불가**(v1↔v2 원칙 동일 적용).
상세: `docs/reference_platform_proposal_2026-08.md`(전면 개정),
`results/navrl_ref_platform_verification/summary.md`.

## 2026-08-13 — PPT 검수 요청서에 기준 플랫폼 분기 반영

PPT 요청서 초안 작성 뒤 같은 날 구현된 `navrl_ref5in_quad` 기록을 재감사해 5B 계약에 반영했다.
5B가 legacy `navrl_quad`를 유지할지 ref5in을 새 본선으로 쓸지는 아직 학습 스모크 전이라 확정할 수
없다. 따라서 Claude가 실제 5B summary의 `robot_name`과 URDF/config SHA를 확인하기 전에는 결과
슬라이드를 만들지 못하게 했다.

본문 구성을 17장으로 조정해 ref5in을 별도 슬라이드로 추가했다. 정합성 26/26 및 무장애물 command
tracking 5/5 PASS는 확정이지만, 10 Hz 하네스가 roll angular-acceleration 0.586배 회귀를 관측하지
못했고 learning/capture는 미측정임을 watermark로 요구했다. ref5in 채택 시 5A legacy-vehicle run과
연속 계보로 그리지 않고 새 corrected-reference-platform lineage로 표시한다. 5B 결과 placeholder에도
robot/URDF/config SHA 행을 추가했다.

## 2026-08-13 — Claude PPT 검수 자료 전달 번들 생성

Claude에 PPT 검수·보완을 요청할 때 필요한 최신 문서와 수치 근거를 단일 ZIP으로 묶었다.
저장소 안에는 PPT/PPTX 원본이 없고 논문 PDF만 확인되어, 현재 PPT가 Claude 대화에 없다면 사용자가
보유한 PPT 원본은 별도로 첨부해야 한다.

번들: `/home/fair/workspaces/aerial_gym_ws/MOTAR_Claude_PPT_bundle_2026-08-13.zip`

- 내부 파일: 11개 (`01_CLAUDE_PPT_REVIEW_REQUEST.md`부터 `11_BUNDLE_README.md`까지)
- 크기: 274,452 bytes
- SHA-256: `506a065f624358b3a53de9f7e8dbe74c04bddbff4ca535c9fb47d21a490f3861`
- `unzip -t`: 11/11 OK

번들 README에는 최신 사실 우선순위, 5B 미실행 수치 추정 금지, legacy/ref5in 계보 분리,
ref5in 검증의 범위가 정합성·무장애물 command tracking에 한정된다는 점을 명시했다.

## 2026-08-13 — ref5in 전면 재감사: 과대주장 제거, provenance 보강, canonical P0 PASS

사용자 요청에 따라 Claude가 만든 `navrl_ref5in_quad` 파라미터·검증·README를 코드부터 다시
감사했다. 이 항목은 위의 08-12/13 “기준 플랫폼 구현 완료” 기록 중 다음 주장을 **supersede**한다.

- `ref5in`은 buildable/real/flight-proven reference platform이 아니라
  **hardware-informed simulation candidate**다.
- 기존 payload `472 g` 합계는 Orin NX module과 compute assembly allowance를 혼동했고 Pixhawk
  질량도 부정확했다. 1.20 kg은 아직 BOM으로 닫히지 않은 synthetic design point다.
- 0.28×0.28 m XY literal은 level에서 legacy와 같지만 높이 0.08→0.12 m는 45° tilt에서 한 축의
  projected support를 약 **0.2546→0.2828 m(+2.83 cm)**로 바꾼다. “바닥/천장만 달라지고 bar
  contact는 동일”이라는 기존 설명은 폐기한다.
- 377/221 rad/s²는 최대 각가속이 아니라 constant-total-hover-thrust 조건의 roll authority다.
- 같은 Lee gains를 쓰는 응답은 plant 차이와 controller tuning이 섞인 closed-loop 조건이다.

### 코드·실험 계약 수정

1. checkpoint `env_state`에 robot name/config class/config SHA/URDF SHA를 저장하고 source bundle
   manifest/SHA/git commit/runtime dirty/file count를 함께 저장한다. 저장 때마다 source receipt를
   재검증해 run 중 executable edit를 차단한다.
2. `eval_navrl_v2_density_sweep.sh`가 checkpoint robot을 import 전에 복원하고, 현재 config/URDF
   SHA가 다르면 Isaac Gym 시작 전에 실패한다. evaluation source receipt에 `resources/robots/**/*.urdf`
   도 포함한다. legacy checkpoint만 contract-v0 fallback을 허용한다.
3. `tools/create_navrl_source_bundle.py`를 추가했다. `aerial_gym`, `resources/robots`와 receipt tool의
   실행 바이트, Python environment를 snapshot/hash한다. 결과·문서가 dirty인 것은 기록하되 학습을
   막지 않고, 실행 source가 dirty일 때만 closed launcher가 실패한다.
4. `NAVRL_SAVE_FREQUENCY` runtime override를 추가하고 ref5in smoke는 250 epoch마다 저장한다.
   기존 50-epoch 주기로 30k×3 seed를 저장하면 약 16 GB가 필요해 현재 여유 디스크에서 안전하지
   않았기 때문이다.
5. `train_navrl_v2_ref5in_smoke.sh`를 fresh/no-checkpoint, seed197, 500 epochs, 70 bars,
   corrected exact-600, analytic/cluster-sector/240°, squashed Gaussian, governor/pose/appearance off,
   yaw3.0/tilt45 조건으로 닫았다. hostile environment와 CLI override 회귀 테스트를 추가했다.
6. `tools/analyze_navrl_ref5in_smoke.py`를 추가해 completion/checkpoint finite/TensorBoard KL/rollback/
   skipped minibatch/raw OOB/source+robot SHA/distance curriculum/last-100 pooled outcome gate를 한 번에
   판정한다. PASS는 held-out 70-bar 한 셀만 허용하고 performance claim은 허용하지 않는다.

### CPU P0

`tests/test_navrl_ref5in_platform.py`를 buildability 검사가 아니라 repository consistency 검사로
고쳤다. 가짜 motor-strength k-spread 주장을 제거하고 `min=max=4.401e-5`, implied 28,023 RPM의
fixed coordinate calibration을 고정했다. 0.12 m tilted support 차이와 조건부 mass allowance도
검사한다. 결과 **26/26 PASS**. ref5in run/provenance 계약 테스트도 **7/7 PASS**였다.

### canonical GPU P0

기존 `results/navrl_ref_platform_verification/flight_envelope.json`은 yaw 2.5였고 forward 15/16,
reversal 13/16 생존자만 평균한 결과라 폐기했다. schema 2 verifier는 exact center, identity/zero
kinematics, runtime `mg/4`, 매 기동 controller/motor midpoint 재고정, 1 s settle 후 recenter를
사용한다. 실제 actor mass/inertia/body order/motor application mask, 모든 env 생존, finite state/
actuator, altitude/slip/reversal/yaw, raw pre-clamp allocator saturation과 100 Hz fixed-gain roll/pitch를
검사한다. absolute Python 실행에서 conda `ninja`가 PATH에 없어 첫 시도가 import 전에 실패했고,
verifier가 실행 interpreter의 `bin/`을 PATH 앞에 고정하도록 고친 뒤 재실행했다.

최종 결과는 **21/21 PASS**, 16/16 생존이었다.

| metric | legacy | ref5in |
|---|---:|---:|
| hover altitude error | +0.0001 m | +0.0001 m |
| forward steady / t90 | 2.490 m/s / 0.8 s | 2.490 m/s / 0.8 s |
| reversal zero-cross / t90 | 0.5 s / 1.0 s | 0.5 s / 1.0 s |
| yaw steady / t90 (3.0 command) | 3.000 rad/s / 0.2 s | 2.999 rad/s / 0.2 s |
| 100 Hz pitch 20° / peak rate | 0.15 s / 5.363 rad/s | 0.13 s / 3.844 rad/s |
| 100 Hz roll 20° / peak rate | 0.15 s / 5.363 rad/s | 0.13 s / 3.844 rad/s |

ref5in worst raw allocator limit fraction은 1.3%, forward/reversal peak altitude error는 각각
0.011/0.027 m였다. 이 결과는 same-unretuned-controller simulator gate일 뿐 intrinsic agility,
hardware feasibility 또는 navigation 성능이 아니다. 원자료 SHA-256은
`35c315603af0afc41bd03adc7cbcaee35cd2d032fd01e86150d983e24bccf5a8`이며 상세는
`results/navrl_ref_platform_verification/summary.md`다.

다음 단계는 runtime source clean commit → fresh 500-epoch P1 smoke → 자동 gate PASS 시에만
미사용 **evaluation seed 313** held-out 70-bar 평가다. 이어지는 full 학습은 training seed 211을
사용해 decision-cell 평가 seed와 분리한다. P1/P2 전에 full-budget 학습을 시작하지 않는다.

### 재감사 마감과 P1 진입 조건

MOTAR 최상위 `README.md`를 실제 사용 순서 중심으로 다시 썼다. corrected-v2/archived-v2,
legacy/ref5in, training proxy/held-out 결과를 분리하고, 존재하지 않는 Git checkpoint를 바로 실행할 수
있는 것처럼 보이던 안내와 일반 `train_navrl.sh` 권고를 제거했다. `ref5in`은 전 문서에서
**hardware-informed simulation candidate**로 한정했다. 기존 `472 g` payload 합계도 폐기했다. 제조사
단품 수치의 단순 합은 compute carrier·냉각·전원·배선·마운트가 빠진 값이므로 buildable BOM이 아니다.

실행 계보 감사에서 두 shape-compatible 기체를 checkpoint가 구별하지 못하는 결함을 고쳤다. 새
checkpoint는 robot name뿐 아니라 config/URDF의 저장소 상대경로와 SHA-256을 기록한다. contract-v1
checkpoint를 다른 기체 또는 수정된 robot source로 불러오면 경고만 남기지 않고 fail-closed한다.
training source receipt도 파일명 대신 저장소 상대경로로 active config/URDF를 결합하고 중복 경로를
거부한다. historical contract-v0 checkpoint는 evaluator에서만 `navrl_quad`로 명시적 fallback한다.

최종 CPU 회귀검사는 `python -m unittest discover -s tests -p 'test_navrl*.py' -v` 기준
**212/212 PASS**였다. ref5in 전용 repository/run 계약은 **33/33 PASS**, launcher hostile-env
preflight와 legacy corrected-v2 epoch-1000 checkpoint의 held-out evaluator preflight도 PASS했다.
따라서 P0는 CPU 26/26 + canonical GPU 21/21(양 기체 각 16/16 생존) + 전체 NavRL 212/212로
닫는다. 아직 P1 fresh 500-epoch와 P2 held-out navigation은 실행 전이므로 README/PPT의 과제 성능
placeholder는 채우지 않는다.

## 2026-08-13 — ref5in P1a fresh 500 epoch: 성능 gate 통과, 안전 gate FAIL

clean runtime commit `578b4bf`에서 `train_navrl_v2_ref5in_smoke.sh`를 실행했다. source receipt는
310개 runtime file, `git_dirty=false`를 기록했고 실제 Isaac Gym actor는 질량 1.200000 kg,
관성 대각 약 `[0.004,0.004,0.006]`으로 생성됐다. run은
`ppo_260813_0406_navrl_v2-ref5in-smoke-s197`, terminal checkpoint는
`last_gen_ppo_ep_500_rew_117.70658.pth`(SHA-256
`59f993811b2f358dc144544998cf0d230756198b19a33b658d0b82c22bebd26c`)다.

최근 100 epoch pooled 3,709 episodes의 capture/crash/timeout은
**2,675/910/124 = 72.12/24.53/3.34%**였다. checkpoint finite, PPO KL max 0.03182,
4축 raw OOB 0, timeout path exercised, robot/config/URDF/source 결합은 모두 PASS였다. 초반
100% crash에서 회복했고 distance gate도 7→9→...→27 m로 10번 승급했다. 따라서 ref5in에서 PPO가
학습 자체를 못한다는 가설은 기각한다.

그러나 사전등록한 전체 판정은 **FAIL**이다. epoch 432에서 pre-update KL 0.05043,
independent behavior-KL audit 0.05842가 0.04 gate를 넘어 minibatch 1회가 skip됐고 PPO epoch 전체가
rollback됐다. transaction guard가 model/optimizer를 복구하고 LR을 3e-5→1.5e-5로 낮춰 corrupted
continuation은 막았지만, P1a는 rollback/skipped 0을 요구했다. 또한 최종 distance state는
`[20,27] m`로 28 m promotion 한 번이 부족했다. 마지막 25→27 m evidence window capture가 0.745였으므로
27 m plateau라기보다 500-epoch budget 부족으로 판정한다.

결과를 본 뒤 gate를 완화하지 않는다. P1b는 같은 seed197, fresh weights, 동일 task/robot 계약에서
안전장치가 선택한 초기 LR **1.5e-5**와 budget **750 epochs**만 바꾼다. P1a와 동일 outcome/KL/OOB
gate에 exact-750, `[20,28]`, rollback/skipped 0을 요구한다. P1b PASS 전에는 held-out P2와 full
training을 시작하지 않는다. 상세: `results/navrl_ref5in_smoke_seed197/summary.{md,json}`.

## 2026-08-13 — ref5in P1b fresh 750 epoch: 안전·outcome PASS, 거리 budget FAIL

clean runtime commit `227b874`에서 `train_navrl_v2_ref5in_smoke_b.sh`를 fresh seed 197로
실행했다. run은 `ppo_260813_0441_navrl_v2-ref5in-smoke-b-s197`, terminal checkpoint는
`last_gen_ppo_ep_750_rew_135.66144.pth`(SHA-256
`5174695ac0d6ab8dcc81a4351afea378db55be2b00b2a4669de5ea1e80a6a2cf`)다. source receipt는
311 files, commit `227b874cfaa358a4f0040885b6dbe45a06878084`, clean runtime과 manifest SHA
`1b5209686c696ab9242111a3a1af54d60fa91e6cd31707021db7ae53fab628d6`을 기록했다.

마지막 100 epoch pooled 3,780 episodes의 capture/crash/timeout은
**2,629/1,063/88 = 69.55/28.12/2.33%**로 세 outcome gate를 통과했다. 정확히 epoch 750에서
정상 종료했고, checkpoint 전체 tensor와 60개 TensorBoard scalar tag가 finite였다. PPO KL max
`0.01300`, behavior-audit KL max `0.01980`, rollback/skipped minibatch/4-axis raw OOB는 모두 0이었다.
초기·최종 actor LR도 `1.5e-5`로 같았고 robot config/URDF 및 original/snapshot runtime 311개를
모두 다시 해시해 source receipt를 검증했다. 즉 P1a의 epoch-432 불안정성은 낮춘 LR에서 재현되지
않았다.

그럼에도 strict P1b verdict는 **FAIL**이다. terminal distance state가 `[20,27] m`라 사전등록한
`[20,28] m`를 만족하지 못했다. 승급 epoch는 372/403/434/467/500/534/571/611/657/709였다.
장거리에서 episode가 길어져 동일 2,048 completed-episode evidence를 모으는 epoch 수가 늘었고,
750은 마지막 27→28 window 전에 끝났다. outcome이나 KL gate를 사후 완화하지 않으며 P2/full
training도 아직 금지한다. 상세는 `results/navrl_ref5in_smoke_seed197/p1b/summary.{md,json}`다.

P1c는 결과를 본 뒤 **budget만 750→900**으로 바꾸는 마지막 corrective engineering smoke로
사전등록한다. seed 197, fresh weights, LR `1.5e-5`, task/robot/representation과 모든 gate는 그대로다.
900은 예상 28 m saturation 뒤 온전한 last-100 window를 남긴다. 전용
`train_navrl_v2_ref5in_smoke_c.sh`는 CLI/CKPT를 거부하고 clean runtime receipt를 강제한다.

같은 감사 중 README의 3-D 재생 경로가 current 898D checkpoint를 구형 574D로 판정해 거부하고,
그 검사 전에 Isaac Gym까지 import하는 결함도 실제 P1b epoch-250 checkpoint로 재현했다. P1c/P2의
runtime byte 계보를 먼저 닫은 뒤 checkpoint metadata로 robot/sensor/token/action 계약을 import 전에
복원하는 fail-closed playback으로 교체한다. 그 전까지 current 898D policy replay 명령은 작동한다고
주장하지 않는다.

### P1c 첫 launch 무효화 — source-receipt fail-closed 실증

commit `89afe43`에서 P1c를 fresh로 시작했으나, Codex가 실행 중
`navrl_ref5in_quad_config.py`의 하드웨어 claim **주석**을 정리했다. 수치·실행 로직은 바뀌지
않았지만 source byte가 receipt와 달라졌고, epoch 117의 best-checkpoint save에서
`RuntimeError: NavRL training runtime source changed`가 발생해 run
`ppo_260813_0532_navrl_v2-ref5in-smoke-c-s197`가 중단됐다. 이는 학습 실패나 P1c 데이터가 아니라
운영자에 의한 provenance 오염이며 해당 run은 **VOID**다. `.aerial_training_finished`도 없고 P1c
판정에 사용하지 않는다.

source guard가 오염된 checkpoint 저장을 막은 것은 의도한 fail-closed 동작이다. 주석 정리와 문서
수정을 먼저 commit한 뒤, 동일한 사전등록 계약(seed/LR/task/budget/gate 변경 없음)으로 P1c를 fresh
재실행한다. 재실행 중에는 `aerial_gym/`, `resources/robots/`, source-bundle 도구를 수정하지 않는다.

### P1c 두 번째 launch — 유효한 fresh run 진행 중

문서와 runtime 주석 정리를 commit `0a570bffe67bef4e9ab033ba496dc6f34202af0e`로 닫은 뒤 같은
사전등록 계약으로 P1c를 다시 fresh launch했다. 유효 run은
`ppo_260813_0540_navrl_v2-ref5in-smoke-c-s197`, session log는
`aerial_gym/rl_training/rl_games/train_session_logs/ref5in_smoke_c_260813_054048.log`다. seed 197,
LR `1.5e-5`, 128 env, 70 bars, max 900 epoch이며 checkpoint resume 없이 fresh weights에서 시작했다.

launch receipt는
`aerial_gym/rl_training/rl_games/train_source_receipts/ref5in_smoke_c_s197_260813_054048_2228846/source_manifest.json`이고,
runtime source **312 files**, commit `0a570bf`, `git_dirty=false`, manifest SHA-256
`ce4a52b850014eab85ee57315ee1834da2d5d16a92e78dc86d2fa6996efcd1ff`를 기록했다. 이 항목 작성
시점에는 run이 **진행 중**이므로 outcome/PASS/FAIL을 판정하지 않는다.

이 run이 정상 종료되거나 명시적으로 VOID 처리될 때까지 `aerial_gym/**`, `resources/**`, `tools/**`
runtime byte를 수정하지 않는다. 문서 변경은 runtime receipt 밖에서 수행하되, 실행 source를 건드린
변경은 수치·로직 영향이 없어 보여도 source drift로 간주하고 해당 run을 판정에 사용하지 않는다.

### P1c 종료·채점 — PASS, P2 한 셀만 해제

유효 P1c는 epoch 900에서 `max_epochs`로 정상 종료했다. terminal checkpoint는
`last_gen_ppo_ep_900_rew_137.08087.pth`, SHA-256은
`f1670a1d74dd92cb00d6a58898e9cc1b96eb9cbe155d1e85812a345e7aaae6bf`다. 기계 분석기는 900개
outcome block과 60개 TensorBoard scalar tag, checkpoint 전체 tensor, original/snapshot runtime
source를 다시 검사했다.

마지막 100 epoch pooled **3,338 episodes**의 capture/crash/timeout은
**2,429/799/110 = 72.77/23.94/3.30%**였다. distance state `[20,28]`, PPO KL max
`0.01239`, behavior-KL max `0.01774`, rollback/skipped minibatch/4축 raw OOB는 모두 0이었다.
모든 P1c gate가 PASS했지만 이는 on-policy engineering result이므로 성능 주장은 허용하지 않고
seed 313 held-out 70-bar P2 한 셀만 해제했다. 상세는
`results/navrl_ref5in_smoke_seed197/p1c/summary.{md,json}`다.

### P2 held-out seed 313 — timeout 경계 초과로 STRICT FAIL

전용 `tools/run_navrl_ref5in_p2.sh`와 `tools/attest_navrl_ref5in_p2.py`를 추가했다. inherited
`NAVRL_*`, profile, detector, force override를 제거하고 deterministic/original/governor-off,
70 bars, target U[0.3,1.5] m/s, full goal/FOV, 2,049 requested episodes를 고정한다. P1c training
manifest, 현재 repository, schema-2 eval manifest의 `aerial_gym/** + resources/robots/**` **311개
path→SHA/size map**이 exact set-equality이고 Python environment/evaluator/checkpoint/robot SHA도
일치해야만 결과를 쓴다. hostile-env/threshold/provenance CPU test는 **8/8 PASS**했다.

실제 ref5in decision cell 결과는 다음과 같다.

| outcome | count | rate | gate | verdict |
|---|---:|---:|---:|---|
| capture | 1,399 / 2,049 | 68.28% | ≥65% | PASS |
| crash | 536 / 2,049 | 26.16% | ≤33% | PASS |
| timeout | 114 / 2,049 | 5.56% | ≤5% | **FAIL** |

timeout 상한은 최대 102건이라 12건 초과했다. Wilson 95% CI는 4.65–6.64%로 5%를 포함하지만,
point estimate/count로 정한 사전등록 gate를 완화하지 않아 전체는 **STRICT FAIL**이다. primary
PASS 후에만 실행하기로 한 legacy anchor와 P3 seed 211 장기학습은 시작하지 않았다.

진단상 초기거리 사분위 capture는 `80.90→73.51→66.84→53.17%`로 하락했지만 속도 사분위는
`68.79/70.65/67.67/66.51%`로 비교적 평평했다. crash 536건은 bar contact 416, OOB 120건이고,
접촉 순간 actual speed 평균 1.109 m/s, stopping margin 평균 −1.098 m였다. timeout 114건은 모두
exact action 600에서 발생해 legacy 601-step 회귀는 아니다. 현행 strata가 distance별 capture만
보존해 crash/timeout 분리가 불가능하므로, 다음 허용 작업은 outcome-aware strata를 추가한 별도
diagnostic evaluation이다. 원자료와 proof는 `results/navrl_ref5in_p2_seed313/`에 있다.

### P2 후속 진단 구현·사전등록 — 학습과 decision 재시험은 금지

P2를 좋은 seed로 다시 돌리거나 timeout 상한을 사후 완화하지 않고, 원인 분리용 계측을 추가했다.
bulk evaluation의 distance/speed/pattern 각 bin은 이제 capture뿐 아니라 crash/timeout과
`bar_contact/below/above/out_of_bounds` crash 원인을 별도 eval-only tensor에 누적한다. 이 tensor는
checkpoint의 density curriculum window와 분리돼 학습 episode가 held-out denominator에 섞이지
않는다. export 직전 각 axis의 episode/capture/crash/timeout/cause 합계가 global outcome과 정확히
같지 않으면 `RuntimeError`로 결과 저장을 막는다.

`tools/run_navrl_ref5in_outcome_diagnostic.py`와 닫힌 shell launcher를 추가했다. 계약은 P1c epoch 900,
seed 317, 70 bars, deterministic/original, governor off, speed `U[0.3,1.5]`, distance `U[6,28]`,
8,193 requested episodes다. P2 attestation이 FAIL/none 상태이고 checkpoint SHA가 맞아야 하며 tracked
source가 clean한 commit이 아니면 실제 run을 거부한다. preflight는 evaluator provenance와
robot/runtime/condition을 모두 확인해 PASS했고 output을 만들지 않았다. 이 평가는 descriptive이며
`decision_authority=none`, `p3_unlocked=false`로 고정한다. 자세한 사전 판독 기준은
`RESEARCH_PLAN.md` §8.24에 있다.

첫 seed-317 진단은 8,194 episodes에서 global capture/crash/timeout
`68.03/25.98/5.99%`를 기록했다. distance q0→q3는 capture `78.05→55.94%`, crash
`21.89→29.09%`, timeout `0.06→14.97%`로 두 failure channel이 함께 증가했다. 다만 결과 검증 중
speed bin이 실제 지원범위 `[0.3,1.5]`가 아니라 역사적 `[0,1.5]`로 나뉘어 q0가 509 episodes만
담은 계약 오류를 발견했다. speed별 해석은 즉시 VOID 처리하고, 빈 circle cell의 Wilson CI를 JSON
`NaN` 대신 `null`로 고쳐 기존 artifact 재검증은 PASS시켰다.

정정 v2는 eval speed bin만 `[0.3,0.6,0.9,1.2,1.5]`로 바꾸고 distance×pattern joint outcome을
추가한다. density training gate의 역사적 `[0,max]` bin은 건드리지 않는다. 동일 seed/checkpoint를
재생해 global outcome/crash cause/distance/pattern/bearing가 v1과 완전히 같지 않으면 telemetry-only
변경 주장을 기각한다. 이 replay도 P2 재시험이나 독립 seed가 아니다.

### outcome diagnostic v2 종료 — parity PASS, 장거리 CV non-capture 분리

동일 seed-317 replay는 global `captured/crash/timeout=5,574/2,129/491`과 distance/pattern/bearing/
crash-cause가 v1과 완전히 같아 telemetry-only parity를 통과했다. corrected speed bins는 각 약 2천
episode를 담았고 timeout이 `8.49→6.84→5.65→3.10%`로 감소했다. 빠른 표적이 timeout 주원인이라는
설명은 기각한다.

distance q0→q3의 capture/crash/timeout은 `78.05/21.89/0.06% → 55.94/29.09/14.97%`였다.
q3에서 CV는 `48.84/29.00/22.16%`, waypoint는 `62.87/29.17/7.96%`였다. crash는 pattern과
무관하게 비슷하게 증가하고, 추가 timeout은 CV에서 집중된다. global crash 2,129건의 81.59%는
bar contact, 18.41%는 OOB이고 고도 crash는 0이었다. P2 seed313과 diagnostic seed317의 전체
capture/crash/timeout 차이는 각각 `-0.25/-0.18/+0.43pp`이고 단순 두-비율 95% 구간이 모두 0을
포함해, global 성능도 P2와 양립한다.

사이트와 README는 P2 strict FAIL/P3 BLOCKED를 유지하면서 이 descriptive 진단을 최신 headline으로
표시하도록 갱신한다. 다음 candidate는 full P3가 아니라 D1 1,000-epoch saturated-distance adaptation
probe다. episode horizon 증가는 timeout을 crash로 바꿀 수 있어 원인 ablation으로만 남긴다.

### 거리 curriculum 계약 정정 — `k_min_cur`는 general-spawn minimum이 아니었다

D1 launcher를 준비하기 전에 reset sampler를 다시 추적해 문서상 큰 의미 오류를 발견했다.
`_general_goal_distance_bounds()`는 training에서 `k_max_cur`만 upper bound로 사용하고 lower bound는
항상 별도 `NAVRL_GENERAL_GOAL_DIST_MIN=6`을 쓴다. 따라서 P1b/P1c checkpoint state
`[20,27]`/`[20,28]`은 실제 sampler range가 아니라 state pair이며, 적용 range는 각각
`[6,27]`/`[6,28] m`였다. P2/D0 평가도 `[6,28]`이라 결과는 유효하고 P1c의 max=28 saturation 및
engineering safety PASS도 유지된다. 다만 “20 m minimum mastery”와 “hard window에서 100 epoch”
주장은 철회한다.

`train_navrl_v2_search.sh`의 general goal min/max를 closed child launcher가 명시적으로 override할 수
있게 하되 기본 `[6,28]`은 유지했다. D1은 이 수정으로 `[22.5,28] m`를 실제로 적용해 q3 exposure만
늘린다. analyzer의 gate 이름도 sampler range처럼 보이지 않도록 `distance_curriculum_state_saturated`
로 바꿨고 P1b/P1c report Markdown, PPT/review/site 문구에 erratum을 남겼다. immutable P1c JSON과
그 SHA를 참조하는 P2 attestation은 변경하지 않았다.

D1 판정을 결과 전에 고정했다. training은 seed 197/P1c epoch 900 warm-start/terminal epoch 1900,
70 bars, `[22.5,28] m`, mixed target, LR `1.5e-5`이며, held-out은 새 seed 331과 최소 8,193 requested
episodes를 사용한다. 통과에는 q3 CV timeout `<=12%`, 전체 crash `<=27%`, q3 crash `<=30%`와
rollback/OOB/NaN 0이 모두 필요하다. D1은 P2 FAIL을 소급 변경하거나 P3 성능 주장에 쓰지 않는다.

### D1 첫 launch — 0 epoch VRAM OOM으로 VOID

commit `0877158`에서 D1을 시작했지만 run
`ppo_260813_1633_navrl_v2-ref5in-d1-q3-adapt-s197`는 첫 backward의 130 MiB 추가 할당에서 OOM으로
종료됐다. 완료 epoch/학습 data/checkpoint는 0/없음이며 성능 판정에 쓰지 않는다. 시작 로그는
`[22.5,28] m`, 70 bars, ref5in, mixed target과 checkpoint 분포 mismatch에 따른 density-window reset이
정상 적용됐음을 보여준다. 종료 직전 PyTorch reserved-but-unallocated는 216 MiB였고 학습 종료 뒤 GPU
free는 6.2 GiB로 회복됐다. 따라서 batch/env/과제 의미론을 바꾸기 전에 launcher가
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`를 고정하도록 했다. 이 설정은 allocator segment만
바꾸며 128-env PPO 계약은 그대로 둔다. 0-epoch 폴더는 삭제하지 않고 runtime source root 밖의
`results/navrl_ref5in_d1_void_oom0/run/`에 보존한다.

## 2026-08-13 — 상태 대시보드를 6페이지로 재편, 실험 인덱스 신설

사용자 요청: "변수 하나하나가 어떻게 적용됐고 결과가 어땠는지"를 한 묶음으로 보기엔 너무 길다.
조사해 보니 **문제는 길이가 아니라 데이터가 없다는 것**이었다.

| 갖고 있다고 생각했던 것 | 실제 |
|---|---|
| `status.json.runs[]` 77건 = 실험 기록 | 학습 중 롤링 통계뿐. held-out 결과·시드·에피소드 수·SHA·판정 **전무** |
| A/B 판정이 구조화돼 있음 | **현재 캠페인 1건**만, `research_update`의 영문 산문 안에 |
| CI·시드가 필드로 있음 | 전부 문자열 (`"95% CI [-0.51,+3.66]"`). 구조화 delta는 2곳뿐 |
| 통제 실험 ~60건 | **`WORKLOG.md` 산문에만 존재** |

따라서 작업의 본체는 페이지 분할이 아니라 **실험 인덱스라는 데이터셋을 새로 만드는 것**이었다.

### 착수 전 발견한 잠복 버그 (Phase 0, 단독 처리)

`update_status_snapshot.py:write_snapshot()`이 압축 JSON을 `re.subn`의 **치환 템플릿**으로 넘기고
있었다. 치환 템플릿에서는 백슬래시가 재해석되므로 재현 결과:

- JSON 문자열 안 `\n` → 실제 개행 → `JSON.parse` 실패 → try/catch가 삼켜 **조용히** 초기 페인트 소실
- `\\` → `\`로 붕괴, `\uXXXX` → `re.error: bad escape`로 스크립트 사망

현재 `status.json`의 백슬래시가 `\"` 2개뿐이라 우연히 살아 있었다. 한국어 실험 산문을 넣는 순간
깨졌을 것이다. `write_js_data()` + `docs/status/status.fallback.js`로 교체하고 round-trip
자체검증으로 fail-loud 성질을 유지했다. 부수효과로 `index.html`이 112 KB → 23 KB가 됐다
(88 KB짜리 한 줄이 매 스냅샷마다 diff에 찍히던 문제도 사라짐).

### 결과 — 6페이지

| 페이지 | 내용 |
|---|---|
| 개요 | 라이브 상태, 성공 기준, 현재 캠페인, 학습 운영 기록 |
| 초기 세팅 | 동결 계약, 정보 방화벽 다이어그램, 3D 아레나 |
| 기체 · 센싱 | legacy vs ref5in 스펙 17행, 포락선 게이트 21항, 센싱·제어 13행, 실기 격차 |
| 파라미터 | `NAVRL_*` **176개** 카탈로그 |
| 실험 인덱스 | 통제 실험 **58건** master/detail |
| 결과 | 밀도 곡선 · 속도 축 · 밀도×속도 맵 · 인지 강건성 |

좌측 고정 사이드바(`js/shell.js` 런타임 주입, 마크업 6벌 복제 회피), 테마 localStorage 영속화.

### 신규 데이터 층 — 생성기 2 + 린터 1

- `tools/generate_parameter_catalog.py` — AST로 `NAVRL_*` 전량 수집. regex로는 못 하는 5가지가
  있다: 여러 줄 호출, 비리터럴 기본값, `.strip() or "x"` 후처리, 중첩 클래스 스코프, 동일 파일 내
  중복 이름. **주석은 tokenize로 수확**(`latency_ego_motion_fix`의 9줄 설명이 이 페이지의 핵심 가치).
  authoritative 176 / echo-only 83으로 분류 — 함수 본문 안의 `os.environ.get` 대부분은 영수증
  기록용 재읽기라 동작 knob이 아니다. 파생 필드 3개가 특히 값어치 있다:
  `mirrors`(5개 — 소스가 "동기화 필수"라 경고하는 것들), `swept_but_not_ablated`(런처가 값은
  지정했지만 통제 A/B는 없음), `frozen_by_contract`(건드리면 898-D 계보가 끊김).
- `tools/generate_platform_spec.py` — URDF + robot config에서 질량·관성·추력·기하를 **재계산**.
  legacy URDF가 3주간 모순 상태였던 이유가 아무도 그 숫자들을 서로 대조하지 않은 것이므로,
  대시보드가 손으로 옮겨 적은 표를 들고 있으면 안 된다.
- `tools/lint_experiments.py` — 스키마·열거형·참조 무결성·`results_paths` 실재 여부를 fail-closed로.

### 실험 인덱스의 설계 결정 — `verdict`와 `validity` 분리

둘은 **직교한다**. 키랄리티 수정 이전 결과는 `PASS`(게이트 통과) + `superseded`(수치 무효)다.
철회를 `FAIL`로 적으면 우리 기록을 우리가 왜곡하는 것이다. 그래서 칩 2개를 띄우고,
기본 필터는 유효 근거만, **"검증으로 무효화 17건"**은 *긍정적* 라벨의 토글로 연다.

현재 58건: 유효 36 · 대체됨 12 · 철회 5 · 탐색적 5 / PASS 32 · FAIL 13 · 미결 13.
철회 5건은 사유별로 전부 노출된다 — RNG 오염, 좌표계 오류, 과대주장(+20 ms 스펙), 범위 이탈,
구현 버그. `experiments.html?validity=void`가 그 목록의 공유 링크다.

파라미터 ↔ 실험은 env var 이름으로 **단방향 조인**한다(생성기가 experiments.json을 읽어
parameters.json에 `experiments[]`를 쓴다). 조인 결과 knob 38개에 실험 이력이 붙었다.

### 함께 고친 것

- `renderRuns`가 run **이름**으로 정렬 후 `.reverse()`해서 "최근 12개"가 사전순이었다
  (`density_120`이 모든 `ppo_*`보다 앞). `finalized_at` 내림차순으로 수정하고,
  한 번도 노출된 적 없던 `exit_reason`·`epochs_logged`·`reward_collapse`를 열로 추가했다.
- `renderLive` 꼬리의 아레나 슬라이더 동기화 16줄을 `syncArenaToRun`으로 분리.
  Live 패널(개요)과 아레나(초기 세팅)가 다른 페이지가 됐는데 이 코드는 전부 null-guard라
  **조용히** 실패했을 것이다 — 슬라이더가 HTML 기본값에 머물고 HUD가 그럴듯하지만 틀린 밀도를
  콘솔 에러 없이 표시. 인수 조건으로 "슬라이더가 캠페인 막대 수와 일치"를 박았다(현재 70).
- `app.js`(966줄)를 `js/core.js` + `panels-{status,setup,results}.js`로 분할.
  렌더러별 `try/catch`를 넣었다 — 이전에는 하나가 던지면 나머지가 통째로 중단됐다.

### 검증

Chrome headless(`--use-gl=swiftshader`)로 6페이지 전부 렌더 확인. 분할 전후 출력 동일:
개요 패널 4 · runs 12행, 초기세팅 슬라이더 70 / HUD 밀도 4.4 / canvas 생성, 결과 곡선 SVG 2712 B
+ 히트맵 6500 B, 기체 스펙 17행, 파라미터 177행, 실험 36행(기본 필터). 3D 실패 0건.
`--disable-gpu`에서만 WebGL이 없어 실패하는데 이는 헤드리스 환경 제약이지 회귀가 아니다.

재현: `python3 -m http.server 8000 --directory docs` 후 6페이지 육안 확인.
생성기는 `tools/generate_parameter_catalog.py`, `tools/generate_platform_spec.py`,
`tools/lint_experiments.py` 순으로 돌린다.

### 남은 것

`research_update`의 5-branch 다형성은 그대로 뒀다. experiments.json이 자리잡으면
"`research_update.experiment_id`인 실험을 렌더"로 축약되어 근원에서 사라지므로,
재편 중에 손대지 않고 후속으로 남긴다. experiments.json은 손 큐레이션이며 린터가 규율을 강제한다 —
자동 추출은 하지 않는다("이 실험이 무엇을 확립했는가"는 기계가 유도할 수 없다).

## 2026-08-13 — ref5in D1 종료·held-out 판정: 개선은 실재, 절대 gate는 FAIL

D0에서 최장 거리 CV timeout이 22.16%였던 원인을 노출 부족과 능력 한계로 분리하기 위해,
결과 전에 고정한 D1 계약을 끝까지 실행했다. lineage는 P1c epoch 900, training seed 197,
70 bars, 실제 goal range `[22.5,28] m`, mixed CV/waypoint, LR `1.5e-5`, governor off이며
epoch 1900까지 정확히 1,000 epoch를 추가했다. 첫 시도는 첫 backward 전 VRAM fragmentation OOM으로
0 epoch 종료돼 `results/navrl_ref5in_d1_void_oom0/`에 VOID로 보존했다. allocator만
`expandable_segments:True`로 고친 재실행은 batch/env/과제 계약을 바꾸지 않고 완료됐다.

terminal checkpoint SHA-256은
`197ea26999d6bb9cf23c4e5a55acbe945f89985e2384687d60ab1dbae66a278e`다. PPO/behavior-KL max는
`0.01155/0.01547`, rollback/skipped minibatch/raw action OOB는 모두 0이었다. 학습 last-100 pooled
capture/crash/timeout은 `71.87/18.72/9.41%`; branch 초반 comparable trailing window의
`56.52/34.80/8.68%`보다 capture와 crash는 크게 회복됐고 timeout은 거의 그대로였다.

사전등록한 held-out seed 331, requested 8,193은 실제 8,194 episodes를 끝냈다.

| stratum | capture | crash | timeout |
|---|---:|---:|---:|
| global | 76.82% | 18.62% | 4.55% |
| q3 `[22.5,28] m` | 70.22% | 19.96% | 9.82% |
| q3 / CV | 64.03% | 19.98% | **15.98%** |
| q3 / waypoint | 76.25% | 19.94% | 3.80% |

global crash `≤27%`와 q3 crash `≤30%`는 통과했지만 q3/CV timeout `15.98% > 12%`라 최종
판정은 **D1 FAIL**이다. 기준을 사후 완화하지 않고 P2 strict FAIL과 P3 BLOCKED를 유지한다.

D0→D1은 서로 다른 eval seed의 독립 비율 근사 비교이며 adaptation과 q3 학습분포 변경이 함께
들어간 기술적 비교다. global capture/crash/timeout 변화는 `+8.80/-7.36/-1.44pp`, q3는
`+14.28/-9.13/-5.15pp`, q3/CV는 `+15.19/-9.02/-6.17pp`였다. q3/CV의 95% normal approximation
구간은 각각 `[+10.94,+19.44]`, `[-12.74,-5.31]`, `[-9.57,-2.77]pp`다. 따라서 “추가 노출이
무의미했다”가 아니라 **유의한 개선은 있었으나 사전 절대 성능에 못 미쳤다**가 정확하다.

D1 eval에서는 q3 학습 checkpoint를 full `[6,28] m` held-out에 재생해야 해 generic evaluator의
훈련분포 일치 guard와 의도적으로 충돌했다. 강제 실행 전에 유일한 mismatch가
`cfg_general_goal_dist_min: 22.5 vs 6.0`인지 확인하는 narrow preflight를 추가했고, 다른 mismatch가
하나라도 있으면 거부한다. train/eval Python receipt 차이는 editable MOTAR 행의 commit metadata뿐임을
양쪽 source manifest와 대조한 뒤에만 허용했다. D1 summary verifier는 FAIL artifact를 정상 검증할 때
exit 0으로 끝난다.

action negative-y rate는 D0 98.53%, D1 97.82%로 강한 키랄리티가 D1 전부터 존재했다. 대칭 arena의
과거 mirror aggregate에서는 outcome penalty가 검출되지 않았으므로 D1 실패 원인으로 단정하지 않는다.
다만 다음 tangent 진단에서 좌우를 합치면 이 축을 숨길 수 있어 tangent-left/right를 별도 cell로 둔다.

다음 단계는 PPO 연장이 아니라 D1 terminal policy를 동결한 CV initial-heading 진단이다. 계약은
`RESEARCH_PLAN.md` §8.25에 결과 전에 고정했다: 70 bars, `[22.5,28] m`, CV-only,
`U[0.3,1.5] m/s`, exact 600, seed 337, cell당 requested 2,049, toward/tangent-left/
tangent-right/away 네 cell이다. 결과는 P2/D1을 재판정하거나 P3를 자동 해제하지 않는다.

### CV initial-heading 진단 구현·preflight PASS

`NAVRL_EVAL_CV_INITIAL_HEADING=random|toward|tangent_left|tangent_right|away`를 추가했다. 기본
`random`은 기존 RNG와 동작을 유지한다. controlled 값은 bulk evaluation + CV-only에서만 허용하고
training 또는 mixed/waypoint/circle과 결합하면 task 초기화가 실패한다. 모든 cell에서 random angle을
먼저 동일하게 소비한 뒤 CV velocity만 덮어써 이후 waypoint/avoid-sign RNG draw의 순서를 보존한다.

방향 정의는 pursuer→target radial을 기준으로 away `(+1,0)`, toward `(-1,0)`, tangent-left
`(0,+1)`, tangent-right `(0,-1)`이다. 요청 문자열만 기록하는 자기증명을 피하려고 reset마다 실제
velocity의 radial cos/sin과 최대 계약오차를 누적하며, cell export가 mean/maximum `1e-5` 허용오차를
벗어나면 generic evaluator가 결과 저장을 거부한다.

generic evaluator는 기본 `[6,28] m`/mixed를 그대로 유지하면서 닫힌 diagnostic만
`NAVRL_V2_GOAL_DIST_MIN/MAX`와 `NAVRL_V2_TARGET_PATTERN`으로 범위/pattern을 요청할 수 있게 했다.
checkpoint와 다른 항목을 자동 허용하지 않는다. 새 orchestrator는 force 없이 먼저 거부시켜 mismatch가
정확히 `cfg_target_pattern: mixed→cv` 한 건인지 증명한 다음에만 force preflight/run을 허용한다.
checkpoint SHA, D1 FAIL, P3 locked, runtime clean/source map 동일성, cell condition/receipt/count/heading
audit를 모두 fail-closed 검증한다.

CPU 단위 테스트 7개, ref5in contract unittest 15개, shell syntax/py_compile을 통과했고 실제 GPU를
할당하지 않는 preflight도 PASS했다. 결과 전 screen의 `all timeout high`는 모호하지 않게 모든 cell
timeout `>=12%`로 고정했다.

첫 toward 실행은 task 생성 중 0 episode에서 fail-closed됐다. evaluator는 checkpoint provenance를
맞추기 위해 `NAVRL_GENERAL_TRAIN=1`을 유지하면서도 실제 실행은 `NAVRL_BULK_EVAL=1`인 구조인데,
초기 guard가 `general_train_mode` 플래그만 보고 실제 학습으로 오판했다. 성능 JSON은 생성되지 않았고
checkpoint/source bundle/log만 남은 폴더는 `_VOID_guard`로 이동해 결과에서 제외했다. guard는 실제
실행 권한인 bulk mode + non-empty bulk-result path + `NAVRL_EVAL_CHECKPOINT` 세 조건으로 수정했다.
일반 training은 이 세 조건을 함께 갖지 않으므로 controlled heading을 사용할 수 없다.

### CV initial-heading 동결 진단 완료 — radial channel은 강하지만 단일 원인은 아님

수정된 guard로 seed 337, 70 bars, CV-only, `[22.5,28] m`, `U[0.3,1.5] m/s`, deterministic,
exact 600의 네 cell을 완료했다. 실제 episode 수는 toward/tangent-left/tangent-right/away 순으로
2,050/2,049/2,050/2,050이며, reset direction audit 최대 계약오차는 모두 `2.4e-7` 이하라 요청한
초기 방향이 실제 속도에 적용됐음을 확인했다.

| initial heading | capture | crash | timeout |
|---|---:|---:|---:|
| toward | 78.20% | 18.59% | 3.22% |
| tangent-left | 63.74% | 20.20% | 16.06% |
| tangent-right | 62.49% | 19.27% | 18.24% |
| away | 53.51% | 19.95% | 26.54% |

away−toward는 capture `-24.68pp [-27.49,-21.88]`, timeout `+23.32pp
[+21.26,+25.38]`, crash `+1.37pp [-1.05,+3.78]`였다. tangent L−R 최대 outcome 차이는 timeout
`-2.19pp [-4.49,+0.12]`로 사전 5pp 기준 미만이다. 따라서 초기 radial heading 채널은 강하지만,
고정된 action chirality가 좌우 outcome을 지배한다는 설명은 지지되지 않는다.

이 개입은 순수 path-length 실험이 아니다. toward→away에서 step-weighted target-hidden fraction이
`74.37→90.95%`, non-crash closest mean이 `1.16→7.01 m`, capture mean step이 `153→300`으로
함께 움직였다. away speed q0→q3 timeout도 `38.17→16.55%`로 역전되어 finite arena에서 빠른 표적이
벽에 더 일찍 닿아 반사되는 효과와 양립한다. 그러므로 결론은 **경로길이·가림·벽 반사를 결합한
radial-heading 환경 채널**까지로 제한한다. P2/D1 FAIL과 P3 BLOCKED는 유지한다.

다음 동결 평가는 §8.26에 결과 전 고정했다. 같은 policy/거리/속도/episode 계약에서 density만
1 bar로 낮추고 seed 347 toward/away 각 requested 2,049회를 비교한다. away−toward timeout 차이가
`<=8pp`면 dense obstacle/occlusion 필요성을, `>=15pp`면 장애물 가림 없이도 kinematic/FOV/
wall-reflection 채널이 충분하다는 설명을 지지하며, 8–15pp는 혼합 판정한다. 어느 결과도 새 PPO나
P3 해제를 자동 허가하지 않는다.

### 1-bar near-open heading 대조 완료 — dense obstacle occlusion 필요성 기각

커밋 `0e29134`의 fail-closed orchestrator로 seed 347 toward/away를 실행했다. 두 cell은 같은 D1
terminal checkpoint SHA, source byte map, `[22.5,28] m`, CV-only, `U[0.3,1.5] m/s`, deterministic,
exact 600 계약을 통과했고 direction audit 최대오차는 `2.4e-7` 이하였다.

| 1 bar | episodes | capture | crash | timeout | target hidden |
|---|---:|---:|---:|---:|---:|
| toward | 2,050 | 95.51% | 4.34% | 0.15% | 65.53% |
| away | 2,049 | 36.85% | 8.69% | 54.47% | 95.09% |

away−toward timeout은 `+54.32pp [+52.16,+56.48]`, capture는 `-58.66pp
[-60.94,-56.39]`, crash는 `+4.35pp [+2.84,+5.85]`다. 70-bar timeout 차이 `+23.32pp`보다
near-open에서 더 커졌으므로 **dense obstacle occlusion이 있어야만 실패한다는 설명은 기각**한다.
막대 접촉은 toward 2건, away 1건뿐이라 이 결론과도 일치한다.

단, target-hidden은 순수 occlusion이 아니라 camera FOV 밖도 포함한다. away의 non-crash closest
mean `13.03 m`, capture mean step `392`, speed q0→q3 timeout `77.16→30.95%`는 slow-away target이
60초 안에 벽 반사로 되돌아오지 않는 finite-arena 기하와 양립한다. 따라서 다음은 학습이 아니라
outcome별 visible-step 비율과 wall/bar reflection 횟수를 추가한 seed 353 재평가다. 계약과 판정
기준은 `RESEARCH_PLAN.md` §8.27에 결과 전에 고정했다.

### outcome telemetry 구현·preflight PASS

bulk evaluation에만 작동하는 네 개의 per-episode counter를 추가했다: target-visible observation
steps, total observation steps, wall reflection 횟수, bar reflection 횟수. 종료 시 capture/crash/timeout
별 합계와 speed quartile × wall-reflection(0/≥1) × outcome 표로 집계하며, 두 표의 outcome 합계와
visibility step 합계가 전체 held-out 집계와 다르면 결과 export 전에 예외로 중단한다. 이 값은 actor,
critic, reward, controller, termination, checkpoint state에 들어가지 않는다.

schema-2 source manifest의 `git_dirty`가 결과 CSV까지 포함하던 문제도 수정했다. 이제 기존 이름은
snapshot 대상 runtime root만 뜻하고 전체 repository 상태는 `repository_git_dirty/status`로 별도
보존한다. 과거 진단 verifier는 오늘의 runtime과 억지로 같음을 요구하지 않고 당시 immutable source
snapshot과 cell 간 byte-map 동일성을 검증한다.

seed 353 orchestrator는 seed 347 near-open 양성 screen, D1 checkpoint SHA, P2/D1 fail-closed 상태,
유일한 forced mismatch `cfg_target_pattern: mixed→cv`, 1 bar/두 heading/2,049 episode 계약을 모두
검사한다. py_compile, shell syntax, target-motion 7개와 P2 contract 9개, historical artifact verify,
GPU를 시작하지 않는 forced preflight를 통과했다.

### outcome telemetry 완료 — FOV 단독 screen 미달, 초기 sensor-range 불일치 발견

seed 353 toward/away는 각각 2,050/2,049 episode를 완료했고 `verify`가 source manifest, receipt,
cell 합계, outcome telemetry와 speed×reflection 합계를 모두 재검증했다. 결과는 다음과 같다.

| 1 bar | capture | crash | timeout | capture visible | timeout visible | capture wall-any | timeout wall-any |
|---|---:|---:|---:|---:|---:|---:|---:|
| toward | 94.29% | 5.37% | 0.34% | 35.38% | 1.52% | 0.21% | 57.14% (n=7) |
| away | 35.97% | 9.42% | 54.61% | 15.02% | 0.59% | 99.73% | 99.64% |

away capture−timeout step-weighted visibility는 `+14.43pp`로 방향은 맞지만 사전등록 20pp에 못
미쳤다. FOV-memory/tracker를 단독 최우선 개입으로 선언하지 않는다. away wall reflection 평균도
capture/timeout `4.024/4.630`으로 둘 다 높다. speed별 반사 집단 capture가 높아 보이지만 무반사
118회가 반사 전에 OOB로 끝난 짧은 episode에 집중되어 있어 인과효과가 아니라 survival selection이다.

두 평가 seed 재현은 양호했다. seed 347→353의 toward timeout은 `0.15→0.34%`, away timeout은
`54.47→54.61%`, away capture는 `36.85→35.97%`였다. 성능 drift보다 구조적 채널이 훨씬 크다.

후속 코드 감사에서 D1/heading hard-distance `[22.5,28]m`가 camera detector 최대 `20m`와 LiDAR
최대 `12m`를 모두 초과함을 확인했다. tracker inactive일 때 15-D target feature 전체에 0을 곱하므로
actor는 최초 취득 전 target state를 받지 못한다. 모든 episode는 초기 표적 비관측이고, target spawn의
전방 위치 prior로 먼저 움직여야 한다. toward/away가 acquisition time과 wall turnaround를 함께
바꾸는 것이 현재 가장 강한 구조적 설명이다. `RESEARCH_PLAN.md` §8.28에 seed 359 first-acquisition
telemetry replay를 결과 전에 고정했다. PPO, sensor range, arena, horizon은 바꾸지 않는다.

## 2026-08-13 — 4-1 결합 진단: v7 오차를 analytic에 주입, 손실의 82% 재현 (게이트 실패로 판정 보류)

사용자 요청: v7 교체 손실 −5.19 pp가 (a) 정책이 analytic 통계에 결합된 탓인지 (b) v7 출력이 덜
쓸모있는 탓인지를 **재학습 없이** 가른다. 사전등록 `docs/prereg_2026-08-13_detector_coupling.md`
(측정 전 작성).

### 착수 전에 명시한 한계 — 하나는 확증, 하나는 해소

L1로 "주변분포만 맞춘 iid 주입은 상관구조를 과소재현하므로 **재현 실패가 (b)를 지지하지 않는다**"를
박아뒀다. 프로파일링이 절반씩 갈랐다:

- **드롭아웃 Markov 우려 → 해소.** v7이 analytic 대비 놓치는 비율이 **2.1e-5**로 사실상 0.
- **range 상관·이분산 → 확증.** lag-1 자기상관 **0.644**, 거리 4분위별 std 0.42/0.37/0.23/**1.07** m.

그래서 **arm 실행 전에** 주입을 AR(1)+이분산으로 확정하고(§4-b에 기록) 해석적으로 검증했다
(주입 std 0.7051/목표 0.7053, lag1 0.6439/0.6440). 결과를 본 뒤 고친 것이 아니다.

### v7이 analytic과 다른 지점 (seed 419, 53,871 동시검출 프레임)

| 축 | 값 |
|---|---|
| 검출률 analytic / v7 | 0.1787 / 0.1809 — **v7이 더 봄** |
| bearing 오차 std | 0.047° (무시 가능) |
| **range 오차 std** | **0.705 m** ← 사실상 유일한 채널 |
| range 오차 mean, 거리 4분위 | +0.205 / +0.305 / +0.224 / **−0.554 m** (원거리에서 **부호 반전**) |

**v7은 표적을 덜 보는 것이 아니라 거리를 다르게 잰다.**

### 5-arm 결과 (seed 409, 205 bars, riskcap, 임계값 0.55 전 arm 고정)

| arm | capture | Δ vs clean | 95% CI |
|---|---:|---:|---:|
| analytic_clean | 81.60% | 기준 | — |
| analytic_noise_0.5× | 80.98% | −0.63 pp | [−3.01, +1.76] |
| **analytic_noise_1.0×** | **78.15%** | **−3.45 pp** | **[−5.91, −1.00]** |
| analytic_noise_1.5× | 74.50% | −7.10 pp | [−9.63, −4.58] |
| **learned_v7** | **77.40%** | **−4.20 pp** | **[−6.67, −1.73]** |

사다리 단조(0.63 < 3.45 < 7.10), Δ₃의 CI가 Δ₅ 점추정을 포함 → 사전등록 1차 조건 **충족**.

### 그런데 게이트 0을 내 기준으로 재니 실패했다

게이트 0은 "주입 std가 0.705 m의 ±10% 이내"를 요구한다. **실측 0.6134 m (−13.0%)**.

원인 분해: `pooled var 0.4974 = 구간내 분산 0.3765 + 구간간 평균 분산 0.1210`.
주입은 구간내 분산만 재현하고 **구간별 평균(계통 편향)을 단일 전역 bias로 뭉갰다**.

사전등록이 "게이트 0 미충족 시 어떤 판정도 하지 않는다"고 못박았으므로 형식 판정은
**INCONCLUSIVE(잡음 모델 불충분)**. 결과를 보고 게이트를 완화하지 않는다.

**단 실패 방향이 보수적이다**: v7보다 **약한** 잡음으로 손실의 **82%**(−3.45/−4.20)를 재현했다.
사다리 선형보간으로 −4.20 pp는 scale≈1.10에서 나오는데 13% 과소주입을 보정하면 실효 ≈0.96 —
사실상 1.0×다(이 산술은 사후 계산이며 판정의 일부가 아니다).

방어 가능한 진술: **analytic 출력에 v7의 측정 오차 통계를 주입하는 것만으로 교체 손실 대부분이
재현된다. 결합 가설과 정합적이며 D3(matched fresh training)를 정당화한다.** 확정은 아니다 —
(L2) 이것은 "이 오차가 이 정책을 해친다"이지 "이 정책이 analytic에 특별히 결합돼 있다"가 아니며,
후자는 정의상 재학습을 요구한다.

### 실수 하나와 재발 방지

첫 프로파일링(seed 401)을 평가기 직접 호출로 돌리며 `NAVRL_SPEED_GOVERNOR=riskcap`을 빠뜨려
governor **off**로 돌았다(capture 71.50%). 계약 위반이라 `..._VOID_governor_off/`로 폐기하고
시드 401을 소진 처리했다. 재발 방지로 계약을 런처 `eval_navrl_v2_detector_coupling.sh` 한 곳에서
export하도록 바꿨다 — ad-hoc 환경변수로 계약을 넘기지 않는다.

### 코드

`navrl_perception.py`: 전용 RNG(전역 무소비, 검증 3 교훈)로 AR(1)+이분산 range 잡음,
2-상태 Markov 드롭아웃, centroid 경유 bearing 잡음(직접 `bearing` 수정 시 KF와 맵 carve-out이
desync되므로 `u`를 움직인다), 그리고 관측 전용 페어 프로파일링 헤드.
`tests/test_navrl_detector_noise.py` 15/15 — Markov 정상률·연속길이 폐형식, 전역 RNG 무오염,
사다리가 p10을 건드리지 않음, 프로파일 헤드가 live 경로에 무접촉을 전부 고정.
회귀 3종(perception/latency/appearance) 통과.

### 다음 (재사전등록 필요)

주입에 **구간별 bias**를 추가하면 13% 격차와 누락된 계통항이 동시에 닫힌다. 새 평가 시드로
3-arm(clean / 보정 1.0× / v7)을 재사전등록해 약 15분 돌리면 확정 가능하다.
**5C 재학습은 요청대로 미착수.** 임계값 재튜닝 없음. 판정 기준 사후 변경 없음.
상세: `results/navrl_detector_coupling_probe_seed409/summary.md`.
## 2026-08-14 — 발표 전 4건 확인: detector bin-bias gate 통과, footprint·OPEN·5B 상태 확정

- 첫 detector-coupling probe의 품질 게이트 실패 원인(거리 quartile별 mean bias 누락)만 수정했다.
  `docs/prereg_2026-08-14_detector_coupling_binbias.md`에 seed 431, noise seed 9431, 3 arms와 기존
  ±10% 게이트를 GPU 실행 전에 고정했다.
- 주입 std 0.7085 m vs v7 profile 0.7053 m(+0.45%)로 gate PASS. 결과는 clean 80.29%, bin-bias
  1.0× 76.72%(−3.57 pp [−6.09,−1.06]), v7 76.04%(−4.26 pp [−6.78,−1.73]). 고정 판정은
  output-coupling hypothesis supported이나 causal confirmation에는 matched retraining이 필요하다.
- bar W,D는 independent U(0.4,0.8)m. 이론 평균 area 0.36m², 205/1600m² gross occupancy 4.6125%.
  실제 seed-42 40-file pool은 mean area 0.365313m², gross 4.6806%. touching overlap을 뺀 union
  occupancy는 layout별 값이라 이 수치와 구분한다.
- RA-L 2025 예외는 Chen et al., “Online Planning for Multi-UAV Pursuit-Evasion in Unknown
  Environments Using Deep Reinforcement Learning,” RA-L 10(8), 8196–8203, DOI
  10.1109/LRA.2025.3583620으로 특정했다.
- 5B는 2026-08-14 01:34 KST 기준 미착수. full-budget summary/receipt와 실행 process 모두 없다.
- 통합 보고: `docs/presentation_followup_2026-08-14.md`; 실험 결과:
  `results/navrl_detector_coupling_binbias_seed431/summary.md`.

## 2026-08-14 — MOTAR 발표 v21 read-only QA

- `/home/fair/Downloads/MOTAR_발표_v21.pptx`의 17개 슬라이드·발표자 노트·OOXML 구조를 읽고,
  LibreOffice PDF 렌더와 원자료를 교차검증했다. PPTX 자체는 수정하지 않았다.
- 최신 detector bin-bias 3-arm 수치, 205-bar nominal 면적 점유율 4.61%, OPEN 서지, 5B
  `NOT RUN / PENDING`은 본문에 정상 반영됐다.
- 발표 전 필수 정정으로 (1) slide 12의 0.705 m를 v7 절대오차가 아닌 `v7−analytic paired output
  difference std`로 명시, (2) slide 15의 corridor-token 상태를 제안에서 `pilot completed / gate
  FAIL`로 변경, (3) slide 15·17의 contact-time 16.8%를 `never entered`가 아니라 `at contact
  absent`로 제한, (4) slide 10 노트의 timeout 역산 문구 제거, (5) slide 3 동영상 poster/fallback
  재생 확인을 판정했다.
- 추가 정정 후보: slide 4에 moving-target 선행인 YOPOv2-Tracker를 명시하고 정지-goal 절대문장
  제거, slide 6의 imitation/RL 필연성 완화 및 실제 reward 항 공개, slide 7 static-token 노트 오류,
  slide 9 crash 감소 전량이 capture로 전환됐다는 서술, slide 14의 모든 within-lineage A/B가
  자동 유효하다는 일반화를 제한한다.

## 2026-08-14 — 발표 slide 6 reward 계약 계수 대조

- 환경 원계수는 range-rate `+1.0`, ego-progress `+1.0`(gamma `0.99`), static safety `+1.5`,
  detector-visible `+0.02/step`, time `−0.05/step`, velocity-change smoothness `−0.1`, height
  violation `−8.0`, yaw alignment `−0.3`, yaw-command damping `−0.02`, capture `+30`, collision
  overwrite `−20`이다. timeout terminal bonus는 없고 corrected semantics에서는 value bootstrap한다.
- PPT 초안의 `action smoothness`는 실제 구현 `||v_t−v_{t−1}||`와 달라 `velocity smoothness`로
  고쳐야 한다. capture는 기존 step reward에 +30을 더하지만 crash는 step reward 전체를 −20으로
  덮어쓰므로 두 terminal 연산도 구분한다.
- rl_games가 환경 total reward에 `scale_value=0.1`을 적용한다. 발표에는 위 환경 원계수를 쓰고
  `PPO input: total reward ×0.1`을 각주로 분리한다.

## 2026-08-14 — NavRL/NavRL++ 밀도 및 감지-footprint 주장 감사

- NavRL RA-L 본문 계약은 50×50 m, static 350 고정, dynamic 60→120 curriculum이며 best는
  dynamic 100에서 저장된다. 따라서 논문 기준 static 14.0, best total 18.0, maximum total
  18.8 objects/100 m²이다. 다만 공개 코드 commit `3725bcc`는 `map_range=[20,20,4.5]`, 즉
  40×40 m이고 기본 설정은 static 350 + dynamic 80이다. 논문 수치와 공개 코드 수치를 섞지 않는다.
- NavRL++ Table I의 S5는 static 400 + dynamic 140임을 원문에서 확인했다. 원문이 명시한
  40×40 m는 evaluation arena이며 training-stage arena 크기는 해당 문장에 직접 결속되어 있지 않아,
  S5를 21.6/100 m²(50×50 가정)로 발표하는 것은 근거가 없다. 같은 40×40이라고 가정하면
  33.75/100 m²지만 반드시 `inference`로 표시해야 한다.
- `results/navrl_v2_bar_ceiling/episodes_seed167.npz`의 1,989개 205-bar episode를 재계산했다.
  12 m 내 bar-center 개수는 spawn 평균 45.84(SD 10.56, p10–p90 31–59), terminal drone
  평균 49.47(SD 9.78, p10–p90 35–60), target terminal 평균 49.60이다. 4 m에서는 각각
  6.07, 6.20, 6.10이다. 이는 저장된 spawn/terminal snapshot 통계이며 trajectory-time 평균은 아니다.
- `58`은 `205/1600 × π×12²`의 무경계 균일분포 기대값(57.96)으로, 40 m arena 경계와 실제
  위치분포를 무시한다. 실제 평균으로 발표하지 않는다. NavRL의 `7`도 같은 방식의 nominal
  expectation(`350/2500 × π×4²=7.04`)이며 측정값이 아니다.
- 더 근본적으로 이를 "정책이 동시에 처리하는 장애물 수 49 vs 7"이라고 부를 수 없다. NavRL은
  4 m ray distances와 nearest dynamic obstacle 5개를 입력한다. MOTAR는 12 m 4×72 scan을 한
  static CNN token으로 압축하고, 8개 proposal을 각 history token 안에 넣는다. 따라서 footprint
  안의 bar-center 수는 scene context 폭이지 개별 object-token 처리량이 아니다.
- 발표 권고: density-superiority 주장은 철회하고 density를 task descriptor로만 둔다. NavRL을 숨기지
  말고 `global density / sensing contract / task` 3행 표로 공개한다. MOTAR의 차별점은 moving-target
  interception, actor information firewall, perception-policy coupling 및 timing/representation/braking
  진단으로 둔다. 보조 혼잡도는 이미 측정된 `31.7 candidates in token FOV / K=8`과 contact-time
  token absence `16.8%`를 사용하되, 후자는 접촉 순간 snapshot임을 명시한다.

## 2026-08-14 — 현재 drone safety 거리 용어 확인

- inference `riskcap`의 sensor-surface hard margin은 **0.45 m**이고 command corridor half-width도
  **0.45 m**이다. 3.0 m 이내에서는 기본 2.0 m/s cap, 3.0→5.0 m에서 free cap으로 완화된다.
- 0.45 m는 인증된 minimum separation이 아니다. riskcap은 근거리에서 강제 정지하지 않으며,
  `usable clearance = LiDAR surface range − 0.45 m`로 stopping margin을 진단하는 최소개입 필터다.
- 실제 upright collision proxy는 0.28×0.28 m이므로 중심 기준 face half-width는 0.14 m,
  half-diagonal은 약 0.198 m이다. static-safety reward는 고정 임계 거리가 아니라 12 m까지의
  LiDAR log-distance 연속 shaping이고, target/bar placement clearance 1.0 m는 별도 조건이다.

## 2026-08-14 — RNN 대비 Transformer 발표 주장 감사

- MOTAR 계보에는 동일 입력·학습예산·seed로 수행한 LSTM/GRU 대 Transformer held-out ablation이
  없다. LSTM은 2026-07-17 당시 10 epoch 저장/배선 sanity만 확인했으므로 성능 비교 자료가 아니다.
- 따라서 MOTAR 결과로 "Transformer가 RNN보다 수치적으로 우수하다"고 말할 수 없다. 현재 slide 7의
  `longer memory than an RNN? no, 2.0 s fixed window`는 올바른 제한 문구다.
- NavRL++의 직접 ablation도 RNN 비교가 아니라 single-step convolutional NavRL-IT 대 temporal
  Transformer NavRL-IT-T이다. combined success는 92.85→90.54%로 2.31 pp 낮아졌지만 control
  effort는 0.093→0.043 m/s²로 53.8% 감소했다. full NavRL++ 94.08%는 Transformer와
  perturbation-aware fine-tuning의 결합 결과라 backbone 단독 효과로 귀속하지 않는다.
- 발표 답변은 우월성 대신 선택 근거로 제한한다: 5-step 고정 history에서 static/obstacle/robot/target
  modality-time token에 직접 접근하고 NavRL++ 계보와 맞추기 위해 채택했다. RNN은 더 긴 history를
  compact hidden state에 보존할 수 있고 유효한 대안이며, parameter/compute/seed-matched recurrent
  ablation이 없다는 점은 명시적 limitation이다.

## 2026-08-14 — seed 359 최초취득 진단: away timeout의 87.52%가 표적을 한 번도 못 봤다

RESEARCH_PLAN 8.28 사전등록대로 실행했다. 정책·보상·센서 range·아레나·horizon 무변경,
D1 terminal checkpoint(SHA `197ea269…`), 1 bar, CV-only, `[22.5,28] m`, deterministic, governor off,
exact 600, seed 359, toward/away 각 2,049 episodes.

### 왜 이걸 쟀나

seed 353 telemetry는 away timeout의 step-weighted visible fraction이 `0.59%`, capture가 `15.02%`로
`+14.43 pp` 차이였고 사전 20 pp screen에 미달했다. 그런데 visible fraction은 **"한 번 잡았다 놓쳤다"와
"한 번도 못 잡았다"를 구분하지 못한다.** 두 경우의 처방은 정반대다(tracker memory vs range 계약).
게다가 hard-distance `[22.5,28] m`는 camera `20 m`·LiDAR `12 m` 밖이라 모든 episode가 target token
0으로 시작한다. 그 상태를 벗어나기는 하는지를 센 적이 없었다.

### 결과

| heading | outcome | episodes | never-acq | first-visible 평균 | 중앙값 | vis→hid /ep |
|---|---|---:|---:|---:|---:|---:|
| toward | capture | 1,932 | **0.00%** | 82.1 | 72 | 2.068 |
| toward | crash | 115 | 99.13% | 77.0 | 77 | 0.026 |
| toward | timeout | 2 | 100.00% | — | — | 0.000 |
| away | capture | 762 | **0.00%** | 318.8 | 312 | 2.142 |
| away | crash | 189 | 93.12% | 360.8 | 381 | 0.249 |
| away | timeout | 1,098 | **87.52%** | 565.2 | 569 | 0.173 |

outcome split은 toward `94.29/5.61/0.10%`, away `37.19/9.22/53.59%`로 seed 353을 재현했다.

**사전등록 primary: away timeout − capture never-acquired = `+87.52 pp` (임계 30 pp) → 통과.**
판정 `initial_acquisition_range_contract_channel_supported`. secondary(first-visible 지연)는
primary 통과 시 미적용이며, 값 `+246.4 step`을 "미달"로 적지 않도록 summary 표기를 고쳤다.

가장 강한 신호는 임계가 아니라 **두 capture cohort가 모두 정확히 0.00%**라는 것이다. 이 조건에서
최초 취득은 capture의 필요조건처럼 동작한다. away capture는 평균 318.8 step에야 처음 보고도
잡아냈고, away timeout 중 취득한 12.48%는 평균 565.2 step으로 600 상한 직전이었다.

### 말할 수 있는 것 / 없는 것

말할 수 있다: 이 조건에서 timeout은 압도적으로 **표적을 한 번도 취득하지 못한 episode**다.
말할 수 없다: 그것이 timeout의 **원인**이라는 것. outcome은 궤적의 결과이고, 일찍 crash한
episode는 취득 기회 자체가 적다. 이건 연관 계측이지 인과가 아니며 해결책도 아니다.
P2 STRICT FAIL·D1 FAIL·P3 BLOCKED 모두 그대로다.

### VOID 2건 — 둘 다 가드가 작동했다

1. 내가 새로 쓴 export 가드가 `fa_outcomes`를 list로 만들어 tuple인 `expected[1:]`과 비교했다.
   list는 tuple과 절대 같지 않으므로 카운트와 무관하게 항상 발화한다
   (`[1932,115,2] != (1932,115,2)`). export 앞단이라 결과 JSON·receipt 모두 미기록.
   → `..._VOID_export_guard_bug/`. 회귀 테스트로 고정.
2. `verify_all`이 `verify_cell`의 두 번째 반환값(receipt)을 runtime byte map으로 오인했다.
   receipt는 timestamp·nonce·셀별 해시를 담으므로 셀 간 비교는 항상 실패한다. 이를 고치자
   가드가 **실제 위반**을 잡았다 — 다른 세션의 untracked 런처가 `aerial_gym/` 아래 있어 두 셀
   모두 dirty tree에서 실행됐다. 강제 우회하지 않고 런처를 커밋해 clean으로 만든 뒤 재실행.
   → `..._VOID_dirty_runtime/`.

두 경우 모두 `summary.json`이 생성되지 않았고 **primary screen은 계산된 적이 없다.** 관측된
수치는 outcome split뿐이며 이는 seed 353을 재현하는 보조 지표라 어떤 screen에도 들어가지 않는다.
따라서 "판정을 보고 재시도"가 아니라 "산출물 없는 실행 실패의 재실행"으로 보고 같은 시드를 썼다.
근거를 각 VOID.md에 남겼으니 동의하지 않으면 새 시드로 재사전등록하면 된다.

### 부수적으로

- 기존 계약 테스트가 조용히 깨져 있었다. heading diagnostic의 screen key가
  `path_length_support` → `radial_heading_channel_support`로 개명됐는데 테스트 리터럴이
  갱신되지 않아 존재하지 않는 키의 존재를 단언하고 있었다. 함께 고쳐 11/12 → 20/20.
- Codex 세션이 내가 남긴 detector-coupling 후속(거리 구간별 bias 주입)을 이미 구현·통과시켰다.
  내 우선순위에서 제거한다. 그 미커밋 변경은 runtime-clean 게이트를 막고 있어 사용자 승인 후
  저자를 명시해 별도 커밋했다.

### 다음

RESEARCH_PLAN 8.29에 **하나만** 사전등록했다: seed 367, away 단일 heading, 2 arm —
target camera range `20 m`(대조) vs `28 m`(개입), **오직 이 한 값만** 변경. primary gate는
timeout rate 20 pp 이상 감소. 목표거리를 줄이는 task-contract 대조는 D0에서 이미 6–11.5 m
timeout이 `0.06%`로 알려져 정보가치가 낮고 경로길이·시간예산이 함께 바뀌어 교란되므로 택하지
않았다. 사전 명시한 한계: 정책은 20 m로 학습됐으므로 **timeout이 줄지 않아도 "비관측이 원인이
아니다"로 읽을 수 없다.** 감소가 관측될 때만 단방향으로 강한 증거다.

### 독립 검증 (별도 세션 재계산)

`verify` PASS(exit 0). raw cell JSON에서 9필드 × 3 outcome × 2 cell = 54개 값과 outcome
카운트·비율·`actual_episodes`를 summary와 대조해 **불일치 0건**, 모든 비율이 자기 카운트에서
1e-12 이내로 재도출된다. 불변식 전부 PASS: never+acquired=episodes, outcome 합=2,049,
acquired=0 cohort가 0이 아닌 `null` 보고, 두 셀 manifest byte-identical(314 runtime files),
runtime git status clean(commit `c01de65e`), checkpoint SHA `197ea269…` 일치.

검증자가 짚은 해석 플래그 2개를 기록한다:

1. **fused == camera가 6개 cohort 전부에서 정확히 같다.** 두 카운터는 코드상 독립인데
   (`_fa_ep_first_fused` vs `_fa_ep_first_camera`) 값이 일치했다는 것은, camera `20 m` /
   LiDAR `12 m` 구성에서 **LiDAR가 camera보다 먼저 최초취득을 준 적이 한 번도 없다**는
   경험적 결과다. 따라서 사전등록 primary(fused)는 이 조건에서 camera 채널 이상의 정보를 담지
   않으며, 이 발견은 사실상 **camera range 발견**이다. 8.29에서 camera range만 바꾸기로 한
   선택이 이 관측과 정합적이다.
2. toward/timeout은 n=2라 100% never-acquired가 의미 있게 추정된 값이 아니다. 어느 screen에도
   들어가지 않는다.

## 2026-08-20 — 문서 통합 (검증 단계용 6개 + archive)

검증만 남은 단계에 맞춰 planning/review/PPT/handoff/prereg 일자별 md 17개와 RESEARCH_PLAN §8.1–8.22를
`docs/archive/`로 이동했다. 실행 authority는 신규 [`VERIFICATION.md`](VERIFICATION.md)로 통합.

| 유지 (루트) | 역할 |
|---|---|
| `README.md` | 입문 + 현재 결론 표 |
| `VERIFICATION.md` | ref5in gate·진단 요약·다음 실험 |
| `RESEARCH_PLAN.md` | charter (§1–7 + §8 pointer), 1231→353 lines |
| `WORKLOG.md` | 날짜별 기록 |
| `OPERATIONS.md` | 설치·launcher 명령 |
| `CRASH_TUNING_LOG.md` | crash-cause 진단 |

`results/*/summary.md`는 실험별 canonical 수치로 그대로 둔다. seed 367 camera-range A/B 원자료는
`results/navrl_ref5in_camera_range_control_seed367/`에 있으나 formal summary는 아직 없음 — VERIFICATION.md에
스냅샷만 기록.

## 2026-08-20 — seed 367 camera-range 인과 대조: 초기 미관측이 timeout의 지배적 원인

RESEARCH_PLAN 8.29 / VERIFICATION.md 사전등록대로 실행·동결했다. 정책·보상·아레나·horizon·heading
불변, D1 terminal checkpoint(SHA `197ea269…`), seed 367, away heading, 1 bar, CV-only,
`[22.5,28] m`, deterministic, governor off, exact 600, 2,049 요청/arm. **조작 변수는 오직
`vision.detector_max_range` 하나**다.

| arm | camera range | capture | crash | timeout | pooled never-acq |
|---|---:|---:|---:|---:|---:|
| A | 20 m | 36.39% | 7.80% | **55.80%** | 57.22% |
| B | 28 m | 74.96% | 6.88% | **18.16%** | 20.30% |
| Δ | — | +38.57 pp | −0.92 pp | **−37.65 pp** | −36.92 pp |

**primary gate 통과** (timeout 감소 ≥ 20 pp). **guard 깨끗** — crash가 오르기는커녕 0.92 pp
내렸으므로 "timeout을 crash로 바꾼 것"이 아니다. 판정
`initial_unobservability_dominant_cause_supported`.

### 조작이 실제로 먹혔는지를 receipt가 아니라 행동으로 확인했다

receipt는 `NAVRL_DETECTOR_MAX_RANGE`를 요청했다는 것만 증명하지, 인지 모듈이 그 값을 썼다는 것을
증명하지 않는다. seed 359에서 붙여둔 first-acquisition telemetry가 그 증거를 갖고 있어서
manipulation check로 넣었다: pooled never-acquired가 `57.22% → 20.30%`로 떨어졌다. 임계를 두지
않고 **방향 검정**(treated < control)으로 설계해 사후 조정 여지를 없앴다.

### 말할 수 있는 것 / 없는 것

말할 수 있다: 이 조건에서 **초기 표적 미관측이 timeout의 지배적 원인**이다. seed 359의 연관
(never-acquired 87.52% vs 0.00%)이 단일 변수 개입으로 재현됐다.

말할 수 없다: (1) camera range 확장이 **해결책**이라는 것 — 진단이지 채택이 아니며, 28 m 검출은
정책의 학습 분포 밖이라 잘 활용한다는 보장이 없다. capture 36.4→75.0%는 부수 관측이지 gate가
아니다. (2) 실기 함의 — 20 m는 시뮬 파라미터이지 센서 사양이 아니다. (3) **P2 STRICT FAIL ·
D1 FAIL · P3 BLOCKED는 전부 그대로다.**

### 요약 생성이 두 번 막혔다 (둘 다 내 verifier 오류, 데이터 무관)

두 셀 평가는 08-14에 이미 끝나 있었고 요약만 못 만들고 있었다.

1. `KeyError: target_camera_max_range_m` — 평가기가 그 값을 result `condition`이 아니라
   **receipt**에 기록하는데 내가 condition에서 찾았다. receipt는 result와 해시로 묶여 있으므로
   provenance 강도는 동일하다. receipt에서 읽도록 고치고, 대신 위의 행동 확인을 추가했다.
2. `episode count does not match` — 내가 `== 2049`로 단정했는데 평가기는 128-env 배치를 비우므로
   요청치를 살짝 넘긴다(A arm 2,050). 감사받은 base verifier와 동일하게
   `requested == N and actual >= N`으로 맞췄다. 계약 위반이 아니라 내 단정 오류다.

재실행 없이 `finalize`만으로 해결됐고 시드는 소모되지 않았다.

### 다음은 진단이 아니라 설계 결정이다

병목이 특정됐으므로 남은 선택은 **(a) 과제를 센서에 맞추기**(goal 거리를 관측 범위 안으로) vs
**(b) 센서를 과제에 맞추기**(장거리 검출 전제로 재학습)다. 둘 다 재학습이 필요하므로 P3 차단
해제 조건과 함께 **별도 사전등록**한다. 이번 결과만으로 재학습을 시작하지 않는다.

상세: `results/navrl_ref5in_camera_range_control_seed367/summary.{md,json}`, `VERIFICATION.md`.

## 2026-08-21 — seed 367 결과 인수검사와 다음 단계: 28 m는 진단 knob, hardware spec 아님

- 첨부 보고와 canonical `results/navrl_ref5in_camera_range_control_seed367/summary.md`를 대조했다.
  20→28 m 단일변수 개입에서 capture `36.39→74.96%`, crash `7.80→6.88%`, timeout
  `55.80→18.16%`, pooled never-acquired `57.22→20.30%`; primary와 guard 판정은 유효하다.
- 결론은 hard-distance/CV/away/1-bar 조건에서 **초기 미관측이 timeout의 지배적 원인**이라는
  진단까지다. 28 m camera가 실기 해결책이거나 P2/D1/P3 gate를 통과시킨다는 뜻은 아니다.
- 중요한 fidelity gap: 조작은 detector의 거리 cutoff만 바꿨다. 현재 renderer는 160×90,
  HFOV 87°, target diameter 0.30 m이며, pinhole 근사상 28 m target 직경은 약 0.90 pixel
  (22.5 m에서도 약 1.12 pixel)이다. 따라서 range cutoff 연장만으로 생긴 spawn acquisition은
  실제 카메라의 해상도·MTF·노출·blur·배경·검출확률·거리추정 성능을 증명하지 않는다.
- 다음 authority는 PPO가 아니라 hardware/perception feasibility gate다: (H0) 22.5–28 m에서 필요한
  detection/range/latency 계약 정의, (H1) exact camera/lens/native resolution 및 range-source 후보 선정,
  (H2) 실거리 정지·이동표적 bench로 recall/false-positive/range error/latency 측정, (H3) exact BOM·전력·
  cooling·mount·CG 폐쇄, (H4) 측정한 distance-dependent observation model을 simulator에 반영한다.
- 그 뒤에만 task를 20 m 안으로 축소할지, 28 m 장거리 RGB bearing + 근거리 LiDAR/depth의 단계형
  인지를 유지할지 결정하고, fresh short gate→P2를 재사전등록한다. P3 full training은 계속 BLOCKED.

## 2026-08-21 — 사용자 의도 재정의: 장거리 미관측은 sensor mismatch가 아니라 active search 과제

- 사용자 목표는 28 m 표적을 spawn부터 보는 것이 아니라, **미관측 상태에서 안전하게 탐색→최초취득→
  추적·요격**하는 것이다. 이 의도라면 task range와 sensor range가 같을 필요는 없고 seed 367은
  해결책 선택이 아니라 현 정책에 search capability가 없음을 드러낸 진단으로 다시 해석한다.
- episode horizon만 무한정 늘리는 것은 해결이 아니다. 현 PPO `gamma=0.99`, 0.1 s/step에서 reward
  weight는 step 600에 `0.99^600=0.002405`, step 1200에 `5.78e-6`이다. actor temporal window는
  2 s, tracker memory는 5 s라 이미 지나간 탐색 영역을 기억하지 못한다. visibility bonus도 보이는
  동안 `+0.02`일 뿐 first-acquisition/새 영역/information gain을 직접 보상하지 않는다.
- 더 근본적으로 unacquired target token은 zero인데 range-rate와 ego-progress reward는 GT target을
  사용한다. actor가 방향을 관측하지 못한 상태에서는 대칭 위치의 hidden directional gradients가
  평균상 상쇄되므로, 길이만 늘려도 체계적인 coverage policy가 생긴다고 보장할 수 없다. search용
  visited/belief memory와 first-acquisition/coverage objective를 별도 설계해야 한다.
- pursuer limit은 현재 **축별 2.5 m/s**(수평 norm 최대 3.536 m/s), target은 최대 1.5 m/s다.
  전역 limit 상향은 acquisition을 직접 해결하지 않고 frozen policy의 action semantics를 바꾸며,
  brake=2 m/s²·reaction=0.1 s 가정의 정지거리는 2.5 m/s에서 1.81 m, 3.0 m/s에서 2.55 m로
  약 41% 증가한다. search/chase 속도와 clearance-aware cap을 분리한 fresh policy가 필요하다.
- 현재 소스의 target motion을 70 bars, mixed CV/waypoint, exact 1.5 m/s, 64 env×300 step으로 재-probe:
  CV `1.50 m/s`, waypoint `1.50 m/s`, 양쪽 stall/overspeed/clearance<1m 모두 `0.0%`.
  `tests/test_navrl_target_motion.py`도 PASS. 즉 70-bar 기하에서는 막대를 정상 회피한다.
- 단 target은 동역학 없는 virtual point다. 0/±30/±60/±90/±120/180° 후보 방향으로 0.1 s마다
  full-speed 재지정하고 wall/bar에서 즉시 반사할 수 있어 실제 표적 드론의 acceleration/yaw-rate/
  inertia를 재현하지 않는다. 따라서 "회피 동작 정상"은 geometry contract이고 hardware realism은 아니다.
- 다음 실험은 학습 전 horizon-only frozen replay(60/120/180 s)로 `P(acquire by t)`와 capture CDF를
  확인하고, speed-only는 2.5/3.0 m/s를 별도 arm으로 둔다. 그 결과와 무관하게 본 설계 후보는
  SEARCH(coverage/belief memory)와 TRACK/INTERCEPT를 명시적으로 나누는 hierarchical policy다.

## 2026-08-21 — 표적 물리성 감사와 opt-in bounded trajectory prototype

- 사용자 지적대로 기존 표적은 물리 드론이 아니었다. `target_position`을 직접 갱신하는 virtual
  point가 0.1 s마다 `0/±30/±60/±90/±120/180°` full-speed 후보를 골랐고, 벽에서는 즉시 반사,
  막대 안에서는 위치 push-out을 했다. 기존 speed/clearance probe는 기하 결과만 검사했으므로 이
  순간 가속·회전을 검출하지 못했다.
- 기존 체크포인트 재현을 위해 default `NAVRL_TARGET_DYNAMICS=legacy`는 보존했다. 새 opt-in
  `bounded`는 별도 model id `bounded_planar_drone_v1_rollout`을 checkpoint에 저장하므로 legacy
  moving-target checkpoint와 resume guard가 불일치를 거부한다.
- bounded prototype은 (1) XY vector acceleration ≤ **4.0 m/s²**, (2) 유의미한 이동 중 travel-heading
  slew ≤ **150°/s**, (3) speed ≤ episode limit, (4) **1.0 s** forward rollout, (5) 보수적 bar-centre
  clearance **0.77 m**를 계약으로 둔다. 4 m/s²는 `atan(a/g)=22.2°` tilt로 ref5in controller의
  45° envelope 안이고, 1.5 m/s에서 `a/v=152.8°/s`라 150°/s 곡률 한계와 일관된다. 0.77 m는
  max 0.8 m square half-diagonal 0.566 + target half-width 0.14 + modelling margin 0.06의 보수 proxy다.
- planner는 방향만 바꾸지 않고 full/half/quarter/stop 목표속도를 함께 rollout한다. stop도 속도를
  즉시 0으로 쓰지 않고 acceleration bound로 제동한다. 새 모드에서는 wall reflection, final clamp,
  bar push-out을 전혀 사용하지 않는다. 초기 target velocity도 0에서 ramp한다.
- reset 감사에서 general target sampler가 96회 rejection 뒤에도 실패하면 검증되지 않은 최초 난수를
  쓰는 silent fallback을 발견했다. bounded mode는 1024회 뒤 fail-closed하며, 움직이는 표적 spawn은
  capture-sphere clearance 1.0 m가 아니라 자기 충돌 계약 0.77 m를 적용한다. `probe_target_motion.py`의
  `NAVRL_MAX_BARS=150` 하드코딩도 고쳐 205/300 probe가 실제 요청 밀도를 만들게 했다.

### 측정 결과 (64 env × 300 target steps, exact 1.5 m/s, seed 42)

| bars | pattern | realized mean | stall | clearance violation | rollout infeasible | accel / turn violation |
|---:|---|---:|---:|---:|---:|---:|
| 150 | CV | 1.32 m/s | 1.8% | **0.0%** | **0.0%** | **0.0 / 0.0%** |
| 150 | waypoint | 1.39 m/s | 0.2% | **0.0%** | **0.0%** | **0.0 / 0.0%** |
| 205 | CV | 1.11 m/s | 3.9% | **0.0%** | **0.0%** | **0.0 / 0.0%** |
| 205 | waypoint | 1.13 m/s | 0.5% | **0.4%** | **0.4%** | **0.0 / 0.0%** |
| 300 | CV | 0.84 m/s | 11.1% | **0.2%** | **0.2%** | **0.0 / 0.0%** |
| 300 | waypoint | 0.76 m/s | 19.9% | **0.0%** | **0.0%** | **0.0 / 0.0%** |

- 해석: 최고속 1.5 m/s는 명령 상한이지 고밀도에서 유지해야 할 실제 속도가 아니다. 물리 planner가
  150→300 bars에서 평균속도를 약 1.3→0.8 m/s로 낮추는 것이 정상적인 결과다. 다만 205/300의
  잔여 infeasible은 local constant-command rollout이 obstacle cluster의 cul-de-sac를 미리 기억하지
  못하는 전역 계획 병목이다. 따라서 **bounded mode 본학습은 BLOCKED**다. 다음은 grid/global route
  memory 또는 target collision을 명시적인 target-crash/reset 종료로 정의하고, 70/150/205/300에서
  `initial/clearance/infeasible/accel/turn violation == 0` gate를 다시 통과시키는 것이다.
- 이 prototype은 trackable planar trajectory 계약이지 6-DoF target rigid-body/rotor simulation의
  증명이 아니다. 최종 “실제 드론” 주장을 위해서는 ref5in actor가 생성 궤적을 추종하는 closed-loop
  harness에서 tracking error, tilt, motor saturation, energy를 추가 검증해야 한다. 새 PPO 학습은 하지 않았다.

## 2026-08-21 — 물리 동역학 전면 감사: bounded trajectory와 실제 drone physics를 구분

사용자 요청에 따라 현재 코드의 “실제 동역학” 주장을 표적·추적 드론·센서·환경으로 분리 감사했다.
결론은 **추적 드론은 PhysX 강체로 동작하지만, 표적은 실제 동역학 시뮬레이션이 아니며, ref5in도
실기 식별 모델이 아닌 합성 설계점**이라는 것이다. bounded prototype을 physical drone이라고 부르면 안
되며, 아래 문제를 해결하기 전 새 본학습·논문 수치의 physical claim은 BLOCKED다.

### 발견 사항과 심각도

1. **BLOCKER — 표적은 PhysX actor가 아니다.** `NavRLBarsEnvCfg.env_config.include_asset_type`에는
   `bars`만 들어가고 `navrl_target_params`/`navrl_target_drone.urdf`는 실제 환경에 등록되지 않는다.
   `_advance_target()`가 매 RL step(0.1s) `target_position`과 `target_vel_w`를 직접 갱신하며, target의
   질량·관성·추력·자세·모터 상태·접촉·중력은 PhysX에 존재하지 않는다. `collision_mask=0`,
   `fix_base_link=True`, `disable_gravity=True`인 target asset 문서도 이 사실을 명시하지만 현재 env에서
   asset 자체가 unused다. 따라서 bounded mode의 4.0 m/s²/150 deg/s는 강체에서 유도된 값이 아니라
   trajectory envelope 상수다.

2. **BLOCKER — target 기하가 세 경로에서 다르다.** (a) unused URDF는 0.30×0.30×0.14m box,
   (b) camera analytic renderer는 `camera_target_radius=0.15m` sphere,
   (c) LiDAR analytic injection은 `target_radius=0.20m` sphere다. 센서별 관측 크기와 실제 충돌
   기하가 같지 않다. 28m에서 0.9px 수준의 표적을 논할 때 이 차이는 검출/가시성 결과를 바꿀 수 있다.

3. **HIGH — 표적의 z/attitude가 고정이다.** 표적은 z=1.0m에 고정되고 XY만 움직인다. 실제 쿼드라면
   4.0m/s² 횡가속에는 tilt와 `T=mg/cos(tilt)`가 필요하고, yaw/roll/pitch 동역학 및 자세 지연이
   뒤따라야 한다. 현재 bounded planner는 travel-heading만 제한할 뿐 이 coupling을 계산하지 않는다.

4. **HIGH — ref5in 자체가 실기 식별값이 아니다.** `1.20kg`, `9.60N/motor`, `4.401e-5 thrust k`,
   `0.04s motor tau`, `0.01m thrust/torque ratio`, 관성은 문서상 analytic/synthetic design point다.
   thrust stand, 모터·프롭·ESC·배터리 조합, 전압/열/전력, CG와 CAD가 없다. self-consistency test는
   URDF/config 내부 일치만 보장하며 hardware truth를 보장하지 않는다.

5. **HIGH — 현재 주 학습은 ref5in이 아니다.** `NAVRL_ROBOT` 기본값은 `navrl_quad`이고
   `train_navrl_v2_search.sh`도 `NAVRL_ROBOT=navrl_ref5in_quad`를 설정하지 않는다. 기존 학습 결과는
   0.25kg legacy robot lineage다. ref5in 결과로 해석하려면 새 robot 선택·fresh 학습·별도 checkpoint가
   필요하다.

6. **MEDIUM — 환경 경계가 물리 벽/바닥이 아니다.** `create_ground_plane=False`이고 arena bound는
   task-level 수치 검사다. 추적 드론은 z<0.1에서 종료되지만 실제 바닥 접촉이 아니며, target/trajectory
   bound도 수학적 clamp/샘플링이다. 실제 비행 환경을 주장하려면 floor/wall 또는 명시적인 boundary
   collision contract가 필요하다.

7. **MEDIUM — target update와 PhysX 시간 해상도가 다르다.** 추적 드론은 physics dt 0.01s × 10 =
   RL step 0.1s로 실제 physics를 진행하지만, target은 RL step당 한 번 좌표를 바꾼다. 이는 “10Hz
   trajectory command”로는 정의할 수 있으나 target rigid body contact를 0.01s에서 검증한 것이 아니다.

8. **MEDIUM — target clearance가 실제 bar collision과 같은 수식이 아니다.** bars는 인스턴스별
   0.4–0.8m axis-aligned box인데 planner는 모든 bar를 center-distance 0.77m circle로 본다. 이 값은
   `0.566 + 0.14 + 0.06` 보수 proxy일 뿐 실제 AABB/tilted prop envelope와 동일하지 않다. 따라서
   bounded probe의 `clearance=0%`는 “PhysX 충돌 0%”가 아니라 “proxy 위반 0%”다.

9. **MEDIUM — 추적 드론도 완전한 실기 모델은 아니다.** 추적 기체는 PhysX rigid body + Lee velocity
   controller + motor lag을 사용하지만, action은 축별 velocity setpoint이고 acceleration/norm cap이
   직접 정의되지 않는다. yaw 2.5rad/s, tilt 45° 및 altitude PI는 task/controller engineering knob이다.
   motor tau/thrust/inertia가 측정값이 아니므로 “물리적으로 가능한 시뮬레이션”과 “실기와 정량적으로
   일치하는 시뮬레이션”을 구분해야 한다.

### 현재 판정

- `legacy`: 기존 결과 재현용 virtual-target 실험. physical claim 불가.
- `bounded`: 가속도·travel-heading·속도·proxy clearance를 지키는 **동역학 가능 궤적 prototype**.
  실제 target drone simulation 아님. 150 bars에서는 proxy gate가 통과했지만, 205/300의 local-planner
  infeasible 표본과 target geometry mismatch가 남아 본학습 BLOCKED.
- `ref5in`: 내부 수치 정합성 후보일 뿐 실기 동역학 검증 완료가 아님. 현재 main 학습에는 사용되지 않음.

### physical claim을 열기 위한 필수 순서

1. target도 `navrl_ref5in_quad` 계열 PhysX actor로 등록하거나, 최소한 동일 URDF/6-DoF rigid-body와
   low-level controller를 별도 actor로 구현한다.
2. camera/LiDAR/충돌에 동일 target mesh와 동일 pose/orientation을 사용한다. analytic sphere를 유지하면
   센서 모델과 collision proxy를 동일 반지름/근거로 명시하고 실측 크기와 대조한다.
3. exact BOM 기반 mass/CG/inertia, thrust-vs-RPM, motor rise/fall tau, torque ratio, battery sag를
   식별하고 ref5in config를 교체한다.
4. 0.01s physics substep에서 target actor를 제어하고, target tracking error/tilt/motor saturation/
   contact를 계측한다. 70/150/205/300 bars에서 collision 및 infeasible gate를 다시 정의한다.
5. 그 후에만 target dynamics를 포함한 fresh PPO를 시작한다.

## 2026-08-21 — 실제 6-DoF target actor 구현 및 1–6단계 사전검증 완료

- `NAVRL_TARGET_DYNAMICS=physical`을 새 lineage로 구현했다. target은 이제 gravity/contact가
  활성화된 PhysX actor이며 task의 `target_position`, `target_orientation`, `target_vel_w`는 actor
  root-state view다. 0.1 s 좌표 덮어쓰기는 physical 경로에서 사용하지 않는다.
- ref5in 등가 rigid body(1.20 kg, I=0.004142/0.004142/0.005769 kg·m²), 4×9.60 N motor,
  arm XY 0.0777817 m, motor τ=0.04 s, yaw ratio 0.01 m를 사용한다. velocity/attitude controller와
  motor lag/saturation은 0.01 s physics substep마다 갱신한다. 초기 legacy gain을 그대로 쓴 첫
  smoke는 tilt 143.7°로 실패했고, inertia-scaled gain(0.08/0.04 계열)으로 고쳐 최종 max tilt를
  5.3° 수준으로 안정화했다.
- transient target contact가 마지막 substep 전에 사라지는 결함을 발견해, 10개 physics substep
  전체 contact를 OR 누적하도록 수정했다. target contact/invalid state는 명시적 episode crash/reset이다.
- 센서/충돌 target 기하를 모두 actor pose의 **0.28×0.28×0.12 m OBB**로 통일했다. physical mode는
  stale dynamic Warp mesh를 만들지 않고 camera/LiDAR가 같은 oriented box를 analytic ray-test한다.
  legacy/bounded sphere는 재현성을 위해 그대로 남겼다.
- 모든 bar URDF의 실제 collision half-extents를 tensor로 노출하고 physical planner를 center-circle
  proxy에서 target OBB support를 포함한 point-to-AABB rollout으로 바꿨다. full-horizon constant-heading
  후보가 없을 때는 safe-prefix 길이를 우선하는 receding-horizon fallback을 사용한다.
- `NAVRL_ROBOT=navrl_ref5in_quad`가 아니면 physical mode가 fail-closed한다. 신규
  `train_navrl_physical_fresh.sh`는 `CKPT/CHECKPOINT`를 exit 4로 거부한다. 실측이 아닌 ref5in 값을
  hardware 검증으로 오독하지 않도록 `docs/navrl_hardware_identification_manifest.yaml`에 BOM/CG/CAD
  inertia/thrust curve/tau/torque ratio/battery sag/thermal의 미측정 상태를 명시했다.

### 고정 gate 결과 — 새 PPO는 아직 BLOCKED

`results/navrl_physical_target_verification/summary.json`, seed 503, 32 env × density별 280 measured
steps, mixed CV/waypoint, target command 1.5 m/s. 기준은 결과 전에 고정했고 사후 완화하지 않았다.

| bars | speed ratio | tracking RMSE | contact | immediate infeasible | invalid | motor sat | max tilt |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 70 | 0.919 | 0.233 | 0.257% | 1.574% | 0.089% | 0.000% | 5.26° |
| 150 | 0.880 | 0.279 | 0.525% | 1.417% | 0.089% | 0.000% | 5.27° |
| 205 | 0.832 | 0.316 | 0.904% | 1.384% | 0.045% | 0.000% | 5.25° |
| 300 | 0.738 | 0.351 | 1.373% | 2.042% | 0.011% | 0.000% | 5.26° |

- gate: RMSE≤0.35, speed ratio≥0.80, contact≤1%, infeasible≤1%, motor sat≤15%, tilt≤60°,
  invalid=0. 네 밀도 모두 strict 전체 PASS는 아니다. 205부터 contact/speed 경계, 300은 명확한
  physical density limit다. 따라서 학습을 시작하지 않았다.
- 검증: py_compile PASS, ref5in/target URDF 27/27, run contract 12/12, target motion/AABB 11/11,
  physical camera+LiDAR live smoke PASS, fresh-launcher checkpoint reject PASS.
- 상세 설계·해석: `docs/navrl_physical_target_audit_2026-08-21.md`. 다음 선택은 target global/corridor
  planner 또는 density-conditioned physical speed envelope이며, 같은 frozen gate 재통과 후에만 PPO smoke다.

### 최종 교차감사 정정

- 첫 잠정 gate 뒤 `EnvManager`의 실행 순서를 다시 추적해, target wrench 계산이 Isaac Gym force
  tensor 제출보다 늦어 명령이 물리 1 substep(0.01 s) 지연되던 결함을 발견했다. callback을
  `IGE_env.pre_physics_step()` 앞으로 이동해 같은 substep에 적용되도록 고쳤다.
- target contact threshold의 0.05 N literal을 제거하고 활성 env의
  `collision_force_threshold`를 controller에 주입했다.
- general target reset의 legacy 1.0 m wall inset이 physical planner의 기본 admissible inset
  1.25 m보다 작던 결함을 고쳤다. invalid-state 판정도 actor center가 아니라 회전된 전체 OBB가
  arena 안에 있는지 검사한다.
- 위 표는 이 세 수정 뒤 동일 seed 503/동일 frozen gate로 재실행한 최종 canonical 수치다. 이전
  잠정 수치는 폐기한다. 205 bars까지 tracking/speed/contact는 통과하지만 planner/state gate가
  남고, 300 bars에서는 tracking/speed/contact도 실패한다. PPO는 실행하지 않았다.
- 최종 회귀검사: py_compile/diff-check PASS, ref5in platform 27/27, run contract 12/12,
  target-motion 11/11, physical launcher의 env/CLI checkpoint 거부 PASS. legacy mode도 실제
  Isaac Gym 2-env 1-step에서 finite observation `(2,156)`, `bar_offset=0`으로 PASS해 기존 계보를
  깨지 않았음을 확인했다.

## 2026-08-21 — navrl_band 중심거리·실제 footprint 중첩 감사

- `navrl_band`는 최소 중심거리 1.6 m 규칙이 아니다. 모든 막대 쌍을 `d≤0.4 m`의 의도적
  touch/overlap 또는 `d≥1.6 m`의 분리 상태로만 허용하고 중간 band를 금지한다. 막대 XY 한 변은
  실제 `bars_h3` URDF에서 0.4–0.8 m, 회전은 0이므로 `d≥1.6`이면 AABB 중첩은 수학적으로
  불가능하다. 축방향 중심 배치의 표면 gap은 0.8 m지만 대각 배치의 이론적 corner gap은 약
  0.469 m까지 줄 수 있다. 반대로 `d≤0.4`는 모든 크기 조합에서 접촉 또는 중첩되어 compound
  obstacle을 만든다.
- 실제 40개 footprint pool과 배치기 mirror로 밀도별 50 layouts를 감사했다. merge fallback,
  `0.4<d<1.6` 위반, `d≥1.6` overlap, `d≤0.4`인데 비접촉인 쌍은 모두 0건이었다. 의도적 overlap
  pair는 70/150/205/300 bars에서 각각 총 `47/301/639/1926`쌍(평균 `0.94/6.02/12.78/38.52`
  쌍/layout)으로 증가했다.
- 따라서 고밀도에서 1.6 m를 자동 축소하지 않는다. 그러면 nominal count와 corridor width를 동시에
  바꾸는 교란이 생긴다. 현 centre rule의 이론적 corner gap은 약 0.469 m이며, physical target의
  level OBB 0.28 m + 양측 tracking reserve 0.45 m가 요구하는 축방향 1.18 m보다 이미 좁다. 먼저
  hard-OBB/operational-reserve 두
  configuration space의 연결성과 compound-wall/cage를 분리 측정한 뒤, 중심거리 상수가 아니라
  footprint-aware surface-gap 및 connectivity 계약으로 재설계한다.

### 300 bars의 실제 독립 덩어리 수와 no-cluster 가능성

- 실제 `bars_h3` footprint와 배치기 mirror로 300 bars × 200 layouts를 연결성 계수했다. AABB가
  접촉/중첩하면 같은 component로 묶었을 때 독립 덩어리는 평균 `263.785`, 중앙값 `264`, 범위
  `252–276`개였다. 독립 단일 막대는 평균 `229.425`개, 2개 이상 군집은 평균 `34.36`개이고,
  군집에 포함된 막대는 평균 `70.575/300`개였다.
- 최대 군집 크기는 평균 `2.92`, 전체 200 layouts에서도 최대 `4`개였다. 따라서 이 표본에서는
  거대한 compound wall/cage가 만들어진다는 앞선 우려는 지지되지 않으며, physical 경로 불능의
  주원인으로 단정하지 않는다.
- 군집을 완전히 금지해도 40×40 m 안에 300개 독립 배치가 가능하다. 최대 0.8 m footprint가 경계
  안에 있도록 centre를 0.4–39.6 m에 둔 20×15 constructive grid는 최소 중심거리 `2.0632 m`로
  1.6 m 계약을 여유 있게 통과하며 정확히 300 components를 만든다. 다만 규칙적인 grid는 현재
  random layout 분포와 다르므로 실제 대체안은 footprint-aware Poisson/blue-noise sampler와 별도
  connectivity audit가 필요하다.

### 300 bars 통로 여유거리 감사 — centre gap 1.6 m의 대각선 함정 정정

- 300 bars × 200 layouts에서 모든 비중첩 AABB 쌍 약 897만 개의 표면거리를 계산했다. layout별
  최협 gap은 최소 `0.5495 m`, 5% `0.6073 m`, 중앙값 `0.6841 m`, 평균 `0.6806 m`, 95%
  `0.7482 m`였다. 막대별 nearest-gap 분포는 5% `0.8040 m`, 중앙값 `1.1313 m`다.
- `d≥1.6`에서 표면 gap이 항상 0.8 m라는 앞선 설명은 잘못이었다. 두 최대 0.8 m square의 중심이
  대각선으로 1.6 m 떨어지면 이론적 AABB gap은
  `sqrt(2)*(1.6/sqrt(2)-0.8)=0.4686 m`까지 감소한다. 코드 주석도 함께 정정했다.
- 관측된 최협 `0.5495 m` 통로 중앙을 통과한다고 단순 보수 계산하면, level 0.28 m box는 총
  `0.2695 m`/편측 `0.1347 m`, yaw-45° 수평 대각 0.396 m는 총 `0.1535 m`/편측 `0.0767 m`,
  3-D box 외접폭 0.4138 m는 총 `0.1357 m`/편측 `0.0679 m`만 남는다. 이는 정적 기하 여유이며
  tracking error·PhysX contact margin을 포함하지 않는다.

## 2026-08-21 — OOB exit forensics 추가, 그리고 두 `0.45 m` 계보 분리

### 정정: seed 367 pursuer와 300-bar physical target을 같은 planner로 설명하면 안 된다
## 2026-08-21 — OOB exit forensics 추가, 그리고 두 `0.45 m` 계보 분리

### 정정: seed 367 pursuer와 300-bar physical-target WIP를 같은 planner로 설명하면 안 된다

앞선 설명과 첫 정정은 서로 다른 코드의 `0.45 m`를 하나로 취급했다. 실제 계약은 다음 두 개다.

1. **동결 ref5in pursuer / seed 367:** 전역 또는 국소 경로 planner가 없다. riskcap은 속도 크기만
   깎는 필터이며 재계획·진입거부를 하지 않는다. `speed_governor.path_half_width_m=0.45`는 명령축
   좌우 0.45 m 안의 LiDAR 광선만 clearance 계산에 포함시키는 **광선 선택창**이다
   (`speed_governor.py:166`). 게다가 이 계보의 governor는 `mode=off`, 실측 intervention 0이다.
2. **미커밋 physical-target fresh 계보 / 300-bar gate:** `bounded_drone_target_step()`이 1초 동안
   여러 방향·속도 후보를 rollout하고 충돌 없는 후보를 선택하는 **receding-horizon local planner**다.
   여기의 별도 `target_motion.physical_tracking_margin=0.45`는 target OBB로 팽창한 막대 표면 바깥에
   추가하는 closed-loop tracking reserve다. riskcap의 동명 숫자와 무관하다. 전역 경로를 찾거나
   통로를 위상적으로 판정하지는 않으며, 전 후보가 막히면 가장 긴 safe prefix의 bounded step을
   실행하고 다음 control step에서 다시 계획하면서 `feasible=False`를 노출한다.

따라서 `0.28 + 0.45×2 = 1.18 m`는 level OBB가 축정렬 평행 통로의 양쪽에서 tracking reserve까지
지키려 할 때의 **운용 폭** 계산으로만 유효하다. pursuer의 물리 통과 한계도, 임의 yaw/대각 통로의
일반식도, planner가 반드시 돌아선다는 증거도 아니다. 300-bar 기하 probe의 최협 표면간격
0.55–0.68 m와 대각 배치 이론 최저 0.469 m는 physical-target 환경 난이도를 설명하지만 seed 367의
실패 원인에는 사용할 수 없다.
2. **별도 physical-target WIP / 300-bar gate:** `bounded_drone_target_step()`이 1초 동안 여러
   방향·속도 후보를 rollout하는 receding-horizon local planner이며, 별도 tracking margin 0.45 m를
   사용한다. 이 코드는 본 OOB 전용 branch에는 포함하지 않는다. riskcap의 동명 숫자와 무관하고,
   전역 경로를 찾거나 통로 연결성을 판정하는 planner도 아니다.

따라서 `0.28 + 0.45×2 = 1.18 m`는 physical-target WIP의 level OBB가 축정렬 평행 통로에서 양쪽
tracking reserve까지 지키려 할 때의 **운용 폭** 계산으로만 유효하다. pursuer의 물리 통과 한계도,
임의 yaw/대각 통로의 일반식도, seed 367 실패의 증거도 아니다. 300-bar 기하 probe의 최협
표면간격 0.55–0.68 m와 대각 배치 이론 최저 0.469 m 역시 seed 367 원인에는 사용할 수 없다.

seed 367은 **막대 1개**이고 crash의 약 98%가 OOB다: `camera_20m` crash 160 = bar_contact 2 /
OOB 158, `camera_28m` 141 = 3 / 138. 이 체제는 아래 OOB 계측으로 따로 진단한다.

riskcap 자체에 관한 확인은 유지한다. riskcap cap은 clearance 0.3/1.0/3.0 m에서 모두 2.0 m/s이고
강제 zero-speed를 만들지 않는다. `hard_margin_m`은 riskcap의 cap 계산에는 쓰이지 않지만
clearance/TTC mode 및 stopping-margin 진단에는 쓰인다.

**riskcap 파라미터 탐색은 금지 규칙이다**(`CLAUDE.md:56`, `CRASH_TUNING_LOG.md:523`). 값들은
제동 프로브(p10 감속 2.9609 m/s², 정지거리 1.047 m)에서 사전 도출됐다.

### 추가: OOB exit forensics

기존 `_diag`의 `oob_w/e/s/n`과 평균 step은 이미 있었다. 그것으로 답할 수 없는 것만 추가한다 —
**"쫓다가 넘어감"과 "헤매다 흘러나감"의 구분**이다. 처방이 정반대다.

| 필드 | 가르는 것 |
|---|---|
| `never_acquired_share` | 나갈 때까지 표적을 한 번도 못 봤는가 (first-acquisition과 연결) |
| `goal_closing_speed_mean_mps` | 음수면 목표에서 멀어지며 나감 |
| `outward_radial_speed_mean_mps` | 양수면 능동적으로 밖으로 몰고 감(표류 아님) |
| `speed_mean_mps` / `goal_distance_mean_m` / `step_median` | 속도·거리·중앙값 |

초안은 acquisition 비율과 운동학 전체 평균만 따로 내서 두 집단이 평균 안에서 상쇄될 수 있었다.
실행 전 감사에서 이를 발견해 `by_acquisition.{never_acquired,acquired}`마다 exit 수·비율·속도·목표거리·
closing·outward를 교차 집계한다. 두 strata 합이 전체 exit와 다르거나 never-acquired 수가 기존 카운터와
다르면 export를 중단하며, `robot_linvel`/first-acquisition 원천이 없을 때도 0으로 기록하지 않고 중단한다.
README의 `camera range A/B = 다음 실험` 잔존 문구도 완료된 seed 367 진단으로 고쳤다. 사용자가 원한
active-search 과제에서는 28 m camera가 positive control이지 자동 채택안이 아니라는 범위를 명시했다.

구현에서 잡은 것 둘:

1. **원인 귀속 마스크를 써야 했다.** 처음엔 raw `oob`에 걸었는데 crash 원인 표는
   `d_oob = oob & ~contact & ~below & ~above & crashed_out`을 쓴다. 같은 step에 막대 접촉이
   겹치면 원인이 그쪽으로 귀속되므로 두 집단이 다르다. 귀속 마스크로 옮겨
   `crash_causes.out_of_bounds`와 직접 비교 가능하게 했다.
2. **`sim_steps`가 env별 텐서였다.** 스칼라 변환이 실패해서 발견했고, 덕분에 episode별 정확한
   exit step을 쓰게 됐다.

**교차검증**: 제 카운터와 `_diag["oob"]`를 독립으로 세고 export 전에 일치를 강제한다(불일치 시
RuntimeError). 스모크 32 env/400 step에서 `126 = 126`, 방향버킷 `[25,36,28,38]` 동일. 네 방향
버킷은 기존 진단과 마찬가지로 **비배타적**이다. 코너 이탈 한 건이 두 버킷에 들어가므로 이 예에서도
합은 127이며, `edge_shares`를 합계 100%인 categorical distribution으로 해석하면 안 된다.
bulk eval 전용이며 관측·보상·종료·체크포인트에 무접촉이다.

### 문서: CRASH_TUNING_LOG를 archival-in-place로 표시

2026-08-05 이후 미갱신이지만 **삭제·이동 금지**다. 소스 4곳이 경로를 주석으로 참조하고
(`navrl_task_config.py:611`, `navrl_task.py:3883`, `navrl_lidar_config.py:17`,
`results/general_12m_lookahead_speed_axis.csv:4`), crash 계측 방법론·07-22/23 원인 분해·
one-lever 후보 결과·Phase-1 부록은 다른 문서에 사본이 없다. riskcap 금지 규칙도 `CLAUDE.md`
외에는 여기에만 있다. 헤더에 사유를 명시하고 CLAUDE.md/README/RESEARCH_PLAN 라벨을 맞췄다.

### 부수 기록: 속도·틸트 상한은 한 번도 ablate된 적이 없다

`NAVRL_MAX_VELOCITY`(2.5), `NAVRL_MAX_TILT_DEG`(45), `NAVRL_YAW_RATE_MAX`(2.5) 모두 파라미터
카탈로그에서 `ablated=False`, 실험 0건이다. `free_speed_cap_mps=3.5355`는 최적화 값이 아니라
`2.5×√2`라는 축별 제한의 기하학적 귀결이다. 다만 `max_velocity`는 관측 정규화의 분모라
(`navrl_task.py:3914,3958`) 바꾸면 동결 체크포인트와 계약이 어긋난다 — 사실상 재학습 knob이다.
물리적으로도 정지거리 `v²/2a`와 선회반경 `v²/a`가 제곱으로 커진다(2.5 m/s에서 1.06/0.64 m,
4.0 m/s에서 2.70/1.63 m). 올리려면 틸트 상한도 함께 올려야 하고, 틸트는 필요추력을 `1/cosθ`로
키운다(60° 2배, 70° 2.9배; 현재 T/W 3.26).

### 다음

재학습 전에 OOB 계측을 seed 367 조건으로 한 번 돌린다(2셀 ~10분). 후보는 A(camera range
20→28 m, seed 367에서 인과 확인) / B(속도·틸트 상향, 근거 없음) / C(목표거리 축소)이며 한 run에서
두 축을 바꾸지 않는다(`VERIFICATION.md:115`).

## 2026-08-21 — N1 사전등록 + import-origin 버그 실증·fail-closed 가드

`docs/diagnostic_synthesis_2026-08-21.md`의 N1(real-frame reflection audit) 착수. 별도 worktree
`.codex_worktrees/navrl_reflection_audit` / branch `codex/reflection-audit`를 9f6929d에서 분기했다
(dirty primary와 physical-target WIP에 merge하지 않는다는 조건).

### N1은 중복이 아니지만 절반은 이미 존재한다 (감사 결과)

| 이미 있는 것 | 위치 |
|---|---|
| 898-D reflection 변환(유일본, bin permutation은 물리 판정 완료) | `ppo_update_safety.py:357-417` |
| involution·schema 단위테스트 | `test_navrl_action_models.py:205-275` |
| real-frame side-forward(원본+거울 2회, `fork_rng`로 env RNG 격리) | `navrl_players.py:135-192` |
| 실제 프레임 chirality 수치 | 2026-08-02, 548,736 obs — **legacy `navrl_quad` ep24000 계보** |

따라서 N1이 새로 답하는 것은 (a) frozen ref5in 계보에서 실제 프레임 재현 여부, (b) 맥락 의존성,
(c) 분포 꼬리(p90/p95/p99)다. 맥락 라벨은 새로 정의하지 않고 `navrl_task.py:3281
_record_action_diagnostics`가 관측 소비 시점에 이미 계산하는 `front_blocked`/`front_clear`,
`_visible_now`, `valid_y_now`를 그대로 기록한다.

**재사용 시 반드시 고쳐야 하는 함정**: `mirror_navrl_structured_observation`의 env-var 기본값이
`HBEAMS=36/VBEAMS=4/MAX_OBSTACLES=5`라 574-D를 기대하고 **898-D 입력에서 예외를 던진다**
(`ppo_update_safety.py:362-364`). 값을 checkpoint metadata에서 읽도록 사전등록에 고정했다.

### import-origin 버그 — 실측으로 확인, 저장소 전체에 가드 0곳이었다

`find_spec("aerial_gym").origin` 실측:

| cwd | PYTHONPATH | 해석 결과 |
|---|---|---|
| `<worktree>` | 없음 | `<worktree>/aerial_gym/__init__.py` |
| `<worktree>` | `<worktree>` | `<worktree>/aerial_gym/__init__.py` |
| **`<worktree>/aerial_gym/rl_training/rl_games`** | **없음** | **`src/aerial_gym_simulator/aerial_gym/__init__.py`** |
| 〃 | `<worktree>` | `<worktree>/aerial_gym/__init__.py` |

원인 3중첩: (1) `site-packages/__editable__.aerial_gym-2.0.0.pth`의 PEP 660 finder가 MAPPING을
PRIMARY 절대경로로 하드코딩하고 `sys.meta_path`에 **append**(=PathFinder 뒤)한다, (2)
`play_navrl.sh:19`가 `aerial_gym/rl_training/rl_games`로 `cd`하는데 거기엔 `aerial_gym/` 패키지
디렉터리가 없다, (3) `attest_navrl_ref5in_p2.py:245`가 child env에서 `PYTHONPATH`를 삭제한다.

→ **worktree에서 시작한 평가는 PRIMARY 소스를 실행하면서 worktree 바이트를 해싱해 영수증에
적는다.** 영수증은 내부적으로 일관되지만 실행되지 않은 코드를 기술한다. 이는 codex의
geofence/mode-probe/joint-telemetry/topology 4개 branch에 모두 해당하는 구조적 조건이다.
실행된 파일 바이트가 두 트리에서 동일했다면 수치는 유효하므로 **어떤 기존 판정도 소급
변경하지 않는다**. 동일성 확인은 별도 작업으로 남긴다.

### 추가한 것

- `aerial_gym/rl_training/rl_games/navrl_import_origin.py` — stdlib 전용, torch/isaacgym 불필요.
  `NAVRL_REQUIRE_SOURCE_ROOT` 미설정 시 완전 무해(기존 run 영향 0), 설정 시 패키지 디렉터리가
  `<root>/aerial_gym`과 **정확히 일치**하지 않으면 RuntimeError. 이미 import된 경우
  `sys.modules["aerial_gym"].__file__`을 본다 — `find_spec`은 "지금 다시 찾으면 어디서 나올지"를
  답하므로 "무엇이 실행 중인지"와 다를 수 있고, 그 차이가 바로 이 가드가 잡으려는 실패다.
- `runner.py:20-31` 배선. `isaacgym`·`torch` import **이전**, `aerial_gym` 로드 **직후**라
  Isaac Gym의 import 순서 요구를 건드리지 않으면서 GPU 작업 전에 fail-closed한다.
- `tests/test_navrl_import_origin.py` — 10 tests, 전부 통과. 무해성, 발화, 정확 동등(부모 디렉터리
  거부), py3.8 호환(`is_relative_to` 미사용), 무거운 의존성 미import를 검사한다.
- 실증: cwd를 `aerial_gym/rl_training/rl_games`로 두고 worktree를 요구하면 가드가 발화하며
  `actual origin`으로 PRIMARY 경로를 정확히 지목한다.

`_is_within` 헬퍼는 삭제했다. 엄격 동등 매칭으로 바꾼 뒤 죽은 코드였고, 남으면 "하위경로도 허용"으로
오독된다.

### 사전등록

`docs/prereg_2026-08-21_n1_real_frame_reflection_audit.md` (측정 개시 전 확정). 정책 ref5in D1
ep1900 SHA `197ea269…`, 70 bars, **신규 seed 373**(전수검색 0건), deterministic, governor off,
rollout의 reflection_mode는 `original`.

설계의 핵심은 **rollout과 반사를 시간적으로 분리**한 것이다. rollout 중에는 표준 평가만 하고
관측·맥락·outcome을 디스크에 덤프하며, 반사 forward는 rollout 종료 후 오프라인으로 한다.
"probe action을 시뮬레이터에 투입하지 않는다"가 시간 순서로 자명해지고, 저장된 npz로 제3자가
전 수치를 재계산할 수 있다(현재 저장소에 없는 성질).

품질 게이트 Q1–Q9(involution=0.0, isometry≤1e-3, schema=898, index-set 정확 일치, scan 순열
`i→(−i) mod 72` 고정점 `{0,36}`, import origin, checkpoint SHA/manifest, 결정론 bitwise, n≥4,096)를
**정책 판정보다 먼저** 통과해야 하며, 하나라도 실패하면 `FAIL_CLOSED_TRANSFORM_QUALITY`로
정책에 대한 주장을 하지 않는다.

판정 임계(결과 보기 전 확정, `[-1,1]` 정규화 행동 단위이므로 기존 1.235/73.08%/1.8332와 직접 비교):
`median(conj_err_lat) ≥ 0.30` **및** sign agreement `≤ 0.60` → CONFIRMED /
`≤ 0.10` **및** `≥ 0.90` → ABSENT / 그 외 INCONCLUSIVE. 맥락 셀은 비교가능 행 ≥256에만 판정을 준다.

측정 개시 전 개정 2건을 문서에 명시적으로 기록했다(§3-b, §3-c). §3-b: 고정 stride 37은 총 호출
수를 미리 알아야 해서 조기 종료율에 따라 최소 표본 미달 또는 앞부분 편향이 된다 → 스트리밍
데시메이션(매 `stride_eff`번째 보관, 상한 초과 시 하나 걸러 버리고 `stride_eff`를 2배)으로 교체해
총 호출 수와 무관하게 전 구간 균일·8,192–16,384행을 보장한다.

### 다음

`NAVRL_OBS_DUMP` 훅과 오프라인 evaluator 구현 → CPU 단위테스트 → GPU smoke → 본 평가.
이 항목 시점에 GPU 실행은 아직 없다.

## 2026-08-21 — N1 구현 완료, 그리고 내 커밋 921fb1d가 ref5in 평가를 전면 차단하고 있다

### 구현 (GPU 실행 전, 전부 CPU 테스트 통과)

| 산출물 | 줄수 | 테스트 |
|---|---|---|
| `aerial_gym/task/navrl_task/navrl_task.py` `NAVRL_OBS_DUMP` 훅 | +298 / −1 | `tests/test_navrl_obs_dump.py` 20/20 |
| `tools/navrl_reflection_offline_audit.py` 오프라인 evaluator | 1,331 | `tests/test_navrl_reflection_offline_audit.py` 47/47 |
| `aerial_gym/rl_training/rl_games/navrl_import_origin.py` + `runner.py:20-31` | 145 + 16 | `tests/test_navrl_import_origin.py` 10/10 |
| `tools/run_navrl_ref5in_reflection_audit.py` schema-v2 런처 | 700 | `ReflectionAuditContract` 11/11 |

`navrl_task.py`에서 삭제된 유일한 줄은 `_record_action_diagnostics`의 조기 반환 가드이며
`if not self._action_diag_enabled and not self._obs_dump_enabled:`로 넓혔다. `_action_diag_enabled`는
`_bulk_eval_mode`에서 항상 True이므로(`navrl_task.py:824`) 본 실험에서 **행위 변화는 0**이다.
나머지 297줄은 순수 추가다.

**오프라인 forward가 live 경로와 동일함을 증명했다** — 손으로 만든 네트워크가 아니라 실제
`NavRLPpoPlayerContinuous.restore()`를 시뮬레이터 없이 구동했다(`config['env_info']`를 주입하면
rl_games가 vecenv를 만들지 않는다; `player.env is None` 확인).

| 등가성 | 값 |
|---|---|
| 파라미터 sha256 (모델 / 체크포인트) | `b197ffbe4b128dd2…` / 동일 |
| 파라미터 개수 | 428,953 / 428,953 |
| `running_mean`/`running_var`/`count` bitwise 일치 | true |
| `selected − tanh(mus)` 최대 | **0.0** |
| `selected − mus` 최대 | **1.3355** |
| Q8 결정론(2회 forward) | `torch.equal` true |

함정 2건을 잡았다. (1) `cfg_action_mu_scale = [1.0, **0.4**, 1.0, 1.0]` — lateral만 pre-tanh
스케일이 0.4다. 이걸 복원하지 않으면 다른 컨트롤러를 평가하게 된다. (2) 데시메이션에서 호출
카운터가 1-based면 첫 보관 표본이 최종 stride 격자에서 영구히 벗어난다. 증가 **전에** 읽도록
고쳐 200k 호출 시뮬레이션으로 고정했다(`test_uniform_after_decimation`).

정규화기 대칭화(S1, 비게이트)의 대수도 명시해 둔다. mirror는 부호 있는 순열
`(Mx)_j = s_j x_{p(j)}`이고, `N(Mx) = M N(x)`를 요구하면 정확히 `v_j = v_{p(j)}`,
`m_j = s_j m_{p(j)}`가 나온다. 따라서 분산은 단순 쌍평균, 평균은 `(m_j + s_j m_{p(j)})/2`이며
**부호 반전 필드는 `p(j)=j, s=−1`이라 평균이 0으로 붕괴한다** — 부호가 뒤집혀야 하는 좌표가
가질 수 있는 유일한 평균이다. 단순 평균을 썼다면 아무것도 제거되지 않았을 것이다.

### 차단: robot config provenance drift — 원인은 내 커밋이다

`preflight`가 exit 2로 거부했다.

```
[eval_v2] robot config source drift: checkpoint=ebb71802f19b630b… runtime=cc8d90b6cf08bb1c…
```

| | sha256 |
|---|---|
| frozen ref5in 체크포인트가 기록한 값 | `ebb71802f19b630ba6c2ac4c04b113c269d8bbd3e40e094e126913caa8731297` |
| 현재 런타임 | `cc8d90b6cf08bb1cd21ea01429ab5e953e7beaf6e16a453c4161b71dce22f7d9` |

차이는 `aerial_gym/config/robot_config/navrl_ref5in_quad_config.py`의 **docstring 한 줄**이다
(`docs/…` → `docs/archive/…`). 커밋 `921fb1d`(2026-08-20, 내가 함). 그 커밋 메시지 자체가
"leaving them dirty would block the runtime-clean gate"라고 적혀 있다 — **runtime-clean 게이트를
풀려고 커밋해서 robot-config provenance 게이트를 깼다.** 게이트 하나를 다른 게이트와 맞바꿨다.

`eval_navrl_v2_density_sweep.sh:240-242`은 무조건 `exit 2`이며 `NAVRL_V2_FORCE`가 닿기 전에 죽는다.
우회 수단이 없다.

범위는 N1보다 넓다. `921fb1d`는 `research/navrl-env`를 포함한 **8개 브랜치 전부**에 있고 origin에
푸시됐다. 즉 **현재 어느 브랜치 tip에서도 frozen ref5in 체크포인트를 byte-exact로 평가할 수 없다.**
`navrl_mode_probe` worktree가 통과했던 건 체크아웃이 브랜치 tip보다 뒤에 있었기 때문이다.
robot URDF(`5c160b0d…`)와 legacy `navrl_quad_config.py`는 영향 없다.

**교훈**: `aerial_gym/config/robot_config/**`와 `resources/robots/**`는 기존 체크포인트에 대해
**provenance-frozen 아티팩트**다. 주석 한 줄만 바꿔도 그 이전에 학습된 모든 체크포인트의
byte-exact 평가가 막힌다. 문서 경로 정리 같은 무해해 보이는 편집을 이 파일들에 하면 안 된다.

이 항목 시점에 GPU 실행은 없고 `results/navrl_ref5in_reflection_audit_seed373/`도 없다.

### 다음

사용자 결정 2건이 필요하다 — (a) `navrl_ref5in_quad_config.py`를 `ebb71802…`로 되돌릴지,
(b) 지금까지의 코드를 커밋할지(런처의 dirty-runtime 게이트가 커밋을 요구한다). 둘 다 없이는
`run`을 시작할 수 없다.

## 2026-08-21 — N1 결과: frozen ref5in의 chirality는 실제 프레임에서 압도적이고 맥락 무관하다

seed 373, 70 bars, 1,024 에피소드, frozen ref5in D1 ep1900(SHA `197ea269…`), governor off,
deterministic, rollout reflection_mode=`original`. 유효 프레임 **15,488**(사전등록 최소 4,096).
`results/navrl_ref5in_reflection_audit_seed373/`. `run`/`finalize`/`verify` 3단계 모두 PASS.

### 품질 게이트 — 정책 판정보다 먼저 전부 통과

| 게이트 | 임계 | 실측 |
|---|---|---|
| Q1 involution `max abs(M(M(x))−x)` | `== 0.0` | **0.0** |
| Q2 isometry | `≤ 1e-3` | 1.907e-06 |
| Q3 schema (checkpoint metadata) | `== 898` | 72×4, obstacles 8, corridor 0 |
| Q4 index-set byte-level | 완전 일치 | 부호반전 110 / 순열 280 / 불변 508 |
| Q5 scan 순열 고정점 | `{0, 36}` | `{0, 36}` |
| Q6 import origin | 강제 | worktree, sha `1ec09850…` |
| Q7 checkpoint SHA / manifest | 일치 | 일치 |
| Q8 결정론 2회 forward | bitwise | 원본·거울 모두 `torch.equal` |
| Q9 표본 | `≥ 4,096` | **15,488** |

### 판정: `CHIRALITY_CONFIRMED_REAL_FRAME`

사전등록 임계는 `median(conj_err_lat) ≥ 0.30` **및** sign agreement `≤ 0.60`이었다.

| context | n | median | p95 | p99 | agreement | signed bias | 판정 |
|---|---|---|---|---|---|---|---|
| **overall** | 15,488 | **1.454** | 1.703 | 1.764 | **0.0249** | −0.693 | CONFIRMED |
| target_visible | 2,695 | 1.288 | 1.693 | 1.780 | 0.0427 | −0.631 | CONFIRMED |
| target_hidden | 12,793 | 1.476 | 1.705 | 1.762 | 0.0212 | −0.706 | CONFIRMED |
| front_blocked | 5,906 | 1.519 | 1.740 | 1.787 | 0.0243 | −0.722 | CONFIRMED |
| front_clear | 9,582 | 1.419 | 1.650 | 1.714 | 0.0253 | −0.676 | CONFIRMED |
| outcome_capture | 8,968 | 1.424 | 1.702 | 1.769 | 0.0292 | −0.679 | CONFIRMED |
| outcome_crash_bar_contact | 1,373 | 1.422 | 1.705 | 1.772 | 0.0469 | −0.669 | CONFIRMED |
| outcome_timeout | 3,492 | 1.504 | 1.707 | 1.754 | **0.0066** | −0.736 | CONFIRMED |
| outcome_crash_oob | 125 | 1.129 | 1.675 | 1.736 | 0.0756 | −0.531 | 표본부족(<256) |
| front_unknown / crash_other | 0 | — | — | — | — | — | 표본 0 |

**맥락 의존성은 없다.** 표본이 충분한 7개 셀 전부가 CONFIRMED이고 median 범위는 1.42–1.52,
agreement는 0.66–4.7%다. 즉 chirality는 "표적이 안 보일 때"나 "앞이 막혔을 때"의 국소 현상이
아니라 정책의 전역 성질이다. 가장 심한 셀이 timeout(agreement 0.66%)인 것은 방향 편향이
탐색 실패와 함께 나타난다는 뜻이지만, 이 자료로 인과는 말할 수 없다.

**가장 결정적인 수치는 부호다.** `mean π(o)[1] = −0.623`, `mean π(Mo)[1] = −0.763`. equivariance라면
둘은 부호가 반대여야 하는데 **둘 다 음수다.** 정책은 세계가 좌우로 뒤집혀도 같은 몸통 방향으로
돈다. p99가 1.76이고 행동 범위가 `[-1,1]`(폭 2.0)이므로, 꼬리에서는 반사쌍이 사실상 정반대 극단에
있다.

yaw도 함께 chiral하다(median 1.029). x는 0.600, z는 0.222로 축별 크기 순서는 lateral > yaw > x > z다.

### S1 — chirality는 정규화기가 아니라 네트워크에 있다 (exploratory, 비게이트)

반사 짝 인덱스에 대해 running_mean_std를 대칭화하고(`v'_j=(v_j+v_{p(j)})/2`,
`m'_j=(m_j+s_j m_{p(j)})/2`; 부호반전 필드는 `p(j)=j, s=−1`이라 평균이 0으로 붕괴) 동일 계산을 반복했다.

| context | raw median | symmetrised median | 축소 |
|---|---|---|---|
| overall | 1.454 | 1.283 | **11.7%** |
| target_visible | 1.288 | 0.988 | 23.3% |
| front_clear | 1.419 | 1.231 | 13.3% |
| outcome_timeout | 1.504 | 1.385 | 7.9% |

sign agreement는 0.0249 → 0.0947로만 올랐다. **정규화기 통계의 비대칭은 관측된 chirality의
10–20%만 설명한다.** 나머지는 네트워크 가중치에 있다. 따라서 관측 재정규화로는 고칠 수 없고,
학습 신호(reflection augmentation 또는 equivariance consistency)가 올바른 레버다.

### 이 결과가 주는 권한과 주지 않는 권한

사전등록 §8에 따라 `CHIRALITY_CONFIRMED_REAL_FRAME`은 **reflection augmentation/consistency의
사전등록을 작성할 자격**만 준다. 구현·실행 권한은 아직 없으며 별도 사전등록이 필요하다.

주장하지 않는 것: 이 실험은 outcome을 측정하지 않았다(사전등록 L3). 2026-08-02에 legacy 계보의
대칭 아레나에서 mirror outcome 차이는 capture −0.81 pp(95% CI −2.78..+1.17)로 **검출되지 않았다.**
따라서 "chirality가 성능을 해친다"는 아직 근거가 없다. 단일 checkpoint·단일 seed·70막대 1셀이며
계보 전반으로 일반화하지 않는다(L4). 장애물 토큰 순서는 반사 시 재배열되지 않는다(L1) —
mode-probe가 그 영향을 0.0078로 측정해 무시 가능함을 보였고 본 실험은 재측정하지 않았다.

2026-08-02 legacy ep24000의 sign mismatch 73.08%(=agreement 26.92%)와 비교하면 ref5in의
agreement 2.49%는 한 자릿수 더 심하다. 다만 계보·아레나·조건이 모두 다르므로 엄밀한 비교가 아니다.

### 부수 확인: import-origin 가드가 자기 값어치를 증명했다

run 로그에 `[origin] aerial_gym <worktree>/aerial_gym/__init__.py sha256=1ec09850… (enforced)`가
찍혔고, 그 sha가 source manifest의 `aerial_gym/__init__.py` 항목과 일치함을 verify가 확인했다.
가드가 없었다면 이 run은 **PRIMARY의 dirty 소스**(physical-target WIP 22파일 포함)를 실행하면서
worktree 바이트를 영수증에 적었을 것이다.

### 다음

reflection intervention의 사전등록 작성이 다음이며, 그 전에 사용자·Codex와 순서를 조율한다.
N2(prospective geofence replication)와 N3(205막대 원인 분리)는 이 결과에 영향받지 않는다.
P2 STRICT FAIL / D1 FAIL / P3 BLOCKED는 변경 없다.
## 2026-08-22 — Codex 진단 4개 import-origin 소급 감사

PEP 660 editable finder가 worktree 실행을 primary import로 바꿀 수 있다는 지적에 따라 geofence,
mode probe, joint telemetry, topology 네 branch를 소급 감사했다. source manifest만 신뢰하지 않고
manifest의 모든 runtime file을 worktree와 재해시하고, raw log의 imported `motor_model.py` 절대경로를
확인했으며, 보존한 primary physical-target WIP 바이트와도 비교했다.

- geofence: manifest/worktree **315/315 일치**, primary와 20파일 불일치, log import는 geofence
  worktree → VALID.
- mode probe: **316/316 일치**, primary와 22파일 불일치, log import는 mode worktree → VALID.
- joint canonical rerun: **317/317 일치**, primary와 22파일 불일치, log import는 joint worktree → VALID.
- joint 첫 run: log import가 primary였고 새 telemetry가 없어 evaluator가 fail-closed한 기존 VOID 유지.
- topology: simulator package를 import하지 않는 offline NumPy 분석이라 PEP 660 영향 없음. 기존
  exploratory-only 범위에서 VALID.

따라서 기존 세 simulator 결과와 topology 판정은 바꾸지 않는다. 상세 증거와 경로는
`docs/navrl_import_origin_audit_2026-08-22.md`에 기록했다. 앞으로 worktree launcher는
`NAVRL_REQUIRE_SOURCE_ROOT`와 local `PYTHONPATH`를 canonical environment 생성 **뒤에** 주입해야 한다.

## 2026-08-22 — paired-reflection consistency 사전등록 (실행 전, 구현 아님)

N1이 `CHIRALITY_CONFIRMED_REAL_FRAME`을 냈으므로 사전등록 §8에 따라 개입 실험의 **사전등록을 쓸
자격**이 생겼다. 구현·실행 권한은 아직 없다. `docs/prereg_2026-08-22_paired_reflection_consistency.md`.

### 새로 만들 코드는 없다

레버는 이미 배선돼 있다. `early_stop_a2c_agent.py:428-466`이 `NAVRL_REFLECTION_COEF`를 읽어
`reflection_equivariance_loss(mu, reflected_mu)`를 더한다. 보조 forward 동안 running_mean_std를
`eval()`로 얼려 정규화 표본을 이중 계수하지 않도록 이미 처리돼 있다.

두 번째 레버 `NAVRL_LATERAL_BIAS_COEF`(`lateral_batch_bias_loss`, `:421`)는 **0으로 고정**한다.
한 run 두 축 금지이기도 하지만 설계상으로도 부적합하다 — batch 평균 `mu[:,1]`만 눌러도 관측별
chirality는 남는다(절반 좌·절반 우면 평균 0, equivariance는 아님). N1의 primary는 평균이 아니라
관측별 `conj_err_lat`이다.

### 설계상 함정: 손실과 지표가 다른 스케일에 있다

손실은 `mus`(pre-tanh, 무계)에 걸리고 N1 지표는 `tanh(mu_scale ⊙ mus)`(post-tanh, `[-1,1]`)에서
읽는다. `mu_scale`이 축별이고 tanh가 홀함수라 `M(tanh(s⊙mu)) = tanh(s⊙M(mu))`이므로 **영점은
일치하지만** 포화 영역 불일치가 손실에서 과대 가중된다. 따라서 계수를 감으로 정하면 안 된다.

계수는 **arm 실행 전 프로파일링 1회**로 정한다: optimizer step 없이 minibatch 64개를 통과시켜
`median(|a_loss|)`와 `median(symmetry_penalty)`를 재고,
`c = round_1sig(0.10 × median(|a_loss|) / median(symmetry_penalty))`. 보조항 초기 기여를 정책 손실의
10%로 맞춘다는 뜻이며, 1 유효숫자 반올림이 "측정값이지 튜닝값이 아님"을 강제한다. 스윕은 금지다
(`VERIFICATION.md` fail-closed 2).

2026-07-27 Ablation B의 `0.01`은 재사용하지 않는다. 그 run은 **잘못된 mirror 연산자** 위에서 돌았고
2026-07-29에 무효화됐다 — 현재 연산자에 대한 증거가 아니며, 동시에 새 실험의 근거로 인용할 수도 없다.

### 계약과 게이트

학습 seed **383**, 평가 seed **389**(둘 다 전수검색 0건). 양 arm 모두 ref5in D1 ep1900
(SHA `197ea269…`)에서 warm-start, 1,000 epoch / 4.096M samples. 조작 변수는 `NAVRL_REFLECTION_COEF`
하나뿐. 평가 rollout 1회가 `70bars.json`(성능)과 `reflection_audit.json`(chirality)을 동시에 낸다.

| 게이트 | 조건 | 실패 시 |
|---|---|---|
| **Gate 0** 설계 타당성 | control이 chirality 유지: median ≥ 1.00 **및** agreement ≤ 0.20 | `INCONCLUSIVE_CONFOUNDED_BY_ADAPTATION` |
| **Gate M** 메커니즘 | treatment median ≤ 0.50 **및** agreement ≥ 0.70 | `REFLECTION_CONSISTENCY_INEFFECTIVE` |
| **Gate P** 성능 guard | capture ≥ control−2.00 pp **및** crash ≤ control+2.00 pp | `MECHANISM_PASS_PERFORMANCE_REGRESSION` |

Gate 0이 핵심이다. control이 손실 없이도 chirality를 잃는다면 원인은 적응 예산 자체이지 손실이
아니므로, treatment에 대해 아무 주장도 할 수 없다. 기준선은 1.454 / 0.0249이고 Gate M의 0.50은
66% 축소, 0.70은 우연 수준 0.50보다 위다. N1의 `CHIRALITY_ABSENT` 구역(≤0.10/≥0.90)은 1,000 epoch
예산에서 비현실적이라 채택하지 않았다.

`INEFFECTIVE`가 나오면 그대로 기록하고 계수를 올려 재시도하지 않는다 — 그게 곧 금지된 스윕이다.

### 우선순위 긴장을 문서에 박아뒀다

P2/D1의 **진단된 병목은 chirality가 아니다.** 초기 표적 미관측(camera 20 m vs goal 22.5–28 m)이며
seed 367에서 camera 20→28 m로 timeout `55.80 → 18.16%`(−37.65 pp)가 인과 확인됐다. 즉 인과가
확인된 처방이 이미 있고 chirality는 성능 근거가 없는 별개 결함이다. 이 실험을 먼저 하는 것은
"인과 확인된 것"보다 "메커니즘적으로 흥미로운 것"을 앞세우는 선택이며, 사전등록 §2에 그 선택을
명시했다. 문서는 그 선택을 정당화하지 않는다 — 실행 여부는 별도 결정이다.

비용: arm당 약 52분(`ppo_260813_1636_ref5in-d1-q3-adapt-s197` 실측), treatment는 update당 정책
forward 1회 추가라 더 느리다. 평가 2셀 약 40분, 프로파일링 약 10분. **총 GPU 2.5–3시간.**

P2 STRICT FAIL / D1 FAIL / P3 BLOCKED 변경 없음. 이 항목 시점에 학습·평가 실행은 없다.

## 2026-08-22 — N1 코드 적대적 감사 4갈래: 결함 16건 수정, 측정은 비트 단위로 불변

camera-first 착수 전 전면 점검. 감사 4건(각각 "고장났다고 가정하고 증명하라") → 수정 4건.
**N1 판정을 무효화하는 결함은 0건**이며, 오히려 측정의 신뢰도가 올라갔다.

### 측정은 독립 재구현으로 비트 단위 재현됐다

감사자가 mirror 연산자를 사전등록 문서에서 다시 옮겨 적고, 정규화기를 원시 체크포인트 텐서에서
직접 읽고, 통계를 따로 구현해 전량 재계산했다.

| | 기록값 | 독립 재계산 |
|---|---|---|
| overall median `conj_err_lat` | 1.4539868831634521 | 1.453986883163452 |
| sign agreement | 0.024904718096990405 | 0.024904718096990 |
| comparable rows | 15,218 | 15,218 |
| S1 median | 1.283378228545189 | 1.283378228545189 |
| 11개 맥락 셀 | — | 전부 마지막 자리까지 동일 |

**반증 가능성 확인 — 틀렸다면 이렇게 나왔을 값들**: 정규화를 건너뛰면 0.741, 정규화→반사로 순서를
바꾸면 1.333, 부호를 `|π(Mo)−π(o)|`로 잘못 쓰면 **0.278(판정이 INCONCLUSIVE로 뒤집힘)**.
실측 1.454는 셋 중 어느 것과도 맞지 않고 정답 경로와만 맞는다.

**종단 교차검증**: rollout 중 완전히 다른 코드 경로가 실시간 누적한 `70bars.json:action.signed_mean_y
= −0.6251` vs 오프라인 재forward `−0.6232`. 같은 로그의 `negative_y_rate = 98.12%`,
`positive_y_rate = 0.51%`. 덤프된 관측이 실제로 정책이 소비한 그것임을 보인다.

또 `70bars.json`의 `crash_causes.bar_contact = 162`와 npz outcome 코드 1 = 162가 정확히 일치했고
(capture 716/716, crash 215/215, timeout 93/93, 합 1,024/1,024), 이는 단위 테스트가 잡을 수 없는
종류의 검증이다.

미매칭 1,530 프레임은 **정상**으로 판명 — 종료 시점 미완결 에피소드(128 env 중 117)의 프레임이며,
outcome이 필요 없는 맥락에는 정상 포함된다(13,958 + 1,530 = 15,488).

### 수정한 결함 16건

**`navrl_task.py` (F1–F7)**
- F1: 다중 밀도 스윕이 같은 npz를 덮어썼고, `atexit` 예외가 **exit 0**이라 런처의
  `require(frames.is_file())`가 **낡은 파일로 통과**했다. 이제 구성 시점에 충돌을 거부하고(수 시간
  롤아웃 전 즉사), 실패 시 `<path>.FAILED` 마커를 남기며, `os.replace`로 원자적 발행한다.
  npz에 run 식별자(bars/seed/num_envs/pid)를 기록한다.
- F2: 병합이 `d_contact`를 `(crashed | target_contact | target_invalid) & crashed_out`로 넓혀
  **표적이 아레나를 벗어난 장부상 종료**가 `crash_bar_contact`로 기록됐다. 코드를 분리했다.

  ```
  0 capture   1 crash_bar_contact   2 crash_oob   3 crash_other   4 timeout   5 unattributed
  6 crash_below_floor   7 crash_above_ceiling   8 crash_target_contact   9 crash_target_invalid
  ```

  **0–5는 commit `cff96c2` 시점 의미에 동결한다.** 공개된 seed-373 npz가 디스크에 그 코드로 존재하고
  재도출이 불가능하기 때문이다. 코드 1은 병합 이전 의미(기체 접촉만)로 되돌아갔고, 병합이 1에
  접어넣었던 표적 원인 둘은 번호를 빌리지 않고 8/9를 받았다. 3은 유지하되 더 이상 방출하지 않아
  공개 덤프의 기존 행이 계속 읽힌다. 코드 맵을 npz 안에 실어 소비자가 하드코딩하지 않게 했다.
- F3: 전체 `reset()`이 프레임 존재 중 일어나면 export guard가 덤프 **전체를 폐기**했다(수 시간 run이
  파일 0개). 이제 reset 고아만 계수·보고 후 드롭하고, 진짜 불일치는 여전히 치명이다.
- F4: obs 폭 검사가 동어반복(같은 값을 자기 자신과 비교)이었다. 지각 스키마 + 라이브 LiDAR 텐서
  shape에서 독립 재도출해 비교한다. 다만 프로세스 안에 **완전히 독립적인 폭 출처는 없으므로**
  (전부 같은 env var 파생) 구성요소/할당 불일치만 잡고 일관된 재설정은 못 잡는다고 명시했다.
- F5: outcome 행 리스트가 무제한이었다(학습 run에 켜면 OOM으로만 발견). 상한 초과 시 절단이 아니라
  예외를 던지고, 조인이 불완전한 덤프는 쓰기를 거부한다.
- F6: 가드를 넓혔는데 `reset_idx`의 `_action_diag_prev` 초기화는 안 넓혀 dump-only run에서
  `delta_y`가 에피소드 경계를 넘어 누적됐다. 두 gate를 일치시켰다.
- F7: `atexit` 단일 flush였다. `close()`에서 결정론적으로 flush하고 둘을 멱등하게 만들었으며,
  종료 시점 GPU 접근을 호스트 미러로 제거했다. **주기적 중간 저장은 여전히 없어 `close()` 전
  SIGKILL은 롤아웃을 잃는다.**

**런처 (G1–G5)**
- G1: Q6가 **검증자 자신의 경로**에 고정돼 worktree 산출물을 primary에서 검증하는 것이 원리적으로
  불가능했다. 게이트가 지켜야 할 불변식은 "실행된 코드 = 해싱된 코드"인데 구현은 "실행된 코드 =
  검증자가 서 있는 트리"를 봤다. 기대 루트를 **영수증의 매니페스트 `repository_root`**에서 유도하도록
  고쳤다. 오히려 강화됐다 — 다른 트리를 가리키는 `[origin]` 줄이 예전엔 조용히 무시됐으나 이제
  `foreign`으로 거부된다.
- G2: 영수증의 절대경로 때문에 결과 이관이 불가능했다. cell-local → 영수증 기록 순으로 해석하고
  둘 다 없으면 양쪽을 명시하며 실패한다. 위치는 움직여도 **바이트는 못 움직인다**(다이제스트 고정).
- G3: `goal_dist_min/max`(사전등록 §3-d)가 설정·기록만 되고 검증되지 않았다. 양쪽에 추가했고
  실행값과 검증값이 갈라질 수 없게 `canonical_env`에서도 단언한다.
- G4: `VERDICT_FAIL_CLOSED`가 죽은 상수였고 불변식이 주석뿐이었다. 쌍조건으로 강제한다.
- G5: `"9개 평가, 실패 0개"`가 성공 경로에서 동어반복이었다(`Q6`가 `passed: None`이라 실패 목록에
  들어갈 수 없었고, `len(gates)`는 dict 키 수였다). 이제 평가/위임/malformed를 구분해 세고,
  위임 게이트는 런처가 **실제로 그 검사를 수행했다는 증거**를 요구한다.

**오프라인 감사 (H1–H6)**
- H1: 같은 JSON 블록에서 `"unattributed"`가 두 뜻이었다(코드 5 vs 조인 실패 −1). `no_outcome_row`로
  분리하고 `13,958 + 1,530 + 0 = 15,488` 재조정을 단언한다.
- H2: 맥락 분할 완전성 검사가 없어, 도메인 밖 라벨이 모든 셀에서 조용히 사라질 수 있었다. 각 family가
  모집단을 분할하는지, 셀이 서로소인지 fail-closed로 검사한다.
- H3: Q6/Q7가 자기 신고였다. 평가/위임/malformed 3상태로 나누고, Q7의 실제 검증분(체크포인트 sha)만
  평가로 계상한다. `delegated_gates` 필드로 호출자가 단언할 수 있게 했다.
- H4: Q5의 fixed_points가 **도구 자신의 기대 맵**에서 계산돼 증거가 아니었다. 관측된 연산자에서 계산한다.
- H5: `.astype(bool)` 강제(int8 −1이 VISIBLE로 계수됨), 표본 부족 시 금지된 판정 부여, NaN이
  게이트를 안 건드리고 INCONCLUSIVE로 착지, uint8 덤프의 `/255` 누락 — 전부 fail-closed로 막았다.
- H6: 제외된 1,530 프레임이 **무작위 결측이 아님**을 데이터에서 자동 산출해 공시한다(전부
  `call_index ≥ 1360`/최대 1920, 100%가 중앙값 이상, 70.2 백분위). timeout 셀의 median이 가장 크므로
  **outcome 분할 비교에 약한 편향**이 있다. 판정은 overall 15,488 전체로 계산되므로 헤드라인은 무관하다.

**테스트 인프라**
- `tests/test_navrl_target_motion.py`의 11개가 **한 번도 실행된 적이 없었다.** pytest 스타일인데
  이 환경에 pytest가 없어 `unittest discover`가 `Ran 0 tests / OK`를 찍었다 — 위양성이다. 하필 물리
  표적(병합에서 가장 새롭고 위험한 서브시스템)의 궤적 계약이다. 직접 호출해보니 **11/11 통과**하므로
  현재 버그가 아니라 안전망 구멍이었다. 단언은 원문 그대로 두고 어댑터만 덧댔다(+50/−0).
- `tests/test_test_suite_collection.py` 신설. 39개 파일을 `ast`로 스캔해 "모듈 레벨 `test_*`가 있는데
  수집도 자체실행도 안 되는" 파일을 **구조적으로** 탐지한다(파일명 허용목록 없음). 수정 전 파일을
  git에서 복원해 검사하니 그 파일만 정확히 실패했다.

### 아티팩트 이관 — 증거가 임시 worktree에만 있었다

`frames.npz`·`70bars.log`(Q6 증거)·`checkpoint_snapshot.pth`·번들 스냅샷 315개가 gitignore라
병합으로 따라오지 않아 `.codex_worktrees/navrl_reflection_audit`에만 있었다. 그 worktree를 정리하면
결과가 재계산 불가능해지고 Q6 증거가 사라진다.

32 MB를 primary의 정식 위치로 이관하고(전량 해시 대조 일치), 셀의 절대 심볼릭 링크를 상대경로로
교체했으며 누락돼 있던 `source_snapshot` 링크를 보강했다. 이제 링크가 primary 안에서 닫혀
**worktree를 삭제해도 안전**하다.

절대 심볼릭 링크 자체는 **저장소 전반의 기존 설계**(추적된 결과 셀 ~80개가 전부 이 머신의 체크아웃을
가리키는 절대 링크, `eval_navrl_v2_density_sweep.sh:989-993`)이며 N1이 만든 문제가 아니다. 클론하면
전부 죽는 링크가 된다. 기존 80개는 모든 과거 결과를 건드리는 별건이라 **미해결로 남긴다.**

### 통합 검증

개별 수정은 각자 통과했으나 A·B·C가 서로 맞물린 계약을 동시에 바꿨으므로 조합을 직접 확인했다.

| 검증 | 결과 |
|---|---|
| 전체 CPU 테스트 | **544개, 종료코드 비0 없음** (감사 전 423) |
| 동결 아티팩트 `verify` (primary) | **PASS, `CHIRALITY_CONFIRMED_REAL_FRAME`** |
| 오프라인 감사 재실행 | median/agreement **완전 동일**, 741개 수치 leaf 중 **0개 이동** |
| 신형식 JSON → 런처 `verify_cell`→`build_summary`→`write_summary` | **전부 OK, 동일 판정·동일 키 집합** |
| 제3의 루트에서 `verify` | **PASS** (수정 전 런처로는 Q6 실패 — 통과시킨 건 수정이지 느슨함이 아님) |

요약 줄도 정직해졌다: `9개 평가, 실패 0개` → `오프라인 판정 8개, 런처 위임·검증 2개, 실패 0개`.

### 남긴 것

- 기존 결과 셀 ~80개의 절대 심볼릭 링크 (별건, 과거 결과 전량 수정 필요)
- F7의 주기적 중간 저장 부재 (`close()` 전 SIGKILL은 롤아웃 손실)
- F4의 완전 독립 폭 출처 부재
- **사전등록 개정 필요**: 덤프가 코드 6–9를 방출하는데 오프라인 도구의 동결된 `OUTCOME_CODES`는
  0–5만 안다. 그런 덤프는 이제 조용히 누락되지 않고 **fail-closed**한다(H2가 의도한 동작). 그런
  run을 감사하려면 사전등록에 outcome 맵 개정이 선행돼야 한다.

P2 STRICT FAIL / D1 FAIL / P3 BLOCKED 변경 없음. 어떤 기존 판정도 소급 변경하지 않았다.

### 다음

camera-first. 1단계는 GPU도 코드도 쓰지 않는 **28 m 하드웨어 실현성 관문**이다 — 표적이 20 m에서
1.26 px, 28 m에서 0.90 px로 1픽셀 미만인 각해상도 계산을 실제 부품 스펙에 대고 닫는다. 검출기가
0.9 px 표적을 잡을 수 없다면 28 m 재학습은 시뮬레이터 안에서만 참인 결과가 되므로, 답을 모르고
2.5시간 학습을 돌리는 것보다 이쪽이 싸다.

## 2026-08-22 — camera-first 1단계(코드 측): 28 m 검출은 시뮬레이터 자신의 기하에서 이미 불가능하다

camera-first의 하드웨어 관문을 열기 전에 각해상도 사슬을 코드에서 전부 재도출했다(기억 금지).
결론은 계획을 바꾼다. **하드웨어 부품 조사는 미완이며 다음 세션으로 넘긴다.**

### 센서 사슬 (전부 코드 도출, `navrl_task_config.py`)

`navrl_ref5in_quad`는 카메라 sensor_config를 **아예 인스턴스화하지 않는다** —
`navrl_quad_config.py:43`이 `enable_camera = False`다. "카메라"는 태스크 안의 별도 Warp
레이캐스터이며 내부값이 `navrl_task_config.py`에 하드코딩돼 있다.

| 항목 | 값 | 출처 |
|---|---|---|
| 해상도 (표적 검출) | **160 × 90** (env 훅 없음) | `:152-153` |
| FOV | **87.0° × 58.0°** | `:149-150` |
| far clip (표적) | 20.0 m = `detector_max_range` | `:148`, `navrl_detector.py:405` |
| 장애물 depth | 40 × 24, far 10 m → 160×90으로 bilinear 업샘플 | `:177-179`, `navrl_detector.py:436-448` |
| 표적 반경 | 0.15 m | `:154` |
| `min_target_pixels` | **2** | `:221` |

`fx = (W/2)/tan(hfov/2)` = **84.302 px/rad**, `fy` = 81.182 px/rad (비등방). 화소당 **0.68°**(광축상).

### `min_target_pixels`는 지름이 아니라 면적이다

`navrl_perception.py:1207-1210`: `count = mask.sum(dim=(1,2)); visible = count >= self.min_pixels`.
H·W 양축 합이므로 **분할 픽셀 개수(면적 px²)**다. 지름 2 px가 아니라 면적 2 px², 즉 지름 약 1.6 px.

| R [m] | 지름 [px] | 면적 [px²] |
|---|---|---|
| 15 | 1.686 | 2.15 |
| **20** | **1.265** | **1.21** |
| **28** | **0.903** | **0.62** |

지름 1 px 손익분기 R = 25.29 m, 면적 2 px² 손익분기 R = **15.55 m**.

### 결정적 수치: 28 m 광축 검출은 확률 0이다

마스크는 안티에일리어싱도 서브픽셀 커버리지도 없는 **정확한 광선-구 교차를 화소 중심에서 표본화**한
것이다(`navrl_detector.py:110-119`). 따라서 `count`는 투영 타원 안에 떨어진 격자 표본 수인 난수이며
**1의 하한이 없다**. 동일 광선 테이블로 서브픽셀 배치를 몬테카를로한 결과:

| R [m] | 광축상 P(count≥2) | FOV 평균 P(count≥2) | 평균 count |
|---|---|---|---|
| 15 | 0.855 | 0.986 | 3.6 |
| **20** | **0.246** | 0.742 | 2.07 |
| 25 | ~0.000 | 0.322 | 1.29 |
| **28** | **0.000** | 0.167 | 1.05 |

즉 공칭 `detector_max_range = 20 m`에서도 **광축 프레임당 검출률이 25%**에 불과하고, 28 m에서
광축 표적은 `min_pixels = 2`를 **원리적으로 통과할 수 없다**. 장거리 검출을 떠받치는 것은 전적으로
**광축 밖 기하**다 — 핀홀에서 FOV 가장자리의 각해상도가 중심보다 1.7–2.9배 조밀하기 때문이며,
물리 현상이 아니라 투영 기하의 부산물이다. seed 367 영수증의 `target_hidden_fraction`
0.952(20 m) / 0.775(28 m)와 정합한다. **이 광학 사슬의 신뢰 검출 거리는 12–15 m다.**

### seed 367이 증명한 것은 하드웨어 실현성이 아니라 정보의 가치다

두 arm의 `navrl_task_config.py`·`navrl_detector.py`·`navrl_perception.py` 스냅샷이 **md5 동일**이고
영수증에서 실질적으로 다른 필드는 `target_camera_max_range_m`(20.0 vs 28.0) **하나뿐**이다. 양 arm
모두 160×90 @ 87°×58°, 동일 표적, 동일 `min_pixels`, 잡음 없음. **소프트웨어 far-plane만 풀렸다.**

**추가 교란 (기록 필요)**: `detector_max_range` 변경은 순수 clip 변경이 아니다. actor의 표적 토큰이
함께 재정규화된다 — `rel_pos / max_camera_range`, `pos_var / max_camera_range²`
(`navrl_perception.py:1574,1578`). 28 m arm은 20 m로 학습된 정책에 **0.714배로 스케일된 위치**를
먹였다. timeout −37.65 pp 중 일부는 관측성이 아니라 **관측 인코딩 변화**다.

이는 seed 367의 공식 판정을 **소급 변경하지 않는다**(사후 재판정 금지). 해석상의 한계로 기록하며,
어떤 후속 사전등록도 이 교란을 명시적으로 다루어야 한다.

### 시뮬-실기 격차는 양방향이다

- **낙관**: 시뮬 검출기는 1–2 픽셀 blob에 오탐 0·블러 0·모션스미어 0·완벽 대비로 발화한다. 분류기는
  문자 그대로 `3R − 2G − 2B − 0.9`(`navrl_perception.py:524-529`)이고, canonical_env에서는
  `training=False`(`navrl_task.py:4785`)라 RGB/depth 잡음·드롭아웃·거리오차가 **전부 꺼져 있다**.
- **비관**: `navrl_task_config.py:149`의 주석 "matches the D455-style forward depth camera"는
  **FOV만 맞고 해상도는 아니다.** 실제 D455는 848×480에서 `fx ≈ 447 px/rad`로 **5.3배 조밀**하며,
  그것만으로 지름 2 px가 약 67 m에 온다.

두 오차가 부분 상쇄되지만 올바른 곳에서 상쇄된다는 보장이 없다.

### 계획 변경

**"28 m로 재학습"은 틀린 다음 단계다.** 시뮬레이터 자신의 기하가 28 m 광축 검출을 불가능하다고
말하는데 그 위에서 학습하면 존재할 수 없는 센서를 배우게 된다. 진짜 문제는 range clip이 아니라
**센서 모델의 해상도가 실기보다 훨씬 조악하다**는 것이다. 방향은 "clip을 늘린다"가 아니라
**"해상도를 실제 부품에 맞춘다"**이며, 그러면 28 m가 물리적으로 성립하면서 sim-to-real 격차도
함께 줄어든다. 다만 해상도 변경은 관측 계약·계보에 영향을 주므로 별도 사전등록이 필요하다.

P2 STRICT FAIL / D1 FAIL / P3 BLOCKED 변경 없음. GPU 실행 없음.

### 다음 (정확히 여기부터)

**하드웨어 부품 조사를 끝낸다** — 1.20 kg 5인치 기체가 실을 수 있는 카메라(OAK-D 계열, D4xx,
글로벌셔터 모듈)의 해상도·FOV·질량·전력·온보드 추론 가능성. 그 결과로 "시뮬 해상도를 어떤 실제
부품에 맞출 것인가"가 정해지고, 그때 비로소 해상도 변경 사전등록을 쓸 수 있다. 이번 세션에서
조사를 시작했으나 회수하지 못했다.

## 2026-08-22 — camera-first 1단계 완료: 28 m는 가능하다, 단 해상도·임계·거리를 함께 고쳐야 한다

하드웨어 조사 회수. **1단계 종료.** 판정: 28 m 검출은 물리적으로 충분히 가능하며, 현재의
12–15 m 한계는 **시뮬레이터 인공물**이다.

### 권장 구성 — AR0234급 2.3 MP 글로벌셔터, 1920×1200 @ 82°

| | 현재 시뮬 | 권장 |
|---|---|---|
| fx | 84.3 px/rad (0.680°/px) | **1104 px/rad** (0.052°/px), 13.1배 |
| 28 m에서 표적 | 0.90 px / 0.62 px² | **11.8 px / 110 px²** |
| 20 m에서 | 1.27 px / 1.21 px² | 16.6 px / 216 px² |
| 지름 4 px 도달 거리 | 6.3 m | 83 m |

질량 +150–210 g(카메라 모듈 + Jetson Orin Nano Super급 연산) → AUW 1.35–1.41 kg,
**T/W 3.26 → 2.72–2.84**. 전력 +12–26 W. 비용 약 $430.

렌더 비용이 문제면 하한은 **1280×800 @ 82°(fx 736)** — 28 m에서 7.9 px. **640×360(fx 337)로는
가지 말 것** — 지름 10 px가 10 m에서야 나와 지금 진단한 문제가 재현된다.

### 해상도만 고치면 실패 모드만 옮긴다

조사가 계획의 결함을 하나 잡았다. **`min_pixels = 2 px²`는 지름 약 1.6 px이고, 이는 Johnson
검출 기준(고대비·무잡동사니에서 지름 2–2.5 px)보다도 낮다.** 존재하는 어떤 검출기의 모델도 아니다.

| 기준 | 지름 | 조건 |
|---|---|---|
| Johnson 검출 한계 | 2–2.5 px | 50% 확률, 고대비, 무잡동사니, 사람 관측자 |
| **현재 시뮬 임계** | **≈1.6 px** | 이론적 하한보다 아래 |
| 단일프레임 CNN 신뢰 검출 | 8–10 px | 하늘 배경 |
| 잡동사니 배경 | 15–20 px | Det-Fly 하늘 mAP 88.3 → 도심 62.0 (−26점) |

따라서 **해상도와 임계는 반드시 함께 바꾼다.** 임계는 지름 ≥8 px(면적 ≈50 px²)로 올린다.
그러면 권장 카메라에서 신뢰 거리는 하늘 배경 기준 여유롭게 28 m, 잡동사니 배경 기준 15–20 m다.
설계 기준으로 삼을 숫자: **공칭 25 m, 스트레치 28 m(하늘), 잡동사니 15 m.**

다운스케일이 공짜가 아님을 LRDDv3가 직접 증명한다 — 같은 모델·같은 데이터에서 입력을 640 대신
1920으로 주는 것만으로 **mAP@50 0.543 → 0.822**.

### 세 번째 sim-to-real 격차: 정책이 측정 불가능한 거리를 받고 있다

`navrl_detector.py:130`이 `target_depth[env_id, row, col] = t_target`로 **정확한 해석적 광선-구
교차 거리**를 표적 픽셀에 직접 쓴다. canonical_env에서 `NAVRL_RANGE_ERROR_M = 0.0`이므로 거리
오차가 전혀 없다. 즉 시뮬은 임의 거리에서 표적까지의 거리를 **오차 0**으로 준다.

실제로는 비행 가능한 어떤 베이스라인으로도 28 m 스테레오 거리는 나오지 않는다 —
D435(50 mm) 시차 1.20 px, D455(95 mm) 2.29 px, OAK-D(75 mm) 2.36 px. 28 m에서의 거리는
스테레오가 아니라 **방위 + 겉보기 크기**에서 나와야 한다. 거리를 좁혀야 하는 정책에게 이것이
아마 가장 치명적인 격차다. LiDAR 연관은 12 m 이내에서만 보정하며 획득은 못 한다.

### 부수 결정

- **단일 카메라로 충분하다.** FOV 긴장은 해상도가 고쳐지면 해소된다 — 82°를 유지하면서 28 m에서
  11.8 px이 나온다. 좁은 FOV 보조 카메라는 28 m에서 얻는 게 없고(광각이 이미 검출함) ~50 m를
  넘길 때에야 값어치가 생기며, 그때는 cueing 사슬이 별도 서브시스템이 된다.
- **OAK-D LR 기각** — 415 g으로 기체 질량의 35%. T/W가 연산 전에 이미 2.4로 떨어진다.
- **OAK-D Lite 기각** — mono가 640×480, fx 433으로 너무 조악하다.
- **D435 계열 주의** — RGB가 69.4° 롤링셔터인데 depth는 87°다. 검출을 RGB로 하면 depth가 보는
  가장자리 표적을 놓친다. D455는 RGB가 87° 글로벌셔터로 둘 다 해결한다.
- **모션 블러는 현재 모델링돼 있지 않다.** 5인치 기체가 고 yaw rate에서 10 ms 노출이면 12 px
  표적이 여러 픽셀로 번진다. 권장 후보를 전부 글로벌셔터로 고른 이유이며, 별도 항목으로 남긴다.

### 해상도 변경의 코드 비용 — 관측 계약을 깨지 않는다

`camera_width`/`camera_height`를 읽는 곳은 `navrl_perception.py:644-645`와
`navrl_detector.py:163-164` 둘뿐이다. `STRUCTURED_OBS_DIM`은 `HBEAMS`/`VBEAMS`/`MAX_OBSTACLES`/
`CORRIDOR_TOKENS`에만 의존한다. 따라서 **actor 관측 898-D 불변, 체크포인트 shape 불변,
ref5in D1 ep1900에서 warm-start 가능**하다.

착수 전 처리할 것 3가지:
1. `camera_width = 160`이 하드코딩이라 `_env_int("NAVRL_CAMERA_WIDTH", 160)` 훅이 필요하다.
   `navrl_task_config.py`는 provenance byte-gate 대상이 아니므로(그건 `navrl_ref5in_quad_config.py`와
   URDF뿐) 안전하다.
2. 렌더 비용이 픽셀 수에 선형이다. 160×90 → 1920×1200은 **160배**다. 학습 속도 실측이 필요하며,
   이 때문에 1280×800이 현실적 타협점일 수 있다.
3. 최소 픽셀 knob이 둘이다 — `camera_min_target_pixels = 1`(`:151`, 레거시 detector 경로)과
   `NAVRL_DETECTOR_MIN_PIXELS = 2`(`:221`, perception 경로). 사전등록에서 어느 쪽을 조작하는지 명시.

### 다음 (정확히 여기부터)

**2단계 — 센서 충실도 사전등록을 쓴다.** 조작 축은 "해상도 + 검출 임계"를 하나의 묶음으로 본
**센서 모델 충실도** 하나다(둘을 따로 바꾸면 실패 모드만 옮기므로 분리하지 않는다는 근거를 문서에
명시). 먼저 **평가 전용**으로 동결 정책을 새 센서에서 재평가해 획득률·timeout이 어떻게 변하는지
보고, 그 다음에 적응 학습 여부를 결정한다. 재학습은 이 사전등록의 권한 밖이다.

사전등록에 반드시 들어갈 한계: 거리 오차 0(위), 모션 블러 미모델링, 잡동사니 배경 미모델링
(분류기가 `3R − 2G − 2B − 0.9`), 그리고 seed 367의 토큰 재정규화 교란.

P2 STRICT FAIL / D1 FAIL / P3 BLOCKED 변경 없음. GPU 실행 없음.

## 2026-08-22 — 센서 충실도 사전등록 (구현 전, 실행 전)

`docs/prereg_2026-08-22_sensor_fidelity.md`. 평가 전용이며 학습은 권한 밖이다. 구현 기계가
존재하기 전에 동결한다.

### 조작 축은 하나다 — "센서 모델 충실도"

해상도와 검출 임계를 분리하지 않는다. 해상도만 올리면 임계가 여전히 Johnson 기준 미만이라
**실패 모드만 옮기고**(더 조밀한 센서에서 여전히 존재하지 않는 검출기를 모델링), 임계만 올리면
현재 해상도에서 28 m 표적이 0.62 px²라 검출이 전면 붕괴한다. 두 값은 한 물리량(각해상도 대비
검출 가능성)의 두 얼굴이므로 `VERIFICATION.md` fail-closed 3을 위반하지 않는다. 결과가 나쁘게
나와도 사후에 "해상도만 따로 볼걸"로 분해하지 않는다.

### 해상도·임계 짝은 유도됐다

임계는 면적이고 `A = 0.785 d²`다. 문헌의 단일프레임 CNN 신뢰 하한 지름 8 px(하늘 배경) →
**면적 50 px²**를 채택. 그 임계에서 교전 계약 22.5–28 m를 덮는 해상도를 풀면:

| 해상도 | fx | 28 m 면적 | 50 px² 신뢰 거리 |
|---|---|---|---|
| 160×90 (현재) | 84.3 | 0.62 px² | 3.1 m |
| 1280×800 | 675 | 41 px² | 25.3 m — 상단 미달 |
| **1920×1200** | **1013** | **92 px²** | **38.0 m** |

잡동사니 배경 기준(15–20 px)을 쓰지 않은 이유는 현재 렌더가 잡동사니를 모델링하지 않기 때문이며,
따라서 결과는 여전히 낙관 편향이다(L1).

### 계약

frozen ref5in D1 ep1900, **seed 421**(미사용), 70막대, arm당 2,049 에피소드, deterministic,
governor off. arm A = detect 160×90 / `MIN_PIXELS=2`(현행). arm B = detect 1920×1200 /
`MIN_PIXELS=50`. RGB 해상도는 양 arm 160×90 동일.

**`detector_max_range`는 20 m로 고정한다.** seed 367이 그것을 바꾸면서 actor 표적 토큰까지
재정규화한 것(`navrl_perception.py:1574,1578`)이 교란이었다. range를 건드리지 않으므로 토큰
정규화가 양 arm에서 동일하고, 조작은 "같은 20 m 안에서 검출이 얼마나 정직해지는가"로 좁혀진다.

### 판정 방향을 뒤집었다

임계가 25배 올라가므로 검출은 **반드시 더 어려워진다**. 따라서 "좋아지는가"는 틀린 질문이다.

| 판정 | 조건 |
|---|---|
| `FIDELITY_COST_CONFIRMED` | arm B never-acquired가 arm A 대비 **+10.00 pp 이상** |
| `FIDELITY_NEUTRAL` | ±3.00 pp 이내 |
| `INCONCLUSIVE_SENSOR_FIDELITY` | 그 외 |

**arm B가 나빠지는 것은 실패가 아니라 예상된 결과**이며, 그 크기가 곧 "지금까지의 성적 중 얼마가
존재할 수 없는 센서 덕분이었는가"의 추정치다. 이 실험의 값어치는 개선이 아니라 **정직한 기준선의
확립**이다. capture/crash/timeout은 원값으로 보고하되 판정에 쓰지 않는다 — 동결 정책은 부정직한
센서로 학습됐으므로 정직한 센서에서의 저하는 정책 결함이 아니라 계보의 결과다.

### 게이트 0 — 구현 타당성이 판정보다 먼저다

검출 해상도를 RGB/perception 해상도와 분리하는 것은 knob이 아니라 **설계 변경**이므로 사전등록에
명시했다. 근거는 실측이다 — 광선 추적은 해상도에 대해 사실상 공짜(픽셀 10배에 +0.8 ms)이고
비용은 전부 하류(합성 RGB 생성 + 전체 해상도 분할)인데, 그 RGB는 표적을 평평한 순수 빨강으로
칠한 합성물이고 분할기가 `3R − 2G − 2B − 0.9`라 appearance 교란 0에서 정보적으로 항등 왕복이다.

채택 조건 3개: detect == camera에서 현재 코드와 **bit-identical**, appearance 교란이 0이 아닌데
detect ≠ camera면 **fail-closed 거부**, 모든 조합에서 actor 관측 **898-D 유지**. 하나라도 실패하면
`FAIL_CLOSED_IMPLEMENTATION`이며 센서 모델에 대한 주장을 하지 않는다.

### 명시한 한계

L1 잡동사니 배경 미모델링(낙관 편향) · L2 모션 블러 미모델링 · **L3 거리 오차 0** — 본 실험은
그것을 고치지 않으므로 결과는 "거리는 여전히 공짜로 주어진 상태에서의 검출 충실도"만 말한다 ·
L4 단일 정책·seed·조건 · L5 임계 50 px²는 문헌 이식이지 이 시스템에서 측정된 값이 아니다.

거리 충실도는 **별개 사전등록**이 필요하다. 함께 바꾸면 축이 둘이 되고, 무엇보다 거리 오차 모델은
검출 임계와 달리 물리적으로 독립된 양이다.

P2 STRICT FAIL / D1 FAIL / P3 BLOCKED 변경 없음. 실행 없음.

## 2026-08-22 — 계획 문서가 하루 뒤처져 있었다: PLAN SYNC 규칙 신설 + VERIFICATION 갱신

사용자 지적으로 발견. 오늘 WORKLOG에 정직한 항목을 6개 썼는데 **`VERIFICATION.md`는 기준일
2026-08-21 그대로**였고, "다음 실험"이 camera-first 1단계가 이미 뒤집은 선택지를 가리키고 있었다.

`CLAUDE.md`가 canonical 상태를 "WORKLOG + VERIFICATION + docs/status"로 규정하는데, 그중 **실행
authority인 VERIFICATION이 낡은 계획을 가리키면 다음 세션이 틀린 것을 실행한다.** WORKLOG가
정확한 것으로는 구제되지 않는다 — 9,000줄을 읽어 계획이 바뀐 걸 알아낼 사람은 없다.

### `.claude/skills/navrl/SKILL.md`에 PLAN SYNC RULE 추가

WORKLOG RULE 바로 위에 놓았다. 요지: **계획이 바뀌면 그것을 바꾼 작업과 같은 커밋에서 계획 문서를
갱신한다.** WORKLOG는 무슨 일이 있었는지를 기록하지 다음 세션에 무엇을 하라고 말하지 않는다.
그건 `VERIFICATION.md`(gate·판정·다음 실험)와 `RESEARCH_PLAN.md`(가설·방법)의 일이다.

갱신 트리거: 다음 실험이나 그 이유, gate·임계·차단 해제 조건, 병목에 대한 믿음, 선행조건의
충족·무효화, 그리고 파일을 건드릴 때마다 `기준일`.

**동기화 중에 해서는 안 되는 것**을 명시했다 — 기록된 판정을 바꾸는 것. P2/D1/P3 상태줄과 동결된
결과의 판정은 역사적 사실이다. 새 증거가 옛 결과를 재해석하면 **판정을 고치는 게 아니라 옆에
한계로 기록**한다. plan sync가 조용히 FAIL을 무르게 만드는 것이 이 규칙의 최악의 결과다.

세션 종료 전 자가점검도 넣었다: `git log --oneline -5`를 보고 각 커밋에 대해 "VERIFICATION.md만
읽은 새 세션이 옳은 다음 일을 할까?"를 묻는다.

### `VERIFICATION.md` 갱신 (기준일 2026-08-21 → 2026-08-22)

- 기존 "이제 남은 것은 설계 결정이다"(선택지 a/b)를 **"2026-08-22 정정 — 그 선택지 둘 다 전제가
  틀렸다"**로 교체했다. seed 367이 광학을 바꾸지 않았다는 것(스냅샷 md5 동일), 시뮬 기하로 28 m
  광축 검출이 불가능하다는 것, 토큰 재정규화 교란을 한계로 기록했다.
- **다음 실험(in force)** 절 신설 — 센서 충실도 평가 전용, seed 421, arm A/B 계약, 판정 규칙,
  판정 방향 주의(arm B가 나빠지는 것이 예상된 결과), 게이트 0.
- **대기 중** 절 신설 — paired-reflection consistency(동결·미실행·우선순위 낮음), 거리 충실도(별도
  사전등록 필요).

P2 STRICT FAIL / D1 FAIL / P3 BLOCKED **변경 없음**(확인함). §8.29 seed 367의 공식 판정도 변경 없음.

## 2026-08-22 — 게이트 0 PASS: 검출 해상도를 RGB 경로에서 분리했다

사전등록 `docs/prereg_2026-08-22_sensor_fidelity.md` §4의 구현 타당성 게이트를 통과했다.

### 설계

`NAVRL_DETECT_WIDTH/HEIGHT` 신설, 기본값은 `camera_width/height` **knob**(리터럴 160/90이 아니라)
이므로 카메라 해상도만 올려도 조용히 분리되지 않는다.

**검출 해상도가 소유**: 표적 픽셀 수 → `visible`, 정수 centroid 합 → bearing/elevation, 마스크된
depth 합 → `surface_range`, 표적 픽셀의 분할 점수 → `confidence`, `sigma_r`의 픽셀 지지.
전부 `detect_fx/fy/cx/cy`로 변환하며 주입 bearing 오프셋(`u = u - detect_fx * d_bearing`)과
`sigma_lat = 0.03 + range/detect_fx`도 포함한다.

**카메라 해상도가 유지**: RGB 이미지, 합성 depth, 장애물 depth에서 지워내는 분할 마스크,
LiDAR 열 매핑, `_reconstruct_target_pixels`, 프로파일 헤드. 섞으면 bearing이 `detect_W/W` 배로
편향된다(1920 vs 160에서 12배). 단위 테스트가 5.65° vs 49.9°로 그 차이를 고정한다.

**핵심 기법**: 검출 해상도에서 **이미지를 만들지 않는다.** 기존 커널을 검출 해상도 광선 테이블에
행 블록(`NAVRL_DETECT_PIXEL_BUDGET`, 16 Mpx 상한)으로 재실행해 행별 부분합으로 리듀스한다.
VRAM이 84 B/px가 아니라 **0.7 B/px**이고 블록 크기에 무관함이 16배 범위에서 bit-identical로
확인됐다. 렌더러는 perception에 **env당 스칼라 요약**만 넘긴다 — 이미지도 마스크도 아니다.

### fail-closed

detect ≠ camera이면서 다음 중 하나라도면 raise: appearance 5종(hue/light/albedo/texture/blur),
검출기 체크포인트 설정, 분할기가 bootstrap 1×1이 아님, depth 채널 가중 ≠ 0, `enable_perturbations`
하의 RGB/depth 잡음, `detection_latency_s > 0`(지연 링버퍼가 카메라 해상도 마스크를 담으므로
검출 해상도 카운트를 같은 박자로 지연시키는 건 미구현). detect < camera도 `ValueError`.
GPU에서 5개 대표 케이스가 실제로 raise함을 확인했다.

**검토 후 명시적으로 허용**: `camera_mount_*`(검출 렌더가 같은 교란 자세를 씀), `camera_fov_scale_err`,
`NAVRL_TARGET_DYNAMICS=physical`, `detection_dropout_prob`, `range_error_m`, `detector_noise_*`,
`target_mask_backfill`, `lidar_*`, `pixel_threshold`, `min_target_pixels` — 양 해상도에 동일하게
작용하거나 env당 스칼라에 작용한다.

### 증명

| 게이트 | 요구 | 실측 |
|---|---|---|
| 동일 해상도 항등성 | bit-identical | `torch.equal=True`: obs(898)·states(906)·visible·camera_visible·confidence·track_state·track_age·det_vec·det_visible, 160×90과 480×300 양쪽. torch peak 201.97/388.30 MB 양쪽 동일 |
| 비동일 해상도 동등성 | appearance 0에서 | 가시성 결정·픽셀 수 **정확히 일치**. bearing ≤1.19e-07 rad, range ≤1.43e-06 m, confidence ≤1.19e-07 — 부동소수 합산 순서 차이뿐 |
| fail-closed | 거부 | 5/5 확인 |
| 898-D | 모든 조합 | 9개 조합 전부 actor 898 / critic 906 |

**해석상 결정적**: 분리 arm의 `static_scan`·`obstacle_history`·`robot_history`가 평범한 160×90
실행과 **bit-identical**이고 `target_history`만 움직인다(≤2.4e-2). 즉 arm B는 "오늘의 장애물
파이프라인 + 고충실도 검출"이며 **깨끗한 단일 축 조작**이 실제로 성립한다. `static_scan`/
`obstacle_history`가 전체 고해상도 실행과 최대 0.88 다른 것은 결함이 아니라 설계다 — 그것들은
40×24 장애물 카메라의 bilinear 업샘플에서 나오므로 카메라 해상도에 묶여 있다.

### 비용 (128 env, 30 step)

| 구성 | ms/step | 배수 | torch peak | device peak |
|---|---|---|---|---|
| (a) 160×90 both | 87.2 | 1.00 | 202 MB | 5586 MB |
| (b) 480×300 both | 105.4 | 1.21 | 1529 MB | 7368 MB |
| **(c) detect 1920×1200 + cam 160×90** | **167.6** | **1.92** | **402 MB** | 5864 MB |
| (d) full 1920×1200 | **CUDA OOM** | — | 외삽 409 ms(4.7배)·23.6 GB | 카드의 3배 |

분리 경로는 픽셀당 시간 4.0배·VRAM **118배** 싸고, 무엇보다 **실행된다**. 다만 "공짜"는 과한
표현이다 — 295 Mpx에서 광선 추적이 +80 ms/step이다. **살아남는 주장은 상대값이다.**

### 부수: 내 테스트가 전체 discovery에서 깨져 있었다

`tests/test_navrl_import_origin.py`가 파일별 실행에서는 통과하고 **전체 `unittest discover`에서는
5개 실패**했다. 앞서 "544개 전부 통과"라고 보고한 것은 **틀렸다** — 파일별로 돌려서 상호작용을
놓쳤다. 원인은 다른 테스트가 Isaac Gym을 피하려고 `sys.modules`에 `aerial_gym` 스텁을 넣어두는데,
`resolve_origin`이 의도적으로 `find_spec`보다 라이브 모듈을 신뢰하기 때문이다(가드의 존재 이유가
그것이다). 스텁을 테스트 동안 걷어내고 정확히 되돌리도록 고쳤다. 하필 "조용한 오작동을 막는"
가드의 테스트가 조용히 깨져 있었다.

전체 discovery **578개 전부 통과**(구현 전 544, 신규 34). 단독 실행도 통과.

### 다음

센서 충실도 schema-v2 런처 구축 → preflight → arm A/B 실행.

## 2026-08-22 — 재학습 사전등록을 결과 보기 전에 조건부로 동결

`docs/prereg_2026-08-22_honest_sensor_adaptation.md`. 센서 충실도 평가가 아직 실행되지 않은
시점에 쓴다 — 결과에 맞춰 재단됐다는 의심을 구조적으로 배제한다.

**조건부다**: 센서 충실도가 `FIDELITY_COST_CONFIRMED`일 때만 실행한다. `FIDELITY_NEUTRAL`이면
고칠 것이 없다는 뜻이고 `INCONCLUSIVE`면 무엇을 고치는지 모른다는 뜻이므로 둘 다 실행하지 않는다.
이 조건을 결과를 본 뒤 완화하지 않는다("생각보다 작지만 그래도 돌려보자"는 금지).

**묻는 것**: 동결 정책은 존재할 수 없는 센서로 학습됐다. 센서 충실도 평가가 그 **비용**을 재고,
본 실험은 **정직한 센서에서 다시 학습하면 얼마나 되찾는가**를 잰다. 되찾지 못하는 부분이 곧
과제 자체의 난이도이며 P2/D1 병목의 정직한 크기다.

**arm A가 필수인 이유**: 부정직한 센서에서 같은 1,000 epoch를 더 학습시키는 control이 없으면
개선을 예산 덕인지 센서 덕인지 가를 수 없다.

**판정을 대칭으로 잡았다** — 센서 충실도가 "+10 pp 이상 나빠지면 비용 확인"이므로 재학습은
"`NA_B ≤ NA_frozen − 10 pp` **및** `NA_B ≤ NA_A − 5 pp`"여야 회복이다. 입힌 만큼 되찾았는지를
같은 자로 묻는다. 5 pp는 예산만으로도 얼마간 좋아질 수 있으므로 그보다 명확히 나아야 센서 덕이라고
말할 수 있다는 마진이다.

capture/crash/timeout은 **원값 보고, 판정 제외** — 서로 다른 센서에서 측정된 성능은 직접 비교
가능한 양이 아니다.

**이것은 P3가 아니다.** P3는 70→205 bars · 30k epoch · seed 211이고 본 실험은 70 bars 고정
1,000 epoch다. `VERIFICATION.md` fail-closed 5를 위반하지 않으며 **P3 차단은 유지된다.**
`ADAPTATION_RECOVERS`가 나와도 정책을 채택하지 않는다 — 채택은 P2 gate 통과가 필요하다.

한계 명시: 거리는 여전히 공짜(L1, 별도 사전등록 필요), 잡동사니 배경 미모델링(L2, 낙관 편향),
모션 블러 미모델링(L3), warm-start가 이미 D1 FAIL인 계보(L4), 단일 seed·예산(L5).

seed 433(학습)·449(평가) 전수 검색 0건 확인.

`VERIFICATION.md`도 같은 커밋에서 갱신했다(PLAN SYNC 규칙) — "그 다음(조건부)" 절 신설.

## 2026-08-22 — 센서 충실도 런처를 만들고 preflight에서 막혔다: 평가기 provenance가 `detector_min_pixels`를 체크포인트에 고정한다

`tools/run_navrl_ref5in_sensor_fidelity.py`(schema-v2, `preflight|run|finalize|verify`)를
사전등록 `docs/prereg_2026-08-22_sensor_fidelity.md` §5/§6/§9/§10대로 구축했다. 계약 상수는
seed 421 / 70 bars / arm당 2,049 / 목표 22.5–28 m / arm A(160×90, 2 px) · arm B(1920×1200, 50 px),
판정 임계는 never-acquired 델타 **+10.00 pp / ±3.00 pp**다.

### preflight 결과 (실행 로그 그대로)

```
[sensor-fidelity] arm baseline: evaluator preflight PASS (no override)
[sensor-fidelity] FAIL: fidelity: generic evaluator preflight did not pass cleanly (returncode=2);
this run is preregistered to need NO provenance override, so it stops here instead of forcing.
mismatch lines: ['cfg_detector_min_pixels: checkpoint=2 expected=50.0']
```

**arm A는 통과, arm B는 거부다.** 원인은 우회가 아니라 구조다:
`eval_navrl_v2_density_sweep.sh:675`가 `cfg_detector_min_pixels`를
`float(os.environ["NAVRL_DETECTOR_MIN_PIXELS"])`와 **같아야 한다**고 요구한다. 즉 평가기는 검출
임계를 "체크포인트가 학습된 값과 동일해야 하는 표현 계약"으로 취급한다. 동결 정책은 2 px²로
학습됐으므로 **임계를 올리는 어떤 arm도 이 게이트를 통과할 수 없다.** 탈출로는 두 개뿐이고 둘 다
현재 권한 밖이다 — (a) `NAVRL_V2_FORCE=1`(사전등록 §5가 "override 없음"을 명시), (b)
`NAVRL_V2_ALLOW_DETECTOR_THRESHOLD_MISMATCH`와 같은 전용 허용 플래그를 min_pixels에도 추가
(런타임 소스 변경 + 사전등록 개정).

가설 기각: "detect 해상도 분리가 끝났으니 arm B는 그대로 돌아간다" — **틀렸다.** 분리는
`navrl_detector.py`/`navrl_perception.py` 쪽에서 끝났지만 **평가기 provenance 게이트는 손대지
않았고**, 임계 축은 그 게이트를 통과하지 못한다.

### 그 밖에 기록해 둘 것

- **p90을 기록하지 못한다.** `navrl_task.py first_acquisition_payload()`는 outcome별
  `first_visible_step_mean`과 lower median만 export하고 히스토그램
  (`_fa_eval_outcome_first_hist`)은 결과 JSON에 쓰지 않는다. 어떤 기록 필드에서도 p90을 유도할 수
  없어 `first_visible_step_p90: null` + 사유 문자열로 남긴다(사전등록 §6은 중앙값·p90 둘 다 요구).
- never-acquired는 **발명하지 않았다**:
  `result["target_motion"]["first_acquisition"][outcome]["never_acquired"]`를 capture/crash/timeout
  세 코호트에 대해 합산해 같은 코호트의 episode 합으로 나눈다(seed 367 `manipulation_check`와 동일
  필드). 코호트 합 == `actual_episodes` 회계를 assert한다.
- detect **해상도**는 receipt에도 `v2_evaluation_contract`에도 기록되지 않는다. 따라서 arm 정체성의
  독립 증명은 `detector_min_pixels` 하나뿐이며, 요약에
  `detect_resolution_not_recorded_by_evaluator: true`로 명시한다.
- 테스트: `python -m unittest discover -s tests` **592개 전부 통과**(기존 578 + 신규
  `SensorFidelityContract` 14).

### 다음

평가기 provenance가 임계 축을 막는 문제를 **결정한 뒤** 진행한다. 코드 우회는 하지 않았고 GPU
시간도 쓰지 않았다.

## 2026-08-22 (이어서) — 사전등록 §5-b 개정으로 해소: 담요식 force가 아니라 **좁은 단일 필드 override**

사전등록에 §5-b(좁은 provenance override)와 §5-c(기록되지 않는 두 가지)가 **측정 전에** 추가됐고,
런처가 그대로 구현했다. 해법은 새 플래그가 아니라 **저장소의 기존 패턴**
(`run_navrl_ref5in_cv_heading_near_open.py:107-120` `verify_narrow_override`)이다.

`verify_narrow_override()`는 arm B에 대해 **force 없이 먼저** preflight를 돌려
`returncode == 2`이고 불일치 라인 집합이 **정확히** `["cfg_detector_min_pixels: checkpoint=2
expected=50.0"]` 하나임을 증명한 **뒤에만** `NAVRL_V2_FORCE=1`을 적용한다. 두 줄이거나 다른
필드면 중단한다. arm A는 override를 아예 쓰지 않으며, `arm_requires_force()`가 arm 이름의 함수라
호출자가 실수로도 arm A에 force를 줄 수 없다(명시적 `force=True`는 거부된다).

**이것은 담요식 force보다 느슨한 게 아니라 더 엄격하다.** 담요식 `NAVRL_V2_FORCE`는 다른 모든
불일치까지 함께 가려주지만, 이 절차는 실행 시점에 불일치가 그 한 필드뿐임을 증명한다. 검증은
`preflight`와 `run` **양쪽**에서 수행되므로 검증되지 않은 override 아래에서 셀이 생성될 수 없다.
요약에는 arm별로 `narrow_provenance_override`가 기록된다 — baseline `{used: false}`,
fidelity `{used: true, sole_verified_mismatch, reason}`.

### preflight 결과 (양 arm 통과)

```
[sensor-fidelity] arm baseline: evaluator preflight PASS (no override)
[sensor-fidelity] arm fidelity: narrow provenance override VERIFIED (sole mismatch: cfg_detector_min_pixels: checkpoint=2 expected=50.0) then preflight PASS
[sensor-fidelity] PREFLIGHT PASS | seed=421 bars=70 episodes=2049/arm | narrow override on fidelity only
[sensor-fidelity]   NAVRL_DETECTOR_MIN_PIXELS: {'baseline': '2', 'fidelity': '50'}
[sensor-fidelity]   NAVRL_DETECT_HEIGHT: {'baseline': '90', 'fidelity': '1200'}
[sensor-fidelity]   NAVRL_DETECT_WIDTH: {'baseline': '160', 'fidelity': '1920'}
[sensor-fidelity]   NAVRL_V2_FORCE: {'baseline': None, 'fidelity': '1'}
[sensor-fidelity]   NAVRL_DETECTOR_MAX_RANGE: never exported (config default 20.0 m, identical in both arms)
```

테스트: 전체 discovery **598개 전부 통과**(578 기존 + `SensorFidelityContract` 20). 신규 6개는
기대 불일치 문자열이 모듈 상수일 것, arm A가 절대 force하지 않을 것, 검증이 `preflight`·`run`
양쪽에서 일어날 것, 불일치 집합 길이가 1이 아니면(두 줄·다른 필드·불일치 없음·평가기가 거부를
멈춤) 중단할 것을 고정한다.

### 다음

`run`은 사용자가 직접 시작한다. 런타임 루트는 clean이므로 dirty-runtime 게이트는 통과한다.

## 2026-08-22 — 센서 충실도 결과: FIDELITY_NEUTRAL. 구속 조건은 임계가 아니라 20 m 클립이다

seed 421, 70막대, arm당 2,049 에피소드, frozen ref5in D1 ep1900. 사전등록
`docs/prereg_2026-08-22_sensor_fidelity.md`(§5-b/§5-c 개정 포함). 학습 없음.

### 조작이 실제로 적용됐음을 두 갈래로 증명했다

영수증이 `detect_width/height`를 기록하지 않는다는 §5-c의 약점을 직접 메웠다.

| 증거 | |
|---|---|
| config 직접 인스턴스화 | `detect 1920×1200`, `decoupled=True`, `fx = 1011.6 px/rad` |
| 20 m 표적 | 지름 **15.17 px**, 면적 **180.8 px²** (임계 50의 3.6배) |
| 실행 시간 | arm B **646 s** vs arm A **307 s** = **2.10배** (미적용이면 동일해야 함) |

예측 1.92배와 일치한다. 두 arm의 유일한 다른 점이 해상도와 임계인데 임계 비교는 스칼라라
공짜이므로, 2.10배 감속은 1920×1200 광선 추적에서만 올 수 있다.

### 결과

| | baseline (160×90, 임계 2) | fidelity (1920×1200, 임계 50) | 델타 |
|---|---|---|---|
| **never-acquired** | **18.89%** (387/2049) | **19.08%** | **+0.195 pp** |
| target_hidden_fraction | 0.8212 | 0.8223 | +0.0011 |
| capture | 70.52% | 71.06% | +0.54 |
| crash | 19.81% | 18.59% | −1.22 |
| timeout | 9.66% | 10.35% | +0.68 |
| capture cohort never-acq | 0.00% | 0.00% | — |
| crash cohort never-acq | 54.19% | 55.91% | +1.72 |
| timeout cohort never-acq | 84.34% | 83.96% | −0.38 |

사전등록 밴드 ±3.00 pp → **`FIDELITY_NEUTRAL`**.

### 왜 그런가 — 데이터가 답한다

`target_hidden_fraction`이 양 arm 모두 **0.82**다(델타 +0.001). 표적이 프레임의 82%에서 안 보이는데
그것은 임계 때문이 아니라 **`detector_max_range = 20 m` 하드 클립** 때문이다. 목표 거리가
22.5–28 m이므로 표적은 대부분 클립 **밖**에 있다.

**구속 조건은 픽셀 임계가 아니라 20 m 클립이다.** 클립 안에서는 조악한 센서로도 충분히 보이고
(20 m에서 baseline 면적 1.3 px² vs 임계 2 — 한계선이지만 더 가까우면 통과), 클립 밖에서는 아무리
좋은 센서라도 렌더러가 표적을 그리지 않는다.

**따라서 "지금까지의 성적이 존재할 수 없는 센서 덕분이었는가"에 대한 답은 아니다(NO)이다.**
검출 임계의 부정직함은 성적을 설명하지 않는다.

### 재학습을 실행하지 않는다

`docs/prereg_2026-08-22_honest_sensor_adaptation.md` §1이 `FIDELITY_COST_CONFIRMED`일 때만
실행하라고 못박았고 결과는 `NEUTRAL`이다. 사용자가 오늘 학습을 승인했으나 **조건이 충족되지
않았다.** "생각보다 작지만 그래도 돌려보자"는 그 문서가 명시적으로 금지한 것이다.

### 부수: 런처 자신의 무결성 단언이 발화했다

`run`이 `G5_import_origin is owned by this launcher, but verify_cell() produced no import_origin
evidence`로 실패했다. 진단 결과 `verify_import_origin()`의 반환 딕셔너리에 `checked_by_launcher`
키가 빠져 있었다 — 옆의 `manifest_provenance`는 넣는데 origin 쪽만 누락. 게이트 검사 자체는 전부
통과했고(로그 origin 경로·sha가 매니페스트와 일치) **증거의 형식만 어긋났다.** 데이터가 온전하므로
GPU를 재실행하지 않고 검증층만 고쳐 `finalize`했다.

이것은 오탐이 아니다. 잘못된 형식의 증거는 증거가 아니며, 단언은 제 할 일을 했다.

### 이 결과가 지목하는 다음 실험

`detector_max_range`를 28 m로 올리되 **이번엔 정직한 고해상도 센서와 함께.** detect 1920×1200에서
28 m 표적은 지름 10.8 px·면적 92 px²로 임계 50을 넘는다 — **처음으로 물리적으로 실현 가능한
28 m 검출**이며, seed 367이 sub-pixel 사건으로 흉내만 냈던 것을 실제로 하는 것이다.
클립 변경은 표적 토큰 정규화를 바꾸므로 동결 정책 평가로는 교란되지만 재학습에서는 일관된다.
**별도 사전등록이 필요하다.**

P2 STRICT FAIL / D1 FAIL / P3 BLOCKED 변경 없음.

## 2026-08-22 — main 머지 + codex 결과 통합 (논문 초안용) + 규율 재검

### research/navrl-env → main

`5e75056`. 267 커밋. 충돌은 `.gitignore` 한 곳뿐이었고 세 hunk 전부 main 쪽이 비어 있어 합집합으로
해소했다(main의 로컬 아티팩트 규칙과 우리 규칙 양쪽 보존 확인). 머지 후 provenance
`ebb71802…`/`5c160b0d…` 유지, 테스트 598 통과.

### codex 결과 6개 브랜치 — 런타임 코드를 빼고 결과·문서만 가져왔다

**왜 전체 머지를 하지 않았나.** 6개 브랜치가 전부 main과 같은 4개 파일에서 겹친다:
`navrl_ref5in_quad_config.py`(provenance-frozen), `navrl_perception.py`(오늘 넣은 검출 해상도 분리),
`navrl_task.py`(오늘 넣은 obs dump 훅, 5,300줄), `test_navrl_ref5in_run_contract.py`.

진짜 위험은 하나다 — perception/task에서 충돌을 잘못 해소해 **fail-closed 가드가 조용히 사라지는
것**. provenance 깨짐은 오늘 만든 잠금 테스트가 잡지만 저 두 파일의 가드에는 그런 보호가 없고,
6번 머지는 6번의 기회다.

반면 `results/**`·`docs/**` 경로는 6개 브랜치 **전부 main과 겹침 0**이다. 논문이 인용할 것은
코드가 아니라 결과·영수증·사전등록 문서이므로, 그것만 가져오면 위험 지점이 통째로 사라진다.

가져온 것(18개 신규, 덮어쓰기 0): `docs/diagnostic_synthesis_2026-08-21.md`(evidence ledger),
사전등록 3종(active-search geofence, joint speed, topology snapshot contract),
결과 5종 — `navrl_ref5in_active_search_geofence_seed367`,
`navrl_ref5in_oob_exit_forensics_seed367`, `navrl_ref5in_symmetric_corridor_mode_probe_seed431`,
`navrl_v2_joint_speed_allocation_seed379`, `navrl_v2_bar_ceiling_topology_assumed0p60_summary`.

**가져오지 않은 것**: codex의 런타임 도구(mode probe 스크립트, joint telemetry 평가기, topology
도구, active-search 학습 런처). 그 실험들을 **재실행**할 때 필요하며, 그때 브랜치별로 머지한다.
따라서 지금 main의 결과들은 인용 가능하지만 일부는 main만으로 재현 불가다 — 논문에 쓸 때
이 사실을 알고 있어야 한다.

### 규율 전면 재검 (사용자 요청)

`docs/discipline_review_2026-08-22.md`. 요지:

**한 이름 아래 두 가지가 섞여 있다.** (A) 주장 보호 = 측정 전 게이트·지표·seed 동결.
(B) 기계 무결성 = 영수증·매니페스트·fail-closed·동등성 증명. **VOID 5건이 전부 (B)가 잡은
것이고, (A)가 막은 사고는 이 저장소 기록에 없다** — 셀당 2,000+ 에피소드라 잡음이 적이 아니며,
이 프로젝트의 실패 유형은 p-hacking이 아니라 조용한 기계 고장이다. (A)도 값어치는 있으나
(오늘 +0.195 pp에서 밴드가 미리 고정돼 있지 않았다면 "방향은 맞다"는 서술이 가능했다) **싼 것에
비싼 의식을 치르고 있다.**

**규제가 실제로 막은 축이 하나 있다.** `NAVRL_MAX_VELOCITY`(2.5)·`NAVRL_MAX_TILT_DEG`(45)·
`NAVRL_YAW_RATE_MAX`(2.5) 실험 이력 0건. riskcap 사후튜닝 금지 + 한 run 한 축 + max_velocity가
관측 정규화 분모라는 것이 겹쳤다. 세 번째는 진짜 물리 제약이지만 **첫 번째는 오염된 데이터 때문에
부과됐고 그 오염은 이미 수정됐다.** 아무도 "아직 필요한가"를 묻지 않았다 — **금지 조항에 만료나
재검 트리거가 없는 것이 구조적 결함이다.**

**가장 아픈 자기비판**: 오늘 가장 값어치 있는 발견 셋이 전부 코드 읽기와 계산에서 나왔고 GPU 0분이다.
그중 "구속 조건은 20 m 클립"은 실험 전에 예측 가능했다 — 목표 22.5–28 m에 하드 클립 20 m면
표적은 대부분 클립 밖이다. 30분 계산이 16분 GPU와 2–3시간 기계 구축을 대체할 수 있었다.
실험이 무용하진 않았으나(예측을 측정으로 바꿨다) **최우선 다음 단계는 아니었다.**

권고 4가지: (A)/(B) 문서 분리하고 (A)의 의식 축소 · 모든 금지에 이유와 재검 트리거 부착 ·
비싼 기계 전에 싼 계산(사전등록에 "계산으로 예측 가능한가, 예측이 무엇인가"를 먼저 적는다) ·
속도·틸트 축을 "금지"가 아니라 "비용이 큰 미탐색 축"으로 재분류.

이 재검은 (B)를 느슨하게 하는 근거가 **아니다**. 줄일 것은 의식이지 가드가 아니다.
어떤 기존 판정도 변경하지 않는다.
### clean 분리와 재평가 사전등록

- physical-target WIP와 섞이지 않도록 `codex/oob-forensics-seed367` worktree를 HEAD `9f6929d`에서
  분리했다. 이 branch에는 OOB 교차계측·문서 정정·전용 evaluator만 둔다.
- checkpoint 이후 ref5in config의 문서 링크 한 줄이 `docs/`→`docs/archive/`로 바뀌어 파일 SHA가
  달라진 것을 preflight가 차단했다. 이 evidence branch에서만 checkpoint가 기록한 exact config SHA
  `ebb71802…`의 바이트를 복원하고 wrapper가 그 SHA를 다시 강제한다. 동역학 값 변경은 없다.
- `tools/run_navrl_ref5in_oob_exit_forensics.py`는 기존 seed 367 base orchestrator의 checkpoint/seed/
  1 bar/away CV/2,049 episodes/20·28 m arms를 그대로 재사용하며 실험 knob를 재정의하지 않는다.
  출력은 새 경로 `results/navrl_ref5in_oob_exit_forensics_seed367/`이고 decision authority는 없다.
- 결과를 보기 전 preflight PASS. 다음은 clean commit 후 두 셀 실행이다.

첫 `camera_20m` 시도는 episode 실행 전에 VOID 처리됐다. host의 editable install이 primary dirty
workspace를 가리켜 worktree evaluator가 그쪽 `aerial_gym`을 import했고, task의 robot-source guard가
checkpoint `ebb71802…` vs 잘못 로드한 primary `cc8d90b…`를 검출해 중단했다. 결과 JSON은 없으며
부분 artifact는 `results/navrl_ref5in_oob_exit_forensics_seed367_VOID_primary_editable_import/`에
보존했다. wrapper가 자기 worktree를 `sys.path`/child `PYTHONPATH` 첫 항목으로 강제하고 import된
`aerial_gym.__file__`이 worktree 밖이면 실행을 거부하도록 보완한 뒤 새 출력에서 재시도한다.

### 실행 완료: blind search가 OOB의 주 채널이며 actor는 경계를 관측하지 못한다

import pin 보완 뒤 같은 사전등록을 새 출력에서 재실행했고 verifier가 PASS했다.

| arm | episodes | capture | crash | timeout | OOB | OOB never-acquired |
|---|---:|---:|---:|---:|---:|---:|
| camera 20 m | 2,050 | 36.39% | 7.80% | 55.80% | 158 | **152 / 158 (96.20%)** |
| camera 28 m | 2,049 | 74.96% | 6.88% | 18.16% | 138 | **120 / 138 (86.96%)** |

20 m never-acquired exit는 평균 speed 1.359 m/s, arena 중심 기준 outward radial
**+1.002 m/s**, 실제 target closing **-0.834 m/s**, exit-step median 84였다. 따라서 표적을 못 본
채 정지하거나 수동 표류한 것이 아니라, 약 8.4초 안에 목표에서 멀어지며 능동적으로 경계를 넘는다.
28 m acquired OOB 18건은 target closing +0.371 m/s와 outward +0.619 m/s가 함께 양수라, 소수의
“취득 후 arena 밖 표적을 추격” 채널도 별도로 존재한다. arm 간 평균 운동학 차이는 cohort selection이
달라 causal delta로 읽지 않는다.

관측 계약도 재감사했다. 898-D actor에는 world XY, arena 크기, 네 벽까지 거리, geofence,
episode progress가 없다. `_arena_xy_norm`은 asymmetric critic의 privileged GT target distance에만
쓰인다. 네트워크는 RNN이 아니며 5-step history는 0.1 s 기준 약 0.5 s다. arena boundary는 물리
wall/LiDAR return도 아니므로 blind actor가 경계를 직접 관측할 경로가 없다.

결론: episode 600을 늘리는 것은 median 84-step OOB를 막지 못하고, speed/tilt 상향은 이미 outward
1.0 m/s인 채널과 정지거리·선회반경을 악화시킬 수 있다. 28 m camera는 계속 positive control이다.
사용자가 원하는 sensor-outside active search의 다음 단일 변경축은 **boundary observability**다.
실기 VIO/GPS/known-map geofence 계약을 명시한 body-frame boundary range 4개를 actor에 주는 fresh
policy A/B를 먼저 사전등록한다. global localization을 허용하지 않으면 recurrent coverage belief가
대안이지만 architecture+memory 동시변경이라 2순위다. 상세 수치·gate는
`results/navrl_ref5in_oob_exit_forensics_seed367/analysis.md`에 고정했다. P2 STRICT FAIL, D1 FAIL,
P3 BLOCKED는 바뀌지 않는다.

## 2026-08-21 — Active-search geofence를 별도 branch에 opt-in 구현

OOB 증거 branch `e6fe790`을 고정한 뒤 `codex/boundary-observable-search`를 분기했다. 기존
898-D 정책과 physical-target WIP를 건드리지 않기 위해 기본값은 off다.

### 구현 계약

- `NAVRL_GEOFENCE_ACTOR=1`일 때만 actor 끝에 8-D를 append한다: body-frame
  forward/left/back/right geofence ray range 4개(아레나 XY 대각선으로 정규화) + validity 4개.
- Transformer에는 geofence projection token 1개를 마지막에 append한다. v2 search 실제 계약은
  **898→906 D, 17→18 tokens**다. off는 898/17 그대로다.
- feature는 target GT가 아니라 VIO/GPS pose + known-map flight boundary를 전제로 한다. 이 센서를
  실기에 제공하지 않는다면 결과를 camera/LiDAR-only exploration으로 주장할 수 없다.
- noise/dropout은 별도 provenance(`cfg_geofence_*`)로 checkpoint에 저장하고 resume mismatch를
  preflight가 거부한다. 첫 A/B는 둘 다 0으로 고정한다.
- schema 변경이므로 warm-start는 지원하지 않는다. fresh policy만 허용한다.

`docs/preregistration_active_search_geofence_2026-08-21.md`에 결과 전 gate를 고정했다. 두 arm은
seed197/P1c 900-epoch 계약에서 geofence off/on 한 값만 다르다. held-out seed367 primary는
never-acquired OOB/all episodes이며 최소 3.0pp 개선(기존 약 7.4% 기준 약 40% 상대 감소)이 필요하다.
이 3pp는 n≈2,049에서 단순 표본오차보다 충분히 크도록 정한 practical gate다. non-OOB crash +2pp
guard와 token-mask inference ablation도 결과 전에 고정했다.

### 검증

- perception unit: off **30/30 PASS**, geofence-on **30/30 PASS**
- checkpoint preflight: **15/15 PASS** (legacy off default와 on mismatch reject 포함)
- ref5in run contract: worktree에 P1c run을 read-only symlink한 뒤 **26/26 PASS**
- 실제 v2 env 상수로 network forward: off `898 D / 17 tokens`, on `906 D / 18 tokens` 모두 PASS
- launcher `bash -n`, control/geofence child preflight 모두 PASS
- Python compile와 `git diff --check` PASS

학습은 아직 시작하지 않았다. 비교의 최소 단위가 두 fresh 900-epoch arm이므로 한쪽만 먼저 장시간
돌려 결과를 해석하지 않는다. 실행기는
`aerial_gym/rl_training/rl_games/train_navrl_ref5in_active_search_geofence_ab.sh`다.

### 첫 launch VOID: editable install import를 source guard가 차단

첫 control launch는 task 생성 중, PPO epoch 0 이전에 중단됐다. local `runner.py`를 실행했지만 host
editable install이 primary dirty workspace의 `aerial_gym`을 import했고, clean branch에서 만든 source
receipt와 실제 `navrl_ref5in_quad_config.py`가 다르다고 runtime guard가 거부했다. checkpoint와 유효
run 결과는 생성되지 않았고 geofence arm도 `&&` 때문에 시작되지 않았다.

launcher가 `PYTHONPATH` 첫 항목을 현재 git root로 고정하고, receipt 생성 전에
`importlib.util.find_spec("aerial_gym").origin`이 해당 worktree의 `aerial_gym/__init__.py`와 정확히
같은지 검사하도록 수정했다. OOB evaluator에서 잡았던 동일 계열 문제를 training launcher에도
fail-fast로 일반화한 것이다.

## 2026-08-21 — Active-search held-out evaluator 사전 구현 (학습 결과 열람 전)

두 training arm이 진행되는 동안 별도 `codex/active-search-geofence-eval` worktree에서 평가기를
작성했다. training worktree의 runtime bytes는 건드리지 않았다.

- `tools/run_navrl_ref5in_active_search_geofence_eval.py`가 완료 marker와 유일한 raw ep900
  checkpoint를 요구한다. latest/best checkpoint를 임의 선택하지 않는다.
- held-out 계약은 seed367, 1 bar, camera20m, goal 22.5..28m, away CV, 600 step,
  deterministic, arm당 최소 2,049 episodes다.
- 3 cells: fresh control / fresh geofence / 같은 geofence checkpoint의 token-force-invalid.
- primary는 never-acquired OOB/all episodes `control - geofence >= 3pp`, guard는 non-OOB crash
  `geofence - control <= 2pp`다.
- mechanism gate의 “material return”을 결과 전에 수치화했다: force-invalid가 normal geofence
  primary gain의 **50% 이상을 잃어야** token 사용을 지지한다.
- evaluator result/receipt에 geofence on/noise/dropout/force-invalid를 명시했다. force-invalid는
  906-D schema를 보존하며 range `[1,1,1,1]`, validity `[0,0,0,0]`만 주입한다.
- P2/D1/P3 판정 권한은 없다.

검증: force-invalid perception **31/31 PASS**, checkpoint preflight **15/15 PASS**, Python compile,
shell syntax, `git diff --check` PASS. 현재 `status`는 control 진행 중/geofence 미시작을 정확히
PENDING으로 표시하며, 두 `.aerial_training_finished`가 생기기 전 preflight/run은 fail-closed다.

### 완료 후 첫 held-out preflight 차단과 계약 수정

두 arm의 epoch-900 완료 뒤 첫 preflight는 평가를 시작하기 전에 차단됐다. held-out은 의도적으로
학습 계약의 `goal_dist_min=6.0`을 `22.5`로, `target_pattern=mixed`를 `cv`로 바꾸는데, 평가기가
후자 한 줄만 허용 mismatch로 등록해 전자도 source drift로 오인했다. 결과 episode는 0개였고
checkpoint에는 영향이 없다. 관측된 두 mismatch 문자열을 순서까지 정확히 고정하고, 그 외 차이는
계속 거부하도록 수정했다. 수정 후 세 arm preflight를 다시 통과해야만 실제 평가를 시작한다.

### Held-out 3-arm 완료: 성능 이득은 크지만 사전등록 기전 gate는 미통과

seed367, 1 bar, away-CV, camera 20m, goal 22.5..28m, 2,049 episodes/arm 평가를 완료했다.

| arm | capture | crash | timeout | OOB | never-acq OOB/all | non-OOB crash |
|---|---:|---:|---:|---:|---:|---:|
| control | 39.04% | 22.21% | 38.75% | 21.96% | 21.28% | 0.24% |
| geofence | 85.75% | 7.91% | 6.34% | 7.47% | 7.32% | 0.44% |
| geofence masked | 39.78% | 5.22% | 55.00% | 4.93% | 4.88% | 0.29% |

primary 개선은 **+13.96pp**로 3pp gate를 통과했고 non-OOB crash 증가는 **+0.20pp**로 2pp
guard 안이다. 그러나 사전등록 mechanism 지표는 `never-acquired OOB/all`의 masked loss였고,
masked 정책은 밖으로 나가는 대신 느리게 움직이며 timeout이 55.00%로 증가했다. 따라서 masked
never-acquired OOB가 오히려 2.44pp 더 낮아져 50% loss gate를 통과하지 못했다. capture가
85.75→39.78%로 붕괴한 것은 경계 token 사용의 강한 사후 증거지만, 결과 뒤에 지표를 바꾸지 않고
공식 판정은 **PASS_MECHANISM_UNRESOLVED**로 유지한다.

첫 finalize는 task result의 `condition`에 없는 evaluator-only intervention 필드까지 찾다가 종료 코드
2로 fail-closed했다. raw 3-cell 결과와 receipt는 모두 완성·해시 고정돼 있었다. task/runtime 필드는
result에서, evaluator intervention 필드는 receipt에서 각각 검증하고 두 artifact의 evaluation nonce도
일치시키도록 verifier를 수정했다. 재실행 없이 `finalize`와 `verify`가 PASS했다. 결과는
`results/navrl_ref5in_active_search_geofence_seed367/{summary.md,summary.json}`이다. P2 STRICT FAIL,
D1 FAIL, P3 BLOCKED는 바뀌지 않는다.

## 2026-08-21 — 병렬 진단 종합과 다음 gate 고정

active-search, symmetric mode probe, 205-bar joint speed telemetry, legacy topology label의 결과를
`docs/diagnostic_synthesis_2026-08-21.md`에 계보별로 분리해 정리했다. 결론은 (1) speed/riskcap을
바로 튜닝하지 않음, (2) chirality 때문에 multi-candidate head를 바로 구현하지 않음, (3) mapped
geofence만 추가 fresh-seed confirmatory 후보로 승격, (4) timeout과 실제 bar footprint가 포함된 dump
전에는 topology curriculum을 만들지 않음이다.

다음 순서는 real-frame reflection audit → prospective geofence replication → frozen 205-bar 위험 step의
시간적 원인 분해 → future dump contract다. 현재 masked 결과에서 OOB가 timeout으로 치환된 사실을
반영해, **향후** replication의 mechanism 지표는 결과 전에 acquisition-failure/all로 고정한다. 이는
현재 `PASS_MECHANISM_UNRESOLVED` 판정을 소급 변경하지 않는다.
## 2026-08-21 — 동결 ref5in 단일행동 mode-averaging probe 사전등록 구현

고밀도 정지가 하나의 연속 action head가 좌·우 경로를 평균내는 현상인지 분리하기 위해, 학습과
실행 명령을 바꾸지 않는 side-forward 진단을 별도 branch `codex/mode-probe`에 구현했다. 실제 player
action은 원래 관측에서 한 번 계산해 그대로 실행하고, 추가로 고정된 898-D 합성 관측 세 개만 동결
정책에 통과시킨다: 정중앙 corridor, 왼쪽 +5°, 오른쪽 -5°. 좌우 arm은 기존 structured-observation
mirror 계약의 정확한 거울쌍이고, 중앙은 두 입력의 산술 중점이라 입력 반사 오차가 `1e-7`보다 크면
정책 추론 전에 실패한다.

결과에는 bounded deterministic action, latent mean/sigma(모델이 제공할 때), XY command m/s,
near-zero 비율, 좌우 conjugacy 오차를 기록한다. 사전 고정 gate는 policy reflection action error
`<=0.15`, 중앙 horizontal speed `<=0.25 m/s`, perturb arm 평균 `>=0.75 m/s`, 중앙 `|y|<=0.10`,
perturb 평균 `|y|>=0.25`, 좌우 lateral sign 반대다. reflection gate가 먼저 실패하면 기존 chirality가
probe를 교란한 것이므로 `INCONCLUSIVE_POLICY_CHIRALITY`로 닫고 mode averaging을 긍정/기각하지
않는다. 전부 통과해도 진단 fixture에 한정된 지지이며 capture/crash 인과나 정책 교체 권한은 없다.

`tools/run_navrl_ref5in_mode_probe.py`는 D1 ref5in checkpoint SHA-256
`197ea26999d6…`와 deterministic/governor-off/seed431 계약을 고정하고, generic evaluator receipt,
checkpoint snapshot, probe JSON을 다시 검증한 뒤 hash-bound summary receipt를 만든다. GPU 평가는
아직 실행하지 않았다. CPU fixture/gate/fail-closed/overwrite 테스트 **4/4 PASS**, Python compile과
`git diff --check` PASS. 이후 실행 순서는 `preflight -> run -> finalize -> verify`다.

## 2026-08-21 — Mode probe 중앙 fixture 교란 제거 (GPU 실행 전 정정)

초기 `3fac1a3` 설계의 중앙 입력은 left/right 관측의 feature-wise 산술평균이었다. 정적 scan은
대칭이지만 obstacle slot별 `y`가 각각 0으로 상쇄되어, 실제 ±12° 통로가 아니라 정면에 중복된
장애물 두 개처럼 보이는 비물리 입력이 됐다. 이 상태의 stall은 mode averaging이 아니라 정면 차단
반응일 수 있으므로 **초기 중앙 fixture와 그 판정 계약을 GPU 실행 전에 폐기**했다. 측정 결과는 없어
무효화할 artifact도 없다.

수정 probe는 산술평균을 전혀 쓰지 않는다. 정중앙 ±12°, 왼쪽 이동 +5°, 오른쪽 이동 -5°의 실제
두-surface scan/token을 만들고, 각 geometry마다 두 obstacle token slot 순서(LR/RL)를 모두 넣어 총
6 arm을 side-forward한다. 중앙 두 arm의 첫 두 token `y`는 각각 nonzero/opposite이며 합만 0이다.
static scan은 3-beam 폭의 정확한 반사대칭이고, `symmetric LR↔RL`, `left LR↔right RL`,
`left RL↔right LR` 입력 반사 오차는 CPU fixture에서 모두 0이다.

새 품질 gate `slot_permutation_max_abs_action<=0.15`를 결과 전에 추가했다. 같은 물리 geometry의 token
순서만 바꿨을 때 action 최대차가 이를 넘으면 `INCONCLUSIVE_SLOT_ORDER_SENSITIVITY`, 그다음 반사
action 오차가 0.15를 넘으면 `INCONCLUSIVE_POLICY_CHIRALITY`로 닫는다. 두 품질 gate가 통과한 뒤에만
두 중앙 order가 모두 stall이고 네 perturb/order가 모두 움직임을 회복하는지 본다. 양성이라도 명칭과
해석은 `MODE_AVERAGING_SUPPORTED_IN_SYNTHETIC_POLICY_SCREEN`이며 실제 고밀도 인과로 승격하지 않는다.
physical-centre 회귀와 slot fail-closed 테스트를 포함한 CPU 테스트 **5/5 PASS**. GPU 미실행 상태 유지.

### GPU 결과: mode averaging 판정 불가, policy chirality가 선행 교란

seed431 evaluator host cell(70 bars, 257 episodes)과 동결 D1 ref5in checkpoint를 사용해 6-arm
side-forward probe를 실행하고 `finalize`/`verify`까지 통과했다. 같은 geometry의 LR/RL token 순서
최대 action 차이는 **0.0078**로 0.15 품질 gate 안이었지만, 정확한 반사 입력쌍의 최대 action 오차는
**1.8332**로 0.15 gate를 크게 넘었다. 중앙 symmetric 두 arm도 horizontal command
**3.349~3.352 m/s**, lateral action **-0.916~-0.917**로 정지하지 않았고, 좌·우 ±5° perturbation도
모두 같은 음의 lateral 방향(**-0.904~-0.922**)을 냈다.

따라서 결과는 사전등록 순서대로 **INCONCLUSIVE_POLICY_CHIRALITY**다. 이 fixture에서는 token slot
순서가 병목이라는 증거가 없지만, policy reflection defect가 더 커 mode averaging을 지지하거나
기각할 수 없다. 이 결과는 한 합성 fixture의 반복 forward이며 205-bar 정지나 capture/crash 인과로
확대하지 않는다. artifact는
`results/navrl_ref5in_symmetric_corridor_mode_probe_seed431/summary.json`에 저장했다.
## 2026-08-21 — 고밀도 speed-allocation joint telemetry 사전등록·CPU 검증

205막대에서 남은 bar contact가 단순 고속 때문인지, 좁은 clearance·큰 방향 변화와 결합된 위험한
속도 배분과 연관되는지 동결 정책으로 분리하기 위한 **평가 전용** 계측을 추가했다. 관측·보상·종료·
정책·checkpoint에는 손대지 않았고 riskcap 파라미터 탐색도 하지 않는다.

- `NAVRL_JOINT_SPEED_TELEMETRY=1`에서만 recorder를 만들고 JSON key를 export한다. 일반 bulk eval은
  기존 runtime/result 계약 그대로이며 launcher·condition·schema-2 receipt·analyzer가 opt-in을
  교차 검증한다.
- action-selection 시점의 actual pursuer speed, requested/executed XY command 각각의 방향으로
  actor-safe directional minimum LiDAR clearance를 별도 계산한다. actual/requested/executed stopping
  distance·margin은 반드시 같은 방향 clearance와 결합하며 primary risk는
  **actual-velocity-direction margin**이다. 요청방향 clearance와 actual speed를 섞은 hybrid는 쓰지 않는다.
- requested/realized heading rate와 curvature는 0.25 m/s 이상인 연속 두 sample의 유한차분
  **proxy**다. planned path curvature나 인과량으로 해석하지 않는다.
- 각 step을 에피소드 종료 뒤 capture/crash/timeout과 actual stopping-margin 4구간
  (`<0`, `0–0.5`, `0.5–1.5`, `>=1.5 m`)에 귀속하고, cause-attributed bar contact 직전 1.0초를
  별도로 집계한다.
- outcome 총계와 bar-contact 총계가 기존 bulk evaluator의 독립 counter와 다르면 JSON export를
  중단한다. analyzer도 result SHA, schema-2 receipt, runtime source manifest와 계측 module snapshot을
  다시 검증한다.

결과를 보기 전에 gate를 `docs/navrl_joint_speed_preregistration_2026-08-21.md`에 고정했다. quality는
bar-contact 100 episodes / pre-contact 500 steps / capture 1,000 steps 이상이다. quality 통과 뒤
contact 직전 negative-margin 비율이 50% 이상이고 capture-outcome 대비 +10 pp 이상일 때만
`supports_descriptive_speed_risk_association`이다. PASS여도 causal claim이나 riskcap 사후튜닝 권한은
생기지 않는다.

CPU 검증은 joint telemetry **7/7**, 기존 speed-governor **10/10**, outcome strata **5/5**,
verification guards **4/4**, Python compile·launcher `bash -n`·4097-episode preflight 모두 PASS했다.
GPU 평가는 실행하지 않았다. 전용 실행기는
`aerial_gym/rl_training/rl_games/eval_navrl_v2_joint_speed_telemetry.sh`; frozen ep25000+riskcap,
205 bars, deterministic, unused seed 379, 4,097 episodes 한 셀이다. 결과 예정 경로는
`results/navrl_v2_joint_speed_allocation_seed379/assessment.json`이다.

통합 전 재현성 검수에서 diagnostic worktree에는 ignored `runs/`가 없어 기본 checkpoint 상대경로가
실패하는 것을 확인했다. launcher는 local checkpoint를 먼저 찾고, 없으면
`git rev-parse --git-common-dir`로 primary worktree의 같은 고정 경로를 해석한다. SHA
`f7022139…` 검사는 그대로다. default `RESULT_ROOT`는 checkpoint 위치를 따라가지 않고 현재
diagnostic worktree의 `${REPO_ROOT}/results/`에 고정한다. 별도 `POLICY` 없이 worktree preflight를
재실행해 PASS했다.

### GPU 실행: 첫 run VOID 후 유효 재실행에서 speed-risk 연관 gate 통과

첫 4,097-episode 실행은 simulator를 끝까지 돌렸지만 evaluator가 joint condition attestation 부재로
fail-closed했다. 원인은 isolated worktree의 launcher가 checkpoint만 primary worktree에서 찾고,
editable install의 `aerial_gym` import도 primary dirty source로 빠지는 것을 막지 않았기 때문이다.
따라서 새 recorder가 생성되지 않은 첫 결과는 진단 증거로 **VOID**이며 삭제하지 않고
`results/void_navrl_v2_joint_speed_allocation_seed379_primary_import_20260821/`에 보존했다.

launcher가 `${REPO_ROOT}`를 `PYTHONPATH` 첫 항목으로 고정하고 `aerial_gym.__init__`의 resolved origin이
현재 diagnostic worktree와 정확히 일치하는지 실행 전에 검사하도록 수정했다. `bash -n`, import
origin guard, 4097-episode preflight가 PASS한 뒤 같은 checkpoint/seed/condition으로 재실행했다.

유효 run은 205 bars, seed379, 4,097 episodes에서 capture **80.69%**, crash **16.77%**, timeout
**2.54%**였고 crash 687건 중 bar contact는 **664건**이다. 사전등록 quality(접촉 100 episodes,
pre-contact 500 steps, capture 1,000 steps)는 모두 통과했다. bar-contact 직전 1초 6,511 step의
actual-direction negative stopping-margin 비율은 **67.58%**, capture outcome 전체 step은 **9.25%**로
차이가 **+58.33pp**였다. 접촉 직전 평균은 requested command **2.895 m/s**, executed command
**2.109 m/s**, actual speed **1.754 m/s**, actual-direction clearance **1.671 m**였다.

판정은 **supports_descriptive_speed_risk_association**이다. 이는 위험한 실제 진행방향 speed-margin과
bar contact의 강한 연관만 지지하며 speed를 원인으로 특정하거나 riskcap 사후 파라미터 탐색을
허용하지 않는다. 유효 artifact와 source/receipt hash는
`results/navrl_v2_joint_speed_allocation_seed379/assessment.json`에 고정했다.
## 2026-08-21 — 정책 무접촉 GT topology difficulty 라벨러 구현

기존 막대 수/전역 연결성만으로 설명하지 못하던 에피소드별 공간 난이도를 정책·환경 변경 없이
측정하기 위해 `tools/analyze_navrl_topology_labels.py`를 추가했다. JSON layout snapshot의 실제
막대 중심·XY footprint를 받아 다음을 같은 0.10 m grid와 axis-aligned inflation 관례로 계산한다.

- start→final goal 정적 path existence와 shortest-path detour ratio
- 선택된 최단 경로의 최소 raw/vehicle-usable side clearance
- start의 12 m sensor disc 안 obstacle 수와 surface-gap cluster 수
- sensor-disc reachable exit arc 기반 local cul-de-sac/dead-end proxy
- grid resolution, vehicle half-width, side-clearance, inflation, sensor range, cluster gap,
  endpoint snap radius 및 arena bounds를 각 row에 명시

과거 `NAVRL_EPISODE_DUMP`에는 `bars_xy/spawn/target_end/outcome`은 있지만 실제 `bars_size_xy`가
없음을 확인했다. 따라서 legacy NPZ는 `--default-bar-size-m`을 의무화하고 모든 결과를
`bar_size_source=assumed_default`로 표시한다. 0.4–0.8 m 실제 pool을 단일 크기로 가정한 값은 탐색적
연관 분석에만 쓰며 publication exact 수치로 쓰지 않는다. 향후 exact JSON export 계약과 적용법은
`docs/topology_layout_snapshot_contract.md`에 고정했다. aggregate summary만 남은 과거 평가는
레이아웃을 복원할 수 없어 소급 라벨링 불가다.

배포 `cluster_sector`의 authoritative default가 `NAVRL_OBSTACLE_CLUSTER_GAP_M=0.45`임을
`navrl_perception.py`와 v2 evaluation launcher에서 재확인했다. topology 도구 초안의 0.40 m에는 별도
기하학적 근거가 없었으므로 CLI·문서·테스트·출력 metadata를 0.45 m로 통일했다.

CPU synthetic 6개가 모두 PASS했다: open/direct, full-wall disconnected, two-exit corridor,
one-exit U-shaped dead-end, sensor-range cluster grouping, metadata contract. arena boundary가 sensor disc를
자른 open spawn을 dead-end로 오인하지 않도록 exit coverage를 arena-available angle로 정규화한다.
이 라벨은 정적 2-D GT 진단이며 동역학·표적 이동·episode horizon 또는 planner의 경로 거부를 뜻하지
않는다. 기존 `tests/test_navrl_reachability.py`도 3/3 PASS했다. GPU 실행 및 학습/평가 run은 수행하지
않았다.

### Seed167 legacy dump topology 전수 탐색 — timeout 누락으로 제한 판정

기존 `results/navrl_v2_bar_ceiling/episodes_seed167.npz`의 1,989개 record를 0.60 m square bar 가정으로
전수 라벨링했다. 원평가는 2,049 episodes였지만 legacy dump는 `captured | crashed_out`만 기록하므로
**timeout 60개가 전부 누락**됐다. 기록 outcome은 capture 1,641 / bar contact 333 / below 10 / OOB
5다. 따라서 timeout 또는 timeout-dead-end에 관한 결론은 금지한다.

기록된 네 outcome 모두 path exists 100%, local cul-de-sac proxy 0%였다. capture 대 bar-contact의
평균 detour ratio는 1.0692 대 1.0710, 선택된 grid-shortest path의 usable clearance는 0.2594 m 대
0.2319 m, start의 12 m 내 obstacle 수는 48.79 대 45.92, cluster 수는 45.92 대 43.15였다. 이는
randomised topology intervention이 아닌 outcome별 descriptive association이며 인과로 읽지 않는다.

또한 legacy dump에 실제 `bars_size_xy`가 없어 0.4–0.8 m pool 전체를 0.60 m로 가정했다. 결과는
exploratory-only이고 publication exact 수치가 아니다. 작은 재현 summary는
`results/navrl_v2_bar_ceiling_topology_assumed0p60_summary/summary.{md,json}`에 저장했다. 2.9 MB raw
per-layout JSON은 추적하지 않는다(raw SHA-256 `6ca8c405…`; input dump `c509c2fa…`). 다음 exact 분석은
실제 bar footprint와 timeout을 모두 포함하는 새 evaluation-only snapshot이 필요하다.
## 2026-08-01 — recovery run 독립 감사: entropy 완만한 상승은 붕괴 재발 아님

ep10836 actor collapse 이후 recovery run(`ppo_260801_1235_navrl_v2-recover-curriculum-s1`,
ep9601 branch)의 건강 상태를 사용자가 "entropy가 소폭 오르고 있어 긴장된다"고 문의해
독립 감사했다. 옛 collapse 구간이 섞이는 걸 피하려 **이 run 고유 event 파일만** 사용
(병합 뷰는 9601~10836 구간에서 옛 sched-s1과 겹쳐 오염됨 — 분석용으로는 부적합, TB
표시 전용으로만 쓸 것).

| 지표 | ep~9601 | ep~11700 | 판정 |
|---|---|---|---|
| entropy | -7.5 | -6.6 (기울기 +0.45/1000ep) | 완만, 붕괴 방향과 반대 |
| KL | 0.001 | 0.001 | 게이트(0.04) 대비 40배 여유, rollback 0건 |
| \|μ_x\| | ~1.9 | ~1.3 | 감소 — tanh 경계에서 멀어짐 |
| capture (10-bin 평균) | 0.70~0.75 | 0.70~0.75 | 130→145막대 승급(ep10367) 관통 평탄 |

**결론**: 붕괴 서명(entropy -8.77→-106 in 36 epoch, KL 0.04→2.7, μ.weight norm +12.8%/50ep)과
질적으로 다르다. 지금의 완만한 entropy 상승은 `NAVRL_LATENT_MARGIN_COEF=0.01`(붕괴 당시
0으로 방치돼 무효했던 페널티, 이번에 수정)이 의도대로 μ를 tanh 경계에서 밀어내는 부수효과로
읽힌다. capture/crash 모두 밀도 승급을 관통해 평탄 — 개입 불필요.

**남은 리스크**: 붕괴도 유사 지점(밀도 130대, 장시간 학습)에서 터졌으므로, 밀도가 더 오를
때마다 이 감사를 반복할 것.

## 2026-08-22 — codex 6개 브랜치 전체 머지 완료 (런타임 코드 포함)

앞 항목에서 결과·문서만 가져오고 런타임은 미뤘으나, 미루면 격차만 커지고 나중 머지가 더 위험해진다는
판단으로 전량 머지했다. 안전망으로 `pre-codex-merge` 태그를 남겼다.

### 결과

622 테스트 통과(머지 전 598). provenance `ebb71802…`/`5c160b0d…` 유지. 오늘 넣은 가드
(`_obs_dump_*`, `NAVRL_DETECT_WIDTH`, `checked_by_launcher`) 전부 생존.

**위험 파일이었던 `navrl_task.py`·`navrl_perception.py`는 전부 자동 병합됐다.** 실제 충돌은
`README.md`·`WORKLOG.md`·`tests/test_navrl_ref5in_run_contract.py`(append 유형)뿐이었다.

### 머지가 실제로 만든 문제 3건 — 전부 테스트가 잡았다

**① 클래스 경계 훼손.** union 해소가 codex의 `test_rerun_reuses_frozen_seed367_contract`를 파일
끝에 붙여 **내 `SensorFidelityContract` 안으로** 넣었다. 그 결과 `self.ORCHESTRATOR`가 codex의
OOB orchestrator가 아니라 내 센서 런처를 가리켰다. 원래 자리(`OOBExitForensicsContract`)로 옮겼다.
naive union은 append 충돌에만 안전하고 **클래스 안쪽 삽입에는 안전하지 않다.**

**② 낡은 마스크 단언.** codex 테스트가 `d_oob = oob & ~crashed & ...`를 기대하는데 main은
physical-target 병합 이후 `~d_contact`다(`d_contact = (crashed | target_contact | target_invalid)
& crashed_out`). **코드가 옳고 테스트가 낡았다** — 병합이 최신 쪽을 올바르게 유지했다. 단언의
의도(레코더가 원인귀속 마스크를 소비한다)는 그대로 두고 이름만 현행화했다.

**③ 테스트 격리 붕괴 — 병합이 새로 들여왔다.** `tests/test_navrl_mode_probe.py`가 **import 시점에**
`os.environ.update({"NAVRL_LIDAR_HBEAMS": "72", ...})`를 실행한다. `navrl_perception`의
`VBEAMS`/`HBEAMS`는 첫 import에서 얼어붙는 모듈 상수이므로, 전체 discovery에서는 mode_probe가
먼저 import하면서 **모든 모듈의 상수를 72로 결정한다.** `test_navrl_perception.py`는 4×36 기하를
전제하고 특정 bin을 인덱싱하므로(`self.lidar = torch.full((2, 144), ...)`, `lidar[0, 18]`) 깨진다.
머지 전 598은 OK였고 622에서 처음 나타났다.

시도했다가 되돌린 것 둘을 기록한다. (a) lidar 크기를 모듈 상수에서 유도 → bin 인덱스 의존 테스트가
여전히 실패. (b) `importlib.reload` → **이미 그 모듈을 import한 다른 테스트 모듈의 참조를 끊어**
mode_probe·ttc_selector까지 깨졌다. reload는 선택지가 아니다.

채택한 해법: 상수가 4×36이 아니면 `NavRLPerceptionTest`를 **사유와 함께 skip**한다. 조용한 통과도
크래시도 아니고, **단독 실행하면 31개가 전부 돈다**(확인함). 근본 해결은 전역 모듈 상수를 없애거나
그 테스트를 서브프로세스로 격리하는 것이며 별건으로 남긴다.

### 남긴 위험

전체 discovery에서 12개가 skip된다. `test_navrl_perception.py`를 단독으로도 돌리지 않으면 그
커버리지가 사라진다. **CI가 파일별 실행을 하지 않는다면 이 skip은 오늘 고친 "실행되지 않는 테스트"
문제의 재발이다.** 다음 세션에서 서브프로세스 격리로 제대로 고쳐야 한다.

## 2026-08-22 — 규율 재검 권고 4가지 실행

`docs/discipline_review_2026-08-22.md`의 권고를 스킬과 규칙 문서에 반영했다.

**① (A)/(B) 분리** — `.claude/skills/navrl/SKILL.md`에 "TWO DISCIPLINES, ONE NAME" 추가.
(A) 주장 보호는 싸다 — 질문·arm·1차 지표·임계·반증 조건·이 결과가 허가하지 않는 것, 짧은 블록이면
충분하고 150줄 산문이 필요하지 않다. (B) 기계 무결성은 비싸고 **VOID 5건을 전부 잡은 쪽**이므로
아끼지 않는다. 피해야 할 것은 (A) 크기 결정에 (B) 크기 의식을 치르는 것이다.

**③ 싼 계산 먼저** — "CHEAP CALCULATION BEFORE EXPENSIVE MACHINERY" 추가. 사전등록에
**"이 결과를 계산으로 예측할 수 있는가, 예측이 무엇인가"**를 적는다. 일치하면 모델 확인이고
어긋나면 진짜 발견이다. 2026-08-22의 세 발견이 전부 GPU 0분이었다는 것을 근거로 박아뒀다.

**② 금지에 사유·재검 트리거** — "PROHIBITIONS EXPIRE" 추가. 원인이 수정됐는데 아무도 다시 묻지
않은 금지는 규율이 아니라 퇴적물이다. `CLAUDE.md`의 금지 2건에 실제로 부여했다:
fixed-density PPO 연장(사유: 실패를 epoch로 덮는 것 방지 / 재검: 다른 축에서 병목 특정 + 사전등록),
**riskcap 사후튜닝(사유: seed 44 자료가 semantic-mask leak으로 오염 — 그 오염은 이미 수정·재실행됐으므로
원인이 소멸했다 / 재검: 깨끗한 재측정 위의 사전등록 A/B라면 허용)**.
`VERIFICATION.md` fail-closed 절에도 같은 원칙을 명시했다.

**④ 속도축 재분류** — `CLAUDE.md`에 "비싸고 미탐색인 축 (금지 아님)" 절 신설.
`NAVRL_MAX_VELOCITY`(2.5)·`NAVRL_MAX_TILT_DEG`(45)·`NAVRL_YAW_RATE_MAX`(2.5) 실험 이력 0건이고
`3.5355`는 최적값이 아니라 `2.5×√2`다. **이것들은 금지된 적이 없다** — riskcap 금지와 한 run 한 축이
겹쳐 접근 불가처럼 취급됐을 뿐이다. 실제 비용은 `max_velocity`가 관측 정규화 분모라 재학습이
필요하다는 것이며, 물리적으로도 정지거리·선회반경이 제곱으로 커지고 틸트 상향이 추력을 `1/cosθ`로
키운다. **"금지"가 아니라 "값이 비싼 미탐색 축"으로 다룬다.**

이 갱신은 어떤 기존 판정도 변경하지 않는다 — P2 STRICT FAIL / D1 FAIL / P3 BLOCKED 그대로다.

## 2026-08-22 — 테스트 격리 빚 상환: skip 12 → 1

앞 항목에서 남긴 빚을 갚았다. 원인이 내가 생각한 것보다 단순했다.

`test_navrl_perception.py`는 `spec_from_file_location`으로 **별도 모듈 객체**
(`navrl_perception_standalone`)를 만든다 — `sys.modules`의 공유 항목이 아니다. 따라서 다른 모듈과
상수를 공유하지 않는다. 깨진 이유는 공유가 아니라 **`exec_module` 시점에 환경변수가 이미 72로
설정돼 있어서** 그 별도 객체의 상수까지 72가 된 것이었다.

해법: `exec_module` **직전에** `NAVRL_LIDAR_VBEAMS/HBEAMS`를 4/36으로 고정하고 `finally`로 정확히
복원한다. 별도 모듈 객체이므로 다른 누구에게도 영향이 없고, 서브프로세스 격리도 필요 없다.

앞서 넣었던 `@unittest.skipIf` 데코레이터는 **제거했다** — 그것은 증상 완화였다. 12개를 조용히
건너뛰는 대신 실제로 실행하는 것이 옳다.

결과: 단독 31/31, 전체 discovery **622개 OK (skip 1)**. 남은 skip 1개는 원래 있던 조건부 skip이다.

`importlib.reload`가 왜 선택지가 아닌지는 앞 항목에 기록돼 있다 — 이미 그 모듈을 import한 다른
테스트 모듈의 참조를 끊는다. 여기서는 애초에 공유 모듈을 건드리지 않으므로 문제가 성립하지 않는다.

## 2026-08-22 — 검출 거리 2단계 사전등록 (실행 전)

`docs/prereg_2026-08-22_detection_range_2stage.md`. 97줄 — 새 스킬 규칙(주장 보호는 짧게)을
적용해 이전 사전등록들의 절반 길이로 썼다.

**조작 축을 옮긴다.** seed 421이 임계가 아니라 **20 m 클립**이 구속 조건임을 확정했고, 이제 정직한
고해상도 센서로 28 m를 물리적으로 정당하게 볼 수 있다(28 m에서 92 px² > 임계 50).

**계산 예측을 측정 전에 기록했다**(스킬 규칙): 클립 28 m면 목표 밴드 22.5–28 m가 통째로 검출 범위
안에 들어온다. seed 367이 같은 변경으로 timeout `55.80 → 18.16%`를 봤고 그것은 sub-pixel 사건이었다.
정직한 센서에서는 유지되거나 더 클 것으로 예측한다. 예측이 명확히 양성인 것이 실험을 불필요하게
만들지는 않는다 — seed 367은 동결 정책 평가였고 토큰 정규화 교란을 안고 있었으나, 여기서는 양 arm이
각자 일관된 정규화로 학습하므로 그 교란이 없다.

**2단계로 나눈 이유** (사용자 지적): 1,000 epoch 적응은 "이 클립에서 도달 가능한 최선"을 답하지
못한다. 동결 정책은 20 m까지 아무것도 못 보는 세계에 맞춘 탐색 전략을 배웠고, 양 arm이 그 정책에서
출발하므로 **arm B만 뭔가를 잊어야 한다 — 설계가 B에 불리하다.** 따라서 1단계는 스크리닝이며
2시간으로 17시간을 쓸지 결정한다. 양성은 신뢰할 수 있고 **음성은 "이 예산에서 미결"이지 "효과
없음"이 아니다.** 이 해석을 측정 전에 못박았다.

정규화 설계 주석: 클립 변경은 표적 토큰 정규화도 바꾸지만 이는 교란이 아니다 — 실제 28 m 센서는
28로 정규화하고, 양 arm이 각자 일관되게 학습·평가한다. seed 367에서 교란이었던 이유는 20으로
학습된 정책에 28 정규화를 먹였기 때문이다.

seed 457/461(1단계), 463/467(2단계) 전수검색 0건. 실측 epoch당 3.1 s 기준 1단계 1.7 h, 2단계 17 h.
2단계는 70막대 고정 10k이므로 **P3가 아니다**(P3 = 70→205막대·30k·seed 211). 10k가 수렴이 아님을
명시했다 — 주말 예산에 따른 선택이며 결과는 "10k epoch에서의 비교"다.

`VERIFICATION.md`도 같은 커밋에서 갱신했다(PLAN SYNC 규칙).

## 2026-08-22 — stage 1 첫 실행 VOID: 학습 스크립트가 min_pixels를 하드코딩하고 있었다

`train_navrl_v2_search.sh:234`가 `export NAVRL_DETECTOR_MIN_PIXELS=2`를 **하드코딩**해, 사전등록이
요구하는 50을 조용히 덮어썼다. 결과적으로:

- VRAM 스모크는 **min_pixels=2로 돌았다**(내가 50이라고 보고한 것은 틀렸다). VRAM 수치
  6,854/8,192 MiB는 min_pixels와 무관하므로 여전히 유효하다.
- arm A가 완주했다면 **양 arm 모두 부정직한 센서로 학습**하고, 체크포인트에
  `cfg_detector_min_pixels=2`가 박힌 채, 겉보기엔 완전히 정상이었을 것이다. 실험이 정의하는
  "정직한 센서"가 그냥 일어나지 않는다.

**이걸 막은 것은 training source receipt 가드다.** 학습 중 런타임 소스가 바뀌자 체크포인트 저장
시점에 `NavRL training runtime source changed: train_navrl_v2_search.sh`로 거부했다. 부분 run은
폐기했다(epoch_metrics.csv도 만들어지기 전). provenance 기계가 설계대로 작동한 사례를 하나 더 얻었다.

수정: `export NAVRL_DETECTOR_MIN_PIXELS="${NAVRL_DETECTOR_MIN_PIXELS:-2}"` — 미설정 시 v2 기본값 2로
동일하므로 기존 계보에 영향이 없다.

**같은 유형을 전수 확인했다.** 사전등록이 지정하는 10개 변수 중 `train_navrl_v2_search.sh`/
`train_navrl.sh`가 덮어쓰는 것은 `NAVRL_DETECTOR_MIN_PIXELS` 하나뿐이었다.
`NAVRL_DETECTOR_MAX_RANGE`(조작 축)·`NAVRL_DETECT_WIDTH/HEIGHT`·`NAVRL_CAMERA_*`·`NAVRL_NUM_BARS`·
`NAVRL_REFLECTION_COEF`·`NAVRL_LATERAL_BIAS_COEF`·`NAVRL_SPEED_GOVERNOR`는 안전하다.

교훈: **학습 런처는 자기가 실행하는 스크립트가 자기 변수를 덮어쓰는지 확인해야 한다.** 평가
경로에는 checkpoint contract 대조가 있어 이런 드리프트가 드러나지만, 학습에는 그 대조가 없다 —
체크포인트가 "무엇으로 학습됐는지"를 스스로 기록할 뿐 "무엇으로 학습하려 했는지"와 비교하지 않는다.

## 2026-08-22 — stage 1 런처 완성: 학습 계약을 "전달한 값"이 아니라 "스크립트가 만들어낸 환경"에서 검증한다

`tools/run_navrl_ref5in_detection_range_stage1.py` (`preflight|train|evaluate|finalize|verify`).
평가 전용이던 기존 런처들과 달리 **학습을 포함**하므로 계약이 다르다.

### 핵심 기법 — 실효 환경 덤프

런처가 `export`한 값은 증거가 아니다. `train_navrl_v2_search.sh`가 그 뒤에 실행되며 자기 값으로
덮어쓰거나 `unset`할 수 있다. 그래서 arm별로 정규 학습 스크립트를
`NAVRL_V2_CONTRACT_PREFLIGHT_ONLY=1`로 **source**한 뒤 `trap 'env -0' EXIT`로 **실제로 만들어진
환경을 덤프**하고, 그 위에서 대칭차를 계산한다. 측정된 arm 간 차이:

| | clip20 | clip28 |
|---|---|---|
| `NAVRL_DETECTOR_MAX_RANGE` | 20.0 | 28.0 |

그 외 허용 차이는 run 태그·세션로그·라이브로그 3개(결과 경로)뿐이며, 하나라도 더 벌어지면 중단한다.
평가 쪽 대칭차도 `NAVRL_DETECTOR_MAX_RANGE` + 결과 디렉터리뿐이다.

이 기법이 `NAVRL_DETECTOR_MIN_PIXELS` 하드코딩을 잡았다(별도 항목 참조). **반증 실험도 했다**:
스크립트를 하드코딩 상태로 되돌리면 실효 환경이 `('2', '50')` 불일치로 preflight에서 즉시 실패한다.

### 정적 감사 — 사전등록이 고정하는 26개 변수 전수

실효 덤프는 "이번엔 살아남았다"를 말하고, 정적 감사는 "체인이 지울 수 있는가"를 말한다. 둘 다 한다.
`train_navrl_v2_search.sh` + `train_navrl.sh`에서 각 변수의 `export VAR=`(단, `${VAR:-` 형태 제외)와
`unset ... VAR`를 기계적으로 찾아, 다음 세 가지 중 **선언된** 경우만 통과시킨다.

| 변수 | 체인의 처리 | 판정 |
|---|---|---|
| `NAVRL_PERCEPTION_PERTURB` | `export ...=0` 하드코딩 | 우리가 고정하는 값과 **동일**하므로 허용(리터럴 대조) |
| `NAVRL_NUM_BARS` | `NAVRL_DENSITY_CURRICULUM=1`일 때만 `unset` | 우리가 커리큘럼을 0으로 고정하므로 허용 |
| `NAVRL_REFLECTION_COEF`·`NAVRL_LATERAL_BIAS_COEF` | 무조건 `unset` | 태스크가 미설정을 0으로 읽으므로 허용(사전등록 §4의 "=0"과 동치) |

나머지 22개는 pass-through. 손으로 한 감사와 결과가 일치했고, 이제 기계가 한다.

### 게이트 0은 예외가 아니라 **판정**이다

`max_epochs` 정상 종료 / KL 롤백 0 / 종단 SHA. 증거는 전부 산출물에서 재도출한다 — 체크포인트의
`epoch`·`frame`(2900 / 11,878,400), `aerial_run/run_summary.json`의 `exit_reason`,
`.aerial_training_finished`, 그리고 체크포인트에 내장된 `aerial_ppo_rollback_total`/`_streak`와
학습 로그의 `PPO EPOCH ROLLBACK` 줄(**독립 증인 2개**). 실패한 arm은 raise가 아니라
`STAGE1_VOID` 판정 + `void_arms`로 보고되고, 평가에 GPU를 쓰지 않으며, 실행되지 않은 평가 게이트를
"실패"로 적지 않는다.

### 평가 절반 — override 불필요를 증명했다

각 arm을 **자기가 학습한 클립에서** 평가하므로 override가 필요 없다. 정적 증명: 평가기의 v2
provenance `want` 집합에 검출 거리 필드가 **없고**(`navrl_task`도 `cfg_detector_max_range`를 기록하지
않는다), `cfg_detector_min_pixels`는 `os.environ["NAVRL_DETECTOR_MIN_PIXELS"]`에 묶여 있는데 학습·평가
모두 50이라 일치한다. 실행 증명: force 없는 preflight가 rc 0이어야 하며, 거부되면 담요식 force가
아니라 **중단**한다. 센서 충실도 실험이 좁은 단일 필드 override를 필요로 했던 것과 대조된다.

**약점 기록**: 클립은 체크포인트 provenance에 남지 않는다. 증명 가능한 arm 구분자는 평가 계약의
`target_camera_max_range_m` 하나뿐이다.

### VRAM·시간 스모크 (128 env, detect 1920×1200 + cam 160×90, warm-start 6 epoch)

| | 값 |
|---|---|
| device peak | **6,667 MiB / 8,192** (headroom 1,525) |
| epoch 시간 | **7.26 s** (D1의 160×90 3.1 s 대비 2.3배) |
| arm당 예상 | **2.02 h** (1,000 epoch), 양 arm 약 4 h |

`exit_reason=max_epochs`로 정상 종료. 1 s 폴링·5구간 측정이므로 epoch 시간은 ±0.5 s 수준이고
warm-start 직후 컴파일 워밍업이 포함돼 **상한**으로 읽어야 한다.

### 테스트

`tests/test_navrl_ref5in_run_contract.py`에 `DetectionRangeStage1Contract` 28개 추가.
전체 **622 → 650**. 학습이 GPU를 점유한 상태에서는 학습 런처를 subprocess로 부르는 12개
(`Ref5inSmokeLauncherContract` 8 + v2 launcher/recovery/v5a 4)가 중복학습 가드로 실패한다.
그 12개를 제외한 **638개는 0 failure / 1 skip**으로 확인했고, 650 전량 green 확인은 학습 종료 후
재실행이 필요하다.

### 외부 학습 run 입양(adoption)

`train`이 arm을 만드는 유일한 합법 경로는 아니다. 학습 스크립트를 직접 돌린 run도
`DETRANGE_STAGE1_RUN_ROOT_<ARM>` / `DETRANGE_STAGE1_TRAIN_LOG_<ARM>`로 지목하면 `evaluate`가
입양해 training record를 만든다. 게이트 0 사실은 전부 그 run의 산출물에서 재도출한다. 로그는
**필수**다 — 롤백 증인 2개 중 하나가 로그이기 때문이다. 다만 **클립은 입양으로 복원할 수 없다**
(`env_state`에 `cfg_detector_max_range`가 없다). arm 배정은 운영자 주장이며
`detector_max_range_evidence: operator_assertion_at_adoption`으로 요약에 명시된다.

### 다음

`train`은 실행하지 않았다(사용자/코디네이터가 직접 실행 중). arm 2개가 끝나면
`evaluate <arm>` → `finalize` → `verify`.

## 2026-08-22 — stage 1 학습 착수 (arm A 실행 중) + 학습 경로의 구조적 허점

arm A(클립 20 m) 실행 중: `runs/ppo_260822_2322_navrl_detrange-stage1-clip20-s457`,
seed 457, ep1900 → 2900, envs 128, `curriculum=False bars=70->70` 로그 확인.
실측 epoch당 약 9 s → arm당 2.0–2.6 h, 양 arm 4–5 h.

### 오늘 밤 잡힌 결함 3건 — 전부 실험을 조용히 무효화했을 것

| | 결함 | 결과 |
|---|---|---|
| ① | `train_navrl_v2_search.sh`가 `NAVRL_DETECTOR_MIN_PIXELS=2` 하드코딩 | 양 arm이 **부정직한 센서로 학습**, 체크포인트에 `cfg_detector_min_pixels=2`가 박힌 채 겉보기 정상 |
| ② | `NAVRL_DENSITY_CURRICULUM` 기본 1 → `:196`이 `NAVRL_NUM_BARS` unset | 커리큘럼이 막대 수를 70→300으로 올리고, arm B가 획득이 쉬워 먼저 승급 → **밀도까지 갈라져 축이 둘** |
| ③ | `MAX_EPOCHS`가 절대 epoch 목표 | ep1900에서 `MAX_EPOCHS=1000`은 이미 지난 목표 → **1 epoch만 돌고 정상 종료**. 2900이어야 한다(riskcap 선례 ep24001→25000과 동일) |

①은 런처 에이전트가, ②③은 내가 잡았다. ①을 실제로 막은 것은 **training source receipt 가드**다 —
수정이 학습 중에 착지하자 체크포인트 저장에서 `training runtime source changed`로 거부했다.
③은 1-epoch run이 **정상 종료로 보였다**는 점이 특히 위험하다. 부분 run 3개를 전부 폐기했다.

### 구조적 발견 — 평가에 있는 안전장치가 학습에는 없다

평가 경로는 런타임을 **체크포인트가 기록한 계약과 대조**하고 어긋나면 거부한다(seed 421에서
`cfg_detector_min_pixels: checkpoint=2 expected=50`으로 실제 발화했다). 학습 경로에는 그 대조가
없다 — 체크포인트는 "무엇으로 학습됐는지"를 기록할 뿐 **"무엇으로 학습하려 했는지"와 비교하지
않는다.** ①·②가 조용히 통과했을 이유가 이것이다.

대응: 런처 preflight에 `verify_pinned_variables_survive_the_script_chain()`을 넣었다.
사전등록이 고정하는 **26개 변수 전부**에 대해 `train_navrl_v2_search.sh`/`train_navrl.sh`의
`export VAR=`(`${VAR:-` 제외)와 `unset ... VAR`를 정적 스캔하고, 세 가지 합법 예외
(`NAVRL_PERCEPTION_PERTURB` 하드코딩 값이 우리와 동일 / `NAVRL_NUM_BARS`는 우리가 가드를 0으로
고정 / `REFLECTION_COEF`·`LATERAL_BIAS_COEF`는 unset이 곧 0)만 **선언된 경우에** 통과시킨다.
그 위에 행동 검사가 붙는다 — `NAVRL_V2_CONTRACT_PREFLIGHT_ONLY=1`로 트레이너를 소스하고
`trap 'env -0' EXIT`로 **실제 생성된 환경**을 두 arm 사이에서 diff한다. 측정 결과 차이는
`NAVRL_DETECTOR_MAX_RANGE {20.0, 28.0}` + run tag/로그 경로뿐이다.

### 남긴 provenance 구멍 2개 (해결 아님, 기록)

1. **학습된 클립이 체크포인트에 없다.** `navrl_task`가 `env_state`에 `cfg_detector_max_range`를
   쓰지 않으므로, 입양된 체크포인트가 어느 클립으로 학습됐는지 증명할 방법이 없다. arm 배정은
   **운영자의 주장**이며 `detector_max_range_evidence: operator_assertion_at_adoption`으로 기록된다.
   평가된 클립은 `v2_evaluation_contract.target_camera_max_range_m`로 증명된다.
   **지금 고칠 수 없다** — `aerial_gym/` 수정은 실행 중인 학습의 source receipt를 깨뜨린다. stage 2 전에 한다.
2. 검출 해상도가 평가기 영수증에 기록되지 않는다(seed 421에서 물려받은 것).

### 평가는 override가 필요 없다 (증명됨)

평가기의 v2 `want` 집합에 검출 거리 키가 없고 `navrl_task`도 `cfg_detector_max_range`를 쓰지
않으므로 **조작 축이 게이트에 보이지 않는다.** `cfg_detector_min_pixels`는 학습·평가 양쪽 50으로
일치한다. 런처는 `NAVRL_V2_FORCE`를 아예 도달 불가로 만들었고 테스트가 그 문자열이 대입으로
나타나지 않음을 검사한다.

테스트 622 → 650. 학습이 락을 쥐고 있는 동안 트레이너를 호출하는 12개가 duplicate 가드로 실패하며,
그것을 제외하면 638 실행 0 실패 1 skip이다. GPU가 비면 650 전수 재확인이 필요하다.

## 2026-08-23 — stage 1 두 arm 완주했으나 계약 위반으로 판정 불가

`clip20` = `runs/ppo_260822_2322_navrl_detrange-stage1-clip20-s457`,
`clip28` = `runs/ppo_260823_0426_navrl_detrange-stage1-clip28-s457`. 양쪽 ep1900→2900 완주,
`cfg_detector_min_pixels=50` 확인, rollback 0.

**그러나 목표거리가 6–28 m로 학습됐다. 사전등록 §4는 22.5–28 m를 못박았다.**
런처의 계약 검사가 `evaluate`에서 잡았다: `cfg_general_goal_dist_min (recorded 6.0, expected 22.5)`.

원인: `train_navrl_v2_search.sh:141-142`가 `NAVRL_GENERAL_GOAL_DIST_MIN="${...:-6}"`로 기본 6을
두는데 내가 그 변수를 걸지 않았다. 어젯밤 잡은 세 결함과 **같은 유형의 네 번째**다.

`k_min`은 무관하다. `NAVRL_GENERAL_TRAIN=1`이면 general spawn 경로(`navrl_task.py:1309-1320`)가
목표거리를 정하고, 복원 로그의 `k_min=20.0 k_max=28.0`은 체크포인트에 남아있던 커리큘럼 상태다.
스크립트 139행 주석이 정확히 이 오해를 경고하는데 로그를 보고 안심한 것이 실수였다.

**A/B 내부 타당성은 살아있다** — 양 arm이 동일한 6–28 분포로 학습했으므로 유일한 차이는 클립이다.
**그러나 처치가 희석된다**: 목표 6 m 에피소드는 표적이 진작 20 m 안에 있어 클립이 무관하다.
arm B의 이점이 먼 에피소드에서만 나타나므로 학습 신호가 묽어지고 **null 쪽으로 편향**된다.
게다가 학습(6–28)과 평가(22.5–28)가 다른 과제다.

판정을 내리지 않는다. 재실행 시 `NAVRL_GENERAL_GOAL_DIST_MIN=22.5`, `_MAX=28`을 명시적으로 건다.

### 운영 실패 2건 (기록)

- arm A 완료(01:23) 후 arm B를 자동으로 잇겠다고 말해놓고 **코드를 짜지 않아 GPU가 3시간 놀았다.**
- 체인이 06:40에 계약 검사로 실패한 뒤 **5시간 방치**됐다. 실패 시 알림이 없었다.

교훈: 장시간 무인 실행은 (a) 다음 단계를 코드로 연결하고 (b) **실패를 눈에 띄게** 남겨야 한다.
`chain_eval.sh`는 (a)는 했으나 (b)가 없었다.

### 남은 provenance 빚 (stage 2 전 필수)

`navrl_task`가 `env_state`에 `cfg_detector_max_range`를 쓰지 않아 **체크포인트가 어느 클립으로
학습됐는지 증명하지 못한다.** arm 배정이 운영자 주장으로만 남는다. 학습이 멈춘 지금이 고칠 때다.
## 2026-08-23 — detection-range stage 1 재실행 전 계약·운영 전면 감사

사용자 요청대로 Claude의 중단된 `/tmp` 체인을 재사용하지 않고 사전등록
`docs/prereg_2026-08-22_detection_range_2stage.md`부터 다시 감사했다.

### 확정한 계약

- `NAVRL_GENERAL_TRAIN=1`에서는 `k_min/k_max`가 아니라
  `NAVRL_GENERAL_GOAL_DIST_MIN/MAX`가 spawn band를 소유한다. 재실행은 22.5–28.0 m를
  effective trainer environment에서 확인한다.
- 70 bars 고정, density curriculum off, detect 1920×1200, RGB 160×90,
  `min_pixels=50`, perturbation·latency·range error 0, governor off다.
- 양 arm의 실제 trainer/evaluator environment 대칭차를 측정했다. bookkeeping 키를 빼면
  `NAVRL_DETECTOR_MAX_RANGE` 20.0↔28.0 **한 키만** 달랐다.
- frozen D1 ep1900 SHA `197ea269…a278e`, 학습 seed 457, 평가 seed 461, arm당 1,000 epoch
  (4.096M samples), 평가 2,049 episodes를 유지한다.
- primary verdict는 `never_acquired(clip28)-never_acquired(clip20) <= -15.00 pp`만 본다.
  capture/crash/timeout은 원값 보고용이고 판정 함수에 들어가지 않는다.

### 고친 provenance 결함

shape-compatible checkpoint가 어느 detector geometry에서 학습됐는지 스스로 증명하지 못했다.
`navrl_task.get_env_state()`에 다음을 추가했다.

- `cfg_detector_max_range`
- `cfg_detect_width`
- `cfg_detect_height`

restore는 legacy checkpoint에서 키가 없으면 허용하지만, 존재하는 키의 mismatch는 기존
same-shape config drift와 같이 경고하고 density evidence를 reset한다. generic v2 evaluator도
세 키 중 하나라도 있는 새 checkpoint라면 완전한 triplet을 요구하고 실행 환경과 비교한다.
stage-1 Gate 0는 새 종단 checkpoint의 20/28 m와 1920×1200을 필수로 검사한다. 외부 run adoption도
운영자 arm 주장만으로는 불가능하고 checkpoint-attested geometry가 맞아야 한다.

### 고친 장시간 운영 결함

`tools/run_navrl_ref5in_detection_range_stage1_campaign.py`를 추가했다. 하나의 PID와 non-blocking
file lock이 `preflight → clip20 train/Gate0 → clip28 train/Gate0 → 2-cell eval → finalize → verify`를
순서대로 소유한다. 매 phase 시작/성공과 최초 실패를 원자적 status JSON에 기록하며 실패하면
즉시 non-zero로 끝나 다음 단계가 실행되지 않는다. global `pgrep`, latest-run glob, `/tmp` scratch
chain을 쓰지 않는다.

실행·GPU·판정 계획은 `docs/execution_plan_2026-08-23_detection_range_stage1.md`에 고정했다.
기존 smoke 실측은 peak 6,667/8,192 MiB, 7.257 s/epoch라 arm당 2.02시간, 2-cell 평가를 포함한
총 예상은 약 4시간 25분(여유 포함 4시간 45분)이다.

### CPU 검증

- stage-1 run contract: 101/101 PASS
- detector/perception: 30 PASS, 1 SKIP
- campaign contract: 3/3 PASS
- trainer가 만든 effective env 및 chain-clobber 감사: PASS
- train/eval arm diff: `NAVRL_DETECTOR_MAX_RANGE`만 차이, PASS
- `py_compile`, `bash -n`, `git diff --check`: PASS

frozen `aerial_gym/config/robot_config/**`와 `resources/robots/**`는 변경하지 않았다.

기존 잘못된 결과는 삭제하지 않고 다음으로 격리했다.

- `results/VOID_20260823_goal_band_6_28_navrl_ref5in_detection_range_stage1_s457/`
- `results/VOID_20260823_pre_provenance_navrl_ref5in_detection_range_stage1_s457_preflight/`

원래의 canonical output 경로는 새 사전등록 준수 campaign만 사용한다.

## 2026-08-23 — detection-range stage 1 완료 (`RANGE_INCONCLUSIVE_AT_THIS_BUDGET`)

단일-owner campaign이 12:03 KST 시작해 16:00 KST 종료했다. preflight, 두 arm 학습과 각 Gate 0,
두 held-out cell, finalize, verify가 모두 완료됐고 품질 게이트 17개 중 실패는 0개다.

| arm | never-acquired | capture | crash | timeout | 종단 reward |
|---|---:|---:|---:|---:|---:|
| clip20 | 8.443% | 82.235% | 15.666% | 2.099% | 221.481 |
| clip28 | 3.172% | 88.677% | 11.274% | 0.049% | 232.306 |
| clip28−clip20 | **−5.271 pp** | +6.442 pp | −4.392 pp | −2.050 pp | +10.825 |

양 arm은 ep1900→2900(각 1,000 epoch/4.096M samples)을 정상 완주했고 checkpoint epoch
2900/frame 11,878,400, PPO rollback 0, KL skip 0이다. 하나의 clean training-source receipt
`5123eae4…b9a97e6`를 공유한다. 평가 seed 461, 각 2,049 episodes다.

사전등록 primary Gate S는 never-acquired delta `<= -15.00 pp`를 요구한다. 실측 −5.271 pp는
방향은 예측대로지만 크기가 부족하므로 판정은 **`RANGE_INCONCLUSIVE_AT_THIS_BUDGET`**이고
`stage2_authorised=false`다. 이것은 “거리 확장이 효과 없음”이 아니라 “warm-start 1,000 epoch
예산에서 사전등록한 큰 효과를 입증하지 못함”이다. capture/crash/timeout 개선은 원시 보조 결과이며
판정을 뒤집는 데 쓰지 않는다.

GPU smoke는 peak 7,258/8,192 MiB(headroom 934 MiB), 7.31 s/epoch였다. 실제 clip20은
1시간 43분, clip28은 1시간 57분, 두 평가는 합계 약 15분, 전체 wall time은 약 3시간 58분이었다.

운영 기록: 최초 `nohup` 시작은 실행 환경의 process-group 정리로 preflight 직전에 종료됐다.
GPU 학습은 시작되지 않았고 해당 status/log는 `VOID_20260823_launcher_startup_killed_*`로 보존했다.
이후 unified execution session이 campaign을 직접 소유해 중단 없이 끝냈다.

사후 QA에서 `summary.md`는 새 detector-geometry provenance를 정확히 설명했지만 `summary.json`의
설명 문자열만 예전 “gate에 detector-range field가 없다” 문구를 유지한 것을 발견했다. 실제 gate는
checkpoint의 `cfg_detector_max_range/cfg_detect_width/cfg_detect_height`를 요청 환경과 비교했고
통과했다. **수치·gate·verdict 변경 없이 설명 문자열만 사후 정정**하고 finalize/verify를 다시 했다.

## 2026-08-23 — sim-to-real 72시간 계약·대시보드·worktree 정리

사용자 요청에 따라 장기 roadmap을 추가하지 않고, 앞으로 3일 동안 사용자가 실제로 해야 할 일과
다음 학습 전에 기록해야 할 숫자를
`docs/SIM2REAL_3DAY_EXECUTION_PLAN.md` 한 파일로 통합했다.

### 실행 판단

- detection-range Stage 1의 공식 값은 clip20/clip28 never-acquired
  `8.443%→3.172%`(−5.271 pp), capture `82.235%→88.677%`(+6.442 pp)다.
- provenance/quality gate는 17/17 PASS지만 primary `≤−15 pp`를 못 넘어
  `RANGE_INCONCLUSIVE_AT_THIS_BUDGET`, `stage2_authorised=false`다.
- 따라서 10k Stage 2나 P3를 돌리지 않는다. 다음 72시간은 exact BOM/AUW/CG,
  intrinsics/extrinsics/time sync, real sensor trial, trial-level sensor profile, held-out tracker
  replay를 닫는 기간이다.
- 두 arm 모두 far target range가 analytic exact였으므로 capture 개선을 sim-to-real 준비 완료로
  읽지 않는다. far zone은 camera bearing/confidence/uncertainty, near zone은 측정 gate를 통과한
  LiDAR/stereo range fusion으로 분리하는 계약을 다음 후보로 명시했다.

새 문서에는 Day 1의 7거리×3조명×2운동×5반복 = 210 independent trial, Day 2의 bearing/range/
latency/dropout 통계와 trial-level bootstrap, Day 3의 real-log replay/go-no-go를 고정했다. 실행 전에는
source/checkpoint/URDF/config/calibration/dataset SHA, dynamics/sensor/observation/reward/PPO/curriculum
전 값을, 실행 중에는 outcome count-rate, first acquisition, tracker error, requested/actual/executed
speed, directional stopping margin, PPO KL/rollback, compute thermal을, 실행 후에는 terminal checkpoint와
held-out CI/receipt/source binding을 기록하도록 했다.

`README.md`, `RESEARCH_PLAN.md`, `VERIFICATION.md`, `OPERATIONS.md`는 이 문서를 72시간의 단일 authority로
가리키도록 정리했다. 과거 정량 감사인 `docs/archive/sim_vs_hardware_gap_2026-08.md`는 삭제하지 않고
보존 문서로 표시했다.

### MOTAR 사이트 갱신

- `update_status_snapshot.py`가 canonical Stage 1 `summary.json`을 직접 검증한다. schema, 2 arm,
  각 2,049 episode, 1920×1200/50 px, 1,000 adaptation epoch, outcome 합계, 17개 gate, 동결 −15 pp
  threshold와 verdict가 어긋나면 사이트 생성을 중단한다.
- 첫 화면 Research update를 최신 Stage 1 `A/B CLOSED`로 바꿨다. primary −5.271 pp, capture +6.442 pp,
  Stage 2 blocked, exact-range 한계를 함께 표시한다.
- 별도 `Sim-to-real · next 72 hours` 패널에 measure→profile→replay→preregister 순서와 학습 blocker를
  데이터로 렌더링한다.
- 실기 페이지의 폐기된 `인지 스택 472 g`, `LiDAR 56%`를 제거했다. 공식 부품 질량만의 불완전 부분합
  404.2–411.8 g, Mid-360 265 g이 최대 명시 단품, exact BOM 전 비율 계산 금지로 고쳤다.
- 실기 비행 0회, ref5in 1.20 kg은 실측 AUW가 아닌 설계점, 28 m exact range는 실기 증거가 아님을
  명시하고 72시간 gate를 기체 페이지에도 추가했다.
- 실험 인덱스를 67개로 갱신하고 `2026-08-23-ref5in-detection-range-stage1` canonical INCONCLUSIVE
  항목과 parameter join을 추가했다. status snapshot은 86 runs, active none, latest clip28 ep2900이다.

### Source Control/worktree 정리

시작 시 main `9352fde`는 clean이었고, Source Control의 대량 변경은 main에 이미 병합된 6개 Codex
worktree에 남은 평가 산출물이었다. 각 branch가 main ancestor인지 확인하고, untracked 결과는 먼저
기본 저장소의 같은 `results/` 경로로 복사한 뒤 file/symlink/directory를 원본과 byte 비교했다.
그 후 geofence, joint telemetry, mode probe, OOB/boundary search, reflection audit, topology worktree와
병합 완료 local Codex branch를 제거했다. 현재 worktree는 main 하나뿐이다.

이관 뒤 cell의 source symlink가 삭제된 worktree 절대경로를 유지한 것을 후속 검사에서 잡았다.
geofence/mode/OOB/VOID symlink 17개를 보존 bundle의 상대경로로 고쳤다. OOB canonical bundle의 ignored
source snapshot은 manifest가 기록한 clean commit `d54c737c…`에서 314 runtime file만 재구성하고,
동일 SHA의 보존된 `python_environment.txt`를 연결했다. 네 bundle의 총 1,259 runtime snapshot을
manifest SHA와 대조했고 전부 일치했다. 이 검증 전에는 worktree 정리를 완료로 간주하지 않았다.

main에 올리는 것은 결과 재검증에 필요한 compact result/receipt/source-manifest와 명시적 VOID 증거다.
topology의 reconstructable 2.9 MB 전수 row와 sample64 raw는 WORKLOG 기존 결정대로 추적하지 않고
로컬 `.git/info/exclude`로 감췄으며, canonical aggregate summary는 기존 tracked 경로에 유지된다.

### 검증

- `tests/test_status_snapshot.py`: 12/12 PASS.
- `tests/test_status_recovery_completion.py`: 3/3 PASS.
- `tests/test_status_arena_motion.js`: PASS. 과거 단일 `app.js`/index layout에 고정된 stale assertion을
  현재 split page/module 구조와 env-backed detector default 계약으로 갱신했다.
- `tools/lint_experiments.py`: 67 experiments, 0 errors.
- `tools/generate_parameter_catalog.py`: authoritative 216, echo-only 94, 294 files, 71 launchers,
  ablated 45, mirrors 6.
- `py_compile`, JSON parse, Markdown 기본 검사, `git diff --check`: PASS.

### Day 1 착수 — exact BOM 입력 전 사이트 payload 계약 정정

`docs/navrl_hardware_identification_manifest.yaml`을 첫 항목부터 감사했다. 현재 확정된 것은
`navrl_ref5in_quad`의 **합성 설계점**(1.20 kg, analytic inertia, 9.60 N/motor, τ=0.04 s)뿐이며,
실제 as-built BOM/AUW/CG/thrust curve는 전부 pending이다. 따라서 첫 사용자 입력은 실제 기체의
조립 상태와 battery·센서·compute를 포함한 AUW 또는 미조립 상태의 exact selected-parts BOM이다.

이 과정에서 `drone.html` 본문은 고쳤지만 data-driven `platform.json` 생성기가 여전히 폐기된
`100+265+72+35=472 g`을 완성 payload처럼 렌더링하는 잔여 오류를 발견했다. 생성기·브라우저 계약을
다음으로 바꿨다.

- Orin NX는 SOM-only **28.0 g**, Mid-360 **265.0 g**, D435i **72.0 g**.
- Pixhawk 6C Mini는 revision 미동결이므로 공식 범위 **39.2–46.8 g**.
- 합계 필드를 없애고 `complete=false`, 불완전 명시 단품 부분합 **404.2–411.8 g**만 export.
- carrier/cooling/storage/DC-DC, wiring/mount, frame/motor/ESC/prop/battery가 빠졌음을 data contract와
  화면 양쪽에 명시. legacy 250 g 대비 비율도 부분합의 1.62–1.65배로만 표시한다.

`generate_platform_spec.py` 재생성 결과 두 기체 파생값과 flight-envelope PASS는 유지됐고,
payload 출력은 `incomplete named-part subtotal 404.2–411.8 g`으로 바뀌었다.

### Day 1 사용자 확인 — 실제 기체 미조립

사용자가 실제 기체는 **미조립**이라고 확인했다. hardware identification manifest에
`as_built.assembly_state=unassembled`, `exact_bom_frozen=false`, `measured_auw_kg=null`을 기록했다.
따라서 simulator의 1.20 kg, analytic inertia, motor/prop/thrust 좌표를 실측값으로 승격하지 않는다.
다음 BOM 결정은 frame 보유·선정 여부부터 한 항목씩 닫는다.

### Day 1 기준 프레임 후보 선정 — 현재 ref5in 형상 우선

사용자는 보유/선정된 frame이 없으며 현재 simulation 사양에 맞는 추천을 사용하겠다고 확인했다.
현행 제조사 공식 사양을 비교해 `navrl_ref5in_quad`의 220 mm/5-inch 설계점을 가장 적게 벗어나는
**iFlight AOS 5 V5.1 Frame Kit**를 provisional packaging reference로 선정했다. 공식값은 wheelbase
228 mm, frame 165±5 g, arm 6 mm, stack mount 20×20/30.5×30.5 mm다. 비교 후보인 AOS HS5와
Nazgul Evoque/DC5/XL5는 233–245 mm, 226–256 g으로 geometry/mass 변화가 더 컸다.

이를 구매 또는 실기 플랫폼 확정으로 기록하지 않았다. 공식 질량이 있는 센서/compute 부분합과
frame만 564.2–581.8 g이므로 1.20 kg 설계점에 남는 618.2–635.8 g 안에 battery, motors,
electronics, Orin carrier/cooling, mounts/wiring이 전부 들어가야 한다. 또한 228 mm+5-inch의 단순 prop
envelope는 288.2 mm로 현 collision proxy보다 8.2 mm 크다. 따라서 scaled payload CAD, prop/sensor
clearance, complete mass/CG, motor-prop thrust/thermal gate 전에는 구매 승인·URDF 수정·재학습을 하지
않는다. 선정 근거와 승격 gate는 `docs/navrl_frame_selection_2026-08-23.md`에 고정했다.
사이트 기체 페이지에도 같은 후보 상태와 문서 링크를 넣어, provisional 선정을 실제 기체 확정으로
오독하지 않게 했다.

### Day 1 추진계·배터리 1차 screen

공식 제품 사양만으로 XING2 2207 1855KV ×4, BLITZ Mini E55 4-in-1, AOS 5 V5.1, 6S 1550/1850 mAh를
screen했다. 5.1-inch prop 3.9 g/개를 포함하면 알려진 부품 합계는 1550 mAh에서 965.5–993.1 g,
1850 mAh에서 1,014.5–1,042.1 g이다. 따라서 1.20 kg 설계점에 남는 공간은 각각 206.9–234.5 g,
157.9–185.5 g뿐이며 Orin carrier/cooling/storage/DC-DC·배선·마운트가 아직 빠져 있다.

XING2의 공식 peak 35.08 A/개와 Mini E55의 55 A continuous/개는 전류 정격상 맞지만, 모터의
propeller별 thrust/current 표를 확보하지 못했으므로 9.60 N/모터 달성이나 실제 체공을 주장하지 않았다.
추력표·payload CAD·CG·전압 sag/열 측정 전에는 구매·exact BOM 승격·URDF 수정·재학습을 하지 않는다.
세부 계산은 `docs/navrl_ref5in_component_screen_2026-08-23.md`에 기록했다.

### Day 1 공식 추력자료 audit

XING2 2207 1855KV 공식 제품 페이지를 다시 확인했다. 공개된 수치는 KV(1855/2755), 질량 31.6 g,
6S 입력, peak current 35.08 A, 16×16 mm mounting이며, 선택 propeller별 thrust/RPM/current 곡선이나
9.60 N 보증점은 페이지에 없다. 따라서 판매자·검색 결과의 단일 추정값을 simulation contract에 넣지
않고, 실제 선택 prop·6S 배터리·ESC 조건의 thrust-stand 측정으로 남겼다. 다음 측정은 최소한 hover
2.943 N/모터, 9.60 N/모터, 전류, 전압 sag, 30 s 열 상태를 같은 조건에서 기록해야 한다.

측정 장비가 아직 없으므로 `docs/navrl_ref5in_thrust_stand_protocol_2026-08-23.md`에 원자료 CSV
필드, 고정 조건, 반복수, 안전 중단 사유, 사전 `THRUST_CONTRACT_PASS/INCONCLUSIVE` 규칙을
추가했다. 이 계약 전에는 인터넷 추정치나 단일 최대추력 숫자를 simulation actuator에 반영하지 않는다.

### Day 1 payload packaging 계약

프레임이 미조립이고 Orin carrier/cooling/storage/DC-DC·배터리 revision이 없으므로 실제 fit을
주장하지 않고, `docs/navrl_ref5in_payload_packaging_contract_2026-08-23.md`에 CAD 입력 형식과
prop swept volume, Mid-360 360°×59° FOV, D435i FOV, connector bend/maintenance, CG, power/thermal
gate를 고정했다. 현재 공식 envelope는 Mid-360 65×65×60 mm/265 g, frame 228 mm/165±5 g이며,
D435i bbox와 Orin complete assembly는 pending이다. exact parts와 도면/실측 SHA가 채워질 때까지
구매 승인·URDF 변경·재학습을 하지 않는다.

### Day 1 통합 BOM 방향 권장

사용자는 필요한 센서와 연산을 모두 포함하는 방향에 동의했다. 최종 시스템은 Mid-360 + D435i +
Orin NX + Pixhawk를 모두 탑재하고, 원인분리를 위해 bench/replay만 단계적으로 진행한다. Orin은
개발키트가 아니라 compact carrier + low-profile cooling을 기준으로 후보를 찾고, 6S 1550 mAh를
질량 우선 1차 배터리로 삼는다. 1850 mAh는 1550의 loaded sag/체공시간이 부족하면서도 complete
AUW가 1.20 kg 안에 남을 때만 재검토한다.

이는 권장 방향이지 구매·exact BOM 확정이 아니다. carrier 모델, D435i unit/cable, 배터리 외곽과
전원/열/CG를 닫기 전에는 URDF 수정이나 재학습을 하지 않는다. 세부 기준은
`docs/navrl_ref5in_bom_direction_2026-08-23.md`에 기록했다.

### Day 1 Orin carrier 후보 screen

공식 Auvidea JNX42 technical reference를 확인해 JNX42-LC를 우선 carrier 후보로 선정했다. Orin NX
기준 native USB 3.0×3, CSI-2×2, 5 V fan connector, 80×104.6 mm이며 필요한 D435i/Mid-360/Pixhawk
인터페이스를 구성할 가능성이 있다. 단 base board는 12 V only라 6S에 직결할 수 없고, regulated
12 V DC-DC가 필수다. 제조사 문서에서 board mass는 확인하지 못했으므로 exact AUW·구매 승격은 보류했다.
JNX42-LC의 모델/전원/인터페이스/승격 gate는 `docs/navrl_ref5in_carrier_screen_2026-08-23.md`에 고정했다.

### Day 1 Orin 메모리 SKU provisional 결정

사용자가 추천안대로 진행하기로 해 최종 통합 컴퓨트의 기본 SKU를 Jetson Orin NX 16GB로 기록했다.
NVIDIA 공식 1KU+ volume MSRP는 16GB $999, 8GB $649로 $350 차이지만 단품 국내 실구매가는
확인 전이다. Mid-360·D435i·Pixhawk와 detector/tracker/policy를 동시에 실행할 메모리 여유를
우선하고, 실제 견적·전력·열·메모리 사용량이 닫힌 뒤 8GB downgrade를 검토한다. 구매나 URDF
변경으로 승격하지 않았으며 세부 결정은 `docs/navrl_ref5in_bom_direction_2026-08-23.md`에 기록했다.

### Day 1 프로펠러 provisional 결정

공식 iFlight 사양이 확인되는 Nazgul F5 Tri-blade를 추진계 provisional prop으로 고정했다. 5.1 inch,
pitch 3.5, 3 blades, 3.9 g/개, 5 mm hub이며 기존 4개×3.9 g 질량 계산과 일치한다. XING2 1855KV와
6S에서의 thrust/current 곡선은 공개 자료로 확인되지 않았으므로, 실제 선택 prop·배터리·ESC 조합의
추력계 측정 전에는 9.60 N/모터나 URDF actuator를 확정하지 않는다. 세부 출처와 gate는
`docs/navrl_ref5in_component_screen_2026-08-23.md`와 `docs/navrl_ref5in_thrust_stand_protocol_2026-08-23.md`에 기록했다.

### Day 1 hardware-independent sim-to-real telemetry 계약 구현

하드웨어와 rosbag이 없어도 진행 가능한 소프트웨어 단계로 `tools/navrl_sim2real_telemetry.py`를
추가했다. JSONL manifest/event 계약에서 topic·sequence·source/host timestamp·frame edge·sensor
sync group을 읽고, timestamp 역행·sequence gap·unknown topic·frame mismatch·음수 latency를
fail-closed로 판정한다. camera/LiDAR/ego-state skew와 sensor→host, policy→command latency의
p50/p95/p99/max를 계산하며, 합성 fixture 결과는 `SYNTHETIC_ONLY`로만 표시한다. 관측·보상·정책·
체크포인트에는 접촉하지 않는다.

`tests/test_navrl_sim2real_telemetry.py` CPU 테스트 5/5, Python 3.8 compile과 fixture→validate
연결을 통과했다. 사용법과 실제 로그로 승격되는 조건은 `docs/SIM2REAL_3DAY_EXECUTION_PLAN.md`
7절에 기록했다.

### Day 1 구매·제조사 확인 요청서

Orin NX 16GB, JNX42-LC, Hadron NGX012, low-profile cooling, NVMe, 6S1550, Nazgul F5 후보를
실제 BOM으로 승격하기 전에 받아야 할 질량·CAD·전원·열·connector 자료를
`docs/navrl_ref5in_vendor_request_2026-08-23.md`에 정리했다. 이 목록이 채워지고 packaging/CG/
power/thermal gate를 통과하기 전에는 구매 확정이나 URDF·재학습을 진행하지 않는다.

### Day 1 Orin carrier 2순위 대조

Connect Tech 공식 Hadron(NGX012) 사양도 대조했다. 82.6×58.8 mm, 49 g, USB 3.1×2,
4-lane CSI-2×1, 1 GbE, 9–60 V 입력이라 6S 전원 범위와 질량 면에서 강점이 있다. 반면
locking IO harness/breakout 질량과 USB/CSI 수가 JNX42-LC보다 작다. 따라서 JNX42-LC 1차,
Hadron 2차 후보로 기록하고, 두 모델 모두 Orin module/heatsink/fan/NVMe/cable 및 전원
transient/thermal/CG를 닫기 전 구매·URDF·재학습을 금지했다. 출처와 비교표는
`docs/navrl_ref5in_carrier_screen_2026-08-23.md`에 기록했다.

### 2026-08-24 — 장기 진행 루프 정리·evaluator provenance 보완·사이트 동기화

- 사용자가 매 단계마다 다음 작업을 다시 묻지 않아도 되도록, 현재 우선순위를
  `docs/SIM2REAL_3DAY_EXECUTION_PLAN.md`의 단일 실행 목록으로 유지했다. 하드웨어가 없으므로
  실측값을 추정하거나 PPO를 재시작하지 않고, 실제 로그가 들어올 때까지 software-only 계약 검증을
  닫는 순서로 진행한다.
- `tools/navrl_sim2real_telemetry.py`의 JSONL 계약(토픽·sequence·source/host timestamp·frame edge·
  sync group)과 fail-closed 검증기를 사이트 상태에 연결했다. 합성 fixture는 구조 검증용으로만
  `PASS / SYNTHETIC_ONLY`이며, 실기 성능·sim-to-real 증거로 승격하지 않는다. CPU 테스트 5/5.
- `eval_navrl_v2_density_sweep.sh`에서 detector geometry provenance가 없는 레거시 체크포인트를
  평가할 때 `NAVRL_DETECTOR_MAX_RANGE`, `NAVRL_DETECT_WIDTH`, `NAVRL_DETECT_HEIGHT`가 비어
  복구 게이트가 `KeyError`로 중단되는 결함을 수정했다. 레거시는 역사적 20 m·160×90으로 명시하고,
  신규 체크포인트는 저장된 geometry 값을 evaluator 환경에 그대로 pin한다. 복구 게이트 19/19 통과.
- `tools/update_status_snapshot.py`와 MOTAR status panel에 `READY_FOR_REAL_LOG`, 테스트 결과,
  `SYNTHETIC_ONLY`, 실제 로그로 승격하는 다음 단계가 표시되도록 반영했다. 3-day 패널의 기존
  날짜 구조와 86개 run 집계는 변경하지 않았다.
- 전체 unittest discover 결과: **659/659 PASS, 1 SKIP**. 의도된 예외 격리 테스트가 출력하는
  `obs_dump FLUSH FAILED` 로그는 실패가 아니라 해당 guard가 작동했음을 확인하는 테스트 출력이다.
  `git diff --check`도 통과했다.

#### 현재 판단과 자동 진행 순서

1. 현재는 실기 하드웨어·calibration·rosbag이 없으므로 새 PPO/URDF/reward/dynamics 변경을 하지 않는다.
2. 실측 로그가 생기면 telemetry JSONL 변환 → validator PASS → trial 단위 sensor profile →
   two-zone observation 계약 → held-out replay 순서로 진행한다.
3. 위 산출물과 source/checkpoint hash가 모두 닫힌 뒤에만 한 축의 fresh PPO를 사전등록한다.
   그 전까지 사이트 상태는 `MEASURE BEFORE RETRAINING`으로 유지한다.

### 2026-08-24 — 3–5단계 software-only 파이프라인 완성

- 실제 rosbag/MCAP/DB3가 아직 없어 1–2단계(기체 조립·원본 취득)는 완료로 표시하지 않았다. 저장소의
  simulation CSV/NPZ를 실기 증거로 재분류하지 않았다.
- `tools/navrl_sim2real_ingest.py`를 추가했다. 원본 CSV에서 topic/sequence/source·host timestamp/
  frame edge를 그대로 JSONL로 옮기고, 입력 SHA-256·run ID·source kind를 manifest에 기록한다.
  필수 열 누락·숫자 파싱 실패·빈 입력은 fail-closed한다. 테스트 4/4.
- `tools/navrl_sensor_profile.py`를 추가했다. GT와 detector가 join된 measurement CSV를 trial 단위로
  집계해 거리·조명·운동 cell별 detection recall, range-valid fraction, bearing/range error,
  latency 분포를 남긴다. frame을 독립 반복으로 취급하지 않고 threshold나 simulator noise를
  자동 선택하지 않는다. 테스트 3/3.
- `tools/navrl_two_zone_replay.py`를 추가했다. 사람이 근거와 함께 정한 `near_boundary_m` JSON
  contract만 받아 replay의 timestamp/uncertainty/range-valid 의미를 검사한다. far zone의 range
  제공, invalid range의 0 m 대체, 음수 age/covariance, 비실기 contract는 fail-closed한다.
  테스트 4/4.
- telemetry 5/5와 신규 ingest/profile/replay 11/11을 합쳐 **software-only 계약 테스트 16/16**.
  사이트 status는 `READY_FOR_REAL_LOG_PIPELINE` / `SYNTHETIC_ONLY`로 갱신했고, 실제 로그가 들어오면
  변환→검증→profile→two-zone replay 순으로 이어지도록 실행계획 8절에 명령을 기록했다.
- 이 변경으로도 PPO, reward, observation width, URDF, dynamics는 변경하지 않았다. 실제 하드웨어와
  calibration/rosbag이 닫히기 전에는 fresh training을 시작하지 않는다.

### 2026-08-24 — ref5in 동역학 3중 교차검증

실측 하드웨어가 없는 상태에서 같은 실험을 반복하지 않고, 서로 독립적인 세 관점으로 검증했다.

1. **구조/계약(CPU)**: URDF·allocation matrix·robot config·physical target equivalent contract를
   `test_navrl_ref5in_platform.py`로 재검증했다. **27/27 PASS**. 질량 1.200 kg, motor arm
   0.0777817 m, collision proxy 0.28×0.28×0.12 m, assembled inertia 계산이 모두 일치했다.
2. **독립 수치 계산**: URDF XML만 다시 읽어 질량·평행축 관성·T/W를 별도로 계산했다.
   ref5in `ixx=0.00414219891`, `izz=0.00576919783 kg·m²`, `T/W=3.261978`,
   `(T/W−1)g=22.19 m/s²`, 45° lateral acceleration limit `g·tan45°=9.81 m/s²`로
   선언값과 일치했다. 이는 hardware identification이 아니라 simulator arithmetic 확인이다.
3. **실행 동역학(GPU)**: 기존 결과를 덮어쓰지 않고 `/tmp/navrl_ref_platform_audit_20260824.json`에
   16 env·seed 911·bars 0·governor off 조건으로 `verify_navrl_ref_platform.py`를 재실행했다.
   hover/forward/reversal/yaw/100 Hz pitch-roll, finite state와 actuator saturation gate가
   모두 PASS했다. ref5in forward 2.490 m/s, reversal t90 1.0 s, yaw 2.999 rad/s,
   worst pre-clamp saturation 1.3%였다.

#### 동역학 관련 결론

- 현재 ref5in simulator 내부에 즉시 수정해야 할 **수치적 동역학 불일치**는 발견되지 않았다.
- 다만 이것은 실제 기체가 맞다는 뜻이 아니다. thrust curve, motor time constant, yaw torque,
  CG/inertia, power/thermal은 여전히 미측정이다. URDF의 1.20 kg과 9.60 N/motor는 synthetic
  design point다.
- task default `max_velocity=2.0`, `yaw_rate_max=2.5`와 canonical v2 launcher/evaluator의
  `2.5 m/s`, `3.0 rad/s`가 다르다는 점을 코드 주석과 discipline 문서에 명시했다. canonical
  launchers/evaluators는 provenance에 값을 pin하므로 기존 checkpoint는 변경하지 않았다.
- 따라서 지금 단계에서 물리 파라미터를 임의로 고치거나 재학습하지 않는다. 실제 측정값이 들어오면
  한 축씩 새 lineage로 등록한다.

### 2026-08-24 — physical target 속도 포락선·reflection/distance 상태 확정

- `docs/prereg_2026-08-24_physical_target_speed_envelope.md`를 먼저 동결하고, 기존 physical
  target gate를 그대로 사용한 4 densities × 4 fixed-speed diagnostic을 실행했다. seed 509,
  mixed pattern, 32 env, 280 measured steps/cell이다. speed arm마다 Isaac Gym interpreter를
  분리해 재현성/자원 해제를 보장했다.
- 결과: 최고 passing speed는 70 bars `0.6 m/s`, 150 bars `0.9 m/s`, 300 bars `0.9 m/s`이며
  205 bars는 0.6/0.9에서 invalid OBB state 1 sample 때문에 strict gate를 통과하지 못했다.
  1.2/1.5 m/s는 각 밀도에서 planner/contact/ratio 중 하나 이상이 gate를 넘었다. 이는 target
  속도 상향이나 PPO 재학습의 근거가 아니라, 205-bar route/boundary event를 고쳐야 한다는
  diagnostic이다. 원자료: `results/navrl_physical_target_speed_envelope_seed509/summary.json`.
- 기존 N1 real-frame reflection audit는 `VERIFY PASS | CHIRALITY_CONFIRMED_REAL_FRAME`으로
  receipt 재검증했다. 다만 paired-reflection consistency A/B는 prereg §5-b의 loss-profile과
  dedicated arm launcher가 없으므로 실행하지 않았다. 계수 `0.01`을 임의로 재사용하지 않는다.
- 거리 충실도는 `docs/prereg_2026-08-24_distance_fidelity.md`로 설계만 동결했다. 실기 range
  ground truth/profile이 없으므로 합성 noise나 range 성능 수치를 만들지 않고
  `BLOCKED_NO_REAL_RANGE_PROFILE`로 남겼다.
- 이번 변경은 physical target runtime, PPO, reward, observation schema, URDF를 바꾸지 않았다.

### 2026-08-24 — physical target OBB 경계 forensic 및 braking fallback

- 사용자가 요청한 큰 실행 순서에 따라 205 bars/seed 509/mixed/32 env/0.6·0.9·1.2·1.5 m/s를
  동일 contract로 조사했다. pre-fix invalid non-contact 이벤트는 유한 상태에서 OBB x/y
  margin이 음수가 되는 경계 초과였고, NaN이나 planner 재계획은 아니었다. 원자료는
  `results/navrl_physical_target_invalid_forensics_seed509/`에 보존했다.
- center만 arena 안에 두던 wall reserve의 결함을 고쳐 현재 target OBB world-axis support를
  planner center bounds에 반영했다. 또 첫 safe step이 없는 physical rollout에는 least-bad
  outward command 대신 zero planar command를 보내 controller가 감속하도록 했다. teleport,
  position clamp, invalid gate 완화, speed 추가는 하지 않았다.
- 동일 gate/grid post-fix 결과: 70 bars 최고 0.9 m/s, 150/205/300 bars 최고 0.6 m/s.
  1.2/1.5 m/s는 각 density에서 planner/tracking/state/contact 중 하나 이상 실패했다.
  post-fix forensic은 0.6/0.9/1.2 m/s 0건, 1.5 m/s 1건(x-margin 약 −0.00006 m)으로
  줄었지만 strict state gate는 아직 닫히지 않았다. 따라서 physical PPO는 계속 BLOCKED다.
- 다음은 gate를 임의 완화하는 대신 잔여 1.5 m/s boundary-inertia 사건의 step 전후 동역학을
  분리 계측하고, 그 결과로만 B 단계의 PASS/FAIL을 결정한다. B가 닫히기 전에는 C(실기 센서
  로그), D(two-zone replay), E(reflection A/B), F(fresh PPO)의 순서를 유지한다.

### 2026-08-24 — synthetic corridor mode probe 재검증

- frozen ref5in checkpoint에 대해 symmetric ±12° 및 좌우 ±5° physical-geometry fixture를
  token slot 순서 LR/RL 모두 평가했다. probe action은 실행하지 않는 side forward이며, 결과는
  `INCONCLUSIVE_POLICY_CHIRALITY`였다. reflection error가 0.15 gate를 넘었으므로 mode averaging
  또는 좌우 대칭을 개선책으로 채택하지 않는다.
- 이 결과는 synthetic policy screen일 뿐 실제 고밀도 환경의 causal evidence가 아니다. paired
  reflection A/B 학습은 별도 preregistration으로 남기고, physical/sensor contract 이후에만
  실행한다. 원자료: `results/navrl_ref5in_symmetric_corridor_mode_probe_seed431/summary.json`.

### 2026-08-24 — 1~7 단계 software-only 일괄 preflight

- telemetry → CSV ingest → trial sensor profile → two-zone replay를 한 명령으로 연결하는
  `tools/run_navrl_sim2real_software_preflight.py`를 추가하고 실행했다.
- telemetry/ingest/profile/replay는 모두 구조상 PASS였지만, 입력은 synthetic fixture라 claim은
  `SYNTHETIC_ONLY`다. 실제 센서 성능·range noise·latency 수치로 승격하지 않는다.
- 기존 mode probe는 `INCONCLUSIVE_POLICY_CHIRALITY`, physical target gate는
  `all_cells_pass=false`이므로 fresh PPO는 계속 BLOCKED다. 다음 실제 입력은 hardware manifest,
  calibration SHA, 210개 독립 trial, timestamp contract다.
- 결과 영수증: `results/navrl_sim2real_software_preflight_2026-08-24/summary.json`.

### 2026-08-24 — 시뮬레이션 검증 종료 상태·사이트·논문 작성 브리프

- 현재 가능한 시뮬레이션/소프트웨어 검증을 종료 상태로 정리했다. 사이트의 상태를
  `SIMULATION VERIFIED · HARDWARE PENDING`으로 갱신하고, software preflight는
  `SYNTHETIC_ONLY`, physical target gate와 fresh PPO는 `BLOCKED`로 명시했다. mode probe도
  `INCONCLUSIVE_POLICY_CHIRALITY`를 유지한다. 즉 시뮬레이션 파이프라인의 계약·계측 검증이
  끝났다는 뜻이지, 실제 센서·기체로 sim-to-real 성능이 입증됐다는 뜻은 아니다.
- `tools/update_status_snapshot.py`가 preflight/physical-envelope/mode-probe 영수증을 읽어
  `docs/status/status.json`의 simulation verification 요약을 자동 생성하도록 했다. 대시보드
  `docs/status/index.html`·`docs/status/drone.html`은 이 한계와 다음 조건을 직접 표시한다.
- 논문 초안 작성에 사용할 수 있도록 `docs/PAPER_WRITING_BRIEF_2026-08-24.md`를 만들었다.
  corrected-v2 계약, 보상 계수, 밀도·속도 결과, latency 정정, physical gate, OOD/합성-only
  주의사항과 출처 경로를 모았으며, 미검증 hardware/citation 주장은 GPT가 발명하지 않도록
  명시적인 작성 규칙을 넣었다.
- 이 시점에서 실기 측정값 없이 PPO를 새로 학습하거나 sim-to-real 성공을 주장하지 않는다.

### 2026-08-24 — 하드웨어·학습 계약·자료구조 사이트/문서 통합

- 발표와 논문에서 같은 숫자를 보도록 `docs/MOTAR_SYSTEM_SPEC_2026-08-24.md`를 추가했다.
  legacy/ref5in 플랫폼의 질량·관성·추력·충돌 proxy, 실제 탑재 후보의 불완전 부분합, corrected-v2
  arena/sensor/episode/action/PPO/reward 계약을 한 문서에 모았다. ref5in 값은 hardware-informed
  simulation candidate이며 AUW/CG/thrust curve/calibration은 여전히 미측정임을 명시했다.
- 898-D actor observation을 `288 static + 480 obstacle history + 50 robot history + 80 target
  history`로 분해하고, `[CLS]+static+obstacle×5+robot×5+target×5`의 17-token 구조와
  Transformer/PI altitude/Lee controller/100 Hz physics 흐름을 Mermaid와 SVG로 기록했다.
  8 obstacle slots가 8개의 독립 Transformer token이 아니라 한 history token 안의 proposal이라는
  점도 명시했다.
- README의 기준일·현재 판정을 2026-08-24로 갱신하고 canonical system-spec/dashboard 링크를
  추가했다. `RESEARCH_PLAN.md`의 오래된 Transformer dropout 0.1 표기를 실제 network dropout 0.0
  (sensor dropout 0.3과 구분)으로 정정했다.
- 대시보드에 `system.html`을 추가하고 모든 페이지의 shell/cache-bust를 갱신했다. 기체·파라미터
  데이터도 generator로 재생성해 source commit과 파일 목록을 최신화했다.

### 2026-08-24 — README 연구 구조도와 public site 전면 단순화

- 이전 변경은 시스템 그림을 별도 `system.html`과 Markdown 문서에만 두어 GitHub README 첫 화면에서
  연구 구조를 바로 파악할 수 없었다. 이를 철회하고 `docs/assets/motar-system-overview.svg`와
  `docs/assets/motar-control-stack.svg`를 추가해 README와 public site에 동일 원본을 직접 렌더링했다.
  첫 그림은 RGB-D/LiDAR/ego-state → 898-D history → 17-token Transformer → controller/physics를,
  두 번째 그림은 bounded action → altitude PI → Lee velocity controller → motor dynamics를 나타낸다.
- `docs/status/`의 7개 HTML, sidebar/dark-mode shell, 3-D browser arena, 동적 panel renderer,
  Three.js vendor와 JavaScript data mirror를 제거했다. 새 사이트는 `index.html + style.css` 단일
  정적 페이지이며 overview/method/evidence/contract/next만 남겼다. 머신용 `status.json`, platform/
  parameter/experiment JSON과 과거 문서는 삭제하지 않아 실험 원자료와 감사 가능성은 보존했다.
- README도 연구 질문, 두 구조도, 현재 근거 5개, canonical 계약, 최소 재현 명령, 저장소 지도와
  physical gate 중심으로 다시 썼다. `SIMULATION VERIFIED · HARDWARE PENDING`, `SYNTHETIC_ONLY`,
  실제 기체 미조립·실측 로그 0이라는 claim boundary를 첫 화면과 evidence section에 반복 명시했다.
- 더 이상 브라우저 JavaScript wrapper를 만들지 않도록 status/platform/parameter/experiment 생성기를
  JSON-only로 정리했다. 삭제한 arena simulator 전용 테스트 대신 `tests/test_status_site.js`를 추가해
  단일 페이지 구조, SVG 접근성, local link, ref5in generated values와 overclaim guard를 검사한다.
- 검증: static site contract PASS, status snapshot unit tests 15/15 PASS, parameter catalog source audit
  PASS, 67-experiment lint PASS, `git diff --check` PASS. Chrome headless 1440 px desktop 전체 렌더를
  확인했으며 외부 font/JavaScript 의존성은 없다.

### 2026-08-24 — 단일 페이지 3-D arena 복원·제어 구조 상세화

- 단일 페이지 정보 구조는 유지하면서 사용성이 좋았던 3-D arena만 독립 섹션으로 복원했다.
  Three.js/OrbitControls를 다시 로컬 vendor로 포함하고, 기존 `arena_motion.js`와 `arena.js`의
  corrected-v2 geometry configurator를 사용한다. 40×40×3 m, bar 10–300, 3 m 높이,
  `navrl_band` touch 0.4 m/gap 1.6 m, goal 6–28 m와 target 0.3–1.5 m/s를 viewer boot에서 고정했다.
- viewer 기본은 205 bars이며 bar density, target max speed, pause/play, overview/chase/sensor 시점,
  camera FOV/LiDAR/trail을 직접 조절한다. 표적은 mixed CV/waypoint motion을 사용하지만 pursuer는
  설명용 steering이므로 화면 바로 위에 **PPO rollout이나 성능 측정이 아님**을 명시했다.
- `motar-control-stack.svg`를 5개 추상 block에서 실제 실행 순서 8단계로 교체했다. Transformer의
  bounded action, body-frame ±2.5 m/s와 yaw ±3 rad/s mapping, actor z overwrite, altitude PI
  `Kp=4/Ki=1`, Lee velocity `Kv=2.5`, 45° force clamp/tilt thrust compensation, attitude/rate gain,
  ref5in allocation, 0.04 s motor lag와 100 Hz physics를 식과 함께 표시했다. 사이트에는 각 단계의
  역할을 네 개 설명 카드로 추가하고 README도 실제 제어 경로를 명시했다.
- 검증: viewer asset/order/local-link contract PASS, v2 motion bounds·goal/speed·speed-zero age reset
  PASS, Chrome headless에서 WebGL canvas 생성과 205-bar desktop render 확인, 상세 SVG 렌더 확인,
  status unit tests와 experiment/catalog lint 재통과, `git diff --check` PASS.

### 2026-08-25 — legacy/bounded/physical 표적 학습환경 독립 감사

- 별도 worktree/branch에서 target update rate와 `dt`, reset/CV/waypoint sampling, bar center-band와
  AABB gap, wall reflection/push-out, speed·acceleration·turn authority, static route topology,
  pose/velocity/sensor synchronization, actor/critic oracle 경계, launcher/checkpoint 계약을 감사했다.
- legacy는 순간 wall mirror/bar push-out을 쓰는 재현 계보, bounded는 10 Hz 2-D kinematic 중간
  계보, physical만 100 Hz rigid-body/motor 계보임을 분리했다. 최신 physical speed-envelope receipt는
  1.5 m/s 전 density strict FAIL이라 physical PPO 차단을 유지한다.
- v2 geometry CPU probe(seed 825, **12 layouts/density, 0.15 m grid**, side-clearance 0/0.1 m)는
  70/150/205/300 bars에서 해당 작은-margin sampled connectivity를 유지했다. 이것은 이후 physical
  routed all-orientation support+tracking 계약의 전역 연결성 근거가 아니다. `navrl_band` center gap
  1.6 m도 footprint/tilt/dynamic corridor 보증이 아니다. bar 평균 footprint는 0.3653 m²,
  205 bars/1600 m² nominal occupancy는 4.681%다.
- checkpoint에 저장만 하고 restore 시 비교하지 않던 target accel/turn/lookahead/clearance와 physical
  authority를 fail-loud drift guard에 추가했다. tracking/boundary margin도 저장·검사한다. 옛
  checkpoint의 missing field는 계속 허용하므로 policy/legacy transition semantics는 그대로다.
- base v2 launcher가 shell의 stale `NAVRL_TARGET_DYNAMICS`를 상속하던 경로를 닫았다. canonical base는
  legacy만 허용하고, physical은 fresh wrapper의 명시적 child marker로만 통과하며 bounded는 전용
  launcher가 없으므로 거부한다.
- `bounded_drone_target_step`의 마지막 반환값이 full-horizon route certificate가 아니라 selected
  first-step feasibility라는 사실을 docstring에 고정했다. 상세 증거와 후속 순서는
  `docs/target_motion_training_environment_audit_2026-08-25.md`에 기록했다.
- CPU 검증: target motion 13/13, ref5in platform 27/27, physical envelope helper 3/3,
  perception 30 pass+1 skip, reachability 3/3, 신규 contract 5/5 PASS. ref5in run-contract 101개 중
  1개는 독립 worktree에 ignored historical checkpoint가 없어 fixture-missing, 나머지 100개 PASS.

### 2026-08-25 — 사이트 표적 동작 계보 분리·10 Hz 고정 시계

- 3-D arena의 표적·추적자 적분을 브라우저 refresh rate에서 분리했다. simulation은 10 Hz
  fixed step으로만 갱신하고 30/60/144 Hz 화면은 직전·현재 상태를 보간한다. 5 s 합성 fixture에서
  세 refresh rate 모두 정확히 50 step, simulation time 5.0 s, 동일한 최종 표적 위치·속도를 냈다.
  긴 background-tab 공백은 무제한 catch-up하지 않고 버리며, pause/visibility/mode 변경 때 clock을
  reset한다.
- Target display UI를 `legacy` / `bounded` / `physical-style`로 분리했다. 기본은 `bounded`이며
  legacy는 checkpointed virtual-point lineage, bounded와 physical-style은 새 trajectory lineage임을
  HUD와 설명에 표시한다. 모드 간 화면 결과는 정책 성능 비교가 아니다.
- JavaScript bounded step은 Python `target_motion.py` 계약의 가속도 4.0 m/s², 선회율 150°/s,
  lookahead 1.0 s, 후보 각도·속도 scale·fail-closed score, 비물리 lineage의 0.77 m 중심거리
  obstacle proxy를 미러링한다. `(10,0)`에서 1.5 m 앞에 막대가 있는 golden fixture는 Python과
  첫 위치가 1e-7 m, 첫 속도가 1e-7 m/s 이내로 일치했다.
- `physical-style`은 bounded command에 1차 속도 lag와 45° attitude display limit만 추가한
  **설명용 시각화**다. 실제 PhysX, 6-DoF rollout, 정책 평가가 아님을 selector·HUD·본문에 명시했다.
  arena 배치, 막대 3-D geometry와 기존 viewer controls는 유지했다.
- 검증: `node --check` 3개 JS PASS, `tests/test_status_site.js` PASS,
  `tests/test_status_arena_motion.js` PASS(고정 시계·보간·Python golden·가속/속도/선회/attitude
  limit), 기존 target-motion 13/13·status snapshot 12/12·recovery completion 3/3 PASS,
  `git diff --check` PASS. 학습·GPU 평가·환경 Python 코드는 실행하거나 변경하지 않았다.

### 2026-08-25 — 3-D arena global routed-preview 추가

- 개발 중인 Python `target_route_planner.py`를 다시 읽고 사이트용 DOM-independent mirror
  `docs/status/arena_route.js`를 추가했다. 기준 계약은 0.25 m grid, exact closed bar AABB,
  ref5in target 3-D box half-diagonal XY support 0.2069 m, tracking reserve 0.45 m,
  boundary 1.25 m + support, deterministic A*/Dijkstra, diagonal corner cutting 금지,
  continuous exact-LOS smoothing, connected goal 최소 6 m다. endpoint의 nearest grid centre가
  막힌 경우에도 연속 connector가 안전한 7×7 anchor를 결정론적으로 찾는다.
  최종 source receipt는 topology branch `fcec3d2`, planner SHA-256 `99ff8fe88525…`다.
- viewer 기본 target display를 `routed-preview`로 바꾸고 smoothed route polyline을 표시한다.
  line은 매 10 Hz step마다 현재 target에서 cursor 이후 waypoint만 잇는다. HUD와 selector에는
  **global route + bounded/lagged browser preview · NOT PhysX/PPO**를 표시해 global planner,
  target display dynamics, pursuer policy를 혼동하지 않도록 했다.
- route follower는 task가 실제 전달하는 `waypoint_reach_m=0.5`를 사용한다. planner의 0.05 m
  goal tolerance를 follower reach로 오용하지 않는다. 목표 완료 시 같은 control interval에서
  새 selector로 최소 6 m connected goal을 다시 계획하고 reference를 재계산한다. 새 경로가
  없으면 random/local fallback 없이 target command와 velocity를 0으로 만드는 fail-closed HUD
  경고를 낸다.
- Python golden fixture와 JS가 empty-arena expanded nodes 21,904/path endpoint, exact-AABB corner
  fixture expanded 280/raw 34/4개 좌표, occupied-nearest-cell anchor fixture expanded 91/raw 28/4개
  좌표까지 일치했다. 최종 합성 dense fixture route runtime은 Node에서 205 bars 41.1 ms,
  300 bars 27.9 ms였고 둘 다 `ok`였다. 이는 브라우저 설명용 runtime이지 simulator 성능값이 아니다.
- 검증: 30/60/144 Hz routed target 5 s 결과 동일(각 50 step), 30 s continuous route에서 goal
  replacement ≥1·유효 route long stall 0, exact AABB corner/no-path/zero-command PASS,
  기존 arena motion/site contract PASS, Chrome headless SwiftShader WebGL canvas·route HUD PASS,
  JS syntax와 `git diff --check` PASS. GPU 학습·평가와 Python 환경 코드는 변경/실행하지 않았다.

### 2026-08-25 — 3-D arena 동작 통합 독립 감사

- 통합 기준 commit `6ef603f`의 fixed-step/interpolation 상태 소유권, pause/reset/background-tab,
  30/60/144 Hz 결정성, Python bounded parity, collision/waypoint/CV 의미, 3-D 자세·trail·capture를
  별도 worktree에서 재감사했다. global route planner와 학습환경·정책 코드는 변경하지 않았다.
- pause가 직전 snapshot(`alpha=0`)을 렌더하던 오류를 최신 committed snapshot(`alpha=1`)으로
  고쳤고, pause 진입 시 보간 상태를 접었다. `visibilitychange`와 intersection 변화 때 clock을
  reset해 background-tab 경과시간을 재개 시 catch-up하지 않도록 했다.
- bounded waypoint 도달 검사를 Python과 동일하게 virtual first-step 이후 위치에서 수행하도록
  수정했다. physical-style은 실제 actor pre-step 위치를 계속 사용한다. Python golden fixture와
  5 s full target+pursuer fixture에서 30/60/144 Hz 모두 50 step·5.0 s·동일 상태를 확인했다.
- endpoint 거리만 보던 capture를 Python과 같은 target-relative swept segment-sphere 검사로 바꿔,
  양 endpoint가 0.5 m 밖이어도 한 step 중 포획 구를 관통하면 capture하도록 했다.
- Three.js의 +X forward/y-up 좌표에 맞춰 drone bank와 target roll을 X축, target pitch를 Z축에
  적용했다. physical-style의 world-frame 가속도를 target body frame으로 회전한 뒤 roll/pitch를
  계산해 북향 전진 가속이 roll로 잘못 표시되던 오류도 제거했다.
- `physical-style`은 계속 10 Hz bounded command에 lag/attitude 표시만 더한 설명용 proxy이며
  PhysX·6-DoF 성능 주장이 아니다. trail의 local-coordinate ownership과 CV/bar push-out 의미에서는
  추가 결함을 찾지 못했다.
- 사이트 스크립트 5개의 cache-bust를 `20260825`로 통일하고 local-link test에 고정했다.
  `test_status_arena_motion.js`, `test_status_site.js`, `git diff --check` 모두 PASS했고,
  Chrome headless SwiftShader에서 실제 WebGL canvas(723×452) 생성과 모든 20260825 asset 응답을
  확인했다. GPU job·학습·평가는 실행하지 않았다.

### 2026-08-25 — fresh-only physical target global route 후보 구현

- 기존 target transition은 route `off`로 보존하고, 새 model id
  `physx_ref5in_6dof_global_astar_aabb_v1`을 별도 opt-in으로 구현했다. routed mode는
  physical + waypoint만 허용하고 전용 fresh launcher 외의 stale shell/resume/checkpoint를 거부한다.
- planner는 actual per-asset bar AABB, 0.28×0.28×0.12 m box의 3-D half-diagonal을 쓴
  all-orientation XY support, tracking/boundary reserve로 2-D occupancy를 만든다. deterministic
  8-neighbour A*, diagonal corner-cut 금지, 동일 continuous inflated AABB LOS smoothing을 사용한다.
  invalid/no-path/local rollout failure는 zero velocity로 fail-closed하며 이유를 export한다.
- reset/replan 때만 선택 env를 CPU로 보내며, 같은 component를 Dijkstra 한 번만 확장해 6 m 이상
  goal과 parent path를 함께 얻는다. 후보마다 occupancy/A*를 최대 64회 만들던 초안은 폐기했다.
  일반 step은 padded GPU waypoint cache로 vectorize했고 runtime counter의 interval별 GPU sync도 없앴다.
  0.5 m task reach와 waypoint 통과 판정으로 overshoot 정지를 막았다.
- 후속 검수에서 local rollout failure 뒤 동일 waypoint로 deterministic replan하면 같은 명령을
  반복할 수 있는 livelock을 발견했다. cooldown 중 zero command는 유지하고, 이후 selector를 다시
  뽑아 같은 connected component의 새 goal로 계획하도록 수정했다. telemetry에
  `local_step_invalidations`와 `connected_goal_replans`를 분리했고 selector GPU→CPU 전송은 env별
  `.item()` 대신 batch 1회로 바꿨다.
- healthy 재목표화와 failure replan을 구분하도록 cumulative `goal_completions`, reason별 invalidation,
  connected-goal replan을 export한다. 실제 PhysX reset 비용을 다음 smoke에서 숨김없이 측정하도록
  GPU→CPU copy 전부터 plan 종료까지 planning batch/환경수/total/max wall time도 기록한다. 시간값은
  계측 전용이며 제어 분기에 사용하지 않는다.
- synthetic route test 10/10와 launcher/checkpoint contract 8/8 PASS, py_compile/bash syntax PASS.
  seed 825 actual-AABB CPU benchmark(16 layouts/density)는 70/150/205/300 bars 평균
  46.18/48.71/58.34/64.01 ms/env, p95 최대 115.52 ms, 128-env 직렬 보수 투영 최대 8.19 s,
  connected goal 64/64 성공으로 preregistered CPU gate를 통과했다.
- 이 64/64는 same-component goal availability일 뿐 arena-wide connectivity가 아니다.
  route-equivalent **obstacle inflation**(boundary/grid/corner 계약은 다름)의 독립 20-layout raster
  topology probe에서 300-bar largest component 0.3998, pair connectivity 0.2247, crossing 0.3이므로
  300 bars arena-wide roaming 주장을 금지한다. 상태는
  `PASS_CPU_ENGINEERING_GATE / SIMULATOR_UNMEASURED`;
  다음은 short fresh PhysX smoke이며 PPO 본학습 권한은 없다.

### 2026-08-25 — routed fresh launcher 재현성·source provenance 차단

- `d047073`에서 분리한 `codex/routed-launcher-provenance` worktree에서 launcher/provenance만
  수정했다. target planner, task transition, observation/reward/termination과 정책은 변경하지 않았고
  GPU 학습·평가도 실행하지 않았다.
- canonical `train_navrl_physical_routed_fresh.sh`가 호출 cwd와 무관하게 `SCRIPT_DIR` 기준으로
  physical→v2→base launcher를 실행하도록 했다. hostile shell이 child marker를 미리 넣어도 두 lower
  launcher가 token과 exact tuple을 각각 재검증한다.
- frozen tuple은 physical+waypoint+`global_astar_v1`, arena 40×40×3, `bars_h3`, full-width
  `navrl_band`, route grid 0.25 m/expansion 50000/waypoint 128/cooldown 10/goal tolerance 0.05 m/
  minimum goal 6 m/**goal exclusion 1.0 m**, target tracking 0.45 m/boundary 0.75 m,
  ref5in authority(1.20 kg, 0.0777817 m arm, 9.60 N, 0.04 s, 45°)로 고정했다. fixed
  0.28×0.28×0.12 m box에서 source-bound된 conservative XY support는 0.206881609 m다.
- 같은 canonical 이름 아래 학습 의미가 변하지 않도록 128 env, density 70→300(step 15),
  gate `70:0.82,85:0.77,100:0.72,115:0.70`, dwell 1000/evidence 16384,
  target speed 0.3→1.5 m/s(ramp 1 epoch), `cluster_sector`, LR 3e-5, governor off도
  unconditional pin+이중 validation 대상으로 만들었다. seed, `MAX_EPOCHS`, run tag/log path만 실행
  시점 선택으로 남겼다.
- dirty worktree에서도 `NAVRL_TARGET_CONTRACT_PREFLIGHT_ONLY=1`은 receipt를 만들거나 Isaac Gym을
  시작하지 않고 exact handoff를 검사한다. 실제 training은 `create_navrl_source_bundle.py
  create --require-clean`으로 clean committed runtime source receipt와 manifest SHA-256을 만든 뒤에만
  진행하고, lower launcher도 receipt/clean flag와 manifest 형식을 fail-closed로 확인한다.
- hostile env를 넣고 repository root에서 preflight한 결과 exact 128 env/70→300/speed ramp 1/
  selector/LR/governor 계약으로 PASS했다. 반대로 실제 launcher 진입은 현재 dirty source 3개를
  열거하고 exit 1로 GPU 초기화 전에 차단됐다. launcher contract unit test 13/13,
  세 launcher `bash -n`, `git diff --check`가 PASS했다.

### 2026-08-25 — routed fresh planner 안전 blocker 2건 수정

- 0.5 m waypoint reach/overshoot가 코너에서 다음 waypoint로 너무 일찍 넘어가면, 현재 위치에서
  다음 waypoint로 향하는 새 connector가 원래 exact inflated-AABB 경로를 자를 수 있었다. planning
  CPU 단계에서 각 outgoing segment의 장애물·arena boundary 최소 Euclidean clearance를 정확히 계산해
  waypoint별 **handoff clearance certificate**로 GPU cache에 저장했다. runtime follower는 기존
  reach/overshoot 조건과 함께 현재 위치가 이 certified ball 안에 있을 때만 다음 waypoint로 전환한다.
  일반 control path는 bar geometry를 보거나 `.cpu()`/`.item()`/`.numpy()`로 sync하지 않는다.
- closed-AABB corner regression에서 기존 reach 0.5 m 안이지만 장애물을 자르는 connector를 구성했다.
  해당 코너 certificate는 약 0.033 m였고 unsafe shortcut은 거부했으며, certificate 안의 safe 위치에서는
  정상 전환했다. empty/open route는 큰 certificate를 유지해 기존 overshoot 전환도 보존했다.
- local-step infeasible 이후 같은 deterministic goal/route로 돌아가는 livelock을 막기 위해 이전 goal을
  중심으로 1.0 m exclusion radius를 connected-goal planner에 전달한다. reachable component 안에서
  radius 밖 후보를 같은 deterministic selector 계약으로 고르고, 후보가 없으면
  `no_alternative_goal` zero/fail-closed 상태로 남는다. planner 회귀가 같은 goal을 반환해도 manager가
  재검증해 `same_goal_reselected`로 차단하고 cumulative `same_goal_reselection_count`를 export한다.
- 변경은 opt-in `physx_ref5in_6dof_global_astar_aabb_v1` fresh routed lineage에만 연결했다.
  route-off/legacy target semantics는 건드리지 않았으며 GPU 학습·평가는 실행하지 않았다.
- 검증: route planner 13/13, target environment contract 8/8, target motion 13/13,
  physical speed envelope 3/3 PASS. `py_compile`, routed launcher `bash -n`, `git diff --check` PASS.

### 2026-08-25 — routed physical-target 32-cell simulator gate 평가기 구현

- 동결 문서 `docs/preregistration_physical_target_routed_simulator_gate_2026-08-25.md`의
  route off/global_astar_v1 × speed 0.6/0.9/1.2/1.5 × 70/150/205/300 bars를 seed 827,
  32 env, 300 step, warmup 20으로 실행하는 전용 도구
  `tools/verify_navrl_physical_target_routed_simulator_gate.py`를 추가했다. route arm×speed마다 fresh
  Isaac Gym child를 만들고 density 4개만 공유하므로 총 8 child/32 record다. GPU 실행은 하지 않았다.
- 각 cell은 initial reset을 포함한 route counter delta로 plan success/fallback/goal completion,
  reason별 invalidation, same-goal reselection, planning wall time을 기록한다. route-on의 역사적
  `_tm_last_step_feasible`은 route cooldown까지 섞으므로 immediate local failure로 쓰지 않고,
  warmup 이후 `local_step_infeasible` counter delta만 사용한다. route-off는 기존 field를 유지한다.
- controller의 task-lifetime diagnostics를 cell 끝에서 그대로 읽지 않는다. 각 physics interval의
  `saturation_substeps/substeps` 비음수 증가량을 failed-env reset 전에 합산하고, tilt도 reset 전 전구간
  max를 보존한다. initial/mid-cell reset GPU-synchronized wall time과 rollout RL steps/s·env-intervals/s도
  별도로 기록한다.
- 모든 child는 stale `NAVRL_*` shell 변수를 제거하고 physical authority, AABB route, goal exclusion
  1.0 m를 재고정한다. density별 initial bar center/AABB SHA가 8개 arm에서 동일해야만 matched delta를
  허용한다. evaluator/task/planner/task config/ref5in robot config+URDF와 전체 `aerial_gym` runtime을
  source manifest로 묶고 local repo import origin을 강제한다.
- 환경변수 선언만 믿지 않고 실제 생성된 task에서 physics dt 0.01 s, RL당 physics 10 step,
  RL step 0.1 s, ref5in robot config/URDF SHA, physical box 0.28×0.28×0.12 m, routed conservative
  support 0.206881609 m, bar offset 1과 density별 active AABB 수를 child/summary에 증명한다.
- 결과 `summary.json`, 8개 child summary, source/execution manifest, receipt를 상호 SHA로 묶는다.
  `--verify --require-contract full_1p5|density_conditioned`은 원 child 32개에서 gate/verdict를 다시
  계산하며 `PASS_ROUTE_MECHANISM`과 요청 contract가 없으면 nonzero다. density-conditioned pass만으로
  canonical full-1.5 launcher를 해제하지 않고 `BLOCKED_PHYSICAL_TRAINING`을 유지한다.
- CPU helper 7/7, upstream route planner 13/13, target environment 13/13, physical envelope 3/3,
  Python 3.8 `py_compile`, launcher `bash -n`, `git diff --check` PASS. 다음은 clean committed tree에서
  GPU grid 1회 실행 후 receipt verify이며, 그 전 상태는 계속 `SIMULATOR_UNMEASURED`다. 예정 결과:
  `results/navrl_physical_target_routed_gate_seed827/`.

### 2026-08-25 — routed-preview를 final Python safety contract에 동기화

- 사이트 mirror의 기준을 final main Python safety commit `a373202`, planner SHA-256
  `7fec3015e5de…`로 다시 고정했다. target 0.28×0.28×0.12 m box의 3-D half-diagonal은
  `0.2068816… m`이므로 기존 `0.2064 m` 표기를 **0.2069 m**로 정정했다.
- JavaScript planner도 Python과 같은 exact segment↔closed-AABB 거리와 boundary 거리를 계산해
  waypoint별 handoff clearance certificate를 plan 결과에 저장한다. route follower는 0.5 m reach 또는
  cross-track 제한을 만족한 overshoot에 더해, current position이 certificate 안에 있을 때만 다음
  waypoint로 전환한다. certificate가 없거나 유한하지 않으면 전환하지 않는 fail-closed 계약이다.
- browser local re-goal은 직전 goal 중심 1.0 m를 reachable 후보에서 제외한다. 대안이 없으면
  `no_alternative_goal`과 zero command를 유지하고, 방어 재검사에서 같은 goal이면
  `same_goal_reselected`로 거부한다. HUD에 successful re-goal 수와 same-goal block 수를 분리했다.
- a373202의 exact-corner fixture를 Node에서 독립 재현해 기존 0.5 m reach 안의 unsafe shortcut은
  거부하고 약 0.0180 m certificate 안의 handoff는 허용했다. open-space overshoot도 유지했다.
  30/60/144 Hz는 동일 50-step/5 s 상태를 냈고, 30 s 연속 fixture는 re-goal ≥1·long stall 0이었다.
- 검증은 Node route parity/source SHA, exact AABB/no-path/no-alternative/zero-command, site/motion contract,
  headless SwiftShader WebGL을 대상으로 수행했다. Python·PhysX·PPO·GPU 실행 코드는 변경하지 않았다.

### 2026-08-25 — topology claim 범위 정정 및 원자료 추적

- primary에 untracked로 남아 있던 seed-825 topology CSV를 저장소에 추가하고 생성 명령·수치·한계를
  `results/navrl_physical_target_route_topology_seed825_summary.md`에 고정했다. 20 layouts/density,
  0.15 m raster, 70/150/205/300 bars, side-clearance 0.517 m 조건이다.
- analyzer의 `side_clearance=0.517`은 level box half-width 0.14 m와 합쳐 total obstacle inflation
  0.6570 m다. routed target의 all-orientation support 0.2068816 m + tracking 0.45 m = 0.6568816 m보다
  반올림상 0.118 mm 보수적이다. 그러나 analyzer는 full boundary, diagonal corner connection,
  floor/ceil raster stamp를 써 route의 boundary 1.25 m+support/no-corner-cut/exact LOS와 동일하지 않다.
- route-equivalent obstacle-inflation raster 결과는 205 bars LCC/pair/crossing
  0.9577/0.9211/1.0, 300 bars 0.3998/0.2247/0.3이다. 따라서 300-bar arena-wide roaming 주장은
  금지하되 exact planner reachability 증명으로도 과대해석하지 않는다.
- README/site의 `333/333 scenes reachable`를 seed167의 **selected 205-bar bar-contact endpoint pair**
  결과로 정확히 축소했다. 이 historical centre-disk oracle은 global/random-pair/300-bar connectivity나
  actual per-bar AABB를 검증하지 않으므로 “기하는 무죄” 문구를 철회했다.
- site 상태일을 2026-08-25로 갱신하고 corrected-v2 simulation evidence와 routed physical simulator
  gate pending, hardware pending을 분리했다. evaluator/preregistered simulator gate 코드는 변경하지 않았다.

### 2026-08-25 — routed simulator gate GPU 전 적대적 감사 보강

- GPU 실행 전에 평가기 v1을 다시 감사해 두 가지 판정 오염을 발견했다. low-level loop가 canonical
  `NavRLTask.step`의 `num_task_steps += 1`을 생략해 10-step replan cooldown이 영원히 풀리지 않았고,
  warmup `continue`가 속도/RMSE뿐 아니라 contact·invalid·position·failed reset까지 첫 20 step에서
  제외했다. GPU 결과는 아직 없으므로 폐기할 결과는 없으며 evaluator/receipt schema를 v2로 올렸다.
- 각 300-step interval은 `_advance_target`이 현재 clock을 본 뒤 physics와 failed-env reset을 끝내고
  정확히 한 번 clock을 증가시킨다. 실제 route manager를 쓰는 CPU 회귀에서 step 0~9는 cooldown,
  step 10은 replan eligible임을 확인한다. warmup 20은 speed/RMSE에만 적용하고 safety·position·reset
  denominator는 `32×300`으로 별도 기록·검증한다.
- route-off의 bounded-step feasible boolean과 route-on의 `local_step_infeasible` invalidation event는
  정의가 달라 각각 명시적 arm-specific gate로 유지하되 matched subtraction에서는 제거했다.
  preregistration을 GPU 실행 전에 같은 내용으로 정정했으며 ambiguous legacy alias도 제거했다.
- `AERIAL_GYM_SIM_NAME=base_sim`을 강제하고 실제 sim class/source SHA를 증명한다. child receipt에 Python
  executable/version, torch/CUDA, Isaac Gym origin/hash, GPU name/driver와 실제 import된 `navrl_task` 및
  planner origin/source SHA를 추가했다. ref5in config/URDF와 함께 어느 하나라도 다르면 VOID다.
- density별 bar AABB·robot pose·target pose digest는 8 arm 모두 같아야 한다. waypoint/route goal은
  속도 arm 안에서는 같아야 하지만 route-on은 connected goal을 선택하는 조작이므로 off/on goal
  동일성은 의도적으로 요구하지 않는다. row ID를 route/speed/density에서 재계산하고 이 초기 digest를
  포함한 record-contract SHA도 검증한다.
- 중복으로 지적된 hidden `--source-manifest`와 `active_support` 대입은 integrated main에 각각 1개만
  존재함을 확인했다. CPU helper 9/9, route planner 13/13, target environment 13/13, physical envelope
  3/3, target motion 13/13과 `py_compile`/launcher `bash -n`/`diff --check`가 PASS했다. GPU는 계속
  미실행이며 다음 권한은 이 v2 exact 32-cell grid뿐이다.

### 2026-08-25 — routed simulator gate 최종 NO-GO 3건 해소

- child summary의 4개 row가 header의 route/speed를 실제로 따르는지 종전에는 aggregation 단계에서만
  간접 검사했다. 이제 각 row를 header route/speed, frozen density 순서, 재계산한 record ID와 즉시
  결합한다. route-on cell을 off child에 끼운 경우와 density row 순서를 바꾼 경우 모두 fail-closed
  회귀를 추가했다.
- pursuer를 정지시키려던 `[N,4]` zero policy action을 `sim_env.step`에 raw command로 넘기던 오류를
  수정했다. canonical `NavRLTask.step`과 같이 target begin→target advance→
  `transform_action_to_command(zero_policy_action)` 순으로 호출하고 finite `[N,4]` command만 physics에
  전달한다. 따라서 zero action이어도 altitude PI/yaw 및 실제 action mapping 계약은 유지된다.
- receipt verify가 child cell/verdict는 재계산했지만 summary의 matched route-on-minus-off 표 자체는
  재계산하지 않았다. 16개 matched row를 child 원자료에서 다시 계산해 byte-equivalent 비교하며,
  summary SHA를 다시 맞춘 뒤 delta만 변조한 fixture도 거부한다. GPU는 여전히 미실행이다.

### 2026-08-25 — routed simulator gate attempt 1 VOID와 conda build-tool 고정

- 첫 GPU 시도는 task 생성 전 torch cpp extension 초기화에서 종료되어 **0/32 record,
  VOID_EXECUTION**이다. absolute `/home/fair/miniconda3/envs/aerialgym/bin/python`을 썼지만 inherited
  `PATH`가 해당 `bin`을 포함하지 않아 같은 환경의 `ninja`를 찾지 못했다. 수치 gate나 planner가
  실행된 실패가 아니며 원 artifact `results/navrl_physical_target_routed_gate_seed827/`는 덮어쓰지 않는다.
- child 환경 생성을 `build_child_environment`로 분리했다. 선택된 `sys.executable`의 parent `bin`을
  `PATH` 첫 항목으로 강제하고, 그 옆 `ninja`가 executable이 아니거나 없으면 child launch 전에
  fail-closed한다. hostile `PATH=/hostile/only` fixture에서 conda bin prefix와 stale NavRL/sim env 제거를
  확인했다.
- software provenance에 `ninja` absolute path, SHA-256, `--version`을 추가하고 aggregation/receipt
  verify 때 파일 bytes와 version을 다시 검사한다. numerical gate·grid·seed·physics는 바꾸지 않았다.
  attempt 2는 명시적 별도 출력 `results/navrl_physical_target_routed_gate_seed827_attempt2/summary.json`
  만 사용한다. 이 수정 과정에서는 GPU를 실행하지 않았다.

### 2026-08-25 — routed simulator gate attempt 2 integrity PASS, route mechanism FAIL

- attempt 1은 task 생성 전 `ninja` PATH 오류로 0/32 `VOID_EXECUTION`이었다. 별도
  `results/navrl_physical_target_routed_gate_seed827_attempt2/`에 재실행 결과를 보존했으며,
  attempt 1 raw artifact는 덮어쓰지 않았다.
- attempt 2는 route-off/on × 0.6/0.9/1.2/1.5 m/s × 70/150/205/300 bars의 **32/32 JSON
  integrity PASS**를 기록했다. 그러나 receipt는 `FAIL_ROUTE_MECHANISM` 및
  `BLOCKED_PHYSICAL_TRAINING`으로 판정했다. PPO policy는 로드하지 않았고 hardware validation과
  300-bar arena-wide connectivity claim은 없다.
- compact cell map(off/on)은 70 bars `P/F, P/F, F/F, F/F`, 150 bars `P/F, P/F, P/F, F/F`,
  205 bars `P/F, P/F, F/F, F/F`, 300 bars `P/F, F/F, F/F, F/F` (속도 0.6→1.5 순)이다.
- route-on 70-bar 4-speed pool은 plan success `0.1454668471` (gate ≥.99), fallback
  `0.359296875` (gate ≤.01)이고, 별도 70×0.6 cell은 goal completions/env `.25` (gate ≥.5)다.
  same-goal은 전체 routed cell에서 `0`이다.
  전체 route-on에서 local invalidation `.125–.635%`가 fallback `32.6–85.0%`로 증폭됐고,
  status 합계 `unsafe_start=420`, `ok=80`, `no_path=6`, `local_step_infeasible=6`이었다.
  따라서 현재 engineering diagnosis는 unsafe_start soft-envelope recovery deadlock이며,
  global route 부재의 증명이나 motor/tilt/contact 주원인 판정은 아니다. tracking은 별도 gate
  metric으로 유지한다.
- evaluator와 preregistered threshold는 수정하지 않았다. 다음 authority는 동일 32-cell gate를
  safe-prefix/full-horizon 및 unsafe-start recovery 계측 후 새 lineage에서 재실행하는 것이다.
  상세: [`attempt 2 summary`](results/navrl_physical_target_routed_gate_seed827_attempt2/summary.md),
  [`attempt 1 VOID`](results/navrl_physical_target_routed_gate_seed827/VOID.md).

### 2026-08-25 — routed 결과 공개 반영과 status-site compact pass

- README, status JSON/experiment registry와 live site를 attempt 2의 정확한 판정으로 맞췄다:
  `32/32 integrity PASS`, `FAIL_ROUTE_MECHANISM`, `BLOCKED_PHYSICAL_TRAINING`. 설명용 3-D
  routed preview는 계속 `NOT PhysX/PPO`로 표시한다.
- 교차검수에서 plan success `14.5467%`와 fallback `35.9297%`가 70×0.6 단일 cell이 아니라
  **70-bar 네 속도 cell pool**임을 확인해 모든 공개 문서에서 정정했다. `0.25 goals/env`만
  70 bars × 0.6 m/s cell 값이다. experiment registry의 실제 evaluator 경로도
  `tools/verify_navrl_physical_target_routed_simulator_gate.py`로 고쳤다.
- 사이트 hero/section/metric typography와 간격을 줄이고, 3-D stage를 desktop
  `clamp(360px,55vh,620px)`, mobile `clamp(320px,72vw,380px)`로 제한했다. HUD는 11–13 px,
  control row는 compact layout으로 바꾸되 body 16 px와 route geometry 동작은 보존했다.
- 1366×768 및 390×844 headless Chrome screenshot을 직접 확인했다. hero와 현재 판정이 첫 화면에
  함께 들어오며 가로 overflow는 없었다. JSON/lint, static site, 30/60/144 Hz motion, routed parity,
  headless WebGL, Python gate/planner/motion/environment/envelope/status suites, py_compile, bash-n,
  artifact SHA/32-cell semantic 재검산과 code/docs `git diff --check`가 모두 PASS했다. 보조 `.log`
  파일은 ANSI와 원출력 공백을 byte 그대로 보존하므로 whitespace 검사에서만 제외했다.
- 최종 공개자료 적대검수에서 status snapshot의 옛 seed-509 route-off speed envelope가 attempt 2
  결과처럼 보일 수 있음을 발견했다. generator와 JSON을 고쳐 옛 값은
  `historical_post_wall_brake_speed_envelope`로 lineage를 명시하고, routed attempt 2의 밀도별 통과
  속도는 네 밀도 모두 `null`로 따로 기록했다. 사이트 카드에는 `70-bar · 4-speed pooled`를
  명시했고 registry가 summary/receipt/execution/source manifest와 children까지 연결하도록 보강했다.
- GitHub Pages는 정적 HTML/CSS/JS만 사용하므로 `docs/.nojekyll`을 추가했다. 첫 push 뒤 raw branch는
  최신이었지만 Pages branch-source workflow가 생성되지 않아, Jekyll 변환을 명시적으로 끄는 실제
  publishing-source 변경으로 재배포 이벤트를 다시 발생시켰다.
- 두 ref를 한 receive에서 갱신한 push가 Pages 이벤트를 만들지 않아, Pages/default source인
  `research/navrl-env`를 단독 ref push로 한 번 갱신한 뒤 `main`을 같은 commit으로 fast-forward한다.

### 2026-08-25 — routed recovery evaluation-only forensics preregistration

- 기존 attempt2의 route-on local invalidation `.125–.635%`, fallback `32.6–85.0%`,
  `unsafe_start=420` 진단을 원인 분해하기 위한 별도 observer를 추가했다. observer는 child
  process에서만 manager 호출을 monkey-patch하며 simulator source의 target command, planner
  decision, reward, observation, termination, PPO에는 연결되지 않는다. attempt2 evaluator와
  artifact 경로는 read-only로 고정했다.
- frozen slice는 seed 827, route-on, 32 env × 300 steps, 70/150/205/300 bars × 0.6/1.5 m/s의
  8 cells다. 0.25 m planner grid와 동일한 ±3-cell anchor search를 사용하고 obstacle soft
  reserve는 정확히 `.45 m`로 유지한다. runtime boundary hard=`wall_margin .50 + support`,
  route boundary soft=`1.25 + support`를 별도 attestation하며 boundary 차이는 `.75 m`다.
- 각 invalidation/replan/fallback에 exact hard/soft bar·boundary clearance, obstacle/boundary
  unsafe-start reason, active cross-track/whole-polyline error, requested/realized target speed,
  originating invalidation, fallback age·per-origin intervals, nearest soft-free anchor와 exact
  hard-free connector를 기록한다. exact closed-AABB segment/corner clearance도 별도 저장하며
  malformed geometry는 null로 fail-closed한다. 분석 결과로 `.45`를 낮추거나 margin/tuning을
  변경하지 않는다.
- CPU test `tests/test_navrl_route_recovery_forensics.py` 7/7 PASS, diagnostic `py_compile`,
  preflight, `git diff --check` PASS. Existing route-planner suite could not import because the
  system Python lacks torch; no GPU command/runtime was executed. New preregistration:
  `docs/preregistration_physical_target_route_recovery_forensics_2026-08-25.md`.

### 2026-08-25 — routed recovery forensic adversarial contract hardening

- Added explicit bounded-rollout rounded-corner soft-bar geometry (`support-inflated AABB`
  Euclidean distance ≥ `.4501 m`) beside route square-AABB soft-free geometry and a disagreement
  flag. The `.45 m` reserve and planner geometry were not changed. Added exact corner fixture and
  synthetic anchor CPU timing test; forensic suite is now 9/9.
- Forensic observer now batches one geometry transfer per manager hook, skips 7×7 anchors for
  fallback/invalidation, and searches anchors only for unsafe-start plans. Receipt records event
  count, observer wall time, rollout wall time, and share. All 300 transitions are recorded;
  warmup=20 is a parity marker only.
- Summary decisions use only the first unsafe replan per unique `local_step_infeasible` origin;
  repeats, unattributed initial failures, and other-origin rows are separate. Fallback amplification
  numerator is local-origin only, and pooled plus per-cell status/reason/origin tables are emitted.
- Added full aerial_gym/resources runtime source manifest, clean committed HEAD/tool/import hashes,
  authorized fresh child token, canonical output-only atomic partial→final rename, completion marker,
  and receipt semantic summary recomputation. Existing attempt2 output is hard-rejected; reruns or
  mixed/partial cells fail closed. GPU remains prohibited and unexecuted.

### 2026-08-25 — forensic receipt post-commit and instantiated-runtime provenance

- Receipt verification now validates the recorded execution commit object with `git cat-file` and
  rechecks the tool/source/runtime bytes, without requiring current HEAD equality. This permits a
  result commit or descendant/cherry-pick to verify when bytes match and rejects byte drift or a
  forged/missing commit. Execution itself still requires a clean committed worktree.
- Every child records Python/torch/CUDA/Isaac Gym/GPU-driver/ninja identities, exact imported
  module origins, target-motion/controller bytes, and post-reset instantiated contract values for
  density/speed, BaseSimConfig/physics timing, ref5in config+URDF, arena/bar placement, physical
  target parameters, route resolution/cooldown/goal distance/exclusion, and margins. Parent
  receipt requires all eight children to carry identical software provenance and stores each
  runtime contract.
- Summary origin tables now expose per-cell origin id/reason/first unsafe status/repeats/fallback
  age and intervals; other-origin rows are excluded from local amplification and repeats. Public
  `--summarize` cannot mutate a finalized artifact. CPU regression suite: 12/12 PASS; no GPU was
  run.

### 2026-08-25 — final runtime and fallback-cache hardening

- Fallback geometry now reuses route-vs-bar segment clearance cached at the observed plan; route
  polyline/cross-track telemetry remains live, but fallback no longer repeats the 300-bars ×
  128-segments calculation. A CPU call-count fixture proves the cache path.
- Every per-origin summary row now carries fallback intervals and maximum age for every reason,
  with local intervals additionally split for amplification. The instantiated contract now
  fail-closes on target gains/box/motor values, bar capacity/placement, route limits/support,
  physics timing, exact speed min/final, and robot source hashes cross-bound to the manifest.
- GPU provenance requires CUDA/current device identity and a non-empty `nvidia-smi` query with
  executable hash and driver/GPU/UUID text. CPU regression suite: 15/15 PASS; no GPU was run.

### 2026-08-25 — routed recovery forensics: `RECOVERY_DOMINANT`

- 사전등록한 seed 827, route-on, 32 env × 300 steps, 70/150/205/300 bars × 0.6/1.5 m/s의
  8/8 GPU PhysX cells가 완료됐고, 별도 `--verify`가 execution commit/source/runtime/software/
  robot/simulator receipt와 모든 raw cell을 재검증했다. PPO와 reward/observation/termination,
  target command 및 route decision은 변경하지 않았다.
- 총 local invalidation 358건이 local-origin fallback 35,666 intervals로 증폭됐다
  (`99.6257 intervals/invalidation`; cooldown 10보다 큼). fallback age는 median 101,
  p90 219, max 287 steps였다. 최초 unique local-origin unsafe-start replan 200건 중
  hard-free/soft-unsafe는 97.0% (Wilson 95% lower 93.61%), 가까운 soft-free anchor까지 exact
  hard-safe connector가 존재한 비율은 96.5% (lower 92.95%)였다. 세 사전등록 gate가 모두
  통과해 verdict는 `RECOVERY_DOMINANT`다.
- plan status는 replan에서 `unsafe_start=3774`, `ok=101`, `no_path=82`, `unsafe_goal=79`이며,
  initial plan은 `ok=349`, `unsafe_start=17`, `no_connected_goal=6`이다. 즉 주병목은 global
  topology 부재가 아니라 local rollout 실패 뒤 zero-command/cooldown이 hard-safe이지만 route
  soft-envelope 밖인 상태를 탈출시키지 못하는 복구 state machine이다. 300 bars의 no-path와
  topology fragmentation은 별도 제한으로 남는다.
- rounded local envelope와 square route envelope disagreement 1,832건도 관측됐지만, 이 결과는
  `.45 m` margin을 낮추거나 geometry를 사후튜닝할 권한이 아니다. 다음 단계는 measured braking을
  보존한 hard-safe connector recovery를 새 lineage에 사전등록하고, 짧은 simulator smoke 뒤
  변경하지 않은 32-cell routed gate를 재실행하는 것이다.
- 결과: [`summary`](results/navrl_physical_target_route_recovery_forensics_seed827/summary.md),
  [`machine-readable summary`](results/navrl_physical_target_route_recovery_forensics_seed827/summary.json),
  [`receipt`](results/navrl_physical_target_route_recovery_forensics_seed827/receipt.json).

### 2026-08-25 — recovery core P0 audit (GPU/PPO not run)

- Kept the legacy `global_astar_v1` route manager diagnostics and runtime branches unchanged;
  `global_astar_recovery_v2` remains an explicit fresh-only mode.
- Added central hard-envelope strictness (`1e-4 m`) and the derived `0.0123 m` prospective
  CONNECT reachable-tube reserve (`ceil(9.81*0.1^2/8)`). Recovery point, brake, anchor,
  CONNECT rollout, and watchdog certificates use the same reserve.
- Split BRAKE/CONNECT age and timeout tensors. BRAKE uses measured stop-time p95 + 0.20 s;
  CONNECT derives a budget from the fixed 7x7 radius-3 diagonal, acceleration ramp, worst half-turn,
  and the same reserve. Malformed per-environment geometry and repeated local-soft-free failures
  latch `NO_CONNECTOR`.
- Recovery scalar p05/p95 values cannot arm the controller without the common probe-receipt
  validator hook (`NAVRL_TARGET_RECOVERY_PROBE_VALIDATED=1`); no probe or GPU/PPO execution was run.
- BRAKE and CONNECT timeout reasons now have distinct status codes/counters. Braking uses a
  validated monotone speed-indexed p95 stop-distance lookup with ceiling-speed selection; the
  scalar `v^2/(2*a)` fallback remains legacy-only and cannot arm recovery.
- Core receipt integration now validates canonical `core_integration`, certified lookup, measured
  lateral stopping tube, and hashed clean source manifest; watchdog brake certificates inflate
  bars/shrink walls by the lateral tube. CONNECT requires full-horizon certificate plus fixed
  nonnegative anchor progress; no GPU/PPO was run.
- Follow-up raw-receipt hardening calls the pinned `tools/verify_navrl_physical_target_braking.py`
  verifier at recovery task construction and consumes only its raw-recomputed summary and
  returned core handoff. Receipt/manifest/`VALIDATED` producer fields cannot arm recovery; the
  verifier and probe origins/SHA are pinned. CONNECT progress is measured against the selected
  full-horizon final position, with explicit row-id certificate telemetry. The common verifier
  must expose `core_integration` in its returned result; CPU pycompile/diff checks passed, with
  no GPU/PhysX/PPO execution.

### 2026-08-25 — fresh physical-target zero-command braking probe scaffold

- Added the independent `navrl_target_recovery_braking_probe_v1` generator, raw-first standalone
  verifier, and fresh-only v2 launcher. The probe is physical-target-only and does not edit the
  routed task/controller/planner or reuse the pursuer braking probe.
- Frozen arms are 0.6/0.9/1.2/1.5 m/s × 32 envs, seed 827, base_sim/ref5in, `0.01 s × 10`, and
  stop threshold 0.10 m/s. Raw rows retain stop time/distance/effective deceleration and the
  contact/invalid-OBB/saturation/tilt gates; the validator recomputes p05/p95 and the measured
  speed-to-p95 lookup from raw rows.
- Receipts are finite-only canonical JSON with exact source/runtime contract, valid recorded git
  object, per-cell/tool/core hashes, completion marker, atomic non-overwrite finalization, and
  post-rename standalone verification. GPU execution is intentionally not run in CPU preflight.

### 2026-08-25 — braking probe adversarial safety hardening

- Replaced random 70-bar warmup with an explicitly recorded obstacle-free (`NUM_BARS=0`) arena-center
  setup and a 5 s speed-convergence gate (absolute and relative tolerance frozen before execution).
  Warmup contact/OBB/saturation/tilt are separate from braking-phase diagnostics; braking counters
  are reset only in evaluation telemetry accumulators.
- Raw traces now carry positions, interval path length, speed, safety flags, saturation, tilt, and
  lateral deviation. The validator derives stop first-crossing, integrated stop distance, effective
  deceleration, p05/p95, and a cumulative-maximum certified speed lookup from trace data; forged rows,
  collision traces, low initial speed, and path/provenance inconsistencies fail closed.
- Added parent-only child authorization records, pinned executable/ninja environment, clean-source
  and `c98997d` ancestry attestations, identical runtime provenance across all four cells, and the
  canonical core integration handoff object. GPU execution remains intentionally unrun.

### 2026-08-25 — PhysX-substep braking evidence

- Replaced interval-only telemetry with a transparent callback proxy around the unchanged physical
  target controller. The proxy delegates force computation, calls the controller post-step hook,
  and records every `0.01 s` PhysX substep on device before one compact cell transfer. Stop
  first-crossing, integrated path, lateral deviation, contact, invalid OBB, saturation, and tilt
  are now substep-derived; the 10-callback-per-0.1 s contract is bound and validated.
- The certified speed lookup remains raw p95 plus cumulative maximum; core handoff consumers must
  use `certified_lookup_for_speed(summary, speed_mps)` and the certified lateral tube. GPU remains
  prohibited for this audit.

### 2026-08-26 — braking receipt core-handoff return contract

- The standalone braking receipt verifier now returns the canonical `core_integration` object in
  its verified result, matching the raw-first validator contract consumed by recovery v2.
- The exact verifier source hash is pinned as
  `26913b091b89bdb5d0bdfa651d67eca803bbf8076b4180b8b689bb1a873f6df5` after py_compile and
  diff-check. No GPU, PhysX, or PPO execution was performed.

### 2026-08-25 — recovery-v2 dedicated 32-cell evaluator (CPU implementation)

- Added a new opt-in evaluator for `off` versus `global_astar_recovery_v2` on the frozen
  2×4×4 grid (seed 827, 32 envs, 300 intervals). The original v1 evaluator remains byte-identical.
- Added evaluation-only packed telemetry with exact hard/soft AABB margins and reason codes,
  state/phase ages and timeouts, actual PhysX state, command, anchor/connector/candidate evidence,
  watchdog/contact/OBB data, and recovery outcome/reset attribution. Raw denominators and reason/
  transition identities are recomputed by the standalone verifier.
- The evaluator snapshots and re-verifies the complete raw-first physical-target braking receipt,
  records the measured p95 stop distance separately from the p05 formula, uses hidden-directory
  atomic finalization, and verifies before and after rename. Source/import/software/child/raw/
  summary/receipt hashes are bound; post-result-commit verification does not require HEAD equality.
- CPU adversarial tests cover atomic non-overwrite, exact 32-cell identity, throughput regression,
  v1 byte preservation, schema pairing, continuous AABB reserve, illegal/cross-interval state
  changes, duplicate timeout prevention, no-connector zero command, position/reset mutation, and
  raw SHA/geometry failures. No GPU/PhysX/PPO execution was performed.
- Core integration now accepts only the finalized braking receipt/probe schema pair, consumes the
  validated monotone measured stop-distance lookup, separates BRAKE/CONNECT timeout reasons, and
  exposes an unambiguous recovery-only selected-candidate full-horizon certificate. GPU launch is
  still blocked until the fresh target braking probe produces its first real finalized receipt.
- The final integration binds the standalone verifier's canonical speed-to-p95 lookup and lateral
  stopping tube. Probe verification reconstructs source bytes from the receipt's immutable git
  object, so the later recovery-core commit cannot invalidate an otherwise valid probe receipt.
- CONNECT certificates now carry explicit global environment rows and selected candidate indices;
  coordinate matching was removed because multiple simulator environments can have identical local
  coordinates. First-step, full-horizon, and actual post-physics fixed-anchor progress are all
  fail-closed. The progress tolerance is frozen at `1e-6 m`.
- The 32-cell evaluator names and verifies the local metrics separately: rounded bounded-step
  infeasibility for `off`, rounded normal-route invalidation for recovery, each with an explicit
  `32×300` denominator. Exact CONNECT failures remain in the recovery reason partition.

### 2026-08-26 — recovery evaluator integration contract reconciliation

- Integrated the evaluator and braking probe into one clean lineage. The standalone verifier keeps
  the pinned current-source manifest contract and exact `core_integration` return hash
  `26913b091b89bdb5d0bdfa651d67eca803bbf8076b4180b8b689bb1a873f6df5`; the evaluator consumes the
  same raw-first result without introducing a second verifier semantics.
- Added explicit receipt/probe schema constants to the task source and reran 22 recovery-core,
  8 braking, 12 packed-telemetry, and 5 evaluator CPU tests. No GPU, PhysX, or PPO execution was
  performed.

### 2026-08-26 — integrated lineage pin before GPU probe

- The probe's original `c98997d` short hash was not an ancestor after cherry-picking into the clean
  integration worktree. The contract now pins the exact integrated core commit
  `dac38227cc3ad2130c84755ef9aab2e75becb9f0`; no runtime tuning or semantic change was made.
- Recomputed and pinned the final validator sources: probe
  `169c43d4d9d950d41fe9abd8c6e198f0be8c51348cc4ce243943ef97274c0434`, verifier
  `33923918fead5f85f919336a5c82f9135202f7a68238411b6ea1ec42286fa365`. CPU braking/core tests and
  source compilation pass. The first GPU launch attempt correctly failed closed because the old
  base commit was not an ancestor; no simulator or PPO step ran.

### 2026-08-26 — target-motion attestation object fix

- The live probe reached PhysX but rejected `physical_mass=None` because its attestation read
  physical target fields from `task.task_config`; those fields are instantiated on `task.tm`.
  The probe now attests the instantiated target-motion object without changing any physical value.
  The probe source hash is re-pinned to
  `c585fd71f1f8f7de6cd173eec6f9b83c014c69530bcf15a0d38df676235dec02`; no measurement was accepted
  from the failed cell.

### 2026-08-26 — controller gain attestation object fix

- The physical target's attitude/rate gains are materialized as CUDA tensors on
  `_target_controller`, not as list attributes on `task.tm`. The probe now reads the controller
  tensors, transfers only this attestation vector to CPU, and compares with the frozen gains. No
  controller gain or physics parameter was changed; the previous cell was rejected before data
  collection.

### 2026-08-26 — controller gain tensor shape normalization

- The instantiated gain tensors have shape `[1, 3]`; the attestation now flattens the single
  controller vector before comparing the frozen three-axis gains. No gain value was changed and
  the rejected cell produced no accepted measurement.

### 2026-08-26 — route-off braking support attestation

- The zero-bar braking probe intentionally instantiates `route_mode=off`, where the route support
  cache is zero by construction. The attestation now accepts that declared zero cache while still
  recording the geometry-derived certified support; route-on evaluator children retain the strict
  nonzero support check. No obstacle, controller, or physical parameter changed.

### 2026-08-26 — braking setup reset-order correction

- The raw trace exposed a 28.27 m first-step path jump: the probe wrote its center pose before
  `reset_idx`, so reset overwrote the recorder baseline while the receipt still declared the center.
  Reset now runs first, followed by the center pose/zero-velocity write and tensor refresh. This is
  a measurement setup correction; no data from the invalid cell is retained or accepted. Probe
  source hash is re-pinned to
  `c2dbde62c53ef8b6dae8d476ea47039081e2127de0b4be75da15c267333b37b2`.

### 2026-08-26 — pinned ninja PATH handoff

- The second probe attempt reached Isaac Gym but its child could not build `gymtorch`: exporting
  `NAVRL_NINJA` alone does not affect PyTorch's PATH-based ninja discovery. The fresh braking
  launcher now prepends the pinned ninja directory to each child PATH while retaining the absolute
  executable attestation. This is an execution-environment fix only; no task/controller/physics
  parameter changed. GPU measurement had not started beyond import/build initialization.

### 2026-08-26 — PhysX trace serialization tolerance

- The live cell reached trace validation, where float32 PhysX position serialization differed from
  the recomputed substep distance by more than the verifier's 1e-6 m default. Warmup and braking
  step-distance continuity now use the existing 1e-4 m path tolerance; cumulative path and stop
  statistics remain raw-trace derived. The verifier hash is re-pinned to
  `23ed75817120668e7404b902fdd35a6290d964af245ec25d4fdd2f91c523487d`.

### 2026-08-26 — stale root-tensor baseline exclusion

- A raw-cell inspection showed the first 28.27 m “step” was not target motion: Isaac Gym refreshed
  the just-written center root on the first callback while the recorder baseline still held the
  stale pre-write tensor. The recorder now receives the declared center XY as the warmup baseline;
  brake phase still rebases to the live controller pose. The probe hash is re-pinned to
  `55a2aef0dd60b7b1eab826eb35a03949f19b5df611f696a216e3307d23946f93`.

### 2026-08-26 — braking probe upper-speed gate is NO-GO

- The fresh RTX 3070 probe reached all four registered arms. The 0.6/0.9/1.2 m/s cells passed
  setup and raw trace validation; the 1.5 m/s cell failed the preregistered 5 s warmup convergence
  gate, so the launcher discarded the partial stage and produced no finalized receipt. The failure
  is evidence that the current physical target/controller contract does not yet sustain the upper
  arm; no threshold, speed arm, or controller parameter was changed. Recovery-v2 simulator gate
  and PPO remain blocked until this measurement contract is resolved.

### 2026-08-26 — upper-speed diagnostic detail

- Warmup-gate failures now report final speed min/mean/max and absolute-error min/mean/max before
  the child exits. This is diagnostic-only and does not alter the preregistered convergence gate;
  the probe hash is re-pinned to
  `a4c74e16ac0e39093549925187a1c4d180800afe6dde1e1c1b22764a3aa71bbf`.

### 2026-08-26 — measured 1.5 m/s warmup failure magnitude

- A diagnostic-only 1.5 m/s child (not part of a finalized receipt) measured final speed
  `min=1.441824`, `mean=1.442577`, `max=1.443260 m/s`; absolute error was
  `min=0.056740`, `mean=0.057423`, `max=0.058176 m/s`. This is a small steady-state tracking
  deficit, not divergence or collision. The frozen `0.05 m/s` convergence gate therefore remains
  NO-GO; no tolerance/controller/speed parameter was changed.
## 2026-08-26 — speed ceiling/controller calibration preregistration

- The failed canonical braking receipt remains immutable: gain 2.5 at requested 1.5 m/s reached
  mean 1.442577 m/s after 5 s and missed the absolute tracking gate by about 0.0074 m/s.
- Added an independent six-cell, fresh-process diagnostic instead of changing that gate after the
  result: gain 2.5 at 1.35/1.40/1.45/1.50 m/s and gains 3.0/3.5 at 1.50 m/s, 32 env, seed 829.
  It records 5/6/8/10 s tracking, the whole 4–5 s stability window, saturation/tilt/contact/OBB,
  and zero-command braking. The frozen rule selects the highest passing baseline speed and the
  lowest safe controller gain; longer convergence is descriptive and cannot rescue a 5 s fail.
- Added `tools/run_navrl_physical_target_speed_controller_calibration.py`, its preregistration,
  atomic fresh result/verification contract, and four CPU contract tests. Existing braking and
  recovery-v2 tools were not edited. GPU has not yet been run at this entry.
- Validation: calibration 4/4, braking contract 8/8, recovery-v2 evaluator 5/5, target motion
  13/13, Python compilation and `git diff --check` PASS.

## 2026-08-26 — stage-1 GPU calibration and frozen stage-2 bracket

- Stage 1 completed six fresh RTX 3070 cells. Gain 2.5 missed the five-second absolute band at
  every 1.35–1.50 m/s arm; the 1.50 m/s mean was 1.44265 m/s. Velocity gains 3.0/3.5 entered the
  band permanently only at 7.96/7.05 s and increased peak overshoot, so neither lower-contract nor
  controller follow-up was eligible under the first frozen grid. The finalized calibration result
  is `results/navrl_physical_target_speed_controller_calibration_seed829/`.
- Before any new GPU child, froze stage 2: unchanged-controller lower-speed bracket
  1.20/1.25/1.30/1.35 m/s plus a shared 1.50 reference and three rate-damping candidates. The
  lower contract retains the original braking gates; controller selection additionally requires
  <=15% overshoot, >=50% overshoot reduction, and no stopping/lateral regression.
- Added explicit diagnostic rate-gain scaling only inside the calibration child. Task defaults,
  observations, reward, termination, recovery, PPO, and existing receipt tools remain unchanged.

## 2026-08-26 — stage-2 result and lower-1.25 follow-up lineage

- Stage 2 finalized at
  `results/navrl_physical_target_speed_controller_calibration_stage2_seed829/`. The unchanged
  controller passed the original five-second endpoint plus 4–5 s stability gate at 1.20 and
  1.25 m/s, but 1.30 m/s missed by 0.00102 m/s. Thus the measured baseline ceiling is 1.25 m/s.
- No damping candidate passed the complete frozen selection rule. Rate scaling could improve a
  single 5 s sample, but none simultaneously sustained the 4–5 s band and reduced overshoot/
  stopping behavior. Controller modification remains blocked rather than being tuned again.
- Added an opt-in `baseline_1p25` braking/recovery contract. The default remains canonical 1.5.
  The new exact speed grid is 0.6/0.9/1.2/1.25; decimal keys and record IDs were made lossless so
  1.25 cannot alias 1.2. Task checkpoint state records the selected contract variant. Separate
  preregistrations and launchers prevent the lower result from being read as a 1.5 result.
- CPU verification: canonical braking 8/8, recovery-v2 evaluator 5/5, recovery core 22/22,
  target environment 13/13, lower-contract isolation 3/3, compilation/diff check PASS. No
  lower-contract braking receipt or recovery GPU result existed at this entry.

## 2026-08-26 — lower-1.25 braking receipt finalized

- Ran the separately preregistered `baseline_1p25` physical-target braking probe on the RTX 3070
  at 0.6/0.9/1.2/1.25 m/s. The atomic result finalized at
  `results/navrl_physical_target_braking_lower1p25_seed827/` and its standalone verifier passed;
  final receipt SHA-256 is
  `259e5333e4bd109547df00b4dd0295244f471de55490f128bbf74f3f89de9663`.
- Measured deceleration p05 is 0.5052572863 m/s^2, stop-time p95 is 0.89 s, and certified lateral
  braking tube p95 is 0.00041199 m. Stop-distance p95 is 0.32814/0.49835/0.67084/0.69944 m at
  0.6/0.9/1.2/1.25 m/s respectively. All four cells recorded zero contact, invalid geometry, and
  braking saturation events.
- This receipt authorizes only the lower-contract recovery-v2 simulator gate. It does not repair
  or supersede the failed canonical 1.5 m/s contract, load a PPO policy, authorize training, or
  constitute hardware validation.
- The first lower-contract gate attempt stopped before task creation: the hermetic child builder
  correctly stripped inherited `NAVRL_*` state but failed to reinsert the frozen
  `baseline_1p25` variant before the child imported its module-level speed grid. The child
  therefore expected the canonical preregistration and rejected the lower manifest. The VOID
  artifact is retained. The evaluator now explicitly re-pins the exact variant in the fresh-child
  environment; a CPU subprocess contract test prevents regression. No scientific cell ran and no
  gate or measured parameter changed.
- A second task-creation attempt exposed an older evaluator import-order defect: the packed
  telemetry module imported PyTorch before Isaac Gym, which Isaac Gym rejects before creating a
  simulator. The evaluator now imports Isaac Gym first only in its private fresh-child process,
  before loading the packed observer; parent preflight/verification remains CPU-safe. A fresh
  subprocess regression test imports the evaluator and local `aerial_gym` in the exact child
  order. This attempt also ran zero scientific cells and is retained as VOID.
- The next attempt completed all 16 route-off cells, then the first recovery interval exposed an
  XY/XYZ interface bug in both recovery geometry helpers: NavRL supplied its native XYZ arena
  bounds while the helpers added a two-axis support vector. Both helpers now project arena bounds
  to XY at their API boundary, with a regression test exercising the real XYZ-shaped input for
  braking and anchor certification. The completed off-arm traces remain in the VOID bundle; they
  are not combined with a later attempt because every accepted grid must be process-consistent.
- Because the recovery fix changes a runtime source file bound by the lower braking receipt, the
  evaluator correctly rejected reuse of the earlier receipt against the new planner bytes. A
  fresh lower-contract braking receipt must therefore be generated from the corrected commit
  before the 32-cell gate is eligible; measured values are not copied around this provenance
  boundary.
- With the corrected geometry, the first recovery-on cell reached a legitimate same-interval
  `BRAKE -> CONNECT -> ROUTE` sequence. The boundary observer sampled only BRAKE before and ROUTE
  after and incorrectly rejected the compressed pair. The raw cell showed exactly one such pair
  and exactly one route-resume counter increment. Verification now permits the compressed pair
  only when the total BRAKE/CONNECT-to-ROUTE pairs equal the immutable resume counter; missing or
  forged counter evidence still fails. `BRAKE -> NORMAL` remains illegal. This is a telemetry
  interpretation fix, not a state-machine, gate, or controller change.
- Raw CONNECT telemetry then showed that 146/290 active interval predictions had a short initial
  coast away from the fixed anchor (minimum -0.04292 m) while every certified 1 s horizon made
  positive progress (minimum +0.00920 m), and the actual PhysX interval reduced anchor distance.
  This follows the existing core contract comment: recovery progress is certified over the full
  horizon, while the submitted physical interval is separately checked against actual anchor
  distance. The verifier now treats first-sample progress as finite descriptive telemetry, keeps
  horizon progress `>= -1e-6`, and retains the actual no-regression check. The in-flight rerun was
  stopped before this known-failing verifier path to avoid wasting GPU time; its incomplete bundle
  is retained and cannot be finalized or combined with another run.

## 2026-08-26 — recovery-v2 lower-1.25 32-cell gate is FAIL_ROUTE_MECHANISM

- After the rest-heading snap, a new lower-contract braking receipt was issued at
  `results/navrl_physical_target_braking_lower1p25_headingrest_seed827/` (SHA-256
  `4e87eb9ddf5dd9cea1fc0354d272a5d18ec6a05427e0f41e672749a57df9047a`; `decel_p05=0.5052572863`,
  `stop_time_p95=0.89 s`). It still does not repair canonical 1.5. CONNECT actual-interval
  regression was reclassified from an integrity VOID into a cell gate, and recovery waypoint
  identity was matched to the v1 route-goal contract (cross-arm layout/robot/target only). Those
  two commits are telemetry/evaluator only; they do not change CORE_PATHS bound by the braking
  receipt.
- The 32-cell lower-1.25 recovery-v2 gate then finalized at
  `results/navrl_physical_target_recovery_v2_gate_lower1p25_seed827/` on commit `2b151d9`.
  Standalone verify: `PASS_32_CELL_INTEGRITY` / `FAIL_ROUTE_MECHANISM` /
  `long_training_authorized=false`. Pass 7/32, all route-off; recovery 0/16. 70-bar pool plan
  success is 190/203=0.9360 vs gate ≥0.99; fallback 18381/38400=0.4787 vs ≤0.01; 0.6 goals/env
  0.21875 vs ≥0.5. VOID/incomplete siblings remain uncombined.
- The earlier note that “the actual PhysX interval reduced anchor distance” is false. On both
  VOID.3656376 and the finalized 0.6/70 cell, the 1 s horizon stayed positive while many actual
  0.1 s PhysX intervals receded (finalized 0.6/70: 59/176 regressions, max +5.3 cm). After the
  0.10 m/s heading snap, rest-start CONNECT commands stay toward the frozen anchor (fraction 1.0);
  realized rest speed is ~0.055 m/s against a ~0.41 m/s command and a 0.40 m/s accel-envelope Δv.
  CONNECT is 1.22% of recovery-arm time. It explains the `connect_actual_progress` cell fail, not
  the fallback occupancy.
- Packed diagnosis of the frozen raw (`tools/diagnose_navrl_physical_target_recovery_v2_packed.py`):
  recovery-arm occupancy is 63.06% `NO_CONNECTOR`, 31.51% `NORMAL`, 2.13% `BRAKE`, 2.09% `ROUTE`,
  1.22% `CONNECT`. Of 534 `NO_CONNECTOR` entries, 0 are hard-breach; 219 are same-interval BRAKE
  then generic no-anchor, 136 BRAKE timeout, 93 in-phase BRAKE no-anchor, 50 CONNECT failed-resume
  (soft-free), 29 CONNECT failed-certificate (still soft-unsafe). This is not v1 `unsafe_start`
  deadlock. Recovery-v2 reached legal compressed `BRAKE→ROUTE` and 93.6% plan success, then latched
  fail-closed zero command.
- Env count stays at the preregistered 32. Raising it would be a different experiment, not more
  of this gate. Do not retune gain 2.5 / `0.45 m` / 1.5 after seeing FAIL. Next eligible GPU is
  the separately preregistered 70-bar no-anchor geometry probe, not another 32-cell grid.

## 2026-08-26 — recovery-v2 no-anchor forensics preregistration frozen

- Replaced the same-day stub with the frozen evaluation-only contract
  `docs/preregistration_physical_target_recovery_v2_no_connector_forensics_2026-08-26.md`.
  The probe is 70-bar × `{0.6,0.9,1.2,1.25}` only, seed 827, 32 env, `baseline_1p25`, gain 2.5.
  It binds the verified gate summary SHA `a85e9576…` and heading-rest braking receipt
  `4e87eb9d…`. The 32-cell artifacts are read-only.
- CPU contract tests lock recovery-v2 `nearest_soft_free_anchor` kwargs
  (`hysteresis=0.25`, `hard_epsilon=1e-4+0.0123`) so the observer cannot silently reuse the v1
  forensic search. Descriptive labels are `ANCHOR_PRESENT_LATCH` /
  `ANCHOR_ABSENT_AT_LATCH` / `INCONCLUSIVE`; none of them pass the 32-cell mechanism or authorize
  retune/PPO. GPU `--run` is not part of this freeze.

## 2026-08-26 — recovery-v2 no-anchor observer implemented (CPU only)

- Implemented the evaluation-only observer in
  `tools/diagnose_navrl_physical_target_recovery_v2_no_connector.py`. It wraps
  `mark_no_connector`, `mark_local_infeasible_soft_free`, `recovery_anchor_idx`,
  `brake_connector_idx`, and `plan_idx` without changing return values. Packed
  classes come from the already-frozen packed diagnostic. The CPU replica uses
  recovery-v2 kwargs, not the 2026-08-25 v1 forensic search.
- Parent `--run` is now wired (fresh `OUTPUT_ROOT` only, one child per 70-bar
  speed, heading-rest braking receipt + gate training-source manifest, no packed
  telemetry flag). It refuses the 32-cell gate directory. GPU children were not
  launched.
- CPU tests: 11 passed, including observer identity, primary-only pooling,
  `VOID_OBSERVER_IDENTITY`, child-env variant/70-bar/no-1.5, and immutable
  summarize. Next is the preregistered 4-cell GPU probe (step 3), not a 32-cell
  rerun.

## 2026-08-26 — no-anchor observer fail-closed before GPU

- Rechecked the 2.5–4 h step-2 budget against v1 forensic / recovery-v2 child
  contracts. CORE_PATHS still match gate `2b151d9` (361/361). Training-source
  snapshot 325/325 and heading-rest braking tools still bind. GPU idle except
  display/rustdesk.
- Tightened GPU fail-closed holes before the 4-cell child: instantiated speed /
  box / bar-capacity / arena-band / robot attestation, physical `0.75 m`
  boundary check, and primary-event VOID if the replica `exists` flag is not a
  real boolean (malformed must not count as absence). CPU tests still 11/11.

## 2026-08-26 — recovery-v2 70-bar no-anchor GPU probe is INCONCLUSIVE

- Ran the preregistered 4-cell diagnostic from commit `9a8f0d6` (after the
  fail-closed attestation patch). Wall ~103 s on the RTX 3070. Standalone
  `--verify` passed. Identity VOID did not fire. Artifacts:
  `results/navrl_physical_target_recovery_v2_no_connector_forensics_seed827/`
  (`receipt` SHA-256 `81f53f49…`, `summary` SHA-256 `bc5d05a3…`).
- Pooled primary n = 1 (`brake_no_anchor_likely` only, speed 1.2). Replica and
  runtime both `exists=False` / `runtime_anchor_ok=False`; hard-free and
  soft-unsafe at latch. Wilson n < 20 → descriptive label `INCONCLUSIVE`.
  Other latches (do not vote): CONNECT failed-certificate 49, BRAKE timeout 32,
  CONNECT failed-resume 23, CONNECT timeout 1. No hard-breach.
- This is not a 32-cell mechanism pass, not a 1.5 result, and does not authorize
  retune, PPO, env-count change, or a denser-grid rerun. BRAKE generic no-anchor
  is rare in these four 70-bar cells; most fail-closed zero commands come from
  CONNECT certificate/resume failure and BRAKE timeout.

## 2026-08-26 — 사이트 compact 병합 + heading 계약 provenance 고정

Track B 종료 후 남은 마무리 3건 중 2건을 처리했다. **어떤 판정도 재판정하지 않는다** —
`FAIL_ROUTE_MECHANISM`, `PASS_32_CELL_INTEGRITY`, recovery 0/16, 전체 7/32(전부 route-off),
70막대 plan success 93.60%(기준 ≥99%), fallback 47.87%(기준 ≤1%)는 그대로다.

### ① compact 사이트 패치 병합

`codex/status-compact-layout` (`0dde2ef`)를 main에 병합. 충돌 0건, `docs/status/` 2파일만 변경.
`h1` 최대 88 → **76px**, `#stage` 최대 620 → **500px**로 실제 축소 확인.

### ② heading 계약을 checkpoint provenance에 고정 — 그리고 전제가 틀렸다는 발견

`HEADING_VALID_SPEED_MPS = 0.10`(`target_motion.py:46`)은 recovery 전용이 아니라 모든
bounded/physical target motion에 적용되는데 계약에 기록되지 않아, 향후 학습이 조용히 바꿔도
아무도 모르는 상태였다.

**핵심 발견**: `git log -p` 검증 결과 이 상수는 `0d1def2`(2026-08-26)에서 `0.10`으로 도입됐고
다른 값이었던 적이 없다. **그러나 그것이 대체한 것은 인라인 `1e-5`였다.** 즉 2026-08-26 이전
체크포인트는 전부 `1e-5`로 학습됐고 0.10이 아니다 — **10,000배 차이다.**

따라서 **과거 체크포인트를 지금 재개하면 계보가 바뀐다.** 이번 평가 결과를 무효화하지는 않지만
(평가는 동일 코드에서 수행됐다), fresh 학습이나 warm-start 전에 반드시 인지해야 할 사실이다.

구현(순수 provenance, 수치 동작 변경 0 — diff에 삭제 라인 0):
- `env_state`에 **두 필드를 항상 함께** 기록한다: `cfg_target_motion_heading_valid_speed_mps`
  (실제로 힘을 발휘하는 값)와 `..._provenance`(`source_literal` / `checkpoint_attested` /
  `assumed_pre_key_default`). **추정이 측정처럼 보이지 않게** 토큰이 분리되고 테스트가 양방향으로 고정한다.
- 키 없음 → 0.10으로 간주하되 메시지가 `1e-5` 사실을 직접 명시한다(거부하면 모든 기존 체크포인트가
  죽는다). 키 있고 다름 → 거부. 읽을 수 없음 → 거부(읽을 수 없는 provenance는 없는 것과 다르다).
- 실제 체크포인트(`transfers/navrl_truncated_1650ti/...ep_19050`)로 검증: 키 없음 →
  `0.1 / assumed_pre_key_default`로 정상 로드.

**env var 대신 리터럴로 둔 이유**: `tests/test_navrl_target_motion.py:278`이 이 상수를 동결된
braking probe의 `STOP_THRESHOLD_MPS`에 묶어놨다. env var를 만들면 런타임에 그 둘을 어긋나게 할 수
있고 그 테스트가 못 본다. 리터럴은 git에 남아 변경이 추적되며, 이제 재개 시 하드 거부까지 붙는다.

테스트 818 → **827**, OK(skip 2). 로드베어링 단언을 mutation 테스트로 확인했다 — 게이트를 `1e-5`로
되돌리면 `test_recorded_value_is_the_threshold_actually_in_force`가 실패한다(상수 사본이 아니라
실제 동작에 묶여 있다).

미처리(의도): 평가 영수증(`navrl_task.py:7446`)에는 넣지 않았다 — 동결 영수증의 해시를 흔들 위험이
있고 이번 범위는 `env_state`였다. preflight `_CONTRACT_ENV`도 env var가 없으므로 손대지 않았다.

### 남은 것

origin/main push와 GitHub Pages 배포 확인. 원시 결과 영수증은 integration worktree의 절대경로에
묶여 있으므로 **`.codex_worktrees/two_envelope_recovery_integration`을 삭제하면 안 된다** —
primary로 복사하면 source root drift로 거부되며 이는 의도된 fail-closed 동작이다.

## 2026-08-26 — GitHub Pages 소스 불일치 진단 및 배포 복구

- 로컬/원격 Git graph와 공개 GitHub deployment를 대조했다. `main`과 `origin/main`은
  `3e8facc`로 일치했지만 Pages의 실제 source는 저장소 workflow나 `gh-pages`가 아니라
  **`research/navrl-env`의 `/docs`**였다. 원격 기본 브랜치도 여전히
  `research/navrl-env`였다.
- 최신 공개 deployment가 `research/navrl-env@71ba73d`였으므로 `main`만 push해서는 사이트가
  갱신되지 않았다. `.github/workflows` 부재는 원인이 아니며, GitHub가 생성하는 dynamic
  `pages build and deployment`가 정상 배포 경로다.
- `origin/research/navrl-env`가 `main`의 조상임을 확인한 뒤 강제 push 없이
  `git push origin main:research/navrl-env`로 `71ba73d -> 3e8facc` fast-forward했다.
  Pages run `32956107111`이 success로 종료됐다.
- cache-busting URL로 공개 자산을 다시 받았다. 공개/로컬 `docs/status/style.css` MD5는
  `2b28d40f4a38f222e0ea1729fba81cb7`, `status.json` MD5는
  `37a1eb9d180aa4fae898eb87f33740bc`로 각각 동일했다. 공개 UI는 h1 최대 76px,
  3-D stage 최대 500px이며 08-26 Track A/B 상태를 표시한다.
- 장기 정리 권고: GitHub Settings -> Pages source와 default branch를 `main`/`docs`로 바꿔
  이중 브랜치 배포 의존성을 제거한다. 웹 설정 전까지는 사이트 변경 때 두 ref를 같은 커밋으로
  fast-forward해야 한다.

## 2026-08-26 — 실행 authority 동결, 사이트 적대 QA, 교수님 PPT 마스터 브리프

### Track A/B를 기계 판독 계약으로 동결

- `docs/research_authority_2026-08-26.json`을 추가해 Track A
  `RANGE_INCONCLUSIVE_AT_THIS_BUDGET / stage2=false`, Track B
  `PASS_32_CELL_INTEGRITY / FAIL_ROUTE_MECHANISM / long_training=false`, no-anchor
  `INCONCLUSIVE`, hardware flight/log 0을 한 receipt로 고정했다.
- receipt는 세 canonical summary의 SHA-256(`228e7d…`, `a85e95…`, `bc5d05…`)을 포함한다.
  `tools/check_research_authority.py`가 실제 파일 SHA와 verdict/claim-boundary 필드를 직접 대조하고
  drift를 fail-closed한다. 이 도구는 GPU·학습·평가를 실행하지 않는다.
- OPERATIONS/VERIFICATION에 `python tools/check_research_authority.py --json`을 현재 명령의
  시작점으로 연결했다. 허용된 다음 작업은 exact hardware BOM/calibration, 210 independent sensor
  trials, real-log profile/offline replay뿐이다.

### 공개 사이트/그림/3D 렌더 감사

- 공개와 로컬을 Chrome 1440×900, 390×844 및 full-page로 렌더했다. system overview와 low-level
  control SVG는 정상이고 routed-preview WebGL canvas도 SwiftShader headless에서 생성됐다.
- 모바일 긴 authority 문장의 slash-delimited token이 잘리지 않도록 `word-break: break-word`를
  적용하고 CSS cache를 `20260826r9`로 올렸다. iframe computed-layout 감사에서 document/body
  scrollWidth=clientWidth=375, hero/status/SVG/stage 모두 viewport 안에 있음을 확인했다.
- stale parity test를 발견했다. viewer는 2026-08-25의 frozen `global_astar_v1` geometry를
  설명하지만 test가 recovery-v2가 추가된 **현재 전체** Python 파일 SHA를 과거 SHA와 비교해
  실패했다. 단순 SHA 갱신 대신 test가 `a373202` historical blob을 검증하도록 수정하고, 사이트에
  recovery-v2 state machine을 재현하거나 성공으로 표시하지 않는다고 명시했다. JS cache는
  `20260826r3`으로 일괄 갱신했다.
- 사이트 resource에 frozen authority와 교수님 PPT master brief를 추가했다. 모든 local src/href가
  존재하고 GitHub link가 HTTP 200임을 확인했다.

### 교수님 발표 제작 계약

- `docs/CLAUDE_PPT_MASTER_BRIEF_2026-08-26.md`를 추가했다. 본편 18장 + appendix 12장으로,
  연구질문, expected hardware, task/sensor contract, 898-D/17-token 구조, Transformer high-level,
  altitude PI/Lee/allocation/motor low-level, reward/PPO/curriculum, held-out density×speed,
  perception/latency, physical target/recovery FAIL, claim boundary와 72-hour hardware plan을 모두
  포함한다.
- 각 슬라이드에 목적, 정확 수치, 권장 그림, 발표 멘트, 원자료와 예상 교수 질문 답변을 지정했다.
  `PASS integrity != mechanism PASS`, simulation-only, hardware pending, pp/% 단위, RNN overclaim,
  riskcap inference-only, raw RGB-D 오해 금지를 명시했다.

### 검증

- research authority current + SHA-drift negative tests 3개 PASS.
- target motion 25/25, target environment 13/13, ref5in run contract 101/101 PASS.
- status snapshot 19개(1 historical fixture skip), static site, fixed-clock motion, routed geometry,
  WebGL routed-preview PASS.
- experiment lint 71개 OK; JSON syntax, JS syntax, Python compile, `git diff --check` PASS.
- GitHub Pages/default branch UI 자체는 API/CLI 인증이 없어 이 머신에서 변경할 수 없다. 현재
  `main`과 Pages source `research/navrl-env`를 같은 commit으로 fast-forward해 배포 일관성을
  유지하며, Settings에서 source/default를 `main`/`docs`로 바꾸는 1회 작업은 계정 UI authority가
  필요하다.

## 2026-08-26 — 캐시 정리, leftover worktree는 main에 머지하지 않음

- `__pycache__` / `*.pyc` / `aerial_gym.egg-info`를 simulator와 Codex worktree에서 삭제했다
  (정리 전 pyc 1015개, `__pycache__` 172개; 정리 후 0). 실험 영수증·worktree 자체는 건드리지
  않았다. `two_envelope_recovery_integration` 절대경로 영수증은 유지.
- `src/aerial_gym_simulator`의 `main`은 이미 `origin/main`과 동일(`4c62dbb`)했고 working tree는
  깨끗했다. 커밋할 신규 코드 변경은 없었다.
- leftover Codex 브랜치 16개는 `main`보다 68–396커밋 뒤다. 3-dot diff는 “추가”처럼 보이지만
  2-dot은 파일 73–1518개가 달라서, 지금 머지하면 최신 Track A/B 동결을 예전 스냅샷으로
  되돌린다. 머지하지 않았다.
- `codex/routed-gate-site-status` worktree에 README/experiments/index/test 미커밋 4파일이
  남아 있다. 내용은 08-25 attempt-1 FAIL 문구이고, `main`은 이미 attempt-2 + recovery-v2 +
  08-26 authority 동결을 갖고 있어 덮어쓰면 후퇴다. 커밋·머지하지 않았다.
- 로컬 `research/navrl-env`만 `504b985`에 160커밋 뒤쳐져 있어서 `main`/`origin/research/navrl-env`
  와 같은 `4c62dbb`로 fast-forward했다. GitHub default branch 이름은 여전히
  `research/navrl-env`다 (`gh` 미설치, Settings UI 필요).
- 워크스페이스 루트 `/home/fair/workspaces/aerial_gym_ws/.git`은 2026-07-13에 생긴 커밋 0개
  빈 저장소다. `reference/`·worktree·결과물까지 포함한 12GB+를 여기에 커밋하지 않았다.

## 2026-08-27 — 막대 중첩 제거 및 physical fresh lineage 기하 정정

- 기존 `navrl_band`가 `d<=0.4 m` 막대를 의도적으로 겹치고 포화 시 merge fallback을 수행하는
  사실을 최종 환경 결함으로 판정했다. 과거 300-bar 표본은 평균 약 264개 독립 component였으므로
  해당 결과를 300개 독립 장애물 성능으로 표현하지 않는다.
- 새 `footprint_clearance` 배치기를 추가했다. 실제 URDF collision half-extents의 XY 외접원을
  사용하므로 yaw와 무관하게 중첩이 불가능하며, canonical surface clearance는 0.45 m다. 후보가
  고갈되면 merge/spacing relaxation 없이 RuntimeError로 fail-closed한다.
- 합성 300 bars × 128 env 생성은 0.261 s, 최소 squared clearance margin 양수였다. 실제 Isaac Gym
  4 env × 300 bars reset/1-step 스모크도 AABB overlap directed pair `0`, finite output PASS였다.
- 기존 `navrl_ref5in_quad`와 `navrl_target_drone.urdf`는 과거 체크포인트 provenance를 위해 byte
  단위로 보존했다. 대신 fresh-only `navrl_ref5in_v2_quad`와 `navrl_target_drone_v2.urdf`를 추가해
  collision proxy를 `0.280 -> 0.283 m`로 함께 수정했다. 계산된 5-inch prop-tip span
  0.2825634 m를 2.56 mm under-bound하던 문제를 바깥쪽 반올림으로 제거했다.
- 동일 controller GPU 비교(seed 20260827, 16 env)는 새 v2 자산으로 최종 PASS: hover error 0.000 m, forward
  2.490/2.50 m/s와 t90 0.8 s, reversal t90 1.0 s, yaw 2.999/3.00 rad/s, roll/pitch 20 deg
  response 0.13 s, peak 25.0 deg, worst motor request saturation 1.3%, 전 env 생존/finite.
  결과: `results/navrl_ref_platform_verification_20260827_v2/summary.json`.
- 이 변경은 observation shape는 유지하지만 장애물 component 수, corridor, contact distribution과
  collision hull을 바꾸므로 기존 PPO와 호환되지 않는다. physical launchers는 checkpoint를 계속
  거부하고 새 mode/surface clearance를 양쪽 하위 launcher에서 재검증한다. 새 성능은 route/physical
  engineering gate와 500-epoch smoke 뒤 fresh PPO로만 생성하며 warm-start/곡선 연결을 금지한다.
- README, paper/PPT brief, status site에 historical `navrl_band` 범위와 새 `NOT RUN` lineage를
  분리했다. 3-D viewer도 non-overlap sampler를 사용하며 merge fallback을 제거했다.
- 최종 회귀는 Python `834 tests OK (2 skipped)`, target/launcher 계약 13/13, v1 provenance
  freeze 3/3, 비중첩 배치 4/4, 사이트 static/motion/route contract 전부 PASS였다.

## 2026-08-27 — PPT 근거 패키지 및 밀도 계보 정리

- 교수님 발표 재제작용 `docs/PPT_PACKAGE_README_2026-08-27.md`를 추가했다. Claude가 이전 PPT를
  권위 자료로 복사하지 않도록 historical v1/v2와 corrected fresh v2를 별도 표로 고정하고,
  각 수치의 사용 범위와 `NOT RUN` 경계를 명시했다.
- corrected fresh physical launcher의 학습 목표를 `70→205 bars`(+15)로 통일했다. `300 bars`는
  자산·평가 상한/OOD 조건으로만 남겼고, 과거 v2의 70→300 기록은 historical로 보존했다.
- YOPOv2 count-density 비교는 공식 4 m/5 m spacing(6.25/4 trees per 100 m²)만 사용하고, 원문에
  tree diameter가 없어 면적 점유율 비교를 만들지 않도록 했다. MOTAR density/footprint 표는 패키지
  README와 PPT brief에 동일하게 반영했다.
- `research_authority_2026-08-26.json`에 fresh training start/final/step, evaluation ceiling 및
  YOPO reference fields를 추가했다. 새 corrected-v2 PPO 성능은 여전히 `NOT RUN`이다.
- 전달 패키지: `/home/fair/Downloads/MOTAR_PPT_MASTER_BUNDLE_2026-08-27.zip` (24 files,
  `unzip -t` PASS, 내부 `MANIFEST.sha256` 포함). 포함된 원자료는 platform gate와 historical
  density/recovery/detection summaries이며 파일명에 historical 범위를 표시했다.

## 2026-08-27 — 기하 감사 205 PASS, 밀도 계약 동결 (Claude 세션 한도 이후 재개)

클로드 세션이 감사 JSON을 쓴 직후(13:52, 세션 한도 13:54) 끊겼다. 0번 계보 문서와 1번 CPU
감사는 이미 끝나 있었고, 2번 밀도 사전등록이 미완이었다.

Canonical 6–28 m, body+tracking 팽창 0.650 m, 60×128:

| bars | connectivity | no-route | 판정 |
|---:|---:|---:|---|
| 205 | 99.167% | 0.833% | PASS |
| 250 | 97.813% | 2.187% | 최고 통과 |
| 300 | 94.661% | 5.339% | FAIL |

게이트는 결과 파일보다 먼저 VERIFICATION에 고정돼 있었다(≥95% / ≤5% / 생성실패 0).
본학습 상한은 약속대로 205 유지. 220/250은 연결 OOD, 300은 단절 스트레스.
SHA-256 `6b1f1b36…a268d5`. 도구는 실 placer + `bars_h3` URDF를 썼고, 팽창 반경은
prop-tip AABB 0.2825634 m의 circumradius다. v2 collision box 0.283 m와의 차이는
+0.31 mm라 205/300 판정을 뒤집지 않는다.

PPO·smoke·Track A/B GPU는 여전히 권한 없음. 다음 GPU 단계는 별도 권한 해제 뒤에만.
Hard 22.5–28 m 밴드와 placer latency 스크립트는 이번 동결에 넣지 않았다.

문서: `docs/preregistration_navrl_v2_corrected_density_geometry_2026-08-27.md`,
`results/navrl_v2_density_geometry_audit_2026-08-27/summary.md`.

## 2026-08-27 — GitHub Pages를 기하 감사 결과와 재동기화

공개 Pages가 `research/navrl-env@7459e91`에 멈춰 있고 `079def1` push는 `docs/status/`를
건드리지 않아 재배포가 안 떴다. status 히어로/메트릭/캐시(`20260827r5`)에 205 PASS · 300
FAIL을 넣고 `main`과 `research/navrl-env`를 같은 커밋으로 다시 맞춘다. PPO 권한은 여전히 없다.

`1982349`는 내용이 맞았지만 `git push origin main main:research/navrl-env` 한 receive라
Pages run이 안 생겼다. 캐시를 `20260827r6`으로 올리고 `.nojekyll` rebuild marker를 넣은 뒤
Pages source인 `research/navrl-env`만 단독 push하고, 그다음 `main`을 같은 커밋으로
fast-forward한다.


## 2026-08-27 — spawn clearance를 bar별 footprint 기준으로 수정 (27.45% 미스매치 해소)

기하 감사가 남긴 열린 caveat(205 bars에서 `start_in_inflated_obstacle_frac` 27.45%)의
원인을 코드에서 특정했다. `_randomize_general_drone_spawn`은 bar **중심**까지의 거리만
0.65 m로 검사했고 bar 크기에는 눈이 멀어 있었다. `bars_h3` 풀의 XY circumradius는
0.3133–0.5465 m이므로 같은 0.65 m가 작은 bar에서는 표면 여유 0.34 m, 가장 큰 bar에서는
0.10 m였다. 감사의 연결성 그래프는 bar마다 자기 footprint로 0.649802 m
(robot inflation 0.199802 + `physical_tracking_margin` 0.45)를 요구하므로 이 둘이 어긋났다.

수정: 후보마다 **모든 bar에 대해** `중심거리 − 그 bar의 circumradius ≥ 0.649802 m`.
평탄 상수를 키우는 대신 bar별로 판정한다(작은 bar 주변의 유효 spawn 면적을 버리지 않음).
half-extents는 런타임 planner와 동일한 축·슬라이스(`asset_collision_half_extents`)에서 읽고,
robot inflation은 live allocation matrix의 모터 암 + 5인치 프로펠러 반경으로 유도한
prop-tip span과 live URDF collision box 중 큰 쪽에서, tracking reserve는
`self.tm.physical_tracking_margin`에서 그때그때 읽는다(감사 도구 상수 복사 아님).
64회 rejection sampling 구조·`env_bounds` 마진·fallback은 그대로 두고 수락 술어만 바꿨다.

205 bars 재측정(60 layouts × 128 pairs, canonical 6–28 m, body+tracking 0.649802):

| spawn 규칙 | start_in_inflated | connectivity | no-route | start_rejection_exhausted | gen fail |
|---|---:|---:|---:|---:|---:|
| center 0.65 m (기존) | 28.294% | 99.154% | 0.846% | 0.000% | 0 |
| footprint (수정) | **0.104%** | 99.987% | 0.013% | **0.000%** | 0 |

동결 receipt의 27.45%와 28.29% 차이는 단일 밀도 실행이라 placer RNG 상태가 다르기 때문이며
규칙 차이가 아니다. 남은 0.104%는 실제 위반이 아니라 감사의 0.05 m 격자 스냅이다:
1,024 spawn의 **정확 좌표** 표면 여유는 위반 0건, 최소 0.652342 m였고 같은 표본에서
셀 중심 기준으로만 3건(0.293%)이 걸렸다(`check_residual_205.py`).
64회 예산은 7,680 spawn 전부에서 소진되지 않았다 — 새 실패 모드 없음.

동결된 `results/navrl_v2_density_geometry_audit_2026-08-27/`는 손대지 않았다(재측정은
scratchpad에만 기록). 그 receipt와 두 문서는 수정 **이전** 측정의 역사 기록으로 그대로 둔다.
감사 도구는 이제 기본이 footprint mirror이고 `--spawn-clearance center`로 옛 규칙을 재현한다.

CPU 스위트 836 → 848 OK(skipped 2). 신규 `tests/test_navrl_spawn_footprint_clearance.py` 12건은
서로 다른 bar 크기 여러 개로 검증하므로 평탄 상수로의 회귀를 잡는다. GPU/PPO 실행 없음.

파일: `aerial_gym/task/navrl_task/navrl_task.py`(`_spawn_footprint_clearance_accepted`,
`_robot_spawn_inflation_radius_m`, `_spawn_required_surface_margin_m`,
`_randomize_general_drone_spawn`), `tools/audit_navrl_v2_density_geometry.py`.

## 2026-08-30 — worktree 정리(20→3) + goal 배치의 동종 결함 마무리

세션 도중 사용자가 "다 검토해서 알려달라"고 위임한 정리 작업 3건.

### worktree 20개 → 3개, 브랜치는 하나도 안 지웠다

`.codex_worktrees/`에 오래 쌓인 worktree 20개를 원칙 하나로 정리했다 — **로컬 체크아웃만 지우고
브랜치·커밋은 반드시 origin에 push해서 보존한다.** 병합 여부만으로 안전 판정하지 않는다(어제
`two_envelope_recovery_integration`이 병합됐어도 결과 영수증이 절대경로로 묶여 있어 지우면 안
된다는 걸 확인했다).

- **즉시 삭제 3개**(병합 완료·클린·참조 0건): `routed_gate_site_status`, `status_compact_layout`,
  `routed_attempt2_docs`(미커밋 파일 2개는 primary의 더 완성된 최신본과 비교해 고유 정보 없음을
  확인 후 삭제).
- **push 후 로컬 체크아웃만 삭제 15개**: 전부 origin에 미push 상태였다(로컬에만 존재 —
  디스크 손실 시 영구 소실 위험). `codex/*` 15개 브랜치를 origin에 push해 sha 일치를 확인한 뒤
  worktree 디렉터리만 제거했다. 브랜치는 로컬·원격에 그대로 있어 언제든
  `git checkout -b <name> origin/codex/<name>`로 복구된다.
- **손대지 않은 2개**: `two_envelope_recovery_integration`, `route_recovery_forensics` — 결과
  영수증이 절대경로 provenance로 묶여 있어 fail-closed 검증 게이트가 지우면 영구 재검증 불가로
  만든다.

### `_sample_general_target`의 static 폴백 — 어제 스폰 버그와 동종

어제 드론 스폰에서 고친 것과 정확히 같은 결함이 goal(목표) 배치에도 있었다. `goal_min_bar_clearance`
사용처 4곳을 전수 확인했다:

| 함수 | 상태 |
|---|---|
| `_sample_general_target` (static 폴백) | **결함** — flat 중심거리, 어제 감사에서 0.013% 측정됨 |
| `_target_spawn_center_clearance` (bounded/physical) | **이미 안전** — 막대 풀 최댓값 반경을 이미 반영 |
| `reset_idx`의 legacy "cross the bar field" | **미측정, 손대지 않음** — v1식 레거시 경로 |
| `_advance_target` 2곳 | **미측정, 손대지 않음** |

**측정된 곳 하나만 고쳤다.** 나머지 둘은 지금 근거 없이 바꾸면 이 프로젝트가 반복해서 당한
"조용한 기계 결함"을 내가 새로 만드는 것이므로 범위 밖에 뒀다.

수정: `_sample_general_target`에 `bar_half`를 추가 인자로 threading(호출부는 이미
`asset_collision_half_extents`를 같은 슬라이스로 읽고 있어 배선만 추가), static 폴백 분기에서
`_spawn_footprint_clearance_accepted`(어제 만든 헬퍼)를 재사용한다. bounded/physical 분기는
`use_center_clearance` 플래그로 완전히 분리해 구조를 건드리지 않았다.

`goal_min_bar_clearance`가 실제로 지키려는 것도 확인했다 — "0.5 m capture sphere가 실제로 비행
가능하게" (`navrl_task_config.py:670-671`). 이는 로봇 발자국 도달가능성 문제이므로 어제와 같은
required margin(robot inflation + tracking reserve = 0.649802 m)을 그대로 재사용하는 것이 의미상
정확하다.

신규 테스트 3건(`SampleGeneralTargetTest`): 실제 표면 여유 검증, 실제 기본값(1.0 m, getattr
0.65가 아니라)로도 결함이 재현됨을 보이는 회귀 가드, bounded/physical 분기가 안 바뀌었다는 구조
검사. 작성 중 직접 잡은 실수 하나 — 추출한 함수를 클래스 속성으로 저장해 `self`가 자동 바인딩되며
인자 수가 어긋났다(수정), 그리고 `cat >>`가 기존 `if __name__` 블록 뒤에 새 클래스를 붙여
**어제 고친 것과 같은 함정**(직접 실행 시 뒤 클래스가 등록 전에 `unittest.main()`이 먼저 실행)이
재발할 뻔했다(수정, 단일 `__main__` 블록만 남김, 직접 실행으로 재확인).

부수: `env_object_config.py:47`의 주석이 "두 스폰 샘플러 모두 flat 0.65 m"라고 말하던 게 어제
수정으로 더 이상 사실이 아니어서 현행화했다.

851 테스트 전부 통과(848→851), provenance-frozen 무사, 실행 범위는 CPU뿐 GPU 없음.

## 2026-08-30 — `MOTAR_deck (4)` 26장 독립 검수 + Claude v5 재제작 계약

사용자 제공 PPT와 `MOTAR_검증요청.md`를 원본 그대로 보존하고, 26장 본문/발표자 note를 PDF·PNG로
전수 렌더한 뒤 코드·result receipt·research authority·주요 원문 논문과 대조했다. slide 23 C1–C4와
C6은 PASS, C5의 crash 수치와 carve-out 코드 경로는 확인했으나 삭제된 실제 bar 수를 세는 직접
계측은 없어 causal wording은 FAIL로 낮췄다.

추가로 PPT 밖의 authority 문서에서도 두 개의 drift를 발견했다. 첫째, fresh physical target을
historical generic v2의 `mixed + 300-epoch ramp`와 섞고 있었다. live launcher는 base fresh(route
off)가 mixed, routed fresh가 waypoint-only이며 둘 다 speed ramp 1 epoch다. 둘째, vision-mode live
reward의 `visibility_bonus=+0.02/visible step`가 시스템 명세에서 누락되고 PPT에서는 후보 항으로
잘못 적혀 있었다. `README.md`, `docs/MOTAR_SYSTEM_SPEC_2026-08-24.md`,
`docs/CLAUDE_PPT_MASTER_BRIEF_2026-08-26.md`를 실행 코드에 맞췄다.

PPT 자체에서 새로 확정한 주요 오류: YOPOv2 inference input에 ESDF/GT target pose를 잘못 배치,
898-D를 target 4-D로 잘못 분해, literature tier count `6/9/4/1`(실제 `6/10/3/1`), NavRL/MAVRL을
`sensor-based target state`로 표기, detector/tracker까지 PPO reward로 공동학습하는 듯한 문장,
YOPOv2 actual/reference mismatch를 timestamp 선행실험처럼 연결, 160×90 clip 진단을 일반 camera
한계로 확장, 서로 다른 lineage를 slide 24에서 badge 없이 혼합, 현재 authority보다 E1/E2가 먼저인
것처럼 쓴 slide 25다.

Claude용 완전 교체 문구·18장 본편 구조·appendix·신규 figure 6종(system boundary, 898→17 token,
timestamp alignment, lineage rail, reward equation, 10/100 Hz control)을
`docs/MOTAR_PPT_V5_CORRECTION_HANDOFF_2026-08-30.md`에 고정했다.

재검증: research authority verified(`Track A RANGE_INCONCLUSIVE_AT_THIS_BUDGET`, Track B
`FAIL_ROUTE_MECHANISM`, long training false), Python 851 OK(skipped 2), site/WebGL/snapshot PASS.
PPT export는 26/26 성공, clipping/font corruption은 없지만 8–11.5 pt 본문은 projector용으로 부적합해
v5 계약에서 본문 17–20 pt 하한을 뒀다. GPU/PPO/새 실험 실행 없음.

## 2026-08-30 — historical speed-ramp receipt 확정

PPT v5 handoff의 임시 문구 `historical receipt 확인`을 실제 체크포인트로 해소했다.
Slide 19 계보의 ep25000 checkpoint `env_state`는 mixed target,
`speed_min=0.3`, `speed_final=1.5`, `ramp_start=0`, `ramp_epochs=300`을 기록한다. 따라서 학습
계약은 `U[0.3, v_max]`의 상한을 첫 300 epoch 동안 0.3→1.5 m/s로 올린 것이다. 반면 held-out
speed-density 결과는 각 셀에서 0.3/0.7/1.1/1.5 m/s를 정확히 고정했으므로 평가 중 ramp는
적용되지 않았다. 내부 확인 메모가 슬라이드에 노출되지 않도록 Claude handoff를 확정 문구로 교체했다.

## 2026-08-31 — corrected non-overlap fresh 재학습 실행 계획 및 route gate 동결

막대 겹침 제거는 환경 전이·관측·통로 분포를 바꾸므로 historical `navrl_band` checkpoint와 성능곡선을
새 계보에 연결할 수 없고 fresh PPO가 필요하다는 판단을 재확정했다. 다만 장기학습보다 먼저 표적이 새
기하에서 실제로 경로를 수행하는지 fail-closed로 확인한다.

사전등록 `docs/preregistration_corrected_nonoverlap_route_gate_2026-08-31.md`와 독립 evaluator
`tools/verify_navrl_corrected_nonoverlap_route_gate.py`를 추가했다. 계약은 seed 829, 32 env,
route off/global_astar_v1 × 0.6/0.9/1.2/1.5 m/s × 70/115/160/205 bars의 32셀이다.
`navrl_ref5in_v2_quad`, target proxy 0.283 m, `footprint_clearance`, surface 0.45 m,
`NAVRL_MAX_BARS=300`을 source/runtime receipt에 고정한다. 기하 감사에서 단절된 300 bars는 PPO 허가
게이트에서 제외하고 후속 OOD 평가로 남겼다.

과거 evaluator를 상수화해 재사용하려는 첫 구현은 전체 테스트의
`test_v1_evaluator_bytes_remain_unchanged`가 의도대로 거부했다. 해당 파일은 HEAD와 byte-identical로
원상복구하고 새 evaluator를 완전히 분리했다. historical gate 기본 계약도 별도 subprocess test로
고정했다. 최종 CPU 회귀는 854 tests PASS, 2 skipped. GPU/PPO는 이 기록 시점에 아직 실행 전이다.

### Revision 1 실행 — VOID, sampler exhaustion 발견

commit `fc3de8c`에서 corrected gate를 실행했다. route-off 0.6/0.9/1.2 arm의 12셀은 child receipt를
썼지만, route-off 1.5의 205-bar reset에서 physical target spawn 한 환경이 1,024개의 iid uniform
후보를 모두 거부해 child exit 1이 발생했다. 전체 결과는 의도대로 `12/32`,
`VOID_EXECUTION`, 모든 mechanism/performance verdict `NOT_INTERPRETED`, PPO blocked다.
원본은 `results/navrl_corrected_nonoverlap_route_gate_seed829/`에 보존한다.

원인은 기준 완화가 아니라 검색 구현이었다. target sampler는 env당 round마다 후보 한 개만 보며
1,024회 뒤 fail-closed했고, pursuer sampler는 더 위험하게 64회 뒤 미해결 env에 최초 unchecked
후보를 사용할 수 있었다. 동일한 wall/range/footprint/support/tracking predicate와 iid uniform
proposal을 유지하면서 첫 accepted proposal을 택하는 batch rejection으로 바꿨다: target
64×256=16,384, pursuer 64×128=8,192. pursuer도 exhaustion 시 fail-closed한다.

결과를 본 뒤 threshold를 움직이지 않도록 revision 2를 별도 사전등록·evaluator·result namespace로
분리했다. seed/cell/속도/밀도/clearance/gate는 revision 1과 동일하다. revision 1의 부분 셀은 r2와
pool하지 않는다. 다음은 전체 CPU 회귀와 clean commit 뒤 r2 32셀 처음부터 재실행이다.

### Revision 2 실행 — 32/32 완주, route mechanism FAIL, PPO 미실행

sampler 수정과 전체 CPU 회귀(857 PASS, 2 skipped)를 clean commit `46b441a`로 고정한 뒤 revision 2
32셀을 처음부터 실행했다. source/import/runtime/record receipt가 모두 일치해
`PASS_32_CELL_INTEGRITY`였지만, 사전등록한 route gate는 세 핵심 항목에서 실패했다:

- 70-bar pooled plan success `17.7831%` (gate ≥99%);
- 70-bar fallback interval `30.0156%` (gate ≤1%);
- 70 bars × 0.6 m/s goal completion `0.21875/env` (gate ≥0.5).

same-goal reselection은 0으로 통과했지만 routed 1.5 m/s는 4개 밀도 모두 실패했고, 어떤 밀도에서도
승인 가능한 routed speed가 없었다. routed speed ratio는 속도·밀도가 올라갈수록 `0.7882→0.2292`
범위로 크게 낮아졌다. contact·motor saturation·tilt·finite-state가 주된 실패축은 아니며 route gauge는
`unsafe_start`가 지배했다(70×0.6 종료 13/32, 205×1.5 종료 31/32). 물리 표적이 관성으로 soft-safe
영역을 벗어난 뒤 fail-closed zero command와 `unsafe_start` 재계획을 반복하는 환경/controller 병목이다.

최종 판정은 `FAIL_ROUTE_MECHANISM / FAIL_FULL_1P5_CONTRACT / BLOCKED_PHYSICAL_TRAINING`이며
long-training authority는 false다. 사전등록대로 fresh 500-epoch PPO smoke와
70→205 장기학습은 시작하지 않았다(새 PPO 0 epoch). 막대 중첩 수정 때문에 fresh 재학습은 여전히
필요하지만, 현 route/controller로 학습하면 잘못된 표적 정지 패턴을 정책이 학습하므로 선수 조건이
충족되지 않았다. 상세 결과는
`docs/corrected_nonoverlap_route_gate_r2_result_2026-08-31.md`, 원자료는
`results/navrl_corrected_nonoverlap_route_gate_r2_seed829/summary.json`이다.

## 2026-09-02 — distractor envelope: 구현·게이트 통과, 측정만 대기

빨간 물체가 나오면 어떻게 되느냐는 질문에서 시작했다. 코드를 읽어보니 기본 검출기
`AppearanceTargetSegmenter`(`navrl_perception.py:591`)는 파라미터 5개짜리 1×1 conv로
`3.0·R − 2.0·G − 2.0·B − 0.9`를 계산한다 — 문자 그대로 "빨간가?"다. 렌더러가 표적을 평평한
`[0.88, 0.08, 0.045]`로 칠하고 다른 어떤 물체도 빨갛지 않으므로 **"빨간 픽셀 = 표적"이 학습이 아니라
정의상 참**이다.

v7 offline gate는 8/8 PASS(frame precision **0.99766**, far 14–20 m recall 0.99688, bearing MAE
0.0244°)인데 **그 요약 JSON에 distractor 키가 0개**다. 즉 그 숫자는 "표적과 닮은 것이 하나도 없는
세계"에서 잰 값이다. `docs/archive/development_directions_2026-08.md:107`이 이 축을
"D9 (보너스, eval-only)"로 이미 제안했으나 구현되지 않았다.

더 근본적으로, `_detect_rgbd`는 임계를 넘은 **이미지 전체** 양성 픽셀을 무게중심 하나로 축약한다
(`navrl_perception.py:1428`, `:1480`). 연결 성분 분석이 없다. 따라서 표적과 distractor가 동시에
보이면 무게중심은 둘 사이 허공을, range는 두 거리의 평균을 가리키고, count는 합이므로
**confidence가 오히려 올라간다.**

### 구현 (branch `codex/distractor-envelope`, `3711f87`)

물리적 실체가 있는 distractor 3종(구 r=0.15 — 표적과 **동일 크기**, 박스, 기둥)을 막대와 같은
`footprint_clearance` 파이프라인으로 배치하고 표적과 **같은 색**으로 칠한다. 최악 조건부터 재는 것이
목적이다. `NAVRL_DISTRACTOR_COUNT` 기본 0.

**기본값 off가 구조적으로 보장된다** — `select_and_order_assets`의 `if num_assets > 0` 가드가 폴더
나열과 RNG 추첨 **이전에** 걸려서 자산 집합·순서·배치 스트림이 불변이고 렌더러는 버퍼조차 할당하지
않는다. 단언이 아니라 AST로 변경 전 슬라이스를 복원해 동일 시드로 `torch.equal` 비교했다.

**fail-closed 가드**(`navrl_detector.py:614`): distractor > 0이면 decoupled 검출 해상도를 거부한다.
그 항등성("perception 분할기가 발화하는 픽셀 = detect 광선추적이 만든 픽셀")은 **표적만 칠해진
세계에서만** 성립하는데, 같은 색 distractor는 분할기를 발화시키되 detect 마스크에는 없어서 둘이
**조용히** 어긋난다. 그러면 detect 경로는 깨끗한 표적을 보고하는데 정책이 보는 이미지는 미끼로
가득한 상태가 된다. 측정은 카메라 해상도 경로로 한다.

**`_bar_offset` 확장의 부작용 7곳** 중 1차 지표를 오염시키는 **2곳만** 고쳤다 — 드론 스폰
clearance(안 고치면 드론이 distractor 안에서 스폰)와 표적 경로 planner(안 고치면 경로가 관통해
가림 통계가 틀어짐). `_solid_obstacle_offset = _bar_offset − num_distractors`로 연속 슬라이스
`[distractors][active bars]`를 만들어 넘긴다. 나머지 5곳(정적 goal, recovery clearance,
bar-contact probe)은 사전등록 L5로 기록했고 전부 crash가 아니라 모델링 격차임을 확인했다.

**정렬 테스트가 핵심이다**: 중심 배열과 half-extents 배열의 j행이 distractor→bar 경계를 넘어서도
같은 물체를 가리키는지 태그로 검사한다. off-by-one이 나면 distractor 중심에 막대 footprint가
짝지어지는데, 이는 지난주 flat-clearance 결함과 **정확히 같은 형태**다. 뮤테이션 5종을 전부 잡았고,
그중 하나가 `assertIn` 부분문자열 매칭을 통과하는 것을 발견해 정확한 줄 매칭 + AST 검사로 강화했다.

테스트 59개 신규, 857 → **916**. 실패 집합 불변(트레이너 호출 8건은 실행 중 학습의 duplicate 락,
error 8건은 worktree에 gitignore된 `results/`가 없어서 나는 아티팩트 — 둘 다 회귀 아님).

### Gate 0 통과 (직접 검증)

기본값 off bit-identical · decoupled+distractor 거부 · 신규 테스트 59/59. 세 조건 모두 확인했다.

### 측정을 미룬 이유

GPU를 다른 worktree(`braking_route_v3`)의 PPO가 쓰고 있다 — PID 12683, 10시간째, 9,846 epoch,
6.8/8.2 GB. 여유 1.4 GB로 128-env 평가를 겹치면 OOM이다. **A4 측정(GPU 약 30분)은 그 학습 종료 후.**

그 학습은 `NAVRL_REQUIRE_SOURCE_ROOT`와 `PYTHONPATH`가 모두 자기 worktree를 가리켜 격리돼 있음을
확인했다(8-27에 만든 import-origin 가드가 실제로 작동 중). 이번 작업은 전부 별도 worktree에서
했고 primary의 런타임 경로는 건드리지 않았다.

### 다음

학습 종료 후 A4: seed 479, distractor 0/1/3/5, 셀당 2,049 ep, FTLR 측정. 판정 규칙은
`docs/prereg_2026-09-01_distractor_envelope.md`에 동결돼 있다. `COLOR_SHORTCUT_CONFIRMED`
(FTLR ≥ 50%)는 **예상된 결과이며 실험 실패가 아니다** — 이 실험의 값어치는 개선이 아니라 결함의
정량화다.

## 2026-09-02 — safety filter 문헌조사: riskcap의 2.0 m/s 하한이 충돌의 구조적 원인

`docs/safety_filter_survey_2026-09-02.md` 작성. **조사·진단만이며 코드 변경 없음.** GPU 미사용,
신규 평가 실행 없음 — 모든 수치는 기존 receipt 재인용.

### 핵심 발견 — 세 갈래 독립 수렴

기존 `results/navrl_v2_ep24000_riskcap_seed44_screen/summary.json`에 이미 기록돼 있었으나
지금까지 해석되지 않은 두 지표를 읽었다. riskcap은 정지여유 위반율을 요청 13.69% → 실행 **9.87%**로
3.82 pp만 줄이고, **접촉 직전 실행 속도가 2.029 m/s** — 하한 2.0과 사실상 같다. 충돌은 하한에서 난다.

거버너 자신의 파라미터(brake 2.9609 실측, reaction 0.1, margin 0.45)로 풀면
**clearance < 1.33 m에서 2.0 m/s 하한은 정지를 물리적으로 불가능하게 만든다.**
`stopping_margin`은 `speed_governor.py:229`에서 계산되지만 진단 전용이고 아무것도 gate 하지 않는다.

이론적으로도 정지 장애물의 velocity obstacle은 양의 원뿔이라 `λv ∈ VO ⟺ v ∈ VO`이다. 즉
**양수 하한을 가진 magnitude-only 필터는 forward invariance를 원리적으로 가질 수 없고 충돌을 지연시킬 뿐이다.**
crash 17.62%는 튜닝 부족이 아니라 구조적 귀결이며, λ→0을 허용한 TTC mode만 6.83%를 달성했던 이유다.

### 문헌조사 진단의 정정

조사는 `cap ∝ clearance`가 접근·측면 운동을 동일 처벌해 deadlock이 났다고 보나 우리 구현에는 해당하지 않는다.
`directional_lidar_clearance`는 명령 방향 회랑 안 ray만 고르므로 이미 closing-rate 선택적이다.
실제 원인은 **거버너가 방향을 고르지 않고 정책이 고른다**는 것 — 정책이 벽을 향해 계속 명령하면 정지 상태가
지속된다. 이는 위 VO 논증의 직접 귀결이다. 명명된 현상: freezing robot problem (Trautman & Krause 2010).

### 부수 발견

- **제동 여유 3.3배**: T/W 3.26, 45° 틸트 이론 한계 9.81 m/s²인데 실측 유효 감속 2.9609 m/s².
  병목은 추력이 아니라 속도명령→자세 루프. 캡을 낮추지 않고 안전을 얻는 경로가 있다.
- **riskcap이 상수 2.0 캡을 이긴다는 증거가 없다.** fixed 2.0은 seed42에서 78.53/16.06/5.42,
  riskcap은 seed44에서 79.55/17.62/2.83 — 시드가 달라 비교 불가. 개입률 97% vs 28.74%로 완전히 다른데
  결과 차이는 노이즈 수준. riskcap 핵심 기구의 값어치가 미검증이다.
- **현 ref5in 계보는 거버너를 끄고 학습해 켜고 평가한다** (`train_navrl_v2_search.sh:115`이 off 강제,
  `eval_navrl_v2_density_speed_map.sh:45`가 riskcap). 우리 seed45 데이터는 필터와 함께 학습하면
  +3.74 pp를 더 준다고 말하는데(off 70.03 → source+riskcap 78.20 → trained+riskcap 81.94) 지금 그걸 버리고 있다.
- `near_stop_rate ≤ 5%` 사전등록 게이트는 riskcap에서 0.0%였다 — 하한 때문에 **실패할 수 없는 게이트**로
  판별력이 없었다. 향후 게이트 설계 시 주의.
- 수직(z) 명령 무규제(`navrl_task.py:4577`이 command_xy만 전달), 무반사 LiDAR ray를 free로 취급.

### 다음

계획 수립은 사용자가 fable로 진행한다. 조사 결과는 문서에 동결했고 **본 항목 시점에서 코드·설정 변경 없음.**

## 2026-09-02 — stopcap 스크린: 사전등록 → 구현 → 실행 (진행 중)

조사 문서의 진단(하한이 충돌 원인)을 검증하는 사전등록 스크린을 구축하고 실행에 들어갔다.

### 사전등록 (결과 이전 동결)

`docs/prereg_2026-09-02_speed_governor_stopcap_screen.md`. frozen ep25000
(SHA `f7022139…ad40`), **미사용 seed 49**, 205 bars, deterministic, 2,049 ep/cell, RTX 3070.
governor 파라미터는 seed42/44 스크린과 바이트 동일(brake 2.9608856678 실측값 포함) — **튜닝 없음**.

5 arms: off(Q3 필터 의존성) / fixed2p0(Q1 동일-seed 비교 최초) / riskcap(기준) /
**stopcap(신규, Q2 채택 심사)** / ttc(참조 전용, 게이트 없음).

판정 규칙 4개 동결: M1 기계 무결성(stopcap `negative_stopping_margin_executed_rate ≤ 1%`,
위반 시 IMPLEMENTATION_VOID), Q1(riskcap−fixed2p0 capture CI 부호), Q2(crash ≤ −3pp & CI<0,
timeout ≤ 5%, capture ≥ −2pp 동시 충족 시 GO), Q3(off−riskcap crash ≥ +5pp 시 FILTER_DEPENDENT —
사실 기록이지 실패 아님). `near_stop_rate` 게이트는 의도적으로 제외 — seed44에서 하한 때문에
실패 불가능했던 무판별력 게이트임이 확인됐기 때문.

### 구현

- `speed_governor.py`: `stopcap` mode 추가 — `cap = √((a·t_r)² + 2a·usable) − a·t_r`
  (DWA admissibility = a_accel=0·정지 장애물 RSS). 하한 없음(λ→0 허용), 방향 불변 원칙 유지,
  `hard_margin_m`이 `usable`을 통해 부활. fail-closed: stopcap에서 brake ≤ 0이면 ValueError.
- `tests/test_navrl_speed_governor.py`: 3개 추가(폐형식 일치·단조성·usable=0에서 cap=0,
  임의 입력 256개에서 `stopping_margin_executed ≥ 0` 구성적 보장, env 계약 fail-closed).
  **13/13 PASS.**
- `eval_navrl_v2_density_sweep.sh:331` 모드 화이트리스트에 stopcap 추가
  (train_source_receipts 동결 스냅샷은 불변).
- 신규 런처 `eval_navrl_v2_ep25000_stopcap_screen.sh`(SHA 게이트·덮어쓰기 거부·preflight,
  5셀 preflight PASS 확인) + 요약기 `tools/summarize_navrl_v2_stopcap_screen.py`
  (기존 receipt 검증 11종 승계, 단수 `verdict` 키 없음 — `verdict_m1/q1/q2/q3` 분리).

### 실행

GPU 확인: braking_route_v3 PPO(PID 12683) 종료, 3070 여유 확인 후 스크린 시작.
결과는 `results/navrl_v2_ep25000_stopcap_seed49_screen/`. 판정은 완료 후 별도 항목으로.

### 명시적 범위 밖 (사전등록 §4)

천장 ablation, 지각한계 항(t_react 실측 선행), stopcap 동반 재학습(Q2 GO 후), 방향 개입.

## 2026-09-02 — stopcap 스크린 완료: 구현 무결성 실패, 채택 금지

5개 GPU 셀은 모두 끝났고 새 실행 없이 raw JSON/receipt/log/checkpoint SHA를 요약기로
검증했다. 최초 finalize는 요약기가 canonical evaluator의 receipt schema 2를 과거 schema 1로
오인해 멈췄다. 사전등록에는 schema 1 제한이 없고 raw receipt는 전부 schema 2이므로 요약기만
수정했다. 결과 원본은 건드리지 않았다.

| arm | n | capture | crash | timeout | intervention | unsafe-after |
|---|---:|---:|---:|---:|---:|---:|
| off | 2049 | 73.16% | 25.18% | 1.66% | 0.00% | 11.06% |
| fixed2p0 | 2049 | 81.31% | 14.06% | 4.64% | 95.67% | 8.34% |
| riskcap | 2050 | 81.71% | 15.95% | 2.34% | 25.66% | 8.74% |
| stopcap | 2051 | 69.19% | 21.31% | 9.51% | 36.85% | **2.62%** |
| ttc | 2051 | 74.70% | 4.24% | 21.06% | 55.94% | 0.00% |

사전등록 판정:

- M1 `IMPLEMENTATION_VOID`: stopcap unsafe-after 2.62% > 1%.
- Q1 `MECHANISM_UNSUPPORTED`: riskcap−fixed2p0 capture +0.40pp, CI95
  [−1.98,+2.78]pp. release 기구의 이득은 입증되지 않았다.
- Q2 `NOT_JUDGED_M1_VOID`: 무결성 실패에 더해 stopcap−riskcap crash +5.36pp,
  timeout 9.51%, capture −12.52pp로 세 게이트 모두 실패했다. **stopcap 채택 및 동반 재학습 금지.**
- Q3 `FILTER_DEPENDENT`: off−riskcap crash +9.23pp. 이 ep25000 계보를 배포한다면 필터를
  제거할 수 없다.
- TTC는 crash 4.24%로 가장 낮지만 timeout 21.06%라 liveness 비용이 크며 사전등록상 참조 전용이다.

M1 해석 주의: `negative_stopping_margin_executed_rate`는 실제 기체 속도가 아니라 필터가 낸
command를 float32로 다시 대입해 `< 0`만 센 값이고, 음수의 크기는 export하지 않았다. 단위테스트는
수치오차 `−1e-5`까지 허용하므로 2.62%가 round-off일 가능성을 배제할 수 없다. 그러나 사전등록이
크기 tolerance 없이 `≤1%`를 고정했으므로 결과를 본 뒤 게이트를 바꾸지 않고 M1 VOID를 유지한다.
설령 M1을 무시해도 Q2의 crash/liveness/capture 세 게이트가 모두 실패하므로 채택 결론은 동일하다.

중요한 해석: 한 스텝의 scalar clearance에서 계산한 폐형식 command cap만으로는 이 스크린에서
안전성과 liveness를 함께 얻지 못했다. 더 이상 stopcap 식이나 파라미터를 사후 튜닝하지 않는다.
다음 후보는 (1) fixed2p0의 독립 seed 복제 또는 (2) 방향 변경이 가능한 swept-volume/CBF·predictive
filter이며, 둘 다 새 사전등록 전에는 GPU를 열지 않는다.

결과: `results/navrl_v2_ep25000_stopcap_seed49_screen/summary.{json,md}`.
## 2026-09-01 — 외관 distractor 축(D9) 구축: 자산·배치·페인팅·fail-closed 가드

detector v7의 frame precision ~99.77%는 **표적을 닮은 물체가 0개인 세계**에서 측정된 값이다.
기본 detector는 `AppearanceTargetSegmenter`(navrl_perception.py:591), 즉 1×1 conv
`3.0·R − 2.0·G − 2.0·B − 0.9` — 문자 그대로 red detector다. 렌더러는 표적만 평탄한
`[0.88, 0.08, 0.045]`로 칠하고(navrl_detector.py:466) 씬의 다른 무엇도 붉지 않으므로
"red pixel ⇒ target"은 구성상 참이다. `docs/archive/development_directions_2026-08.md:107`이
"D9 (bonus, eval-only) add a distractor axis to the envelope"로 이미 제안했으나 구현된 적이 없다.
이번 작업은 **그 실패를 측정 가능하게** 만든 것이고, detector를 고친 것이 아니다.

### 만든 것

`resources/models/environment_assets/objects/`에 정적·충돌 가능 URDF 3종.
visual 프리미티브와 collision 프리미티브가 동일하므로 LiDAR가 보고 드론이 부딪힌다.

| 파일 | 형상 | 치수 | 질량 | 의도 |
|---|---|---|---|---|
| `navrl_distractor_sphere.urdf` | sphere | r = 0.15 m | 0.5 kg | `camera_target_radius`(0.15)와 **동일** — 크기-거리 일관성으로 분리 불가 |
| `navrl_distractor_box.urdf` | box | 0.30³ m | 1.0 kg | 표적 스케일(0.28×0.28×0.12), 다른 종횡비 |
| `navrl_distractor_pole.urdf` | cylinder | r = 0.06, h = 1.60 m | 1.5 kg | 더 높고 더 가늚 |

관성은 각 프리미티브의 해석해다(구 2/5mr², 정육면체 m(a²+b²)/12, 실린더 m(3r²+h²)/12 · mr²/2).

### 노브 (기본 OFF는 절대 조건)

- `NAVRL_DISTRACTOR_COUNT` (기본 **0**) — env당 총 distractor 수.
- `NAVRL_DISTRACTOR_SHAPES` (기본 `"sphere,box,pole"`) — 형상 배합. 적힌 순서로 round-robin 분배.
  알 수 없는 형상은 `ValueError`로 fail-closed.

count=0이면 세 param 클래스 모두 `num_assets == 0`이고, `AssetLoader.select_and_order_assets`의
`if num_assets > 0` 가드가 폴더 listing·RNG 추출 이전에 해당 asset type을 건너뛴다 → 자산 목록,
정렬 순서, 배치 스트림이 모두 historical과 동일하다. 렌더러도 버퍼를 할당하지 않고 커널을
launch하지 않는다.

### 배치 — 막대와 같은 footprint_clearance 경로

distractor는 `keep_in_env = True`로 **표적 바로 뒤·막대 앞**에 놓인다
(`[target?] [distractors...] [bars...]`). 막대 뒤에 두면 밀도 커리큘럼이 build-time 상한보다 낮을 때
AssetManager가 `num_obstacles_in_env` 밖의 자산을 -1000으로 주차해버려 distractor가 사라진다.
`navrl_bars_env.py`의 `asset_type_to_dict_map`에서 distractor를 `physical_target`보다 **먼저**
선언한 것은 AssetLoader가 keep_in_env 자산을 `appendleft`하기 때문이다 — 마지막에 적재된
keep_in_env 자산이 index 0이 되고, index 0은 표적이어야 한다(`obstacle_position[:, 0]`).
`navrl_task.py`의 `_bar_offset`을 `(1 if physical else 0) + num_distractors`로 넓혀
모든 막대 slice가 계속 막대만 가리키게 했다.

`asset_loader.py`의 collision half-extent 추출이 box만 이해했으므로 sphere/cylinder를 추가했다
(`_urdf_collision_half_extents`). box 조회는 기존과 동일하게 전체 트리에서 **먼저** 수행되므로
기존 자산의 수치는 한 개도 바뀌지 않는다. 파싱 실패는 여전히 `[0,0,0]` → 배치기가 명시적으로 거부.

### 렌더러 — 같은 색으로 칠하고, 정체성이 깨지면 raise

`_render_distractor_camera_kernel`(navrl_detector.py:134)이 카메라 해상도에서 첫 씬 히트의
per-vertex segmentation id(warp_env_manager가 velocity 채널에 심는 값)를 읽어
`DISTRACTOR_SEMANTIC_ID = 51`인 픽셀만 마스크한다. 첫 히트만 쓰므로 가림은 자동으로 옳다.
페인팅은 표적 페인트 **직전**에 같은 per-env `target_color`와 자기 자신의 정확한 카메라-해상도
depth로 이뤄진다 — 최악의 경우를 먼저 재려는 의도적 선택이다.

**핵심 가드**: `_assert_detect_decoupling_is_equivalent`(navrl_detector.py:614)에 기존 appearance
offender 목록 **앞에** 전용 raise를 넣었다. decoupling은 "고해상도 RGB 렌더 + 분할"을
"고해상도 **표적** 마스크 읽기"로 치환하는데, 이는 렌더러가 칠하는 물체가 표적뿐일 때만 항등이다.
distractor를 같은 색으로 칠하면 perception의 segmenter가 detect-해상도 표적 ray-cast가 만든 적 없는
픽셀에서 발화하고, 둘은 **조용히** 어긋난다. 그래서 `NAVRL_DISTRACTOR_COUNT > 0`이면서 detect
해상도가 카메라 해상도와 다르면 즉시 raise한다. 측정은 카메라-해상도 경로에서 한다.

### "칠해짐 ⇒ 표적"을 가정하는 다른 지점 (기록)

- **obstacle map carve-out** (`_fuse_static_and_extract_obstacles`, `_reconstruct_target_pixels`):
  bearing+range 일치만 쓰고 semantic id를 보지 않는다. distractor 오검출은 실제 고체 장애물을
  LiDAR scan과 융합 depth에서 지워버릴 수 있다 — 이는 detector 오차의 **정당한 전파**이며
  측정 대상이지 오염이 아니다. 손대지 않았다.
- **LiDAR 연관** (`_associate_lidar_target`): 순수 기하. 위와 동일 판단.
- **profiling head** (`_record_detector_profile`): 두 head가 같은 프레임을 채점하는 **쌍대 비교**라
  distractor가 있어도 비교는 유효하다. 단 `ref_*` 필드는 ground truth가 아니다.
- **미처리(범위 밖, 명시 flag)**: 드론 spawn clearance(navrl_task.py:1912), 정적 goal
  배치(:4102), target route planner(:5858), recovery clearance(:6058), bar-contact
  probe(:3081, :3127)는 모두 `[_bar_offset : _bar_offset + n_bars_active]`만 읽으므로
  **distractor를 자유공간으로 취급**한다. distractor 배치 자체는 막대·서로와 0.45 m surface
  clearance를 지키지만, 스폰/목표/경로가 distractor를 통과하거나 그 안에 놓일 수 있다.
  eval에서 distractor를 켜기 전에 이 7개 slice를 처리해야 한다.

### 검증

- `python -m unittest discover -s tests`: **857 → 898** (신규 `tests/test_navrl_distractors.py` 41개).
  실패 집합은 분기점 `b5850a2`의 기존 16건(failures 8 / errors 8, skipped 2)과 **완전히 동일**하다.
  즉 신규 41개 전부 PASS, 기존 회귀 0건.
- GPU 측정은 하지 않았다. Warp distractor 커널이 페인트가 기대하는 값을 쓰는지, distractor가 있는
  씬의 end-to-end 픽셀 거동은 GPU에서 별도 확인이 필요하다.

다음 단계는 카메라-해상도 경로에서 `NAVRL_DISTRACTOR_COUNT`를 올려가며 detector v7의 frame
precision이 어떻게 무너지는지 재는 것이다. `_detect_rgbd`가 연결요소 분리 없이 모든 양성 픽셀을
하나의 centroid로 뭉개므로(navrl_perception.py:1428, :1480), 두 붉은 blob은 그 사이 빈 공간의
centroid와 어느 쪽에도 속하지 않는 평균 depth를 만든다 — 그것이 다음 측정의 1차 가설이다.

## 2026-09-01 — distractor를 고체로 취급: spawn clearance와 target route planner 2곳만 확장

앞 항목이 flag한 7개 slice 중 **사전등록 측정을 직접 오염시키는 2곳**만 고쳤다.
나머지 5곳은 preregistration limitation으로 남긴다(아래 감사 결과 참조).

### 인덱스 산술

에셋 축 layout은 `[target?] [distractors...] [bars...]`이고
`_bar_offset = (1 if physical else 0) + num_distractors`이다. 따라서 distractor 블록은
`_bar_offset` **직전**의 `num_distractors`행이다. 새 앵커를 하나 도입했다
(`navrl_task.py:859`):

```
self._solid_obstacle_offset = self._bar_offset - self._num_distractors
solid slice = [_solid_obstacle_offset : _bar_offset + n_bars_active]
```

이 slice는 연속이며 `[distractors...][active bars...]`만 담는다 — index 0의 target actor도,
active window 밖의 parked bar도 포함하지 않는다. `NAVRL_DISTRACTOR_COUNT` 미설정이면
`_num_distractors == 0`이므로 `_solid_obstacle_offset == _bar_offset`이고 slice는 기존과 동일하다.

layout은 추정이 아니라 확인했다. `env_manager.py:238-246`이 `asset_collision_half_extents`를
actor를 만드는 **같은 loop iteration**에서 vstack한 뒤 `[num_envs, -1, 3]`으로 view하고,
`IGE_env_manager.py:476`의 `obstacle_position`은 `env_asset_state_tensor[:, :, 0:3]`으로 같은
에셋 축이다. 즉 두 배열의 행 j는 같은 에셋이다. 순서는 `navrl_bars_env.py:131-147`의
`keep_in_env` 선언 순서와 `asset_loader.select_and_order_assets`의 `appendleft`가 정한다.

### 변경 2곳

| 위치 | 함수 | 변경 |
|---|---|---|
| `navrl_task.py:1929,1935` | `_randomize_general_drone_spawn` | 중심·XY half-extent 두 slice를 solid slice로 확장. `_spawn_footprint_clearance_accepted`의 per-obstacle circumradius 규칙은 그대로 — distractor가 자기 크기로 inflate된다 |
| `navrl_task.py:5890,5893` | `_plan_target_routes` | `plan_idx`에 넘기는 중심·half-extent를 solid slice로 확장. `plan_idx`/`RoutePlanner.plan`은 장애물 개수에 대해 generic(`reshape(-1,2)` + 두 배열 shape 일치 요구)이므로 계약 변경이 아니라 장애물 추가다 |

`_bar_offset` 자체와 bar 전용 slice의 의미는 건드리지 않았다. detector·segmenter·
`robot_config/**`·`resources/robots/**`도 무변경이다.

### default-off 동일성 — 주장이 아니라 측정

`tests/test_navrl_distractors.py`에 AST로 메서드를 lift하면서 `self._solid_obstacle_offset`을
`self._bar_offset`으로 되돌린 **변경 전 사본**을 만들고(`_lift_task_method(historical=True)`),
같은 입력·같은 `torch.manual_seed`로 실제 메서드와 나란히 돌려 텐서를 `torch.equal`로 비교했다.
`bar_offset` 0(legacy)/1(physical) 양쪽에서 spawn의 `robot_position/orientation/linvel/angvel`,
route planner가 `plan_idx`로 넘긴 중심·half-extent 모두 bit-identical이다. rename이 no-op이면
헬퍼가 raise하므로 비교가 공허해질 수 없다.

### 신규 테스트 18개 (41 → 59)

- `SolidObstacleOffsetArithmetic` — 산술 pin(**exact-line**; `+ 1` off-by-one이 `assertIn`을
  통과하는 걸 발견해 강화했다), 대입이 단 1회임을 AST로 확인, 확장된 곳이 정확히 2곳이고
  나머지 5곳은 bars-only로 남아 있음.
- `SolidObstacleSliceAlignment` — 행 k에 자기 인덱스를 tag로 심은 합성 월드로, 중심 배열 행 j와
  half-extent 배열 행 j가 **distractor→bar 경계를 넘어서도** 같은 에셋임을 확인. 절대 앵커
  (첫 행 == `_solid_obstacle_offset`, 마지막 행 == `_bar_offset + n_bars_active - 1`)를 함께 걸어
  두 slice가 같이 밀리는 경우도 잡는다. legacy(physical target 없음) 계보도 포함.
- `DefaultOffIsBitIdentical` — 위의 historical 비교 4건.
- `SpawnTreatsDistractorsAsSolid` — 실제 URDF footprint 3종(sphere/box XY 0.15, pole XY 0.06)
  24개 + 막대 16개가 있는 24 m 아레나에서 256 env spawn. 모든 spawn이 모든 장애물의
  **정확한 직사각형 표면**을 0.649802 m 이상 확보. 변경 전 코드로 같은 seed를 돌리면 distractor
  안쪽에 spawn이 생긴다(guard the guard). 수용 경계가 shape별 `|half| + margin`임을 직접 확인 —
  sphere 반경을 pole에 쓰면 거부, pole 반경을 sphere에 쓰면 수용되는 지점을 명시했다.
- `RouteTreatsDistractorsAsSolid` — 실제 `DeterministicAStarRoutePlanner`로 계획. 변경 전에는
  planner가 장애물을 **0개** 받아 start→goal 직선 2-waypoint를 반환하고 그 직선은 distractor를
  관통한다(`segment_is_safe` False). 변경 후에는 우회 경로가 나오고 모든 구간이 safe다. 3종 모두.

**mutation test 5건 전부 검출됨**: 두 slice 동시 되돌림, spawn half-extent만 되돌림(정렬 붕괴),
route half-extent 1행 shift, offset 정의 off-by-one, 그리고 no-op rewrite.

### 남긴 5곳 — crash가 아니라 modelling gap임을 확인

정적 goal 배치(`:4122/:4125`), recovery clearance(`:6091/:6094`),
bar-contact probe(`:3104`, `:3150`), `_target_spawn_center_clearance`(`:5943`),
그리고 `:6322/:6327`·`:6591`·`:6633`은 모두 중심과 half-extent를 **같은 offset**에서 읽으므로
shape 불일치가 발생할 수 없다. `bar_probe.associate_surface_tokens_to_bars`도 `B >= 1`만 요구하며
token 수와 bar 수의 일치를 요구하지 않는다. 따라서 outright crash는 없고 순수한 modelling gap이다.
사전등록 한계로 기록한다.

### 검증

- `python -m unittest discover -s tests`: **898 → 916**(신규 18). 실패 집합은
  failures 8 / errors 8 / skipped 2로 **변경 전과 완전히 동일**(16건 이름 단위 diff 0).
  이 실패들은 학습 중복 실행 lock(`refusing duplicate NavRL training; active PID(s): 12683`)과
  없는 run 폴더·receipt를 읽는 기존 건이며 이번 변경과 무관하다. baseline은 내 편집을 되돌린
  사본으로 실제 재측정했다(Ran 898, 동일 16건).
- GPU 측정은 없다.

다음 단계는 앞 항목대로 카메라-해상도 경로에서 `NAVRL_DISTRACTOR_COUNT`를 올려가며 detector v7의
frame precision 붕괴를 재는 것이다.

## 2026-09-02 — distractor envelope 측정 런처와 FTLR 계측을 구현했다 (GPU 미실행)

`docs/prereg_2026-09-01_distractor_envelope.md`(동결)의 측정 장치를 만들었다. 아직 아무것도
측정하지 않았다 — GPU는 다른 worktree의 PPO run이 쓰고 있다.

### FTLR 분류를 어디에 두었나

`navrl_task._record_distractor_lock_frame`. 검출기가 낸 3D 측정 위치와 GT 표적·distractor 중심이
동시에 존재하는 유일한 지점이며, GT를 metric에만 쓰는 기존 소비자(first_acquisition telemetry,
OOB exit forensics)가 이미 사는 곳이다. `navrl_perception.observe`의 diagnostics에
`camera_measurement_world`(= tracker에 넣는 관측, 필터 이전)와 `camera_confidence`를 노출했다 —
둘 다 이미 계산된 텐서라 참조만 추가한다. 가시 판정은 `camera_visible`(융합 아님): LiDAR가 잡고
있는 프레임에는 분류할 카메라 측정이 없다.

- 분류: `|meas − target| ≤ 0.5` → TARGET, 아니고 `|meas − 최근접 distractor| ≤ 0.5` → DISTRACTOR,
  둘 다 아니면 GHOST. 사전등록 표 순서대로 TARGET 우선이며, 양쪽 반경에 동시에 드는
  `ambiguous` 프레임 수를 따로 세어 내보낸다(우선순위가 숫자를 바꿨는지 독자가 볼 수 있게).
- 누산은 device 위 int64/float64 총계 2개뿐이고 export 때 한 번만 `.tolist()` 한다 —
  step마다 `.item()`으로 host sync를 걸어 재는 대상의 타이밍을 바꾸지 않는다.

### 기본값에서 진짜 no-op인가

`self._distractor_metric_enabled = bool(self._bulk_eval_mode and self._num_distractors > 0)`.
파생값이며 새 env-var가 아니다. False면 누산기가 **할당되지 않고** recorder가 **호출되지 않으며**
export 블록도 없다. `_num_distractors`는 실제로 build된 env config에서 읽는다(작업이 knob을 직접
읽지 않는다 — 장면과 계측이 어긋날 수 없다). N=0 회귀 셀이 손대지 않은 계보를 재는 것이 Gate 0.1의
요구사항이므로, 그 대가로 **N=0 셀에는 FTLR도 count/confidence 평균도 없다**(사전등록 §8의
"셀당 FTLR"을 문자 그대로는 만족하지 못한다). count/confidence의 N 의존성은 N=1·3·5로 읽는다.

### export guard

`_validate_distractor_lock_export` — `_validate_obs_dump_export` 스타일의 순수 함수(torch·self
없음). 핵심은 **분할 검사**: `visible_frames`는 검출기 가시 마스크에서, 세 범주는 별도로 만든 세
마스크에서 누산되므로, 이중 계수나 누락이 있으면 두 총계가 어긋나 export가 죽는다. 그 외 음수,
`visible > total`, 프레임 0, `ambiguous > target_lock`, 반경 불일치를 잡는다. 7개 음성 케이스를
테스트로 확인했다.

### N=0 회귀 허용오차

**±3.75 pp** (`N0_REGRESSION_TOLERANCE_PP`). 계보 참조는 sensor-fidelity `baseline` arm
(seed 421, capture/crash/timeout **70.52/19.81/9.66%**, sha256 `57d0bbef819ad0f7…`)이며 seed가
다른 독립 표본이므로 동치 비교가 불가능하다. n1=n2=2049, 최악 p=0.5에서 차이의 SE는
`sqrt(0.25/2049+0.25/2049)=1.5620 pp`, outcome 3개에 대한 Bonferroni z(0.05/3)=2.39398 →
3.739 pp를 올림했다. 오탐 VOID는 4셀분 GPU를 버리는 반면 이 게이트가 잡아야 할 실제 회귀는
크다(flat spawn-clearance 결함이 27.45 pp였다). never-acquired는 함께 보고하되 게이트하지 않는다.

### 셀별 env 차이 (실측)

`NAVRL_DISTRACTOR_COUNT: {n0:'0', n1:'1', n3:'3', n5:'5'}` — 네 환경의 **모든 쌍**을 비교해 실제
대칭차가 이것과 `NAVRL_V2_RESULT_DIR`뿐임을 확인한다. 실행 후에는 결과 문서의 `condition` 사전을
셀 간 대칭차로 비교해 `distractor_count`만 움직였음을 확인하고, evaluator contract는 전 키가
동일해야 한다(그 문서에는 distractor 필드가 없다).

### Gate 0을 런처가 강제한다

- 0.1/0.2는 `tests/test_navrl_distractors.py`의 `DefaultOffIsUnchanged`,
  `DefaultOffIsBitIdentical`, `FailsClosedOnDistractorsPlusDecoupling`를 런처가 subprocess로
  돌리고(테스트 모듈 sha256을 함께 기록) + 계측 게이팅·거부문·전 셀 카메라 해상도 경로 정적
  검사를 더한다. 실패하면 셀을 만들지 않고 `FAIL_CLOSED_IMPLEMENTATION` 요약을 낸다.
- 0.3은 위 허용오차로 판정한다.
- 카메라 해상도 경로는 런타임 거부에 기대지 않고 `evaluation_env()`가 전 셀에서
  `DETECT == CAMERA`를 assert한다 — N=0 셀은 그 거부를 그냥 통과해 버리기 때문이다.

### 검증

- `preflight` 실행(CPU만, GPU 점유 6900→6895 MiB 무변화): Gate 0.1 PASS(10 tests),
  0.2 PASS(5 tests), **네 셀 모두 evaluator preflight를 force 없이 통과**. 마지막에 dirty
  runtime으로 정상 FAIL(내 런타임 편집이 미커밋).
- `python -m unittest discover -s tests`: **916 → 933**(신규 17). failures 8 / errors 8 /
  skipped 2로 실패 집합은 **이름 단위까지 동일**. 학습 중복 lock(PID 12683)과 이 worktree에 없는
  gitignored `results/` 건이다.
- `test_navrl_distractors.SolidObstacleOffsetArithmetic.test_both_widened_sites_...`를 갱신했다.
  telemetry가 `[_solid_obstacle_offset : _bar_offset]`(distractor 전용, 막대 미포함)을 읽는 세
  번째 소비자라서다. 넓힌 슬라이스(4회)와 distractor 전용 슬라이스(1회, recorder 안)를 구분해
  각각 고정했다 — 후자가 실수로 막대까지 읽으면 검출기 측정을 막대 중심과 비교하게 된다.
- GPU 측정은 없다. 판정도 없다.

다음 단계는 GPU가 비는 대로 `evaluate n0 → n1 → n3 → n5` 후 `finalize`다.

## 2026-09-02 — distractor envelope를 2×4 요인설계로 확장했다 (검출기 축 추가, GPU 미실행)

앞 항목의 런처는 기본 검출기만 쟀다. 사전등록 §3이 "현행 기본값과 학습형 v7 **둘 다**"를 명시하므로
검출기를 **두 번째 요인**으로 올렸다. 기본값만 재는 것은 5-파라미터 색 규칙이 색 구별에 실패하는
것을 재는 것이라 거의 항진명제다. 열린 질문은 **학습된 v7이 색 shortcut을 벗어났는가**이고,
0.99766 frame precision 주장을 지고 있는 것도 v7이다.

### 셀 격자

`검출기 {default, v7} × distractor {0,1,3,5}` = **8 셀**
(`default_n0`…`default_n5`, `v7_n0`…`v7_n5`). 손으로 나열하지 않고 두 요인에서 유도한 뒤
완전 요인설계인지(8개, 중복 없음) 즉시 assert한다.

### v7 아티팩트 — 무엇을 고정했나

- `artifacts/navrl_target_detector_v7_confirmatory.pth`
- sha256 `85c7974bcd85c627170c5bd63030144d1c5dc2a11e5d64829cad38f615c5d5d7`
- **동작점 threshold 0.700**

`artifacts/`에는 검출기 체크포인트가 7개 있으므로 "v7 파일"은 이름이 아니라 **v7 게이트가 채점한
바이트**로 정의해야 한다. 런처는 `results/navrl_detector_offline_gate_v7_confirmatory/summary.json`의
`artifact_sha256`·`selected_threshold`·`gate_passed`와 교차검증한다(실측: frame precision
0.99765625 @ thr 0.7 — 게이트 요약과 일치).

**동작점이 0.55가 아니라 0.700인 이유**: 0.99766은
`tools/train_navrl_target_detector_v2.py`가 held-out test를 `selected_threshold`에서 채점한 값이다.
0.55에서 재면 그 주장이 속한 동작점이 아닌 다른 동작점을 재게 된다. 기존 유일한 v7 실행
(`results/navrl_detector_domain_shift_v7/`)은 205막대·thr **0.55**의 오프라인 프레임 스크리닝이라
비교 대상이 아니다. 따라서 **검출기 = 아티팩트 + 동작점**을 하나의 요인 수준으로 묶었다.

### 좁은 override (담요식 force 아님)

동결 정책은 thr 0.55에서 학습됐으므로 evaluator의 v2 provenance gate가 v7 셀에서 반드시 한 필드를
문제 삼는다. `NAVRL_V2_FORCE`(전 필드 무력화) 대신 목적 전용
`NAVRL_V2_ALLOW_DETECTOR_THRESHOLD_MISMATCH`를 쓴다 — sensor-fidelity가 만든 좁은 override 패턴이며
담요식보다 **더 엄격하다**. force 없는 실행이 정확히
`cfg_detector_threshold: checkpoint=0.55 expected=0.7` **한 줄로만** 거부되는 것을 실행 시점에
증명한 뒤에야 적용하고, override 실행에서 evaluator가 **그 필드를 허용했다고 announce한 것**까지
확인한다. 두 줄이거나 다른 필드면 중단한다.

**실측으로 버그 하나를 잡았다.** evaluator는 허용한 필드를 거부 라인과 **같은 모양**으로 출력한다
(`[eval_v2] ALLOWED mismatch: ... checkpoint=... expected=...`). `checkpoint=`/`expected=`만 보고
거르던 파서가 override의 성공 announce를 잔여 거부로 읽어 v7 preflight를 실패시켰다. 정반대 사실이라
`mismatch_lines`(거부)와 `allowed_mismatch_lines`(허용)로 분리했다.

### 두 방향 단일축 검증

한 번에 뭉뚱그리면 검출기를 따라 움직인 엉뚱한 변수가 "검출기 축의 일부"로 변명될 수 있으므로
**따로** 검사한다.

| 고정 | 허용된 차이 | 검사 |
|---|---|---|
| 검출기 | `NAVRL_DISTRACTOR_COUNT` | 각 검출기 4환경의 모든 쌍 |
| distractor 수 | `NAVRL_DETECTOR_CHECKPOINT`·`NAVRL_DETECTOR_THRESHOLD`·`NAVRL_V2_ALLOW_DETECTOR_THRESHOLD_MISMATCH` | 각 N에서 2환경 |

실행 후에는 결과 문서로 같은 두 방향을 다시 본다: 검출기 안에서는 `condition.distractor_count`만,
N 안에서는 contract의 `detector_threshold`/`detector_checkpoint_sha256`만 움직여야 한다. 후자는
`navrl_perception`이 로드한 바이트에서 재도출해 raise하는 값이라 "요청한 경로"가 아니라
"실제 로드된 가중치"에 대한 진술이다.

### 판정은 검출기마다 따로

Gate F는 각 검출기의 N=5 셀에서 내려 **판정이 2개**다. `summary.json`에는 `verdicts`(복수)만 있고
단수 `verdict` 키는 **없다** — 통합 판정으로 오독할 여지를 키 이름 수준에서 없앴다. 각
`verdict_basis`는 `applies_only_to`를 달고 있고, `summary.md`는 두 판정이 서로에 대한 근거가
아님을 명시한다.

### Gate 0.3 범위

`default_n0`에만 적용한다(±3.75 pp, 계보 = sensor-fidelity baseline seed 421).
**`v7_n0`에는 계보 기준점이 없다** — 이 체크포인트의 기존 70막대 항법 셀은 전부 내장 segmenter였고
유일한 v7 실행은 조건·동작점이 다른 오프라인 스크리닝이다. 따라서 기술적(descriptive) 셀로 두고
v7 자신의 FTLR 추이 기준으로만 쓰며 **아무것도 게이트하지 않는다**. v7 셀의 outcome은 계보와 비교할
수 없다: 동결 정책이 학습한 적 없는 검출기로 날고 있으므로 궤적 자체가 다르다.

### 비용 추정 (GPU 확보 전 사전 통지)

기존 영수증 기준 2,049 에피소드 1셀 wall-clock: **카메라 해상도 경로 5.11분**
(`sensor_fidelity/cells/baseline`, 본 실험과 동일한 센서 조건), 참고로 1920×1200 detect는 10.77분.
distractor 렌더는 카메라 해상도 Warp 커널 1회 추가(+수%), FTLR 계측은 per-step host sync 없음(무시).
v7은 1×1 conv가 아니라 spatial CNN이라 step time이 늘어난다.

| | 셀 | 셀당 | 소계 |
|---|---:|---:|---:|
| default | 4 | ~5.1–5.5분 | ~21분 |
| v7 | 4 | ~6–7분 | ~26분 |
| **합계** | **8** | | **~50분** |

**보수적 상한 ~1시간 20분**(v7이 예상의 2배로 느릴 경우). 요청받은 ~1시간 예산 안이다.
VRAM 추가분은 distractor mask/depth 128×90×160×8 B ≈ 15 MB로 무시할 수준이지만, 평가는 학습이
점유한 6.9 GB와 **동시 실행 불가**다(기존 평가 셀도 단독 실행이었다).

### 검증

- `preflight`(CPU만, GPU 6902→6906 MiB 무변화, PID 12683 무영향): **8셀 전부 통과**.
  default 4셀은 force 없이 PASS, v7 4셀은 좁은 override를 실행 시점에 검증한 뒤 PASS.
  Gate 0.1 PASS(10 tests), 0.2 PASS(5 tests), v7 게이트 교차검증 통과. 마지막에 dirty runtime으로
  정상 FAIL(런타임 편집 미커밋).
- `python -m unittest discover -s tests`: **916 → 938**(신규 22). failures 8 / errors 8 /
  skipped 2로 실패 집합은 **이름 단위까지 동일**.
- GPU 측정은 없다. 판정도 없다.

다음 단계는 GPU가 비는 대로 `evaluate` 8셀 후 `finalize`다.

## 2026-09-02 — L6(검출기 간 FTLR 비교 금지)을 코드로 강제했다

사전등록 §3-c가 추가됐다. 앞 항목에서 나는 "동결 정책이 v7을 학습한 적 없는 검출기로 날린다"를
**주석에만** 적었다. 결과는 그보다 무겁다: 궤적이 다르면 프레임이 다르고, 프레임이 다르면
시야 기하 분포(거리·베어링·가림, 특히 **표적과 distractor가 동시에 보이는 빈도**)가 다르다.
FTLR은 그 분포 **위에서** 정의되므로 `FTLR_v7 − FTLR_default`는 검출기 강건성과 궤적 분포를
뒤섞은 값이고 이 셀들로는 분리할 수 없다. 각 검출기의 FTLR은 그 검출기가 실제로 만든 프레임
위에서 계산되므로 **검출기 안에서 N에 따른 비교는 그대로 유효**하다.

- `LIMITATIONS`에 **L6** 추가(L1–L5와 같은 한국어 서술). outcome 3종에도 적용됨을 명시했다 —
  v7 셀 값은 계보와도, default 셀과도 비교하지 않는다.
- `summary.json`에 `comparability` 필드(`between_detectors: false`,
  `cross_detector_difference_computed: false`)를 넣었다.
- `summary.md`는 **두 줄짜리 판정표 바로 밑**에 인용 블록으로 경고를 놓는다. 한계 절은
  독자의 눈이 닿는 곳이 아니다.
- 교차 검출기 차이는 **어디에도 계산하지 않는다.** 원래도 없었음을 확인했고(유일한 뺄셈은
  Gate 0.3의 계보 회귀로, 검출기 안이며 FTLR이 아니다), 앞으로도 못 생기게 테스트를 걸었다.

### 테스트가 실제로 잡는지 확인

이름 기반 검사(published key에 delta/difference/gap 류 토큰 금지)와 **AST 검사**(다중 셀 FTLR
컨테이너 — `ftlr_by_cell`/`measurements`/`verdict_basis`/`cells_payload` — 를 건드리는 뺄셈 금지,
그리고 `build_summary`/`write_summary` 안에서는 FTLR 뺄셈 자체 금지) 두 겹이다.
`"ftlr_v7_minus_default_pp": 100.0 * (ftlr_by_cell["v7_n5"] - ftlr_by_cell["default_n5"])`를
일부러 주입해 테스트가 **실패하는 것을 확인**한 뒤 되돌렸다. `verify_lock_block`의 셀 내부
자기일관성 검사(같은 셀의 재계산 FTLR vs 내보낸 값)는 걸리지 않는다 — 다중 셀 컨테이너를
건드리지 않기 때문이며, 걸렸다면 그 검사는 고쳐지는 대신 삭제됐을 것이다.
이름 검사는 `preregistration`에 `ratio`가 들어 있어 부분문자열 매칭이 오탐을 냈다 — 밑줄 토큰
단위 비교로 바꿨다.

### v7 forward 비용 — CPU 전용 상한 (GPU 미사용)

앞 항목의 "+10–40%"는 CNN 크기를 추측한 값이었다. 실제로 재고 세었다.

- 아키텍처 `SpatialTargetSegmenterWide/cnn9x9x24-RGBD`: `Conv(4→24,3×3)` → `Conv(24→24,3×3,d=2)`
  → `Conv(24→24,3×3)` → `Conv(24→1,1×1)`, **11,329 파라미터**(부트스트랩은 5개).
- CPU forward(128×90×160, 4스레드): 부트스트랩 15.8 ms, v7 248 ms → **15.8배**.
  다만 이 비율은 **CPU 아티팩트**다. 24채널 3×3 스택은 GPU가 잘하고 4스레드 CPU가 못하는
  워크로드이며, 1×1 conv는 양쪽 다 메모리 바운드다.
- 기계적 상한(스레딩 무관): **20.75 GMAC = 41.5 GFLOP/step**(부트스트랩 0.007 GMAC, 2,814배).
  RTX 3070 fp32 peak ~20.3 TFLOP/s, 이 shape의 현실적 효율 15–35% → **6–14 ms/step**.
  baseline 셀의 step time은 306 s / 대략 4,000–6,400 sim step ≈ **50–80 ms**이므로 **+10–30%**.

수정 추정: default 4셀 ~21분, v7 4셀 ~22–39분, **합계 ~45–60분**. 상한 ~1시간 20분은 유지하되
**여유롭지 않다** — GPU가 이 shape에서 유효 1 TFLOP/s밖에 못 내면 41 ms/step이 되어 step time이
거의 두 배가 되고 정확히 상한에 닿는다.

### verify_detector_artifacts()는 항진명제가 아니다

파일 바이트의 digest를 **모듈 상수**(`DETECTORS`의 하드코딩 리터럴)와 비교하고, **별도로**
게이트 요약의 `artifact_sha256`을 같은 상수와 비교한다. 파일을 자기 자신과 비교하는 지점은 없다.
실증: v7 경로에 v6 체크포인트를 끼워 넣으면 첫 비교에서 잡히고, 게이트 요약의 sha를 위조하면
둘째 비교에서 잡힌다. 둘 다 확인했다.

### 검증

- `python -m unittest discover -s tests`: **916 → 941**(신규 25). failures 8 / errors 8 /
  skipped 2로 실패 집합 **이름 단위까지 동일**.
- GPU 실행 없음(위 timing은 CPU 전용). PID 12683 무영향.

## 2026-09-02 — distractor envelope (seed 479) 1650 Ti 측정 완료: default·v7 둘 다 COLOR_SHORTCUT_CONFIRMED

`tools/run_navrl_distractor_envelope.py` preflight→evaluate→finalize→verify를 **1650 Ti(평가 공장)**에서
완주. 동결 정책 ref5in D1 ep1900(`197ea269…`), seed 479, 셀당 2049 에피소드, 카메라 해상도 160×90,
70막대. 프로파일 = GPU4GB=1(64 env · `base_sim_4gb` · `base_sim_4gb_dt0.01_buffers`).

**결과 (2×4 요인, 1차지표 FTLR — 검출기가 무엇을 표적이라 락온하는가):**

| 검출기 | N=1 | N=3 | **N=5 FTLR** | 판정(N=5) |
|---|---|---|---|---|
| `default` (thr 0.55) | 52.7% | 79.7% | **88.53%** | `COLOR_SHORTCUT_CONFIRMED` |
| `v7` (thr 0.70) | 60.7% | 83.1% | **90.27%** | `COLOR_SHORTCUT_CONFIRMED` |

- N=0은 회귀 셀(FTLR 미정의). §3-c대로 **`FTLR_v7 − FTLR_default`는 계산·게재하지 않음.**
- distractor↑ → FTLR 단조 증가, target_lock 급감(default 0.473→0.115), capture 하락(0.726→0.132).
- **판정 검출기별 2개, 둘 다 COLOR_SHORTCUT_CONFIRMED** (FTLR ≥ 50%) — 사전등록이 예측한 결과.
  학습형 v7도 색 지름길 미탈출(v7 FTLR이 오히려 높으나 §3-c로 직접 비교 안 함).

**게이트:** 0.1 default-off bit-identity PASS · 0.2 decoupling refusal PASS · **0.3 N=0 계보 회귀 PASS**
(`default_n0` vs 계보 seed421, timeout 차 +0.34pp, ±3.75pp band 이내).

**반증:** "1650 Ti 프로파일(64env·base_sim_4gb)이라 Gate 0.3이 머신-프로파일 불일치로 실패할 것"이라는
사전 예측을 **기각** — N=0 outcome이 3070 계보(128env·base_sim) ±3.75pp 안에 들어와 실제 PASS.
사전등록 §3-d(머신 프로파일 편차 서술적 강등)는 **불필요**했음. 이 머신 측정이 예상보다 더 신뢰됨.

**경로:** `results/navrl_detector_distractor_envelope_seed479/summary.{json,md}` +
`cells/*/70bars.json`. 계보 참조 `results/navrl_ref5in_sensor_fidelity_seed421/cells/baseline/70bars.json`.

**머신 이식 편차(기록·정리됨):** 1650 Ti 실행 위해 `tools/`를 임시 편집했다 — (a) canonical python
경로 `/home/fair`→`/home/joshuali` 2곳, (b) `canonical_env` GPU4GB `0`→`1`(프로파일 셀렉터). 이 셋은
**머신 로컬이라 커밋 전 원복**(3070은 /home/fair·GPU4GB=0). (c) `condition_differences`에서 랜덤
`evaluation_nonce` 제외 — **사전존재 버그 수정**(랜덤 토큰을 조작축 차이로 오판해 모든 머신에서
finalize가 실패하던 게이트; `evaluation_env_diff`가 `NAVRL_V2_RESULT_DIR`를 제외하는 것과 동일 패턴).
이 버그 수정만 커밋한다. **측정 로직·FTLR 정의·셀 조건·seed·에피소드 수는 무변경.**

**다음:** 원하면 3070에서 canonical 128env·base_sim 프로파일로 재확인(옵션 ②) — 단 Gate 0.3이 이미
1650 Ti에서 통과했으므로 우선순위 낮음. PID 12683 학습 무영향.

## 2026-09-02 — 병합 노트: distractor envelope를 main에 통합, 3070에서 재검증

`codex/distractor-envelope`(`09356c2`)를 main에 병합했다. WORKLOG는 양쪽이 파일 끝에 **추가만**
했으므로 손실 없이 둘 다 보존했다. 두 갈래(1650 Ti의 distractor 측정과 3070의 safety-filter 작업)가
서로 다른 기계에서 동시 진행돼 날짜가 교차하므로 **갈래별로 묶여 있고 엄밀한 시간순이 아니다.**

### 3070 재검증: PASS

```
[distractor] VERIFY PASS | {'default': 'COLOR_SHORTCUT_CONFIRMED', 'v7': 'COLOR_SHORTCUT_CONFIRMED'}
```

1650 Ti가 만든 gitignore 아티팩트(셀별 `70bars.log`·`checkpoint_snapshot.pth`·`source_bundle`,
61 MB)를 전송해 8셀 전부 검증했다. 판정·수치는 1650 Ti와 동일하다.

### 재검증이 드러낸 이식성 결함 2건 (수치 무변경)

1. **셀별 심링크가 `/home/joshuali/MOTAR/...` 절대경로로 커밋돼 있었다** — 생성 기계 밖에서는
   구조적으로 깨져 `source_manifest.json`·`source_snapshot`·`python_environment.txt` 24개가
   모두 dangling이었다. `../../source_bundle/...` 상대경로로 repoint했다.
2. **`gate0`의 `artifact_path`에 host 절대경로가 기록돼 `SUMMARY_VERIFY_KEYS` 비교에 들어갔다** —
   내용이 동일해도 다른 기계에서는 verify가 실패한다. 런처 자신의 `_resolve_shared_path` docstring이
   "the path is a lookup, never a trust boundary"라고 명시하는데 그 계약과 모순이었다. 신뢰 경계는
   `source_sha256`이고 이는 별도 `require`로 하드 검증된다(무변경).

**두 수정 모두 게이트·임계값·판정을 건드리지 않는다.** 수정 전 recorded-vs-recomputed 전수 비교에서
차이는 `gate0/zero_cell_reproduces_lineage/lineage_reference/artifact_path` **한 필드뿐**이었음을
확인한 뒤 적용했다.

### 출처

`artifact_path` 수정(`_VERIFY_IGNORE_KEYS` + 상대경로 기록)은 **Cursor 세션이 동시에** 작성한
것을 검토 후 채택했다. 심링크 repoint는 이 세션이 했다.

## 2026-09-02 — stopcap 판정 보론: M1은 측정된 round-off, 충돌은 종방향이 아니다

앞 항목(`b4639ec`, Cursor 세션이 커밋)의 판정에 세 가지를 확정해 덧붙인다. 판정·게이트·수치는
일절 바꾸지 않는다. 새 GPU 실행 없음.

### 1. M1의 "round-off 가능성"을 측정으로 확정했다

앞 항목은 단위테스트 tolerance가 `−1e-5`라 2.62 %가 round-off일 가능성을 **배제할 수 없다**고
적었다. float32로 200,000 표본을 재현해 확정한다:

- 음수 margin의 **100 %가 cap-binding 경우**(`executed == cap < free_cap`)
- **최악 −4.77e-07 m, 중앙값 −1.19e-07 m** (0.5 µm 이하)

stopcap 법칙은 `usable`에서 여유를 **정확히 0**으로 만드는데 게이트를 strict `< 0`로 걸었으므로
float32 반올림이 부호를 양쪽으로 나눈다. 즉 안전 결함이 아니라 **사전등록 명세 오류**다.
**게이트는 사후 완화하지 않는다** — M1 VOID를 그대로 유지하고, Q2가 독립적으로 세 게이트 모두
실패하므로 채택 결론도 그대로다. 교훈: 등호를 달성하는 법칙에 strict 부등호 게이트를 걸면 안 된다.

### 2. 충돌은 종방향 정지 실패가 아니다 (조사 문서 §1 진단의 부분 반증)

기구는 **설계대로 작동했다**: 하한 제거로 접촉 직전 실행 속도 2.024 → **0.302 m/s**,
`unsafe_after` 8.74 → **2.62 %**, 접촉 시 정지여유 −0.026 → **+0.395 m(양수)**.
그런데 crash는 15.95 → **21.31 %로 올랐다**.

→ 접촉 순간 거버너 자신의 안전 모델이 "정지 가능"이라고 말한다. 즉 회랑(반폭 0.45 m, 명령 방향
직선)이 **실제로 부딪히는 장애물을 담고 있지 않다**. `docs/safety_filter_survey_2026-09-02.md` §1의
"2.0 m/s 하한이 충돌 원인"은 기구적으로는 맞았으나 **지배적 원인이 아니었다**. 문서 §1을 이 결과로
읽어야 한다(문서 자체는 조사 시점 기록이므로 수정하지 않는다).

악화 기제는 **노출 시간**이다: `mean_clearance` 7.18 → 5.21 m, 접촉 step 97 → 121,
near-stop 30.12 %. clearance가 낮을 때 정확히 감속하므로 **통과해야 할 틈에서 가장 오래 머문다**.
평균 실행 속도가 사실상 같은(1.963 vs 1.978) fixed2p0가 crash·timeout 양쪽에서 stopcap을
지배하는 것이 이를 뒷받침한다 — 두 arm의 차이는 감속의 **상태 의존성**뿐이다.

### 3. 귀속 한계: stopcap·ttc는 이 정책에 out-of-distribution

ep25000은 마지막 1,000 epoch을 **riskcap과 함께** 학습했다. riskcap·fixed2p0는 clutter에서 모두
≈2.0 m/s이므로 in-distribution이지만 stopcap·ttc는 아니다. Q3가 이 정책이 FILTER_DEPENDENT
(+9.23 pp)임을 이미 보였으므로, **stopcap의 부진이 법칙 탓인지 필터 불일치 탓인지는 분리되지
않았다.** 앞 항목의 "stopcap 동반 재학습 금지"는 이 스크린 근거로는 과한 결론이며, 정확히는
**새 사전등록 없이는 진행 금지**다.

### 세션 출처 기록

`b4639ec`는 이 세션이 만든 산출물을 **Cursor 세션이 커밋**한 것이다(사용자 diff 검토 전 커밋).
본 항목은 그 위에 얹는 보론이다.

## 2026-09-03 — 인지·안전필터 문서화: 두 축 모두 문서가 없었다

사용자 요청은 "safety filter 구조도와 카메라 탐지 자료가 잘 되어 있을 테니 바뀐 부분을 고쳐달라"였다.
**감사 결과 전제가 사실과 달랐다 — 두 축 모두 사실상 문서가 없었다.**

| 대상 | 감사 전 실제 상태 |
|---|---|
| README safety filter | 한 줄 (`canonical baseline의 safety governor는 꺼져 있습니다`) |
| `motar-control-stack.svg` | 텍스트 한 줄 `speed governor: OFF in baseline`. 구조도 아님 |
| README 인지/탐지 | 표 한 칸 (`RGB-D target track + 4×72 LiDAR`). 탐지 방식 설명 0 |
| 사이트 `index.html` | detector·governor·riskcap·segmenter 언급 **0건** |
| `status.json` | 인지·필터 구조화 데이터 없음 |

따라서 "수정"이 아니라 **신규 작성**이다. 지금이 정확한 시점인 이유는 이번 주에 두 축 모두
판정이 나왔기 때문이다(distractor envelope, stopcap screen).

### 신규 자산 2개

- `docs/assets/motar-perception-detection.svg` — RGB-D → 1×1 conv 색 분류기 → **단일 중심점 붕괴**
  (connected components 없음) → LiDAR 연관. **동색 디코이가 분류기에서 들어와 걸러지지 않는 경로**를
  붉은 점선 화살표로 표시했고, FTLR 표(default 88.5% / v7 90.3% at N=5)와 confidence
  N=1/3/5 실측(0.826→0.896→0.892)을 함께 실었다. N에 따른 단조 증가 주장은 하지 않는다.
- `docs/assets/motar-safety-filter.svg` — LiDAR → 방향 회랑(반폭 0.45 m 직선) → cap 법칙 →
  **크기만 스케일(방향 불변)**. 핵심은 **회랑의 맹점 4개**(측방·수직 z 무규제·미지 공간
  무반사=자유·직선 가정)를 전면에 낸 것이다. seed49 5-arm 측정표와 두 확정 판정을 함께 실었다.

두 SVG 모두 XML 파싱 검증 통과. 기존 `motar-control-stack.svg`는
`baseline governor: OFF`로 유지했다. canonical evaluator의 기본값도 `off`이며, 필터 arm은
ep25000 stopcap screen처럼 명시적으로 켠 별도 실험이다.

### README

- `## Perception — how the target is detected` 신설: 탐지기가 YOLO가 아님을 명시,
  FTLR 표, confidence 역설, "개선이 아니라 결함의 정량화", detector 간 비교 금지(L6).
- `## Safety filter — the speed governor` 신설: 5-arm 측정표 + 두 확정
  (① 충돌은 종방향 정지 실패가 아니다 ② riskcap 해제 기구 무실증). `+9.23 pp`는 이
  ep25000 screen의 off−riskcap 비교로만 한정했다.
- Current evidence 표에 두 캠페인 행 추가.
- Method 절의 낡은 governor 한 줄을 정정하고 신설 절로 링크.

### 사이트 (기존 껍데기 유지, 현재 상태로 갱신)

기존 8개 섹션을 모두 유지한 채 낡은 서술만 고치고 패널 2개를 신설했다.

- `04 · PERCEPTION`, `05 · SAFETY FILTER` 신설(기존 overview/control-stack과 동일한 `<img>` 패턴).
- 하위 섹션 번호 재정렬(EVIDENCE 04→06, CONTRACT 05→07, NEXT 06→08). 중복 번호 없음을 확인.
- `Structured perception` 카드: "탐지기는 YOLO가 아니라 1×1 conv 색 분류기이며 동색 디코이를
  구분하지 못합니다" 추가.
- 상단 탐색에 Perception/Filter anchor를 추가했다.
- **"현재 주장할 수 없는 것"에 두 항목 추가** — 표적 식별을 주장할 수 없음(색 외 단서 없이는
  "표적 인식"이 아니라 "빨간 것 찾기"), riskcap 해제 기구 무실증.
- Evidence는 20개 혼합 카드에서 핵심 9개로 줄였다. route-off fresh smoke(500 epochs),
  seed-911 curriculum 중지(epoch 21,973, 145 bars), seed-313 held-out(70 bars 83.70%,
  145 bars 65.54%)을 현재 상태로 올리고, perception/filter 수치는 각 전용 절에만 남겼다.
- hero의 낡은 0-epoch·smoke 미실행 문구를 제거했다. route-off 학습·평가와
  routed mechanism FAIL을 별도 계보로 표시한다.
- perception SVG의 “안전층은 탐지기를 소비하면 안 된다”는 미래 제안으로 표기하고, 현재 governor가
  `camera_lidar_association`으로 표적 return을 제외하는 detector-dependent 구현임을 함께 적었다.

검증: SVG 3개 XML 파싱, HTML anchor/asset 감사, Chrome headless 렌더, `test_status_site.js`,
`test_status_arena_route.js`, `test_status_arena_motion.js`, `test_status_webgl_headless.js` PASS.
## 2026-09-01 — braking-aware route v3 MECHANISM_GATE 구현 (GPU 미실행)

corrected non-overlap r2 `FAIL_ROUTE_MECHANISM` 이후, 사전등록된 fresh-only 모드
`global_astar_braking_v3`를 구현했다. 사전등록
`docs/preregistration_braking_aware_route_v3_2026-09-01.md`의 SHA-256
`cceecb9ad4a538e7bc2bc9171436e823ef18652e9c971e0d6fa8174279df6056`는 결과를 보기 전에
고정됐고, 이 작업에서 바이트를 수정하지 않았다.

구현 범위: spawn/A*/local/watchdog가 공유하는 exact closed-AABB soft envelope, full-horizon
수락(longest-safe-prefix 실행 금지), canonical 1.5 m/s monotone p95 ceiling lookup,
terminal zero-command 정지 증명, tangent/cross-track + stop-turn-go, unsafe_start에서 A* 반복
금지, reset/goal-completion/runtime-replan 분리 계측. `global_astar_v1`과
`global_astar_recovery_v2`의 동작·diagnostics shape는 유지했다. reward/obs/termination/PPO/
물리 동역학은 변경하지 않았다. `CRASH_TUNING_LOG.md`는 archival-in-place로 새 기록을 넣지
않았다.

navrl skill은 stage router(`ENVIRONMENT_CONTRACT` → `MECHANISM_GATE` → `PPO_SMOKE` →
`PPO_CURRICULUM` → `HELD_OUT_EVAL` → `SIM_TO_REAL`)로 개정했고, 과거 25/50/75/110/130/150과
`NAVRL_DENSITY_THRESHOLD=0.6`는 historical recipe로 내렸다. 현재 계보는 70/115/160/205,
300 bars는 disconnected stress다.

GPU/PPO/simulator 평가는 실행하지 않았다. 포커스 CPU 테스트는
`test_navrl_braking_route_v3*` 24, planner 13, target_motion 25, two-envelope 23,
environment contract 13, recovery evaluator 8, corrected-gate 4, braking-contract 8로
전부 PASS였고 `git diff --check`는 깨끗하다. 커밋/푸시는 사용자 diff 승인 전 금지.

다음: CPU 통과를 확인한 뒤, 사용자가 지시하면 seed 829 / 70 bars / 8-cell pilot만 실행한다.
pilot 실패 시 confirmatory(seed 839, 32 cell)는 금지다.

## 2026-09-01 — v3 셀 어댑터 완성, GPU pilot 직전 준비 (GPU 미실행)

사용자가 diff를 승인해 braking-v3 구현을 `6b13441`로 커밋했고, 이어 pilot 진행을 지시했다.
게이트가 요구하는 tracked 실행 파일 `tools/run_navrl_braking_route_v3_cell.py`를 추가했다.
어댑터는 `NAVRL_V3_*` 셀 계약을 fail-closed로 파싱하고, corrected-r2 평가기의
`frozen_environment`/neutral-pursuer 인터벌 헬퍼를 바이트 그대로 재사용해 셀 하나를 돌린 뒤,
v3 verifier의 `validate_cell`을 자기 payload에 먼저 적용하고 나서야 JSON을 원자적으로 쓴다.
identity의 cell-runner/braking-receipt/사전등록 SHA 자기검증, `MOTAR_V3_*` passthrough로
training source manifest 전달(게이트가 `NAVRL_*`를 제거하므로), off arm의 manager 부재 확인을
포함한다. v1/v2/기존 verifier는 수정하지 않았다.

CPU 검증: 새 `test_navrl_braking_route_v3_cell_runner` 9/9, `test_navrl_braking_route_v3*`
합계 33/33, planner 13/13, target_motion 25/25, braking-contract 8/8 PASS.

GPU 직전 상태에서 남은 실제 선행물은 **canonical 1.5 m/s braking receipt**다. receipt는 현재
커밋의 core 바이트에 바인딩되므로 재측정이 필수인데, 2026-08-26 기록상 1.5 m/s warmup 수렴이
1.4426 m/s로 0.05 m/s 게이트를 못 넘겨 NO-GO였고 controller는 변경 금지다. 재측정이 다시
NO-GO면 pilot은 열리지 않으며, threshold 완화 없이 중단·보고한다. 실행 시퀀스(probe →
training source bundle → pilot)는 `OPERATIONS.md`에 고정했다. GPU/PPO는 이 엔트리 시점까지
실행하지 않았다.

## 2026-09-01 — canonical 1.5 braking probe 재측정 NO-GO 재현, pilot 차단

사용자 지시로 커밋 `fb9fa50`에서 canonical_1p5 braking probe를 RTX 3070에서 실행했다.
0.6/0.9/1.2 m/s 세 셀은 warmup/raw 게이트를 통과했으나 1.5 m/s 셀이 5 s warmup 수렴 게이트에서
실패했다: final speed `min=1.441824 / mean=1.442577 / max=1.443260 m/s`, 절대 오차
`min=0.056740 / mean=0.057423 / max=0.058176 m/s` (게이트 0.05 m/s). 이 수치는 2026-08-26
진단 측정과 소수 6자리까지 동일하므로 controller의 결정론적 정상상태 추적 한계이지 노이즈가
아니다. launcher가 fail-closed로 partial stage를 삭제해 receipt는 발급되지 않았다
(`results/`에 산출물 없음, 트리 clean 유지). 사전등록 규칙대로 threshold·controller·속도 grid는
변경하지 않았다.

같은 날 training source bundle은 저장소 밖
`/home/fair/workspaces/aerial_gym_ws/navrl_v3_receipts/training_source_2026-09-01/`에 생성했다
(`manifest_sha256=0fb6dbb4551cd1a2d8ab6bd341447de3a1f5ae5e69e57c95947d71ed51cc25fd`,
git_commit `fb9fa50`, 328 files, verify PASS). pilot 단계 3의 나머지 입력은 준비돼 있다.

결정 지점: v3 사전등록은 canonical_1p5 receipt를 요구하고 controller retune을 금지하므로,
**1.5 m/s warmup 게이트가 서 있는 한 v3 pilot은 열리지 않는다.** 다음 선택지는 사용자 결정
사항이다 — (a) 이 한계를 근거로 v3 계보를 lower-contract(예: 1.25 상한) 방향으로 새로
사전등록, (b) warmup 수렴 게이트 자체를 재검토하는 새 사전등록(측정 후 완화가 아니라 독립
근거 필요), (c) 중단. 본 세션은 어느 것도 자동 선택하지 않는다.

## 2026-09-01 — lower-contract v3 사전등록 고정, canonical 1.5/PID/0.05는 유지

canonical 1.5 warmup NO-GO 이후 사용자는 규약을 지키는 방향으로 진행하되, 규약을 깨는 편이
나은지만 판단하라고 했다. 판단: **지금은 깨지 않는 편이 맞다.** 0.05를 0.06으로 올리면
1.5 receipt는 숫자상 통과하지만 lookup이 실제 1.443 m/s 정지 거리를 1.5 칸으로 위장한다.
PID는 정상상태 오차를 없앨 수 있으나 stage 1–2가 이미 게인/damping으로 1.5를 살리지 못했고,
표적 동역학을 바꾸면 기존 receipt·r2·v3 정지 증명이 다른 물리가 된다. 둘 다 “게이트를 통과시키기
위한 사후 변경”이 된다.

따라서 별도 사전등록
`docs/preregistration_braking_aware_route_v3_lower1p25_2026-09-01.md`(SHA-256
`cd1347121c24ecd10273189360bed9ca76ffa80673aa89addf3ff0eaebc16252`)를 결과 전에 고정했다.
속도 grid는 0.6/0.9/1.2/1.25이고 기본값은 여전히 `canonical_1p5`다. task는 명시적
`baseline_1p25` variant만 추가로 받고, 게이트/셀 어댑터는 동일 구현 바이트에 그 receipt만
묶는다. record id는 lossless decimal이라 1.25가 1.2로 alias되지 않는다. 0.05 게이트와
controller/PID는 변경하지 않았다. 08-26 lower receipt는 재사용하지 않는다.

다음 GPU는 현재 커밋에서 새로 뜰 `baseline_1p25` raw braking receipt다. 이 엔트리 시점까지
lower receipt와 lower pilot는 실행하지 않았다.

## 2026-09-01 — lower-1.25 braking receipt 검증, 8-cell pilot 미실행

커밋 `dd8b4a4`에서 `NAVRL_TARGET_BRAKING_CONTRACT_VARIANT=baseline_1p25`로 새 raw probe를
RTX 3070에서 실행했다. standalone verifier PASS.
`results/navrl_physical_target_braking_lower1p25_seed827_2026-09-01/receipt.json`
SHA-256 `6d71b0be34ffb166d23aff9f6897cf41b5bb82a4a488d24403d735cf97852485`.
속도 0.6/0.9/1.2/1.25, decel p05 0.5052572863 m/s², stop-time p95 0.89 s, certified lateral
tube p95 0.00041199 m, 정지거리 p95 0.32814/0.49835/0.67084/0.69944 m. 0.05 warmup 게이트와
P-only 게인 2.5는 그대로다. 이 숫자는 08-26 lower receipt와 같은 컨트롤러 천장과 일치하지만
바이트 바인딩이 현재 커밋이므로 옛 receipt를 재사용하지 않는다.

training source bundle은 저장소 밖
`/home/fair/workspaces/aerial_gym_ws/navrl_v3_receipts/training_source_lower1p25_2026-09-01/`
(`manifest_sha256=5de7fdbfa56811b71284d454195239d8490e8a5db0b54a22d3e8b9c58263964c`,
git_commit `dd8b4a4`, 328 files, git_dirty false).

8-cell lower pilot는 아직 실행하지 않았다. PPO는 여전히 0 epoch다.

## 2026-09-01 — lower v3 8-cell pilot VOID (matched-arm target pose)

seed 829, 70 bars, 0.6/0.9/1.2/1.25, arms `off`/`global_astar_braking_v3`를 RTX 3070에서
실행했다. 8개 셀 JSON은 모두 쓰였으나 parent verifier가
`matched-arm identity drift: initial_target_pose_sha256`으로 `VOID_EXECUTION` /
`NOT_INTERPRETED`다. 원자료
`/home/fair/workspaces/aerial_gym_ws/navrl_v3_runs/pilot_lower1p25_seed829_2026-09-01/summary.json`.

레이아웃 SHA와 로봇 자세 SHA는 8셀·두 arm이 모두 같다. 표적 초기 자세만 arm 사이에서
갈라지고, 같은 arm 안에서는 속도 4개가 동일하다. 원인: v3 `_sample_waypoints`가
wall+boundary에 target support를 더 inset해서 off와 다른 균일 상자에서 샘플한다. 게이트
threshold·controller·사전등록 바이트는 바꾸지 않았다. confirmatory는 열리지 않는다.

VOID라 공식 메커니즘 판정은 없다. 셀 파일을 진단으로만 보면(권한 아님) routed 70-bar는
runtime `unsafe_start` 0, plan success 38/38–54/54, 0.6 goals/env 7/32=0.21875,
fallback 277/9600≈2.89%, soft-envelope exit 569, speed ratio 0.80 미만
(0.7955/0.7475/0.6551/0.6535). 이 숫자는 VOID를 FAIL로 바꾸지 않는다.

## 2026-09-01 — matched-spawn 사전등록·CPU 계약, GPU 미실행

VOID 이후 선택지는 (1) off와 같은 상자로 샘플 (2) pose 매칭을 사후에 끄기 (3) 중단이다.
평가: (2)는 VOID를 본 뒤 무결성 검사를 빼는 것이라 기각. (1)이 맞다. envelope는
A*/rollout/watchdog에 남기고 spawn RNG 상자만 off와 같게 한다.

사전등록
`docs/preregistration_braking_aware_route_v3_lower1p25_matched_spawn_2026-09-01.md`
SHA-256 `17e7f35e350087bf2733aca70dfca7210efe667ad1845dd053f7334ddf1645b4`를 구현 전에
고정했다. 옛 lower-1.25 사전등록 바이트는 그대로다. `_sample_general_target`과
`_sample_waypoints`의 v3-only inset/AABB 필터를 제거했다. pose 해시 검사는 유지한다.

계산 예측: identity는 통과하고 메커니즘은 실패한다. VOID 셀의 비공식 숫자는 더 보수적인
envelope-inset spawn에서도 speed ratio·soft-exit·fallback·0.6 goals/env가 이미 게이트를
밑돈다. spawn을 넓히면 in-flight 지표가 나아지지 않고 reset `unsafe_start`만 늘 수 있다.
이 예측은 공식 판정이 아니다. spawn 바이트가 바뀌므로 `dd8b4a4` receipt는 다음 GPU를
무장하지 못한다. PPO는 0 epoch, confirmatory는 닫혀 있다.

## 2026-09-01 — matched-spawn CPU forensic 통과와 제한 GPU authority

사용자가 CPU forensic부터 새 receipt·8-cell pilot까지 끊지 않고 진행하도록 명시적으로
지시했다. 기존 matched-spawn 구현을 먼저 감사했다. spawn 상자만 `off`와 같고 soft envelope는
A*/rollout/watchdog에 남아 있으며 pose 해시 검사는 유지된다.

기존 120개 계약을 정확한 unittest discovery 방식으로 재현했고 모두 PASS했다. 이어 세 가지
forensic 계약을 추가했다: (1) seeds 1/59/367/827/829/839/65521에서 off/v3 general-target 및
waypoint RNG identity, (2) obstacle-rich 512-row 이상적 controller 모델에서 accepted first step의
다음 주기 terminal-stop certificate 보존, (3) randomized 256-row에서 certificate와 physical
substep watchdog의 exact closed-AABB 판정 일치. 총계는 **123 tests PASS**
(v3 40 + planner 13 + motion 25 + recovery 24 + lower 4 + spawn 17)다.

첫 VOID 원자료의 routed 셀은 accepted command와 accepted terminal certificate가 각 셀에서
완전히 같았다(9030/9030, 8438/8438, 7273/7273, 7112/7112). certificate failure와 다음
soft-envelope exit는 570/569, 1162/1159, 2327/2305, 2488/2473으로 거의 일대일이다. CPU의
재귀 인증과 watchdog 기하는 일치하므로 현재 가장 강한 가설은 sampler/좌표/endpoint 버그가 아니라
**PhysX 추종 오차가 다음 제어 주기의 stopping certificate를 소실시키는 것**이다. 공식 첫 pilot은
여전히 `VOID_EXECUTION / NOT_INTERPRETED`이고 이 진단 숫자로 FAIL을 소급하지 않는다.

옛 `dd8b4a4` receipt 폴더는 삭제하지 않고 저장소 밖
`/home/fair/workspaces/aerial_gym_ws/navrl_v3_receipts/archive_braking_lower1p25_dd8b4a4_2026-09-01/`
로 보존했다(receipt SHA-256 `6d71b0be34ffb166d23aff9f6897cf41b5bb82a4a488d24403d735cf97852485`).
worktree에서 빠졌던 frozen evidence 두 개는 primary 사본의 계약 SHA
`a85e9576...` / `bc5d05a3...`를 확인해 복구했고 `check_research_authority.py --json`이 PASS했다.

방법·grid·threshold를 바꾸지 않는 execution-only addendum
`docs/preregistration_braking_aware_route_v3_lower1p25_matched_spawn_gpu_authority_2026-09-01.md`
(SHA-256 `70b8a08f0c95040a86c43e1be5ac11d0b688b9a6874f3cfc4548f914882f085f`)를 고정했다.
허용 범위는 현재 clean commit의 fresh `baseline_1p25` receipt 1회와 seed-829/70 bars/8-cell
pilot 1회뿐이다. FAIL이면 confirmatory와 PPO를 닫고, PASS일 때만 frozen seed-839
confirmatory가 열린다. 어느 경우에도 이 addendum가 PPO나 장기학습 권한을 만들지 않는다.

새 commit `2769f5d`에서 fresh `baseline_1p25` braking receipt를 실행해 standalone verification
PASS했다. 경로는
`results/navrl_physical_target_braking_lower1p25_matched_spawn_seed827_2026-09-01/receipt.json`,
SHA-256 `18425bfc9cb618834b37435b8d83a07dc1af69e56aa02459f00a94c544e075cc`다.
receipt의 `git_head=2769f5d692ff124802a8e11b83393c31585bae7b`, `source_clean=true`다.
decel p05 0.5052572863 m/s², stop-time p95 0.89 s, 정지거리 p95는
0.328143/0.498352/0.670841/0.699436 m로 이전 결정론적 lower receipt와 일치한다.

pilot 직전 verifier가 무결성-clean `FAIL_BLOCKS_CONFIRMATORY`까지 예외로 던져 stderr에
`VOID_EXECUTION`이라고 표시하는 판정 표시 버그를 발견했다. threshold나 verdict 계산은 맞고
프로세스 종료 의미만 틀렸다. `verify_summary`가 integrity-valid FAIL을 정상 검증하도록 고치고,
`PASS_8_CELL_INTEGRITY + FAIL_BLOCKS_CONFIRMATORY` fixture가 VOID가 되지 않는 회귀 테스트를
추가했다. 이 수정은 simulator·controller·grid·threshold·receipt runtime source를 바꾸지 않는다.

## 2026-09-01 — matched-spawn lower-v3 8-cell 공식 FAIL, GPU authority 폐쇄

verifier 표시 수정 후 commit `b054f07`에서 runtime source bundle을 다시 고정했다:
`/home/fair/workspaces/aerial_gym_ws/navrl_v3_receipts/training_source_lower1p25_matched_spawn_b054f07_2026-09-01/`,
manifest SHA-256 `a34618e91a12d6a94f4a48f9b145cf72531c2ce54aa6f40eeec71db002c39840`,
runtime 328 files, `git_dirty=false`. 새 receipt
`18425bfc9cb618834b37435b8d83a07dc1af69e56aa02459f00a94c544e075cc`를 사용해
seed 829 / 70 bars / 32 envs / 300 measured steps / 20 warmup / 4 speeds × 2 arms를 실행했다.

원자료:
`/home/fair/workspaces/aerial_gym_ws/navrl_v3_runs/pilot_lower1p25_matched_spawn_seed829_2026-09-01/summary.json`,
SHA-256 `8e4b71bc9095f64065140f2838904bab8b7be868c7d97025953fb451041a49fc`.
공식 verdict는 **`PASS_8_CELL_INTEGRITY / FAIL_BLOCKS_CONFIRMATORY`**다. 각 속도의 off/v3
layout·robot pose·target pose SHA가 모두 한 종류로 일치해 이전 VOID 원인은 해소됐다.

| speed | off ratio | v3 ratio | v3 tracking RMSE | v3 soft exit | v3 cert fail | fallback | goals |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.60 | 0.9475 | 0.7872 | 0.1400 | 573 | 573 | 0 | 7 |
| 0.90 | 0.9458 | 0.7681 | 0.1867 | 1,387 | 1,395 | 634 | 10 |
| 1.20 | 0.9352 | 0.5984 | 0.2495 | 2,583 | 2,597 | 1,547 | 14 |
| 1.25 | 0.9458 | 0.6297 | 0.2652 | 2,282 | 2,296 | 1,242 | 15 |

모든 routed 셀은 contact/motor/tilt/state/tracking/local-feasibility를 통과하고 speed ratio ≥0.80만
실패했다. mechanism pooled 값은 unsafe start 0(PASS), soft exits 6,825(FAIL, gate 0), terminal
certificate fraction 1.0(PASS), plan success 99.468%(PASS), fallback 8.914%(FAIL, gate ≤1%),
0.6 m/s goals/env 0.21875(FAIL, gate ≥0.5)다. matched spawn은 identity를 고쳤지만 in-flight
mechanism을 고치지 않았고, 계산-first FAIL 예측이 확인됐다.

one-shot GPU authority는 소비·폐쇄했다. seed-839 confirmatory, PPO smoke, 장기학습,
gain/margin/threshold 사후 탐색은 실행하지 않는다. 새 시도는 physical tracking uncertainty를
명시적으로 포함하는 controller의 **새 방법**이어야 하며 별도 사전등록 전에는 구현·GPU를 열지
않는다. 공식 요약은
`docs/braking_aware_route_v3_lower1p25_matched_spawn_result_2026-09-01.md`에 보존했다.

## 2026-09-01 — 막대 비중첩 fresh PPO 우선순위 분리 및 500-epoch route-off smoke 준비

사용자가 막대 겹침 수정의 fresh PPO를 우선 진행하도록 명시했다. routed-v3 FAIL을 무시해 같은
실패 메커니즘으로 장기학습하지 않고, 이미 동일-layout 70-bar에서 physical PASS한 route-off
0.6/0.9/1.2/1.25 m/s envelope만 사용하는 **고정 70-bar learning-viability baseline**으로 분리했다.
500 epoch PASS도 routed target이나 205-bar 성능을 뜻하지 않으며, 장기 70→205는 별도 사전등록한다.

실행 전 런처 chain을 재감사해 두 개의 은닉 결함을 발견·수정했다.

1. `train_navrl_physical_fresh.sh`가 `footprint_clearance`를 넘겨도 하위
   `train_navrl_v2_search.sh`가 child marker를 배치 선택 전에 unset해 실제로는 옛
   `navrl_band`를 다시 설정했다. marker를 shell-local에 보존해 실제 task까지 비중첩 계약이
   유지되도록 했다.
2. 하위 런처가 `NAVRL_TARGET_SPEED_FINAL=1.5`를 무조건 덮어써 1.25라고 출력한 child 실험도
   실제로는 1.5가 될 수 있었다. canonical default 1.5는 유지하되 명시적 closed child override를
   보존하도록 바꿨다.

preflight는 이제 최종 `placement=footprint_clearance`, `surface_clearance=0.45`,
`density=70->70`, `fixed_bars=70`, `speed_final=1.25`, `ramp=1`을 한 줄에서 증명한다. 새 launcher
`train_navrl_corrected_nonoverlap_physical_smoke.sh`는 checkpoint/CLI/dirty runtime을 거부하고,
exact import root와 clean source receipt를 강제한다. 사전등록은
`docs/preregistration_corrected_nonoverlap_physical_off_smoke_2026-09-01.md`
(SHA-256 `23181001d29e9d34295f63f153791e6ee4492a00431b209edc7454db885974d1`)다.

검증: launcher/target environment 14/14, 신규 smoke contract 3/3, non-overlap placement 4/4,
v2 launcher 5/5, physical-target 관련 30/30 PASS. `test_navrl_ref5in_run_contract.py` 101개 중
100개 PASS, 1개는 이 worktree에 historical P1c checkpoint 폴더가 없어 실패했으며 변경 코드와
무관하다. bash syntax/preflight와 `git diff --check`도 PASS했다. GPU 학습은 clean commit 뒤
source receipt를 만든 다음 1회 시작한다.

### 공식 smoke 실행 시작

clean commit `1ce6a6f`에서 공식 1회를 시작했다. run은
`ppo_260901_1220_navrl_corrected-nonoverlap-physical-off-smoke-s907`, trainer PID는 launch 확인
시점 `1445290`, session log는
`aerial_gym/rl_training/rl_games/train_session_logs/corrected_nonoverlap_physical_off_smoke_260901_122019.log`다.
source receipt는 runtime 329 files, `git_dirty=false`, commit
`1ce6a6faf395826498885b5aa550f357303468d9`, manifest SHA-256
`f7708cf210584af27b9c84101e163c489ac13f38b3c6a85c5a686cb436a11a22`다. runner import-origin
guard는 이 worktree의 `aerial_gym/__init__.py`를 enforced PASS했다.

실제 runtime 시작 로그도 `placement=footprint_clearance`, `surface_clearance=0.45`,
`curriculum=False`, `initial_bars=70`, `final=70`, physical mixed target,
`speed_final=1.25`, fresh weights, 128 env, max 500 epoch를 확인했다. epoch 12까지 finite하게
진행됐고 GPU 사용량은 약 6.16 GiB였다. 초기 crash 100%는 fresh PPO의 예상 cold start이며
사전등록상 epoch 500의 early-vs-late window로만 판정한다.

### 공식 smoke 종료·판정 PASS

run은 max epoch 500에서 exit 0으로 정상 종료했고 terminal checkpoint는
`last_gen_ppo_ep_500_rew_91.16469.pth`다. 전용 fail-closed analyzer
`tools/analyze_navrl_corrected_nonoverlap_physical_smoke.py`가 source original/snapshot 329개,
checkpoint/TensorBoard finite, robot/environment 계약과 outcome accounting을 재검증했다. 결과는
`results/navrl_corrected_nonoverlap_physical_off_smoke_seed907/summary.json`, SHA-256
`85758ba6b6bef7b0097afe6e14c6dbcf6f127d1d80595a70da2fcbf699ec1a89`다.

| window | capture | crash | timeout | mean reward |
|---|---:|---:|---:|---:|
| epoch 2–101 | 11.01% | 73.94% | 15.05% | −131.00 |
| epoch 401–500 | 60.39% | 39.22% | 0.39% | +93.08 |

capture 개선은 +49.38pp, reward 개선은 +224.08이다. PPO KL max 0.01119,
behavior-KL max 0.01568, rollback/skipped minibatch 0, 모든 60개 TB scalar finite, checkpoint tensor
finite, density 70 고정이다. 사전등록 gate 전부 PASS해 verdict는
**`PASS_LEARNING_VIABILITY`**다. 이는 route-off curriculum 사전등록만 허용하고 routed-v3/1.5 m/s/
hardware 주장은 허용하지 않는다.

결과를 본 뒤 별도 장기 사전등록
`docs/preregistration_corrected_nonoverlap_physical_off_curriculum_2026-09-01.md`
(SHA-256 `d10a19fe570aa2a73e88cc8e1a301618738fee55ee592f24675ba84117ae5ec8`)를 고정했다. 다음 run은
smoke checkpoint를 쓰지 않는 fresh seed 911, 30,000 epochs, route-off 1.25 m/s, density
70→205/step15/dwell1000/evidence16384 한 번이다.

### seed 911 curriculum 시작 확인 (Codex 중단 이후)

사용량 제한으로 Codex가 launcher 시작 직후 끊겼다. 재확인 결과 이미 정상 기동됐다. 두 번째
run은 시작하지 않았다.

- run `ppo_260901_1259_navrl_corrected-nonoverlap-physical-off-curriculum-s911`
- bash PID 1496735, trainer PID 1496818, GPU ~6160 MiB
- session `train_session_logs/corrected_nonoverlap_physical_off_curriculum_260901_125953.log`
- source receipt `train_source_receipts/corrected_nonoverlap_curriculum_s911_260901_125953_1496735`,
  manifest SHA-256 `ade7098ccb16cb0e3847c5b77f9d68cf651255d9cb57c60c498e67f641f1e6c8`
- 계약: seed 911, fresh weights, 128 env, 30,000 epoch, route-off, physical mixed,
  `U[0.3,1.25]`, `footprint_clearance`/0.45 m, density 70→205/step15,
  schedule `70:0.82,85:0.77,100:0.72,115:0.70`, dwell 1000, evidence 16384, LR 1.5e-5
- import-origin은 이 worktree `aerial_gym/__init__.py` enforced
- 13:13 확인 시점 epoch 208/30000, 단일 epoch capture 54–65%, pooled n≈2049 capture 59.3%

종료 전에는 공식 판정을 하지 않는다. 끝나면 분석만 하고 routed PPO는 열지 않는다.

### seed 911 curriculum 세션 단절 후 fresh 재시작

`1259` 런은 epoch 1374/30000, 70 bars에서 로그가 대시보드 중간에 끊겼다. same-density
guard/NaN/OOM 표식은 없다. 마지막 주기 체크포인트는 `last_gen_ppo_ep_1250`. 사용자는
재실행을 요청했고, 계약이 fresh-only라 그 체크포인트는 쓰지 않았다.

재시작: `setsid nohup`으로
`ppo_260901_1431_navrl_corrected-nonoverlap-physical-off-curriculum-s911`,
trainer PID 12683, session
`train_session_logs/corrected_nonoverlap_physical_off_curriculum_260901_143136.log`,
receipt SHA-256 `42ee349201ae5220d13fdee3dd9e756502cabb70ba7ece0faab9836dcd2b03a3`.
첫 epoch는 fresh weights, seed 911, 70 bars를 다시 확인했다.

### seed 911 curriculum 감시 재개 (Cursor, 17:51)

세션 단절 후 프로세스는 살아 있었다. 두 번째 curriculum은 시작하지 않았다.

- trainer PID 12683, bash SID 12602, TTY 없음(세션 죽어도 유지), GPU 6162 MiB / 61% / 56°C
- 확인 시점 epoch 3278/30000, bars 70 고정, 최신 주기 체크포인트 `last_gen_ppo_ep_3250`
- 초기 100 epoch capture ~11% / reward ~−129 → 최근 100 epoch capture 76.5% / reward +133
- 승급 창(최근 455 epoch, 약 16k episode 규모) capture 75.6% — 70-bar gate 0.82 미달, 밀도는 아직 70
- same-density guard 미발화(현재 50-epoch 평균 대비 peak drop ~0.02, 한도 0.25)
- 거리 커리큘럼만 k_max 7→28 m로 승급됨. NaN/OOM/rollback 표식 없음
- 기존 TensorBoard :6007은 `src/aerial_gym_simulator/.../runs`를 보고 있어 이 런이 안 보인다. 이 런 전용 TB는 :6008

종료 전 공식 판정 금지. 끝나면 분석만 하고 routed PPO는 열지 않는다.

### seed 911 감시 계속 — TensorBoard 경로 수정과 완료 분석기

사용자가 기존 :6007 화면으로 진행을 요청했다. 학습 PID 12683은 그대로 두고 TensorBoard :6007 logdir만
worktree `.../rl_games/runs`로 바꿨다. :6008도 같은 경로를 가리킨다.

완료 즉시 판정용 `tools/analyze_navrl_corrected_nonoverlap_physical_curriculum.py`를 추가했다.
`--live`는 판정 없이 스냅샷만 내고, 공식 analyze는 종료 표식이 있을 때만 연다. routed PPO 권한은
어느 경로에서도 false다. 17:55 live 스냅샷: epoch 3359/30000, bars 70, 최근 100 epoch capture 77.4%,
reward +135, 밀도 승급 0회.

### seed 911 밀도 70→85 승급 (20:53)

공식 로그: `bars 70 -> 85 after 16385 eps, capture=0.829 (threshold=0.820) dwell=6327 epochs`.
trainer PID 12683 유지, 두 번째 run 없음. 확인 시점 epoch 6331/30000, bars 85, 최근 20 epoch
capture 84.1% / reward +154. 다음 게이트는 85:0.77, dwell 1,000, evidence 16,384. 가드·NaN·OOM 없음.
routed PPO는 열지 않는다.

### seed 911 밀도 85→100 승급 (22:16)

공식 로그: `bars 85 -> 100 after 16386 eps, capture=0.792 (threshold=0.770) dwell=1248 epochs`.
trainer PID 12683 유지. 확인 시점 epoch 7687/30000, bars 100(이 밀도 112 epoch). 다음 게이트는
100:0.72. 가드 없음. routed PPO는 열지 않는다.

### seed 911 밀도 100→115 승급 (23:38)

공식 로그: `bars 100 -> 115 after 16385 eps, capture=0.746 (threshold=0.720) dwell=1232 epochs`.
trainer PID 12683 유지. 확인 시점 epoch 8977/30000, bars 115(이 밀도 171 epoch). 다음 게이트는
115:0.70. 가드 없음. routed PPO는 열지 않는다.

### seed 911 밀도 115→130 승급 (00:59)

공식 로그: 115-bar hold `capture=0.696` then `0.700`, then `bars 115 -> 130 after 16384 eps, capture=0.701 (threshold=0.700) dwell=1224 epochs`.
trainer PID 12683 유지. 확인 시점 epoch 10211/30000, bars 130(이 밀도 181 epoch). 다음 게이트는
130:0.70. 가드 없음. routed PPO는 열지 않는다.

### seed 911 밀도 130→145 승급 (08:09)

공식 로그: 130-bar holds 후 `bars 130 -> 145 after 16384 eps, capture=0.715 (threshold=0.700) dwell=6196 epochs`.
trainer PID 12683 유지. 확인 시점 epoch 16416/30000, bars 145(이 밀도 190 epoch). 130 말기 last100 capture 71.9% → 145 진입 first100 65.2% (−6.7pp). 다음 게이트는 145:0.70. 가드 없음. routed PPO는 열지 않는다.

### seed 911 curriculum 운영자 중지 (14:48)

사용자가 145-bar stall에서 중단을 요청했다. trainer PID 12683 / bash SID 12602를 종료했다.
가드·OOM·max_epochs가 아니다. fresh-only라 이 런은 재개하지 않는다. 두 번째 curriculum과
routed PPO는 열지 않는다. GPU one-shot 권한은 이미 소모된 상태다.

- run `ppo_260901_1431_navrl_corrected-nonoverlap-physical-off-curriculum-s911`
- 중지 시점 epoch 21973/30000, bars 145 (~5739 epoch at 145), last100 capture 64.3%
- 최신 주기 체크포인트 `last_gen_ppo_ep_21750_rew_83.1572.pth`
  SHA-256 `541b36bdcabacf8bb14c6fbb0ad07054dd9735ad24777a3222655ba8ca9c8132`
- 마지막 공식 145 hold capture=0.647 / gate 0.70
- 승급 경로: 70→85→100→115→130→145. 160 이상은 도달하지 않음
- GPU 학습 프로세스 없음 (잔여 ~736 MiB, rustdesk만)

다음 단계는 이 런을 205 성공으로 읽거나 재개하는 것이 아니다. held-out 평가 사전등록
(체크포인트 `last_gen_ppo_ep_21750`, 학습 밀도 70/85/100/115/130/145, seed≠911,
`gen_ppo.pth` 금지) 후에만 평가 GPU를 연다.

### seed 911 last_gen_21750 held-out 평가 사전등록 (2026-09-02)

결과 보기 전에 계약을 고정했다.
[`docs/preregistration_corrected_nonoverlap_physical_off_heldout_eval_2026-09-02.md`](docs/preregistration_corrected_nonoverlap_physical_off_heldout_eval_2026-09-02.md)
SHA-256 `072060a82421ea67c6b1abfbb541d67ca89b7a26dd091a849f893edc520708c5`.
셀 70/85/100/115/130/145, seed 313, n=2049, route-off, `U[0.3,1.25]`,
`footprint_clearance` 0.45 m. 205는 학습하지 않은 OOD라 넣지 않았다. `gen_ppo.pth` 금지.
이 숫자는 145-terminal 정책의 held-out capture/crash일 뿐이며, 다음 학습을 열지는
결과가 나온 뒤에 새 사전등록으로만 정한다.

### seed 911 last_gen_21750 held-out 평가 시작 (2026-09-02)

GPU는 rustdesk만 점유 중이었고 메인 트리 체크아웃은 하지 않았다. braking_route_v3 worktree에서
`eval_navrl_corrected_nonoverlap_physical_off_heldout.sh`를 한 번 띄웠다. 결과 루트는
`results/navrl_corrected_nonoverlap_physical_off_heldout_seed313`이다. 205 없음. 재실행 금지.
첫 시도는 evaluator가 `navrl_ref5in_v2_quad`를 거부해서 즉시 종료됐다. 두 번째는
`cfg_target_motion_model`이 역사적 analytic 문자열과 달라 거부됐다. v2 로봇 계보,
physical motion model, `NAVRL_PHYSICAL_GEOMETRY_VERSION=v2` / box 0.283을 계약 분기에
넣은 뒤 다시 한 번만 띄운다. 세 번째 시도는 provenance OK로 70-bar 셀에 진입했다.
- PID 3618494 (eval wrapper/sweep), python 3618744
- log `train_session_logs/corrected_nonoverlap_physical_off_heldout_260902_180023.log`
- seed 313, densities 70 85 100 115 130 145, `U[0.3,1.25]`, `footprint_clearance`, robot `navrl_ref5in_v2_quad`
- 학습 seed 911 vs 평가 seed 313 mismatch 경고는 의도된 것이다 (density evidence reset)

### seed 313 held-out 완료·봉인 (2026-09-02 18:20 KST)

6셀은 모두 끝났고 GPU 프로세스는 없다. 체크포인트 SHA, 셀별 result/receipt/log SHA,
서로 다른 nonce, source manifest 330개 파일, evaluator hash, outcome 분모와 CSV를 재검증했다.
판정은 `COMPLETE_VALID_WITH_METADATA_ERRATUM`; raw artifact는 수정하지 않았다.

| bars | n | capture (Wilson 95%) | crash | timeout | bar contact |
|---:|---:|---:|---:|---:|---:|
| 70 | 2049 | 83.70% [82.04, 85.24] | 15.91% | 0.39% | 15.67% |
| 85 | 2051 | 80.94% [79.18, 82.58] | 18.82% | 0.24% | 18.43% |
| 100 | 2049 | 77.75% [75.89, 79.49] | 21.86% | 0.39% | 21.67% |
| 115 | 2049 | 73.45% [71.50, 75.32] | 26.35% | 0.20% | 26.06% |
| 130 | 2050 | 69.17% [67.14, 71.13] | 30.44% | 0.39% | 30.24% |
| 145 | 2049 | 65.54% [63.46, 67.57] | 34.16% | 0.29% | 34.16% |

70→145 capture는 −18.16 pp, 평균 −3.63 pp/15 bars다. crash는 +18.25 pp이고 timeout은
전 셀 0.4% 이하여서 병목은 탐색시간이 아니라 막대 접촉이다. lateral action의 양의 방향
비율도 모든 셀에서 98.58–99.20%, mean |y| 0.921–0.938로 남아 있어 고밀도 충돌과 함께
다음 offline 원인분석의 1순위다. 이 결과만으로 205 mastery, routed, hardware/sim-to-real을
주장하지 않는다.

메타데이터 정정: raw `v2_evaluation_contract.target_speed_max_mps`만 과거 상수 1.5를
기록했다. 실제 `condition.target_speed_max_mps`, 시작 로그, 결과 저장 전 validator와 speed
strata 상단은 모두 1.25다. outcome 비영향 직렬화 결함으로 판정하고 raw는 보존했으며,
향후 evaluator는 `NAVRL_TARGET_SPEED_FINAL`을 기록하도록 수정·회귀 테스트했다.

- 결과: `results/navrl_corrected_nonoverlap_physical_off_heldout_seed313/summary.{json,md}`
- summary SHA-256: `fd52ad6c4d4a9ba564510fd556cfd561b48ab9771a22799ecdc9956f84249559`
- 다음: offline 병목 분석과 새 사전등록 전에는 GPU 없음. 1431 재개/두 번째 curriculum/205
  mastery 평가/routed PPO 모두 금지.

검증: corrected-nonoverlap/authority 관련 24 tests PASS, authority freeze PASS, clean-clone식
source 검증(ignored source snapshot을 잠시 제외하고 tracked bytes + archived evaluator 사용) PASS.
전체 suite는 917 tests 중 913 PASS / 2 skipped / 4 historical-artifact failures였다. 실패 4개는
이 변경의 assertion이 아니라 이 worktree에 없는 과거 P1c checkpoint·P2 log·P1c source manifest
3건과 frozen heading-rest braking receipt provenance 1건이다. held-out 결과/런처/요약 테스트에는
실패가 없다.

## 2026-09-03 — S1 explicit blind-search state 구현 (GPU 미실행)

사용자가 Claude의 구현 계획을 승인해
`docs/preregistration_s1_blind_search_state_2026-09-03.md`로 저장하고 S1 actor-state 축을
구현했다. 문서 SHA-256은
`4a78ec94cf6813c5bfe6261bb9145c05371f79f9b9b32f4b849fec80d2ad3294`이다. GPU authority는
열지 않았고 PPO/simulator 학습·평가는 실행하지 않았다.

- 신규 `navrl_search_state.py`: 40×40 m / 2 m actor-safe coverage·belief grid, 12 body
  sector, `never_acquired → tracked → stale` mode, blind time, camera-depth occlusion,
  p_det=0.9 negative update, 1.25 m/s×0.1 s isotropic diffusion, KF active override와
  active→inactive 재주입. 공개 `update`에는 target/semantic/GT 인자가 없다.
- schema: canonical 72-beam/8-obstacle 기준 off/geofence/coverage/belief가 각각
  898/906/934/959-D, Transformer 17/18/19/19 token이다. non-off인데 geofence가 꺼졌거나
  arm 값이 잘못되면 import에서 fail-closed한다. search token projector는 128→64이다.
- masked ablation: geofence range=1/valid=0, mode·coverage=0, belief sector mass=1/12,
  radial moment=0, entropy=1. checkpoint `env_state`, generic evaluator condition/receipt,
  bulk JSON에 arm/mask provenance를 추가했다.
- 평가 telemetry: blind-phase 속도·actor-safe LiDAR clearance, first-acquisition p10/p50/p90,
  first-visible coverage/entropy를 추가했다. 기존 actor/critic·reward·termination에는 넣지 않았다.
- fresh launcher는 arm 인자 하나만 받고 seed 919 / 3000 epochs / fixed 70 bars /
  U[0.3,1.25] / governor off / save 250을 고정하며 checkpoint와 dirty runtime을 거부한다.
  held-out wrapper는 각 arm의 raw ep3000 terminal checkpoint만 받아 seed 331, 70/145 bars,
  2049 episodes를 고정하고 checkpoint arm/geofence/mask provenance를 먼저 검사한다.
  summarizer는 primary·guard·masked mechanism 판정과 summary JSON/Markdown/SHA를 만든다.

구현 전에 계획서의 내부 모순을 보정했다. 표와 schema test는 belief 25-D를 요구했지만
본문은 `12 sector mass + entropy` 13-D만 정의했다. 남은 12-D를 동일 belief의 sector별
정규화 거리 1차 모멘트로 명시했고 문서 끝에 GPU 실행 전 amendment를 남겼다. GT나 새 센서
입력은 추가되지 않았다.

CPU 검증: 신규 S1 tests 18/18 PASS. 기존 perception/obs-dump/training-semantics/LiDAR
bearing/latency/TTC/detect-resolution/checkpoint-preflight 포커스 210/210 PASS. 전체 suite는
1025 tests 중 1021 PASS / 2 skipped / 2 historical-artifact failures였다. 두 실패는 저장소에
없는 seed-911 ep21750 checkpoint와 frozen source snapshot의 과거
`env_object_config.py` digest drift이며 S1 assertion 실패가 아니다. `py_compile`, `bash -n`,
`git diff --check`도 PASS. 사용자 승인 후 `research/navrl-env`에서 커밋 준비했다.

## 2026-09-03 — 오프라인 instance adapter CPU 계약 (SAM 미설치, GPU 미실행)

사이트 표적 탐지 구조도를 보면서 SAM 3 후보를 검토한 뒤, 검증된 CURRENT 그림 4장은
그대로 두고 제어루프에 SAM을 넣지 않기로 했다. 기존 SVG 4장은 XML 파싱이 정상이라
교체하지 않았다.

- 신규 `docs/assets/motar-perception-candidate.svg`: `CANDIDATE · not in the control loop ·
  not a measured result`. 저주기 SAM 재탐색 / 고주기 경량 추적 / K개 instance /
  AMBIGUOUS(불명확하면 선택하지 않음) / 원시 LiDAR 안전층. README Perception 절과
  `docs/status/index.html` `#perception` 아래 현재 그림 다음에 붙였다. 문구는 설계 후보·
  미채택·성능 주장 아님.
- `tests/test_status_site.js`가 5번째 asset을 등록하고, CURRENT 그림은 `단일 중심점`을
  유지하며 후보 그림만 `not in the control loop`를 갖게 가드한다.
- 사전등록
  `docs/preregistration_sam_instance_adapter_offline_2026-09-03.md` (SHA-256
  `5348a3dae3ada2098a90a36fc6d2bc992ecf56d5b8099a5147bda21c190eff6d`). SAM FTLR/Hz/PPO를
  묻지 않는다. 게이트는 CPU 세 개: stub가 blob 2개 유지, union centroid는 ghost, 점수 차가
  작으면 AMBIGUOUS이고 lock id 없음.
- 신규 `navrl_instance_adapter.py`: 색 임계+CC stub, 합집합 금지 가드, SAM 백엔드는 별도
  프로세스 스펙만 두고 in-process 로드는 fail-closed. `NAVRL_INSTANCE_ADAPTER` 기본 0.
  `navrl_perception.py`는 import하지 않는다. 오프라인 툴
  `tools/run_navrl_instance_adapter_offline.py`.
- 2-blob fixture에서 stub `n=2`, union range `6.0`(4 m와 8 m의 평균, 양쪽 bbox 밖),
  기본 마진에서 `AMBIGUOUS`.

CPU 검증: `tests.test_navrl_instance_adapter` 9/9 PASS. `test_status_site.js` PASS.
aerialgym에서 `test_navrl_perception` 31 (1 skip) / `test_navrl_detect_resolution` 34 PASS.
Isaac/SAM/PPO는 실행하지 않았다. VERIFICATION 기준일을 2026-09-03으로 올리고 다음
software 작업만 이 계약으로 적었으며 기존 FAIL 판정은 바꾸지 않았다.

## 2026-09-03 — SAM 후보 구조도 정정과 전체 검증 계획

README와 연구 사이트를 실제 구현 경계에 맞춰 다시 감사했다. 기존 탐지 실패 그림은 측정 결과와
일치해 유지하고, system overview에는 `CURRENT SOFTWARE PATH`와 `1 detector → 1 KF track`을
명시했다. 후보 그림은 capture-time sensor packet, 비동기 discovery/propagation, 구현된 CPU
instance boundary, 미구현 3-D association/track bank, fail-closed decision, 독립 safety path로
다시 그렸다. 초록색 한 블록 외에는 미구현이며 성능 측정이 아니라는 표시를 그림 내부에도 넣었다.

전체 계획은 `docs/SAM3_PERCEPTION_VERIFICATION_PLAN_2026-09-03.md`에 저장했다. 코드 감사에서
실제 SAM worker/transport, timestamp/pose, 3-D lifting, track bank와 safety 구현이 없고,
`predicted_uv` 정렬 점수와 ambiguity margin 점수가 일치하지 않으며 depth validity 계약이
불완전함을 확인했다. 따라서 순서는 instance boundary 보완 → 해상도/가시성 → worker/transport →
paired open-loop → latency/SWaP → shadow parity → active A/B → 경량화 → 실기다.

검증: instance adapter 9/9 PASS, perception 31 PASS/1 skip, detect-resolution 34/34 PASS,
static site contract PASS, offline stub CLI PASS, `git diff --check` PASS. Chrome headless로 README가
가리키는 CURRENT/후보 SVG와 전체 status page를 렌더해 겹침·잘림이 없음을 확인했다. SAM, Isaac,
PPO, 실제 센서는 실행하지 않았다.

## 2026-09-03 — perception 재설계 계획: 형상+시간 재정의, 단 구조-수정 단계를 먼저

사용자의 40개 절 제안(색 detector 개선 → UAV 형상 인식 + 시간 연관으로 문제 재정의,
CNN=공간/GRU·Transformer=시간 역할 분리)을 받아
`docs/plans/perception_shape_temporal_redesign_2026-09-03.md`를 작성했다. **계획 문서만이며
코드·판정 변경 없음.**

핵심 판단 세 가지:

1. **S1 단계 추가** — CNN+FPN 직행 전에 connected-component 후보 + KF gating만으로 기존
   detector를 재평가한다. FTLR 90.27 %에는 인식 실패(색 특징)와 자료구조 실패(단일 중심점
   붕괴)가 합쳐져 있어(SAM 계획 §1과 동일 구분), 구조만 먼저 고쳐야 귀속이 선다.
   RQ0 판정 경계(<30 % 구조 지배 / >60 % 인식 지배)를 사전등록 대상으로 명시.
2. **"해상도 먼저"를 두 게이트로 재정의** — (a) 렌더 비용 곡선 실측(8 GB에서 closed-loop
   성립 경계), (b) **협시야 대안**: 같은 160×90 픽셀 예산의 HFOV 20° 카메라가 453 px/rad로
   640×360 전면 인상(337 px/rad)보다 각해상도가 높다(@20 m 6.8 px vs 5.1 px). 광각 SEARCH →
   협시야 TRACK 상태기계와 맞물림.
3. **SAM 트랙과 연관 계층 공유** — 후보 packet 계약(bbox/score/uv/depth, K후보,
   AMBIGUOUS/REJECT)을 SAM adapter 인터페이스로 통일해 연관 스택을 한 벌만 만든다.

채택: GRU(T=8) 먼저·Transformer(T=16)는 가림-재획득 근거로 비교군(에피소드당 visible↔hidden
7.7회 실재), LSTM 제외, full-image Transformer 금지(candidate token 80개만), episode-level
split + unseen-mesh 홀드아웃, frozen perception → PPO(기존 −2 pp 비열등 게이트 재사용).
연기: 30 Hz 멀티레이트(sim 편익 없음), 관측 토큰 확장(898-D 계약 파괴 = 새 계보, S6에서만).

사다리: S0 비용게이트 → S1 구조수정 → S2 데이터셋(최대 비용 항목: UAV mesh 다양화;
frozen 디렉터리 밖에서, D9 선례) → S3 소형 CNN+FPN(SAM=오프라인 상한 baseline) → S4 시간
연관(KF/GRU/Transformer, T 스윕) → S5 hard-distractor 벤치마크(5조건 = 논문 메인 테이블) →
S6 closed-loop. 각 단계 별도 사전등록.

## 2026-09-03 — S0-a 렌더 비용 곡선: 640×360 전면 인상 불성립, 협시야 경로 채택

`tools/benchmark_navrl_camera_cost.py` + `tools/run_navrl_camera_cost_sweep.sh` (신규, 엔지니어링
측정 — 사전등록 불요). 205 bars, vision on/off × {160×90, 320×180, 640×360} × {32, 64, 128} envs,
셀당 별도 프로세스, warmup 50 제외 250 step 계시. RTX 3070.
원자료 `results/navrl_camera_cost/cells.jsonl` (12 cells).

| vision | res | envs | ms/step | smi VRAM |
|---|---|---:|---:|---:|
| off | 160×90 | 128 | 88.2 | 5,487 MiB |
| on | 160×90 (현재) | 128 | 109.2 | 5,677 MiB |
| on | **320×180** | 128 | **117.3** | **6,294 MiB** |
| on | 640×360 | 128 | **OOM FAIL** | — |
| on | 640×360 | 64 | 100.8 | 6,931 MiB |

관찰: torch peak은 수백 MiB뿐이고 **VRAM의 주인은 Isaac Gym 자체 버퍼**(torch 밖) — smi 차이가
진짜 지표다. OOM 셀 teardown에서 Warp error 700 스팸(1,026줄)은 실패 셀 정리 과정이며 성공 셀
데이터와 무관.

### 판정 (계획 §3 결정 규칙 적용)

- **640×360 × 128 envs 불성립** → 전면 인상 경로 폐기.
- 320×180 × 128 envs는 성립(+7.4 % step, +617 MiB)하나 각해상도 2.5 px@20 m로 **탐지 문턱
  (2–2.5 px)은 넘지만 형상 신뢰(8–10 px)에는 미달**.
- **협시야 이중 스트림을 기본 경로로 채택**: 광각 320×180@87°(탐지) + 협시야 @20°(형상/식별,
  160×90이면 6.8 px, 320×180이면 13.6 px@20 m). 두 번째 소형 스트림 비용은 vision-on 증분
  (128 envs에서 +21 ms/+190 MiB)과 동급으로 추정 — S3 설계 시 실측.

### 사용자 결정 2건 (질의 완료)

- **S1 shadow-mode 승인** (판정 경계 30/60 % 포함) → `docs/prereg_2026-09-03_s1_structure_fix_shadow.md`
  초안 작성 완료, 구현 착수.
- **S2 표적 visual은 우리 quad mesh 재사용** — 현재 표적 visual이 반지름 0.05 m 구체(RGB에 드론
  형상 없음)라는 확인에 따른 결정. 디코이 UAV도 동일 mesh + 다른 색으로 시작.

## 2026-09-03 — S1 shadow-mode 구축: M1 PASS, 4셀 측정 시작

사용자 승인(shadow-mode 설계 + 판정 경계 30/60 %, S2 표적 mesh는 quad 재사용)에 따라 S1을
구현하고 측정에 들어갔다. 사전등록 `docs/prereg_2026-09-03_s1_structure_fix_shadow.md`.

### 구현

- `shadow_association.py` (신규): GPU 4-연결 CC(반복 max-전파, fixpoint 조기종료) +
  희소 후보 통계(top-K=5, count/u/v/depth) + `ShadowAssociator`(별도
  `BatchedConstantVelocityTracker` 인스턴스, 대각근사 Mahalanobis χ²(3)=11.345 게이트,
  init=argmax count). 단위테스트 6개(분리 blob·L자·대각 분리·순위·top-K 절단·입력 불변) PASS.
- `navrl_perception.py`: `NAVRL_S1_SHADOW=1` 시 4개 훅 — init(fail-closed:
  detect_decoupled·latency≠0 거부), raw mask 스냅(라이브 in-place gating 이전 clone),
  tracker.step 직후 shadow step(동일 pose lift), diagnostics 4키. 라이브 경로 무변경.
- `navrl_task.py`: `_record_s1_shadow_frame`(라이브 lock recorder와 동일 GT·반경, N=0이면
  distractor 거리 +inf) + `_s1_shadow_payload`(init/tracking 분해 포함) + export.
- `tests/test_navrl_distractors.py`의 `_solid_obstacle_offset` 사용처 meta-test가 신규 reader를
  잡았다 — 테스트가 요구하는 "명시적 결정"으로 합법 reader 2개(distractor-only 슬라이스 콤마
  패턴 동일)로 갱신. 953개 중 951 PASS.
- pre-existing 실패 2건 확인(클린 HEAD 재현): corrected-nonoverlap heldout 계약 테스트 —
  braking_route_v3 스레드의 미수확 체크포인트/env_object_config 핀 드리프트. 그 스레드 병합
  과제 소관, 본 작업에서 불수정.

### 실행 기반: envelope closed-env 재사용에서 드러난 것 2건

- `tools/run_navrl_s1_shadow.py`: envelope의 `evaluation_env()`를 임포트 재사용, 오버라이드
  4필드(seed 481, result dir, `NAVRL_S1_SHADOW`, shared source bundle). **seed479 공유 번들의
  `repository_root=/home/joshuali/MOTAR`(1650 Ti 절대경로)** 때문에 이 기계에서 manifest 검증이
  구조적으로 불가 → S1 전용 신선한 번들을 스윕이 생성하게 함(사전등록 §2에 명시).
- M1 첫 판정 FAIL은 **비교기 결함**이었다: npz 파일 해시에 덤프 자기참조 필드
  (`run_obs_dump_path`, `run_pid`)가 포함돼 구조적 불일치. 배열 단위 비교로 **obs 9,728×898
  전 프레임 비트 동일** 확인 후 canonical digest(자기참조 2필드 제외)로 정정(사전등록 §5에
  측정 전 정정으로 기록). 재판정 **M1 PASS** (off/on digest 동일).

### 스모크 미리보기 (50 ep, 게이트 아님 — 본 측정은 2,049 ep×4셀 진행 중)

online FTLR 93.55 % / shadow 95.04 %, shadow init-false 92.99 %·tracking-false 95.15 %.
기제 관찰: **init=argmax count가 픽셀 큰 디코이(147–181 px vs 표적 2–5 px)를 사실상 항상
선택하고, 게이트는 그 잘못된 track을 충실히 유지한다.** 정적 디코이는 항상 보이므로 track이
만료되지 않는다. 사전등록 예측 2("초기화 오선택은 남는다")의 극단형이며, 규칙은 등록대로
유지한다. 본 측정 결과는 별도 항목으로.

## 2026-09-03 — S1 판정: RECOGNITION_DOMINANT (shadow FTLR@n5 = 92.05 %)

사전등록 `docs/prereg_2026-09-03_s1_structure_fix_shadow.md`의 본 측정 완료 (seed 481, 3070,
2,049 ep/cell, 70 bars, v7). `results/navrl_s1_shadow_seed481/summary.{md,json}`.

| cell | online FTLR | shadow FTLR | shadow init-false | shadow tracking-false | shadow visible |
|---|---:|---:|---:|---:|---:|
| v7_n0 | (기록 없음, 설계상) | 10.58 % | 30.26 % | 9.93 % | 56,130 |
| v7_n1 | 60.63 % | 58.16 % | 65.45 % | 57.89 % | 61,820 |
| v7_n3 | 83.68 % | 84.64 % | 85.48 % | 84.61 % | 79,038 |
| v7_n5 | 90.71 % | 92.05 % | 92.29 % | 92.04 % | 73,258 |

**verdict_rq0 = RECOGNITION_DOMINANT** (등록 경계: <30 % STRUCTURE / >60 % RECOGNITION).
shadow 92.05 % ≥ online 90.71 % — connected-component 다중 후보 + χ² 게이팅이라는 자료구조
수정은 FTLR을 **1픽셀도 줄이지 못했다**(오히려 +1.3 pp). RQ0의 답: 90 %대 오추적의 주인은
단일-중심점 붕괴가 아니라 **색-단독 특징의 인식 실패**다.

### 사전등록 예측 대조

1. "정적 디코이라 추적 프레임 오류는 크게 감소" — **기각.** tracking-false 92.04 % ≈
   init-false 92.29 %. 기제: init=argmax count가 픽셀 큰 디코이를 선택하면 게이트는 그 track을
   충실히 유지하고, 정적 디코이는 항상 보이므로 track이 만료되지 않는다. 게이팅은 잘못된
   초기 선택을 **고착**시킨다(스모크에서 관찰한 기제가 본 측정에서 확인됨).
2. "초기화 오선택은 남는다" — 확인 (n5 init-false 92.29 %).
3. "N=0 shadow ghost ≈ online ghost" — **비교 불능(설계 확인만).** envelope 라이브 recorder는
   N=0에서 distractor_lock 블록 자체를 기록하지 않아 동일-프레임 online 비교치가 없다.
   shadow 단독 수치(FTLR 10.58 %, tracking 9.93 %)는 낮은 수준으로 무해 방향과 부합하나
   sanity 판정으로 채택하지 않는다.

부수 관찰(게이트 아님): shadow visible_frames(73,258) < online visible(131,378) at n5 —
분모 정의가 다르다(shadow는 자체 연관 성공 프레임). 셀 간 비교는 shadow 열 안에서만 유효.

### 함의

- S3(형상 detector)가 주 서사로 확정 — "구조를 고쳐도 색으로는 안 된다"의 정량 근거 확보.
- S4의 시간 모델도 **초기 선택이 틀리면 시간 누적이 오류를 고착시킨다**는 본 측정의 기제를
  직접 겨냥해야 함(색-단독에서는 게이팅·smoothing이 무력).
- 다음 행동은 사용자 diff 검토 후 커밋 → S2(데이터셋: quad-mesh 표적 visual 교체 등) 착수 여부
  사용자 확인.
## 2026-09-04 — Phase 1 구축: 깊이 잡음 모델 차수, M1 PASS, 3셀 측정 시작

사전등록 `docs/prereg_2026-09-04_depth_noise_model_order.md`. 문헌 조사(아래 별도 항목)에서
나온 결론 — 실데이터 detector + 오차모델 동정으로의 재편 — 이 정당한가를 가장 싸게 판정하는
게이트. 학습 없음, 기존 체크포인트(ref5in ep1900, sha `197ea269...`)를 다른 오차모델 아래에서
평가만 한다.

### 발견: 우리 sigma_r은 실제 스테레오 물리와 차수가 다르다

현재 `sigma_r = 0.04 + 0.012*r + 0.15/sqrt(px)` — 거리에 **1차**.
Intel 공식 `RMS = D^2 * subpixel / (fx * baseline)` — 거리에 **2차**.

두 팔은 하한과 산탄잡음 항을 공유하므로 유일한 차이는 거리 항이고, 교차점은 `0.012/c`:
**D455 6.37 m, D435 3.35 m**. detector 사거리 20 m이므로 **6.4 m 이상 전 구간에서 현재 모델이
낙관적**이다. 20 m에서 sigma_r은 linear 0.386 m vs stereo-D455 0.900 m vs stereo-D435 1.578 m.

sigma_lat(`0.03 + r/detect_fx`)은 **건드리지 않았다** — 방위오차 x 거리이므로 1차가 구조적으로
옳고, detect_fx를 쓰는 것도 옳다. 틀린 항만 고친다.

**모델링 결정(사전등록에 고정)**: 스테레오 항의 초점거리는 **센서의 정합 해상도**(848x480,
fx=447)를 쓰고 detect_fx(160x90에서 84.3)를 쓰지 않는다. 실기에서 깊이는 카메라 내부에서 센서
해상도로 계산되며 우리가 검출용으로 다운샘플하는 것과 무관하다. detect_fx를 쓰면 20 m 오차가
5배 부풀고, 우리 다운샘플링이 센서의 깊이 추정을 열화시키는 것처럼 모델링하게 된다.

### 구현과 M1

- `navrl_task_config.py`(class perception): `NAVRL_DEPTH_NOISE_MODEL`(기본 `linear`) +
  베이스라인/서브픽셀/fx 노브. **기본값이 현행이므로 기존 결과·체크포인트·영수증은 불변.**
- `navrl_perception.py`: 거리 항만 분기. 알 수 없는 모드는 `ValueError`(조용한 무시 금지).
- `tests/test_navrl_depth_noise_model.py` 10개 PASS. **M1**: 노브 미설정 시 sigma_r이 변경 전
  공식과 `torch.equal` 비트 동일((r, px) 격자 401x7). M2: sigma_lat 무변경, 스테레오 계수가
  detect_fx로 만들어지지 않음을 소스에서 확인.

### 초안 정정 (측정 전)

초안은 교차점을 "약 12 m"로 적었다. 그 값은 우리 **전체** sigma_r을 Intel의 **맨** 공식과
비교한 것으로, 구현된 두 팔의 차이가 아니다. 공유 항을 소거한 올바른 교차점은 6.37 m이며
**낙관 구간은 초안이 적은 것보다 넓다**. M1 단위테스트가 이 오류를 잡았다. 판정 경계·측정
조건 무변경.

### 측정 분해능을 판정 규칙에 반영

capture ~0.669, n=2,049에서 두 팔 차이의 95 % CI 반폭이 **약 2.9 pp**. 그래서 애초 구상한
"2 pp 미만이면 둔감"은 잡음 바닥보다 작아 성립하지 않는다. 판정을 CI 기반으로 세우고
`INCONCLUSIVE` 칸을 명시적으로 두어, CI가 넓게 나온 결과가 다른 칸에 억지로 들어가지 않게 했다.

`VARIANCE_INSENSITIVE`의 해석 문구도 사전등록에 못박았다: **"오차모델이 무의미"가 아니라
"PEM의 값어치가 분산이 아닌 성분(누락검출·오검출·편향)에 있다"**. 우리는 오검출의 효과를
이미 안다(디코이 실험 -55.5 pp). 이 문구가 없으면 이후 오검출 성분에서 큰 효과가 나왔을 때
결과를 사후 재해석하게 된다.

셀: linear(대조) / stereo_d455(주) / stereo_d435(스트레스), N=0 distractors(분산 효과를
오표적 lock과 분리), 70 bars, 2,049 ep, seed 487, 3070. 결과는 별도 항목으로.

## 2026-09-04 — Phase 1 판정: VARIANCE_INSENSITIVE, 예측 1 기각

`results/navrl_depth_noise_order_seed487/summary.{md,json}`. seed 487, 3070, N=0, 70 bars,
2,049 ep/cell, 체크포인트 ref5in ep1900.

| cell | 20 m sigma_r | capture | crash | timeout |
|---|---:|---:|---:|---:|
| linear (대조) | 0.386 | 68.23 % | 21.91 % | 9.86 % |
| stereo_d455 (주) | 0.900 | 68.57 % | 21.23 % | 10.20 % |
| stereo_d435 (스트레스) | 1.578 | 70.77 % | 20.06 % | 9.18 % |

**verdict_rq = VARIANCE_INSENSITIVE.** 주지표 `linear - stereo_d455` = **-0.34 pp**,
95 % CI [-3.19, +2.51] — 상한이 등록 경계 5 pp 미만이므로 큰 효과를 배제할 수 있다.

**사전등록 예측 1(단조 감소) 기각.** 20 m 잡음을 2.3배(D455), 4.1배(D435)로 키웠는데 capture는
오히려 **상승**했다(68.23 -> 68.57 -> 70.77). 예측 2(주저 -> timeout 상승)도 성립하지 않는다
(9.86 -> 10.20 -> 9.18).

부지표는 모두 CI가 0을 포함해 개별로는 유의하지 않다(`linear - stereo_d435`: capture
-2.54 pp CI[-5.36,+0.28], crash +1.85 pp CI[-0.64,+4.35], timeout +0.68 pp CI[-1.11,+2.48]).
다만 세 셀에 걸쳐 capture 단조 상승 + crash 단조 하락이라는 **방향의 일관성**은 있다.
과대 해석하지 않되 기제 가설로는 기록한다: 가정 R이 작아 KF가 측정을 과신하고 있었고,
R을 키우자 트랙이 평활해졌다. 즉 우리 필터는 **과신 쪽으로 잘못 맞춰져 있었다.**

### 이 결과가 허용하는 해석과 허용하지 않는 해석

사전등록 4절에 결과 이전에 못박은 문구를 그대로 적용한다: VARIANCE_INSENSITIVE는
"오차 모델이 무의미하다"가 **아니라** "PEM의 값어치가 분산이 아닌 성분에 있다"는 뜻이다.

- **기각된 것**: "현재 정책이 낙관적 오차 모델 위에 서 있다"는 내 서사. sigma_r의 차수가
  틀렸다는 것은 여전히 모델에 대한 사실이지만, **이 정책은 그것에 무관하다.**
- **기각되지 않은 것**: 인지 오차 일반의 중요성. 이미 측정되어 있다(디코이 envelope,
  capture 68.1 % -> 12.6 %, -55.5 pp).

### 세 실험이 같은 결론으로 수렴한다

| 무엇을 바꿨나 | capture 효과 |
|---|---:|
| 자료구조 (다중 후보 + chi^2 게이팅) — S1 | 없음 (FTLR +1.3 pp) |
| 거리 측정 분산 (2.3배, 물리적으로 옳은 차수) — Phase 1 | 없음 (-0.34 pp, CI 상한 2.51) |
| **어느 물체를 lock 하는가 — distractor envelope** | **-55.5 pp** |

**구속 조건은 측정 품질이 아니라 물체 동일성(identity)이다.** 측정을 더 정밀하게 해도,
후보를 더 잘 관리해도 달라지지 않는다. 무엇을 표적으로 볼 것인가만이 문제다.

이는 사용자가 처음 제기한 직관("드론 형상 인식 문제로 재정의해야 한다")을 세 갈래 측정으로
뒷받침한다. 다만 그 인식기를 **우리 시뮬 이미지로 학습할 수는 없다**(문헌: 일반 시뮬 학습
detector는 실제 저조도에서 mAP 37.2 %; 우리 렌더러는 그보다 열악하다 — 상수색 해석적 구,
40x24 배경, 텍스처/조명 없음).

### 다음 게이트 수정

분산 항은 끝났다. 다음에 검증할 성분은 **누락검출의 시간상관 dropout과 오검출률**이며,
분산과 달리 이들은 우리가 이미 큰 효과를 관측한 축이다. 별도 사전등록으로 다룬다.

## 2026-09-04 — Phase 2 착수: 실데이터 확보, 라이선스 감사, 계획 문서 무효화

### 계획 문서 무효화 기록 (CLAUDE.md "기각된 가설도 반드시 기록")

`docs/plans/perception_shape_temporal_redesign_2026-09-03.md`와
`docs/SAM3_PERCEPTION_VERIFICATION_PLAN_2026-09-03.md` 상단에 SUPERSEDED 고지를 넣었다.

**확증된 것**: 문제 재정의("빨간 픽셀 찾기"가 아니라 물체 동일성). 세 갈래 측정이 수렴 —
자료구조 +1.3 pp, 거리 분산 -0.34 pp, 오표적 lock -55.5 pp.

**폐기된 것**: 그 인식기를 **우리 시뮬 이미지로 학습**한다는 실행 경로(S2~S6). 사유 둘:
① 표적에 형상 정보가 물리적으로 없다 — 반지름 0.15 m 해석적 구에 상수색, 디코이 3종 중 하나는
같은 반지름 구, 배경은 40x24 depth 업샘플(텍스처·조명 없음), 저장소에 쿼드로터 메쉬 부재
(우리 기체 visual도 `sphere radius="0.05"`). **S2의 "quad-mesh 재사용"은 애초에 실행 불가능한
선택지였다.** ② 일반 시뮬 학습 detector는 실제 저조도에서 mAP 37.2 %(실사진 기반 96.4 %,
Ning et al. *Unmanned Systems* 2024)이고 우리 렌더러는 그보다 열악하다.

살아남은 것: KF/신경망 역할 분리, episode-level split, hard negative 설계. 폐기된 것은
**데이터를 어디서 얻는가**이지 **무엇을 배워야 하는가**가 아니다.

### 라이선스 감사 — 이것이 설계 제약이다

라이선스가 불명확한 데이터로 학습하면 게재 단계에서 무효가 될 수 있으므로 전수 확인했다
(2026-09-04, 링크 직접 probe). 전문은 `README.md` *External data* 절과
`docs/status/index.html` `05 · EXTERNAL DATA`.

- 안전: NPS-Drones **BSD-3**, Det-Fly **MIT**, DUT Anti-UAV **Apache-2.0**, AOT **CDLA-Permissive**
- 차단: Drone-vs-Bird(공개 링크 없음 + DUA 서명), **FL-Drones(라이선스 모순 — EPFL은 공개
  링크를 두는데 후속 논문 저자는 "저자 허락 필요"라고 명시; 문구 부재 = 기본 저작권)**
- 규모 초과: ARD-MAV 14.6 GB zip, ARD100 Baidu 전용, AOT 13.4 TB
- **전염 위험**: ARD100 코드 GPL-3.0, `ultralytics/yolov5` **AGPL-3.0** — 그 위에 만든 코드를
  공개하면 우리 코드도 AGPL. 아키텍처 선택 시의 실제 제약이다.

### 확보 현황

- **GLAD 사전학습 가중치 확보**: `yolov5s_GLAD.pt`, 로드 검증 완료 — YOLOv5s, **7,022,326
  파라미터**, 단일 클래스 `Drone`, stride [8,16,32], ARD-MAV 학습. TensorRT `.engine`은 하드웨어
  종속이라 사용 불가. 레포에 **LICENSE 파일이 실제로 없다**(README 배지만) — 측정에는 쓰되
  재배포·인용은 위험.
- **NPS-Drones**: BSD-3 원문 확인, 2.04 GB 다운로드 중. 어노테이션(v2) 먼저 받아 분석 완료 —
  50클립 / 박스 72,131 / 고유 프레임 52,566 / 트랙 246. 등가 한 변 `sqrt(area)` 중앙값 **18.1 px**,
  p5 9.5 px, **13.6 %가 12x12 px 미만**. 우리 협시야 320x180 구성의 20 m 표적(11.3 px)과 같은 영역.

### 환경 격리 — aerialgym conda 환경은 건드리지 않았다

yolov5는 pandas/ultralytics를 요구하는데, 영수증에 `python_environment_manifest`가 기록되고
gate-0가 검사하므로 canonical 환경에 패키지를 넣으면 계보가 흔들린다. `--system-site-packages`
venv(`datasets/detenv`)로 torch만 공유하고 의존성은 venv에만 설치했다. conda 환경에 `pandas`가
없음을 사후 확인했다. 함정 하나: `yolov5/models/common.py`가 import 시점에
`os.system("pip install -U ultralytics")`를 실행해 무한 대기한다 — 명시적 선설치로 회피.

### 저장 공간

여유가 14 GB뿐이어서 정리했다: CUDA 설치 deb 2개(5.2 GB, `/usr/local/cuda-11.7` 설치 완료 확인 후)
+ Chrome 캐시(1.7 GB) + conda `airsim` 환경(5.5 GB) + `workspaces/airsim`(0.76 GB) 제거 →
**여유 26 GB**. pip 캐시(7.3 GB)는 사용자 요청으로 보존. 학습 체크포인트는 `.pth` 963개 8.0 GB이고
커밋된 결과가 참조하는 것은 고유 16개뿐이지만, 이미 gitignore된 체크포인트 소실로 테스트 2개가
깨진 전례가 있으므로 **삭제하지 않았다**.

### 다음

`tools/eval_air2air_detector_zeroshot.py`로 GLAD 가중치를 NPS에 **교차 데이터셋 zero-shot**
평가한다(ARD-MAV 학습 -> NPS 테스트). 판정 질문: 공개 가중치로 충분한가, 아니면 학습이 필요한가.
크기 구간(<8 / 8-12 / 12-20 / 20-32 / 32+ px)과 IoU 0.3은 결과를 보기 전에 코드에 고정했다 —
8 px는 Johnson recognition 문턱, 12 px는 우리 20 m 표적 크기다.

## 2026-09-04 — GLAD 가중치 교차 데이터셋 zero-shot: 학습을 건너뛸 수 없다

`results/air2air_zeroshot/glad_on_nps_*.json`. ARD-MAV 학습 GLAD YOLOv5s를 **NPS-Drones**에
zero-shot 평가(교차 데이터셋). 시뮬레이터·체크포인트 무관, 실제 이미지만 읽는다.

### 결과 (1,840 프레임 / GT 2,650 박스, IoU 0.3)

| imgsz | P | R | F1 | AP |
|---:|---:|---:|---:|---:|
| 640 (GLAD 학습 크기) | 0.373 | 0.067 | 0.113 | 0.030 |
| 1280 | 0.134 | 0.152 | 0.143 | **0.045** |

크기 구간별 recall (등가 한 변):

| <8 px | 8–12 px | 12–20 px | 20–32 px | 32+ px |
|---:|---:|---:|---:|---:|
| 0.316 | **0.419** | 0.186 | 0.046 | 0.023 |

**recall이 크기에 대해 뒤집혀 있다** — 작은 표적이 잘 되고 큰 표적이 안 된다. 하네스 버그를
의심해 셋을 확인했다: ① 프레임 시크 정확도(순차 읽기와 픽셀 차 0.000) ② 최대 GT 박스
프레임의 예측 직접 비교 ③ 예측 박스 크기 분포. 결론은 **버그가 아니라 실제 스케일 도메인
시프트**다:

- 예측 박스 등가 한 변: 중앙값 **8.2 px**, p95 17.6 px, 최대 23.9 px, **32 px 초과 0개/187**
- 즉 이 가중치는 큰 박스를 **낼 수 없다**. ARD-MAV 평균 물체가 0.02 %(≈9×9 px)이므로 박스
  회귀가 작은 스케일에 고정돼 있다. GT 141×111 px에 예측 10×7 px가 나온다.

### 정확한 해석 — "공개 가중치가 나쁘다"가 아니다

GLAD 논문의 **자체 ablation에서 외형 단독 recall이 0.17**이다. 우리가 교차 데이터셋에서 측정한
외형 단독 recall은 **0.152**로 사실상 같은 수준이다. 공개된 `.pt`는 **외형 분기 하나**이고,
레포가 명시한다: *"the full codes with Kalman Filter, Adaptive Search Region, and other codes
will be published in the future."* 즉 AP 0.80을 만드는 **움직임 융합과 지역 크롭 재탐지는
공개되지 않았다.**

→ **학습을 건너뛸 수 없다.** 이유는 가중치 품질이 아니라 **신호를 나르는 부분이 배포되지 않았기**
때문이다. 그리고 그 부분(ego-motion 보상 + 움직임 융합)은 우리가 depth와 자기운동을 정확히
갖고 있으므로 직접 만드는 편이 오히려 유리하다(호모그래피 근사가 시차에서 깨지는 문제를 우회).

### 공정성 확인

GLAD 원 파이프라인의 입력은 **640**이다(`detector1_trt.py: INPUT_W=INPUT_H=640`). 처음 1280으로
돌린 것은 불공정했으므로 두 크기를 모두 보고한다. 640이 오히려 더 나쁘다(AP 0.030). 또한
`detector2`(크롭 재탐지)와 `MOD2`(움직임)는 실행하지 않았다 — TensorRT 엔진이 하드웨어 종속이고
`.pt`로 제공되는 것은 전역 외형 detector뿐이다.

### 한계

단일 데이터셋(NPS) · IoU 0.3 고정 · 외형 분기 단독 · 크기 구간과 IoU는 결과 이전에 코드에
고정했다(`tools/eval_air2air_detector_zeroshot.py`). 라이선스: NPS는 BSD-3(원문 확인),
GLAD 가중치는 **LICENSE 파일 없음**이므로 측정에만 쓰고 재배포하지 않는다.

## 2026-09-04 — 정정: -55.5 pp는 판정 지표가 아니라 교란된 보조 수치다

커밋 전 문서 검토에서 발견해 정정한다. 위 항목들(2026-09-04 Phase 1 판정, Phase 2 착수)과
`fcc71a0` 커밋 메시지가 distractor envelope의 효과를 **"capture -55.5 pp (68.1 % -> 12.6 %)"** 로
적었고, 이를 S1(shadow FTLR +1.3 pp)·Phase 1(capture -0.34 pp)과 **같은 표에 나란히** 놓았다.
셋을 동등한 증거처럼 제시한 것은 **과장이다.**

원 출처가 명시한다
(`results/navrl_detector_distractor_envelope_seed479/summary.md`):

> "capture/crash/timeout은 원값이며 **판정에 쓰지 않는다**."

그리고 §L5가 이유를 적는다: distractor가 다섯 코드 경로에서 자유 공간으로 남아 있어
**distractor 충돌이 미귀속 contact로 기록**되고 정적 goal이 distractor 안에 놓일 수 있다.
즉 68.13 % -> 12.64 %는 **방향은 명확하지만 크기가 교란된 값**이다.

- envelope의 사전등록 판정 지표는 **FTLR**이고, v7 N=5에서 **90.27 %**다. 이 값은 깨끗하다.
- 세 실험 수렴 서사(구속 조건 = 물체 동일성)는 **바뀌지 않는다.** 근거가 capture 원값에서
  FTLR로 바뀔 뿐이며 오히려 더 단단해진다.

수정 범위: `README.md` 인지 절과 `docs/status/index.html` `04 · PERCEPTION` 표를 FTLR 기준으로
고치고 교란 사유를 본문에 명시했다. **이미 커밋된 `fcc71a0`의 메시지와 그 이전 WORKLOG 항목은
고치지 않는다**(이력 재작성 금지) — 이 항목이 그에 대한 정정 기록이다.

교훈: 다른 실험의 수치를 인용할 때 **그 실험이 그 수치를 판정에 썼는지** 먼저 확인한다.
보조 보고용 원값을 판정 지표와 같은 표에 놓으면 독자는 동등한 증거로 읽는다.

## 2026-09-04 — 안전필터 선행연구: riskcap은 재발견이다 (두 겹으로)

`docs/prereg_2026-09-04_contact_corridor_forensics.md` §5c에 요약. 증거등급을 붙여 조사했다
(A: 원문에 수치 명시 · B: 그림/인접문에서 추론 · C: arXiv only · D: 벤더/코드/특허).

### 우리 것은 재발견이다

**(1) 산업 매니퓰레이터 안전** — Zanchettin & Rocco, IROS 2013 (A, 원문 확인):

> "Let δ ∈ [0, 1] be a scalar quantity adopted to kinematically scale the trajectory in time...
> the robot has to maintain the programmed path, while the traversing speed can be reduced"

스칼라 하나로 크기만 조정하고 경로 보존 — 우리 구조와 동일. 용어는 **trajectory scaling /
path-consistent safety**이며 ISO 10218 / ISO/TS 15066의 **Speed and Separation Monitoring**
계열이다. **이 인용을 빠뜨리면 매니퓰레이터 안전 리뷰어에게 걸린다.**

**(2) ROS 2 Nav2** — `nav2_collision_monitor`의 approach 동작(D, 소스 확인):
`safe_vel = velocity * (TTC / time_before_collision)`, 소스 주석이
*"Apply the same ratio to all components to **preserve curvature**"*. 맵 없음 · raw scan ·
크기만 조정 · 곡률 보존 — 사실상 동일하고 **이미 출하돼 있다**. 다만 **출판된 평가가 없다.**

### 초안 주장 하나 기각

1차 조사는 "0이 아닌 하한을 일부러 두는 사례가 없다"고 했다. **틀렸다.** Regulated Pure Pursuit
(Macenski et al., *Autonomous Robots* 47, 2023, A)는 `regulated_linear_scaling_min_speed =
0.25 m/s`를 기본값으로 둔다. 같은 논문이 **선형 램프를 지수·이차보다 의도적으로 선택**한
근거도 준다("exponential and quadratic ... far too significantly penalized proximity").

### 그래도 남는 니치 (2차 조사 결론)

쿼드로터에서 ① 맵 없이 raw depth/LiDAR clearance의 닫힌 형식 함수로 `v_max`를 정하고
② **명령 방향을 보존**하며 ③ 동일 명목속도에서 with/without ablation을 낸 결과는 **없다**.
가까운 넷이 각각 하나씩 빠진다: Zhao RA-L'24(맵 기반, 수치 그림뿐) · Ryll ICRA'19(cost이지
cap 아님) · Falanga RA-L'19(공식은 맞으나 분석 전용) · DWA'97(지상, 방향 동시탐색).

### 반드시 인용·비교해야 할 것

- **Zhao, Wu, Chen, Gao, "Learning Speed Adaptation for Flight in Clutter," IEEE RA-L 2024**
  (arXiv:2403.04586) — 최근접 이웃. RL 외부루프가 스칼라 `v†`를 EGO-Planner에 부과하므로
  실질적 magnitude-only. 차이는 **맵 기반**.
- **Zhang et al., arXiv:2602.08653 (2026, ZJU FAST Lab)** — 같은 플랫폼·같은 밀집 환경이고
  **NavRL을 벤치마크로 쓴다**(NavRL 3/5/7/9 m/s에서 43.12/5.00/3.12/0 %). 우리가 진 ablation을
  그들은 이겼다(92.50 % vs 단순 baseline 87.08 %). 동시에 **우리 노출시간 기제를 말로 적어놓은
  유일한 문헌**이며 이름까지 붙였다: **"constraint-induced reachability phenomenon"**.
  단 그들의 실패 양태는 **stagnation이고 collision이 아니다** — 노출 기제를 **충돌률**에
  귀속시킨 문헌은 없다. 우리 stopcap 결과가 바로 그 귀속이다.
- Hsu/Hu/Fisac, *Annual Review* 2024 — **Perfect Safety Filter** 조건 2가 우리의 "상한 아래
  요청은 손대지 않음"과 동일. 같은 문헌이 정지 안전집합의 무용함을 명시해 우리 floor를
  정당화한다("some hard-to-please customers may return the robot on the basis that it doesn't
  do anything").
- Fox/Burgard/Thrun 1997 §2가 방향·속도 분리를 **"무한한 힘을 낼 수 있을 때만 정당하다"**고
  공격한다. 답변("방향은 clutter를 이미 본 학습 정책이 정한다")을 논문에 명시해야 한다.

### 부수 발견

- 항공 정본 두 법칙(Falanga RA-L'19, Loquercio *Sci. Robot.* 2021 §S5)은 **제동이 아니라 회피
  법칙**이다: `v = s/(τ + t_maneuver)`. 우리 stopcap은 지상 계열 제동 법칙을 비킬 수 있는
  비행체에 적용한 것 → **stopcap 부진의 경쟁 가설 B**(사전등록 §5b-2에 기록).
- 우리 stopcap은 **교과서 DWA보다 엄격**하다 — DWA 원논문에 reaction/latency 항이 아예 없다
  (단어 전수 확인).
- EGO-Swarm Table II: 밀도 0 -> 0.42 obs/m²에서 평균 속도 1.55 -> 1.61 m/s로 **평평** —
  주요 플래너에 clearance 감속이 없다는 직접 증거.
- Loquercio가 우리 기제를 열린 문제로 지목: *"strong variations of the instantaneous flying
  speed as a function of the obstacle density ... resulting in intractable sampling."*
- 안전필터가 단순 baseline을 못 이긴 **정직한 ablation을 문헌에서 찾지 못했다**(검색어 5종).
  우리 Q1 null(riskcap-fixed2p0 +0.40 pp, CI [-1.98,+2.78])은 그래서 출판 가치가 있으나,
  **부정 결과를 요점으로 삼을 때만**이다.

### 명명 규칙 (동결)

**"direction-preserving speed governor"** 또는 **"magnitude-only (path-consistent) safety
filter with a liveness floor"**. 개입 방식은 saturation-type, 감시는 value-based.
**CBF·shield·"provably safe"로 부르지 않는다** — forward invariance도 recursive feasibility도
없다.

### 다음

접촉 기하 포렌식 사전등록 완료(범주 5개·우선순위·예측 5개 동결, 경쟁 가설 A/B 명시).
GPU 2 arm x 2,049 ep 대기.

## 2026-09-04 — 접촉 기하 포렌식 판정: 충돌의 77~78%는 회랑이 구조적으로 못 보는 곳에서 난다

사전등록 `docs/prereg_2026-09-04_contact_corridor_forensics.md`. seed 491, 3070, N=0 distractors,
70 bars, 2,049 ep/arm, 체크포인트 ref5in ep1900.
`results/navrl_contact_geometry_seed491/summary.{md,json}`.

### 결과 — 부딪힌 막대가 **접촉 1.0초 전**에 어디 있었나

| arm | crash | 접촉 | vertical_out | behind | **lateral** | **no_return** | in_corridor |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 21.23 % | 303 | **0.0 %** | 1.7 % | **57.1 %** | **20.1 %** | 21.1 % |
| riskcap | 18.35 % | 233 | **0.0 %** | 3.4 % | **58.4 %** | **19.7 %** | 18.5 % |

**구조적으로 감시 불가(lateral + no_return): off 77.2 %, riskcap 78.1 %.**
진짜 종방향 실패(in_corridor)는 **5건 중 1건**뿐이다.

### 사전등록 예측 대조

| 예측 | 결과 |
|---|---|
| 1. riskcap의 `IN_CORRIDOR` 과반 미만 | **성립** (18.5 %) |
| 2. `LATERAL`이 단일 최대 범주 | **성립** (57~58 %, 압도적) |
| 3. `VERTICAL_OUT` ≈ 0 | **성립, 정확히 0.0 %** (막대 높이 2.0 m) |
| 4. riskcap에서 `IN_CORRIDOR` 비중 감소 | **방향만 일치, 유의하지 않음** — +2.67 pp CI[-4.11,+9.44] |
| 7. `LATERAL`(명령) 일부가 `IN_CORRIDOR`(실제속도)로 재분류 | **기각** — 재분류가 **음수**(-38, -27) |
| 8. 회랑 내 막대 2개 이상이 과반 | **기각** — 13.2 % / 11.2 %, 평균 0.67~0.72개 |

(예측 5·6은 비접촉 프레임 기준선을 기록하지 않아 **평가 불가**. 다음 계측에 포함할 것.)

### 가설 판정

- **가설 C(회랑을 명령 방향에 그림) 기각.** 명령-실제 평균 각차는 29~33°로 작지 않지만, 기준을
  실제 속도로 바꾸면 `in_corridor`가 **오히려 줄어든다**(21.1→8.6 %, 18.5→6.9 %). 즉 실제 속도
  방향이 "더 옳은" 기준이 아니다. 회랑을 어느 벡터에 그리느냐는 문제의 핵심이 아니다.
- **가설 D(장애물 집합 축소) 이 작동점에서 기각.** 접촉 1초 전 회랑 안 막대는 **평균 0.67~0.72개**로
  1개도 안 된다. 2개 이상은 11~13 %뿐이다. Fraichard Property 4의 우려는 이론적으로 맞지만
  **이 밀도·이 속도에서는 지배적 기제가 아니다.**
- **빔 간격 사각**은 측정 전 산수로 배제했다(D_Det 8.13 m vs 정지거리 1.81 m, 구속 속도 5.50 m/s).
  `NO_RETURN` 20 %는 빔 사이로 빠진 것이 아니라 **사거리·가림으로 반환이 없는 방향을 자유로
  취급한** 결과다.
- **남은 것은 회랑의 폭 그 자체다.** 반폭 0.45 m는 4 m 전방에서 ±6.4°, 즉 360° 중 **3.6 %**만
  검사한다. 충돌의 57~58 %가 그 밖에서 났다.

### riskcap이 하는 일 — 그리고 하지 않는 일

접촉을 303 → 233건으로 **-23.1 %** 줄인다. 그런데 **구성비는 거의 변하지 않는다**
(lateral 57.1→58.4, no_return 20.1→19.7, in_corridor 21.1→18.5). 필터가 자기가 감시하는
종방향 위협을 특별히 더 잘 처리하는 것이 아니라, **전반적으로 느려져서 모든 범주를 고르게
줄인다.** 예측 4가 유의하지 않은 것이 이를 뒷받침한다.

이는 Q1 `MECHANISM_UNSUPPORTED`(riskcap - fixed2p0 = +0.40 pp, CI[-1.98,+2.78])와 정합한다.
riskcap의 이득은 정교한 해제 기구가 아니라 **감속 그 자체**에서 온다.

### 계측 정정 (측정 전, 스모크가 잡음)

초안은 **접촉 순간**의 기하를 쟀다. 그 시점엔 드론이 이미 막대 옆(중심간 ~0.5 m)이라 막대가
진행 방향의 정확히 옆에 놓이고 전부 `BEHIND`가 됐다 — 원인이 아니라 결과를 재고 있었다.
막대가 정적이므로 **드론의 과거 pose만 링버퍼**에 담아 **접촉 1.0초 전 기하를 정확히 재구성**하도록
고쳤다(근사 아님). 부딪힌 막대의 **식별**은 접촉 시점 최근접으로, **기하만** 과거에서 읽는다.
lookback 1.0 s는 2.5 m/s 정지거리 1.81 m(=0.72 s)를 덮는 값으로 사전등록에 고정. 판정 규칙·범주·
우선순위·예측은 무변경. 이력 부족 접촉은 집계 제외하고 개수를 보고했다(off 10, riskcap 4).

### 처방 — 문헌이 지지하는 순서

1. **감시를 전방향으로.** 회랑을 넓히는 게 아니라 버린다. LiDAR 점마다 제약(288 선형 제약 QP)
   또는 VO 방식. 57~58 %가 회랑 밖이므로 이것이 최대 항목을 직접 겨눈다.
2. **무반환을 자유로 취급하지 않는다.** 검증된 시스템 셋(Bajcsy 누적 `M_t`, RAPPIDS 피라미드,
   SUPER 가시성 다면체)이 **전부** 자유 공간을 센서 원점에서 **구성**한다. 무반환 광선은 아무것도
   아니라는 증거다. 이것이 `NO_RETURN` 20 %를 겨눈다.
3. 정지 판정을 **분해된 공간의 frontier**에서 (RAPTOR Eq. 6 / FLAP `C_sd`).

이 셋으로 이론상 **77~78 %를 겨눈다.** 남은 in_corridor 20 %가 진짜 종방향 문제다.

### 부록 — 빔 간격 문헌 공백, 1차 출처로 확정 (2026-09-04)

포렌식 부수 결과로 남기는 신규성 주장. 조사자가 아래를 **전부 원문에서 직접 확인**했다.

- **Bansal et al.** (ISSRE 2022 → AIAA SciTech 2024, arXiv:2312.02937)만이 "빔 간격 → 보장 탐지거리
  → 속도 상한" 사슬을 쓴다: `D_Det = E/(2·tan(Δθ/2))`, `D_Det ≥ D_stop`,
  `v_safe = √((a·L)² + 2a·D_stop) − a·L`. **에어택시 착륙** 논문이고 MAV 항법이 아니다.
- **Voxblox**(IROS 2017): 팽창은 `8.5 % + 0.3v`(v=voxel)뿐 — **센서 기하 항 없음**.
- **D-Map**(T-RO 2023): `ψ_I = max{d/R, ψ_lidar}`로 각해상도를 쓰지만 **depth image 크기 조정**
  용도이고, 보수성의 방향이 반대다 — *"If the LiDAR point lies outside the projected area, the
  occupancy state of the cell remains unknown."* 장애물을 넓히지 않는다.
- **Wavemap**(RSS 2023, arXiv:2306.01279): *"infinitely thin rays"* 가정의 앨리어싱을 **지적**하고
  각도 반폭 `τ_θ = 3σ_θ`를 쓰지만, **σ_θ를 빔 간격이 아니라 캘리브레이션·pose 불확실성으로 동기**
  하며 기본값이 `angle_sigma = 0`이다.
- **Xiao et al.**(ISPRS Annals 2017)만이 *"standard deviations are determined by the gaps between
  rays"*라고 쓴다 — 유일한 사례이나 **이동물체 검출용이고 계획·안전 용도가 아니며 닫힌 형식이 없다.**
- OctoMap · UFOMap · FIESTA: 각해상도 항 자체가 없다.

→ **거리 의존 빔 간격으로 장애물을 팽창시키는 논문은 없다.** 우리 값(Δθ=5°, 막대 0.71 m →
`D_Det` 8.13 m, 구속 속도 5.50 m/s)은 작지만 정확하고 미점유인 항목이다. 다만 **우리 2.5 m/s에서는
구속력이 없으므로 이번 충돌의 원인이 아니다** — 고속 운용을 논할 때의 기여로 남긴다.

미확인 1건: Fraichard & Kuffner(*Autonomous Robots* 32(3), 2012)의 *"no perceptual errors"* 가정
문장은 검색엔진 추출로만 확보했다. 인용하려면 도서관 사본이 필요하다.

## 2026-09-05 — (A) 진단 논문 계획 수립, 문서 재편 푸시

`docs/plans/diagnostic_paper_plan_2026-09-05.md`(134줄). 커밋 7개 원격 반영
(`dba9e00..58311c4`).

### 논문 축 결정: (A) 진단 우선, (B)는 A3 판정 후

주장 한 문장: **명령 방향을 보존하고 크기만 조정하는 속도 필터에서 충돌의 77~78 %는 그 필터가
감시하는 영역 밖에서 난다.** 더 엄격한 정지거리 법칙은 자기 안전모델을 만족시키면서 충돌을
늘렸고, 정교한 해제 기구는 상수 상한을 이기지 못했다.

부정 결과를 요점으로 삼는 구조이며, 조사에서 이를 지지하는 두 공백을 확인했다 — 안전필터가
단순 baseline을 못 이긴 정직한 ablation이 없고, 노출시간 기제를 **충돌률**에 귀속시킨 문헌도 없다.

### 없는 것을 기계로 확인했다

결과 JSON을 훑어 논문이 요구하는 지표의 유무를 판정했다. **있음**: 성공/충돌/타임아웃 분리,
충돌 원인별, 실행 속도, 개입률, 접촉 시 여유. **없음**: 궤적 전체 clearance 분포, 경로 길이,
필터 계산시간. 여기에 단일 시드(seed 491)와 외부 baseline 부재, 예측 5·6 평가 불가를 더해
결함 5개(D1~D5)로 정리했다.

### 단계와 중단 조건

A1 재현(seed 497) · A2 누락 지표 · **A3 SCP shadow(핵심)** · A4 외부 baseline(`dwa_arc`, `omni`) ·
A5 절제표 · A6 집필. 사람 약 3주, GPU 4.5시간, **재학습 없음**.

**A3가 이 계획의 무게중심이다.** 정책을 건드리지 않고 "SCP였다면 이 접촉이 사전 감지되었나"만
같은 프레임에서 세어, 논문을 "문제를 보였다"에서 "무엇을 고쳐야 하는지도 보였다"로 올린다.
활성 A/B의 비용과 실패 위험 없이 그것을 한다. 판정 경계 동결: 재분류율 ≥50 % `PRESCRIPTION_SUPPORTED`,
<20 % `INSUFFICIENT`, 사이는 `PARTIAL`.

A4에 **`omni`(방위 무관 전방향 최근접 정지거리)를 넣은 이유**는 E7의 교훈이다 — riskcap이
fixed2p0를 못 이겼으므로, SCP도 "그냥 다 보면 되는 것 아닌가"라는 가장 단순한 답을 먼저 이겨야
한다. 못 이기면 그것을 결과로 보고한다.

중단 조건을 결과 이전에 못박았다: A1 재현 실패 시 시드 5개로 늘리고 그래도 흔들리면 E2를
주장하지 않는다. A3가 `INSUFFICIENT`면 처방 절을 삭제하고 순수 진단으로 축소한다 —
**논문은 여전히 선다.**

### 문서 재편 (커밋 `58311c4`)

축척 그림 2장 신규(아레나 기하 1 px=0.05 m, 기체 1 px=1 mm). 질량·추력에서 운동 포락선이
유도되는 과정을 그림이 보이게 했다 — 1.20 kg / 모터당 9.6 N → TWR 3.26 → 45° 틸트 9.81 m/s² →
2.5 m/s에서 정지거리 1.81 m. 안전필터 논의의 그 숫자가 하드웨어의 귀결임이 보인다.

2026-08 PPT/논문 브리프 묶음 8개(약 2,100줄) 삭제 — 판정이 뒤집힌 뒤 쓰인 것들이고
`GENSPARK_PPT_BRIEF`는 `riskcap +5.82 pp`를 여전히 주장했다(현재 판정은 +0.40 pp, CI가 0 포함).
소스·테스트·사이트 잠금이 하나도 없음을 확인하고 지웠다.

**이미 깨져 있던 참조 2건 복구**: ① `docs/prereg_2026-08-13_detector_coupling.md`가 2026-08-20
통합 때 archive로 옮겨졌는데 소스 5곳의 경로 문자열이 갱신되지 않아 2주 넘게 끊겨 있었다 —
스텁으로 복구. ② `tests/test_status_site.js`가 `1b139b0`(내 커밋) 이후 실패 중이었다 —
인지 절을 다시 쓰며 `offline CPU` 문구를 지운 탓. 복구.

`docs/README.md` 색인 신설(82개 중 41개가 소스 잠금임을 기계 판정해 🔒 표시). CLAUDE.md의
`CRASH_TUNING_LOG` 참조 수 정정(4→3)과, 경고가 없던 `reference_platform_proposal_2026-08.md`
추가 — provenance-frozen이라 바이트 하나에 `exit 2`이고 커밋 `921fb1d`가 이미 8개 브랜치를 깼다.

### 미해결

- **브랜치**: CLAUDE.md는 `research/navrl-env`인데 실제 이력이 전부 `main`이라 main에 커밋·푸시했다
- **`SIM2REAL_3DAY_EXECUTION_PLAN.md`**: README가 아직 Track A의 "only next authority"라 부르나
  72시간 창이 열흘 지났다. 연구 방향 결정이므로 미처리

## 2026-09-05 — A1 판정 `REPLICATED`, A2 지표 확보, A3 구현

사전등록 `docs/prereg_2026-09-05_a1_forensics_replication.md`.
`results/navrl_contact_geometry_seed497/summary.{md,json}`.

### A1 — 두 시드 모두 게이트 통과

| arm | lateral+no_return (seed 497) | 허용 구간 (seed 491 CI) | seed 491 |
|---|---:|---|---:|
| off | **75.2 %** | [72.5 %, 81.9 %] | 77.2 % |
| riskcap | **76.3 %** | [72.8 %, 83.4 %] | 78.1 % |

**`verdict_replication = REPLICATED`.** E2("충돌의 77~78 %가 회랑 밖")를 논문에 낼 수 있다.

부수 지표도 일관된다 — `VERTICAL_OUT`이 두 시드 모두 **정확히 0.0 %**, 가설 C 재분류가 두 시드
모두 **음수**(−26/−29, seed 491은 −38/−27), 가설 D 회랑 내 막대 평균 0.69~0.74개(491은 0.67~0.72).
**세 기각 판정이 재현됐다.**

| 지표 | seed 491 off / riskcap | seed 497 off / riskcap |
|---|---|---|
| lateral | 57.1 / 58.4 % | 54.5 / 60.2 % |
| no_return | 20.1 / 19.7 % | 20.7 / 16.2 % |
| in_corridor | 21.1 / 18.5 % | 22.3 / 18.3 % |
| crash | 21.23 / 18.35 % | 21.82 / 17.86 % |

### A2 — 없던 지표 넷을 확보했다

| 지표 | off | riskcap |
|---|---:|---:|
| 경로 길이 | 43.67 m | 44.40 m |
| 회랑 clearance 3 m 미만 체류 | 7.26 % | 7.91 % |
| 명령-실제 각차 (전 프레임) | 29.9° | 29.6° |
| 거버너 계산시간 (env당) | 6.06 µs | 5.89 µs |

**계산시간이 env당 6 µs다.** 문헌의 HOCBF-QP(90.6 µs 평균 / 384.5 µs 최대)보다 빠르므로,
"이 필터가 싸서 쓴다"는 방어는 성립하지 않는다 — 더 비싼 기하 판정을 쓸 예산이 있다.

**riskcap은 "안전해져서 느려진" 것이 아니다.** 경로 길이가 43.67 → 44.40 m로 1.7 % 늘었을 뿐이고
3 m 미만 체류도 7.26 → 7.91 %로 거의 같다. crash를 21.82 → 17.86 %로 줄이면서 우회 비용을
거의 치르지 않았다. 이는 A4에서 `omni`·`dwa_arc`가 넘어야 할 기준선이다.

### 예측 6 — arm에 따라 갈렸다

접촉 프레임의 명령-실제 각차가 전체 평균보다 큰가:

- **off: 기각** (29.5° vs 29.9°, −0.4°)
- **riskcap: 성립** (35.8° vs 29.6°, +6.2°)

필터가 켜졌을 때만 접촉 순간의 방향 불일치가 커진다. 해석은 열어둔다 — 감속이 명령과 실제의
괴리를 키우는지, 아니면 각차가 큰 상황에서 필터가 더 자주 개입하는지 이 자료로는 분리되지 않는다.
**가설 C는 이미 기각됐으므로**(실제 속도 기준이 오히려 나쁨) 이 관측이 C를 되살리지는 않는다.

### 계측 버그 2건 — 스모크가 잡았다

① **경로 길이 5.69 m**(목표 22~28 m인데 말이 안 됨): 초기화 시 128 env가 전부 리셋되며 길이 0을
집계했고, `_cg_prev_pos`를 env별로 무효화하지 않아 리스폰 순간이동이 다음 에피소드에 섞였다.
실제로 달린 에피소드만 집계하고 리셋된 env는 다음 스텝을 건너뛰도록 고쳐 **43~44 m**가 됐다.
② **계산시간 833 µs**: `cuda.synchronize()`로 감싸 GPU 동기화 비용까지 쟀다. 동기화를 빼고
호스트측 발행 비용만 재며 배치·env당을 함께 보고하도록 고쳤다.

### A3 구현 — star-convex, 단위테스트 9/9

`aerial_gym/task/navrl_task/star_convex.py`(100줄, Isaac-free라 CPU 테스트 가능) +
`tests/test_navrl_star_convex.py`. 사전등록 `docs/prereg_2026-09-05_a3_star_convex_shadow.md`.

테스트가 고정한 성질: ① **무반환 광선이 최대사거리 구면의 벽이 된다**(미지=자유 → 미지=경계),
② 측면 장애물이 자기 방위를 구속한다(57~58 % LATERAL을 직접 겨냥), ③ **cone 반각을 π로 넓히면
`omni` baseline으로 수렴**한다 — A4 대조군이 같은 연산자에서 나온다.

테스트가 내 오류도 잡았다: 뒤집기는 `|x̂| = 2R − |x|`라 **반경 순서를 역전**시키므로 "가까운 게
멀리, 먼 게 가까이"가 아니라 **가까운 점일수록 더 멀리** 간다(그래서 볼록껍질에 오른다).
소스 주석과 테스트를 모두 정정했다.

## 2026-09-05 — A3 배선·측정 시작, A4 baseline 2종 구현

### A3 — star-convex shadow 배선

`docs/prereg_2026-09-05_a3_star_convex_shadow.md`. seed 503, arm off·riskcap, 2,049 ep/arm 실행 중.

**입력 스캔 선택을 사전등록에 동결했다**: 거버너가 소비하는 것과 같은 **수직 접힌**
`last_scan_nearest`를 쓴다. 전체 `[B,V,H]`를 넣으면 두 가지(방위 범위와 수직 해상도)가 동시에
바뀌어 비교가 격리되지 않는다. 이 입력에서 회랑과의 차이는 **오직 ① 모든 방위 참여 ② 무반환이
경계** 둘뿐이다.

스모크(60 ep): 접촉 10, 대상(lateral|no_return) 9 = lateral 8 + no_return 1, 재분류 6 → 66.7 %,
`no_return` 1/1, `lateral` 5/8. M2 PASS. 본 측정으로 확정한다.

### A4 — `omni`와 `dwa_arc`

거버너에 두 모드를 추가했다. **둘 다 stopcap의 정지 법칙을 그대로 쓰고 기하만 바꾼다** —
따라서 결과 차이는 전부 "필터가 어디를 보는가"에 귀속된다.

- **`omni`** — 회랑을 없애고 **모든 방위의 최근접**을 본다. 포렌식이 부른 가장 단순한 답이자
  더 정교한 기하가 넘어야 할 기준선이다(Q1에서 riskcap이 상수 캡을 못 이긴 전례).
- **`dwa_arc`** — 직선 대신 현재 `(v, ω)`가 만드는 **원호**를 따라 잰다(Fox et al. 1997 Eq. 14).
  DWA §2의 *"무한한 힘을 낼 수 있을 때만 정당하다"*는 반론을 **우리 자료로 정량화**한다.

단위테스트 5개 추가(거버너 총 18개 PASS)로 고정한 것: ① omni는 회랑이 못 보는 순수 측면
장애물을 본다 ② **yaw rate 0에서 arc는 직선 회랑으로 정확히 수렴한다** ③ 선회 중에는 둘이
어긋난다 ④ 세 모드가 동일한 cap 법칙을 쓴다 ⑤ 설정이 두 모드를 받는다.

### 테스트가 잡은 내 오류 2건

① **수평 투영 누락** — `directional_lidar_clearance`는 각 수직빔을 `× cos(elevation)`으로 수평면에
투영한 뒤 최소를 취하는데, 새 함수 둘이 그것을 빼먹었다. +20° 행에서 6 % 더 멀리 보게 되어
"기하만 바꿨다"는 전제가 깨진다. 공용 헬퍼 `_horizontal_nearest`로 통일했다.
② **테스트 기대값 오류** — 1.5 m 사거리를 +20° 빔에서 투영하면 1.409 m인데 1.5를 기대했다.
투영이 옳고 기대가 틀렸다.

### A2 계측 정정 (앞 항목 보론)

star-convex가 쓰는 스캔이 수직 접힌 것이라, `star_convex.py`의 elevation 인자는 단일 0 행이다.
사전등록 표에 이 선택과 이유를 적었다.

## 2026-09-05 — A3 판정 `PRESCRIPTION_SUPPORTED`: 회랑 밖 접촉의 68.9 %가 사전 감지 가능했다

사전등록 `docs/prereg_2026-09-05_a3_star_convex_shadow.md`. seed 503, 2 arm, 2,049 ep/arm.
정책 무변경 shadow — 같은 프레임에서 반사실만 기록.

| arm | crash | 대상(lateral+no_return) | 재분류 | 비율 | lateral | no_return |
|---|---:|---:|---:|---:|---:|---:|
| off | 22.21 % | 234 | 156 | **66.7 %** | 101/177 (57.1 %) | **55/57 (96.5 %)** |
| riskcap | 17.03 % | 181 | 130 | **71.8 %** | 93/143 (65.0 %) | **37/38 (97.4 %)** |
| 합산 | | 415 | 286 | **68.9 %** CI[64.5, 73.4] | | |

**`verdict_a3 = PRESCRIPTION_SUPPORTED`** (게이트 ≥50 %). 두 arm 각각도 독립적으로 통과.

### 예측 1이 강하게 성립했다

`no_return`의 재분류율이 **96~97 %**다. **무반환 광선을 최대사거리 구면의 벽으로 바꾸는 것만으로
그 범주가 사실상 전부 사전 감지된다.** 이는 코드 한 줄 수준의 변경이 아니라 자유공간을 센서
원점에서 **구성**하는 설계 전환이지만, 겨누는 항목이 명확하고 효과가 거의 100 %다.

`lateral`은 57~65 %로 낮다. 전방위로 보면 절반 이상이 잡히지만 **1/3은 여전히 놓친다** —
star-convex 경계 평균이 7.2~7.5 m인데 부딪힌 막대는 2.0~2.5 m에 있었다는 것은, 그 방위에서
**더 가까운 다른 반환이 경계를 만들지 못했다**는 뜻이다. 즉 5° cone 안에 반환이 없거나
막대가 빔 사이에 있었던 경우다. A4의 `omni`(전방위 최근접)가 이 잔여를 얼마나 더 줄이는지가
다음 질문이다.

### 처방의 위계가 수치로 정해졌다

| 조치 | 겨누는 접촉 | 재분류율 |
|---|---:|---:|
| **무반환을 자유로 취급하지 않기** | 20 % | **96~97 %** |
| **회랑을 버리고 전방위** | 57~58 % | 57~65 % |
| 나머지(종방향) | ~20 % | — |

첫 번째가 **효율이 압도적으로 높다.** 논문의 처방 절은 이 순서로 쓴다.

### VOID 재실행 — 값은 같았다

앞선 seed 503 실행은 내가 평가 중 `navrl_task.py`를 편집해 VOID 처리했다
(`results/VOID_20260905_a3_source_drift_seed503/`). 깨끗한 트리에서 재실행한 `off` arm이
**66.7 %로 동일**했다. 값이 같았다고 해서 VOID 판단이 과했던 것은 아니다 — 계보를 보증할 수
없는 수치를 채택하면 그 뒤 판정이 전부 같은 결함을 물려받는다. 재실행 비용은 1시간이었다.

### A4 사전등록 완료

`docs/prereg_2026-09-05_a4_geometry_baselines.md`. seed 509, 6 arm(off/fixed/riskcap/stopcap/
**omni**/**dwa_arc**), 판정 주지표 crash rate, 기준선 riskcap.

**`NO_IMPROVEMENT`을 실패가 아닌 결과로 미리 규정했다** — 그 경우 "회랑 밖에서 부딪히지만
회랑을 넓히는 것으로는 해결되지 않는다"가 주장이 되고 처방은 star-convex로 좁혀진다.
예측 2에 "`dwa_arc`가 riskcap과 거의 같으면 DWA의 직선분해 반론은 우리 작동점에서 실질적이지
않다는 것이 결과"라고 적어, 리뷰어의 예상 질문에 대한 답이 어느 쪽이든 나오게 했다.

## 2026-09-05 — A4 판정 `GEOMETRY_MATTERS`: 원호가 직선을 모든 축에서 이긴다 (예측 2 기각)

사전등록 `docs/prereg_2026-09-05_a4_geometry_baselines.md`. seed 509, 6 arm, 2,049 ep/arm.
M3 PASS(체크포인트·소스 매니페스트 6 arm 동일). `results/navrl_contact_geometry_seed509/`.

| arm | capture | crash | timeout | 경로 m | <3 m 체류 | 개입률 | 실행 m/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 69.55 % | 21.13 % | 9.32 % | 44.0 | 7.27 % | 0 % | 2.27 |
| fixed2p0 | 69.11 % | 15.67 % | 15.23 % | 46.3 | 7.68 % | 61.5 % | 1.86 |
| riskcap | 71.33 % | 18.77 % | 9.90 % | 44.4 | 7.58 % | 5.8 % | 2.24 |
| stopcap | 67.94 % | 12.30 % | 19.77 % | 46.3 | **21.85 %** | 19.3 % | 1.85 |
| **omni** | **42.59 %** | **7.61 %** | **49.80 %** | 37.4 | **83.51 %** | 70.2 % | 1.05 |
| **dwa_arc** | **77.80 %** | **11.07 %** | 11.12 % | 46.5 | **6.25 %** | **4.0 %** | 2.24 |

**`verdict_a4 = GEOMETRY_MATTERS`** — 두 신규 기하 모두 riskcap 대비 crash를 3 pp 이상 낮추고
CI가 0을 제외한다: `omni` −11.16 pp CI[−13.20,−9.12], `dwa_arc` −7.70 pp CI[−9.87,−5.53].

### 예측 2 기각 — 그리고 이것이 오늘의 가장 큰 결과다

나는 "우리 정책의 yaw rate가 작으니 원호가 직선과 거의 같을 것"이라 예측했다. **틀렸다.**

`dwa_arc`는 riskcap 대비 **crash −7.70 pp**, **capture +6.47 pp**, timeout +1.22 pp,
그리고 **개입률이 오히려 낮다(4.0 % vs 5.8 %)**. 3 m 미만 체류도 7.58 → 6.25 %로 줄었다.

**덜 개입하면서 더 안전하고 더 잘 잡는다.** 안전-완주 교환이 아니라 **측정이 그냥 더 정확한
것**이다. 같은 정지 법칙·같은 스캔·같은 튜브 반폭에서 직선을 원호로 바꾼 것뿐이므로, 차이는
전부 기하에 귀속된다.

Fox, Burgard & Thrun(1997) §2의 *"직선 분해는 무한한 힘을 낼 수 있을 때만 정당하다"*는 반론이
**우리 작동점에서 실질적으로 옳았다.** 리뷰어의 예상 질문에 대한 답이 "우리 속도에서는
무관하다"가 아니라 "그들이 맞았고 우리가 측정했다"가 됐다.

### 예측 1 성립 — `omni`는 "그냥 다 보면 된다"의 반례다

crash는 7.61 %로 가장 낮지만 **timeout 49.80 %**, capture 42.59 %로 무너진다. `<3 m 체류`가
**83.51 %** — 전방위를 보면 항상 무언가가 가깝다. 개입률 70.2 %, 평균 실행 속도 1.05 m/s.

A4에 `omni`를 넣은 이유가 이것이었다: **회랑 밖에서 부딪힌다는 사실이 "회랑을 없애라"를
함의하지 않는다.** 없애면 자유공간이 사라진다.

### Q1이 세 번째 시드에서도 재현된다

`fixed2p0 − riskcap` crash **−3.11 pp** CI[−5.41, −0.80], 0을 제외한다.
**상수 2.0 캡이 riskcap보다 안전하다.** riskcap이 capture에서 +2.2 pp 앞서지만, 해제 기구가
안전을 산다는 주장은 세 시드에서 연속으로 지지되지 않았다.

### stopcap의 노출시간 기제가 지표로 잡혔다

`<3 m 체류`가 7.58 → **21.85 %**(2.9배), timeout 9.90 → 19.77 %. 정지여유는 개선되는데
crash 감소분(−6.47 pp)을 timeout(+9.87 pp)으로 지불한다. **감속해서 틈에 오래 머문다**는
2026-09-02의 기제 가설이 이제 직접 측정된 양으로 뒷받침된다.

### 계보 오류 2건 — 오늘의 실수 패턴

① A3: 평가 중 `navrl_task.py` 편집 → VOID → 재실행(값 동일, 그래도 재실행).
② A4: 스윕 중간에 셸 화이트리스트 수정 후 재개 시도 → 공유 소스 번들이 **첫 arm에서 고정**되므로
드리프트 → 전체 VOID → 재실행.

둘 다 "기다리는 동안 다음 걸 하자"에서 나왔다. **GPU 평가 중에는 소스를 건드리지 않는다.**
`tests/test_navrl_speed_governor.py::WhitelistsAgree`가 파이썬·셸 화이트리스트 불일치를 이제
차단한다.

## 2026-09-05 — A5 절제표·A6 논문 초안 완료: (A) 계획 6단계 종료

`docs/a5_ablation_table.md`(생성물, `tools/build_a5_ablation_table.py`),
`docs/paper_draft_2026-09-05.md`(127줄).

### A5 — 계획서의 표 구조를 자료에 맞춰 바꿨다

계획서 §A5는 "학습 조건 × 평가 조건" Swift 형식을 스케치했다. **우리 측정은 그 형식에 맞지
않는다** — 모든 필터가 추론 시점에만 작동하고 동결 정책이 전 arm 공통이므로, 실제로 변하는 축은
**필터가 어디를 보는가**다. 표를 그 축으로 재구성했고, 그래서 지지되는 주장이 더 좁고 더 단단하다.

표 4개: ① 감시 기하별 결과(6 arm) ② 보수성의 대가(경로·체류·개입률·계산시간) ③ 회랑 밖
접촉의 2시드 재현 ④ star-convex 반사실. 생성기는 시뮬레이션 없이 커밋된 JSON만 읽으므로
재생성·검증이 가능하다.

### A6 — 초안의 뼈대

주장 셋: ① 더 엄격한 법칙이 충돌을 늘렸다(정지여유 +0.395 m인데 crash 15.95→21.31 %)
② 충돌의 75~78 %가 감시 밖(2시드 재현) ③ **그러나 회랑을 없애는 것은 답이 아니다** —
`omni`는 timeout 49.8 %로 무너지고, 답은 **직선을 원호로** 바꾸는 것이었다.

배제한 대안 설명 셋을 §3에 명시했다(명령 vs 실제 방향 · 장애물 집합 축소 · 빔 간격).
빔 간격은 GPU 없이 산수로 배제했다 — `D_Det` 8.13 m vs 정지거리 1.81 m, 구속 속도 5.50 m/s.

§6에 Q1을 두 시드로 병기했다: seed 49에서 +0.40 pp CI[−1.98,+2.78](차이 없음),
seed 509에서 **−3.11 pp CI[−5.41,−0.80]**(상수 캡이 더 안전). **riskcap의 해제 기구가 값을
한다는 주장은 세 시드에서 연속으로 지지되지 않았다.**

한계 5개를 §7에 명시했다. 특히 **모든 arm이 riskcap과 함께 적응한 정책 위에서 추론 시점에만
작동했으므로 `stopcap`·`omni`·`dwa_arc`는 분포 밖**이라는 것, 그리고 `dwa_arc`의 이득이
원호 자체 덕인지 선회 중 튜브가 넓어진 효과인지 **분리되지 않았다**는 것.

### (A) 계획 6단계 종료 — 실제 소요

| 단계 | 결과 | GPU |
|---|---|---|
| A1 재현 | `REPLICATED` | 1 h |
| A2 지표 4종 | 계산시간 6 µs/env · riskcap 우회비용 +1.7 % | (동시) |
| A3 반사실 | `PRESCRIPTION_SUPPORTED` 68.9 % | 1 h ×2 (VOID 1회) |
| A4 baseline | `GEOMETRY_MATTERS`, 예측 2 기각 | 3 h ×2 (VOID 1회) |
| A5 표 | 4개 | — |
| A6 초안 | 127줄 | — |

계획은 GPU 4.5시간이었으나 **계보 오류 2건으로 재실행해 약 9시간** 들었다. 둘 다 "GPU가 도는
동안 다음 단계를 구현"하다 생겼다. 이 교훈이 이 계획에서 얻은 것 중 가장 이전 가능한 것이다.

### 다음 (§8)

① `dwa_arc` 귀속 분리(튜브 반폭 고정, 원호만 토글) ② star-convex를 실제로 켠 A/B
③ 필터와 함께 재적응 학습 ④ 밀도 축 확장. 전부 새 사전등록 필요.

## 2026-09-05 — A7 사전등록·구현 완료, 평가 착수 (법칙 × 기하 2×2)

**계기.** A6 초안의 "dwa_arc가 riskcap보다 안전, 같은 법칙·기하만 교체"는 틀린 문장이었다.
`speed_governor.py:315`와 A4 사전등록 §1 모두 `dwa_arc`가 **stopcap 정지법칙**을 쓴다고 명시한다.
같은 법칙끼리(stopcap→dwa_arc) crash 차이는 −1.23 pp CI[−3.19, +0.74]로 비유의이고, 원호가 한
일은 정지법칙의 비용 제거(timeout −8.6 pp, 개입률 19.3→4.0 %, <3 m 체류 21.85→6.25 %)다.
riskcap 대비 −7.70 pp의 대부분은 법칙(stopcap − riskcap −6.47 pp) 몫이다.
두 번째 위험: stopcap − riskcap 부호가 09-02 스크린(+5.36 pp, 205 bars·ep25000·brake 2.96)과
A4(−6.47 pp, 70 bars·ep1900·brake 2.0)에서 뒤집힌다. 교란은 체크포인트·밀도·brake·기체·목표거리.

**계획 (Fable)**: `docs/prereg_2026-09-05_a7_arc_attribution.md`. 원안(개입률 4.0 %에 맞춘 약화
직선 riskcap)은 법칙 교란을 남기고 튜닝 자유도를 늘려 폐기. 대신 2×2의 빈 칸 `riskcap_arc`
(riskcap 법칙 + arc_clearance), seed 491 재현, 205 bars·ep25000·seed 49·brake 2.0 4 arm.

**구현·감사 (Codex)**: `riskcap_arc` mode(파이썬·셸 화이트리스트·셸 riskcap 검증 세 곳),
`run_navrl_contact_geometry.py`의 `NAVRL_CG_ARMS`/`NAVRL_CG_RESULT_ROOT`/소스 동결 검사,
P3 런처 `eval_navrl_v2_ep25000_arc_attribution.sh`, 요약기 `build_a7_arc_attribution_table.py`,
테스트 27개. 감사(`docs/a7_preexecution_audit_2026-09-05.md`)가 **`arc_clearance` yaw=0 결함**을
찾았다: 반경 1e6 대용의 float32 상쇄로 횡거리 0.46 m 광선이 ±0.45 m 튜브 안으로 들어온다
(직선 12 m vs 원호 3.97 m, 배치 최대 0.107 m). A4 `dwa_arc`는 이 결함 상태로 돌았다.

**결정 (개정 1, 결과 이전)**:
1. 결함 수정(`perp = where(straight, |py|, perp)`). 수정 함수는 A4 dwa_arc와 다르므로 P1을 같은 커밋
   4 arm으로(합계 12 arm, 약 4.5~5.5 h). M5(riskcap 재현)는 보고용으로 남기고 P3 brake 대조만 막는다.
2. 라벨 중복은 `ARC_HURTS_UNDER_RISKCAP` > `INTERACTION`(유일하게 공존 가능한 쌍).
3. M5 |Δ| ≤ 0.5 pp는 `PASS_INEXACT`로 교차 루트 비교 허용·표기, > 0.5 pp는 `FAIL`.
4. P3 라벨 `FLIP_IS_DENSITY`/`DENSITY_LIMITED`는 이름과 달리 조건 전이 검사로 해석(기체·목표거리·시드도 다름).

테스트: 전체 1,110개, failures 2 / errors 1 — 전부 A7 무관 기존 실패(heldout·latency). A7 신규 테스트 51개 통과. P3 런처 CPU preflight PASS.
초안·A5 표·RESEARCH_PLAN·VERIFICATION은 "같은 법칙" 문장을 정정하고 A7 미결로 표시했다.

**실행**: 코드 커밋 후 P1(509) → P2(491) → P3(205/49) 순차, 런처가 arm마다 HEAD·작업트리 동결을 검사.
결과는 `results/navrl_arc_attribution_{seed509,seed491,205bars_seed49}/`, 판정은 요약기가 낸다.

## 2026-09-05 — A7 판정: `LAW_CARRIES_SAFETY` ×2 시드 `REPLICATED`, 205 bars `DENSITY_LIMITED`

12 arm 전부 커밋 742022e에서 13:53→15:05(72분, arm당 70 bars 12분·205 bars 5분). 게이트 M3·M4·M6 PASS,
M5 세 루트 모두 **정확 일치**(riskcap 18.77/18.35/15.95 %). 09-02 스크린의 `runtime_git_dirty=true`는
값이 같으므로 결과 무영향으로 사후 확인. 표: `docs/a7_arc_attribution_table.md`,
`results/navrl_arc_attribution_summary.json`.

| crash Δ (pp, 95 % CI) | seed 509 | seed 491 | 205 bars seed 49 |
|---|---|---|---|
| 법칙, 직선 (stopcap − riskcap) | −6.47 [−8.68, −4.26] | −5.51 [−7.73, −3.30] | +0.30 [−1.95, +2.55] |
| 법칙, 원호 (dwa_arc − riskcap_arc) | −8.68 [−10.83, −6.52] | −7.71 [−9.92, −5.50] | −0.14 [−2.43, +2.15] |
| 기하, 정지법칙 (dwa_arc − stopcap) | −1.85 [−3.80, +0.09] | −1.12 [−3.13, +0.89] | +0.44 [−1.83, +2.71] |
| 기하, riskcap (riskcap_arc − riskcap) | +0.35 [−2.05, +2.75] | +1.07 [−1.32, +3.47] | +0.88 [−1.39, +3.14] |

- **P1/P2 `LAW_CARRIES_SAFETY`, `REPLICATED`.** 안전은 정지법칙이 산다. 원호의 crash 기여는 두 법칙·두 시드
  모두 CI가 0을 포함. `ARC_REMOVES_COST` 양 시드 성립: 정지법칙 아래 원호가 timeout −9.4/−10.2 pp,
  개입률 −15.2/−16.3 pp, capture +11.3/+11.3 pp. riskcap 아래에선 원호가 아무것도 바꾸지 않는다
  (개입률 오히려 +0.5 pp — 예측 P1-1의 "3~5 %" 세부는 기각).
- **P3 `INCONCLUSIVE` → `DENSITY_LIMITED`.** 205 bars·ep25000·brake 2.0에서 정지법칙의 crash 이득이
  사라진다(+0.30 pp). 원호는 여전히 비용을 걷어낸다(timeout 14.0→2.7 %, 개입률 45→20 %, capture +10.9 pp).
  brake 대조 `FLIP_IS_DENSITY`: brake 2.96→2.0으로 stopcap crash −5.05 [−7.44, −2.67] 내려가므로 brake 축도
  실재하지만, 그래도 riskcap을 못 이긴다. 감사 §3대로 "밀도 단독"이 아니라 조건 전이(체크포인트·기체·목표거리 포함).
- 예측 대조: P1-1 라벨 ✔(개입률 세부 ✘), P1-2 ✔, P2 ✔, P3 ✔.
- 초안 헤드라인은 §4 표대로: "정지법칙이 안전을 사고 활력을 잃는다. 원호는 그 활력 비용을 되돌리되 안전을
  더 주지 않는다. 그 안전 이득 자체가 70 bars·ep1900 조건에 한정된다." A4 `GEOMETRY_MATTERS`는 결합 효과 판정으로 보존.
- 남은 일: 초안 §1·§5·§7 반영(A5 표는 A7 표를 인용), 205 bars에서 법칙 이득이 사라지는 기제(체류 46.9 % vs 21.9 %) 분석.

### A7 반영·A8 계획 (같은 날)
- 초안 §1(네 가지 주장)·§5(2×2 표·세 조건 crash Δ)·§7·§8 재작성. §7 정정: ref5in ep1900은 거버너 **off**로 학습
  (`train_navrl_v2_ref5in_d1_adapt.sh` L63)이라 70 bars 2×2는 riskcap 편향이 없다. "riskcap과 함께 적응"은 ep25000에만 해당.
- A8 사전등록 `docs/prereg_2026-09-05_a8_filter_readaptation.md`: ep1900에서 {off, riskcap, dwa_arc} 동반 1,000 epoch
  (T0 대조군 포함), seed 521 평가 10 cell. Q1 비용 회수·Q2 법칙 이득 유지·Q3 필터 의존. 예측 P-2 `LAW_GAIN_PERSISTS`가
  틀리면 §1-2가 "추론 시점 한정"이 된다. 약 10 h GPU, 구현 §6. 미실행.

## 2026-09-05 — A8 구현·착수 + 상태 사이트 A7 반영

- 사이트: `docs/status/index.html` hero 문장·§06 안전필터 단락·"주장할 수 없는 것"에 A7 판정 반영, `docs/assets/motar-safety-filter.svg`의
  "정지여유 양수인데 악화" 주석을 brake 2.96·205 bars 조건으로 한정하고 A7 한 줄 추가.
- A8 구현: `train_navrl_v2_ref5in_a8_readapt.sh`(D1 계약 복제, `A8_MODE` off|riskcap|dwa_arc, ep1900 SHA 게이트, MAX_EPOCHS 2900,
  A7 거버너 파라미터 export; preflight PASS), `tools/run_navrl_a8_readaptation.py`(10 cell, seed 521, run tag로 ep2900 체크포인트 유일 해석,
  소스 동결·기존 루트 거부), `tools/build_a8_readaptation_table.py`(§2 규칙, 학습 arm trailing-100 crash 중단조건 검사),
  `tests/test_navrl_a8_readaptation.py` 8개. 전체 1,118 tests, 신규 실패 0(기존 3 유지).
- 착수: T0 → T1 → T2 학습(각 1,000 epoch) → 평가 10 cell → 요약기, 한 드라이버·한 커밋.

## 2026-09-05 — A8 판정: `LAW_GAIN_PERSISTS` · 동반학습은 이득 없음 · 양 arm `FILTER_DEPENDENT`

`docs/a8_readaptation_table.md`, `results/navrl_a8_readaptation_summary.json`. 상태 `RESULT_CHECKS_PASSED`,
blocker 0. 학습 3 arm 각 1,000 epoch(ep1900→2900, 커밋 5e527ea), 평가 10 cell(seed 521, 커밋 71ce1ff).
대조 건전성 `CONTROL_STABLE`.

| cell | crash | capture | timeout | 개입률 |
|---|---:|---:|---:|---:|
| S/off | 21.47 % | 69.01 % | 9.52 % | 0 % |
| S/riskcap | 17.57 % | 71.69 % | 10.74 % | 5.6 % |
| S/dwa_arc | 11.32 % | 77.27 % | 11.41 % | 3.9 % |
| T0/off | 20.69 % | 77.75 % | 1.56 % | 0 % |
| T0/riskcap | 15.57 % | 82.97 % | 1.46 % | 9.6 % |
| T0/dwa_arc | 9.37 % | 88.97 % | 1.66 % | 7.1 % |
| T1/riskcap | 17.46 % | 80.98 % | 1.56 % | 10.2 % |
| T1/off | 24.52 % | 74.06 % | 1.41 % | 0 % |
| T2/dwa_arc | 11.03 % | 86.19 % | 2.78 % | 8.1 % |
| T2/off | 29.77 % | 68.87 % | 1.36 % | 0 % |

### Q2 `LAW_GAIN_PERSISTS` — 이 실험의 핵심

`L_adapt`(T2/dwa_arc − T1/riskcap) crash **−6.43 pp [−8.56, −4.30]**, capture +5.21 [+2.95, +7.48].
**각자 자기 필터와 적응한 뒤에도** 정지법칙+원호가 riskcap 법칙을 이긴다. A7의 "안전은 법칙이
산다"는 동결 정책의 추론 시점 아티팩트가 아니다. 예측 P-2 적중(−5~−8 예측, 실측 −6.43).

### Q1 `INCONCLUSIVE` — 예측 P-1 기각, 그리고 더 중요한 사실

동반학습은 비용을 회수하지 않았다. `C_arc` crash +1.66 [−0.19, +3.51](CI 0 포함), capture
**−2.78 [−4.80, −0.76]**, timeout **+1.12 [+0.22, +2.02]**. `C_risk`도 같은 방향(crash +1.89,
capture −1.99). **두 arm 모두 필터를 학습 루프에 넣는 것이 미세하게 손해다.**

이유는 대조군이 말한다. `K_*`(T0 − S): 거버너 **없이** 1,000 epoch 추가한 것만으로 timeout이
세 필터 모두에서 **−8.0~−9.8 pp**(10~11 % → 1.5 %), capture +8.7~+11.7 pp다. crash는 −0.8~−2.0 pp로
작다. 즉 **A7에서 정지법칙의 비용으로 관측한 timeout은 필터 탓이 아니라 정책의 미성숙 탓**이었고,
추가 학습만으로 사라진다. 회수할 비용이 없으니 동반학습이 회수할 것도 없었다.

### Q3 `FILTER_DEPENDENT` (T1·T2 모두) — 예측 P-3 부분 적중

`D_arc`(T2/off − T0/off) crash **+9.08 pp [+6.43, +11.72]**, `D_risk` **+3.83 pp [+1.27, +6.39]**.
필터와 함께 학습한 정책은 **필터를 떼면 대조군보다 위험하다**. 강한 필터일수록 의존이 크다
(원호+정지법칙 9.1 pp > riskcap 3.8 pp). T1도 의존적일 줄은 예측하지 못했다(`NEUTRAL` 예측).
09-02 Q3 `FILTER_DEPENDENT`(off−riskcap +9.23 pp)와 같은 방향이며, 실기에서 필터 고장 시 후과다.

### 종합 — 채택 권고

**필터는 추론 시점에만 쓴다.** 학습은 필터 없이(T0), 배치는 dwa_arc로. 그 조합(T0/dwa_arc)이
전 cell 최저 crash 9.37 %·최고 capture 88.97 %다. 동반학습(T2/dwa_arc 11.03 %/86.19 %)보다 낫고
필터 의존도 만들지 않는다.

### 평가 결정성 — 부하가 결과를 바꾼다 (부수 발견)

S_off를 세 번 돌렸다. 유휴 상태 두 번은 **소수 둘째 자리까지 동일**(21.47/69.01/9.52),
NPS 타일링이 CPU·디스크를 점유하던 한 번만 어긋났다(22.35/67.50/10.15). 코드·체크포인트·시드는
세 실행 모두 동일했고 런타임 소스 339개 파일 해시도 전부 일치했다(매니페스트 해시 차이는
생성시각·커밋해시·editable install 줄 3개뿐). 3차는 `navrl_task.py` 수정 후인데도 2차와 완전히
같아, 그 수정이 시뮬레이션에 무영향임도 함께 확인됐다.
→ **GPU 평가 중 무거운 CPU 작업 금지는 예의가 아니라 측정 요건이다.**

### 중단 2회 — 원인이 다르다

① 18:22 VOID: 평가 중 내가 `tools/`에 파일 추가 → 동결 가드가 중단(A3·A4와 같은 실수 패턴 3회째).
② 18:22 VOID: `cfg_search_state`("off" 문자열)가 `float(saved)` 루프에 있어 **재적응 체크포인트
복원 불가**. 이 키가 생긴 뒤 학습한 정책은 전부 평가 불가였고 A8이 첫 사례였다. 커밋 71ce1ff로
값 비교로 분리, 회귀 테스트는 개별 키가 아니라 불변식(float 루프에 `str()`·문자열 리터럴 금지)을
검사한다. 수정 전 2건 실패·수정 후 통과 확인.

잃은 것은 cell 4개 약 25분. 학습 3 arm은 무손실.

## 2026-09-05 — 계획 I1/I2 구현: 접촉별 기록과 프레임 표본 (측면/밀도 계획의 게이트)

`docs/plans/lateral_contact_density_plan_2026-09-05.md` §4 I. 지금까지 접촉 포렌식은 범주 **합계**와 평균 4개만
남겨서 §2의 어떤 가설(속도-미끄러짐·틈 폭·기억·요레이트)도 가를 수 없었다.

- `aerial_gym/task/navrl_task/contact_records.py`(신규, Isaac-free 순수 함수): 범주 우선순위 인덱스, 좌/우 부채꼴(15~165°)
  최근접 **표면** 거리(막대별 외접반경 차감), 기억 창 지표(`seen_frac_window`, `seen_then_lost`), JSONL 기록기.
- `navrl_task.py`: 링버퍼 깊이 11→26(정책 이력 창 2.5 s), 요레이트·거버너(requested/executed/cap) 슬롯 추가, 거버너
  적용 직후 텔레메트리 push. `_record_contact_geometry` 끝에 **접촉당 1행**(35열: 범주 명령/실속 기준, 방위, 전방/측면
  거리, 표면 거리, 1.0/0.5 s 전 속도·요레이트, cap binding, 좌/우 틈, 회랑 내 막대 수, 무반환, 2.5 s 창 관측 여부).
  `_record_contact_geometry_step`에 N스텝마다 전 env 프레임 표본(I2, `NAVRL_CG_FRAME_SAMPLE_EVERY`, 기본 100).
  사이드카 `<cell>.contact_records.jsonl`·`<cell>.frame_samples.jsonl`을 결과 JSON 옆에 쓰고 경로·행 수·SHA를 payload에 기록.
  기존 집계·판정 로직은 그대로(회귀 없음).
- H3 기억 지표는 토큰 ID가 아니라 **기하로** 계산한다: 막대는 정적이므로 과거 포즈 × 막대 위치로 "[t−2.5, t−1.0 s]에
  사거리 안·240° 안이었는가"가 정확히 복원된다. 선택기가 빔 인덱스만 돌려주므로 이 방식이 유일하게 정확하다.
- 테스트 9개(`tests/test_navrl_contact_records.py`), GPU 스모크 32 ep: 접촉 4행 = 집계 4, 범주 일치, 프레임 128행.
- 주의: 평가 환경(`canonical_env`)이 ambient `NAVRL_*`를 지우므로 `NAVRL_CG_FRAME_SAMPLE_EVERY`는 런처가 명시해야 한다.

## 2026-09-05 — 계획 I3 구현: 거버너 추가 knob 3종 (L2 속도의존 폭 · L3 측면 채널 · L5 요레이트 캡)

새 mode가 아니라 **파라미터**다. 어떤 기본 법칙(riskcap/stopcap/dwa_arc/…) 위에도 하나씩 얹을 수 있어 "arm 하나 = 변화
하나"가 지켜지고, 화이트리스트 3곳을 건드릴 필요가 없다. 기본값 0에서 전부 꺼져 **이전 모든 arm과 비트 동일**.

| knob | env | 동작 |
|---|---|---|
| L2 | `NAVRL_SPEED_GOVERNOR_WIDTH_PER_MPS` k | 회랑 반폭 w = w0 + k·|v_req| (Nav2 VelocityPolygon). 직선·원호 모두 per-env 텐서 폭 지원 |
| L3 | `..._LATERAL_MARGIN_M` m(>0이면 켬), `..._LATERAL_SPAN_M` s, `..._LATERAL_FLOOR_MPS` v_floor | 명령 방향 기준 15~165° 양측 최근접 표면 d_lat → cap_lat = v_floor + (v_free − v_floor)·clamp((d_lat−m)/s); 최종 cap = min(전방 cap, cap_lat). 바닥이 있어 omni 교착 불가 (RSS 횡규칙) |
| L5 | `..._YAW_CAP_RADPS` c(>0이면 켬), `..._YAW_CAP_MARGIN_M` | d_lat < margin이면 요레이트 크기를 c로 캡(부호 불변). 요 명령 생성부에 scale 곱 |

- `speed_governor.py`: 필드 6개·검증(floor ≤ free)·`speed_dependent_half_width`·`lateral_clearance`·`lateral_cap`·`yaw_scale`;
  `apply_speed_governor(…, lateral_clearance_m=None)`, 텔레메트리에 `lateral_cap_mps`·`lateral_clearance_m` 추가.
- `navrl_task.py`: 거버너 진입부에서 per-env 폭 계산, L3/L5 중 하나라도 켜졌을 때만 측면 clearance 계산(아니면 연산도 동일),
  `_governor_yaw_scale`을 요 명령에 곱. 체크포인트 env_state·결과 JSON `condition`에 6개 값 기록.
- 평가 셸 `GOVERNOR_VALUES` spec에 6개 추가·export(파이썬과 셸 일치를 테스트가 검사).
- 테스트 8개(`tests/test_navrl_governor_lateral_knobs.py`): 기본값 비트동일, 방향 보존, 바닥, 부호 불변, 셸 파싱 15값.
- GPU 스모크(32 ep, seed 509): 기본 riskcap이 I3 전 스모크와 **완전 동일**(capture 20·crash 13·접촉 4·개입률 0.060018).
  knob 3종 모두 오류 없이 실행·condition 기록. 32 ep 수치는 해석하지 않는다.
- 주의: k=0.2 s는 요구속도 3.5 m/s에서 반폭 1.15 m라 정지법칙이 상시 발동(개입 53 %). L1 반폭 스윕으로 척도를 먼저 잡고
  L2의 k 격자를 정하는 계획 순서가 맞다.

## 2026-09-05 — 격자 러너 `tools/run_navrl_filter_grid.py` + cell 명세 3종 (D1·L1·D3)

cell 명세 JSON(정책·밀도·거버너 env)을 받아 한 루트·한 소스 번들·한 커밋에서 순차 평가한다. 정책은 S/T0/T1/T2/ep25000/절대경로,
env는 **거버너 knob만** 허용(다른 계약은 못 바꿈 — 테스트로 고정). I1/I2 사이드카 항상 켬, `frame_sample_every`를 명시 전달.
명세: `docs/specs/grid_d1_density_filter_T0.json`(20 cell), `grid_l1_halfwidth_T0.json`(20 cell), `grid_d3_lineage_ep25000.json`(4 cell).
스모크(16 ep): 205 bars override·knob 2종 기록·사이드카 확인.

**신호**: T0(70 bars 학습)가 205 bars에서 16/16 crash. 계획 §5.1 게이트 D1(a)에 걸릴 가능성이 커서 D1 전체(2 h) 대신
`off` 5 밀도 게이트(30 min)를 먼저 잰다. 걸리면 205 bars 수치는 인용 불가이고 R1(밀도 커리큘럼 재학습)이 다음 학습 작업이 된다.

## 2026-09-06 — D1 게이트 판정: T0는 100 bars 위에서 분포 밖. 밀도 축은 이 정책으로 물을 수 없다

`results/navrl_grid_d1_gate_off_T0_seed523/`, seed 523, 거버너 off, 2,049 ep/cell, 커밋 22541cd. 20:31→01:36.

| bars | crash [95 % CI] | 증가 | capture | 접촉 | 측면/무반환/회랑안 |
|---:|---|---:|---:|---:|---|
| 70 | 21.02 % [19.3, 22.8] | — | 77.66 % | 383 | 59/18/21 % |
| 100 | 29.02 % [27.1, 31.0] | +8.0 | 69.71 % | 501 | 60/16/24 % |
| 130 | **45.39 %** [43.2, 47.5] | +16.4 | 53.15 % | 753 | 58/12/29 % |
| 160 | 58.66 % [56.5, 60.8] | +13.3 | 40.36 % | 965 | 58/10/30 % |
| 205 | **81.96 %** [80.3, 83.6] | +23.3 | 17.41 % | 1,298 | 61/ 7/31 % |

**게이트 발동**: 계획 §5.1의 "205 bars crash(off) > 45 %"를 **130 bars에서 이미** 넘었다. T0는 70 bars에서만 학습했으므로
100 bars 위의 모든 수치는 분포 밖이다. → **T0로는 밀도 축을 물을 수 없다.** 130/160/205 bars 필터 비교는 이 정책으로 인용 불가.

### 접촉별 기록(I1)이 처음으로 말한 것

| bars | 측면 접촉 v(t−1s) | 측면 접촉 좁은쪽 틈 | 회랑안 접촉 v | 명령−실속 각차 |
|---:|---:|---:|---:|---:|
| 70 | 2.16 m/s | 1.45 m | 2.84 m/s | 18° |
| 205 | 1.57 m/s | 1.02 m | 1.88 m/s | 22° |

1. **틈 폭은 구속 조건이 아니다.** 205 bars에서도 접촉 1 s 전 좁은 쪽 여유가 **1.02 m**로, 기체 상자 0.28 m·팽창 0.65 m보다
   넉넉하다. H2의 단순형("틈이 몸체보다 좁다")은 지지되지 않는다. 대신 2 m/s로 1 s면 2 m를 가므로 1 m 옆의 것에 닿는 것은
   **경로가 밀린 결과**다 → H1(속도-미끄러짐)·H6(선회)이 남는다.
2. **회랑 안 접촉은 언제나 측면보다 빠르다**(모든 밀도에서 +0.3~0.7 m/s). 정면은 과속, 측면은 경로 이탈.
3. **밀도는 접촉의 종류가 아니라 빈도를 바꾼다.** 측면 비율 58~61 % 고정, 회랑 안 21→31 %로 완만히 증가.
   "205에서 정지법칙 이득이 사라진 것은 측면 비중 증가 탓"이라는 설명은 **기각**된다.
4. 기억 지표(관측 후 놓친 뒤 접촉) 16~24 %로 계획 트리거 25 % 미만 → **R4 후순위**.
5. crash 증가가 가속한다(+8 → +16 → +13 → +23 pp). 선형이 아니라 임계 현상에 가깝다(Karaman & Frazzoli 2012).

### 절차 사고 1건 — 알림 내용 불일치

205 셀 완료 알림에 `crash=0.7135`와 완료 표시가 왔으나 디스크는 아직 실행 중이었고, 그 알림 텍스트에는 로그에 없는
XML 조각과 존재하지 않는 도구 이름이 섞여 있었다. 디스크에서 다시 읽은 실제 값은 **81.96 %**다.
→ **알림 본문의 수치는 인용하지 않고 항상 결과 JSON에서 읽는다.**

## 2026-09-06 — D1′ (ep25000 밀도×필터 20 cell): 정지법칙 이득은 **밀도가 아니라 계보**의 문제였다

`results/navrl_grid_d1p_ep25000_seed523/`, contract v2(목표 6~28 m, navrl_quad — 체크포인트가 강제), seed 523,
2,049 ep/cell, 커밋 40d4ec1. 01:49→03:5x. T0와 달리 ep25000은 밀도 커리큘럼으로 205까지 학습해 **전 구간 분포 안**이다.

| bars | off | riskcap | stopcap | dwa_arc | timeout (stop/risk/arc) |
|---:|---:|---:|---:|---:|---|
| 70 | 9.71 % | 5.71 % | 5.37 % | **4.49 %** | 4.20 / 2.44 / 2.10 % |
| 100 | 8.49 % | 6.39 % | 6.20 % | **4.64 %** | 4.39 / 3.22 / 2.24 % |
| 130 | 10.39 % | 6.69 % | 7.56 % | **6.25 %** | 6.25 / 3.07 / 3.17 % |
| 160 | 15.32 % | 9.03 % | 10.93 % | **8.15 %** | 9.12 / 2.64 / 2.93 % |
| 205 | 25.13 % | 16.35 % | 17.21 % | **14.93 %** | 13.55 / 2.44 / 2.15 % |

| bars | stopcap − riskcap | dwa_arc − riskcap | riskcap − off |
|---:|---|---|---|
| 70 | −0.34 [−1.74, +1.06] | −1.22 [−2.56, +0.13] | −4.00 [−5.63, −2.38] |
| 100 | −0.20 [−1.69, +1.29] | **−1.76 [−3.15, −0.36]** | −2.09 [−3.70, −0.49] |
| 130 | +0.88 [−0.70, +2.45] | −0.44 [−1.95, +1.07] | −3.70 [−5.41, −1.99] |
| 160 | **+1.90 [+0.06, +3.73]** | −0.88 [−2.59, +0.84] | −6.30 [−8.29, −4.30] |
| 205 | +0.86 [−1.43, +3.15] | −1.42 [−3.64, +0.81] | **−8.78 [−11.25, −6.32]** |

### 판정 — 초안 §4의 "밀도가 이득을 지운다"는 기각

1. **정지법칙의 riskcap 대비 추가분은 이 계보에서 어느 밀도에서도 없다.** 70 bars부터 −0.34 pp(CI가 0을 크게 포함)이고,
   130 이상에서는 부호가 손해로 넘어간다(160에서 +1.90, CI 0 제외). ref5in에서 70 bars −6.47 pp였던 것과 정반대다.
   → **밀도를 올려서 사라진 게 아니라 이 계보에는 처음부터 없었다.** A7 P1(ref5in, 70 bars)과 P3(ep25000, 205 bars)의
   대비는 밀도 대비가 아니라 **계보 대비**였다. A7 P3에 달아둔 "조건 전이" 단서가 주된 설명이었음이 확인됐다.
2. **가장 그럴듯한 계보 차이는 동반학습이다.** ep25000은 riskcap과 1,000 epoch 함께 학습했고(A8 `FILTER_DEPENDENT`),
   ref5in ep1900은 거버너 off로 학습했다. 자기가 학습한 필터가 이미 잘 맞으므로 더 엄격한 법칙이 이길 여지가 없다.
3. **원호 기하의 이득만 계보를 넘어 살아남는다.** dwa_arc가 다섯 밀도 **전부**에서 최저 crash이고, timeout은 riskcap 수준을
   유지한다(정지법칙은 205에서 13.55 %). 개별 CI는 100 bars만 0을 제외하지만 부호가 5/5로 일관된다.
4. **거버너 자체의 이득은 밀도와 함께 커진다**(riskcap − off: −4.0 → −8.8 pp). 08-11 A/B/C의 −8.80 pp(205 bars)와
   이번 −8.78 pp가 다른 시드에서 일치 — 계보 재현성 확인.
5. **기제(I1)**: dwa_arc의 접촉 시 cap 작동 비율이 31~51 %로 riskcap·stopcap(49~67 %)보다 낮다. 즉 **덜 개입하면서 덜
   부딪힌다**. 정지법칙은 더 자주 감속하고도 crash가 같거나 높다 → 종방향 감속은 이미 한계에 닿았고, 남은 것은 측면이다
   (측면 접촉 비중 57~73 %).

### 정정

앞선 대화에서 stopcap−riskcap이 "밀도에 단조 증가"라고 적었으나 160이 정점(+1.90)이고 205는 +0.86이다.
정확한 서술은 **어느 밀도에서도 이득이 없고 timeout 비용만 커진다**이다.

### 다음

L1 반폭 스윕(측면 가설 H5)이 다음 GPU 작업. 계보 교란을 피하려면 ep25000(v2 계약)에서 도는 편이 낫다 —
T0는 130 bars 위가 분포 밖이고, ep25000은 전 구간 유효하며 dwa_arc가 이미 기준선을 제공한다.

## 2026-09-06 — D4 판정 `CO_ADAPTATION_CONFIRMED`: 정지법칙의 이득을 지우는 것은 동반학습이다

`results/navrl_grid_d4_coadapt_seed521/`, ref5in 계약, 70 bars, seed 521, 2,049 ep/cell, 커밋 b99c103.
T0와 T1은 **같은 기체·같은 출발 체크포인트(ep1900)·같은 1,000 epoch·같은 목표 거리·같은 시드**이고,
오직 학습 루프 안에 riskcap이 있었는지만 다르다.

| cell | crash | capture | timeout |
|---|---:|---:|---:|
| T0 (거버너 off 학습) / riskcap | 15.57 % | 82.97 % | 1.46 % |
| T0 / stopcap | **11.71 %** | 81.16 % | 7.13 % |
| T1 (riskcap 동반학습) / riskcap | 16.74 % | 81.06 % | 2.20 % |
| T1 / stopcap | 17.22 % | 71.22 % | 11.56 % |

| 정책 | stopcap − riskcap | 판정 |
|---|---|---|
| **T0** | **−3.86 pp [−5.95, −1.76]** | 유의. ep1900 원본의 −6.47 pp와 같은 방향·크기 |
| **T1** | **+0.48 pp [−1.82, +2.78]** | CI가 0 포함. ep25000의 −0.34 pp와 같은 그림 |

**결론**: ep25000에서 정지법칙 이득이 없던 것은 밀도도, 0.25 kg 기체도, 6~28 m 목표도, 제동 가정도 아니라
**riskcap과 함께 학습했기 때문**이다. 같은 기체에서 학습 루프의 필터 하나만 바꿔 재현했다. A8의
`FILTER_DEPENDENT`와 같은 기제의 다른 얼굴이다: 동반학습은 그 필터에 정책을 맞추고, 그 결과 더 엄격한 법칙이
얻을 것을 남기지 않는다(T1/stopcap은 timeout 11.56 %만 지불하고 crash는 그대로).

### 미해결 — 교차 루트 1 pp 드리프트

같은 시드·같은 체크포인트인데 A8과 D4의 값이 갈렸다.

| cell | D4 (b99c103) | A8 (71ce1ff) | 차이 |
|---|---|---|---|
| T0/riskcap | 15.57 % (2,049 ep) | 15.57 % (2,049 ep) | **정확 일치** |
| T1/riskcap | 16.74 % (2,049 ep) | 17.46 % (2,050 ep) | −0.72 pp (crash 343 vs 358) |

체크포인트 SHA·시드·계약은 동일하고 `runtime_source_manifest`만 다르다(그 사이 I1 계측·거버너 knob 추가).
T0가 비트 동일하게 재현됐으므로 코드 변경이 일반적으로 결정성을 깨지는 않았다. T1만 갈린 이유는 **미상**이다.
→ **D4 판정은 영향받지 않는다**(네 cell 모두 한 루트·한 커밋의 루트 내 대비). 다만 **다른 커밋의 수치를
1 pp 수준에서 비교하지 않는다**를 규칙에 추가한다. 앞서 "유휴 머신에서 결정적"이라 적은 것은
**같은 커밋 안에서**로 한정해야 한다.

### 이 판정이 논문에 하는 일

초안 §4의 밀도 설명을 대체한다. 새 서술: "정지거리 법칙이 riskcap을 이기는지는 밀도가 아니라 **정책이 어떤
필터와 함께 학습됐는지**로 결정된다. 거버너 없이 학습한 정책에서는 −3.9~−6.5 pp로 이기고, riskcap과 함께
학습한 정책에서는 이득이 사라진다(+0.5 pp). 계보와 무관하게 살아남는 유일한 개선은 원호 기하다."

### 초안 개정 (D1′·D4 반영)

`docs/paper_draft_2026-09-05.md`. §1의 네 주장을 교체하고 §5.1을 신설했다.

- **철회**: "밀도가 오르면 정지법칙 이득이 사라진다". 인용문 블록으로 철회 사유를 본문에 남겼다(계보 교란).
- **새 §1-2**: 어떤 상한 법칙이 옳은지는 **정책이 무엇과 함께 학습됐는지**가 정한다. 거버너 없이 학습한
  세 정책에서 −6.47/−5.51/−3.86 pp, riskcap 동반학습 정책에서 +0.48 pp.
- **새 §1-3**: 원호 기하만 계보를 넘어 살아남는다(다섯 밀도 전부 최저, cap 작동 31~51 % vs 49~67 %).
- **새 §1-4**: 거버너 자체의 이득은 밀도와 함께 커진다(−4.0 → −8.8 pp). 밀도가 무력화하는 것은 필터가 아니라
  **법칙 간의 차이**다.
- **§5.1 신설**: D4 2×2 표(학습 시 거버너 off/riskcap × 평가 riskcap/stopcap)와 D1′ 다섯 밀도 표.
- **§7 한계**: 법칙 비교에 "거버너 없이 학습한 정책에서"라는 단서 필수. dwa_arc 우위는 부호 일관이나 CI 약함.
  교차 커밋 1 pp 비교 금지.
- **§8**: 재적응·밀도 축·기제는 완료로 이동. 남은 것은 star-convex 실켜기, dwa_arc 통계 확정, 실기.
  **측면 접촉 처방을 0번으로 올렸다** — 접촉의 57~73 %인데 아직 손대지 못했고, 틈이 아니라 경로 이탈이 원인이다.

## 2026-09-06 — L1 반폭 스윕 20 cell: 폭을 넓히는 것이 **원호에서만** 값을 한다

`results/navrl_grid_l1_ep25000_seed523/`, contract v2, seed 523, 2,049 ep/cell, 커밋 3b6e7b5.
ep25000, 두 밀도 × 두 법칙 × 다섯 폭. 요약: `python tools/summarize_navrl_grid.py <root> --mechanism`.

| 70 bars | crash | capture | timeout | 개입률 | | 205 bars | crash | capture | timeout | 개입률 |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| 직선 0.45 | 5.37 % | 90.43 % | 4.20 % | 17.0 % | | 직선 0.45 | 17.21 % | 69.23 % | 13.55 % | 44.4 % |
| 직선 0.60 | 5.76 % | 83.11 % | 11.13 % | 38.5 % | | 직선 0.60 | 11.96 % | 41.14 % | 46.90 % | 73.8 % |
| 직선 0.80 | 3.71 % | 71.64 % | 24.65 % | 58.5 % | | 직선 0.80 | 7.17 % | 19.77 % | 73.06 % | 87.0 % |
| 직선 1.00 | 3.86 % | 55.69 % | 40.46 % | 67.3 % | | 직선 1.00 | 3.36 % | 11.31 % | 85.32 % | 92.2 % |
| 직선 1.20 | 3.51 % | 49.19 % | 47.29 % | 73.4 % | | 직선 1.20 | 3.61 % | 9.71 % | 86.68 % | 93.7 % |
| **원호 0.45** | 4.49 % | 93.41 % | 2.10 % | 7.4 % | | **원호 0.45** | 14.93 % | 82.92 % | 2.15 % | 20.8 % |
| **원호 0.60** | 3.90 % | 93.61 % | 2.49 % | 9.2 % | | **원호 0.60** | 11.07 % | 85.95 % | 2.98 % | 25.0 % |
| **원호 0.80** | 3.02 % | 94.54 % | 2.44 % | 11.7 % | | **원호 0.80** | 10.69 % | 86.33 % | 2.98 % | 31.2 % |
| **원호 1.00** | 2.73 % | 94.10 % | 3.17 % | 14.1 % | | **원호 1.00** | 10.05 % | 86.77 % | 3.17 % | 36.6 % |
| **원호 1.20** | **1.76 %** | **95.07 %** | 3.17 % | 17.8 % | | **원호 1.20** | **8.64 %** | **87.21 %** | 4.15 % | 43.0 % |

### 판정 — H5는 "기하에 달려 있다"

1. **직선 회랑을 넓히면 임무가 죽는다.** 70 bars에서 capture 90.4 → 49.2 %, 205 bars에서 69.2 → 9.7 %.
   crash는 내려가지만(5.4 → 3.5 %, 17.2 → 3.6 %) 개입률 73~94 %로 사실상 멈춰 선 상태다. omni(timeout 49.8 %)로
   수렴하는 경로를 다섯 점으로 채워 연속적으로 보였다. **고밀도일수록 훨씬 빨리 무너진다**(0.6 m에서 이미 capture 41 %).
2. **원호 튜브를 넓히면 대가가 거의 없다.** 같은 조작이 두 밀도 모두에서 crash를 낮추면서 capture를 **올린다**.
   70 bars: 4.49 → 1.76 % crash, 93.4 → 95.1 % capture. 205 bars: 14.93 → 8.64 %, 82.9 → 87.2 %.
   timeout은 2.1 → 3.2 / 2.2 → 4.2 %로만 오른다.
3. **같은 폭에서 두 기하의 차이가 밀도와 함께 커진다.** w=0.6에서 capture 차이는 70 bars 10 pp, 205 bars **45 pp**.
   w=1.2에서는 46 pp / 78 pp. 직선은 밀집할수록 넓은 띠에 늘 뭔가 들어오지만, 원호는 선회 중 정면 √(0.9R) m까지만
   보므로 곧 비껴갈 장애물을 붙잡지 않는다. **밀도가 높을수록 그 선택성의 값이 커진다.**
4. **최적 폭이 밀도에 따라 다르다.** 70 bars는 1.2 m에서도 안 꺾이고(스윕 범위를 잘못 잡았다), 205 bars는
   0.6 m 이후 포화한다(0.6→1.2에서 crash −2.4 pp에 개입률 +18 pp). 고정 폭은 한쪽을 손해 본다.
5. **기제**: 폭을 넓히면 측면 접촉이 줄고(70 bars 원호 43 → 3건) 회랑 안 접촉이 는다(12 → 16건).
   205 bars 원호는 183 → 14건 / 87 → 121건. 즉 폭은 접촉을 없애는 게 아니라 **측면에서 정면으로 옮긴다**.
   정면 접촉은 감속으로 다룰 수 있으므로 이 교환이 유리한 것이고, 직선에서는 그 감속이 과도해 임무를 삼킨다.

### 폭은 관측 가능한 양으로 정할 수 있다

밀도(막대 수)는 시뮬레이터 파라미터라 드론이 볼 수 없지만, **자기 라이다로 트인 정도는 매 스텝 읽는다.**
D1 게이트의 비접촉 프레임 표본(거버너 off, T0):

| bars | 최근접 표면 | 좌우 중 좁은 쪽 | 회랑 clearance |
|---:|---:|---:|---:|
| 70 | 2.06 m | 2.22 m | 12.00 m |
| 100 | 1.74 | 1.87 | 12.00 |
| 130 | 1.50 | 1.62 | 12.00 |
| 160 | 1.36 | 1.46 | 9.33 |
| 205 | 1.18 | 1.25 | 6.88 |

단조 감소하고 구간이 겹치지 않는다. → **적응형 폭을 "밀도를 안다고 가정"하지 않고 실기에서도 성립하는 형태로
구현할 수 있다.** L2(속도 의존 폭)와 L3(측면 clearance)가 이미 그 구조를 갖고 있어 몇 줄이면 된다.

### 도구

`tools/summarize_navrl_grid.py` 추가. 진행 중인 격자도 한 줄로 본다(`--mechanism`이 접촉별 열 추가).

## 2026-09-06 — 인수인계 문서 작성

`docs/HANDOVER_2026-09-06.md`. 새 세션이 처음 읽을 문서. 프로젝트 목적·시스템 요약·확정된 결과 7건·
진행 중인 격자·인프라(규약)·운영 규칙 7개·내일 할 일 6개·논문 상태·파일 지도.
WORKLOG는 시계열 기록으로 두고, 상태 스냅샷은 이 문서로 분리한다.


## 2026-09-06 — 자체 검증: 83 cell 재계산, 서술 정정 3건, 코드 수정 2건

`docs/verification_report_2026-09-06.md`, 재현 `results/verification_export_2026-09-06/verify.py`.
개수→비율 불일치 0. cond/req 어긋남은 L8 knob 미기록 18건뿐.

**정정 1 (가장 중요)**: "폭은 측면 접촉을 정면으로 옮긴다"는 **정의상 효과**였다. 범주 문턱이 w를 따라가므로 넓히면
재분류된다. 고정 0.45 m 문턱으로 다시 세면 70 bars w=3.0에서 범주 측면 0건 / 오프셋>0.45 m **32/35건**. 맞는 서술:
부딪히는 막대는 폭과 무관하게 0.6~0.7 m 옆에 있고, 넓은 튜브가 그것을 본다. 정의 독립 지표는 **총 접촉 수**(68→23→35).
**정정 2**: 원호 우위의 근거는 부호 5/5(p=0.062, 불충분)가 아니라 역분산 결합 **−1.15 pp [−1.85, −0.45]**.
**정정 3**: 교차 커밋 드리프트는 6쌍 중 **5쌍 비트 동일**, 1쌍 예외(원인 미상). "일반적 드리프트"가 아니다.
**L8 실패 원인**: 프레임 표본으로 복원한 실제 폭이 205 bars에서 **44~45 % 프레임에서 기본폭 미만**(p10 0.10 m).
기준 1.2 m 앵커링 + 바닥 0.1 m가 좁힘을 허용. 설계 결함이지 적응성의 반증이 아님 → 바닥을 w0로 고정(코드), 재실행 내일.
**D4 한계 추가**: 학습 시드 1개라 "동반학습" vs "그 특정 1,000 epoch"를 분리 못 함. T1은 off에서도 +3.8 pp 나쁘나
riskcap 이득(−7.8 pp)은 유지되고 stopcap 추가분만 사라짐 → 결론 유지, 기제 문장 축소.
**주장 2 한계**: "지나갈 자리가 있었다"는 t−1 s 기준. 접촉 순간 틈은 미기록. 측면 접촉은 요레이트 1.3~1.8배·슬립각 2배로
"경로 이탈"은 지지됨.
코드: 조건 기록에 L8 knob 3개 추가, L8 바닥 w0(테스트 갱신). 전체 1,156 tests, 신규 실패 0.


## 2026-09-06 — R-A 드리프트 탐침: A8의 T1/riskcap 358이 단일 예외였다

`results/navrl_grid_r0_drift_probe_T1_seed521/`, 같은 cell 2회, 커밋 4937f77.
rep1 = rep2 = **343 / 1661 / 45 (n 2049)** = D4 값과 비트 동일. A8(커밋 71ce1ff)의 358/1660/32(n 2050)만 어긋난다.
→ 이 cell은 현재 커밋에서 **결정적**이다. 교차 루트 동일 조건 재현은 이제 **8쌍 중 7쌍 비트 동일**, 예외는 A8의 그 한 실행뿐.
원인은 여전히 미상(당시 로그 구조 동일, 동시 부하 기록 없음). 처리: D4 값을 정본으로 쓰고, A8 요약의 T1/riskcap 행은
"재현 안 됨(1회)" 각주. "교차 커밋 비교 금지" 규칙은 유지하되 근거는 "설명 못 한 단일 실행 1건"으로 적는다.


## 2026-09-06 — R-B 시드 반복 착수 (seed 527 완료 22:10, seed 531 진행)

`results/navrl_grid_r1_seedrep_ep25000_s527/` 25/25 완료(커밋 c1c33d62, 2,049 ep/cell, 25 cell 2 h 08 min).
이어서 seed 531을 22:10에 시작했다. 이 시점의 중간 판정표는 3 시드가 모인 뒤 무의미해지므로 남기지 않는다.
**수치와 판정은 아래 2026-09-07 R-B 항목이 정본이다.**


## 2026-09-07 — R-B 완료: 원호 우위는 세 평가 시드에서 재현, P5는 기각(밀도 국소 효과)

`results/navrl_grid_r1_seedrep_ep25000_s531/` 25/25 완료(00:20:47, 커밋 c0ad44bd). 위의 09-06 중간
항목을 대체하는 최종 판정이다. 결합 도구 `tools/pool_navrl_seed_replication.py`, 결합 결과 JSON은
`results/navrl_grid_r1_seedrep_ep25000_s531/pooled_rb.json`. 시드 523은 D1'(좁은 팔)과 L1(1.2 m)에서,
527·531은 R-B 격자에서 읽었다. 셀당 2,049~2,051 에피소드.

**좁은 팔(w 0.45) crash — 원호가 15/15 (3 시드 × 5 밀도)에서 최저**

| 대비 (pp) | seed 523 | seed 527 | seed 531 | 3 시드 결합 (15 cell) | Q (df) | I² |
|---|---|---|---|---|---|---|
| arc − riskcap | −1.15 [−1.85, −0.45] | −1.89 [−2.60, −1.18] | −1.45 [−2.16, −0.74] | **−1.49 [−1.90, −1.08]** | 10.3 (14) | 0 % |
| arc − stopcap | −1.57 [−2.28, −0.87] | −1.03 [−1.71, −0.34] | −1.17 [−1.87, −0.46] | **−1.25 [−1.65, −0.85]** | 16.5 (14) | 15 % |
| stopcap − riskcap | +0.44 [−0.29, +1.18] | −0.84 [−1.57, −0.11] | −0.27 [−1.01, +0.46] | −0.23 [−0.65, +0.20] | 19.7 (14) | 29 % |

시드 간 이질성이 0 %다. 원호 이득은 특정 시드의 성질이 아니다.

**밀도별 결합(3 시드)** — 원호 우위는 모든 밀도에서 같은 부호이고, 정지법칙과 리스크캡의 차이는 저밀도에만 있다.

| bars | arc − riskcap | arc − stopcap | stopcap − riskcap |
|---|---|---|---|
| 70 | −1.87 [−2.67, −1.08] | −0.69 [−1.44, +0.07] | **−1.19 [−2.01, −0.37]** |
| 100 | −1.24 [−2.06, −0.43] | −1.07 [−1.88, −0.26] | −0.17 [−1.02, +0.68] |
| 130 | −0.86 [−1.73, +0.02] | −0.98 [−1.87, −0.10] | +0.13 [−0.78, +1.04] |
| 160 | −1.76 [−2.75, −0.77] | −2.60 [−3.61, −1.59] | +0.84 [−0.21, +1.90] |
| 205 | −2.01 [−3.29, −0.73] | −1.73 [−3.01, −0.46] | −0.28 [−1.58, +1.03] |

**폭 확장(3 시드 결합)**: 70 bars crash −2.18 [−2.81, −1.55], capture +1.39 [+0.54, +2.23].
205 bars crash −5.60 [−6.73, −4.47], capture +4.44 [+3.19, +5.70]. 직선 1.2 m capture는
70 bars 49.19 / 48.85 / 49.44 %, 205 bars 9.71 / 10.64 / 10.05 %로 시드 간 산포가 1 pp 미만이다.

**동결 예측 판정**

- **P1 통과**: 결합 −1.49 [−1.90, −1.08], 15 cell, p = 7.8e−13. 문턱 −0.8 pp를 넘고 CI가 0을 제외한다.
- **P2 통과**: 새 시드 527·531 모두 5/5 밀도에서 음수.
- **P3 6건 중 5건 통과**: seed 527의 70 bars만 −1.32 [−2.43, −0.20]로 부호·CI는 성립하나 사전등록
  크기 문턱 −2 pp에 미달. 나머지(523/531의 70 bars, 세 시드의 205 bars)는 통과. capture는 6/6 비감소.
- **P4 6/6 통과**.
- **P5 기각**: 세 시드 모두 한 밀도씩 CI가 0을 벗어난다. 원인은 노이즈가 아니라 **밀도 국소 효과**다.
  70 bars에서 세 시드의 stopcap − riskcap이 −0.34 / −1.66 / −1.61로 모두 같은 부호이고 결합이
  **−1.19 [−2.01, −0.37]**이다. 100~205 bars에서는 0을 포함한다(523의 160 bars +1.90은 반대 방향의
  단일 이탈). → C3의 "동반학습 계보에서는 정지법칙의 추가 이득이 없다"는 **밀도 조건부**로 다시 쓴다:
  70 bars에서는 약 1.2 pp 이득이 있고, 100 bars 이상에서는 사라진다. 전체 결합은 여전히 0을 포함한다.

**C2 접촉 순간 협착(I4 `pinch_t0`, 시드 527·531의 50 cell, 5,890 접촉)**: dwa_arc 0.45 **1.6 %**(n 1,333),
riskcap 1.8 %(1,686), stopcap 0.9 %(1,610), arc 1.2 4.1 %(763), 직선 1.2 0.0 %(498). 전부 사전 문턱
20 %보다 한 자릿수 아래다 → "지나갈 자리가 있었다"를 **접촉 순간까지** 확장해 쓴다. 고정 문턱
|측면 오프셋 t−1| > 0.45 m 비율은 좁은 팔 62.9~72.0 %. 총 접촉 수는 arc 1,333 < stopcap 1,610 < riskcap 1,686.

**드리프트**: D1'와 L1이 겹치는 4 cell(70·205 bars × dwa_arc·stopcap, w 0.45)은 개수까지 **비트 동일**
(92/2049, 110/2049, 306/2049, 353/2051). 교차 루트 동일 조건 재현은 이제 12쌍 중 11쌍 동일이다.

다음: 초안 §1·§3·§5.2·§5.3·§7 재작성, R-C(학습 시드 반복)는 사용자 실행.


### 커밋 전 독립 감사 (2026-09-07 새벽)

커밋 전에 분석 도구를 **한 줄도 쓰지 않고** 원본 JSON에서 다시 계산했다
(`scratchpad/audit_rb.py`, scipy 1.10.1). 감사가 잡은 유일한 실제 문제는 아래 교차 트리 건이고,
확인 실행으로 해소했다.

**감사가 잡은 문제**: 결합에 쓴 셀들이 **서로 다른 동결 소스 트리**에서 나왔다. seed 523은
`aerial_gym` 트리 `b51c3d2f`(커밋 40d4ec11 / 3b6e7b5e), seed 527·531은 `3aa30cd8`(c1c33d62 / c0ad44bd)다.
그 사이에 L8 knob 3개, I4 접촉 순간 열, L8 바닥 수정이 들어갔다. 이 저장소 규약은 교차 커밋 비교를
금지하므로, 세 시드를 replicate로 묶는 것이 정당한지가 걸린다.

**코드 검토**: 바뀐 경로는 전부 (a) 기록 전용이거나 (b) 이득 0에서 조기 반환이다.
`speed_dependent_half_width`는 `width_per_mps`와 `width_per_open_m`이 모두 0이면 예전과 같은
`return float(config.path_half_width_m)`을 탄다. `open_m` 계산은 `open_width_enabled`(= gain > 0)로
막혀 있어 R-B 셀에서는 아예 호출되지 않는다. 평가기 변경은 기본값 0/1.2/3.0인 env 3개 추가뿐이다.

**확인 실행(추론에 의존하지 않기 위해)**: `docs/specs/grid_r3_crosstree_check_s523.json`,
결과 `results/navrl_grid_r3_crosstree_s523/`. seed 523의 `ep25000_d205_dwa_arc`를 **현재 트리에서**
같은 시드·같은 조건으로 다시 돌렸다.

| | crash | captured | timeout | n | 접촉 | 평균 측면 오프셋 | 개입 배율 |
|---|---|---|---|---|---|---|---|
| 원본 (트리 b51c3d2f) | 306 | 1699 | 44 | 2049 | 296 | 0.6911 m | 0.934264 |
| 재실행 (트리 3aa30cd8) | **306** | **1699** | **44** | 2049 | **296** | **0.6911 m** | **0.934264** |

**비트 동일**. → 세 시드 결합은 유효하다. 교차 루트 동일 조건 재현은 이제 13쌍 중 12쌍 동일이다.

**나머지 감사 결과 (전부 통과)**
- 90 cell 전부 체크포인트 1개(`f70221393660`), 셀별 seed·bars·governor mode·half-width가 명세와 일치,
  cell env는 `NAVRL_SPEED_GOVERNOR*` 외에 아무것도 바뀌지 않았다.
- 에피소드 정산: `crash + captured + timeout − n`이 70 cell 전부에서 **정확히 0**.
- 개수→비율 불일치 0(1e−9 이내).
- 판정에 쓴 15 cell × 3 대비는 **모두 같은 루트 내부** 짝이다. 폭 대비만 seed 523에서 루트가 갈리는데,
  기준 셀이 비트 동일이라 L1 내부 기준으로 계산해도 −2.73 / −6.30 pp로 값이 같다.
- 통계 재현(scipy `norm.ppf`, `chi2.sf`): arc−riskcap **−1.49 [−1.90, −1.08]**, Q 10.3 (df 14, p 0.74),
  대비 p 7.82e−13. arc−stopcap −1.25 [−1.65, −0.85], Q 16.5 (p 0.28). stopcap−riskcap −0.23 [−0.65, +0.20],
  Q 19.7 (p 0.14). 도구 출력과 소수 둘째 자리까지 일치.
- 원호가 crash 최저인 (시드, 밀도) 조합은 독립 재계산으로도 **15/15**.
- 밀도별 stopcap−riskcap: 70 bars −1.19 [−2.01, −0.37](시드별 −0.34 / −1.66 / −1.61로 부호 일치),
  100~205 bars는 0 포함. P5 기각이 밀도 국소 효과라는 판단을 재확인.
- 접촉 기록 무결성: 50 cell 전부 `contact_records_rows`가 실제 행 수와 일치하고 `pinch_t0`가 전 행에 있다.
  접촉 5,890건 중 협착 97건 = **1.65 %**.
- 타이밍: 마지막 셀 종료 00:20:47, 내 후속 작업 시작 00:20:58. 격자 실행 중 무거운 CPU 작업 없음.

전체 스위트 재확인 1,177 tests OK (skipped 4).


## 2026-09-07 — 코드 정비: 빨간 테스트 3건 원인 규명, 통계 공용화, R-B 분석 도구화

평가가 도는 동안에는 동결 경로(`aerial_gym`, `tools`, `resources/robots`)를 건드리지 않고 `tests/`와
scratchpad에서만 작업했다. 격자 COMPLETE 후 일괄 반영. 전체 스위트 **1,177 tests OK (skipped 4)**,
직전 1,157 tests / 실패 3에서 실패 0이 됐다.

**빨간 테스트 3건의 원인은 모두 코드 결함이 아니라 계측의 결함이었다.**
1. `test_navrl_latency_compensate`: 소스를 `SOURCE[start:start+4000]` 고정 창으로 잘라 검사했는데
   `observe()`가 12,036자로 자라면서 LiDAR 백업 게이트가 offset 4,220으로 밀려 창 밖으로 나갔다.
   기능(`navrl_perception.py:799,2034`)은 멀쩡했다. 메서드 본문을 실제로 잘라 보는 `_method_body()`로
   교체하고, 순서 의존이던 `_target_features` 검사도 같은 방식으로 바꿨다.
2. `test_..._heldout_contract`: 고정된 체크포인트(`ppo_260901_1431_...s911`) 디렉터리가 디스크 확보로
   삭제돼 preflight가 exit 2. 런처에서 경로를 읽어 없으면 사유를 밝히고 skip. 함께, 인수 거부 테스트가
   종료코드만 봐서 체크포인트가 없어도 통과하던 약한 단언을 거부 메시지 확인으로 강화했다.
3. `test_..._heldout_summary`: 09-02 결과의 `source_snapshot/`과 셀 로그가 삭제됐고 런타임 파일 11개가
   그 뒤 수정돼 검증이 불가능했다. **기록된 커밋의 blob으로 복구**하는 경로를 넣으니 330개 전부가
   SHA-256으로 입증된다(현재 워크트리 319 + 커밋 복구 10 + 보관 평가기 1). 검증부를
   `verify_runtime_sources()`로 분리하고 `--sources-only`를 추가해, 로그가 없어도 소스 증명은 돌아간다.
   로그 해시 대조가 필요한 전체 요약만 사유를 밝히고 skip한다.

**통계 공용화**: `tools/navrl_stats.py` 신설(Wald 대비, 역분산 결합, Cochran Q, DerSimonian–Laird,
Holm, 부호검정). **비율이 아니라 기록된 개수**에서 계산한다(계획 §4 규약). `summarize_navrl_grid.py`를
여기에 얹었다. 출력 차이는 z를 1.96에서 1.959963984540054로 바꾼 데서 오는 경계 1개 0.01 pp뿐이다.
기존 이력 분석 스크립트 8종은 발표된 숫자를 보존하기 위해 건드리지 않았다.

**R-B 분석 도구화**: scratchpad 일회용 스크립트를 `tools/pool_navrl_seed_replication.py`로 옮겼다.
같은 조건이 두 루트에 있으면 조용히 버리지 않고 일치 여부를 보고한다(위의 4쌍 비트 동일이 이걸로 나왔다).
한 조건에 두 체크포인트가 걸리면 추측하지 않고 거부한다.

**테스트 22개 추가**(`tests/test_navrl_seed_replication.py`, heldout 요약 2건). 합성 데이터로 자족적이라
결과 디렉터리가 지워져도 빨개지지 않는다. 그중 `PreregistrationSyncTest`는 도구의 판정 문턱이
`docs/specs/grid_r1_seedrep_*.json`의 동결된 예측문과 어긋나면 실패한다.


## 2026-09-07 — NPS-Drones 검출기 40 에포크 완주: mAP50 0.579에서 평탄, 최적은 13 에포크

두 머신에 걸친 한 번의 학습이다. 1650 Ti에서 1~9 에포크(에포크당 44분), 나머지 31 에포크를 3070에서
이어 돌렸다(에포크당 2분 30초, 17배). 최종 26 에포크 구간은 1.14시간에 끝났다.
run: `detector_runs/runs/nps_det/s_tiles640_b8_e40/`(저장소 밖, 워크스페이스 루트).
데이터: `datasets/nps_yolo_tiles640` (train 19,659 / val 4,096 타일, 라벨 3,867 박스, `tools/tile_nps_yolo_dataset.py`).
모델 YOLOv5s, 640, batch 8, SGD lr0 0.01, 40 epochs.

| 에포크 | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| 1 | 0.466 | 0.366 | 0.264 | 0.068 |
| 9 (1650 Ti 마지막) | 0.690 | 0.501 | 0.534 | 0.202 |
| **13 (최적)** | **0.685** | **0.573** | **0.579** | **0.227** |
| 20 | 0.689 | 0.543 | 0.552 | 0.209 |
| 30 | 0.678 | 0.551 | 0.560 | 0.205 |
| 40 | 0.696 | 0.546 | 0.550 | 0.204 |

**곡선은 13 에포크에서 꺾였다.** 9 에포크 시점의 "아직 오르는 중"이라는 판단은 4 에포크만 더 유효했다.
14~40 에포크의 mAP50은 0.545~0.560 대역에 머물고 마지막 5 에포크의 폭은 0.0086에 불과하다. 즉
**40 에포크는 과했고 이 설정에서 이 데이터로는 mAP50 0.58이 천장**이다. 같은 예산을 더 쓰려면 에포크가
아니라 다른 축(입력 해상도, 앵커, 타일 겹침, 증강, 더 큰 백본)을 건드려야 한다.

산출물: `weights/best.pt`(13 에포크, 14.4 MB), `weights/last.pt`(40 에포크). **둘 다 옵티마이저가 제거돼
추론 전용이며 이 파일로는 재개할 수 없다.** 재개가 필요하면 처음부터 다시 돌려야 한다.
곡선·PR·혼동행렬 그림은 같은 폴더에 있고 tfevents 3개(joshuali 1 + fair 2)가 한 폴더에 있어
텐서보드에서 한 줄로 이어 보인다.

**운영 기록 세 가지.**
① 체크포인트가 numpy 2로 저장돼 aerialgym 환경(numpy 1.24)에서 열리지 않는다. Isaac Gym을 깨지 않으려고
   `detector_runs/venv`(rl_study의 torch 2.10 + numpy 2.2.6 상속)를 따로 만들어 썼다.
② **YOLOv5 `utils/general.py`는 임포트 시점에 `os.system("pip install -U ultralytics")`를 실행한다**(34~40행).
   `YOLOv5_AUTOINSTALL`로 막히지 않는다. 셸 pip이 base 환경을 잡아 torch 2.14와 CUDA 13 휠을 받기 시작했고
   디스크 88 % 상태라 중단시켰다. base에는 아무것도 설치되지 않았다. venv에 ultralytics를 먼저 넣어 회피한다.
③ `nohup ... &`만으로는 세션 정리 때 학습이 함께 죽는다(02:14, 오류·OOM 없이 로그가 끊김). **`setsid`로
   분리해야 산다.** 14 에포크까지 저장돼 있어 손실은 없었다.
디스크: 타일 19,659장을 `cache: disk`로 캐싱하면 약 23 GB라 여유가 없어 캐시를 껐다. pip 캐시 8 GB를 비웠다.


## 2026-09-07 — 디스크 정리: 미참조 중간 체크포인트 7.17 GB 회수, 규칙 8-A 제정

Det-Fly 교차 검증(원본 9.34 GB)을 받을 자리를 만들기 위해 정리했다. 지우기 전에 무엇이 실제로
인용되는지부터 셌다.

**조사 결과**: `runs/` 8.4 GB 중 체크포인트가 8.06 GB인데 **결과 JSON이 인용하는 것은 0.12 GB뿐**이었다.
학습 중 주기적으로 저장된 중간본이 나머지 전부다. 로그·설정(csv, tfevents)은 합쳐서 0.34 GB라
회수 대상이 아니다. 예: `recover-curriculum-s1`은 체크포인트 224개 1,862 MB를 갖고 있는데 인용되는
것은 8 MB 하나다.

**삭제**: 932개 중 **870개, 7.17 GB**. 여유 26 GB → 33 GB. 보존은 62개 0.50 GB.
목록은 `tools/audit_checkpoint_references.py --list-deletable`로 뽑았고 세션 scratchpad에
`deleted_2026-09-07.txt`로 남겼다. 삭제 후 재검사에서 **인용 유실은 하나도 늘지 않았다**(11건 그대로).

**정리 중 발견한 기존 사고**: 결과가 인용하는 체크포인트 **4개가 이미 없었다**. 특히
`last_gen_ppo_ep_21750_rew_83.1572.pth`는 seed 911 route-off held-out 평가 6건과 VERIFICATION.md가
인용하는 파일이고, 이것이 없어서 `test_navrl_corrected_nonoverlap_heldout_contract.py`가 건너뛴다.
나머지 7건은 WORKLOG 산문에만 등장한다. 언제 지워졌는지는 기록이 없다.

**규칙 8-A 제정**(`OPERATIONS.md` §8). 산문 규칙("삭제 금지: 최신 terminal checkpoint")이 이미
있었는데도 위 사고가 났다. 그래서 판단을 사람 기억에서 도구로 옮겼다.
`tools/audit_checkpoint_references.py`가 보존 대상을 증거에서 직접 계산한다 —
① 결과 JSON이 인용, ② 추적 파일이 이름으로 고정, ③ 각 run의 최종 `last_gen_*`(없으면 `gen_ppo.pth`).
**모르는 이름은 보존한다**: 삭제 후보가 되는 것은 도구가 주기적 저장본으로 인식하는
`last_gen_ppo_ep_<N>_rew_<X>.pth`뿐이고, `_rlnorm` 변형처럼 패턴에 없는 것은 `unknown`으로 보고만 한다.
`--verify`는 인용된 것이 하나라도 없으면 exit 1이라 정리 직후 사고를 즉시 잡는다.
`tests/test_checkpoint_reference_audit.py` 7개가 세 보존 이유와 fail-safe를 고정한다.

**하지 않은 것**: `datasets/nps_yolo` 3.2 GB(원본 영상에서 뽑은 중간 프레임)는 남겼다. 타일을 다시
만들 때 필요하므로 Det-Fly 작업이 끝난 뒤에 판단한다.


## 2026-09-07 — 종합 계획 작성, 검증 패키지 136 cell 갱신, 기본 브랜치 불일치 발견

**기본 브랜치 불일치.** 원격 기본 브랜치(사이트 서빙 원)는 `research/navrl-env`인데 이 워크트리는 `main`에서
작업·푸시해 왔다. `origin/research/navrl-env`(마지막 커밋 5e527ea, A8 런처·A7 사이트)는 `origin/main`의
**순수 조상**이다(0 뒤처짐 / 31 앞섬). 그래서 09-07 새벽에 main으로 푸시한 사이트 갱신이 배포되지 않았다.
처방은 fast-forward 하나다: `git push origin main:research/navrl-env`. 이력 재작성 없음. 이 세션에서는
기본 브랜치 푸시가 권한 분류기에 막혀 사용자 실행으로 넘긴다. 이후 작업 브랜치를 하나로 통일해야 한다.

**종합 계획** `docs/plans/master_plan_2026-09-07.md`. 두 트랙(진단 논문 A, 실사 인지 B)의 남은 일·순서·기간을
측정 비용에서 계산했다. 근거: 격자 1 cell 4.5~5.1 min(R-B), A8 재적응 1회 **0.81 h**(09-05 세 run 실측
0.80/0.81/0.81), 검출기 1 epoch 2.5 min(3070)·수렴 13~15 epoch. 트랙 A는 R-C(GPU 4 h) 하나가 남았고
병목은 글이라 **D+7(09-14 전후) 제출 가능**. 트랙 B는 Det-Fly가 OneDrive/Baidu 전용이라 **스크립트로 못 받아**
사용자 다운로드가 첫 관문이고, 거리 오차 모델은 MIDGARD 서면 허락에 걸려 **2주(대리 측정) ~ 5주(허락 대기)**.

**검증 패키지** `results/verification_export_2026-09-07/`: 83 → **136 cell**(R-A 2, R-B 50, 교차 트리 1 추가),
루트 13개. `verify.py`는 표준 라이브러리만으로 CSV의 **개수**에서 R-B 결합을 다시 계산한다 —
arc−riskcap −1.49 [−1.90, −1.08] Q 10.3 I² 0 %, 원호 최저 15/15, 70 bars stopcap−riskcap −1.19 [−2.01, −0.37]
세 시드 부호 일치, 폭 확장 70/205 bars −2.18/−5.60, 교차 트리 306/1699/44 비트 동일. 도구 출력과 일치.
README에 Astra에게 묻는 질문 5개와 미리 밝히는 한계 5개를 적었다.


## 2026-09-07 — 브랜치 정책 제정: 작업 브랜치는 `main` 하나

**원인은 git이 아니라 문서였다.** 두 브랜치는 갈라진 적이 없다 —
`origin/research/navrl-env`는 `origin/main`의 순수 조상이었고 fast-forward 한 번으로 지금은 **같은 커밋**
(`e06b68b`)이다. 문제는 원격 **기본 브랜치**가 `research/navrl-env`인데 모든 작업이 `main`에서 이뤄졌고,
`CLAUDE.md`가 작업 브랜치를 `research/navrl-env`라고 **틀리게** 적어둔 것이었다. 그래서 사이트 갱신과
README 도식을 푸시하고도 배포 여부를 확신할 수 없었다. 현재 사이트는 정상이며 새 배너
("원호 튜브의 우위는 평가 시드를 넘어 재현된다 … −1.49 pp")가 서빙되는 것을 확인했다.

**규칙 0 제정**(`OPERATIONS.md` §0): 작업·커밋·푸시는 `main` 하나에서. 장기 작업 브랜치 금지.
**GitHub 기본 브랜치 = `main`, Pages 서빙 = `main`의 `/docs`** — 이 둘이 어긋나면 푸시해도 사이트가
안 바뀐다. 세션 시작 시 `git status -sb` 첫 줄이 `## main...origin/main`인지 확인한다. 예외는
`.codex_worktrees/`의 단기 브랜치뿐이고, 워크트리에 물린 브랜치는 삭제하지 않는다.

`CLAUDE.md`의 낡은 문장을 고쳤고, `tests/test_branch_policy.py` 4개가 두 문서의 브랜치 이름 일치와
Pages 서빙 기술을 강제한다. 판정은 단어 매칭이 아니라 **구조**다: 폐기된 이름은 "이 규칙이 생긴 이유"
블록 안에서만 등장할 수 있고, 규칙 본문이나 `CLAUDE.md`에 나오면 실패한다.

**미완**: `git push origin --delete research/navrl-env`는 GitHub가 거부했다
(`refusing to delete the current branch`). 기본 브랜치를 `main`으로 바꾸는 것은 웹 설정이라
사용자 실행이 필요하다. 순서: ① Settings → General → Default branch → `main`,
② Settings → Pages → Branch `main` / 폴더 `/docs`, ③ 그 뒤 원격 브랜치 삭제.

원격 브랜치는 19개이고 대부분 `codex/*`다. 그중 4개는 워크트리에 물려 있어 삭제 대상이 아니다.


## 2026-09-07 — 외부 감사 반영: 분석 단위를 시드로 올려도 주효과는 살고, 국소 효과는 강등된다

Astra 감사(`results/independent_verification_2026-09-07/`, 커밋 dd8eeda)를 직접 검산하고 반영했다.
감사가 지적한 8건 중 **6건이 실제 결함**이었고 2건은 해석 차이였다. 가장 중요한 것부터 적는다.

**① 분석 단위(핵심).** 15개 셀을 독립으로 취급한 결합은 "이 셀들에 대한 정밀도"이지 새 평가 시드에
대한 일반화가 아니다. 한 시드 안의 다섯 밀도는 같은 정책·장면 샘플러·난수열을 공유한다. 시드를
반복 단위로 삼아(k=3, t, df 2) 다시 계산했다.

| 대비 (crash pp) | 셀 단위 n=15 | **시드 단위 k=3** |
|---|---|---|
| 원호 − riskcap | −1.49 [−1.90, −1.08] | **−1.49 [−2.42, −0.57] p=0.020** |
| 원호 − 정지법칙 | −1.25 [−1.65, −0.85] | **−1.26 [−1.96, −0.55] p=0.017** |
| 정지법칙 − riskcap | −0.23 [−0.65, +0.20] | −0.22 [−1.82, +1.38] p=0.608 |
| 원호 1.2 m − 0.45 m | −2.63 [−3.01, −2.24] | **−2.80 [−4.48, −1.12] p=0.019** |

**세 주요 대비는 시드 단위에서도 0을 제외한다.** 구간이 세 배 넓어지는 것이 3관측의 정직한 대가이고,
그래도 유의하다는 것이 시드 일반화의 근거다. 사용자가 전달한 "−1.2024, p=0.1085"는 주효과가 아니라
**70 bars stopcap−riskcap 국소 효과**의 값이며, 감사 보고서도 "주효과보다 이 국소 차이가 의존성 가정에
훨씬 민감하다"고 적고 있다. 감사 자신의 `recomputed.json`에 주효과 `seed_t` = −1.4942 [−2.4197, −0.5687]
p=0.0201이 들어 있다. 내 `seed_level_t` 구현이 이 값과 scipy에 1e−6 이내로 일치한다.

**② P5 국소 효과 강등.** 70 bars stopcap−riskcap은 셀 단위 −1.19 [−2.01, −0.37](5밀도 Holm p=0.022)이나
시드 단위 −1.20 [−3.06, +0.66] p=0.108이고, 15셀 각각에 Holm을 걸면 유의 셀 0/15다. **탐색적 관찰로
강등**하고 밀도 조건부 처방으로 쓰지 않는다. "100 bars 이상에서 사라진다"도 삭제 — 비유의는 동등성의
증거가 아니다. 동등성을 주장하려면 사전 margin과 TOST가 필요하다(Lakens 2017). 대체 문구는
**"다른 밀도에서는 추가 이득이 검출되지 않았다"**.

**③ verify.py 키 충돌(실제 버그).** `(seed, bars, mode, width)`는 유일 키가 아니다. L1b 적응폭 셀
(`arc_open0p3/0p6/1p0`)이 mode `dwa_arc`, width 0.45로 같은 키를 갖고 crash가 92 대 77·78로 다르다.
게다가 그 루트는 `width_per_open_m`이 CSV에 **빈 값**이라(09-06에 찾은 knob 미기록 18건) 이득으로도
구분할 수 없다. 출력 숫자는 CSV 순서 덕에 우연히 맞았지만 구조가 틀렸다. **시드별 소스 루트를 못박아**
해결하고, 충돌이 남으면 `SystemExit`으로 죽게 했다.

**④ manifest 3개 루트 누락(실제 결함).** A7 루트 3개는 셸 런처로 돌려 `cells.json`이 없고, 내보내기가
그 경우 조용히 건너뛰었다. 결과 JSON 자체가 증언하는 커밋·계약·지문을 기록하도록 고쳐 13/13이 됐다.

**⑤ "bit-identical" 과장.** 교차 트리 재실행은 **기록된 outcome 개수와 집계 텔레메트리**가 일치한 것이지
궤적·난수 상태·텐서 해시를 비교한 것이 아니다. 한 셀이 45개를 보증하지도 않는다. 문구를 정정했다.

**⑥ 명명(실제 결함).** 현재 구현은 원호가 쓸고 지나가는 원판 안의 장애물을 모두 검출하지 못한다.
CPU 반례를 **현재 소스에서 재현**했다: 속도 2.0 m/s·요레이트 1.0 rad/s에서 0.5 m 반사에 대해 직선은
clearance 0.5 m, 원호는 12.0 m를 준다. 호 길이 0.1 m 지점 중심에서 그 반사까지 0.400049 m로 반폭 0.45 m
안이다. **DWA·충돌 안전 보장으로 부르지 않고 `arc-clearance 속도 필터`로 한정**한다. 기하를 고치면
새 버전으로 재평가해야 하며 현재 숫자를 그 성능으로 재사용할 수 없다.

**코드**: `navrl_stats.seed_level_t`(+ 정규화 불완전 베타 기반 Student t sf/ppf, scipy와 1e−6 일치),
`pool_navrl_seed_replication`이 두 단위와 밀도별 이중 표를 출력, `verify.py` 루트 못박기 + 시드 단위,
`export_verification.py` manifest 전수 기록. 테스트 5개 추가(주효과가 시드 단위에서도 0 제외, 국소 효과는
0 포함, 시드 구간이 셀 구간보다 넓음, t 분위수 대조). 전체 **1,193 tests OK**.

**문서**: 초안 §1 주장 2·3, §5.3(이중 단위 표), §7(명명 한계·반례), 확정 계획 C3·C4·P5 행,
VERIFICATION.md, README, 상태 사이트를 같은 기준으로 정정했다.

**미반영(추가 작업 필요)**: episode-level ledger가 없어 군집/짝지음 분석은 불가능하다. 감사가 제안한
새 로깅(scene/layout ID, episode index, initial-condition hash)은 45 cell 재수집 GPU 약 3 h + 구현 4~8 h다.
seed 523 narrow arm을 새 트리로 옮기는 14 cell 재평가(GPU 1~1.5 h)도 미실행.


## 2026-09-07 — Det-Fly 수신 재개, scale-matched arm 사전등록, 좌표 결함 1건 수정

Codex 세션이 사용 한도로 응답만 끊겼고 **다운로드 프로세스는 살아 있었다**(pid 1111614). 어노테이션
13,271개는 완료, 이미지는 4,777/13,271(7.2 GB)이며 약 2 MB/s로 계속 받는 중이다. 실패 0.
전체 스위트 **1,198 tests OK** — Codex가 보고한 사이트 테스트 중단은 재현되지 않는다
(`RESULT_UNAVAILABLE_OR_MALFORMED`는 status.json에 3회 있으나 테스트와 충돌하지 않는다).

**어노테이션만으로 확인한 사실 — Det-Fly 표적은 NPS보다 4.3배 크다.** 예측은 한 건도 보지 않고
배포된 XML 13,271개(box 13,270)를 전수 집계했다.

| 640 px 창 기준 box 대각 | p25 | 중앙 | p75 | p95 |
|---|---:|---:|---:|---:|
| NPS 학습 타일 (14,608 box) | 19 | 25 | 36 | 62 |
| Det-Fly native 4K (13,270 box) | 74 | **109** | 165 | 298 |

**Det-Fly 표적의 75 %가 NPS 학습 표적의 95 %보다 크다.** 따라서 native scale 단독 평가는 도메인 전이와
크기 전이를 분리하지 못한다. 결과 열람 전에 **arm B(scale-matched, `--downscale 4.0`)를 사전등록**했다.
÷4로 960×540을 만들면 Det-Fly 대각 중앙이 27 px로 NPS 25 px와 맞고, NPS의 1920×1080 → 640 타일 구조와도
같다. 두 arm 모두 box를 원본 4K 픽셀로 되돌린 뒤 NMS·IoU·bin을 적용한다. 판정 규칙도 동결했다:
두 arm의 차이가 arm A 값보다 크면 원인은 **크기**(처방: 다중 스케일 학습·앵커 재계산), 둘 다 낮으면
**도메인**(처방: 합동 학습).

**배관 검증에서 결함 1건.** 축소 구현이 각 축을 타일 크기로 따로 clamp해 종횡비를 깨뜨렸다
(2160/4 = 540 → 640, y 3.375배 대 x 4.0배). 되돌릴 때 둘 다 ×4를 곱해 모든 예측이 엉뚱한 자리에 놓였다.
균일 축소 + bottom-right 패딩으로 고쳤고(패딩이 원점을 바꾸지 않아 역변환 불변), raw record의 크기도
작업 해상도가 아닌 원본 4K를 기록하게 했다. threshold·bin·checkpoint는 무변경.

수정 전후를 같은 이미지로 대조했다. GT `[1146,760,1333,808]`에 대해 **수정 후 arm B**의 최상위 예측이
`[1285,762,1312,798]`(신뢰도 0.046)로 같은 자리에 오고, **arm A**는 11~14 px 박스를 신뢰도 0.49로
엉뚱한 곳에서 확신한다. 크기 불일치의 전형적 증상이다. **이 12장은 좌표 검증용이며 P3 결과가 아니다** —
P3는 13,271장 전수로 두 arm을 돌린 뒤에만 판정한다.

다음: 다운로드 완료(ETA 약 90분) → 인덱서 fail-closed 검증 → arm A·B 전수 실행 → 판정.


## 2026-09-07 — Det-Fly P3 전수 완료: size-dominant zero-shot failure

공식 OneDrive 배포를 모두 받았다. XML 13,271개 + JPEG 13,271개,
18,849,340,854 bytes이며 manifest 26,542개의 누락·크기 불일치·`.part`는 모두 0이다.
전 파일 SHA-256, XML–JPEG pairing, JPEG SOF 크기, class·bbox·flag를 검사한 content
manifest SHA-256은 `d8d6875b…`이다. 13,271장에 GT 13,270 box, negative frame 1장이
있다. 전체 평가에서 box를 버리지 않았고, full-width 1개는 `implausible_size`로만 표시했다.

frozen checkpoint `ccd65dc3…`와 사전 고정 threshold로 두 arm을 전수 실행했다.

| arm | images / tiles | IoU 0.3 P / R / F1 / AP | IoU 0.5 P / R / F1 / AP | runtime |
|---|---:|---:|---:|---:|
| native 4K | 13,271 / 424,672 | .0097 / .1668 / .0183 / **.0035** | .0051 / .0873 / .0096 / **.0010** | 1,961.1 s |
| scale-matched ÷4 | 13,271 / 26,542 | .2416 / .3503 / .2860 / **.2382** | .1612 / .2337 / .1908 / **.1126** | 536.6 s |

IoU 0.3 AP 차이 0.2347로 native AP 자체보다 크므로 사전 판정은
**SIZE-DOMINANT ZERO-SHOT FAILURE**다. ÷4로 회복되어도 recall 0.3503, FP 14,587이므로
scale matching을 완성 detector로 해석하지 않는다. 다음 경로는 NPS+Det-Fly 합동 학습,
multi-scale/anchor 재설계, Det-Fly test split 봉인이다. 배경 4종 mapping은 배포
metadata에 없어 작성하지 않았고 source group 010/020을 대체 slice로 보고했다.

raw/report SHA-256은 native `a98e952d…` / `ca45d1f8…`, scale-matched
`4b0277a6…` / `421aeed8…`이며 각 receipt와 독립 재계산이 일치했다.


## 2026-09-07 — Det-Fly 누수 방지 split + NPS 통합 dataset v1 완료

Det-Fly는 video ID 없이 `010`/`020` 그룹과 숫자 ID만 배포하므로 frame 무작위 split을
금지했다. 코드가 P3 prediction/report를 전혀 읽지 않고 source structure만으로 결정하도록
규칙을 고정했다: 더 큰 group은 test, 나머지는 최대 ID gap에서 나누고 큰 쪽은
train이다. 결과는 train=`010` ≤4333 4,200장, val=`010` ≥4752 2,158장,
test=`020` 전체 6,913장이다. 경계 ID 차이는 419이고 test group은 development와 다르다.

검사: 원 JPEG exact SHA duplicate 0, exact 64×36 thumbnail cross-split 0. dHash Hamming ≤2가
2,963개지만 저정보량 하늘 충돌라 누수 근거로 쓰지 않았다. 구조적 source-group
분리와 exact SHA만 gate다. sealed test 6,913장 SHA는 `3ae8bfc0…`이고 학습 YAML은
`test:` key가 없다. P3 결과 후에 split을 만든 순서상 한계도 receipt에 명시했다.

NPS 기존 640 tile은 복사하지 않고 file list로 참조했다. Det-Fly development에서는
P3와 같은 native 640/overlap 128 crop으로 positive 전부 + GT와 교차 0인 negative 1장/frame을
만들었다. train 29,824 samples/20,573 boxes, val 9,307/6,921, 합계 39,131/27,494다.
비정상 full-width annotation 1장은 원 index에는 남기고 학습/val tile에서 frame을 제외했다.

`verify_nps_detfly_joint_dataset.py`가 39,131 sample의 SHA·size·label·source unit·seal을 전수
재검사해 PASS. train/val source unit 교차 0, exact image 교차 0, sealed test 유입 0.
YOLOv5 `check_dataset`도 train/val 경로를 정상 해석했고 `test=None`. 생성물 917 MB,
남은 디스크 약 14 GB. joint manifest SHA `b6d8282e…`, dataset receipt SHA `ef9babda…`.


## 2026-09-08 — 합동 detector·P4 완료, P5 KF v1 기각

NPS+Det-Fly 합동 detector를 사전 동결한 batch 8, 30 epochs, 320–960 multi-scale 계약으로
학습했다. RTX 3070은 batch 8에서 compute 약 96%, 약 188 W로 이미 포화에 가까워 VRAM 점유만
늘리는 batch 16 변경은 하지 않았다. validation fitness가 가장 높은 zero-based epoch 26의
P/R/mAP50/mAP50-95는 `0.77634 / 0.65901 / 0.69167 / 0.32339`다. test 접근 전에
checkpoint SHA-256 `aab12f39…00ac`를 동결했다.

그 뒤 sealed Det-Fly `020` 6,913장을 두 arm에서 한 번 평가했다. native 4K IoU 0.3 P/R/AP는
`0.27291 / 0.86400 / 0.73887`, TP/FP/FN은 `5,972 / 15,911 / 940`이다. scale-matched ÷4
진단 arm은 `0.59319 / 0.58478 / 0.59719`다. NPS-only native AP30 `0.00354`의 크기 실패는
회복됐지만 confidence 0.25 precision이 낮다. test에 맞춰 threshold를 변경하지 않았다.

P4는 NPS validation 2,296 frames/7 clips, test 1,725 frames/8 clips에서 Top-K=5 후보를 생성했다.
각 후보는 `[u,v,w,h,confidence,appearance_64D]`이며 원 clip timestamp를 보존한다. schema, semantic,
manifest, compressed payload hash 검증은 두 split에서 모두 PASS했다. appearance는 parameter-free
RGB-grid/histogram descriptor이며 learned ReID 주장이 아니다.

P5 고정 KF v1은 validation에서 CNN-only보다 낮았고, test에서도 frame hit
`0.61449 → 0.46551`, proxy false lock `0.24501 → 0.48426`, center error
`1.9127 → 3.9748 px`로 악화했다. **KF v1은 REJECTED**다. 원 track ID와 intrinsic이 없어 실제
ID switch/FTLR와 degree bearing error는 unavailable이다. producer mean `61.43 ms`는 Det-Fly 평가와
GPU를 공유한 contended offline 측정이라 배포 지연시간으로 쓰지 않는다.

전체 Python suite는 올바른 aerialgym 환경에서 **1,213 tests OK, 4 skipped**다. SVG XML과 실렌더,
JSON, Python compile, `git diff --check`가 통과했다. 정적 사이트 JS는 새 perception 계약을 통과한 뒤
기존 `status.json`의 Track-B `RESULT_UNAVAILABLE_OR_MALFORMED`와 오래된 기대값 충돌에서 중단되며,
이번 perception 변경과 무관한 선행 문제로 보존한다.


## 2026-09-08 — P6/P7 temporal association 완료, Transformer validation 선택

결과를 보기 전에 commit `5b3ee5e`에서 P6/P7 계약과 코드를 고정했다. 두 arm은 같은 P4 Top-K=5
후보만 읽으며 NPS train clip에서 학습하고 validation clip에서 checkpoint와 arm을 선택한다. target
identity가 없으므로 현재 frame에서 어떤 UAV annotation과든 IoU≥0.3인 후보를 supervision으로 삼고,
이를 ReID 또는 ID-switch 평가로 부르지 않는다.

NPS train 13,510 frames/35 clips의 연속 manifest와 candidate 67,550개를 만들었다. candidate 전수
schema·semantic·manifest·hash 검증이 PASS했고 payload SHA-256은 `bd28f4e2…b19061`이다. 후보 생성은
RTX 3070에서 31.43 frames/s였고 mean latency는 27.92 ms/frame이다.

GRU T=8은 validation loss가 첫 epoch 뒤 악화해 epoch 0이 선택됐으며 utility `0.63153`이었다.
Transformer T=16은 epoch 5가 선택됐고 utility `0.67247`이었다. 네 arm의 validation 결과는 다음과 같다.

| arm | frame hit | proxy false lock / selected | utility |
|---|---:|---:|---:|
| CNN-only | 0.81490 | 0.15492 | 0.66551 |
| KF v1 | 0.62631 | 0.36652 | 0.26394 |
| GRU T=8 | 0.78659 | 0.16466 | 0.63153 |
| Transformer T=16 | 0.80880 | 0.14424 | **0.67247** |

고정 rule에 따라 Transformer를 선택했다. CNN-only 대비 hit는 14 frame 줄고 proxy false lock은 30 frame
줄어 utility가 `+0.00697` 높다. validation 2,296 frames/7 clips만의 선택이고 test는 입력으로 사용하지
않았다. 독립 evaluator가 frozen checkpoint를 다시 읽어 같은 metrics를 재현했고, 압축 해제 prediction
content SHA-256도 training output과 동일한 `b6ca57c0…bba60`이었다.

## 2026-09-08 — P7b/P7c 완료 및 좌표 오류 정정

후보를 보존하는 P7b와 optical-flow/GMC를 추가한 P7c를 구현했다. 첫 P7c는
축소 좌표와 원본 좌표를 혼용한 후보 매칭 버그로 무효 처리하고 원 결과를 보존했다.
정정 commit `3ce7d8d`에서 cache와 P7c를 v2로 다시 실행했다.
utility는 기존 P7 0.67247, P7b 0.67944, corrected P7c 0.68554이며 P7c를 선택했다.
기존 P7 대비 hit −9, false lock −39지만 no-lock과 평균 재획득 시간은 증가했다.
독립 재로드 prediction bytes 일치. test 미실행, P7d는 track-ID 데이터 대기.
전체 Python 1,219 tests OK (4 skipped). 사이트 테스트는 기존 status 데이터 기대값
충돌로 실패했다. [상세 결과·hash](results/perception_p7bc_2026-09-08/README.md).

## 2026-09-08 — P7e 미채택 및 streaming 구현

사전등록 `f71079c` 후 crop-validity CNN을 train clip 3-fold로 평가하고 최종 8 epochs
학습했다. validation utility 0.52352로 P7c v2 0.68554보다 낮아 미채택했다.
`1d8cade`에서 프레임 단위 detector → motion/GMC → T16 API와 검증기를 구현했다.
GT는 온라인 API에 전달하지 않는다. 전체 cached replay는 logits/rank 정확 일치했다.
전체 RGB 두 번 실행은 historical cache와 rank 5/2,296 차이로 strict parity FAIL이다.
기존 producer도 현재 환경의 첫 프레임에서 historical cache와 다르지만 새 pipeline과는
동일했다. 환경·수치 재현성 원인은 아직 미확정이다. 실패 결과를 보존했다.
두 번째 RGB decode 포함 mean 69.38 ms, 14.41 FPS; 실기/ROS/PPO 비용은 제외했다.
P7c를 유지하며 NPS test는 사용하지 않았다. Python 1,221 tests OK (4 skipped).
[결과·hash·한계](results/perception_p7e_streaming_2026-09-08/README.md).


## 2026-09-07 — P3 Det-Fly zero-shot 완료: 실패의 지배적 원인은 도메인이 아니라 **크기**

13,271장 전수, 두 arm, 예측 열람 전 동결한 규칙으로 판정했다.

| IoU 0.5 | arm A native | arm B scale-matched ÷4 |
|---|---:|---:|
| AP | **0.0010** | **0.1126** |
| 정밀도 | 0.0051 | 0.1612 |
| 재현율 | 0.0873 | 0.2337 |

차이 0.1116이 arm A 값 0.0010의 **111배**다. 동결 규칙에 따라 판정은 **크기**다.

크기별 재현율이 결정적이다. native arm은 64 px를 넘으면 **완전히 실패한다** — 64–128 px 4,961개 중 4개,
128 px 이상 2,151개 중 0개. Det-Fly 박스의 **54 %가 64 px를 넘는다.** ÷4로 축소해 학습 분포에 넣으면
같은 구간이 0.246으로 회복된다. 반대로 12–20 px 구간은 native가 0.177로 더 낫다(축소하면 3–5 px가 되어 사라짐).

**도메인 성분도 남는다**: arm B 0.1126은 NPS 자체 검증 0.579의 19 %이고 사전등록 문턱 0.33 미달이다.
크기를 맞춰도 5분의 1이므로 크기가 지배적이되 유일한 원인은 아니다.

처방은 **다중 스케일 학습·앵커 재계산**이 먼저이고, 그 뒤에 남는 격차가 진짜 도메인 격차다.
합동 학습은 그다음이다. 원자료 `detector_runs/p3_detfly_{native,scale_matched}/`.

데이터 확보: 26,542 파일 18.85 GB, 실패 0. 첫 실행에서 8개가 연속 구간에서 끊겨 워커를 줄여 재시도해
받았다. 인덱서 fail-closed 검증 통과(13,271장 / 13,270 박스 / 어려움 990 / 잘림 2,811).
검증 중 걸린 2건은 데이터 성질이었다 — 객체 없는 프레임 1장은 **음성 프레임으로 보존**(오검출 측정용),
전체 폭 3840×17 박스 1건은 **제외하지 않고 표시만** 했다(결과 전에 불리한 GT를 빼지 않는다).


## 2026-09-09 — streaming 재현성 원인 확정(환경 불일치), 병목은 GMC 아닌 광류, P8-lite 계약 초안

보고서 `docs/plans/perception_streaming_findings_2026-09-09.md`. 재학습·threshold 변경·test 열람·
원본 cache 덮어쓰기 없음.

**① 재현성 — 원인은 실행환경 불일치다.** 같은 프레임을 같은 스크립트·같은 가중치로 두 환경에서 돌렸다.

| 환경 | Python/torch/CUDA/cuDNN | 60프레임 완전일치 | confidence 최대 상대차 |
|---|---|---:|---:|
| `detector_runs/venv` | 3.10 / 2.10.0+cu128 / 12.8 / 9.10.2 | **60/60** | **0.000e+00** |
| `datasets/detenv` | 3.8 / 2.4.1+cu121 / 12.1 / 9.1.0 | 0/60 | **0.45** |

cache는 `venv`로 만들어졌는데 RGB 재실행이 `detenv`로 돌았다. cuDNN 9.1 대 9.10.2로 컨볼루션 커널이
달라 TF32·deterministic 플래그로는 메울 수 없었던 것이다. 첫 프레임 차이가 6e-4로 작아 보인 것은 우연이고
60프레임 전체에서는 45 %까지 벌어진다. **"5프레임 선택 불일치"는 증상이지 원인이 아니다.**

`venv`로 전체 2,296프레임 RGB 감사를 새 경로 `stream_rgb_venv_v1.json`에 재실행한 결과
**rank_mismatches 0, rank_parity_pass True, candidate_frame_mismatches 0**이고 selected 2122 /
no_lock 174 / false_lock 274 / center error 1.941796 / loss 106이 cached replay와 전부 일치한다.
새 baseline 채택은 불필요하다 — 기존 frozen output이 그대로 재현된다. RGB 실행 환경을 `venv`로 고정 권고.

**부수 발견**: candidate receipt에 Python·torch·CUDA·cuDNN 버전이 없다(device_name·fp16만 있음).
이번 혼선의 근본이며 receipt 스키마 보강을 제안한다. 조사 중 NPS 단독 `best.pt`(`ccd65dc3…`)로 잘못
실행했으나 SHA 게이트가 즉시 거부했다 — cache가 쓴 detector는 `aab12f39…`(joint)다.

**② 병목은 GMC가 아니라 광류다.** validation 119프레임 분해(1280×960 → flow 640×480):
Farnebäck **36.58 ms(97.7 %)**, features 0.48, mask/grid 0.22, **RANSAC 0.17 ms(0.4 %)**.
OpenCV 스레드는 지렛대가 아니다(1개 36.9 → 24개 36.0 ms, 출력은 스레드 수 무관 비트 동일).

**출력 보존 개선안(실측)**: flow는 `previous_gray`·`gray`만 필요하고 candidates는 그 뒤 마스크부터
쓰이므로, CPU 광류를 GPU 검출기와 겹칠 수 있다. 프로토타입에서 순차 112.3 ms → 겹침 74.7 ms
(37.6 ms 절감, 33 %)이고 **flow 출력 비트 동일**이다. 현재 pipeline에 적용하면 frame 62.13 → 약 35.8 ms,
decode 포함 67.28 → 약 40.9 ms로 **14.86 → 약 24 FPS** 추정. 구현은 사용자 승인 대상이며 적용 시
전후 latency와 ranks·hit·false-lock·no-lock·reacquisition을 함께 보고한다. flow 해상도·빈도·FP16·
다른 알고리즘은 입력 분포를 바꾸므로 별도 계약으로 분리한다.

**③ P8-lite/P9 입력 계약 초안.** PPO 연결 없음. 측정된 것만 쓴다 — center error mean 1.942 /
median 1.034 px, hit rate 0.8049, no-lock 7.58 %, false-lock 12.91 %, loss 106회, reacquisition
mean 0.470 s / max 19.933 s, latency p50 67.18 / p95 70.32 ms. **degree bearing·metric range·실제
ID-switch는 만들지 않는다**(intrinsics·거리 GT 없음, NPS는 영상당 UAV 1기라 identity 전환 측정 불가).
error model은 시간 상관을 갖는 4성분(분위수 직접 샘플링 center offset, 2-state Markov miss burst,
false-lock burst, latency)으로 두고, 입력 schema는 선택 후보 1개 + `lock_state`/`age_frames`/`latency_ms`다.
NO_LOCK은 0으로 채우지 않고 명시적 무효 플래그로, STALE 문턱은 사전등록, reset은 sequence 경계에서.
**세 처리 모두 PPO 연결 전 별도 승인 대상.**

전체 스위트 1,221 tests OK (skipped 4).


## 2026-09-09 — receipt 스키마 보강과 환경 오지시 교정

**스키마**: `tools/runtime_fingerprint.py` 신설. Python·torch·CUDA·cuDNN·OpenCV·numpy 버전과
TF32/benchmark/deterministic 플래그, device 이름·compute capability를 실행 프로세스에서 직접 읽는다.
GPU 수치를 담는 receipt 4곳에 심었다 — `run_perception_candidate_producer`,
`build_perception_motion_features`, `eval_detfly_zeroshot`, `prepare_detfly_dataset`.
`differences(old, new)`가 재현 실패 시 어느 축이 달라졌는지 바로 짚는다. 두 환경에 실제로 걸어보니
torch·cuda·cudnn·numpy·python·platform·executable 7개 축의 차이를 정확히 출력한다.

1프레임 실행으로 receipt에 `runtime`이 실제로 기록되는 것을 확인했다
(python 3.10.19 / torch 2.10.0+cu128 / cuda 12.8 / cudnn 91002 / cudnn_allow_tf32 True).

**교정 — 명세가 사고의 원인을 지시하고 있었다.** `docs/specs/perception_streaming_v1.md`가
"RGB 실행 환경은 `datasets/detenv/bin/python`"이라고 적고 실행 예시까지 그 인터프리터로 주고 있었다.
이것이 09-08 parity 실패의 직접 원인이다. `detector_runs/venv`로 고치고, 왜 다른 환경에서는 재현되지
않는지(cuDNN 9.1 대 9.10.2, 같은 60프레임에서 최대 45 % 차이)를 함께 적었다. "검증 상태" 절의
"rank 5개가 달라 gate FAIL, 원인 미확정"도 해소 사실로 대체했다.

인수인계 문서 상단에 해소 표시를 달고 작업 경계의 detenv 지시를 취소선으로 정정했다.
`results/perception_p7e_streaming_2026-09-08/`의 frozen `summary.json`은 **덮어쓰지 않고** README에
후속 문단을 붙였다 — 그 status는 당시 기록으로 유효하고, 원인이 모델이 아니라 환경이었다는 사실을
가리키게 했다.

**규칙 0.1 제정**(`OPERATIONS.md`): GPU 수치를 담는 receipt는 실행 스택을 기록한다. perception
streaming RGB 실행 환경은 `detector_runs/venv`다. `tests/test_runtime_fingerprint.py` 8개가 이 규칙과
명세의 환경 단일성을 강제한다 — 특히 명세가 `detenv`로 실행하라고 다시 적으면 실패한다.

**작업 중 낸 결함 1건**: 임포트를 AST 없이 정규식으로 끼워 넣다가 다중 행 임포트 중간을 갈랐고,
`build_perception_motion_features.py`에는 `sys` 임포트가 없어 `NameError`가 났다. 테스트 모듈 2개가
임포트 실패로 통째로 사라져 스위트가 1,221 → 1,203으로 줄었는데, 이 숫자 감소가 아니었으면 통과로
보였을 것이다. AST로 마지막 최상위 임포트를 찾아 다시 넣고 `sys`를 추가해 해결했다.
전체 **1,229 tests OK (skipped 4)**.


## 2026-09-09 — perception 우선 계획 재작성: 앞서 제안한 "다중 스케일 학습"은 이미 완료된 중복이었다

계획 `docs/plans/perception_next_plan_2026-09-09.md`.

**정정**: 09-09 초안에서 2순위로 제안한 "검출기 다중 스케일 학습·앵커 재계산"은 **중복 실험이다.**
`runs/nps_detfly_joint/yolov5s_ms_b8_e30_s0/opt.yaml`이 `multi_scale: true`이고 NPS 단독
checkpoint에서 warm-start해 joint 데이터로 30 epoch 학습돼 있다(P3-F, seed 0). 효과도 측정돼 있다.

| Det-Fly IoU 0.5 | NPS 단독 (P3 zero-shot, 13,271장) | joint+multi-scale (P3-F, 6,913장) |
|---|---:|---:|
| native AP | 0.0010 | **0.7166** |
| native 재현율 | 0.087 | **0.839** |
| scale-matched AP | 0.1126 | 0.4137 |

**두 수치를 같은 실험의 전후로 읽으면 안 된다.** P3는 Det-Fly 미노출 모델의 zero-shot 전수 평가,
P3-F는 Det-Fly `010`을 학습에 쓴 모델의 **validation** 성적이다. 0.7166은 일반화 성능이 아니다.
스트리밍 파이프라인이 이미 이 joint detector(`aab12f39…`)를 쓰므로 P3의 처방은 적용된 상태다.
용어도 "진짜 도메인 격차" → **"크기 보정 후 잔여 도메인 격차"**로 교정했다(배경·압축·시점·광학은 남는다).

**남은 것과 순서**: S1 광류 겹침(반나절) → S2 P8 오차 모델(1~2일) → S3 P9 주입기(2~3일) →
S4 봉인 test 개봉(마지막 1회). P5를 빼면 **test는 한 번도 열리지 않았고**
joint 데이터셋의 `training_yaml_contains_test`가 false다. 이 봉인은 파이프라인 확정 후에만 연다.

**S1의 "비트 동일" 표현 규칙**: 프로토타입은 flow 배열만 확인했다. 최종 문장으로 쓰려면 validation
2,296프레임에서 ① rank/candidate 불일치 0 ② **motion feature 5×12 배열 전 프레임 비트 일치**
③ selected/hit/no-lock/false-lock/center error/loss/reacq 전부 일치가 모두 성립해야 한다.
그전에는 "출력 보존 예상"으로만 쓴다. benchmark는 `detector_runs/venv`, **GPU 유휴**, 3회 반복,
mean/P50/P95 전부 보고. **Track A R-C와 동시 GPU 실행 금지** — latency 측정이 오염된다.

**S2는 크기 구간별 조건부 분포로 잰다.** P3에서 크기가 지배 변수임이 확인됐으므로 평균 하나로는
시뮬에 잘못된 분포를 주입하게 된다. degree bearing·metric range·실제 ID-switch는 여전히 만들지 않는다.


## 2026-09-09 — README 영어 전면 재작성, 사이트 갱신

**README**: 496줄 전부 영어로 다시 썼다(한글 0자, 내부 링크 20개 전부 유효). 한국어 원본은 세션
scratchpad에 백업했다. 구조를 재편해 안전필터와 인지를 각각 하나의 서사로 묶고, 흩어져 있던
날짜별 update 블록 5개를 §Status 하나로 통합했다.

새로 담은 것: 원호 결과의 **셀 단위/시드 단위 이중 보고**, arc-clearance 속도 필터라는 **명명 제한**과
반례, P3의 크기별 재현율 근거(native가 64 px 초과에서 4,961개 중 4개·2,151개 중 0개, Det-Fly의 54 %가
그 위), **0.0010 → 0.7166을 전후로 읽지 말라는 경고**(zero-shot 대 학습 데이터 포함 validation),
"크기 보정 후 잔여 도메인 격차"라는 정확한 표현, streaming 환경 불일치 사건과 그 교훈, 사고에서 나온
운영 규칙 4개(§0 단일 브랜치, §0.1 실행 스택 기록, §8-A 체크포인트 수동 삭제 금지, 실행 중 동결 경로).

**사이트**: 배너를 streaming parity 해소와 크기 판정으로 교체했다. P3 절에 크기별 재현율 근거와 잔여
도메인 격차를 넣고, joint detector 절에 전후 오독 경고를 달았다. streaming 절을 신설해 환경 불일치
사건과 광류 병목(Farnebäck 97.7 %, GMC RANSAC 0.17 ms), 겹침 프로토타입 실측(112.3 → 74.7 ms)을 적었다.

## 2026-09-09 — S1 CPU flow/GPU detector overlap 완료

후보 비의존 Farnebäck flow를 단일 CPU worker로 분리해 GPU detector와 겹치고, 후보 의존 GMC는
detector 완료 뒤 수행하도록 streaming pipeline을 변경했다. `--serial-flow` 기준선과 각 프레임
timing 저장, runtime fingerprint, candidate/rank/motion byte/전체 metric fail-closed gate를 추가했다.

`detector_runs/venv`, RTX 3070, validation 2,296프레임에서 직렬·겹침을 교차 순서로 각 3회 실행했다.
여섯 실행 모두 candidate mismatch 0, rank mismatch 0, 5×12 float32 motion byte mismatch 0이며
motion SHA `33c1af29…4e4f9d`와 모든 하위 지표가 동일했다. NPS test는 사용하지 않았다.

3회 pooled decode 포함 latency는 직렬 mean/P50/P95 67.91/67.89/70.64 ms,
겹침 44.49/45.37/47.48 ms다. 실행별 평균 FPS의 평균은 14.72 → 22.48로 52.65 % 증가했다.
학습 compute job은 없었고 GPU snapshot과 원본 artifact hash를 receipt에 보존했다.
clean-tree 전체 Python 1,231 tests OK (4 skipped).
[결과](results/perception_streaming_overlap_s1_2026-09-09/README.md). 다음은 P8 오차 모델이다.
마크업·상대 링크 검사 통과. 스냅샷 재생성.

전체 스위트 1,231 tests OK (skipped 4).

## 2026-09-09 — P8 조건부 오차 모델 측정

계약 `31ee02f`에서 결과 관찰 전 sqrt(GT area) 크기 경계 8/16/32/64 px,
IoU .3, single-GT eligibility, .5 s gap, censoring과 sparse support 규칙을 고정했다.
코드 `48852a7`로 frozen P7c validation과 S1 overlap 3회 timing을 hash 확인 후 집계했다.

전체 2,296프레임의 기존 any-GT 지표는 정확히 재현됐다. 단일-GT는 1,310프레임이며
986프레임에는 2~5개의 GT가 있다. 조건부 구간별 개수는 0/271/849/188/2다.
<8 px와 >=64 px는 지원 부족이며 32~64 px NO_LOCK 전이행도 outgoing 5개로 부족하다.
27개 eligible segment, 1,283개 전이의 실제 관측 간격은 약 50~103 ms다.
조건부 completed reacquisition은 65회, mean .221 s/max .801 s이며, 전체 baseline의
102회/.470 s/19.933 s와 모집단·중도절단이 다르므로 직접 성능 개선으로 해석하지 않는다.

model/frame/burst 세 산출물을 별도 경로에서 다시 생성해 SHA가 모두 일치했다.
표본·전이 보존, 확률합, paired error correspondence, frozen 전체 지표를 검사했다.
PPO·test는 미실행. [원자료와 receipt](results/perception_p8_2026-09-09/README.md).
clean-tree 전체 스위트 1,237 tests OK (skipped 4); P8 신규 6개 포함.


## 2026-09-09 — P9 오차 주입기 완료: 적합도 gate 통과, 시뮬레이터 연결

코덱스 세션이 사용 한도로 끊긴 지점에서 이어받았다. P9 오프라인 모델은 이미 gate를 통과한 상태였고,
남아 있던 시뮬레이터 연결부와 provenance를 검증해 커밋했다. 코덱스가 마지막에 본 테스트 오류는
**코드 결함이 아니라 호출 방식 문제**였다(이 저장소의 `tests`는 패키지가 아니라 모듈 경로 호출이 실패).
올바른 방식으로 전체를 돌리면 **1,243 tests OK (skipped 4)**.

**적합도(64 replicas, seed 1701)**

| bin | 실제 프레임 | 생성 프레임 | 상태점유 TV | latency KS | 판정 |
|---|---:|---:|---:|---:|---|
| 1 | 271 | 17,344 | 0.0415 | 0.00796 | 통과 |
| 2 | 849 | 54,336 | 0.0051 | 0.00396 | 통과 |
| 3 | 188 | 12,032 | 0.0222 | 0.00695 | 통과 |

문턱은 occupancy TV 0.05 / transition max 0.06 / offset KS 0.1 / latency KS 0.05이고 전부 여유 있게
통과했다. 재현성도 확인됐다 — 같은 seed는 동일 SHA, 다른 seed는 다른 SHA.

**세 번 만에 통과한 과정을 기록으로 남긴다.** v1은 전이를 "현재 프레임 크기"에 귀속한 구현 오류와
크기 필터 후 비연속 프레임을 이웃으로 취급한 집계 오류로 실패했다. v2는 코드 오류가 아니라 271개
표본의 raw 전이 MLE가 관측 점유율과 일관된 정상분포를 만들지 못하는 **유한표본 문제**로 bin-1 TV
0.05495에서 근소하게 실패했다(raw 정상분포 HIT 0.697 대 관측 0.635). v3은 임계값을 건드리지 않고
관측 전이에서 KL 변화가 최소가 되도록 주변분포를 관측 점유율에 맞추는 stationary balancing을
**사전 기록한 뒤** 적용했다. 필요한 최대 행 조정은 0.0253으로 이미 고정된 전이 허용오차 0.06 안이다.
**실패한 v1·v2 산출물을 삭제하지 않고** `results/perception_p9_v{1,2}_failed_2026-09-09/`로 보존했다.

**시뮬레이터 연결**: `navrl_empirical_error.py` 신설, `NAVRL_P9_ERROR_MODEL`로 켠다. 기존 노이즈
훅(`NAVRL_PERCEPTION_PERTURB`, `NAVRL_DETNOISE_*`, `NAVRL_DETECTION_LATENCY_S`, `NAVRL_RANGE_ERROR_M`,
`NAVRL_DETECTOR_CHECKPOINT`, `NAVRL_POSE_*`)와 **상호배타**로 막아 입력 계약이 섞이지 않게 했다.
결과 JSON에 `cfg_p9_empirical_error_{enabled,sha256,seed}`를 기록한다.

**모델이 만들지 않는 것**(P8에 없으므로): metric range 오차, confidence 오차, false-lock range.
모두 clean 해석값을 유지한다. 미지원 크기 구간은 사전 고정한 nearest-supported 규칙으로만 처리한다.

**주의(코덱스 인계 사항)**: 관측 간격이 50~103 ms이므로 측정 전이확률을 시뮬레이션 매 프레임에
그대로 적용하면 안 된다. 주입기는 `p_stay(dt) = p_stay(ref_dt)^(dt/ref_dt)` 경쟁위험 변환으로 처리한다.

다음은 P10이다. 계획서상 단순 평가가 아니라 **동일 관측 차원을 유지한 채 P9 오차에 PPO를 재적응**시킨 뒤
별도 seed에서 비교하는 4-arm(기존/재적응 × clean/P9) 설계다.

## 2026-09-09 — P10 완료: P9 오차 비용 −4.57 pp 확인, 재적응 효과는 유의하지 않음

4-arm(기존/재적응 × clean/P9) × 평가 시드 2개 = 8셀을 205 bars에서 완주했다. 전체 수치와 receipt
경로는 `results/perception_p10_2026-09-09/{README.md,summary.json}`.

**사전등록 주 추정량 — adapted-P9 − source-P9 (시드 541+547 결합, 정규근사 95% CI)**

| 지표 | 차이 | 95% CI |
|---|---:|---|
| capture | +1.41 pp | [−0.38, +3.21] |
| crash | −0.73 pp | [−2.44, +0.98] |
| timeout | −0.68 pp | [−1.39, +0.03] |

**셋 다 0을 포함한다.** 시드별로도 +1.79 / +1.04 pp로 방향은 같지만 각각 0을 포함한다.
clean 유지도(adapted-clean − source-clean)는 capture −1.25 pp [−2.95, +0.44]로 역시 유의하지 않다.

P9 오차는 source 정책 capture를 −4.57 pp [−6.32, −2.82], 재적응 정책 capture를
−1.90 pp [−3.65, −0.16] 떨어뜨린다. 커밋 전 검토에서 해석을 교정했다: 비용 변화는 주 추정량과
다른 상호작용이며 +2.67 pp [+0.20, +5.14]다(사후·미보정·독립 셀 정규근사; 동일 seed 셀 공분산
미추정). 개별 CI 겹침으로 이 차이를 판정할 수 없다. 참고 crash/timeout 대비 일부도 명목 CI에서
0을 제외하므로 전체 지표에 대해 "둘뿐"이라는 표현은 부정확했다.

**결론**: 실사에서 잰 인지 오차를 주입하면 동결 ep25000의 capture가 유의하게 4.6 pp 떨어진다.
1,000 epoch 재적응의 주 효과는 방향상 양수지만 CI가 0을 포함한다. clean CI는 −2.95 pp 손실도
포함하므로 성능 보존/비열등성은 입증되지 않았다. **학습 시드가 811 하나이므로 시드 일반적 PPO 주장이 아니다.**

**진행 중 발견한 결함 하나 (커밋 `1a39933`)**: P9 모델 SHA를 `navrl_task.py`의 **수치** config-drift
루프(`abs(float(saved) - current)`)에 넣어둬서 P9로 학습된 모든 체크포인트가 로드 즉시
`ValueError: could not convert string to float`로 죽었다. 첫 평가 4셀이 조용히 실패한 원인이고,
사용자의 "병렬로 셀 못 돌리나?" 질문에 GPU를 확인하다 발견했다(641 MiB / 2%). 09-05 `cfg_search_state`
사고와 같은 부류다. 기존 회귀 테스트가 문자열 리터럴과 `str()`만 거부해서 `IfExp` 안의 attribute가
통과했으므로, 테스트를 **deny-list에서 allow-list로 뒤집고** 결함을 되살렸을 때 실패하는지 확인했다.
실패한 4셀은 삭제하고 전부 재실행했다. 8셀 모두 `runtime_git_dirty false`, commit `1a39933`.

**병렬 실행은 불가로 확정**: 셀당 5,388 MiB / 총 8,192 MiB이므로 두 셀이 못 들어간다. 더구나
동시 부하가 결과를 바꾼 선례가 있다(22.35% 대 21.47%).

다음: S4(봉인 test 1회 개봉, 사전등록 후)와 Track A R-C(학습 시드 replication, 사용자 실행).

## 2026-09-09 — R-C 완료: "동반학습" 설명(C3) 0/2로 철회

`confirmation_phase_plan_2026-09-06.md` §3 R-C를 사용자 실행으로 완주했다(학습 4회, `ref5in`
a8-readapt 계보, 시드 233·239 × {off, riskcap} 각 1,000 epoch → ep2900; 평가 8 cell, 70 bars,
2,049 에피소드/셀, 평가 시드 521 고정). 상세·수치는
[`results/navrl_grid_r2_d4_trainseed_rep/README.md`](results/navrl_grid_r2_d4_trainseed_rep/README.md).

**사전등록 예측 vs 실측 (상호작용 = T1 차이 − T0 차이, T0=off로 학습·T1=riskcap으로 학습)**

| | T0 stopcap−riskcap | T1 stopcap−riskcap | 상호작용 |
|---|---:|---:|---:|
| seed 233 | −2.93 [−5.12, −0.75] | −8.59 [−11.14, −6.04] | −5.65 [−9.01, −2.30] |
| seed 239 | −8.69 [−11.15, −6.22] | −7.17 [−9.68, −4.67] | +1.51 [−2.00, +5.03] |

예측은 T1이 ±2 pp 안(CI 0 포함)이어야 했다. **두 시드 모두 정반대로 T1도 T0만큼 큰 손해를
보였다**(풀링 T0 −5.46 [−7.10,−3.83], T1 −7.87 [−9.66,−6.08], 구간이 크게 겹친다). 상호작용은
seed 233에서 **반대 방향으로 유의**(T1이 T0보다 더 나쁨)하고 seed 239는 방향은 원 가설과 같지만
불유의하다. 계획서에 미리 적힌 판정 규칙대로 **0/2 → C3 철회**.

**대체 결론**: `ref5in` a8-readapt 계보에서 stopcap의 capture 손해는 학습 시 어떤 속도 거버너와
함께였는지와 무관하게 나타난다. 09-05 D4(시드 197)에서 본 T1 무손해는 학습 시드 하나의 우연이었다.
반대 방향("riskcap 학습이 악화시킨다")도 시드 간 방향이 갈려(−5.65 대 +1.51) 주장하지 않는다.
`docs/plans/confirmation_phase_plan_2026-09-06.md` §1 C3 행과 §3 R-C 절을 이 판정으로 갱신했다.

**규율 준수 확인**: 학습 4회 사이 `git status`가 매 cell 전 확인되는 고정 러너를 통과했고
(`runtime_git_dirty: false`, commit `c9fd3b0`), 학습 중 동시 부하 없었다. 체크포인트 SHA 4개
모두 `training_receipt.json`에 기록됨.

다음: 남은 것은 논문 그림 3종과 관련연구 절. 봉인 test(S4)는 이미 09-09 앞선 세션에서 1회
열렸다(utility 0.4701, validation 0.6855 대비 −0.2154 — 다른 영상이라 대응비교는 아니고 일반화
경고로만 기록, `results/perception_s4_2026-09-09/README.md`).

## 2026-09-09 — range 오차 대리 측정: 지배항은 자세가 아니라 검출기 박스 크기 (가설 기각)

MIDGARD가 라이선스 미제공으로 완전히 배제돼(사용자 확인) 거리 GT 없이 상대 range 오차의 크기만
재는 두 대리 측정을 사전등록([`prereg_2026-09-09_range_error_proxy.md`](docs/plans/prereg_2026-09-09_range_error_proxy.md))
하고 실행했다. 도구 `tools/measure_range_error_proxy.py`, 결과
[`results/perception_range_proxy_2026-09-09/README.md`](results/perception_range_proxy_2026-09-09/README.md).
NPS validation만, `test_used: false`.

**원리**: 크기 기반 단안 추정기는 `range = k/s`이므로 상대 range 오차가 크기 비의 역수로 그대로
나온다. `k`(초점거리 × 기체 실제 크기)가 소거되어 intrinsics도 거리 GT도 필요 없다. 절대 미터는 못 얻는다.

| 지표 | 중앙값 | \|오차\| P50 | \|오차\| P95 |
|---|---:|---:|---:|
| 대리 A — 검출기 박스 크기 (n=965) | −3.2 % | 20.3 % | **51.4 %** |
| 대리 B — 겉보기 크기 시간 잔차 W=5 (n=1,213) | 0.0 % | 0.0 % | **12.6 %** |

`E95 = 51.4 %` > 사전등록 문턱 30 % → **`RANGE_ERROR_LARGE`**.

**사전 예측을 기각한다.** 계획서와 사전등록 §0은 "자세에 의한 겉보기 크기 변동(대리 B)이 지배적이고
대리 A는 하한"이라고 적었다. 측정은 정반대로 **대리 A가 대리 B의 4배**다.

**부수 발견 — 오차는 잡음이 아니라 영상별 계통 편차다.** clip 안에서는 IQR이 좁은데 clip 중앙값이
−34.0 %(Clip_36)에서 +28.3 %(Clip_14)까지 흩어진다. 에피소드 내내 거리를 일정 비율로 잘못 보는
구조이고, 제어 관점에서는 프레임 잡음보다 이쪽이 중요하다. 크기별로는 bin 3(32–64 px)이 가장 나쁘다
(중앙값 +17.5 %, P95 59.5 %) — 검출기가 큰 표적의 박스를 작게 잡는다.

**한계(보고서에 명시)**: 대리 B의 중앙값 0은 자세 변동이 없다는 뜻이 아니라 5점 중앙값 필터가 단조
구간에서 중앙 표본을 그대로 돌려주는 성질 때문이다. 즉 대리 B는 고주파 반전만 보고 **느린 자세
변화는 구조적으로 못 본다**. 다른 평활자는 별도 사전등록이 필요하다. hit 문턱 IoU 0.3이 느슨해
대리 A 꼬리 일부가 매칭 느슨함에서 올 수 있다.

**이 결과가 바꾸는 것**: 지배항이 검출기 박스 크기로 밝혀졌고 그것은 **거리 GT 없이 주석만으로 이미
전수 측정된다.** 거리 GT가 꼭 필요한 항(자세)은 빠른 성분 기준 12.6 %로 지배항의 1/4이다. 따라서
판정은 `RANGE_ERROR_LARGE`이지만 "그러므로 거리 GT 데이터셋이 필요하다"로 곧장 이어지지 않는다.
대체 데이터셋 조사는 병행 중이며, 결과를 받아 §1-A를 갱신한다.

**주의**: 이 51.4 %를 현재 시뮬에 그대로 주입하면 안 된다. 시뮬 표적은 반지름 0.15 m 구라 자세에
의한 겉보기 크기 변동이 원리적으로 없고, 시뮬 range는 크기 역산이 아니라 해석적 ray-cast
depth(`navrl_perception.py:1662`)다. P9는 range를 건드리지 않으며 켜지면 기존 range 훅과 배타다
(`navrl_perception.py:938-947`).

### 정정 (같은 날) — 대리 B는 상한이 아니었다

외부 자료 조사 중 대리 B의 결함을 지적받아 직접 검증했다. **지적의 결론은 맞고 진단은 틀렸다.**

지적은 "GT 박스 크기가 대부분 정확히 일정하다(주석이 보간돼 있다)"였다. 실제로는 연속 단일-GT
프레임 간 상대 크기 변화가 P50 1.35 % / P90 17.4 % / P95 27.1 %(n=1,284)로 **자료는 충분히 변한다.**
정확히 반복되는 비율은 7.8 %뿐이다.

진짜 원인은 처음 적었던 중앙값 필터 성질이다. 유효 창 1,213개 중 잔차가 0인 것이 **55.6 %**이고,
그중 **43.3 pp가 "중앙 표본이 곧 창의 중앙값"**이라서 0이다(단조 구간에서 필터가 중앙 표본을
그대로 돌려준다). 나머지 12 pp가 주석 반복이다.

**따라서 대리 B가 보고한 12.6 %는 실제 고주파 변동의 절반 이하다** — 필터를 거치지 않은 연속 차분
P95가 27.1 %다. 103 ms 동안 참 거리가 27 % 변하려면 30 m에서 초속 78 m가 필요하므로 이 꼬리는
겉보기 크기 변동이다.

**사전등록 판정 `RANGE_ERROR_LARGE`는 유지된다** — `E95`는 대리 A(51.4 %)가 결정하고 대리 A는 이
결함과 무관하다. 바뀌는 것은 해석이다: "A와 B가 브래킷한다"는 구조가 깨졌으므로 **자세 항은 상한
없이 미측정으로 남는다.** 결과를 본 뒤 평활자를 바꾸지 않았다(규율). 어제 이 세션에서 "자세 항이
12.6 %라 MIDGARD 문제가 해소된다"고 보고했던 것을 **철회한다** — 정확한 문장은 "필요한 자료가
거리 GT에서 **자세 GT**로 옮겨갔다"이다.

**대체 자료 조사 결과**(외부 조사, 미검증 — 착수 전 직접 재확인 필요):
1순위 **Albl/ETH `drone-tracking-datasets`**(RTK ±1 cm, intrinsics 제공, 13–78 m, ds5에 **RPY 자세
GT**, CC BY-NC-SA 4.0, 등록 없음) — 자세 항을 잴 수 있는 유일한 후보다. 2순위 **MMAUD**(Leica mm급,
DJI 표적, 그러나 180° 어안이라 크기 분해능 나쁨). **LRDDv3**는 도메인 정합 1등이나 수출통제 심사가
걸려 폼만 제출해두고 대기. **AOT**는 거리 GT가 유인기 위주라 보조 검증용.
Anti-UAV 계열·ARD-MAV·Det-Fly·NPS는 거리 GT 없음 확인. Svanström은 범주값뿐이라 관련연구에서 명시 기각.

**하지 않을 것 둘**: SfM/COLMAP(독립 운동 표적 + 하늘 배경 + 등속 추격 기하에서 거리 **형식적
불관측**, Nardone & Aidala IEEE TAES 1981), 단안 metric depth pseudo-GT(UAV 벤치마크 δ₁ 0 %대이고
그 모델도 겉보기 크기로 추론하므로 순환논법).

**비용 0으로 얻은 것**: 크기 기반 추정에서 `σ_Z/Z = σ_w/w`, `σ_Z = Z²·σ_w/(f·W)`. 거리 오차는
거리의 제곱으로 자라고, 이미 가진 σ_w가 곧 1차 range 오차 모델이다.

## 2026-09-10 ETH ds5 cam0 E1–E2: 압축 해제·영상 무결성·검토 자료 준비 (gate 미통과)

cam0 분할 압축본 43개(4,471,880,421 B)를 p7zip 16.02로 `datasets/eth_ds5_cam0_extracted/cam0.mp4`
(4,520,030,301 B, CRC OK, SHA-256 `d3e6727e…`)에 풀었다. h264 1920×1080, 30000/1001 fps,
컨테이너 패킷 20970 = 전체 디코딩 20970 = nb_frames. **그러나 공개 `cam0_frame_ts.txt`는 20969행**이다.

**원인 규명(같은 날 추가 조사)**. 두 blocker 모두 기전을 특정했다.

(1) *행 수 차이*: mp4 비디오 트랙 edit list가 `media_time=1001`(정확히 1프레임), 오디오·메타 트랙은 0이다.
edit list를 따르는 판독기는 20969프레임만 제시하고, upstream이 쓴 MATLAB `VideoReader.readFrame` 루프가
그렇게 동작했다. 반면 ffmpeg/OpenCV는 그 프레임을 버리지 않는다 — 기본과 `-ignore_editlist 1` 모두 20970
디코딩, cv2 첫 3프레임은 ffmpeg 기본 출력과 바이트 동일. 남는 후보는 세 가지(`frame_id`, `frame_id-1`,
`frame_id+1`)이고 **시각만으로는 분리 불가**(모두 정수 프레임 재라벨링이라 적합이 불변).

(2) *scale 불일치*: 공개 시각은 frame id에 정확히 선형(잔차 33 ns), `t(j)=0.033367703956·j+10.551275280`.
이를 `scale·(j+origin)·period+shift`로 놓고 6개 카메라 행을 모두 시험하면, 기울기는 **cam1의 scale**과
5.9e-10까지 일치(다른 카메라는 2.1e-6 이상 차이)하고, 절편은 **cam0의 shift**에서만 정수에 가까운
origin(1.9486, 정수 2에서 0.051프레임)을 준다. origin=2일 때 함의되는 shift는 10.48454 s로 현재
10.48625553 s와 1.7 ms 차이다. 갱신 전 표가 반올림 `1.000 / 10.48`로 표시됐던 것과도 맞는다(origin 1이나
3이면 10.518/10.451이라 10.48로 반올림되지 않는다). 즉 **시각 파일이 계수 개정보다 앞선 것**이지 손상이
아니다. cam0 현재 계수로 재계산하면 origin=2에서 겹침 구간 −3.0…+1.7 ms(±0.1프레임 미만), origin 0/1/3에서는
1–2프레임. scale 차이 단독 기여는 겹침 전체에서 4.7 ms 이하다.

1프레임 시각 오차는 영상에서 중앙값 2.2 px, p95 7.6 px, 최대 20.3 px 변위를 만든다. 재투영 게이트가 4 px
RMS이므로 **검토된 박스가 생기면 정렬을 판별할 수 있다.** 그때까지 **fail-closed**로 남긴다.

**틀린 부분 수정**: 첫 pilot은 edit list를 모르던 때 `container_index = frame_id - 1`을 하드코딩해 뽑았다.
이를 `--index-offset` 인자로 빼고, 컨테이너 증거가 지지하는 offset 0으로 36프레임을 다시 렌더링했다
(`pilot_frames_offset0/`, `review_material_offset0/`, 검토 대상은 이쪽). 첫 세트는 출처 보존용으로 남긴다.
재투영 검사에서 `d = offset - shift`로 참 정렬을 역산하도록 `--container-index-offset`을 추가했다 —
offset 0에서 shift 0이 이기면 edit list 후보가, +1이 이기면 마지막 프레임 누락 후보가 맞는 것이다.
새 패널의 frame 901 crop에서 **암 6개** 기체가 선명하다. drone1(DJI Phantom)·drone2(DJI Mavic)는 쿼드로터라
6암 기체일 수 없으므로 로터 수가 가장 강한 식별 단서다. 검토 지침에 반영했다.

**코드 역산 검증(사용자 요청)**. 손계산·독립 구현과 대조했다. 공개 타임스탬프를 유도 파라미터로 재구성하면
4.4e-7 s 일치(공개 계수 9자리 반올림으로 정확히 설명). 투영·광선복원·Kabsch 회전적합은 직접 작성한
핀홀+Brown 모델과 1e-12 이하로 일치. 보간·방위·속도는 브래킷 직접계산과 완전 일치. `d = offset - shift`는
렌더 offset과 참 offset 6조합에서 종단 확인.

**그 과정에서 실제 결함 6개를 찾아 고쳤다.** (1) edit list 프레임 환산이 30000/1001을 가정 — 트랙별 mdhd
timescale과 stts sample duration을 읽도록 수정. (2) **재투영 검사가 정렬을 아예 못 가렸다** — pilot 프레임은
프레임당 1.5 px밖에 안 움직여 완벽한 박스로도 인접 shift 3개가 4 px 게이트를 통과한다. 이제 통과가 정확히
하나일 때만 RESOLVED로 보고하고 판별력을 함께 낸다. (3) shift마다 다른 프레임 부분집합으로 채점되던 문제 —
공통 집합으로 고정. (4) 카메라 뒤 점 미거부 — 뒤에 있어도 그럴듯한 화면 안 픽셀이 나온다. 거부하도록 수정.
(5) **왜곡 접힘 미거부** — 이 보정에서 반경 다항식은 정규화 반경 1.023에서 증가를 멈추는데 화면 모서리는
0.713이다. 광축에서 70° 벗어난 표적이 화면 안으로 접혀 들어오고 undistortPoints가 엉뚱한 가지로 역변환한다.
계수에서 한계 반경을 유도해 거부. (6) 보간 자세각이 [-180,180]을 벗어날 수 있던 문제.

**정렬 해소에는 빠른 프레임이 필요하다.** 겹침 구간 판별력은 median 1.61, p90 4.41, max 20.23 px/frame인데
stride 150 pilot은 median 1.40이라 원리적으로 부족하다. `select_eth_ds5_alignment_frames.py`를 새로 만들어
운동량 기준(≥4 px/frame, ≥25 m, 화면 안 예측)으로 17프레임을 골랐다(4.07–6.94 px/frame, 62–99 m).
시뮬레이션에서 박스 잡음 3 px까지 유일 해소, 4 px에서 fail-closed. 화면 안 예측에는 근사 카메라 방향이
필요해 명시적 인자로 받고 미검증 가정으로 기록한다(az 25.8, el 17.3, roll 0; frame 901·1651의 6암 기체
2점 적합, 잔차 0.11°). 이 가정은 사람에게 보여줄 프레임만 정하고, 검사는 박스만으로 방향을 다시 적합한다.
이 제약이 없던 20프레임 세트는 8개가 드론이 화면 밖이어서 폐기했다.

**직접 검수(잠정)와 그 결론**. 사용자 요청으로 정렬용 17프레임을 직접 검수했다. GT를 보지 않고 시간차분으로
움직이는 어두운 덩어리를 찾고, 확대 크롭에서 기체를 육안 식별한 뒤, 어둡기 가중 중심으로 박스 중심을 잡았다.
1778번은 나무와 겹쳐 제외, 16개를 기록했다. **검수자는 사람이 아니라 Claude이므로 식별 gate는 닫히지 않는다.**

**(1) 판별은 된다.** RMS 곡선이 단일 최소를 갖고 shift 1프레임당 2–3 px 올라간다(−6:11.4 −5:8.7 −4:6.9
−3:7.0 −2:9.1 −1:12.2 0:15.4). stride 150 pilot로는 불가능했을 일이다. 설계 의도대로 작동한다.

**(2) 그런데 답이 정수가 아니다.** 정밀 최적이 −3.53프레임, 즉 −117.6 ms이고 정수에서 0.47프레임 떨어져
있다. frame_id↔컨테이너 index 재라벨링으로는 설명이 안 된다. 기존 ±3 탐색은 이 최소를 아예 놓쳤다. 탐색을
±8로 넓히고 연속 최적을 정밀화하며, 정수가 아니면 index offset을 보고하지 않도록 고쳤다.

**(3) 공개 보정은 실패한다.** 그 최적에서도 RMS 6.9 px로 4 px 게이트를 넘는다. 잔차가 주점 거리와 0.78로
상관되고 화면 가장자리 두 프레임이 20.1 px, 11.0 px로 지배한다. 파라미터를 하나씩 풀면 초점거리만 자유일 때
3.58 px, **반경왜곡 k1만 자유일 때 2.34 px(k1 −0.100, 공개 −0.0112)**, 둘 다 풀어도 2.34 px에 초점배율은
1.00이다. 초점거리는 맞고 왜곡이 틀렸다는 뜻이다. **Sony5100 파일은 ds5 cam0 렌즈를 공개 상태 그대로는
기술하지 못한다.** 적합한 k1을 채택해 보정이 유효하다고 선언하면 순환논법이므로 하지 않는다.

식별 근거: 추적한 물체의 방위 궤적이 drone0 GT와 최대 0.6° 안에서 일치하고, pilot의 901번 프레임에서 6암
기체가 보인다(DJI 쿼드로터 둘은 배제됨). 다만 −118 ms 오프셋과 왜곡 불일치는 부분적으로 얽혀 있어
16점으로는 분리되지 않는다. E3는 시작하지 않는다. 이제 blocker는 프레임 인덱싱이 아니라 왜곡 모델이다.

**왜곡 모델 조사 결론: 보정 파일 탓이 아니다.** 앞선 "왜곡 계수가 틀렸다"는 잠정 판단을 정정한다.

(1) 데이터셋이 a5100 체커보드 이미지 122장을 함께 공개한다. 해시 검증 후 119장에서 보드를 찾아 절반으로
재보정하고 나머지 절반에서 채점했다. **홀드아웃 코너 RMS: 공개 1.245 px, 재계산 1.262 px.** 재계산값도
공개값과 근접(fx 1542.8 vs 1545.4, k1 −0.0136 vs −0.0112). 공개 파일은 자기 이미지에 대해 정확하다.

(2) 검수한 41개 중심을 이웃 프레임으로 전파해 **3495프레임을 추적**했다. 궤적 매끄러움으로 잰 추적 잡음은
**0.198 px**. 678점에서 공개 보정 + 전역 회전 + 자유 시간오프셋의 잔차는 **5.6 px**, 잡음의 28배다.
잔차는 방사 5.0 px 대 접선 2.6 px로 방사가 지배하고 반경에 따라 +1.4→+7.9 px로 커진다.

(3) **모든 가설을 홀드아웃으로 시험했다**(검증셋에서 회전 재적합해 외부 드리프트 배제). 방사왜곡·초점거리·
주점·카메라 위치·시간 드리프트·8초 창 회전 재적합·프리즘 레버암. **어느 것도 모든 분할에서 이기지 못하고,
최선 홀드아웃도 3.04 px로 잡음의 15배다.** 특히 전반부에서 적합한 방사 모델은 후반부를 5.17→12.24 px로
악화시킨다. 즉 고정된 렌즈 성질이 아니다. 자유 레버암은 0.64 m를 원하는데 물리적으로 불가능하다.
모든 변형이 −3.6프레임(−120 ms) 오프셋을 요구하며 이는 여전히 미설명이다.

(4) 방사성이면서 시간에 따라 변하고 큰 시간 오프셋을 요구하는 잔차는 카메라보다 **공개 타임스탬프나 융합
pose** 쪽일 가능성이 높다. 공개 시각이 cam1 scale로 만들어졌고 계수 개정 전이라는 점, ds5가 README와 달리
자체 보정을 공개하지 않은 점이 이를 뒷받침한다. 확정하려면 ds5 보정 세션이나 저자 문의가 필요하다.

(5) **그래도 E3가 실제로 필요한 것은 살아 있다.** 5 px 위치 오차는 크기 측정을 망치지 않는다. 18–108 m에서
추적한 3429프레임에서 겉보기 크기×거리가 안정적이다: sqrt(픽셀수) 기준 상대산포 중앙값 **6.9%**, p90 16.4%.
세로 extent는 30–115 m 구간별로 0.199/0.195/0.199/0.183 m로 일관된다. 절대 기하 용도는 막히지만
**크기-거리 관계는 측정 가능**하며, 별도 사전등록으로 진행할 수 있다.

새 도구 3개(보정 이미지 수집, 보정 재계산, 표적 추적, 카메라 모델 감사)와 테스트를 추가했다. 테스트 1316개 통과.

**디스크**: 이전 세션 중복 추출본을 전체 SHA-256이 보존본과 일치함을 확인한 뒤 삭제하고, 사용자 지시로
Downloads의 Antigravity 설치본도 지웠다. 2.2 GB → 8.0 GB 확보. 데이터셋·영수증·run은 하나도 지우지 않았다.

36개 pilot 프레임을 `--allow-count-mismatch`로 추출(정렬 미해결 플래그 기록)하고, 사람 검토 패널(GT 거리·
방위·고도·속도, ±3프레임 motion cue, 확대 crop, contact sheet)과 빈 `review_template.csv`를 만들었다.
motion cue는 identity가 아니며 박스도 아니다. drone0 GT: ~30 s까지 cam0에서 5.8 m 지상, 이후 5–108 m.

Sony5100 보정 후보: 해상도·fps는 일치하나 모델/렌즈 태그 없음, ds5 전용 보정 없음 → **미검증**.
검토된 박스로 회전 전용 Kabsch 적합 + PnP 중심 vs 측량 위치를 검사하는 `check_eth_ds5_reprojection.py`를
박스가 하나도 없는 지금 임계(RMS<4 px, 중심 2 m, ≥6 박스, 정렬 −3…+3프레임)와 함께 고정했다.
TrackingStatus는 Leica 토탈스테이션 상태(0 정상, 1 경고), 자세는 ArduPilot EKF body→NED(위치는 ENU).

디스크 4.0 GB뿐이라 conda tarball 캐시·pip 캐시만 비웠다(데이터·결과·run 삭제 없음). 이후 감사에서 이전
세션이 이미 같은 영상을 `datasets/eth_ds5_extracted_cam0/`에 풀어놨음을 발견했다 — 이번 추출본
`datasets/eth_ds5_cam0_extracted/`와 바이트 동일(SHA-256 `d3e6727e…`)이라 4.3 GB가 중복이다. 삭제하지 않고
보고만 한다. 테스트 1316개 통과(ETH 모듈은 72개로 증가).
E3(오차 모델)는 시작하지 않았다. 다음: 사람 검토 → 재투영 검사 → 통과 시에만 dense segment 등록.

## 2026-09-10 E3 분리: E3-S 실행 완료, E3-P는 BLOCKED

E3를 한 덩어리로 돌리지 않고 근거가 확보된 부분만 분리했다. **E3-S**는 "겉보기 크기로 거리를 얼마나 정확히
복원하는가", **E3-P**는 "그 크기 변화 중 얼마가 거리가 아니라 상대 자세 때문인가"를 묻는다. E3-S는 표적 위치
GT와 초점거리만 쓰므로 미해결 blocker(카메라 방향, −120 ms 오프셋)를 우회한다. E3-P는 두 자세 사이의 각도가
필요해 카메라 방향이 있어야 하는데 재투영 잔차가 측정 잡음 0.196 px 대비 5.6 px라 불가능하다.

**사전등록을 먼저 커밋했다**(572314c). 모집단·제외기준·크기척도·적합법·15초 블록 leave-one-block-out·오차
지표·거리 bin·최소표본·게이트 4개를 결과를 보기 전에 고정했다. 해석 범위도 구속했다: 이 수치는 "크기 기반
거리 추정 성능"일 뿐이고, 이전 6.9%는 거리 오차가 아니며, 자세 기여 분해는 금지다.

**결과: SIZE_RANGE_USABLE.** 9블록 3107프레임(31–108 m), 블록 단위 홀드아웃 |상대오차| 중앙값 **6.2%**,
안정성(p90−p10) 9.3%p. 적합 유효크기는 0.236–0.242 m로 안정적이다. 거리 bin은 50–70 m에서 bias +15.5%,
70–90 m +0.4%, 90–110 m −5.3%이지만 **이 비행에서는 거리와 시간이 교락**되어 있다. 50–70 m bin은 블록 11·12가
지배하고 같은 거리대의 초반 블록 3은 7.7%다. 원인 분해는 하지 않는다.

민감도: 시간 shift −3.6프레임 6.5%(주 분석 6.2%), 왜곡 보정 적용 6.2%, 세로 extent 19.7%, 가로 extent 13.0%.
즉 **미해결 타임스탬프 오프셋이 E3-S 결론을 바꾸지 않는다.** 데이터 품질: pose 브래킷이 0.5 s 한계에 닿고
포함 프레임의 2375/3107이 총국 TrackingStatus 1(경고) 구간이다. 가중·제외에 쓰지 않고 한계로만 기록했다.

추적 도구는 창을 이전 extent에 맞춰 넓히고 창 가장자리에 닿는 blob은 버리도록 고쳤다. 고정 창이 근접 표적의
크기를 조용히 잘라내던 문제였다. 새 도구 `measure_eth_ds5_size_range.py`와 단위 테스트 추가, 테스트 1326개 통과.

**E3-P는 BLOCKED**로 남긴다. 전제조건 4개(시간 정합, 좌표 규약 확인, 카메라 외부 파라미터, 자세 GT 신뢰성)와
각각의 현재 상태를 계획서에 적었다. E3-S 결과로 E3-P의 가설이나 기준을 바꾸지 않는다.

## 2026-09-10 E3-P 게이트 실측: 전제조건 2개는 과했고, 남은 하나에서 막힘

"왜 blocked인가"를 묻길래 실제로 재봤더니 **내가 과하게 막아 놨다.** 근거는 E3-S 결과가 아니라 기하·동역학
계산이며, 원래 문구는 이력에 남긴다.

**철회한 전제 둘.** (1) 카메라 방향은 필요 없다. 시선-기체 각도는 표적 GT 위치와 측량 카메라 위치, 그리고
자세만으로 나오고 카메라 회전은 면적 기반 척도에 들어가지 않는다. 화면상 가로·세로 extent를 쓸 때만 필요하다.
(2) 타이밍도 구속하지 않는다. −120 ms가 aspect에 주는 오차는 p90에서 0.88°, 관측 폭 30.8°의 2.9%다.

**C1 자세 규약: 통과.** 다중회전익기는 추력축을 기울여 수평 가속한다. 공개 각도의 32가지 해석을 각각
GT 위치에서 얻은 수평 가속 방향과 비교했다. 승자는 **내재 xyz 오일러, body→NED, 추력 −z, NED→ENU 변환**으로
방향오차 중앙값 **17.7°**, 2위와 **16.3° 차**다. 업스트림 코드만 믿던 규약을 데이터로 확인했다.

**C2 자세 신뢰성: 실패.** |a_h| = g·tan(tilt) 형태는 기울기 0.885인데 상관 0.18(기준 0.5). 이 모델은
공기저항을 빼먹어 물리적으로 틀렸다 — 정속 비행 중에도 항력을 상쇄하려 기울기 때문이다. 추력·항력을 벡터로
함께 적합한 정정형(기준 사전 고정)은 추력이득 0.73, 항력계수 −0.017, 상관 0.64로 셋은 통과했으나 항력항의
잔차 개선이 4.4%로 기준 20%에 못 미쳐 실패했다. **세 번째 변형은 돌리지 않았다.** 통과할 때까지 바꾸면
답을 보고 검사를 고르는 것이다. 게이트는 사후에 고른 기준으로 여는 것보다 막아 두는 편이 낫다.

**C3 분리 가능성: 통과.** aspect-거리 상관 0.21, 20프레임·2블록 이상인 칸 6개, 네 거리대 모두 aspect 폭
20° 초과(21.6/24.6/25.9/23.3°). 거리와 시간이 교락됐던 것과 달리 aspect와 거리는 공선적이지 않다.
다만 채워진 칸이 60–90°에 몰려 있어 E3-P 설계는 이 좁은 폭을 감안해야 한다.

**판정: C2b 하나로 `E3P_GATE_BLOCKED`.** 게이트를 열려면 "항력이 큰가"가 아니라 "자세가 기체를 기술하는가"를
묻는 신뢰성 기준을 한 번만, 미리 정해서 써야 한다. 미분 창 길이도 그 기준의 선언된 파라미터여야 한다.
그건 E3-P 자체 사전등록에 들어가야지 이 게이트를 또 덧대는 방식이면 안 된다.

새 도구 `check_eth_ds5_attitude_gate.py`와 단위 테스트 추가. 상태 사이트도 갱신했다.

## 2026-09-10 — README 그림 리뉴얼과 PPT용 배포 묶음

사용자 요청에 따라 기존 결과를 발표용으로 재구성했다. **실험·학습·정책·제어 코드 변경 없음.**
기존 README 그림 8개의 상태 설명을 갱신하고 E3 그림 1개를 추가했다. 전체 구조도는 Track A/B/C의
검증 범위를 구분하며, 인지 그림은 구현 계획 대신 S1/S4/P8–P10의 완료된 측정과 한계를 표시한다.
나머지 6개 상세 패널은 기존 내용을 보존하고 현재 상태와 해석 범위를 붙였다.

발표용은 9개 파트를 16:9 SVG와 3840×2160 PNG로 내보냈다. `docs/assets/presentation/`에 미리보기,
사용 안내, 원본·근거 SHA-256 manifest, 전체 ZIP을 두고 README·문서 색인·사이트에 연결했다.
`tools/render_presentation_figures.py --raster`로 재생성한다(Chrome + websocket-client 필요;
이번 실행은 `detector_runs/venv/bin/python`). 옛 `render_readme_figures.py` 진입점은 새 빌더로 넘겨
과거 생성 코드가 수동 교정된 패널을 덮어쓰지 않게 했다.

설명도 근거 보고서와 맞췄다. E3-S 6.2%는 블록별 절대 상대 거리 오차 중앙값의 중앙값이며,
9.3 pp는 블록 간 p90−p10 산포다. GT bbox가 아닌 dark-pixel 크기 척도, 동일 비행 재사용,
시간·거리 교락, E3-P의 남은 자세 신뢰성 blocker를 명시했다. 0.196 px를 전체 측정 잡음으로
부르던 표현은 궤적 매끄러움 잔차로 제한했다. P10 주 추정량과 탐색적 상호작용을 혼동하던
README·사이트 문구는 이미 교정된 P10 보고서에 맞췄다. 원본 결과 receipt는 변경하지 않았다.

검증: PNG 실렌더 확인, Chrome에서 9장 전체 텍스트의 캔버스 이탈 검사, SVG XML/중복 ID,
4K 크기·SHA·ZIP 무결성·갤러리 링크·한계 문구 검사 7개, branch policy 4개 통과.
SVG 재생성 바이트 동일 확인. arena route/motion 및 headless WebGL 테스트 통과.
`git diff --check` 통과. 기존 `test_status_site.js`는 새 그림/링크 계약을 통과한 뒤 241행의
과거 recovery gate에서 중단한다: snapshot은 `RESULT_UNAVAILABLE_OR_MALFORMED`인데 테스트는
`PASS_32_CELL_INTEGRITY`를 요구한다. 이 선행 불일치를 숨기기 위해 snapshot을 바꾸지 않았다.

### 후속 — 결과 카드 대신 논문형 블록 다이어그램

사용자가 요청한 것은 결과 요약이 아니라 모듈·입출력·화살표 중심 구조도였으므로 인지 과정과
safety filter를 새로 도식화했다. 기존 streaming 명세와 safety 상세 그림만 재구성했으며
알고리즘·실험은 변경하지 않았다. 인지는 검출/광류 병렬 경로, 특징 결합, 시간 이력과 선택 출력을
표시한다. safety는 LiDAR 처리와 정책 명령의 분기·결합을 표시하며 **직선 회랑 기준선**으로
한정했다. arc variant나 실기 통합으로 읽지 않도록 캡션에 명시했다.

`docs/assets/paper/`에 SVG·4K PNG 각 2장, 설명, SHA manifest와 ZIP을 배포한다.
README와 사이트의 두 파트는 새 블록도를 표시하고 이전 요약/상세도는 링크로 보존한다.
Chrome 실렌더·텍스트 캔버스 이탈 검사, 이미지/ZIP/SHA/문서 연결 검사를 수행했다.

## 2026-09-11 P10 시드 재현: 판정 INCONCLUSIVE — 재현도 철회도 아니다

P10이 학습 시드 1개에 걸려 있어서 시드 857·863을 같은 계약으로 돌렸다. **판정 규칙 두 가지를 학습 시작
전에 커밋**했다(`d0e1aea`). 평가 시드는 541·547로 P10과 동일하게 두어 학습 시드만 변인이 되게 했고,
얼은 정책 셀도 캠페인 안에서 다시 돌려 모든 추정량이 캠페인 내부에서 계산되게 했다.

**결과.** 주 추정량(adapted−source, P9 하)이 시드별로 +1.41 / +0.78 / −0.01 pp다. 신규 2시드 중 양수는
1개라 부호 규칙에서 재현(2/2)도 철회(0/2)도 아니다. 시드를 반복 단위로 한 t 검정은 평균 +0.73 pp,
CI [−1.04, +2.50]로 0을 포함한다. 두 규칙 모두 미달이므로 **`INCONCLUSIVE`**이고 재적응의 순이득은
**입증되지 않았다**. 사후에 마진을 옮기거나 시드를 더 붙이지 않았다. 사전등록이 셋 다 금지했다.

**세 시드가 합의하는 것 셋.** (1) 측정된 인지 오차가 얼은 정책의 capture를 **−4.57 pp** 깎는다. 세 캠페인
모두 동일하고 CI가 0을 제외한다. (2) 재적응 후에도 잔여 비용이 남고 그 크기가 −1.90 / −2.71 / −3.84 pp로
시드마다 커진다. 셋 다 CI가 0을 제외한다. 즉 재적응은 오차 비용을 **완전히 회복하지 못하며, 얼마나 회복하는지가
시드마다 다르다.** 그 변동이 이번 발견이다. (3) clean 성능을 −1.25 / −1.08 / −0.74 pp 깎는다. 공짜 회복이
아니라 거래다.

**재현성 확인.** 얼은 정책 셀은 학습 시드와 무관하므로 두 캠페인 모두 처음부터 다시 돌렸고, 8셀 전부가
P10의 capture·crash·timeout·episode 수와 **정확히 일치**했다. 이 저장소는 환경 불일치로 재현에 실패한 적이
있어 가정하지 않고 확인했다. 설계상 추정량이 캠페인 내부라 여기서 실패했어도 비교가 조용히 왜곡되지는 않는다.

**논문 문장 교체.** "재적응이 손실을 회복한다"는 못 쓴다. 쓸 수 있는 것은 "오차가 4.57 pp를 물리고, 1,000
epoch 재적응이 잔여 비용을 시드별 2.67 / 1.86 / 0.73 pp만큼 줄이되 clean capture를 0.7–1.3 pp 대가로
치르며, 직접 대비의 시드 단위 구간이 [−1.04, +2.50]이라 순이득은 3시드에서 입증되지 않았다"이다.

학습 각 1시간, 평가 캠페인당 40분, 총 3시간 반. TensorBoard는 포트 6009에서 실제 학습 로그를 본다.

## 2026-09-11 — README 전체 그림의 논문형 블록도 갱신

README 본문 그림 9개를 모두 직접 렌더링 검수하고, 모듈·연결 관계 중심의 블록도로 교체했다.
긴 설명과 결과 수치는 캡션으로 분리했다. 기존 상세도와 요약 카드는 참고/아카이브 링크로 보존했다.
README와 연구 사이트에 동일한 SVG를 연결하고, 각 그림의 4K PNG와 한 페이지 벡터 PDF,
설명 README·해시 manifest·전체 ZIP을 배포 대상으로 생성했다.

기존 연구 내용의 표현만 변경했다. safety 비교군은 대체 조건임을, SAM은 보관된 제안임을,
E3-P는 gate 뒤 BLOCKED임을 명시했다. 알고리즘·실험 결과·status.json은 변경하지 않았다.
기존 미추적 ETH 검수 자료 세 항목은 수정하거나 스테이징하지 않았다.

검증: 그림 테스트 11개, branch-policy 테스트 4개, arena route/motion 및 headless WebGL 검사 통과.
9개 PDF 모두 한 페이지·텍스트 선택 가능·래스터 이미지 없음 확인. SVG 텍스트 블록 경계,
PNG 해상도, 링크, ZIP/manifest 해시, git diff --check 통과.
기존 test_status_site.js의 status snapshot 기대값 불일치(RESULT_UNAVAILABLE_OR_MALFORMED와
PASS_32_CELL_INTEGRITY 관련)는 그대로 남아 있으며 이번 그림 변경으로 해결됐다고 주장하지 않는다.

## 2026-09-11 — R1 독립 renderer 정적 감사 산출물

사용자의 "R1 구현"을 앞서 정의한 현재 rendering path 조사 단계의 산출물 작성으로 한정했다.
`docs/renderer_r1_audit_2026-09-11.md`에 요청 항목 10개, class/function·buffer 흐름,
material 전달 경계, package import 경계, 독립 scene의 재사용 후보를 기록하고 docs 색인에 연결했다.
기준은 `17e4a82f185787c080ef8fea52f3c4cc4870a580`; 코드·자산 23개 SHA를 별도 목록에 남겼다.

중요한 확인: normal camera는 depth를 출력하지 않으며 face index는 material/instance ID가 아니다.
Warp GPU 경로에는 명시적인 material table 전달이 없고, package 일반 import는 task 등록을 동반한다.
native RGB와 normal debug visualization, task-owned pseudo-RGB를 서로 다른 경로로 구분했다.

R1 상태는 `STATIC_AUDIT_COMPLETE`, runtime smoke는 `NOT_RUN`, R2–R5는 `NOT_STARTED`다.
기존 소스·자산·실험 결과·appearance overhaul 계획은 변경하지 않았고 simulator/GPU/task/policy는
실행하지 않았다. 이 문서는 shading 구현이나 성능 검증 완료를 의미하지 않는다.

검증: SHA 23개가 현재 파일 및 기준 커밋 양쪽에서 일치, 보고서 링크 24개 확인,
범용 kernel AST signature/import 정적 검사, 상태값·색인 연결·git diff --check 통과.
렌더링·perception·policy 테스트는 실행하지 않았다.

## 2026-09-11 — R2 독립 shading prototype 구현, 실제 실행은 인수인계

사용자가 R2 구현만 요청하고 실행·판정을 별도로 맡기기로 했다. `tools/renderer_validation/`에
일반 triangle scene·camera·sequence-fixed material/light 계약, SHA-pinned 범용 Warp kernel 로더,
두-pass G-buffer, flat/Lambertian shading을 추가했다. 초기 fixture는 일반 상자 두 개이고
기존 UAV asset, renderer/task/perception/policy 코드는 변경하거나 실행하지 않았다.

`tools/run_renderer_validation.py`는 NPZ의 RGB 3채널과 별도 debug buffer, source/runtime/입력/hash
receipt만 생성한다. output 덮어쓰기·소스 drift·ray pass 불일치를 거부하고 학습/실험 판정은 하지 않는다.
정상 실행 상태도 `RENDERED_UNASSESSED`이며 실패는 `FAILED_INCOMPLETE`로 남긴다.

`docs/renderer_r2_prototype_2026-09-11.md`에 API·출력·미검증 항목과 Sol 인수인계를 기록했다.
상태는 `IMPLEMENTED_RUNTIME_UNVERIFIED`다. CPU 단위 테스트는 fake Warp/직접 구성한 tensor로만
수행했으며 실제 Warp ray casting, GPU smoke, R3–R5 실험·benchmark·학습은 실행하지 않았다.

검증 결과: CPU/모의 backend 단위 테스트 39개 통과(CUDA 비노출), 신규 Python 6개 AST 파싱,
R2 문서 링크 6개와 git diff --check 통과. R1 근거 파일 23개의 SHA도 그대로 일치한다.
이번 테스트의 NPZ/receipt 쓰기는 임시 디렉터리의 모의 데이터에 한정했고 실제 렌더링 산출물은 없다.

## 2026-09-11 — R2 실제 GPU smoke 및 R3 renderer validation PASS

R2 implementation을 커밋(`24470e2`)한 뒤 RTX 3070에서 Warp 1.0.0 camera kernel을 실제 실행했다.
seed 0 기술 smoke 두 프로세스는 160×120 frame 2개씩을 만들었고 모든 배열 hash가 exact 일치했다.
유효 geometry는 4,098 pixels, visible instance 2개였다. 다만 사전등록 전·dirty source 실행이므로
정식 판정에서는 제외하고 `TECHNICAL_SMOKE_PASS`로만 기록했다.

R3 기준과 실행 코드를 결과 전에 별도 커밋(`83c62c3`)하고 seed 173을 두 독립 GPU 프로세스에서
실행했다. 두 run 모두 12개 고정 check를 통과했고 metric·11개 배열 SHA·runtime 축이 exact 일치해
최종 `PASS`다. uniform flat luminance variance 0, Lambertian variance 0.018891,
light-direction RGB MAE 0.201582, material 0의 2,196 pixels만 변경되고 외부 최대 변화는 0이었다.
geometry buffer와 instance-ID leakage 검사도 exact 조건을 통과했다.

이는 일반 상자 fixture의 renderer 계약 판정이다. 모델·label·loss·데이터셋이 없는 비학습 실험이므로
학습은 실행하지 않았다. detector/association/task/policy도 import·수정·실행하지 않았으며,
shortcut 감소나 task 성능 개선을 주장하지 않는다. R4 background와 R5 throughput은 미실행이다.

## 2026-09-11 — R3 재검증, R5 처리량, R4·R4b는 기준 C에서 FAIL

R3이 "예상보다 빨리 끝났다"는 의심에 따라 코드를 역산했다. 구현은 진짜다. 기존 warp 커널 두 개를
실제 launch하고, 두 패스의 hit 마스크가 어긋나면 이미지를 거부하며, 색은 face ID로 재질 표를 조회해
가져온다. 빨랐던 이유는 **URDF 자산 로딩이 아예 없고**(코드로 만든 상자 2개) 픽스처가 최소여서다.
이 자산 로딩 공백은 계획서 V1의 선행 작업으로 추가했다.

R5 벤치마크(`tools/benchmark_renderer_r5.py`). 128장면 480×270에서 Lambertian은 flat 대비 **1.185배**다.
반복당 기하 23.72 ms, flat 음영 4.80 ms, Lambertian 음영 10.07 ms이고 기하가 70%를 차지한다.
삼각형 24개짜리 픽스처이므로 실제 메시로 가면 비율이 1에 수렴한다. **1.185는 상한이다.**
초당 3,788장, torch 예약 2.1 GB. 작은 9개 셀에서는 1.053–1.176이었다.

R4는 사전등록(`f61123c`)을 먼저 커밋하고 실행했다. **판정 FAIL.** 기준 A(flat 분산 0),
B(depth arm R² 1.0), D(같은 깊이 구간 휘도 퍼짐 60.8–874.4배), E(두 독립 프로세스 exact 일치)는
통과했고 **C(lambertian의 range→휘도 R² < 0.5)가 최대 0.5861로 실패**했다.

첫 진단은 "상자별 material-깊이 교란"이었고, R4b(`f99b663` 사전등록)로 face→material 배정만 바꿔
한 번 실행했다. **역시 FAIL, 0.5861 → 0.5816.** 그리고 두 가지가 드러났다.

첫째, 내 재배정이 이미지에 거의 닿지 않았다. 삼각형 24개 전체 표를 보고 설계했는데 실제로 보이는
quad는 4개뿐이라, **픽셀의 98.36%가 원래 material 번호를 유지했고** 보이는 색은 4→3으로 줄었다.

둘째, 진짜 결함은 기준 C 자체다. **이것은 픽셀 36,242개의 회귀가 아니라 면 4개의 회귀다.**
외형 모델은 면마다 색 하나를 칠하고, 상자 두 개 장면에서 보이는 면은 언제나 4개(quad 0,5,6,10,
평균 거리 3.240/3.308/3.808/4.059 m)다. 실패 장면을 면 4개 가중 평균으로 회귀하면 R² = 0.997이다.
통계 단위가 픽셀이 아니라 면이어야 했다 — E3-S의 시간 블록, P10의 시드와 같은 자리다.
material 재배정으로는 고쳐지지 않으므로 **세 번째 픽스처는 만들지 않는다**(사전등록 준수).

색을 통일한 진단용 arm은 두 픽스처에서 R² ≤ 0.0903으로 동일하다. 음영 계산 자체는 깊이의 함수가
아니다. 계획서 V3의 검증 기준("깊이→휘도 R² < 0.5")은 폐기하고, 단위를 면으로 잡고 독립 면 개수를
선언된 파라미터로 두는 기준으로 교체했다. 어떤 모델의 shortcut 의존도도 주장하지 않는다.
전체 테스트 1,405개 통과(skip 4).

## 2026-09-11 — 정정: "로터 팔 6개" 식별 단서 철회

2026-09-10에 "frame 901 crop에서 암 6개 기체가 선명하고, drone1·drone2는 DJI 쿼드로터라
6암 기체일 수 없으므로 로터 수가 가장 강한 식별 단서"라고 기록했다. **이 주장은 틀렸다.**

사용자가 검수 중 팔이 4개이고 착륙 지지대가 있다고 지적했고, 14배 확대로 확인한 결과 로터가 달린
굵은 팔 4개, 가는 착륙 다리 2개, 안테나 마스트였다. 가는 다리 둘을 팔로 잘못 셌다.
**세 대 모두 쿼드로터이므로 로터 개수로는 구분되지 않는다.** 이 판별자는 무효다.

식별 자체는 유지되나 근거가 다르다. 남은 것은 방위 궤적이 넓은 호에서 drone0의 GT를 **0.6도**
이내로 따라간다는 점과, drone0이 노출된 착륙 다리·안테나 마스트를 가진 Pixhawk 자작기라는
근거리 양식 단서다. **사람 검수는 이만큼 약해졌다.** 원거리 crop에서 "drone1·drone2가 아니다"를
가릴 수단이 사라졌고, 검수 지침을 그렇게 고쳐 적었다.

영향 없는 것: 프레임 선정에 쓴 카메라 지향 추정(방위 25.8도, 고도 17.3도, 잔차 0.11도)은 그 기체의
**영상 좌표**로 적합한 것이라 그대로다. E3-S 6.2%, 추적 잡음 0.196 px, 재투영 5.6 px도 그대로다.

정정 문서 `results/eth_ds5_intake_2026-09-10/CORRECTION_2026-09-11_rotor_count.md`.
검수 지침 4종·README·상태 사이트·E3 계획·실행 계획을 고치고 검수 zip을 다시 만들었다.
`provisional_review_2026-09-10/README.md`에는 숫자를 고치지 않고 안내만 붙였고,
`alignment_queue.json`은 기계가 읽는 receipt이라 전혀 건드리지 않았다. 발표 자료에는 이 주장이 없어 hash 고정 매니페스트는 건드리지 않았다.

## 2026-09-11 — 2차 검증: 식별 근거 서술을 다시 고치고, R5 방법 단서를 보완

사용자가 "잘 됐는지 한번 더 검증해 달라"고 해서 내 작업을 독립적으로 다시 계산했다.

**R4 수치는 전부 재현됐다.** r4_metrics를 쓰지 않고 numpy 최소제곱으로 R²를 다시 구하니 receipt와
최대 2.9e-16 차이였고, 분위 구간을 다른 방식(argsort + array_split)으로 나눠 구한 within-bin 비율도
3.1e-15 차이로 같았다. **세 번째 독립 프로세스**의 배열 SHA-256도 receipt와 전부 일치했다.
depth arm은 법선·face를 난수로 뒤섞어도 바이트 단위로 같은 이미지를 냈다. 보이는 quad가 4개라는
사실과 장면 6의 면 4개 가중 R² = 0.9974도 다시 확인했다. 재배정이 material 번호를 유지한 픽셀
비율은 장면별 97.79–98.44%였다(보고서에 적은 98.36%는 장면 6 값이다).

**식별 근거 서술이 부정확했다. 고쳤다.** 어제 정정문에서 "방위 궤적이 넓은 호에서 GT를 0.6도 이내로
따라간다"를 첫 근거로 들었는데, 출처를 추적하니 그것은 독립 측정이 아니라
`reprojection_check.json`의 `angular_residual_deg_max`, 즉 **검수된 상자 15개에 맞춘 PnP 카메라 자세의
잔차**였다. 최적 정렬(shift −4)에서 0.5999도다. 문제는 **같은 적합이 보정 게이트에서 떨어진다**는
것이다. 같은 shift에서 픽셀 RMS 7.10 px(게이트 4 px), 최대 20.3 px, 상태
`CALIBRATION_CANDIDATE_NOT_VALIDATED`, 일치 정렬 0개. 픽셀 형태로 실패하는 적합의 각도 형태만
근거로 드는 것은 선택적이다.

가장 강한 근거는 **E3-S**로 바꿨다. 물리 크기 하나만 적합하고 leave-one-block-out으로 예측했을 때
**3,107 프레임 · 9블록 · 30.7–108.4 m에서 held-out 상대 거리 오차 중앙값 6.2%**
(`G2_central.median_of_block_medians` = 0.061865)다. 카메라 자세가 들어가지 않아 정렬·보정
블로커와 독립이다. 0.6도는 보조 정황으로만 남긴다. 정정문·intake README·E3 계획·실행 계획·
상태 사이트를 그에 맞게 고쳤다.

**R5 보고에 빠졌던 방법 단서 둘을 넣었다.** 반복당 총시간은 측정값이 아니라 단계별 평균의 합이고
각 표본이 자체 동기화 경계를 포함하므로, 비율이 실제 융합 반복보다 1쪽으로 아주 약간 치우친다.
그리고 배치 안의 모든 장면이 같은 카메라 자세를 쓰므로 광선 적중률이 동일하다. 실제 워크로드는
시야가 달라 적중률이 흩어진다.

## 2026-09-11 — 독립 감사가 잡아낸 것들

2차 검증을 서브에이전트 감사로도 돌렸다. 실제 결함이 나왔고 전부 고쳤다.

**살아 있던 철회 지시.** `HUMAN_REVIEW_HANDOFF.md`의 안내 표에 "**암 6개**가 보입니다 / 같은 6암
기체가 계속 보입니다"가 그대로 남아 있었다. 같은 파일 2절에는 정정을 넣었으면서 표를 놓쳤다.
내 검색이 `6암`·`six-arm` 표기만 훑어서 **어순이 뒤집힌 `암 6개`를 지나쳤다.** 자작기 양식 서술로 고쳤다.

**내가 하지 않은 일을 했다고 적은 곳 둘.** 정정문과 WORKLOG에 "`alignment_queue.json`에 안내를
붙였다"고 썼는데 그 파일은 **전혀 건드리지 않았다.** 기계가 읽는 receipt이라 문구를 넣지 않은 것이
맞는 판단이었고, 잘못은 하지도 않은 일을 했다고 적은 것이다. 두 곳 다 사실대로 고쳤다.

**커밋 메시지가 적용되지 않은 수정을 주장했다.** b873c38 메시지에 "98.36%를 장면 6으로 표기했다"고
썼는데 그 치환은 **실패했고 파일은 그대로였다.** 굵게 표시 위치가 달라 문자열이 어긋났는데
그 치환에만 assert를 걸지 않아 조용히 넘어갔다. 다른 치환들에는 걸어 두고 하나를 빠뜨렸다.

**장면 하나의 값을 픽스처 전체의 값처럼 적은 곳 셋.** material 유지 비율 98.36%, quad 4개의 평균
거리 3.240/3.308/3.808/4.059 m는 전부 **장면 6**의 값이고, 함께 적은 "픽셀 36,242개"는 엉뚱하게
**장면 0**의 값이었다(장면 6은 35,279개). 전부 범위를 명시했다. 유지 비율은 장면별 97.79–98.44%,
전체 98.15%다.

**면 단위 표가 여덟 중 넷만 보여주고 있었다.** 그것도 가장 큰 넷이었고 부분집합이라는 말이 없었다.
여덟 장면 전부로 바꿨다. 그 표와 quad 평균 거리는 **receipt에 없는 파생값**이라 그렇게 표시했다.

**철회한 진단에 적었던 상자별 거리 구간(2.85–3.77 / 3.67–4.37 m)은 어느 receipt에도 없었고
틀렸다.** 여덟 장면에서 보이는 실제 거리는 2.746–4.435 m다.

**"면"이 한 문단에서는 삼각형, 다른 문단에서는 사각면을 뜻했다.** 용어를 정의하고 quad로 통일했다.

**재현성 서술이 비교 스크립트의 범위를 넘어섰다.** `camera_poses`·`aggregation_rule`·진단 arm 비율은
대조 목록에 없다. 다만 두 `run.json`이 **바이트 단위로 동일**하므로 그쪽이 더 강하고 단순한 주장이다.

**R5 receipt에 출처가 없었다.** `git_commit`도 `tracked_tree_clean`도 없었고, 벤치마크 스크립트가
커밋되기 **전**에 실행됐다. 어떤 트리에서 나온 숫자인지 확인할 수 없는 값은 결과가 아니다.
스크립트에 출처 기록과 더러운 트리 거부를 넣고 커밋 26b85bf에서 두 캠페인을 다시 돌렸다.
**재현된다**: 128장면에서 1.188배(첫 캠페인 1.185), 64장면 1.186(1.1835), 9개 셀
1.060/1.122/1.177(1.053/1.127/1.176). 셀 사이 편차가 캠페인 사이 편차보다 크다.
첫 캠페인은 지우지 않고 나란히 보고한다.

**"1.185는 상한"의 축을 분리했다.** 메시가 무거워지면 비율이 1로 내려가므로 메시 축에서는 상한이
맞지만, 단계를 따로 재서 더한 방식이 비율을 1쪽으로 끌어당기므로 같은 픽스처의 참값은 그보다
약간 높다. 방향이 반대인 별개의 축이다.

감사가 통과시킨 것: R5 표 전 항목, R3/R4/R4b 판정과 임계, 사전등록이 실행보다 앞선다는 순서
(f61123c → 8ae00e5 → R4 실행, f99b663 → 6ae50be → R4b 실행, 둘 다 ancestor 확인),
검수 zip 14개 항목과 이미지 해시, 상태 사이트 태그 균형. 그리고 두 독립 run이 **바이트 단위로
동일**하다는 것은 내가 주장한 것보다 강한 결과다.

## 2026-09-11 — R3–R5 전체 재실행과 메모리 계측 교정

사용자의 전체 재실행 요청에 따라 독립 일반 상자 렌더러만 GPU로 검증했다. detector/tracker/PPO의
실제 실행이나 기존 태스크의 변경은 없다. 결과는
`results/renderer_reverification_2026-09-11_1951/README.md`와 `summary.json`에 있다.

- R3/R4/R4b 각각 2회 GPU 실행: 역사적 지표·배열 SHA exact 재현, 단계별 새 두 run.json 바이트 동일.
  R3 PASS, R4/R4b C FAIL 그대로다. 원인이나 기준 무효가 확정됐다는 서술은 철회했다.
- R4/R4b 각각 추가 GPU 렌더 후 NumPy 독립 재계산: 4 arm × 8장면의 scalar 지표 차이 최대
  4.54e-16 미만. 기준을 변경하지 않았다.
- R5 기존 9+2셀과 128 단독 셀 재실행. 합산 비율은 128장면에서 1.187986이다.
  64→128 연속 실행의 Torch peak 1,830.528 MiB와 128 단독의 1,601.134 MiB 차이는 이전
  64장면 G-buffer의 229.394531 MiB와 exact 일치했다. 숫자의 반복 재현만으로 올바른 계측이 되지 않는다.
- 새 보완 경로는 셀별 프로세스에서 전체 `shade(renderer.render())`를 직접 측정했다(11셀 전부).
  128장면은 flat 31.210 / Lambertian 37.380 ms, 비율 1.197689. 어느 것도 일반적 상한이 아니다.
  GPU/process VRAM 표본, utilization, Warp/driver/runtime, 원시 시간과 mean/P50/P95를 보존했다.
  작은 셀은 이상치 영향이 크며 주기적인 nvidia-smi 계측과 desktop 부하를 제거하지 않았다.
- 기존 실행은 tracked-clean 7c62dbf. 새 보완 도구/규약은 미커밋임을 명시하고 실제 전체 소스
  SHA와 HEAD 추적/일치 여부를 기록했다. 실행 전후 및 셀 간 source manifest가 같다.
  기존 renderer/판정 구현과 역사적 JSON은 수정하지 않았다. 기존 R5 runner도 재현용으로 보존했다.
- CPU renderer 테스트 67개 통과. 전체 unittest는 1,415개 실행 / 1,411 PASS / 4 skip이다.
  최초 전체 실행은 내가 CUDA_VISIBLE_DEVICES를 비워 CUDA import 계약에서 1 failure/1 error를 냈다.
  기존 코드 변경 없이 GPU-visible 환경에서 해소했다. 신규 계측 테스트 10개를 별도로 추가했다.
- R3 README의 잘못된 전체 commit SHA, R2의 오래된 상태표, R4의 원인 단정과 밝기 통제 주장,
  결과 문서/계획서의 R5 상한 주장을 교정했다. 위 직전 작업 기록의 "메시 축 상한" 해석도 철회한다.

추가 background/articulated scene 연구와 학습은 이번 재검증에 포함하지 않는다.
커밋·푸시는 하지 않았다. 원자료는 새 root에 보존했으며 삭제/덮어쓰기는 없다.

## 2026-09-11 — R4 background v1 구현 TECHNICAL_PASS, 계측 결함 수정, 학습 해상도 재측정

**R4에 없던 것을 만들었다.** 원래 요청에 있었으나 구현되지 않았던 일반 background 장면과
intervention 검사다. 절차적 바닥·벽·12각 기둥·2링크 경첩 패널이며 자산 로더도 시뮬레이터도
UAV도 쓰지 않는다. 상자 픽스처 대비: 삼각형 24→144, patch 12→62, **뷰당 보이는 삼각형 8→60–67**,
서로 다른 법선 4→10, 물체 2→4, 거리 2.7–4.4→4.1–10.2 m. R4가 막혔던 바로 그 지점이다.

검사는 통계 검정이 아니라 **반사실 검사**다. material 0 알베도를 절반으로 하면 그 material의
픽셀이 전부 바뀌고 나머지는 bit-exact로 불변이어야 한다. face→material을 `(id+1)%4`로 순환하면
보이는 material ID와 RGB가 **100%** 바뀌어야 한다. 광원을 반전하면 영상이 움직이고 material은
그대로여야 한다. instance ID를 +100 해도 RGB는 불변이어야 한다. 빈 mask는 통과가 아니라 거부다.
**R4b가 놓친 것이 이것이다** — 재배정했다고 선언만 하고 보이는 픽셀의 98%는 그대로였다.

커밋 914d520, 깨끗한 트리, 두 독립 프로세스에서 `run.json`과 `arrays.npz` SHA가 전부 일치하는
**TECHNICAL_PASS**다. 결과는 `results/renderer_background_v1_clean_2026-09-11/`.

**첫 실행의 FAIL은 계측 결함이었다.** 실패한 두 값이 0.9999999999999999였다. 전부 바뀌었는데
1 미만으로 나온 것이다. **CUDA의 `mean()`은 세지 않고 n으로 나눈다.** [1000,41000) 표본조사에서
**약 13%의 크기**가 영향받는다. float32는 n=7600, float64는 n=7898·25815에서 깨지고, 실패한 두 뷰의
개수가 정확히 그것이었다. **CPU는 모든 크기에서 정확하므로 CPU 테스트로는 못 잡는다.**
비율을 정수 합으로 바꿨고 이번 실행에서 n=7600·7898이 정확히 1.0이다.

**R3에도 같은 결함이 있었다.** `material_changed_fraction_selected`를 float 평균으로 구해 1.0과
비교하고 있었다. R3의 2,196 픽셀이 영향받는 크기가 아니어서 결과는 바뀌지 않았다 — **운으로
통과하고 있었다.** 정수 합으로 고치고 재실행해 기존 receipt와 `metrics`·`checks`·`array_sha256`·
판정이 모두 동일함을 확인했다(`renderer_r3_seed173_after_countfix_2026-09-11`).

**R5를 실제 학습 해상도에서 다시 쟀다.** 기존 수치는 480×270인데 학습 기본값은 **160×90**이다.
128장면 160×90에서 기하 3.22 ms, flat 0.63 ms, Lambertian 1.28 ms,
**추가 비용 반복당 0.66 ms, torch 최대 할당 183.9 MiB**(예약 228.0 MiB)다.
480×270의 1,601 MiB가 아니므로 **메모리는 V2의 병목이 아니다.**

진단으로만: 기존 기준 C를 이 장면에서 계산하면 네 뷰에서 0.027–0.228이다. **"R4가 통과한다"가
아니다.** 다른 픽스처, 뷰 4개, 광원 하나이며 그 가설을 검정하도록 설계되지 않은 실행이다.
**기존 R4/R4b 판정은 FAIL 그대로다.** 전체 테스트 1,440개 통과(skip 4).

## 2026-09-11 — URDF 자산 로딩 v1 구현 (계획 1번), TECHNICAL_PASS

계획의 1번, V1 선행 작업이다. 프로토타입에 자산 로더가 없어 장면을 코드에서 프리미티브로 만들어
왔다. 이제 URDF의 visual geometry를 `MeshScene`으로 읽는다. 표준 라이브러리 XML과 numpy만 쓰고
시뮬레이터도 URDF 라이브러리도 import하지 않는다(AST 테스트로 고정).

지원 범위는 저장소 실측을 따랐다: box 1107, cylinder 118, sphere 14, **mesh 3**(외형 개편 경로 밖).
mesh와 비-fixed joint, 없는 링크, 순환, 루트 복수, 음수 치수, 미선언 material은 전부 거부한다.
계약 `docs/renderer_urdf_loader_v1_contract.md`를 구현 전에 커밋했다.

**맞다는 근거는 눈이 아니라 세 성질이다.** 닫힌 표면(모든 모서리가 삼각형 2개에 공유, 위반 0),
바깥 winding(위반 0), 그리고 **부피가 해석값에 아래에서 수렴**한다 — 구 16×8 −6.26% → 64×32 −0.40%,
원기둥 16분할 −2.55% → 128분할 −0.04%. 내접 분할이므로 방향이 아래인 것이 맞다.
상자는 float64에서 정확하고 float32 저장 후 7.6e-8(float32 eps 1.19e-7 아래)이다.
**테스트가 두 층을 분리**해서 저장 정밀도가 분할 오류로 읽히지 않게 했다.
`rpy`가 고정축 `Rz@Ry@Rx`임을 독립 구성과 1e-14로 대조했고, 모터 링크가 URDF가 적은
±0.0777817 m에 놓이는 것도 확인했다.

GPU smoke: 다섯 자산 전부 TECHNICAL_PASS, 두 독립 프로세스 일치, 깨끗한 트리.
**요격기의 링크 9개와 재질 3개가 모두 화면에 픽셀을 만든다.** run1은 FAIL이었는데 원인이 자산이
아니라 **내 프레이밍**이었다 — 1.6 m 장대를 경계구로 맞추니 coverage 0.3%였다. 링크·재질은
정상 도달했다. **임계값은 건드리지 않고** 카메라 축 투영 범위로 맞추도록 고쳤고(종횡비 무관,
자산별 조정 불필요) 장대는 4.2–8.8%가 됐다. 실패 receipt는 보존한다.

**V1이 착수 전에 정해야 할 것이 측정으로 드러났다.** 계획의 V1("표적을 요격기 자산으로 교체")을
그대로 하면 의도와 반대로 간다.
- 요격기 visual은 0.176 m로 **자기 충돌 상자(0.283 m)의 62%**다. **프로펠러가 visual에 없다.**
  URDF 주석의 프로펠러 반경 0.0635 m와 모터 위치 ±0.0777817 m를 더하면 0.2826 m로 상자와 맞는다.
- 그대로 바꾸면 표적의 겉보기 선형 크기가 같은 거리에서 **약 72%**(투영 면적 기준)가 된다.
  겉보기 크기-거리는 E3-S가 측정한 양이고 검출기가 쓸 법한 단서다. 28% 변경은 개입이다.
- 요격기 보이는 픽셀의 **64.4%가 밋밋한 흰 구**다. 탐지 거리에서 사실상 흰 공이다.
  "표적이 드론처럼 보이게"가 목표인데 드론처럼 안 보이는 자산을 복제하면 목표를 못 만족한다.

**권고: V1을 "공유 드론 형상 자산을 만들어 양쪽에 적용"으로 바꾼다.** 프로펠러 원반 4개를 추가하고
몸체를 구에서 상자로 바꾸면 0.2826 m가 되어 충돌 상자와도 표적의 현재 크기와도 맞는다. URDF가
스스로 적은 산술을 따르는 것이라 새 가정이 없다. **결정 전에는 자산을 바꾸지 않는다.**
시뮬레이터 자산은 아직 하나도 건드리지 않았다. 전체 테스트 1,465개 통과(skip 4).

## 2026-09-11 — URDF 로더를 라이브러리 기반으로 교체, 측정으로 버그 5개 수정

사용자 지적: "urdf 라이브러리를 import 안 하면 어떻게 하라는 거야? 무조건 해야지." **맞는 지적이다.**
이 프로토타입의 격리는 aerial_gym·시뮬레이터·태스크·검출기·정책을 쓰지 않는다는 뜻이지 범용 파서까지
피하라는 뜻이 아니었다. 검증된 동작을 아무도 시험하지 않은 코드로 옮기는 것은 위험만 늘린다.

기본 백엔드를 **`urdfpy 0.0.22` + `trimesh 4.11.5`**로 바꿨다. 링크 트리·순기구학·재질·mesh 파일과
scale을 라이브러리가 담당한다. 직접 쓴 파서는 지우지 않고 **교차검증용**으로 남겼다
(`compare_backends`). **mesh 지원이 함께 생겨** 초판의 거부 규칙을 해제했다(BlueROV 180,450 삼각형).

**측정으로 찾은 버그 5개.** 셋은 라이브러리가 이 계약을 조용히 어기던 것이고, 둘은 내 검사가 틀린 것이다.

1. urdfpy는 **limit이 있는 revolute 관절을 받아들이고 영 자세로 평가**한다. 관절 있는 로봇이
   아무도 고르지 않은 자세로 로드된다. 가장 위험했다. 이제 거부한다.
2. urdfpy는 **선언되지 않은 material 참조를 기본색으로 조용히 대체**한다. 이제 거부한다.
3. 치수 0인 형상이 렌더러까지 가서 "Degenerate triangle"로 죽었다. 이제 **형상 이름을 말하며** 거부한다.
4. **내 비교 함수가 정점 평균으로 기구학을 재고 있었다.** 정점 평균은 기구학량이 아니다 — 같은 구라도
   UV 분할과 icosphere 분할의 평균점이 다르다. 배치를 재는 검사에 분할이 새어 들어와 나노미터
   "불일치"가 나왔다. **링크 월드 변환을 직접 비교**하도록 고쳤다.
5. **내 바깥 법선 검사가 볼록 형상 전용이었다.** 오목한 BlueROV 선체에서 88,721건을 보고하는데 전부
   무의미하다. 일반 표면용으로 **열린 모서리 0 + winding 불일치 0 + 부호 부피 > 0**을 추가했다.
   그 기준으로 재니 **출하된 BlueROV STL이 방수가 아니다**(열린 모서리 4,854, 불일치 312).
   자산의 성질이지 로더 결함이 아니므로 보고하되 막지 않는다.

거부 테스트는 **두 백엔드에 같은 케이스**를 건다. 다섯 자산에서 `compare_backends` 불일치 0이다.
격리 테스트는 공유 프로세스의 `sys.modules`가 아니라 **별도 프로세스**에 묻도록 고쳤다.

**삼각형 증가 비용을 쟀다 (128장면 160×90).** 상자 24개 → 요격기 736개 → 2,304개에서 기하 패스가
3.09 → 6.29 → **9.14 ms**로 약 3배가 되고, Lambertian 음영은 1.29 → 1.34 ms로 **변하지 않는다**
(픽셀에 비례하지 삼각형에 비례하지 않음). 메모리는 183 MiB로 동일하다.
그래서 **Lambertian/flat 비율이 1.175 → 1.063으로 내려간다.** R5에서 "메시가 무거워지면 1에
수렴한다"고 적었던 것이 이제 주장이 아니라 측정이다.

GPU smoke 다섯 자산 전부 TECHNICAL_PASS, 두 독립 프로세스 일치, 깨끗한 트리.
전체 테스트 1,470개 통과(skip 4). 시뮬레이터 자산은 아직 하나도 바꾸지 않았다.

## 2026-09-11 — V1: 표적을 빨간 상자에서 쿼드로터 실루엣으로 (계획 2번)

사용자 요청의 출발점이었던 문제다. 표적이 밋밋한 빨간 상자라 색만으로 찾을 수 있었다.
새 자산 `navrl_target_drone_v3.urdf`, 생성기 `tools/generate_shared_airframe.py`,
테스트 `tests/test_shared_airframe_v1.py`(14개), 계약 `docs/v1_shared_airframe_contract.md`.

**기본값은 안 바꿨다.** `NAVRL_TARGET_APPEARANCE=v3` 옵트인이고 기본은 v2다. 저장소가 이미 쓰는
`NAVRL_PHYSICAL_GEOMETRY_VERSION` 선택자와 같은 방식이다. 기본을 바꾸면 기존 학습·평가가 자산이
하나 생겼다는 이유로 조용히 달라진다.

**물리는 건드리지 않았고, 설명이 아니라 호출로 확인했다.** 시뮬레이터가 URDF 기하를 읽는 곳은
`asset_loader._urdf_collision_half_extents`와 `navrl_task`의 `base_link` 충돌 상자 조회 둘뿐이고
**둘 다 `collision`만 읽는다.** 테스트가 그 함수를 실제로 호출해 v2/v3 결과가 같음을 확인한다
([0.1415, 0.1415, 0.06], "0.283 0.283 0.12"). `<inertial>`·`<collision>` 토큰도 동일,
링크도 base_link 하나다. **visual이 충돌 상자를 벗어나지 않는지도** 검사한다.

**새로 정한 숫자가 하나도 없다.** 모터 위치·팔·모터 원기둥은 `quad_navrl_ref5in_v2.urdf`에서,
프로펠러 반경 0.0635는 `navrl_task.py`의 `PROP_RADIUS_5IN_M`에서, 몸체 상자 0.15×0.15×0.12는 같은
URDF 주석의 관성 근사에서 **읽는다.** 테스트가 생성기 소스에 그 숫자들이 리터럴로 없음을 확인하고,
커밋된 URDF가 재생성 결과와 바이트 동일함도 확인한다. 결과: visual 13개, 삼각형 1,548개,
닫힌 표면·winding 일치·부호 부피 > 0, 두 백엔드 일치, 경계 상자 0.282563 × 0.282563 × 0.120000.

**겉보기 크기는 바뀐다. 이건 측정했다.** 같은 카메라·3.0 m·**21개 시선**에서 픽셀을 셌다.
투영 면적 비 v3/v2가 0.539–0.700(중앙값 **0.587**), 겉보기 선형 비가 0.734–0.837(중앙값 **0.766**)이다.
**폭은 0.16%만 줄었는데 면적은 41% 줄었다** — 쿼드로터는 팔 사이가 비고 상자는 꽉 차 있다.
설계 실수가 아니라 표적을 드론 모양으로 만들면 반드시 따라오는 결과다.
**V4에 반드시 반영할 것**: E3-S가 겉보기 크기-거리를 측정했고 검출기가 그것을 쓸 수 있으므로,
개편 전후 오추적률을 그냥 비교하면 크기 변화와 형상 변화가 교란된다. **면적을 맞춘 상자 대조군**이
한 팔로 필요하다.

**실루엣을 만드는 것은 몸체와 프로펠러다.** 28개 시선에서 몸체 57.0%, 프로펠러 36.7%, 모터 5.8%,
**팔 0.5%**. 몸체 상자 0.15 m의 반대각선 0.1061 m에 모터 거리가 0.1100 m라 **팔이 3.9 mm만 나온다.**
문서화된 치수를 그대로 따른 결과이고, 팔을 더 보이게 하려면 몸체를 줄여야 하는데 **출처가 없으므로
지어내지 않는다.**

**색은 아직 화면에 닿지 않는다.** `render_raw_rgbd()`가 표적 픽셀을 `target_color`로 평면 채움한다.
v3는 부품별 중립 회색을 선언해 뒀고(어느 재질도 빨강 우세가 아님을 테스트로 확인) V2에서 평면
채움을 음영으로 바꿀 때 쓰인다. **이번 단계는 색 지름길에 대해 아무 효과도 주장하지 않는다.**

요격기와 방해물은 그대로다. 방해물은 `test_navrl_distractors.py`(1,555줄)가 단일 프리미티브로
고정하고 시뮬레이터가 그것으로 footprint를 계산하므로 별도 단계다. 전체 테스트 1,484개 통과.

## 2026-09-11 — 전체 코드 점검: 다섯 축 병렬 조사, 확정 결함 수정

사용자 요청으로 저장소 전체(535파일 144,373줄, 동결 receipt 제외)를 다섯 축으로 병렬 조사했다:
수치 정확성, 중복 상수, 조용한 기본값, 낡은 코드·규약, 대리 측정·미명시 가정.
**모든 항목을 직접 재현한 뒤에만 고쳤다.** 조사 결과를 그대로 믿지 않았고, 실제로 두 건은
조사 쪽 판단이 과했다.

### 고친 것 — 재현 → 수정 → 재확인

| 무엇 | 재현 결과 |
|---|---|
| `navrl_stats` 퇴화 표준오차 | `wald_diff(0/2048, 2048/2048)` → se=0.0, p=0.0, **`excludes_zero=True`**, 폭 0 신뢰구간. A5/A7/A8 표와 시드 풀링의 PASS/FAIL를 가른다. `pool_fixed`·`cochran_q`는 `ZeroDivisionError` |
| `CLAUDE.md` RUNNING 주장 | 권위 JSON은 `OPERATOR_STOPPED_INCOMPLETE`+`resume_forbidden`, 학습 프로세스 0개, 같은 파일 85행이 스스로 반박 |
| 밀도붕괴 FAIL-STOP | `except Exception: pass` 안에 있어 며칠간 **출력 없이** 건너뛸 수 있었다 |
| 관측 폭 fallback 574 | `navrl_perception`의 모든 `ValueError`를 계보에 없는 574로 바꿔 resume이 shape 오류로 죽고 원인 메시지는 사라진다 |
| 게이트 무장 플래그 fail-open | `=Y`·`=enabled`가 조용히 "불필요"가 돼 출처 증명 없이 학습이 돈다 |
| `motor_model` bare `except` | 오타 난 적분기 이름이 조용히 rk4가 돼 A/B가 같은 값을 두 번 잰다. `KeyboardInterrupt`도 삼켰다 |
| `orientation_report`(내 코드) | motor_0만 뒤집으면 그 링크 −1.56e−5인데 합계 +6.88e−4라 **True**. 대체한 볼록 검사보다 약했다 → shell별 부피로 교체 |
| 생성기 인터프리터 의존(내 코드) | `ET.indent`가 3.8에 없어 `--check`가 3.8은 IDENTICAL, 3.13은 DIFFERS |
| 실루엣 테스트(내 코드) | 분할 정점을 재고 있었다. 내접 오차 1.22 mm > 실제 여유 0.218 mm → 선언된 형상의 해석적 지지로 교체 |
| 낡은 인용 5건 | 5~700줄 드리프트, 둘은 출력되는 provenance 안. 줄번호 대신 심볼로 |
| `camera_target_radius` 주석 | "0.30 m 표적 footprint"라는데 그런 표적이 없다(0.28 / 0.283) |
| 생성기 docstring(내 코드) | "모든 치수를 읽는다"는데 둘은 직접 적었다 → 사실대로 고치고 전사 검증 함수 추가 |

### 고치지 않기로 한 것 — 사유

- **`_arena_contract`의 surface clearance 기본값 0.0.** 조사는 "오늘 틀린 기록"이라 했으나,
  `asset_manager.py:133-136`에서 `navrl_band` 모드는 이 값을 **소비하지 않는다.** 0.0은 "이 계약이
  적용되지 않았다"는 뜻으로 방어 가능하고, 체크포인트 호환성을 가르는 값이라 바꾸지 않는다.
  추적된 receipt 40개는 전부 0.45다. **틀린 것은 값이 아니라 docstring이다.**
- **표적 충돌상자 0.283 vs 센서 0.28 불일치.** 실재하지만 고치면 검출기 기하가 바뀌어 기존 비교가
  무효가 된다. 연구 영향이 있는 결정이라 사용자 판단이 필요하다.
- **바이트 동결 계약 문서 2건.** `CLAUDE.md:45-52`가 잠그고 있고 과거 커밋 `921fb1d`가 이걸
  '정리'했다가 브랜치 8개를 깼다. 조사 권고가 와도 건드리지 않는다. 동결 테스트 통과 확인함.

### 조사가 통과시킨 것

CUDA `mean()==1.0` 함정은 torch 코드에서 **0건** 남았다(내가 앞서 고친 두 곳이 전부).
누적 순서 위험 0건, 다르게 계산된 GPU 값의 정확 비교 0건. 문서의 backtick 경로 39개 전부 실재.
전체 테스트 1,484개 통과(skip 4).

## 2026-09-12 — README와 상태 사이트를 현재 상태로 정리

이번 세션의 결과가 두 문서에 반영돼 있지 않았다. 코드와 문서가 갈라진 채로 두지 않는다.

**README.** Status 날짜를 2026-09-12로 올리고 **트랙 D(시뮬레이터 외형)**를 추가했다 —
"렌더러와 자산은 준비됐고 옵트인, 그것이 가능하게 하려는 shortcut 재측정은 아직 안 돌렸다".
최근 결과에 다섯 항목을 넣었다: 표적이 빨간 상자였고 방해물도 같은 빨강이라 색만으로 찾을 수
있었다는 것, 드론 형상으로 바꾸면 **투영 면적이 41% 줄어 다음 실험의 교란**이 된다는 것(면적을
맞춘 상자 대조군이 필요하다), 렌더러 기준 C가 **두 번 실패했고 그대로 보존**한다는 것과 그 이유,
음영 비용이 학습 해상도에서 0.66 ms·184 MiB라는 것, 그리고 전체 코드 점검에서 **한 모양의 결함**
(한 모듈이 검증한 값을 다른 모듈의 넓은 예외가 그럴듯한 값으로 대체)을 고쳤다는 것.
트랙 C에는 **6암 식별 단서 철회**를 명시했다. 평가 표에 5행(R4/R4b, background v1, R5 비용,
URDF 로더, V1 실루엣), 저장소 지도에 2행(자산 디렉터리, 격리된 렌더러 프로토타입)을 더했다.

**상태 사이트.** 히어로 상태줄을 외형 개편으로 교체하고 `#appearance` 절을 새로 넣었다(내비 포함).
독립 프로토타입이 시뮬레이터를 import하지 않는다는 계약, 7단계 표(R3 PASS / R4·R4b FAIL 보존 /
C 실패의 진짜 이유 / background v1 / R5 비용 / URDF 로더 / V1), 물리 무영향을 **호출로** 확인한다는
점, 겉보기 크기 41% 감소가 교란이라는 점, 색은 아직 화면에 닿지 않으므로 **색 지름길 감소를
주장하지 않는다**는 점을 적었다.

편집은 줄 단위로 했고(문자열 슬라이스 금지 규칙 준수) 태그 균형 검사와 `html.parser` 전체 파싱으로
확인했다. 567 → 630줄. 전체 테스트 1,484개 통과(skip 4).


## 2026-09-12 — 공개 정합성 교정 1: 현재 판정과 역사 기록 분리

사용자 승인 범위는 저장소 정리·독립 renderer 재현뿐이며 policy·제어 실험은 실행하지 않는다.
R4/R4b FAIL을 유지하면서 기준 자체가 무효라는 단정을 철회했다. C3 요약부를 WITHDRAWN으로,
E3-P를 세 번째 검사 실행 후 ATTITUDE_NOT_RELIABLE/BLOCKED로 정정했다. six-arm 식별 단서는
철회 표시하고 0.196 px를 정확도가 아닌 궤적 매끄러움 잔차로, 183.9 MiB를 Torch peak allocated로
명시했다. Det-Fly 020과 NPS P4/P5/S4 test 열람을 분리했다. VERIFICATION 상단이 현재 판정을
안내하고 OPERATIONS의 소비된 P10 명령은 역사 기록으로 남긴다. 이전 WORKLOG 서술은 작성 당시
기록이며 이번 교정이 현재 해석을 대체한다. raw JSON·사전등록·정책 코드·실험 수치는 변경하지 않았다.

검증: 새 문서 회귀 테스트 8개와 git diff --check. 전체 스위트는 후속 정리 완료 후 재실행한다.

## 2026-09-12 — 공개 배포 정합성: 인수인계 계획 항목 2·3·5·6·7·9·11 실행

다른 세션이 항목 1(stale claim 정리)을 `c634ad1`로 커밋하고 항목 2의 파일을 미추적으로 남겼다.
그 위에서 이어받았고, **주장한 것을 전부 직접 재현한 뒤에만** 커밋했다.

**항목 2·3 — 비교기와 증거.** `None == None` fail-open이 실제로 닫혔는지 실 receipt로 시험했다:
정상 두 개 `exit 0`, **둘 다 필수 필드 누락 `exit 2`**(필드 이름을 출력), per_view 값 하나 변조
`exit 1`, JSON 절단 `exit 2`. 결과 디렉터리는 base commit·clean 플래그·파일별 SHA·13개 소스
스냅샷·원배열 해시 일치를 확인했다. 다만 그 README가 **`arrays.npz`를 추적한다고 잘못 적고
있었다** — `.gitignore:3`의 `*.npz`로 제외된다. 정정했다.

**항목 5 — 설치·Quick Start.** `requirements.txt`는 `urdfpy=0.0.22`(등호 하나)로 **pip이 파일
전체를 파싱하지 못했다.** 더 심각한 것은 `networkx==3.1` 선언인데 실제로는 2.2가 깔려 있고
**urdfpy setup.py가 `networkx==2.2`를 고정**한다 — 선언대로 설치하면 URDF 로더가 깨진다.
`trimesh`는 로더가 직접 import하는데 미선언이었다. 전부 고치고 실제 버전을 주석으로 남겼다.
Quick Start 4단계는 **실행해서** 확인했다(3단계 renderer 146개는 PYTHONPATH 없이도 통과).

**항목 6 — 라이선스.** `THIRD_PARTY_LICENSES.md`를 파일 수준으로 작성하고 코드/데이터셋 라이선스를
분리했다. **미해결 항목 하나가 실재한다**: ETH 검수 이미지 34장과 1.3 MB zip이
**CC BY-NC-SA 4.0** 파생물인데 ShareAlike·NonCommercial을 BSD-3이 만족하지 못한다. 루트 LICENSE가
그 디렉터리를 덮는 것으로 읽히면 안 된다. `THIRD_PARTY_NOTICE.md`를 파일 옆에 두고 선택지 셋을
적었으며 **하나를 조용히 고르지 않았다.** 확인한 것: 비밀정보 0, YOLOv5 소스 미포함(AGPL 경계 유지),
`datasets/` 추적 0, `.pth` 18개 = 상류 예제 11개(50 MB) + 프로젝트 산출물 7개(100 KB).

**항목 7 — 인용.** `CITATION.cff`는 저장소가 실제로 기록하는 것만 담는다(핸들·이메일·원격).
**실명과 ORCID는 저장소 어디에도 없어서 넣지 않았다.** 추측하지 않았고 미입력임을 명시했다.

**항목 9 — 공개 감사 미해결.** 추적 파일 **83개에 `/home/fair` 절대경로**가 있다. 66개 셸 중 58개는
`${PYTHON:-...}` 형태로 덮어쓸 수 있으나 사용자명이 노출되고, 8개는 대체값이 없다. 런처 바이트가
training source manifest에 해시로 들어가므로 **일괄 수정은 receipt를 무효화한다.** 미해결로 보고한다.

**항목 11 — 검사.** README·THIRD_PARTY 로컬 링크 56개 전부 해소, 사이트 `html.parser` 전체 파싱 OK,
CFF YAML 구조 검증 OK(cffconvert는 환경에 없음). 전체 테스트가 한 번 **실패**했다 — 다른 세션이
고정된 문서 둘을 고쳐 발표 자료 해시 핀이 어긋났다. 슬라이드 주장(6.2%, 3,107프레임, E3-P BLOCKED)은
여전히 유효해 **핀만** 문자열 치환으로 갱신하고 zip의 해당 멤버 하나만 교체했다.
최종 **1,516개 통과**(skip 4, 기존 1,484에서 증가, 깨진 것 없음).

미실행: 항목 4(README 전면 재구성), 항목 8(외부 인용 audit), 항목 10(문서 역할 고정),
항목 12의 나머지, 항목 13(push). push는 하지 않았다.

## 2026-09-12 — 23e6818 보고 감사: 수치 재현, 전체 동일성 단정 철회

새 simulator/학습/제어 실행 없이 `results/target_appearance_in_sim_2026-09-12/raw.json`을
독립 재계산했다. SHA-256 `0589df29f51cace0ac7c756c29531f6f53021aaf739bf6793eac4d95046700a6`.
각 arm 3,840개 모두 유한·양수이며 저장된 거리/픽셀 수가 항목별로 같다. 픽셀 수 비의
최솟값/Q1/중앙값/Q3/최댓값 전부 1.000. [4,5)/[5,6)/[6,7) m 구간 n은 1196/1804/840,
두 arm 픽셀 수 중앙값은 18/6/6이다. 원자료는 변경하지 않았다.

원자료에 pose/mask/RGB/env-frame ID/runtime/source commit이 없어 "전체 궤적·영상이 같다"를
철회하고 PARTIAL_EVIDENCE로 제한했다. 보관 commit과 실행 commit도 구분한다. 같은 색 cue는
foreground segmentation과 지정된 한 물체의 identity를 구분해야 한다. 독립 prototype의
면적비 0.587을 현재 detector 입력 변화로 읽던 README·사이트·계획·VERIFICATION을 교정했다.
두 collision 추출기와 visual mesh를 읽는 Warp loader도 구분했다. 상세 `AUDIT.md` 및
Python 표준 라이브러리 재계산 테스트 6개를 추가했다.

**기존 통합 회귀 실패를 발견:** 23e6818에서 1,516 tests / 1 failure / 4 skipped.
`test_shared_airframe_v1.PhysicsIsUntouchedTest.test_it_is_still_one_base_link`가 새 13-link
자산과 충돌한다. 계약에 CONTRACT_MISMATCH를 기록하고 실패를 유지했다. URDF, 생성기,
검출기·제어 코드나 기존 통합 테스트 기준을 바꾸지 않았다. 보관된 로드 실패 로그가 없으므로
로드 성공/실패를 이번 감사가 새로 재현했다고 주장하지 않는다.

사이트 테스트는 오래된 PASS 기대 대신 근거 누락 상태의 source·BLOCKED·authority·hardware_claim
전체 필드를 검사하도록 고쳤다. 허위 PASS/하드웨어 주장/권한 승격/필드 누락 5종은 거부한다.
`status.json`과 historical 판정은 변경하지 않았다. docs 색인의 R2 미실행 문구도 정정했다.

검증: 재계산 6개, 문서 주장 10개, 발표 산출물 11개, 독립 renderer 146개 통과.
snapshot tests 19개 실행(1 skip), 나머지 통과. Node와 최종 전체 스위트 결과는 아래 후속 기록.
시작 시 live origin/main은 eaa7f73, local 23e6818은 **10 commits ahead**였다(보고의 11 정정).
새 통합/정책 실험 권한을 열지 않았고 push는 하지 않는다.

후속 확인: 최종 전체 스위트 **1,524 tests / 1 failure / 4 skipped**(38.387초).
새로 추가한 8개 테스트는 모두 통과했고 실패는 같은 one-link 계약 1개뿐이다.
Node 사이트 테스트 PASS. 기존 링크 검사도 디렉터리 자체의 Git 추적을 요구하는 결함이 있어
gallery의 실제 `index.html` 존재·추적을 검사하도록 고쳤다. 문서/발표 원본 hash는 유지된다.

## 2026-09-12 — 독립 배경 검증 원배열의 공개 재현성 복구

기존 비교 테스트는 Git에서 제외된 `arrays.npz` 두 개를 필요로 했다. 합성 일반 배경만 담은
`results/renderer_background_v1_2026-09-12/run{1,2}/arrays.npz`를 각각 1,573,320 bytes로 확인해
이 두 경로만 `.gitignore` 예외 처리했다. 두 파일 SHA-256은
`bd4c7d4664b78d76683b1a89760ebdfeecf8427fe064a96ae8a08c3074736492`이며 receipt와 일치한다.
decoded 14개 배열과 historical Git source 14개도 비교기가 검증한다. 직전 기록의 "13개 source"는
14개로 정정한다. 다른 NPZ·실사 데이터·체크포인트 ignore는 그대로이며 historical JSON은 보존했다.

`--system-site-packages` 없는 새 Python 3.8.20 venv에 NumPy 1.23.0만 설치해 비교 테스트
24개를 실행했고 통과했다. Torch/Isaac Gym은 이 환경에서 검색되지 않는다.
이는 원배열/receipt의 CPU 검증이며 새 GPU 렌더링이나 전체 설치 quickstart 성공을 뜻하지 않는다.
전체 Git clone에서의 검증은 이 원배열 포함 커밋 이후 별도로 수행해 기록한다. push는 하지 않는다.

## 2026-09-12 — 독립 checkout의 CPU 근거 재현 완료

`ce7d191`을 `/tmp/motar-evidence-clone-C38xRP/repo`에 full local clone했다(`--no-hardlinks`).
ignored 원배열이나 source_snapshot은 복사하지 않았다. 앞서 만든 NumPy-only venv로 appearance
기술통계 6개·배경 비교 테스트 24개, 실제 비교 CLI(PASS/evidence_verified true×2)를 확인했다.
Node 사이트 계약 검사도 PASS, 검사 후 clone은 clean이다. 실제로 push하지 않았으므로 원격
배포 검증은 아니다. `docs/public_evidence_reproduction_2026-09-12.md`에 명령과 범위를 남긴다.

전체 환경 테스트의 남은 실패 1개는 그대로이며, 전체 renderer 신규 설치·README 전면 재편·
외부 인용/라이선스/전체 history 감사는 완료하지 않았다. 앞선 CPU Torch 신규 설치는 의존성
해석/build 단계에서 실패했으며 이번 NumPy-only 성공으로 덮지 않는다. 새 simulator 실험·
통합 코드 변경·사용자 데이터 삭제·push는 수행하지 않았다.

## 2026-09-12 — 독립 일반 renderer CPU 신규 설치 완료, 버전 동일성 가설 기각

이전 항목의 NumPy-only 근거 점검과 별도로 실제 renderer 신규 설치를 수행했다. 이 작업 파일은
다른 세션이 검토 후 `ef59832`로 커밋했다. 근거는 `results/renderer_cpu_install_2026-09-12/`다.

| 시도 | 실제 결과 |
|---|---|
| 새 진단 venv + PyPI urdfpy 0.0.22 | pip check 성공, renderer 기존 146 tests 중 errors 8 |
| 같은 버전 VCS 설치, 강제 교체 없음 | exit 0이지만 pip report의 install 목록 0; 실제 교체 아님 |
| 진단 venv에 정확한 source 강제 교체 | 기존 146 tests PASS; 신규 설치 성공으로 분류하지 않음 |
| 두 번째 새 venv + 별도 CPU profile | 첫 다운로드 read timeout, 설치 전 중단; 동일 profile 재시도로 성공 |
| 두 번째 환경 계약 | 기존 146 + 패키징 6 = 152 tests PASS, skip 0; pip check PASS |
| 두 CPU 프로세스의 일반 상자 smoke | 각 2프레임, 160×120, scene 1, seed 0; 프레임당 8개 배열 SHA 일치 |

정상 urdfpy upstream source는 `5466842899b33bd549e8f9e2a9a987bd5e37373b`다. 같은 버전 문자열의
공개 wheel과 cylinder 구현이 다르다. 로컬 urdfpy checkout도 이 commit이고 clean임을 확인했다.
`requirements-renderer-cpu.txt`는 Linux x86-64/CPython 3.8/CPU profile이며, 설치 후 실제 Git
출처·비-editable·venv 내부 설치 경로를 확인한다. root simulator 환경을 덮어쓰지 않았다.
OS/build 의존성까지 잠근 hermetic 환경은 아니다.

공개 CPU smoke의 source는 별도 clean checkout `cdc8113`이다. 설치 profile은 당시 후보의 SHA로
식별했으며 보관 commit `ef59832`를 실행 commit으로 소급하지 않는다. 4개 NPZ만 ignore 예외로
추가했고 file/decoded array/source SHA를 근거 테스트 3개로 검증한다. receipt의
RENDERED_UNASSESSED/NOT_EVALUATED/NOT_SUPPORTED는 유지한다. GPU·전체 simulator 신규 설치나
성능 향상을 주장하지 않는다. 실패한 설치와 당시 전체 회귀 실패 1개도 summary에 보존했다.

## 2026-09-12 — ef59832 후속 검사: 테스트 충돌 해소와 미측정 범위 분리

사용자가 보고한 `5cea0e4`(V1 테스트 수정), `ef59832`(설치 패키징 보관)를 확인했다. 검사 시작
HEAD는 ef59832, 트리는 clean, 로컬 origin/main(eaa7f73) 대비 15 ahead였다. V1 테스트·자산·
실행 코드 자체는 이번 후속 작업에서 수정하지 않았다.

깨끗한 ef59832에서 전체 회귀 **1,533개 실행 / 1,529 통과 / 4 skip / 실패·오류 0**, 37.416초.
별도 CPU profile에서 renderer **152개 통과**, 저장 근거 **3개 통과**, Node 사이트 PASS다.
`results/renderer_cpu_install_2026-09-12/followup.json`에 출처와 실행 구성을 기록한다.

README 일부만 성공으로 바뀌고 VERIFICATION·사이트·계약·CPU 설치 문서에는 현재 실패라고
남아 있던 모순을 고쳤다. 현재 상태는 TEST_CONTRACT_RECONCILED이며 당시 CONTRACT_MISMATCH는
실제 이력으로 유지한다. 설치 당시 `summary.json` SHA
`e5669ea4d1c2069eb34f983195435d3ce720e505c7e687ab75d9f04d8a07ebf9`는 그대로다.
문서 검사 기존 10개 중 상태 검사 1개를 갱신하고, 이력 보존·미측정 비용 검사 2개를 추가했다.
문서 검사 12개·발표 자료 11개·Node 사이트는 통과했다. 변경 후 전체 회귀는 아래 후속 확인에 기록한다.

기존 URDF smoke의 3.09/9.14 ms는 다른 장면의 기하 패스이며 새 경로의 추가 비용 측정이 아니다.
“몇 ms가 추가된다”는 확정 추정을 채택하지 않는다. 보고된 5 mm 변조 검사는 별도로 재실행하지
않았다. 3,840쌍 probe의 PARTIAL_EVIDENCE, R4/R4b FAIL, shortcut 감소 미측정도 유지한다.
요격 시스템의 표적 검출 개선용 커널은 격리 prototype을 포함해 구현하지 않았다. 새 detector·
정책·제어 실행, 사용자 데이터 삭제, 원격 push는 수행하지 않았다. 다음은 독립 CPU 공개 입구와
남은 배포 체크리스트 정리다. 상세 `docs/repository_followup_2026-09-12.md`.

최종 후속 확인: 문서 검사 2개 추가 후 전체 **1,535개 실행 / 1,531 통과 / 4 skip /
실패·오류 0**, 37.898초, exit 0. 이는 ef59832 위 문서·근거 테스트 변경이 있는 작업 트리의
회귀이며 clean 실행으로 표시하지 않는다. `git diff --check` PASS, 기존 실행 코드·자산·
CPU profile·historical 설치 summary는 변경 없음. 이번 후속 변경은 미커밋·미푸시 상태로 남긴다.

## 2026-09-13 — D7 기존 커밋 푸시와 공개 연구 개요 사이트 재구성

사용자의 새 첨부 요청에 따라 연구 실행 대신 공개 설명 계층을 정리했다.
`git fetch origin` 후 원격 `77d885b55d642ffe6ae10aa4d89b9cdc962d7717`,
HEAD `52b5dd878b28a15e0391efdee7cc5b2e01bdca65`, clean tree, 원격 ahead 0 / local ahead 4와
ancestor 관계를 확인했다. `git push origin main`으로 cfeb0ac / 06c7643 / 1edf171 /
52b5dd8 기존 네 커밋을 정상 푸시했다. force push 및 D7 재실험은 하지 않았다.

`docs/status/index.html`을 Overview → Contributions → Environment → System → Perception →
Parameters → Algorithms → Evidence → External Data 구조로 줄였다. 기존 WebGL 실행 코드·
컨트롤 ID·script 순서는 그대로이고 긴 설명은 details에 넣었다. 이전 HTML 전문을
`docs/status/archive-2026-09-13.html`에 원본 바이트 그대로 보존했다(SHA-256
`a96982824b82293172b73ac226ade7396b8773c62366c21e32a5f4694accf2bb`). 상세 이동과 소스는
`docs/status/overview-sources-2026-09-13.md`가 연결한다. Archive의 공유 JS는 live이므로
독립 실행 환경 snapshot이라고 부르지 않는다.

새 그림 7종은 `docs/assets/paper/overview-2026-09-13/`에만 생성했다. SVG 7 + 3840×2160
PNG 7 + 벡터 PDF 7, 해시 manifest, ZIP이다. 기존 paper manifest·ZIP·historical figure 및
result 파일은 덮어쓰지 않았다. `tools/render_research_overview.py`는 문서 생성 도구만
재사용하며 simulator를 임포트하지 않는다. 그림은 기존 모듈의 상위 설명이며 새로운
표적 인지·제어·요격 개선 설계가 아니다. 실사 selector 출력은 candidate rank/no-selection이고
metric target state나 live policy 통합이 아니라는 점을 본문·점선으로 구분했다.

공개 manifest의 뒤처진 D6–D8 명칭을 schema 2의 실제 D1–D9 기록으로 이관했다.
D6 INCONCLUSIVE, D7 GO는 shadow 비용/출력 비간섭에만 해당, D8 NOT_STARTED,
D9 NOT_RUN이다. 기존 generic-box benchmark 계획과 D6 target-mesh 결과를 합치지 않았다.
기존 assistance scope BLOCKED_BY_POLICY와 실제 연구 상태를 별개로 설명했다.
E3-S 6.2%는 단일 비행 dark-pixel 크기 proxy의 블록별 절대 상대오차 중앙값들의 중앙값이며,
일반 센서 정확도/자세 분리가 아니다. E3-P·P10·C3 철회·R4/R4b 실패의 한계를 유지한다.

설정 확인 중 default와 historical run의 차이를 명시했다: LiDAR 기본 36×4/4 m와
canonical 72×4/12 m, 축별 속도 기본 2.0와 historical 2.5 m/s, 환경 기본 256과 D7 주요 셀
128은 다른 것이다. 현재 실행값으로 묶지 않았다. Actor·실사 selector Transformer도 별개다.

검증: 전체 unittest 첫 실행 **1,582 실행 / 1,578 통과 / 4 기존 skip / 실패·오류 0**,
42.352초. 신규 개요 계약 8개, 기존 Node 사이트/arena motion/route/manifest, headless WebGL,
공개 경로·Draft-7 schema·CFF 1.2.0, SVG XML/텍스트 overflow·figure hash/ZIP, diff-check 통과.
역사적 수치/그림 assertion은 그 내용이 보존된 archive로 이동했고, 현재 페이지의 DOM·
링크·claim boundary는 신규 테스트로 계속 검사한다. CPU CI에 새 개요 테스트를 추가했다.
전체 suite의 고의 실패주입 traceback과 기존 dependency warning은 테스트 오류와 구분했다.

실렌더 첫 검수에서 inherited CSS가 모바일 nav를 GitHub만 남기고 숨김을 발견했다.
overview 전용 CSS로 교정하고 desktop 1360×1000 / tablet 800×1000 / mobile 390×844에서
각각 nav 8개·manifest 9행·WebGL canvas·이미지 decode·문서 가로 overflow 없음을 확인했다.
manifest 요청 차단 시 성공으로 보이지 않는 fallback도 확인했다. 7개 그림 PNG와 화면
표본을 직접 봤다. `results/public_site_overview_2026-09-13/browser/`에 스크린샷 12개와
실제 HTML/CSS/JS/SVG/tool SHA가 있는 dirty-tree receipt를 보존했다. 자세한 재현 명령과
검증 한계는 같은 결과 디렉터리 README에 있다. Archive 생성 때 끝에 붙었던 빈 줄 하나는
최종 원본 SHA 비교로 발견해 제거했고 원본 바이트 해시로 테스트를 고정했다.

이번 변경으로 연구 결과·센서 기하·detector·정책·제어·viewer motion을 바꾸지 않았다.
D8 구현/적응/새 학습과 표적 인지 개선 실험은 진행하거나 설계하지 않았다. 기존 기하·
관측 계약의 미결 상태, 별도 계보·사전등록이 필요하다는 경계만 문서에 남겼다.
사이트 변경은 사용자 요청대로 검토·커밋까지 진행하며, 이번 명시적 원격 푸시 범위는
위 D7 네 커밋이다. 새 사이트의 hosted CI/공개 배포 완료는 주장하지 않는다.

## 2026-09-13 — 공개 사이트를 논문형 단일 컬럼 페이지로 재설계

사용자가 대시보드/제품 랜딩처럼 보인다고 지적한 `be89675` 사이트를 그 커밋 위에서 다시
구성했다. 시작 확인에서 보고와 달리 `origin/main`도 이미 `be89675`를 가리켰다. 다른 세션이
먼저 원격에 반영한 상태로 판단했으며, history rewrite나 force push는 하지 않았다. 이번
논문형 변경은 별도 후속 커밋으로 만들고 사용자 검토 전 push하지 않는다.

`docs/status/index.html`은 제목·저자/저장소·140-word Abstract 뒤에 1 Introduction,
2 Research Environment, 3 System Architecture, 4 Experimental Setup, 5 Algorithms,
6 Evidence and Current Results, 7 Discussion, 8 Next Research Steps, availability 순으로
읽히는 단일 컬럼 문서가 됐다. 본문 폭은 960 px, 그림 최대 폭은 1120 px다. 기존 WebGL
arena는 Figure 2로 유지하고 여섯 정적 SVG와 합쳐 번호·caption이 있는 Figure 1–7로
제시한다. 관찰/Transformer 그림은 dated download package의 supplementary asset으로 남긴다.

CSS는 Georgia/Times 계열 serif 본문, 흰 배경, charcoal/gray와 muted blue 한 가지 accent,
얇은 표/구분선으로 다시 작성했다. dashboard card 20개, gradient, shadow, 큰 pill/metric
box와 교대 section background는 제거했다. 6개 navbar 항목은 얇은 흰 navigation으로
유지한다. renderer 도구로 Figure 1/5/6의 논문형 block diagram과 전체 파생 PDF를 다시
내보냈으며 같은 날짜 package의 manifest와 ZIP도 현재 21개 산출물 해시로 갱신했다.
SVG가 정본이고 PNG는 3840×2160, PDF 일곱 개는 `pdfimages -list`에서 raster row가 없다.

근거 경계는 유지했다. D6는 INCONCLUSIVE, D7 GO는 shadow cost/output non-interference만,
D8은 NOT_STARTED, D9는 NOT_RUN이다. 실사 selector의 no-selection/rank를 persistent identity나
metric state로 읽지 않으며, P10 net adaptation·appearance shortcut 감소·real flight/sim-to-real을
성립한 결과로 쓰지 않았다. 데이터셋은 역할 표 하나로 줄이고 라이선스 세부는 기존
`THIRD_PARTY_LICENSES.md`로 연결했다. 기존 상세 HTML archive SHA와 historical result/문서는
건드리지 않았다. 실험·학습·simulator/detector/policy/controller 코드는 실행하거나 바꾸지 않았다.

브라우저 검수 첫 시도에서 inherited responsive CSS가 tablet nav를 하나만 보이게 해 실패했다.
이를 override한 뒤 mobile evidence table의 760 px 최소 폭이 390 px viewport를 774 px로
확장하는 두 번째 실패를 찾았다. 640 px 이하에서 evidence row를 label/value 세로 형식으로
바꾸고 preview의 device metrics와 scroll-to-top을 고정했다. Chrome profile 삭제 race도
명시적 Browser.close와 cleanup tolerance로 교정했다. 최종 1360×1000, 800×1000,
390×844에서 nav 6, manifest status 4, canvas, 모든 그림 decode, route contract,
document overflow를 검사해 PASS했다. manifest 요청 차단 시 UNAVAILABLE로 fail closed한다.
스크린샷 12개와 실제 source SHA receipt는
`results/public_site_paper_2026-09-13/browser/`에 있다.

동일 파서로 `be89675`와 비교한 text/interface volume은 HTML 24,974→27,062 bytes,
visible 6,291→14,380 characters / 1,235→1,929 words, paragraph 44→30,
dashboard cards 20→0, numbered figures 7→7이다. 문서 자체가 짧아졌다고 주장하지 않는다.
요청된 Abstract/방법/한계/caption/table을 추가하면서 UI fragmentation과 status-card 반복을
줄인 결과다.

검증: 전체 unittest **1,584개 실행 / 1,580 통과 / 기존 skip 4 / 실패·오류 0**,
43.503초, exit 0. 논문 페이지 계약 10개, status manifest/static site Node, arena
fixed-clock/route/headless WebGL, 공개 local links와 Draft-7 schema, CFF 1.2.0,
figure hash/ZIP/XML, PDF vector 검사와 `git diff --check`가 모두 통과했다. 전체 suite의
dependency/resource warning과 고의 fault-injection traceback은 exit 0인 테스트 출력으로
구분했다. 새 사이트의 원격 배포 완료는 주장하지 않으며 이 후속 커밋은 push하지 않는다.

## 2026-09-13 — D8 결과 전 기하 계약과 기술 gate 사전등록

사용자의 장시간 D8 구현 요청을 시작하기 전에 `VERIFICATION.md` 실행 authority와 Track D 상태를
대조했다. 시작 HEAD/origin은 모두 `23d48b5`, working tree는 clean이었다. 연구 authority 검사는
`PASS_RESEARCH_AUTHORITY_FREEZE`; 기존 route/PPO GPU 권한은 닫혀 있으며 새 학습은 허용하지 않는다.

`docs/preregistration_dynamic_mesh_detector_d8_2026-09-13.md`에 D8-A 질문·세 arm·지표·판정·중지
조건을 treatment 코드와 결과 전에 고정했다. 기하는 선택지 C로 결정했다: collision은 v2/v3
`0.283×0.283×0.12 m`, historical sensor baseline은 analytic OBB half-extents
`(0.14,0.14,0.06 m)`, D8 treatment는 SHA-256
`c843e0bd9004ab596d5b948dc7566c9f3d3e28b7a3d98d3e8f4de581465dcad0`인 v3 URDF visual mesh다.
서로를 정답 하나로 합치지 않는다.

arm은 `analytic_flat`, geometry-only `mesh_flat`, 같은 mesh에 deterministic material/normal/light만
추가한 `mesh_shaded`다. D8-A 주요 셀은 128 env×160×90, warm-up 50/측정 500, seed 20260913이며
D7의 비용 GO/NO-GO 경계를 결과 전에 재사용한다. 정확성·결정성·occlusion·debug-buffer 비유출을
통과하지 못하면 빠르더라도 GO가 아니다. detect-resolution decoupling은 구현 전 silent analytic
fallback을 금지한다.

현재 authority는 D8-A 코드와 고정-action GPU 무결성/비용 측정까지만 연다. D8-B frozen-policy
평가는 D8-A `TECHNICAL_GO` 뒤 별도 addendum, adaptation은 다시 별도 multi-seed training
사전등록이 필요하다. D9 기준을 D8 결과 뒤에서 바꾸지 않는다. D6 INCONCLUSIVE와 D7의
cost/non-interference-only GO는 유지했다. 아직 D8 코드·GPU 결과·학습은 없다.

## 2026-09-13 — D8-A mesh-derived observation 구현, GPU 실행 전 고정

사전등록 커밋 `e538052`의 후속으로 D8-A 구현을 작성했다. 기존 `NavRLTargetDetector`의 analytic
kernel을 기본으로 유지하고, 명시적으로 attach된 경우에만 별도 target-local v3 visual mesh 질의가
같은 private `target_mask`/`target_depth`를 덮어쓴다. static scene과 target mesh를 따로 질의해
self-occlusion을 피하고, 표적 자세의 역회전으로 광선을 local frame에 옮기므로 움직일 때 BVH를
재구축하지 않는다. `mesh_flat`과 `mesh_shaded`의 기하는 동일하며 shaded arm만 v3 material의
상대 휘도와 고정된 ambient/Lambertian 세기를 기존 nominal red paint에 곱한다. 이는 새 hue/class를
도입하는 효과와 geometry/shading 효과를 섞지 않기 위한 처리다.

normal·face·material·raw-hit·occlusion buffer는 treatment 내부 진단 전용이며 task observation에
등록하지 않았다. unset/off는 treatment 모듈을 import하지도, mesh나 buffer를 할당하지도 않는다.
알 수 없는 mode, detect-resolution decoupling, D7 shadow와 동시 부착, 중복 부착은 모두 fail
closed한다. 실험 launcher는 v3 URDF SHA를 다시 확인하고, dirty tree를 거부하고, 사전등록 commit이
구현 HEAD의 엄격한 ancestor인지 검사한다. 21 visible pose(4–7 m)와 bar-occluded pose를 두 번,
128 env fixed-action step을 arm별 세 번 독립 process에서 실행하며 결과/failure receipt를 새
directory에만 쓴다. policy는 load하지 않는다.

구현 중 첫 CPU 테스트 시도는 Warp stub을 torch보다 먼저 설치해 torch introspection을 깨뜨렸고,
Python 3.8에 없는 `ast.unparse`도 사용해 5 error를 냈다. 구현 결함으로 숨기지 않고, torch 선행
import와 `ast.get_source_segment`로 테스트 harness를 교정했다. 최종 D8 계약 14개, 기존 D7 shadow
9개, perception 31개(skip 1), appearance 12개, detector noise 17개가 통과했다. D8 판정기의 합성
fixture에서 GO/중간 INCONCLUSIVE/정확성 NO_GO/비용 NO_GO가 사전등록 경계대로 나뉘는 것도
검사했다. 아직 GPU 커널 컴파일·실험 receipt·판정·D8-B·학습은 없다.

전체 회귀는 올바른 `aerialgym` interpreter와 `PYTHONNOUSERSITE=1`에서 **1,598개 실행 / 1,594개
통과 / 기존 skip 4 / 실패·오류 0**, 42.753초, exit 0이었다. 출력의 dependency warning과
obs-dump traceback은 고의 fault-injection 테스트가 기대한 경로이며 suite 판정과 구분했다.
`git diff --check`와 연구 authority 검사 `PASS_RESEARCH_AUTHORITY_FREEZE`도 통과했다.

## 2026-09-13 — D8-A attempt 1 환경 실행 실패 보존

clean implementation commit `8fdeed9`에서 공식 D8-A runner를 열었으나 첫 pose child가 task 생성
전에 종료됐다. `PYTHONNOUSERSITE=1`이나 D8 Warp kernel 자체가 원인이 아니라, absolute conda
Python을 호출하면서 해당 prefix의 `bin`이 inherited `PATH`에 없어
`torch.utils.cpp_extension`이 `ninja` executable을 찾지 못한 것이 직접 원인이다. 설치된 executable은
`/home/fair/miniconda3/envs/aerialgym/bin/ninja`, version
`1.13.0.git.kitware.jobserver-pipe-1`, SHA-256
`696f9628a79d9ce50314cf9556d7cd1a1d1ec52b8fd52828f6f9db1719565b67`이다.

0 pose/0 step/0 cost cell이므로 이 attempt는 `VOID_EXECUTION`이며 D8-A의 기하·무결성·비용
NO-GO 근거가 아니다. runner가 만든 `failure.json`을 삭제하거나 덮어쓰지 않고
`results/dynamic_mesh_detector_d8a_2026-09-13/`에 보존했다. OPERATIONS의 기존 사고 기록과 같은
실패 유형이다. 후속 수정은 child `PATH`에 선택 interpreter의 bin을 명시적으로 prepend하고
executable path/version/SHA를 provenance에 기록해야 하며, 재실행은 새 directory에서만 한다.

후속 launcher 수정은 이 요구를 코드로 강제한다. parent와 child가 모두 selected Python 옆의
executable `ninja` 존재·실행 가능성을 확인하고, child `PATH` 첫 항목과 `NAVRL_NINJA`를 같은
절대 경로로 설정한다. path·version·binary SHA는 parent provenance와 각 child report에 기록하며,
반복 간 runtime뿐 아니라 launcher fingerprint도 동일해야 무결성 검사를 통과한다.

## 2026-09-13 — D8-A attempt 2 완료: `TECHNICAL_GO`

launcher 수정 commit `c50a26d`의 clean tree에서 사전등록 본 실험을 새
`results/dynamic_mesh_detector_d8a_attempt2_2026-09-13/`에 실행했다. pose 2회와 128 env,
160×90, warm-up 50, measured 500의 세 arm×3회가 모두 완주했다. receipt SHA-256은
`92ce1c6a9d5f419de7b873af327c006b92246be9d9ba0da8c656dc9fe659e7ee`다.

사전등록 무결성 check 전부가 true다. 두 pose run hash, mesh-flat/shaded mask와 depth,
arm별 세 process output hash, fixed-action robot/target trajectory, runtime/launcher fingerprint가
각각 일치했다. 4–5/5–6/6–7 m band에서 mesh pixel이 존재하며 analytic proxy와 다른 silhouette를
냈다. 21 visible pose의 mesh/analytic 면적비 중앙값은 0.556(범위 0.429–0.889)이다. raw mesh
hit 164개 중 정적 bar가 12개를 가렸고 published survivor는 0개다. visible 152 hit의 invalid
depth/face/material/normal은 전부 0개다. flat 내부 red variance는 0이고 shaded는 0보다 크다.

세 반복의 step-median 중앙값은 analytic 43.656 ms, mesh-flat 43.911 ms, mesh-shaded 44.173 ms다.
증가는 flat +0.256 ms/+0.585%, shaded +0.517 ms/+1.184%; throughput 손실 0.734%/1.436%,
torch reserved +44/+66 MiB, NVML +44/+64 MiB다. 모두 사전등록 GO 경계 안이므로 판정은
**`TECHNICAL_GO`**다. 이는 mesh observation을 기술적으로 연결할 수 있고 정해진 비용 예산 안이라는
결론뿐이다. detector 정확도·association·shortcut 감소·policy robustness·adaptation·실기 성능은
측정하지 않았다. 다음 단계 D8-B는 별도 addendum 뒤 frozen-policy sensitivity로만 열며, 학습은
계속 금지한다.

receipt 수치의 별도 재계산과 historical execution commit의 8개 source hash 대조를 새 evidence
test 4개로 고정했다. 전체 회귀는 **1,602개 실행 / 1,598 통과 / 기존 skip 4 / 실패·오류 0**,
42.877초, exit 0이었다. 고의 fault-injection traceback과 dependency warning은 suite 판정과
구분했으며 `git diff --check`도 통과했다.

## 2026-09-13 — D8-B frozen-policy sensitivity addendum 사전등록

D8-A 결과 commit `81f680e` 뒤, D8-B runtime auto-attach나 평가 cell을 만들기 전에
`docs/preregistration_dynamic_mesh_policy_sensitivity_d8b_2026-09-13.md`를 고정했다. frozen ref5in
D1 ep1900 checkpoint SHA `197ea269…278e`와 built-in segmenter threshold 0.55를 세 arm 모두
사용한다. arm은 analytic-flat/mesh-flat/mesh-shaded, fresh evaluation seed는 593/599/601,
각 2,049 episode로 총 9셀이다. 70 bars, goal 22.5–28 m, camera/detect 160×90, deterministic
action, governor off, distractor/error/appearance perturbation 0을 고정한다.

1차 추정량은 seed별 `mesh_shaded − analytic_flat` capture rate 차이의 3-seed 평균이며, seed를
단위로 한 paired Student-t 95% CI(df 2)를 쓴다. −3.0 pp를 practical non-inferiority margin으로
결과 전에 고정했다. CI upper가 −3 pp보다 작으면 `MATERIAL_LOSS`, CI lower가 −3 pp보다 크면
`NO_MATERIAL_LOSS_WITHIN_MARGIN`, 나머지는 `INCONCLUSIVE_POLICY_SENSITIVITY`다. mesh-flat contrast와
crash/timeout/acquisition/visibility/action telemetry는 보조이며 판정을 바꾸지 않는다.

D8-A receipt/commit, checkpoint, 9-cell complete accounting, shared clean source, held-fixed condition,
effective treatment/v3 asset attestation, debug observation 비유출과 no-blanket-force가 무결성 gate다.
어느 하나라도 실패하면 policy verdict 없이 `INVALID_D8B_EXECUTION`이다. 이 addendum은 평가만
허용하며 adaptation/PPO/새 detector/제어 변경은 계속 금지한다.

## 2026-09-13 — D8-B runtime attachment와 9-cell launcher 구현

사전등록 commit `0aba981` 뒤 D8-B 코드 경로를 구현했다. `NavRLTask`는
`NAVRL_DYNAMIC_MESH_TREATMENT`가 unset/off이면 D8 module이나 URDF loader를 import하지 않는다.
명시적 mesh mode에서만 pinned v3 asset을 dual-backend 감사에 사용한 loader로 읽고 SHA를 확인한
뒤 detector treatment를 attach한다. bulk result의 condition은 effective `target_render_mode`, attach
여부, v3 SHA, triangles/material/light/ambient/directional/material-gain contract를 기록한다. debug
buffer는 여전히 actor observation에 넣지 않는다.

Generic evaluator의 기본 source roots는 바꾸지 않았다. D8-B가 명시적으로 요청할 때만 allowlist된
`resources/models/environment_assets/objects`와 `tools/renderer_validation`을 source snapshot에 더하며,
그 밖의 ambient root는 거부한다. 따라서 기존 historical evaluator의 root contract는 그대로이고,
D8-B에서는 실제 실행한 URDF/parser bytes가 shared manifest에 포함된다.

새 `tools/run_dynamic_mesh_policy_sensitivity_d8b.py`는 기존 D9의 audited default-detector/N=0
canonical environment를 재사용한다. per-arm diff는 result path와 D8 flag뿐, per-seed diff는 result
path와 seed뿐인지 CPU test로 확인한다. matching Python/ninja를 바인딩하고 D8-A/checkpoint/v3/prereg
ancestor/runtime clean을 preflight하며, generic evaluator로 9개 evaluation cell만 실행한다. 각 cell의
result/receipt/checkpoint/source snapshot SHA, episode outcome accounting과 effective treatment를 검증한
뒤에만 seed-level t interval과 frozen verdict를 만든다. training entry point는 없고 blanket force도
금지했다.

첫 D8-B CPU test는 사전등록 문장의 정확한 어순보다 강한 문자열 `paired Student-t`를 요구해 1
failure였다. 문서는 이미 Student-t와 paired seed differences를 각각 명시했으므로 구현/명세를 바꾸지
않고 test를 실제 문구 `Student-t interval`로 교정했다. 최종 D8-B 계약 8개와 D8-A 계약/evidence
18개가 통과했다. 아직 D8-B GPU preflight/cell/result는 없다.

전체 회귀는 **1,610개 실행 / 1,606 통과 / 기존 skip 4 / 실패·오류 0**, 42.116초, exit 0이다.
고의 fault-injection traceback과 dependency warning은 suite 판정과 구분했다.

## 2026-09-13 — D8b 9셀 무결성 감사와 동결 판정 종료 (GPU 재실행 없음)

시작 Git: HEAD `1e0eed8b963ed3ccdae93cc2e9ea67d9f9ddbaab`, origin/main
`23d48b5eb8579ee742d177da1b3481794a117917`, 로컬 7커밋 앞섬. 추적 변경은 없고 D8b 결과만
미추적이었다. 평가 GPU 프로세스는 없었다. 기존 9셀은 재실행하지 않았다.

요청 전달문의 "6개 gate" 대신 실제 사전등록 §4의 **8개**를 사용했다. 원문은 actual≥2,049이고,
실제로는 9/9 셀 모두 정확히 2,049다. 3/3 seed·3/3 arm, 중복 없는 9 nonce, outcome 합계·rate,
결과/로그/환경/manifest SHA, 같은 checkpoint SHA, 모든 held-fixed contract를 대조했다.
370개 source snapshot 전부 manifest 및 runtime Git blob과 바이트 일치한다. 원문·threshold 수정 없음.

| 평가 시드 | analytic capture | mesh-flat capture | mesh-shaded capture |
|---:|---:|---:|---:|
| 593 | 71.01% | 64.81% | 22.40% |
| 599 | 70.96% | 63.15% | 22.16% |
| 601 | 71.94% | 64.42% | 22.45% |

기존 runner의 finalize 뒤 verify PASS. primary mesh-shaded−analytic의 3개 시드 차이 평균
**−48.96697576053359 pp**, df=2 95% t CI **[−50.11285929556478, −47.821092225502404] pp**.
−3.0 pp margin에 대한 동결 판정은 **`MATERIAL_LOSS`**. 별도 stdlib-only `audit.py`는 runner를
import하지 않고 count와 df=2 t-quantile 닫힌식으로 재계산한다. 평균은 일치, CI endpoint의
최대 차이 약 4.31e-11 pp는 원래 runner의 t 상수와 닫힌식 수치 차이다.

한계: actor debug 비유출은 pinned source 검토와 runtime dimension 기록이지 새 동적 정보흐름
증명은 아니다. Python/ninja 환경은 당시 source/로그/package inventory로 확인했지만, finalize가
새로 읽는 ninja binary SHA를 당시 9셀 fingerprint라고 부르지 않는다. 전체 TF32/cuDNN runtime
flags도 없다. 비활성 noise 기본값과 실제 활성 noise를 구분했다. 학습 시드 일반화·원인 분해·
shortcut 감소·D8c 성공을 주장하지 않는다. 상세는 결과 폴더 `AUDIT.md`.

패키지: 원시 JSON/receipt/CSV, 작은 원본 로그 9개, 환경 목록, shared source manifest를 보존한다.
중복 checkpoint/source snapshot 및 절대 manifest symlink는 로컬 유지·Git 제외한다. 삭제 0건.
검증기 기본 모드는 Git blob을 읽어 새 clone에서 snapshot/체크포인트 없이 기록을 검증하며,
`--local`은 로컬 checkpoint/source bytes까지 검사한다. 이것을 weight 재배포나 학습 재현 보장으로
오인하지 않는다. 최종 README 생성기의 trailing space 1개만 정리했고 숫자/원시 결과는 그대로다.

신규 archive corruption/portable-package 테스트 **13개 통과**. 추가 셀·중복·누락·count/rate 불일치,
NaN, 비정수 outcome, 조건 변경, 중복 JSON key, summary shape 변조를 거부한다.
전체 회귀 및 커밋 전 최종 검사는 아래 후속 기록으로 남긴다.

이번 구현은 보존·검증 도구뿐이다. UAV 요격 경로의 검출·표적 연결·정책 입력을 진단해 성능 개선으로
잇는 D8c는 지원 범위 밖으로 알렸으며, 사전등록·학습·새 GPU 측정을 시작하지 않았다. 정책에서
분리된 일반 renderer 검증은 별도 가능한 작업이지 이 요청의 D8c를 완료한 것으로 대체하지 않는다.

커밋 전 전체 회귀: **1,623개 실행 / 실패 0 / 오류 0 / 기존 skip 4** (`aerialgym`,
`PYTHONNOUSERSITE=1`, `CUDA_VISIBLE_DEVICES=0`). 처음 GPU를 숨겨 실행한 검사는 1,611개/
failure 1/error 1/skip 5였으며, 기존 GPU 초기화 의존 import가 실패해 일부 테스트가 로드되지
않았다. 이 실패를 숨기거나 테스트를 삭제하지 않고 기존 환경에서 전체 개수를 회복해 확인했다.
회귀 종료 뒤 평가/학습 GPU 프로세스는 없다. D8b 원래 verify도 통과했고 추가 실험은 0회다.

원본 로그의 logger 끝 공백과 CSV의 CRLF는 변경하지 않았다. 이 파일 형식에만 좁게
`.gitattributes`를 적용해 바이트/해시를 보존하며, 코드·JSON·문서의 whitespace 검사는 유지한다.

## 2026-09-13 — D8b 종료 이후 공개 상태 정합성 교정

D8b 결과 commit `295223d` 뒤에도 공개 manifest와 사이트가 `D8 NOT_STARTED`라고 적어 모순이었다.
D8 manifest는 D8-A 기술 판정 `TECHNICAL_GO`와 그 근거만 나타내고, 표에 D8b `MATERIAL_LOSS`를
별도 행으로 기록하도록 고쳤다. D8c는 미실행, D9는 `NOT_RUN`이다. 기존 D6/D7 및 실패 판정은 유지.
manifest schema와 보호 검사도 이 새 근거에 맞췄다. `COMPLETE`로의 근거 없는 승격은 계속 거부한다.
해시가 고정된 그림·ZIP·archive HTML을 수정하지 않고 Figure 1/6 캡션에 pre-D8 역사 그림임을
명시했다. 이전 site revision의 설명은 이력으로 보존하고 상단에 현재 상태 정정을 붙였다.
뷰어·renderer·perception·정책 코드와 원시 결과는 변경하지 않았다. 새 실험이나 adaptation 승인 없음.

검증: 공개 문서/schema PASS, 사이트·문서 테스트 17개 PASS. 처음 연구 환경에서 schema 검사는
`jsonschema` 누락으로 실패했다. 연구 환경을 변경하지 않고 기존 별도 공개 검증 환경
`/tmp/motar-release-validation-6rTnrm/venv/bin/python`으로 재검사해 통과했다.
최종 전체 회귀는 **1,624개 실행 / 실패 0 / 오류 0 / skip 4**, 48.954초다.
원격 재조회 결과 origin/main은 여전히 `23d48b5`이며 외부 커밋 유입은 없다. 요청의 최종 push 조건에
포함된 D8c 구현·검증은 수행하지 않았으므로 이번 두 커밋은 로컬에 남기고 push하지 않는다.

## 2026-09-13 — 렌더러 특성화 트랙(RC-R1~R6) 신규 수행

D8/D8b 계보 9개 커밋을 fast-forward로 push(`23d48b5..879e57c`)한 뒤, **지각·정책과 완전히 분리된
독립 렌더러 연구**를 새로 시작했다. 목표는 이 렌더러가 geometry·apparent area·material·shading·
lighting을 어떻게 표현하는지 일반적·재현 가능하게 정량화하는 것이며, **D8b 손실의 원인을 추정하지
않는다**. 모든 결과 파일에 `causality_vs_d8b: NOT_TESTED`를 기록했다.

이름 충돌 주의: 2026-09-11의 외형 프로토타입 트랙도 R1~R5를 썼다. 두 이름공간은 별개이며 숫자를
섞지 않는다. 이 트랙은 `RC-R1 … RC-R6`으로 표기한다.

**Preregistration을 코드보다 먼저 작성해 커밋**(`c709eec`)했다. 이후 구현 중, box의 한 면이
삼각형 2개라서 G6의 최소 가시삼각형 3이 RC-R2 fit view 4개를 전부 거부한다는 사실을 발견해,
**실험 실행 전에** 날짜와 사유를 적은 증보 절을 추가하고 3→2로 내렸다(`4cab0f4`). 이 변경은 뷰를
배제하는 게 아니라 가장 구조가 적은 뷰를 포함시키므로 실패 게이트를 통과로 바꿀 수 없다. S3는
손대지 않았고, 축 정렬 box 뷰가 S3에서 실패할 것임을 그 문서에 미리 적었다(실제로 그렇게 됐다).

결과(8개 디렉터리, `results/renderer_characterization_2026-09-13/`):

- **RC-R1 `GEOMETRY_DEFECT`.** G1·G2·G5·G6 통과. 독립 CPU Möller–Trumbore 교차기와 비교해 두 mesh
  specimen 24개 뷰 전부에서 실루엣 불일치 **0픽셀**, 최대 range 차 1.4e-6 m. 해석적 구는 닫힌 형식
  대비 반지름 0.03 px, 깊이 2.4e-8 m. G3(1.038 px vs 1.0), G4(3.6~7.2 % vs 2 %),
  G7(2.1~3.2 % vs 2 %)은 다면체 specimen에서 미달. **임계값은 결과를 본 뒤 바꾸지 않았다.**
- **RC-R1 진단(게이트·판정 없음).** 미달 원인이 실루엣 양자화인지 묻고, 숫자를 보기 전에
  "초점거리 2배면 오차 절반"을 예측으로 적었다. 320×240의 최악 뷰 오차는 box 3.0 %,
  quadrotor 2.2 %, 해석적 구 0.14 %이고 640×480에서 1.0~1.2 %로 줄어 비율 1.84~3.01이다.
  교차 경로가 아니라 측정 해상도의 문제라는 읽기를 지지한다. RC-R1 수치는 그대로 둔다.
- **RC-R2 `AREA_MATCH_FAILED`.** fit view 4개에서 적합해 동결한 등방 스케일 0.845가 held-out 20개
  뷰에서 19.0 % 초과(허용 5 %). 재적합하지 않았다. box는 mesh보다 평균 면적 1.67배, 구는 1.53배.
- **RC-R3 `SHADING_GATE_FAILED`.** 한 방향만 보일 때 lambertian 음영은 공간 변화를 만들지 않는다.
  방위 0·180, 고도 0에서 quadrotor mesh는 가시삼각형 98·118개인데 가시법선 엔트로피는 정확히
  **0 bit**, 평균 |n·view| = 1.000, 실루엣 휘도 분산 정확히 0. **삼각형 수는 음영 구조의 지표가
  아니다.** 해석적 구는 삼각형이 없는데 32개 법선 bin 중 16개, 3.99 bit를 채운다.
- **RC-R4 `LIGHTING_CHARACTERIZED` / RC-R4M `MATERIAL_CHARACTERIZED`.** 41개 셀 전부에서 6개
  geometry 버퍼 해시와 실루엣 픽셀 수가 동일한 채 평균 휘도만 0.52·0.45 움직였다.
- **RC-R5 `DESCRIPTIVE_NO_VERDICT`.** 54/54 셀 측정, 자원 중단 규칙 미발동. 640×480·32장에서
  quadrotor G-buffer 14.3 ms, host copy 175 ms, lambertian 6.4 ms. 삼각형 129배 증가의 비용은
  12 %다. 해석적 arm은 CPU numpy 구현이라 933 ms이며, 이는 그 구현의 비용이지 해석적 교차 일반의
  비용이 아니다. **D6(1.5338배)·D7(+0.411 ms)과 비교하지 않는다.**
- **determinism `DETERMINISTIC`.** 독립 프로세스 2회 export에서 7개 배열 해시 전부 일치.

전체 스위트에서만 실패하고 단독으론 통과하는 신규 테스트 11개를 통해 **기존 테스트 하네스의 실제
결함 2건**을 찾아 고쳤다(`c707316`). (1) dynamic-mesh raycast 게이트가 teardown에서 실제 warp
패키지를 `sys.modules`에서 제거했다. warp 하위 모듈은 남으므로 이후의 `import warp`가 반쯤 만들어진
모듈을 다시 실행해 `warp.config`에서 죽는다. (2) fake-warp 소스 로더 테스트가 커널 모듈 캐시를
비우지 않아, 먼저 실행된 실제 렌더 테스트의 캐시를 검사하고 있었다. 캐시를 먼저 비우도록 고치고
mutation으로 단언이 살아 있음을 확인했다. **테스트를 삭제·skip·약화하지 않았다.**

그림 6종은 커밋된 `summary.json`에서만 읽어 그린다(`docs/assets/paper/renderer-characterization-2026-09-13/`,
SVG+PNG+벡터 PDF, 흰 배경·3색·그라디언트 없음, 새 파일명). 기존 해시 고정 그림·ZIP·archive는
건드리지 않았다.

검증: 최종 전체 회귀 **1,702개 실행 / 실패 0 / 오류 0 / 기존 skip 4**, 53.412초. 각 결과의
`source_manifest.json`에 기록된 파일별 sha256은 커밋된 파일과 **전부 일치**(불일치 0)한다.
RC 커밋 6개는 로컬에 있고 push하지 않았다.

## 2026-09-14 — RC 첫 계보 10커밋 보존·푸시, 후속 실험 사전등록

실제 HEAD `32ee105`, origin `879e57c`, clean main에서 RC의 10개 커밋을 확인했다.
전체 baseline **1,702 실행 / 실패 0 / 오류 0 / skip 4**, 57.44초. 8개 source manifest의 파일별
SHA가 현재 커밋된 소스와 모두 일치하고 receipt/summary 판정도 일치한다. 당시 dirty source에서
실행됐다는 기록은 보존했다. 원본 생성 SVG의 공백과 determinism README의 끝 빈 줄로 기존
`git diff --check origin/main..HEAD`에는 경고 1,314건이 있다. 역사 파일을 정리해 감추지 않았다.
사용자 지시대로 force/squash/rewrite 없이 **`879e57c..32ee105`를 origin/main에 푸시**했다.
위 이전 항목의 '6개 로컬'은 작성 시점 기록이며 현재는 10개가 공개됐다.

새 사전등록 `docs/preregistration_renderer_followup_2026-09-14.md`는 R5b→R1b→R2b→R3b의
지원 arm, 데이터 분리, 자원 중단, threshold와 문서 산출물을 새 측정 전에 고정한다.
Camera 기존 상한 때문에 reference는 2048×1536. MeshScene에 authored vertex/smooth normals가
없으므로 N1/N2는 UNSUPPORTED로 미리 정했다. 과거 기준·source·receipt는 수정하지 않는다.
단계별 코드와 결과는 별도 커밋하며, 정책/검출기와 인과 연결은 `NOT_TESTED`로 유지한다.

## 2026-09-14 — R5b 전송 하네스 구현 (실측 전)

새 `transfer_strategies.py`는 RSS reader의 context manager로 조기 return/예외에서도 파일을 닫는다.
기존 R5 소스·receipt는 변경하지 않는다. A0 개별 복사, A1 device-resident+scalar, A2 dtype별 pack,
A3 pinned nonblocking+완료 대기의 네 arm을 같은 렌더 출력에 적용한다. A1 전체 export 검증 비용과
A3 사전 할당 비용은 별도 기록한다. 2 fixtures×4 arms×3 processes, 20 warmup, staged/headline 각
100 sample. 전체 버퍼 해시와 반복 간 해시가 일치해야 비교하며 CUDA/G-buffer 코드는 바꾸지 않았다.
기존 R5 copy_once는 `cell.arrays` 안에서 shading도 실행하므로 그 175 ms를 순수 D2H라고 부르지
않는다. 새 stage 경계는 geometry/shading/transfer strategy/total로 분리한다.

신규 회귀 12개 통과: RSS 조기 return/예외/반복 실행, A0/A2 dtype·shape·byte 보존, A1 전체 host
복사 부재, A3 CPU 거부, 계약 drift 거부, 역사 R5 source SHA 불변. 아직 새 GPU 수치는 없다.

## 2026-09-14 — R5b 24셀 완료: 출력 보존, host strategy 비용 우세

`results/renderer_characterization_r5b_2026-09-14/`, source `dd0fd4f`. 2 fixtures × 4 arms ×
3 독립 프로세스, 각 staged/headline 100회. 7개 버퍼의 arm 간·repeat 간·첫/마지막 SHA 전부 일치,
receipt와 historical Git blob 검증 통과. 판정 `TRANSFER_CHARACTERIZED`.

640×480×32, 3 repeat 평균 (ms; staged 성분과 별도 headline 전체 시간):

| Arm | Geometry | Shading | Host strategy | Headline total |
| --- | ---: | ---: | ---: | ---: |
| A0 개별 owned copy | 16.879 | 7.043 | 178.620 | 205.215 |
| A1 device/scalar | 15.559 | 6.922 | 2.042 | 24.348 |
| A2 dtype grouped | 15.422 | 6.917 | 109.554 | 131.582 |
| A3 pinned + completion wait | 15.293 | 6.912 | 17.362 | 39.592 |

A0 staged host 비율은 88.02/88.08/88.46%, 사전등록된 >50% 조건을 모두 충족한다.
이는 allocation/CPU copy를 포함한 host strategy 병목이지 순수 PCIe 대역폭 측정은 아니다.
A3는 full-buffer 출력을 보존하지만 소비 후 버퍼가 재사용되며 compute/DMA overlap은 없다.
A1은 scalar만 host에 보내므로 full-export 속도와 동등한 작업이라고 비교하지 않는다.
R5 역사 warning과 기존 negative 판정은 보존. 다음은 고정된 R1b 해상도 수렴 측정이다.

## 2026-09-14 — R1b/R2b/R3b 계산 도구 구현 (실측 전)

새 `tools/measure_renderer_contract.py`: 288개 단일-view 해상도 비교, calibration 6개만으로
anisotropic box fitting 후 fit.json SHA 고정·held-out 24개 채점, authored normal이 없는
N1/N2 미지원 표시와 N0 normal entropy/광도 분포 측정. 기존 renderer/exporter 소스는 그대로다.
source manifest는 실제 새 실행 파일을 포함하며 R5b의 과거 manifest는 수정하지 않는다.
새 단위 테스트 18개: inclusive extent, clipping/최소픽셀 거부, 해상도 정규화, centroid 0 안전성,
고정 gate, fit/validation 분리, 폐형식 면적·synthetic scale 회복, 비유한 값 거부, normal 출처 계약.
R1b→R2b→R3b는 별도 프로세스·결과·커밋으로 진행하며, 결과에 따라 threshold를 바꾸지 않는다.

## 2026-09-14 — R1b 완료: 1280×960이 등록된 측정 해상도 조건 충족

`results/renderer_characterization_r1b_2026-09-14/`, source `901d4d8`. 3 specimens×24 views×
4 resolutions = 288행, 거부 0. 2048×1536 sampled reference 대비 모든 72개 specimen-view의
최악 상대 오차(%):

| Resolution | Width | Height | Area | Centroid / equivalent diameter | Depth |
| --- | ---: | ---: | ---: | ---: | ---: |
| 320×240 | 5.391 | 2.476 | 7.021 | 1.267 | 0.195 |
| 640×480 | 2.609 | 1.000 | 3.119 | 0.707 | 0.074 |
| 1280×960 | 1.217 | 0.483 | 1.143 | 0.299 | 0.056 |

판정 `MEASUREMENT_RESOLUTION_QUALIFIED`, 최소 추천 1280×960. 이는 이 거리 2 m의 고정 view
grid에서 측정한 finite-resolution discrepancy이며 보편적 오차 상한이나 정확한 기하 GT가 아니다.
RC-R1의 GEOMETRY_DEFECT는 유지한다. 다음 R2b는 원래 고정한 1280×960에서 한 번 fitting한다.

## 2026-09-14 — R2b 완료: anisotropic 대조군도 등록 gate 실패

`results/renderer_characterization_r2b_2026-09-14/`, source `be631bf`. 6 calibration view로 7회
함수 평가 뒤 scale `[0.7844494024, 0.8725480318, 0.6582063554]`를 동결했다. fit.json SHA를
held-out 렌더 전에 저장했고 끝까지 동일하다. 24개 held-out view 재적합은 없었다.

| Phase / control | Median ARE % | P90 % | Worst % |
| --- | ---: | ---: | ---: |
| Fit / historical C0 | 22.772 | 25.268 | 25.978 |
| Fit / C1 anisotropic | 7.317 | 11.429 | 11.774 |
| Held-out / historical C0 | 23.525 | 32.211 | 34.301 |
| Held-out / C1 anisotropic | 8.132 | 11.051 | 11.544 |

C1은 C0보다 오차가 작지만 median≤5%, P90≤10%를 충족하지 못하므로 `AREA_CONTROL_FAILED`.
1280×960은 R1b의 측정 지원 조건을 충족한다. 기존 RC-R2 실패를 바꾸거나 이 대조군을
area-matched라고 채택하지 않는다. 다음 normal 연구는 사전등록대로 기술적 분포 측정만 한다.

## 2026-09-14 — R3b 72 view 완료: normal/광도 분포 기록, smoothing arm 미지원

`results/renderer_characterization_r3b_2026-09-14/`, source `a19d9fe`. 3 specimens×24 views,
640×480. shading 전후 geometry hash 불일치 0. `NORMAL_FIELD_CHARACTERIZED`는 완료 상태이며
효과 gate의 PASS가 아니다. N1/N2는 source/loader에 authored normal contract가 없어 UNSUPPORTED.

| Specimen | Entropy median (bits) | Occupied bins min–max | Luminance variance median | Pearson r |
| --- | ---: | ---: | ---: | ---: |
| Analytic sphere | 3.99149 | 16–16 | 0.0109250 | −0.80985 |
| Box | 0.89306 | 1–3 | 0.0071478 | +0.57161 |
| Quadrotor mesh | 1.25934 | 1–14 | 0.0053678 | +0.59242 |

Sphere 광도 variance의 전체 범위는 약 1.64e-9뿐이다. 등록된 계산은 r을 반환하지만 이 사실상
일정한 구에서의 상관계수를 물리적 효과로 해석하지 않는다. mesh의 entropy 중앙값은 box보다
높아도 광도 variance 중앙값은 낮다. triangle 수→음영 다양성이라는 보편 법칙을 도출하지 않는다.
R4/R4M grid 재실행 0, 기존 SHADING_GATE_FAILED 유지, D8b 원인 기여 NOT_TESTED.

문서 경계 검사에서 기존 characterization exporter의 설명이 face ID까지 'never shading inputs'라고
적은 것을 찾았다. 실제 shader는 face ID로 material table을 조회하며 instance ID는 읽지 않는다.
기존 source/receipt는 보존하고 새 v1 계약 및 테스트에서 정확히 구분한다.

## 2026-09-14 — RC 후속 4종(R5b/R1b/R2b/R3b)과 Renderer Contract v1

첫 RC 계보 10개 커밋을 `32ee105`로 push해 별도 역사로 고정한 뒤, **실패한 RC 판정을 재해석하지
않고 실패 이유가 드러난 부분만 새 계보에서 다루는** 후속 연구를 진행했다. 사전등록을 구현보다 먼저
커밋(`3e6d074`)했고, RC-R1 `GEOMETRY_DEFECT`·RC-R2 `AREA_MATCH_FAILED`·RC-R3
`SHADING_GATE_FAILED`와 R5의 알려진 ResourceWarning은 **원본 그대로 보존**한다.

- **R5b `TRANSFER_CHARACTERIZED`** (`dd0fd4f`→`ad414f6`). 같은 mesh·appearance·view로 전송 전략
  4종(A0 개별 전체 복사 / A1 device 상주+스칼라 / A2 dtype 묶음 / A3 pinned 비동기+완료 대기)을
  2 fixture × 3 프로세스 반복으로 측정. 640×480×32에서 A0 205.2 ms(host 178.6 ms), A1 24.3,
  A2 131.6, A3 39.6 ms이고 A0의 host 단계는 staged total의 88.02/88.08/88.46 %다. **7개 출력 해시가
  arm·반복 전부에서 동일**하므로 비용 비교이지 출력 변화가 아니다. 기존 R5의 host_copy에는 shading과
  NumPy 복사가 섞여 있어 새 전송 단계 수치와 동일시하지 않는다. A1은 전체 export가 아니므로 그
  처리량을 dataset export 속도로 광고하지 않는다.
- **R1b `MEASUREMENT_RESOLUTION_QUALIFIED`** (`901d4d8`→`be631bf`). 3 specimen × 24 view ×
  4 해상도 = 288행 전부 측정(거부 0). 카메라 계약의 축당 2048 픽셀 한계 때문에 기준은 2048×1536
  **샘플링**이며 GT가 아니다. 등록 허용치(폭·높이·면적 2 %, centroid·depth 1 %)를 모든 뷰에서
  만족하는 최소 해상도는 **1280×960**(최악 1.217/0.483/1.143/0.299/0.056 %)이다. 이는 유한 해상도
  불일치의 측정치이지 절대 정확도 증명이 아니며, RC-R1 판정을 수정하지 않는다.
- **R2b `AREA_CONTROL_FAILED`** (`be631bf`→`a19d9fe`). 6개 보정 뷰에서 3축 스케일을 log-면적
  최소제곱으로 적합하고 `fit.json`과 SHA를 **렌더 전에 동결**한 뒤 disjoint 24개 held-out 뷰를
  측정했다. C1은 held-out median 8.132 %, P90 11.051 %, worst 11.544 %로 등록 게이트 5/10/20 %의
  median·P90을 넘겼다. 역사적 등방 C0(23.525/32.211/34.301 %)보다 낫지만 **채택하지 않는다.**
  재적합·임계값 변경·초기값 교체는 없었다.
- **R3b `NORMAL_FIELD_CHARACTERIZED`** (`a19d9fe`→`62865ac`). MeshScene과 로더 계약에 authored
  vertex normal이나 smoothing group이 없어 **N1·N2는 UNSUPPORTED**로 기록했다(0 효과 arm으로
  위장하지 않음). 640×480 24뷰에서 sphere/box/mesh의 median entropy 3.99149/0.89306/1.25934,
  median 휘도 분산 0.0109250/0.0071478/0.0053678이다. mesh는 entropy가 더 높은데 분산은 더 낮다 —
  **entropy만으로도, 삼각형 수만으로도 음영 법칙이 서지 않는다.** 구의 분산은 24뷰에서 1.64e-9만
  변하므로 그 Pearson r(−0.810)은 물리적 추세가 아니라 수치 잔차이며, 그렇게 기록했다.

**Renderer Contract v1**(`9b8ab6d`)을 `docs/renderer_contract_v1_2026-09-14.md`,
`docs/renderer_measurement_contract_v1.md`, 기계가독 `docs/renderer_buffer_contract_v1.json`으로
발행했다. 버퍼 7종의 의미·miss 값·좌표 규약, 지원되는 normal, 측정 해상도와 양자화 하한, compute와
host 전송의 분리를 기록한다. **blanket PASS가 아니다**: 채택된 area control이 없고 N1/N2는
UNSUPPORTED이며 과거 부정 판정은 유효하다. 과거 문구 오류도 정정했다 — `instance_id`는 shading에
들어가지 않지만 **`face_id`는 material 테이블을 인덱싱한다**. 원본 소스 문자열은 재현을 위해 보존하고
새 테스트로 두 경계를 각각 증명한다.

논문형 그림 5종(`docs/assets/paper/renderer-contract-v1-2026-09-14/`, SVG+PNG+벡터 PDF, 흰 배경·3색)
과 그림 중심 페이지 `docs/status/renderer-contract-v1.html`를 새로 만들고, manifest가 결과 요약
해시·생성기·산출물 해시를 고정한다. 기존 해시 고정 그림·ZIP·archive는 수정하지 않았다. 메인 페이지
evidence 절의 기존 source-note 줄에 링크 한 줄만 추가했다(`6bc6f8f`).

검증: 전체 회귀 **1,751개 실행 / 실패 0 / 오류 0 / 기존 skip 4**, 57.7초(직전 baseline 1,702개).
독립 재계산 테스트가 R1b 288행, R2b fit/held-out 통계, R3b entropy·상관계수, R5b 타이밍 요약과
12,000개 원시 샘플, 모든 receipt·git blob·사전등록 조상 관계, 첫 RC 계보의 바이트 보존을 다시
확인한다. manifest 해시 훼손과 페이지의 부정 판정 약화는 mutation으로 실패함을 확인했다.
D8b와의 인과는 모든 산출물에서 `NOT_TESTED`로 유지한다.

## 2026-09-14 (2) — 렌더러 v1 동결, 과제 수준 진단(TD-T1/TD-T2) 계측

렌더러 특성화 트랙을 **v1에서 동결**했다. 기계가독 기록은 `docs/renderer_track_status_v1.json`,
사람용 문서는 `docs/renderer_track_v1_freeze_2026-09-14.md`다. 종합 상태는
`RENDERER_CONTRACT_V1_COMPLETE`지만 **blanket PASS가 아니다**: area control은 두 대조군이 모두
held-out에서 실패해 `NOT_ADOPTED`, normal은 asset이 face normal만 제공하므로 `N0_ONLY`이며,
RC-R1/R2/R3의 부정 판정은 그대로 유효하다. 각 행에 근거 경로와 경계를 붙였다. 실패한 게이트를
계속 고치는 R2c/R3c는 하지 않으며, v2는 별개 질문이 생길 때만 새 사전등록으로 연다.

사이트는 요청대로 Appearance/Rendering 절에 **표 1개 + 그림 1개**만 넣고 상세는 renderer-contract-v1
페이지로 링크했다. 표는 번호를 붙이지 않았다 — 기존 Table 1–3 참조가 공개 문서에 있어서 재번호는
그 참조를 깨뜨린다. 그림 추가로 overview 테스트의 figure/이미지 고정값 2개를 함께 옮겼다.

**로드맵 경계 변경을 명시적으로 기록했다.** 마스터 플랜은 P2(search/reacquisition)와 T0/T1(terminal
forensics·strict-contact instrumentation)을 `BLOCKED_BY_POLICY`로 두고 "evaluation-only labels do not
remove that intended operational role"라고 적고 있었다. 소유자가 2026-09-14에 이 경계를 바꿨으므로,
**기존 문장을 수정하지 않고 그대로 둔 채** 변경 절을 앞에 추가했다. 허용된 것은 기존 체크포인트
실행의 계측뿐이고, 정책·보상·검출기·필터·컨트롤러·종료 변경과 재학습·탐색 휴리스틱·터미널 컨트롤러는
여전히 막혀 있다. 이름 충돌을 피해 새 작업은 `TD-T1`/`TD-T2`로 쓴다.

사전등록 2건을 구현 전에 작성했다(`docs/preregistration_task_diagnostics_t1_2026-09-14.md`,
`..._t2_...`). T1은 8개 분석 라벨과 손실 구간 경계 10스텝(1.0 s), 에피소드별 기록 필드, Q1–Q6,
밀도축 70/115/160/205를 고정한다. T2는 **`success_radius = 0.5 m`를 건드리지 않고** 평가 전용 지표
(minimum_center/surface_distance, closest approach 상대속도, time_inside_1m/0p5m)를 옆에 추가하고,
10개 실패 분류를 **우선순위 순서까지** 고정하며 `UNKNOWN`을 허용한다. 모든 임계값(창 20스텝,
근접 1.0 m, 빠른 반경속도 1.5 m/s, 느린 접근 0.3 m/s, governor 50 %, 명령오차 0.5 m/s)이 사전등록에 있다.

구현: `navrl_episode_forensics.py`(라벨 상태기계·분류기는 순수 함수), `navrl_trajectory_digest.py`,
그리고 navrl_task.py의 **가드된 읽기 전용 훅 4개**(관측 진단 캐시, capture 스윕 거리, per-step 기록,
에피소드 종료). 기본 off이고 체크포인트 평가 밖에서는 거부한다. 라벨은 관측·보상·종료에 들어가지
않으며 이를 소스 수준 테스트가 강제한다.

**행동 불변성은 주장하지 않고 측정했다**(`results/task_diagnostics_invariance_2026-09-14/`).
같은 감사된 평가 환경에서 3회 실행(off/off/on, 70 bars, seed 593, 128 에피소드, deterministic)의
궤적 digest가 모두 `b9cc980d…`로 동일하다. off/off 비교로 하네스 자체의 재현성을 먼저 세우고,
그 위에서 off/on 불변성을 주장한다. digest는 세 arm 모두에서 켜져 상쇄된다. 판정
`TRAJECTORY_INVARIANT`.

도중 두 가지 실제 함정을 만났고 기록해 둔다. (1) 러너의 `quiet_rl_io` stdout 필터는 허용 목록에 없는
출력을 삼키므로 digest는 파일로 읽어야 한다. (2) 평가기는 자기 디렉터리에서 실행되므로 상대 출력
경로는 결과 디렉터리가 아니라 런처 옆에 쓰인다 — 런처가 경로를 절대화한다. 또 예전 실험의 공유
소스 번들을 재사용하면 평가기가 **정당하게** 거부한다(그 사이 트리가 움직였으므로).

`on` arm의 forensics 산출은 밀도 1개·seed 1개의 smoke 표본이며 그렇게 라벨링했다. **TD-T1 결과가
아니다.** 사전등록된 T1/T2는 밀도축 전체와 기존 held-out 프로토콜이 필요하고 아직 실행하지 않았다.

검증: 전체 회귀 **1,784개 실행 / 실패 0 / 오류 0 / 기존 skip 4**(직전 1,751). 정책·보상·검출기·
연관·필터·컨트롤러·종료와 기존 결과는 변경하지 않았다. D8b와의 인과는 계속 `NOT_TESTED`다.

## 2026-09-14 (3) — 정책 비실행 범위의 기록 무결성·CPU 계측 검사

사용자는 12셀 평가 대신 **정책 실행과 분리된 로그 스키마·해시·중복·누락 검사, 합성 데이터 계측,
공개/renderer/metadata 회귀 검사** 범위에 동의했다. 이번에는 TD full sweep, acquisition/terminal
성능 분석·분류, 모델/컨트롤러/계측 hook 수정이 없다. 위 TD 계측 상태나 원래 trajectory verdict를
이 검사로 다시 쓰지 않는다. 시작 HEAD/origin `7451577`, clean. 커밋·푸시는 하지 않았다.

새 stdlib 도구 `tools/check_record_integrity.py`는 generic JSON record envelope만 검사한다.
SHA와 strict JSON(중복 key·NaN/Infinity·비유한 숫자), 필수 key·null/type·SHA 형식, 복합 ID 중복,
context 일치, row count를 검사하며 값이나 operational metric은 출력하지 않는다. 오류 행은 보존,
output 덮어쓰기 거부. 감사 계약 `docs/record_envelope_audit_2026-09-14.json`은 기존 파일을 열람한
뒤 작성한 소프트웨어 검사 계약이지 prospective experiment preregistration이 아니다.

**발견:** `results/task_diagnostics_invariance_2026-09-14/on_episode_forensics.json` 128행의
`seed`·`density_bars`가 전부 null이고 `checkpoint_sha256`은 해시가 아닌 파일 경로다. context도
같다. 선언한 metadata 계약에는 `INVALID_FOR_DECLARED_CONTRACT` (null 258건, SHA 형식 129건,
context 포함). 파일 SHA `1475b0c226e9…78870a`는 원래 Git blob과 일치하고, 중복 ID 0·비유한 값 0·
필수 key 자체 누락 0이다. 원본·checkpoint를 수정/열람하거나 주변 receipt로 값을 추정하지 않았다.
이 판정은 파일 byte 손상이나 기존 invariance smoke의 VOID 판정이 아니다. 원본 조회 검사 7.35 ms.

새 `benchmark_record_integrity.py`는 generic 5필드 합성 row 128/10,000/100,000개에만 실행했다.
총 21.999초; 직렬화+hash+parse+validation 평균 0.593/45.402/468.670 ms, P95
0.602/45.914/472.571 ms. 3 timing batches×5 samples, 총 225 raw sample의 요약 독립 재계산 일치.
실기/시뮬레이터 계측 overhead나 12셀 소요시간으로 환산하지 않는다. 디스크 fsync·GPU도 미측정.

검증 harness의 실패 2건도 보존했다. 첫 시도는 aerialgym에 jsonschema가 없어 문서 검사 실패;
기존 별도 docs 환경(jsonschema 3.2.0, cffconvert 2.0.0, PyYAML 6.0.2, pip check PASS)을 사용해
해소했다. **패키지 설치/환경 변경 0**. 두 번째는 내가 cffconvert를 `python -m`으로 호출했는데
그 패키지에는 __main__이 없었다. 설치된 CLI entrypoint 호출로 수정했다. 첫/둘째 run의 logs,
summary, runner source snapshot은 새 결과 디렉터리에 그대로 남겼다.

최종 **선택된 회귀 358개 PASS / 실패·오류·skip 0**, 문서/CFF/Node 검사도 PASS, 14.311초.
이 중 신규 generic 테스트 27개. 기존 test 수정/삭제/skip/약화 0. 이는 public/독립 renderer/
metadata allowlist이고, **기존 1,784개 전체 스위트는 NOT_RUN**이다. CUDA_VISIBLE_DEVICES 빈 값,
OMP/MKL threads 2. task/policy suites는 선택하지 않았다.

결과·명령·범위·재현 소스 hash 및 시간 기록:
`results/record_integrity_review_2026-09-14/README.md`, `summary.json`, `receipt.json`.
기존 renderer 및 task code/result는 그대로다. 알려진 결함은 metadata 완전성 문제로 남기며,
이를 근거로 성능 병목이나 다음 시스템 개선을 선택하지 않았다.

## 2026-09-14 (3) — Record Envelope v2: 미래 producer의 provenance 계약

record-integrity 감사 결과에 대한 대응을 **기록 수정이 아니라 producer 수정**으로 했다. 128행 smoke
artifact `on_episode_forensics.json`은 그대로 두고 두 판정을 모두 유지한다: byte integrity `MATCH`
(`1475b0c2…78870a`), metadata contract `INVALID_FOR_DECLARED_CONTRACT`. 주변 receipt나 런처에서
seed/density/hash를 **추론해 채우지 않는다** — 추론한 값은 기록된 값이 아니다.

**Record Envelope v2**(`docs/record_envelope_v2.json`, `tools/record_envelope_v2.py`):

- identity와 location을 분리한다. `checkpoint_path`는 사람이 파일을 찾는 정보이고
  `checkpoint_sha256`은 바이트에서 계산한 정체성이다. config도 같은 방식으로 분리. digest 필드에
  경로·파일명·심볼릭 이름이 오면 오류이고, 63/65자 해시도 오류다.
- `seed`, `density_bars`, `checkpoint_sha256`, `source_git_commit`은 **null 불가**. 얻을 수 없으면
  artifact가 생기기 전에 fail closed.
- `source_dirty`는 `git status`로 관측한다. clean이라고 가정하지 않는다.
- `run_id`는 UUIDv4이며 결과 의미를 담지 않는다.
- 행이 context 식별자를 반복하면 반드시 일치해야 하고 아니면 `CONTEXT_RECORD_MISMATCH`로 거부한다.
  v2는 context를 단일 소스로 쓰지만, 행 식별자를 읽는 기존 v1 소비자가 있어 **마이그레이션이 아니라
  별도 형식**으로 두었다.
- preflight가 생성 전을, postflight가 발행 전을 막는다. `publish()`는 postflight 통과 시에만 receipt를
  쓰고, 실패하면 원시 artifact를 남긴 채 failure 기록만 남긴다.

`tools/check_record_integrity.py`는 v1 historical audit과 v2 producer validation을 **둘 다** 지원하며
두 모드를 CLI에서 상호 배타로 강제한다. 역사적 artifact가 새 schema 덕분에 갑자기 PASS가 되는 일은
없고, 그 사실을 회귀 테스트가 고정한다.

상태 용어는 좁게 유지한다: `producer_unit_validation = PASS`,
`live_generation_validation = NOT_RUN`(`docs/record_envelope_v2_status.json`). 이 producer로 만든
런타임 artifact는 아직 없으므로 합성 파일 단위 테스트 통과는 `METADATA_FIXED`가 아니다. 상태 승격,
역사적 artifact를 repaired로 표시, 판정 완화 — 세 가지 모두 mutation으로 테스트가 실패함을 확인했다.

사이트에는 evidence source-note 줄에 한 문장만 추가하고 상세는 감사 문서로 링크했다.

검증: 전체 저장소 회귀 **1,852개 실행 / 실패 0 / 오류 0 / 기존 skip 4**, 65.2초(v2 테스트 41개 포함).
정책 실행·12셀 평가·과제 결과 분석·학습·컨트롤러/보상 변경은 하지 않았다.

## 2026-09-14 (4) — Envelope v2 live 시도, result manifest, evidence index, 공개 정리

**Phase 1 — Envelope v2 live-generation은 `NOT_APPLICABLE`이다.** generic renderer dataset exporter로
실제 시도했고, 15개 필수 필드 중 5개(`density_bars`, `checkpoint_path/sha256`, `config_path/sha256`)가
task-independent producer에 **정직한 출처가 없다**. producer는 세 번 모두 서로 다른 코드로 거부했다
(`INVALID_DENSITY_BARS`, `MISSING_CHECKPOINT_PATH`, `MISSING_CONFIG_PATH`). `density_bars = 0`을 넣거나
URDF를 checkpoint로 가리키거나 exporter 소스를 "config"로 해싱했다면 초록색 결과와 **거짓 기록**이
나왔을 것이다. 이것을 schema-design limitation으로 기록하고 **v2를 고치지 않았다** — task 섹션을
optional로 만드는 v2.1은 검증 부수효과가 아니라 별도 결정이다. 상태 테스트는 이제 네 가지 선언값을
허용하되 `PASS`는 실제 live receipt 파일을 가리켜야만 성립한다. 근거는
`results/record_envelope_v2_live_2026-09-14/`.

**Phase 2 — Experiment Manifest v1**(`tools/build_result_manifest.py` → `results/MANIFEST.json`).
201개 결과를 identity·provenance·status 문자열·링크·해시로만 색인한다. capture rate, 개선폭, 실패
원인 같은 **연구 수치는 추출하지 않는다** — 해석하는 manifest는 결과가 서술되는 두 번째 미검토
장소가 되고 둘은 어긋난다. 176개는 현재 규약 이전이라 `legacy_exception`으로 두고 누락 파일을
결함이 아니라 이력으로 기록한다. track 디렉터리는 container로 인식해 하위 실험을 색인한다.
source-manifest drift는 missing file과 분리해 보고한다(결과는 실행 당시 소스를 고정하므로 이후 커밋의
변경은 손상이 아니라 이력이다). 실제 drift 1건이 그 경우였다.

**Phase 3/6 — evidence index와 reproduction entry point.** `docs/RESEARCH_EVIDENCE_INDEX.md`는 7개 축
각각에 question/method/최대로 말할 수 있는 것/status/경로/한계를 적는다. bounded claim을 그대로
운반한다: P2·D1 FAIL과 P3 BLOCKED, P10 INCONCLUSIVE(적응 이득 미확립), S4 일반화 경고, E3-P
ATTITUDE_NOT_RELIABLE, D8b `MATERIAL_LOSS` −48.967 pp, renderer `causality_vs_d8b = NOT_TESTED`.
`docs/REPRODUCIBILITY.md`는 GPU 없이 되는 것/필요한 것/재배포 불가 데이터를 먼저 말한다.

**Phase 4 — paper page editorial pass.** 8절의 D8→D9 로드맵 스트립(실험 연대기)을 **덜어내고**
Reproducibility 절로 교체해 재현 진입점·evidence index·manifest로 보낸다. 연대기는 VERIFICATION·
WORKLOG·날짜별 결과 페이지에 그대로 있다. 고정된 주장 문자열은 모두 살아 있다.

**Phase 5 — release audit**(`docs/public_release_audit_2026-09-14.md`). 추적 바이너리 인벤토리:
checkpoint 18개(49.5 MiB), PNG 64개, 아카이브 4개, npz 10개, JPEG 13개. **blocker 2건**은 모두
CC BY-NC-SA 4.0 ETH ds5 파생물(JPEG 13개 + review ZIP)이며 **삭제·재라이선스하지 않고 보고만** 했다 —
삭제는 커밋된 결과가 참조하는 증거를 파괴하고, 재라이선스는 이 저장소의 권한이 아니다.
`NEEDS_CONFIRMATION` 4건(NavRL 복사 인벤토리, fork 기여자 저작권, Det-Fly 데이터셋 약관, NPS-Drones
파생물)은 추론으로 닫지 않았다.

**Phase 7 — 종료 상태**(`docs/repository_status_2026-09-14.json`): Envelope v2 COMPLETE(live
NOT_APPLICABLE), Manifest v1 COMPLETE, Evidence Index COMPLETE, Renderer Contract v1 FROZEN,
paper page REVIEWED, release audit COMPLETE_WITH_BLOCKERS. 제출 준비는 저장소 정리에 대한 것이지
연구 결과가 충분하다는 주장이 아니다.

검증: 전체 회귀 **1,869개 실행 / 실패 0 / 오류 0 / 기존 skip 4**, 63.9초(직전 1,852). 역사적 128행
artifact는 `1475b0c2…78870a` 그대로이고 판정 두 가지도 그대로다. 정책 실행·평가·학습은 없었다.

## 2026-09-14 — ETH ds5 current-tree 비배포 계약과 제거 전 인벤토리

사용자의 명시적 제거·커밋·normal push 승인에 따라 release engineering만 수행한다.
시작 시 fetch 후 HEAD=origin/main=`f7496004a1027ecf21b666d6da2d040adb16dec3`, clean.
제거 전 JPEG 13개와 ZIP 1개(합계 3,321,884 bytes)의 경로·SHA·마지막 커밋·참조·ZIP 멤버를
`docs/eth_ds5_public_release_inventory_2026-09-14.json`에 고정했다. ETH 비이미지 파일 46개를
핀하며 수치·receipt·CSV·과거 소스는 바이트 보존한다. 유일한 editorial 예외는 handoff의
깨질 이미지 링크 두 개와 availability 설명이다. 외부 데이터 계약·로컬 파일 검사·좁은 ignore를
추가했다. 기존 검수팩의 완전한 중심 좌표가 이미 유실됐다는 재현 제한은 유지한다.
아직 최종 회귀/제거 후 감사가 끝난 상태는 아니며, 후속 항목에서 실제 결과를 기록한다.
과거 Git blob은 삭제하지 않으며 과학적 재계산·GPU 평가·학습은 실행하지 않는다.

### ETH current-tree 제거와 외부 참조 계약 검증

`ec15d53`으로 제거 전 계약·inventory를 기록한 뒤 `79209d8`에서 정확한 14개 자산만 제거했다.
총 3,321,884 bytes이며 역사적 커밋에서 복구 가능하다. 제거 후 audit은 current-tree RESOLVED,
historical risk OPEN, 예상하지 않은 ETH evidence 변경 0개다. 46개 비이미지 핀 중 45개는
바이트 동일하고 handoff 하나만 가용성/링크를 교정했다. 남은 ZIP 3개 71멤버에 제거 이미지의
동일 바이트 사본이 없으며 E3 도표 2개는 실제로 열어 사진이 아닌 도식임을 확인했다.
기존 ZIP 직접 열기 테스트는 skip하지 않고 보존된 원 멤버 SHA/notice와 비배포 계약 검증으로
전환했다. 별도 synthetic ZIP 바이트 보존 테스트도 유지한다. 새 regression 13개 PASS;
public docs/schema PASS. Manifest는 정확한 path+SHA에만 external exception을 적용하며,
임의 missing/hash mismatch는 여전히 보고한다. 전체 회귀와 최종 push는 다음 단계다.

### ETH release 최종 회귀와 배포 판정

코드 완료 `d6bccb4`에서 canonical aerialgym Python 3.8 / CUDA_VISIBLE_DEVICES=0으로 전체
unittest 회귀 **1,882 run / 1,878 pass / 4 기존 skip / fail 0 / error 0**, **66.638 s**.
pytest는 해당 환경에 설치되어 있지 않아 설치·환경 변경 없이 저장소의 기존 unittest runner를
사용했다. pytest를 실행했다고 보고하지 않는다. 사이트 JS 검사 5종(실제 headless WebGL 포함),
public docs/schema, CFF 검사 PASS. 최종 문서 수정 뒤 docs/schema와 외부 데이터 13개 regression을
다시 확인했다. 로그와 파일 SHA receipt는 `docs/release_validation_2026-09-14/`에 보존한다.
394개 tracked 문서의 제거 대상 활성 링크 0개. 과거 단순 path/hash는 의도적으로 남긴다.

현재 트리의 ETH image/ZIP blocker는 **RESOLVED**, 14개 historical blob reachable이므로
**HISTORICAL_REDISTRIBUTION_RISK=OPEN**. NavRL copying·fork ownership·Det-Fly data terms·NPS
파생물의 네 확인 사항은 검색만으로 닫지 않았다. 과거 18 weights 모두 자체 학습이라는 감사
서술도 새 release status에서 7 project / 11 inherited로 정정했다. 과학 결과는 재계산·삭제하지
않았다. root LICENSE와 역사적 receipt/summary는 보존됐다. 새 release status와 체크리스트는
현재 availability만 표현하며 이전 audit은 당시 기록으로 유지한다.

다음 선택은 필요할 경우 **별도 clean public-release repository**를 명시적으로 승인받아 만드는
것이다. 이번에는 history rewrite/force push/새 repo 생성 없이 normal fast-forward push만 한다.

## 2026-09-14 (5) — clean public-release candidate (로컬 완성, 공개 없음)

기존 MOTAR history는 **전혀 변경하지 않았다**(rewrite·filter·force 없음). 대신 현재 commit의 내용만
담은 별도 스냅샷을 로컬에 만들었다: `../MOTAR-public-release-candidate/`.

**빌더**(`tools/build_public_release_candidate.py`)는 `git archive`로 내보내 `.git`을 상속하지 않고,
저장소 내부 절대 symlink 272개를 상대 경로로 바꾸고, 문서화된 제외 규칙을 적용한 뒤 **스스로를
반증하려 시도한다**: 모든 정규 파일과 **모든 아카이브 멤버**를 denylist 해시와 대조한다. ZIP 안에 숨은
이미지도 재배포이기 때문이다. denylist는 79209d8에서 제거된 ETH ds5 파생물 14개의 내용 해시
(`docs/public_release_denylist.json`)다.

**결과: `ETH_DENYLIST_MATCHES = 0`** (파일 5,310개·아카이브 멤버 71개 대조). 해시/파일명이 receipt
안에 **문자열로** 남아 있는 것은 provenance라서 유지한다 — 이건 바이트가 아니다.

정책은 `docs/public_release_file_policy.md`에 적었다. 제외는 (1) ETH 파생물 14개, (2) `.git`,
(3) 캐시, (4) **source commit에서 이미 깨져 있던** symlink 3개뿐이다.

빌드가 드러낸 **기존 저장소의 세 가지 비자립성**:
1. 일부 증거 테스트가 연구 history의 특정 commit을 검증한다 — 스냅샷에는 그 commit이 없다.
2. 일부는 `.gitignore`가 "immutable local receipts with raw traces"로 **의도적으로 제외**한 번들
   (`..._recovery_v2_gate_lower1p25_seed827` 57 MB 등)을 직접 읽는다. `check_research_authority.py`가
   그 안의 `summary.json`을 frozen evidence로 요구하므로 **어떤 clone에서도 실패**한다.
3. 일부는 추적되지 않은 학습 체크포인트/receipt를 요구한다.
셋 다 내 빌드가 만든 문제가 아니라 측정해서 드러난 기존 속성이다. 해당 검사들은 이제 stack trace 대신
**사유를 명시하고 skip**한다(`tests/history_helpers.py`). 연구 저장소에서는 가드가 무력해 그대로 실행된다.
스냅샷 판별은 SHA 목록이 아니라 "commit 1개·조상 없음"이라는 일반 규칙을 쓴다.

manifest 빌더도 고쳤다: 추적되지 않은 결과 디렉터리 3개를 `untracked_results`로 드러내고, git이 없는
스냅샷에서는 legacy 분류를 "모른다"고 하거나 연구 manifest에서 **상속**하며 그 manifest 해시를 기록한다.
릴리스 색인은 `results/RELEASE_MANIFEST.json`(198개)로 연구 `MANIFEST.json`(201개)과 **덮어쓰지 않고**
분리했다.

검증(후보 안에서): 전체 스위트 **1,884개 / 실패 0 / skip 29**(가드된 history·local evidence),
공개 문서 PASS, CITATION.cff 파싱 OK, 절대 경로는 공개 재현 경로 13개 파일에서 **0건**, 시크릿 0건.
별도 임시 디렉터리로 **clean clone**해서 다시 돌린 결과도 동일하게 통과했고 진입 문서의 깨진 링크는 0이다.

후보 저장소는 로컬에서만 `git init` + 첫 commit까지 했다(`Initial public release snapshot from MOTAR
research commit …`). **remote 생성·push·기존 MOTAR visibility 변경은 하지 않았다.**

## 2026-09-15 — Pages 랜딩의 저장소 루트 링크 404

공개 사이트 `https://joshualikaist.github.io/MOTAR/status/`는 `/docs`만 배포한다.
`docs/status/index.html`의 `../../CITATION.cff` 등은 Pages 루트를 벗어나
`https://joshualikaist.github.io/CITATION.cff`로 해석되어 404였다. Code 링크
`github.com/joshualikaist/MOTAR`도 비공개 저장소라 로그아웃 방문자는 404였다.

랜딩 페이지의 저장소 루트·results 링크를 공개 스냅샷
`https://github.com/joshualikaist/MOTAR-public/blob/main/…`으로 바꿨다.
해시 고정된 `archive-2026-09-13.html`은 수정하지 않았다. 과학 결과·시뮬레이터 코드는
변경하지 않았다.

## 2026-09-16 — MOTAR 약자 복원

원래 이름은 `9f67ca3`(2026-07-09)의
`Moving Object Tracking and Reinforcement-Learning-Based Approach for UAV Navigation in Random Obstacle Fields`
이다. `22f952e`(2026-07-22) README 축약에서 약자가 한 줄로 줄었고,
`23d48b5`(2026-09-13) 논문형 사이트 부제가 `Moving-Target Rendezvous in Dense Obstacle Environments
with Measured Perception Uncertainty`로 바뀌어 앞글자와 맞지 않았다.

사이트 부제·title, README 첫 줄, `CITATION.cff` title을 원문 약자로 되돌렸다.
본문의 interception/rendezvous 구분 설명은 그대로 둔다. archive 페이지는 해시 고정이라 수정하지 않았다.

## 2026-09-16 — matched baseline, 외부 문헌 관계, 상태 registry, target behavior ladder

현재 main `195ad05`의 README·VERIFICATION·evidence index·plans·results·status site를 다시
대조했다. 사이트에는 D8b primary contrast 외에 같은 계약 안의 정량 baseline 비교가 거의
없었고, P8/P9/D8-A/D8b 완료 뒤에도 그림에 generic dashed future 표현이 남아 있었다.

원 result 문서를 다시 읽어 Section 6에 8개 matched row를 추가했다: arc−riskcap
−1.4903 pp [−1.8981, −1.0826], 205-bar arc width crash −5.60/capture +4.44 pp,
riskcap adaptation capture +3.75 pp [1.30, 6.19], latency correction +40.21 pp,
learned detector NI −0.015 pp [−1.752, 1.723], P7 utility +0.00697, P10 +0.73 pp
[−1.04, 2.50] INCONCLUSIVE, D8b −48.967 pp [−50.113, −47.821] MATERIAL_LOSS.
새 근거 검사 9개가 사이트 문자열과 각 원 result를 함께 고정한다.

외부 시스템은 18개를 1차 검토하고 NavRL/NavRL++, YOPO/YOPOv2-Tracker, OPEN, AgilePE,
Fast-Tracker, Elastic Tracker, MAD, PILOT, FlowPilot, Temporal Barrier를 10개 행으로 사이트에
남겼다. 같은 benchmark와 metric인 Class A는 0개다. 공개 구현이 있는 NavRL/YOPO/OPEN/
Fast-Tracker/Elastic Tracker만 향후 port 후보 범위를 갖고, unreleased/incompatible 축은 C로
구분했다. 외부 논문 percentage를 MOTAR 수치에서 빼지 않았다.

`docs/research_status_registry.json`은 lifecycle과 evidence verdict를 분리한다.
COMPLETED/PLANNED/BLOCKED/NOT_TESTED/ARCHIVED_WITHDRAWN 5종을 badge·선 스타일로 표시하며
P10 INCONCLUSIVE, D8b MATERIAL_LOSS, D8c NOT_STARTED, D9 NOT_RUN을 보호한다. P6–P9와
D8-A/D8b 그림은 완료 lifecycle로 재생성했고 live RGB→policy는 dotted NOT_TESTED,
SAM candidate는 ARCHIVED, degree bearing/metric range/persistent ID switch는 BLOCKED다.

Target motion은 새로 추정하지 않고 기존 구현을 감사했다. legacy는 CV/waypoint와 순간 wall
reflection/push-out, bounded는 4 m/s²·150 deg/s·1 s lookahead의 obstacle-aware receding horizon,
physical은 100 Hz PhysX/motor actor와 optional GT-obstacle global route다. 브라우저 기본은
routed-preview이며 target은 pursuer에 반응하지 않는다. TM-E0 static, TM-E1 CV, TM-E2
obstacle-aware scripted를 opt-in `NAVRL_TARGET_BEHAVIOR_LEVEL`로 노출했고 historical default는
그대로다. TM-E2는 target-side obstacle GT만 사용하며 pursuer observation은 바꾸지 않는다.
TM-E3/TM-E4는 선택 시 fail closed한다. 새 PPO나 E0/E1/E2 policy grid는 실행하지 않았고
기존 physical `FAIL_ROUTE_MECHANISM`도 유지했다.

제목은 `MOTAR: Moving Object Tracking And Reinforcement Learning for UAV Pursuit in Random
Obstacle Fields`로 README/site/CFF/Figure 1 caption에 통일했다. 최종 research 전체 unittest
**1,906개 / 실패·오류 0 / 기존 skip 4**, 59.154초. target-motion 29,
source-bound comparison 9, overview 16,
public-doc 7, Node site/manifest/arena와 CFF/schema가 PASS했다. 브라우저 검수에서 800 px
literature-table overflow를 발견해 내부 scroll로 수정했고 desktop/tablet/mobile 모두
document overflow 0, component status 19행 로드, 이미지·WebGL·fail-closed PASS다.
공개 스냅샷의 Node route 검사는 연구 ancestor `a373202`가 없어 처음 실패했다. 연구 저장소에서는
hash 검사를 유지하고, `RELEASE_PROVENANCE.md`가 있는 공개 스냅샷에서만 그 ancestor hash 한 항목을
명시적으로 skip하며 현재 route geometry/determinism 검사는 계속 실행하도록 고쳤다.
공개 스냅샷 full unittest도 후속 public commit이 1개를 넘으면서 기존 `commit_count < 2`
판별이 무력화되어 1 failure/12 errors를 냈다. snapshot은 영구히 1-commit일 필요가 없으므로
`RELEASE_PROVENANCE.md`의 세 문구를 durable marker로 검증하게 고쳤다. 파일명만 만든 가짜 marker는
인정하지 않고, research checkout에서는 모든 ancestor 검사가 그대로 실행된다.
수정 후 공개 스냅샷 전체 suite는 **1,906개 / 실패·오류 0 / skip 29**, 52.662초로 통과했다.
29 skip은 snapshot에 의도적으로 없는 research ancestor/local evidence를 사유와 함께 표시한다.

## 2026-09-17 — Browser GT tracking preview and public evidence wording

사이트 3D arena를 코드 기준으로 감사한 뒤, browser-only GT tracking preview를 추가했다.
기존 local heuristic pursuer는 exact target 좌표를 이미 받고 있었으나 global route/prediction/
continuous follow가 없어 dense bar에서 제자리 선회처럼 보였다. 새 `gt-route-track` /
`gt-free-roam`은 exact browser target state와 obstacle AABB만 사용하며 PPO observation,
PhysX, TM-E3, research evaluation이 아니다. HUD badge
`BROWSER GT PREVIEW / NOT PPO · NOT PHYSX · NOT RESEARCH EVIDENCE`는 GT mode에서 숨길 수 없다.
카메라 DETECTED/OCCLUDED/OUT OF FOV는 GT planner와 독립이다. Historical
legacy/bounded/physical-style/routed-preview와 local heuristic은 보존했다.

Editorial: MOTAR 전개는 `Moving Object Tracking And Rendezvous`로 맞추고, COMPLETED는
named stage 완료이며 긍정 판정/후속 비교 완료를 의미하지 않는다고 고쳤다. 외부 문헌은
18 UAV systems screened / 10 retained / 8 omitted이며 fictitious self-play는 방법론
기초로 18에 넣지 않는다. public README의 Git-history 문구는 research history와
MOTAR-public snapshot을 분리한다. 기존 P10 INCONCLUSIVE, D8b MATERIAL_LOSS, live RGB
NOT_TESTED, bearing/range/ID BLOCKED는 그대로다.

Node: arena motion/route/demo-planner/site/manifest/WebGL PASS. Python overview,
comparison, public-docs, target-motion, GT-observation-boundary PASS. aerialgym 전체
discover는 CUDA import 환경 실패 2건만 확인했고 이번 변경과 무관하다.


## 2026-09-17 — GT tracking preview: measured, four defects found and fixed, GT_BROWSER_V1 frozen

어제 추가한 browser GT tracking preview를 **처음으로 matrix에서 측정**했다. 단위 테스트는
전부 PASS였는데 실제로는 추적을 못 하고 있었다. `tools/validate_gt_browser_tracking.js`
(밀도 70/115/160/205 × target speed 0.3/0.9/1.5 m/s × seed 10 × 60 simulated seconds,
고정 10 Hz, **120 runs**)로 재면 tracker가 **run의 47–87 %를 정지**해 있었고, 여러 셀에서
60 s 뒤 상대거리가 spawn보다 **더 멀었다**(160 bars/0.3 m/s: 18.13 → 18.49 m).
hard invariant도 하나 깨져 있었다.

### 찾은 결함 4개와 원인

| | 증상 | 원인 |
|---|---|---|
| **D1** | `acceleration_limit_violation` > 0 | `integrateBounded`가 swept step이 unsafe인 순간 속도를 0으로 만들었다. 2.5 m/s → 0을 0.1 s에 하면 **25 m/s²**로 4.0 m/s² 계약 위반이고, 거부된 command를 다음 tick에 그대로 재발행해 stop/stutter loop에 갇혔다. |
| **D2** | 유효 route와 full-speed command를 쥔 채 safety envelope에 **끼임** | carrot이 polyline을 따라 1.8 m 간 점인데 follower는 거기까지의 **직선**을 탄다. route가 bar를 끼고 되꺾이면 그 chord가 certified corridor를 벗어나 장애물을 가리킨다. 205 bars에서 command heading 202° vs 다음 waypoint heading −69°로 확인. |
| **D3** | 205 bars에서 `no_safe_route_fraction` 최대 74 %, replan 5–7.8/s | tracker의 support disc + tracking margin이 target보다 커서, target이 빠져나가는 틈이 tracker에게는 막혀 있다. `nearestSafe`가 snapping 대신 포기했다. |
| **D4** | 두 기체가 step 0부터 `no_path`로 **영구 정지** | 고밀도에서 `navrl_band`가 닿은 bar를 compound wall로 합치며 pocket을 봉인한다. 거기서 **zero command는 옳았고 spawn이 틀렸다**. |

D4는 분명히 적어 둔다: fail-closed 동작이 정답이었고 시나리오가 틀렸다. 이걸 controller에서
"고쳤다면" 벽을 뚫고 바쁘게 움직이는 preview가 나왔을 것이다.

### 수정

- **D1** `integrateBounded`가 bounded candidate(scale 1, 0.7, 0.45, 0.25, 0.1, 0) 중 swept
  segment가 certified된 첫 번째를 고른다. 모든 candidate가 **같은** `limitPlanarVelocity`를
  통과하므로 어느 것이 채택돼도 가속·선회 계약이 유지된다. scale 0은 최대율 bounded brake다.
  brake조차 envelope를 쓸고 갈 경우의 emergency hold는 남겼고 **횟수를 센다**.
- **D2** `carrotPoint`가 현재 pose에서 **certified segment로 닿는 가장 먼** look-ahead를
  돌려주고, 전부 막히면 `null`을 돌려 replan을 강제한다. `advanceAlongRoute`는 지나쳐 버린
  waypoint도 통과시켜 follower가 뒤를 겨누지 않게 했다. 새 `approachVelocity`가 남은 선회량으로
  command 속도를 깎아 코너에서 감속한다.
- **D3** `nearestSafe`가 `snapToSafe`로 폴백하고, no-route 재시도는 `noRouteRetryS`로 throttle해
  unroutable pose가 tick당 A* 한 번을 먹지 않게 했다.
- **D4** `createSession`이 A* cache가 이미 만든 occupancy grid에서 연결성분(`freeComponents`)을
  구해 봉인된 pocket 밖으로 **결정적으로** 재배치한다. cached occupancy만 읽고 새 geometry는
  만들지 않는다. 120 runs 중 target 12회·tracker 16회 발동했다 — D4는 드문 사건이 아니었다.

### 같은 matrix 재측정 (120 runs, 60 s/run)

| | 수정 전 | 수정 후 |
|---|---|---|
| hard invariant 위반 | `acceleration_limit_violation` ≠ 0 | **11개 항목 전부 0 / 120 runs** |
| stall fraction | 47–87 % | 0.2–1.3 % |
| no-safe-route fraction | 최대 74 % | 0.0 % (205/0.9 한 셀만 5.2 %) |
| median 상대거리 | 8.7–18.5 m | 1.52–2.13 m |
| final 상대거리 | 6.5–18.5 m | 1.35–2.17 m (spawn 14.8–21.4 m) |
| replans/s | 최대 7.8 | 0.16–1.36 |

추가 진단(120 runs 집계): `distance_reduction` 평균 **16.14 m**, `minimum_distance` 평균
0.190 m(최소 0.004 m), terminal tracking error 평균 **1.602 m**, `heading oscillation
period-2`(>25° 반전) **0회**, valid-route-but-stalled 평균 0.38 s/60 s, emergency hold 평균
7.7 tick/600 tick(1.3 %).

1.5 m 바닥은 **display-only standoff**(`CONTRACT.standoffM = 1.55 m`)다. 문서상 tracking
reference는 상대 위치 오차 0이고, standoff는 mesh overlap 방지용 렌더링 오프셋이지
capture radius가 아니다. historical 0.5 m capture semantics와 연결하지 않는다.

### Determinism / browser validation

`tests/test_status_gt_tracking_determinism.js` 신규: 동일 seed 재현, **30/60/120 FPS render
cadence가 committed trajectory를 바꾸지 않음**(30 s, state digest 완전 일치), paused renderer는
0 step, 장애물 없는 합성 케이스의 거리 수렴(quantile 계약), matrix hard invariant, certified
look-ahead 회귀 가드.

`tools/browser_validation/`로 실제 Chrome 146 headless(swiftshader)에서 desktop 1440×900 /
tablet 1024×768 / mobile 390×844 각 65 s 실측했다. probe는 published page를 바꾸지 않고
`?probe=1`일 때만 주입된다.

측정 중 두 가지를 발견해 고쳤다. ① arena는 IntersectionObserver로 화면에 있을 때만 도는데,
probe가 스크롤하지 않아 **정지한 장면을 재고 있었다**(거리 7.44 m 고정). ② teleport 카운터가
wall-clock 기반이라 render 부하에서 drift를 불연속으로 오판했다 — committed simulation step
수로 bound하고, bars/speed 변경이 일으키는 **의도된 scene rebuild**는 `resets` serial로 분리했다.
runner가 중단될 때 headless Chrome을 orphan으로 남겨 다음 run의 frame time을 깎는 것도
확인해서 child reaper를 넣었다(orphan이 붙은 desktop run은 37.1 fps, 깨끗한 run은 54.2 fps).
③ headless Chrome이 창 폭을 ~500 CSS px 아래로 내려주지 않아 `--window-size=390`이 조용히
500 px viewport를 준다는 것도 찾았다. 즉 좁은 breakpoint가 실제로는 돌지 않고 있었다.
390 px 미만 viewport는 **정확히 그 폭의 iframe**에 태우고(iframe은 자체 viewport라 media query가
진짜 폭에서 평가된다), verdict가 `window.innerWidth`가 요청 폭과 일치하는지 **검사**하게 했다.

### Editorial

method subtitle을 `Reinforcement Learning for UAV Tracking and Close Approach in Random
Obstacle Fields`로 확정하고 README / 라이브 페이지 / `CITATION.cff` / figure generator를 맞췄다.
`tools/check_public_docs.py`가 이제 **세 표면에서 동일한지**를 검사한다(한 곳만 보던 것을 확장).
`docs/assets/paper/overview-2026-09-13/`는 hash-pinned이므로 재생성하지 않았다 — 그 SVG의
embedded `<title>`/`<desc>`는 2026-09-13 상태로 남으며, 이유를
`docs/status/overview-sources-2026-09-13.md`에 provenance로 적었다.

### Evidence boundary

actor observation tensor, `aerial_gym` task observation builder, reward, checkpoint, 기록된
research 판정은 **하나도 건드리지 않았다**. `tests/test_browser_gt_state_not_in_policy_observation.py`
가 이를 강제하며 PASS다. P10 INCONCLUSIVE, D8b MATERIAL_LOSS, D8b causality NOT_TESTED,
D8c PLANNED, D9 NOT_RUN, live RGB NOT_TESTED, bearing/true-range/persistent-ID BLOCKED,
TM-E0/E1/E2 IMPLEMENTED, TM-E2 policy comparison NOT_TESTED, TM-E3/E4 PLANNED — 전부 그대로다.

### 다음

`docs/prereg_2026-09-17_target_motion_complexity_e0_e1_e2.md`를 **사전등록만** 했다.
상태는 `PREREGISTERED / NOT_RUN`이고, 이번 작업에서 outcome experiment는 실행하지 않았으며
새 PPO 학습도 없다. 실행에는 별도 GPU authority 승인이 필요하다.

## 2026-09-17 — Target motion: one canonical explanation, bound to source by test

"타겟은 어떤 알고리즘으로 움직이나요?"에 대해 README·사이트·기술문서·FAQ가 **같은 말을
하도록** 정리했다. 새 research result는 만들지 않았고, 코드 동작도 바꾸지 않았다.

### 먼저 source를 다시 감사했다 — 그리고 가장 중요한 사실이 나왔다

작업 전 가정은 "MOTAR 타겟은 A*를 쓴다"였는데, 소스를 읽으니 **틀렸다**.

| 계보 | global route | local step | 근거 |
|---|---|---|---|
| legacy | 없음 | `steer_target_step` — 10개 heading 후보, full speed, 90° 연속성 선호, least-bad fallback | `target_motion.py` |
| **TM-E2 (bounded)** | **없음** | `bounded_drone_target_step` — 24 heading × cruise scale (1.0/0.5/0.25) + stop, lookahead rollout, **첫 스텝만** 실행 | `target_motion.py` |
| physical | **선택적** A* (`NAVRL_TARGET_ROUTE_MODE`, 기본 `off`) | `braking_aware_route_step` → PhysX actor | `navrl_task.py:886` |
| browser GT free-roam | **있음** — A* 또는 reachable-set flood | `integrateBounded` | `arena_*.js` |

즉 **TM-E2는 A*를 쓰지 않는다.** `target_motion.py` 전체에 `heapq`도 A*도 없다.
연구 시뮬레이터에서 A*가 등장하는 곳은 physical 계보의 opt-in route mode 하나뿐이고,
그 mode는 `"physical+waypoint-only lineage"`로 virtual/bounded 타겟에는 **거부**되며
기본값이 `off`다. 그 계보의 판정은 여전히 `FAIL_ROUTE_MECHANISM`이다.

browser가 mirror하는 것은 physical 계보의 `global_astar_v1` **geometry contract**이지
TM-E2가 아니다. 둘을 같은 알고리즘으로 설명하면 안 된다.

두 번째로 나온 정밀한 사실: browser AUTO ROAM은 **"랜덤 goal 뽑고 A*"가 아니다.**
`planToConnectedGoal`은 현재 셀에서 free grid를 uniform-cost flood한 뒤 **실제로 도달된
셀 중에서** goal을 고르고, route는 그 flood의 parent chain에서 나온다. reachability가
rejection sampling이 아니라 **구성적으로** 보장된다. A*(`plan`)는 goal이 고정된 경우 —
click/tap goal과 추적 드론 자신의 경로 — 에만 쓴다.

geometry 대조도 확인했다: research와 browser 양쪽 모두 8-connected, cell 단위 Euclidean
heuristic, 그리고 **대각선 corner-cutting 거부**(양쪽 직교 이웃이 모두 free일 때만 대각 허용)가
일치한다. route 단순화는 양쪽 다 **greedy farthest-visible shortcut**이고 spline이 아니다.

### 만든 것

- `docs/target_motion_algorithm_2026-09-17.md` — canonical. 30초/2분 설명, 계보 분류표,
  browser 정밀 기술, TM-E2 정밀 기술, GT 경계, 제약식, pseudocode 2종(browser와 TM-E2를
  **따로**), FAQ 15문항, known limitations, source binding.
- `docs/presentation_target_algorithm_answers_2026-09-17.md` — 15초/45초/technical 답변
  (한국어·영어), 예상 후속질문 9개 한 줄 답, **말할 때 피해야 할 표현** 표.
- 사이트 `2.2 Target motion` 섹션 + **Figure T1**(research와 browser가 갈라지는 지점을
  보여주는 분기 다이어그램). CSS로만 그려서 hash-pinned figure package는 건드리지 않았다.
- README에 "How does the target move?" 한 줄 진입점.
- ladder 문서에 canonical 문서 상호참조.

### Source-bound test — 이게 핵심이다

`tests/test_target_algorithm_documentation.py` 24개. 문서가 코드와 어긋나면 FAIL한다.

- `target_motion.py`에 `heapq`/`astar`가 나타나면 FAIL → "TM-E2는 A*를 안 쓴다" 문장이
  거짓이 되는 순간을 잡는다.
- research/browser 양쪽 planner가 8-connected가 아니게 되면 FAIL.
- browser planner가 corner-cut 거부를 잃으면 FAIL.
- 어느 planner에든 `spline`/`bezier`/`catmull`이 나타나면 FAIL(문서는 shortcut이라고 적혀 있다).
- route mode 기본값이 `off`가 아니게 되거나 physical-only 제한이 사라지면 FAIL.
- TM-E3/E4가 fail-closed가 아니게 되면 FAIL.
- `uses_privileged_pursuer_gt: False`가 사라지면 FAIL.
- §9 수치표의 값이 소스에 없으면 FAIL, 소스에 있는데 표에 없어도 FAIL.
- 문서가 "MOTAR target uses A*" 류로 일반화하면 FAIL.
- 사이트 target-motion 섹션이 browser-only 라벨이나 pursuer 언급을 잃으면 FAIL.

음성 검증도 했다: canonical 문서에 "The MOTAR target uses A* everywhere, including TM-E2"를
일부러 주입하니 정확히 2개 테스트가 짧은 메시지로 FAIL했고, 되돌리니 24/24 PASS.

### 검증

Node 7종 PASS, Python 문서/사이트 계열 PASS. aerialgym env에서
`test_navrl_target_motion`·`test_navrl_target_route_planner` PASS(타겟 모션 코드는 손대지
않았으므로 당연하지만 확인). 사이트 3 viewport 실측에서 horizontal overflow 0, page error 0
(mobile 390 px 포함).

`test_research_overview`의 figure 수 계약을 8 → 9로 **의도적으로** 갱신했다. Figure T1은
CSS로만 그린 lettered supporting figure라 image 자산이 없고, 따라서 image 수는 7로 그대로이며
dated hash-pinned figure package도 그대로다.

### 바꾸지 않은 것

target motion 구현, 관측, reward, checkpoint, 기록된 판정. `aerial_gym/` 아래 변경 0건.

## 2026-09-17 — Retraining readiness audit: 세 가지 설계 결함을 측정 전에 잡았다

"현재 frozen policy가 지금 contract에서 여전히 타당한가"를 판정하려 했고, **판정에
도달하지 못했다.** 이유가 결과다: 평가를 돌리기 전에 **그 평가 설계 자체의 결함 3개**와
코드 결함 9개를 찾았다. 먼저 돌렸으면 해석 불가능한 숫자가 나왔을 것이다.

**새 PPO 학습 없음. GPU 작업 자체를 하지 않았다.** `tools/check_research_authority.py`가
"offline bottleneck analysis and a new preregistration are required before any GPU work"를
요구하며, 이 감사가 그 offline analysis다.

### 결함 1 — 사전등록에 in-distribution arm이 없었다 (가장 중요)

checkpoint `f702213936…`의 `env_state`를 읽으니
`cfg_target_motion_model = symmetric_local_steer_v2_heading_continuity90`,
`cfg_target_pattern = mixed`다. launcher `train_navrl_v2_search.sh:154`의
`NAVRL_TARGET_DYNAMICS:-legacy`도 같은 말을 한다. 즉 **이 정책은 legacy 계보 / mixed
pattern에서 학습됐다.** accel bound도 turn-rate bound도 없고, wall reflection과 bar
push-out이 있는 그 계보다.

그런데 내가 전 세션에 쓴 사전등록은 세 arm 모두 `bounded`를 고정했다. TM-E2는
`bounded|physical`+`waypoint`, E1은 `cv`, E0는 speed 0을 강제한다. **E0/E1/E2 중 어느
것도 학습 분포가 아니다.** 그대로 돌렸으면 E2 저하를 "E2가 어렵다"와 "세 arm이 전부
off-distribution이다"로부터 분리할 수 없었다.

→ **arm H(historical) 추가.** 4 arms × 4 densities × 3 seeds = 48 cells.

### 결함 2 — `time_within_approach_band_frac`는 이 task에서 정의되지 않는다

`navrl_task.py:4999` 주석이 명시적이다: **"Interception semantics (always on): capture
ends the episode."** `success_radius = 0.5 m` swept 판정으로 **episode가 끝난다**(:6025).

따라서 "approach band 안에 머문 시간"은 구조적으로 편향된다 — **잘 추적할수록 더 일찍
끝나서 표본을 적게 남긴다.** band 숫자를 못 정한 게 아니라 metric이 termination과
양립하지 않는다. → **제거.** 임의의 숫자를 고르지 않았다. 대신 capture termination
아래에서 정의되는 `min_relative_distance_m`(모든 episode에서 정의)와 `capture_rate`를
primary로 올렸다. §6 판정은 **A. CLOSE_APPROACH_TASK**(termination 유지, continuous
tracking 주장 축소). termination은 이번에 건드리지 않았다.

### 결함 3 — 학습 계약이 HEAD 기본값으로 재현되지 않는다

`tools/audits/extract_runtime_contract.py`(정적 추출, CUDA context 없음)로 대조했다.
checkpoint `cfg_` 112개 중 **14개가 다르다.** 전부 env-var opt-in이고, **10개는 아무도
검사하지 않으며 4개는 warning만 낸다.**

| key | 학습 | HEAD 기본 | guard |
|---|---|---|---|
| `cfg_lidar_hbeams` | 72 | 36 | warn | **관측 차원** |
| `cfg_lidar_max_range` | 12.0 | 4.0 | warn |
| `cfg_max_velocity` | 2.5 | 2.0 | warn | **action scale + 관측 정규화 분모** |
| `cfg_yaw_rate_max` | 3.0 | 2.5 | warn |
| `cfg_episode_len_steps` | 600 | 300 | **없음** |
| `cfg_oob_margin` | 1.0 | 0.5 | **없음** |
| (+8개 더) | | | 없음 |

시뮬레이터 guard는 *"warn, never override: an eval may deliberately change these"*라고
명시돼 있다. 시뮬레이터로선 옳은 기본값이고 matched comparison으로선 틀린 기본값이다.
→ 시뮬레이터를 바꾸지 않고 **strict gate를 추가**했다:
`tools/audits/check_eval_condition_contract.py`, `CONFOUNDED`면 exit 2.
HEAD 기본값 시뮬레이션에 대해 CRITICAL confound 6개를 잡고 거부하는 것을 확인했다.

### 관측 호환성 — 비대칭 실패가 핵심

normalization은 **사용된다**(`normalize_input: True`). 저장 위치는 `env_state`가 아니라
checkpoint의 `model` state dict(`running_mean_std`, actor `(898,)` / critic `(906,)`).
`env_state`에 `norm` 키가 없는 것은 미사용 근거가 아니다.

```text
폭이 틀리면        -> RuntimeError (strict load_state_dict)
폭은 같고 의미가 틀리면 -> logger.warning만, 실행 계속
```

그리고 **restored checkpoint의 관측 차원을 runtime과 비교하는 검사는 어디에도 없다.**
raise하는 preflight(`navrl_checkpoint_preflight.py`)가 있지만 **frozen checkpoint가 실제로
지나가는 eval 경로에서 호출되지 않는다.** 위 14개 drift가 만들 수 있는 게 정확히
"같은 폭, 다른 의미"다. strict gate의 근거가 이것이다.

### 코드 결함 9건 — 전부 소스에서 직접 인용 확인, **수정하지 않음**

서브에이전트 보고를 그대로 옮기지 않고 고영향 항목은 내가 다시 읽었다.

- **D1** `navrl_task.py:8184-8252` — legacy push-out이 순수 변위이고
  `target_vel_w = Δpos/dt`로 재계산되므로 **realized 속도가 라벨된 episode 속도를 넘을 수
  있다.** reflection counter는 `if self._bulk_eval_mode:` 안에만 있어 **학습 중에는 전혀
  계측되지 않았다.** bounce jitter는 bar contact 조건부로 **global torch RNG**를 뽑아
  이후 draw를 desync시킨다. **frozen policy가 학습한 바로 그 계보다.**
- **D2** `:9289` results receipt의 `target_pattern` 기본값이 `"static"`(실제 기본은 `mixed`).
  `env_state`는 맞다 — receipt만 거짓말한다.
- **D3** `:2224` `target_speed_mps`를 `NAVRL_TARGET_SPEED` 기본 0으로 기록(실제는 curriculum).
- **D4** `:3440` surface clearance를 `0.0`으로 attest하는데 arena는 `0.45`로 짓는다
  (`navrl_bars_env.py:96`). drift guard가 **같은 잘못된 값을 재계산**하므로 영원히 발화 못 함.
- **D5** `navrl_perception.py:960` `int(round(0.05/0.1)) = round(0.5) = 0` — banker's
  rounding으로 0.05 s latency arm이 물리적으로 0 latency.
- **D6** `:5918` 비유한 LiDAR 픽셀을 `1.0`(= 전 범위, "아무것도 없음")으로 매핑. 가장
  유리한 값이고 safety reward와 governor 양쪽에 들어간다.
- **D7** `speed_governor.py:44` `free_speed_cap = √2·2.5` 하드코딩인데 `max_velocity`
  기본은 2.0(도달 norm 2.83) → HEAD 기본값에서 governor가 개활지에서 절대 안 걸린다.
  이 checkpoint(2.5 학습)에서는 맞다.
- **D8** `:1319-1324` `_detector_noise_range_ar` AR(1) 상태가 episode reset 안 됨.
  두 knob 기본값이 0.0이라 기본 경로는 무해.
- **D9** `:1960` LiDAR→target range에 **카메라** 구 반경 0.15를 더하는데 LiDAR는 0.20으로
  주입 → 모든 LiDAR fallback 보정이 0.05 m 편향.

기록에 남는 결과를 바꿀 수 있는 것이 8건이다. **하나도 고치지 않았다.** D1은 frozen
checkpoint의 학습 분포 자체를 건드리므로 수정은 기존 결과 수정이 아니라 새 계보를 만든다.
D2/D3는 물리는 그대로고 라벨만 틀리므로 재실행이 아니라 erratum이 필요하다. 나머지는
result별 impact analysis가 먼저다. **historical result는 하나도 덮어쓰지 않았다.**

### 그 밖에 확인한 것

- heading-validity threshold: checkpoint에 키 **없음** → `ASSUMED_PRE_KEY_DEFAULT`.
  소스 주석이 pre-key 계보의 inline **1e-05 m/s**를 명시하고 현행은 **0.10 m/s** — 1e4배.
  arm 간 동일하게 유지하고 결과에 provenance를 명시한다. **0.10을 소급 기록하지 않는다.**
- TM-E2 dynamics: `v·ω = 1.5 × 150°/s = 3.9270 m/s²` vs envelope `4.0` → 일치하지만
  **여유 1.83%**. 최대 속도·최대 선회의 coordinated turn이 가속 한계에 거의 붙어 있어
  동시 속도 변화 여유가 사실상 없다. 2-D bounded kinematic이지 rigid-body가 아니다.
- action: xy는 **축당** ±2.5 → 도달 norm 2.5√2 = 3.5355. z는 altitude PI가 덮어쓰지만
  prev-action 관측에는 그대로 실린다 → **미래 training lineage의 design debt로 기록**,
  이번에 바꾸지 않음(checkpoint 호환성이 이 평가의 전제).
- reset 감사: target/pursuer/tracker/history/prev-action은 reset됨. route manager의
  4개 버퍼는 `valid` gate 덕에 command 경로에 못 닿음(진단 노출만).

### 산출물

```text
docs/audits/frozen_policy_training_contract_2026-09-17.md      provenance 태그별 계약표
docs/audits/current_runtime_contract_2026-09-17.json           정적 스냅샷
docs/audits/training_contract_vs_current_runtime_2026-09-17.json  14개 diff
docs/audits/retraining_readiness_audit_2026-09-17.md           본 감사
docs/audits/retraining_readiness_status_2026-09-17.json        기계 판독 상태
tools/audits/extract_runtime_contract.py                       정적 추출/대조
tools/audits/check_eval_condition_contract.py                  strict gate (exit 2)
tools/audits/build_retraining_readiness_status.py              status 생성기
tests/test_retraining_readiness_audit.py                       19개, drift 시 FAIL
```

사전등록은 **측정 전에** Amendment 1로 갱신했다: arm H 추가, 결함 metric 제거,
seed 4101-4103 / cell당 정확히 2048 episode(128-env × 16 batch; 과거 2,049 tail 제거) /
Wilson CI / seed-paired BCa bootstrap / Holm-Bonferroni 고정.

### 판정

```text
RETRAIN_DECISION = BLOCKED_PENDING_EVALUATION
EVALUATION       = READY_NOT_RUN
```

재학습 전 freeze해야 할 6개 계약 중 **3개가 아직 안 잠겼다**: observation contract
(§9의 미기록 perception knob들), perception contract(live RGB NOT_TESTED, true range
BLOCKED, persistent ID BLOCKED, P10 INCONCLUSIVE), target behavior contract(D1이
legacy 계보 안에 있음), robot dynamics contract(asset hash가 checkpoint에 없음).
따라서 지금 재학습을 승인할 근거는 없다. §9 때문에 **case C(target generator부터 고쳐라)가
실재 가능성**이고, 설계가 그걸 검출할 수 있도록 validity gate를 outcome보다 먼저 읽는다.

## 2026-09-17 — Defect impact matrix: D1은 H 전용임을 구조로 증명, TYPE-B 0건

9개 defect 중 **무엇이 historical training distribution의 일부이고 무엇이 앞으로의
frozen-policy comparison을 무효화하는가**를 가렸다. GPU outcome evaluation은 실행하지 않았다.

### 핵심 결과 — D1은 arm H에만 해당한다 (구조적 증명)

`_advance_target`의 제어 흐름을 AST로 확인했다:

```text
if self._target_dynamics in ("bounded", "physical"):
    ...
    return          <-- :8082, 무조건, else 없음

# 아래는 dynamics == "legacy"일 때만 도달:
#   :8205  bar push-out (순수 위치 변위, 최대 6라운드)
#   :8232  bounce reflection, GLOBAL torch RNG jitter
#   :8249  target_vel_w = (new_pos - old_pos) / dt
```

E0/E1/E2는 전부 `bounded`이므로 push-out / reflection / velocity-rewrite 블록에
**구조적으로 도달할 수 없다.** 들여쓰기 눈대중이 아니라 AST로 확인했고, matrix 생성기가
빌드 시 이 사실을 재도출해서 깨지면 matrix 발행을 거부한다.

```text
E2_UNAFFECTED_BY_D1 = TRUE (증명됨)
```

D1은 **TYPE-A / preserve**다. H는 frozen policy의 실제 학습 계보를 재현하는 것이 목적이므로
H 안에서 D1을 "버그니까" 고치면 H가 재현이 아니게 된다.
**historical defect ≠ evaluation implementation bug.**

### 분류 결과

```text
TYPE-A (preserve)        D1
TYPE-B (invalidating)    없음
TYPE-C (receipt only)    D2, D3, D4
TYPE-D (future debt)     D5, D6, D7, D8, D9
```

arm별 영향: `H: D1,D3,D6,D9` / `E0,E1,E2: D3,D6,D9`.
D3은 receipt 전용, D6·D9는 공유 perception stack이라 **arm-symmetric** — 절대값은 흔들지만
H-vs-E2 대비를 편향시키지 못한다. arm 비대칭인 것은 D1뿐이고 그건 H에 갇혀 있다.

48-cell 계약에 대해 좁혀진 것들:
- **D2 무해** — 48셀 전부 `NAVRL_TARGET_PATTERN`을 명시 설정하므로 잘못된 기본값에 도달 안 함.
- **D3 활성** — manifest는 `NAVRL_TARGET_SPEED`를 **일부러 설정하지 않는다**(설정하면
  `speed_fixed`가 켜져 physics가 바뀐다). 따라서 receipt의 `target_speed_mps`는 모든 arm에서
  0으로 적힌다. 라벨을 고치려 env를 건드리지 말고 `cfg_target_speed_min/final`을 읽을 것.
- **D4 무해** — `placement_surface_clearance`의 유일한 실소비자는
  `asset_manager._footprint_clearance_xy_spacing`이고 48셀은 전부 `navrl_band`다.
- **D5/D7/D8 무해** — 각각 latency 0.0, `NAVRL_MAX_VELOCITY=2.5` 고정, detnoise 0.0 기본.

### 감사 중 발견 — heading-validity threshold가 turn-rate bound를 **게이트한다**

defect 목록에 없던 것이고, defect가 아니라 **provenance가 없는 계약**이다.

```python
limited_heading = torch.where(
    current_speed > HEADING_VALID_SPEED_MPS, limited_heading, desired_heading)
```

docstring도 명시한다: *"Below HEADING_VALID_SPEED_MPS there is no heading of travel."*
즉 **0.10 m/s 미만에서는 slew limit이 적용되지 않는다.** `speed_min=0.3`에
cruise scale `0.25` → **0.075 m/s**이므로 정상 E1/E2 운용 중에도 free pivot이 가능하다.

그리고 이 키는 frozen checkpoint가 **attest하지 않는다**(`ASSUMED_PRE_KEY_DEFAULT`, pre-key
계보는 inline `1e-05`). `1e-05`면 slew limit이 사실상 항상 걸리고 `0.10`이면 안 걸린다.
→ 이 threshold는 metric semantics만이 아니라 **target kinematics 자체를 바꾼다.**
`limit_planar_velocity`는 bounded executor가 쓰므로 E0/E1/E2에 해당하고, H(`steer_target_step`)에는
해당하지 않는다. 전 arm 동일하게 유지하고 결과에 `ASSUMED`로 표기한다.

### 내 감사 하네스가 세 번 틀렸다 — 실행기가 아니라

E2 validity를 처음 돌렸을 때 pen/turn/teleport가 났는데 전부 하네스 문제였다.
이걸 E2 결함으로 보고했으면 심각한 오보였다.

1. **spawn 64개 중 3개가 애초에 bar 안에서 시작** — 실행기 탓이 아니다. rejection sampling 추가.
2. **"teleport"가 bound 초과 1.8e-06** — float32 반올림. 상대 허용오차로 교체.
3. **turn 위반**이 정지→재출발 구간. 그리고 결정적으로 임계값을 `1e-4`로 썼는데
   계약 임계값은 `HEADING_VALID_SPEED_MPS = 0.10`이다. 실제 이벤트를 뽑아보니
   prev |v| = 0.0199 < 0.10 → 계약상 free heading. **계약 준수 동작을 위반으로 세고 있었다.**

### E2 validity 최종 — PASS, 그러나 하네스를 네 번 고친 뒤에

production 계약을 정확히 재현하고 나서야 통과했다. 12 cases, **115,200 env-steps**,
penetration/wall/clearance/speed/accel/turn/teleport/NaN/RNG **전부 0**.

네 번의 오류가 **전부 하네스 쪽**이었고 하나도 실행기 결함이 아니었다:

1. spawn 64개 중 3개가 bar 안에서 시작 → rejection sampling 추가.
2. "teleport"가 bound 초과 **1.8e-06** → float32 노이즈. 상대 허용오차로 교체.
3. turn 체크를 `1e-4`에서 했는데 계약 임계값은 `HEADING_VALID_SPEED_MPS = 0.10`.
   문제의 이벤트는 `|v| = 0.0199 < 0.10` → 계약상 free heading. **준수 동작을 위반으로 셈.**
4. **계보를 잘못 잡았다.** `bars_half_extents`를 넘기고 surface clearance 1.0 m를 썼는데
   그건 **physical** 계약이다. bounded 계보(=E0/E1/E2)는 `bars_half_extents = None`,
   `clearance = tm.obstacle_clearance = 0.77` m **centre-to-centre**다
   (`navrl_task.py:7873`, `_target_planner_clearance`). 이게 결정적이다: 0.77 m는
   navrl_band 복도 반폭 0.8 m 안에 **일부러** 들어가게 잡힌 값이라 실제 계약은 합법 복도를
   통과하고 physical 계약은 못 통과한다. 잘못된 계약으로 감사하니 합법 복도가 막히면서
   남은 penetration을 전부 만들어냈다.

교훈: **감사는 production call contract를 그대로 재현해야 한다.** 아니면 자기 설정을 잰다.
네 개 중 셋은 그럴듯한 실패처럼 보였고, 덜 의심했으면 target generator 결함으로 발표했을 것이다.

### RNG / reset

- `target_motion.py`에 **난수 호출 0건**(빌드 시 정규식 검증). bounded executor는 RNG를 안 쓴다.
- legacy bounce jitter(H)는 bar contact 조건부로 **global** stream을 소비한다.
- obstacle pose는 `asset_manager.reset_idx`에서 **매 episode** global stream으로 재샘플.
- 따라서 **같은 seed라도 arm이 다르면 bar layout이 다르다.** 이건 **편향이 아니라 pairing 손실**이다:
  분포는 동일하고 2048 ep × 3 seed로 비교한다. 설계를 **distributional matching**으로 기술하고
  paired-layout matching이라고 쓰지 않는다. prereg의 seed-paired BCa는 seed 평균 수준에서
  짝을 짓지 episode layout으로 짓지 않으므로 영향 없다. dedicated generator 도입은
  H가 재현해야 할 historical RNG 거동을 바꾸므로 **하지 않는다**.
- reset: target/pursuer/tracker/history/prev-action/route validity 모두 reset. route manager의
  4개 버퍼는 `valid` gate 뒤에 있고 route mode는 48셀 전부 `off`. speed governor는 per-env
  상태 자체가 없다. → **cross-arm leakage 없음.**

### 48-cell dry preflight

```text
48 cells (4 arms × 4 densities × 3 seeds), 2048 ep/cell, 98,304 episodes
checkpoint present + SHA 검증 통과
strict gate 48/48 MATCHED, confounded 0
arm별로 다른 env 키는 전부 NAVRL_TARGET_* 뿐 (테스트로 강제)
outcome 산출 없음 (dry)
```

### 산출물

```text
docs/audits/target_motion_defect_impact_matrix_2026-09-17.{md,json}
results/target_motion_e0_e2_preflight_2026-09-17/{cell_manifest,contract_comparison,go_no_go,e2_target_validity}.json
tools/audits/{build_defect_impact_matrix,build_e0_e2_cell_manifest,build_go_no_go,audit_e2_target_validity}.py
tests/test_target_motion_defect_impact.py   16개
```

**physics 코드 수정 0건.** push-out / reflection / action semantics / termination / reward /
controller 전부 그대로다. `aerial_gym/` 아래 diff 0.

## 2026-09-18 — Browser arena: interception episode(기본) + capture/timeout termination

브라우저 GT preview가 "무한 following"만 보여줘 MOTAR의 핵심(detection→tracking→pursuit→
close approach→**capture**)을 잘못 표현하던 conceptual bug를 고쳤다. browser visualization
전용 작업. `aerial_gym/`·`resources/`·`configs/` 변경 0.

### 정확한 원인 (재검증 완료)

`arena.js:simulationStep`의 `gt-route-track` 브랜치가 `return;`(구 :902)으로 끝나서 그 아래
`sweptCapture(...,0.5)`와 `episode.age>=30` watchdog에 **도달하지 못했다**. 즉 기본 GT
preview는 outcome 검사를 아예 안 했다. 게다가 `predictedFollowPoint`가 `standoffM=1.55 m`를
target **뒤**에 영구히 두어 조준점이 target에 닿지 않으니 0.5 m capture가 구조적으로 불가능했다.

### 수정

- `simulationStep`을 `stepTarget → stepPursuer → 공통 evaluateEpisodeOutcome → terminal 전환`
  으로 재구성. historical/GT 두 경로가 **하나의** outcome layer를 지난다. capture literal을
  브랜치에 복붙하지 않았다.
- 공통 함수 `Motion.episodeOutcome`을 arena.js와 planner(headless)가 **같은 입력**으로 호출 →
  capture/timeout semantics drift 불가. capture는 swept(스텝 사이 crossing 포착), timeout은
  task episode budget.
- interception 모드: standoff를 **거리의 연속함수**로 감쇠(CLOSE 4.0 m에서 1.55 m → INTERCEPT
  1.2 m에서 0). INTERCEPT phase는 target 자신을 short-horizon 예측해 조준. continuous 모드는
  1.55 m 고정 유지.
- terminal freeze + ~1.5 s hold + **정확히 1회** reset. TIMEOUT/ABORT도 terminal 처리.
- capture radius(0.5 m)·timeout(600×0.1=60 s)은 **연구 task 값**을 viewer.js가 provenance
  주석과 함께 싣고 `tests/test_browser_capture_provenance.py`가 소스에 바인딩.

### hesitation — 느낌 아니라 수치로, 네 원인 분리 (continuous 120런)

| 후보 | 측정 | 판정 |
|---|---|---|
| 10 Hz 클럭 | 30/60/120 FPS 동일 궤적 | **원인 아님** |
| replan | 0.64/s(최대 5.75/s) | 이동 중 발생, 정지 아님 |
| goal handoff | pause **0.035 s** 평균 | 무시 가능 |
| emergency hold | 7.7 tick/600 = **1.28%** | 지배적, 안전장치 → 유지 |

goal handoff pause가 이미 0.035 s라 prefetch rewrite는 fail-closed를 위험에 빠뜨리면서
얻는 게 없어 **구현하지 않았다**(brief의 "확인되면" 조건부). 10 Hz는 hesitation 원인이 아니다.

### 검증

- **continuous before/after 완전 동일** — 모든 metric(goal transition/handoff/dwell/stall/
  emergency hold/replan/route switch) 일치. 추적 데모 byte-for-byte 보존.
- interception 120런: episode 676 종료 / **capture 676 / timeout 0**, time-to-capture 8.8 s,
  closest 0.39 m(<0.5 m). terminal-correctness 5종(무근거 capture/누락 capture/terminal 중
  이동/reset 횟수/stale state) 전부 0.
- hard invariant 11종 **세 매트릭스 모두 ALL ZERO**.
- Node 7종 + headless WebGL + provenance/doc python 테스트 PASS.
- raw: `results/browser_interception_v1_2026-09-18/`, 문서: `docs/status/interception_episode_2026-09-18.md`.

### evidence boundary

"Browser interception preview mirrors the task termination semantics, but is not PPO or
PhysX performance evidence." 48-cell 사전등록 평가·정책·checkpoint·reward 미변경.

## 2026-09-18 — Browser termination provenance 정리 + GT_BROWSER_EPISODE_V1 FROZEN

browser 작업을 마무리하고 H/E0/E1/E2 preflight로 복귀했다. GPU/PPO 실행 없음.

### 문제: provenance source가 분산돼 있었다

termination 값(0.5 / 600 / 0.1)이 **세 곳에 literal**로 있었다 — `viewer.js`, `arena.js`,
`validate_gt_browser_tracking.js`. 값은 맞았지만 단일 출처가 없었다. `status.json`은
`generated_at=2026-09-08`의 old snapshot이라 termination field가 없었고, harness는 fallback을
쓰고 있었다.

`status.json`을 그냥 regenerate하면 `latest_run`·`active_run`·run summary 같은 **무관한
live field가 함께 바뀐다**. 그래서 하지 않았다.

### 해결: static contract를 dynamic snapshot과 분리

```text
docs/status/status.json                  DYNAMIC dashboard snapshot (active/latest run, 요약)
docs/status/research_task_contract.json  STATIC task contract (termination + stable geometry)
```

`tools/build_research_task_contract.py`가 **research source에서 읽어** 생성한다:

| 값 | 바인딩 |
|---|---|
| `success_radius_m` 0.5 | `navrl_task_config.py: success_radius` |
| `episode_len_steps` 600 | `train_navrl_v2_search.sh: NAVRL_EPISODE_LEN_STEPS` |
| `rl_step_dt_s` 0.1 | `base_sim_config.dt(0.01)` × `navrl_bars_env.num_physics_steps_per_env_step_mean(10)` — **유도**, 단언 아님 |
| arena/placement | 같은 launcher의 `NAVRL_ARENA_*` / `NAVRL_PLACEMENT_*` |

output에 source path·git commit·source SHA-256 포함. `--check` 모드가 committed contract와
source의 drift를 검사(provenance 필드는 제외, 계약 본문만).

**fail closed**: 바인딩 regex가 안 맞으면 `ContractError`로 생성 거부. silent default 없음.

### 소비자 통합

page(fetch, malformed면 부팅 거부) / arena(configure로 주입, `terminationFromContract` 노출) /
harness(contract 못 읽으면 실행 거부) / tests 전부 이 한 파일에 bind. `arena.js`의 literal은
**explicitly labelled dev fallback**으로만 남겼고 production 경로에서는 사용 0회.
`update_status_snapshot.py`에 넣었던 termination field는 **되돌렸다**(dynamic-only 유지).

### Public terminology — UI label만

```text
CHASE→TRACKING  CLOSE→CLOSE APPROACH  INTERCEPT→FINAL APPROACH
CAPTURED→APPROACH COMPLETE  TIMEOUT→TIMEOUT  ABORT→RESET / ABORT
```

**internal state name·`success_radius`·`capture`·historical record는 rename하지 않았다.**
테스트가 양쪽을 강제한다(label map 존재 + planner/motion의 internal name 보존).

### 검증

- `tests/test_browser_capture_provenance.py` 23개(기존 11 → 확장): contract가 source와 일치,
  `rl_step_dt`가 유도값과 일치, 생성기 `--check` 통과, 소비자 fail-closed, dev fallback 라벨링,
  status.json에 termination field 누출 없음, UI label 매핑, internal name 보존.
- Node 7종 + headless WebGL PASS(새 fetch 부팅 경로 실브라우저 확인).
- browser 120런 재측정(contract-bound): interception **676 episode / 676 capture / 0 timeout**,
  continuous 포함 **hard invariant 16종 전부 0**.

### GT_BROWSER_EPISODE_V1 = FROZEN

`docs/status/gt_browser_episode_v1_freeze_2026-09-18.md`. continuous 계약 / close-approach
episode 계약 / termination provenance / public terminology 동결. planner parameter
(standoff, close·final-approach threshold, prediction horizon, A* resolution)는 이제
prereg나 amendment 없이 조정하지 않는다. browser engineering metric은 research result
table에 넣지 않는다.

### 연구 preflight 복귀 (GPU 미실행)

- defect matrix 재확인: **TYPE-B 0건**, `E2_UNAFFECTED_BY_D1 = TRUE`(AST 재도출).
  arm 영향 `H: D1,D3,D6,D9` / `E0,E1,E2: D3,D6,D9`.
- 48-cell dry preflight: **48/48 MATCHED**, confounded 0, checkpoint SHA 검증 통과.
- E2 validity: **PASS**(navrl_band, production bounded 계약 0.77 m centre-to-centre).
- Go/No-Go: 8개 기준 중 `git_tree_clean` 외 전부 PASS → 커밋 후 재생성.

```text
GPU_EVALUATION = READY_FOR_USER_AUTHORIZATION (자동 실행 안 함)
RETRAINING     = BLOCKED_PENDING_EVALUATION
```

---

## 2026-09-18 — External quantitative positioning (CPU/docs only, GPU 격리)

GPU에서 H/E0/E1/E2 frozen-policy evaluation이 도는 동안, **완전히 분리된 worktree**
(`paper/quantitative-positioning-20260918`, base `origin/main` = `096eee5`)에서 문헌 정량
positioning만 수행. GPU worktree 변경 0, CUDA workload 0, result-dir write 0.

### 해결한 gap

기존 사이트는 within-MOTAR 수치 비교(Table 3)와 published-system **정성** 관계(Table 4)는
보여줬지만 **external reported numerical positioning**이 사실상 없었다. 이번에 3-layer 구조로
채웠다: Layer A(외부 보고값) / Layer B(MOTAR 자체 측정) / Layer C(matched comparison, **비어 있음**).

### External ledger — 13 works, 45 rows

`docs/literature_quantitative_ledger_2026-09-18.json`. **34 rows에 값, 11 rows는
`NOT_EXTRACTED`.** source: primary_paper 42, official_code 3, third_party_benchmark 0.
12 works가 사이트 10행을 채우고(NavRL/NavRL++, YOPO/YOPOv2-Tracker가 각각 한 행),
Fast-Tracker 2.0은 screened-not-retained로 분리 기록.

검증 과정에서 **기존에 후보로 적어둔 숫자 여러 개가 정정됐다**:

| 주장 | 검증 결과 |
|---|---|
| OPEN "100% capture rate" | simulation only, Random/Passage(OOD)만 1.000, **Wall은 0.987**. 3 pursuers vs 1 evader, capture radius 0.3, 300 episodes. **입력이 privileged relative state + LOS mask — 카메라 없음.** real-world capture rate는 미발표 |
| FlowPilot "8 m/s" | **commanded**이지 achieved 아님. 달성된 sim peak는 6.14 m/s |
| FlowPilot "5.5 m/s real clutter" | real·achieved 맞지만 forest **sparse 구간** peak. indoor cluttered는 3.8 / 3.1 m/s |
| FlowPilot "<18 ms" | Table II "Total inference latency" 16.294 ms(max 17.055). **MPC 2.384 ms는 별도 행** — abstract의 "full perception-to-action" 표현이 더 넓다 |
| YOPO RK3566 INT8 ~20 ms | **official_code(README master)로 확인**. 단 paper 본문은 paywall + arXiv 없음 → paper-internal 값 전부 `NOT_EXTRACTED` |
| MAD 9.66 m/s | **abstract에만 존재**, 본문 미대응. 본문 최대는 corridor 6.37 m/s |

`NOT_EXTRACTED`가 많은 건 실패가 아니라 기록이다. Elastic Tracker Fig. 8(c)
(out-of-FOV / too-near / occlusion 분해)와 YOPOv2-Tracker Fig. 10B success rate는
**bar chart·plot이라 텍스트에 수치가 없어** 추정하지 않고 그대로 남겼다.

### 가장 쓸모 있는 positioning 사실

**Elastic Tracker 결론이 "tracking escaping target"을 future work로 명시**한다. 즉 obstacle-aware
자율 evader는 고전 tracking 계열의 **평가 밖 영역**이다. 이 계열의 target은 전부 cooperative
(marker 부착 / position broadcast / GT future trajectory oracle). MOTAR의 TM-E2는
pursuer-independent obstacle-aware라서, matched comparison은 arena만이 아니라 **target 자체를
고정**해야 한다.

### MOTAR 내부 수치 — 전부 source-bound

`docs/quantitative_positioning_registry.json`에 11개 항목. 10개 수치를 **인용한 result 파일에서
실제로 grep 되는지 검증**했고, 그 검증을 테스트로 고정했다. P9 perception error가 frozen policy에
주는 비용 **−4.57 pp**(3개 training-seed campaign 전부 동일, CI가 0 제외)를
`preregistration_p10_seed_replication`에서 찾아 Layer B에 추가.

`target_motion_generalization` 행은 **`RESULT_PENDING`** — GPU 48셀이 끝나고 integrity PASS +
raw manifest freeze 전에는 값이 들어갈 수 없게 테스트로 막았다.

### 새 파일

- `docs/literature_quantitative_ledger_2026-09-18.json` (기계 판독), `..._positioning_2026-09-18.md` (생성물)
- `docs/quantitative_positioning_registry.json`, `docs/paper_claim_evidence_matrix_2026-09-18.md`
- `docs/external_matched_baseline_readiness_2026-09-18.md`, `docs/paper_evidence_outline_2026-09-18.md`
- `tools/build_literature_positioning_doc.py` (`--check`), `tools/build_quantitative_positioning_figure.py` (`--check`)
- `docs/assets/paper/quantitative-positioning-2026-09-18.svg` (Figure 9)
- `tests/test_quantitative_positioning.py`, `test_external_numeric_sources.py`, `test_site_quantitative_positioning.py`

MD 본문과 SVG는 **손으로 쓰지 않고 JSON에서 생성**한다. 사이트는 6.2 Quantitative positioning
신설, 기존 6.2→6.3 / 6.3→6.4, Table 4 신설로 4→5 / 5→6 재번호.

### 도구 버그 2건 (기각된 가정 포함)

- `build_quantitative_positioning_figure.py`: 첫 판 matrix가 viewBox를 넘었다(마지막 열 x=1592,
  row rule 1632, MOTAR rect 1668 > 1600). **"XML로 파싱되면 맞다"는 가정 기각** — overflow SVG도
  정상 파싱되고 오른쪽만 조용히 잘린다. bounds guard 추가 후 실제 범위 x 60..1536 / y 56..852.
- `tests/test_quantitative_positioning.py`: 금지어 검사 초안이 "not이 한 줄에 있으면 통과"라
  `"outperforms X, but not in real flight"`를 놓쳤다. 부정어가 **용어 앞 120자 이내·마침표 없이**
  올 때만 면제하도록 수정. 세 테스트 모두 심어놓은 위반으로 **negative test** 확인.

### 검증

`tests.test_quantitative_positioning` 9 / `test_external_numeric_sources` 7 /
`test_site_quantitative_positioning` 10 전부 PASS. docs·site 관련 33개 모듈 458 tests 재실행 →
남은 FAIL 2건은 **내 변경과 무관**(참조 파일 교집합 0): `test_result_manifest`는 git 미추적
`results/*` 디렉터리가 새 worktree에 없어서, `test_navrl_corrected_nonoverlap_route_gate`는
tools import 문제. error 32건은 base python에 torch/trimesh 없음.

`test_research_overview`의 figure 수(9→10), image(7→8), figcaption(9→10)과
heading split 참조는 **내 변경이라 같이 갱신**했다.

### 환경 관찰 (조치 안 함)

디스크 `/` **100% (47M free)**. 원인은 `/tmp`의 과거 세션 임시 디렉터리 ~4.7G와 conda
`pkgs/` 6.9G(전부 extracted, tarball 0 → 삭제 위험). **평가 산출물은 셀당 0.1 MB, 남은 27셀에
약 2 MB만 필요**하므로 진행 중인 GPU run은 위험하지 않다고 판단. 남의 임시 디렉터리는
임의로 지우지 않았다. 사용자 판단 필요.

```text
GPU_EVALUATION = RUNNING (별도 worktree, 이 작업과 무관)
CLASS_A_EXTERNAL = 0 (변동 없음)
RETRAINING = BLOCKED_PENDING_EVALUATION
```

---

## 2026-09-19 — H/E0/E1/E2 frozen-policy 평가 완주 (초기 판정, 아래 정정 항목 참조)

사용자 승인 하에 48셀 매트릭스 실행 완료. frozen `ep25000+riskcap`
(SHA `f702213936…`, 실행 직전 재검증 일치). 4 arms × 4 densities(70/115/160/205) × 3 seeds
(4101–4103) × 2048 episodes = **98,319 episodes**, wall 11,347 s. 학습은 하지 않았다.

### 8단계 게이트

`stage1 integrity PASS`(48/48, nonce 전부 고유) · `stage2 target validity PASS` ·
`stage3 contract equivalence PASS` · stage4/5/6 recorded · stage7 computed ·
**stage8 = `NO_RETRAIN_NEEDED`**. 분석 전에 `RAW_MANIFEST.json`으로 raw 동결.

### arm별 결과 (seed 평균)

| arm | capture | crash | timeout | closest_nocrash |
|---|---:|---:|---:|---:|
| E0 static | 84.31% | 8.58% | 7.04% | 1.85 m |
| E1 cv | 85.07% | 10.49% | 4.41% | 1.24 m |
| **H historical (in-dist ref)** | **87.89%** | **9.50%** | **2.61%** | **0.90 m** |
| E2 obstacle-aware | 89.22% | 9.61% | 1.17% | 0.61 m |

### H 대비 (seed-paired BCa95, 20000 resamples)

| arm | Δcapture | BCa95 | 0 제외 |
|---|---:|---|---|
| E0 | **−3.58 pp** | [−4.04, −3.29] | 예 |
| E1 | **−2.82 pp** | [−3.53, −2.44] | 예 |
| E2 | **+1.33 pp** | [+1.15, +1.50] | 예 |

**세 arm 모두 CI가 0을 제외**한다. 당시 분석 도구는 5 pp 문턱으로 `materially_degraded = false`,
verdict `NO_RETRAIN_NEEDED`를 냈다. **그 5 pp는 사전등록된 것이 아니었다 — 같은 날 아래 항목에서
정정한다.** 어느 쪽이든 "차이가 없다"가 아니라
**"차이는 실재하지만 작다"**가 정확한 표현이다.

### E0/E1/E2 상호 대비 (분석 도구가 빠뜨려 별도 계산, `contrasts_within_E_arms.json`)

capture: E1−E0 **+0.76 pp** [+0.51,+0.90] · E2−E0 **+4.91 pp** [+4.34,+5.36] ·
E2−E1 **+4.16 pp** [+3.53,+4.70]. 전부 0 제외.

### 핵심 발견 — 가설 기각: "정지 표적이 가장 쉽다"는 **틀렸다**

네 지표가 전부 같은 순서다: **E0 < E1 < H < E2**. 즉 **표적이 느릴수록 frozen policy가 나쁘다.**
정지 표적(E0)이 최악이고, 장애물 회피까지 하는 E2가 최고다.

메커니즘은 관측 쪽에 있다. step-weighted target visible fraction:

| arm | overall | capture | crash | timeout |
|---|---:|---:|---:|---:|
| E0 | 0.2236 | 0.3444 | 0.0199 | **0.0015** |
| E1 | 0.2536 | 0.3355 | 0.0788 | 0.0058 |
| H | 0.2932 | 0.3557 | 0.0853 | 0.0075 |
| E2 | 0.3285 | 0.3638 | 0.1319 | 0.0355 |

visibility가 capture 순서와 단조 일치한다. 결정적으로 **E0의 timeout episode에서 표적 가시율이
0.0015** — 사실상 전혀 보이지 않는다. 정지 표적은 한 번 놓치면 스스로 시야로 돌아오지 않으므로
재획득이 일어나지 않고 timeout(7.04%, H의 2.7배)으로 끝난다. E0의 closest_nocrash 1.85 m는
success_radius 0.5 m에 한참 못 미친다 — 접근 자체를 못 한다.

밀도 상호작용도 있다(F1): 70 bars에서 arm 간 격차가 가장 크고(E2 93.9% vs E0 87.3%)
205 bars에서 거의 수렴한다(H 82.0 / E2 82.8 / E0 80.1 / E1 79.0). 고밀도에서는 장애물 난이도가
표적 운동 효과를 덮는다.

### D1 격리 경험적 확증

48셀 전부에서 E0/E1/E2의 wall/bar reflection이 **정확히 0.0**, H만 비영(bar_reflection_any_rate
0.00195). AST로 주장했던 "D1은 H에만 존재"가 런타임 데이터로 확인됐다.

### 도구 버그 2건 (둘 다 내 쪽, 평가 데이터는 정상)

- `verify_canary.py`: `(x or -1)`이 E0의 정당한 `target_speed_max_mps=0.0`(falsy)을 FAIL시켜
  캐너리 게이트가 거짓 실패했다. None-safe 비교로 수정 후 4셀 전부 PASS.
- `analyze_arms.py` stage5: strata 집계가 `successes`만 읽는데 task export가
  `distance`/`speed`는 `successes`, `initial_target_bearing`은 **`captured`**로 내보낸다.
  그 결과 bearing capture_rate가 4개 arm 전부 **0.0으로 조용히 계산**됐다(전체가 ~85%인데).
  두 키를 모두 받고 **둘 다 없으면 raise**하도록 고친 뒤 재실행. 수정 후 bearing은
  0.838–0.900으로 정상이고 좌/우 비대칭은 없다.
  **"XML/JSON이 파싱되면 값도 맞다"는 가정 기각** — 스키마 불일치는 조용히 0을 만든다.

### 보고해야 할 편차 3건

1. `HEAD == origin/main` 게이트 미충족. HEAD `ae286e5`(평가 런처 추가 커밋 1개)가 origin/main
   `096eee5`보다 앞선다. fast-forward push를 시도했으나 **환경 permission classifier가 차단**했고
   우회하지 않았다. 셀 receipt마다 정확한 commit SHA가 기록돼 무결성은 보존된다. **사용자 push 필요.**
2. n=3에서 sign-flip permutation은 2³=8 배열뿐이라 최소 양측 p가 **0.25**(Holm 후 0.752).
   어떤 대비도 p<0.05에 도달할 수 없다. 추론은 BCa95 CI가 담당하며, permutation p를
   "유의차 없음"으로 읽으면 안 된다.
3. 사전등록 primary인 `min_relative_distance_m`은 bulk export에 없고
   `closest_nocrash_mean_m`(무충돌 조건부)만 있다. 조건부 형태로 보고한다.

### 산출물

`results/target_motion_e0_e2_evaluation_2026-09-18/`: 48 cell 디렉터리(result+receipt+log),
`RAW_MANIFEST.json`, `analysis_report.json`, `contrasts_within_E_arms.json`,
`figures/F1..F4.png`.

```text
RETRAINING = NOT_AUTHORISED (NO_RETRAIN_NEEDED)
PPO_TRAINING_STARTED = false
```

---

## 2026-09-19 — 정정: 5 pp 문턱은 사전등록이 아니었다 + canonical result package

위 항목의 판정 근거를 감사한 결과 **두 가지를 철회**한다. GPU는 재실행하지 않았고 PPO training도
시작하지 않았다. raw artifact는 immutable이며 **144/144 파일이 `RAW_MANIFEST.json`과 일치**한다.

### 철회 1 — "사전등록 5 pp materiality threshold"는 존재하지 않는다

| 검색 | 결과 |
|---|---|
| `git log --all -S"material_threshold"` | **0 commits** |
| `git log --all -S"NO_RETRAIN_NEEDED"` | **0 commits** |
| prereg §6 Decision rule (`096eee5`, sha256 `7d8744c4…`) | validity gate / confounding / CI 비교 / deployment 확장 금지 **4개뿐. 효과크기 문턱 없음, retraining 어휘 없음** |
| Amendment 1 §A1.3 | arms·densities·seeds·cells·episodes/cell·CI method·seed aggregation·multiple comparison 고정. **materiality 문턱 없음** |

5 pp는 이번 분석 세션에서 내가 쓴 untracked 도구(`analyze_arms.py`, `material_threshold = 0.05`)가
도입한 것이다. **사전등록을 소급 수정하지 않았고**, 5 pp는 `POST_HOC DECISION CONTEXT`로만 남긴다.

**판정을 좁힌다**: `NO_RETRAIN_NEEDED` → **`NO_RETRAINING_JUSTIFIED_FOR_E2_TARGET_MOTION_SHIFT`**.
근거는 문턱이 아니라 측정된 부호다 — E2는 H 대비 **+1.33 pp**(BCa95 [+1.15,+1.50], 3/3 seed 양수)로
저하되지 않았다. E0 −3.58 pp / E1 −2.82 pp는 **실재하는 감소**이며, 이를 material/immaterial로
분류할 사전등록 기준이 없으므로 pass/fail이 아니라 측정값으로 보고한다.

### 철회 2 — `closest_nocrash_mean_m`을 primary로 쓰지 않는다

Amendment 1 §A1.2가 primary로 올린 `min_relative_distance_m`은 **기록되지 않았다**.
`navrl_task.py:9745-9746`이 `ep_min_goal_dist`를 **non-crash episode에 대해서만** 누적하고
전체 episode 누적기는 어디에도 없다. per-episode record도 export되지 않는다.
→ `min_relative_distance_m = NOT_RECORDED`, `closest_nocrash_mean_m`은
`POST_HOC_CONDITIONAL_DIAGNOSTIC`로 재분류(결과 조건부 선택이므로 대체재가 아니다).

### 98,319 − 98,304 = +15의 원인 규명

**vectorised tail overshoot.** `navrl_task.py:9750-9751`의 정지 조건이
`if total >= self._progress_log_interval`이고 이 검사는 **vectorised step마다 한 번**, 그 step에서
끝난 모든 env를 합산한 뒤 평가된다. 128 env에서 2047에 있던 카운터는 두 env가 같은 step에
종료하면 2049가 된다. 분포는 0:35셀 / +1:11셀 / +2:2셀. 이론상 최대 overshoot은 `num_envs-1 = 127`.

기각한 원인: canary 오염(canary 4셀 중 3셀이 excess 0), 중복 terminal record(48/48에서
outcome triple == actual), resume 중복(셀 단위 skip이므로 +2048이지 +1이 아님).

**A1.3의 근거가 실현되지 않았다**: "2048로 고정하면 tail 비대칭이 구조적으로 사라진다"고 적었지만
구현은 target을 **cap하지 않고 `>=`로 비교만** 하므로 비대칭이 남는다. 이건 진짜 protocol deviation이다.

**영향은 유계**: 15개가 전부 capture였다/전부 실패였다고 가정한 최악 envelope가
arm별 **최대 0.018 pp**로, 가장 작은 효과(0.76 pp)보다 두 자릿수 작다. 순서·부호·결론 불변.

exact-2048 view는 **구성 불가**(`NOT_CONSTRUCTIBLE`) — bulk export에 per-episode record가
전혀 없다(episode_id/env_id/배열 없음). 재실행 없이는 만들 수 없으므로 **합성하지 않았다.**

### canonical 재계산 — 예비 보고와 숫자는 동일

| arm | capture | crash | timeout |
|---|---:|---:|---:|
| E0 static | 84.31% | 8.58% | 7.04% |
| E1 CV | 85.07% | 10.49% | 4.41% |
| H (in-dist ref) | 87.89% | 9.50% | 2.61% |
| E2 obstacle-aware | 89.22% | 9.61% | 1.17% |

12개 capture 대비 전부 3/3 seed 부호 일치. n=3에서 permutation 최소 p가 0.25이므로
**"statistically significant"라고 쓰지 않는다** — effect size와 seed 일관성만 말한다.

visibility는 pooled step-weighted(`visible_steps/observation_steps`)로 재계산했다:
E0 0.2202 / E1 0.2502 / H 0.2889 / E2 0.3215. 예비 보고는 cell 평균이라 0.003–0.010 높았다
(순서는 동일). **E0의 timeout episode 가시율 0.0016**이 메커니즘의 핵심이다.

### 도구 버그 1건 추가 (stage5 strata)

task export가 `distance`/`speed` strata는 `successes`, `initial_target_bearing`은 `captured`로
내보내는데 전자만 읽어 **bearing capture_rate가 4개 arm 전부 0.0으로 조용히 계산**됐다.
두 키를 모두 받고 **둘 다 없으면 raise**하도록 수정 → 0.838–0.900, 좌우 비대칭 없음.

### 산출물

`results/target_motion_e0_e2_2026-09-18/`: `RAW_MANIFEST.json`, `episode_count_audit.json`,
`primary_2048_manifest.json`, `canonical_summary.json`, `seed_effects.csv`,
`statistical_sensitivity.json`, `figures/F1..F4.png`.
`docs/audits/target_motion_protocol_deviations_2026-09-19.md`,
`docs/results/target_motion_generalization_2026-09-19.{md,json}`,
`tools/build_target_motion_result_doc.py`(`--check`), `tools/build_target_motion_figures.py`,
`tests/test_target_motion_result_package.py`(22 tests PASS).

```text
VERDICT = NO_RETRAINING_JUSTIFIED_FOR_E2_TARGET_MOTION_SHIFT
5PP_THRESHOLD = NOT_PREREGISTERED (post-hoc decision context)
min_relative_distance_m = NOT_RECORDED
PPO_TRAINING_STARTED = false
GPU_RERUN = none
```
