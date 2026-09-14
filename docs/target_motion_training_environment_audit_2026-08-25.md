# 표적 동작·학습환경 독립 감사 (2026-08-25)

## 결론

현재 환경이 전부 잘못된 것은 아니다. `dt`, reset 순서, 센서 pose 동기화, vision actor의 oracle
차단, ref5in 기체/표적의 질량·관성·충돌박스 정합성은 코드와 CPU 검사에서 일관됐다. 그러나 세
계보는 같은 의미가 아니다.

- **legacy**는 체크포인트 재현용 kinematic 환경이다. 벽 반사, 막대 push-out 및 heading jitter가
  순간적으로 일어나므로 실제 표적 비행의 근거로 쓰면 안 된다.
- **bounded**는 0.1 s마다 가속도·방향전환을 제한하는 2-D kinematic 중간 계보다. 순간 teleport는
  제거됐지만 6-DoF, motor lag, contact가 없으며 공식 재현 launcher도 없다.
- **physical**은 100 Hz PhysX actor와 motor lag를 쓰지만, 현재 사전등록 strict gate는 1.5 m/s에서
  실패한다. 따라서 fresh physical PPO를 시작할 상태가 아니다.

이번 감사에서 안전하게 수정한 코드 결함은 두 가지다. checkpoint에 저장만 하고 restore 때
비교하지 않던 target acceleration/turn/lookahead/clearance 및 physical authority 값을 fail-loud
계약에 넣었다. 오래된 checkpoint는 필드가 없으면 기존처럼 통과하므로 legacy semantics는 바뀌지
않는다. 그리고 base v2 launcher가 interactive shell의 stale `NAVRL_TARGET_DYNAMICS`를 상속하던
경로를 닫아 legacy를 기본 강제하고 physical wrapper만 명시적 child marker로 통과시켰다. 또한
planner의 `feasible` 반환값이 full-horizon 인증처럼 적힌 docstring을 실제 의미인
**selected first-step feasibility**로 바로잡았다.

## 감사 범위와 기준

코드 기준점은 branch 생성 시 `6dc84f9`이다. GPU 학습·평가는 실행하지 않았다. 읽은 주 소스는
다음과 같다.

- motion/runtime: `aerial_gym/task/navrl_task/{navrl_task.py,target_motion.py,physical_target.py}`
- geometry/placement: `aerial_gym/config/env_config/navrl_bars_env.py`,
  `aerial_gym/config/asset_config/env_object_config.py`, `aerial_gym/env_manager/asset_manager.py`
- dynamics: `resources/models/environment_assets/objects/navrl_target_drone.urdf`, ref5in robot config/URDF
- launch contract: `train_navrl_v2_search.sh`, `train_navrl_physical_fresh.sh`
- existing receipts: `results/navrl_physical_target_speed_envelope_post_wall_brake_seed509/summary.json`

CPU geometry probe는 v2 40 m arena, `navrl_band`, seed 825, 20 layouts/density, 0.2 m grid에서
실행했다. 이는 static XY topology diagnostic이지 동적 경로 존재 증명이 아니다.

## 계보별 실행 계약

| 항목 | legacy | bounded | physical |
|---|---|---|---|
| 표적 상태 | task-side virtual point | task-side 2-D virtual point | dynamic PhysX rigid body |
| 명령 갱신 | 10 Hz | 10 Hz | planner 10 Hz, controller/motor 100 Hz |
| 적분 `dt` | 0.1 s | 0.1 s | physics 0.01 s × 10/RL step |
| 속도 | episode speed를 즉시 사용 | episode speed 상한 | velocity reference; realized speed는 dynamics 결과 |
| 가속/turn | 제한 없음 | 4 m/s², 150 deg/s 기본 | planner는 동일; 45° tilt, 0.04 s motor lag |
| 벽 | position+velocity mirror, final clamp | rollout bounds, clamp/reflect 없음 | OBB-support-aware bound + unsafe first-step zero command |
| 막대 | center-distance steer 후 composite push-out/bounce | rollout; center clearance | bar AABB + oriented target support + tracking margin |
| collision/contact | 없음 | 없음 | exact box contact/invalid termination |
| checkpoint 위치 | 기존 계보 유지 | shape는 같지만 다른 transition | fresh-only launcher |

### update rate와 순서

`base_sim.dt=0.01`, `navrl_bars_env.num_physics_steps_per_env_step_mean=10`, 따라서
`NavRLTask.step_dt=0.1`. 매 RL step에서 이전 sensor observation으로 action을 정한 뒤 target
command/virtual transition을 준비하고, pursuer와 physical target이 같은 physics interval을
진행한다. reward는 interval 끝의 두 상태로 계산한다. physical callback은 매 PhysX step 전 force를
쓰고 refresh 후 contact를 누적한다. 0.01/0.1 혼동은 발견되지 않았다.

### 초기화와 sampling

- v2 general spawn은 start/goal을 arena 내부에서 뽑고 target spawn은 bar rejection을 거친다.
  physical은 conservative center clearance에 최대 bar half-diagonal, target half-diagonal, tracking
  reserve를 포함한다.
- episode speed는 canonical v2에서 `U[0.3, vmax]`, `vmax→1.5 m/s`; physical wrapper는 ramp를
  1 epoch로 두며 base launcher가 이를 보존한다.
- `mixed`는 CV/waypoint 50:50이다. CV initial heading은 uniform `[0,2π)`. waypoint는 wall margin
  안 uniform이지만 **bar-free waypoint를 보장하지 않는다**. local planner가 그 방향을 회피할 뿐
  global route 또는 waypoint reachability 인증은 아니다.
- reset은 target position을 먼저 쓰고 motion state를 샘플한 뒤 realized velocity를 zero로 시작한다.
  physical은 orientation identity, linear/angular velocity zero, hover motor state로 함께 reset한다.

## 발견사항

### H1 — physical 1.5 m/s strict gate 실패 (높음, 학습 차단)

**증거.** 최신 보존 결과 `results/navrl_physical_target_speed_envelope_post_wall_brake_seed509/
summary.json`에서 1.5 m/s는 70/150/205/300 bars 모두 fail이다. speed ratio는 각각
0.824/0.780/0.739/0.720, immediate infeasible fraction은 10.74/8.02/8.23/4.16%이며 일부 cell에는
strict invalid sample도 남는다. 사전 gate는 speed ratio ≥0.8, immediate infeasible ≤1%, invalid=0다.

**영향.** 현재 1.5 m/s physical target을 “동일 난이도의 실제 동역학 환경”으로 부를 수 없고,
fresh physical PPO를 시작하면 실패한 환경 generator에 정책을 맞추게 된다.

**권장.** gate를 완화하거나 legacy reward를 조절하지 않는다. 사전등록된 density-conditioned speed
envelope(현재 모든 density에서 확실히 통과한 값은 0.6 m/s)를 과제 계약으로 채택할지 먼저 결정하고,
결정 후 별도 500-epoch smoke를 preregister한다.

### H2 — planner `feasible`은 global/full-lookahead route 인증이 아님 (높음, 해석)

**증거.** `bounded_drone_target_step`는 후보 전체 rollout 안전성을 selection에 쓰지만 반환 boolean은
선택 후보의 `immediate_feasible`. full horizon 후보가 하나도 없어도 첫 step이 안전하면 true다.
physical caller는 이를 hard AABB first-step check로 다시 계산한다. 기존 speed-envelope의
`planner_infeasible_fraction`은 정확히 말해 immediate hard-step failure다.

**영향.** 해당 비율이 0이어도 target이 목적지까지 갈 route가 있다는 뜻이 아니다. “경로 불능”과
“timeout”도 같은 개념이 아니다.

**조치.** 반환값 docstring을 바로잡았다. 이후 평가 전용으로 `full_horizon_available`,
`safe_prefix_steps`, waypoint age/reach rate를 별도 export해야 한다. control/reward에는 넣지 않는다.

### M1 — center-distance placement는 AABB corridor 보증이 아님 (중간)

**증거.** bar pool 40개의 width/depth는 0.4123–0.7883 / 0.4029–0.7902 m이고 평균 footprint는
0.3653 m²다. `navrl_band`는 center 거리 `≤0.4`(merge) 또는 `≥1.6`만 검사한다. 최대 square의
대각 배치에서는 표면 gap이 약 0.469 m까지 줄 수 있다. 0.28 m level box는 0.189 m 총 여유만 남고,
tilt/추종/제동 reserve는 포함하지 않는다.

초기 CPU topology probe(seed 825, **12 layouts/density**, 0.15 m grid)는 작은 side-clearance
0/0.1 m에서 70/150/205/300 bars의 sampled connectivity가 거의 1이었다. 이것은 level 0.28 m box의
half-width 0.14 m에 작은 여유만 더한 옛 계약이며 physical routed contract의 all-orientation support와
0.45 m tracking reserve를 포함하지 않는다.

후속 20-layout **route-equivalent obstacle-inflation raster probe**는 side-clearance 0.517 m를 사용해
총 obstacle inflation을 0.6570 m로 만들었다(route의 0.2068816+0.45=0.6568816 m보다 0.118 mm
보수적). largest-component/random-pair/crossing은 205 bars에서 0.9577/0.9211/1.0,
300 bars에서 0.3998/0.2247/0.3이었다. 따라서 300-bar sampled free space는 명확히 분절됐다.
다만 이 probe는 0.15 m raster, diagonal corner-cut 허용, full-arena boundary를 쓰므로 route planner의
0.25 m/no-corner-cut/exact LOS/1.25 m+support boundary 계약과 동일한 reachability 증명은 아니다.

**권장.** 환경 배치 규칙을 지금 바꾸면 기존 연구와 다른 task가 된다. 다음 physical lineage에서만
footprint-aware placement 또는 exact configuration-space route receipt를 preregister하고, legacy/v2
결과에는 center-band 배치라고 명시한다.

### M2 — legacy와 bounded는 물리적 표적 근거가 아님 (중간, claim boundary)

legacy wall mirror, composite bar push-out, CV specular bounce와 ±10° jitter는 위치/방향을 순간 변경한다.
bounded는 vector acceleration/heading slew를 지키고 teleport/push-out을 제거했지만, rigid-body attitude,
thrust allocation, motor saturation/contact를 모델링하지 않는다. 둘은 학습 curriculum/ablation에는 쓸 수
있어도 실기 가능성의 근거는 physical gate뿐이다.

### M3 — bounded canonical launcher 부재 (중간, 재현성)

코드/config/test에는 bounded mode가 있으나 `NAVRL_TARGET_DYNAMICS=bounded`를 고정하고 checkpoint
정합성을 검사하는 전용 training launcher가 없다. shell 환경에 의존한 수동 실행은 speed ramp,
airframe 및 provenance drift 위험이 있다.

**권장.** bounded를 더 학습할 계획이 생길 때만 fresh-only launcher와 preflight receipt를 만든다.
지금 launcher를 추가하면 사용하지 않을 계보를 canonical처럼 보이게 하므로 이번에는 만들지 않았다.

### M4 — checkpoint restore가 motion authority drift를 놓쳤음 (중간, 수정 완료)

accel/turn/lookahead/clearance와 physical arm/yaw ratio/max tilt 값은 checkpoint에 저장됐지만 restore
비교 목록에 없었다. tracking/boundary margin은 저장도 되지 않았다. shape-compatible checkpoint가
다른 target transition 아래서 조용히 load될 수 있었다.

**조치.** 모든 필드를 저장·비교하고 mismatch 시 기존 계약대로 경고+curriculum evidence reset한다.
필드가 없는 옛 checkpoint는 그대로 허용한다. observation/reward/termination/policy tensor는 변하지
않는다.

### M5 — base launcher가 stale target lineage를 상속했음 (중간, 수정 완료)

`train_navrl_v2_search.sh`는 fresh/checkpoint CLI는 닫았지만 `NAVRL_TARGET_DYNAMICS`를 고정하지 않았다.
따라서 shell에 `bounded`/`physical`이 남아 있으면 canonical v2 명령이 shape-compatible한 다른 state
transition으로 시작될 수 있었다. base launcher는 이제 legacy만 허용하며, physical은
`train_navrl_physical_fresh.sh`의 child marker와 matching ref5in 설정을 함께 요구한다.
bounded는 canonical launcher가 없으므로 명시적으로 거부한다.

### M6 — 계보별 초기 transient가 같지 않음 (중간, 비교 해석)

모든 계보가 reset 직후 `target_vel_w=0`으로 시작하지만 legacy는 첫 0.1 s step에 episode speed를
즉시 적용한다. bounded는 4 m/s² acceleration limit로 ramp하고, physical은 같은 reference 위에 motor/
attitude response가 추가된다. physical orientation은 reset 때 항상 identity이며 random CV heading과
정렬해 초기화하지 않는다. square XY hull이라 collision footprint의 yaw bias는 작지만, initial yaw
transient/motor effort는 계보별로 다르다. speed label만 맞춘 첫-second 비교를 동역학 동등 비교로
해석하지 않는다. 이를 바꾸면 task semantics가 달라지므로 이번에는 수정하지 않았다.

### L1 — orientation/velocity/sensor sync (낮음, 통과)

physical target pose/velocity는 actor tensor가 단일 source이며 camera와 analytic LiDAR에 동일
orientation이 전달된다. reset에서 pose/orientation/linear/angular/controller state가 함께 초기화되고,
step 뒤 sensor buffer가 새 pose로 동기화된다. virtual 계보의 realized velocity도 실제 displacement로
다시 계산돼 range-rate reward와 일치한다. 확인된 sync bug는 없다.

### L2 — reward/observation leakage (낮음, 통과·조건부)

perception actor는 sensor/ego/history만 받고 `target_position`은 renderer 입력으로만 사용된다.
GT relative pose/target velocity는 분리된 central-value `states`에만 들어가는 asymmetric critic이며,
reward/termination은 simulator GT를 사용한다. 이는 leakage가 아니라 현재 training contract다.
non-vision legacy actor는 GT goal state를 직접 관측하므로 sensor-only 결과와 섞어 비교하면 안 된다.
CPU `test_navrl_perception.py`의 oracle-API guard도 통과했다.

## CPU 검사 결과

| 검사 | 결과 |
|---|---|
| target motion math/contracts | 13/13 PASS |
| ref5in/physical target URDF·config | 27/27 PASS |
| physical speed-envelope helper math | 3/3 PASS |
| perception/oracle API | 30 PASS, 1 optional skip |
| reachability oracle regression | 3/3 PASS |
| 신규 checkpoint/launcher guards | 5/5 PASS |
| ref5in run-contract suite | 100 PASS, 1 fixture-missing (ignored `runs/`가 독립 worktree에 없음) |
| v2 static topology (old small margin) | 70/150/205/300 bars × margin 0/0.1 m, 12 layouts/density에서 sampled connectivity 유지 |
| routed obstacle-inflation raster topology | 20 layouts/density; 205 pair 0.9211, 300 pair 0.2247/crossing 0.3; exact route reachability 아님 |

fixture-missing 한 건은 test가 요구하는 historical checkpoint가 main worktree의 ignored `runs/`에만
있고 감사 worktree에는 없어서 발생했다. 소스 assertion failure로 계산하지 않았다.

## 다음 순서

1. **학습 금지 유지:** physical 1.5 m/s strict gate가 닫혔다고 쓰지 않는다.
2. **과제 결정:** density-conditioned target speed를 채택할지, 1.5 m/s controller/arena 문제를 별도
   공학 과제로 남길지 결정한다. 둘을 한 run에서 섞지 않는다.
3. **평가 전용 계측:** full-horizon candidate existence, safe-prefix length, waypoint reach/age를 추가해
   “local first-step failure”와 “global route unavailable”을 분리한다.
4. **그 뒤에만 smoke:** 새 physical task contract가 고정되면 fresh 500 epochs + held-out cells.
5. legacy/bounded checkpoint 결과는 historical baseline으로 보존하고 physical claim으로 승격하지 않는다.

## 2026-08-25 routed PhysX gate 결과 (seed 827)

attempt 1은 task 생성 전 conda `ninja` PATH 문제로 0/32 record의 `VOID_EXECUTION`이었다. 이
실행은 수치 결과가 아니며, 원자료는 별도 보존했다. PATH를 고친 별도 attempt 2는 정확한 32-cell
grid를 모두 기록해 **execution integrity 32/32 PASS**를 얻었지만, route mechanism은 FAIL이고
physical PPO는 `BLOCKED_PHYSICAL_TRAINING`이다. `summary.json`의 claim boundary는 PPO policy
미탑재, hardware validation 없음, 300-bar arena-wide connectivity claim 금지를 그대로 명시한다.

route-off/on 결과를 `P/F`로 요약하면 다음과 같다(각 셀은 off/on strict gate).

| bars \ speed | 0.6 | 0.9 | 1.2 | 1.5 |
|---:|:---:|:---:|:---:|:---:|
| 70 | P / F | P / F | F / F | F / F |
| 150 | P / F | P / F | P / F | F / F |
| 205 | P / F | P / F | F / F | F / F |
| 300 | P / F | F / F | F / F | F / F |

route-on local invalidation은 0.125–0.635%로 보이지만 fallback은 32.6–85.0%까지 증폭됐다.
70-bar 4-speed pool의 plan success는 `0.1454668471` (gate ≥0.99), fallback은
`0.359296875` (gate ≤0.01)이고, 별도 70×0.6 cell의 goal completions/env는 `0.25`
(gate ≥0.5)다. same-goal reselection은 전체 routed cell에서 0이라 해당
guard는 원인이 아니다. 16 route-on cell의 status counter 합계는 `unsafe_start` 420,
`ok` 80, `no_path` 6, `local_step_infeasible` 6으로, 현재 가장 강한 진단은 작은 local
invalidation 뒤 soft-envelope recovery가 `unsafe_start`에서 되돌지 못하는 deadlock이다.
이는 global route 부재의 증명이 아니다. motor saturation·tilt·contact을 주원인으로 승격하지
않으며 tracking은 별도 gate metric으로 유지한다. 다음에는 safe-prefix/full-horizon 및 unsafe-start recovery를 별도 계측하고,
threshold/evaluator/preregistration은 바꾸지 않은 채 동일 grid를 재실행한다.

attempt 2 provenance: summary `e5e4560464dc3a2080d904c2f8d2247e0c65e671dd63ea08d1b507ec65fc7197`,
execution manifest `896b05c9bf2cea672aad5e95a8e2a893d6eaa6999c56d5ab8ab60fc4c0de4291`, source
manifest `9af2f58b42176c4800cf5f8795dbf8f009097d3ae5f74c9ef16937730d6a1aad`, receipt
`bd840ef6dd157aa317dfeccfbf347235ac858d541e1e35e9e50c130084fc7771`. 상세 표와 raw links는
[`attempt 2 summary`](../results/navrl_physical_target_routed_gate_seed827_attempt2/summary.md)와
[`attempt 1 VOID`](../results/navrl_physical_target_routed_gate_seed827/VOID.md)다.

## 2026-08-26 supersession addendum

위 “다음 순서”와 attempt 2 말미의 동일-grid 재실행 문구는 2026-08-25 당시 권고이며 현행
authority가 아니다. 후속 recovery-v2 lower-1.25는 32/32 integrity를 통과했지만
`FAIL_ROUTE_MECHANISM`이다: 7/32 PASS(모두 route-off), recovery 0/16, 70-bar plan `93.60%`,
fallback `47.87%`, 70×0.6 goals/env `0.21875`, `NO_CONNECTOR` occupancy `63.06%`.
사전등록된 no-anchor probe도 실행·검증됐지만 primary `n=1`, identity disagreement `0`으로
`INCONCLUSIVE`다.

따라서 이 감사는 역사적 원인 분석으로 보존하되, 추가 Track B GPU/PPO/retune/1.5/env-count/32-cell
rerun을 승인하지 않는다. 현재의 유일한 다음 authority는
[`SIM2REAL_3DAY_EXECUTION_PLAN.md`](SIM2REAL_3DAY_EXECUTION_PLAN.md)의 hardware BOM/calibration,
210 trials와 real-log replay이며, hardware/log가 없으면 GPU 작업은 없다.
