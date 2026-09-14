# MOTAR / NavRL — 작업 규칙 (매 세션 로드)

## 작업 방식 — 스킬 먼저, 무거운 조사는 서브에이전트 (토큰 절약)

substantive한 요청(학습·평가·분석·감사·구현)을 받으면:

1. **스킬 먼저 탐색·호출한다.** 연구 루프는 `/navrl` 스킬로 시작한다
   (`.claude/skills/navrl/`). 스킬은 **먼저 `VERIFICATION.md`에서 현재 stage를 읽고**,
   학습을 자동으로 시작하지 않는다. routed v3는 **MECHANISM_GATE_FAIL_CLOSED**다.
   route-off 70→205 curriculum(seed 911)은 **RUNNING이 아니다** — 2026-09-11 정정. ep21973에서
   operator가 중단했고 `docs/research_authority_2026-08-26.json`이
   `route_off_curriculum.status = OPERATOR_STOPPED_INCOMPLETE`, `resume_forbidden = true`로 잠근다.
   두 번째 run을 시작하지 않는 이유는 **RUNNING이어서가 아니라 GPU authority가 소진돼서**다.
   `tools/check_research_authority.py`가 이 상태를 기계로 검사하니 상태는 그쪽을 믿을 것.
   대시보드 갱신은 저장소
   `research-status` 스킬.
2. **무거운 조사는 서브에이전트(Agent 도구)에 위임한다** — 파일 다수 읽기, `runs/**/epoch_metrics.csv`
   파싱, `nn/*.pth` 로드, 긴 학습 로그 스캔. 메인 루프는 짧은 구조화 보고만 회수한다.
   독립 조사는 **한 메시지에 여러 Agent**로 병렬 실행. 서브에이전트 `*.output` 트랜스크립트는
   **절대 셸로 읽지 말 것**(컨텍스트 폭발). 서브에이전트가 세션/주간 한도로 죽으면 read-only 직접
   조사로 폴백하거나 리셋 후 재개.
3. **메인 컨텍스트에 `runs/**/*.csv`·`nn/*.pth`·긴 로그를 직접 로드하지 않는다.**

## 프로젝트 현재 상태 (재도출 금지 — 여기 요약)

- **현재 실행 단계(2026-09-11 갱신)**: corrected non-overlap route r2는
  `FAIL_ROUTE_MECHANISM`이다. route-off 500-epoch smoke는 `PASS_LEARNING_VIABILITY`이고
  seed 911 70→205 curriculum은 **OPERATOR_STOPPED_INCOMPLETE**(ep21973, `resume_forbidden`)다.
  seed-313 held-out sweep은 완료됐다. **학습·평가 프로세스는 없다.**
  두 번째 curriculum/routed PPO는 여전히 금지이며, 사유는 RUNNING이 아니라 authority 소진이다.
  (2026-09-01판은 RUNNING이라고 적어 두었는데 아래 85번 줄과도, 권위 JSON과도 모순이었다.)
  canonical 1.5 v3 receipt는 NO-GO다.
  matched-spawn lower-v3 재실행은 pose/layout/robot 해시가 일치해
  `PASS_8_CELL_INTEGRITY`였지만 공식 판정은 **`FAIL_BLOCKS_CONFIRMATORY`**다. routed speed
  ratio 0.7872/0.7681/0.5984/0.6297, soft exits 6,825, fallback 8.914%, 0.6 goals/env
  0.21875다. 제한 GPU authority는 소비됐고 confirmatory/PPO는 닫혀 있다. 새 방법
  사전등록 없이 rerun·retune하지 않는다. 0.05 warmup 게이트와 PID/게인은 바꾸지 않는다.
  현재 계보 밀도는 **70/115/160/205**이고 **300 bars는 disconnected stress**다.
- **주제**: 센서 전용 UAV 요격. actor 관측에 GT 표적(semantic id/mask, bearing/range, `target_position`,
  GT visibility) **절대 금지** — detector supervision·reward·종료판정·critic·평가 metric에만 사용.
- **현재 방향(2026-07-22 피벗)**: NavRL++식 **학습형 인지(RGB-D 카메라 + LiDAR detector/tracker) +
  Transformer 시계열 정책**. 계획서 = `RESEARCH_PLAN.md`.
- **현재 관측 계약(2026-07-30)**: actor **898차원** / critic **906차원**(=898 + GT 8). 장애물 토큰 8개를
  `cluster_sector` 셀렉터로 전방 240°에서 선택, LiDAR 72×4 @12 m, 17-token Transformer.
  폐기된 계보: 156(GT LiDAR) → 305(해석적 semantic) → 1265(vision CNN). 체크포인트 shape 불일치의
  원인이 되므로 섞지 말 것.
- **canonical 현재 상태**: `WORKLOG.md`(맨 아래) + `VERIFICATION.md`(ref5in gate·다음 실험) +
  `docs/status/` 라이브 대시보드. `RESEARCH_PLAN.md`는 charter(가설·방법). 실무는 `OPERATIONS.md`,
  과거 crash 진단 기록은 `CRASH_TUNING_LOG.md`(내용은 2026-08-05에 동결, archival-in-place —
  **소스 3곳**이 경로를 참조하므로 이동·삭제 금지 — 줄번호는 드리프트하므로 파일만 적는다:
  `navrl_task_config.py`, `navrl_lidar_config.py`, `navrl_task.py`. 확인은
  `grep -rn CRASH_TUNING_LOG aerial_gym/`). 역사 문서는 `docs/archive/`.
- **경로가 계약인 문서 2건 — 이동·삭제·바이트 변경 금지**:
  ① `docs/reference_platform_proposal_2026-08.md` — `navrl_ref5in_quad_config.py`의 docstring이
  가리키고 그 파일이 **provenance-frozen**이라, 바이트가 하나만 바뀌어도
  `eval_navrl_v2_density_sweep.sh:240-242`가 `robot config source drift`로 `exit 2`한다
  (`NAVRL_V2_FORCE`로 우회 불가). 커밋 `921fb1d`가 이걸 '정리'했다가 8개 브랜치를 깼다.
  잠금: `tests/test_navrl_ref5in_provenance_freeze.py:39,86`.
  ② `docs/prereg_2026-08-13_detector_coupling.md` — 원본은 archive에 있고 이 경로는 **스텁**이다.
  소스 5곳이 이 경로를 인용하므로 스텁을 지우면 인용이 다시 깨진다(2026-09-05 복구).
- **현재 상태(2026-08-03)**: recovery curriculum continuation
  `ppo_260802_0020_navrl_v2-recover-curriculum-continue-s1`은 사용자 결정으로 ep24010에서 안전 중단했다.
  canonical artifact는 ep24000 checkpoint(SHA-256 `82f7978b42d…`)이며 학습 프로세스는 없다.
  205 bars에서 4,910 epoch/10회 hold 뒤 stochastic gate는 68.94%(최근 7회 평균), 동일 checkpoint의
  held-out deterministic/stochastic capture는 72.44%/67.35%다. PPO 발산과 free-space 단절은 주원인에서
  기각했다. frozen ep24000의 mirror/second-seed 평가는 완료됐다. seed43 capture는 72.77%(seed42 대비
  +0.33 pp, 재현 PASS)이고 초기 target-bearing 좌/우 capture 차이는 -0.21 pp로 outcome asymmetry는
  검출되지 않았다. 반면 정확한 reflected-observation pair에서 lateral action MAE 1.235, sign mismatch
  73.08%로 학습된 turn-direction chirality는 명확하다. fixed-speed는 0.3→1.5 m/s에서 capture
  73.26→67.35%(-5.91 pp), bar contact 22.11→29.97%(+7.86 pp)였다. ep19100→ep24000은 uniform
  +4.65 pp, fixed-1.5 +3.25 pp로 **망각 없이 개선**했다. 1650 Ti의 70-bar TTC A/B도 capture
  +9.86 pp/crash -8.06 pp로 PASS했다. main RTX 3070 고정-205 `cluster_sector` baseline은
  ep24001→25000, 4.096M samples를 `max_epochs`로 정상 완료했다. final SHA는 `169ddcddb83c…`이며
  KL max 0.00757, behavior-KL max 0.01236, rollback/OOB 0이었다. 이 checkpoint의 seed42
  deterministic/original 205-bar held-out도 2,049회에서 capture/crash/timeout
  **69.50/28.99/1.51%**로 완료됐다. ep24000 대비 capture -2.94 pp/crash +3.92 pp이며 KL이 정상인
  느린 action drift다. 이어서 같은 예산의 main `ttc_sector` arm과 2,051회 held-out도 완료했다.
  결과는 **70.21/29.50/0.29%**로 사전 등록한 primary gate(≥71.50/≤26.99%)와 ep24000 replacement
  floor(≥72.44/≤25.07%)를 모두 FAIL했다. baseline 대비 capture +0.71 pp, crash +0.51 pp이므로
  current TTC mode는 채택하지 않는다. 또한 selector switch가 candidate FOV 240°→360°를 함께
  바꾸므로 pure ranking effect는 판정 불가다. 이어서 sensor-only control-risk를 분리했다. semantic-mask
  leak과 LiDAR 수직행 역순을 찾아 앞선 speed-governor 자료를 무효·격리한 뒤 처음부터 재실행했다.
  corrected complete-stop clearance/TTC는 timeout 16.59/23.57%로 기각했다. 멈추지 않는 단일 `riskcap`
  후보는 seed44에서 capture/crash **79.55/17.62%**(off 72.83/24.63%)로 gate를 통과했고, 정확히
  1,000 epoch/4.096M samples 적응한 checkpoint(SHA `f70221393660…`)는 새 seed45 uniform에서
  **81.94/15.67/2.39%**를 기록했다. source+riskcap 78.20/17.80/4.00% 대비 capture +3.75 pp이며,
  seed46 fixed 0.3/0.9/1.5 m/s도 capture↑/crash↓ 방향을 3/3 통과했다. 이 ep25000+riskcap을 현재
  navigation/control candidate로 동결한다. 다음은 learned detector/perception robustness다.
  **금지 2건 — 사유와 재검 조건 (2026-08-22 부여, `docs/discipline_review_2026-08-22.md`)**:
  ① fixed-density PPO 연장 금지 — 사유: 실패를 epoch 추가로 덮는 것을 막는다. 재검: 밀도가
  아닌 축에서 병목이 특정되고 그 처방이 사전등록됐을 때. ② riskcap 사후튜닝 금지 — 사유:
  seed 44 speed-governor 자료가 semantic-mask leak으로 오염됐었다. **그 오염은 이미 수정·재실행**
  됐으므로 이 금지의 원인은 소멸했다. 재검: 깨끗한 재측정 위에서 사전등록된 A/B라면 허용. 현재 학습/평가 프로세스는 없다.
  핵심 보고는 `results/navrl_v2_ep24000_limit_audit.{md,json}`과
  `results/navrl_v2_riskcap_postadapt/summary.{md,json}`이다.
- **하드웨어**: RTX 3070 8GB(학습 공장) + GTX 1650 Ti 4GB(평가 공장 — 학습은 N=128이라 ~10-15pt 약함,
  결과 섞지 말 것). RTX 50번대(Blackwell)는 Isaac Gym Preview4와 비호환 — 사지 말 것.

## 비싸고 미탐색인 축 (금지 아님 — 값을 매겨서 다룰 것)

`NAVRL_MAX_VELOCITY`(2.5) · `NAVRL_MAX_TILT_DEG`(45) · `NAVRL_YAW_RATE_MAX`(2.5) 는 **실험 이력
0건**이다. `free_speed_cap_mps = 3.5355`도 최적값이 아니라 `2.5 x sqrt(2)`라는 축별 제한의 기하학적
귀결일 뿐이다. 이것들은 **금지된 적이 없고**, riskcap 사후튜닝 금지 + 한 run 한 축이 겹쳐 사실상
접근 불가처럼 취급돼 왔다(2026-08-22 재검에서 확인).

실제 비용은 이것이다: `max_velocity`는 관측 정규화의 분모다(`navrl_task.py:3914,3958`). 바꾸면
동결 체크포인트와의 계약이 어긋나므로 **재학습이 필요한 축**이다. 물리적으로도 정지거리 `v²/2a`와
선회반경 `v²/a`가 제곱으로 커지고(2.5 m/s에서 1.06/0.64 m, 4.0 m/s에서 2.70/1.63 m), 올리려면 틸트
상한도 함께 올려야 하며 틸트는 필요추력을 `1/cosθ`로 키운다(60° 2배, 70° 2.9배; 현재 T/W 3.26).

**"금지"가 아니라 "값이 비싼 미탐색 축"으로 다룬다.** 손대려면 재학습 예산과 사전등록이 필요하다.

## 핵심 함정 (재학습 금지)

- **밀도 커리큘럼 run 평가는 `last_gen_ppo_ep_XXXX.pth`로.** `gen_ppo.pth`(best-reward)는 저밀도 정책이라
  고밀도 평가 시 ~15%로 오독됨.
- warm-start: `--checkpoint <p> --max_epochs <N>` (runner가 critic `_orig_mod` 자동 정규화 + max_epochs override).
- 커리큘럼 knob은 전부 env-var: 밀도 `NAVRL_DENSITY_*`, 거리 `NAVRL_K_*`. v2 recovery의 density
  gate는 단일 0.6이 아니라 measured knot schedule `0.82@70, 0.77@85, 0.72@100, 0.70@115+`,
  16,384 episodes, 최소 dwell 1,000 epoch다. rolling tail을 이 gate나 held-out 수치로 오독하지 않는다.
- canonical epoch 계보는 smoke 9501–9600 + curriculum 9601–20700 + continuation 20701–24010이다.
  이전 curriculum run의 20701–20746은 ep20700에서 재학습된 중복 구간이므로 최종 통계에서 제외한다.

## WORKLOG 규칙 — 예외 없음

**모든 작업은 `WORKLOG.md` 항목 작성으로 끝난다.** 루프 한 바퀴만이 아니라 아래 각각에 대해:

- 학습 run (시작/완료/사망/중단 — 어느 쪽인지와 이유를 기록)
- 평가·스윕 (실제 숫자를 기록. "좋아졌다"가 아니라 값)
- 코드 변경 (무엇을, 무엇 대비로 측정했는지)
- 진단 — **틀린 것으로 판명된 가설도 반드시**. 기각된 가설도 결과이며, 다음 세션이 같은 걸 다시
  돌리는 걸 막아준다.

형식 (최신이 `WORKLOG.md` **맨 아래**):
- `## YYYY-MM-DD — <한 줄 헤드라인>`
- 측정 숫자 (셀이 2개 이상이면 표로)
- 그로 인한 결정과 다음 구체적 단계
- run 폴더 / 체크포인트 / `results/*.csv` 경로 (숫자를 재도출할 수 있게)
- 반증된 주장은 명시 ("가설 X 기각: Y로 측정됨")

**diff 검토를 요청하기 전에** 작성해서 코드와 문서가 같은 커밋에 들어가게 한다. 세션 예산이
부족해도 WORKLOG 항목은 **마지막까지 자르지 않는다** — 다음 세션을 싸게 만드는 게 이것이다.
장문의 메커니즘 설명은 `WORKLOG.md`에 날짜별로 남긴다(`CRASH_TUNING_LOG.md`는 2026-08-05에서
멈춘 과거 기록이며 새 항목을 추가하지 않는다).

## 커밋 규칙
- **커밋/푸시 전 사용자가 diff 검토·승인.** 자율 커밋 지양. **작업 브랜치는 `main` 하나**, 원격
  joshualikaist/MOTAR. 장기 작업 브랜치를 만들지 않는다 — 규칙과 사유는 `OPERATIONS.md` §0.
- `runs/`(4GB+)·`nn/`·tfevents·로그는 gitignore. 결과는 `results/<sweep>.csv`로 소량만 추적.
- 여러 클로드/커서 세션이 동시에 작업할 수 있음 — 미커밋 변경이 남의 WIP일 수 있으니 함부로 커밋하지 말 것.
