# MOTAR 운영 가이드

이 문서는 “어떤 명령을 복사해야 하는가”와 “결과를 어떻게 잃지 않는가”만 다룹니다. 연구 가설과
검증 gate와 다음 실험은 [`VERIFICATION.md`](VERIFICATION.md), charter는 [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md),
프로젝트 입구는 [`README.md`](README.md), 날짜별 기록은 [`WORKLOG.md`](WORKLOG.md)를 보세요.

## 현재 운영 상태 — 2026-09-12

P9·P10·S4·R-C·P10 시드 반복은 완료됐다. P10 반복은 INCONCLUSIVE, C3는 WITHDRAWN,
E3-P는 ATTITUDE_NOT_RELIABLE 검사 후 BLOCKED다. 현재 근거는 [VERIFICATION](VERIFICATION.md).
아래 날짜별 launcher는 **소비된 실행 계약의 보존 기록**이며 재실행 지시가 아니다.
현재 작업은 문서·재현성·공개 배포 정리와 정책에서 분리된 renderer smoke만 포함한다.
새 학습·통합 제어 실험은 이 문서 개편으로 승인되지 않는다.

> 아래 절의 역사적 기준일: 2026-09-01
>
> Track A는 exact BOM·calibration·210개 sensor
> trial·real-log replay를
> [`docs/SIM2REAL_3DAY_EXECUTION_PLAN.md`](docs/SIM2REAL_3DAY_EXECUTION_PLAN.md)에 따라 진행합니다.
> 실제 hardware나 real log가 없으면 GPU 작업을 시작하지 않습니다. Track B recovery-v2는
> `FAIL_ROUTE_MECHANISM` 뒤 no-anchor probe까지 `INCONCLUSIVE`로 종료됐으며 추가
> GPU/PPO/retune/rerun authority가 없습니다. corrected non-overlap r2는
> `FAIL_ROUTE_MECHANISM`이고 새 PPO는 0 epoch입니다. canonical 1.5 v3 receipt는 NO-GO입니다.
> software 쪽의 별도 execution addendum가 허용한 **새 baseline_1p25 receipt 1회와
> matched-spawn lower-v3 8-cell pilot 1회는 모두 소비됐습니다.** 결과는
> `PASS_8_CELL_INTEGRITY / FAIL_BLOCKS_CONFIRMATORY`입니다.
> 직전 8-cell pilot는 `VOID_EXECUTION`(matched-arm target pose drift)입니다. spawn 바이트가
> `dd8b4a4` receipt와 두 pilot 출력 루트는 재사용하지 않습니다. confirmatory와 PPO는
> 차단됐고, 아래 L1–L3 명령은 provenance 기록일 뿐 다시 실행할 authority가 아닙니다.
> 별도 route-off 70-bar/500-epoch fresh smoke는 `PASS_LEARNING_VIABILITY`로 끝났습니다. 이에 따라
> 별도 사전등록된 fresh route-off 70→205 curriculum 1회만 열렸습니다. routed PPO 권한은 아닙니다.

GPU 명령을 찾기 전에 아래 동결 receipt를 먼저 검사합니다. 이 명령은 학습이나 평가를 실행하지
않고, Track A/B 원자료 SHA와 `Stage 2=false`, `long training=false`를 fail-closed로 확인합니다.

```bash
cd /home/fair/workspaces/aerial_gym_ws/.codex_worktrees/braking_route_v3
/home/fair/miniconda3/envs/aerialgym/bin/python tools/check_research_authority.py --json
```

기계 판독 계약은 [`docs/research_authority_2026-08-26.json`](docs/research_authority_2026-08-26.json)이다.
matched-spawn receipt/pilot과 route-off curriculum 권한은 모두 소비됐다.

## Corrected non-overlap route-off learning smoke (1회)

```bash
cd /home/fair/workspaces/aerial_gym_ws/.codex_worktrees/braking_route_v3/aerial_gym/rl_training/rl_games
nohup ./train_navrl_corrected_nonoverlap_physical_smoke.sh \
  > train_session_logs/corrected_nonoverlap_smoke_nohup.out 2>&1 &
echo $!
./watch_navrl_training.sh
```

고정값은 fresh seed 907, 500 epochs, 128 env, fixed 70 bars,
`footprint_clearance`/surface 0.45 m/overlap fallback off, physical route-off mixed target,
`U[0.3,1.25] m/s`, LR `1.5e-5`다. checkpoint/CLI override를 거부하며 source receipt와 정확한 import
root를 강제한다. 판정은
[`preregistration_corrected_nonoverlap_physical_off_smoke_2026-09-01.md`](docs/preregistration_corrected_nonoverlap_physical_off_smoke_2026-09-01.md)를 따른다.

위 smoke는 완료됐으므로 다시 실행하지 않는다. seed 911 curriculum
(`ppo_260901_1431_navrl_corrected-nonoverlap-physical-off-curriculum-s911`)은 운영자 중지로
`OPERATOR_STOPPED_INCOMPLETE`다. 재개 금지. 직전 `1259`는 세션 단절로 중단됐다.

그 런의 `last_gen_ppo_ep_21750` held-out 평가는 완료됐다. 계약은
[`preregistration_corrected_nonoverlap_physical_off_heldout_eval_2026-09-02.md`](docs/preregistration_corrected_nonoverlap_physical_off_heldout_eval_2026-09-02.md)를
따랐고, 결과는
[`summary.md`](results/navrl_corrected_nonoverlap_physical_off_heldout_seed313/summary.md)에 있다.
다시 실행하지 않는다. 기본 density sweep `70 150 210 280`도 쓰지 않는다.

```bash
cd /home/fair/workspaces/aerial_gym_ws/.codex_worktrees/braking_route_v3
/home/fair/miniconda3/envs/aerialgym/bin/python tools/check_research_authority.py --json
# 과거 실행의 provenance 확인용이다. P10 시드 반복도 이미 완료됐다.
```

## 완료된 실행 기록 — P10 시드 재현 (2026-09-10)

두 새 학습 시드와 평가가 완료됐으며 판정은 INCONCLUSIVE다.
[결과](results/perception_p10_seed_replication_2026-09-10/README.md). 아래 명령은 당시 기록이다.

계약: [`preregistration_p10_seed_replication_2026-09-10.md`](docs/preregistration_p10_seed_replication_2026-09-10.md).
학습 시드 857, 863 각각 1,000 epoch(약 1시간), 이어서 캠페인당 4셀 × 평가 시드 2개(약 40분).
평가 시드는 541·547로 P10과 동일하게 유지하고 source 셀도 캠페인 안에서 다시 돌립니다.

```bash
cd /home/fair/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games
P10_SEED=857 nohup bash train_navrl_v2_p10_empirical_readapt.sh   > train_session_logs/p10_seed857_nohup.out 2>&1 &
```

TensorBoard는 `aerial_gym/rl_training/rl_games/runs`를 logdir으로 포트 **6009**에 띄웁니다.
학습·평가가 도는 동안 `aerial_gym/`, `tools/`, `resources/robots`는 동결 경로입니다.

fresh seed 911 curriculum은 이미 소비됐다. 밀도 70→205/step 15/dwell 1,000, route off,
`U[0.3,1.25] m/s`, 비중첩 surface 0.45 m. 학습 계약은
[`preregistration_corrected_nonoverlap_physical_off_curriculum_2026-09-01.md`](docs/preregistration_corrected_nonoverlap_physical_off_curriculum_2026-09-01.md)를 따른다.

현재 software MECHANISM_GATE(`global_astar_braking_v3`)의 CPU 계약은 아래와 같습니다.

```bash
cd /home/fair/workspaces/aerial_gym_ws/.codex_worktrees/braking_route_v3
export PYTHONNOUSERSITE=1
/home/fair/miniconda3/envs/aerialgym/bin/python -m unittest discover -s tests -p 'test_navrl_braking_route_v3*.py'
/home/fair/miniconda3/envs/aerialgym/bin/python -m unittest discover -s tests -p 'test_navrl_target_route_planner.py'
/home/fair/miniconda3/envs/aerialgym/bin/python -m unittest discover -s tests -p 'test_navrl_target_motion.py'
/home/fair/miniconda3/envs/aerialgym/bin/python -m unittest discover -s tests -p 'test_navrl_two_envelope_recovery.py'
git diff --check
```

canonical 1.5 v3 단계 1은 2026-09-01에 NO-GO로 재현됐습니다(1.5 warmup mean 1.442577 m/s).
현재 GPU 순서는 **lower-contract v3**입니다. conda `aerialgym` Python, `PYTHONNOUSERSITE=1`,
**커밋된 clean tree**, `NAVRL_TARGET_BRAKING_CONTRACT_VARIANT=baseline_1p25`. 셀 어댑터는
tracked `tools/run_navrl_braking_route_v3_cell.py`입니다. 2026-08-26 lower receipt는 재사용하지
않습니다.

**단계 L1 — 완료·소비됨, 재실행 금지.** 속도는 0.6/0.9/1.2/1.25였습니다.

```bash
cd /home/fair/workspaces/aerial_gym_ws/.codex_worktrees/braking_route_v3
export PYTHONNOUSERSITE=1
export NAVRL_TARGET_BRAKING_CONTRACT_VARIANT=baseline_1p25
NAVRL_BRAKING_PYTHON=/home/fair/miniconda3/envs/aerialgym/bin/python \
NAVRL_NINJA=/home/fair/miniconda3/envs/aerialgym/bin/ninja \
/home/fair/miniconda3/envs/aerialgym/bin/python \
  tools/run_navrl_physical_target_braking_v2_fresh.py \
  --output results/navrl_physical_target_braking_lower1p25_matched_spawn_seed827_2026-09-01
sha256sum results/navrl_physical_target_braking_lower1p25_matched_spawn_seed827_2026-09-01/receipt.json
```

**단계 L2 — 완료·소비됨, provenance 기록.**

```bash
cd /home/fair/workspaces/aerial_gym_ws/.codex_worktrees/braking_route_v3
/home/fair/miniconda3/envs/aerialgym/bin/python tools/create_navrl_source_bundle.py create \
  --require-clean \
  --output /home/fair/workspaces/aerial_gym_ws/navrl_v3_receipts/training_source_lower1p25_matched_spawn_b054f07_2026-09-01
export MOTAR_V3_TRAINING_SOURCE_MANIFEST=/home/fair/workspaces/aerial_gym_ws/navrl_v3_receipts/training_source_lower1p25_matched_spawn_b054f07_2026-09-01/source_manifest.json
export MOTAR_V3_TRAINING_SOURCE_MANIFEST_SHA256=<create 출력의 manifest_sha256>
```

**단계 L3 — 완료·FAIL, 재실행 금지 (GPU, 8 cells, seed 829, 70 bars).**
2026-09-01 첫 실행은 `VOID_EXECUTION`이다. 아래 출력 루트는 unique-root 계약상 재사용 금지.
matched-spawn 수정 뒤에는 **새 커밋에서 뜬 새 L1 receipt**와 새 디렉터리가 필요하다.
같은 명령을 다시 돌리려면 새 디렉터리 이름이 필요하다.

```bash
cd /home/fair/workspaces/aerial_gym_ws/.codex_worktrees/braking_route_v3
export PYTHONNOUSERSITE=1
export NAVRL_TARGET_BRAKING_CONTRACT_VARIANT=baseline_1p25
NAVRL_V3_OUTPUT_ROOT=/home/fair/workspaces/aerial_gym_ws/navrl_v3_runs/pilot_lower1p25_matched_spawn_seed829_2026-09-01 \
NAVRL_V3_CELL_RUNNER=$PWD/tools/run_navrl_braking_route_v3_cell.py \
NAVRL_V3_BRAKING_RECEIPT=$PWD/results/navrl_physical_target_braking_lower1p25_matched_spawn_seed827_2026-09-01/receipt.json \
NAVRL_V3_BRAKING_RECEIPT_SHA256=<단계 L1 sha256sum 값> \
bash tools/run_navrl_braking_route_v3_pilot.sh
```

실측은 `PASS_8_CELL_INTEGRITY / FAIL_BLOCKS_CONFIRMATORY`이므로 여기서 중단한다. confirmatory는
lower pilot 8/8 PASS일 때만 seed 839, bars 70/115/160/205였으나 열리지 않았다. 어느 단계도
0.05 warmup 게이트나 PID를 바꾸지 않습니다. confirmatory PASS가 여는 것은 별도 사전등록할
500-epoch PPO smoke뿐이며, 장기학습 authority는 만들지 않습니다.

## 0. 브랜치 규칙 (2026-09-07 제정) — 작업 브랜치는 `main` 하나다

**모든 작업·커밋·푸시는 `main`에서 한다.** 장기 작업 브랜치를 만들지 않는다. 원격은
`git@github.com:joshualikaist/MOTAR.git` 하나뿐이다.

- **GitHub 기본 브랜치 = `main`**, **Pages 서빙 = `main` 브랜치의 `/docs` 폴더**. 이 둘이 어긋나면
  푸시해도 사이트가 안 바뀐다.
- 세션 시작 시 `git status -sb` 첫 줄이 `## main...origin/main`인지 확인한다. 아니면 그 자리에서 멈추고
  왜 다른 브랜치인지 확인한다.
- 예외는 **codex 워크트리**뿐이다(`.codex_worktrees/`). 이건 병렬 에이전트가 쓰는 단기 작업 공간이고,
  끝나면 `main`으로 병합한 뒤 브랜치를 지운다. 워크트리에 물려 있는 브랜치는 삭제하지 않는다
  (`git worktree list`로 확인).
- 문서에 브랜치 이름을 적을 때는 `CLAUDE.md`와 이 파일이 **같은 이름**을 말해야 한다.
  `tests/test_branch_policy.py`가 그 일치를 강제한다.

**이 규칙이 생긴 이유**: 원격 기본 브랜치가 `research/navrl-env`인데 실제 작업은 `main`에서 이뤄졌고,
`CLAUDE.md`는 `research/navrl-env`라고 적고 있었다. 그래서 2026-09-07에 사이트 갱신과 README 도식을
푸시했는데도 배포되지 않았다. 두 브랜치는 갈라진 적이 없었고(`research/navrl-env`가 `main`의 순수 조상,
32 커밋 뒤짐) fast-forward 한 번으로 해소됐지만, 원인은 **어느 브랜치가 정본인지 문서가 틀리게 적어둔 것**이었다.

## 0.1 규칙 — GPU 수치를 담는 receipt는 실행 스택을 기록한다 (2026-09-09 제정)

`runs/`나 `results/`에 **GPU에서 나온 숫자**를 쓰는 도구는 receipt에
`tools/runtime_fingerprint.py`의 `runtime_fingerprint()`를 넣는다. Python·torch·CUDA·cuDNN·OpenCV·
numpy 버전과 TF32/benchmark/deterministic 플래그, device 이름과 compute capability가 들어간다.

**이 규칙이 생긴 이유**: 2026-09-08에 frozen candidate cache를 재현하지 못하는 사고가 있었다.
cache는 `detector_runs/venv`(torch 2.10 / cuDNN 9.10.2)에서 만들어졌는데 재실행이
`datasets/detenv`(torch 2.4 / cuDNN 9.1)로 돌았고, 두 스택이 서로 다른 컨볼루션 커널을 골라 같은
60프레임에서 confidence가 최대 45 % 어긋났다. receipt에 device 이름과 fp16만 있고 인터프리터·가속
스택이 없어서, 이 차이가 **비결정성으로 오인돼** TF32·benchmark·deterministic 플래그를 하루 종안
뒤졌다. 그 플래그로는 애초에 메울 수 없는 차이였다.

재현이 실패하면 `runtime_fingerprint.differences(옛_receipt['runtime'], 새_receipt['runtime'])`가
어느 축이 달라졌는지 바로 알려준다. `tests/test_runtime_fingerprint.py`가 이 규칙과, 스트리밍 명세가
cache를 만든 환경 하나만 지시하는지를 강제한다.

**perception streaming RGB 실행 환경은 `detector_runs/venv`다.** 다른 환경에서는 frozen output이
재현되지 않는다.

## 1. 처음 설치할 때

필수 조건은 Linux, NVIDIA GPU, Miniconda, Isaac Gym Preview 4입니다. Isaac Gym은 NVIDIA 계정으로
직접 내려받아야 하며 기본 경로는 `~/isaacgym`입니다.

```bash
mkdir -p ~/workspaces/aerial_gym_ws/src
cd ~/workspaces/aerial_gym_ws/src
git clone https://github.com/joshualikaist/MOTAR.git aerial_gym_simulator
cd aerial_gym_simulator

./bootstrap_second_machine.sh
conda deactivate
conda activate aerialgym
```

Isaac Gym 위치가 다르면 한 번만 경로를 넘깁니다.

```bash
ISAACGYM_PATH=/absolute/path/to/isaacgym ./bootstrap_second_machine.sh
```

설치 후 가장 먼저 CPU 계약 검사를 실행하세요.

```bash
cd ~/workspaces/aerial_gym_ws/src/aerial_gym_simulator
export PYTHONNOUSERSITE=1
python -m unittest discover -s tests -p 'test_navrl*.py'
```

`~/.local`의 NumPy가 conda 환경을 덮으면 Warp가 깨질 수 있습니다. bootstrap이 아래 설정을 넣지만,
수동 설치라면 직접 실행하고 conda 환경을 다시 활성화하세요.

```bash
conda env config vars set PYTHONNOUSERSITE=1 -n aerialgym
conda deactivate && conda activate aerialgym
```

## 2. 두 GPU의 역할

| 머신 | canonical 역할 | 주의점 |
|---|---|---|
| RTX 3070 8 GB | main profile 학습, 128 env, formal 평가 | 다른 NavRL 학습을 동시에 시작하지 않음 |
| GTX 1650 Ti 4 GB | `GPU4GB=1` held-out 평가와 가벼운 진단 | main 학습 seed와 같은 결과표에 합치지 않음 |

현재 corrected-v2 main 계약은 128 env입니다. 과거 문서의 256 env나 generic 64-env 학습 명령은 현행
ref5in 계보의 시작 명령이 아닙니다. 4 GB profile은 PhysX buffer를 줄이므로 결과 receipt에 profile을
반드시 남깁니다.

RTX 50 계열처럼 Isaac Gym Preview 4가 지원하지 않는 새 CUDA architecture는 구매·이전 전에 실제
호환성을 따로 확인하세요. 이 저장소의 현재 검증 머신은 RTX 3070입니다.

## 3. 지금 실행할 명령 찾기

현재 명령 authority는 `VERIFICATION.md`와 sim-to-real 72시간 계약이 정합니다. 아래 P0–P3 도식과
학습 명령은 계보 재현을 위한 역사적 운영 참고이며, 현재 GPU 실행 허가가 아닙니다.

```text
P0 repository/simulator gate
  └─ P1 fresh learning smoke
       └─ P2 held-out 70-bar decision cell
            └─ P3 full-budget seed 211
```

2026-08-13 현재 P1c의 preflight와 실제 실행은 다음과 같습니다. 이 문장은 P1c가 끝난 뒤 역사 명령이
되므로, 다시 실행하기 전 README와 WORKLOG의 최신 판정을 확인하세요.

```bash
cd ~/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games

REF5IN_PREFLIGHT_ONLY=1 ./train_navrl_v2_ref5in_smoke_c.sh
./train_navrl_v2_ref5in_smoke_c.sh
```

고정 launcher는 CLI 인자와 `CKPT` resume를 거부하고, runtime source가 commit되지 않았으면 실제
학습을 시작하지 않습니다. 실패한 smoke를 임의로 이어 돌리는 대신 새 corrective 계약을 먼저
RESEARCH_PLAN에 기록합니다.

## 4. 학습 상태 보기

고정 launcher는 터미널에 로그를 보여 주고, 같은 내용을 `train_session_logs/`에도 보존합니다.

```bash
cd ~/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games

./watch_navrl_training.sh
tail -f train_session_logs/current_ref5in_smoke_c.log
```

실행 중인 프로세스와 GPU는 다음처럼 확인합니다.

```bash
pgrep -af 'runner.py .*--task navrl_task .*--train'
nvidia-smi
```

TensorBoard는 run마다 따로 서버를 띄우지 말고 상위 `runs` 하나만 엽니다.

```bash
tensorboard \
  --logdir ~/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/runs \
  --port 6006
```

중간 run이 여러 줄로 보이는 것은 각 run이 독립 event file을 갖기 때문입니다. 하나의 새 학습을 이전
run과 시각적으로 이어 붙이는 것은 provenance를 숨기므로 하지 않습니다. 같은 run을 resume할 때만
전용 continuation launcher가 명시적으로 branch lineage를 기록합니다.

## 5. 정상 종료를 확인하는 법

끝났다는 판단은 콘솔 마지막 한 줄이 아니라 아래 세 가지를 함께 봅니다.

```bash
RUN=/absolute/path/to/runs/ppo_YYMMDD_HHMM_name

test -f "$RUN/.aerial_training_finished"
python -m json.tool "$RUN/aerial_run/run_summary.json" | tail -40
ls -lh "$RUN"/nn/last_gen_ppo_ep_*.pth | tail
```

- `.aerial_training_finished`가 있어야 정상 terminal path입니다.
- `run_summary.json`의 `exit_reason`과 `last_epoch`를 확인합니다.
- curriculum 끝 정책은 `gen_ppo.pth`가 아니라 terminal `last_gen_ppo_ep_*.pth`입니다.
- formal gate는 전용 analyzer/attestation 결과를 사용합니다. 마지막 epoch의 capture 한 값만 보지
  않습니다.

강제 중단이 필요하면 먼저 PID를 확인하고 해당 프로세스에 `SIGINT`를 한 번 보냅니다. 무관한 Python
전체를 kill하지 마세요.

```bash
pgrep -af 'runner.py .*--task navrl_task .*--train'
kill -INT <확인한_PID>
```

## 6. 평가 규칙

범용 corrected-v2 density evaluator의 기본 사용법은 다음과 같습니다.

```bash
cd ~/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games

CKPT=/absolute/path/to/last_gen_ppo_ep_XXXX_rew_YY.pth
NAVRL_SEED=313 \
NAVRL_V2_ACTION_MODE=deterministic \
NAVRL_V2_DENSITIES="70" \
./eval_navrl_v2_density_sweep.sh "$CKPT" 2049
```

단, 연구 gate에는 README/RESEARCH_PLAN에 고정된 seed·밀도·조건과 깨끗한 wrapper를 사용합니다.
shell에 남은 `NAVRL_*`, `GPU4GB`, governor, detector, pose/appearance 변수가 조건을 바꿀 수 있기
때문입니다. evaluator는 다음을 fail-closed로 확인합니다.

- checkpoint의 arena, sensor, token selector, action contract;
- robot name과 contract-v1 config/URDF SHA;
- learned detector가 필요하면 detector SHA;
- 요청 seed, action mode, density, full goal/FOV 조건;
- checkpoint와 runtime source snapshot receipt.

`2049`는 requested completed episodes의 최솟값입니다. 128개 vector environment가 동시에 진행되므로
실제 완료 수는 더 클 수 있고, 비율의 분모는 JSON의 `actual_episodes`입니다.

### 6.1 필터 격자와 결합 분석 (거버너 기하 트랙)

여러 조건을 한 번에 돌릴 때는 evaluator를 직접 부르지 말고 **격자 러너**를 씁니다. 셀마다 조건을 검증하고,
`aerial_gym`·`tools`·`resources/robots`가 커밋된 상태인지 **매 셀 직전에** 다시 확인합니다.

```bash
cd ~/workspaces/aerial_gym_ws/src/aerial_gym_simulator
python tools/run_navrl_filter_grid.py docs/specs/<spec>.json results/<root>
```

**실행 중에는 저 세 경로를 절대 건드리지 마세요.** 파일을 하나 추가하거나 문서 커밋으로 HEAD를 옮기기만 해도
남은 셀이 VOID 처리됩니다(실제로 세 번 겪었습니다). 편집이 필요하면 `docs/`, `tests/`, scratchpad에서만 하고
격자가 끝난 뒤 옮깁니다. 무거운 CPU/디스크 작업도 같이 피합니다 — 동시 부하가 같은 셀의 결과를 바꾼 적이 있습니다.

진행 상황과 결과 보기:

```bash
python tools/summarize_navrl_grid.py results/<root>              # 한 줄씩 표, 진행 중에도 됨
python tools/summarize_navrl_grid.py results/<root> --mechanism  # 접촉별 열 추가
```

여러 시드·밀도를 하나의 추정으로 묶고 사전등록 예측을 판정할 때:

```bash
python tools/pool_navrl_seed_replication.py \
    results/navrl_grid_d1p_ep25000_seed523 results/navrl_grid_l1_ep25000_seed523 \
    results/navrl_grid_r1_seedrep_ep25000_s527 results/navrl_grid_r1_seedrep_ep25000_s531 \
    --mechanism --json results/<root>/pooled.json
```

역분산 결합·Cochran Q·DerSimonian–Laird를 함께 출력하고, 같은 조건이 두 루트에 있으면 버리지 않고 일치 여부를
보고합니다. 통계는 `tools/navrl_stats.py` 한 곳에 정의돼 있고 **비율이 아니라 기록된 개수**에서 계산합니다.

## 7. checkpoint와 결과를 다른 컴퓨터로 옮기기

Git에는 `runs/`와 `.pth`가 들어가지 않습니다. 최소 이 묶음을 옮기세요.

```text
runs/<run>/nn/last_gen_ppo_ep_*.pth
runs/<run>/aerial_run/
runs/<run>/summaries/
train_session_logs/<matching log>
train_source_receipts/<matching receipt>/
results/<matching evaluation>/
```

먼저 checkpoint hash를 적습니다.

```bash
sha256sum /absolute/path/to/last_gen_ppo_ep_XXXX_rew_YY.pth
```

같은 네트워크라면 `rsync`가 가장 단순합니다.

```bash
rsync -avh --progress \
  /absolute/path/to/runs/ppo_YYMMDD_HHMM_name/ \
  user@other-host:/absolute/path/to/MOTAR/aerial_gym/rl_training/rl_games/runs/ppo_YYMMDD_HHMM_name/
```

파일 하나만 전달해야 하면 checkpoint를 복사하되, 최소한 SHA·run 이름·source manifest를 같이
전달하세요. shape가 같아도 robot 또는 observation 의미론이 다를 수 있습니다.

## 8. 디스크 정리

### 규칙 8-A — checkpoint는 손으로 지우지 않는다 (2026-09-07 제정)

**`runs/**/*.pth`를 사람이 골라서 지우는 것을 금지한다.** 반드시 이 도구가 만든 목록으로만 지운다.

```bash
python tools/audit_checkpoint_references.py                  # 보존/삭제 가능/이미 유실 보고
python tools/audit_checkpoint_references.py --verify         # 인용된 것이 하나라도 없으면 exit 1
python tools/audit_checkpoint_references.py --list-deletable > /tmp/del.txt
#  ↑ 목록을 사람이 확인한 다음에만 지운다. 도구는 절대 지우지 않는다.
```

도구가 계산하는 **보존 대상**은 셋이다. 하나라도 걸리면 남긴다.

1. `results/` 아래 결과 JSON이 `checkpoint`로 인용하는 것 — 그 수치를 다시 뽑을 수 있어야 한다.
2. 추적되는 런처·명세·문서가 이름으로 고정한 것 — 그 실험이 그 파일에 묶여 있다.
3. 각 run의 **최종** `last_gen_*`(없으면 `gen_ppo.pth`) — 최종 가중치가 없는 run은 평가 자체가 불가능하다.

**모르는 이름은 남긴다.** 도구가 주기적 저장본으로 인식하는 `last_gen_ppo_ep_<N>_rew_<X>.pth`만
삭제 후보가 된다. `_rlnorm` 변형이나 손으로 이름을 바꾼 것처럼 패턴에 없는 파일은 `unknown`으로
보고하고 보존한다. "이게 뭔지 모르겠다"가 "지워도 된다"가 되어선 안 된다.

**지운 뒤에는 반드시 `--verify`를 돌린다.** exit 1이면 인용된 체크포인트를 지운 것이므로 즉시
WORKLOG에 무엇을 잃었는지 적는다.

**이 규칙이 생긴 이유**: 산문 규칙("삭제 금지: 최신 terminal checkpoint")이 이미 §8에 있었는데도
결과가 인용하는 체크포인트 4개가 지워졌다. 그중 `last_gen_ppo_ep_21750`은 seed 911 route-off
held-out 평가 6건이 인용하는 파일이고, 그것이 없어서
`tests/test_navrl_corrected_nonoverlap_heldout_contract.py`가 지금 건너뛴다. 판단을 사람 기억에
맡기면 또 진다. 현재 유실 목록은 도구가 매번 출력한다(2026-09-07 기준 11건, 그중 결과 인용 4건).

### 그 밖의 정리

삭제 전에 `run_summary.json`, terminal checkpoint SHA, results summary가 WORKLOG에 기록됐는지
확인합니다. smoke run을 정리할 때도 canonical 결과의 근거 파일은 남깁니다.

- 보존: 논문 표에 쓰인 terminal checkpoint, `aerial_run`, `summaries`, source/eval receipt.
- archive 가능: 실패한 짧은 smoke의 TensorBoard event.
- 삭제 금지: 사용 중인 run, 최신 terminal checkpoint, 결과가 아직 문서화되지 않은 evaluation.
- **로그는 지우지 않는다.** `runs/` 8.4 GB 중 체크포인트가 8.06 GB이고 로그·설정은 0.34 GB뿐이다.
  회수는 체크포인트에서 나오므로 csv와 tfevents를 건드릴 이유가 없다.

TensorBoard 화면만 단순하게 만들고 싶으면 old `summaries/`를 저장소 밖 archive로 이동한 뒤
WORKLOG에 원래 run과 이동 위치를 남깁니다. 기록 없이 여러 run을 합치거나 event file을 편집하지
않습니다.

## 9. 자주 만나는 메시지

### duplicate NavRL training

```text
refusing duplicate NavRL training; active PID(s): ...
```

오류가 아니라 중복 학습 방지입니다. `pgrep`와 `nvidia-smi`로 기존 run을 확인하세요. 의도적으로 두
학습을 같은 GPU에서 돌리는 것은 canonical 계약이 아닙니다.

### `libtinfo.so.6: no version information available`

conda의 bash/terminfo 경고인 경우가 많습니다. 그 뒤 launcher가 계속되고 epoch가 증가하면 학습 실패
원인은 아닙니다. 반드시 뒤의 실제 exit message를 봅니다.

### `torch/fx/experimental/symbolic_shapes ... unknown range`

PyTorch compile의 경고일 수 있습니다. traceback, NaN/Inf fail-stop, process 종료가 없고 epoch가
증가하면 단독으로 실패 판정하지 않습니다.

### GPU OOM

- 3070 main: 다른 GPU 프로세스를 먼저 정리합니다. canonical 128 env를 임의로 낮추지 않습니다.
- 1650 Ti: formal evaluator에서 `GPU4GB=1`을 사용합니다.
- `NUM_ENVS`만 바꾸면 PPO minibatch와 physics profile이 동시에 달라질 수 있으므로 generic 수정은
  금지합니다.

### terminal을 닫자 학습이 끝남

사용자가 직접 장기 run을 띄울 때는 고정 launcher가 제공하는 로그 경로를 확인하고 `nohup` 또는
`tmux`를 사용합니다. 다만 동일 launcher를 두 번 실행하지 마세요.

```bash
nohup ./<audited-fixed-launcher>.sh > train_session_logs/manual_nohup.out 2>&1 &
echo $!
```

### evaluator provenance mismatch

강제 옵션으로 덮기 전에 checkpoint와 현재 code/robot/detector가 진짜 같은 계보인지 확인합니다.
formal result에는 force로 만든 셀을 넣지 않습니다.

## 10. 하지 않는 것

- `gen_ppo.pth`를 density curriculum의 끝 정책으로 평가하지 않음.
- archived 601-action 결과와 corrected exact-600 결과를 합치지 않음.
- legacy `navrl_quad` checkpoint를 `navrl_ref5in_quad` 성능으로 부르지 않음.
- training log capture를 held-out 성공률로 부르지 않음.
- 한 training seed의 좁은 episode CI를 multi-seed 재현성으로 부르지 않음.
- 결과를 본 뒤 gate를 낮추거나 실패 run을 몰래 이어 돌리지 않음.

과거 recovery/riskcap/1650Ti 실험의 세부 명령은 Git history와 WORKLOG에 남아 있습니다. 현재 계보를
시작할 때는 이 문서와 README의 고정 launcher만 사용하세요.

## 2026-08-25 routed physical gate 운영 기록

attempt 1은 conda Python의 inherited `PATH`에 matching `ninja`가 없어 0/32
`VOID_EXECUTION`이었다. attempt 2는 별도 output directory에서 32/32 JSON integrity PASS를
기록했지만 `FAIL_ROUTE_MECHANISM`과 `BLOCKED_PHYSICAL_TRAINING`이다. 따라서 이 결과를 PPO
학습 명령이나 hardware validation 명령으로 재사용하지 않는다.

당시 후속 진단 전에 확인하도록 기록한 항목:

- [`attempt 2 summary`](results/navrl_physical_target_routed_gate_seed827_attempt2/summary.md)의
  route-on/off 32-cell 표와 receipt SHA를 확인한다.
- route-on의 `unsafe_start` soft-envelope recovery deadlock을 safe-prefix/full-horizon 계측으로
  분리한다. local invalidation 0.125–0.635%와 fallback 32.6–85.0%를 같은 지표로 합치지 않는다.
- 70-bar 4-speed pool의 plan success `.1454668471`, fallback `.359296875`가 각각 `.99`, `.01`
  gate를 못 넘고, 별도 70×0.6 cell의 goal/env `.25`가 `.5` gate를 못 넘는 것을 확인한다.
  same-goal은 0이다.
- evaluator와 preregistered threshold를 수정하거나 300-bar arena-wide connectivity를 주장하지
  않는다. physical PPO는 새 preregistered lineage에서 동일 32-cell gate가 닫힌 뒤에만 검토한다.

## 2026-08-26 Track B 종료 상태

Recovery-v2 lower-1.25는 32/32 integrity를 통과했지만 7/32만 PASS(모두 route-off), recovery
0/16으로 `FAIL_ROUTE_MECHANISM`이다. 70-bar pool은 plan success `93.60%`, fallback `47.87%`,
70×0.6 goal completions/env `0.21875`, recovery `NO_CONNECTOR` occupancy `63.06%`다.

후속 no-anchor observer probe는 primary `n=1`, identity disagreement `0`으로 유효하게 검증됐지만
최소 `n=20`을 못 채워 `INCONCLUSIVE`다. 이 결과는 32-cell FAIL을 바꾸지 않는다. Track B에서
gain 2.5, `0.45 m`, PPO, 1.5 m/s, env 수, 32-cell rerun 또는 추가 GPU probe를 위한 운영 명령은
없다. 다음 실행은 Track A의 hardware/real-log 조건이 충족될 때만 72시간 계약에서 찾는다.
