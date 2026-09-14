# Detection-range Stage 1 재실행 계획 — 2026-08-23

상태: **실행 전 계약 감사 완료 후 재실행 대기**

사전등록: `docs/prereg_2026-08-22_detection_range_2stage.md`

실행 소유자: `tools/run_navrl_ref5in_detection_range_stage1_campaign.py`

## 1. 왜 다시 실행하는가

2026-08-22/23의 두 arm은 ep1900→2900을 완주했지만 `cfg_general_goal_dist_min=6.0`으로
학습됐다. 사전등록은 22.5–28 m를 고정하므로 둘 다 **VOID**다. `k_min/k_max`는
`NAVRL_GENERAL_TRAIN=1` 경로의 spawn band를 소유하지 않으며 이번 실험에서 조작하지 않는다.

VOID run은 삭제하거나 평가하지 않는다.

- clip20: `runs/ppo_260822_2322_navrl_detrange-stage1-clip20-s457`
- clip28: `runs/ppo_260823_0426_navrl_detrange-stage1-clip28-s457`

## 2. 고정 계약과 유일한 조작

| 항목 | 고정값 |
|---|---:|
| warm start | ref5in D1 ep1900, SHA `197ea269…a278e` |
| 학습 / 평가 seed | 457 / 461 |
| 학습 예산 | arm당 1,000 epoch = 4,096,000 samples, absolute ep2900 |
| 평가 예산 | arm당 최소 2,049 episode |
| 목표 거리 | 22.5–28.0 m (`GENERAL_GOAL_DIST_*`) |
| 장애물 | 70 bars, density curriculum off |
| detector / RGB | 1920×1200, min_pixels 50 / 160×90 |
| appearance·latency·range error | 0 |
| governor / reflection·lateral coefficient | off / unset(=0) |

유일한 arm 차이는 `NAVRL_DETECTOR_MAX_RANGE`: clip20=20.0 m, clip28=28.0 m다. 정규
trainer가 만든 **effective environment 두 개의 대칭차**를 측정하며, run tag·로그 경로 외에는 이
키 하나만 달라야 한다.

## 3. 실행 순서와 중단 규칙

1. CPU 계약 검사: pinned checkpoint SHA, frozen robot lineage, script-chain clobber, source origin,
   effective train/eval env diff.
2. 6-epoch GPU smoke: 8 GiB 적재와 epoch 시간을 확인한다.
3. clip20 학습 → 즉시 Gate 0.
4. clip28 학습 → 즉시 Gate 0.
5. clip20, clip28 held-out 평가.
6. finalize → 독립 재검산 verify.

단일 campaign PID가 위 순서를 소유한다. 단계 하나가 예외·non-zero exit·Gate 실패를 내면 다음
단계는 실행하지 않고 durable status를 `failed`로 원자 기록한다. 글로벌 `pgrep`, latest-run glob,
`/tmp` 체인으로 다른 run을 입양하지 않는다. lock이 중복 campaign을 거부한다.

Gate 0은 arm별로 다음을 모두 요구한다.

- 정확히 ep2900 / frame 11,878,400
- 정상 `max_epochs` 종료와 finished marker
- PPO epoch rollback 0 (체크포인트 카운터 + 로그 양쪽)
- 종단 checkpoint SHA-256
- 동일한 clean training-source receipt
- 체크포인트에 20/28 m, detect 1920×1200, min_pixels=50, goal 22.5–28, bars=70 기록

## 4. 판정 — 결과를 본 뒤 움직이지 않는다

primary metric은 모든 capture/crash/timeout cohort를 합친 pooled never-acquired rate다.

`delta = never_acquired(clip28) − never_acquired(clip20)` (percentage points)

- `delta <= -15.00 pp`: `RANGE_HELPS`
- 그 외: `RANGE_INCONCLUSIVE_AT_THIS_BUDGET`
- 어느 arm이든 Gate 0 실패: `STAGE1_VOID`, 평가·효과 판정 금지

capture/crash/timeout은 원값과 Wilson interval을 보고하지만 verdict에는 들어가지 않는다.
`RANGE_HELPS`도 P2/D1/P3를 바꾸지 않으며 Stage 2 작성 자격만 준다. inconclusive이면 Stage 2를
실행하지 않는다.

## 5. GPU 시간 예산

2026-08-22 RTX 3070 smoke 실측은 peak 6,667/8,192 MiB, headroom 1,525 MiB,
7.257 s/epoch였다.

| 단계 | 예상 GPU 시간 |
|---|---:|
| preflight smoke | 1–2분 |
| clip20 1,000 epoch | 2.02시간 |
| clip28 1,000 epoch | 2.02시간 |
| 평가 2×2,049 episode | 약 20–35분 |
| finalize/verify | CPU 1분 미만 |
| 합계 | **약 4시간 25분, 여유 포함 4시간 45분** |

VRAM이 8,192 MiB를 넘거나 headroom 측정이 없으면 학습을 시작하지 않는다. epoch 시간이 smoke
대비 지속적으로 2배 이상이면 campaign을 임의 종료하지 말고 상태와 GPU 프로세스를 먼저
점검한다.

## 6. 관찰·복구

- live log: `results/navrl_ref5in_detection_range_stage1_s457_campaign.log`
- durable status: `results/navrl_ref5in_detection_range_stage1_s457_campaign_status.json`
- 최종: `results/navrl_ref5in_detection_range_stage1_s457/{summary.json,summary.md}`

프로세스가 비정상 종료되면 status·partial log·run folder를 보존한다. 같은 경로에 덮어쓰거나
중간 체크포인트에서 임의 resume하지 않는다. 어느 phase까지 증명됐는지 분류한 뒤 새 run/새
사전등록이 필요한지 결정한다.
