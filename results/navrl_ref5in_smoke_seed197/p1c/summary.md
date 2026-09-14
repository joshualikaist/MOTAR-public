# ref5in P1c learning-viability smoke — PASS

> **2026-08-13 sampler erratum:** checkpoint state `k=[20,28]` was recorded correctly, but
> general-spawn sampling used the separate configured range `[6,28] m`. The table below records
> state saturation, not proof that every training goal was 20–28 m. P1c engineering PASS and P2
> lineage remain valid; the minimum-distance mastery interpretation is withdrawn.

판정일: 2026-08-13
범위: **engineering gate only** — held-out 성능이나 하드웨어 타당성 주장이 아님

## 판정

P1c는 사전등록된 gate를 모두 통과했다. P1b에서 바꾼 것은 fresh budget `750 → 900 epoch`뿐이며,
seed 197, LR `1.5e-5`, 128 env, 70 bars, `navrl_ref5in_quad`, governor off와 corrected-v2 계약은
그대로 유지했다.

| 검사 | 결과 |
|---|---:|
| 정상 종료 | epoch 900 / frame 3,686,400 / `max_epochs` |
| 마지막 100 epoch pooled | 3,338 episodes |
| capture / crash / timeout | **72.77% / 23.94% / 3.30%** |
| distance curriculum state | **[20, 28] m** |
| PPO KL max / behavior-KL max | 0.01239 / 0.01774 |
| rollback / skipped minibatch | 0 / 0 |
| raw action OOB | 전 축 0 |
| TensorBoard | 60 scalar tags, empty/non-finite 0 |
| runtime source | 312 files, clean receipt, 재해시 PASS |

Terminal checkpoint:

```text
aerial_gym/rl_training/rl_games/runs/ppo_260813_0540_navrl_v2-ref5in-smoke-c-s197/nn/last_gen_ppo_ep_900_rew_137.08087.pth
SHA-256 f1670a1d74dd92cb00d6a58898e9cc1b96eb9cbe155d1e85812a345e7aaae6bf
```

기계 판정 원문은 `summary.json`이다. 이 PASS가 허용하는 다음 단계는 **seed 313, deterministic,
original reflection, 70 bars, U[0.3,1.5] m/s, 2,049+ episodes의 held-out P2 한 셀**뿐이다.
P2 전에는 full-budget 학습이나 legacy 대비 우월성 주장을 허용하지 않는다.

## 한계

- training seed 하나의 on-policy 결과다.
- mass, inertia, actuator와 0.12 m collision height가 함께 바뀐 whole-platform 개입이다.
- inherited controller를 그대로 사용해 plant와 controller tuning이 분리되지 않았다.
- exact BOM/CAD, 전력·열·비행시간 및 실기 비행은 검증하지 않았다.
