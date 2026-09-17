# Frozen policy training contract — `ep25000 + riskcap`, reconstructed 2026-09-17

What environment the currently-frozen policy was actually trained in, reconstructed
from evidence rather than from memory. No result is changed and no training was run.

```text
checkpoint  aerial_gym/rl_training/rl_games/runs/
            ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1/
            nn/last_gen_ppo_ep_25000_rew_39.742134.pth
sha256      f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40   (re-hashed, matches)
epoch       25000        frame  102,400,000        env_state keys  127
```

## Provenance vocabulary

| Tag | Meaning |
|---|---|
| `ATTESTED_BY_CHECKPOINT` | the value is stored in this checkpoint's own `env_state` |
| `ATTESTED_BY_RECEIPT` | recorded in a run receipt or launcher that this run demonstrably used |
| `RECONSTRUCTED_FROM_SOURCE` | read from source at HEAD; the checkpoint does **not** record it |
| `ASSUMED_PRE_KEY_DEFAULT` | the key did not exist when this checkpoint was written |
| `UNKNOWN` | not recoverable from any of the above |

Nothing was promoted to `ATTESTED` by inference. Where the checkpoint is silent the
row says so.

## The headline

**This policy was trained on the `legacy` target lineage with the `mixed` pattern.**
`cfg_target_motion_model = symmetric_local_steer_v2_heading_continuity90` is the
historical `steer_target_step` law, and `train_navrl_v2_search.sh:154` resolves
`NAVRL_TARGET_DYNAMICS:-legacy`. That lineage applies **no acceleration bound and no
turn-rate bound** to the target; it uses a 90° heading-continuity *preference*, plus
wall reflection and post-step obstacle push-out.

TM-E2 requires `bounded|physical` dynamics and the `waypoint` pattern. TM-E1 forces
`cv`; TM-E0 forces speed 0. **None of E0/E1/E2 is this policy's training
distribution**, which is why the preregistration was amended to add an
in-distribution arm H before any measurement.

## Contract table

| Item | Value | Provenance | Note |
|---|---|---|---|
| arena size | `40` | `ATTESTED_BY_CHECKPOINT` |  |
| arena height | `3` | `ATTESTED_BY_CHECKPOINT` |  |
| placement mode | `navrl_band` | `ATTESTED_BY_CHECKPOINT` |  |
| placement touch / gap | `cfg_placement_touch_m=0.4 / cfg_placement_gap_m=1.6 m` | `ATTESTED_BY_CHECKPOINT` |  |
| bar pool | `bars_h3` | `ATTESTED_BY_CHECKPOINT` |  |
| bar band x ratio | `0 .. 1` | `ATTESTED_BY_CHECKPOINT` |  |
| bar density (final) | `205` | `ATTESTED_BY_CHECKPOINT` | HEAD default is `150` via `NAVRL_DENSITY_FINAL` — **unguarded** |
| density curriculum step | `15` | `ATTESTED_BY_CHECKPOINT` |  |
| density threshold sched | `70:0.82,85:0.77,100:0.72,115:0.70` | `ATTESTED_BY_CHECKPOINT` |  |
| | | | |
| target motion model | `symmetric_local_steer_v2_heading_continuity90` | `ATTESTED_BY_CHECKPOINT` |  |
| target pattern | `mixed` | `ATTESTED_BY_CHECKPOINT` |  |
| target speed range | `U[0.3, vmax], vmax -> 1.5 m/s` | `ATTESTED_BY_CHECKPOINT` |  |
| target speed ramp | `300` | `ATTESTED_BY_CHECKPOINT` | HEAD default is `3000` via `NAVRL_TARGET_SPEED_RAMP_EPOCHS` — **unguarded** |
| target max acceleration | `not in env_state` | `UNKNOWN` | no `cfg_target_max_accel_mps2` key; the legacy lineage applies no acceleration bound |
| target max turn rate | `not in env_state` | `UNKNOWN` | no `cfg_target_max_turn_rate_degps` key; legacy uses a 90 deg heading-continuity *preference*, not a rate bound |
| target lookahead | `not in env_state` | `UNKNOWN` | no `cfg_target_lookahead_s` key; legacy has no receding-horizon rollout |
| target heading-valid speed | `0.10 m/s in force` | `ASSUMED_PRE_KEY_DEFAULT` | key ABSENT from env_state. Source notes the pre-key lineage ran an inline 1e-05 m/s epsilon: a 1e4 difference |
| | | | |
| observation dimension | `actor 898 / critic 906` | `RECONSTRUCTED_FROM_SOURCE` | no `obs_dim` key in env_state; derived from the recorded LiDAR 72x4, 8 obstacle tokens and 17-token layout |
| LiDAR beams | `72 x 4` | `ATTESTED_BY_CHECKPOINT` |  |
| LiDAR max range | `12` | `ATTESTED_BY_CHECKPOINT` | HEAD default is `4.0` via `NAVRL_LIDAR_RANGE` — warn-only guard |
| obstacle tokens | `8` | `ATTESTED_BY_CHECKPOINT` |  |
| obstacle selector | `cluster_sector` | `ATTESTED_BY_CHECKPOINT` |  |
| token FOV | `240` | `ATTESTED_BY_CHECKPOINT` |  |
| critic privileged state | `actor + 8 GT` | `RECONSTRUCTED_FROM_SOURCE` | no env_state key; from the documented 898/906 asymmetric contract |
| | | | |
| detector threshold | `0.55` | `ATTESTED_BY_CHECKPOINT` |  |
| detection dropout | `0.3` | `ATTESTED_BY_CHECKPOINT` |  |
| depth noise std | `0.02` | `ATTESTED_BY_CHECKPOINT` |  |
| RGB noise std | `0.015` | `ATTESTED_BY_CHECKPOINT` |  |
| detector checkpoint | `` | `ATTESTED_BY_CHECKPOINT` |  |
| | | | |
| success radius | `0.5 m` | `RECONSTRUCTED_FROM_SOURCE` | no `cfg_success_radius` key in env_state; `navrl_task_config.py:720` at HEAD |
| episode length | `600` | `ATTESTED_BY_CHECKPOINT` | HEAD default is `300` via `NAVRL_EPISODE_LEN_STEPS` — **unguarded** |
| goal distance range | `6 .. 28 m` | `ATTESTED_BY_CHECKPOINT` |  |
| out-of-bounds margin | `1` | `ATTESTED_BY_CHECKPOINT` | HEAD default is `0.5` via `NAVRL_OOB_MARGIN` — **unguarded** |
| reward components | `range_rate, time cost, static safety, smooth, height, yaw align, yaw damping, ego progress, +capture bonus, collision penalty` | `RECONSTRUCTED_FROM_SOURCE` | no reward key in env_state; from the task docstring at HEAD, NOT attested by the checkpoint |
| | | | |
| action policy | `squashed_gaussian` | `ATTESTED_BY_CHECKPOINT` |  |
| action mu scale | `1.0,0.4,1.0,1.0` | `ATTESTED_BY_CHECKPOINT` |  |
| action std | `0.35,0.35,0.05,0.08` | `ATTESTED_BY_CHECKPOINT` |  |
| max velocity (per axis) | `2.5` | `ATTESTED_BY_CHECKPOINT` | HEAD default is `2.0` via `NAVRL_MAX_VELOCITY` — warn-only guard |
| max tilt | `45` | `ATTESTED_BY_CHECKPOINT` |  |
| yaw rate max | `3` | `ATTESTED_BY_CHECKPOINT` | HEAD default is `2.5` via `NAVRL_YAW_RATE_MAX` — warn-only guard |
| altitude hold vmax | `2.5` | `ATTESTED_BY_CHECKPOINT` |  |
| speed governor mode | `riskcap` | `ATTESTED_BY_CHECKPOINT` |  |
| governor free / fixed | `3.53553390593 / 2 m/s` | `ATTESTED_BY_CHECKPOINT` |  |
| governor TTC | `1.2` | `ATTESTED_BY_CHECKPOINT` |  |
| | | | |
| robot asset | `not in env_state` | `UNKNOWN` | `cfg_robot_config_sha256` is emitted by current code but ABSENT from this checkpoint |
| physics dt | `0.01` | `ATTESTED_BY_CHECKPOINT` |  |
| physics steps / RL step | `10` | `ATTESTED_BY_CHECKPOINT` |  |
| RL step dt | `0.1` | `ATTESTED_BY_CHECKPOINT` |  |
| | | | |
| training seed | `1` | `ATTESTED_BY_CHECKPOINT` |  |
| training envs | `128` | `ATTESTED_BY_CHECKPOINT` |  |
| training epochs | `25000` | `ATTESTED_BY_CHECKPOINT` | top-level `epoch`; `frame` = 102,400,000 |
| training file | `ppo_navrl_perception_transformer.yaml` | `ATTESTED_BY_CHECKPOINT` |  |
| training profile | `main` | `ATTESTED_BY_CHECKPOINT` |  |
| source commit | `not in env_state` | `UNKNOWN` | this checkpoint carries no source-commit key; the run receipt lineage is the only route |

## Reproducibility of this contract

The contract is **not** reproducible from HEAD defaults. Fourteen knobs differ
(`training_contract_vs_current_runtime_2026-09-17.json`); ten of them are checked by
nothing at all, and the four that are checked only emit `logger.warning` and continue.

It **is** reproducible by replaying the launcher environment block:

```text
aerial_gym/rl_training/rl_games/train_navrl_v2_search.sh
    NAVRL_ARENA_XY=40            NAVRL_EPISODE_LEN_STEPS=600
    NAVRL_LIDAR_HBEAMS=72        NAVRL_LIDAR_RANGE=12
    NAVRL_MAX_VELOCITY=2.5       NAVRL_YAW_RATE_MAX=3.0
    NAVRL_TARGET_SPEED_MIN=0.3   NAVRL_TARGET_SPEED_FINAL=1.5
    NAVRL_OOB_MARGIN=1.0         NAVRL_TARGET_DYNAMICS:-legacy
aerial_gym/rl_training/rl_games/train_navrl_v2_ep24000_speed_governor.sh
    NAVRL_SPEED_GOVERNOR=riskcap and the governor geometry
```

Any evaluation of this checkpoint must replay that block. The gate
`tools/audits/check_eval_condition_contract.py` refuses otherwise.

## What the checkpoint does not record

`success_radius`, the reward composition, the robot asset hash, the source commit,
the target acceleration / turn-rate / lookahead bounds, the observation dimension and
the heading-validity threshold are **absent** from this `env_state`. Rows above marked
`RECONSTRUCTED_FROM_SOURCE` are read from HEAD and are therefore only as good as the
assumption that those files did not change since the run — an assumption this audit
does **not** verify and does not claim.
