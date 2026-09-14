# Preregistration — corrected non-overlap physical route-off PPO smoke

Frozen: 2026-09-01, before observing any PPO result.

## Question and scope

Can a fresh PPO policy learn at all in the corrected `footprint_clearance` environment when the
physical target uses the already-passing **local route-off** motion contract?

This is a one-axis engineering baseline. It isolates the bar-placement correction from the failed
`global_astar_v1`, recovery-v2 and braking-v3 mechanisms. A PASS does not validate a global route,
hardware flight, sim-to-real transfer or 205-bar mastery.

## Frozen execution tuple

| item | value |
|---|---|
| launcher | `train_navrl_corrected_nonoverlap_physical_smoke.sh` |
| initialization | fresh; checkpoint/resume forbidden |
| seed / epochs / envs | 907 / 500 / 128 |
| robot and target | `navrl_ref5in_v2_quad`; 6-DoF physical target |
| target motion | route off; mixed CV/waypoint; `U[0.3,1.25] m/s`; 1-epoch ramp |
| arena / bars | 40×40×3 m; fixed 70 bars; asset pool ceiling 300 |
| placement | `footprint_clearance`; surface clearance 0.45 m; merge/overlap fallback forbidden |
| perception / policy | canonical v2 cluster-sector Transformer and squashed Gaussian |
| learning rate | `1.5e-5` |
| governor | off |
| provenance | clean runtime receipt plus enforced exact import source root |

Why 1.25 m/s: the matched-layout route-off measurements at 70 bars passed every physical gate at
0.6/0.9/1.2/1.25 m/s. The 1.5 m/s physical contract and every routed-v3 arm remain failed/blocked.
This smoke must not be renamed as the canonical 1.5 m/s or routed result.

## Frozen verdict rules

`PASS_LEARNING_VIABILITY` requires all of:

1. exact epoch 500 completion and terminal checkpoint;
2. source receipt, import origin, robot/config/URDF and environment-state contracts valid;
3. fixed density 70 throughout; zero curriculum promotion;
4. finite checkpoint and TensorBoard scalars; no NaN/Inf fail-fast;
5. zero skipped minibatches and zero PPO epoch rollbacks;
6. last-100-epoch pooled capture at least 10 percentage points above the first-100 value, and
   last-100 mean reward above first-100 mean reward.

Any execution/provenance mismatch is `VOID_EXECUTION`. Any clean completion missing another gate is
`FAIL_LEARNING_VIABILITY`. Thresholds will not be changed after seeing the result.

## Authority after the smoke

- PASS authorizes only a separately preregistered route-off 70→205 curriculum run.
- FAIL blocks that curriculum and triggers failure analysis.
- Either result leaves routed PPO, 1.5 m/s claims and hardware/sim-to-real claims blocked.
