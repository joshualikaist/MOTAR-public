# Preregistration revision 2 — corrected non-overlap route/physical gate

Frozen on 2026-08-31 (Asia/Seoul), after revision 1 became `VOID_EXECUTION` and before any routed
revision-2 cell was run or read.

## Why a revision is allowed

Revision 1 did not produce a route-mechanism or physical-performance verdict. Its fourth route-off
child exited while resetting one failed environment: the physical target sampler drew one uniform
candidate per round and found no accepted point in 1,024 attempts. The execution manifest contains
12/32 cells and therefore correctly marks every scientific verdict `NOT_INTERPRETED`.

The only runtime change before this retry is proposal batching in the existing rejection samplers:

- target: 64 iid uniform proposals/round × 256 rounds = 16,384 maximum proposals;
- pursuer: 64 iid uniform proposals/round × 128 rounds = 8,192 maximum proposals;
- the first accepted proposal is retained, so the accepted conditional distribution and every
  wall, range, footprint, target-support, and tracking-clearance predicate are unchanged;
- exhaustion now fails closed for both actors; the old pursuer path could silently use an unchecked
  initial point.

No clearance, target speed, controller gain, route setting, density, seed, step count, outcome
threshold, or PPO parameter changes. Revision-1 artifacts remain immutable and are not pooled.

## Frozen contract and gates

The complete revision-1 contract is incorporated unchanged:

- seed 829, 32 envs, 300 policy intervals/cell, 20-interval tracking warm-up;
- route off vs `global_astar_v1`;
- speeds `0.6/0.9/1.2/1.5 m/s`;
- training densities `70/115/160/205`, with asset ceiling 300;
- `navrl_ref5in_v2_quad`, target proxy `0.283×0.283×0.12 m`;
- `footprint_clearance`, surface clearance 0.45 m, no overlap fallback;
- identical initial bar/robot/target digests across matched arms;
- exact 32-cell/source/import/runtime/receipt integrity.

Per-cell physical gates remain RMSE ≤0.35 m/s, speed ratio ≥0.80, contact ≤0.01, applicable
local-infeasibility/invalidation ≤0.01, motor saturation ≤0.15, tilt ≤60°, invalid-state =0.

The route-mechanism gate remains: pooled 70-bar plan success ≥0.99, fallback ≤0.01, no same-goal
reselection anywhere, and 70-bar/0.6 m/s goal completions ≥0.5 per environment. Every routed
1.5 m/s cell must pass for `PASS_FULL_1P5_CONTRACT`.

## Decision

Only exact integrity + route mechanism + full 1.5 contract authorizes a separately frozen
70-bar/500-epoch fresh PPO smoke. Any other valid verdict stops before PPO. A new sampler exhaustion
is an interpretable environment-feasibility failure only if all 32 children complete and the
evaluator's predefined metrics encode it; another incomplete grid remains VOID. Long training,
warm-start, threshold relaxation, and 300-bar training remain unauthorized.
