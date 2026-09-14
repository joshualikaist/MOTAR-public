# NavRL physical routed-target recovery forensics — seed 827

## Verdict

`RECOVERY_DOMINANT` (evaluation-only; no PPO/fix/tuning authority).

The frozen 8-cell GPU PhysX diagnostic completed and the standalone verifier passed all raw cells,
the immutable receipt, execution commit, imported sources, runtime configuration, ref5in robot and
simulator provenance.

## Preregistered pooled decision

| Test | Result | Frozen gate | Pass |
|---|---:|---:|:---:|
| Unique local-origin unsafe-start replans | 200 | >= 64 | yes |
| Hard-free but route-soft-unsafe | 97.0% (Wilson 95% lower 93.61%) | lower > 50% | yes |
| Exact hard-safe connector to nearby soft-free anchor | 96.5% (lower 92.95%) | lower > 50% | yes |
| Local fallback amplification | 99.63 intervals/invalidation | > cooldown 10 | yes |

The first independent event per `local_step_infeasible` origin is used for the two Wilson gates;
3,136 repeated unsafe-start replans and 438 unattributed replans are not treated as independent
samples. Of all fallback intervals, 35,666 are attributed to local invalidation. Fallback age is
median 101, p90 219 and maximum 287 steps.

## Cell breakdown

| bars | target m/s | local invalidations | local fallback intervals | amplification | unique unsafe origins | plan OK | no path | unsafe goal |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 70 | 0.6 | 12 | 2,204 | 183.7x | 11 | 38 | 0 | 0 |
| 70 | 1.5 | 38 | 2,838 | 74.7x | 19 | 69 | 0 | 0 |
| 150 | 0.6 | 24 | 4,218 | 175.8x | 23 | 38 | 0 | 0 |
| 150 | 1.5 | 73 | 5,218 | 71.5x | 26 | 82 | 0 | 0 |
| 205 | 0.6 | 28 | 4,756 | 169.9x | 24 | 36 | 0 | 0 |
| 205 | 1.5 | 78 | 5,138 | 65.9x | 33 | 81 | 0 | 0 |
| 300 | 0.6 | 33 | 5,916 | 179.3x | 31 | 33 | 29 | 29 |
| 300 | 1.5 | 72 | 5,378 | 74.7x | 33 | 73 | 53 | 50 |

Pooled replan status is `unsafe_start=3774`, `ok=101`, `no_path=82`, `unsafe_goal=79`.
Initial-plan status is `ok=349`, `unsafe_start=17`, `no_connected_goal=6`.

## Interpretation boundary

The dominant failure is a recovery state-machine deadlock: after a local rollout failure, the
zero-command/cooldown path leaves a target in hard-safe space but outside the route planner's soft
envelope, so replanning repeatedly rejects `unsafe_start`. This is not evidence that the 0.45 m
reserve should be reduced. Rounded-local versus square-route geometry disagreed on 1,832 observed
events, while 300-bar `no_path` remains a separate topology limitation.

The authorized next step is to preregister a hard-safe connector recovery using the existing
measured-braking contract, run a short simulator gate, and only then rerun the unchanged 32-cell
routed gate. Physical PPO remains blocked until that gate passes.
