# Preregistration — routed physical-target simulator gate

Frozen: 2026-08-25, before any routed PhysX cell was executed.

## Question and claim boundary

This is an engineering evaluation of the new target-motion environment, not PPO training and not
hardware validation. It asks whether a deterministic global route removes unreachable random
waypoints without introducing contacts, invalid rigid-body states, tracking failure, or persistent
zero-command stalls.

The only manipulated factor in the matched comparison is route mode:

- `route_off`: physical ref5in target, waypoint pattern, existing bounded local planner;
- `global_astar_v1`: the same target and waypoint task with exact inflated-AABB global routes.

The pursuer policy is not loaded and receives no route information. Simulator ground-truth bar
geometry is used only by the environment controller that moves the target. Results cannot support
a real-hardware claim, a pursuer-policy improvement claim, or arena-wide connectivity at 300 bars.

## Frozen grid

- seed: `827`;
- route arms: `off`, `global_astar_v1`;
- densities: `70, 150, 205, 300` bars;
- exact target speeds: `0.6, 0.9, 1.2, 1.5 m/s`;
- `32` environments per cell;
- `300` RL intervals per cell. The first `20` are excluded **only** from realized-speed and
  tracking-RMSE metrics. Contact, arm-specific local failure, invalid state, position envelope,
  controller counters, and failed-environment resets cover all `300` intervals;
- physics `dt=0.01 s`, `10` physics steps per RL interval (`10 Hz` command update);
- ref5in target, `40 x 40 x 3 m`, `bars_h3`, `navrl_band`, waypoint pattern;
- the pursuer policy action is the neutral all-zero `[N,4]` tensor, but it is never sent directly
  to the simulator. Every interval calls the canonical `NavRLTask.transform_action_to_command`
  after target advance and sends its finite `[N,4]` command to physics. This preserves the task's
  altitude/yaw controller semantics while removing learned pursuer behavior;
- route geometry: actual per-asset axis-aligned bar AABBs, all-orientation target support,
  `0.45 m` tracking reserve, `0.25 m` grid, exact continuous AABB line-of-sight checks;
- each route arm and speed is created in a fresh process. Density cells share only that process and
  report counter deltas per cell.

The fixed grid is not expanded, narrowed, or retuned after seeing results. A failed cell is retained.

## Existing physical-controller gates

Each cell is evaluated against the previously used limits, unchanged:

| metric | gate |
|---|---:|
| tracking RMSE | `<= 0.35 m/s` |
| realized/requested mean speed ratio | `>= 0.80` |
| target contact-step fraction | `<= 0.01` |
| route-off bounded-step infeasible fraction | `<= 0.01` |
| routed local-step invalidation fraction | `<= 0.01` |
| motor saturation fraction | `<= 0.15` |
| maximum tilt | `<= 60 deg` |
| finite but invalid OBB state fraction | `0` |

The two local-feasibility rows share the previously used `0.01` engineering threshold but are
**arm-specific observables, not a matched metric**. Route-off samples the bounded local step's
per-environment feasible boolean every interval. Route-on counts `local_step_infeasible`
invalidations emitted by the routed manager. They differ in event semantics and therefore are gated
inside their own arm, reported under distinct names, and never subtracted from each other. Neither
is proof that no global route exists. The ambiguous historical `planner_infeasible_fraction` alias
is not emitted by this evaluator.

## Routed-mechanism quality gates

These gates apply to `global_astar_v1` and are checked before interpreting speed performance:

1. source/config receipt and all 32 cell records must be present; a missing or crashed arm is
   `VOID_EXECUTION`;
2. route plan successes divided by attempts must be at least `0.99` at 70 bars;
3. fail-closed fallback intervals divided by all commanded intervals must be at most `0.01` at
   70 bars;
4. at `70 bars x 0.6 m/s`, completed global goals per environment must be at least `0.5` during
   the 30 s run; this rejects a planner that succeeds once and then silently parks;
5. routed planning wall time, including device-to-host geometry transfer, is reported. It is an
   engineering throughput measurement, not a policy metric; no post-hoc latency threshold is added
   to this simulator gate because the separate CPU preregistration already froze that threshold.

High-density route failures and fallback are reported, not hidden by selecting only large connected
components. The 70-bar mechanism gate is the minimum viable curriculum-start contract.

## Verdicts

- `PASS_ROUTE_MECHANISM`: all routed-mechanism gates pass.
- `PASS_FULL_1P5_CONTRACT`: route mechanism passes and every routed `1.5 m/s` density cell passes
  all existing physical-controller gates.
- `PASS_DENSITY_CONDITIONED_ENVELOPE`: route mechanism passes and every density has at least one
  preregistered passing speed; the highest passing speed per density is reported without interpolation.
- `BLOCKED_PHYSICAL_TRAINING`: route mechanism fails, any density has no passing registered speed,
  or execution integrity is void.

Even `PASS_FULL_1P5_CONTRACT` authorizes only a separately preregistered short fresh PPO smoke. It
does not authorize checkpoint reuse or a long training run.

## Matched-arm interpretation

For every density-speed cell, route-on minus route-off deltas are reported only for speed ratio,
tracking RMSE, contact, and invalid state. The arm-specific local-feasibility metrics above are
explicitly excluded. With one seed these are descriptive mechanism checks, not confidence-bounded
causal effects. No claim requires a favorable delta; safety and route gates determine the verdict.

## Runtime and matched-state receipt

Every low-level evaluation interval mirrors `NavRLTask.step`: `_advance_target` observes the current
`num_task_steps`, the neutral policy action passes through the canonical task action mapping, physics
and any failed-environment reset complete, and the clock increments exactly once. This is required
for the frozen 10-step route-replan cooldown to become eligible. Raw policy actions are never used as
simulator commands.

The launcher pins `AERIAL_GYM_SIM_NAME=base_sim`. Every child records and hashes its actual Python,
torch/CUDA, Isaac Gym, GPU/driver, imported `navrl_task` and `target_route_planner`, instantiated sim
config, robot config and URDF. A missing or mismatched origin makes execution void.

At each cell's initial reset, separate digests bind bar AABBs, robot pose, target pose, task waypoint,
and routed final goal. Bar layout and robot/target pose must match across all route/speed arms at a
density. Waypoints must match across speeds within each route arm. Route-off and route-on goals are
intentionally not required to match: choosing a reachable connected-component goal is the mechanism
being evaluated, not an accidental uncontrolled covariate.

## Execution note (does not change any numerical gate)

Attempt 1 ended before task creation with `VOID_EXECUTION` and zero cell records because the inherited
child `PATH` did not expose the selected conda Python's `ninja`. Its artifact is retained and never
interpreted. Attempt 2 uses a new output directory; the evaluator prefixes the selected Python's
`bin` to child `PATH` and binds the executable `ninja` path, hash, and version in provenance. No grid,
seed, physics, metric, or threshold changed.
