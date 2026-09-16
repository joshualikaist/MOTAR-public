# Target behavior ladder and current-motion audit — 2026-09-16

This document separates the target/evader difficulty generator from the pursuer's information
contract. It does not reinterpret any historical result and does not report a new policy
comparison.

> **Information boundary:** Evader may use privileged GT obstacle information. Pursuer may not.

For MOTAR's sensor-only perception claims, the pursuer is limited to the camera/perception path,
allowed obstacle sensing and its declared ego/history inputs. Target-side obstacle geometry is a
controlled benchmark generator; it is not appended to the actor observation. The historical
non-vision LiDAR baseline injects a simulator goal frame containing target GT and must not be
reported as this sensor-only contract.

## What the current code actually does

The repository has three target-motion dynamics lineages and a separate browser illustration:

| Lineage | Motion law | Obstacles / bounds | Dynamics limits | Pursuer-aware? | Evidence boundary |
|---|---|---|---|---|---|
| `legacy` | Per-episode `cv` or random waypoint (50:50 default); optional held-out circle | Local symmetric steering, wall reflection and final bar push-out | Episode speed is applied immediately; no acceleration or yaw-rate bound | No, except an explicit evaluation-only initial CV-heading intervention | Historical checkpoint reproduction; can reflect/push instantaneously |
| `bounded` | Same nominal CV/waypoint/circle references | Uses simulator GT bar positions in a receding-horizon candidate screen; no position clamp, reflection or push-out | Default 4 m/s² acceleration, 150 deg/s travel-heading rate, 1 s lookahead, speed limit and arena bounds | No | 2-D kinematic implementation, not 6-DoF flight evidence |
| `physical` | Bounded planner supplies a velocity reference to a ref5in target actor | Bar AABBs, oriented target support, wall reserve; optional global A* route and braking certificates | 100 Hz PhysX, motor lag, tilt/thrust limits plus the planner envelope | No | Implemented and evaluated historically, but the strict physical route gate is `FAIL_ROUTE_MECHANISM` |
| Browser `routed-preview` | Deterministic global route with bounded/lagged display following | Exact displayed bar AABBs and workspace bounds | Display acceleration/turn-rate limits | No | Explanatory WebGL only; explicitly not PhysX/PPO |

Therefore the current target is not simply an obstacle-blind moving point. Obstacle-aware bounded
and routed implementations already exist. The important defects are lineage-specific:

1. `legacy` may change velocity discontinuously at walls and may reposition after obstacle
   penetration. It is a historical kinematic contract, not physical target evidence.
2. A random legacy/bounded waypoint is sampled inside wall margins but is not itself guaranteed
   reachable; local steering is not a global route certificate.
3. `bounded` has no rigid-body attitude, motors or contact.
4. `physical` uses the required dynamics, but its preserved route/recovery evaluations failed
   their preregistered mechanism gates. Implementation does not equal a passing benchmark.
5. None of the implemented target laws reacts to the pursuer after reset. The optional initial
   CV heading can be chosen relative to the pursuer only in a declared evaluation intervention.
6. Browser motion is visually obstacle-aware but is not a recorded simulator rollout.

Primary sources: `aerial_gym/task/navrl_task/navrl_task.py`,
`aerial_gym/task/navrl_task/target_motion.py`,
`aerial_gym/task/navrl_task/target_route_planner.py`, `docs/status/arena_motion.js`, and the
[earlier independent audit](target_motion_training_environment_audit_2026-08-25.md).

## Independent behavior axis

The public ladder is named `TM-E0`…`TM-E4` to avoid collision with the existing E3 range studies
and the older evader-plan phase names.

| Level | Target behavior | Implementation status | Result status |
|---|---|---|---|
| **TM-E0** | Static target | Existing speed-zero baseline; explicit `e0_static` profile | No new comparison |
| **TM-E1** | Constant-velocity target | Existing CV law; explicit `e1_cv` profile | Historical results remain lineage-specific |
| **TM-E2** | Obstacle-aware scripted target | Existing bounded waypoint executor; explicit `e2_obstacle_aware` profile | **Implemented; policy comparison NOT TESTED** |
| **TM-E3** | Reactive obstacle-aware evader using pursuer relative state | `PLANNED`; selecting it fails closed | No result |
| **TM-E4** | Learned evader / self-play, with historical-opponent evaluation | `PLANNED`; selecting it fails closed | No result |

`NAVRL_TARGET_BEHAVIOR_LEVEL` is an opt-in independent variable. Its default is `historical`, so
existing commands and checkpoints keep their prior target sampling. The explicit profiles select:

```text
e0_static             -> speed 0, CV placeholder
e1_cv                 -> persistent CV nominal reference
e2_obstacle_aware     -> waypoint nominal reference + bounded GT-obstacle executor
e3_reactive           -> refuse: PLANNED
e4_learned            -> refuse: PLANNED
```

TM-E2 requires `NAVRL_TARGET_DYNAMICS=bounded` or `physical`. It consumes privileged target-side
bar geometry but never pursuer ground truth. The existing bounded executor evaluates symmetric
heading candidates at full, half and quarter speed plus a stop candidate, rolls each candidate
through the configured acceleration/turn envelope, rejects workspace/obstacle violations, and
executes only its first receding-horizon step. This is the implemented E2 algorithm; the earlier
illustrative weighted-sum expression is not presented as a published equation.

## Constraints and tests

The target command is subject to:

- per-episode maximum speed;
- Euclidean planar acceleration bound;
- travel-heading/yaw-rate bound;
- wall-support bounds;
- bar-clearance screening, including AABB/support geometry in the physical lineage.

Tests cover static/CV/E2 profile selection, no pursuer-GT authority for E2, obstacle avoidance,
speed/acceleration/turn bounds, no teleporting in bounded mode, wall support, and fail-closed
selection of TM-E3/TM-E4.

## Comparison status

No E0/E1/E2 policy-performance grid is run here. Existing execution authority forbids silently
opening a new PPO/evaluation lineage, and the user request also forbids inventing new results.
A future matched comparison requires a preregistered common checkpoint/task/sensor contract and
must report target collision, infeasible-step, speed, acceleration, turn-rate and discontinuity
checks before capture/crash/timeout are interpreted. Until then, the public site labels TM-E2
`IMPLEMENTED; POLICY COMPARISON NOT TESTED`.
