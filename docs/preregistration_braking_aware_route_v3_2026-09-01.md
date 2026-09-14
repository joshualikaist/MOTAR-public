# Preregistration — corrected non-overlap braking-aware route v3

Date frozen: 2026-09-01 (Asia/Seoul)  
Implementation branch: `codex/braking-route-v3`  
Base: `b5850a215de689219dac72e50d6ea979b0025b66`

## Question and calculation-first prediction

Can a fresh-only route/controller that preserves a measured terminal stopping invariant remove the
runtime `unsafe_start`/fallback deadlock without weakening the corrected non-overlap geometry,
physical target dynamics, or route-mechanism gates?

The prediction is **yes at 70 bars**.  The frozen r2 receipt recorded zero `no_path` statuses at
70 bars, 1,104/1,254 planner calls were runtime replans, and fallback-excluded active speed ratios
were 0.969--0.996.  The separate geometry audit measured 99.961% random-pair connectivity at
70 bars under body+tracking inflation.  Therefore the expected failure mechanism is the mismatch
between the cached A* soft envelope, the rounded/local safe-prefix controller, and a velocity-free
waypoint path, not 70-bar topology or low-level thrust saturation.

This prediction is falsified if the v3 pilot preserves the exact soft envelope and terminal stop
certificate but still misses any frozen route-mechanism gate.

## Frozen intervention

New route mode: `global_astar_braking_v3`.  It is fresh-only and may not alias or mutate
`global_astar_v1` or `global_astar_recovery_v2`.

The intervention consists only of:

1. one exact closed-AABB soft-envelope contract shared by spawn, A*, local rollout, and watchdog;
2. full-horizon rollout acceptance (ordinary longest-safe-prefix execution is forbidden);
3. a vectorised terminal zero-command stopping certificate using the already validated canonical
   1.5 m/s speed-to-p95-distance lookup and lateral p95 tube;
4. path-tangent/cross-track following with collision-certified corner fillets when available and
   pre-brake/stop-turn-go otherwise;
5. cause-separated reset, goal-completion, runtime-replan, soft-exit, braking, and certificate
   telemetry.

No reward, observation, termination, pursuer action, detector, PPO, target mass/thrust/controller
gain, speed ceiling, acceleration ceiling, turn-rate ceiling, bar geometry, 0.45 m tracking margin,
or 0.75 m boundary reserve may change in this intervention.

## Development pilot (not a confirmatory claim)

- Seed: 829 (deliberate replay of the diagnosed r2 layouts).
- Bars: 70.
- Speeds: 0.6, 0.9, 1.2, 1.5 m/s.
- Arms: `off`, `global_astar_braking_v3`.
- Environments and rollout length: unchanged from the corrected r2 gate.

The pilot advances only if all eight cells pass integrity and the pooled routed arm satisfies:

- runtime `unsafe_start` replans = 0;
- first soft-envelope exits = 0;
- accepted-command terminal-stop certificate rate = 100%;
- all-cause plan success >= 99%;
- fallback interval fraction <= 1%;
- realised/commanded speed ratio >= 0.80 in every routed cell;
- goal completions per environment at 0.6 m/s >= 0.50;
- every unchanged per-cell physical-controller/contact/displacement gate from corrected r2 passes.

Failure of any item blocks the 32-cell run.  Pilot outcomes may diagnose code but may not change the
thresholds or confirmatory seed.

## Confirmatory mechanism gate

- Seed: **839**, frozen before implementation results.
- Bars: 70, 115, 160, 205.
- Speeds: 0.6, 0.9, 1.2, 1.5 m/s.
- Arms: `off`, `global_astar_braking_v3` (32 cells total).
- Same environment count, rollout length, physical geometry v2, footprint-clearance placement,
  target box, and source/import integrity requirements as corrected r2.

The 70-bar route-mechanism gates are identical to the pilot gates.  All 32 cells must also pass the
unchanged corrected-r2 controller and integrity gates.  A cell that cannot be certified is a FAIL,
not an invitation to reduce the reserve, raise speed/acceleration, increase the episode length, or
retry a different seed.

## Machinery integrity requirements

- The canonical braking raw receipt must be reverified at process start; declared scalars or a
  `VALIDATED` marker alone cannot arm v3.
- Missing/mismatched route schema, source manifest, import origin, physical robot/URDF/config hash,
  braking lookup, geometry contract, or checkpoint environment field fails closed.
- The per-step v3 path must remain GPU-vectorised; a CPU planner is allowed only at reset, goal
  replacement, or a certified stopped replan.
- Continuous swept-segment tests are required; endpoint-only collision checks do not count.
- Existing v1 and recovery-v2 tests and diagnostic payloads must remain unchanged.

## Decision and authority boundary

- Pilot PASS authorises only the seed-839 32-cell mechanism gate.
- Confirmatory PASS authorises a separate preregistered fresh 500-epoch corrected non-overlap PPO
  smoke; it does not itself authorise long training or claim capture performance.
- Any confirmatory FAIL leaves corrected non-overlap PPO at 0 epoch and requires mechanism analysis.
- 300 bars remains a disconnected stress condition and is not part of this route gate.

