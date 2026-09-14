# Preregistration — lower-contract v3 matched-spawn amendment (1.25 m/s)

Date frozen: 2026-09-01 (Asia/Seoul)  
Implementation branch: `codex/braking-route-v3`  
Base: lower-1.25 8-cell pilot `VOID_EXECUTION` at
`/home/fair/workspaces/aerial_gym_ws/navrl_v3_runs/pilot_lower1p25_seed829_2026-09-01/`

The frozen lower-1.25 preregistration
(`docs/preregistration_braking_aware_route_v3_lower1p25_2026-09-01.md`, SHA-256
`cd1347121c24ecd10273189360bed9ca76ffa80673aa89addf3ff0eaebc16252`) is not edited.  This
document replaces only the spawn/waypoint sampling contract for the next `baseline_1p25` v3
gate.  It does not reinterpret that VOID, the canonical 1.5 warmup NO-GO, or any frozen FAIL.

## Question and calculation-first prediction

Does restoring identical target spawn and waypoint sampling to the `off` arm make matched-arm
`initial_target_pose_sha256` agree, while leaving A*/rollout/watchdog on the shared soft
envelope?

**Identity prediction: yes.**  The VOID reason was v3-only support inset and v3-only closed-AABB
spawn filtering in `_sample_general_target` / `_sample_waypoints`.  Removing those branches
leaves the physical wall+boundary box that `off` already uses.

**Mechanism prediction: fail.**  The VOID cells are not a scientific verdict, but they already
measured FAIL-class routed 70-bar numbers *on the more conservative envelope-inset spawn*:
speed ratio 0.7955/0.7475/0.6551/0.6535 (gate ≥ 0.80), 0.6 m/s goals/env 7/32 = 0.21875
(gate ≥ 0.50), fallback 277/9600 ≈ 2.89% (gate ≤ 1%), soft-envelope exits 569–2473 (gate = 0).
Sampling like `off` cannot repair those in-flight metrics and can only add reset-time
`unsafe_start`.  This prediction is falsified only by a later integrity-clean GPU gate that
passes every frozen threshold.

Do not drop `initial_target_pose_sha256` matching after seeing VOID.  Do not retune 0.05 m/s
warmup, PID, margins, speeds, or thresholds.

## Frozen intervention

Keep `global_astar_braking_v3` and `NAVRL_TARGET_BRAKING_CONTRACT_VARIANT=baseline_1p25`.
Default everywhere remains `canonical_1p5`.

Change only this: v3 target spawn and waypoint draws use the same physical wall+boundary box
and the same bar-clearance rule as `off`.  The exact closed-AABB soft envelope remains the
shared contract for A*, local rollout, and watchdog.  A start or goal outside that envelope
is an `unsafe_start` / plan rejection, not a different RNG box.

Unchanged from the lower-1.25 preregistration: speeds 0.6/0.9/1.2/1.25 m/s; pilot seed 829,
70 bars, 8 cells; confirmatory seed 839, bars 70/115/160/205, 32 cells; 32 envs, 300 steps,
20 warmup; every route-mechanism and physical threshold; matched-arm layout/robot/target pose
digests; unique output root; runtime-clean tracked sources.

## Braking receipt requirement

A spawn-byte change in `navrl_task.py` supersedes the `dd8b4a4` lower receipt.  The next GPU
pilot may arm only from a **fresh `baseline_1p25` raw braking receipt generated at the commit
that contains this spawn amendment**, verified by the unchanged standalone raw-first verifier.
The 2026-08-26 receipt and the 2026-09-01 `dd8b4a4` receipt may not arm this gate.

## Decision and authority boundary

- This document authorises CPU tests of the spawn-identity contract.  It does not itself start
  GPU, confirmatory, or PPO.
- A later integrity-clean pilot PASS authorises only the seed-839 32-cell confirmatory.
- Any FAIL blocks PPO.  Thresholds, seeds, speeds, and the controller may not be adjusted in
  response.
- Confirmatory PASS still authorises only a separately preregistered 500-epoch PPO smoke inside
  the lower envelope.
- No stage creates long-training, hardware, or sim-to-real authority.
