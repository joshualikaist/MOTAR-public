# Execution authority — lower-contract v3 matched-spawn pilot

Date frozen: 2026-09-01 (Asia/Seoul)  
Implementation branch: `codex/braking-route-v3`  
Parent preregistration SHA-256:
`17e7f35e350087bf2733aca70dfca7210efe667ad1845dd053f7334ddf1645b4`

This is an execution-only addendum. It does not change the intervention, seeds, grid, metrics,
thresholds, or scientific decision rule in the matched-spawn preregistration. It records the
explicit user instruction on 2026-09-01 to continue through the fresh braking receipt and the
8-cell pilot after the CPU forensic gate passed.

## CPU forensic gate

The following contracts must pass from a committed clean tree before GPU starts:

- all focused v3, planner, motion, recovery, lower-contract, and spawn tests;
- off/v3 target spawn and waypoint RNG identity across seeds 1, 59, 367, 827, 829, 839, 65521;
- recursive terminal-stop certification under the ideal controller model on randomized
  obstacle-rich states;
- exact closed-AABB parity between the v3 certificate and the physical substep watchdog;
- frozen research evidence SHA verification.

At authorization time these checks passed: 40 v3 + 13 planner + 25 motion + 24 recovery +
4 lower-contract + 17 spawn = **123 tests**. The old VOID telemetry also showed that every
accepted v3 command carried a terminal-stop certificate, while certificate failures and later
soft-envelope exits were nearly one-for-one. The CPU model preserved the recursive certificate.
Therefore the remaining failure hypothesis is physical tracking loss relative to the certified
model, not an observed sampler, coordinate, endpoint, or watchdog-geometry defect.

## Narrow GPU authority

Only these executions are authorized:

1. one fresh `baseline_1p25` four-speed raw braking receipt at the committed matched-spawn source;
2. one seed-829, 70-bar, 8-cell matched-arm pilot using that receipt.

Use new output roots containing `matched_spawn`; never overwrite or reuse the `dd8b4a4` receipt or
the first `pilot_lower1p25_seed829_2026-09-01` VOID root. The raw receipt, source manifest, import
origin, runtime-source hashes, and off/v3 layout, robot-pose, and target-pose hashes remain
fail-closed.

## Prediction and stop rules

The preregistered prediction remains **mechanism FAIL**. Restoring matched spawn is expected to
repair execution identity, not speed ratio, goal completion, fallback, or in-flight soft exits.

- Any matched-arm digest mismatch is `VOID_EXECUTION`; do not interpret gate numbers.
- Integrity-clean pilot FAIL closes confirmatory and PPO.
- Integrity-clean 8/8 PASS opens only the already frozen seed-839 confirmatory grid.
- No result from this addendum authorizes PPO, threshold/gain/margin retuning, long training,
  hardware claims, or sim-to-real claims.

