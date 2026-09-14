# Preregistration — recovery-v2 lower-1.25 physical-target 32-cell gate

Frozen before the first lower-contract recovery GPU child. It is a separate result lineage from
the failed canonical 1.5 m/s braking contract and the unrun canonical recovery-v2 evaluator.

The exact grid is route `{off,global_astar_recovery_v2}` × bars `{70,150,205,300}` × speed
`{0.6,0.9,1.2,1.25}` m/s: 32 cells, seed 827, 32 env, 300 RL intervals, 20 tracking warmup
intervals, and the unchanged gain-2.5 physical controller. All telemetry, row gates, route
mechanism gates, source/receipt binding, process partition, atomicity, and claim boundaries are
identical to `preregistration_physical_target_recovery_v2_gate_2026-08-25.md`.

The only manipulated contract is the preregistered upper target speed, selected prospectively by
the two-stage calibration. Results may establish a lower-speed simulator mechanism envelope only.
They cannot establish 1.5 m/s validity, PPO performance, training authorization, or hardware
performance. The explicit variant token is `baseline_1p25`; default execution remains canonical.
