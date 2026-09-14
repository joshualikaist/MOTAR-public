# Preregistration — baseline-controller lower-speed braking contract

Frozen after the two calibration stages and before this lineage's first receipt-bearing GPU run.
The unchanged gain-2.5 controller passed the original five-second tracking and safety contract at
1.20 and 1.25 m/s and failed at 1.30 m/s. Controller candidates were ineligible. Therefore this
new lineage preserves all canonical braking-probe semantics and replaces only the upper registered
speed `1.50` with `1.25 m/s`: `{0.6, 0.9, 1.2, 1.25}`.

The seed (827), 32 env, 5 s warmup, zero-command braking, raw substep trace, stop threshold,
contact/OBB/saturation/tilt gates, source receipt, and atomic finalization are identical to the
canonical probe. It is selected by the explicit environment contract
`NAVRL_TARGET_BRAKING_CONTRACT_VARIANT=baseline_1p25`; the default remains `canonical_1p5`.

This receipt authorizes only a separately named lower-contract recovery-v2 simulator gate. It
does not reinterpret the failed 1.5 result, authorize the canonical evaluator, PPO, or hardware.
