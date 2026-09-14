# Preregistration — physical-target speed ceiling and controller calibration

Frozen before the first calibration GPU child is launched. This is an evaluation-only
diagnostic and does not replace or relax the existing four-speed braking contract. The existing
`2.5` velocity-gain lineage and every prior result remain immutable.

## Fixed experiment

- seed `829`, `32` environments, `base_sim`, `navrl_ref5in_quad`, physical target, route off,
  zero bars, 40×40×3 m arena, and `0.01 s × 10` physics per `0.1 s` control interval;
- one fresh Isaac Gym process per unique `(velocity_kp, requested_speed)` cell;
- exactly six cells:
  `(2.5,1.35)`, `(2.5,1.40)`, `(2.5,1.45)`, `(2.5,1.50)`,
  `(3.0,1.50)`, `(3.5,1.50)`;
- `10 s` commanded-speed phase followed by a zero-command braking phase of at most `10 s`;
- descriptive tracking snapshots at exactly `5`, `6`, `8`, and `10 s`.

No gain, speed, horizon, threshold, or selection rule may be changed after a GPU result is seen.
The gain values are diagnostic candidates, not new canonical defaults.

## Frozen gates and decision rule

A cell passes the five-second tracking gate only when all 32 environments are within both
`0.05 m/s` and `10%` of the requested speed at 5 s and remain within those bounds for the whole
`4–5 s` window. It must additionally have zero contact, zero invalid OBB state, warmup and
braking motor-saturation fraction at most `0.15`, maximum tilt at most `60 deg`, speed overshoot
at most `0.05 m/s`, and all environments must stop below `0.10 m/s` within the braking budget.

The baseline attainable speed is the greatest passing member of
`{1.35,1.40,1.45,1.50}` at gain `2.5`. The controller candidate is the lowest gain in
`{3.0,3.5}` that passes at `1.50 m/s`, has p95 stopping distance no more than `110%` of the
gain-2.5/1.50 diagnostic, and has p95 lateral deviation no more than that baseline plus `0.05 m`.
If neither passes, there is no selected controller. The 6/8/10 s values diagnose finite-time
convergence but never retroactively turn a five-second failure into a pass.

## Claim boundary

This calibration can authorize two separately named follow-ups only: a lower-speed contract
using the unchanged gain-2.5 controller, and a controller-lineage contract retaining 1.5 m/s.
It cannot authorize PPO, hardware claims, or editing the existing braking receipt/evaluator.
Every follow-up needs its own source-bound receipt and recovery-v2 result.
