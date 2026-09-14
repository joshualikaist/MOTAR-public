# Preregistration — physical-target lower ceiling and damping calibration, stage 2

Frozen after stage 1 and before any stage-2 GPU child. Stage 1 found that the unchanged gain-2.5
controller missed the five-second absolute tracking gate even at 1.35 m/s, while velocity gains
3.0/3.5 reached the band only after 7.96/7.05 s and increased transient overshoot. Those findings
fix the reason for this second, independent grid; stage-1 files and thresholds are not edited.

## Fixed grid

The common setup remains seed 829, 32 env, physical ref5in target, obstacle-free arena centre,
10 s command plus zero-command braking, and fresh process per cell.

- lower-speed bracket, unchanged controller: 1.20, 1.25, 1.30, 1.35 m/s;
- shared 1.50 m/s reference: velocity gain 2.5, rate gain scale 1.0;
- damping candidates at 1.50 m/s, in frozen selection order:
  `(velocity_kp=2.5, rate_scale=1.5)`, `(3.0,1.5)`, `(3.0,2.0)`.

Rate scaling multiplies the existing `[0.04,0.04,0.03]` N m/(rad/s) tensor only for this
diagnostic. It is not a task default or a training authorization.

## Decisions

The lower contract ceiling is the highest unchanged-controller speed that satisfies the original
five-second endpoint and whole 4–5 s tracking band plus the original contact/OBB/saturation/tilt/
stop gates. Transient overshoot is reported but is not added retroactively to the original braking
contract.

A damping candidate is eligible only if it satisfies those same gates at 1.50 m/s, has peak
overshoot no greater than 15% of command and at least 50% below the shared reference, p95 stopping
distance no greater than 110% of the reference, and p95 lateral deviation no greater than the
reference plus 0.05 m. The first eligible candidate in the order above is selected. If no cell is
eligible, the corresponding follow-up stays blocked. Results authorize only separate, newly
named braking/recovery contracts; they do not modify v1 or authorize PPO.
