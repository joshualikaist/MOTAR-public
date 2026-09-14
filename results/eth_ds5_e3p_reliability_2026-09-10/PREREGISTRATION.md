# E3-P attitude reliability criterion, fixed 2026-09-10 before it ran

This replaces the two earlier forms of the reliability check, both of which failed and both of which
asked the wrong question. It is written and committed before the criterion is evaluated.

## Why the earlier forms were wrong

`|a_h| = g·tan(tilt)` (C2) discards the direction information and assumes drag is negligible; it gave
correlation 0.18. Adding a drag term (C2b) fixed the physics but the pass condition demanded that the
drag term improve the residual by at least 20 %, which tests whether drag is large, not whether the
attitude is trustworthy. Both are kept on the record.

## What this criterion asks

**Does the published attitude carry real, frame-specific information about how the aircraft moved?**

A multirotor accelerates horizontally by tilting its thrust axis, so a genuine attitude stream predicts
the horizontal acceleration measured independently from position ground truth. A stream that is
mislabelled, misaligned in time, or reconstructed from something other than this aircraft does not.

**What a pass does and does not certify.** A pass says the attitude is authentic and tied to this
aircraft's motion. It does **not** bound the attitude's accuracy in degrees. E3-P must therefore carry an
attitude-accuracy limitation whatever this returns, and may not quote a per-frame attitude uncertainty.

## Model

`a_h = A · g·tan(tilt) · û_thrust,h + B · |v| · v_h`, fitted as vectors, where `û_thrust,h` is the
horizontal unit vector of the thrust axis under the convention C1 identified (intrinsic `xyz`, body to
NED, thrust along body −z, mapped to ENU). `A` is the thrust gain, `B` the drag coefficient.

## The window is swept, not chosen

Horizontal acceleration comes from the second derivative of a local quadratic fit to the ground-truth
position over a half-window of `h` pose samples. `h` changes both the noise and the attenuation of real
dynamics, so **no single value is privileged**. The criterion is evaluated at every
`h ∈ {2, 3, 4, 5, 7, 9}` and must hold at **all** of them. A criterion that must survive every window
cannot be passed by choosing one.

## Held-out evaluation

The flight is split at the median pose time. Coefficients are fitted on one half and scored on the other,
in both directions. Nothing is reported from the half a coefficient was fitted on.

## Null model

The attitude rows are randomly permuted against the frames 200 times, destroying the frame-specific
correspondence while preserving every marginal distribution. Each permutation is refitted and scored the
same way. This is the null that a mislabelled or misaligned attitude stream would produce.

## Pass conditions, all required, at every window and in both split directions

1. **Beats the null**: held-out correlation exceeds the 99th percentile of the 200 permuted correlations.
2. **Physically signed**: thrust gain `A` within `[0.5, 1.5]`, drag coefficient `B` negative.
3. **Non-trivial**: held-out correlation at least `0.35`, so the attitude explains at least about an
   eighth of the out-of-sample variance in horizontal acceleration.

Any window or direction failing any condition fails the criterion. Verdict `ATTITUDE_RELIABLE` or
`ATTITUDE_NOT_RELIABLE`, naming what failed.

## Committed in advance

No fourth variant will be written. If this fails, E3-P stays blocked and the reason is that the dataset's
attitude could not be shown to be trustworthy by an independent measurement, which is a finding about the
dataset and is reported as one.
