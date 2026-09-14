# E3-P attitude reliability: authentic everywhere, but the criterion fails and stands

**Verdict: `ATTITUDE_NOT_RELIABLE`.** Two of six differentiation windows fail, so E3-P stays blocked.
Receipt: `run/attitude_reliability.json`. The criterion was
[preregistered and committed](PREREGISTRATION.md) at 362fe95 before it ran.

## The result

Held-out correlation between the attitude-predicted and the measured horizontal acceleration, with the
99th percentile of 200 permutations of the attitude-to-frame assignment as the null.

| half-window | samples | direction | correlation | null p99 | thrust gain | drag | pass |
|---:|---:|---|---:|---:|---:|---:|---|
| 2 | 844 | early→late | 0.242 | 0.041 | 0.705 | −0.011 | **fail** |
| 2 | 844 | late→early | 0.163 | 0.043 | 0.697 | −0.022 | **fail** |
| 3 | 792 | early→late | 0.396 | 0.026 | 0.751 | −0.013 | pass |
| 3 | 792 | late→early | 0.297 | 0.045 | 0.701 | −0.022 | **fail** |
| 4 | 753 | early→late | 0.545 | −0.007 | 0.739 | −0.010 | pass |
| 4 | 753 | late→early | 0.434 | 0.013 | 0.716 | −0.021 | pass |
| 5 | 740 | early→late | 0.671 | −0.003 | 0.743 | −0.008 | pass |
| 5 | 740 | late→early | 0.565 | 0.010 | 0.730 | −0.021 | pass |
| 7 | 681 | early→late | 0.850 | 0.043 | 0.753 | −0.010 | pass |
| 7 | 681 | late→early | 0.773 | 0.107 | 0.748 | −0.018 | pass |
| 9 | 663 | early→late | 0.893 | 0.055 | 0.743 | −0.009 | pass |
| 9 | 663 | late→early | 0.844 | 0.124 | 0.734 | −0.017 | pass |

## What passed, and it is the interesting part

**Authenticity passed at every window, in both directions, by a wide margin.** Even the worst cell,
h = 2 late→early, scores 0.163 against a null 99th percentile of 0.043. Permuting the attitude rows
against the frames destroys the signal completely; the true assignment does not. The published attitude
contains frame-associated signal under this diagnostic. This does not exclude all misalignment,
misidentification or measurement error, and does not certify attitude accuracy.

**The thrust gain is remarkably stable**: 0.697 to 0.753 across all twelve cells, and the drag
coefficient is negative in all twelve. A gain near 0.73 rather than 1.0 means the tilt implies more
horizontal acceleration than is measured, consistently.

## What failed, and why the verdict stands anyway

Only the correlation floor of 0.35 fails, and only at the two shortest windows. Correlation rises
monotonically with window length, from 0.24 at h = 2 to 0.89 at h = 9. That is the signature of noise in
the **target** variable being averaged down: differentiating position twice over 0.5 s is noisy, and the
noise may affect the acceleration estimate. This pattern alone does not localise all error to
acceleration or exclude attitude error.

So the natural reading is that the short windows measure the differentiator, not the attitude. **The
verdict still stands.** The window sweep was preregistered as all-or-nothing precisely so that a window
could not be chosen after the fact, and it was committed before the numbers existed. Reading the pattern
and then declaring a pass is the failure mode the sweep was designed to prevent.

**No fourth variant will be written, as committed.**

## The cost of this, stated plainly

Having now seen these numbers, no honest preregistration of a different correlation floor for this test
on this dataset is possible. The design choice is spent. What remains available is **independent
evidence**: a measurement that does not reuse the acceleration test at all, for instance whether the
airframe's projected orientation in the image tracks the published attitude at close range.
The earlier six-arm identification cue is WITHDRAWN: all three aircraft are quadrotors.
This describes a possible independent evidence source, not approval for another test on these data.

## What this would have certified, and what it never could

A pass would have said the attitude is authentic. It would **not** have bounded the attitude's accuracy in
degrees. Any future E3-P has to carry an attitude-accuracy limitation whatever this test returns, and may
not quote a per-frame attitude uncertainty from it.

## Reproducing

```bash
PY="env PYTHONNOUSERSITE=1 /home/fair/miniconda3/envs/aerialgym/bin/python"
$PY tools/check_eth_ds5_attitude_reliability.py \
  --dataset /home/fair/workspaces/aerial_gym_ws/datasets/eth_ds5 \
  --output results/eth_ds5_e3p_reliability_2026-09-10/run
```
