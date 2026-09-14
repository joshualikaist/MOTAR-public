# D8-B — frozen-policy sensitivity to mesh-derived observation

## Verdict: **`MATERIAL_LOSS`**

The verdict uses only the preregistered mesh-shaded minus analytic capture contrast.
Mean: **-48.967 pp**, seed-level 95% t CI [-50.113, -47.821] pp; margin -3.0 pp.

| seed | analytic capture | mesh-flat capture | mesh-shaded capture |
|---:|---:|---:|---:|
| 593 | 71.01% | 64.81% | 22.40% |
| 599 | 70.96% | 63.15% | 22.16% |
| 601 | 71.94% | 64.42% | 22.45% |

All nine cells use the same frozen policy/checkpoint and built-in detector. No PPO or adaptation ran. The seed is the uncertainty unit; episodes/frames are not independent replicates for the reported CI.

This result does not measure color-shortcut reduction because D8-B contains no distractors. D9 remains a separate preregistered audit. See `summary.json` and cell receipts for full secondary metrics and provenance.
