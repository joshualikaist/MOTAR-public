# NavRL v2 ep24000 fixed-target-speed evaluation

Frozen policy, 205 bars, seed 42, deterministic/original inference, mixed motion, 2,049 requested episodes per independent cell.

| speed (m/s) | episodes | capture (95% CI) | crash | timeout | bar contact |
|---:|---:|---:|---:|---:|---:|
| 0.3 | 2,049 | 73.26% (71.30%..75.13%) | 23.04% | 3.71% | 22.11% |
| 0.9 | 2,049 | 72.62% (70.65%..74.51%) | 25.09% | 2.29% | 24.26% |
| 1.5 | 2,049 | 67.35% (65.29%..69.35%) | 30.75% | 1.90% | 29.97% |

High-minus-low capture: **-5.91%** (95% CI -8.70%..-3.11%).
High-minus-low crash/bar-contact/timeout: **+7.71% / +7.86% / -1.81%**.
Lateral executed-edge98 changes from **2.41%** to **4.92%**; mean flight speed changes only 2.386→2.400 m/s while mean command speed is already 2.958→2.969 m/s.
Pre-registered material speed sensitivity: **YES**.
Capture monotonically non-increasing: **YES**.

This is an inference-only causal slice; it does not update the optimizer, running statistics, or checkpoint.
