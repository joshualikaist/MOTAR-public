# Detector coupling bin-wise-bias rerun (seed 431)

**Frozen verdict: `SUPPORTED_OUTPUT_COUPLING_CONSISTENT_MATCHED_RETRAINING_REQUIRED`.**

Quality gate: injected std 0.7085 m vs profiled 0.7053 m (+0.45%), ±10% gate = **PASS**.

| arm | episodes | capture | crash | timeout | Δ capture vs clean |
|---|---:|---:|---:|---:|---:|
| `analytic_clean` | 2,050 | 80.29% | 17.22% | 2.49% | baseline |
| `analytic_noise_1p0_binbias` | 2,049 | 76.72% | 21.03% | 2.24% | -3.57 pp [-6.09, -1.06] |
| `learned_v7` | 2,049 | 76.04% | 21.43% | 2.54% | -4.26 pp [-6.78, -1.73] |

Interpretation ceiling: an eval-only reproduction supports that v7-shaped output errors hurt this frozen analytic-trained policy. Causal confirmation that the policy is specifically coupled requires matched retraining; this experiment cannot supply it.
