# Detector threshold diagnostic (fresh seeds 191/193)

| arm | pooled capture |
|---|---:|
| analytic_t055 | 80.58% |
| learned_t055 | 75.38% |
| learned_t070 | 76.82% |

- D1_model_at_matched_055: -5.192 pp [-6.982, -3.401]
- D2_original_combined_070: -3.752 pp [-5.523, -1.981]
- D3_threshold_within_v7_070_minus_055: +1.439 pp [-0.407, +3.285]

diagnostic only: D1 isolates detector statistics at matched threshold; D3 isolates the runtime threshold within v7; no threshold selection or adoption from these cells
