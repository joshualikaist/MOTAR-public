# RC-R1 — geometry representation across four specimens

Verdict: **GEOMETRY_DEFECT**

Failed gates: `G3_box_closed_form_box_proxy`, `G4_inverse_distance_area_matched_box`, `G4_inverse_distance_box_proxy`, `G4_inverse_distance_quadrotor_mesh`, `G7_resolution_consistency_area_matched_box`, `G7_resolution_consistency_box_proxy`, `G7_resolution_consistency_quadrotor_mesh`

| arm | views | silhouette px (min / median / max) | equivalent diameter px (median) |
|---|---|---|---|
| `analytic_sphere` | 24 | 1361 / 1361 / 1361 | 41.63 |
| `box_proxy` | 24 | 697 / 1610 / 1678 | 45.27 |
| `area_matched_box` | 24 | 525 / 1137 / 1182 | 38.05 |
| `quadrotor_mesh` | 24 | 405 / 949 / 1197 | 34.76 |

The area-matched box scale was fitted on the four fit views only: `0.845078`.

Scope: this is a measurement of how the renderer represents these four shapes. It is
not a detector, tracking or policy result, and it makes no causal claim about the
closed D8b frozen-policy result (`causality_vs_d8b: NOT_TESTED`).
