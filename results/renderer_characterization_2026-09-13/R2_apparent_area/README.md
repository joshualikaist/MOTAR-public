# RC-R2 — apparent area across four specimens

Verdict: **AREA_MATCH_FAILED**

Primary estimand: mean silhouette area in pixels over the 20 validation views.

| arm | mean area px | ratio to quadrotor mesh |
|---|---|---|
| `analytic_sphere` | 1361.0 | 1.5318 |
| `area_matched_box` | 1057.2 | 1.1899 |
| `box_proxy` | 1486.8 | 1.6734 |
| `quadrotor_mesh` | 888.5 | 1.0000 |

The isotropic scale `0.845078` was fitted on the four fit views and applied unchanged
to the twenty validation views. On those views the matched box covers 1.1899 of the
mesh's mean area, a relative error of 0.1899 against a tolerance of 0.05.

This says how large each specimen looks to this renderer. It says nothing about what a
detector or a policy does with that, and it proposes no cause for the closed D8b
result (`causality_vs_d8b: NOT_TESTED`).
