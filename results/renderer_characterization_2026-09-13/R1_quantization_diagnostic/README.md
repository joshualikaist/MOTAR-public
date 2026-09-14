# RC-R1 diagnostic — is the unmet-gate disagreement quantization?

**This is a diagnostic, not an experiment.** It carries no gate and no verdict, and it
changes nothing in RC-R1, which stands exactly as recorded.

Question: Is the RC-R1 G3/G4/G7 disagreement dominated by binary-silhouette quantization rather than by the intersection path?

Prediction, written before the numbers were read: a quantization-dominated relative error halves when the focal length doubles (ratio near 2.0).

| arm | error vs finest at 320x240 | at 640x480 | consecutive ratios | perimeter/area at 320x240 |
|---|---|---|---|---|
| `analytic_sphere` | 0.0014 | 0.0002 | 6.78 | 0.0852 |
| `box_proxy` | 0.0302 | 0.0100 | 3.01 | 0.1071 |
| `area_matched_box` | 0.0222 | 0.0121 | 1.84 | 0.1256 |
| `quadrotor_mesh` | 0.0222 | 0.0101 | 2.20 | 0.1518 |

A ratio near 2 means the error halves as the focal length doubles, which is what
silhouette quantization does. A ratio near 1 would mean the error does not depend on
resolution, and the quantization reading would be wrong.

Nothing here is evidence about the detector, the policy, or the cause of the closed
D8b result (`causality_vs_d8b: NOT_TESTED`).
