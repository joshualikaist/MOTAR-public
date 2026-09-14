# D8-B addendum — frozen-policy sensitivity to mesh-derived observation

Date frozen: 2026-09-13, after committed D8-A `TECHNICAL_GO` (`81f680e`) and before D8-B
runtime attachment code, any D8-B cell, or any D8-B outcome is observed.

This is simulation-only, evaluation-only sensitivity analysis. It does not authorize PPO,
adaptation, controller changes, real-flight claims, target identity claims, or deployment.

## 1. Question and lineage

How does one existing frozen policy respond when only its target renderer/perception input changes
from the historical analytic OBB to the D8 v3 visual mesh, first without and then with deterministic
within-object shading?

The same frozen ref5in D1 policy is used in every cell:

- checkpoint:
  `aerial_gym/rl_training/rl_games/runs/ppo_260813_1636_navrl_v2-ref5in-d1-q3-adapt-s197/nn/last_gen_ppo_ep_1900_rew_182.11377.pth`
- SHA-256: `197ea26999d6bb9cf23c4e5a55acbe945f89985e2384687d60ab1dbae66a278e`
- built-in `AppearanceTargetSegmenter`, threshold 0.55; no learned-detector substitution.

This policy was trained on the analytic arm. Mesh arms therefore measure out-of-distribution
sensitivity, not the quality of an adapted policy.

## 2. Factorial and held-fixed condition

Three observation arms × three fresh evaluation seeds = nine cells:

| arm | target geometry | target appearance |
|---|---|---|
| `analytic_flat` | historical analytic OBB | uniform nominal red |
| `mesh_flat` | v3 visual mesh | uniform nominal red |
| `mesh_shaded` | same v3 visual mesh | fixed material/normal/light modulation of nominal red |

Seeds are `593`, `599`, `601`. Before freezing this document, exact seed declarations were absent
from README/WORKLOG/VERIFICATION/RESEARCH_PLAN/docs(non-archive)/tools/tests/aerial_gym. Each cell
requests 2,049 completed episodes. Arms sharing a seed are paired by the seed and held-fixed
environment contract, not by episode identity; asynchronous termination means episode rows are not
assumed to be paired.

Held fixed: 70 bars, 40×40 m arena, goal distance 22.5–28.0 m, target speed U[0.3,1.5] m/s and mixed
pattern, deterministic action selection, original reflection, speed governor off, camera/detect
160×90, min pixels 2, detector range 20 m, no distractors, no P9 error injection, and every existing
appearance/noise/latency perturbation zero. Detect-resolution decoupling remains forbidden.

The only per-arm environment change is `NAVRL_DYNAMIC_MESH_TREATMENT` with canonical values
`off`, `mesh_flat`, or `mesh_shaded`. The runtime must attest the effective arm, v3 URDF SHA,
treatment mode, material/light constants, and `target_render_mode` in every result.

## 3. Estimands and analysis unit

Primary estimand: for each seed, `capture_rate(mesh_shaded) - capture_rate(analytic_flat)`, then the
arithmetic mean across the three seed differences. The uncertainty interval is the two-sided 95%
Student-t interval over those three paired seed differences (`df=2`). The seed is the inferential
unit. Episodes and adjacent frames are not treated as independent replicates.

The non-inferiority margin is **−3.0 percentage points**, fixed from the practical resolution used
for this renderer sensitivity gate rather than from D8-B results.

- `MATERIAL_LOSS`: the 95% CI upper endpoint is below −3.0 pp.
- `NO_MATERIAL_LOSS_WITHIN_MARGIN`: the 95% CI lower endpoint is above −3.0 pp.
- otherwise `INCONCLUSIVE_POLICY_SENSITIVITY`.

This single primary rule controls the D8-B verdict. If the point estimate is positive, it is still
described as a response in this one frozen lineage, not an improvement claim.

Secondary, descriptive estimands use the same three per-seed differences but do not change the
verdict: `mesh_flat - analytic_flat` capture; both contrasts for crash and timeout; first-acquisition
never-acquired rate; target-visible fraction; episode step summaries; and action/motion telemetry.
No threshold is fitted or changed from those outputs.

## 4. Integrity gates before the verdict

Every condition must hold or the campaign is `INVALID_D8B_EXECUTION` and no policy-sensitivity
verdict is issued:

1. D8-A receipt SHA is
   `92ce1c6a9d5f419de7b873af327c006b92246be9d9ba0da8c656dc9fe659e7ee` and its verdict is
   `TECHNICAL_GO`.
2. Preregistration and implementation/result commits form an ancestor chain; runtime tree and
   shared source bundle are clean and identical across nine cells.
3. Checkpoint SHA and copied snapshot SHA match the value above; no policy or detector artifact
   differs across arms.
4. Each result reports requested 2,049 and actual at least 2,049 episodes, with
   `captured + crash + timeout = actual`.
5. Every held-fixed runtime condition is identical within and across seeds except the declared
   seed, result nonce/path, and treatment arm.
6. `analytic_flat` attests no D8 object; mesh arms attest an attached treatment with the pinned v3
   URDF and D8 constants. No debug normal/face/material/GT field appears in actor observation.
7. The launcher binds and records the matching Python/ninja environment. Existing evaluator
   provenance refusal is never bypassed with a blanket force.
8. All nine cells exist before finalization; partial cells remain partial evidence and cannot be
   pooled into a verdict.

## 5. Stop and interpretation rules

- Never overwrite a cell, receipt, or historical D8/D9 result.
- Preserve environment/import failures as `VOID_EXECUTION`; retry only in a new campaign path or
  an explicitly suffixed cell path.
- Do not alter thresholds, seeds, episode count, analysis unit, or primary metric after observing
  any cell.
- Do not compare this campaign's absolute rates to historical seeds as if paired.
- Do not call a null-crossing interval equivalence, robustness, or no effect.
- `MATERIAL_LOSS` permits drafting a separate multi-seed adaptation preregistration; it does not
  start training. Other verdicts do not authorize adaptation.
- D9 remains a separate shortcut audit. D8-B has no distractors and cannot establish shortcut
  reduction, correct association, identity tracking, or motion-cue robustness.
