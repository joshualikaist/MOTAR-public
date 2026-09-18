# Paper evidence outline — 2026-09-18

This supersedes [`paper_draft_2026-09-05.md`](paper_draft_2026-09-05.md) as the structural plan for
a manuscript. That draft predates the target-motion ladder, the renderer contract freeze, the
record-envelope work and the quantitative positioning layer, and it must not be used as a final
manuscript.

This is an outline of **what may be claimed and on what evidence**, not prose. Every section names
its claim, the MOTAR evidence, the external citation, the figure or table, and the exact wording the
evidence licenses.

## Contribution candidates

Deliberately stated without superiority language. None of these asserts precedence or a ranking.

| # | Contribution | Supporting number | Figure / table | Result path | External relation |
|---|---|---|---|---|---|
| **C1** | An end-to-end evidence chain from measured perception error to policy outcome, retaining negative links | The chain in [the claim matrix](paper_claim_evidence_matrix_2026-09-18.md) | Fig. `quantitative-positioning` | `results/MANIFEST.json` (201 entries) | No screened work publishes an equivalent chain |
| **C2** | Quantified perception-to-policy sensitivity driven by a *measured* error distribution | −4.57 pp capture cost, CI excludes zero, 3/3 campaigns | Table 3 row "Measured perception-error cost" | `results/perception_p10_seed_replication_2026-09-10/` | NavRL++ reports a >5 % drop from its own perturbation model |
| **C3** | Obstacle-density and target-motion generalization of a frozen policy | Density sweeps recorded per cell; H/E0/E1/E2 `RESULT_PENDING` | Table 3; target-motion row | `docs/target_behavior_ladder_2026-09-16.md` | NavRL sweeps dynamic-obstacle count; no work sweeps a target-motion ladder |
| **C4** | Safety-filter geometry diagnosis with matched configured arms | −1.4903 pp crash, CI [−1.8981, −1.0826], 15/15 cells | Table 3 rows "Safety-filter geometry", "Arc geometry width" | `results/independent_verification_2026-09-07/` | Temporal Barrier gives a certificate-style comparator in an obstacle-free domain |
| **C5** | A renderer/observation contract with measured frozen-policy sensitivity and explicit untested causality | −48.967 pp, CI [−50.113, −47.821], `MATERIAL_LOSS`, causality `NOT_TESTED` | Table 5 row "Appearance D8b" | `results/dynamic_mesh_policy_sensitivity_d8b_2026-09-13/` | None found |

Banned in every section: `state of the art`, `best`, `outperforms all prior work`, `first ever`.
`tests/test_quantitative_positioning.py` enforces this across the public documents.

## Section plan

### 1 Introduction
* **Claim.** A simulation-only study of perception-aware motion toward a moving object in dense
  obstacle fields, whose contribution is a connected chain of measurements.
* **MOTAR evidence.** The contribution table above.
* **External citation.** NavRL, YOPOv2-Tracker, OPEN, Elastic Tracker as the surrounding field.
* **Figure/table.** Fig. 1 research overview; the positioning figure.
* **Allowed wording.** "We report a measured chain from perception error to policy outcome, including
  its negative and inconclusive links." Not: any claim of being first or best.

### 2 Related Work
* **Claim.** MOTAR's combination of axes differs in *scope* from the screened systems.
* **MOTAR evidence.** [Relation to published systems](relation_to_published_systems_2026-09-16.md):
  18 screened, 10 retained, Class A = 0.
* **External citation.** All twelve retained works, with the ledger's per-work conditions.
* **Figure/table.** Site Table 5 (published-system relation); the new quantitative positioning table.
* **Allowed wording.** "No screened system shares MOTAR's task contract, sensor model and metric, so
  no numerical comparison is available." Not: a claim that MOTAR does better on any of their metrics.

### 3 System
* **Claim.** A documented UAV dynamics, control, camera and LiDAR stack with a declared
  sensor-only observation contract.
* **MOTAR evidence.** Site Table 1 parameters; the component/lifecycle registry.
* **External citation.** NavRL for the PPO-plus-shield architecture family.
* **Figure/table.** Figs. 1, 6, 7; Table 1.
* **Allowed wording.** "The observation contract is declared and machine-checked." Note that 14 of
  112 config values differ from HEAD defaults and are env-var opt-in, so the training contract is
  **not reproducible from HEAD defaults**.

### 4 Perception and Error Modeling
* **Claim.** Detector and temporal-association error was measured on real imagery and injected into
  simulation, and the injected error costs the frozen policy a specific amount of capture.
* **MOTAR evidence.** E3-S 6.2 % median absolute relative range error over 3,107 frames;
  P8 → P9 → **−4.57 pp**; P10 **INCONCLUSIVE** at +0.73 pp, CI [−1.04, +2.50].
* **External citation.** NavRL++ perturbation analysis (perception failure the largest factor, >5 %).
* **Figure/table.** Table 3 rows; E3 analysis diagram.
* **Allowed wording.** "The measured perception error costs the frozen policy 4.57 pp of capture,
  with an interval clear of zero in all three campaigns; readaptation recovers part of it and the
  net benefit is not established." Not: that readaptation works.

### 5 Navigation / Close Approach
* **Claim.** Capture, crash and timeout are recorded per cell under a fixed evaluation contract.
* **MOTAR evidence.** `P2 held-out` **STRICT FAIL** (timeout 5.56 %, 114 events against 102 allowed);
  `D1 adaptation` **FAIL**; `P3 full budget` **BLOCKED**; detection-range Stage 1
  `RANGE_INCONCLUSIVE_AT_THIS_BUDGET`.
* **External citation.** Fast-Tracker's tracking rate; YOPOv2-Tracker's real-world speeds.
* **Figure/table.** VERIFICATION gate table.
* **Allowed wording.** "The preregistered held-out gate is a FAIL and no later diagnostic changes
  that." Capture ends the episode, so following-phase integrals are structurally biased.

### 6 Safety Analysis
* **Claim.** Two configured filter geometries produce different crash rates, with no guarantee.
* **MOTAR evidence.** −1.4903 pp, CI [−1.8981, −1.0826], 15/15 cells; arc width −5.60 pp crash and
  +4.44 pp capture at 205 bars.
* **External citation.** Temporal Barrier (aTTC-CBF), explicitly as a *formal* comparator MOTAR is not.
* **Figure/table.** Table 3; safety-filter diagram.
* **Allowed wording.** "A configured comparison in simulation, with no formal collision-safety
  guarantee." The C3 explanation is WITHDRAWN and route-mechanism gates are `FAIL_ROUTE_MECHANISM`.

### 7 Target-Motion Generalization
* **Claim.** The target ladder separates static, constant-velocity and obstacle-aware targets, and a
  frozen policy's validity across them is an open measured question.
* **MOTAR evidence.** TM-E0/E1 implemented; TM-E2 `IMPLEMENTED; POLICY_COMPARISON_NOT_TESTED`;
  H/E0/E1/E2 evaluation **`RESULT_PENDING`**.
* **External citation.** Elastic Tracker's cooperative target and its "escaping target" future-work
  statement; OPEN's scripted evader.
* **Figure/table.** Target-motion ladder; the pending positioning row.
* **Allowed wording.** Until the evaluation is finalized: "not yet reported." **No partial cell may
  be cited.** Note the frozen policy was trained on the `legacy`/`mixed` lineage, so none of
  E0/E1/E2 is its training distribution and arm H exists as the in-distribution reference.

### 8 Renderer / Observation Contract
* **Claim.** A frozen policy scored substantially worse under a mesh-shaded observation treatment,
  and the cause was not tested.
* **MOTAR evidence.** D8b −48.967 pp, CI [−50.113, −47.821], `MATERIAL_LOSS`; Renderer Contract v1
  frozen with `NOT_ADOPTED` area control, `N0_ONLY` normals, and RC-R1/R2/R3 failures retained.
* **External citation.** None found.
* **Figure/table.** Fig. 7; Table 5.
* **Allowed wording.** "A frozen policy scored worse under a new observation treatment, and nothing
  more." `causality_vs_d8b = NOT_TESTED` must appear wherever D8b is cited.

### 9 Quantitative Positioning
* **Claim.** The field's reported numbers and MOTAR's measured numbers can be shown together without
  being combined.
* **MOTAR evidence.** The three-layer registry; Class A = 0.
* **External citation.** All ledger entries, with `NOT_EXTRACTED` preserved.
* **Figure/table.** The positioning figure; site section 6.2.
* **Allowed wording.** "Reported under each publication's own benchmark; not head-to-head."

### 10 Limitations
* No real-flight validation of anything.
* `P2` and `D1` are FAIL; `P3` is BLOCKED.
* No adaptation benefit established; S4's generalization warning stands.
* No causal attribution for D8b.
* Target-behavior comparison unfinished; TM-E3/TM-E4 planned only.
* Class A external comparisons: **0**.
* Licensing unresolved for several third-party datasets (`NEEDS_CONFIRMATION`).
* Permutation-based inference on 3 seeds is structurally limited; interval estimates carry the
  inference.
