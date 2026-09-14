# Research evidence index

One page, seven axes, bounded claims. This is not a development log: it names the question, how it
was answered, the strongest thing the evidence supports, and what that evidence cannot carry.
Commit-by-commit history is in [WORKLOG](../WORKLOG.md); execution authority is
[VERIFICATION](../VERIFICATION.md); every result directory is indexed in
[`results/MANIFEST.json`](../results/MANIFEST.json).

Nothing here promotes a result. Where a track ended negative, inconclusive or blocked, it says so.

---

## 1. Navigation and RL

**Question.** Can a learned policy reach a moving target in a dense bar field from vision and LiDAR
alone, under a fixed control stack?

**Method.** Transformer-PPO on structured observations; held-out density sweeps at a fixed
evaluation contract; outcome strata by distance, speed and target pattern.

**Best established evidence.** Held-out density sweeps are recorded per cell with receipts and
per-outcome strata. Capture, crash and timeout rates exist for the evaluated checkpoints at the
evaluated densities.

**Status.** `P2 held-out` **STRICT FAIL** (timeout 5.56 %, 114 events against 102 allowed);
`D1 adaptation` **FAIL** (q3/CV timeout 15.98 % > 12 %); `P3 full budget` **BLOCKED** until P2
passes. Stage 1 of the detection-range arm is `RANGE_INCONCLUSIVE_AT_THIS_BUDGET`
(never-acquired 8.443 → 3.172 %, −5.271 pp against a preregistered −15 pp gate); Stage 2 is not
authorized.

**Primary result path.** [`VERIFICATION.md`](../VERIFICATION.md) — the gate table is authoritative.

**Known limitation.** No deployment claim, no real-flight validation, and no diagnostic result
retroactively changes a FAIL or unblocks P3.

---

## 2. Perception

**Question.** How well does the detector plus temporal association find and hold the target?

**Method.** Detector and association lineages P1–P7 on recorded imagery, with held-out splits and a
separate selector evaluation.

**Best established evidence.** Detector and association performance is recorded per lineage and
per dataset, with the selector evaluated separately from the detector.

**Status.** Recorded. Identity improvement is **not** claimed.

**Primary result path.** [results overview, Track B](results_overview_2026-09-12.md).

**Known limitation.** Dataset- and lineage-specific. [S4](../results/perception_s4_2026-09-09/README.md)
carries an explicit generalization warning, and it stands.

---

## 3. Perception error modelling

**Question.** What is the measured error of the perception stack, and what happens when that
measured error is injected into the simulator?

**Method.** P8 measured error distribution; P9 simulator injector driven by it; P10 policy
readaptation under the injected error; E3-S image-size-to-range proxy on the ETH ds5 flight.

**Best established evidence.** A measured error distribution exists (P8) and is injectable (P9).
E3-S is `SIZE_RANGE_USABLE`: held-out median absolute relative range error **6.2 %** over 3,107
frames, 9 blocks, 30.7–108.4 m.

**Status.** P10 replication **INCONCLUSIVE** — a net adaptation benefit is **not** established.
E3-P is `ATTITUDE_NOT_RELIABLE` and the pose/range decomposition stays **BLOCKED**.

**Primary result path.** [P10 replication](../results/perception_p10_seed_replication_2026-09-10/README.md),
[E3-S](../results/eth_ds5_e3s_2026-09-10/README.md),
[E3-P](../results/eth_ds5_e3p_reliability_2026-09-10/README.md).

**Known limitation.** E3-S is one flight and an apparent-size proxy, not ground-truth boxes and not
a causal separation of attitude. Range and time are confounded in that flight.

---

## 4. Safety filter

**Question.** What do the configured speed-governor geometries actually do, and where do they fail?

**Method.** Configured comparisons over recorded runs, with crash-cause attribution and replication.

**Best established evidence.** Geometry-dependent failure modes are exposed and recorded, with
replication.

**Status.** Diagnosis recorded. The C3 explanation is **WITHDRAWN**. Route-mechanism gates are
`FAIL_ROUTE_MECHANISM` (recovery-v2 lower-1.25 32-cell, and the corrected non-overlap r2), and the
70-bar no-anchor follow-up is `INCONCLUSIVE` at n = 1.

**Primary result path.** [replication](../results/navrl_grid_r2_d4_trainseed_rep/README.md).

**Known limitation.** No formal safety guarantee of any kind. These are configured comparisons in
simulation.

---

## 5. Appearance and renderer

**Question.** How does the renderer represent geometry, area, normals, shading, lighting and cost —
and does a mesh-derived observation change a frozen policy's performance?

**Method.** Track D (D1–D8b) for the simulator-integrated appearance path; the independent renderer
characterization track (RC-R1…R6 and the R1b/R2b/R3b/R5b follow-ups) for the graphics contract.

**Best established evidence.** [Renderer Contract v1](renderer_track_v1_freeze_2026-09-14.md),
frozen: intersection verified against an independent intersector, measurement resolution qualified
at 1280×960 for its tested coverage, lighting/material isolation and determinism verified, transfer
cost characterized. D8b measured a frozen-policy contrast of **−48.967 pp**, 95 % CI
[−50.113, −47.821] pp, verdict `MATERIAL_LOSS`.

**Status.** Renderer Contract v1 **FROZEN**, and it is **not** a blanket renderer pass: area control
`NOT_ADOPTED` after two controls failed on held-out views, normals `N0_ONLY`, and RC-R1
`GEOMETRY_DEFECT`, RC-R2 `AREA_MATCH_FAILED`, RC-R3 `SHADING_GATE_FAILED` all stand. D6 remains
`INCONCLUSIVE`; D9 is `NOT_RUN`.

**Primary result path.** [renderer track freeze](renderer_track_v1_freeze_2026-09-14.md),
[D8b](../results/dynamic_mesh_policy_sensitivity_d8b_2026-09-13/README.md).

**Known limitation.** **`causality_vs_d8b = NOT_TESTED`.** No renderer result establishes that area
mismatch, shading or mesh geometry caused the D8b loss; that experiment was never run. D8b says a
frozen policy scored worse under a new observation treatment, and nothing more.

---

## 6. Reproducibility

**Question.** Can an outsider install the independent tools, re-run the generic checks, and tell
what produced each result?

**Method.** A CPU-only renderer profile, public document/schema checks, per-result receipts and
source manifests, a record-integrity checker, Record Envelope v2 for future producers, and a
repository-wide result manifest.

**Best established evidence.** The CPU renderer path installs and its validation runs without a
GPU. Every modern result directory carries a receipt and a source manifest whose file hashes are
verifiable. The result manifest indexes **201** result entries with their links and hashes.

**Status.** Record Envelope v2: `producer_unit_validation = PASS`,
`live_generation_validation = NOT_APPLICABLE` — the contract is task-shaped and does not fit a
generic producer, which is
[recorded rather than worked around](record_envelope_v2_2026-09-14.md). The historical 128-row
artifact keeps both verdicts: byte integrity `MATCH`, metadata contract
`INVALID_FOR_DECLARED_CONTRACT`.

**Primary result path.** [REPRODUCIBILITY](REPRODUCIBILITY.md),
[record-integrity review](../results/record_integrity_review_2026-09-14/README.md).

**Known limitation.** Hosted CI is unverified. Several datasets cannot be redistributed, so some
results are reproducible only in their recorded form, not from raw data.

---

## 7. Open limitations

* **No real-flight validation of anything.** Every number in this repository is simulation or
  recorded-dataset analysis.
* **P2 and D1 are FAIL, P3 is BLOCKED.** No later diagnostic changes that.
* **No adaptation benefit is established** (P10 INCONCLUSIVE), and S4's generalization warning
  stands.
* **No causal attribution for D8b.** Renderer work is a side branch with `NOT_TESTED` causality.
* **No area-matched control geometry exists**, and the renderer supports face normals only.
* **Task-level failure structure is instrumented but unmeasured.** TD-T1 and TD-T2 recorders exist
  and are trajectory-invariant; the preregistered density sweep has not been run.
* **Licensing is unresolved for several third-party datasets** and is tracked as
  `NEEDS_CONFIRMATION` rather than assumed; see the release audit.
