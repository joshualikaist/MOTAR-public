# MOTAR master roadmap — evidence and implementation boundaries

This is the current roadmap for **repository maintenance and independent generic graphics**.
It does not authorize an operational interception/engagement programme under renamed terminology.
Historical interception, capture and contact metrics keep their original meanings and paths.
No new PPO, reward tuning, checkpoint schema change or detector/control integration is started.

## Stage ledger

| Stage | Question and why it matters | Current evidence | Decision / dependency |
|---|---|---|---|
| R0 repository/reproducibility | Can an outsider install the independent tools and inspect evidence? | CPU profile, archived smoke, doctor, public docs and CI | Implement/test now; hosted CI remains unverified until run |
| R1 appearance/rendering | Are static graphics outputs reproducible, and what is their full-loop cost? | Existing independent renderer/background; historical statistical failures preserved | Generic boxes/background only; no target-specific control fixture |
| P0 perception support audit | Are existing claims labelled with their actual source/support limits? | P8/streaming/S4 records are historical, not new environment coverage | Documentation/index only; new operational support measurements BLOCKED_BY_POLICY |
| P1 identity/association | Requested target identity improvement | Historical identity limits remain | BLOCKED_BY_POLICY |
| P2 search/reacquisition | Requested target reacquisition capability | No new claim established by this cycle | BLOCKED_BY_POLICY |
| E0 range/uncertainty | Are existing range and attitude claims separated? | E3-S measured; E3-P ATTITUDE_NOT_RELIABLE | Evidence indexing now; new decomposition BLOCKED_BY_DATA/EVIDENCE |
| N0 navigation baseline | Requested new vehicle-policy baseline | Historical results preserved | BLOCKED_BY_POLICY in this interception context |
| S0 remaining safety-filter experiments | What has actually been reported? | Existing diagnostic and replication records, no formal guarantee | Historical bookkeeping only; operational performance experiments BLOCKED_BY_POLICY |
| T0 terminal failure forensics | Requested diagnosis to improve final physical approach | No new analysis performed | BLOCKED_BY_POLICY |
| T1 strict contact metric | Requested additional target-contact evaluation | Historical metrics are not changed | BLOCKED_BY_POLICY |
| T2 terminal controller comparisons | Requested physical approach/contact control comparison | No new controller implemented | BLOCKED_BY_POLICY |
| H0 supervised/RL hybrid study | Requested integrated target estimator/mode manager/controller development | No new architecture validated | BLOCKED_BY_POLICY |

## R0 — implementable work contract

**Question / importance:** can installation and evidence checks fail explicitly and reproducibly?
**Current evidence:** [CPU installation](../../results/renderer_cpu_install_2026-09-12/README.md),
[follow-up regression](../repository_followup_2026-09-12.md).
**Inputs:** full Git checkout, the separate CPU and documentation profiles, stored synthetic evidence.
**Outputs:** landing README, citation, notices, doctor JSON, CI definition, history audit, checklist.
**Implementation:** `motar_doctor.py`, `check_public_docs.py`, `audit_public_history.py`, schema/site index.
**Experiment:** technical installation/smoke/metadata checks only, not policy evaluation.
**Metrics:** process exit codes, explicit missing dependencies, file/decoded hashes, schema validity,
test execution/failure/error/skip counts; no hardcoded changing test count in the landing page.
**Gate:** local checks pass and hosted CI is independently recorded; legal unknowns remain explicit.
**Stop:** dependency/network failure records incomplete state; suspected secrets prevent new publication.
**Human time estimate:** review 2–4 hours; **GPU:** none. Estimates are not measured agent runtime.
**Dependency:** source history, accessible upstream licence texts; maintainer decisions for ownership.
**Contribution:** reproducible negative-result reporting and portable evidence, not a new control method.

## R1 — independent static graphics contract

**Question / importance:** do appearance interventions preserve geometry and debug-ID independence?
**Current evidence:** loader/renderer technical checks; R4/R4b FAIL remains a separate statistical result.
**Inputs:** fixed procedural boxes or floor/wall/column/panel geometry; camera pose; independent material
and light seeds; resolution and batch. No UAV/target assets, task state, detector or policy.
**Outputs:** separate RGB/depth/range/normal/face/instance/valid arrays and a provenance receipt.
**Implementation:** public exporter, export verifier, full-loop static benchmark, existing kernels reused.
**Experiment:** five export interventions and a declared finite technical benchmark matrix.
**Metrics:** decoded hashes; RGB changes under material/light; geometry invariance; ID renumber invariance;
timing samples, mean/median/P95; correctly scoped memory figures.
**Gate:** each technical condition is reported, no changed seed/criterion after inspecting failure.
Benchmark completion requires all requested cells or an explicit incomplete record; no guessed cells.
**Stop:** budget exceeded, source drift, nonfinite buffers, process failure or failed invariant.
**Human time estimate:** engineering/review 2–5 hours; **GPU:** approximately 0–0.25 hours for small
generic timing runs, not training. Actual timing is recorded by receipts, not this estimate.
**Dependency:** verified CPU profile or existing compatible GPU runtime; committed clean benchmark source.
**Contribution:** independently reproducible graphics infrastructure; no inference about shortcut reduction.

The requested target-specific 21-view area-matched box and quadrotor benchmark are not included.
Their intended role in this project is to support target-perception improvement, so they are
**BLOCKED_BY_POLICY**, not merely awaiting compute or permission. The implemented alternative
is general static graphics validation, not a renamed target fixture.

## P0 and E0 — evidence support, not a new inference pipeline

**Question:** are reported support and uncertainty limits distinguishable from new measurements?
**Why:** unsupported data and missing GT must not be converted to PASS by wording.
**Current evidence/input:** historical [P8](../../results/perception_p8_2026-09-09/README.md),
[S4](../../results/perception_s4_2026-09-09/README.md),
[E3-S](../../results/eth_ds5_e3s_2026-09-10/README.md),
[E3-P](../../results/eth_ds5_e3p_reliability_2026-09-10/README.md).
**Output/implementation now:** cited evidence overview and protected status fields; no new episode logger.
**Experiment/metric:** document and source-reference consistency only. Historical bins/results are not
changed and no newly claimed size, visibility, track-age, range or association distribution is invented.
**Gate/stop:** source missing or contradicted → BLOCKED_BY_EVIDENCE; missing reliable attitude data →
BLOCKED_BY_DATA. These are evidence limitations, not permission to perform the blocked operational work.
**Human time estimate:** 0.5–1 hour of document review; **GPU:** 0.
**Dependency:** original result records, no new dataset download required.
**Contribution:** honest separation of measured quantities and untested hypotheses.

## S0 — historical backlog status

The requested hook inventory must distinguish existence, unit tests, preregistration and results.
This cycle does not review operational hook implementations to propose performance improvements.

| Historical hook label | Implemented? | Tested? | Preregistered? | Result? | Current action |
|---|---|---|---|---|---|
| speed-dependent width | NOT_AUDITED_THIS_CYCLE | NOT_AUDITED_THIS_CYCLE | NOT_AUDITED_THIS_CYCLE | Not promoted from code existence | BLOCKED_BY_POLICY operational audit |
| open-space adaptive width | NOT_AUDITED_THIS_CYCLE | NOT_AUDITED_THIS_CYCLE | NOT_AUDITED_THIS_CYCLE | Not promoted from code existence | Same boundary |
| lateral clearance | NOT_AUDITED_THIS_CYCLE | NOT_AUDITED_THIS_CYCLE | NOT_AUDITED_THIS_CYCLE | Not promoted from code existence | Same boundary |
| yaw cap | NOT_AUDITED_THIS_CYCLE | NOT_AUDITED_THIS_CYCLE | NOT_AUDITED_THIS_CYCLE | Not promoted from code existence | Same boundary |

**Inputs/current evidence:** existing [verification](../../VERIFICATION.md) and dated worklog only.
**Output:** truthful backlog classification, no claim of implementation readiness or formal safety.
**Experiment/metric/gate:** no new experiment; preserve negative/withdrawn findings.
**Stop:** request becomes tuning, controller design or improved interception evaluation.
**Time/GPU:** no operational implementation estimate; 0 GPU allocated.
**Contribution/dependency:** evidence bookkeeping; original author-reviewed result records needed to fill
unknown cells. Unknown is not equivalent to “absent”, “failed” or “complete”.

## Boundary change, 14 September 2026 — diagnostics authorized, architecture still blocked

The repository owner changed this boundary on 2026-09-14. The text of the blocked-stage section
below is **kept unedited** as the record of what the boundary said before, because a policy that can
be silently rewritten is not a policy.

**Now authorized, and only this:** evaluation-only instrumentation of runs produced by *existing*
checkpoints — per-episode search/acquisition/reacquisition records (TD-T1) and terminal
close-approach forensic records with additional evaluation-only distance metrics and a documented
failure taxonomy (TD-T2). The purpose is to measure which failure modes actually dominate before any
architecture is proposed.

**Still blocked, unchanged:** any policy, reward, detector, association, safety-filter, controller or
termination change; any retraining or new training lineage; search heuristics; terminal controller
replacement; and any operational capability claim. The historical capture criterion
(`success_radius = 0.5 m`) is not modified — new distance metrics are additional and
evaluation-only. A diagnostic result does not authorize the architecture it might motivate; that
needs its own preregistration.

**Naming:** the new work is written `TD-T1` and `TD-T2` (task diagnostics). The stage codes T0/T1/T2
in the ledger above are a different, older naming and are not the same items.

## Blocked stages P1/P2/N0/T0/T1/T2/H0 — text as it stood before 2026-09-14

For **each** of these rows, the requested input is integrated vehicle/target state or trajectories;
the requested output would diagnose or improve target acquisition, physical approach or contact.
**Implementation, experiments, metrics, decision thresholds, control architecture and training recipes
are not specified here.** Evaluation-only labels do not remove that intended operational role.
This includes terminal telemetry/classification, reward-component exports for retuning, strict-contact
instrumentation, hybrid modes, privileged-teacher distillation and target-prediction model comparisons.

**Status:** BLOCKED_BY_POLICY. **Prerequisite:** a genuinely separate non-engagement task; compute,
new approval, a renamed objective or a simulation-only wrapper does not by itself change this boundary.
**Alternative now:** static general graphics, metadata/evidence validation and repository maintenance.
**Stop condition:** any requested connection to target acquisition or final physical contact improvement.
**Human/GPU estimate:** N/A for blocked work; 0 training/GPU budget allocated to it.
**Scientific contribution:** none newly claimed; historical source records remain accessible.

## Current status, 14 September 2026

```text
Perception P1-P10        COMPLETE / bounded claims
Safety diagnosis         COMPLETE
Renderer Contract v1     COMPLETE  (frozen; not a blanket renderer pass)

Search diagnostics       NEXT      (TD-T1, evaluation-only instrumentation)
Terminal diagnostics     NEXT      (TD-T2, evaluation-only instrumentation)

Search architecture      NOT STARTED
Terminal controller      NOT STARTED
New training lineage     NOT JUSTIFIED YET
```

The renderer track is closed at [Contract v1](../renderer_track_v1_freeze_2026-09-14.md) and is a
side methodological branch, not the main line.

## Boundary change, 16 September 2026 — target difficulty axis documented and TM-E2 exposed

The repository owner explicitly authorized a target-behavior difficulty ladder and an opt-in
TM-E2 obstacle-aware scripted target implementation. This does **not** authorize a new PPO run,
checkpoint retuning, or an E0/E1/E2 performance claim.

- `historical` remains the default and preserves every existing run.
- TM-E0/TM-E1 select the existing static/CV baselines.
- TM-E2 selects the existing bounded waypoint executor, which may use privileged simulator
  obstacle geometry only for the target.
- The pursuer observation remains unchanged and receives no privileged target position or
  obstacle map.
- TM-E3 reactive evasion and TM-E4 self-play remain `PLANNED`; selecting either fails closed.
- The prior physical routed-target `FAIL_ROUTE_MECHANISM` is not changed by exposing this axis.

The code, limitations and future preregistration requirement are in
[target behavior ladder](../target_behavior_ladder_2026-09-16.md).

## Evidence map

```text
Perception
    v
Measured perception error
    v
Navigation policy
    v
Safety filter
    v
Search / reacquisition diagnostics      <- TD-T1, instrumentation only
    v
Terminal approach diagnostics           <- TD-T2, instrumentation only

side branch:  Renderer Contract v1  (frozen, causality_vs_d8b = NOT_TESTED)
```

D8b frozen-policy sensitivity stays `MATERIAL_LOSS` at −48.967 pp, and no renderer or diagnostic
result in this repository establishes its cause.

## Execution order and completion rule

R0 implementation and tests → R1 generic export checks → committed-source generic benchmark →
final local regression and release checklist → remote review/CI if publication gates permit.
P0/E0 document checks run with R0. No new policy training is a successor step of this cycle.
Each failed dependency/licence/evidence check keeps its own BLOCKED reason without stopping
unrelated safe tooling. A local green suite is not hosted CI success or blanket public-release clearance.
