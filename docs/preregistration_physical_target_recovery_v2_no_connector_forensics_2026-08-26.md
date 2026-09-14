# Preregistration — recovery-v2 lower-1.25 NO_CONNECTOR geometry forensics

Date frozen: 2026-08-26, after the verified lower-1.25 32-cell
`FAIL_ROUTE_MECHANISM` and the CPU packed occupancy diagnosis, and **before** any
no-anchor GPU child. This document replaces the same-day stub. Evaluation-only.

It does not authorize a 32-cell rerun, controller/gain change, `0.45 m` retune, PPO,
a 1.5 m/s claim, an env-count change, or edits to the 32-cell evaluator or its artifacts.

## Provenance

The diagnostic refuses to run if the preserved lower-1.25 gate `summary.json` does not have
SHA-256 `a85e95764061b7b20cacaa622efc44e2d7e31e054e398f57cfdf48ec98e6c04f`. It also binds
receipt SHA-256 `707636fcbcfe0c855267b39e307af7ac133a0feabbf25d2e7feba726465f1f96`.
The 32-cell gate directory is read-only input and remains the authority for the prior FAIL.
Outputs must never be written into
`results/navrl_physical_target_recovery_v2_gate_lower1p25_seed827/`.

The GPU child must instantiate the same runtime bytes the gate bound at commit
`2b151d9a4c4fe078ecc027152e5642fa857a2e2f` (CORE_PATHS under `aerial_gym/` and `resources/`).
A later documentation/diagnostic commit is allowed only when those runtime bytes still match
the gate `source_manifest`. The diagnostic tool itself is new and is hashed in the forensic
receipt, not in the 32-cell receipt.

The heading-rest braking receipt remains
`results/navrl_physical_target_braking_lower1p25_headingrest_seed827/` with SHA-256
`4e87eb9ddf5dd9cea1fc0354d272a5d18ec6a05427e0f41e672749a57df9047a`. Variant token
`baseline_1p25`. Default execution remains canonical 1.5 and must not be used here.

## Frozen probe

Seed `827`, 32 environments, 300 RL intervals, 20-step warmup marker (parity only; every
interval is eligible for a latch event), physical waypoint target, `base_sim`,
`navrl_ref5in_quad`, gain 2.5, route mode `global_astar_recovery_v2`.

Cells: the four 70-bar recovery-v2 speeds `{0.6, 0.9, 1.2, 1.25}` only. One fresh child per
speed. Density knots 150/205/300, the route-off arm, and any 1.5 m/s cell are out of scope.
No cell may be added, removed, or substituted after events are inspected.

Packed diagnosis already showed that 70-bar fallback is the mechanism pool and that BRAKE-origin
generic `NO_CONNECTOR` grows with speed. This probe isolates that latch at the 70-bar
mechanism density. It does not re-open the 32-cell grid.

Run command (GPU, not run as part of this CPU-only freeze):

```text
NAVRL_TARGET_BRAKING_CONTRACT_VARIANT=baseline_1p25 \
<aerialgym-python> tools/diagnose_navrl_physical_target_recovery_v2_no_connector.py --run
```

Outputs go only under
`results/navrl_physical_target_recovery_v2_no_connector_forensics_seed827/`.

## Observer and claim boundary

The 32-cell packed npz has state/status/margins but not bar poses, so it cannot recompute the
7×7 connector. The observer attaches at evaluation time to the existing
`recovery_anchor_idx` / `mark_no_connector` / resume-replan calls. It consumes the same
position, bar AABB, half-extents, arena bounds, support, wall margin, and speed values the
transition already uses. CPU conversion occurs only in the diagnostic child. The observer is
not imported by the task and is absent from the normal control path. It never supplies a
command or changes a planner decision. A malformed/non-finite diagnostic value is reported as
null and cannot become a passing zero.

At every `NO_CONNECTOR` entry the record stores:

- recovery state before the latching call, status after, realized planar speed;
- exact hard and soft signed AABB clearances (support-only hard; support+`0.45 m` soft);
- packed class using the already-frozen labels
  `brake_no_anchor_likely`, `same_interval_brake_no_anchor_likely`, `brake_timeout`,
  `connect_failed_resume_likely`, `connect_failed_certificate_likely`,
  `connect_timeout`, `route_to_no_connector`, `hard_breach`;
- the **runtime** `recovery_anchor_idx` boolean for that environment in that interval;
- an independent CPU replica of `nearest_soft_free_anchor` with the **recovery-v2** kwargs,
  not the v1 forensic kwargs: `resolution_m=0.25`, `radius_cells=3`,
  `tracking_margin_m=0.45`, `soft_hysteresis_m=0.25`,
  `hard_epsilon_m=1e-4+0.0123`, `hard_boundary_margin_m=0.50`,
  `soft_boundary_margin_m=1.25`;
- whether `brake_connector_idx` certified the stopping tube on that interval (descriptive;
  BRAKE timeout remains a separate class).

For CONNECT-origin latches with positive soft margin, also store the resume-replan
`status_code` after the existing `_plan_target_routes(..., is_replan=True)` call. Do not
invent a new planner status.

The replica must disagree with a v1-style search (no hysteresis, `hard_epsilon=1e-4` only)
on at least the CPU fixtures in
`tests/test_navrl_recovery_v2_no_connector_forensics.py`. Copying the 2026-08-25 forensic
search is a contract failure.

## Analysis rules fixed before data

Primary denominator: BRAKE-origin generic no-anchor entries in the four 70-bar cells, i.e.
packed classes `brake_no_anchor_likely` and `same_interval_brake_no_anchor_likely` only.
BRAKE timeout, CONNECT, ROUTE, and hard-breach entries are reported separately and do not
vote.

1. Report per-cell and pooled counts of every packed class.
2. Among the primary denominator, report the fraction for which the CPU replica returns
   `exists is True` and `hard_connector_safe is True`. Also report the fraction that are
   hard-free and soft-unsafe at latch. Wilson 95% intervals are required when `n >= 20`.
3. Report CONNECT-origin positive-soft-margin resume-replan status counts. This is not the
   primary label.
4. Before inspecting GPU events, freeze this descriptive rule (not a 32-cell gate, not a
   planner change):
   - `ANCHOR_PRESENT_LATCH` if primary `n >= 20` and the Wilson 95% lower bound on replica
     hard-safe-anchor existence is `> 0.5`;
   - `ANCHOR_ABSENT_AT_LATCH` if primary `n >= 20` and the Wilson 95% lower bound on
     *absence* is `> 0.5`;
   - otherwise `INCONCLUSIVE`.
5. If the runtime `recovery_anchor_idx` boolean and the CPU replica disagree on any primary
   event, the receipt is `VOID` rather than a scientific FAIL. That is an observer-identity
   defect, not evidence about anchors.
6. Do not lower `0.45 m`, change gain 2.5, change env count, change radius 3, change
   hysteresis, rerun a cell after seeing data, or treat a completed probe as a recovery-v2
   gate pass, a 1.5 m/s result, training authorization, or hardware validation.

`ANCHOR_PRESENT_LATCH` means the 7×7 hard-safe connector already existed and the state
machine still latched. `ANCHOR_ABSENT_AT_LATCH` means the neighbourhood was empty when
BRAKE needed it. Neither label authorizes a fix.

## Frozen deliverables

Each cell JSON is hashed in `receipt.json`. The receipt contains this contract, tool hash,
runtime source hashes, the 32-cell summary/receipt SHAs, braking-receipt SHA, and explicit
`gate_artifacts_read_only` and `original_evaluator_unchanged` markers. A receipt or cell
with a contract mismatch is invalid and must be discarded rather than repaired in place.
VOID/incomplete siblings cannot be combined with a later attempt.
