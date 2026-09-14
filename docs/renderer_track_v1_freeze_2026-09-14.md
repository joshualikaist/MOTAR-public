# Renderer characterization track — v1 frozen, 14 September 2026

The renderer characterization track is closed at v1. Machine-readable record:
[`renderer_track_status_v1.json`](renderer_track_status_v1.json).

```text
RENDERER CHARACTERIZATION V1

Geometry intersection        VERIFIED
Measurement resolution       QUALIFIED
Area control                 NOT ADOPTED
Normal representation        N0 ONLY
Lighting/material isolation  CHARACTERIZED
Determinism                  VERIFIED
Transfer cost                CHARACTERIZED
Dataset exporter             CONTRACTED

Overall:
RENDERER_CONTRACT_V1_COMPLETE
```

## What this is not

`RENDERER_CONTRACT_V1_COMPLETE` means the preregistered v1 questions were answered and the track is
closed. **It is not a blanket renderer PASS.** Two of the answers are negative, and they stay
negative:

* **Area control NOT ADOPTED.** Both the historical isotropic control (RC-R2 `AREA_MATCH_FAILED`)
  and the three-axis control fitted and frozen in R2b (`AREA_CONTROL_FAILED`, held-out median
  8.132 % against a 5 % gate) failed on views they were not fitted on. This repository has no
  area-matched control geometry.
* **Normal representation N0 ONLY.** Face normals are all the asset and loader contract carry. The
  N1 interpolated and N2 smooth arms are `UNSUPPORTED`, which is a statement about the asset, not a
  null experimental result.

Each row of the block carries its own boundary in the JSON record. "Geometry intersection VERIFIED"
refers to the intersection path — the independent-intersector and closed-form gates — while RC-R1's
own verdict remains `GEOMETRY_DEFECT` because its silhouette-measurement tolerances were not met at
the resolution it used. "Determinism VERIFIED" holds for one recorded source, configuration and
environment across two processes, and says nothing about other hardware or library versions.

## Closure rule

There is no R2c or R3c. A failed renderer gate is a preserved scientific result, not a backlog item
to be improved until it passes. A **Renderer Contract v2** is opened only if a separate question
justifies it, under its own preregistration — not to retry a failed v1 gate.

## Boundary against D8b

D8b frozen-policy sensitivity remains `MATERIAL_LOSS` at −48.967 pp, 95 % CI [−50.113, −47.821] pp,
closed and preserved. Every renderer artefact carries `causality_vs_d8b = NOT_TESTED`. Nothing in
this track establishes that area mismatch, shading or mesh geometry caused that loss; the track was
never an experiment about its cause.

## Where the evidence lives

| Layer | Path |
| --- | --- |
| First lineage, negative verdicts included | [`results/renderer_characterization_2026-09-13/`](../results/renderer_characterization_2026-09-13/README.md) |
| Follow-ups R1b / R2b / R3b / R5b | [R1b](../results/renderer_characterization_r1b_2026-09-14/README.md) · [R2b](../results/renderer_characterization_r2b_2026-09-14/README.md) · [R3b](../results/renderer_characterization_r3b_2026-09-14/README.md) · [R5b](../results/renderer_characterization_r5b_2026-09-14/README.md) |
| Contract | [contract](renderer_contract_v1_2026-09-14.md) · [measurement](renderer_measurement_contract_v1.md) · [buffers](renderer_buffer_contract_v1.json) |
| Figures and page | [page](status/renderer-contract-v1.html) · [assets](assets/paper/renderer-contract-v1-2026-09-14/manifest.json) |

## What follows this track

Renderer work is a side methodological branch and does not continue as the main line. The next
research is task-level failure structure — search/acquisition/reacquisition diagnostics and terminal
close-approach forensics — under their own preregistrations, as evaluation instrumentation before
any architecture or training decision.
