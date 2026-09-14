# Visual research overview: source and migration map

## D8b archival closure follow-up

After the presentation revisions described below, D8-A recorded `TECHNICAL_GO` and
D8b was finalized as `MATERIAL_LOSS`; [result and provenance](../../results/dynamic_mesh_policy_sensitivity_d8b_2026-09-13/AUDIT.md).
The live manifest now describes D8-A as a technical result only (`D8A_TECHNICAL_ONLY`);
Table 3 separately reports D8b. D8c is not started and D9 remains `NOT_RUN`.
Figures 1 and 6 are labeled historical pre-D8 diagrams rather than silently rewriting
their pinned SVG/PNG/PDF/ZIP bytes. The sections below retain the state at each earlier
presentation revision; their old `NOT_STARTED` wording is not the current D8 status.
This change records existing evidence only and does not implement perception adaptation.

## Paper-page revision on `be89675`

The second presentation pass replaces the dashboard-like overview introduced by
`be89675` with a single-column research-paper page. It changes presentation and
information architecture only: no simulator, detector, policy, controller, viewer
motion, result receipt or historical evidence file is changed. The page now follows
title/authorship/abstract, numbered sections, numbered figures with captions,
experimental tables, bounded evidence, discussion and next-steps end matter.

The public page contains seven numbered figures: six static diagrams plus the existing
interactive WebGL arena as Figure 2. The dated figure package still contains seven
static diagrams because the observation/Transformer diagram remains a supplementary
download; it is not silently discarded. Static diagrams were re-exported in a shared
paper palette and without dashboard cards, gradients or shadows. D6 remains
`INCONCLUSIVE`; D7 `GO` remains limited to shadow cost and output non-interference;
D8 remains `NOT_STARTED`; D9 remains `NOT_RUN`.

The first-pass site remains recoverable from Git history. The pre-redesign detailed
page is still preserved byte-for-byte as
[archive-2026-09-13.html](archive-2026-09-13.html), with the SHA-256 pinned by tests.
The redesign does not turn that live-resource archive into a standalone executable
snapshot.

## First presentation pass

This is a presentation-only reorganization based on source `52b5dd8`. No experiment,
detector, policy, controller, geometry or viewer motion implementation changed.
The new overview is `index.html`; the complete previous HTML is preserved in
[archive-2026-09-13.html](archive-2026-09-13.html). Its body is historical, while shared
viewer assets and the manifest loader remain live resources. It is not a frozen executable bundle.

## Detail mapping

| Previous detail | Current overview | Preserved source |
|---|---|---|
| P3–P10 / S4 metrics | Perception + bounded summary | [Results overview](../results_overview_2026-09-12.md), [verification](../../VERIFICATION.md), [old perception section](archive-2026-09-13.html#perception) |
| E3-S / E3-P | Evidence range card | [E3-S](../../results/eth_ds5_e3s_2026-09-10/README.md), [E3-P](../../results/eth_ds5_e3p_reliability_2026-09-10/README.md) |
| Filter comparison and numerical outcomes | Algorithms / safety figure | [Old filter section](archive-2026-09-13.html#safety-filter), [verification](../../VERIFICATION.md) |
| Routed preview and lineage explanation | Environment technical details | [Old arena](archive-2026-09-13.html#arena), [system specification](../MOTAR_SYSTEM_SPEC_2026-08-24.md) |
| Appearance R1–R5 and D5 audit | Appearance figure / manifest | [Old appearance section](archive-2026-09-13.html#appearance), [probe audit](../../results/target_appearance_in_sim_2026-09-12/AUDIT.md) |
| D6 / D7 | Evidence manifest | [D6](../../results/dynamic_mesh_raycast_feasibility_2026-09-12/README.md), [D7](../../results/dynamic_mesh_integrated_cost_2026-09-12/README.md) |
| Licensing / size / access | Three external data cards | [Third-party terms](../../THIRD_PARTY_LICENSES.md) |
| Old figures and presentation ZIPs | Historical gallery links | [Paper gallery](../assets/paper/index.html), [result overview](../results_overview_2026-09-12.md) |
| Execution authority / unfinished learning | Detail links, no new run claims | [Authority](../research_authority_2026-08-26.json), [worklog](../../WORKLOG.md) |

Historical evidence assertions now inspect the archived detail page. The new overview
has separate active-page structural, local-link, scope, viewer and manifest tests.
This moves assertions with the content rather than removing negative-outcome checks.

## Parameters

Numbers describe explicit sources, not one universally active configuration. No
Isaac Gym imports or configuration execution were needed for these read-only checks.

| Display | Source at `52b5dd8` | Interpretation |
|---|---|---|
| Camera 160×90, HFOV 87° | [task config](../../aerial_gym/config/task_config/navrl_task_config.py) `camera_width`, `camera_height`, `detector_hfov_deg` | Defaults; runtime overrides may differ |
| LiDAR 36×4 / 4 m default | [LiDAR config](../../aerial_gym/config/sensor_config/lidar_config/navrl_lidar_config.py) | Environment-variable defaults, NOT the 72×4 / 12 m canonical run |
| Canonical scan 72×4 / 12 m; 898-D / 17 tokens | [System specification §4](../MOTAR_SYSTEM_SPEC_2026-08-24.md) | Historical observation contract; not claimed for every mode |
| Five policy history samples; selector T16 | [task config](../../aerial_gym/config/task_config/navrl_task_config.py), [streaming interface](../specs/perception_streaming_v1.md) | Different models / histories |
| Arena 40×40×3, density lineage 70–205 | [System specification](../MOTAR_SYSTEM_SPEC_2026-08-24.md), [environment config](../../aerial_gym/config/env_config/navrl_bars_env.py) | Named research geometry, not all historical layouts |
| ref5in 1.20 kg / motor lag 0.04 s / tilt 45° | [System specification §2](../MOTAR_SYSTEM_SPEC_2026-08-24.md), [robot config](../../aerial_gym/config/robot_config/navrl_ref5in_quad_config.py) | Simulated candidate, not real hardware validation |
| Velocity 2.0 m/s default versus historical 2.5 | [task config](../../aerial_gym/config/task_config/navrl_task_config.py) `max_velocity`, [specification](../MOTAR_SYSTEM_SPEC_2026-08-24.md) | Per-axis command setting, not measured flight speed |
| Physics dt 0.01 s | [base simulation config](../../aerial_gym/config/sim_config/base_sim_config.py) | 100 Hz configuration; viewer clock is 10 Hz and not PhysX |
| 256 default environments versus D7 128 | [task config](../../aerial_gym/config/task_config/navrl_task_config.py), [D7 receipt report](../../results/dynamic_mesh_integrated_cost_2026-09-12/README.md) | Default and experiment cell are distinguished |
| Browser target 0.3–1.5 m/s | [arena motion](arena_motion.js) | Illustration contract only; individual experiments use their receipts |

## Manifest migration: schema 1 → 2

The old public manifest used D6 for a proposed generic full-loop benchmark and D7/D8
for integration/shortcut work. That naming lagged the actual Track D record.
Schema 2 adopts [the recorded D1–D9 sequence](../track_d_state_2026-09-12.md):
D6 mesh feasibility `INCONCLUSIVE`, D7 shadow cost `GO`, D8 integration `NOT_STARTED`,
D9 shortcut remeasurement `NOT_RUN`. The generic box/background benchmark remains a
separate [public tool](../renderer_public_tools.md); it is not the D6 target-mesh result.
`SHADOW_COST_ONLY` replaces the stale global cost-unmeasured flag. It does not claim
that end-to-end mesh-derived perception cost or performance has been measured.

Research status and assistance scope are distinct: old `BLOCKED_BY_POLICY` was a
boundary on implementation assistance, not a measured experiment result. No D8
target-identification improvement, policy adaptation or engagement experiment is
implemented or designed by this documentation change.

## Figure contracts

New assets live only in `docs/assets/paper/overview-2026-09-13/`. SVG is authoritative;
PNG and vector PDF are derived with Chrome. Rebuild from the repository root:

```bash
python tools/render_research_overview.py --export
```

Requires Chrome and websocket-client. The SVG generator imports documentation
helpers only. Existing `docs/assets/paper/manifest.json`, the old ZIP, its figures,
and historical result packages are not rewritten. The dated new manifest hashes
the 21 new diagram artifacts; the new ZIP contains those artifacts and its manifest.

## Interpretive corrections

- The real-image API emits frame-local rank/no-selection, not persistent identity
  or an independently validated metric state. The policy link is dashed.
- D7 records shadow cost and non-interference only; D6 stays INCONCLUSIVE.
- E3-S 6.2% is the median of block medians of absolute relative range error using
  a dark-pixel size proxy in one previously explored flight. It is not GT-box
  accuracy, an independent-flight test or causal attitude decomposition. No temporal
  embargo was applied at block boundaries; the result README retains that limitation.
- P10 net adaptation benefit is not established. The historical C3 withdrawn
  claim, E3-P reliability failure and R4/R4b criterion failure remain preserved.
- D8 prerequisites remain unresolved documentation/authority boundaries. This
  site update neither selects a new geometry contract nor opens new experiments.
