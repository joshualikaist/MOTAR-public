# Routed physical-target simulator gate — seed 827, attempt 2

This is a **32-cell integrity-PASS diagnostic**, not a PPO result. The exact route-off/on ×
0.6/0.9/1.2/1.5 m/s × 70/150/205/300-bar grid was recorded with 32 environments per cell.
The receipt verdicts are `PASS_32_CELL_INTEGRITY`, `FAIL_ROUTE_MECHANISM`,
`FAIL_DENSITY_CONDITIONED_ENVELOPE`, and `BLOCKED_PHYSICAL_TRAINING`. The claim boundary in the
JSON explicitly keeps `ppo_policy_loaded: false`, `hardware_validation: false`, and
`arena_wide_connectivity_300_claim: false`.

## Compact route-on/off grid

Each entry is `route-off / route-on`; `P` is the cell-level strict gate result and `F` is fail.
The route-on arm is not a route-mechanism pass anywhere in this grid.

| bars \ target speed | 0.6 m/s | 0.9 m/s | 1.2 m/s | 1.5 m/s |
|---:|:---:|:---:|:---:|:---:|
| 70 | P / F | P / F | F / F | F / F |
| 150 | P / F | P / F | P / F | F / F |
| 205 | P / F | P / F | F / F | F / F |
| 300 | P / F | F / F | F / F | F / F |

The route-off arm is a neutral-pursuer feasibility baseline, not a learned-policy evaluation. Its
bounded local-step infeasibility ranges from 0.094% to 4.250%; route-on local invalidation ranges
from 0.125% to 0.635%, but route fallback is 32.6%–85.0%. Route-on plan success is only
3.846%–17.330% across cells. These are controller/planner diagnostics, not capture rates.

For the four routed 70-bar cells pooled across all speeds, `receipt.json` records plan success and
fallback below. Goal completion is the separately preregistered 70 bars × 0.6 m/s cell:

| metric | observed | preregistered gate |
|---|---:|---:|
| plan success fraction, 70-bar 4-speed pool | 0.1454668471 | ≥ 0.99 |
| fallback interval fraction, 70-bar 4-speed pool | 0.359296875 | ≤ 0.01 |
| goal completions / env, 70 bars × 0.6 m/s | 0.25 | ≥ 0.5 |
| same-goal reselection count, all routed cells | 0 | 0 required |

The same-goal guard is therefore not the cause. The route status counters across all 16 route-on
cells are dominated by `unsafe_start` (420 observations, versus 80 `ok`; only 6 `no_path` and 6
`local_step_infeasible`). The evidence is consistent with a soft-envelope recovery deadlock:
small local invalidation events leave the target in an unsafe start state, after which recovery
falls back repeatedly. This is the current engineering diagnosis; it is not a claim that the
arena has no global route. Motor saturation, tilt, and contact are not the primary failure mode in
this decomposition; tracking remains a separately gated metric. The route-off arm has the expected low controller telemetry
at its passing cells, while route-on fails before those route guarantees can be interpreted.

## Verdict and next authority

Do not start physical PPO, relax a preregistered threshold, or call 300 bars arena-wide connected.
Do not convert this diagnostic into hardware or sim-to-real evidence. First instrument safe-prefix /
full-horizon availability and repair the `unsafe_start` recovery path in a new, explicitly
preregistered engineering lineage; then rerun the unchanged 32-cell gate.

## Artifact provenance

| artifact | SHA-256 |
|---|---|
| `summary.json` | `e5e4560464dc3a2080d904c2f8d2247e0c65e671dd63ea08d1b507ec65fc7197` |
| `execution_manifest.json` | `896b05c9bf2cea672aad5e95a8e2a893d6eaa6999c56d5ab8ab60fc4c0de4291` |
| `source_manifest.json` | `9af2f58b42176c4800cf5f8795dbf8f009097d3ae5f74c9ef16937730d6a1aad` |
| `receipt.json` | `bd840ef6dd157aa317dfeccfbf347235ac858d541e1e35e9e50c130084fc7771` |
| evaluator source (receipt) | `fe9740bc09d49cdc1705f49b634cd10f223b4984a198390533962423538e0025` |
| target-route planner (source/receipt) | `7fec3015e5dee667b8cd64d145d29b9244c18eb0c79b4af12020be44b503cb83` |
| ref5in URDF (source/receipt) | `5c160b0d19caebf9a4a3c38be861a77637ee0fb2b80febf4ac54d8b143db6a32` |

Raw JSON remains authoritative: [`summary.json`](summary.json),
[`execution_manifest.json`](execution_manifest.json), [`source_manifest.json`](source_manifest.json),
[`receipt.json`](receipt.json), and the 32 files under [`children/`](children/).
