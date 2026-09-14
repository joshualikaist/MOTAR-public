# Legacy seed167 topology labels — exploratory summary

## Verdict

**Exploratory only.** All 1,989 dumped episodes have a static 2-D path and none trigger the local
cul-de-sac proxy under the assumed geometry. This supports only the narrow statement that static
grid disconnection was not observed among the episodes present in the dump. It does **not** show
that timeout episodes were connected or free of dead ends.

## Coverage and outcome contract

The original frozen-policy evaluation contains 2,049 episodes, but the legacy dump writer records
only `captured | crashed_out`. Its 60 timeout episodes are therefore absent.

| recorded outcome | n | path exists | local cul-de-sac proxy |
|---|---:|---:|---:|
| capture | 1,641 | 100% | 0% |
| bar contact | 333 | 100% | 0% |
| below | 10 | 100% | 0% |
| out of bounds | 5 | 100% | 0% |
| **recorded total** | **1,989** | **100%** | **0%** |
| timeout | **0 present / 60 actual** | **not measured** | **not measured** |

Consequently, this analysis cannot answer whether timeout was caused by a local dead end and must
not be cited for any timeout/dead-end conclusion.

## Capture versus bar-contact descriptives

| topology label | capture, n=1,641 | bar contact, n=333 | capture − contact |
|---|---:|---:|---:|
| mean shortest-path detour ratio | 1.0692 | 1.0710 | −0.0018 |
| mean minimum usable side-clearance | 0.2594 m | 0.2319 m | +0.0275 m |
| mean obstacles within 12 m | 48.79 | 45.92 | +2.88 |
| mean clusters within 12 m | 45.92 | 43.15 | +2.76 |

The detour distributions are nearly identical at the mean. Bar-contact episodes have 2.75 cm
less mean usable clearance on the selected grid-shortest path, while their start-local obstacle
and cluster counts are lower. These are descriptive outcome associations only: the groups were
not randomised by topology, and the values do not establish clearance, count, or clustering as a
cause of contact.

## Geometry assumptions

- 205 bars in every layout; each is treated as a **0.60 × 0.60 m square**.
- Actual per-bar footprints were not saved and the asset pool spans 0.4–0.8 m. Therefore all rows
  carry `bar_size_source=assumed_default`; they are not publication-exact geometry.
- Grid resolution 0.10 m; vehicle half-width 0.14 m; requested side-clearance 0.20 m; total
  obstacle inflation 0.34 m.
- Sensor range 12.0 m; deployed representation cluster surface-gap contract 0.45 m; endpoint snap
  radius 0.60 m.
- Static 2-D labels omit vehicle dynamics, target motion, stopping distance, and the 600-step
  episode horizon.

## Provenance and reproduction

- Layout dump: `results/navrl_v2_bar_ceiling/episodes_seed167.npz`
  - SHA-256: `c509c2fa79a245a6ea34376651e76051e5ccfa9f149b01f9e79184c7d1ac9cb8`
- Untracked per-layout output:
  `results/navrl_v2_bar_ceiling_topology_assumed0p60_all1989.json`
  - SHA-256: `6ca8c40599d51d9c9547de18a7528a1b3f73559916684455a2d0aa6c85973ac8`
- Original evaluation: `results/navrl_v2_bar_ceiling/instrumented/205bars.json`
- Topology tool contract: commit `8a2eaf3`

```bash
/home/fair/miniconda3/envs/aerialgym/bin/python \
  tools/analyze_navrl_topology_labels.py \
  /home/fair/workspaces/aerial_gym_ws/src/aerial_gym_simulator/results/navrl_v2_bar_ceiling/episodes_seed167.npz \
  --default-bar-size-m 0.60 \
  --output results/navrl_v2_bar_ceiling_topology_assumed0p60_all1989.json
```

The 2.9 MB per-layout output is intentionally left untracked. This compact summary is the
versioned evidence index; exact future analysis requires a new evaluation snapshot containing the
actual `bars_size_xy` and all termination types, including timeout.
