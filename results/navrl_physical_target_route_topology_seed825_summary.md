# Physical-target route-equivalent obstacle-inflation raster topology

This is a tracked receipt for the seed-825 CPU topology CSV. It is evaluation-only geometry
evidence; it is not a simulator, policy, training, or exact route-planner pass.

Recovered generating command (the original untracked output is stored here with LF line endings):

```bash
/home/fair/miniconda3/envs/aerialgym/bin/python \
  tools/analyze_navrl_density_feasibility.py \
  --densities 70 150 205 300 \
  --margins 0.517 --trials 20 --resolution 0.15 --seed 825 \
  --csv-out results/navrl_physical_target_route_topology_seed825.csv
```

The analyzer defines obstacle inflation as a level 0.28 m box half-width (`0.14 m`) plus
`side_clearance_m`. Thus `0.14 + 0.517 = 0.657000 m`. The routed target uses all-orientation
support plus tracking reserve: `0.2068816 + 0.45 = 0.6568816 m`. The raster probe is therefore
0.118 mm more conservative in **obstacle inflation** due to the three-decimal command value.

| bars | largest component | random-pair connectivity | arena crossing | free area |
|---:|---:|---:|---:|---:|
| 70 | 0.999914 | 0.999828 | 1.0 | 0.800533 |
| 150 | 0.996561 | 0.993164 | 1.0 | 0.596434 |
| 205 | 0.957675 | 0.921124 | 1.0 | 0.471755 |
| 300 | 0.399788 | 0.224651 | 0.3 | 0.306514 |

## Claim boundary

“Route-equivalent” applies only to obstacle inflation. The analyzer uses a 0.15 m raster with
floor/ceil inclusive obstacle stamps, permits diagonal component connections without the route
planner's no-corner-cut rule, and uses the full arena rather than excluding the routed contract's
1.25 m boundary margin plus target support. It also does not run exact continuous LOS smoothing.
Therefore these values are contextual topology evidence, not exact planner reachability.

The 300-bar sampled raster free space is strongly fragmented, so arena-wide roaming at 300 bars
must not be claimed. The separate route benchmark's 16/16 per-density result means only that a
same-component goal was found in those layouts.

The historical `333/333` reachability result asks a different question: in selected seed-167,
205-bar episodes that ended in bar contact, a spawn-to-final-target path existed in a centre-disk
oracle at radii 0.40/0.65/0.766 m. It does not measure global/random-pair or 300-bar connectivity
and does not use actual per-bar AABBs.
