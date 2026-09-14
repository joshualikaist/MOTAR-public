# Physical-target global route CPU benchmark

Verdict: **PASS_CPU_ENGINEERING_GATE**. This is not a simulator, policy, or training pass.

Command:

```bash
/home/fair/miniconda3/envs/aerialgym/bin/python \
  tools/benchmark_navrl_target_route_planner.py \
  --trials 16 --seed 825 \
  --output results/navrl_target_route_cpu_benchmark_seed825/summary.json
```

Contract: actual per-asset `bars_h3` AABBs, deterministic `navrl_band` placement,
0.25 m route grid, 0.2069 m all-orientation target support (exact 0.2068816 m), 0.45 m tracking reserve,
1.25 m boundary reserve, same-component goal at least 6 m away. Each row contains 16
deterministic layouts planned sequentially on CPU.

| bars | ok | mean ms/env | p95 ms/env | max ms/env | serial 128-env projection |
|---:|---:|---:|---:|---:|---:|
| 70 | 16/16 | 46.18 | 57.08 | 63.08 | 5.91 s |
| 150 | 16/16 | 48.71 | 65.67 | 77.61 | 6.24 s |
| 205 | 16/16 | 58.34 | 87.32 | 123.94 | 7.47 s |
| 300 | 16/16 | 64.01 | 115.52 | 137.01 | 8.19 s |

All preregistered limits pass: mean ≤100 ms/env, p95 ≤150 ms/env, projected serial
128-env reset ≤10 s, and 70-bar availability ≥90%. The projection deliberately assumes no
parallel speedup and is not a measured synchronous Isaac Gym reset stall.

## Claim boundary

The benchmark chooses a new goal in the start's connected component, so 16/16 only means it
found a local connected destination in these layouts. It does **not** show arena-wide
connectivity. An independent 20-layout route-equivalent **obstacle-inflation raster probe** found 300-bar
largest-component and sampled-pair connectivity of 0.3998 and 0.2247, with arena crossing in
only 0.3 of layouts: arena-wide roaming at 300 bars is prohibited as a
claim. Its 0.15 m grid allows diagonal corner connections and omits the route's
1.25 m + support boundary exclusion, so it is contextual evidence rather than an exact route
reachability certificate. GPU/PhysX tracking, contact rate, reset stalls, and policy effects remain
unmeasured.

Machine-readable values and the frozen gate verdict are in `summary.json`; the preregistration is
[`docs/preregistration_physical_target_global_route_2026-08-25.md`](../../docs/preregistration_physical_target_global_route_2026-08-25.md).
