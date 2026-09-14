# Preregistration — corrected non-overlap physical route-off 70→205 curriculum

Frozen: 2026-09-01, after the preregistered seed-907 smoke returned
`PASS_LEARNING_VIABILITY` and before this curriculum produced any result.

## Question and claim boundary

How far can a fresh PPO policy progress through the corrected non-overlap density curriculum when
the physical target uses the independently passing local route-off motion envelope?

This is the current **route-off baseline**, not a successful global-route method. It cannot support
claims about braking-v3, 1.5 m/s target motion, hardware validation or sim-to-real transfer.

## Frozen execution tuple

| item | value |
|---|---|
| launcher | `train_navrl_corrected_nonoverlap_physical_curriculum.sh` |
| initialization | fresh; no smoke checkpoint, warm-start or resume |
| seed / budget / envs | 911 / 30,000 epochs / 128 |
| robot/target | `navrl_ref5in_v2_quad`; physical 6-DoF; route off; mixed CV/waypoint |
| target speed | `U[0.3,1.25] m/s`, one-epoch ramp |
| arena/placement | 40×40×3 m; `footprint_clearance`; surface 0.45 m; fallback/merge 0 |
| density | 70→205, step 15; minimum dwell 1,000 epochs per density |
| evidence window | 16,384 completed episodes |
| promotion schedule | 70:0.82, 85:0.77, 100:0.72, 115 and above:0.70 |
| perception/action | canonical v2 cluster-sector Transformer; squashed Gaussian |
| learning rate / governor | `1.5e-5` / off |
| provenance | clean runtime source receipt and exact import-root guard |

Seed 911 is independent of the engineering-smoke seed 907. The 500-epoch checkpoint is not reused.
The target-speed cap, optimizer setup and every environment input are identical to the passing
smoke except that density is now curriculum-owned and the budget is longer.

## Frozen interpretation

1. Density promotion uses only the schedule above. Thresholds, dwell, evidence size and density
   step may not be changed after observing a stall.
2. A same-density capture-guard stop is a valid stopped result, not permission to resume, lower a
   gate or jump density.
3. A clean max-epoch finish does not itself prove 205-bar performance. Final claims require a
   separately preregistered held-out density evaluation of the terminal `last_gen_ppo` checkpoint.
4. `gen_ppo.pth` is not the terminal curriculum policy and must not be used for the density curve.
5. 220/250 are later OOD evaluation cells. 300 is disconnected stress only and is not trained.
6. Any source/import/checkpoint mismatch or non-finite state is `VOID_EXECUTION`.

## Follow-up authority

- The run may proceed once from clean committed runtime bytes.
- Completion authorizes only offline run analysis and a new held-out evaluation preregistration.
- It never authorizes routed PPO, parameter search, hardware flight or sim-to-real claims.
