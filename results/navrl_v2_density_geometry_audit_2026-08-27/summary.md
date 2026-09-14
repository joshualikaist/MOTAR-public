# Corrected-v2 density geometry audit — canonical 6–28 m

Date: 2026-08-27. CPU only. No Isaac Gym, no PPO.

Receipt: [`density_geometry_canonical_6_28.json`](density_geometry_canonical_6_28.json)
SHA-256 `6b1f1b36cf73409d0c09483c3e1767b7ff196aecf0b95769f0e07d8dffa268d5`.
Schema `navrl.v2-density-geometry-audit.v1`. 60 layouts × 128 pairs per density.
Runtime 1400.5 s.

Gates were frozen in `VERIFICATION.md` **before** this file existed: random-pair
connectivity ≥ 95%, no-route ≤ 5%, generation failures 0. The binding inflation is
**body + tracking reserve**, not a point robot.

## Binding result (`body_plus_tracking`, inflation 0.650 m)

| bars | connectivity | no-route | gen fail | gate |
|---:|---:|---:|---:|---|
| 64 | 99.961% | 0.039% | 0 | PASS |
| 70 | 99.961% | 0.039% | 0 | PASS |
| 100 | 99.961% | 0.039% | 0 | PASS |
| 130 | 99.948% | 0.052% | 0 | PASS |
| 160 | 99.753% | 0.247% | 0 | PASS |
| 190 | 99.440% | 0.560% | 0 | PASS |
| **205** | **99.167%** | **0.833%** | **0** | **PASS** |
| 220 | 98.945% | 1.055% | 0 | PASS |
| 250 | 97.813% | 2.187% | 0 | PASS |
| **300** | **94.661%** | **5.339%** | **0** | **FAIL** |

Highest passing density under the frozen gates: **250 bars**.
Training cap stays **205** (pre-committed; 220/250 passing does not raise it).
300 bars is geometrically disconnected on this inflation, so it is not a connected
OOD cell.

`body_only` (inflation 0.200 m) passes every audited density including 300. That
arm is diagnostic. A yaw-invariant body disk without tracking reserve is not the
task's support.

## What this does and does not authorize

- It answers the open claim “205 is only a YOPOv2 density analogy.” Under
  `footprint_clearance` layouts and canonical 6–28 m start/goal sampling, 205 is
  connected.
- It does **not** authorize PPO, 500-epoch smoke, Track A GPU, or Track B reruns.
- Hard goal band 22.5–28 m is not in this receipt.
- Spawn still uses 0.65 m bar-**center** clearance. At 205 bars,
  `start_in_inflated_obstacle_frac` is 27.45% before the 0.6 m snap used by the
  connectivity test. That is a spawn/inflation mismatch, not a generation failure.

## Provenance notes

Layouts come from the real `AssetManager._footprint_clearance_xy_spacing` and the
`bars_h3` URDF pool. Inflation uses the documented prop-tip AABB 0.2825634 m
(`0.5 * span * √2` = 0.19980 m) plus `physical_tracking_margin` 0.45 m.
The fresh `quad_navrl_ref5in_v2.urdf` collision box is 0.283 m, +0.31 mm of
circumradius. That delta cannot move 205 from 99.17% onto the 95% knife-edge, and
it cannot rescue 300.

Tool: `tools/audit_navrl_v2_density_geometry.py --band canonical_6_28`.
Density contract: [`docs/preregistration_navrl_v2_corrected_density_geometry_2026-08-27.md`](../../docs/preregistration_navrl_v2_corrected_density_geometry_2026-08-27.md).
