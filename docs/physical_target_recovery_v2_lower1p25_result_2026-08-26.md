# Recovery-v2 lower-1.25 physical-target gate result — seed 827

The separately preregistered `baseline_1p25` recovery-v2 32-cell gate completed and verified.
Integrity passed; the route mechanism failed. This is a different FAIL from v1 attempt 2. It
does not repair the canonical 1.5 m/s contract, authorize PPO, or change the frozen `0.45 m`
margin, gain 2.5, or 32-env cell size.

| Diagnostic | Result |
|---|---:|
| Integrity | `PASS_32_CELL_INTEGRITY` |
| Route mechanism | `FAIL_ROUTE_MECHANISM` |
| Cells | **7 / 32** pass, all route-off; recovery **0 / 16** |
| 70-bar pool plan success | `190 / 203` = **93.60%** (gate ≥99%) |
| 70-bar pool fallback | `18381 / 38400` = **47.87%** (gate ≤1%) |
| 70 bars × 0.6 m/s goals/env | `7 / 32` = **0.21875** (gate ≥0.5) |
| Recovery occupancy in `NO_CONNECTOR` | `96854 / 153600` = **63.06%** |
| Recovery occupancy in `CONNECT` | `1871 / 153600` = **1.22%** |
| Hard-breach `NO_CONNECTOR` entries | **0 / 534** |

Source commit `2b151d9a4c4fe078ecc027152e5642fa857a2e2f`. Bound lower-contract braking receipt
`results/navrl_physical_target_braking_lower1p25_headingrest_seed827/`, SHA-256
`4e87eb9ddf5dd9cea1fc0354d272a5d18ec6a05427e0f41e672749a57df9047a`.

## Cell grid

`off / recovery-v2` at speeds `{0.6, 0.9, 1.2, 1.25}` and bars `{70, 150, 205, 300}`.

| bars \ speed | 0.6 | 0.9 | 1.2 | 1.25 |
|---:|:---:|:---:|:---:|:---:|
| 70 | off P / on F | off P / on F | off F / on F | off F / on F |
| 150 | off P / on F | off P / on F | off F / on F | off F / on F |
| 205 | off P / on F | off P / on F | off F / on F | off F / on F |
| 300 | off P / on F | off F / on F | off F / on F | off F / on F |

Off-arm 1.2 and 1.25 fail `local_feasibility` (300 also contact/speed). Recovery-arm cells all
fail `speed`; all but `1.25/150` also fail `connect_actual_progress`. Watchdog fails except some
0.6 cells.

## Why this is not v1 `RECOVERY_DOMINANT`

v1 attempt 2 pooled 70-bar plan success was 14.55% with replans dominated by `unsafe_start`.
Recovery-v2 reached 93.60% plan success and compressed legal `BRAKE → CONNECT → ROUTE`
handoffs. The remaining mechanism failure is occupancy in latched `NO_CONNECTOR` (zero planar
command), which drives fallback and realized-speed collapse.

Packed-telemetry classes for the 534 `NO_CONNECTOR` entries (descriptive; not new status codes):

| Class | Count | Share |
|---|---:|---:|
| same-interval BRAKE then no-anchor | 219 | 41.0% |
| BRAKE timeout | 136 | 25.5% |
| BRAKE, still in-phase, no-anchor | 93 | 17.4% |
| CONNECT, soft-free, failed resume replan | 50 | 9.4% |
| CONNECT, still soft-unsafe, failed certificate | 29 | 5.4% |
| ROUTE → `NO_CONNECTOR` | 5 | 0.9% |
| CONNECT timeout | 2 | 0.4% |
| hard breach | 0 | 0.0% |

Same-interval no-anchor grows with speed (0 at 0.6/70; 23 of 35 entries at 1.25/70). At 0.6 the
dominant packed classes are in-phase BRAKE no-anchor and CONNECT failed resume.

## CONNECT tracking on the same frozen raw

The rest-heading snap (`HEADING_VALID_SPEED_MPS = 0.10`) did its job: in every recovery cell,
when `|v| < 0.10` m/s the submitted CONNECT command points at the frozen anchor (fraction 1.0).
Candidate index 0 is always selected. Certified 1 s horizons stay positive.

Actual PhysX intervals still recede. On 0.6/70: 59/176 CONNECT intervals increased anchor
distance (max +5.3 cm) while rest-start realized speed was 0.055 m/s against a 0.41 m/s
command and a 4 m/s² × 0.1 s envelope of 0.40 m/s. CONNECT is only 1.22% of recovery-arm time.
It explains the `connect_actual_progress` cell gate, not the 47.87% fallback.

An earlier WORKLOG claim that “the actual PhysX interval reduced anchor distance” was false
when rechecked on VOID and finalized raw.

## Claim boundaries

- Do not call 1.25 a 1.5 success.
- Do not combine VOID/incomplete bundles with this receipt.
- Do not retune gain 2.5, `0.45 m`, or env count after seeing FAIL.
- Do not rerun the 32-cell gate.
- Raising env count above the preregistered 32 is a different experiment.

## Next authority

The evaluation-only no-anchor geometry
[probe](physical_target_recovery_v2_no_connector_forensics_result_2026-08-26.md) completed and
verified with primary `n=1`, identity agreement, and verdict `INCONCLUSIVE`. It does not alter
this 32-cell FAIL and creates no authority for a cell/grid rerun, retuning, PPO, 1.5 m/s, or an
environment-count change.

Evidence: [gate summary](../results/navrl_physical_target_recovery_v2_gate_lower1p25_seed827/summary.json),
[receipt](../results/navrl_physical_target_recovery_v2_gate_lower1p25_seed827/receipt.json),
[gate preregistration](preregistration_physical_target_recovery_v2_lower1p25_gate_2026-08-26.md),
[packed diagnostic](../tools/diagnose_navrl_physical_target_recovery_v2_packed.py).
