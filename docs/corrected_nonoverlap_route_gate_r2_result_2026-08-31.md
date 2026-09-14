# Corrected non-overlap route/physical gate r2 result

Frozen result date: 2026-08-31 (Asia/Seoul)

Preregistration: `docs/preregistration_corrected_nonoverlap_route_gate_r2_2026-08-31.md`

Raw result: `results/navrl_corrected_nonoverlap_route_gate_r2_seed829/summary.json`

## Decision

`PASS_32_CELL_INTEGRITY / FAIL_ROUTE_MECHANISM / BLOCKED_PHYSICAL_TRAINING`

The corrected, footprint-aware non-overlap environment does require a fresh PPO lineage. However,
the preregistered environment-side route/physical mechanism did not pass the gate that must precede
PPO. Therefore the 500-epoch smoke and the 70→205 long run were **not started**. No historical
checkpoint or historical capture curve is reused as corrected-environment performance.

## Frozen gate results

| Item | Measured | Frozen gate | Decision |
|---|---:|---:|---|
| execution/source/import integrity | 32/32 cells | exactly 32/32 | PASS |
| 70-bar routed plan success | 17.7831% | ≥99% | FAIL |
| 70-bar routed fallback intervals | 30.0156% | ≤1% | FAIL |
| 70 bars × 0.6 m/s goals/env | 0.21875 | ≥0.5 | FAIL |
| same-goal reselections | 0 | 0 | PASS |
| routed 1.5 m/s cells passing | 0/4 | 4/4 | FAIL |
| highest authorized speed by density | none at 70/115/160/205 | at least one passing routed speed | FAIL |

The routed mean-speed ratios were:

| target speed (m/s) | 70 bars | 115 bars | 160 bars | 205 bars |
|---:|---:|---:|---:|---:|
| 0.6 | 0.7882 | 0.6206 | 0.5604 | 0.5298 |
| 0.9 | 0.6713 | 0.4702 | 0.3755 | 0.3226 |
| 1.2 | 0.6336 | 0.4795 | 0.3893 | 0.3325 |
| 1.5 | 0.6432 | 0.4303 | 0.3895 | 0.2292 |

Contact, motor saturation, tilt and finite-state checks were generally within their frozen bounds.
The dominant routed end state was `unsafe_start`, not bar contact or motor saturation: for example,
70 bars × 0.6 m/s ended with 19 valid and 13 `unsafe_start` environments; 205 bars × 1.5 m/s ended
with 31 `unsafe_start`, zero valid, and one local-step infeasible environment. The physical target's
inertia can carry it out of the route planner's soft safe set; the fail-closed route state then
submits zero velocity and repeatedly replans from `unsafe_start`. This is an environment/controller
mechanism failure, not a learned-policy result—the pursuer action was neutral and no PPO policy was
loaded.

## Revision-1 execution defect and fix

Revision 1 is preserved separately as `VOID_EXECUTION` at
`results/navrl_corrected_nonoverlap_route_gate_seed829/`. It stopped after 12/32 cells because one
physical-target reset exhausted 1,024 scalar proposals. The retry did not relax geometry or gates:
it changed rejection sampling to retain the first accepted iid proposal from target 64×256 and
pursuer 64×128 proposal budgets. Pursuer exhaustion also changed from an unchecked silent fallback
to fail-closed. Revision 2 then completed all 32 cells.

## Claim and execution boundary

- Supported: the corrected non-overlap geometry is implemented; the exact 32-cell route diagnostic
  completed; its current target route/controller mechanism failed the preregistered gate.
- Not supported: corrected-environment capture/crash/timeout, 205-bar PPO mastery, successful physical
  target routing, hardware validation or sim-to-real performance.
- Not authorized in this execution: threshold relaxation, warm-start, 500-epoch smoke, long PPO,
  300-bar training or presenting integrity PASS as mechanism PASS.
- Next valid software step: design and independently preregister a target trajectory/controller that
  preserves a braking-aware safe-state invariant, then rerun a new mechanism gate before PPO.

## Integrity hashes

- summary SHA-256: `4a464521015c110805d46392b0355297c7328ac5b006be368f6b9a23dcf389d7`
- receipt SHA-256: `74dfcbe10eeb39ea7bdd060e115ec460af58992b07949faed2382c3f9ec2ce95`
- execution manifest SHA-256: `15355dd45c9d73fd8d4707f7c4c50f7127a62ed6aabf50911ffef0a7eb850af2`
