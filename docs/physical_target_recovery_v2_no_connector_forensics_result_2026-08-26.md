# Recovery-v2 70-bar no-anchor forensics result — seed 827

The preregistered four-cell evaluation-only GPU probe completed and independently verified.
The descriptive verdict is **`INCONCLUSIVE`**: only one event entered the frozen primary
denominator, below the required `n >= 20`. This result does not alter the verified 32-cell
`FAIL_ROUTE_MECHANISM`.

| Diagnostic | Result |
|---|---:|
| Cells | 4 / 4 recorded: 70 bars × `{0.6, 0.9, 1.2, 1.25}` m/s |
| `NO_CONNECTOR` entries | 106 |
| Primary events | **1** |
| Replica hard-safe anchor present | 0 / 1 |
| Hard-free and soft-unsafe at latch | 1 / 1 |
| Observer identity disagreement | **0** |
| Descriptive verdict | **`INCONCLUSIVE`** |

The sole primary event occurred in the 1.2 m/s cell. Runtime
`recovery_anchor_idx` and the independent CPU replica both reported no anchor, so the
fail-closed `VOID_OBSERVER_IDENTITY` rule did not fire. Its one-sample Wilson 95% interval for
anchor absence is `[0.2065, 1.0]`; the preregistered minimum sample size prevents either
`ANCHOR_PRESENT_LATCH` or `ANCHOR_ABSENT_AT_LATCH`.

## Per-cell records

| speed (m/s) | primary n | failed certificate | BRAKE timeout | failed resume | CONNECT timeout |
|---:|---:|---:|---:|---:|---:|
| 0.60 | 0 | 3 | 2 | 10 | 0 |
| 0.90 | 0 | 8 | 12 | 5 | 1 |
| 1.20 | 1 | 15 | 10 | 4 | 0 |
| 1.25 | 0 | 23 | 8 | 4 | 0 |
| **pooled** | **1** | **49** | **32** | **23** | **1** |

The 23 positive-soft-margin CONNECT resume failures reported 22 `missing` and one
`unsafe_goal` replan status. BRAKE timeout, CONNECT-origin entries, and CONNECT timeout are
descriptive context only and do not vote in the primary rule.

## Relation to the frozen packed diagnosis

The same four 70-bar cells contained 106 packed `NO_CONNECTOR` entries, including 59
BRAKE-origin generic no-anchor labels (43 same-interval and 16 in-phase). Packed telemetry
sampled state at interval start. This observer instead sampled state immediately before the
latching call, as preregistered. Consequently, its per-class split is not expected to reproduce
the packed split; it found only one event in the frozen primary denominator. This timing
difference is not an observer-identity VOID because the runtime and replica anchor booleans
agreed on the one primary event.

The probe therefore cannot answer whether a 7×7 hard-safe connector usually already existed
at the relevant latch. It does show that, under the pre-latch observer semantics, these four
cells were dominated by CONNECT certificate/resume failures and BRAKE timeout rather than
eligible generic BRAKE no-anchor events.

## Provenance and claim boundary

- Probe source HEAD: `9a8f0d62882fafe500e6c7ed7d3fb3f3889c506b`
- Receipt SHA-256:
  `81f53f49f7897c8978a0be0d5ab8466515591c5695fd96220f28db0d5a0f63ae`
- Summary SHA-256:
  `bc5d05a3f36da9213989c84ae28eda9ad65b0ba60b55dc2bbe4b5d262a12b8ee`
- Gate artifacts remained read-only; the original evaluator remained unchanged.
- The run was bound to the lower-1.25 gate and heading-rest braking receipts.

This result is descriptive only. It does not pass the 32-cell mechanism, authorize a rerun,
retune gain 2.5 or the `0.45 m` margin, change 32 environments, authorize PPO, validate
hardware, or support a 1.5 m/s claim.

## Observer-hardening backlog (separate from this verified result)

Luna review findings are follow-up engineering hardening only: add an explicit untracked-cleanliness
attestation, enforce strict real-boolean and schema validation rather than truthy coercion, and
validate the expected `.COMPLETE` marker before accepting an observer artifact. The current artifact
remains verified because its frozen receipt, runtime/replica identity check, and independent result
verification passed. Tool bytes must not be changed in this result commit; any hardening belongs to a
separate future change and does not authorize another Track B run.

Evidence: [summary](../results/navrl_physical_target_recovery_v2_no_connector_forensics_seed827/summary.json),
[receipt](../results/navrl_physical_target_recovery_v2_no_connector_forensics_seed827/receipt.json),
[preregistration](preregistration_physical_target_recovery_v2_no_connector_forensics_2026-08-26.md),
and [32-cell result](physical_target_recovery_v2_lower1p25_result_2026-08-26.md).
