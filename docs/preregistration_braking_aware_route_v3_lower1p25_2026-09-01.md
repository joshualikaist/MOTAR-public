# Preregistration — braking-aware route v3, baseline-controller lower contract (1.25 m/s)

Date frozen: 2026-09-01 (Asia/Seoul)  
Implementation branch: `codex/braking-route-v3`  
Base: `fe734b2` (canonical 1.5 braking probe NO-GO rerun recorded)

## Why a lower contract, and what it may not reinterpret

The canonical braking-aware route v3 gate
(`docs/preregistration_braking_aware_route_v3_2026-09-01.md`, SHA-256
`cceecb9ad4a538e7bc2bc9171436e823ef18652e9c971e0d6fa8174279df6056`) requires the canonical
1.5 m/s raw braking receipt.  That receipt is deterministically unobtainable under the frozen
controller: the 1.5 m/s five-second warmup convergence gate (absolute error <= 0.05 m/s) failed on
2026-08-26 and again on 2026-09-01 with digit-identical statistics
(`final mean = 1.442577 m/s`, `abs error mean = 0.057423 m/s`), and the measured baseline
controller ceiling from the two frozen calibration stages is **1.25 m/s** (1.30 missed by
0.00102 m/s; every damping candidate failed the frozen selection rule).  Controller, thresholds,
and physical dynamics may not be retuned to force the canonical arm.

This lineage therefore replaces only the top registered speed, exactly as the recovery-v2
lower-1.25 lineage did.  It does **not** reinterpret the canonical 1.5 warmup NO-GO, does not
unlock the canonical v3 arm (which remains NOT RUN and blocked), and does not weaken any
route-mechanism, physical-controller, integrity, or placement gate.

## Frozen intervention and grid

Route mode: `global_astar_braking_v3`, unchanged implementation bytes.  The lower contract is
selected only by the explicit environment declaration
`NAVRL_TARGET_BRAKING_CONTRACT_VARIANT=baseline_1p25`, mirroring the braking-probe variant
contract; the default everywhere remains `canonical_1p5`.

- Speeds: **0.6, 0.9, 1.2, 1.25 m/s**.  Record identifiers use lossless decimal keys so
  1.25 can never alias 1.2.
- Development pilot: seed **829**, 70 bars, arms `off` / `global_astar_braking_v3`, 8 cells.
- Confirmatory: seed **839**, bars 70/115/160/205, same arms and speeds, 32 cells.
- Environments 32, rollout 300 steps, warmup 20 steps — unchanged from corrected r2.
- Every pooled 70-bar route-mechanism gate and per-cell physical gate keeps the canonical v3
  thresholds byte-for-byte: runtime `unsafe_start` = 0, first soft-envelope exits = 0,
  terminal-stop certificate rate = 100%, all-cause plan success >= 99%, fallback fraction <= 1%,
  realised/commanded speed ratio >= 0.80 per routed cell, goal completions per environment at
  0.6 m/s >= 0.50, and the unchanged corrected-r2 controller/contact/displacement gates.

## Braking receipt requirement

The arming receipt is a **fresh `baseline_1p25` raw braking receipt generated at the current
commit**, produced by the unchanged preregistered probe
(`docs/preregistration_navrl_physical_target_braking_lower1p25_2026-08-26.md`) and verified by
the unchanged standalone raw-first verifier at gate start.  Declared scalars or a `VALIDATED`
marker alone cannot arm the mode.  The 2026-08-26 lower receipt is historical evidence only; it
binds superseded core bytes and may not arm this gate.

## Machinery integrity

Identical to the canonical v3 preregistration: runtime-clean tracked sources, tracked executable
cell adapter, source-manifest and import-origin binding, matched-arm layout/pose digests,
fail-closed cell validation before write, and a unique output root outside the repository.

## Decision and authority boundary

- Pilot PASS authorises only the seed-839 32-cell lower-contract confirmatory gate.
- Any pilot or confirmatory FAIL blocks PPO and requires mechanism analysis; thresholds, seeds,
  the speed grid, and the controller may not be adjusted in response.
- Confirmatory PASS authorises only a **separately preregistered** fresh 500-epoch corrected
  non-overlap PPO smoke whose target-speed contract must stay within this lower envelope.
- No stage of this lineage creates long-training, hardware, or sim-to-real authority, and none of
  it modifies the recorded canonical 1.5 NO-GO or any frozen FAIL verdict.
