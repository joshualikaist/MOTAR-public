# Preregistration — recovery-v2 physical-target 32-cell evaluator

Frozen 2026-08-25 before any recovery-v2 GPU result. This evaluator is new and opt-in; the
existing `global_astar_v1` evaluator and its result lineage remain byte-identical.

## Fixed grid and execution

The exact grid is `route={off,global_astar_recovery_v2}` ×
`bars={70,150,205,300}` × `speed={0.6,0.9,1.2,1.5 m/s}`: 32 cells, seed 827,
32 environments, 300 RL intervals, 20 tracking warm-up intervals, neutral pursuer action,
`base_sim`, `navrl_ref5in_quad`, physical waypoint target, 40×40×3 m arena,
`bars_h3/navrl_band`, and `0.01 s × 10` physics per 0.1 s interval. One fresh child owns the
four density cells for each route/speed pair, matching the immutable v1 evaluator process
partition. The task is reset and reseeded before each density. `NAVRL_NUM_BARS=70` and
`NAVRL_MAX_BARS=300` are both runtime-attested.

The target-specific braking result must first pass its standalone raw-first verifier. Its entire
finalized directory is copied into the partial result and hash-bound. The common validator alone
sets `NAVRL_TARGET_RECOVERY_PROBE_VALIDATED=1`. The registered speed's measured p95 stop distance
is recorded separately from `v²/(2*a_p05)`; recovery safety uses the standalone verifier's
monotone-certified ceiling-speed lookup and its p95 lateral braking tube, not the formula. The
child task independently revalidates the copied receipt, source manifest, lookup, and lateral tube.

## Packed evaluation telemetry

Every cell writes one device-packed NPZ with fixed denominators `32×300` intervals and
`32×300×10` substeps. It records state/status transitions, general/BRAKE/CONNECT ages and phase
timeouts, exact hard/soft signed AABB margin and wall/bar reason, actual PhysX pose/velocity,
submitted command, measured and formula braking distances, anchor/cell/connector clearance,
selected candidate row/index/count/horizon/safe prefix/final rollout endpoint, fixed-anchor first/
horizon/actual progress, route/no-connector reasons, watchdog/contact/OBB/
geometry/motor/tilt traces, and evaluator-attributed reset/position-write counts.

Verification is raw-first and enforces legal within- and cross-interval transitions, phase-field
completeness, full-horizon CONNECT certificates, strictly hard-safe connectors, zero command in
NO_CONNECTOR, no target pose write/reset from recovery, unique timeout events, and the identity
`entries = resumes + no_connector + reset_during_recovery + open_at_end`. Reason counts partition
all no-connector transitions. Runner resets are attributed after the recorded interval and are
the only permitted cross-interval discontinuity.

The recovery-only helper returns the selected candidate's explicit environment row plus exact
index/count/horizon/safe-prefix/full-horizon/final-position certificate; the evaluator does not
infer identity from coordinates or the first velocity, because environments can share coordinates
and acceleration saturation can alias multiple candidates. Any optional evaluator recomputation and
connector attribution time is reported separately. The operational throughput gate conservatively
uses total instrumented wall time; the observer-adjusted value is descriptive only. Matched
recovery/off throughput must be at least 0.50.

## Existing gates and decision

Per-cell gates remain: tracking RMSE ≤0.35 m/s, realized/requested speed ≥0.80, contact ≤0.01,
arm-specific local infeasibility ≤0.01 (off: rounded bounded-step infeasible; recovery: rounded
normal-route invalidation only; each numerator divided by exactly `32×300` intervals), motor
saturation ≤0.15, tilt ≤60°, invalid state 0,
watchdog hard breach 0, direct target-position write 0, and reset-during-target-advance 0.
The 70-bar route mechanism requires plan success ≥0.99, fallback ≤0.01, and at speed 0.6 at
least 0.5 goal completions/environment. A scientific FAIL is finalized rather than discarded;
only execution-integrity failures produce a VOID directory. PPO and long training remain
unauthorized.

## Atomicity and provenance

The parent writes a hidden sibling directory, uses exclusive temporary files plus fsync/replace,
refuses every existing final path, verifies all child/raw/source/probe hashes, writes summary,
execution manifest, and receipt, verifies before rename, atomically renames the directory, and
verifies again. Standalone verification recomputes cell gates, matched identities, telemetry
summaries, verdict, source/import origin, child/summary equality, and all artifact hashes.
Source, evaluator, packed observer, braking verifier/probe, robot/URDF, Python/Torch/CUDA/
Isaac Gym and `nvidia-smi` provenance are bound. Committing a result may change HEAD but may not
change any recorded runtime byte.
