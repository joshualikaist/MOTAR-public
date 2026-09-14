# Preregistration — physical-target zero-command braking probe (`v1`)

Frozen 2026-08-25 on the `two-envelope-recovery` lineage. This is an evaluation-only
measurement of the existing physical target controller. It does not change planner decisions,
target commands used by the routed task, reward, observations, termination, PPO, or the original
attempt-2 artifacts/evaluator. The pursuer braking probe is a different experiment and is not
used here.

## Frozen experiment

The four registered target-speed arms are exactly `0.6`, `0.9`, `1.2`, and `1.5 m/s`; every arm
uses 32 environments, seed `827`, `base_sim`, `navrl_ref5in_quad`, physical target dynamics,
`NAVRL_NUM_BARS=0` (an obstacle-free certified-center setup), `NAVRL_MAX_BARS=300`,
`bars_h3`/`navrl_band`, and a fresh Isaac Gym child process. PhysX is `0.01 s × 10` per `0.1 s`
control interval. The source/config/robot/URDF tuple
is recorded and exact-attested in the receipt. Missing GPU, driver, import origin, tool hash,
or instantiated physical parameter is a hard refusal.

Each child places the actor at the arena center with recorded clearance, commands the existing
target controller at its registered world-frame speed for a 5 s convergence warmup, then submits
a zero target-velocity command. The stop threshold is frozen at `0.10 m/s`; warmup must converge
within both 0.05 m/s and 10% of the requested speed. The raw row for every environment contains
requested and measured initial speed, integrated interval-path stop distance (not endpoint
displacement), stop time, effective deceleration, lateral deviation from the initial velocity
ray, contact, invalid-OBB, braking-phase motor saturation, and braking-phase maximum tilt. The
trace retains every `0.01 s` PhysX callback sample (exactly 10 callbacks per `0.1 s` interval),
including position, speed, path, contact, invalid-OBB, saturation, and tilt.

## Analysis and gates

The standalone validator recomputes all statistics from raw rows using linear interpolation on
the sorted `n-1` probability index. The measured lookup is keyed by the exact decimal speed and
contains `p95_stop_time_s`, `p95_stop_distance_m`, and `p05_effective_deceleration_mps2`. A result
is accepted only when warmup and braking contact/invalid-OBB counts are zero, saturation is at most `0.15`,
tilt is at most `60°`, all 32 environments stop within the fixed child budget, and every value
is finite. Summary values supplied by a producer are never trusted: rows and p05/p95 are recomputed
from the trace. Raw speed-specific p95 distance is also emitted as a cumulative-maximum certified
lookup for the recovery handoff, with a separately certified lateral-deviation tube.

The probe is a measurement, not a tuning loop. No margin (`0.45` or otherwise), acceleration,
turn limit, lookahead, or controller gain may be changed after seeing these data. The resulting
digest is an input receipt for recovery evaluation only; it does not authorize PPO or a topology
claim.

## Receipt and fresh-only contract

The launcher creates a new partial directory, writes canonical finite-only JSON, verifies it,
atomically renames it to the requested output, writes no existing path, and verifies again after
rename. `complete.marker`, `receipt.json`, `summary.json`, `source_manifest.json`, and four raw
`cells/speed_*.json` files are required. The receipt binds the valid recorded git commit object,
all core physical-target/config/robot/URDF bytes, this preregistration, generator/validator/
launcher bytes, Python/Torch/CUDA/Isaac Gym/GPU-driver/nvidia-smi provenance, and per-cell hashes.
Post-commit verification checks the recorded commit object and current manifest bytes; it does
not require current HEAD equality.

The v2 launcher is fresh-only. It rejects unknown continuation options, existing output
directories, attempt-2 paths, and partial-directory reuse. There is no checkpoint/resume input
in this experiment. GPU execution is deliberately not part of CPU preflight or unit tests.
