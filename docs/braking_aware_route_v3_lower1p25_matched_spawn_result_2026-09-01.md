# Result — lower-contract v3 matched-spawn pilot

Date: 2026-09-01 (Asia/Seoul)

## Verdict

- execution integrity: **PASS_8_CELL_INTEGRITY**
- scientific gate: **FAIL_BLOCKS_CONFIRMATORY**
- mechanism pass: false
- all corrected-r2 per-cell physical gates pass: false
- confirmatory/PPO/long training: **not authorized**

Source summary:
`/home/fair/workspaces/aerial_gym_ws/navrl_v3_runs/pilot_lower1p25_matched_spawn_seed829_2026-09-01/summary.json`
(SHA-256 `8e4b71bc9095f64065140f2838904bab8b7be868c7d97025953fb451041a49fc`).

The off/v3 pairs matched on layout, robot pose, and target pose at all four speeds. This resolves
the previous `VOID_EXECUTION`; it does not reinterpret that run. The new run is independently
integrity-clean and therefore its FAIL is official.

## Frozen pilot inputs

- commit containing runtime source: `b054f072da8ce9e1d5584c6eba868d5667490812`
- parent method preregistration SHA-256:
  `17e7f35e350087bf2733aca70dfca7210efe667ad1845dd053f7334ddf1645b4`
- execution authority SHA-256:
  `70b8a08f0c95040a86c43e1be5ac11d0b688b9a6874f3cfc4548f914882f085f`
- braking receipt SHA-256:
  `18425bfc9cb618834b37435b8d83a07dc1af69e56aa02459f00a94c544e075cc`
- training runtime manifest SHA-256:
  `a34618e91a12d6a94f4a48f9b145cf72531c2ce54aa6f40eeec71db002c39840`
- seed 829, 70 bars, 32 envs, 300 measured steps, 20 warmup steps
- arms `off` and `global_astar_braking_v3`; speeds 0.6/0.9/1.2/1.25 m/s

## Per-cell result

| arm | speed m/s | speed ratio | tracking RMSE m/s | local invalid | physical pass |
|---|---:|---:|---:|---:|---|
| off | 0.60 | 0.9475 | 0.0998 | 0.00135 | PASS |
| off | 0.90 | 0.9458 | 0.1422 | 0.00208 | PASS |
| off | 1.20 | 0.9352 | 0.2037 | 0.00615 | PASS |
| off | 1.25 | 0.9458 | 0.2083 | 0.00281 | PASS |
| v3 | 0.60 | 0.7872 | 0.1400 | 0.00000 | FAIL: speed |
| v3 | 0.90 | 0.7681 | 0.1867 | 0.00083 | FAIL: speed |
| v3 | 1.20 | 0.5984 | 0.2495 | 0.00146 | FAIL: speed |
| v3 | 1.25 | 0.6297 | 0.2652 | 0.00146 | FAIL: speed |

All contact, motor, tilt, state-displacement, tracking, and arm-specific local-feasibility gates
passed. Every routed cell failed only the preregistered mean-speed-ratio minimum of 0.80.

## Mechanism inputs

| metric | measured | gate | result |
|---|---:|---:|---|
| runtime replan unsafe starts | 0 | 0 | PASS |
| soft-envelope exits | 6,825 | 0 | FAIL |
| terminal certificate fraction | 1.000 | 1.000 | PASS |
| plan success fraction | 0.99468 | at least 0.99 | PASS |
| fallback interval fraction | 0.08914 | at most 0.01 | FAIL |
| 0.6 m/s goals per env | 0.21875 | at least 0.50 | FAIL |

Per-speed routed soft exits were 573/1,387/2,583/2,282; certificate failures were
573/1,395/2,597/2,296. Accepted commands all carried terminal-stop certificates. Together with
the CPU recursive-certificate and watchdog parity tests, this supports the diagnosis that the
ideal stopping certificate is not robust to the physical target's realized tracking error.

## Decision

The matched-spawn change repaired experiment identity but did not repair the route mechanism.
The preregistered failure prediction was confirmed. Do not run seed-839 confirmatory, PPO smoke,
long training, or post-result gain/margin/threshold search. A future controller redesign is a new
method and requires a new preregistration; this pilot cannot be reused as its tuning set.

