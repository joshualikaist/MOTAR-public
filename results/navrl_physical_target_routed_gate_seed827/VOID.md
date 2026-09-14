# VOID — routed physical-target gate, seed 827 (attempt 1)

`summary.json` and `receipt.json` classify this execution as `VOID_EXECUTION`: the execution
manifest has `record_count: 0`, `integrity_ok: false`, and the only child returned exit code 1.
This is an execution-environment failure, not a physical-target or planner measurement.

The child traceback ends at PyTorch C++ extension initialization with:

```text
RuntimeError: Ninja is required to load C++ extensions
```

The command used the absolute conda Python, but the inherited `PATH` did not contain the matching
conda `bin` directory. No route cell was measured, no numerical gate was evaluated, and no PPO
authority was created. The artifact is retained unchanged; attempt 2 is a separate lineage.

| field | value |
|---|---|
| seed / intended grid | 827 / route off+on × 4 speeds × 4 densities |
| records | 0/32; `VOID_EXECUTION` |
| child | `route_off__speed_0p6`, exit 1 |
| source manifest SHA-256 | `6ce758e004a87956236889cd4a8326c9f8c91ac25a10df730227b02c32648b1c` |
| execution manifest SHA-256 | `6886776c42ccf95328299f500e464dcab7136ea1582b8a6fa04d222cce8734ba` |
| summary SHA-256 | `e3800a1afcb4d4634bac779c26eee329d40cead1a61ca7bbc8d8fe9b6da873c7` |
| receipt SHA-256 | `cc82e2f9fc1b57b0bbf23741ccc803196b73c1059fdd36bbc4a6d0f00ac3d184` |

Raw files: [`summary.json`](summary.json), [`execution_manifest.json`](execution_manifest.json),
[`receipt.json`](receipt.json), and [`logs/route_off__speed_0p6.log`](logs/route_off__speed_0p6.log).
