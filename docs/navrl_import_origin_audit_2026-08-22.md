# NavRL diagnostic import-origin audit — 2026-08-22

## Question

PEP 660 editable installation can resolve `import aerial_gym` to the primary worktree even when
`runner.py` is launched from a diagnostic worktree. This audit checks the four Codex diagnostic
branches used on 2026-08-21. A source manifest alone is not sufficient: it describes the bytes
snapshotted by the evaluator, not necessarily the module imported by Python.

## Method

For each simulator-backed result:

1. Read the result source manifest and its declared `repository_root`.
2. Re-hash every `runtime_files` entry against the diagnostic worktree.
3. Read the raw simulator log for absolute Python warning paths. `motor_model.py` is imported through
   the `aerial_gym` package during task startup, so its absolute path identifies the package root that
   Python actually executed.
4. Compare the manifest runtime bytes with the preserved primary physical-target WIP. A difference
   means a primary import would be material, not harmless.

The topology branch is treated separately because it is an offline NumPy analysis and never imports
the simulator package.

## Results

| branch/result | manifest vs diagnostic worktree | manifest vs preserved primary | raw log import root | verdict |
|---|---:|---:|---|---|
| `codex/active-search-geofence-eval` | 315/315 identical | 20/315 differ | `.codex_worktrees/navrl_geofence_eval/aerial_gym` | VALID |
| `codex/mode-probe` | 316/316 identical | 22/316 differ | `.codex_worktrees/navrl_mode_probe/aerial_gym` | VALID |
| `codex/joint-telemetry`, canonical rerun | 317/317 identical | 22/317 differ | `.codex_worktrees/navrl_joint_telemetry/aerial_gym` | VALID |
| `codex/joint-telemetry`, first run | manifest described diagnostic bytes | material primary differences | `src/aerial_gym_simulator/aerial_gym` | VOID (already labelled and archived) |
| `codex/topology-labels` | offline tool at commit `8a2eaf3` | not applicable | no `aerial_gym` import | VALID as exploratory offline output |

For all three valid simulator results, `navrl_task.py` in the manifest differs from the preserved
primary WIP but exactly matches the diagnostic worktree. Therefore the absolute worktree import paths
are decision-relevant evidence: had Python imported primary, those runs would be VOID. It did not.

The invalid first joint run is already archived at
`results/void_navrl_v2_joint_speed_allocation_seed379_primary_import_20260821/` in its diagnostic
worktree. Its missing joint telemetry attestation caused the evaluator to fail closed before a result
claim was accepted. The canonical rerun added an explicit local `PYTHONPATH` and origin check and is
the only joint result cited.

## Decision

- Existing geofence, mode-probe, canonical joint-telemetry and exploratory topology numbers retain
  their prior verdicts.
- The first joint run remains VOID; it is not rehabilitated by byte comparison.
- Future worktree simulator runs must set `NAVRL_REQUIRE_SOURCE_ROOT` and reinject local
  `PYTHONPATH` **after** any helper such as `P2.canonical_env` clears it.
- `aerial_gym/config/robot_config/**` and `resources/robots/**` remain frozen checkpoint provenance;
  comments and docstrings are load-bearing bytes.
