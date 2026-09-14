# Behaviour invariance of the TD instrumentation — 14 September 2026

**Verdict: `TRAJECTORY_INVARIANT`.**

Turning the evaluation-only episode forensics on does not change what the simulator did.

| Arm | Forensics | Trajectory digest (sha256) |
| --- | --- | --- |
| `off_a` | off | `b9cc980d26a35879879b6c64cb1c407a5f9bd7eeda8fd2156375b78901004a99` |
| `off_b` | off | `b9cc980d26a35879879b6c64cb1c407a5f9bd7eeda8fd2156375b78901004a99` |
| `on` | on | `b9cc980d26a35879879b6c64cb1c407a5f9bd7eeda8fd2156375b78901004a99` |

Two comparisons, not one. `off_a == off_b` shows this harness reproduces itself at all; without it an
`off ≠ on` difference could not be attributed to the logger. `off_a == on` is then the invariance
claim. The digest hashes the robot position, orientation and executed command of all 128
environments at every one of the 256 steps, and it is enabled in every arm so it cancels out.

Conditions: the same audited evaluation environment the D8b sensitivity launcher used, reached
through the same loader — 70 bars, seed 593, deterministic action selection, 128 episodes per arm,
checkpoint `last_gen_ppo_ep_1900_rew_182.11377.pth`. All three arms ran back to back on one source
tree; the evaluator re-checks its runtime source hashes and refuses on drift.

## What this is not

The `on` arm also wrote `on_episode_forensics.json`, and its counts are recorded in `summary.json`
under `forensics_smoke_summary`. **That is not the TD-T1 result.** It is one density and one seed,
produced only to show that the recorder runs end to end and emits well-formed records. The
preregistered TD-T1 and TD-T2 results need the full density axis (70, 115, 160, 205) under the
existing held-out protocol, and neither has been run.

Nothing here changes a policy, a reward, a detector, an association module, the safety filter, the
controller or a termination rule, and nothing was trained. `causality_vs_d8b: NOT_TESTED`.

## What was pruned before committing

Each arm's evaluation directory kept its result, receipt, log, environment record and source
manifest. The 8.8 MB `checkpoint_snapshot.pth` and the `source_snapshot/` tree were removed from all
three arms after the run: their sha256 values are recorded inside each `source_manifest.json` and
`70bars.receipt.json`, the checkpoint is the one named above and still in the repository, and three
more copies of a checkpoint do not belong in a public result directory. Nothing that carries a
number was removed.

## Reproducing

```bash
python tools/run_forensics_invariance_check.py \
  --checkpoint <checkpoint.pth> --output <absolute output directory> --games 128
```

The output path must be absolute: the evaluator runs from its own directory, and a relative path
would put the digest beside the launcher instead of in the result directory.
