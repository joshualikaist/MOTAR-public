# NavRL v2 verification 5A — corrected-semantics engineering smoke

Date: 2026-08-12
Verdict: **CONDITIONAL PASS (engineering only; no performance claim)**

## Contract

- Fresh PPO, seed 197, 128 environments, 1,000 epochs; no checkpoint.
- Canonical v2 environment: 40×40×3 m, 70 bars at the smoke endpoint, mixed moving target.
- Analytic detector, 8 `cluster_sector` tokens, 240° token FOV, squashed-Gaussian action,
  governor off, pose noise off.
- Intervention under test: exact 600-action horizon plus rl_games `time_outs` bootstrap signal.
- Run: `aerial_gym/rl_training/rl_games/runs/ppo_260812_1620_navrl_v2-v5a-semantics-smoke-s197`
- Canonical endpoint checkpoint:
  `nn/last_gen_ppo_ep_1000_rew_140.1189.pth`, SHA-256
  `f53489aa91580d9087732ff8a3320d120066018bfdfe5b494a04bff1b0ac13f0`.

## Passed engineering checks

- Exit code 0, `max_epochs`, completion marker `epoch=1000`; no active GPU process afterward.
- Final checkpoint records `cfg_episode_len_steps=600`,
  `cfg_rlgames_timeout_info_key=time_outs`, analytic detector (no checkpoint),
  `cluster_sector`, 240°, squashed Gaussian, bars=70.
- The live run exercised the timeout path (530 timeouts in the parsable training epochs). Unit
  tests independently assert exact action 600 and exclusion of capture/crash terminals from
  `time_outs`.
- All 1,000 PPO scalar records were finite. PPO KL max **0.015831**; behavior-KL audit max
  **0.022902**, below the 0.04 rollback threshold; rollback count **0**; skipped minibatches **0**.
- Raw action out-of-bounds rate was exactly 0 on x/y/z/yaw for all 1,000 epochs.
- The four predeclared semantic-critical file hashes were identical before and after the run.

## Descriptive training outcomes

These are on-policy training outcomes at 70 bars, not held-out evaluation and not an algorithmic
comparison. Epochs with no finished episode are omitted; all rates below pool raw counts rather
than averaging noisy per-epoch percentages.

| Window | Episodes | Capture | Crash | Timeout |
|---|---:|---:|---:|---:|
| all parsable epochs | 44,383 | 73.02% | 25.78% | 1.19% |
| speed-ramp epochs 1–300 | 15,533 | 66.95% | 31.63% | 1.42% |
| post-ramp epochs 301–1000 | 28,850 | 76.29% | 22.63% | 1.07% |
| last 250 epochs | 9,953 | 78.55% | 20.36% | 1.10% |
| last 100 epochs | 4,134 | 78.69% | 20.30% | 1.02% |
| last 50 epochs | 2,088 | 78.40% | 20.59% | 1.01% |

The stable recent windows and zero rollback support the narrow claim that corrected timeout
semantics do not prevent fresh PPO from learning. The run does not establish superiority over the
legacy semantics.

## Warnings and untested scope

1. **Density promotion was not tested.** `NAVRL_DENSITY_WARMUP=1000`; the endpoint reaches exactly
   `num_task_steps=32000`, so post-warmup density evidence starts only on the next epoch. The zeroed
   density accumulator is therefore expected. Distance curriculum did saturate at [20, 28] m.
2. **No held-out evaluation and only one training seed.** Do not cite 78–79% as final navigation
   performance or as an algorithmic effect.
3. **Partial provenance only.** The worktree was dirty and this launcher did not create a full
   runtime source manifest. Four critical hashes were manually frozen and matched, but 5B must run
   from a clean commit with a complete source receipt.
4. **Forward action saturation remains visible.** At epoch 1000, x `edge95=49.8%` and
   `edge99=18.6%` (run maxima 64.6% and 43.0%). Raw OOB is zero, so this is not the old unbounded
   action bug, but it is a control-behavior warning for later dense-clutter analysis.
5. The automatic run summary's 97.1% peak is a small single-epoch sample and must not be used.

## Post-run defect correction

Epoch 1000 was both a 50-epoch save boundary and `max_epochs`, so the trainer wrote two differently
named but recursively identical checkpoints. The existing files are preserved as evidence. The
save logic now skips the redundant terminal write when a periodic checkpoint already exists and
uses the canonical scalar-reward filename otherwise. Regression test added.

## Decision

5A passes its engineering purpose. Do **not** run the excluded full-budget step now. Before 5B,
commit/freeze the current source, add a full training source manifest/receipt, and preregister
matched 2–3 training seeds plus held-out evaluation seeds. 5B should be the first run that actually
crosses the density warmup/gate.
