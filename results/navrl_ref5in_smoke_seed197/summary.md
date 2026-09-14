# ref5in P1a fresh 500-epoch smoke — FAIL, corrective rerun required

Date: 2026-08-13 KST

This was a learning-viability engineering gate, not a held-out performance comparison. The run used
fresh weights, training seed 197, 128 environments, 70 bars, the corrected exact-600 semantics,
analytic perception, eight cluster-sector tokens, governor off, and `navrl_ref5in_quad`.

## Exact result

Run: `ppo_260813_0406_navrl_v2-ref5in-smoke-s197`
Terminal checkpoint: `last_gen_ppo_ep_500_rew_117.70658.pth`
Checkpoint SHA-256: `59f993811b2f358dc144544998cf0d230756198b19a33b658d0b82c22bebd26c`

The last 100 epochs pooled **3,709** completed episodes:

| outcome | count | rate | preregistered gate |
|---|---:|---:|---:|
| capture | 2,675 | **72.12%** | >=65% PASS |
| crash | 910 | **24.53%** | <=33% PASS |
| timeout | 124 | **3.34%** | <=5% PASS |

The run completed exactly 500 epochs, all checkpoint tensors were finite, all four raw-action OOB
rates were zero, PPO KL stayed below 0.04 (`max=0.03182`), and the clean source receipt bound 310
runtime files to commit `578b4bf668058c2b67ac734512fd450b16abb5a0`.

## Why the verdict is still FAIL

At epoch 432, the pre-update behavior KL reached `0.05043`; the independent audit reached
`0.05842`, over the fixed `0.04` gate. The transaction guard skipped one minibatch, restored the
entire PPO epoch, and reduced the learning rate from `3e-5` to `1.5e-5`. The guard worked as
designed—there was no corrupted continuation—but P1a preregistered **zero** rollback and **zero**
skipped minibatches, so this lineage cannot be called an uninterrupted PASS.

The distance curriculum ended at `[20,27] m`, one evidence-window promotion short of the required
`[20,28] m`. It had promoted monotonically ten times, with the final `25 -> 27 m` window scoring
74.5% capture. This is an insufficient engineering budget, not evidence of a 27 m plateau.

## Corrective decision made before rerun

Do not relax the completed run's gates and do not start held-out P2 or full training. Run a second
fresh smoke with the same seed and task contract, changing only:

- initial PPO learning rate: `3e-5 -> 1.5e-5`, the value selected by the safety backoff;
- budget: `500 -> 750 epochs`, allowing the final distance evidence window to complete.

P1b must finish exactly at epoch 750 with `[20,28] m`, no rollback/skipped minibatch, behavior/PPO
KL below 0.04, all-axis raw OOB zero, and the same last-100 outcome gates. Only a P1b PASS unlocks
the held-out 70-bar decision cell. The source of truth for every check is `summary.json`.
