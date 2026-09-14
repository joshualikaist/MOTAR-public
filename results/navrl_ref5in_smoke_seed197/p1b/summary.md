# ref5in P1b fresh 750-epoch smoke — FAIL (distance budget only)

> **2026-08-13 sampler erratum:** `[20,27]` is checkpoint curriculum state. The applied
> general-spawn range was `[6,27] m`; this run failed the preregistered max-state gate, not an
> applied 20 m minimum-distance gate.

Date: 2026-08-13 KST

P1b repeated the P1a engineering gate from fresh weights with the same training seed 197 and
task/robot contract. Only the initial actor learning rate (`3e-5 -> 1.5e-5`) and maximum budget
(`500 -> 750`) changed. This is an on-policy learning-viability check, not held-out performance.

## Exact result

- Run: `ppo_260813_0441_navrl_v2-ref5in-smoke-b-s197`
- Terminal checkpoint: `last_gen_ppo_ep_750_rew_135.66144.pth`
- Checkpoint SHA-256: `5174695ac0d6ab8dcc81a4351afea378db55be2b00b2a4669de5ea1e80a6a2cf`
- Runtime source: clean commit `227b874cfaa358a4f0040885b6dbe45a06878084`, 311 files,
  manifest SHA-256 `1b5209686c696ab9242111a3a1af54d60fa91e6cd31707021db7ae53fab628d6`

The last 100 epochs pooled 3,780 completed episodes:

| outcome | count | rate | preregistered gate |
|---|---:|---:|---:|
| capture | 2,629 | **69.55%** | >=65% PASS |
| crash | 1,063 | **28.12%** | <=33% PASS |
| timeout | 88 | **2.33%** | <=5% PASS |

## Gate accounting

Every gate except distance saturation passed:

- normal epoch-750 completion and one unambiguous terminal checkpoint;
- every checkpoint tensor and all 60 TensorBoard scalar tags finite;
- PPO KL max `0.01300`, behavior-audit KL max `0.01980`, both below `0.04`;
- rollback, skipped minibatches, and all-axis raw action OOB: exactly zero;
- initial and terminal actor LR both `1.5e-5`;
- robot config/URDF SHA and every original/snapshotted runtime file re-hashed successfully;
- exact seed/main/128-env/Transformer/cluster-sector/governor-off contract.

The terminal distance window was `[20,27] m`, not the required `[20,28] m`. Promotion epochs were
372, 403, 434, 467, 500, 534, 571, 611, 657, and 709. As goals became longer, fewer episodes
finished per epoch, so the fixed 2,048-episode evidence window took progressively more epochs. The
750 budget ended before the final `27 -> 28 m` evidence window filled. This is not a measured
27 m performance plateau, but it is still a strict P1b FAIL.

## Corrective decision before P1c

Do not relax P1b and do not run held-out P2 from its checkpoint. P1c remains fresh seed 197 with
the same LR and all task coordinates; only the budget becomes 900 epochs. This gives the final
promotion enough completed episodes and leaves at least one full 100-epoch reporting window after
the projected saturation point. P1c must pass every unchanged gate before P2 is allowed.

Machine-readable source of truth: [`summary.json`](summary.json).
