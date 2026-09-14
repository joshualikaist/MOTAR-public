# NavRL v2 ep24000 core failure and limit audit

Generated: `2026-08-02T01:48:07.276263+00:00`

## Decision

The curriculum continuation is closed. More epochs under the unchanged 205-bar stochastic gate are not justified.
The final deployment-style deterministic policy clears 70% (72.44%), while the sampled training policy does not (67.35%). The next training candidate is a fixed-density, one-variable representation A/B, but causal evaluations come first.

## Frozen artifact

- checkpoint: `aerial_gym/rl_training/rl_games/runs/ppo_260802_0020_navrl_v2-recover-curriculum-continue-s1/nn/last_gen_ppo_ep_24000_rew_44.73549.pth`
- epoch/SHA-256: `24000` / `82f7978b42d9d9e95adcc638a40ae85fb3736fd897ae39cb8aa8333be39cf23f`
- canonical training: 14,510 epochs = 59,432,960 samples
- 205 bars alone: 4,910 epochs = 20,111,360 samples; 10 complete holds
- last seven gate windows: mean 68.94% (16,384+ episodes each)

## Held-out deterministic density sweep

| bars | /100m² | episodes | capture (Wilson 95%) | crash | timeout | bar contact / all eps |
|---:|---:|---:|---:|---:|---:|---:|
| 130 | 8.125 | 2049 | 84.77% [83.15%, 86.26%] | 12.74% | 2.49% | 11.18% |
| 160 | 10.000 | 2050 | 79.66% [77.86%, 81.34%] | 16.88% | 3.46% | 15.61% |
| 190 | 11.875 | 2049 | 73.99% [72.04%, 75.84%] | 22.65% | 3.37% | 21.43% |
| 205 | 12.812 | 2050 | 72.44% [70.46%, 74.33%] | 25.07% | 2.49% | 24.29% |
| 220 | 13.750 | 2050 | 68.49% [66.44%, 70.46%] | 29.76% | 1.76% | 28.78% |

A linear interpolation between 205 and 220 puts the deterministic 70% crossing near 214.3 bars. This is a descriptive frontier, not a proof of a hard limit.

## Action-selection A/B at 205 bars

| mode | episodes | capture (Wilson 95%) | crash | timeout | lateral edge98 |
|---|---:|---:|---:|---:|---:|
| deterministic | 2050 | 72.44% [70.46%, 74.33%] | 25.07% | 2.49% | 3.37% |
| stochastic | 2049 | 67.35% [65.29%, 69.35%] | 30.41% | 2.24% | 7.08% |

Deterministic minus stochastic: capture +5.09 pp (approx. 95% CI +2.28 pp to +7.89 pp), crash -5.33 pp (CI -8.07 pp to -2.60 pp). Exploration also doubles lateral edge98 from 3.37% to 7.08%.

Verdict: the gate is not a random false alarm. It measures the sampled training policy; the deterministic deployment policy is materially better. Keep both metrics and do not silently replace one with the other.

## Stable failure structure

At 205 bars, deterministic distance strata fall from 81.42% (6–11.5 m) to 61.41% (22.5–28 m); stochastic strata fall from 75.35% to 55.06%. The fastest stochastic speed bin is 64.26%. CV is weaker than waypoint in both modes. This is accumulated collision exposure over long/high-speed trajectories, not a timeout or stationary-drone bottleneck.

The 205-bar 0.2 m-clearance geometry audit found crossing=1.000, largest component=0.999150, random-pair connectivity=0.998308, and no placement fallback. Disconnected free space is therefore rejected as the primary cause.

PPO is also rejected as the primary cause: behavior KL stayed below the 0.04 rollback threshold, learning rate remained 5e-6, explained variance was healthy, and no rollback or out-of-bounds action input occurred. Failures remain overwhelmingly bar contacts.

## Literature-calibrated interpretation

| system | arena / density | task and result | transferable lesson |
|---|---|---|---|
| NavRL | 50×50 m; 350 static + 60→120 dynamic | fixed-goal navigation; curriculum SR 94.33→68.65%; best saved at 100 dynamic | stopping before the hardest stage and deploying the distribution mean/safety shield are legitimate design choices |
| NavRL++ | 40×40 m; evaluation up to 400 static + 100 dynamic | high-dynamic SR 83.96%; ~200 RTX4090 GPU-hours, 1,024 robots | use static ray geometry plus temporal structured obstacles; curriculum performance need not improve monotonically |
| Ours | 40×40 m; 205 static bars = 12.81/100m² | moving-target interception; 72.44% deterministic / 67.35% stochastic | raw SR must not be ranked against fixed-goal navigation; our immediate gap is risk selection and exploration, not obstacle count alone |

Primary sources: [NavRL](https://arxiv.org/html/2409.15634v2), [NavRL++](https://arxiv.org/html/2605.15559v1), [Anticipatory Risk-Guided RL](https://arxiv.org/abs/2607.23565), [Self-Paced Contextual RL](https://proceedings.mlr.press/v100/klink20a.html).

## Next training candidate (prepared, not authorized)

Run `cluster_sector` versus `ttc_sector` from the byte-frozen ep24000 checkpoint at fixed 205 bars. Both arms receive exactly 4,096,000 samples: 1,000 epochs on main or 2,000 on the 4GB profile. TTC advances only if held-out capture improves by at least 2 pp and crash falls by at least 2 pp versus its same-profile baseline.

Launcher: `aerial_gym/rl_training/rl_games/train_navrl_v2_ep24000_ttc_ab.sh`.

Do not change density threshold, action sigma, reward, speed limit, and selector in one run. If TTC fails, the next isolated experiment is action-noise reduction. If it passes, repeat deterministic/stochastic evaluation before deciding whether the curriculum gate should remain a robustness gate or be paired with a separate deployment gate.

## Remaining unknowns

- The large positive lateral action remains a symptom, not a chirality verdict; a paired mirrored-layout evaluator is still required.
- The current actor uses an analytic target detector, so learned perception remains a later research stage.
- The full 4×72 static scan is present in addition to eight obstacle tokens. Calling this simply an '8-token capacity limit' is inaccurate; the open question is whether the actor uses dense geometry and threat ordering effectively.
- The action-evaluation outcome and strata are valid, but its old goal-centered context bucket included the rear cone. That diagnostic has now been fixed and is excluded here.

## Pending causal checks before another training run

This document closes the unchanged 205-bar curriculum, but it is not the end of the causal audit. At this core-audit snapshot the next step was a mirror evaluation of the frozen ep24000 policy; that evaluation and the second seed are completed in the addendum below. Fixed-speed cells, a forgetting comparison, target-trajectory reachability, and the real 1650 Ti fixed-205 gate remain pending before the next main training run.

## 2026-08-02 causal 1--3 addendum

The frozen mirror and second-seed checks are now complete; this original core-audit snapshot remains
unchanged for provenance. Seed 43 reproduced 205-bar capture at 72.77% versus seed42 72.44%
(+0.33 pp). Original versus mirror-conjugate aggregate capture differed by -0.81 pp, and initial
negative/positive-y target-bearing capture differed by -0.21 pp, so no outcome-side asymmetry was
detected. Exact reflected-observation actions nevertheless failed equivariance: lateral MAE 1.235
and sign mismatch 73.08%. H-BIAS is therefore action-level supported but outcome-neutral in the
current symmetric arena. See `results/navrl_v2_ep24000_causal_1to3/summary.md`. Fixed-speed,
forgetting, trajectory-reachability, and real 1650 Ti gates remain pending; no new training has run.
