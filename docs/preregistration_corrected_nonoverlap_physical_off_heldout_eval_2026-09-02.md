# Preregistration — seed-911 route-off curriculum held-out density evaluation

Frozen: 2026-09-02, after the seed-911 curriculum was operator-stopped at 145 bars and **before**
this evaluation produced any held-out result.

## Question and claim boundary

At each density the operator-stopped seed-911 policy actually trained on, what are the held-out
capture / crash / timeout rates of the terminal periodic checkpoint?

This is a measurement of one incomplete route-off baseline. It cannot support claims about
205-bar curriculum success, braking-v3, 1.5 m/s target motion, hardware validation or
sim-to-real transfer. Training-log capture is on-policy inside a curriculum window and is not
this evaluation.

## Frozen execution tuple

| item | value |
|---|---|
| parent run | `ppo_260901_1431_navrl_corrected-nonoverlap-physical-off-curriculum-s911` |
| launcher | `eval_navrl_corrected_nonoverlap_physical_off_heldout.sh` |
| checkpoint | `nn/last_gen_ppo_ep_21750_rew_83.1572.pth` |
| checkpoint SHA-256 | `541b36bdcabacf8bb14c6fbb0ad07054dd9735ad24777a3222655ba8ca9c8132` |
| forbidden checkpoint | `gen_ppo.pth` (best-reward / low-density policy) |
| eval seed | 313 (independent of training seed 911; 911 is forbidden) |
| episodes / envs / action | 2049 requested completed episodes per cell / 128 / deterministic |
| densities | **70 / 85 / 100 / 115 / 130 / 145** only |
| 205 bars | **not included**. It was never trained. It is untrained OOD and is not this evaluation. |
| robot / target | `navrl_ref5in_v2_quad`; physical 6-DoF; route off; mixed CV/waypoint |
| target speed | `U[0.3,1.25] m/s`, one-epoch ramp |
| arena / placement | 40×40×3 m; `footprint_clearance`; surface 0.45 m; fallback/merge 0 |
| perception / governor | canonical v2 cluster-sector Transformer; governor off |
| contract flag | `NAVRL_V2_EVAL_CONTRACT=corrected_nonoverlap_physical_off` |

Default `eval_navrl_v2_density_sweep.sh` densities `70 150 210 280` and the historical
`navrl_band` / `U[0.3,1.5]` / 300-epoch ramp contract are the wrong evaluation for this
checkpoint. Blind use of that default is VOID.

## Frozen interpretation

1. Report per-cell capture, crash and timeout from `actual_episodes`. Do not rename a 145-bar
   held-out number as 205-bar mastery, and do not treat on-policy curriculum holds as held-out.
2. 145 is in-distribution for this checkpoint (the run stalled there) and is still below the
   frozen 0.70 promotion gate in the training log. Held-out 145 may agree or disagree with that
   log; either way is a result, not a reason to change this tuple.
3. There is no capture-rate pass/fail gate here. The output is the six-cell table. A later
   training run, if any, needs a **new** preregistration after these numbers are seen.
4. `VOID_EXECUTION` if checkpoint SHA, seed 911, `gen_ppo.pth`, density set, speed cap, placement
   mode, route mode or import origin disagree with this document.
5. This evaluation does not authorize resume of the 1431 run, a second curriculum, routed PPO,
   1.5 m/s claims or hardware/sim-to-real claims.

## Authority

- One GPU pass of the six cells above, launched from the braking-route-v3 worktree so the main
  tree's stopcap screen is not disturbed.
- Completion authorizes only offline reading of those six cells and a decision about whether to
  write a **new** training preregistration.
- It never authorizes routed PPO, parameter search, hardware flight or sim-to-real claims.
