# Codex independent review — verification 1–4 and verification 5 decision

Date: 2026-08-12
Reviewed material: Claude commits through `47d836b`, verification briefs 1–2 and 3–4, raw result
JSON/receipts/source manifests, and current runtime source.

## Executive decision

- Verification 1 remains valid: learned-v2 detector navigation non-inferiority replicated.
- Verification 2's system-level failure is real, but its original causal explanation was
  confounded. A fresh diagnostic now isolates the cause: learned-v7 output statistics, not its
  0.70 threshold, are incompatible with the analytic-trained actor.
- Verification 3's constant clock-offset ladder remains a valid sensitivity experiment. Its
  position/yaw cells were confounded by global RNG consumption and are superseded by the
  isolated-RNG campaign. No hardware tolerance can be claimed from these single-seed ladders.
- Verification 4's original reachability percentages were invalid because the oracle used the
  wrong arena frame. Corrected static 2-D connectivity is 100% at all three radii, but this does
  not acquit dynamics, target motion, or the 600-step horizon. Contact-time token and braking
  diagnostics are descriptive, not yet causal.
- Do not start a multi-change fresh PPO. Use a corrected-semantics engineering smoke first, then
  matched fresh-training arms that isolate perception adaptation from appearance randomization.

## Verification 1

Independent recomputation over seeds 97/101:

- analytic: 3,271 / 4,098 captures
- learned-v2: 3,272 / 4,100 captures
- learned−analytic: −0.0145 pp, 95% CI [−1.752, +1.723]
- non-inferiority lower bound > −2.0 pp: PASS

All four cells have matching policy, detector, result, receipt, and source hashes. The launcher
summary failure after cell completion is disclosed; it did not corrupt cell data.

## Verification 2

The original 8-cell result is numerically valid as a whole-system comparison:

- nominal analytic 80.92%
- nominal learned-v7@0.70 75.90%
- envelope analytic 49.88%
- envelope learned-v7@0.70 66.63%

However, nominal E1 changed detector implementation and threshold simultaneously. The corrective
fresh-seed diagnostic (`results/navrl_v2_detector_threshold_diagnostic_seed191_193`) found:

| contrast | effect | 95% CI |
|---|---:|---:|
| v7@0.55 − analytic@0.55 | −5.192 pp | [−6.982, −3.401] |
| v7@0.70 − analytic@0.55 | −3.752 pp | [−5.523, −1.981] |
| v7@0.70 − v7@0.55 | +1.439 pp | [−0.407, +3.285] |

Thus threshold 0.70 is not the cause of the nominal loss. The learned detector changes the
mask/range/visibility/carve-out distribution seen by an actor trained with the analytic detector.
The appropriate remedy is matched fresh training with that detector, not post-hoc threshold
tuning. The envelope result remains a selected simulator stress envelope, not an independently
validated real deployment distribution.

Additional corrections:

- broad `NAVRL_V2_FORCE=1` was replaced with a detector-threshold-only override;
- v7 receipt seeds were corrected from stale constants 71/73/79 to actual 137/139/149;
- the receipt writer now records CLI seeds rather than module constants;
- v6 failure followed by v7 redesign is sequential model development, not one untouched
  confirmatory chain; the 1-pixel tolerance and selected envelope must be disclosed accordingly.

## Verification 3

Clock-offset cells draw no random numbers and remain valid as constant offset sensitivity. They do
not test clock rate skew or jitter. `+20 ms allowed`, `−50 ms safe`, and the proposed mechanism via
forward carve-out were not preregistered or directly instrumented, so they must not be presented as
hardware specifications or proven mechanism.

The original position/yaw implementation consumed the global PyTorch RNG, changing simulator reset
and scene randomness across arms. A dedicated seeded generator now isolates pose noise. The fresh
seed181 corrective campaign found:

- position 1/3/10 cm: +0.20/−0.69/−1.47 pp; every CI includes zero;
- yaw 0.5°: −0.69 pp; CI includes zero;
- yaw 2°: −3.28 pp, CI [−5.86, −0.70];
- yaw 5°: −12.75 pp, CI [−15.47, −10.03].

The defensible claim is limited to step-wise iid Gaussian perturbations on one environment seed.
Position loss up to 10 cm was not detected; yaw sensitivity is detected at 2° and severe at 5°.
No 1° cell exists. Bias, drift, correlated odometry error, and v7-stack interaction remain untested.

## Verification 4

The dump is in the 0..40 m arena frame, while the original oracle assumed −20..20 m. The corrected
oracle and regression test produce 333/333 connected contact episodes at 0.40, 0.65, and 0.766 m
inflation. This supports only: a static 2-D path from spawn to the final target position existed.

It does not test:

- turn radius, acceleration, or braking;
- the moving target trajectory;
- whether the path fits inside 600 actions;
- time-varying reachability.

The contact probe reports 83.2% token association and 85.3% hit-in-FOV, but these events are not
nested: `hit_token_given_fov` uses the intersection divided by hit-in-FOV, while `hit_token` also
includes matches outside the configured token FOV through the full scan/history representation.
Therefore the raw values are not algebraically inconsistent. They remain contact-time snapshots;
pre-contact coverage is required before attributing crashes to token absence.

Negative contact-time stopping margin and high contact-time clearance may be two observations of
the same missed-obstacle channel. A causal control/representation split requires matched successful
near-miss states and a pre-contact time window.

## Implementation defects fixed

1. Episode-dump indentation had accidentally gated normal perception, previous-action, and
   visibility resets on dump enablement. The reset path is restored and AST-guarded.
2. Reachability arena coordinates are corrected to 0..40 m and tested with an upper-half scene and
   a full-height blocking wall.
3. Pose perturbations have a dedicated seeded generator; tests prove they neither advance global
   RNG nor lose repeatability.
4. Narrow threshold override is defined in every evaluator Python process and regression-tested.
5. Detector training receipts record the actual requested split seeds.

## Verification 5 decision

Reject option C (corrected semantics + v7 + appearance randomization + token/control changes in one
run). It cannot identify which change helped or hurt.

Recommended sequence:

1. Engineering smoke: exact-600 and `time_outs` bootstrap only; analytic detector and current
   representation/control; 500–1,000 epochs; no performance claim.
2. Corrected-semantics baseline: full-budget fresh PPO. A single training seed is a demonstration,
   not an algorithmic result; use at least two, preferably three, training seeds for a claim.
3. Learned-v7 nominal arm: same seeds, architecture, budget, curriculum, and evaluation seeds.
4. Learned-v7 + appearance-randomized arm only after step 3, so detector adaptation and domain
   randomization remain separable.

Do not add pose randomization, token capacity/FOV changes, or governor changes to these first arms.
The fresh launcher must pin no-checkpoint semantics, horizon/bootstrap keys, detector artifact and
threshold, curriculum/budget, source receipts, and stop rules before GPU launch.

## Verification 5A result

Verification 5A completed on 2026-08-12 as run
`ppo_260812_1620_navrl_v2-v5a-semantics-smoke-s197`: fresh seed 197, 128 environments,
1,000 epochs, analytic detector, current cluster-sector/squashed-Gaussian contract. Verdict:
**conditional engineering PASS; no performance claim**. The endpoint checkpoint is
`last_gen_ppo_ep_1000_rew_140.1189.pth` (SHA-256 `f53489aa9158…`).

- exact-600/`time_outs` checkpoint contract present; timeout path exercised;
- PPO KL max 0.015831 and behavior-KL audit max 0.022902 (<0.04), zero rollback, zero skipped
  minibatches, and zero raw OOB on every action axis;
- pooled post-speed-ramp training outcomes: 22,010/28,850 capture = 76.29%; last 100 epochs:
  3,253/4,134 = 78.69%. These are descriptive on-policy 70-bar data only;
- distance max-state reached 28 m, but the general-spawn sampler actually remained `[6,28] m`;
  `k_min_cur=20` was not its applied minimum. Density promotion was **not tested** because epoch
  1,000 is exactly the configured density-warmup boundary;
- the dirty worktree and lack of a full training source manifest limit provenance. Four critical
  source hashes were frozen and unchanged, but 5B requires a clean commit and full receipt.

The full report is `results/navrl_v2_v5a_semantics_smoke_seed197/summary.md`. The excluded
full-budget step was not started. A duplicate terminal checkpoint naming defect discovered at the
endpoint was fixed after the run; the two historical files are semantically identical and remain
preserved.
