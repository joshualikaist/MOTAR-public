# P10 seed replication — preregistration, 2026-09-10

Registered before any training was launched. P10's readaptation result rests on **one** training seed,
and this repository has already withdrawn one claim (C3) that a single training seed appeared to support.
This runs the same contract on two further training seeds and fixes the decision rule in advance.

## The claim under test

P10 (training seed 811) reported the preregistered primary estimand
`adapted-P9 − source-P9` capture as **+1.41 pp, 95% CI [−0.38, +3.21]**, pooled over evaluation seeds
541 and 547, with both per-seed point estimates positive (+1.795, +1.036) and neither CI excluding zero.
The claim is that **readaptation under the measured perception error recovers capture**.

## Design — identical to P10 except the training seed

- Training seeds **857** and **863**. Neither has been used for training or evaluation in this repository.
- Warm start, contract, injector, governor, arena, 1,000 epochs, 205 bars: exactly
  `train_navrl_v2_p10_empirical_readapt.sh` with `P10_SEED` set. Nothing else changes.
- Evaluation seeds stay **541 and 547**, so the training seed is the only varying factor and the new
  campaigns are directly comparable with P10's.
- Four cells per evaluation seed: source-clean, source-P9, adapted-clean, adapted-P9. 2,049 requested
  episodes, deterministic actions, exact-600 semantics, `riskcap`.
- The frozen-policy (`source`) cells are **re-run inside each campaign**, not borrowed from P10. The
  primary estimand is always computed within one campaign, so a determinism failure cannot silently
  distort it.

## Determinism check, reported either way

The source cells do not depend on the training seed, so they must reproduce P10's counts exactly. Whether
they do is reported. If they do not, the replication still stands — every estimand is within-campaign —
but the discrepancy is reported as a finding about reproducibility, and this repository has already seen
one environment-mismatch reproducibility failure.

## Decision rule — both parts fixed now

**R1, sign rule (the R-C precedent).** Count the new training seeds whose within-campaign pooled primary
estimand is positive.

**R2, seed as the replication unit (the arc-geometry precedent).** Across the three training seeds
811, 857 and 863, take the mean of the per-seed pooled primary estimand with a two-sided 95 % CI from
Student's t at df = 2.

| outcome | condition |
|---|---|
| `REPLICATED` | R1 gives 2 of 2 positive **and** R2's CI excludes zero with a positive mean |
| `WITHDRAWN` | R1 gives 0 of 2 positive |
| `INCONCLUSIVE` | anything else, including 2 of 2 positive with an R2 CI spanning zero |

No margin is adjusted afterwards. No seed is re-run because its result is unwelcome. No fourth seed is
added to move an interval. If the outcome is `INCONCLUSIVE`, that is the reported outcome and the paper
says the readaptation claim is directional but not established.

## Reported alongside, never gating

Crash and timeout differences; clean retention (`adapted-clean − source-clean`), which P10 measured at
−1.25 pp and which makes readaptation a trade rather than a free recovery; and the cost of P9 error on
each policy separately, which P10 measured at −4.57 pp on source and −1.90 pp on adapted.

## Scope this cannot exceed

Three training seeds at one density, one arena, one injector, in simulation. It is not a seed-general PPO
claim, and it says nothing about real flight.
