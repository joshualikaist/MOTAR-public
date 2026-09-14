# P10 seed replication — `INCONCLUSIVE`

Two further training seeds were run under the contract fixed in
[the preregistration](../../docs/preregistration_p10_seed_replication_2026-09-10.md), committed at
`d0e1aea` before any training started. Receipt: `summary.json`. Re-derive with `analyse.py`.

**The readaptation claim is neither replicated nor withdrawn. It is directional and not established,
and the paper has to say so.**

## The two rules, applied as written

| rule | requirement | result |
|---|---|---|
| R1 sign rule | 2 of 2 new seeds positive for `REPLICATED`; 0 of 2 withdraws | **1 of 2** |
| R2 seed-level t (df = 2) | interval excludes zero | mean **+0.73 pp**, CI **[−1.04, +2.50]** |

Neither `REPLICATED` (needs 2 of 2 and an interval clear of zero) nor `WITHDRAWN` (needs 0 of 2).

## Per training seed

Capture percentage points, each computed inside its own campaign.

| training seed | primary, adapted−source under P9 | clean retention | P9 cost on source | P9 cost on adapted |
|---|---:|---:|---:|---:|
| 811 (original) | +1.41 | −1.25 | −4.57 * | −1.90 * |
| 857 | +0.78 | −1.08 | −4.57 * | −2.71 * |
| 863 | −0.01 | −0.74 | −4.57 * | −3.84 * |
| mean | **+0.73** | −1.02 | −4.57 | −2.82 |

`*` marks a 95 % CI excluding zero. The primary estimand's interval contains zero for every seed
individually and for the seed-level mean.

## What the three seeds do agree on

**The measured perception error costs the frozen policy 4.57 pp of capture**, identically in all three
campaigns, with an interval clear of zero. That is not a replication artefact: the frozen-policy arms are
the same computation three times, and they came out identical to the episode.

**Readaptation always leaves a residual cost.** Its size grows across the three seeds — −1.90, −2.71,
−3.84 pp — and every one of those intervals excludes zero. So readaptation never fully recovers what the
error costs, and how much it recovers is exactly what varies between seeds. That variation is the finding.

**Readaptation costs clean performance in all three**, by −1.25, −1.08 and −0.74 pp. It is a trade, not a
free recovery, which was already visible on one seed and is now visible on three.

## Reproducibility, checked and passed

The frozen-policy cells do not depend on the training seed, so both new campaigns re-ran them from
scratch. All eight reproduced P10's counts **exactly** — captured, crash, timeout and episode totals. This
repository has previously had a result fail to reproduce because of an environment mismatch, so this was
worth confirming rather than assuming, and the design put every estimand inside its own campaign so that a
failure here could not have distorted the comparison silently.

## What this changes in the write-up

The P10 sentence cannot be "readaptation recovers the loss". Supportable instead:

> The measured perception error costs the frozen policy 4.57 pp of capture. One thousand epochs of
> readaptation reduces that residual cost, by 2.67, 1.86 and 0.73 pp across three training seeds, while
> costing 0.7 to 1.3 pp of clean-condition capture. The direct adapted-minus-source contrast under error
> is +0.73 pp on average with a seed-level interval of [−1.04, +2.50], so the net benefit is not
> established at three seeds.

No margin was moved, no seed was re-run, and no fourth seed was added to narrow the interval. The
preregistration forbade all three.

## Scope

Three training seeds, two evaluation seeds each, 205 bars, one arena, one injector, in simulation. Not a
seed-general PPO claim and not a real-flight result.

## Reproducing

```bash
cd aerial_gym/rl_training/rl_games
P10_SEED=857 bash train_navrl_v2_p10_empirical_readapt.sh    # and 863
cd -
PY=/home/fair/miniconda3/envs/aerialgym/bin/python
for s in 857 863; do for e in 541 547; do
  $PY tools/run_navrl_filter_grid.py \
     results/perception_p10_seed_replication_2026-09-10/spec_s${s}_eval${e}.json \
     results/perception_p10_seed_replication_2026-09-10/grid_s${s}_eval${e} evaluate
done; done
$PY results/perception_p10_seed_replication_2026-09-10/analyse.py --repo . \
   --root results/perception_p10_seed_replication_2026-09-10 --new-seeds 857 863 \
   --output results/perception_p10_seed_replication_2026-09-10/summary.json
```
