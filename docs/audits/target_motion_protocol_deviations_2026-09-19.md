# Target-motion evaluation — protocol deviations — 2026-09-19

Four deviations between the preregistration and the executed campaign. Each is recorded with its
cause and its consequence. None is repaired by re-running: the raw artifacts are frozen and no
further GPU evaluation was authorised.

Preregistration audited: `docs/prereg_2026-09-17_target_motion_complexity_e0_e1_e2.md` at commit
`096eee5`, sha256 `7d8744c469c59be9ab568916bcd8cc4f70da386fef892b2e049363c62a5464a4`.

Raw artifacts: `results/target_motion_e0_e2_evaluation_2026-09-18/`, immutability verified —
**144/144 files match `RAW_MANIFEST.json`**.

---

## D-1 Episode count: 98,319 recorded against 98,304 preregistered

**Status: EXPLAINED, bounded, does not affect any conclusion.**

Amendment 1 §A1.3 fixed "episodes per cell = 정확히 2048" (exactly 2048) and justified it:

> 직전 held-out 설계는 `actual >= 2,049`를 요구해 9/9 cell이 2,049가 됐는데, 그 +1은 128-env
> tail이 만든 잔여물이다. 2048로 고정하면 그 **tail 비대칭이 구조적으로 사라지고** 표본 크기는
> 실질적으로 동일하다.

**That expectation was not met.** The +15 distribution:

| excess | cells |
|---|---|
| 0 | 35 |
| +1 | 11 |
| +2 | 2 |

**Cause — vectorised tail overshoot.** `aerial_gym/task/navrl_task/navrl_task.py:9750-9751`:

```python
total = self._succ_agg + self._crash_agg + self._to_agg
if total >= self._progress_log_interval:      # == 2048
```

The stop condition is a `>=` test evaluated **once per vectorised step**, after every environment
that finished in that step has been aggregated. With 128 parallel environments a counter at 2047
becomes 2049 when two environments terminate in the same step. Fixing the target at 2048 changes
which number is compared; it does not cap the counter, so the tail asymmetry survives. The
theoretical maximum overshoot is `num_envs - 1 = 127`; the observed maximum is 2.

**Causes ruled out, with evidence:**

| Candidate | Ruled out because |
|---|---|
| Canary contamination | The four promoted canary cells are the `160bars/seed4101` cells. Three of the four have excess 0, and cells never run as canaries do carry excess. Excess does not track canary provenance. |
| Duplicate terminal record | The outcome triple (`captured + crash + timeout`) equals `actual_episodes` in **48/48** cells. |
| Resume / retry duplication | The launcher skips any cell that already has `result.json` + `receipt.json`; a retry would duplicate a whole cell (+2048), not +1. |
| Deficit anywhere | No cell has fewer than 2048 records. |

**Consequence — bounded.** Assuming all 15 excess episodes were captures, or all were failures,
gives the worst-case envelope on each arm's pooled capture rate:

| arm | excess | max possible shift |
|---|---:|---:|
| H historical | 4 | 0.0143 pp |
| E0 static | 4 | 0.0137 pp |
| E1 CV | 2 | 0.0069 pp |
| E2 obstacle-aware | 5 | 0.0182 pp |

The largest possible distortion is **0.018 pp**, against a smallest observed effect of 0.76 pp —
roughly two orders of magnitude smaller. No ordering, sign or conclusion can change.

---

## D-2 The exact-2048 primary view cannot be constructed

**Status: NOT_CONSTRUCTIBLE from frozen artifacts.**

Restoring a prereg-compliant 2048/cell dataset requires selecting the first 2048 episode records
per cell. **The frozen bulk-eval export contains no per-episode records at all.** `result.json`
holds only aggregates (`outcome`, `strata`, `target_motion`, `speed_governor`, `action`,
`crash_causes`, `condition`, `search_state`); there is no `episode_id`, no `env_id` and no
per-episode array.

Per-episode rows exist only in the task's *general* evaluation mode
(`general_trial_records`, `navrl_task.py:2176`), which this campaign did not use, and
`NAVRL_TRAJECTORY_DIGEST` was not set by the launcher.

Constructing the view would require re-running the evaluation, which is not authorised. No
synthetic 2048/cell view was fabricated. The bounded envelope in D-1 is the mitigation.

---

## D-3 `min_relative_distance_m` was never recorded

**Status: NOT_RECORDED. Primary metric unavailable.**

Amendment 1 §A1.2 promoted `min_relative_distance_m` to **primary**, described as
"모든 episode에서 정의됨" (defined on every episode), sourced from `ep_min_goal_dist`.

The quantity exists per-environment at runtime (`navrl_task.py:772`, updated at `:4996`), but the
export accumulates it **only over non-crash finished episodes**:

```python
nocrash = finished & ~(crashes > 0)
if nocrash.any():
    self._mindist_sum += float(torch.sum(self.ep_min_goal_dist[nocrash]).item())   # :9745
    self._nc_agg      += int(torch.sum(nocrash).item())                            # :9746
```

There is no all-episode accumulator anywhere, and per-episode records are not exported (D-2).
Therefore the preregistered primary is **irrecoverable from frozen data**.

`closest_nocrash_mean_m` is **not** a substitute: it conditions on non-crash episodes, which is a
selection on outcome. It is relabelled throughout as:

```text
POST_HOC_CONDITIONAL_DIAGNOSTIC
```

`capture_rate`, the other Amendment 1 primary, **is** available and carries the primary analysis.

---

## D-4 The 5 pp materiality threshold was never preregistered

**Status: NOT_PREREGISTERED. Prior claim withdrawn.**

The preliminary report wrongly called the 5 pp rule preregistered and issued the verdict
`NO_RETRAIN_NEEDED`. **Both are withdrawn.**

Search evidence:

| Search | Result |
|---|---|
| `git log --all -S"material_threshold"` | **0 commits** |
| `git log --all -S"NO_RETRAIN_NEEDED"` | **0 commits** |
| prereg §6 Decision rule | four rules: validity gate, confounding check, CI-based comparison, no deployment extension. **No effect-size threshold. No retraining vocabulary.** |
| Amendment 1 §A1.3 | fixes arms, densities, seeds, cells, episodes/cell, CI method, seed aggregation, multiple-comparison correction. **No materiality threshold.** |

The only occurrences are the untracked analysis tool written during the analysis session
(`material_threshold = 0.05`) and the WORKLOG entry drafted from its output. Both post-date
measurement.

**The preregistration was not edited retroactively.** Any 5 pp rule is retained only as
**POST_HOC DECISION CONTEXT** and is never described as preregistered. The verdict is narrowed to
`NO_RETRAINING_JUSTIFIED_FOR_E2_TARGET_MOTION_SHIFT`, which rests on the measured sign of the E2
effect rather than on any threshold.

---

## What the deviations do not change

D-1 is bounded at 0.018 pp. D-2 and D-3 remove a primary metric and a robustness view but do not
alter the `capture_rate` analysis, which was computed from the preregistered seed-level
aggregation. D-4 changes the **wording and scope of the verdict**, not the measured effects. The
arm ordering, all twelve contrasts and their intervals are unchanged from the raw data.
