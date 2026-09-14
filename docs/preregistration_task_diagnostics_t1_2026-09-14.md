# TD-T1 preregistration — search, acquisition and reacquisition instrumentation

Written 2026-09-14, **before** the instrumentation was implemented and before any diagnostic run.
Everything fixed here is fixed before results exist.

## 0. Scope, and three name clashes

**What this is.** Evaluation-only instrumentation that records, per episode, when an existing policy
first sees the target, whether it loses it, and whether it gets it back. It runs on runs produced by
**existing checkpoints**.

**What this is not.** No policy, reward, detector, association, safety-filter, controller or
termination change. No retraining, no search heuristic, no new observation. The analysis labels
below are written to a log; they are never an input to anything. If they ever become a policy input
that is a new observation lineage and needs its own preregistration.

Three names already exist in this repository and are **not** this work:

* The master roadmap's stage codes `T0/T1/T2` are an older, different naming. This work is `TD-T1`.
* `aerial_gym/task/navrl_task/navrl_search_state.py` is a *policy-input* search feature (the S1
  lineage, arms `off/geofence/coverage/belief`). TD-T1 shares no state with it.
* The renderer track's `R1…R6` and `RC-R1…RC-R6` are unrelated.

The repository's roadmap listed this area as `BLOCKED_BY_POLICY`. The owner changed that boundary on
2026-09-14 for evaluation-only instrumentation; the change and its limits are recorded in
[the roadmap](plans/moving_target_rendezvous_master_plan_2026-09-12.md).

## 1. Chronology and the unit of analysis

Time is counted in **observation steps**: steps where the policy consumed a finite observation, the
same denominator the existing visible-fraction and first-acquisition telemetry use. The unit of
analysis is the **episode**. Steps within an episode are not independent and no per-step confidence
interval is claimed.

"Visible" means the fused perception visibility flag already computed for the actor observation
(`diagnostics["visible"]`), evaluated only on valid observation steps. The camera-only flag is
recorded beside it, never substituted for it.

## 2. Analysis labels

Per-step labels, assigned in this order, with no other state:

| Label | Rule |
| --- | --- |
| `SEARCH` | The episode has not yet had a visible observation. |
| `ACQUIRED` | The first visible observation of the episode. |
| `TRACKING` | Visible, and the episode has been acquired before. |
| `LOST_SHORT` | Not visible after an acquisition, with the current loss run ≤ **10** observation steps. |
| `LOST_LONG` | Not visible after an acquisition, with the current loss run > **10** observation steps. |
| `REACQUIRED` | The first visible observation after a loss run. |
| `TERMINATED` | The final observation step of the episode, whatever the previous label would be. |

Episode-level: `NEVER_SEEN` when no visible observation occurred at all. An episode carries exactly
one episode-level label; `NEVER_SEEN` is mutually exclusive with every other outcome-bearing label.

**The 10-step threshold** (1.0 s at the 0.1 s control interval) separates a single-frame dropout from
a real loss. It is one tenth of the tracker's 5.0 s memory saturation and is a *labelling* threshold,
not a gate: every record also stores raw loss-run lengths, so any other threshold can be applied to
the stored data afterwards.

## 3. Per-episode record

Fields, all recorded for every finished episode:

```text
env_index, episode_index, seed, density_bars, checkpoint_sha256, outcome, crash_cause
observation_steps, visible_steps, visibility_fraction
ever_acquired, first_acquisition_step, first_camera_acquisition_step
loss_count, total_lost_steps, longest_lost_steps, longest_invisible_steps
reacquired, reacquisition_count, first_reacquisition_latency_steps,
mean_reacquisition_latency_steps
track_age_at_first_acquisition_s, final_track_age_s
distance_at_acquisition_m, distance_at_first_loss_m, distance_at_first_reacquisition_m
min_distance_m, final_distance_m
label_step_counts (one count per label above), episode_label
```

Distances are the centre-to-centre range between the vehicle and the target at the named step.
A field that does not exist for an episode is written as `null`, **never** as 0 or −1 folded into a
mean: an episode that never acquired contributes to `ever_acquired` and to nothing else.

## 4. Questions

* **Q1** What fraction of episodes never acquire the target at all?
* **Q2** What is the distribution of first-acquisition latency, in observation steps, among episodes
  that do acquire?
* **Q3** Among acquiring episodes, what fraction lose the target at least once?
* **Q4** Given a loss, what fraction are followed by a reacquisition, and with what latency?
* **Q5** What fraction of episodes that end in a crash or a timeout had a `LOST_LONG` run in their
  final 20 observation steps?
* **Q6** How do Q1–Q5 vary with obstacle density?

These are descriptive. TD-T1 declares no pass/fail gate on policy quality.

## 5. Evaluation protocol

Density axis **70, 115, 160, 205** bars, using the existing held-out density protocol and its
existing seeds and checkpoints. No new test set is constructed, and no set is re-used for tuning:
nothing is tuned in TD-T1. Episodes per cell and seeds are those of the existing protocol; if a cell
cannot be run it is reported as not run, never imputed.

## 6. Verdict

The first result may be no stronger than **`SEARCH_GAP_CHARACTERIZED`**: the failure structure is
measured and reported. A search policy or architecture is **not** implemented on the strength of it.
Any architecture decision requires its own preregistration, written after these numbers exist.

## 7. Behaviour invariance requirement

The instrumentation is opt-in and defaults to off. With it on, the simulated trajectory must be
identical: the same checkpoint, seeds and configuration must produce the same trajectory digest with
the logger off and on. If the digests differ, the instrumentation is defective and no diagnostic
result is reported from it.
