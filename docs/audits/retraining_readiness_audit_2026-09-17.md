# Retraining readiness audit — 2026-09-17

Does the currently-frozen policy still make sense under today's target-motion,
observation and dynamics contract, and does that justify retraining?

**No new PPO training was run. No GPU work was performed at all.** This is the
offline analysis that the research-authority gate requires *before* any GPU work.

---

## 1. Executive verdict

```text
RETRAIN_DECISION = BLOCKED_PENDING_EVALUATION
```

The decision gate needs the frozen-policy E0/E1/E2 evaluation, and that evaluation
could not be run responsibly yet — not because of machine availability, but because
**the audit found three defects in the experiment's own design and several in the
code it would have run through.** Running first and auditing afterwards would have
produced a number nobody could interpret.

What the audit established, in order of consequence:

1. **The frozen policy was trained on the `legacy` target lineage with the `mixed`
   pattern** — not on any of E0/E1/E2. The preregistration as written had **no
   in-distribution reference arm**, so an E2 drop could not have been separated from
   "all three arms are off-distribution". Fixed by adding arm H before measuring.
2. **`time_within_approach_band_frac` is not measurable in this task.** Capture at
   0.5 m *ends the episode*, so better tracking yields fewer samples. This is not a
   missing threshold; the metric is incompatible with the termination. Removed.
3. **The training contract is not reproducible from HEAD defaults.** Fourteen knobs
   differ; ten are unchecked and four only warn. Six of them change observation
   dimension, observation normalisation, action scale or termination geometry.
4. **Nine verified code defects**, five of which can change a recorded result.

None of this says the policy is stale. It says the measurement that would decide
that was not yet trustworthy. It is now specified so that it will be.

---

## 2. Frozen policy provenance

Full table: [`frozen_policy_training_contract_2026-09-17.md`](frozen_policy_training_contract_2026-09-17.md).

```text
sha256   f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40  (re-hashed, matches)
epoch    25000    frame 102,400,000    env_state 127 keys
target   cfg_target_motion_model = symmetric_local_steer_v2_heading_continuity90
         cfg_target_pattern      = mixed
         => legacy lineage: no acceleration bound, no turn-rate bound,
            90 deg heading-continuity preference, wall reflection, bar push-out
```

The checkpoint is silent on `success_radius`, reward composition, robot asset hash,
source commit, target dynamics bounds, observation dimension and the heading-validity
threshold. Those rows are reconstructed from HEAD and labelled as such.

---

## 3. Training contract vs current runtime

Machine-readable: [`training_contract_vs_current_runtime_2026-09-17.json`](training_contract_vs_current_runtime_2026-09-17.json),
produced by `tools/audits/extract_runtime_contract.py` (static extraction; no task
instantiation, no CUDA context, so it reproduces on a GPU-less machine).

```text
checkpoint cfg_ keys      112
DEFAULT_PRESERVED          15
CHANGED_BUT_OPT_IN         14      <- every one is env-var controlled
NOT_ATTESTABLE             83      <- computed or nested; not guessed
```

The fourteen, with what breaks if an evaluation forgets them:

| Key | trained | HEAD default | guard | consequence |
|---|---|---|---|---|
| `cfg_lidar_hbeams` | 72 | 36 | warn-only | **observation dimension** |
| `cfg_lidar_max_range` | 12.0 | 4.0 | warn-only | observation scale, sensing horizon |
| `cfg_max_velocity` | 2.5 | 2.0 | warn-only | **action scale AND obs normalisation denominator** |
| `cfg_yaw_rate_max` | 3.0 | 2.5 | warn-only | yaw action scale |
| `cfg_episode_len_steps` | 600 | 300 | **none** | every rate metric's denominator |
| `cfg_oob_margin` | 1.0 | 0.5 | **none** | termination geometry |
| `cfg_density_final` | 205 | 150 | none | obstacle density |
| `cfg_general_goal_dist_min/max` | 6 / 28 | 4 / 18 | none | task difficulty |
| `cfg_target_speed_min/final` | 0.3 / 1.5 | 0.0 / 0.0 | none | target speed distribution |
| `cfg_target_speed_ramp_epochs` | 300 | 3000 | none | curriculum |
| `cfg_density_check_eps` | 16384 | 2048 | none | density gate sample size |
| `cfg_general_train` | True | 0 | none | task family |

The simulator's own guard is explicitly documented as *"warn, never override: an eval
may deliberately change these"*. That is the right default for the simulator and the
wrong one for a preregistered matched comparison, so this audit adds a strict gate
(`tools/audits/check_eval_condition_contract.py`, exit 2 on `CONFOUNDED`) rather than
changing the simulator's behaviour.

Verified against a simulated HEAD-defaults launch: the gate flags 6 CRITICAL
confounds and refuses. Verified against an identical environment: `MATCHED`.

---

## 4. Target-motion shift

Per-lineage algorithms are documented in
[`../target_motion_algorithm_2026-09-17.md`](../target_motion_algorithm_2026-09-17.md).
What matters here is the distance from the *training* lineage:

| | trained (legacy) | TM-E1 `cv` | TM-E2 `bounded` |
|---|---|---|---|
| pattern | mixed (cv + waypoint) | cv only | waypoint only |
| acceleration bound | **none** | none | yes |
| turn-rate bound | **none** (90° preference) | none | yes |
| wall handling | reflection | reflection | feasibility-screened |
| obstacle handling | post-step push-out | push-out | receding-horizon screen |
| global route | none | none | none |

Moving to E2 changes the motion law, the pattern *and* adds two dynamic bounds
simultaneously. It is a lineage change, not a difficulty dial. Arm H exists so that
this distance is measured rather than assumed.

### 4.1 Heading-validity threshold — `ASSUMED`, and the gap is 10 000×

The checkpoint has **no** `heading_valid` key. `resolve_heading_valid_speed_contract`
classifies this as `ASSUMED_PRE_KEY_DEFAULT`, and the source comment states that
pre-key lineages ran an inline **1e-05 m/s** epsilon against today's **0.10 m/s**.

The threshold decides when residual speed counts as a heading of travel, so it changes
what every moving-target metric means. It is held identical across arms (so it cannot
confound the contrast) and recorded as `ASSUMED` in every result. **0.10 is not
back-dated onto this checkpoint.**

---

## 5. Observation shift and checkpoint compatibility

Observation normalisation **is** used (`normalize_input: True` for both actor and
central value) and lives in the checkpoint's `model` state dict as
`running_mean_std.{running_mean,running_var,count}` — shape `(898,)` actor, `(906,)`
critic. It is not in `env_state`, which is why no `norm` key appears there; that
absence is not evidence that normalisation is unused.

Dimension is derived, not literal: `4·72 + 5·8·12 + 5·10 + 5·16 = 898`, `+8` privileged
`= 906`. Feature order is fixed in exactly two places that must agree — the producer
(`navrl_perception.py`, `obs_parts` concatenation) and the consumer
(`navrl_transformer_network.py`, offset slicing).

**Failure behaviour is asymmetric, and this is the important part:**

```text
wrong WIDTH      -> hard RuntimeError   (strict load_state_dict)
right width,
wrong SEMANTICS  -> logger.warning only, run continues
```

There is **no** check anywhere that compares a restored checkpoint's observation
dimension against the runtime's. A standalone preflight that *would* raise
(`navrl_checkpoint_preflight.py`) exists but is **not invoked by the evaluation path**
the frozen checkpoint runs through. So a same-width semantic mismatch — exactly what
the fourteen drifted knobs could produce — is silent apart from a log line.

This is the single strongest argument for the strict gate in §3.

---

## 6. Dynamics and control audit

**Action semantics** (task docstring, verified):

```text
action[0:2]  vehicle-frame velocity, INDEPENDENT PER-AXIS limit +/- max_velocity
             => per-axis 2.5 m/s, attainable horizontal NORM 2.5*sqrt(2) = 3.5355 m/s
action[2]    retained for checkpoint compatibility, OVERWRITTEN by the altitude PI loop,
             and still persisted into the prev-action observation
action[3]    yaw rate
```

The unused-but-observed z component is recorded as **design debt for a future training
lineage**, not changed here: altering it would break checkpoint compatibility, which is
the one thing this evaluation depends on.

**TM-E2 dynamics consistency** (arithmetic, not a claim about realism):

```text
v * omega = 1.5 m/s * 150 deg/s = 3.9270 m/s^2   vs envelope 4.0 m/s^2
headroom  = 1.83 %
```

Consistent, but only just: a coordinated turn at maximum speed and maximum turn rate
sits essentially on the acceleration bound, leaving almost no budget for simultaneous
speed change. This is a 2-D bounded kinematic envelope and **not** rigid-body UAV
dynamics.

**Command-vs-actual telemetry** required before any outcome is interpreted, so that a
policy failure is not confused with a controller limitation: commanded vs actual
horizontal speed, heading error, yaw-rate request vs actual, action edge95/98/99,
governor intervention rate, acceleration and turn-rate saturation. No controller is
modified by this work.

---

## 7. Target generator validity

Before any E2 outcome is read, the generator itself must be shown valid: obstacle
penetration, wall violation and infeasible-step rates; realized speed distribution;
acceleration and turn rate; stationary fraction; candidate-switch rate; safe-prefix
fallback rate; no-complete-lookahead rate; waypoint completion and resampling; path
length; heading chattering and period-2 candidate switching.

**A non-zero validity gate means the outcome is not interpreted.** A comparison against
a target generator that is violating its own contract is not a difficulty comparison.

The physical lineage keeps its recorded verdict `FAIL_ROUTE_MECHANISM`. It is not
presented as ready to be the canonical target, and it is not modified or retrained here.

---

## 8. Termination and reward consistency

`navrl_task.py` is explicit: *"Interception semantics (always on): capture ends the
episode"*, with `success_radius = 0.5 m` applied as a swept capture test.

**Verdict: `A. CLOSE_APPROACH_TASK`.** Keep the historical 0.5 m termination and narrow
the continuous-tracking claim accordingly. The consequence is not cosmetic: any metric
integrated over a "following phase" is conditioned on *not having succeeded yet*, so
better tracking produces fewer samples and the statistic is biased by construction.
That is why `time_within_approach_band_frac` was removed rather than given a threshold.

A genuine continuous-following task would need different termination semantics and
therefore a separate task lineage (`NavRL-ContinuousFollow` alongside
`NavRL-CloseApproach`). **Not implemented here, and deliberately not bundled with any
target-behaviour change.**

---

## 9. Code defects

Nine verified findings. Each was re-read and quoted at source before being recorded
here; none is reported on a subagent's say-so alone. **No silent fixes were made** —
nothing below is repaired in this commit, because several would change what recorded
results mean and that requires impact analysis first.

| # | Location | Defect | Class | Can change a recorded result |
|---|---|---|---|---|
| D1 | `navrl_task.py:8184-8252` | Legacy target push-out is pure displacement, then `target_vel_w = Δpos/dt` — so realized target speed can **exceed the labelled episode speed** at push-out events. Reflection counters increment only under `_bulk_eval_mode`, so training-time corrections are uninstrumented. The bounce jitter draws from the **global** torch RNG conditionally on bar contacts, desynchronising subsequent draws. | SCIENTIFIC_CONFOUND | **YES** — and this is the lineage the frozen policy trained on |
| D2 | `navrl_task.py:9289` | Results receipt records `target_pattern` with default `"static"`; the simulator's real default is `"mixed"`. `env_state` is correct; only the receipt lies. | BUG | **YES** — a moving-target eval archives as static |
| D3 | `navrl_task.py:2224` | Results receipt records `target_speed_mps` from `NAVRL_TARGET_SPEED` default `0`, but that maps to `speed_fixed = -1.0` (disabled); real speed comes from the curriculum. | BUG | **YES** — curriculum-speed eval archives as 0 m/s |
| D4 | `navrl_task.py:3440` vs `navrl_bars_env.py:96` | Surface clearance attested as `0.0` while the arena is built with `0.45`. The drift guard recomputes the *same wrong value*, so it can never fire. | BUG | **YES** — undetectable geometry drift |
| D5 | `navrl_perception.py:960` | `int(round(latency_s / step_dt))`: a 0.05 s latency becomes **0 steps** (`round(0.5) == 0`, banker's rounding) while `env_state` records 0.05 s. | BUG | **YES** — a latency arm is physically a zero-latency arm |
| D6 | `navrl_task.py:5918` (+3 sites) | `nan_to_num(nan=1.0, posinf=1.0, neginf=1.0)` maps a non-finite LiDAR pixel to **full range = "nothing there"**, the most favourable value, feeding both the safety reward and the governor. The `-max_range` sentinel maps to the opposite meaning. | SCIENTIFIC_CONFOUND | **YES** |
| D7 | `speed_governor.py:44` | `free_speed_cap_mps = sqrt(2)*2.5` hard-coded while `max_velocity` defaults to `2.0` (attainable norm 2.83). At HEAD defaults the free cap sits above the attainable speed, so the governor never binds in open space. | SCIENTIFIC_CONFOUND | **YES at HEAD defaults**; correct for this checkpoint, which trained at 2.5 |
| D8 | `navrl_perception.py:1319-1324` | `_detector_noise_range_ar` AR(1) state is never reset per episode; the previous episode's terminal range error bleeds into the next in the same slot. | BUG | **NO by default** — both gating knobs default to 0.0; affects detector-noise arms only |
| D9 | `navrl_perception.py:1960` | LiDAR→target range adds the **camera** sphere radius `0.15` to a surface range from a LiDAR that injects `0.20`. | SCIENTIFIC_CONFOUND | **YES** — every LiDAR-fallback correction biased 0.05 m |

Plus a family of perception knobs (`detector_noise_*`, `latency_*`, `pose_noise_*`,
`lidar_assoc_*`) that change what the actor observes and appear in **neither**
`env_state` **nor** the eval receipt: two arms differing only in these are
bit-indistinguishable in their stored provenance.

**Impact triage, deliberately not acted on here:** D1 touches the training
distribution of the frozen checkpoint itself, so any fix creates a new lineage rather
than amending the old one. D2/D3 mislabel receipts without changing the physics, so
the affected receipts need an erratum, not a re-run. D4/D5/D6/D7/D9 need per-result
impact analysis before anything is rewritten. **No historical result is overwritten.**

---

## 10. Frozen-policy evaluation — plan, not results

State: **`READY_NOT_RUN`**.

The preregistration was amended *before any measurement*
([`../prereg_2026-09-17_target_motion_complexity_e0_e1_e2.md`](../prereg_2026-09-17_target_motion_complexity_e0_e1_e2.md),
Amendment 1): arm H added, `time_within_approach_band_frac` removed, and every
previously-unfrozen number fixed.

```text
arms              H (historical, in-distribution) / E0 / E1 / E2
densities         70 / 115 / 160 / 205
seeds             4101, 4102, 4103
cells             48
episodes per cell exactly 2048        (16 x 128-env batches; removes the historical
                                       2,049 tail artifact at equal sample size)
primary           min_relative_distance_m, capture_rate
guards            crash_rate, timeout_rate
validity gates    target penetration / wall violation / infeasible step  == 0
CI                Wilson per cell; seed-paired BCa bootstrap for arm contrasts
correction        Holm-Bonferroni over the 3 primary contrasts vs arm H;
                  per-density contrasts are exploratory and uncorrected
gate              tools/audits/check_eval_condition_contract.py, exit 2 == do not run
```

Analysis order is fixed: integrity → target mechanism → matched-condition check →
policy behaviour → outcome → decision. **Not results first.**

Interpretation is fixed too: this measures **frozen historical-policy generalization
under changed target-motion distributions**. It is not a training comparison, and an
E2 drop is not evidence that "E2 is intrinsically harder" until distribution shift,
target dynamics, occlusion change, relative-velocity change, route geometry and
controller demand have been separated.

**Blocked by:** `tools/check_research_authority.py` — *"offline bottleneck analysis and
a new preregistration are required before any GPU work"*. This audit is that offline
analysis and the amended preregistration is that preregistration; authorising the GPU
run is the operator's call, not this document's.

---

## 11. Retraining decision

```text
DECISION = PENDING_EVALUATION
```

No case A–E can be selected without the measurement. Recorded here so the decision
cannot drift afterwards:

| Case | Condition | Decision |
|---|---|---|
| A | small E0/E1/E2 differences vs H, mechanisms valid, controller diagnostics similar | `NO_RETRAIN_REQUIRED_FOR_CURRENT_SCOPE` |
| B | E2 degrades materially, mechanism valid, controller not dominant | `RETRAINING_JUSTIFIED` |
| C | E2 degradation **and** target-mechanism pathology | `DO_NOT_RETRAIN; FIX_TARGET_GENERATOR_FIRST` |
| D | loss dominated by command-vs-actual / governor saturation | `DO_NOT_ATTRIBUTE_TO_POLICY; CONTROL_STACK_REVIEW` |
| E | perception loss dominates | `DO_NOT_RETRAIN_NAVIGATION_YET; FREEZE_PERCEPTION_CONTRACT_FIRST` |

Given §9, **case C is a live possibility that the design must be able to detect** —
which is why the target-generator validity gates run before any outcome is read.

---

## 12. Conditions before any new training

Every one of these must be frozen first. Three are not:

| Contract | State |
|---|---|
| final controller | frozen |
| final safety filter | frozen |
| final observation contract | **not frozen** — see §5 and the unrecorded perception knobs in §9 |
| final perception contract | **not frozen** — live RGB→policy `NOT_TESTED`, true metric range `BLOCKED`, persistent ID `BLOCKED`, P10 `INCONCLUSIVE` |
| final target behaviour contract | **not frozen** — D1 is in the legacy lineage; TM-E3/E4 PLANNED |
| final robot dynamics contract | **not frozen** — robot asset hash absent from this checkpoint |

If retraining is later justified, the design is fixed in advance and **not executed
here**: arm H (historical source policy) vs arm E2 (same architecture, observation,
reward, controller, safety filter and arena curriculum; target behaviour = E2), at
least 3 training seeds, cross-evaluated `policy_H × {E0,E1,E2}` and
`policy_E2 × {E0,E1,E2}` so that E2 specialisation, general improvement and
catastrophic forgetting can be told apart. Warm-start and fresh training are **separate
questions** and are not mixed; reward is **not** retuned in the same step as the target
distribution, or neither change is identifiable.
