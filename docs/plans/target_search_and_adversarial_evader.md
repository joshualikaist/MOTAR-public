# MOTAR blind search, reacquisition, and autonomous-evader planning brief

Status: planning input; no implementation or performance claim  
Prepared: 2026-09-03  
Audience: the next implementation/planning agent (including Claude)

## Executive decision

MOTAR currently solves short target occlusions with sensor fusion and a constant-velocity tracker,
but it does not explicitly plan where to search before first acquisition or after a stale track.
Its target is a gravity/contact-enabled 6-DoF rigid body, yet the target's behavioral command is a
scripted planar CV/random-waypoint reference. It does not observe the pursuer, optimize an evasion
reward, or learn an opponent model.

The next research lineage should preserve the current reproducible target and held-out benchmark as
Tier 0, then change two axes separately:

1. add explicit `SEARCH -> TRACK -> REACQUIRE` state/belief and measure first-acquisition failures;
2. add increasingly autonomous targets: reactive rule-based first, then learned evaders trained
   against a historical opponent population.

Do not combine both axes in the first causal experiment. Do not describe the current target as an
intelligent or adversarial evader.

## 1. What the current pursuer does when the target is hidden

### 1.1 Short occlusion after acquisition

The perception module maintains a six-state constant-velocity Kalman track:

- world position `(x, y, z)`;
- world velocity `(vx, vy, vz)`;
- position/velocity covariance;
- time since the last measurement and an active flag.

Camera measurements initialize/correct the track. On camera-missed frames, sensor-gated LiDAR
association can also correct it. While hidden, the filter predicts with constant velocity and grows
its covariance. The policy-facing target token contains relative predicted position/velocity,
normalized covariance, camera/LiDAR confidence, track age, and target radius.

Relevant code:

- `aerial_gym/task/navrl_task/navrl_perception.py`,
  `BatchedConstantVelocityTracker.step/correct` (around lines 648-701);
- the same file, `_target_features` (around lines 1821-1856);
- `aerial_gym/config/task_config/navrl_task_config.py`, `vision.tracker_memory_s = 5.0`.

The structured policy has five robot, target, and obstacle history samples. Samples are refreshed at
0.5 s intervals, so the window spans about two seconds. The four-layer Transformer consumes this
fixed window but declares `is_rnn() == False`; it has no memory beyond the supplied tokens.

Relevant code:

- `aerial_gym/task/navrl_task/navrl_perception.py`: `ROBOT_HISTORY`, `TARGET_HISTORY`,
  `OBSTACLE_HISTORY`, and `history_stride`;
- `aerial_gym/rl_training/rl_games/navrl_transformer_network.py`, especially `is_rnn` and token
  parsing.

Interpretation: a brief loss is handled as "continue toward the extrapolated track." This is useful
for smooth CV-like motion but is not a deliberate occlusion-clearing or information-gathering plan.
It will degrade quickly against a target that intentionally turns after breaking line of sight.

### 1.2 Before first acquisition or after track expiry

The tracker starts inactive. After more than five seconds without an accepted correction it becomes
inactive again. `_target_features` multiplies the complete target token by the active flag, so the
policy receives zeros rather than a target-direction belief.

In the corrected route-off held-out policy, `geofence_actor` is false. The actor does not receive
world XY, arena side, distance to the four task boundaries, a visited map, target-location
probability, or episode progress. The arena edge is not physical geometry and therefore produces no
LiDAR return. Blind motion is consequently an implicit behavior learned from training statistics,
not coverage planning.

The earlier diagnosis is recorded in:

- `results/navrl_ref5in_oob_exit_forensics_seed367/analysis.md`;
- `results/navrl_ref5in_active_search_geofence_seed367/summary.md`.

The mapped-geofence A/B improved one old 1-bar condition from 39.04% to 85.75% capture and reduced
never-acquired OOB, but its masked mechanism test did not pass. Its recorded conclusion is
`PASS_MECHANISM_UNRESOLVED`, not a completed solution and not a result that can be transferred to
the current corrected route-off checkpoint.

## 2. Current held-out evidence for blind-search failure

Source of truth:

- `results/navrl_corrected_nonoverlap_physical_off_heldout_seed313/70bars.json`;
- `results/navrl_corrected_nonoverlap_physical_off_heldout_seed313/145bars.json`;
- `results/navrl_corrected_nonoverlap_physical_off_heldout_seed313/summary.md`.

| condition | episodes | never acquired | share | acquisition facts |
|---|---:|---:|---:|---|
| 70 bars | 2,049 | 175 | 8.54% | 169 crashed, 6 timed out; all 5 OOB exits were never-acquired |
| 145 bars | 2,049 | 406 | 19.81% | 400 crashed, 6 timed out |

Every captured episode acquired the target. At 70 bars, captured episodes had a median first-visible
step of 23 and averaged 2.25 visible-to-hidden transitions; at 145 bars the corresponding values
were 31 and 2.78. These observations are consistent with short gaps being survivable and blind
initial search being a separate bottleneck, but they do not by themselves prove the causal value of
any proposed search module.

The current held-out headline remains 83.70% capture at 70 bars and 65.54% at 145 bars. The decline
is dominated by bar contact overall. The never-acquired split above identifies an additional,
specific failure channel; it must not be presented as the sole cause of all density loss.

## 3. Current target dynamics and effective degrees of freedom

### 3.1 Physical layer: genuine 6-DoF rigid-body dynamics

With `NAVRL_TARGET_DYNAMICS=physical`, the target is one PhysX rigid body with:

- 3D position/orientation and linear/angular velocity;
- gravity and contact;
- four bounded motor thrusts reduced to a body-frame wrench;
- first-order motor response (`tau = 0.04 s` by default);
- roll, pitch, and yaw torque authority;
- a 1.20 kg synthetic ref5in design point and 45-degree tilt envelope.

The controller and motor allocation live in
`aerial_gym/task/navrl_task/physical_target.py`. The declared parameters live in
`aerial_gym/config/task_config/navrl_task_config.py::target_motion`.

This supports the narrow statement "the target is simulated as a physical 6-DoF multirotor." It
does not support the stronger statement "the target has a 6-DoF behavioral policy."

### 3.2 Behavioral layer: scripted, planar, and non-reactive

The corrected held-out contract uses:

- `target_motion_model = physx_ref5in_6dof_motor_wrench_v2_same_substep`;
- `target_pattern = mixed`, choosing CV or random waypoint 50:50 per episode;
- speed `U[0.3, 1.25] m/s` (the raw redundant 1.5 serializer field has a documented erratum);
- `target_route_mode = off`;
- fixed altitude of 1 m;
- yaw automatically aligned to the planar velocity heading.

The planner supplies only XY desired velocity. The physical controller supplies altitude hold and
the attitude/motor response needed to realize it. The target cannot voluntarily climb, descend,
choose independent yaw, perform a strategic roll/pitch maneuver, or select an action in response to
the pursuer. It uses bar geometry for safe local steering, but that is collision avoidance rather
than opponent awareness.

Relevant code:

- `aerial_gym/task/navrl_task/navrl_task.py::_sample_target_motion`;
- `aerial_gym/task/navrl_task/navrl_task.py::_advance_target`;
- `aerial_gym/task/navrl_task/physical_target.py::set_command`.

Operationally, an autonomous target requires a policy `pi_e(a_e | o_e, h_e)`, an observation
contract, action authority, reward, and memory/opponent model. The current target has none of these;
it has a sampled trajectory generator and a tracking controller.

## 4. External primary-source comparison

### Initial search and reacquisition

1. **E-VAT (IEEE RA-L 2022)** explicitly switches between exploration and tracking so the target
   need not start in the camera's immediate vicinity. Its official implementation documents a
   target probability/obstacle map and separate exploration/tracking behavior:
   <https://github.com/isarlab-department-engineering/e-vat>
2. **MATT-Diff (L4DC 2026)** represents target estimates as Gaussian densities and learns
   exploration, tracking, and reacquisition behavior from frontier, uncertainty-aware RRT*, and
   time-based hybrid experts:
   <https://proceedings.mlr.press/v331/liu26a.html>
3. **OA-VAT (CVPR 2026)** combines a confidence-aware Kalman tracker with a learned
   occlusion-aware trajectory planner that moves around obstacles to restore visibility:
   <https://arxiv.org/abs/2604.21453>
4. Earlier end-to-end VAT showed that a ConvNet-LSTM policy can sometimes restore tracking after
   an occasional loss, but its target paths were largely predefined and this is weaker than an
   explicit search belief or recovery planner:
   <https://proceedings.mlr.press/v80/luo18a.html>

Implication for MOTAR: the Kalman state is useful but insufficient. The missing component is an
actionable spatial belief/coverage state and a search or visibility-recovery objective.

### Learned adversarial targets

1. **AD-VAT (ICLR 2019)** learns both tracker and target policies. The target is deliberately
   tracker-aware: it receives the tracker's observation and action and predicts the tracker reward
   as an auxiliary task. A partial zero-sum reward applies competition near the observable range
   while penalizing trivial flight far outside it:
   <https://openreview.net/pdf?id=HkgYmhR9KX>
2. **Towards Distraction-Robust Active Visual Tracking (ICML 2021)** trains a target and multiple
   distractors as a cooperative team against the tracker, producing adversarial occlusion and
   confusing behavior:
   <https://proceedings.mlr.press/v139/zhong21b.html>
3. **AgilePE (arXiv, submitted 2026-08-14; under review)** models both UAVs as 6-DoF bodies and
   maps onboard state observations directly to collective-thrust/body-rate actions. It compares
   naive self-play, fictitious self-play (FSP), and prioritized FSP with historical opponent pools.
   The reported policies discover flanking, lateral oscillation, and pursuit-turn-latency
   exploitation:
   <https://arxiv.org/abs/2608.14135>
4. Fictitious self-play formalizes training approximate best responses against mixtures of past
   opponent strategies instead of only the latest nonstationary opponent:
   <https://proceedings.mlr.press/v37/heinrich15.html>

AgilePE is strategically stronger than the current MOTAR target, but it is not an overall
head-to-head superiority result: its reported core setup is state-based 1v1 and its own paper notes
the obstacle-free setting. MOTAR's comparative strengths are sensor-path auditing, RGB-D/LiDAR
fusion, motor/contact dynamics in dense clutter, and failure-accounted held-out evaluation.
MOTAR's comparative weaknesses are explicit search, long-occlusion recovery, target behavioral
DoF, and opponent-conditioned learning.

## 5. Recommended contracts for an autonomous evader

Create two explicitly separate evaluation contracts. Never mix their claims.

### Contract A: realistic onboard evader

The evader actor receives only quantities that a physical target could sense or estimate:

- own attitude, velocity, angular rate, previous action, and battery/actuator state if modeled;
- egocentric obstacle sensing and mapped/geofenced boundary distance;
- relative pursuer estimate only while within the evader's declared sensors;
- a recurrent or fixed temporal history with a documented duration.

This contract measures mutual autonomy under symmetric sensor limitations.

### Contract B: privileged red-team evader

The evader may receive pursuer ground truth, pursuer observation/action, or simulator-wide geometry,
as in tracker-aware adversarial training. This is a worst-case robustness generator, not a realistic
deployed target. A privileged critic is acceptable during training, but actor-visible privileged
features must be separately enumerated.

### Initial action interface

Reuse the existing physical controller first and expose a bounded high-level command:

`[vx, vy, vz, yaw_rate]` or `[ax, ay, az, yaw_rate]`.

This creates real three-dimensional behavioral freedom without immediately destabilizing motor-level
learning. CTBR (`[collective thrust, body rates]`) can be a later ablation after the strategic policy
is established. Do not call the high-level version end-to-end motor control.

### Evader reward

A candidate reward must balance:

- positive survival time and tactical separation;
- positive line-of-sight break or pursuer uncertainty increase;
- negative capture/inter-agent collision;
- negative bar contact, OOB, overspeed, excessive tilt, motor saturation, energy, and jerk;
- an encounter-envelope penalty preventing the trivial optimum of fleeing permanently beyond
  sensing range.

Use a partial zero-sum core only where interaction is meaningful. Log every reward component and
test for degeneracies: stationary hiding, permanent range escape, wall hugging, unsafe oscillation,
and exploiting simulator-only geometry.

## 6. Experimental roadmap

### Phase S0: freeze and characterize the baseline

- Keep the corrected seed-911 route-off policy/checkpoint and seed-313 artifacts immutable.
- Add no performance claims beyond the six trained density cells at 70-145 bars.
- Export acquisition-conditioned outcome, track expiry, hidden-duration, and reacquisition latency
  using evaluation-only diagnostics where possible.
- Establish deterministic CPU tests for belief propagation and episode-state transitions.

Exit condition: all proposed metrics reproduce existing aggregate capture/crash/timeout totals and
partition episodes without overlap or omission.

### Phase S1: explicit blind-search state, one causal axis

Hold target dynamics, camera range, reward, density, speed, PPO budget, and safety governor fixed.
Compare fresh policies, because adding actor observations changes the schema.

Candidate observation additions:

- noisy/dropout-tested body-frame geofence distances;
- egomotion-integrated visited or occupancy map;
- target-location belief and entropy;
- mode/age indicators distinguishing never-acquired, tracked, and stale/lost.

Compare at least:

1. current implicit blind policy;
2. geofence only;
3. geofence plus coverage/visited state;
4. explicit belief/frontier search.

Primary metrics: never-acquired rate, first-acquisition time, never-acquired OOB, and crash. Capture
is secondary because it conflates acquisition with post-acquisition navigation.

### Phase S2: occlusion-aware reacquisition

- Keep the Tier-0 target scripted.
- Replace a single indefinitely extrapolated point with uncertainty-aware hypotheses. Candidate
  baselines are CV Kalman, interacting multiple-model filtering, and particle/Gaussian-mixture
  belief.
- Switch from TRACK to REACQUIRE using track age/covariance, not detector visibility alone.
- Evaluate occlusion-clearing viewpoints or collision-free paths around the occluding bar.

Primary metrics: reacquisition probability and latency stratified by hidden duration, target speed,
trajectory type, and density. Guardrails: bar contact, false lock, OOB, and compute latency.

### Phase E1: rule-based reactive evader

Add a bounded, deterministic/reactive evader before self-play. It may choose among physically
feasible headings or accelerations based on relative pursuer bearing, time-to-capture, free space,
and FOV escape. Include named policies such as:

- radial-away;
- lateral/FOV-break;
- obstacle-screening;
- bounded random reversal;
- mixed policy sampled per episode.

This creates interpretable failure cases and verifies the observation/action/reward plumbing without
the nonstationarity of two learned agents.

### Phase E2: single-opponent adversarial training

- Warm-start low-level flight control, not necessarily the strategic head.
- Alternate pursuer and evader updates or freeze one side for declared rounds.
- Begin with privileged critics and explicitly scoped actor observations.
- Measure cross-play against all scripted E1 opponents after every round.

This phase is diagnostic. Naive latest-vs-latest self-play is not sufficient for a final robustness
claim because cycling and catastrophic forgetting are expected.

### Phase E3: historical-population FSP/PFSP

- Archive pursuer and evader policies at fixed, preregistered intervals.
- Train best responses against a mixture of historical opponents.
- Add prioritization only after uniform historical sampling is stable.
- Keep simple scripted opponents in the sampling/evaluation mixture so prioritization does not
  overspecialize against complex learned opponents.
- Seal held-out evader policies/seeds before the final evaluation.

### Phase J: joint search plus adversarial evader

Only after S and E axes separately pass should they be combined. Use a factorial comparison:

| pursuer search | target type |
|---|---|
| implicit baseline | scripted CV/waypoint |
| explicit search/reacquire | scripted CV/waypoint |
| implicit baseline | held-out reactive/learned evader |
| explicit search/reacquire | held-out reactive/learned evader |

This separates whether gains come from better search or from adaptation to a particular opponent.

## 7. Required evaluation matrix and gates

Report at minimum:

- capture, crash, timeout, OOB, and bar-contact rates with Wilson intervals;
- first-acquisition rate/time and never-acquired outcomes;
- visible fraction and visible-to-hidden transition count;
- reacquisition success/time by hidden-duration bin;
- target and pursuer realized speed, acceleration, turn rate, tilt, and motor saturation;
- cross-play payoff matrix over current and historical opponent policies;
- performance against scripted CV, waypoint, held-out circle, rule-based reactive, and sealed
  learned evaders;
- latency/noise/domain-randomization sensitivity;
- wall-clock inference and training cost.

Predeclare thresholds before long training. Suggested qualitative gates:

1. search changes must reduce never-acquired failure without increasing non-OOB collision;
2. evader changes must obey the same physical envelope and may not use undeclared oracle inputs;
3. a pursuer improvement must hold against unseen opponent policies, not only the co-trained evader;
4. no superiority claim may compare different sensor ranges, speed envelopes, obstacle densities, or
   safety layers as if only the policy changed;
5. no hardware or zero-shot sim-to-real claim without physical deployment evidence.

## 8. Claude handoff checklist

Before proposing code, Claude should:

1. read this document and the three current source files cited above;
2. read the two corrected held-out JSON files and preserve their claim boundary;
3. choose exactly one first experimental axis: S1 search state or E1 reactive evader;
4. write a preregistration naming treatment/control, seeds, checkpoint policy, metrics, gates, and
   abort conditions;
5. identify every new actor-visible and critic-only field;
6. state whether the evader contract is onboard-realistic or privileged red-team;
7. add unit tests and observation-schema fail-closed checks before training;
8. run a short deterministic smoke before allocating a fresh PPO campaign;
9. keep raw artifacts immutable and produce receipts/source manifests like the existing evaluation;
10. update README/status only after the corresponding held-out gate completes.

The preferred first task is **S1 explicit blind-search state**, because the current held-out evidence
already isolates never-acquisition as a measured failure and no adversarial target is needed to test
that mechanism. The preferred second task is **E1 rule-based reactive evader**, which establishes a
reproducible opponent suite before introducing self-play.

