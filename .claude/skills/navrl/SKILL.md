---
name: navrl
description: >-
  Stage-routed MOTAR/NavRL research loop. Read VERIFICATION.md first, identify the
  current stage, and never auto-start PPO. Invoke for the next research cycle,
  "다음 학습", "iterate", "run the sweep", diagnosis, or "how is training going".
---

# NavRL sensor-only research loop

You are driving the MOTAR / NavRL "Phase 3 vision pivot": a **sensor-only** UAV interception policy
(no ground-truth target in the actor observation). This skill is the repeatable loop.

**Token rule:** orchestrate and record; DELEGATE every step that reads many files, parses CSVs, or
scans logs to a subagent. Never read `runs/**/epoch_metrics.csv`, `nn/*.pth`, or long training logs
into the main context.

Working dir: `aerial_gym/rl_training/rl_games`. Env: conda `aerialgym`, always `PYTHONNOUSERSITE=1`.
Sensor-only mode = `NAVRL_VISION=1`.

## Rule priority — do not invert

1. The user's current decision
2. `VERIFICATION.md` — current execution authority
3. `RESEARCH_PLAN.md` — hypotheses and method
4. The current-lineage preregistration
5. This skill — shared procedure
6. Historical command examples inside this skill — reference only, not authority

If a command in this file disagrees with `VERIFICATION.md`, follow `VERIFICATION.md`.

## Session start — mandatory

1. Read `VERIFICATION.md` (current stage, GPU/PPO authority, frozen FAILs).
2. Do **not** launch training, eval, or simulator GPU work unless that document currently
   authorizes it.
3. Identify the stage below. Routed v3 is `MECHANISM_GATE_FAIL_CLOSED`. The separately
   authorized route-off curriculum was operator-stopped at epoch 21973 / 145 bars; its seed-313
   held-out sweep is complete. No GPU authority remains. Do not resume it, start a second
   curriculum, evaluate 205 as mastery, or open routed PPO without a new preregistration.
4. `CRASH_TUNING_LOG.md` is archival-in-place. Do not append. Use `WORKLOG.md` and a separate
   diagnostic document.

## Stage router

| Stage | What it is | May run | Must not run |
|---|---|---|---|
| `ENVIRONMENT_CONTRACT` | geometry/placement/spawn/provenance CPU gates | CPU tests, audits | PPO, GPU eval |
| `MECHANISM_GATE` | target route/controller does the job it claims | CPU tests; GPU gate **only** if VERIFICATION authorizes | PPO, threshold retune |
| `PPO_SMOKE` | separately preregistered short fresh PPO | that smoke only | long curriculum, sim-to-real claims |
| `PPO_CURRICULUM` | density curriculum training | the preregistered curriculum | held-out substitution, mixing GPU factories |
| `HELD_OUT_EVAL` | frozen checkpoint evaluation | registered eval cells | training, post-hoc threshold edits |
| `SIM_TO_REAL` | hardware/real-log contract | BOM/calibration/210 trials/real-log replay | GPU without hardware/logs |

Corrected non-overlap software lineage is in `MECHANISM_GATE`: r2 is
`FAIL_ROUTE_MECHANISM`, fresh PPO is **0 epoch**, and braking-aware route v3
(`global_astar_braking_v3`) must pass its preregistered gate before any PPO smoke.

Track A (perception/sim-to-real) remains hardware/real-log only. Track B recovery-v2 is closed.

## TWO DISCIPLINES, ONE NAME — know which one you are paying for

`docs/discipline_review_2026-08-22.md` found that "preregistration" here means two different things
with very different track records. Keep them apart; they cost differently and fail differently.

**(A) Claim protection** — freeze the gate, the metric, the seed, and the verdict rule BEFORE
measuring, so the answer cannot be steered afterwards. This is cheap. It should be a short block:
question, arms, primary metric, thresholds, what would falsify it, and what the result does NOT
authorise. It does not need 150 lines of prose. Write it, freeze it, commit it before the machinery
that runs it exists.

**(B) Machinery integrity** — receipts, source manifests, checkpoint SHA pinning, runtime-clean
gates, import-origin enforcement, fail-closed guards, equivalence proofs. This is expensive and it
is what has actually saved this project: all five VOIDed runs (`governor_off`, `dirty_runtime`,
`export_guard_bug`, `source_drift`, `guard`) were caught by (B), not by (A). With 2,000+ episodes a
cell, noise is not the adversary — silent machinery failure is. **Do not economise here.**

The failure mode to avoid is paying (B)-sized ceremony for an (A)-sized decision.

## CHEAP CALCULATION BEFORE EXPENSIVE MACHINERY

Before preregistering an experiment, write down: **can this result be predicted by calculation, and
if so what is the prediction?** Put the answer in the preregistration.

- If calculation gives a confident prediction, ask whether the experiment is still the highest-value
  next step. Often a 30-minute derivation replaces hours of GPU and days of tooling.
- If you run it anyway, a matching result confirms the model and a mismatch is a real discovery.
  Either way you learn more than from an unanchored measurement.

This is not hypothetical. On 2026-08-22 the three most valuable findings of the day — that 28 m
detection is geometrically impossible in-sim, that `min_target_pixels` is an area rather than a
diameter, and that the 20 m clip rather than the pixel threshold is what binds — all came from
reading code and doing arithmetic with the GPU idle.

## PROHIBITIONS EXPIRE — every one carries a reason and a review trigger

When you forbid something, write **why** and **what would make it reviewable again**. A prohibition
whose cause has been fixed and which nobody has re-examined is not discipline, it is sediment.

Distinguish two things that look alike:
- **Forbidden** — doing it would corrupt a claim or destroy provenance. Example: editing
  `aerial_gym/config/robot_config/**` or `resources/robots/**`, which are byte-frozen against every
  existing checkpoint.
- **Expensive and unexplored** — legitimate, just costly. Example: the speed/tilt/yaw ceilings.
  `max_velocity` is the observation-normalisation denominator, so changing it breaks the checkpoint
  contract and forces a retrain. That is a price, not a prohibition. Say so, and price it.

Never let the second quietly become the first.

## PLAN SYNC RULE — non-negotiable, and the one most often forgotten

**When the plan changes, update the planning documents in the same commit as the work that changed
it.** A WORKLOG entry records what happened; it does NOT tell the next session what to do. That is
`VERIFICATION.md`'s job (execution authority: gates, verdicts, the next experiment) and
`RESEARCH_PLAN.md`'s job (charter: hypotheses and method).

Update `VERIFICATION.md` whenever ANY of these change:
- the next experiment, or the reason for it
- a gate, threshold, or the conditions that unblock a blocked stage
- what the current bottleneck is believed to be
- a prerequisite being satisfied or invalidated
- the `기준일` — bump it every time you touch the file

Update `RESEARCH_PLAN.md` when the hypothesis or the method changes, not when a number lands.

**What you may NOT do while syncing**: change a recorded verdict. P2 / D1 / P3 status lines and any
frozen result's judgement are historical facts. If new evidence reinterprets an old result, record
the reinterpretation as a LIMITATION next to it — never by editing the verdict.

Self-check before you end a session: `git log --oneline -5` and ask, for each commit, "would a fresh
session reading only VERIFICATION.md do the right next thing?" If not, the sync is missing.

## WORKLOG RULE — non-negotiable

**Every piece of work ends with a `WORKLOG.md` entry. No exceptions, no "I'll add it later".**

This applies to *each* of these, not just a full loop iteration:
- a training run (launched, finished, died, or was killed — record which, and why)
- an eval / sweep (record the actual numbers, not "it improved")
- a code change (what changed, what it was measured against)
- a diagnosis, **including one that turned out wrong** — a refuted hypothesis is a result and stops
  the next session from re-running it

Entry format (newest at the BOTTOM of `WORKLOG.md`):
- dated `## YYYY-MM-DD — <one-line headline>`
- the measured numbers, in a table when there is more than one cell
- the decision it led to, and the next concrete step
- run folder / checkpoint / `results/*.csv` paths so the numbers can be re-derived
- when a claim was **disproved**, say so explicitly ("hypothesis X refuted: measured Y")

Write the entry **before** asking the user to review a diff. If a session is running out of budget,
the WORKLOG entry is the LAST thing to cut. `CRASH_TUNING_LOG.md` is archival-in-place (stopped
2026-08-05); do not add new records there.

## Hard-won rules (do not relearn these)

- **Evaluate a curriculum run with `last_gen_ppo_ep_XXXX.pth`, NOT `gen_ppo.pth`.** `gen_ppo.pth`
  is the *best-reward* checkpoint, and in a density curriculum reward is highest at LOW density.
- **Warm-start works now**: `--checkpoint <path> --max_epochs <N>` resumes and extends.
- **1650 Ti (4 GB, N=128) trains ~10-15 pt WEAKER than the 3070 (N=256)** — never mix their training
  numbers in one curve.
- Current corrected non-overlap densities are **70/115/160/205**. **300 bars is a disconnected
  stress condition**, not a training or route-gate cell.
- Do not retune wall/tracking margins, speed/accel ceilings, or reward/obs/termination to make a
  mechanism gate pass.

## Current MECHANISM_GATE procedure (braking-v3)

Canonical preregistration (frozen, do not edit; 1.5 receipt is NO-GO):
`docs/preregistration_braking_aware_route_v3_2026-09-01.md`

Current lower-contract preregistration (frozen, do not edit; VOID under extra spawn inset):
`docs/preregistration_braking_aware_route_v3_lower1p25_2026-09-01.md`

Current matched-spawn preregistration (frozen, do not edit):
`docs/preregistration_braking_aware_route_v3_lower1p25_matched_spawn_2026-09-01.md`

Current execution-only GPU authority (frozen, do not edit):
`docs/preregistration_braking_aware_route_v3_lower1p25_matched_spawn_gpu_authority_2026-09-01.md`

1. CPU tests only until they pass: focused v3 tests plus v1/recovery regression.
2. The one-shot GPU addendum is consumed. The matched-spawn pilot is
   `PASS_8_CELL_INTEGRITY / FAIL_BLOCKS_CONFIRMATORY`; **stop** before any rerun, confirmatory,
   retune, or PPO.
3. The prior seed-829 run remains `VOID_EXECUTION` and the integrity-clean matched-spawn run is a
   separate official FAIL. Do not drop the target-pose hash, reuse either output root, or arm from
   the `dd8b4a4` receipt.
4. Confirmatory (32 cells): seed 839, bars 70/115/160/205, same speeds/arms. Thresholds stay as
   written in the lower preregistration. Do not reuse the 2026-08-26 lower receipt.
5. Confirmatory PASS authorises only a **separate** preregistered 500-epoch PPO smoke inside the
   lower envelope. It does not itself start training. Do not retune the 0.05 warmup gate or add PID
   to pass a mechanism gate.

```bash
# CPU only — current allowed work
cd /path/to/repo
export PYTHONNOUSERSITE=1
python -m unittest tests.test_navrl_braking_route_v3 tests.test_navrl_braking_route_v3_gate \
  tests.test_navrl_target_route_planner tests.test_navrl_target_motion \
  tests.test_navrl_two_envelope_recovery
git diff --check
```

Prepared GPU commands live in `OPERATIONS.md` and `tools/run_navrl_braking_route_v3_{pilot,confirmatory}.sh`.
Do not run them from this skill.

## Historical recipes — not current authority

These values belong to earlier PPO-first loops. Do not use them as the current density grid or
as a reason to launch training.

- Eval/train densities **25/50/75/110/130/150**
- `NAVRL_DENSITY_THRESHOLD=0.6` as a single sensor-only default
- Overlap-era 70/150/205/**300** route-gate grid (300 is now disconnected stress)
- Auto-launching `./train_navrl.sh` at the start of a session

Historical launch templates, kept for reconstruction only:

```bash
# historical density curriculum (NOT current authority)
NAVRL_VISION=1 NAVRL_DENSITY_CURRICULUM=1 NAVRL_DENSITY_WARMUP=6000 NAVRL_DENSITY_THRESHOLD=0.6 \
  ./train_navrl.sh --seed 1 --max_epochs 12000
# historical eval densities 25/50/75/110/130/150 via last_gen_ppo_ep_*.pth
NAVRL_VISION=1 NAVRL_NUM_BARS=<D> NAVRL_MAX_BARS=150 NUM_ENVS=256 HEADLESS=True \
  PLAY_GAMES_NUM=2500 ./play_navrl.sh <last_ckpt>
```

v2 recovery's density gate was a measured knot schedule `0.82@70, 0.77@85, 0.72@100, 0.70@115+`,
not a single 0.6 threshold. Do not mix that recipe with the corrected non-overlap 70→205 charter.

## If and only if VERIFICATION authorises PPO_SMOKE or later

Then, and only then: check GPU idle, pick the preregistered launcher, wait with a completion
monitor, delegate eval/CSV parsing, and record a WORKLOG entry. Commit/push only after the user
reviews the diff.

## Running subagents well (token-efficient)

- Delegate reads, keep conclusions. Launch independent agents in one message.
- Give each agent exact paths, exact commands, "read-only", and the structured output you want.
- Never read a subagent's `*.output` transcript from the shell.
