# Active-search geofence A/B preregistration (2026-08-21)

## Motivation and frozen diagnosis

Seed 367, 1-bar, away-CV evaluation found that camera-20m OOB exits were 152/158
(96.20%) never-acquired. Those exits had outward radial speed +1.002 m/s and target closing speed
-0.834 m/s. The 898-D actor has no world XY, arena-boundary distance or geofence, and its explicit
history spans only five 0.1 s steps. See
`results/navrl_ref5in_oob_exit_forensics_seed367/analysis.md`.

This experiment tests one claim only: **making the mapped flight boundary observable helps a policy
learn bounded active search while the target is outside camera range.** It does not test a longer
camera, faster drone, longer horizon, new reward or obstacle representation.

## Sensor/task contract

The intervention appends one Transformer token containing eight actor features:

1. body-forward/left/back/right ray distance to the mapped XY geofence, each divided by the arena
   XY diagonal;
2. four corresponding validity flags.

The feature presumes a deployable VIO/GPS pose and known flight geofence. It is not presented as a
camera/LiDAR-only capability. The first causal A/B fixes noise and dropout to zero. Robustness to
localization error is a later experiment and may not be introduced before the first result.

## Training arms

Both arms are fresh PPO, seed 197, 900 epochs, ref5in, P1c's same general-spawn/distance curriculum,
70-bar start, detector 20 m and otherwise identical launcher contract.

| arm | `NAVRL_GEOFENCE_ACTOR` | observation |
|---|---:|---|
| control | 0 | historical structured actor |
| geofence | 1 | control + 8-D geofence token |

No checkpoint warm-start is permitted because the actor schema changes. No arm may change camera
range, goal range, episode length, max velocity/tilt, reward, speed governor, target motion,
obstacle tokens or density schedule.

## Held-out decision screen

After both training arms finish, evaluate deterministic policies at seed 367 under the exact
20 m / 1 bar / `[22.5,28] m` / away-CV / 600-step contract and at least 2,049 episodes per arm.

Primary metric is never-acquired OOB episodes divided by all episodes. The geofence arm must improve
the control by at least **3.0 percentage points**. Guards: non-OOB crash may not worsen by more than
2.0 pp; episode accounting and source/checkpoint receipts must pass; no NaN/KL safety failure may
occur. Capture and timeout are secondary and cannot rescue a failed primary/guard.

If the geofence arm passes, run a predeclared inference ablation that replaces all four ranges by
1 and all validity flags by 0. “Material return” is fixed before evaluation as losing at least
**50% of the normal geofence arm's primary gain**. Passing that threshold supports token use;
absence of a change leaves mechanism attribution unresolved even if outcome improved.

## Interpretation limits

- PASS supports mapped-geofence active search, not localization-free exploration.
- FAIL does not prove boundary information is useless; 900 epochs or the feed-forward 0.5 s history
  may be insufficient. Escalation is then a recurrent coverage belief, as a separate architecture
  experiment.
- The old 28 m camera arm remains a positive control and is not part of training.
