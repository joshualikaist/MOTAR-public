# TD-T2 preregistration — terminal close-approach forensics

Written 2026-09-14, **before** the instrumentation was implemented and before any diagnostic run.
The classification rules and every threshold below are fixed here, before any result is seen.

## 0. Scope

Evaluation-only forensic logging of the final approach in episodes produced by **existing**
checkpoints. No policy, reward, detector, association, safety-filter, controller or termination
change; no retraining; no terminal controller. Scope, naming and the 2026-09-14 boundary change are
as stated in [TD-T1](preregistration_task_diagnostics_t1_2026-09-14.md).

## 1. The historical capture criterion is not touched

`success_radius = 0.5 m` stays exactly as it is, and every historical success number keeps its
meaning. TD-T2 adds **evaluation-only** metrics beside it; none of them replaces it, and none of them
is used to relabel a historical result:

```text
legacy_capture_0p5m                 the unchanged criterion, recorded as-is
minimum_center_distance_m           swept centre-to-centre minimum over the episode
minimum_surface_distance_m          minimum_center_distance_m - 0.15 m target radius
relative_speed_at_closest_approach_mps
time_inside_1m_steps
time_inside_0p5m_steps
```

`minimum_center_distance_m` uses the **same swept segment distance the capture test itself uses**, so
the two cannot drift apart. The 0.15 m target radius is the configured `camera_target_radius`; the
surface distance is defined only under that sphere model and is reported as `null` if the radius is
unavailable.

## 2. Per-step terminal record

Recorded every observation step, retained for the final **20** observation steps (2.0 s) of each
episode plus per-episode extrema:

```text
target_range_m, swept_min_range_m
radial_relative_velocity_mps, tangential_relative_velocity_mps
bearing_error_rad, yaw_error_rad
visible, track_age_s, range_sigma_m
commanded_horizontal_speed_mps, actual_horizontal_speed_mps
governor_scale, governor_intervened, action_saturated
nearest_obstacle_clearance_m
```

Relative velocity is the vehicle velocity minus the realized target velocity, decomposed along and
across the line of sight. `range_sigma_m` is the square root of the tracker covariance's position
trace along the line of sight. `governor_intervened` is true when the executed command speed is below
the requested command speed by more than 1e-3 m/s. `action_saturated` is true when either horizontal
action component is within 1e-3 of ±1.

## 3. Failure taxonomy

Assigned only to episodes that did **not** satisfy `legacy_capture_0p5m`. Rules are evaluated in this
fixed priority order and the first match wins. The order is an attribution choice, fixed here; every
episode also stores the raw inputs, so any other rule can be applied to the stored data afterwards.

| Priority | Class | Rule |
| ---: | --- | --- |
| 1 | `NEVER_ACQUIRED` | The episode never had a visible observation. |
| 2 | `OBSTACLE_CONTACT` | Outcome is a crash whose recorded cause is bar/target contact. |
| 3 | `LOST_TRACK` | Not visible on the final observation step, and the final loss run exceeds 10 steps. |
| 4 | `TERMINAL_OVERSHOOT` | `minimum_center_distance_m` ≤ 1.0 m, radial speed at closest approach ≥ 1.5 m/s, and the range increases over the steps after the closest approach. |
| 5 | `FLY_BY` | `minimum_center_distance_m` ≤ 1.0 m and the tangential speed at closest approach is at least the magnitude of the radial speed. |
| 6 | `FILTER_LIMITED` | `governor_intervened` on at least 50 % of the final 20 observation steps. |
| 7 | `CONTROL_TRACKING_ERROR` | Mean \|commanded − actual\| horizontal speed over the final 20 steps ≥ 0.5 m/s, with no governor intervention in that window. |
| 8 | `SLOW_APPROACH` | Outcome is a timeout, the target is visible at the end, and the mean closing (radial) speed over the final 20 steps is < 0.3 m/s while the range exceeds 0.5 m. |
| 9 | `TIMEOUT_WITH_TRACK` | Outcome is a timeout with the target visible at the end, not matched above. |
| 10 | `UNKNOWN` | Everything else. |

`UNKNOWN` is a legitimate outcome and is reported as such. An ambiguous episode is **not** forced
into a class to make a table look complete, and the `UNKNOWN` share is part of the result.

Thresholds fixed here: window **20** steps, near range **1.0 m**, fast radial speed **1.5 m/s**,
slow closing speed **0.3 m/s**, governor share **50 %**, command error **0.5 m/s**, loss run
**10** steps. None of them moves after a result is seen.

## 4. Questions

* **Q1** Among capture failures, how near did the vehicle actually get?
* **Q2** Was the radial speed at closest approach too high to stop?
* **Q3** How large is the tangential fly-by share?
* **Q4** How often did the safety filter limit the final approach?
* **Q5** How often is the dominant discrepancy between the commanded and the actual speed?
* **Q6** Do terminal failures coincide with low obstacle clearance?

Descriptive. TD-T2 declares no pass/fail gate and proposes no controller.

## 5. Verdict and what it does not authorize

The result is a distribution over the classes above, per density. A dominant class does **not**
authorize the architecture it suggests: a terminal controller, a search policy, a new filter family
or a retraining lineage each need their own preregistration written after these numbers exist.

## 6. Behaviour invariance requirement

As in TD-T1: opt-in, default off, and the trajectory digest with the logger off must equal the digest
with it on. A digest mismatch invalidates any diagnostic result from that run.
