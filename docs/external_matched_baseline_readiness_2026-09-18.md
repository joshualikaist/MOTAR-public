# External matched-baseline readiness — 2026-09-18

A Class A comparison does not exist and cannot be created by argument. It requires porting an
external system into a frozen, preregistered MOTAR contract and running it there. This document
assesses which Class B candidates could support that, and what each port would actually cost.

**Selection criterion is benchmark compatibility, not expected outcome.** A candidate is not
preferred because MOTAR would likely score well against it. Where a port would be cheap but the
resulting comparison meaningless, that is recorded as a reason against, not for.

**Nothing here was executed.** No external system was installed, built, trained or run. No GPU
workload was started for this assessment. Every row is a documentation-level judgement from the
sources cited in
[the quantitative ledger](literature_quantitative_ledger_2026-09-18.json), and the rows say plainly
which fields were verified and which were not.

## Verification status of this table

| Field | How it was established |
|---|---|
| Public code exists | Repository URL fetched and confirmed to resolve |
| License | Only recorded where the repository states it; otherwise `NEEDS_CONFIRMATION` |
| Released checkpoint | **`NOT_VERIFIED` for every candidate** — no repository release artifact was inspected |
| Sensor / action / task mismatch | Read from the primary paper's own description of its inputs and outputs |
| Port work estimate | Engineering judgement, not a measurement |

## Readiness table

| Candidate | Public code | License | Checkpoint | Sensor match | Action match | Task match | Tracks a moving target | Obstacle benchmark | Port work |
|---|---|---|---|---|---|---|---|---|---|
| [OPEN](https://github.com/thu-uav/Multi-UAV-pursuit-evasion) | Yes, official | MIT (stated in repo) | `NOT_VERIFIED` | **Severe mismatch** — privileged relative state and an LOS bit; no camera or depth anywhere | CTBR vs MOTAR's velocity/yaw-rate command | Pursuit-evasion, but **3 pursuers vs 1 evader** | Yes, a scripted evader | Yes — 4-5 cylinders in a small arena | **High.** Would need a single-pursuer reduction and an observation bridge that does not hand the policy privileged state |
| [NavRL](https://github.com/Zhefan-Xu/NavRL) | Yes, official | `NEEDS_CONFIRMATION` | `NOT_VERIFIED` | Closest of the set — RGB-D plus state, Isaac Sim lineage | PPO velocity command, close to MOTAR | **Goal navigation, not pursuit** | No — dynamic obstacles are avoided, not tracked | Yes — 350 static plus 60-120 dynamic | **Medium.** Same simulator family and same PPO shape; the missing piece is a pursuit task wrapper |
| [YOPO](https://github.com/TJU-Aerial-Robotics/YOPO) | Yes, official | `NEEDS_CONFIRMATION` | Pretrained weights stated in README (6 m/s); artifact `NOT_VERIFIED` | Depth only | Motion primitive selection, not a velocity policy | Obstacle avoidance only | No | Flightmare/Unity forests | **High.** Different simulator stack and a primitive-library action space |
| YOPOv2-Tracker | **No** — repository is a placeholder | n/a | n/a | Depth 160x96 | Primitive selection | **Closest task semantics of any tracker: single tracker, uninstrumented evader UAV, onboard vision** | Yes | Poisson forest, 1/16 tree/m² | **Blocked.** Code unreleased, so no port is possible |
| [Fast-Tracker](https://github.com/ZJU-FAST-Lab/Fast-tracker) | Yes, official | `NEEDS_CONFIRMATION` | n/a — classical planner, no learned weights | Depth plus a marker-based target detector | Trajectory tracking, not a policy action | Target tracking | Yes, but **cooperative and marker-based**; the simulation benchmark feeds the planner the target's ground-truth future trajectory | Yes — 20x20x3 m with 140 obstacles | **Medium.** No training needed, but its target model must be replaced, and the oracle future trajectory removed |
| [Elastic Tracker](https://github.com/ZJU-FAST-Lab/Elastic-Tracker) | Yes, official | `NEEDS_CONFIRMATION` | n/a — classical planner | Depth plus a separate target camera | Trajectory tracking | Target tracking with an explicit visibility objective | Yes, but **cooperative** — the target's position is broadcast to the chasers | Yes, in the released configuration | **Medium.** Same as above; its visibility objective is the most transferable idea in the set |

## What each candidate would actually establish

These are role assignments, not a ranking. None of them is "the strongest system."

* **OPEN** — closest *pursuit task semantics*. It is the only candidate whose evaluated objective is
  capturing an evader. Its input contract is also the furthest from MOTAR's, so a port tests whether
  MOTAR's task is reproducible at all outside a privileged-state setting.
* **NavRL** — closest *architecture and simulator lineage*. A port mostly tests the task wrapper
  rather than the policy, which makes it the cheapest honest Class A candidate.
* **Fast-Tracker / Elastic Tracker** — closest *single-UAV tracking system semantics*, and the only
  non-learned comparators available. They would separate "does the policy help" from "does any
  competent planner suffice" — a question MOTAR currently cannot answer about itself.
* **YOPO** — closest *learned obstacle-aware planning architecture* that is neither PPO nor a
  tracker.

## The obstacle that no port removes

Every candidate tracker evaluates a **cooperative** target: marker-carried, position-broadcast, or
supplied as an oracle future trajectory. MOTAR's TM-E2 target is obstacle-aware and
pursuer-independent, and TM-E3/TM-E4 would be reactive and learned. Elastic Tracker's conclusion
names generalizing to an escaping target as future work, which locates that boundary in the
classical tracking literature rather than in MOTAR's framing of it.

This means a matched comparison must **fix the target**, not only the arena. Running an external
tracker against its own cooperative target and MOTAR against TM-E2 would produce two numbers that
still are not comparable.

## Future Class A protocol (frozen here, not executed)

A Class A entry requires all of the following to be identical across arms:

* the same arena and the same obstacle layouts, from the same seeds;
* the same start states and the same target initial states;
* the same target trajectory family, produced by the same target executor;
* the same vehicle speed and acceleration limits, where the ported system's dynamics permit;
* the same success definition (`success_radius`) and the same timeout;
* the same evaluation seeds and the same episode count per cell.

Where sensor or action representation cannot be made identical — which is expected for OPEN, and
certain for YOPO — the run is recorded as **`PARTIALLY_MATCHED`**, with the specific unmatched
dimensions named in the receipt. A `PARTIALLY_MATCHED` run does not become Class A, and it does not
license a superiority statement in either direction.

## Current state

**Class A count: 0.** No port has been started, and none is authorized by this document. Selecting a
candidate is a separate preregistration decision.
