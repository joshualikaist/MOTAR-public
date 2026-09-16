# Relation to published systems — 2026-09-16

This is a scope and reproducibility comparison, not a performance ranking. The search first
screened UAV obstacle-navigation, tracking, pursuit-evasion, safety-filter and temporal-planning
systems, then retained the ten closest to MOTAR's combined question. No paper below uses MOTAR's
same repository, task contract, dataset, sensor model, policy lineage and outcome metric.

Comparability classes:

- **A — direct numerical comparison:** same dataset/benchmark and same metric.
- **B — candidate for a future matched baseline:** usable public code exists, but it has not been
  ported and run under a frozen MOTAR contract.
- **C — architectural relation only:** the environment, sensors, method or metric differ, or a
  runnable implementation was not found.

There are currently **no Class A systems**. Class B means “potentially rerunnable after a separate
port and preregistration,” not “already comparable.”

| Work | Pursuit / tracking | Obstacles | Perception | Method / temporal model | Class | What MOTAR can borrow |
|---|---|---|---|---|---|---|
| [NavRL](https://arxiv.org/abs/2409.15634) ([code](https://github.com/Zhefan-Xu/NavRL)) | Goal navigation; pursuit is not the core task | Dynamic navigation environments | LiDAR/state navigation inputs | PPO navigation and a velocity-based safety layer | **B** | The closest open navigation lineage for a frozen port; obstacle representation and shield design. |
| [NavRL++](https://arxiv.org/abs/2605.15559) | Goal navigation; not target pursuit | Static/dynamic navigation scenarios | Perception-to-control navigation stack | Improved RL training, temporal network, perturbation-aware fine-tuning and safety shield | **C** | Temporal policy and perturbation-training ablations. No released implementation was verified in this review. |
| [YOPO](https://doi.org/10.1109/LRA.2024.3399589) ([code](https://github.com/TJU-Aerial-Robotics/YOPO)) | No target pursuit | Dense static navigation scenes | Depth | Guidance learning over trajectory primitives; no RL pursuit policy | **B** | A public depth-based one-stage planner that could be ported as an obstacle-navigation baseline. |
| [YOPOv2-Tracker](https://arxiv.org/abs/2505.06923) ([project](https://github.com/TJU-Aerial-Robotics/YOPO-Tracker)) | Fast moving-target tracking | Obstacle-dense real/sim scenes | RGB-D / multimodal visual input | Guidance learning, objectness-scored motion primitives, trajectory/control generation | **C** | Joint visibility-aware trajectory scoring. The project page still states that tracking/navigation code will be released. |
| [OPEN](https://doi.org/10.1109/LRA.2025.3583620) ([code](https://github.com/thu-uav/Multi-UAV-pursuit-evasion)) | Multi-UAV pursuit-evasion | Unknown environments | Partial state observations with evader prediction | MAPPO, attention, adaptive environment generation, CTBR control | **B** | The strongest open pursuit candidate: evader prediction, adaptive curriculum and cross-scenario evaluation. A port is still required. |
| [AgilePE](https://arxiv.org/abs/2608.14135) | Bilateral UAV pursuit-evasion | Core reported setting is not MOTAR's random dense-bar benchmark | Onboard state observations | Self-play, FSP/PFSP opponent pools, direct CTBR control | **C** | Historical-opponent pools and cross-play for a later TM-E4 learned evader. No official runnable code was verified. |
| [FlowPilot](https://arxiv.org/abs/2608.00635) | No target pursuit | Increasing navigation clutter | Depth and future depth geometry | Flow-matching world/action model; C² Bernstein trajectory output | **C** | Trackable trajectory parameterization and future-geometry prediction. Different task and no verified public implementation. |
| [Role-based MADDPG pursuit](https://doi.org/10.1109/ICRA48891.2023.10160919) ([preprint](https://arxiv.org/abs/2303.01799)) | Multi-target pursuit and tracking | Random obstacles | Multi-agent state/target observations | Heterogeneous scout/pursuer roles with MADDPG and Voronoi exploration | **C** | Explicit search-versus-track roles for later target-search studies; multi-agent benchmark differs from MOTAR. |
| [PILOT](https://arxiv.org/abs/2608.14082) | No target pursuit | Cluttered navigation | Historical depth images and odometry | Privileged imitation learning with a TCN and structured trajectory layer | **C** | Teacher/student separation and temporal partial-observability handling. |
| [Temporal Barrier](https://arxiv.org/abs/2608.14239) | Independent pursuit and formation scenarios, not target tracking | Other aircraft are dynamic collision agents | Relative multi-agent state | Adversarial TTC control-barrier function with a neural surrogate | **C** | A formal temporal-risk reference for future safety work; it must not be conflated with MOTAR's non-certified arc-clearance filter. |

## MOTAR's distinct combination

MOTAR currently combines moving-object perception records, temporal candidate selection, measured
perception-error injection, PPO pursuit/navigation in random obstacle fields, and matched
arc-clearance/riskcap ablations. That combination is a difference in research scope, not evidence
of superiority over the systems above.

MOTAR has no real-flight validation, no external Class A benchmark, no validated persistent target
identity, and no learned evader. A future external numerical comparison must first freeze a common
task, sensor and metric contract and run a Class B implementation inside it.

## Screening boundary

Seventeen systems/papers were screened. The ten above were retained. The following seven were
read but omitted from the compact site table because a selected row already covered the same axis
more directly or because they were farther from single-pursuer perception-plus-clutter:

- [Meta-RL target tracking and obstacle avoidance](https://github.com/shenmuxin/uav_tracking_metaRL);
- [ARVP-MADDPG dynamic-obstacle pursuit](https://doi.org/10.1109/AIAHPC66801.2025.11290656);
- [HOCBF safe-RL multi-drone planning](https://doi.org/10.3390/drones8090481);
- [CBF-based multi-UAV/UGV tracking](https://doi.org/10.1109/IROS60139.2025.11247497);
- [zero-shot CLF/CBF-filtered navigation](https://arxiv.org/abs/2605.01787);
- [YOPO-Rally](https://arxiv.org/abs/2505.18714); and
- [fictitious self-play](https://proceedings.mlr.press/v37/heinrich15.html), retained as a
  methodological foundation rather than a UAV system row.

Reported percentages from any screened paper are not copied into the public table because none is
measured on the MOTAR benchmark.
