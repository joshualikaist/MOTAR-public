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
In the table, “dynamic obstacles” excludes the tracked target/evader itself.

| Work | Pursuit / tracking | Static / random obstacles | Dynamic obstacles | Perception | Method / temporal tracking | Class | What MOTAR can borrow |
|---|---|---|---|---|---|---|---|
| [NavRL](https://arxiv.org/abs/2409.15634) / [NavRL++](https://arxiv.org/abs/2605.15559) ([NavRL code](https://github.com/Zhefan-Xu/NavRL)) | Goal navigation; pursuit is not the core task | Randomized clutter | NavRL tracks detected moving obstacles | RGB-D plus state | PPO, Kalman obstacle tracks and velocity-obstacle shield; NavRL++ adds a temporal policy and perturbation training | **B/C** | NavRL is the closest open PPO safety port; NavRL++ supplies temporal/perturbation ablations but no verified release. |
| [YOPO](https://doi.org/10.1109/LRA.2024.3399589) / [YOPOv2-Tracker](https://arxiv.org/abs/2505.06923) ([YOPO code](https://github.com/TJU-Aerial-Robotics/YOPO); [Tracker placeholder](https://github.com/TJU-Aerial-Robotics/YOPO-Tracker)) | YOPO: no; Tracker: fast target tracking | Dense simulated/real clutter | No established non-target moving-obstacle benchmark | Depth / RGB-D | Guidance-learned motion primitives; Tracker adds objectness, EKF state and spatiotemporal consistency | **B/C** | YOPO can become an obstacle-planner port; Tracker is the closest unified visual tracking/planning architecture but its code is unreleased. |
| [OPEN](https://arxiv.org/abs/2409.15866) ([code](https://github.com/thu-uav/Multi-UAV-pursuit-evasion)) | Multi-UAV pursuit-evasion | Randomized unknown environments | Evader only | Relative state, LOS mask, nearest obstacles | MAPPO, LSTM evader prediction, adaptive curriculum, CTBR | **B** | Open pursuit baseline candidate; history predictor, occlusion mask and automatic difficulty curriculum. |
| [AgilePE](https://arxiv.org/abs/2608.14135) | Bilateral pursuit-evasion | Core setting is not MOTAR's random dense bars | Evader only | Onboard state | Self-play, FSP/PFSP opponent pools, direct CTBR | **C** | Historical-opponent pools and cross-play for TM-E4; no verified public implementation. |
| [Fast-Tracker](https://arxiv.org/abs/2011.03968) ([code](https://github.com/ZJU-FAST-Lab/Fast-tracker)) | Agile target tracking | Random clutter | Not demonstrated; explicitly future work | AprilTag/cameras plus depth map | EKF/FIFO target history, Bézier prediction, kinodynamic search and optimization | **B** | Classical target-prediction/reacquisition comparator and reproducible random-field protocol. |
| [Elastic Tracker](https://arxiv.org/abs/2109.07111) ([code](https://github.com/ZJU-FAST-Lab/Elastic-Tracker)) | Aerial target tracking | Clutter and occlusion-aware corridors | No non-target moving-obstacle benchmark | Depth mapping plus target camera | EKF target prediction, visibility-aware search, safe corridor and spatiotemporal optimization | **B** | Visibility/occlusion objective and planner comparator separating tracking from navigation safety. |
| [MAD](https://arxiv.org/abs/2606.04534) ([base DiffAero code](https://github.com/flyingbitac/diffaero)) | No target pursuit | Random cubes/spheres/cylindrical forests | Qualitative moving-pedestrian test only | Low-resolution depth plus proprioception/VIO | Recurrent occupancy/visibility world model with Dreamer, PPO or SHAC | **C** | Visible-free/occupied/unknown spatial memory and memory ablations; MAD implementation unavailable. |
| [PILOT](https://arxiv.org/abs/2608.14082) | No target pursuit | Randomized clutter/gates | No demonstrated moving-obstacle evaluation | Depth history, odometry and goal | Privileged MPC imitation, causal TCN, structured trajectory output | **C** | Teacher/student separation and temporal partial-observability handling. |
| [FlowPilot](https://arxiv.org/abs/2608.00635) | No target pursuit | Randomized forests and real clutter | No demonstrated moving-obstacle benchmark | Depth plus state/goal | Flow-matching world/action transformer, future depth, C² Bernstein trajectory | **C** | Future-geometry auxiliary prediction and trackable trajectory parameterization. |
| [Temporal Barrier](https://arxiv.org/abs/2608.14239) | Pursuit/formation scenarios, not a target tracker | No static-clutter evaluation | Adversarial aerial agents | Relative agent state | Learned adversarial TTC surrogate inside a CBF-QP | **C** | Formal temporal-risk comparator; must not be conflated with MOTAR's non-certified arc-clearance filter. |

## MOTAR's distinct combination

MOTAR currently combines moving-object perception records, temporal candidate selection, measured
perception-error injection, PPO pursuit/navigation in random obstacle fields, and matched
arc-clearance/riskcap ablations. That combination is a difference in research scope, not evidence
of superiority over the systems above.

MOTAR has no real-flight validation, no external Class A benchmark, no validated persistent target
identity, and no learned evader. A future external numerical comparison must first freeze a common
task, sensor and metric contract and run a Class B implementation inside it.

## Screening boundary

Eighteen UAV systems/papers were screened. The ten above were retained. The following eight UAV
systems were read but omitted from the compact site table because a selected row already covered
the same axis more directly or because they were farther from single-pursuer perception-plus-clutter:

- [BPMP-Tracker](https://arxiv.org/abs/2408.04266);
- [Role-based MADDPG pursuit](https://arxiv.org/abs/2303.01799);
- [Meta-RL target tracking and obstacle avoidance](https://github.com/shenmuxin/uav_tracking_metaRL);
- [HOCBF safe-RL multi-drone planning](https://doi.org/10.3390/drones8090481);
- [Fast-Tracker 2.0](https://arxiv.org/abs/2103.06522);
- [Eva-Tracker](https://arxiv.org/abs/2602.12549);
- [AMS-DRL](https://arxiv.org/abs/2304.03443);
- [MatrixWorld](https://arxiv.org/abs/2307.14854).

[Fictitious self-play](https://proceedings.mlr.press/v37/heinrich15.html) is a methodological
foundation for later TM-E4 design. It is not a UAV system and is not counted among the 18
screened systems.

Reported percentages from any screened paper are not copied into the public table because none is
measured on the MOTAR benchmark.
