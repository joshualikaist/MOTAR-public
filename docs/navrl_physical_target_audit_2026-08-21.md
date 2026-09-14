# NavRL physical-target implementation and pre-training gate

## Decision

`NAVRL_TARGET_DYNAMICS=physical` is now a distinct, fresh-only simulation lineage. It replaces the
0.1 s virtual-point write with a dynamic PhysX actor and a 0.01 s four-motor controller. This does
not turn the synthetic ref5in design point into a hardware-identified aircraft. Hardware evidence
is tracked separately in `docs/navrl_hardware_identification_manifest.yaml`.

No PPO was started. The preregistered four-density gate did not pass, so fresh PPO remains blocked.

## Implemented contract (steps 1–6)

1. **Actor:** target is a gravity-enabled, collision-enabled 6-DoF rigid body. Its root state is the
   only target pose/velocity used by rewards and observations.
2. **Geometry:** collision, camera and LiDAR use the same oriented 0.28×0.28×0.12 m box and actor
   pose. The old 0.15/0.20 m sensor spheres are retained only for legacy/bounded lineages.
3. **Dynamics:** equivalent ref5in rigid-body mass/inertia, four 9.60 N motors, 0.0777817 m XY arms,
   0.04 s first-order motor response and 0.01 m yaw torque ratio. These are synthetic assumptions.
4. **Substep control:** motor allocation, saturation, attitude and velocity feedback run on every
   0.01 s PhysX step. Contact is accumulated over all ten substeps so a transient hit cannot vanish
   before the 0.1 s task boundary.
5. **Planner geometry:** each selected bar URDF exposes its exact collision half-extents. The target
   OBB support inflates each bar for point-to-AABB rollout; the old one-radius center proxy is not
   used by physical mode.
6. **Guards and verification:** physical mode requires `NAVRL_ROBOT=navrl_ref5in_quad`; the fresh
   launcher rejects every checkpoint argument. A fixed gate measures tracking, speed, PhysX
   contact, immediate plan feasibility, motor saturation, tilt and invalid state at 70/150/205/300.

The raw camera/LiDAR intersections and PhysX contact geometry are unified. The downstream camera
tracker still adds a 0.15 m target-size prior when converting visible surface depth to an estimated
centre range. That is an onboard estimator approximation, not hidden ground truth; it can bias the
range estimate for a yawed/tilted box and must be calibrated or randomized before a real-camera
claim.

## Preregistered validation result

Source: `results/navrl_physical_target_verification/summary.json`, seed 503, 32 envs × 280 measured
steps per density, mixed CV/waypoint, command 1.5 m/s.

| bars | speed ratio | tracking RMSE | contact steps | planner infeasible | invalid state | motor saturation | max tilt | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 70 | 0.919 | 0.233 m/s | 0.257% | 1.574% | 0.089% | 0.000% | 5.26° | FAIL |
| 150 | 0.880 | 0.279 m/s | 0.525% | 1.417% | 0.089% | 0.000% | 5.27° | FAIL |
| 205 | 0.832 | 0.316 m/s | 0.904% | 1.384% | 0.045% | 0.000% | 5.25° | FAIL |
| 300 | 0.738 | 0.351 m/s | 1.373% | 2.042% | 0.011% | 0.000% | 5.26° | FAIL |

Fixed gates were RMSE ≤0.35 m/s, speed ratio ≥0.80, contact ≤1%, immediate infeasible ≤1%, motor
saturation ≤15%, tilt ≤60°, invalid state =0. They were not moved after observing results.

## Interpretation and next decision

The controller is no longer the principal defect: motor saturation is zero in this run and attitude
stays near level. The density trend is instead a navigation/geometry limit. The 205-bar contact gate
now passes narrowly, but immediate route feasibility still misses its fixed 1% gate. At 300 bars the
target realizes only 73.8% of 1.5 m/s, contact exceeds 1%, and tracking RMSE also misses its gate.
Even 70/150 retain rare planner/boundary failures, so a fresh PPO would train against a target
environment that fails its own safety contract.

The final run also includes two audit corrections found after the first provisional measurement:
motor forces are submitted in the same PhysX substep in which they are computed (not one 0.01 s step
late), and arena validity checks the complete OBB support rather than its centre only. Physical
target resets now start inside the planner's own wall reserve. Earlier provisional numbers are not
canonical.

Next work is not “train longer.” First choose and preregister one of:

- a global/corridor route planner for the target, retaining the 1.5 m/s command but allowing safe
  commanded slowing; or
- a density-conditioned target-speed envelope, explicitly changing the experimental question from
  fixed-speed interception to physically feasible evasive flight.

After the chosen target contract passes the same frozen gate, run a short PPO smoke and only then a
fresh full lineage through `train_navrl_physical_fresh.sh`.

## Verification evidence

- Python compilation: PASS.
- ref5in/target URDF equivalence: 27/27 PASS.
- ref5in run contract: 12/12 PASS.
- target-motion/AABB tests: 11/11 PASS.
- live physical camera+LiDAR OBB sensor smoke: finite observations, PASS.
- checkpoint rejection launcher smoke: exit 4, PASS.
- four-density physical motion gate: FAIL as detailed above.
