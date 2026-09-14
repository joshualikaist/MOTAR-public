#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

# Fresh-only physical+waypoint lineage. The parent physical launcher and base v2 launcher each
# repeat the checkpoint/dynamics guards; this wrapper adds the exact route contract.
if [[ -n "${CKPT:-}" || -n "${CHECKPOINT:-}" ]]; then
  echo "[physical-routed] refusing CKPT/CHECKPOINT: global target routes require fresh PPO" >&2
  exit 4
fi
for arg in "$@"; do
  case "${arg}" in
    --checkpoint|--checkpoint=*|--resume_in_place|--branch_run)
      echo "[physical-routed] refusing resume/checkpoint flag: ${arg}" >&2
      exit 4
      ;;
  esac
done
if [[ "${NAVRL_TARGET_ROUTE_MODE:-}" == "global_astar_recovery_v2" ]]; then
  echo "[physical-routed] refusing recovery_v2 in the immutable global_astar_v1 launcher; use the recovery preflight" >&2
  exit 4
fi

# This is a frozen experiment tuple, not a convenient collection of defaults.  Assign every
# geometry/dynamics input unconditionally so a stale interactive shell cannot create a different
# route lineage under the canonical name.  Both lower launchers independently validate the tuple.
export NAVRL_ROBOT=navrl_ref5in_v2_quad
export NAVRL_PHYSICAL_GEOMETRY_VERSION=v2
export NAVRL_TARGET_BOX_XY_M=0.283
export NAVRL_TARGET_DYNAMICS=physical
export NAVRL_TARGET_ROUTE_MODE=global_astar_v1
export NAVRL_TARGET_PATTERN=waypoint
export NAVRL_PHYSICAL_ROUTED_CHILD=1
export NAVRL_ROUTED_CONTRACT_TOKEN=physx_ref5in_6dof_global_astar_aabb_nonoverlap_v2
export NAVRL_ARENA_XY=40
export NAVRL_ARENA_Z=3
export NAVRL_BAR_POOL=bars_h3
export NAVRL_BAR_X_MIN=0
export NAVRL_BAR_X_MAX=1
export NAVRL_PLACEMENT_MODE=footprint_clearance
export NAVRL_PLACEMENT_SURFACE_CLEARANCE_M=0.45
unset NAVRL_PLACEMENT_GAP_M NAVRL_PLACEMENT_TOUCH_M
export NAVRL_MAX_BARS=300
export NAVRL_TARGET_ROUTE_RESOLUTION_M=0.25
export NAVRL_TARGET_ROUTE_MAX_EXPANSIONS=50000
export NAVRL_TARGET_ROUTE_MAX_WAYPOINTS=128
export NAVRL_TARGET_ROUTE_REPLAN_COOLDOWN_STEPS=10
export NAVRL_TARGET_ROUTE_GOAL_TOLERANCE_M=0.05
export NAVRL_TARGET_ROUTE_MIN_GOAL_DISTANCE_M=6.0
export NAVRL_TARGET_ROUTE_GOAL_EXCLUSION_M=1.0
export NAVRL_TARGET_MAX_ACCEL=4.0
export NAVRL_TARGET_MAX_TURN_RATE_DEG=150.0
export NAVRL_TARGET_LOOKAHEAD_S=1.0
export NAVRL_TARGET_OBSTACLE_CLEARANCE=0.77
export NAVRL_TARGET_MASS_KG=1.20
export NAVRL_TARGET_MOTOR_ARM_XY_M=0.0777817
export NAVRL_TARGET_MAX_MOTOR_THRUST_N=9.60
export NAVRL_TARGET_MOTOR_TAU_S=0.04
export NAVRL_TARGET_YAW_TORQUE_RATIO_M=0.01
export NAVRL_TARGET_MAX_TILT_DEG=45.0
export NAVRL_TARGET_VEL_KP=2.5
export NAVRL_TARGET_ALT_KP=4.0
export NAVRL_TARGET_TRACKING_MARGIN_M=0.45
export NAVRL_TARGET_BOUNDARY_MARGIN_M=0.75
export NAVRL_V2_ALLOW_RESUME=0
export NAVRL_V2_PROFILE=main
export NUM_ENVS=128
export NAVRL_VISION=1
export NAVRL_PERCEPTION=1
export NAVRL_GENERAL_TRAIN=1
export NAVRL_GENERAL_GOAL_DIST_MIN=6
export NAVRL_GENERAL_GOAL_DIST_MAX=28
export NAVRL_DENSITY_CURRICULUM=1
export NAVRL_DENSITY_START=70
export NAVRL_DENSITY_FINAL=205
export NAVRL_DENSITY_STEP=15
export NAVRL_DENSITY_THRESHOLD_START=0.80
export NAVRL_DENSITY_THRESHOLD_END=0.70
export NAVRL_DENSITY_THRESHOLD_SCHEDULE=70:0.82,85:0.77,100:0.72,115:0.70
export NAVRL_DENSITY_WARMUP=1000
export NAVRL_DENSITY_CHECK_EPS=16384
export NAVRL_DENSITY_STRATIFIED_GATE=0
export NAVRL_DENSITY_STRATIFIED_FLOOR=0.55
export NAVRL_DENSITY_STRATIFIED_MIN_EPS=512
export NAVRL_DENSITY_MIN_EPOCHS=1000
export NAVRL_OBSTACLE_SELECTOR=cluster_sector
export NAVRL_GEOFENCE_ACTOR=0
export NAVRL_GEOFENCE_NOISE_STD_M=0
export NAVRL_GEOFENCE_DROPOUT=0
export NAVRL_DETECTOR_MIN_PIXELS=2
export NAVRL_TARGET_SPEED_MIN=0.3
export NAVRL_TARGET_SPEED_FINAL=1.5
export NAVRL_TARGET_SPEED_RAMP_EPOCHS=1
export NAVRL_LEARNING_RATE=3e-5
export NAVRL_SPEED_GOVERNOR=off

# Preflight validates the complete handoff even from a dirty worktree, but cannot create a source
# receipt or start Isaac Gym.  A real run snapshots clean committed runtime bytes and makes the task
# re-verify them at every checkpoint save.
ROUTED_PREFLIGHT="${NAVRL_TARGET_CONTRACT_PREFLIGHT_ONLY:-${NAVRL_V2_CONTRACT_PREFLIGHT_ONLY:-0}}"
if [[ "${ROUTED_PREFLIGHT}" == "1" ]]; then
  export NAVRL_TRAINING_SOURCE_MANIFEST=/preflight/routed_source_manifest.json
  export NAVRL_TRAINING_SOURCE_MANIFEST_SHA256="$(printf '0%.0s' {1..64})"
  export NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT=0
  export NAVRL_REQUIRE_CLEAN_TRAINING_SOURCE=0
else
  if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "${HOME}/miniconda3/envs/aerialgym/bin/python" ]]; then
      PYTHON="${HOME}/miniconda3/envs/aerialgym/bin/python"
    elif [[ -x /home/fair/miniconda3/envs/aerialgym/bin/python ]]; then
      PYTHON=/home/fair/miniconda3/envs/aerialgym/bin/python
    else
      PYTHON=python
    fi
  fi
  export PYTHON
  export PATH="$(dirname "${PYTHON}"):${PATH}"
  export PYTHONNOUSERSITE=1
  RECEIPT_ROOT="${SCRIPT_DIR}/train_source_receipts/physical_routed_s${SEED:-1}_$(date +%y%m%d_%H%M%S)_$$"
  RECEIPT_JSON="$(
    "${PYTHON}" "${REPO_ROOT}/tools/create_navrl_source_bundle.py" \
      create --output "${RECEIPT_ROOT}" --require-clean
  )"
  export NAVRL_TRAINING_SOURCE_MANIFEST="$(
    "${PYTHON}" -c 'import json,sys; print(json.loads(sys.argv[1])["manifest"])' "${RECEIPT_JSON}"
  )"
  export NAVRL_TRAINING_SOURCE_MANIFEST_SHA256="$(
    "${PYTHON}" -c 'import json,sys; print(json.loads(sys.argv[1])["manifest_sha256"])' "${RECEIPT_JSON}"
  )"
  export NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT=1
  export NAVRL_REQUIRE_CLEAN_TRAINING_SOURCE=1
fi

echo "[physical-routed] model=physx_ref5in_6dof_global_astar_aabb_nonoverlap_v2 fresh=1"
echo "[physical-routed] route=global_astar_v1 pattern=waypoint grid=0.25m goal_exclusion=1.0m fail_closed=1"
echo "[physical-routed] frozen | arena=40x40x3 pool=bars_h3 placement=footprint_clearance surface=0.45m overlap_fallback=off tracking=0.45 boundary=0.75 support=0.208912661m"
echo "[physical-routed] authority | mass=1.20kg thrust=9.60N arm=0.0777817m tau=0.04s tilt=45deg accel=4.0mps2 turn=150degps"
echo "[physical-routed] training | envs=128 density=70:15:205 max_pool=300 gate=70:0.82,85:0.77,100:0.72,115:0.70 dwell=1000 evidence=16384 speed=0.3:1.5@1 selector=cluster_sector lr=3e-5 governor=off"
echo "[physical-routed] source | preflight=${ROUTED_PREFLIGHT} receipt_required=${NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT} clean_required=${NAVRL_REQUIRE_CLEAN_TRAINING_SOURCE}"
exec "${SCRIPT_DIR}/train_navrl_physical_fresh.sh" "$@"
