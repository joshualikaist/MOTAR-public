#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

_require_routed_value() {
  local name="$1" expected="$2" actual="${!1-}"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "[physical-fresh] routed contract mismatch: ${name}=${actual:-<unset>} expected=${expected}" >&2
    exit 4
  fi
}

_validate_routed_contract() {
  _require_routed_value NAVRL_ROUTED_CONTRACT_TOKEN physx_ref5in_6dof_global_astar_aabb_nonoverlap_v2
  _require_routed_value NAVRL_ROBOT navrl_ref5in_v2_quad
  _require_routed_value NAVRL_PHYSICAL_GEOMETRY_VERSION v2
  _require_routed_value NAVRL_TARGET_BOX_XY_M 0.283
  _require_routed_value NAVRL_TARGET_DYNAMICS physical
  _require_routed_value NAVRL_TARGET_ROUTE_MODE global_astar_v1
  _require_routed_value NAVRL_TARGET_PATTERN waypoint
  _require_routed_value NAVRL_ARENA_XY 40
  _require_routed_value NAVRL_ARENA_Z 3
  _require_routed_value NAVRL_BAR_POOL bars_h3
  _require_routed_value NAVRL_BAR_X_MIN 0
  _require_routed_value NAVRL_BAR_X_MAX 1
  _require_routed_value NAVRL_PLACEMENT_MODE footprint_clearance
  _require_routed_value NAVRL_PLACEMENT_SURFACE_CLEARANCE_M 0.45
  _require_routed_value NAVRL_MAX_BARS 300
  _require_routed_value NAVRL_TARGET_ROUTE_RESOLUTION_M 0.25
  _require_routed_value NAVRL_TARGET_ROUTE_MAX_EXPANSIONS 50000
  _require_routed_value NAVRL_TARGET_ROUTE_MAX_WAYPOINTS 128
  _require_routed_value NAVRL_TARGET_ROUTE_REPLAN_COOLDOWN_STEPS 10
  _require_routed_value NAVRL_TARGET_ROUTE_GOAL_TOLERANCE_M 0.05
  _require_routed_value NAVRL_TARGET_ROUTE_MIN_GOAL_DISTANCE_M 6.0
  _require_routed_value NAVRL_TARGET_ROUTE_GOAL_EXCLUSION_M 1.0
  _require_routed_value NAVRL_TARGET_MAX_ACCEL 4.0
  _require_routed_value NAVRL_TARGET_MAX_TURN_RATE_DEG 150.0
  _require_routed_value NAVRL_TARGET_LOOKAHEAD_S 1.0
  _require_routed_value NAVRL_TARGET_OBSTACLE_CLEARANCE 0.77
  _require_routed_value NAVRL_TARGET_MASS_KG 1.20
  _require_routed_value NAVRL_TARGET_MOTOR_ARM_XY_M 0.0777817
  _require_routed_value NAVRL_TARGET_MAX_MOTOR_THRUST_N 9.60
  _require_routed_value NAVRL_TARGET_MOTOR_TAU_S 0.04
  _require_routed_value NAVRL_TARGET_YAW_TORQUE_RATIO_M 0.01
  _require_routed_value NAVRL_TARGET_MAX_TILT_DEG 45.0
  _require_routed_value NAVRL_TARGET_VEL_KP 2.5
  _require_routed_value NAVRL_TARGET_ALT_KP 4.0
  _require_routed_value NAVRL_TARGET_TRACKING_MARGIN_M 0.45
  _require_routed_value NAVRL_TARGET_BOUNDARY_MARGIN_M 0.75
  _require_routed_value NAVRL_V2_ALLOW_RESUME 0
  _require_routed_value NAVRL_V2_PROFILE main
  _require_routed_value NUM_ENVS 128
  _require_routed_value NAVRL_VISION 1
  _require_routed_value NAVRL_PERCEPTION 1
  _require_routed_value NAVRL_GENERAL_TRAIN 1
  _require_routed_value NAVRL_GENERAL_GOAL_DIST_MIN 6
  _require_routed_value NAVRL_GENERAL_GOAL_DIST_MAX 28
  _require_routed_value NAVRL_DENSITY_CURRICULUM 1
  _require_routed_value NAVRL_DENSITY_START 70
  _require_routed_value NAVRL_DENSITY_FINAL 205
  _require_routed_value NAVRL_DENSITY_STEP 15
  _require_routed_value NAVRL_DENSITY_THRESHOLD_START 0.80
  _require_routed_value NAVRL_DENSITY_THRESHOLD_END 0.70
  _require_routed_value NAVRL_DENSITY_THRESHOLD_SCHEDULE 70:0.82,85:0.77,100:0.72,115:0.70
  _require_routed_value NAVRL_DENSITY_WARMUP 1000
  _require_routed_value NAVRL_DENSITY_CHECK_EPS 16384
  _require_routed_value NAVRL_DENSITY_STRATIFIED_GATE 0
  _require_routed_value NAVRL_DENSITY_STRATIFIED_FLOOR 0.55
  _require_routed_value NAVRL_DENSITY_STRATIFIED_MIN_EPS 512
  _require_routed_value NAVRL_DENSITY_MIN_EPOCHS 1000
  _require_routed_value NAVRL_OBSTACLE_SELECTOR cluster_sector
  _require_routed_value NAVRL_GEOFENCE_ACTOR 0
  _require_routed_value NAVRL_GEOFENCE_NOISE_STD_M 0
  _require_routed_value NAVRL_GEOFENCE_DROPOUT 0
  _require_routed_value NAVRL_DETECTOR_MIN_PIXELS 2
  _require_routed_value NAVRL_TARGET_SPEED_MIN 0.3
  _require_routed_value NAVRL_TARGET_SPEED_FINAL 1.5
  _require_routed_value NAVRL_TARGET_SPEED_RAMP_EPOCHS 1
  _require_routed_value NAVRL_LEARNING_RATE 3e-5
  _require_routed_value NAVRL_SPEED_GOVERNOR off
}

_validate_routed_source_contract() {
  local preflight="${NAVRL_TARGET_CONTRACT_PREFLIGHT_ONLY:-${NAVRL_V2_CONTRACT_PREFLIGHT_ONLY:-0}}"
  if [[ "${preflight}" == "1" ]]; then
    _require_routed_value NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT 0
    _require_routed_value NAVRL_REQUIRE_CLEAN_TRAINING_SOURCE 0
    return
  fi
  _require_routed_value NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT 1
  _require_routed_value NAVRL_REQUIRE_CLEAN_TRAINING_SOURCE 1
  if [[ ! -f "${NAVRL_TRAINING_SOURCE_MANIFEST:-}" \
        || ! "${NAVRL_TRAINING_SOURCE_MANIFEST_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[physical-fresh] routed training requires a valid source manifest path and SHA-256" >&2
    exit 4
  fi
}

# New physical-platform lineage. Deliberately fresh-only: legacy/bounded checkpoints encode a
# different target state transition and cannot be resumed merely because tensor shapes match.
if [[ -n "${CKPT:-}" || -n "${CHECKPOINT:-}" ]]; then
  echo "[physical-fresh] refusing CKPT/CHECKPOINT: physical 6-DoF target requires fresh PPO" >&2
  exit 4
fi
for arg in "$@"; do
  case "${arg}" in
    --checkpoint|--checkpoint=*|--resume_in_place|--branch_run)
      echo "[physical-fresh] refusing resume/checkpoint flag: ${arg}" >&2
      exit 4
      ;;
  esac
done

export NAVRL_ROBOT=navrl_ref5in_v2_quad
export NAVRL_PHYSICAL_GEOMETRY_VERSION=v2
export NAVRL_TARGET_BOX_XY_M=0.283
export NAVRL_TARGET_DYNAMICS=physical
if [[ "${NAVRL_PHYSICAL_ROUTED_CHILD:-0}" == "1" ]]; then
  _validate_routed_contract
  _validate_routed_source_contract
  export NAVRL_V2_ROUTED_CHILD=1
else
  if [[ "${NAVRL_TARGET_ROUTE_MODE:-off}" != "off" ]]; then
    echo "[physical-fresh] canonical physical lineage refuses target route; use train_navrl_physical_routed_fresh.sh" >&2
    exit 4
  fi
  export NAVRL_TARGET_ROUTE_MODE=off
  unset NAVRL_V2_ROUTED_CHILD NAVRL_ROUTED_CONTRACT_TOKEN
fi
unset NAVRL_PHYSICAL_ROUTED_CHILD
# Explicit child-entry marker. The base v2 launcher otherwise forces legacy dynamics so a stale
# interactive-shell NAVRL_TARGET_DYNAMICS cannot silently switch the canonical training lineage.
export NAVRL_V2_PHYSICAL_FRESH_CHILD=1
export NAVRL_VISION="${NAVRL_VISION:-1}"
export NAVRL_PERCEPTION="${NAVRL_PERCEPTION:-1}"
export NAVRL_GENERAL_TRAIN="${NAVRL_GENERAL_TRAIN:-1}"
export NAVRL_ARENA_XY="${NAVRL_ARENA_XY:-40}"
export NAVRL_ARENA_Z="${NAVRL_ARENA_Z:-3}"
export NAVRL_BAR_POOL="${NAVRL_BAR_POOL:-bars_h3}"
export NAVRL_BAR_X_MIN="${NAVRL_BAR_X_MIN:-0}"
export NAVRL_BAR_X_MAX="${NAVRL_BAR_X_MAX:-1}"
export NAVRL_PLACEMENT_MODE="${NAVRL_PLACEMENT_MODE:-footprint_clearance}"
export NAVRL_PLACEMENT_SURFACE_CLEARANCE_M="${NAVRL_PLACEMENT_SURFACE_CLEARANCE_M:-0.45}"
export NAVRL_MAX_BARS="${NAVRL_MAX_BARS:-300}"
export NAVRL_DENSITY_FINAL="${NAVRL_DENSITY_FINAL:-205}"
export NAVRL_TARGET_SPEED_FINAL="${NAVRL_TARGET_SPEED_FINAL:-1.5}"
export NAVRL_TARGET_SPEED_RAMP_EPOCHS="${NAVRL_TARGET_SPEED_RAMP_EPOCHS:-1}"

echo "[physical-fresh] robot=$NAVRL_ROBOT target=$NAVRL_TARGET_DYNAMICS fresh=1"
echo "[physical-fresh] placement=$NAVRL_PLACEMENT_MODE surface_clearance=${NAVRL_PLACEMENT_SURFACE_CLEARANCE_M}m overlap_fallback=off"
echo "[physical-fresh] WARNING: ref5in is an internally consistent simulation design point; hardware BOM/ID is pending"
exec "${SCRIPT_DIR}/train_navrl_v2_search.sh" "$@"
