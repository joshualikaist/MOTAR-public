#!/usr/bin/env bash
# Task v2 "search arena": NavRL-scale environment + interception objective. FRESH training.
#
# Why v2 (WORKLOG 2026-07-31): in the v1 24 x 24 arena the goal (4..16 m) was almost always
# inside the sensor horizon (LiDAR 12 m, forward camera 20 m), so the task degenerated into
# visible-target pursuit -- no search. v2 matches the reference NavRL scale (40 x 40 map,
# ~22 obstacles/100 m^2; reference env.py map_range [20,20], 350 obstacles) and pushes the
# goal-distance curriculum past the camera horizon so the drone must FIND the target with
# LiDAR + a 87-degree forward camera before it can intercept it.
#
# v2 vs v1 deltas (each env-gated; defaults preserve v1 exactly):
#   arena          24x24x3        -> 40x40x3          (NAVRL_ARENA_XY=40)
#   bar pool       2.0 m bars     -> 3.0 m full height (NAVRL_BAR_POOL=bars_h3; no fly-over,
#                                     like NavRL's mostly 4-6 m obstacles in a 4.5 m map)
#   placement      random+relax   -> navrl_band       (slit-free: touching-or->=0.8 m-gap;
#                                     the legacy relax made ~2.2 impassable slits/layout at 150 bars)
#   goal distance  4..16 m        -> 6..28 m           (28 m > camera 20 m -> search regime;
#                                     competence curriculum ramps k_max 10->28 as before)
#   episode        300 steps/30 s -> 600 steps/60 s    (search + longer traversals need time;
#                                     NavRL uses 2200 x 0.016 s = 35 s for navigation alone)
#   density        25..150 bars   -> 70..300 bars      (4.4..18.8/100m^2 over the full 1600 m^2
#                                     arena; step 15 = v1's 5-bar step scaled)
#   bar x band     0.13..0.96     -> 0.0..1.0          (the legacy window kept v1's left-to-right
#                                     spawn strip clear; with uniform v2 spawns it only left 17%
#                                     of the arena obstacle-free and inflated reported density)
#   promote gate   flat 0.70      -> 0.85 @70 -> 0.70 @300 (linear in bar count: demand mastery
#                                     at the easy end, avoid a permanent stall at the hard end)
#   target speed   U[0,1.5] / 3000-> U[0.3,vmax] over 300 epochs (always moving; the short ramp
#                                     recovers early survival learning and finishes 700 epochs
#                                     before density evidence starts, so the two gates do not overlap)
# Deliberately UNCHANGED (variable control): observation contract 898-D, LiDAR 12 m 72x4,
# 8 cluster-sector tokens, camera 87 deg @20 m, v_max 2.5, yaw 3.0, mixed pattern,
# squashed-Gaussian action policy, PPO hyperparameters, seed.
#
# VRAM: measured 6314 MiB / 8192 at 128 envs, 40x40, 300 full-height bars (2026-07-31).
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

_require_routed_value() {
    local name="$1" expected="$2" actual="${!1-}"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "[v2-search] routed contract mismatch: ${name}=${actual:-<unset>} expected=${expected}" >&2
        exit 2
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
        echo "[v2-search] routed training requires a valid source manifest path and SHA-256" >&2
        exit 2
    fi
}

# Prefer the current user's aerialgym environment so this launcher remains portable between the
# 3070 and 4 GB hosts. Keep the original 3070 path as a compatibility fallback; callers can still
# override either choice with PYTHON=/path/to/python.
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

# Target transition is checkpoint-compatible in tensor shape but not in meaning. Keep this base
# entry point on the historical legacy transition unless the physical-fresh wrapper marks
# the child invocation. In particular, do not inherit a stale NAVRL_TARGET_DYNAMICS=bounded or
# =physical from an interactive shell.
_TARGET_DYNAMICS_REQUESTED="${NAVRL_TARGET_DYNAMICS:-legacy}"
_TARGET_ROUTE_REQUESTED="${NAVRL_TARGET_ROUTE_MODE:-off}"
# Preserve the validated handoff in a shell-local value before clearing the environment marker.
# The old code tested NAVRL_V2_PHYSICAL_FRESH_CHILD again *after* unsetting it below, so every
# physical-fresh invocation silently fell through to the historical navrl_band placement.  That
# made the wrapper print footprint_clearance while the actual task received navrl_band.
_PHYSICAL_FRESH_CHILD="${NAVRL_V2_PHYSICAL_FRESH_CHILD:-0}"
if [[ "${_PHYSICAL_FRESH_CHILD}" == "1" ]]; then
    if [[ "${_TARGET_DYNAMICS_REQUESTED}" != "physical" ]]; then
        echo "[v2-search] physical child marker requires NAVRL_TARGET_DYNAMICS=physical" >&2
        exit 2
    fi
    if [[ "${_TARGET_ROUTE_REQUESTED}" != "off" && "${_TARGET_ROUTE_REQUESTED}" != "global_astar_v1" ]]; then
        echo "[v2-search] unsupported physical target route: ${_TARGET_ROUTE_REQUESTED}" >&2
        exit 2
    fi
    if [[ "${_TARGET_ROUTE_REQUESTED}" == "global_astar_v1" ]]; then
        if [[ "${NAVRL_V2_ROUTED_CHILD:-0}" != "1" ]]; then
            echo "[v2-search] global route requires canonical routed child handoff" >&2
            exit 2
        fi
        _validate_routed_contract
        _validate_routed_source_contract
    elif [[ "${NAVRL_V2_ROUTED_CHILD:-0}" == "1" || -n "${NAVRL_ROUTED_CONTRACT_TOKEN:-}" ]]; then
        echo "[v2-search] stale routed child marker/token with route=off" >&2
        exit 2
    fi
else
    if [[ "${_TARGET_DYNAMICS_REQUESTED}" != "legacy" ]]; then
        echo "[v2-search] refusing inherited target dynamics: ${_TARGET_DYNAMICS_REQUESTED}" >&2
        echo "[v2-search] use train_navrl_physical_fresh.sh for physical; bounded has no canonical launcher." >&2
        exit 2
    fi
    if [[ "${_TARGET_ROUTE_REQUESTED}" != "off" ]]; then
        echo "[v2-search] target route requires the dedicated physical routed fresh launcher" >&2
        exit 2
    fi
fi
export NAVRL_TARGET_DYNAMICS="${_TARGET_DYNAMICS_REQUESTED}"
export NAVRL_TARGET_ROUTE_MODE="${_TARGET_ROUTE_REQUESTED}"
unset NAVRL_V2_PHYSICAL_FRESH_CHILD
unset NAVRL_V2_ROUTED_CHILD NAVRL_ROUTED_CONTRACT_TOKEN

# This entry point is fresh-training by default.  Its CLI is deliberately a closed contract:
# fresh runs accept no runner arguments, while a dedicated continuation wrapper may pass only the
# exact checkpoint tuple below.  Appending arbitrary arguments after our pinned flags would let a
# caller override --file/--task/--seed/--max_epochs and silently invalidate the experiment.
if [[ "${NAVRL_V2_ALLOW_RESUME:-0}" != "1" ]]; then
    if [[ -n "${CKPT:-}" ]]; then
        echo "[v2-search] refusing inherited CKPT in fresh mode: ${CKPT}" >&2
        echo "[v2-search] use train_navrl_v2_recover_safe.sh for recovery." >&2
        exit 2
    fi
    if (( $# != 0 )); then
        echo "[v2-search] fresh mode accepts no CLI arguments: $*" >&2
        echo "[v2-search] configure the pinned launcher through its documented environment only." >&2
        exit 2
    fi
    # Recovery provenance is meaningful only when a dedicated continuation wrapper supplies it.
    # An old interactive-shell export must not label an unrelated fresh run as audited recovery.
    unset NAVRL_RECOVERY_STAGE NAVRL_RECOVERY_SOURCE_EPOCH
    unset NAVRL_RECOVERY_SOURCE_SHA256 NAVRL_RECOVERY_SMOKE_REQUIRED_EPOCHS
    unset NAVRL_RECOVERY_SMOKE_BARS NAVRL_RECOVERY_EVAL_ATTESTATION_SHA256
    unset NAVRL_RECOVERY_EVAL_ATTESTATION_B64
else
    if (( $# != 3 )); then
        echo "[v2-search] continuation mode requires exactly: --checkpoint \"\$CKPT\" --branch_run" >&2
        exit 2
    fi
    if [[ -z "${CKPT:-}" || "$1" != "--checkpoint" || "$2" != "${CKPT}" || "$3" != "--branch_run" ]]; then
        echo "[v2-search] refusing non-canonical continuation arguments: $*" >&2
        echo "[v2-search] expected exactly: --checkpoint \"${CKPT:-<unset>}\" --branch_run" >&2
        exit 2
    fi
fi

export MAX_EPOCHS="${MAX_EPOCHS:-30000}"
export SEED="${SEED:-1}"
case "${NAVRL_V2_PROFILE:-main}" in
    main)
        unset GPU4GB
        export AERIAL_GYM_SIM_NAME=base_sim
        export NUM_ENVS=128
        ;;
    4gb)
        export GPU4GB=1
        export AERIAL_GYM_SIM_NAME=base_sim_4gb
        export NUM_ENVS=64
        ;;
    *)
        echo "[v2-search] NAVRL_V2_PROFILE must be main or 4gb; got ${NAVRL_V2_PROFILE}." >&2
        exit 2
        ;;
esac
export FILE=ppo_navrl_perception_transformer.yaml
export TASK=navrl_task
export HEADLESS=True
export NAVRL_SEED="${SEED}"
export AERIAL_RUN_TAG="${AERIAL_RUN_TAG:-v2-search-fresh-s${SEED}}"
export TRAIN_SESSION_LOG="${TRAIN_SESSION_LOG:-train_session_logs/v2_search_fresh_$(date +%y%m%d_%H%M%S).log}"
# Training-only entry point: stale evaluation/viewer variables must not mutate reset logic or
# terminate through an evaluation path while still looking like a normal v2 run.
unset NAVRL_GENERAL_EVAL NAVRL_INTERACTIVE NAVRL_BULK_EVAL NAVRL_BULK_EVAL_JSON
unset NAVRL_GENERAL_RESULTS_JSON NAVRL_EVAL_CHECKPOINT NAVRL_EVAL_TARGET_SPEED_FINAL
unset NAVRL_EVAL_FULL_DISTRIBUTION
unset NAVRL_EVAL_RUN_NONCE NAVRL_EVAL_PROFILE NAVRL_SIM_PHYSICS_CONTRACT
unset NAVRL_LEGACY_VISION NAVRL_OOB_PROBE
unset NAVRL_NETWORK_OVERRIDE NAVRL_V2_FORCE ALLOW_CONCURRENT

# ---- v2 environment ----
export NAVRL_ARENA_XY=40
export NAVRL_ARENA_Z=3
export NAVRL_BAR_POOL=bars_h3
if [[ "${_PHYSICAL_FRESH_CHILD}" == "1" ]]; then
    export NAVRL_PLACEMENT_MODE=footprint_clearance
    export NAVRL_PLACEMENT_SURFACE_CLEARANCE_M=0.45
    unset NAVRL_PLACEMENT_TOUCH_M NAVRL_PLACEMENT_GAP_M
else
    export NAVRL_PLACEMENT_MODE=navrl_band
    export NAVRL_PLACEMENT_TOUCH_M=0.4
    export NAVRL_PLACEMENT_GAP_M=1.6
    unset NAVRL_PLACEMENT_SURFACE_CLEARANCE_M
fi
export NAVRL_EPISODE_LEN_STEPS=600
export NAVRL_MAX_BARS=300
# Full-width obstacle band. The legacy 0.13..0.96 window kept the v1 left-to-right spawn strip
# clear; v2 spawns drone AND target uniformly (NAVRL_GENERAL_TRAIN=1), so that window only left
# 17% of the arena permanently empty -- episodes spawning there were straight-line pursuit at any
# density -- and inflated reported density (1328 m^2 band vs 1600 m^2 flyable).
export NAVRL_BAR_X_MIN=0.0
export NAVRL_BAR_X_MAX=1.0

# ---- v2 objective: goal beyond the sensor horizon ----
# Keep the canonical 6..28 m default, but let a closed child launcher deliberately oversample one
# preregistered radial band. Hardcoding these values made NAVRL_K_MIN_FINAL=20 look like an applied
# minimum even though general-spawn sampling always read this separate 6 m variable.
export NAVRL_GENERAL_GOAL_DIST_MIN="${NAVRL_GENERAL_GOAL_DIST_MIN:-6}"
export NAVRL_GENERAL_GOAL_DIST_MAX="${NAVRL_GENERAL_GOAL_DIST_MAX:-28}"
export NAVRL_K_COMPETENCE=1
export NAVRL_K_FINAL=28
export NAVRL_K_MIN_FINAL=20
export NAVRL_K_MIN_RAMP_START=2000
export NAVRL_K_MIN_RAMP_EPOCHS=3000
export NAVRL_K_WARMUP=3000
export NAVRL_K_THRESHOLD=0.6
export NAVRL_K_STEP=2.0
export NAVRL_K_CHECK=2048

# ---- density curriculum, v1-equivalent per-area schedule ----
# Overridable so a fixed-density probe (train_navrl_v2_ceiling_probe.sh) can freeze the curriculum
# while inheriting the rest of this contract verbatim. Unset => the normal curriculum run.
export NAVRL_DENSITY_CURRICULUM="${NAVRL_DENSITY_CURRICULUM:-1}"
export NAVRL_DENSITY_START="${NAVRL_DENSITY_START:-70}"
export NAVRL_DENSITY_FINAL="${NAVRL_DENSITY_FINAL:-300}"
export NAVRL_DENSITY_STEP=15
# Per-density threshold ramp (2026-07-31): flat 0.70 risks the curriculum stalling forever if the
# achievable capture ceiling keeps falling with density (the same failure mode behind v1's 100-bar
# plateau). Require MORE capture to leave the easy end, LESS to leave the hard end. Unset either
# var to fall back to the flat NAVRL_DENSITY_THRESHOLD.
#
# START was 0.85 (a design guess) and got MEASURED at 70 bars on 2026-07-31 (run ppo_260731_1722,
# 2300 epochs): two full 16,384-episode gate windows scored 0.816 and 0.837, crash decayed to an
# extrapolated floor of 13.1% (fit tau~650 epochs, already converged), and with the ~2.6% timeout
# base that puts the achievable capture ceiling at ~0.843 -- structurally BELOW 0.85. Root cause is
# representation, not geometry: 23.6% of crashed bars were outside the 240-degree token window and
# 11.7% of in-window hits had no token (capacity 8 vs ~12 bars in FOV). 0.80 sits under the
# measured ceiling with margin while still demanding near-ceiling mastery before promotion.
export NAVRL_DENSITY_THRESHOLD_START="${NAVRL_DENSITY_THRESHOLD_START:-0.80}"
export NAVRL_DENSITY_THRESHOLD_END="${NAVRL_DENSITY_THRESHOLD_END:-0.70}"
# Explicit per-density gate (overrides the ramp above). The achievable ceiling is not linear in
# density, so a two-point ramp cannot express it: 70 bars measured 0.843, and the early levels
# should still demand near-ceiling mastery while the dense end settles at the 0.70 floor. Every
# density the curriculum can occupy (70, 85, 100, 115, ...) is a knot, and the schedule holds the
# last knot beyond 115.
export NAVRL_DENSITY_THRESHOLD_SCHEDULE="${NAVRL_DENSITY_THRESHOLD_SCHEDULE:-70:0.82,85:0.77,100:0.72,115:0.70}"
export NAVRL_DENSITY_WARMUP="${NAVRL_DENSITY_WARMUP:-1000}"
export NAVRL_DENSITY_CHECK_EPS="${NAVRL_DENSITY_CHECK_EPS:-16384}"
export NAVRL_DENSITY_STRATIFIED_GATE="${NAVRL_DENSITY_STRATIFIED_GATE:-0}"
export NAVRL_DENSITY_STRATIFIED_FLOOR="${NAVRL_DENSITY_STRATIFIED_FLOOR:-0.55}"
export NAVRL_DENSITY_STRATIFIED_MIN_EPS="${NAVRL_DENSITY_STRATIFIED_MIN_EPS:-512}"
export NAVRL_DENSITY_EASY_GOAL_MIX=0
export NAVRL_DENSITY_EASY_GOAL_MIN=5.0
export NAVRL_DENSITY_EASY_GOAL_MAX=10.0
# Dwell at each density for at least this many epochs before promoting, even when the capture gate
# already passes. Without it the curriculum chains promotions as fast as evidence windows fill, so
# no level ever converges and every metric only ever tracks rising difficulty.
export NAVRL_DENSITY_MIN_EPOCHS="${NAVRL_DENSITY_MIN_EPOCHS:-1000}"
# NAVRL_NUM_BARS pins the active density; clearing it is what lets the curriculum own the value.
# A fixed-density probe sets it deliberately, so only clear it when the curriculum is actually on --
# otherwise the probe's density silently reverts to the config default.
if [[ "${NAVRL_DENSITY_CURRICULUM}" == "1" ]]; then
    unset NAVRL_NUM_BARS
fi
unset NAVRL_FIXED_BARS NAVRL_CONTROLLED_ABLATION

# ---- UNCHANGED sensor/representation/action contract (variable control) ----
export NAVRL_VISION=1
export NAVRL_PERCEPTION=1
export NAVRL_GENERAL_TRAIN=1
export NAVRL_PERCEPTION_PERTURB=0
export NAVRL_TILT_COMP=1
export NAVRL_MAX_OBSTACLES=8
export NAVRL_OBSTACLE_FOV_DEG=240
# Overridable so a representation A/B (train_navrl_v2_ttc_ab.sh) can swap the selector while
# inheriting the rest of this contract. Hardcoding it silently discarded the experimental variable
# and made both arms run the SAME condition -- an A/B failure that is invisible because both arms
# still train normally. Unset => cluster_sector, the v2 default.
export NAVRL_OBSTACLE_SELECTOR="${NAVRL_OBSTACLE_SELECTOR:-cluster_sector}"
export NAVRL_OBSTACLE_CLUSTER_GAP_M=0.45
export NAVRL_OBSTACLE_SECTORS=8
export NAVRL_OBSTACLE_SUPPRESS_DEG=10
export NAVRL_OBSTACLE_TTC_IDLE_S=30.0
export NAVRL_OBSTACLE_TTC_MIN_SPEED=0.15
export NAVRL_CORRIDOR_TOKENS=0
export NAVRL_CORRIDOR_HORIZON_M=6.0
export NAVRL_CORRIDOR_MIN_WIDTH_M=0.55
# Opt-in mapped-geofence observation. Defaults keep the historical 898-D schema; the dedicated
# active-search A/B launcher sets actor=1 for its fresh geofence arm.
export NAVRL_GEOFENCE_ACTOR="${NAVRL_GEOFENCE_ACTOR:-0}"
export NAVRL_GEOFENCE_NOISE_STD_M="${NAVRL_GEOFENCE_NOISE_STD_M:-0}"
export NAVRL_GEOFENCE_DROPOUT="${NAVRL_GEOFENCE_DROPOUT:-0}"
export NAVRL_LIDAR_HBEAMS=72
export NAVRL_LIDAR_VBEAMS=4
export NAVRL_LIDAR_RANGE=12
export NAVRL_MAX_VELOCITY=2.5
export NAVRL_ALT_HOLD_VMAX=2.5
export NAVRL_YAW_RATE_MAX=3.0
export NAVRL_MAX_TILT_DEG=45.0
export NAVRL_FOV_CURRICULUM_EPOCHS=3000
# Overridable so a sensor-model experiment (detection-range stage 1, prereg 2026-08-22) can train
# at the Johnson/CNN-grounded 50 px^2 detection floor while inheriting the rest of this contract.
# Hard-coding it did the same damage the selector comment above describes, but more quietly: a
# child launcher exporting 50 would have trained at 2, both arms would still have trained
# normally, and the resulting checkpoints would carry cfg_detector_min_pixels=2 -- so the honest
# sensor the experiment is defined by would simply not have happened. Unset => 2, the v2 default.
export NAVRL_DETECTOR_MIN_PIXELS="${NAVRL_DETECTOR_MIN_PIXELS:-2}"
export NAVRL_DETECTOR_THRESHOLD=0.55
unset NAVRL_DETECTOR_CHECKPOINT
export NAVRL_DETECTION_DROPOUT=0.3
export NAVRL_RGB_NOISE_STD=0.015
export NAVRL_DEPTH_NOISE_STD=0.02
# Target speed: always moving at >=0.3 m/s, with the upper support increasing to 1.5 m/s over a
# short 300-epoch ramp. The measured no-ramp v3 pilot learned, but reached 10-epoch rolling capture
# 0.50 at epoch 140 versus 53 for the old ramped run; the comparison is confounded by the simultaneous
# full-width bar-band change, so keep only a short scaffold. It ends at epoch 300, well before density
# evidence starts at epoch 1000, avoiding the old 3000-epoch overlap with the density curriculum.
export NAVRL_TARGET_SPEED_MIN=0.3
export NAVRL_TARGET_SPEED_FINAL="${NAVRL_TARGET_SPEED_FINAL:-1.5}"
export NAVRL_TARGET_SPEED_RAMP_EPOCHS="${NAVRL_TARGET_SPEED_RAMP_EPOCHS:-300}"
if [[ "${NAVRL_TARGET_ROUTE_MODE}" == "global_astar_v1" ]]; then
    export NAVRL_TARGET_PATTERN=waypoint
else
    export NAVRL_TARGET_PATTERN=mixed
fi
unset NAVRL_TARGET_SPEED
if [[ "${NAVRL_TARGET_CONTRACT_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "[v2-search] TARGET CONTRACT PREFLIGHT PASS | dynamics=${NAVRL_TARGET_DYNAMICS} route=${NAVRL_TARGET_ROUTE_MODE} pattern=${NAVRL_TARGET_PATTERN} placement=${NAVRL_PLACEMENT_MODE} surface_clearance=${NAVRL_PLACEMENT_SURFACE_CLEARANCE_M:-unset} density=${NAVRL_DENSITY_START}->${NAVRL_DENSITY_FINAL} fixed_bars=${NAVRL_NUM_BARS:-curriculum} speed_final=${NAVRL_TARGET_SPEED_FINAL} ramp=${NAVRL_TARGET_SPEED_RAMP_EPOCHS}"
    exit 0
fi
export NAVRL_ACTION_POLICY=squashed_gaussian
export NAVRL_ACTION_STD=0.35,0.35,0.05,0.08
export NAVRL_ACTION_MU_SCALE=1.0,0.4,1.0,1.0
export NAVRL_ENTROPY_COEF=0.0
export NAVRL_ACTION_DIAG=1
# 1e-4 remained quiet for thousands of epochs, then produced KL 0.8--2.7 and destroyed the
# actor in ~50 epochs. 3e-5 is the conservative fresh default selected below that failed LR;
# recovery uses the directly smoke-tested 5e-6 in its dedicated launcher.
export NAVRL_LEARNING_RATE="${NAVRL_LEARNING_RATE:-3e-5}"
export NAVRL_RESET_ACTOR_OPTIMIZER=0
export NAVRL_PPO_LOG_RATIO_CLAMP=10.0
export NAVRL_PPO_KL_STOP=0.04
export NAVRL_PPO_EPOCH_ROLLBACK=1
export NAVRL_PPO_ROLLBACK_LR_FACTOR=0.5
export NAVRL_PPO_ROLLBACK_MIN_LR=1e-6
export NAVRL_PPO_ROLLBACK_PATIENCE=5
export NAVRL_DENSITY_GUARD_WINDOW_EPOCHS=50
export NAVRL_DENSITY_GUARD_MIN_EPOCHS=100
export NAVRL_DENSITY_GUARD_MIN_PEAK=0.50
export NAVRL_DENSITY_GUARD_DROP=0.25
export NAVRL_DENSITY_GUARD_PATIENCE=25
# Protect every pre-tanh action axis. The former y-only setting left x/z/yaw free to saturate and
# made tanh action replay non-invertible even while the lateral diagnostic looked healthy.
export NAVRL_LATENT_MARGIN=2.0,1.25,2.0,2.0
export NAVRL_LATENT_MARGIN_Y=1.25
export NAVRL_LATENT_MARGIN_COEF=0.01
unset NAVRL_LATERAL_BIAS_COEF NAVRL_REFLECTION_COEF NAVRL_TRUNCATED_DMIN
export NAVRL_OOB_MARGIN=1.0
export NAVRL_CRASH_DIAG=1
export NAVRL_BAR_PROBE=1

mkdir -p train_session_logs
ACTIVE_PIDS="$(pgrep -f '[r]unner.py .*--task navrl_task .*--train' | tr '\n' ' ' || true)"
if [[ -n "${ACTIVE_PIDS// }" ]]; then
    echo "[v2-search] refusing duplicate NavRL training; active PID(s): ${ACTIVE_PIDS}" >&2
    exit 3
fi
exec 9>train_session_logs/.navrl_training.lock
if ! flock -n 9; then
    echo "[v2-search] another NavRL launcher holds the global training lock." >&2
    exit 3
fi

V2_RUN_KIND="FRESH"
if [[ "${NAVRL_V2_ALLOW_RESUME:-0}" == "1" ]]; then
    V2_RUN_KIND="CONTINUATION"
fi
echo "[v2-search] ${V2_RUN_KIND} | arena=${NAVRL_ARENA_XY}m pool=${NAVRL_BAR_POOL} placement=${NAVRL_PLACEMENT_MODE} target=${NAVRL_TARGET_DYNAMICS} route=${NAVRL_TARGET_ROUTE_MODE} pattern=${NAVRL_TARGET_PATTERN}"
echo "[v2-search] executable | profile=${NAVRL_V2_PROFILE:-main} file=${FILE} task=${TASK} sim=${AERIAL_GYM_SIM_NAME} envs=${NUM_ENVS} seed=${SEED}"
echo "[v2-search] goal ${NAVRL_GENERAL_GOAL_DIST_MIN}..${NAVRL_GENERAL_GOAL_DIST_MAX}m (camera 20m -> search) episode=${NAVRL_EPISODE_LEN_STEPS} steps"
DENSITY_START_PER_100M2="$(awk -v bars="${NAVRL_DENSITY_START}" 'BEGIN {printf "%.2f", bars / 16.0}')"
DENSITY_FINAL_PER_100M2="$(awk -v bars="${NAVRL_DENSITY_FINAL}" 'BEGIN {printf "%.2f", bars / 16.0}')"
echo "[v2-search] density ${NAVRL_DENSITY_START}->${NAVRL_DENSITY_FINAL} bars step=${NAVRL_DENSITY_STEP} (${DENSITY_START_PER_100M2}->${DENSITY_FINAL_PER_100M2} /100m2 over 1600m2) threshold ${NAVRL_DENSITY_THRESHOLD_SCHEDULE:-${NAVRL_DENSITY_THRESHOLD_START}->${NAVRL_DENSITY_THRESHOLD_END}}"
echo "[v2-search] target speed U[${NAVRL_TARGET_SPEED_MIN}, vmax] m/s, vmax->${NAVRL_TARGET_SPEED_FINAL} by epoch ${NAVRL_TARGET_SPEED_RAMP_EPOCHS} | bar band x=[${NAVRL_BAR_X_MIN}, ${NAVRL_BAR_X_MAX}]"
echo "[v2-search] PPO safety | lr=${NAVRL_LEARNING_RATE} KL=${NAVRL_PPO_KL_STOP} epoch_rollback=${NAVRL_PPO_EPOCH_ROLLBACK} latent_margin=${NAVRL_LATENT_MARGIN}@${NAVRL_LATENT_MARGIN_COEF}"
if [[ "${NAVRL_SPEED_GOVERNOR:-off}" != "off" ]]; then
    echo "[v2-search] speed governor | mode=${NAVRL_SPEED_GOVERNOR} fixed=${NAVRL_SPEED_GOVERNOR_FIXED_MPS:-2.0} free=${NAVRL_SPEED_GOVERNOR_FREE_MPS:-3.5355} slow=${NAVRL_SPEED_GOVERNOR_SLOW_M:-3.0} release=${NAVRL_SPEED_GOVERNOR_RELEASE_M:-5.0}"
fi
if [[ "${NAVRL_V2_CONTRACT_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "[v2-search] PREFLIGHT PASS (child handoff validated; training not started)"
    exit 0
fi
exec "${SCRIPT_DIR}/train_navrl.sh" --seed "${SEED}" --max_epochs "${MAX_EPOCHS}" --disable_collapse_early_stop "$@"
