#!/usr/bin/env bash
# Held-out density sweep for a TASK-V2 (search arena) checkpoint.
#
# v2 differs from v1 in the TASK, not the observation width (both 898-D), so a v2 checkpoint
# loads without error in the v1 arena and would be scored on a completely different problem.
# This script pins the v2 evaluation contract, checks checkpoint provenance, and requires one
# machine-readable bulk-evaluation JSON result per density before writing a consolidated CSV.
#
# ALWAYS pass last_gen_ppo_ep_* -- gen_ppo.pth is the best-reward (low-density) policy.
#
# Usage:
#   ./eval_navrl_v2_density_sweep.sh runs/ppo_XXXX_v2/nn/last_gen_ppo_ep_9000.pth
#   ./eval_navrl_v2_density_sweep.sh <ckpt> 2049                 # episodes per cell
#   GPU4GB=1 ./eval_navrl_v2_density_sweep.sh <ckpt> # 4 GB machine (1650 Ti)
#
# Optional outputs/conditions:
#   NAVRL_V2_DENSITIES="70 150 210 280"
#   NAVRL_V2_RESULT_DIR=/absolute/or/caller-relative/output/directory
#   NAVRL_V2_ACTION_MODE=deterministic|stochastic  # deployed mean vs on-policy gate
#   NAVRL_EVAL_REFLECTION_MODE=original|conjugate # inference-only mirror audit
#   NAVRL_V2_FIXED_TARGET_SPEED=0.9                # fixed-speed causal evaluation
#   NAVRL_SPEED_GOVERNOR=off|fixed|clearance|ttc|riskcap  # inference-only R2 speed-risk screen
#   NAVRL_V2_SHARED_SOURCE_BUNDLE=/abs/path         # reuse one immutable source snapshot across arms
#   NAVRL_V2_EVAL_CONTRACT=corrected_nonoverlap_physical_off
#       footprint_clearance / U[0.3,1.25] / ramp 1 / route off; default densities 70 85 100 115 130 145
set -euo pipefail

CALLER_PWD="${PWD}"
# Resolve the evaluator before changing directory.  When invoked from the repository root,
# BASH_SOURCE[0] is relative to CALLER_PWD; resolving it again after cd would point at a
# non-existent doubly-prefixed path and make a valid evaluation exit before the first cell.
EVALUATOR_SCRIPT="$(readlink -f -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${EVALUATOR_SCRIPT}")"
CKPT_INPUT="${1:?usage: $0 <last_gen_ppo_ep_XXXX.pth> [games_per_cell]}"
GAMES="${2:-2049}"
if (( $# > 2 )); then
    echo "[eval_v2] unexpected arguments: ${*:3}" >&2
    exit 2
fi

cd "${SCRIPT_DIR}"

export PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
if [[ "${PYTHON}" == */* ]]; then
    export PATH="$(dirname "${PYTHON}"):${PATH}"
fi
export PYTHONNOUSERSITE=1
# A held-out sweep is neither the single-environment interactive viewer nor the legacy general
# evaluator. Inherited flags used to make the canonical 128-env bulk evaluation abort only after
# Isaac Gym had initialized, wasting the run and leaving no attestation.
unset NAVRL_GENERAL_EVAL NAVRL_INTERACTIVE NAVRL_GENERAL_RESULTS_JSON
unset NAVRL_BULK_EVAL NAVRL_BULK_EVAL_JSON NAVRL_EVAL_CHECKPOINT NAVRL_EVAL_RUN_NONCE
unset NAVRL_EVAL_ACTION_MODE
unset NAVRL_LEGACY_VISION NAVRL_OOB_PROBE

if [[ "${CKPT_INPUT}" == /* ]]; then
    CKPT="${CKPT_INPUT}"
else
    # A relative argument belongs to the directory from which the user invoked this script, not
    # this script's own directory (we cd above so all local launchers/configs remain resolvable).
    CKPT="${CALLER_PWD}/${CKPT_INPUT}"
fi
if [[ ! -f "${CKPT}" ]]; then
    echo "[eval_v2] checkpoint not found: ${CKPT}" >&2
    exit 2
fi
CKPT="$(readlink -f -- "${CKPT}")"
REQUESTED_DETECTOR_CHECKPOINT="${NAVRL_DETECTOR_CHECKPOINT:-}"
if [[ -n "${REQUESTED_DETECTOR_CHECKPOINT}" ]]; then
    if [[ "${REQUESTED_DETECTOR_CHECKPOINT}" != /* ]]; then
        REQUESTED_DETECTOR_CHECKPOINT="${CALLER_PWD}/${REQUESTED_DETECTOR_CHECKPOINT}"
    fi
    if [[ -f "${REQUESTED_DETECTOR_CHECKPOINT}" ]]; then
        REQUESTED_DETECTOR_CHECKPOINT="$(readlink -f -- "${REQUESTED_DETECTOR_CHECKPOINT}")"
    fi
fi

# The token selector is part of the learned observation semantics. Recover it from the checkpoint
# instead of trusting an inherited shell value; this also makes the registered cluster-vs-TTC A/B
# evaluable without letting the caller accidentally score an arm with the other arm's selector.
CHECKPOINT_META="$(
    "${PYTHON}" - "${CKPT}" <<'PY'
import math
import os
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
state = checkpoint.get("env_state") or {}
selector = str(state.get("cfg_obstacle_selector", "")).strip()
if selector not in {"cluster_sector", "ttc_sector"}:
    raise SystemExit(
        "[eval_v2] checkpoint has no supported v2 obstacle selector: %r" % selector
    )
force = os.environ.get("NAVRL_V2_FORCE", "0") == "1"
allow_detector_threshold = (
    os.environ.get("NAVRL_V2_ALLOW_DETECTOR_THRESHOLD_MISMATCH", "0") == "1"
)
defaults = {
    "cfg_obstacle_ttc_idle_s": 30.0,
    "cfg_obstacle_ttc_min_speed": 0.15,
    "cfg_fov_curriculum_epochs": 3000.0,
    "cfg_detector_min_pixels": 2.0,
    "cfg_detector_threshold": 0.55,
    # Detector geometry was added after the first v2 checkpoints.  Keep the historical
    # configuration as the explicit legacy default, then pin these values below so an
    # inherited shell cannot silently change the sensor contract.
    "cfg_detector_max_range": 20.0,
    "cfg_detect_width": 160.0,
    "cfg_detect_height": 90.0,
    "cfg_detector_checkpoint_sha256": "",
    "cfg_perception_perturb": False,
    "cfg_detection_dropout": 0.3,
    "cfg_detection_latency_s": 0.0,
    "cfg_range_error_m": 0.0,
    "cfg_rgb_noise_std": 0.015,
    "cfg_depth_noise_std": 0.02,
    "cfg_max_tilt_deg": 45.0,
    "cfg_tilt_comp": True,
}
legacy_optional = {
    "cfg_detection_latency_s",
    "cfg_range_error_m",
    # Geometry provenance was introduced after the first v2 checkpoints.  Legacy checkpoints
    # retain the historical 20 m / 160x90 contract; newer checkpoints carry these fields and
    # are pinned to their recorded values below.
    "cfg_detector_max_range",
    "cfg_detect_width",
    "cfg_detect_height",
}
missing = [key for key in defaults if key not in state and key not in legacy_optional]
if missing and not force:
    raise SystemExit(
        "[eval_v2] checkpoint lacks same-shape perception/control provenance: "
        + ", ".join(missing)
    )
if missing:
    print(
        "[eval_v2] WARNING (forced): inferring legacy defaults for " + ", ".join(missing),
        file=sys.stderr,
    )

def value(key):
    return state.get(key, defaults[key])

numeric_keys = {
    "cfg_obstacle_ttc_idle_s",
    "cfg_obstacle_ttc_min_speed",
    "cfg_fov_curriculum_epochs",
    "cfg_detector_min_pixels",
    "cfg_detector_threshold",
    "cfg_detector_max_range",
    "cfg_detect_width",
    "cfg_detect_height",
    "cfg_detection_dropout",
    "cfg_detection_latency_s",
    "cfg_range_error_m",
    "cfg_rgb_noise_std",
    "cfg_depth_noise_std",
    "cfg_max_tilt_deg",
}
for key in numeric_keys:
    number = float(value(key))
    if not math.isfinite(number):
        raise SystemExit(f"[eval_v2] checkpoint has non-finite {key}: {number}")
for key in ("cfg_fov_curriculum_epochs", "cfg_detector_min_pixels"):
    number = float(value(key))
    if not number.is_integer() or number < 0:
        raise SystemExit(f"[eval_v2] checkpoint has invalid integer {key}: {number}")
detector_sha = str(value("cfg_detector_checkpoint_sha256")).strip()
if detector_sha and (
    len(detector_sha) != 64
    or any(char not in "0123456789abcdef" for char in detector_sha.lower())
):
    raise SystemExit("[eval_v2] checkpoint detector SHA-256 is malformed")
robot_name = str(state.get("cfg_robot_name", "navrl_quad")).strip()
if robot_name not in {"navrl_quad", "navrl_ref5in_quad", "navrl_ref5in_v2_quad"}:
    raise SystemExit(f"[eval_v2] unsupported checkpoint robot lineage: {robot_name!r}")
robot_contract_version = int(state.get("cfg_robot_contract_version", 0) or 0)
robot_config_sha = str(state.get("cfg_robot_config_sha256", "")).strip()
robot_asset_sha = str(state.get("cfg_robot_asset_sha256", "")).strip()
robot_asset_file = str(state.get("cfg_robot_asset_file", "")).strip()
if robot_contract_version >= 1:
    for name, digest in (
        ("cfg_robot_config_sha256", robot_config_sha),
        ("cfg_robot_asset_sha256", robot_asset_sha),
    ):
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
            raise SystemExit(f"[eval_v2] checkpoint {name} is missing or malformed")
    if not robot_asset_file:
        raise SystemExit("[eval_v2] checkpoint cfg_robot_asset_file is missing")
elif robot_name != "navrl_quad":
    raise SystemExit(
        "[eval_v2] a non-legacy robot checkpoint must carry robot contract v1 provenance"
    )
print(
    selector,
    robot_name,
    robot_contract_version,
    robot_config_sha or "-",
    robot_asset_file or "quad_navrl_collide.urdf",
    robot_asset_sha or "-",
    str(state.get("cfg_recovery_stage", "")).strip() or "-",
    int(checkpoint.get("epoch", -1)),
    int(state.get("cfg_recovery_source_epoch", -1)),
    int(state.get("cfg_recovery_smoke_required_epochs", -1)),
    int(state.get("n_bars_active", -1)),
    float(value("cfg_obstacle_ttc_idle_s")),
    float(value("cfg_obstacle_ttc_min_speed")),
    int(float(value("cfg_fov_curriculum_epochs"))),
    int(float(value("cfg_detector_min_pixels"))),
    float(value("cfg_detector_threshold")),
    float(value("cfg_detector_max_range")),
    float(value("cfg_detect_width")),
    float(value("cfg_detect_height")),
    float(value("cfg_detection_dropout")),
    float(value("cfg_rgb_noise_std")),
    float(value("cfg_depth_noise_std")),
    float(value("cfg_max_tilt_deg")),
    detector_sha or "-",
    int(bool(value("cfg_perception_perturb"))),
    int(bool(value("cfg_tilt_comp"))),
)
PY
)"
read -r CHECKPOINT_SELECTOR CHECKPOINT_ROBOT ROBOT_CONTRACT_VERSION ROBOT_CONFIG_SHA \
    ROBOT_ASSET_FILE ROBOT_ASSET_SHA RECOVERY_STAGE CHECKPOINT_EPOCH RECOVERY_SOURCE_EPOCH \
    RECOVERY_REQUIRED_EPOCHS CHECKPOINT_BARS TTC_IDLE_S TTC_MIN_SPEED FOV_CURRICULUM_EPOCHS \
    DETECTOR_MIN_PIXELS DETECTOR_THRESHOLD DETECTOR_MAX_RANGE DETECT_WIDTH DETECT_HEIGHT \
    DETECTION_DROPOUT RGB_NOISE_STD DEPTH_NOISE_STD \
    MAX_TILT_DEG DETECTOR_SHA PERCEPTION_PERTURB TILT_COMP <<< "${CHECKPOINT_META}"

# This must be exported before runner.py imports navrl_task_config.py.  Both robot lineages have
# identical policy tensor shapes, so relying on a caller's shell value would silently replay the
# checkpoint with different rigid-body dynamics.
export NAVRL_ROBOT="${CHECKPOINT_ROBOT}"
export ROBOT_CONTRACT_VERSION
export NAVRL_EXPECTED_ROBOT_CONFIG_SHA256="${ROBOT_CONFIG_SHA}"
export NAVRL_EXPECTED_ROBOT_ASSET_SHA256="${ROBOT_ASSET_SHA}"

# Fail before allocating Isaac Gym when the current source cannot reproduce the checkpoint's
# vehicle.  The bulk-result check below repeats this against runtime metadata, but discovering a
# mismatch after thousands of episodes wastes GPU time and can leave tempting invalid CSV files.
if [[ "${ROBOT_CONTRACT_VERSION}" -ge 1 ]]; then
    case "${CHECKPOINT_ROBOT}" in
        navrl_quad)
            ROBOT_CONFIG_PATH=../../config/robot_config/navrl_quad_config.py
            ;;
        navrl_ref5in_quad)
            ROBOT_CONFIG_PATH=../../config/robot_config/navrl_ref5in_quad_config.py
            ;;
        navrl_ref5in_v2_quad)
            ROBOT_CONFIG_PATH=../../config/robot_config/navrl_ref5in_v2_quad_config.py
            ;;
        *)
            echo "[eval_v2] unsupported robot lineage: ${CHECKPOINT_ROBOT}" >&2
            exit 2
            ;;
    esac
    ROBOT_ASSET_PATH=../../../resources/robots/quad/${ROBOT_ASSET_FILE}
    if [[ ! -f "${ROBOT_CONFIG_PATH}" || ! -f "${ROBOT_ASSET_PATH}" ]]; then
        echo "[eval_v2] checkpoint robot source is missing: ${ROBOT_CONFIG_PATH} / ${ROBOT_ASSET_PATH}" >&2
        exit 2
    fi
    CURRENT_ROBOT_CONFIG_SHA="$(sha256sum "${ROBOT_CONFIG_PATH}" | awk '{print $1}')"
    CURRENT_ROBOT_ASSET_SHA="$(sha256sum "${ROBOT_ASSET_PATH}" | awk '{print $1}')"
    if [[ "${CURRENT_ROBOT_CONFIG_SHA}" != "${ROBOT_CONFIG_SHA}" ]]; then
        echo "[eval_v2] robot config source drift: checkpoint=${ROBOT_CONFIG_SHA} runtime=${CURRENT_ROBOT_CONFIG_SHA}" >&2
        exit 2
    fi
    if [[ "${CURRENT_ROBOT_ASSET_SHA}" != "${ROBOT_ASSET_SHA}" ]]; then
        echo "[eval_v2] robot URDF source drift: checkpoint=${ROBOT_ASSET_SHA} runtime=${CURRENT_ROBOT_ASSET_SHA}" >&2
        exit 2
    fi
fi

if [[ ! "${GAMES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[eval_v2] games_per_cell must be a positive integer, got: ${GAMES}" >&2
    exit 2
fi

# Pin the runtime physics profile as part of the evaluation contract.  Merely recording the
# caller's inherited AERIAL_GYM_SIM_NAME is insufficient: base_sim_4gb has a different timestep,
# so the same policy can receive a materially different score while the JSON otherwise looks
# canonical.  GPU4GB is the only supported profile selector; NUM_ENVS and the sim name themselves
# are outputs of that selector, not caller-overridable inputs.
case "${GPU4GB:-0}" in
    0|"")
        export GPU4GB=0
        export AERIAL_GYM_SIM_NAME=base_sim
        export NUM_ENVS=128
        export NAVRL_EVAL_PROFILE=main
        export NAVRL_SIM_PHYSICS_CONTRACT=base_sim_dt0.01
        ;;
    1)
        export GPU4GB=1
        export AERIAL_GYM_SIM_NAME=base_sim_4gb
        export NUM_ENVS=64
        export NAVRL_EVAL_PROFILE=4gb
        export NAVRL_SIM_PHYSICS_CONTRACT=base_sim_4gb_dt0.01_buffers
        ;;
    *)
        echo "[eval_v2] GPU4GB must be 0 or 1; got: ${GPU4GB}" >&2
        exit 2
        ;;
esac

NAVRL_V2_ACTION_MODE="${NAVRL_V2_ACTION_MODE:-deterministic}"
case "${NAVRL_V2_ACTION_MODE}" in
    deterministic|stochastic) ;;
    *)
        echo "[eval_v2] NAVRL_V2_ACTION_MODE must be deterministic or stochastic; got: ${NAVRL_V2_ACTION_MODE}" >&2
        exit 2
        ;;
esac
export NAVRL_V2_ACTION_MODE
export NAVRL_EVAL_ACTION_MODE="${NAVRL_V2_ACTION_MODE}"
NAVRL_EVAL_REFLECTION_MODE="${NAVRL_EVAL_REFLECTION_MODE:-original}"
case "${NAVRL_EVAL_REFLECTION_MODE}" in
    original|conjugate) ;;
    *)
        echo "[eval_v2] NAVRL_EVAL_REFLECTION_MODE must be original or conjugate; got: ${NAVRL_EVAL_REFLECTION_MODE}" >&2
        exit 2
        ;;
esac
if [[ "${NAVRL_EVAL_REFLECTION_MODE}" != "original" && "${NAVRL_V2_ACTION_MODE}" != "deterministic" ]]; then
    echo "[eval_v2] conjugate reflection evaluation requires deterministic actions." >&2
    exit 2
fi
export NAVRL_EVAL_REFLECTION_MODE

# Inference-only speed-risk intervention. Validate and normalize every numeric field before the
# simulator starts; the task and result validator both record the normalized contract.
NAVRL_SPEED_GOVERNOR="${NAVRL_SPEED_GOVERNOR:-off}"
case "${NAVRL_SPEED_GOVERNOR}" in
    off|fixed|clearance|ttc|riskcap) ;;
    *)
        echo "[eval_v2] NAVRL_SPEED_GOVERNOR must be off, fixed, clearance, ttc or riskcap; got: ${NAVRL_SPEED_GOVERNOR}" >&2
        exit 2
        ;;
esac
GOVERNOR_VALUES="$(${PYTHON} - <<'PY'
import math
import os

spec = (
    ("NAVRL_SPEED_GOVERNOR_FIXED_MPS", 2.0, True),
    ("NAVRL_SPEED_GOVERNOR_FREE_MPS", math.sqrt(2.0) * 2.5, True),
    ("NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M", 0.45, True),
    ("NAVRL_SPEED_GOVERNOR_MARGIN_M", 0.45, False),
    ("NAVRL_SPEED_GOVERNOR_SLOW_M", 3.0, True),
    ("NAVRL_SPEED_GOVERNOR_RELEASE_M", 5.0, True),
    ("NAVRL_SPEED_GOVERNOR_TTC_S", 1.0, True),
    ("NAVRL_SPEED_GOVERNOR_BRAKE_MPS2", 2.0, True),
    ("NAVRL_SPEED_GOVERNOR_REACTION_S", 0.1, False),
)
values = []
for name, default, positive in spec:
    raw = os.environ.get(name, "").strip()
    try:
        value = float(raw) if raw else float(default)
    except ValueError as exc:
        raise SystemExit(f"[eval_v2] {name} must be numeric; got {raw!r}") from exc
    if not math.isfinite(value) or value < 0.0 or (positive and value <= 0.0):
        relation = "positive" if positive else "non-negative"
        raise SystemExit(f"[eval_v2] {name} must be finite and {relation}; got {value!r}")
    values.append(value)
if values[4] <= values[3]:
    raise SystemExit("[eval_v2] governor slow distance must exceed hard margin")
if os.environ.get("NAVRL_SPEED_GOVERNOR", "off").strip().lower() == "riskcap":
    if values[5] <= values[4]:
        raise SystemExit("[eval_v2] riskcap release distance must exceed slow distance")
    if values[1] < values[0]:
        raise SystemExit("[eval_v2] riskcap free speed must be >= fixed cap")
print(" ".join(f"{value:.12g}" for value in values))
PY
)"
read -r NAVRL_SPEED_GOVERNOR_FIXED_MPS NAVRL_SPEED_GOVERNOR_FREE_MPS \
    NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M NAVRL_SPEED_GOVERNOR_MARGIN_M \
    NAVRL_SPEED_GOVERNOR_SLOW_M NAVRL_SPEED_GOVERNOR_RELEASE_M \
    NAVRL_SPEED_GOVERNOR_TTC_S \
    NAVRL_SPEED_GOVERNOR_BRAKE_MPS2 NAVRL_SPEED_GOVERNOR_REACTION_S <<< "${GOVERNOR_VALUES}"
export NAVRL_SPEED_GOVERNOR NAVRL_SPEED_GOVERNOR_FIXED_MPS NAVRL_SPEED_GOVERNOR_FREE_MPS
export NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M NAVRL_SPEED_GOVERNOR_MARGIN_M
export NAVRL_SPEED_GOVERNOR_SLOW_M NAVRL_SPEED_GOVERNOR_RELEASE_M NAVRL_SPEED_GOVERNOR_TTC_S
export NAVRL_SPEED_GOVERNOR_BRAKE_MPS2 NAVRL_SPEED_GOVERNOR_REACTION_S
export NAVRL_SPEED_GOVERNOR_DIAG=1

# ---- v2 ARENA / TASK contract (fixed evaluation condition) ----
export NAVRL_ARENA_XY=40
export NAVRL_ARENA_Z=3
export NAVRL_BAR_POOL=bars_h3
export NAVRL_PLACEMENT_MODE=navrl_band
export NAVRL_PLACEMENT_TOUCH_M=0.4
export NAVRL_PLACEMENT_GAP_M=1.6
export NAVRL_V2_EVAL_CONTRACT="${NAVRL_V2_EVAL_CONTRACT:-}"
EVAL_CONTRACT="${NAVRL_V2_EVAL_CONTRACT}"
if [[ "${EVAL_CONTRACT}" == "corrected_nonoverlap_physical_off" ]]; then
    export NAVRL_PLACEMENT_MODE=footprint_clearance
    export NAVRL_PLACEMENT_SURFACE_CLEARANCE_M=0.45
    export NAVRL_TARGET_ROUTE_MODE=off
    export NAVRL_TARGET_DYNAMICS=physical
    export NAVRL_PHYSICAL_GEOMETRY_VERSION=v2
    export NAVRL_TARGET_BOX_XY_M=0.283
fi
export NAVRL_EPISODE_LEN_STEPS=600
export NAVRL_MAX_BARS=300
export NAVRL_BAR_X_MIN=0.0
export NAVRL_BAR_X_MAX=1.0
read -r EVAL_GOAL_DIST_MIN EVAL_GOAL_DIST_MAX EVAL_TARGET_PATTERN <<< "$(
    "${PYTHON}" - "${NAVRL_V2_GOAL_DIST_MIN:-6}" "${NAVRL_V2_GOAL_DIST_MAX:-28}" \
        "${NAVRL_V2_TARGET_PATTERN:-mixed}" <<'PY'
import math
import sys

try:
    minimum, maximum = map(float, sys.argv[1:3])
except ValueError as exc:
    raise SystemExit("[eval_v2] goal-distance overrides must be numeric") from exc
pattern = sys.argv[3].strip().lower()
if not all(math.isfinite(value) for value in (minimum, maximum)) or minimum < 1.0:
    raise SystemExit("[eval_v2] goal-distance overrides must be finite with min >= 1 m")
if maximum < minimum + 1.0 or maximum > 28.0:
    raise SystemExit("[eval_v2] goal-distance overrides require min+1 <= max <= 28 m")
if pattern not in {"mixed", "cv", "waypoint", "circle"}:
    raise SystemExit("[eval_v2] NAVRL_V2_TARGET_PATTERN must be mixed|cv|waypoint|circle")
print(f"{minimum:.12g} {maximum:.12g} {pattern}")
PY
)"
export NAVRL_GENERAL_GOAL_DIST_MIN="${EVAL_GOAL_DIST_MIN}"
export NAVRL_GENERAL_GOAL_DIST_MAX="${EVAL_GOAL_DIST_MAX}"
export NAVRL_K_COMPETENCE=1
export NAVRL_K_FINAL=28
export NAVRL_K_MIN_FINAL=20

# A sweep cell must stay at the requested density. An inherited curriculum flag could otherwise
# promote density during a long evaluation even though NAVRL_NUM_BARS selected the initial value.
export NAVRL_DENSITY_CURRICULUM=0
export NAVRL_DENSITY_START=70
export NAVRL_DENSITY_FINAL=300
export NAVRL_DENSITY_STEP=15
export NAVRL_DENSITY_THRESHOLD_START=0.80
export NAVRL_DENSITY_THRESHOLD_END=0.70
export NAVRL_DENSITY_THRESHOLD_SCHEDULE="70:0.82,85:0.77,100:0.72,115:0.70"
export NAVRL_DENSITY_CHECK_EPS=16384
export NAVRL_DENSITY_MIN_EPOCHS=1000
unset NAVRL_FIXED_BARS NAVRL_CONTROLLED_ABLATION

# v2 trains on moving targets sampled from U[0.3, 1.5] m/s after the short 300-epoch ramp.  The
# canonical sweep uses that final distribution.  A causal speed evaluation may explicitly set
# NAVRL_V2_FIXED_TARGET_SPEED within the training support; the result validator below then requires
# the task's measured condition to be fixed at exactly that value.
export NAVRL_TARGET_SPEED_MIN=0.3
export NAVRL_TARGET_SPEED_FINAL=1.5
export NAVRL_TARGET_SPEED_RAMP_EPOCHS=300
if [[ "${EVAL_CONTRACT}" == "corrected_nonoverlap_physical_off" ]]; then
    export NAVRL_TARGET_SPEED_FINAL=1.25
    export NAVRL_TARGET_SPEED_RAMP_EPOCHS=1
    export NAVRL_DENSITY_FINAL=205
fi
export NAVRL_TARGET_PATTERN="${EVAL_TARGET_PATTERN}"
export NAVRL_EVAL_CV_INITIAL_HEADING="${NAVRL_EVAL_CV_INITIAL_HEADING:-random}"
FIXED_TARGET_SPEED="${NAVRL_V2_FIXED_TARGET_SPEED:-}"
if [[ -n "${FIXED_TARGET_SPEED}" ]]; then
    FIXED_TARGET_SPEED="$(${PYTHON} - "${FIXED_TARGET_SPEED}" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError as exc:
    raise SystemExit("[eval_v2] NAVRL_V2_FIXED_TARGET_SPEED must be numeric") from exc
if not math.isfinite(value) or not 0.3 <= value <= 1.5:
    raise SystemExit(
        "[eval_v2] NAVRL_V2_FIXED_TARGET_SPEED must be within the trained [0.3, 1.5] m/s support"
    )
print(f"{value:.12g}")
PY
    )"
    export NAVRL_V2_FIXED_TARGET_SPEED="${FIXED_TARGET_SPEED}"
    export NAVRL_TARGET_SPEED="${FIXED_TARGET_SPEED}"
    TARGET_SPEED_DESCRIPTION="fixed ${FIXED_TARGET_SPEED}m/s"
else
    unset NAVRL_V2_FIXED_TARGET_SPEED NAVRL_TARGET_SPEED
    TARGET_SPEED_DESCRIPTION="U[${NAVRL_TARGET_SPEED_MIN},${NAVRL_TARGET_SPEED_FINAL}]m/s"
fi
# Evaluation is defined on the final training distribution, independent of the checkpoint's saved
# curriculum clock. Without this explicit override, an early checkpoint silently evaluates at the
# ramp's then-current upper speed while the result used to claim U[0.3,1.5].
export NAVRL_EVAL_TARGET_SPEED_FINAL=1
# Goal-distance and visibility difficulty are held-out conditions, not checkpoint state.  This
# makes an early or malformed curriculum clock incapable of receiving an easier evaluation.
export NAVRL_EVAL_FULL_DISTRIBUTION=1

# ---- representation contract (fixed; observation width alone cannot detect semantic drift) ----
export NAVRL_VISION=1
export NAVRL_PERCEPTION=1
export NAVRL_GENERAL_TRAIN=1
export NAVRL_PERCEPTION_PERTURB="${NAVRL_PERCEPTION_PERTURB:-${PERCEPTION_PERTURB}}"
export NAVRL_TILT_COMP="${NAVRL_TILT_COMP:-${TILT_COMP}}"
export NAVRL_LIDAR_RANGE=12
export NAVRL_LIDAR_HBEAMS=72
export NAVRL_LIDAR_VBEAMS=4
export NAVRL_MAX_OBSTACLES=8
export NAVRL_OBSTACLE_SELECTOR="${CHECKPOINT_SELECTOR}"
export NAVRL_OBSTACLE_CLUSTER_GAP_M=0.45
export NAVRL_OBSTACLE_SECTORS=8
export NAVRL_OBSTACLE_SUPPRESS_DEG=10
export NAVRL_OBSTACLE_FOV_DEG=240
export NAVRL_OBSTACLE_TTC_IDLE_S="${TTC_IDLE_S}"
export NAVRL_OBSTACLE_TTC_MIN_SPEED="${TTC_MIN_SPEED}"
export NAVRL_CORRIDOR_TOKENS=0
export NAVRL_GEOFENCE_ACTOR="${NAVRL_GEOFENCE_ACTOR:-0}"
export NAVRL_GEOFENCE_NOISE_STD_M="${NAVRL_GEOFENCE_NOISE_STD_M:-0}"
export NAVRL_GEOFENCE_DROPOUT="${NAVRL_GEOFENCE_DROPOUT:-0}"
export NAVRL_CORRIDOR_HORIZON_M=6.0
export NAVRL_CORRIDOR_MIN_WIDTH_M=0.55
export NAVRL_MAX_VELOCITY=2.5
export NAVRL_ALT_HOLD_VMAX=2.5
export NAVRL_YAW_RATE_MAX=3.0
export NAVRL_MAX_TILT_DEG="${MAX_TILT_DEG}"
export NAVRL_FOV_CURRICULUM_EPOCHS="${FOV_CURRICULUM_EPOCHS}"
export NAVRL_DETECTOR_MIN_PIXELS="${NAVRL_DETECTOR_MIN_PIXELS:-${DETECTOR_MIN_PIXELS}}"
export NAVRL_DETECTOR_THRESHOLD="${NAVRL_DETECTOR_THRESHOLD:-${DETECTOR_THRESHOLD}}"
export NAVRL_DETECTOR_MAX_RANGE="${DETECTOR_MAX_RANGE}"
export NAVRL_DETECT_WIDTH="${DETECT_WIDTH}"
export NAVRL_DETECT_HEIGHT="${DETECT_HEIGHT}"
export NAVRL_DETECTION_DROPOUT="${NAVRL_DETECTION_DROPOUT:-${DETECTION_DROPOUT}}"
export NAVRL_DETECTION_LATENCY_S="${NAVRL_DETECTION_LATENCY_S:-0}"
export NAVRL_RANGE_ERROR_M="${NAVRL_RANGE_ERROR_M:-0}"
# Compensation knobs (fixes, not perturbations), pinned so a cell cannot inherit one from the
# caller's shell. P0/P1/P2 default off -- they were measured and not adopted. P3 defaults ON
# because it is the correct measurement model rather than a compensation (WORKLOG 2026-08-06);
# pass NAVRL_LATENCY_EGO_MOTION_FIX=0 to reproduce the superseded R3 latency arms.
export NAVRL_LATENCY_COMPENSATE="${NAVRL_LATENCY_COMPENSATE:-0}"
export NAVRL_LATENCY_LIDAR_BACKUP="${NAVRL_LATENCY_LIDAR_BACKUP:-0}"
export NAVRL_LATENCY_OBSTACLE_FIX="${NAVRL_LATENCY_OBSTACLE_FIX:-off}"
export NAVRL_LATENCY_EGO_MOTION_FIX="${NAVRL_LATENCY_EGO_MOTION_FIX:-1}"
export NAVRL_POSE_CLOCK_OFFSET_S="${NAVRL_POSE_CLOCK_OFFSET_S:-0}"
export NAVRL_POSE_NOISE_POS_M="${NAVRL_POSE_NOISE_POS_M:-0}"
export NAVRL_POSE_NOISE_YAW_DEG="${NAVRL_POSE_NOISE_YAW_DEG:-0}"
export NAVRL_TARGET_MASK_BACKFILL="${NAVRL_TARGET_MASK_BACKFILL:-0}"
export NAVRL_LIDAR_TARGET_ASSOC="${NAVRL_LIDAR_TARGET_ASSOC:-1}"
export NAVRL_LIDAR_RANGE_ONLY_UPDATE="${NAVRL_LIDAR_RANGE_ONLY_UPDATE:-0}"
export NAVRL_LIDAR_ASSOC_GATE_M="${NAVRL_LIDAR_ASSOC_GATE_M:-0}"
export NAVRL_LIDAR_SILENT_CORRECT="${NAVRL_LIDAR_SILENT_CORRECT:-0}"
# Appearance domain-shift knobs (검증 2): pinned to nominal unless the arm sets them.
export NAVRL_APP_HUE_DEG="${NAVRL_APP_HUE_DEG:-0}"
export NAVRL_APP_LIGHT_GAIN="${NAVRL_APP_LIGHT_GAIN:-0}"
export NAVRL_APP_ALBEDO_JITTER="${NAVRL_APP_ALBEDO_JITTER:-0}"
export NAVRL_APP_TEXTURE_STD="${NAVRL_APP_TEXTURE_STD:-0}"
export NAVRL_APP_MOTION_BLUR="${NAVRL_APP_MOTION_BLUR:-0}"
export NAVRL_CAM_MOUNT_ROT_DEG="${NAVRL_CAM_MOUNT_ROT_DEG:-0}"
export NAVRL_CAM_MOUNT_TRANS_M="${NAVRL_CAM_MOUNT_TRANS_M:-0}"
export NAVRL_CAM_FOV_SCALE_ERR="${NAVRL_CAM_FOV_SCALE_ERR:-0}"
export NAVRL_RGB_NOISE_STD="${NAVRL_RGB_NOISE_STD:-${RGB_NOISE_STD}}"
export NAVRL_DEPTH_NOISE_STD="${NAVRL_DEPTH_NOISE_STD:-${DEPTH_NOISE_STD}}"
export NAVRL_OOB_MARGIN=1.0
export FILE=ppo_navrl_perception_transformer.yaml
export TASK=navrl_task

if [[ "${DETECTOR_SHA}" == "-" ]]; then
    if [[ -n "${REQUESTED_DETECTOR_CHECKPOINT}" && -f "${REQUESTED_DETECTOR_CHECKPOINT}" ]]; then
        ACTUAL_DETECTOR_SHA="$(
            "${PYTHON}" - "${REQUESTED_DETECTOR_CHECKPOINT}" <<'PY'
import hashlib
from pathlib import Path
import sys

digest = hashlib.sha256()
with Path(sys.argv[1]).open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
        )"
        export NAVRL_DETECTOR_CHECKPOINT="${REQUESTED_DETECTOR_CHECKPOINT}"
        export NAVRL_EXPECTED_DETECTOR_SHA256="${ACTUAL_DETECTOR_SHA}"
    else
        unset NAVRL_DETECTOR_CHECKPOINT
        export NAVRL_EXPECTED_DETECTOR_SHA256=""
    fi
else
    if [[ -z "${REQUESTED_DETECTOR_CHECKPOINT}" || ! -f "${REQUESTED_DETECTOR_CHECKPOINT}" ]]; then
        echo "[eval_v2] checkpoint requires its learned detector; set NAVRL_DETECTOR_CHECKPOINT to the matching file." >&2
        exit 2
    fi
    ACTUAL_DETECTOR_SHA="$(
        "${PYTHON}" - "${REQUESTED_DETECTOR_CHECKPOINT}" <<'PY'
import hashlib
from pathlib import Path
import sys

digest = hashlib.sha256()
with Path(sys.argv[1]).open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
    )"
    if [[ "${ACTUAL_DETECTOR_SHA}" != "${DETECTOR_SHA}" ]]; then
        echo "[eval_v2] detector checkpoint SHA-256 mismatch." >&2
        exit 2
    fi
    export NAVRL_DETECTOR_CHECKPOINT="${REQUESTED_DETECTOR_CHECKPOINT}"
    export NAVRL_EXPECTED_DETECTOR_SHA256="${DETECTOR_SHA}"
fi

# Evaluation must use the distribution encoded in the checkpoint. Inherited experiment variables
# would otherwise select a different action model or network while appearing to be the same run.
unset NAVRL_ACTION_POLICY NAVRL_ACTION_STD NAVRL_ACTION_MU_SCALE NAVRL_TRUNCATED_DMIN
unset NAVRL_ENTROPY_COEF NAVRL_LEARNING_RATE NAVRL_NETWORK_OVERRIDE

if [[ "${RECOVERY_STAGE}" == "smoke" ]]; then
    if [[ -n "${FIXED_TARGET_SPEED}" ]]; then
        echo "[eval_v2] fixed target speed is forbidden for a recovery-smoke attestation." >&2
        exit 2
    fi
    if [[ "${NAVRL_V2_ACTION_MODE}" != "deterministic" ]]; then
        echo "[eval_v2] recovery smoke attestation requires deterministic action selection." >&2
        exit 2
    fi
    if [[ "${NAVRL_V2_FORCE:-0}" == "1" ]]; then
        echo "[eval_v2] NAVRL_V2_FORCE is forbidden for a curriculum-unlocking recovery evaluation." >&2
        exit 2
    fi
    if [[ "${NAVRL_EVAL_PROFILE}" != "main" ]]; then
        echo "[eval_v2] recovery smoke attestation requires the canonical main/base_sim/128-env runtime." >&2
        exit 2
    fi
    export NAVRL_SEED=42
else
    export NAVRL_SEED="${NAVRL_SEED:-42}"
fi
export NAVRL_BULK_EVAL=1
export NAVRL_EVAL_CHECKPOINT="${CKPT}"

# Same per-area schedule as the v1 sweep, over the full 1600 m^2 v2 arena:
# 70/150/210/280 bars = 4.4 / 9.4 / 13.1 / 17.5 per 100 m^2.
# The corrected-nonoverlap route-off held-out contract uses the trained knots instead.
if [[ "${EVAL_CONTRACT}" == "corrected_nonoverlap_physical_off" ]]; then
    DENSITIES_TEXT="${NAVRL_V2_DENSITIES:-70 85 100 115 130 145}"
else
    DENSITIES_TEXT="${NAVRL_V2_DENSITIES:-70 150 210 280}"
fi
read -r -a DENSITIES <<< "${DENSITIES_TEXT}"
if [[ "${#DENSITIES[@]}" -eq 0 ]]; then
    echo "[eval_v2] NAVRL_V2_DENSITIES must contain at least one density" >&2
    exit 2
fi
for N in "${DENSITIES[@]}"; do
    if [[ ! "${N}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[eval_v2] density must be an integer in [1, ${NAVRL_MAX_BARS}], got: ${N}" >&2
        exit 2
    fi
    N_VALUE=$((10#${N}))
    if (( N_VALUE > NAVRL_MAX_BARS )); then
        echo "[eval_v2] density must be an integer in [1, ${NAVRL_MAX_BARS}], got: ${N}" >&2
        exit 2
    fi
    if [[ "${EVAL_CONTRACT}" == "corrected_nonoverlap_physical_off" ]] && (( N_VALUE == 205 )); then
        echo "[eval_v2] 205 bars is untrained OOD for this checkpoint and is not in the held-out prereg." >&2
        exit 2
    fi
done

# A recovery smoke evaluation is a machine-enforced promotion gate, not a casual sweep. Refuse an
# incomplete/intermediate checkpoint or a cheaper/different condition before spending GPU time.
if [[ "${RECOVERY_STAGE}" == "smoke" ]]; then
    if (( ${#DENSITIES[@]} != 1 || 10#${DENSITIES[0]} != 130 )); then
        echo "[eval_v2] recovery smoke requires exactly NAVRL_V2_DENSITIES=130." >&2
        exit 2
    fi
    if (( GAMES < 2049 )); then
        echo "[eval_v2] recovery smoke requires at least 2049 held-out episodes; got ${GAMES}." >&2
        exit 2
    fi
    if (( RECOVERY_SOURCE_EPOCH != 9500 || RECOVERY_REQUIRED_EPOCHS != 100 \
          || CHECKPOINT_EPOCH != 9600 || CHECKPOINT_BARS != 130 )); then
        echo "[eval_v2] recovery smoke checkpoint is incomplete or malformed: epoch=${CHECKPOINT_EPOCH} bars=${CHECKPOINT_BARS}." >&2
        exit 2
    fi
    RECOVERY_RUN_ROOT="$(dirname "$(dirname "${CKPT}")")"
    if [[ ! -f "${RECOVERY_RUN_ROOT}/.aerial_training_finished" ]]; then
        echo "[eval_v2] recovery smoke lacks its normal-completion marker." >&2
        exit 2
    fi
fi

# ---- provenance gate: refuse a checkpoint that was not trained under the v2 contract ----
# NAVRL_V2_FORCE is honored inside the gate so an explicit override works under `set -e`.
NAVRL_V2_FORCE="${NAVRL_V2_FORCE:-0}" "${PYTHON}" - "${CKPT}" <<'PY'
import os
import sys
import torch

force = os.environ.get("NAVRL_V2_FORCE", "0") == "1"
allow_detector_threshold = (
    os.environ.get("NAVRL_V2_ALLOW_DETECTOR_THRESHOLD_MISMATCH", "0") == "1"
)
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
state = ckpt.get("env_state") or {}
# Every fixed task/representation field whose mismatch could still load at the same observation
# width. Action-policy provenance is deliberately restored by runner.py rather than hard-coded.
want = {
    "cfg_arena_xy": 40.0,
    "cfg_arena_z": 3.0,
    "cfg_bar_pool": "bars_h3",
    "cfg_placement_mode": os.environ["NAVRL_PLACEMENT_MODE"],
    "cfg_placement_gap_m": 1.6,
    "cfg_placement_touch_m": 0.4,
    "cfg_episode_len_steps": 600.0,
    "cfg_bar_x_min": 0.0,
    "cfg_bar_x_max": 1.0,
    "cfg_general_goal_dist_min": float(os.environ["NAVRL_GENERAL_GOAL_DIST_MIN"]),
    "cfg_general_goal_dist_max": float(os.environ["NAVRL_GENERAL_GOAL_DIST_MAX"]),
    "cfg_lidar_max_range": 12.0,
    "cfg_lidar_hbeams": 72.0,
    "cfg_lidar_vbeams": 4.0,
    "cfg_max_obstacles": 8.0,
    "cfg_token_fov_deg": 240.0,
    "cfg_obstacle_suppress_deg": 10.0,
    "cfg_obstacle_selector": os.environ["NAVRL_OBSTACLE_SELECTOR"],
    "cfg_obstacle_cluster_gap_m": 0.45,
    "cfg_obstacle_sectors": 8.0,
    "cfg_obstacle_ttc_idle_s": float(os.environ["NAVRL_OBSTACLE_TTC_IDLE_S"]),
    "cfg_obstacle_ttc_min_speed": float(os.environ["NAVRL_OBSTACLE_TTC_MIN_SPEED"]),
    "cfg_corridor_tokens": 0.0,
    "cfg_corridor_horizon_m": 6.0,
    "cfg_corridor_min_width_m": 0.55,
    "cfg_fov_curriculum_epochs": float(os.environ["NAVRL_FOV_CURRICULUM_EPOCHS"]),
    "cfg_detector_min_pixels": float(os.environ["NAVRL_DETECTOR_MIN_PIXELS"]),
    "cfg_detector_threshold": float(os.environ["NAVRL_DETECTOR_THRESHOLD"]),
    "cfg_max_velocity": 2.5,
    "cfg_yaw_rate_max": 3.0,
    "cfg_max_tilt_deg": float(os.environ["NAVRL_MAX_TILT_DEG"]),
    "cfg_tilt_comp": float(os.environ["NAVRL_TILT_COMP"]),
    "cfg_target_motion_model": "symmetric_local_steer_v2_heading_continuity90",
    "cfg_target_pattern": os.environ["NAVRL_TARGET_PATTERN"],
    "cfg_target_speed_min": 0.3,
    "cfg_target_speed_final": float(os.environ["NAVRL_TARGET_SPEED_FINAL"]),
    "cfg_target_speed_fixed": -1.0,
    "cfg_target_speed_ramp_epochs": float(os.environ["NAVRL_TARGET_SPEED_RAMP_EPOCHS"]),
    "cfg_target_speed_ramp_start_epochs": 0.0,
    "cfg_general_train": 1.0,
    "cfg_oob_margin": 1.0,
    "cfg_alt_hold_vmax": 2.5,
    "cfg_action_policy": "squashed_gaussian",
    "cfg_action_std": "0.35,0.35,0.05,0.08",
    "cfg_action_mu_scale": "1.0,0.4,1.0,1.0",
}
if os.environ.get("NAVRL_V2_EVAL_CONTRACT") == "corrected_nonoverlap_physical_off":
    want.update(
        {
            "cfg_placement_surface_clearance_m": 0.45,
            "cfg_target_route_mode": "off",
            "cfg_target_dynamics": "physical",
            "cfg_target_motion_model": "physx_ref5in_6dof_motor_wrench_v2_same_substep",
        }
    )
# Detector geometry provenance was added after the frozen ref5in checkpoint lineage.  Preserve
# legacy loadability, but once a checkpoint carries any of the new fields, require the complete
# triplet and compare it to the requested evaluation sensor.  This makes new experiments fail
# closed without retroactively making every older checkpoint unevaluable.
bad = []
detector_geometry_keys = {
    "cfg_detector_max_range": float(os.environ["NAVRL_DETECTOR_MAX_RANGE"]),
    "cfg_detect_width": float(os.environ["NAVRL_DETECT_WIDTH"]),
    "cfg_detect_height": float(os.environ["NAVRL_DETECT_HEIGHT"]),
}
present_detector_geometry = [key for key in detector_geometry_keys if key in state]
if present_detector_geometry:
    missing_detector_geometry = [
        key for key in detector_geometry_keys if key not in state
    ]
    if missing_detector_geometry:
        bad.append(
            "incomplete detector geometry provenance: missing "
            + ", ".join(sorted(missing_detector_geometry))
        )
    want.update(detector_geometry_keys)
checkpoint_robot = str(state.get("cfg_robot_name", "navrl_quad")).strip()
if checkpoint_robot != os.environ["NAVRL_ROBOT"]:
    bad.append(
        f"cfg_robot_name: checkpoint={checkpoint_robot!r} "
        f"runtime={os.environ['NAVRL_ROBOT']!r}"
    )
if "cfg_robot_contract_version" in state:
    want.update(
        {
            "cfg_robot_contract_version": 1.0,
            "cfg_robot_name": os.environ["NAVRL_ROBOT"],
            "cfg_robot_config_sha256": os.environ[
                "NAVRL_EXPECTED_ROBOT_CONFIG_SHA256"
            ],
            "cfg_robot_asset_sha256": os.environ[
                "NAVRL_EXPECTED_ROBOT_ASSET_SHA256"
            ],
        }
    )
if state.get("cfg_recovery_stage") == "smoke":
    # The held-out recovery cell unlocks further training, so it must also prove that the smoke
    # was produced by the exact executable/safety contract rather than merely a shape-compatible
    # policy. Check this before spending 2,049 GPU episodes.
    want.update(
        {
            "cfg_training_seed": 1.0,
            "cfg_training_num_envs": 128.0,
            "cfg_training_file": "ppo_navrl_perception_transformer.yaml",
            "cfg_training_task": "navrl_task",
            "cfg_training_sim": "base_sim",
            "cfg_training_profile": "main",
            "cfg_ppo_horizon": 32.0,
            "cfg_action_learning_rate": 5e-6,
            "current_action_learning_rate": 5e-6,
            "cfg_ppo_log_ratio_clamp": 10.0,
            "cfg_ppo_kl_stop": 0.04,
            "cfg_ppo_epoch_rollback": 1.0,
            "cfg_ppo_rollback_lr_factor": 0.5,
            "cfg_ppo_rollback_min_lr": 1e-6,
            "cfg_ppo_rollback_patience": 5.0,
            "cfg_density_guard_window_epochs": 50.0,
            "cfg_density_guard_min_epochs": 100.0,
            "cfg_density_guard_min_peak": 0.5,
            "cfg_density_guard_drop": 0.25,
            "cfg_density_guard_patience": 25.0,
            "cfg_latent_margin": "2.0,1.25,2.0,2.0",
            "cfg_lateral_latent_margin_coef": 0.01,
            "num_task_steps": 307200.0,
            "k_max_cur": 28.0,
            "k_min_cur": 20.0,
        }
    )
    if int(ckpt.get("frame", -1)) != 39321600:
        bad.append(
            f"checkpoint.frame: checkpoint={ckpt.get('frame')!r} expected=39321600"
        )
verb = "WARNING (forced)" if force else "REFUSING"
missing = [key for key in want if state.get(key) is None]
if missing:
    print(
        "[eval_v2] %s: checkpoint lacks required v2 provenance (%s).\n"
        "          Re-check the run, or set NAVRL_V2_FORCE=1 to override deliberately."
        % (verb, ", ".join(sorted(missing))),
        file=sys.stderr,
    )
    if not force:
        sys.exit(2)
for key, expected in want.items():
    got = state.get(key)
    if got is None:
        continue
    ok = (
        str(got).strip() == expected
        if isinstance(expected, str)
        else abs(float(got) - expected) <= 1e-6
    )
    if not ok:
        if key == "cfg_detector_threshold" and allow_detector_threshold:
            print(
                "[eval_v2] ALLOWED mismatch: cfg_detector_threshold: "
                f"checkpoint={got} expected={expected}",
                file=sys.stderr,
            )
            continue
        bad.append(f"{key}: checkpoint={got} expected={expected}")
if bad:
    print("[eval_v2] %s: v2 contract mismatch:\n  " % verb + "\n  ".join(bad), file=sys.stderr)
    if not force:
        sys.exit(2)
if missing or bad:
    sys.exit(0)
print(
    "[eval_v2] provenance OK | %.0fx%.0f m, pool=%s, placement=%s, %.0f steps"
    % (
        state["cfg_arena_xy"],
        state["cfg_arena_xy"],
        state["cfg_bar_pool"],
        state["cfg_placement_mode"],
        state["cfg_episode_len_steps"],
    )
)
PY

echo "[eval_v2] robot=${CHECKPOINT_ROBOT} contract=v${ROBOT_CONTRACT_VERSION} asset=${ROBOT_ASSET_FILE} urdf_sha=${ROBOT_ASSET_SHA:0:12} config_sha=${ROBOT_CONFIG_SHA:0:12}"
echo "[eval_v2] same-shape | fov_curriculum=${FOV_CURRICULUM_EPOCHS} detector_pixels=${DETECTOR_MIN_PIXELS} detector_threshold=${DETECTOR_THRESHOLD} ttc=${TTC_IDLE_S}/${TTC_MIN_SPEED}"
echo "[eval_v2] runtime=${NAVRL_EVAL_PROFILE}/${AERIAL_GYM_SIM_NAME} envs=${NUM_ENVS} physics=${NAVRL_SIM_PHYSICS_CONTRACT}"
echo "[eval_v2] action_selection=${NAVRL_EVAL_ACTION_MODE} reflection=${NAVRL_EVAL_REFLECTION_MODE}"
echo "[eval_v2] speed_governor=${NAVRL_SPEED_GOVERNOR} fixed=${NAVRL_SPEED_GOVERNOR_FIXED_MPS} free=${NAVRL_SPEED_GOVERNOR_FREE_MPS} margin=${NAVRL_SPEED_GOVERNOR_MARGIN_M} slow=${NAVRL_SPEED_GOVERNOR_SLOW_M} release=${NAVRL_SPEED_GOVERNOR_RELEASE_M} ttc=${NAVRL_SPEED_GOVERNOR_TTC_S}"
if [[ "${NAVRL_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "[eval_v2] PREFLIGHT PASS (evaluation not started)"
    exit 0
fi

RUN_NAME="$(basename "$(dirname "$(dirname "${CKPT}")")")"
STAMP="$(date +%y%m%d_%H%M%S)"
RESULT_DIR="${NAVRL_V2_RESULT_DIR:-train_session_logs/eval_v2_${RUN_NAME}_${STAMP}}"
if [[ "${RESULT_DIR}" != /* ]]; then
    # User-supplied relative output paths follow the same caller-relative rule as checkpoints.
    if [[ -n "${NAVRL_V2_RESULT_DIR:-}" ]]; then
        RESULT_DIR="${CALLER_PWD}/${RESULT_DIR}"
    else
        RESULT_DIR="${PWD}/${RESULT_DIR}"
    fi
fi
if [[ -e "${RESULT_DIR}" ]]; then
    echo "[eval_v2] refusing to overwrite existing result directory: ${RESULT_DIR}" >&2
    exit 2
fi
mkdir -p "${RESULT_DIR}"
RESULT_CSV="${RESULT_DIR}/results.csv"
CHECKPOINT_SNAPSHOT="${RESULT_DIR}/checkpoint_snapshot.pth"
SHARED_SOURCE_BUNDLE="${NAVRL_V2_SHARED_SOURCE_BUNDLE:-}"
if [[ -n "${SHARED_SOURCE_BUNDLE}" ]]; then
    if [[ "${SHARED_SOURCE_BUNDLE}" != /* ]]; then
        SHARED_SOURCE_BUNDLE="${CALLER_PWD}/${SHARED_SOURCE_BUNDLE}"
    fi
    SOURCE_BUNDLE_DIR="$(readlink -m -- "${SHARED_SOURCE_BUNDLE}")"
else
    SOURCE_BUNDLE_DIR="${RESULT_DIR}"
fi
SOURCE_SNAPSHOT_DIR="${SOURCE_BUNDLE_DIR}/source_snapshot"
SOURCE_MANIFEST="${SOURCE_BUNDLE_DIR}/source_manifest.json"
PYTHON_ENVIRONMENT="${SOURCE_BUNDLE_DIR}/python_environment.txt"

sha256_file() {
    "${PYTHON}" - "$1" <<'PY'
import hashlib
from pathlib import Path
import sys

digest = hashlib.sha256()
with Path(sys.argv[1]).open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
}

# Snapshot the actual runtime source bytes before Isaac Gym starts.  A git commit alone is not
# enough: a dirty worktree can be scientifically valid, but only if the evaluated bytes are
# preserved and checked after every cell.  Include every tracked/untracked runtime text source
# under aerial_gym plus the robot URDF assets under resources/robots (ignored runs/results/logs are
# excluded by git), plus a Python package manifest.  Omitting URDF used to let two different rigid
# bodies share an otherwise identical evaluation receipt.
# A campaign launcher may point several evaluator invocations at one immutable shared bundle so
# all arms are provably evaluated from identical source bytes instead of merely similar commits.
CREATE_SOURCE_BUNDLE=1
if [[ -n "${SHARED_SOURCE_BUNDLE}" && -e "${SOURCE_BUNDLE_DIR}" ]]; then
    if [[ ! -d "${SOURCE_BUNDLE_DIR}" || ! -f "${SOURCE_MANIFEST}" \
          || ! -f "${PYTHON_ENVIRONMENT}" || ! -d "${SOURCE_SNAPSHOT_DIR}" ]]; then
        echo "[eval_v2] shared source bundle is partial or malformed: ${SOURCE_BUNDLE_DIR}" >&2
        exit 3
    fi
    CREATE_SOURCE_BUNDLE=0
fi
if (( CREATE_SOURCE_BUNDLE )); then
    mkdir -p "${SOURCE_SNAPSHOT_DIR}"
    "${PYTHON}" - "${SCRIPT_DIR}" "${SOURCE_SNAPSHOT_DIR}" "${SOURCE_MANIFEST}" \
        "${PYTHON_ENVIRONMENT}" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

script_dir = Path(sys.argv[1]).resolve()
snapshot_dir = Path(sys.argv[2]).resolve()
manifest_path = Path(sys.argv[3]).resolve()
environment_path = Path(sys.argv[4]).resolve()
repo = Path(
    subprocess.check_output(
        ["git", "-C", str(script_dir), "rev-parse", "--show-toplevel"], text=True
    ).strip()
).resolve()

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def git_paths(*args):
    raw = subprocess.check_output(["git", "-C", str(repo), *args])
    return [Path(item.decode("utf-8")) for item in raw.split(b"\0") if item]

extensions = {".py", ".pyx", ".sh", ".yaml", ".yml", ".toml", ".json", ".csv", ".urdf"}
runtime_roots = ("aerial_gym", "resources/robots")
paths = set(git_paths("ls-files", "-z", "--", *runtime_roots))
paths.update(
    git_paths("ls-files", "--others", "--exclude-standard", "-z", "--", *runtime_roots)
)
paths = sorted(
    path for path in paths
    if path.suffix.lower() in extensions and "__pycache__" not in path.parts
)
if not paths:
    raise SystemExit("[eval_v2] no runtime sources found for the provenance snapshot")

entries = []
for relative in paths:
    source = (repo / relative).resolve()
    if not source.is_file() or repo not in source.parents:
        raise SystemExit(f"[eval_v2] invalid runtime source path: {relative}")
    destination = snapshot_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    entries.append(
        {
            "path": relative.as_posix(),
            "sha256": digest(source),
            "size_bytes": source.stat().st_size,
            "snapshot": destination.relative_to(manifest_path.parent).as_posix(),
        }
    )

try:
    freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--all"], text=True, stderr=subprocess.STDOUT
    )
except subprocess.CalledProcessError as exc:
    freeze = f"pip freeze failed ({exc.returncode})\n{exc.output}"
environment_text = (
    f"python_executable={Path(sys.executable).resolve()}\n"
    f"python_version={sys.version.replace(chr(10), ' ')}\n"
    f"platform={platform.platform()}\n"
    "\n[pip-freeze]\n"
    + freeze
)
environment_path.write_text(environment_text, encoding="utf-8")

runtime_status = subprocess.check_output(
    [
        "git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all",
        "--", *runtime_roots,
    ],
    text=True,
).splitlines()
repository_status = subprocess.check_output(
    ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"],
    text=True,
).splitlines()
manifest = {
    "schema_version": 2,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "repository_root": str(repo),
    "git_commit": subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip(),
    # Backward-compatible names describe executable bytes only. Result CSVs and Markdown drafts
    # cannot change the rollout and are retained separately as repository-wide metadata.
    "git_dirty": bool(runtime_status),
    "git_status": runtime_status,
    "repository_git_dirty": bool(repository_status),
    "repository_git_status": repository_status,
    "runtime_roots": list(runtime_roots),
    "python_environment": environment_path.relative_to(manifest_path.parent).as_posix(),
    "python_environment_sha256": digest(environment_path),
    "runtime_file_count": len(entries),
    "runtime_files": entries,
}
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
for path in [environment_path, *snapshot_dir.rglob("*")]:
    if path.is_file():
        path.chmod(0o444)
manifest_path.chmod(0o444)
PY
fi

# Keep the standard per-result paths available as symlinks.  The attestation verifier resolves
# these links to the one shared manifest, while a human opening any cell directory still finds the
# complete provenance entry point in the expected place.
if [[ -n "${SHARED_SOURCE_BUNDLE}" ]]; then
    ln -s "${SOURCE_MANIFEST}" "${RESULT_DIR}/source_manifest.json"
    ln -s "${PYTHON_ENVIRONMENT}" "${RESULT_DIR}/python_environment.txt"
    ln -s "${SOURCE_SNAPSHOT_DIR}" "${RESULT_DIR}/source_snapshot"
fi
SOURCE_MANIFEST_SHA256="$(sha256_file "${SOURCE_MANIFEST}")"
export NAVRL_EVAL_SOURCE_MANIFEST="${SOURCE_MANIFEST}"
export NAVRL_EVAL_SOURCE_MANIFEST_SHA256="${SOURCE_MANIFEST_SHA256}"
export NAVRL_EVAL_SOURCE_SNAPSHOT_DIR="${SOURCE_SNAPSHOT_DIR}"

# Learned-detector bytes are an external artifact rather than repository source.  Evaluate an
# immutable snapshot just like the PPO checkpoint; the expected SHA remains the artifact identity.
if [[ -n "${NAVRL_DETECTOR_CHECKPOINT:-}" ]]; then
    SOURCE_DETECTOR_CHECKPOINT="${NAVRL_DETECTOR_CHECKPOINT}"
    DETECTOR_SNAPSHOT="${RESULT_DIR}/detector_snapshot.pth"
    cp --reflink=auto -- "${SOURCE_DETECTOR_CHECKPOINT}" "${DETECTOR_SNAPSHOT}"
    chmod 0444 "${DETECTOR_SNAPSHOT}"
    if [[ "$(sha256_file "${DETECTOR_SNAPSHOT}")" != "${NAVRL_EXPECTED_DETECTOR_SHA256}" ]]; then
        echo "[eval_v2] detector snapshot SHA-256 mismatch; refusing evaluation." >&2
        exit 3
    fi
    export NAVRL_EVAL_SOURCE_DETECTOR="${SOURCE_DETECTOR_CHECKPOINT}"
    export NAVRL_DETECTOR_CHECKPOINT="${DETECTOR_SNAPSHOT}"
else
    export NAVRL_EVAL_SOURCE_DETECTOR=""
fi

# Evaluate an immutable byte snapshot and bind it back to the named source checkpoint.  Checking
# the source both before and after every cell prevents a path swap from attaching an old score to
# new policy bytes.  A reflink is used when supported, but it is still a distinct CoW inode.
SOURCE_CHECKPOINT_SHA256="$(sha256_file "${CKPT}")"
cp --reflink=auto -- "${CKPT}" "${CHECKPOINT_SNAPSHOT}"
chmod 0444 "${CHECKPOINT_SNAPSHOT}"
SNAPSHOT_CHECKPOINT_SHA256="$(sha256_file "${CHECKPOINT_SNAPSHOT}")"
if [[ "${SNAPSHOT_CHECKPOINT_SHA256}" != "${SOURCE_CHECKPOINT_SHA256}" ]]; then
    echo "[eval_v2] checkpoint snapshot SHA-256 mismatch; refusing evaluation." >&2
    exit 3
fi
EVALUATOR_SCRIPT_SHA256="$(sha256_file "${EVALUATOR_SCRIPT}")"

append_result_csv() {
    local result_json="$1"
    local expected_bars="$2"
    local cell_log="$3"
    local receipt_json="$4"
    local evaluation_nonce="$5"
    local started_at_utc="$6"
    "${PYTHON}" - "${result_json}" "${RESULT_CSV}" "${expected_bars}" "${GAMES}" \
        "${CKPT}" "${cell_log}" "${NAVRL_SEED}" "${SOURCE_CHECKPOINT_SHA256}" \
        "${CHECKPOINT_SNAPSHOT}" "${SNAPSHOT_CHECKPOINT_SHA256}" \
        "${EVALUATOR_SCRIPT}" "${EVALUATOR_SCRIPT_SHA256}" "${receipt_json}" \
        "${evaluation_nonce}" "${started_at_utc}" <<'PY'
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import sys
from pathlib import Path

(
    json_path,
    csv_path,
    bars_text,
    games_text,
    checkpoint,
    log_path,
    seed_text,
    checkpoint_sha256,
    checkpoint_snapshot,
    checkpoint_snapshot_sha256,
    evaluator_script,
    evaluator_script_sha256,
    receipt_path,
    expected_nonce,
    started_at_utc,
) = sys.argv[1:]
expected_bars = int(bars_text)
expected_games = int(games_text)
expected_seed = int(seed_text)

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

source_manifest_path = Path(os.environ["NAVRL_EVAL_SOURCE_MANIFEST"]).resolve()
source_manifest_sha256 = os.environ["NAVRL_EVAL_SOURCE_MANIFEST_SHA256"]
if sha256_file(source_manifest_path) != source_manifest_sha256:
    raise SystemExit("[eval_v2] runtime source manifest changed during evaluation")
source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
if source_manifest.get("schema_version") != 2:
    raise SystemExit("[eval_v2] unsupported runtime source manifest schema")
repository_root = Path(source_manifest["repository_root"]).resolve()
for entry in source_manifest.get("runtime_files") or []:
    original = (repository_root / entry["path"]).resolve()
    snapshot = (source_manifest_path.parent / entry["snapshot"]).resolve()
    expected = entry["sha256"]
    if not original.is_file() or sha256_file(original) != expected:
        raise SystemExit(f"[eval_v2] runtime source changed during evaluation: {entry['path']}")
    if not snapshot.is_file() or sha256_file(snapshot) != expected:
        raise SystemExit(f"[eval_v2] runtime source snapshot changed: {entry['snapshot']}")
environment_path = (
    source_manifest_path.parent / source_manifest["python_environment"]
).resolve()
if sha256_file(environment_path) != source_manifest["python_environment_sha256"]:
    raise SystemExit("[eval_v2] Python environment manifest changed during evaluation")

with open(json_path, encoding="utf-8") as stream:
    payload = json.load(stream)

if payload.get("schema_version") != 1:
    raise SystemExit(f"[eval_v2] unsupported bulk JSON schema: {payload.get('schema_version')!r}")
if int(payload.get("requested_episodes", -1)) != expected_games:
    raise SystemExit("[eval_v2] bulk JSON requested_episodes does not match the sweep cell")
actual = int(payload.get("actual_episodes", -1))
if actual < expected_games:
    raise SystemExit(f"[eval_v2] incomplete bulk result: {actual} < {expected_games} episodes")

condition = payload.get("condition") or {}
if condition.get("robot_name") != os.environ["NAVRL_ROBOT"]:
    raise SystemExit("[eval_v2] bulk JSON robot does not match the checkpoint lineage")
if int(os.environ.get("ROBOT_CONTRACT_VERSION", "0")) >= 1:
    if condition.get("robot_config_sha256") != os.environ[
        "NAVRL_EXPECTED_ROBOT_CONFIG_SHA256"
    ]:
        raise SystemExit("[eval_v2] runtime robot config SHA differs from the checkpoint")
    if condition.get("robot_asset_sha256") != os.environ[
        "NAVRL_EXPECTED_ROBOT_ASSET_SHA256"
    ]:
        raise SystemExit("[eval_v2] runtime robot URDF SHA differs from the checkpoint")
if int(condition.get("bars", -1)) != expected_bars:
    raise SystemExit("[eval_v2] bulk JSON density does not match the sweep cell")
if int(condition.get("seed", -1)) != expected_seed:
    raise SystemExit("[eval_v2] bulk JSON seed does not match the pinned held-out seed")
if condition.get("action_selection") != os.environ["NAVRL_EVAL_ACTION_MODE"]:
    raise SystemExit("[eval_v2] bulk JSON action selection does not match the requested mode")
if condition.get("reflection_mode") != os.environ["NAVRL_EVAL_REFLECTION_MODE"]:
    raise SystemExit("[eval_v2] bulk JSON reflection mode does not match the requested mode")
if condition.get("speed_governor_mode") != os.environ["NAVRL_SPEED_GOVERNOR"]:
    raise SystemExit("[eval_v2] bulk JSON speed governor does not match the requested mode")
if condition.get("speed_governor_target_exclusion") != "camera_lidar_association":
    raise SystemExit(
        "[eval_v2] speed governor target exclusion is not actor-safe camera/LiDAR association"
    )
joint_speed_requested = os.environ.get("NAVRL_JOINT_SPEED_TELEMETRY", "0").strip().lower() in {
    "1", "true", "yes", "on"
}
if joint_speed_requested:
    if condition.get("joint_speed_telemetry") is not True:
        raise SystemExit("[eval_v2] requested joint-speed condition attestation is missing")
    joint = payload.get("joint_speed_allocation")
    if not isinstance(joint, dict) or joint.get("evaluation_only") is not True:
        raise SystemExit("[eval_v2] requested joint-speed telemetry is missing or malformed")
elif "joint_speed_allocation" in payload or "joint_speed_telemetry" in condition:
    raise SystemExit("[eval_v2] unrequested joint-speed telemetry changed the canonical result")
governor_numeric = {
    "speed_governor_fixed_mps": "NAVRL_SPEED_GOVERNOR_FIXED_MPS",
    "speed_governor_free_mps": "NAVRL_SPEED_GOVERNOR_FREE_MPS",
    "speed_governor_half_width_m": "NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M",
    "speed_governor_margin_m": "NAVRL_SPEED_GOVERNOR_MARGIN_M",
    "speed_governor_slow_m": "NAVRL_SPEED_GOVERNOR_SLOW_M",
    "speed_governor_release_m": "NAVRL_SPEED_GOVERNOR_RELEASE_M",
    "speed_governor_ttc_s": "NAVRL_SPEED_GOVERNOR_TTC_S",
    "speed_governor_brake_mps2": "NAVRL_SPEED_GOVERNOR_BRAKE_MPS2",
    "speed_governor_reaction_s": "NAVRL_SPEED_GOVERNOR_REACTION_S",
}
for field, env_name in governor_numeric.items():
    try:
        matches = abs(float(condition.get(field)) - float(os.environ[env_name])) <= 1e-6
    except (TypeError, ValueError):
        matches = False
    if not matches:
        raise SystemExit(f"[eval_v2] bulk JSON {field} does not match {env_name}")
if int(condition.get("num_envs", -1)) != int(os.environ["NUM_ENVS"]):
    raise SystemExit("[eval_v2] bulk JSON num_envs does not match the pinned runtime profile")
if condition.get("evaluation_nonce") != expected_nonce:
    raise SystemExit("[eval_v2] bulk JSON nonce does not match this evaluator process")
expected_sim_class = (
    "BaseSimConfig" if os.environ["NAVRL_EVAL_PROFILE"] == "main" else "BaseSim4GBConfig"
)
physics_expected = {
    "runtime_sim_config_class": expected_sim_class,
    "physics_dt_s": 0.01,
    "physics_substeps": 1,
    "physics_steps_per_rl_step": 10,
    "rl_step_dt_s": 0.1,
}
for name, expected in physics_expected.items():
    got = condition.get(name)
    if isinstance(expected, str):
        matches = got == expected
    else:
        try:
            matches = math.isfinite(float(got)) and abs(float(got) - expected) <= 1e-9
        except (TypeError, ValueError):
            matches = False
    if not matches:
        raise SystemExit(
            f"[eval_v2] measured physics mismatch for {name}: {got!r} != {expected!r}"
        )
expected_goal_min = float(os.environ["NAVRL_GENERAL_GOAL_DIST_MIN"])
expected_goal_max = float(os.environ["NAVRL_GENERAL_GOAL_DIST_MAX"])
if abs(float(condition.get("goal_dist_min_m", -1.0)) - expected_goal_min) > 1e-6:
    raise SystemExit("[eval_v2] bulk JSON goal-distance minimum is not the requested value")
if abs(float(condition.get("goal_dist_max_m", -1.0)) - expected_goal_max) > 1e-6:
    raise SystemExit("[eval_v2] bulk JSON goal-distance maximum is not the requested value")
if condition.get("full_goal_distribution") is not True:
    raise SystemExit("[eval_v2] bulk JSON did not use the full goal-distance distribution")
if condition.get("fov_curriculum_saturated") is not True:
    raise SystemExit("[eval_v2] bulk JSON did not use the final FOV distribution")
if condition.get("target_pattern") != os.environ["NAVRL_TARGET_PATTERN"]:
    raise SystemExit("[eval_v2] bulk JSON target pattern is not the requested condition")
if condition.get("cv_initial_heading") != os.environ["NAVRL_EVAL_CV_INITIAL_HEADING"]:
    raise SystemExit("[eval_v2] bulk JSON CV initial heading is not the requested condition")
heading_mode = os.environ["NAVRL_EVAL_CV_INITIAL_HEADING"]
motion = payload.get("target_motion") or {}
if motion.get("cv_initial_heading") != heading_mode:
    raise SystemExit("[eval_v2] target-motion heading audit mode mismatch")
if heading_mode != "random":
    expected_radial = {
        "toward": (-1.0, 0.0),
        "tangent_left": (0.0, 1.0),
        "tangent_right": (0.0, -1.0),
        "away": (1.0, 0.0),
    }.get(heading_mode)
    if expected_radial is None:
        raise SystemExit("[eval_v2] unsupported controlled CV initial heading")
    if int(motion.get("initial_heading_samples", 0)) < actual:
        raise SystemExit("[eval_v2] controlled CV heading audit has too few reset samples")
    if abs(float(motion.get("initial_heading_mean_radial_cos")) - expected_radial[0]) > 1e-5:
        raise SystemExit("[eval_v2] controlled CV heading radial-cos audit failed")
    if abs(float(motion.get("initial_heading_mean_radial_sin")) - expected_radial[1]) > 1e-5:
        raise SystemExit("[eval_v2] controlled CV heading radial-sin audit failed")
    if float(motion.get("initial_heading_max_contract_error", 1.0)) > 1e-5:
        raise SystemExit("[eval_v2] controlled CV heading contract-error audit failed")
fixed_speed_text = os.environ.get("NAVRL_V2_FIXED_TARGET_SPEED", "").strip()
fixed_speed = float(fixed_speed_text) if fixed_speed_text else None
if fixed_speed is None:
    if condition.get("target_speed_mode") != "uniform":
        raise SystemExit("[eval_v2] bulk JSON target speed mode is not uniform")
    if abs(float(condition.get("target_speed_min_mps", -1.0)) - 0.3) > 1e-6:
        raise SystemExit("[eval_v2] bulk JSON target-speed minimum is not 0.3 m/s")
    expected_speed_max = float(os.environ["NAVRL_TARGET_SPEED_FINAL"])
    if abs(float(condition.get("target_speed_max_mps", -1.0)) - expected_speed_max) > 1e-6:
        raise SystemExit(
            f"[eval_v2] bulk JSON target-speed maximum is not {expected_speed_max:g} m/s"
        )
else:
    if condition.get("target_speed_mode") != "fixed":
        raise SystemExit("[eval_v2] bulk JSON target speed mode is not fixed")
    for name in ("target_speed_mps", "target_speed_min_mps", "target_speed_max_mps"):
        if abs(float(condition.get(name, -1.0)) - fixed_speed) > 1e-6:
            raise SystemExit(
                f"[eval_v2] bulk JSON {name} is not the requested {fixed_speed:g} m/s"
            )
if abs(float(condition.get("oob_margin_m", -1.0)) - 1.0) > 1e-6:
    raise SystemExit("[eval_v2] bulk JSON OOB margin is not the trained 1.0 m")
if condition.get("pursuer_speed_limit_semantics") != "per_axis_xy":
    raise SystemExit("[eval_v2] pursuer speed limit is not recorded as per-axis XY")
if abs(float(condition.get("pursuer_per_axis_speed_limit_mps", -1.0)) - 2.5) > 1e-6:
    raise SystemExit("[eval_v2] pursuer per-axis speed limit is not 2.5 m/s")
if abs(
    float(condition.get("pursuer_max_horizontal_request_norm_mps", -1.0))
    - math.sqrt(2.0) * 2.5
) > 1e-6:
    raise SystemExit("[eval_v2] pursuer maximum XY request norm is not sqrt(2)*2.5 m/s")
if int(condition.get("policy_output_dim", -1)) != 4:
    raise SystemExit("[eval_v2] policy output dimension is not 4")
if condition.get("policy_z_output_overwritten_by_altitude_pi") is not True:
    raise SystemExit("[eval_v2] policy-z altitude-PI overwrite is not attested")
if condition.get("policy_z_persisted_in_prev_action_observation") is not True:
    raise SystemExit("[eval_v2] indirect policy-z prev_action channel is not attested")
if Path(payload.get("checkpoint", "")).resolve() != Path(checkpoint).resolve():
    raise SystemExit("[eval_v2] bulk JSON checkpoint does not match the requested checkpoint")
if sha256_file(checkpoint) != checkpoint_sha256:
    raise SystemExit("[eval_v2] source checkpoint changed during evaluation")
if sha256_file(checkpoint_snapshot) != checkpoint_snapshot_sha256:
    raise SystemExit("[eval_v2] evaluated checkpoint snapshot changed during evaluation")
if checkpoint_snapshot_sha256 != checkpoint_sha256:
    raise SystemExit("[eval_v2] evaluated snapshot is not byte-identical to the source checkpoint")
if sha256_file(evaluator_script) != evaluator_script_sha256:
    raise SystemExit("[eval_v2] evaluator script changed during evaluation")

payload.update(
    {
        "checkpoint_sha256": checkpoint_sha256,
        "evaluated_checkpoint_snapshot": str(Path(checkpoint_snapshot).resolve()),
        "evaluated_checkpoint_snapshot_sha256": checkpoint_snapshot_sha256,
        "evaluator_script": str(Path(evaluator_script).resolve()),
        "evaluator_script_sha256": evaluator_script_sha256,
        "runtime_source_manifest": str(source_manifest_path),
        "runtime_source_manifest_sha256": source_manifest_sha256,
        "runtime_source_snapshot": str(
            Path(os.environ["NAVRL_EVAL_SOURCE_SNAPSHOT_DIR"]).resolve()
        ),
        "runtime_git_commit": source_manifest["git_commit"],
        "runtime_git_dirty": bool(source_manifest["git_dirty"]),
        "python_environment_manifest": str(environment_path),
        "python_environment_manifest_sha256": source_manifest[
            "python_environment_sha256"
        ],
        "evaluation_receipt": str(Path(receipt_path).resolve()),
    }
)

outcome = payload.get("outcome") or {}
counts = [int(outcome.get(name, -1)) for name in ("captured", "crash", "timeout")]
if any(value < 0 for value in counts) or sum(counts) != actual:
    raise SystemExit(
        f"[eval_v2] invalid outcome accounting: captured+crash+timeout={sum(counts)} actual={actual}"
    )
for name in ("capture_rate", "crash_rate", "timeout_rate"):
    value = float(outcome.get(name, float("nan")))
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise SystemExit(f"[eval_v2] invalid {name}: {value!r}")
for name, count in zip(("capture_rate", "crash_rate", "timeout_rate"), counts):
    reported = float(outcome[name])
    measured = count / actual
    if abs(reported - measured) > 1e-9:
        raise SystemExit(
            f"[eval_v2] {name}={reported:.12g} disagrees with count/actual={measured:.12g}"
        )

bearing = (payload.get("strata") or {}).get("initial_target_bearing") or {}
bearing_counts = [bearing.get(name) or {} for name in (
    "negative_y", "centered_5deg", "positive_y"
)]
if sum(int(cell.get("episodes", -1)) for cell in bearing_counts) != actual:
    raise SystemExit("[eval_v2] initial-target-bearing strata do not account for every episode")
for outcome_name, expected_count in zip(("captured", "crash", "timeout"), counts):
    if sum(int(cell.get(outcome_name, -1)) for cell in bearing_counts) != expected_count:
        raise SystemExit(
            "[eval_v2] initial-target-bearing %s counts do not match the bulk outcome"
            % outcome_name
        )

causes = payload.get("crash_causes") or {}
action = payload.get("action") or {}
speed_governor = payload.get("speed_governor") or {}
if action.get("policy") != "squashed_gaussian":
    raise SystemExit("[eval_v2] bulk JSON action policy is not squashed_gaussian")
if speed_governor.get("mode") != os.environ["NAVRL_SPEED_GOVERNOR"]:
    raise SystemExit("[eval_v2] speed_governor payload mode mismatch")
if speed_governor.get("sensor_only") is not True or speed_governor.get("direction_preserved") is not True:
    raise SystemExit("[eval_v2] speed governor must attest sensor-only direction-preserving execution")
if int(speed_governor.get("samples", 0)) <= 0:
    raise SystemExit("[eval_v2] speed governor diagnostics contain no samples")

# A contract field is not evidence by itself.  Prove from the per-outcome diagnostic that every
# measured timeout occurred on action 600.  This catches a regression back to the legacy > limit
# (all legacy timeout summaries are exactly 601).
timeout_steps = (speed_governor.get("outcome_steps") or {}).get("timeout") or {}
if counts[2] > 0:
    if int(timeout_steps.get("count", -1)) != counts[2]:
        raise SystemExit("[eval_v2] timeout-step diagnostic does not account for every timeout")
    for field in ("mean", "p10", "p50", "p90"):
        if abs(float(timeout_steps.get(field, -1.0)) - 600.0) > 1e-9:
            raise SystemExit(
                f"[eval_v2] timeout {field} is not exactly action 600: {timeout_steps.get(field)!r}"
            )
elif int(timeout_steps.get("count", 0)) != 0:
    raise SystemExit("[eval_v2] timeout-step diagnostic has samples but outcome timeout count is zero")

def action_y(name):
    values = action.get(name) or []
    if len(values) < 2:
        raise SystemExit(f"[eval_v2] bulk JSON action.{name} lacks the lateral component")
    value = float(values[1])
    if not math.isfinite(value):
        raise SystemExit(f"[eval_v2] invalid action.{name}[1]: {value!r}")
    return value

# Retain a compact evaluator-level contract beside the task's measured condition so downstream
# plotting can validate the whole sweep without relying on shell history.
payload["v2_evaluation_contract"] = {
    "schema_version": 2,
    "episode_limit_steps": 600,
    "episode_limit_comparator": "gte",
    "timeout_observed_at_step": 600,
    "pursuer_speed_limit_semantics": "per_axis_xy",
    "pursuer_per_axis_speed_limit_mps": 2.5,
    "pursuer_max_horizontal_request_norm_mps": math.sqrt(2.0) * 2.5,
    "policy_output_dim": 4,
    "policy_z_output_overwritten_by_altitude_pi": True,
    "policy_z_persisted_in_prev_action_observation": True,
    "runtime_sim": os.environ["AERIAL_GYM_SIM_NAME"],
    "runtime_profile": os.environ["NAVRL_EVAL_PROFILE"],
    "runtime_num_envs": int(os.environ["NUM_ENVS"]),
    "action_selection": os.environ["NAVRL_EVAL_ACTION_MODE"],
    "reflection_mode": os.environ["NAVRL_EVAL_REFLECTION_MODE"],
    "speed_governor_mode": os.environ["NAVRL_SPEED_GOVERNOR"],
    "speed_governor_fixed_mps": float(os.environ["NAVRL_SPEED_GOVERNOR_FIXED_MPS"]),
    "speed_governor_free_mps": float(os.environ["NAVRL_SPEED_GOVERNOR_FREE_MPS"]),
    "speed_governor_half_width_m": float(os.environ["NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M"]),
    "speed_governor_margin_m": float(os.environ["NAVRL_SPEED_GOVERNOR_MARGIN_M"]),
    "speed_governor_slow_m": float(os.environ["NAVRL_SPEED_GOVERNOR_SLOW_M"]),
    "speed_governor_release_m": float(os.environ["NAVRL_SPEED_GOVERNOR_RELEASE_M"]),
    "speed_governor_ttc_s": float(os.environ["NAVRL_SPEED_GOVERNOR_TTC_S"]),
    "speed_governor_brake_mps2": float(os.environ["NAVRL_SPEED_GOVERNOR_BRAKE_MPS2"]),
    "speed_governor_reaction_s": float(os.environ["NAVRL_SPEED_GOVERNOR_REACTION_S"]),
    "speed_governor_target_exclusion": "camera_lidar_association",
    **({"joint_speed_telemetry": True} if joint_speed_requested else {}),
    # Perception arm: WHICH perturbation was injected and WHICH compensation was enabled. Without
    # these, two cells of a robustness sweep produce byte-identical provenance and the archive
    # cannot say which arm a number came from.
    "perception_perturb": os.environ.get("NAVRL_PERCEPTION_PERTURB", "0") == "1",
    "perception_detection_dropout": float(os.environ.get("NAVRL_DETECTION_DROPOUT", 0.0)),
    "perception_detection_dropout_active": (
        float(os.environ.get("NAVRL_DETECTION_DROPOUT", 0.0))
        if os.environ.get("NAVRL_PERCEPTION_PERTURB", "0") == "1"
        else 0.0
    ),
    "perception_detection_latency_s": float(os.environ.get("NAVRL_DETECTION_LATENCY_S", 0.0)),
    "perception_range_error_m": float(os.environ.get("NAVRL_RANGE_ERROR_M", 0.0)),
    "perception_latency_compensate": os.environ.get("NAVRL_LATENCY_COMPENSATE", "0") == "1",
    "perception_latency_lidar_backup": os.environ.get("NAVRL_LATENCY_LIDAR_BACKUP", "0") == "1",
    "perception_latency_obstacle_fix": os.environ.get("NAVRL_LATENCY_OBSTACLE_FIX", "off"),
    "perception_latency_ego_motion_fix": os.environ.get("NAVRL_LATENCY_EGO_MOTION_FIX", "0") == "1",
    "perception_pose_clock_offset_s": float(os.environ.get("NAVRL_POSE_CLOCK_OFFSET_S", 0.0)),
    "perception_pose_noise_pos_m": float(os.environ.get("NAVRL_POSE_NOISE_POS_M", 0.0)),
    "perception_pose_noise_yaw_deg": float(os.environ.get("NAVRL_POSE_NOISE_YAW_DEG", 0.0)),
    "perception_pose_noise_seed": int(os.environ.get("NAVRL_POSE_NOISE_SEED", 9163)),
    "perception_target_mask_backfill": os.environ.get("NAVRL_TARGET_MASK_BACKFILL", "0") == "1",
    "perception_lidar_target_assoc": os.environ.get("NAVRL_LIDAR_TARGET_ASSOC", "1") == "1",
    "perception_lidar_range_only_update": os.environ.get("NAVRL_LIDAR_RANGE_ONLY_UPDATE", "0") == "1",
    "perception_lidar_assoc_gate_m": float(os.environ.get("NAVRL_LIDAR_ASSOC_GATE_M", 0.0)),
    "perception_lidar_silent_correct": os.environ.get("NAVRL_LIDAR_SILENT_CORRECT", "0") == "1",
    "appearance_appearance_hue_deg": float(os.environ.get("NAVRL_APP_HUE_DEG", 0.0)),
    "appearance_appearance_light_gain": float(os.environ.get("NAVRL_APP_LIGHT_GAIN", 0.0)),
    "appearance_appearance_albedo_jitter": float(os.environ.get("NAVRL_APP_ALBEDO_JITTER", 0.0)),
    "appearance_appearance_texture_std": float(os.environ.get("NAVRL_APP_TEXTURE_STD", 0.0)),
    "appearance_appearance_motion_blur": float(os.environ.get("NAVRL_APP_MOTION_BLUR", 0.0)),
    "camera_mount_rot_deg": float(os.environ.get("NAVRL_CAM_MOUNT_ROT_DEG", 0.0)),
    "camera_mount_trans_m": float(os.environ.get("NAVRL_CAM_MOUNT_TRANS_M", 0.0)),
    "camera_fov_scale_err": float(os.environ.get("NAVRL_CAM_FOV_SCALE_ERR", 0.0)),
    "sim_physics_contract": os.environ["NAVRL_SIM_PHYSICS_CONTRACT"],
    "runtime_sim_config_class": condition["runtime_sim_config_class"],
    "physics_dt_s": float(condition["physics_dt_s"]),
    "physics_substeps": int(condition["physics_substeps"]),
    "physics_steps_per_rl_step": int(condition["physics_steps_per_rl_step"]),
    "rl_step_dt_s": float(condition["rl_step_dt_s"]),
    "arena_xy_m": 40.0,
    "goal_dist_min_m": expected_goal_min,
    "goal_dist_max_m": expected_goal_max,
    "full_goal_distribution": True,
    "fov_curriculum_saturated": True,
    "target_speed_distribution": "fixed" if fixed_speed is not None else "uniform",
    "target_speed_mps": fixed_speed,
    "target_speed_min_mps": fixed_speed if fixed_speed is not None else 0.3,
    "target_speed_max_mps": fixed_speed if fixed_speed is not None else 1.5,
    "target_pattern": os.environ["NAVRL_TARGET_PATTERN"],
    "cv_initial_heading": os.environ["NAVRL_EVAL_CV_INITIAL_HEADING"],
    # RESEARCH_PLAN 8.29: the observability arm identity. Without it the two arms of the
    # camera-range control produce provenance that cannot tell them apart.
    "target_camera_max_range_m": float(os.environ.get("NAVRL_DETECTOR_MAX_RANGE", 20.0)),
    "geofence_actor": os.environ.get("NAVRL_GEOFENCE_ACTOR", "0") == "1",
    "geofence_noise_std_m": float(os.environ.get("NAVRL_GEOFENCE_NOISE_STD_M", 0.0)),
    "geofence_dropout": float(os.environ.get("NAVRL_GEOFENCE_DROPOUT", 0.0)),
    "geofence_force_invalid": os.environ.get("NAVRL_GEOFENCE_FORCE_INVALID", "0") == "1",
    "lidar_beams": [4, 72],
    "lidar_range_m": 12.0,
    "obstacle_tokens": 8,
    "obstacle_fov_deg": 240.0,
    "obstacle_effective_fov_deg": (
        360.0 if os.environ["NAVRL_OBSTACLE_SELECTOR"] == "ttc_sector" else 240.0
    ),
    "obstacle_suppress_active": os.environ["NAVRL_OBSTACLE_SELECTOR"] == "greedy_suppress",
    "obstacle_selector": os.environ["NAVRL_OBSTACLE_SELECTOR"],
    "obstacle_ttc_idle_s": float(os.environ["NAVRL_OBSTACLE_TTC_IDLE_S"]),
    "obstacle_ttc_min_speed": float(os.environ["NAVRL_OBSTACLE_TTC_MIN_SPEED"]),
    "fov_curriculum_epochs": int(os.environ["NAVRL_FOV_CURRICULUM_EPOCHS"]),
    "detector_checkpoint_sha256": os.environ["NAVRL_EXPECTED_DETECTOR_SHA256"],
    "detector_min_pixels": int(os.environ["NAVRL_DETECTOR_MIN_PIXELS"]),
    "detector_threshold": float(os.environ["NAVRL_DETECTOR_THRESHOLD"]),
    "perception_perturb": bool(int(os.environ["NAVRL_PERCEPTION_PERTURB"])),
    "detection_dropout": float(os.environ["NAVRL_DETECTION_DROPOUT"]),
    "detection_dropout_active": (
        float(os.environ["NAVRL_DETECTION_DROPOUT"])
        if bool(int(os.environ["NAVRL_PERCEPTION_PERTURB"]))
        else 0.0
    ),
    "detection_latency_s": float(os.environ["NAVRL_DETECTION_LATENCY_S"]),
    "range_error_m": float(os.environ["NAVRL_RANGE_ERROR_M"]),
    "rgb_noise_std": float(os.environ["NAVRL_RGB_NOISE_STD"]),
    "depth_noise_std": float(os.environ["NAVRL_DEPTH_NOISE_STD"]),
    "max_tilt_deg": float(os.environ["NAVRL_MAX_TILT_DEG"]),
    "tilt_comp": bool(int(os.environ["NAVRL_TILT_COMP"])),
    "oob_margin_m": 1.0,
    "seed": expected_seed,
}
Path(json_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

receipt = {
    "schema_version": 2,
    "producer": "eval_navrl_v2_density_sweep.sh",
    "started_at_utc": started_at_utc,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "evaluation_nonce": expected_nonce,
    "source_checkpoint": str(Path(checkpoint).resolve()),
    "source_checkpoint_sha256": checkpoint_sha256,
    "evaluated_checkpoint_snapshot": str(Path(checkpoint_snapshot).resolve()),
    "evaluated_checkpoint_snapshot_sha256": checkpoint_snapshot_sha256,
    "result_json": str(Path(json_path).resolve()),
    "result_sha256": sha256_file(json_path),
    "log_file": str(Path(log_path).resolve()),
    "log_sha256": sha256_file(log_path),
    "evaluator_script": str(Path(evaluator_script).resolve()),
    "evaluator_script_sha256": evaluator_script_sha256,
    "runtime_source_manifest": str(source_manifest_path),
    "runtime_source_manifest_sha256": source_manifest_sha256,
    "runtime_source_file_count": int(source_manifest["runtime_file_count"]),
    "runtime_git_commit": source_manifest["git_commit"],
    "runtime_git_dirty": bool(source_manifest["git_dirty"]),
    "python_environment_manifest": str(environment_path),
    "python_environment_manifest_sha256": source_manifest[
        "python_environment_sha256"
    ],
    "source_detector_checkpoint": os.environ.get("NAVRL_EVAL_SOURCE_DETECTOR", ""),
    "evaluated_detector_snapshot": os.environ.get("NAVRL_DETECTOR_CHECKPOINT", ""),
    "evaluated_detector_snapshot_sha256": os.environ.get(
        "NAVRL_EXPECTED_DETECTOR_SHA256", ""
    ),
    "bars": expected_bars,
    "seed": expected_seed,
    "requested_episodes": expected_games,
    "actual_episodes": actual,
    "action_selection": os.environ["NAVRL_EVAL_ACTION_MODE"],
    "reflection_mode": os.environ["NAVRL_EVAL_REFLECTION_MODE"],
    "goal_dist_min_m": expected_goal_min,
    "goal_dist_max_m": expected_goal_max,
    "target_pattern": os.environ["NAVRL_TARGET_PATTERN"],
    "cv_initial_heading": os.environ["NAVRL_EVAL_CV_INITIAL_HEADING"],
    # RESEARCH_PLAN 8.29: the observability arm identity. Without it the two arms of the
    # camera-range control produce provenance that cannot tell them apart.
    "target_camera_max_range_m": float(os.environ.get("NAVRL_DETECTOR_MAX_RANGE", 20.0)),
    "geofence_actor": os.environ.get("NAVRL_GEOFENCE_ACTOR", "0") == "1",
    "geofence_noise_std_m": float(os.environ.get("NAVRL_GEOFENCE_NOISE_STD_M", 0.0)),
    "geofence_dropout": float(os.environ.get("NAVRL_GEOFENCE_DROPOUT", 0.0)),
    "geofence_force_invalid": os.environ.get("NAVRL_GEOFENCE_FORCE_INVALID", "0") == "1",
    "speed_governor_mode": os.environ["NAVRL_SPEED_GOVERNOR"],
    "speed_governor_target_exclusion": "camera_lidar_association",
    **({"joint_speed_telemetry": True} if joint_speed_requested else {}),
    "perception_perturb": os.environ.get("NAVRL_PERCEPTION_PERTURB", "0") == "1",
    "perception_detection_dropout": float(os.environ.get("NAVRL_DETECTION_DROPOUT", 0.0)),
    "perception_detection_latency_s": float(os.environ.get("NAVRL_DETECTION_LATENCY_S", 0.0)),
    "perception_range_error_m": float(os.environ.get("NAVRL_RANGE_ERROR_M", 0.0)),
    "perception_latency_compensate": os.environ.get("NAVRL_LATENCY_COMPENSATE", "0") == "1",
    "perception_latency_lidar_backup": os.environ.get("NAVRL_LATENCY_LIDAR_BACKUP", "0") == "1",
    "perception_latency_obstacle_fix": os.environ.get("NAVRL_LATENCY_OBSTACLE_FIX", "off"),
    "perception_latency_ego_motion_fix": os.environ.get("NAVRL_LATENCY_EGO_MOTION_FIX", "0") == "1",
    "perception_pose_clock_offset_s": float(os.environ.get("NAVRL_POSE_CLOCK_OFFSET_S", 0.0)),
    "perception_pose_noise_pos_m": float(os.environ.get("NAVRL_POSE_NOISE_POS_M", 0.0)),
    "perception_pose_noise_yaw_deg": float(os.environ.get("NAVRL_POSE_NOISE_YAW_DEG", 0.0)),
    "perception_pose_noise_seed": int(os.environ.get("NAVRL_POSE_NOISE_SEED", 9163)),
    "perception_target_mask_backfill": os.environ.get("NAVRL_TARGET_MASK_BACKFILL", "0") == "1",
    "perception_lidar_target_assoc": os.environ.get("NAVRL_LIDAR_TARGET_ASSOC", "1") == "1",
    "perception_lidar_range_only_update": os.environ.get("NAVRL_LIDAR_RANGE_ONLY_UPDATE", "0") == "1",
    "perception_lidar_assoc_gate_m": float(os.environ.get("NAVRL_LIDAR_ASSOC_GATE_M", 0.0)),
    "perception_lidar_silent_correct": os.environ.get("NAVRL_LIDAR_SILENT_CORRECT", "0") == "1",
    "appearance_appearance_hue_deg": float(os.environ.get("NAVRL_APP_HUE_DEG", 0.0)),
    "appearance_appearance_light_gain": float(os.environ.get("NAVRL_APP_LIGHT_GAIN", 0.0)),
    "appearance_appearance_albedo_jitter": float(os.environ.get("NAVRL_APP_ALBEDO_JITTER", 0.0)),
    "appearance_appearance_texture_std": float(os.environ.get("NAVRL_APP_TEXTURE_STD", 0.0)),
    "appearance_appearance_motion_blur": float(os.environ.get("NAVRL_APP_MOTION_BLUR", 0.0)),
    "camera_mount_rot_deg": float(os.environ.get("NAVRL_CAM_MOUNT_ROT_DEG", 0.0)),
    "camera_mount_trans_m": float(os.environ.get("NAVRL_CAM_MOUNT_TRANS_M", 0.0)),
    "camera_fov_scale_err": float(os.environ.get("NAVRL_CAM_FOV_SCALE_ERR", 0.0)),
}
receipt_file = Path(receipt_path)
temporary_receipt = receipt_file.with_name(
    f"{receipt_file.name}.tmp.{os.getpid()}"
)
temporary_receipt.write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
os.replace(temporary_receipt, receipt_file)

row = {
    "bars": expected_bars,
    "seed": expected_seed,
    "density_per_100m2": expected_bars / 16.0,
    "target_speed_distribution": (
        f"fixed:{fixed_speed:g}"
        if fixed_speed is not None
        else f"U[0.3,{os.environ['NAVRL_TARGET_SPEED_FINAL']}]"
    ),
    "target_pattern": os.environ["NAVRL_TARGET_PATTERN"],
    "requested_episodes": expected_games,
    "actual_episodes": actual,
    "captured": counts[0],
    "crash": counts[1],
    "timeout": counts[2],
    "capture_rate": float(outcome["capture_rate"]),
    "crash_rate": float(outcome["crash_rate"]),
    "timeout_rate": float(outcome["timeout_rate"]),
    "bar_contact_rate": int(causes.get("bar_contact", 0)) / actual,
    "below_rate": int(causes.get("below", 0)) / actual,
    "above_rate": int(causes.get("above", 0)) / actual,
    "out_of_bounds_rate": int(causes.get("out_of_bounds", 0)) / actual,
    "action_policy": action.get("policy", ""),
    "action_selection": condition.get("action_selection", ""),
    "reflection_mode": condition.get("reflection_mode", ""),
    "lateral_task_input_oob_rate": action_y("task_input_oob_rate"),
    "lateral_executed_edge98_rate": action_y("executed_edge98_rate"),
    "lateral_mean_abs": action_y("mean_abs"),
    "mean_abs_delta_y": float(action.get("mean_abs_delta_y", float("nan"))),
    "sign_flip_y_rate": float(action.get("sign_flip_y_rate", float("nan"))),
    "speed_governor_mode": speed_governor.get("mode", ""),
    "speed_governor_target_exclusion": speed_governor.get("target_exclusion_source", ""),
    "governor_intervention_rate": float(speed_governor.get("intervention_rate", float("nan"))),
    "governor_mean_requested_speed_mps": float(speed_governor.get("mean_requested_speed_mps", float("nan"))),
    "governor_mean_executed_speed_mps": float(speed_governor.get("mean_executed_speed_mps", float("nan"))),
    "governor_contact_actual_speed_mps": float((speed_governor.get("contact") or {}).get("mean_actual_speed_mps", float("nan"))),
    "json_file": str(Path(json_path).resolve()),
    "log_file": str(Path(log_path).resolve()),
}
fields = list(row)
csv_file = Path(csv_path)
write_header = not csv_file.exists()
with csv_file.open("a", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    if write_header:
        writer.writeheader()
    writer.writerow(row)

print(
    "[eval_v2] result | bars=%d episodes=%d capture=%.4f crash=%.4f timeout=%.4f "
    "lateral_oob=%.4f lateral_edge98=%.4f"
    % (
        expected_bars,
        actual,
        row["capture_rate"],
        row["crash_rate"],
        row["timeout_rate"],
        row["lateral_task_input_oob_rate"],
        row["lateral_executed_edge98_rate"],
    )
)
PY
}

echo "[eval_v2] arena=${NAVRL_ARENA_XY}m pool=${NAVRL_BAR_POOL} placement=${NAVRL_PLACEMENT_MODE} \
episode=${NAVRL_EPISODE_LEN_STEPS} goal=${NAVRL_GENERAL_GOAL_DIST_MIN}..${NAVRL_GENERAL_GOAL_DIST_MAX}m"
echo "[eval_v2] target=${TARGET_SPEED_DESCRIPTION} pattern=${NAVRL_TARGET_PATTERN} \
density_curriculum=${NAVRL_DENSITY_CURRICULUM} OOB_margin=${NAVRL_OOB_MARGIN}m seed=${NAVRL_SEED}"
echo "[eval_v2] lidar=${NAVRL_LIDAR_RANGE}m scan=${NAVRL_LIDAR_VBEAMS}x${NAVRL_LIDAR_HBEAMS} \
tokens=${NAVRL_MAX_OBSTACLES} selector=${NAVRL_OBSTACLE_SELECTOR} corridor=${NAVRL_CORRIDOR_TOKENS}"
echo "[eval_v2] ckpt=${CKPT} games/cell=${GAMES} densities=${DENSITIES[*]}"
echo "[eval_v2] results=${RESULT_DIR}"

for N in "${DENSITIES[@]}"; do
    DENSITY_PER_AREA="$(${PYTHON} -c "print(f'{${N}/16.0:.1f}')")"
    CELL_PREFIX="${RESULT_DIR}/${N}bars"
    RESULT_JSON="${CELL_PREFIX}.json"
    CELL_LOG="${CELL_PREFIX}.log"
    RECEIPT_JSON="${CELL_PREFIX}.receipt.json"
    EVALUATION_NONCE="$(${PYTHON} -c 'import secrets; print(secrets.token_hex(32))')"
    EVALUATION_STARTED_AT="$(${PYTHON} -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).isoformat())')"
    echo "======== v2 density ${N} bars (${DENSITY_PER_AREA}/100m2) ========"
    NAVRL_NUM_BARS="${N}" \
    NAVRL_BULK_EVAL_JSON="${RESULT_JSON}" \
    NAVRL_EVAL_RUN_NONCE="${EVALUATION_NONCE}" \
    NUM_ENVS="${NUM_ENVS}" \
    HEADLESS=True \
    PLAY_GAMES_NUM="${GAMES}" \
        ./play_navrl.sh "${CHECKPOINT_SNAPSHOT}" 2>&1 | tee "${CELL_LOG}"
    if [[ ! -s "${RESULT_JSON}" ]]; then
        echo "[eval_v2] evaluation completed without a bulk result: ${RESULT_JSON}" >&2
        echo "[eval_v2] inspect: ${CELL_LOG}" >&2
        exit 4
    fi
    if [[ "$(sha256_file "${CKPT}")" != "${SOURCE_CHECKPOINT_SHA256}" \
          || "$(sha256_file "${CHECKPOINT_SNAPSHOT}")" != "${SNAPSHOT_CHECKPOINT_SHA256}" \
          || "$(sha256_file "${EVALUATOR_SCRIPT}")" != "${EVALUATOR_SCRIPT_SHA256}" ]]; then
        echo "[eval_v2] checkpoint or evaluator bytes changed during the cell; refusing result." >&2
        exit 4
    fi
    append_result_csv "${RESULT_JSON}" "${N}" "${CELL_LOG}" "${RECEIPT_JSON}" \
        "${EVALUATION_NONCE}" "${EVALUATION_STARTED_AT}"
done

if [[ "${RECOVERY_STAGE}" == "smoke" ]]; then
    "${PYTHON}" ../../../tools/navrl_v2_recovery_attestation.py \
        "${CKPT}" "${RESULT_DIR}/130bars.json"
fi

echo "[eval_v2] done | CSV=${RESULT_CSV} | per-cell JSON/logs=${RESULT_DIR}"
