#!/usr/bin/env bash
# Fresh one-lever active-search A/B. Invoke once per arm: control, then geofence.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if (( $# != 1 )) || [[ "$1" != "control" && "$1" != "geofence" ]]; then
    echo "usage: $0 {control|geofence}" >&2
    exit 2
fi
ARM="$1"
AS_PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
AS_PREFLIGHT="${ACTIVE_SEARCH_PREFLIGHT_ONLY:-0}"

if [[ -n "${CKPT:-}" ]]; then
    echo "[active-search-${ARM}] refusing checkpoint warm-start: ${CKPT}" >&2
    exit 2
fi
while IFS= read -r name; do
    case "${name}" in
        NAVRL_*|AERIAL_RUN_TAG|AERIAL_GYM_SIM_NAME|TRAIN_SESSION_LOG|TRAIN_LIVE_LOG|MAX_EPOCHS|SEED|NUM_ENVS|FILE|TASK|GPU4GB|CKPT|ALLOW_CONCURRENT|PYTORCH_CUDA_ALLOC_CONF)
            unset "${name}"
            ;;
    esac
done < <(compgen -v)
export PYTHON="${AS_PYTHON}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(git rev-parse --show-toplevel)"
# The host editable install points at the primary worktree. Pin this experiment to its audited
# branch before creating a receipt or importing any aerial_gym module; otherwise source receipt
# and executed code can diverge even though runner.py itself is local.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
EXPECTED_IMPORT="$(realpath "${REPO_ROOT}/aerial_gym/__init__.py")"
ACTUAL_IMPORT="$(${PYTHON} -c 'import importlib.util,os; print(os.path.realpath(importlib.util.find_spec("aerial_gym").origin))')"
if [[ "${ACTUAL_IMPORT}" != "${EXPECTED_IMPORT}" ]]; then
    echo "[active-search-${ARM}] aerial_gym import escaped worktree: ${ACTUAL_IMPORT}" >&2
    exit 2
fi
if [[ "${AS_PREFLIGHT}" != "1" ]]; then
    RUNTIME_DIRTY="$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all -- \
        aerial_gym resources/robots tools/create_navrl_source_bundle.py)"
    if [[ -n "${RUNTIME_DIRTY}" ]]; then
        echo "[active-search-${ARM}] refusing dirty runtime sources; commit first." >&2
        printf '%s\n' "${RUNTIME_DIRTY}" | sed -n '1,20p' >&2
        exit 2
    fi
fi

export MAX_EPOCHS=900
export SEED=197
export NAVRL_V2_PROFILE=main
export NAVRL_ROBOT=navrl_ref5in_quad
export NAVRL_YAW_RATE_MAX=3.0
export NAVRL_MAX_TILT_DEG=45.0
export NAVRL_ALT_HOLD_VMAX=2.5
export NAVRL_SPEED_GOVERNOR=off
export NAVRL_PERCEPTION_PERTURB=0
export NAVRL_POSE_CLOCK_OFFSET_S=0
export NAVRL_POSE_NOISE_POS_M=0
export NAVRL_POSE_NOISE_YAW_DEG=0
export NAVRL_APP_HUE_DEG=0
export NAVRL_APP_LIGHT_GAIN=0
export NAVRL_APP_ALBEDO_JITTER=0
export NAVRL_APP_TEXTURE_STD=0
export NAVRL_APP_MOTION_BLUR=0
export NAVRL_CAM_MOUNT_ROT_DEG=0
export NAVRL_CAM_MOUNT_TRANS_M=0
export NAVRL_CAM_FOV_SCALE_ERR=0
export NAVRL_DENSITY_FINAL=205
export NAVRL_DENSITY_WARMUP=1000
export NAVRL_LEARNING_RATE=1.5e-5
export NAVRL_SAVE_FREQUENCY=250
export NAVRL_GEOFENCE_NOISE_STD_M=0
export NAVRL_GEOFENCE_DROPOUT=0
if [[ "${ARM}" == "geofence" ]]; then
    export NAVRL_GEOFENCE_ACTOR=1
else
    export NAVRL_GEOFENCE_ACTOR=0
fi
export AERIAL_RUN_TAG="v2-ref5in-active-search-${ARM}-s197"
export TRAIN_SESSION_LOG="train_session_logs/active_search_${ARM}_$(date +%y%m%d_%H%M%S).log"
export TRAIN_LIVE_LOG="train_session_logs/current_active_search_${ARM}.log"
export NAVRL_V2_CONTRACT_PREFLIGHT_ONLY="${AS_PREFLIGHT}"

if [[ "${AS_PREFLIGHT}" == "1" ]]; then
    export NAVRL_TRAINING_SOURCE_MANIFEST=/preflight/source_manifest.json
    export NAVRL_TRAINING_SOURCE_MANIFEST_SHA256="$(printf '0%.0s' {1..64})"
    export NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT=0
    export NAVRL_REQUIRE_CLEAN_TRAINING_SOURCE=0
else
    RECEIPT_ROOT="train_source_receipts/active_search_${ARM}_s${SEED}_$(date +%y%m%d_%H%M%S)_$$"
    RECEIPT_JSON="$(${PYTHON} "${REPO_ROOT}/tools/create_navrl_source_bundle.py" \
        create --output "${RECEIPT_ROOT}" --require-clean)"
    export NAVRL_TRAINING_SOURCE_MANIFEST="$(${PYTHON} -c 'import json,sys; print(json.loads(sys.argv[1])["manifest"])' "${RECEIPT_JSON}")"
    export NAVRL_TRAINING_SOURCE_MANIFEST_SHA256="$(${PYTHON} -c 'import json,sys; print(json.loads(sys.argv[1])["manifest_sha256"])' "${RECEIPT_JSON}")"
    export NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT=1
    export NAVRL_REQUIRE_CLEAN_TRAINING_SOURCE=1
fi

echo "[active-search-${ARM}] prereg=docs/preregistration_active_search_geofence_2026-08-21.md"
echo "[active-search-${ARM}] fresh=1 seed=${SEED} epochs=${MAX_EPOCHS} robot=${NAVRL_ROBOT} geofence=${NAVRL_GEOFENCE_ACTOR}"
echo "[active-search-${ARM}] import_root=${ACTUAL_IMPORT}"
echo "[active-search-${ARM}] camera=20m reward/horizon/speed/tilt/density=P1c-frozen noise=0 dropout=0"
exec ./train_navrl_v2_search.sh
