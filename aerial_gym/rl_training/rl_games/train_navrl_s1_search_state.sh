#!/usr/bin/env bash
# Fresh-only S1 explicit blind-search state arm.
# Prereg: docs/preregistration_s1_blind_search_state_2026-09-03.md
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if (( $# != 1 )); then
    echo "[s1-train] usage: $0 <off|geofence|coverage|belief>" >&2
    exit 2
fi
ARM="$1"
case "${ARM}" in
    off|geofence|coverage|belief) ;;
    *)
        echo "[s1-train] invalid arm: ${ARM}; expected off|geofence|coverage|belief" >&2
        exit 2
        ;;
esac
if [[ -n "${CKPT:-}" || -n "${CHECKPOINT:-}" ]]; then
    echo "[s1-train] refusing checkpoint/warm-start; every S1 arm is fresh-only." >&2
    exit 2
fi

S1_PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
S1_PREFLIGHT="${NAVRL_S1_SEARCH_PREFLIGHT_ONLY:-0}"
while IFS= read -r name; do
    case "${name}" in
        NAVRL_*|AERIAL_RUN_TAG|AERIAL_GYM_SIM_NAME|TRAIN_SESSION_LOG|TRAIN_LIVE_LOG|MAX_EPOCHS|SEED|NUM_ENVS|FILE|TASK|GPU4GB|CKPT|CHECKPOINT|ALLOW_CONCURRENT|PYTORCH_CUDA_ALLOC_CONF)
            unset "${name}"
            ;;
    esac
done < <(compgen -v)

export PYTHON="${S1_PYTHON}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(git rev-parse --show-toplevel)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NAVRL_REQUIRE_SOURCE_ROOT="${REPO_ROOT}"
EXPECTED_IMPORT="$(realpath "${REPO_ROOT}/aerial_gym/__init__.py")"
ACTUAL_IMPORT="$(${PYTHON} -c 'import importlib.util,os; print(os.path.realpath(importlib.util.find_spec("aerial_gym").origin))')"
if [[ "${ACTUAL_IMPORT}" != "${EXPECTED_IMPORT}" ]]; then
    echo "[s1-train] aerial_gym import escaped worktree: ${ACTUAL_IMPORT}" >&2
    exit 2
fi

if [[ "${S1_PREFLIGHT}" != "1" ]]; then
    RUNTIME_DIRTY="$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all -- \
        aerial_gym resources/robots tools/create_navrl_source_bundle.py)"
    if [[ -n "${RUNTIME_DIRTY}" ]]; then
        echo "[s1-train] refusing dirty runtime sources; commit the audited runtime first." >&2
        printf '%s\n' "${RUNTIME_DIRTY}" | sed -n '1,20p' >&2
        exit 2
    fi
fi

export MAX_EPOCHS=3000
export SEED=919
export NAVRL_V2_PROFILE=main
export NAVRL_TARGET_ROUTE_MODE=off
export NAVRL_DENSITY_CURRICULUM=0
export NAVRL_NUM_BARS=70
export NAVRL_DENSITY_START=70
export NAVRL_DENSITY_FINAL=70
export NAVRL_TARGET_SPEED_FINAL=1.25
export NAVRL_TARGET_SPEED_RAMP_EPOCHS=1
export NAVRL_LEARNING_RATE=1.5e-5
export NAVRL_SAVE_FREQUENCY=250
export NAVRL_SPEED_GOVERNOR=off
export NAVRL_CORRIDOR_TOKENS=0
export NAVRL_SEARCH_STATE="${ARM}"
export NAVRL_SEARCH_STATE_FORCE_INVALID=0
export NAVRL_GEOFENCE_ACTOR=0
if [[ "${ARM}" != "off" ]]; then
    export NAVRL_GEOFENCE_ACTOR=1
fi
export NAVRL_GEOFENCE_NOISE_STD_M=0
export NAVRL_GEOFENCE_DROPOUT=0
export AERIAL_RUN_TAG="s1-search-${ARM}-s919"
export TRAIN_SESSION_LOG="train_session_logs/s1_search_${ARM}_$(date +%y%m%d_%H%M%S).log"
export TRAIN_LIVE_LOG="train_session_logs/current_s1_search_${ARM}.log"
export NAVRL_TARGET_CONTRACT_PREFLIGHT_ONLY="${S1_PREFLIGHT}"

if [[ "${S1_PREFLIGHT}" == "1" ]]; then
    export NAVRL_TRAINING_SOURCE_MANIFEST="/preflight/s1_search_${ARM}_source_manifest.json"
    export NAVRL_TRAINING_SOURCE_MANIFEST_SHA256="$(printf '0%.0s' {1..64})"
    export NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT=0
    export NAVRL_REQUIRE_CLEAN_TRAINING_SOURCE=0
else
    RECEIPT_ROOT="train_source_receipts/s1_search_${ARM}_s${SEED}_$(date +%y%m%d_%H%M%S)_$$"
    RECEIPT_JSON="$(${PYTHON} "${REPO_ROOT}/tools/create_navrl_source_bundle.py" \
        create --output "${RECEIPT_ROOT}" --require-clean)"
    export NAVRL_TRAINING_SOURCE_MANIFEST="$(${PYTHON} -c 'import json,sys; print(json.loads(sys.argv[1])["manifest"])' "${RECEIPT_JSON}")"
    export NAVRL_TRAINING_SOURCE_MANIFEST_SHA256="$(${PYTHON} -c 'import json,sys; print(json.loads(sys.argv[1])["manifest_sha256"])' "${RECEIPT_JSON}")"
    export NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT=1
    export NAVRL_REQUIRE_CLEAN_TRAINING_SOURCE=1
fi

echo "[s1-train] arm=${ARM} search=${NAVRL_SEARCH_STATE} geofence=${NAVRL_GEOFENCE_ACTOR} fresh=1"
echo "[s1-train] seed=${SEED} epochs=${MAX_EPOCHS} bars=${NAVRL_NUM_BARS} density_curriculum=${NAVRL_DENSITY_CURRICULUM}"
echo "[s1-train] route=${NAVRL_TARGET_ROUTE_MODE} speed=U[0.3,${NAVRL_TARGET_SPEED_FINAL}]@${NAVRL_TARGET_SPEED_RAMP_EPOCHS} governor=${NAVRL_SPEED_GOVERNOR} save=${NAVRL_SAVE_FREQUENCY}"
echo "[s1-train] run_tag=${AERIAL_RUN_TAG} session_log=${TRAIN_SESSION_LOG} live_log=${TRAIN_LIVE_LOG}"
echo "[s1-train] source_root=${REPO_ROOT} import=${ACTUAL_IMPORT} receipt_required=${NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT}"
if [[ "${S1_PREFLIGHT}" == "1" ]]; then
    exit 0
fi
exec "${SCRIPT_DIR}/train_navrl_physical_fresh.sh"
