#!/usr/bin/env bash
# Fresh route-off 70->205 curriculum after the corrected non-overlap smoke passed.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if (( $# != 0 )); then
    echo "[nonoverlap-curriculum] no CLI arguments are accepted." >&2
    exit 2
fi
if [[ -n "${CKPT:-}" || -n "${CHECKPOINT:-}" ]]; then
    echo "[nonoverlap-curriculum] refusing checkpoint/warm-start; this lineage is fresh-only." >&2
    exit 2
fi

RUN_PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
RUN_PREFLIGHT="${CORRECTED_NONOVERLAP_CURRICULUM_PREFLIGHT_ONLY:-0}"
while IFS= read -r name; do
    case "${name}" in
        NAVRL_*|AERIAL_RUN_TAG|AERIAL_GYM_SIM_NAME|TRAIN_SESSION_LOG|TRAIN_LIVE_LOG|MAX_EPOCHS|SEED|NUM_ENVS|FILE|TASK|GPU4GB|CKPT|CHECKPOINT|ALLOW_CONCURRENT|PYTORCH_CUDA_ALLOC_CONF)
            unset "${name}"
            ;;
    esac
done < <(compgen -v)

export PYTHON="${RUN_PYTHON}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO_ROOT="$(git rev-parse --show-toplevel)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NAVRL_REQUIRE_SOURCE_ROOT="${REPO_ROOT}"
EXPECTED_IMPORT="$(realpath "${REPO_ROOT}/aerial_gym/__init__.py")"
ACTUAL_IMPORT="$(${PYTHON} -c 'import importlib.util,os; print(os.path.realpath(importlib.util.find_spec("aerial_gym").origin))')"
if [[ "${ACTUAL_IMPORT}" != "${EXPECTED_IMPORT}" ]]; then
    echo "[nonoverlap-curriculum] aerial_gym import escaped worktree: ${ACTUAL_IMPORT}" >&2
    exit 2
fi
if [[ "${RUN_PREFLIGHT}" != "1" ]]; then
    RUNTIME_DIRTY="$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all -- \
        aerial_gym resources/robots tools/create_navrl_source_bundle.py)"
    if [[ -n "${RUNTIME_DIRTY}" ]]; then
        echo "[nonoverlap-curriculum] refusing dirty runtime sources; commit first." >&2
        printf '%s\n' "${RUNTIME_DIRTY}" | sed -n '1,20p' >&2
        exit 2
    fi
fi

export MAX_EPOCHS=30000
export SEED=911
export NAVRL_V2_PROFILE=main
export NAVRL_TARGET_ROUTE_MODE=off
export NAVRL_DENSITY_CURRICULUM=1
export NAVRL_DENSITY_START=70
export NAVRL_DENSITY_FINAL=205
export NAVRL_DENSITY_THRESHOLD_START=0.80
export NAVRL_DENSITY_THRESHOLD_END=0.70
export NAVRL_DENSITY_THRESHOLD_SCHEDULE=70:0.82,85:0.77,100:0.72,115:0.70
export NAVRL_DENSITY_WARMUP=1000
export NAVRL_DENSITY_CHECK_EPS=16384
export NAVRL_DENSITY_MIN_EPOCHS=1000
export NAVRL_TARGET_SPEED_FINAL=1.25
export NAVRL_TARGET_SPEED_RAMP_EPOCHS=1
export NAVRL_LEARNING_RATE=1.5e-5
export NAVRL_SAVE_FREQUENCY=250
export NAVRL_SPEED_GOVERNOR=off
export AERIAL_RUN_TAG=corrected-nonoverlap-physical-off-curriculum-s911
export TRAIN_SESSION_LOG="train_session_logs/corrected_nonoverlap_physical_off_curriculum_$(date +%y%m%d_%H%M%S).log"
export TRAIN_LIVE_LOG=train_session_logs/current_corrected_nonoverlap_physical_off_curriculum.log
export NAVRL_TARGET_CONTRACT_PREFLIGHT_ONLY="${RUN_PREFLIGHT}"

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
    export NAVRL_TRAINING_SOURCE_MANIFEST=/preflight/corrected_nonoverlap_curriculum_source.json
    export NAVRL_TRAINING_SOURCE_MANIFEST_SHA256="$(printf '0%.0s' {1..64})"
    export NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT=0
    export NAVRL_REQUIRE_CLEAN_TRAINING_SOURCE=0
else
    RECEIPT_ROOT="train_source_receipts/corrected_nonoverlap_curriculum_s${SEED}_$(date +%y%m%d_%H%M%S)_$$"
    RECEIPT_JSON="$(${PYTHON} "${REPO_ROOT}/tools/create_navrl_source_bundle.py" \
        create --output "${RECEIPT_ROOT}" --require-clean)"
    export NAVRL_TRAINING_SOURCE_MANIFEST="$(${PYTHON} -c 'import json,sys; print(json.loads(sys.argv[1])["manifest"])' "${RECEIPT_JSON}")"
    export NAVRL_TRAINING_SOURCE_MANIFEST_SHA256="$(${PYTHON} -c 'import json,sys; print(json.loads(sys.argv[1])["manifest_sha256"])' "${RECEIPT_JSON}")"
    export NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT=1
    export NAVRL_REQUIRE_CLEAN_TRAINING_SOURCE=1
fi

echo "[nonoverlap-curriculum] scope=route-off corrected-geometry 70->205 baseline; routed claim=forbidden"
echo "[nonoverlap-curriculum] fresh=1 seed=${SEED} epochs=${MAX_EPOCHS} density=70:15:205 dwell=1000 evidence=16384"
echo "[nonoverlap-curriculum] placement=footprint_clearance surface=0.45m overlap_fallback=off speed=U[0.3,1.25]@1"
echo "[nonoverlap-curriculum] source_root=${REPO_ROOT} import=${ACTUAL_IMPORT} receipt_required=${NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT}"
exec "${SCRIPT_DIR}/train_navrl_physical_fresh.sh"
