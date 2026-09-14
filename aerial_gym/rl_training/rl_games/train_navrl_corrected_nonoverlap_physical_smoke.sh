#!/usr/bin/env bash
# Fresh 500-epoch learning-viability smoke for the corrected non-overlap environment.
#
# This is deliberately a route-off baseline.  It isolates the obstacle-placement correction from
# the failed global route mechanisms; it neither authorises nor claims a routed physical target.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if (( $# != 0 )); then
    echo "[nonoverlap-smoke] no CLI arguments are accepted." >&2
    exit 2
fi
if [[ -n "${CKPT:-}" || -n "${CHECKPOINT:-}" ]]; then
    echo "[nonoverlap-smoke] refusing checkpoint/warm-start; corrected geometry is fresh-only." >&2
    exit 2
fi

SMOKE_PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
SMOKE_PREFLIGHT="${CORRECTED_NONOVERLAP_SMOKE_PREFLIGHT_ONLY:-0}"
while IFS= read -r name; do
    case "${name}" in
        NAVRL_*|AERIAL_RUN_TAG|AERIAL_GYM_SIM_NAME|TRAIN_SESSION_LOG|TRAIN_LIVE_LOG|MAX_EPOCHS|SEED|NUM_ENVS|FILE|TASK|GPU4GB|CKPT|CHECKPOINT|ALLOW_CONCURRENT|PYTORCH_CUDA_ALLOC_CONF)
            unset "${name}"
            ;;
    esac
done < <(compgen -v)

export PYTHON="${SMOKE_PYTHON}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(git rev-parse --show-toplevel)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NAVRL_REQUIRE_SOURCE_ROOT="${REPO_ROOT}"
EXPECTED_IMPORT="$(realpath "${REPO_ROOT}/aerial_gym/__init__.py")"
ACTUAL_IMPORT="$(${PYTHON} -c 'import importlib.util,os; print(os.path.realpath(importlib.util.find_spec("aerial_gym").origin))')"
if [[ "${ACTUAL_IMPORT}" != "${EXPECTED_IMPORT}" ]]; then
    echo "[nonoverlap-smoke] aerial_gym import escaped worktree: ${ACTUAL_IMPORT}" >&2
    exit 2
fi

if [[ "${SMOKE_PREFLIGHT}" != "1" ]]; then
    RUNTIME_DIRTY="$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all -- \
        aerial_gym resources/robots tools/create_navrl_source_bundle.py)"
    if [[ -n "${RUNTIME_DIRTY}" ]]; then
        echo "[nonoverlap-smoke] refusing dirty runtime sources; commit the audited runtime first." >&2
        printf '%s\n' "${RUNTIME_DIRTY}" | sed -n '1,20p' >&2
        exit 2
    fi
fi

# Frozen one-axis smoke contract.  The 0.3..1.25 m/s route-off envelope passed the preceding
# physical gate at 70 bars.  The failed 1.5 m/s and global-route mechanisms are not smuggled in.
export MAX_EPOCHS=500
export SEED=907
export NAVRL_V2_PROFILE=main
export NAVRL_TARGET_ROUTE_MODE=off
export NAVRL_DENSITY_CURRICULUM=0
export NAVRL_NUM_BARS=70
export NAVRL_DENSITY_START=70
export NAVRL_DENSITY_FINAL=70
export NAVRL_TARGET_SPEED_FINAL=1.25
export NAVRL_TARGET_SPEED_RAMP_EPOCHS=1
export NAVRL_LEARNING_RATE=1.5e-5
export NAVRL_SAVE_FREQUENCY=100
export NAVRL_SPEED_GOVERNOR=off
export AERIAL_RUN_TAG=corrected-nonoverlap-physical-off-smoke-s907
export TRAIN_SESSION_LOG="train_session_logs/corrected_nonoverlap_physical_off_smoke_$(date +%y%m%d_%H%M%S).log"
export TRAIN_LIVE_LOG=train_session_logs/current_corrected_nonoverlap_physical_off_smoke.log
export NAVRL_TARGET_CONTRACT_PREFLIGHT_ONLY="${SMOKE_PREFLIGHT}"

if [[ "${SMOKE_PREFLIGHT}" == "1" ]]; then
    export NAVRL_TRAINING_SOURCE_MANIFEST=/preflight/corrected_nonoverlap_source_manifest.json
    export NAVRL_TRAINING_SOURCE_MANIFEST_SHA256="$(printf '0%.0s' {1..64})"
    export NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT=0
    export NAVRL_REQUIRE_CLEAN_TRAINING_SOURCE=0
else
    RECEIPT_ROOT="train_source_receipts/corrected_nonoverlap_physical_off_s${SEED}_$(date +%y%m%d_%H%M%S)_$$"
    RECEIPT_JSON="$(${PYTHON} "${REPO_ROOT}/tools/create_navrl_source_bundle.py" \
        create --output "${RECEIPT_ROOT}" --require-clean)"
    export NAVRL_TRAINING_SOURCE_MANIFEST="$(${PYTHON} -c 'import json,sys; print(json.loads(sys.argv[1])["manifest"])' "${RECEIPT_JSON}")"
    export NAVRL_TRAINING_SOURCE_MANIFEST_SHA256="$(${PYTHON} -c 'import json,sys; print(json.loads(sys.argv[1])["manifest_sha256"])' "${RECEIPT_JSON}")"
    export NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT=1
    export NAVRL_REQUIRE_CLEAN_TRAINING_SOURCE=1
fi

echo "[nonoverlap-smoke] scope=route-off corrected-geometry learning viability; routed claim=forbidden"
echo "[nonoverlap-smoke] fresh=1 seed=${SEED} epochs=${MAX_EPOCHS} bars=70 fixed speed=U[0.3,1.25]@1"
echo "[nonoverlap-smoke] placement=footprint_clearance surface=0.45m overlap_fallback=off route=off"
echo "[nonoverlap-smoke] source_root=${REPO_ROOT} import=${ACTUAL_IMPORT} receipt_required=${NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT}"
exec "${SCRIPT_DIR}/train_navrl_physical_fresh.sh"
