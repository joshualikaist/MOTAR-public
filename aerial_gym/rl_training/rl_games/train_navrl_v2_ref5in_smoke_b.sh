#!/usr/bin/env bash
# Corrective ref5in learning-viability smoke after P1a's single KL rollback.
#
# P1a demonstrated learning but failed its zero-rollback gate at epoch 432 and stopped one
# distance-curriculum promotion short.  This fresh run changes only the initial PPO learning rate
# (3e-5 -> the safety-selected 1.5e-5) and the engineering budget (500 -> 750 epochs).  It is not a
# performance comparison and may not resume P1a.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if (( $# != 0 )); then
    echo "[ref5in-smoke-b] no CLI arguments are accepted." >&2
    exit 2
fi
if [[ -n "${CKPT:-}" ]]; then
    echo "[ref5in-smoke-b] refusing inherited CKPT: ${CKPT}" >&2
    exit 2
fi

SMOKE_PYTHON="${PYTHON:-}"
SMOKE_PREFLIGHT="${REF5IN_PREFLIGHT_ONLY:-0}"
SMOKE_ALLOW_DIRTY="${REF5IN_ALLOW_DIRTY_SMOKE:-0}"
while IFS= read -r name; do
    case "${name}" in
        NAVRL_*|AERIAL_RUN_TAG|AERIAL_GYM_SIM_NAME|TRAIN_SESSION_LOG|TRAIN_LIVE_LOG|MAX_EPOCHS|SEED|NUM_ENVS|FILE|TASK|GPU4GB|CKPT|ALLOW_CONCURRENT)
            unset "${name}"
            ;;
    esac
done < <(compgen -v)
if [[ -n "${SMOKE_PYTHON}" ]]; then
    export PYTHON="${SMOKE_PYTHON}"
elif [[ -x /home/fair/miniconda3/envs/aerialgym/bin/python ]]; then
    export PYTHON=/home/fair/miniconda3/envs/aerialgym/bin/python
else
    export PYTHON=python
fi
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONNOUSERSITE=1

REPO_ROOT="$(git rev-parse --show-toplevel)"
if [[ "${SMOKE_PREFLIGHT}" != "1" && "${SMOKE_ALLOW_DIRTY}" != "1" ]]; then
    RUNTIME_DIRTY="$(
        git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all -- \
            aerial_gym resources/robots tools/create_navrl_source_bundle.py
    )"
    if [[ -n "${RUNTIME_DIRTY}" ]]; then
        echo "[ref5in-smoke-b] refusing dirty runtime sources; commit the audited runtime first." >&2
        printf '%s\n' "${RUNTIME_DIRTY}" | sed -n '1,20p' >&2
        echo "[ref5in-smoke-b] non-publishable wiring only: REF5IN_ALLOW_DIRTY_SMOKE=1" >&2
        exit 2
    fi
fi

export MAX_EPOCHS=750
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
export AERIAL_RUN_TAG=v2-ref5in-smoke-b-s197
export TRAIN_SESSION_LOG="train_session_logs/ref5in_smoke_b_$(date +%y%m%d_%H%M%S).log"
export TRAIN_LIVE_LOG=train_session_logs/current_ref5in_smoke_b.log
export NAVRL_V2_CONTRACT_PREFLIGHT_ONLY="${SMOKE_PREFLIGHT}"

if [[ "${SMOKE_PREFLIGHT}" == "1" ]]; then
    export NAVRL_TRAINING_SOURCE_MANIFEST=/preflight/source_manifest.json
    export NAVRL_TRAINING_SOURCE_MANIFEST_SHA256="$(printf '0%.0s' {1..64})"
    export NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT=0
    export NAVRL_REQUIRE_CLEAN_TRAINING_SOURCE=0
else
    RECEIPT_ROOT="train_source_receipts/ref5in_smoke_b_s${SEED}_$(date +%y%m%d_%H%M%S)_$$"
    CREATE_ARGS=(create --output "${RECEIPT_ROOT}")
    if [[ "${SMOKE_ALLOW_DIRTY}" != "1" ]]; then
        CREATE_ARGS+=(--require-clean)
    fi
    RECEIPT_JSON="$(${PYTHON} "${REPO_ROOT}/tools/create_navrl_source_bundle.py" "${CREATE_ARGS[@]}")"
    export NAVRL_TRAINING_SOURCE_MANIFEST="$(${PYTHON} -c 'import json,sys; print(json.loads(sys.argv[1])["manifest"])' "${RECEIPT_JSON}")"
    export NAVRL_TRAINING_SOURCE_MANIFEST_SHA256="$(${PYTHON} -c 'import json,sys; print(json.loads(sys.argv[1])["manifest_sha256"])' "${RECEIPT_JSON}")"
    export NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT=1
    export NAVRL_REQUIRE_CLEAN_TRAINING_SOURCE="$([[ "${SMOKE_ALLOW_DIRTY}" == "1" ]] && echo 0 || echo 1)"
fi

echo "[ref5in-smoke-b] scope=corrective learning-viability gate (NO performance/hardware claim)"
echo "[ref5in-smoke-b] fresh=1 robot=${NAVRL_ROBOT} seed=${SEED} epochs=${MAX_EPOCHS} bars=70"
echo "[ref5in-smoke-b] lr=${NAVRL_LEARNING_RATE} yaw=${NAVRL_YAW_RATE_MAX} tilt=${NAVRL_MAX_TILT_DEG} checkpoint_every=${NAVRL_SAVE_FREQUENCY}"
echo "[ref5in-smoke-b] source_manifest=${NAVRL_TRAINING_SOURCE_MANIFEST} sha=${NAVRL_TRAINING_SOURCE_MANIFEST_SHA256:0:12}"

exec ./train_navrl_v2_search.sh
