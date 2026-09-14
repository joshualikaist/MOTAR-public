#!/usr/bin/env bash
# D1 development probe: oversample the diagnosed 22.5..28 m band without changing the controller.
# This is not P3, not a publication run, and cannot turn the failed P2 cell into a PASS.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if (( $# != 0 )); then
    echo "[ref5in-d1] no CLI arguments are accepted." >&2
    exit 2
fi

D1_PREFLIGHT="${REF5IN_D1_PREFLIGHT_ONLY:-0}"
D1_PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
while IFS= read -r name; do
    case "${name}" in
        NAVRL_*|AERIAL_RUN_TAG|AERIAL_GYM_SIM_NAME|TRAIN_SESSION_LOG|TRAIN_LIVE_LOG|MAX_EPOCHS|SEED|NUM_ENVS|FILE|TASK|GPU4GB|CKPT|ALLOW_CONCURRENT|PYTORCH_CUDA_ALLOC_CONF)
            unset "${name}"
            ;;
    esac
done < <(compgen -v)
export PYTHON="${D1_PYTHON}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONNOUSERSITE=1
# The 128-env PhysX + compiled Transformer footprint sits close to the 8 GB board limit.  The
# first zero-epoch launch had 216 MiB reserved-but-unused and failed on a 130 MiB backward buffer.
# This allocator setting changes memory segmentation only, not the model, batch or task contract.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(git rev-parse --show-toplevel)"
CKPT="${REPO_ROOT}/aerial_gym/rl_training/rl_games/runs/ppo_260813_0540_navrl_v2-ref5in-smoke-c-s197/nn/last_gen_ppo_ep_900_rew_137.08087.pth"
EXPECTED_CKPT_SHA=f1670a1d74dd92cb00d6a58898e9cc1b96eb9cbe155d1e85812a345e7aaae6bf
if [[ ! -f "${CKPT}" ]]; then
    echo "[ref5in-d1] missing P1c checkpoint: ${CKPT}" >&2
    exit 2
fi
ACTUAL_CKPT_SHA="$(${PYTHON} - "${CKPT}" <<'PY'
import hashlib,sys
h=hashlib.sha256()
with open(sys.argv[1], 'rb') as f:
    for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
print(h.hexdigest())
PY
)"
if [[ "${ACTUAL_CKPT_SHA}" != "${EXPECTED_CKPT_SHA}" ]]; then
    echo "[ref5in-d1] P1c checkpoint SHA mismatch: ${ACTUAL_CKPT_SHA}" >&2
    exit 2
fi

if [[ "${D1_PREFLIGHT}" != "1" ]]; then
    "${PYTHON}" "${REPO_ROOT}/tools/run_navrl_ref5in_outcome_diagnostic_v2.py" verify
    RUNTIME_DIRTY="$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all -- \
        aerial_gym resources/robots tools/create_navrl_source_bundle.py)"
    if [[ -n "${RUNTIME_DIRTY}" ]]; then
        echo "[ref5in-d1] refusing dirty runtime sources; commit first." >&2
        printf '%s\n' "${RUNTIME_DIRTY}" | sed -n '1,20p' >&2
        exit 2
    fi
fi

export CKPT
export NAVRL_V2_ALLOW_RESUME=1
export MAX_EPOCHS=1900
export SEED=197
export NAVRL_V2_PROFILE=main
export NAVRL_ROBOT=navrl_ref5in_quad
export NAVRL_YAW_RATE_MAX=3.0
export NAVRL_MAX_TILT_DEG=45.0
export NAVRL_ALT_HOLD_VMAX=2.5
export NAVRL_SPEED_GOVERNOR=off
export NAVRL_PERCEPTION_PERTURB=0
export NAVRL_GENERAL_GOAL_DIST_MIN=22.5
export NAVRL_GENERAL_GOAL_DIST_MAX=28
export NAVRL_DENSITY_CURRICULUM=0
export NAVRL_NUM_BARS=70
export NAVRL_DENSITY_START=70
export NAVRL_DENSITY_FINAL=70
export NAVRL_LEARNING_RATE=1.5e-5
export NAVRL_SAVE_FREQUENCY=250
export AERIAL_RUN_TAG=v2-ref5in-d1-q3-adapt-s197
export TRAIN_SESSION_LOG="train_session_logs/ref5in_d1_q3_adapt_$(date +%y%m%d_%H%M%S).log"
export TRAIN_LIVE_LOG=train_session_logs/current_ref5in_d1_q3_adapt.log
export NAVRL_V2_CONTRACT_PREFLIGHT_ONLY="${D1_PREFLIGHT}"

if [[ "${D1_PREFLIGHT}" == "1" ]]; then
    export NAVRL_TRAINING_SOURCE_MANIFEST=/preflight/source_manifest.json
    export NAVRL_TRAINING_SOURCE_MANIFEST_SHA256="$(printf '0%.0s' {1..64})"
    export NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT=0
    export NAVRL_REQUIRE_CLEAN_TRAINING_SOURCE=0
else
    RECEIPT_ROOT="train_source_receipts/ref5in_d1_q3_adapt_s${SEED}_$(date +%y%m%d_%H%M%S)_$$"
    RECEIPT_JSON="$(${PYTHON} "${REPO_ROOT}/tools/create_navrl_source_bundle.py" \
        create --output "${RECEIPT_ROOT}" --require-clean)"
    export NAVRL_TRAINING_SOURCE_MANIFEST="$(${PYTHON} -c 'import json,sys; print(json.loads(sys.argv[1])["manifest"])' "${RECEIPT_JSON}")"
    export NAVRL_TRAINING_SOURCE_MANIFEST_SHA256="$(${PYTHON} -c 'import json,sys; print(json.loads(sys.argv[1])["manifest_sha256"])' "${RECEIPT_JSON}")"
    export NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT=1
    export NAVRL_REQUIRE_CLEAN_TRAINING_SOURCE=1
fi

echo "[ref5in-d1] scope=development adaptation probe (NO P2 override/P3/performance claim)"
echo "[ref5in-d1] source=ep900 -> terminal=1900 seed=197 robot=${NAVRL_ROBOT} bars=70"
echo "[ref5in-d1] applied_goal_range=${NAVRL_GENERAL_GOAL_DIST_MIN}..${NAVRL_GENERAL_GOAL_DIST_MAX}m mixed target"
echo "[ref5in-d1] policy/reward/governor unchanged lr=${NAVRL_LEARNING_RATE}"

exec ./train_navrl_v2_search.sh --checkpoint "${CKPT}" --branch_run
