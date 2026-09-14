#!/usr/bin/env bash
# P10 readaptation: continue the frozen ep25000 riskcap policy for exactly 1,000 epochs with the
# P9 empirical perception-error injector active in the training loop.
# Contract: docs/plans/perception_p9_p10_execution_2026-09-09.md "P10 adaptation and held-out
# evaluation", registered before P9 goodness-of-fit, test opening, training or PPO evaluation.
#
# Usage: bash train_navrl_v2_p10_empirical_readapt.sh
#        P10_SEED=<n> to name the single reported training seed (default 811).
#
# Everything except the perception-error arm and the checkpoint is the ep25000 contract verbatim:
# riskcap with its screened parameters, 205 bars fixed, LR 5e-6, cluster_sector, 898-D observation.
# The training seed must not appear in the evaluation seeds.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if (( $# != 0 )); then
    echo "[p10] no CLI arguments are accepted; set P10_SEED." >&2
    exit 2
fi

P10_PREFLIGHT="${P10_PREFLIGHT_ONLY:-0}"
P10_PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
while IFS= read -r name; do
    case "${name}" in
        NAVRL_*|AERIAL_RUN_TAG|AERIAL_GYM_SIM_NAME|TRAIN_SESSION_LOG|TRAIN_LIVE_LOG|MAX_EPOCHS|SEED|NUM_ENVS|FILE|TASK|GPU4GB|CKPT|ALLOW_CONCURRENT|PYTORCH_CUDA_ALLOC_CONF)
            unset "${name}"
            ;;
    esac
done < <(compgen -v)
export PYTHON="${P10_PYTHON}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONNOUSERSITE=1
# 128-env PhysX plus the compiled Transformer sits close to the 8 GB board limit; this changes
# allocator segmentation only, never the model, batch or task contract.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(git rev-parse --show-toplevel)"
CKPT_PATH="${REPO_ROOT}/aerial_gym/rl_training/rl_games/runs/ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1/nn/last_gen_ppo_ep_25000_rew_39.742134.pth"
EXPECTED_CKPT_SHA=f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40
P9_MODEL="${REPO_ROOT}/results/perception_p9_2026-09-09/p9_error_injector.json"

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "[p10] missing frozen ep25000 checkpoint: ${CKPT_PATH}" >&2
    exit 2
fi
if [[ ! -f "${P9_MODEL}" ]]; then
    echo "[p10] missing P9 injector model: ${P9_MODEL}" >&2
    exit 2
fi

sha_of() {
    "${PYTHON}" - "$1" <<'PY'
import hashlib, sys
digest = hashlib.sha256()
with open(sys.argv[1], "rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
}

ACTUAL_CKPT_SHA="$(sha_of "${CKPT_PATH}")"
if [[ "${ACTUAL_CKPT_SHA}" != "${EXPECTED_CKPT_SHA}" ]]; then
    echo "[p10] ep25000 checkpoint SHA mismatch: ${ACTUAL_CKPT_SHA}" >&2
    exit 2
fi
P9_MODEL_SHA="$(sha_of "${P9_MODEL}")"

# The P9 gate must have passed before any policy sees this model.
"${PYTHON}" - "${REPO_ROOT}/results/perception_p9_2026-09-09/goodness_of_fit.json" <<'PY'
import json, sys
report = json.loads(open(sys.argv[1]).read())
if not report.get("passed"):
    raise SystemExit("[p10] P9 goodness-of-fit did not pass; refusing to train")
if report.get("test_used"):
    raise SystemExit("[p10] P9 report claims test usage; refusing to train")
PY

if [[ "${P10_PREFLIGHT}" != "1" ]]; then
    RUNTIME_DIRTY="$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all -- \
        aerial_gym resources/robots tools/create_navrl_source_bundle.py)"
    if [[ -n "${RUNTIME_DIRTY}" ]]; then
        echo "[p10] refusing dirty runtime sources; commit first." >&2
        printf '%s\n' "${RUNTIME_DIRTY}" | sed -n '1,20p' >&2
        exit 2
    fi
fi

export CKPT="${CKPT_PATH}"
export NAVRL_V2_ALLOW_RESUME=1
export MAX_EPOCHS=26000                     # 25,000 source + exactly 1,000 adaptation epochs
export SEED="${P10_SEED:-811}"              # the single reported training seed; not an eval seed
export NAVRL_V2_PROFILE=main
export NAVRL_OBSTACLE_SELECTOR=cluster_sector

# P9 empirical perception error: the only arm that differs from the source contract. The module
# refuses to run alongside any other noise hook, so an accidental mix cannot go unnoticed.
export NAVRL_P9_ERROR_MODEL="${P9_MODEL}"
export NAVRL_P9_EXPECTED_SHA256="${P9_MODEL_SHA}"
export NAVRL_P9_SEED="${P10_SEED:-811}"
export NAVRL_PERCEPTION_PERTURB=0

# riskcap, byte-identical to the screened parameters the ep25000 lineage was adapted under.
export NAVRL_SPEED_GOVERNOR=riskcap
export NAVRL_SPEED_GOVERNOR_FIXED_MPS=2.0
export NAVRL_SPEED_GOVERNOR_FREE_MPS=3.53553390593
export NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M=0.45
export NAVRL_SPEED_GOVERNOR_MARGIN_M=0.45
export NAVRL_SPEED_GOVERNOR_SLOW_M=3.0
export NAVRL_SPEED_GOVERNOR_RELEASE_M=5.0
export NAVRL_SPEED_GOVERNOR_TTC_S=1.0
export NAVRL_SPEED_GOVERNOR_BRAKE_MPS2=2.0
export NAVRL_SPEED_GOVERNOR_REACTION_S=0.1
export NAVRL_SPEED_GOVERNOR_DIAG=1

# Fixed 205 bars, exactly as the source checkpoint was trained.
export NAVRL_DENSITY_CURRICULUM=0
export NAVRL_NUM_BARS=205
export NAVRL_DENSITY_START=205
export NAVRL_DENSITY_FINAL=205
export NAVRL_DENSITY_WARMUP=0
export NAVRL_DENSITY_MIN_EPOCHS=0
export NAVRL_RESET_DENSITY_WINDOW=1
export NAVRL_LEARNING_RATE=5e-6
export NAVRL_RESET_ACTOR_OPTIMIZER=0
export NAVRL_SAVE_FREQUENCY=250

export AERIAL_RUN_TAG="v2-p10-empirical-readapt-s${SEED}"
export TRAIN_SESSION_LOG="train_session_logs/p10_empirical_readapt_$(date +%y%m%d_%H%M%S).log"
export TRAIN_LIVE_LOG="train_session_logs/current_p10_empirical_readapt.log"
export NAVRL_V2_CONTRACT_PREFLIGHT_ONLY="${P10_PREFLIGHT}"

echo "[p10] source ckpt   ${ACTUAL_CKPT_SHA}"
echo "[p10] P9 model      ${P9_MODEL_SHA}"
echo "[p10] training seed ${SEED} (must not appear in evaluation seeds)"
echo "[p10] epochs        25000 -> ${MAX_EPOCHS}"

if [[ "${P10_PREFLIGHT}" == "1" ]]; then
    export NAVRL_TRAINING_SOURCE_MANIFEST=/preflight/source_manifest.json
    export NAVRL_TRAINING_SOURCE_MANIFEST_SHA256="$(printf '0%.0s' {1..64})"
fi

exec ./train_navrl_v2_search.sh --checkpoint "${CKPT}" --branch_run
