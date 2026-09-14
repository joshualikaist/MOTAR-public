#!/usr/bin/env bash
# Held-out density sweep of the operator-stopped seed-911 route-off curriculum.
# Prereg: docs/preregistration_corrected_nonoverlap_physical_off_heldout_eval_2026-09-02.md
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if (( $# != 0 )); then
    echo "[heldout] no CLI arguments are accepted." >&2
    exit 2
fi

RUN_PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
RUN_PREFLIGHT="${CORRECTED_NONOVERLAP_HELDOUT_PREFLIGHT_ONLY:-0}"
while IFS= read -r name; do
    case "${name}" in
        NAVRL_*|GPU4GB|CKPT|CHECKPOINT|SEED|ALLOW_CONCURRENT|PYTORCH_CUDA_ALLOC_CONF)
            unset "${name}"
            ;;
    esac
done < <(compgen -v)

export PYTHON="${RUN_PYTHON}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONNOUSERSITE=1
export GPU4GB=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(git rev-parse --show-toplevel)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NAVRL_REQUIRE_SOURCE_ROOT="${REPO_ROOT}"
EXPECTED_IMPORT="$(realpath "${REPO_ROOT}/aerial_gym/__init__.py")"
ACTUAL_IMPORT="$(${PYTHON} -c 'import importlib.util,os; print(os.path.realpath(importlib.util.find_spec("aerial_gym").origin))')"
if [[ "${ACTUAL_IMPORT}" != "${EXPECTED_IMPORT}" ]]; then
    echo "[heldout] aerial_gym import escaped worktree: ${ACTUAL_IMPORT}" >&2
    exit 2
fi

CKPT="${SCRIPT_DIR}/runs/ppo_260901_1431_navrl_corrected-nonoverlap-physical-off-curriculum-s911/nn/last_gen_ppo_ep_21750_rew_83.1572.pth"
EXPECTED_SHA256="541b36bdcabacf8bb14c6fbb0ad07054dd9735ad24777a3222655ba8ca9c8132"
RESULT_ROOT="${REPO_ROOT}/results/navrl_corrected_nonoverlap_physical_off_heldout_seed313"
DENSITIES="70 85 100 115 130 145"
SEED=313
GAMES=2049

if [[ "$(basename "${CKPT}")" == "gen_ppo.pth" || "${CKPT}" == *"/gen_ppo.pth" ]]; then
    echo "[heldout] gen_ppo.pth is forbidden; this evaluation uses last_gen_ppo_ep_21750 only." >&2
    exit 2
fi
if [[ ! -f "${CKPT}" ]]; then
    echo "[heldout] checkpoint not found: ${CKPT}" >&2
    exit 2
fi
ACTUAL_SHA256="$(sha256sum "${CKPT}" | awk '{print $1}')"
if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
    echo "[heldout] checkpoint SHA-256 mismatch: ${ACTUAL_SHA256}" >&2
    exit 2
fi

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
    echo "[heldout] seed=${SEED} ckpt=last_gen_ppo_ep_21750 densities=${DENSITIES} episodes=${GAMES} route=off placement=footprint_clearance speed_final=1.25 ramp=1 action=deterministic ood_205=0 gen_ppo=forbidden"
    exit 0
fi

if [[ -e "${RESULT_ROOT}" ]]; then
    echo "[heldout] refusing to overwrite existing result root: ${RESULT_ROOT}" >&2
    exit 2
fi

export NAVRL_V2_EVAL_CONTRACT=corrected_nonoverlap_physical_off
export NAVRL_V2_DENSITIES="${DENSITIES}"
export NAVRL_SEED="${SEED}"
export NAVRL_V2_ACTION_MODE=deterministic
export NAVRL_SPEED_GOVERNOR=off
export NAVRL_V2_RESULT_DIR="${RESULT_ROOT}"
export NAVRL_EVAL_REFLECTION_MODE=original
export NAVRL_PHYSICAL_GEOMETRY_VERSION=v2
export NAVRL_TARGET_BOX_XY_M=0.283

echo "[heldout] evaluation only; frozen last_gen_ppo_ep_21750; no training; no 205"
echo "[heldout] checkpoint=${CKPT} sha256=${EXPECTED_SHA256}"
echo "[heldout] seed=${SEED} densities=${DENSITIES} games=${GAMES}"

exec ./eval_navrl_v2_density_sweep.sh "${CKPT}" "${GAMES}"
