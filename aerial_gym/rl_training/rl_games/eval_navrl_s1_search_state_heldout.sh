#!/usr/bin/env bash
# Preregistered held-out evaluation for one S1 search-state terminal checkpoint.
# Usage: eval_navrl_s1_search_state_heldout.sh <off|geofence|coverage|belief> <checkpoint> [--masked]
set -euo pipefail
CALLER_PWD="${PWD}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if (( $# < 2 || $# > 3 )); then
    echo "[s1-eval] usage: $0 <off|geofence|coverage|belief> <last_gen_ppo_ep_3000_*.pth> [--masked]" >&2
    exit 2
fi
ARM="$1"
CKPT_INPUT="$2"
MASKED=0
case "${ARM}" in off|geofence|coverage|belief) ;; *)
    echo "[s1-eval] invalid arm: ${ARM}" >&2; exit 2 ;;
esac
if (( $# == 3 )); then
    if [[ "$3" != "--masked" ]]; then
        echo "[s1-eval] the only optional argument is --masked" >&2
        exit 2
    fi
    MASKED=1
fi
if [[ "${ARM}" == "off" && "${MASKED}" == "1" ]]; then
    echo "[s1-eval] off has no search-state mechanism to mask" >&2
    exit 2
fi
if [[ "${CKPT_INPUT}" == /* ]]; then
    S1_CHECKPOINT="${CKPT_INPUT}"
else
    S1_CHECKPOINT="${CALLER_PWD}/${CKPT_INPUT}"
fi
if [[ ! -f "${S1_CHECKPOINT}" ]]; then
    echo "[s1-eval] checkpoint not found: ${S1_CHECKPOINT}" >&2
    exit 2
fi
S1_CHECKPOINT="$(readlink -f -- "${S1_CHECKPOINT}")"
if [[ "$(basename "${S1_CHECKPOINT}")" == "gen_ppo.pth" ]]; then
    echo "[s1-eval] gen_ppo.pth is forbidden" >&2
    exit 2
fi
if [[ ! "$(basename "${S1_CHECKPOINT}")" =~ ^last_gen_ppo_ep_3000_rew_.*\.pth$ ]]; then
    echo "[s1-eval] expected an arm terminal last_gen_ppo_ep_3000_rew_*.pth checkpoint" >&2
    exit 2
fi

RUN_PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
RUN_PREFLIGHT="${NAVRL_S1_EVAL_PREFLIGHT_ONLY:-0}"
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

export NAVRL_SEARCH_STATE="${ARM}"
export NAVRL_SEARCH_STATE_FORCE_INVALID="${MASKED}"
export NAVRL_S1_SEARCH_TELEMETRY=1
export NAVRL_GEOFENCE_ACTOR=0
if [[ "${ARM}" != "off" ]]; then export NAVRL_GEOFENCE_ACTOR=1; fi
export NAVRL_GEOFENCE_NOISE_STD_M=0
export NAVRL_GEOFENCE_DROPOUT=0
export NAVRL_GEOFENCE_FORCE_INVALID=0

if ! "${PYTHON}" - "${S1_CHECKPOINT}" "${ARM}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
state = checkpoint.get("env_state") or {}
expected = sys.argv[2]
actual = str(state.get("cfg_search_state", "")).strip().lower()
if actual != expected:
    raise SystemExit(
        "[s1-eval] checkpoint search-state mismatch: checkpoint=%r requested=%r"
        % (actual, expected)
    )
if bool(state.get("cfg_search_state_force_invalid", False)):
    raise SystemExit("[s1-eval] training checkpoint was created with the evaluation mask enabled")
expected_geofence = expected != "off"
if bool(state.get("cfg_geofence_actor", False)) != expected_geofence:
    raise SystemExit("[s1-eval] checkpoint geofence/search-state contract mismatch")
PY
then
    exit 2
fi

SUFFIX="${ARM}"
if [[ "${MASKED}" == "1" ]]; then SUFFIX="${SUFFIX}_masked"; fi
export NAVRL_V2_EVAL_CONTRACT=corrected_nonoverlap_physical_off
export NAVRL_V2_DENSITIES="70 145"
export NAVRL_SEED=331
export NAVRL_V2_ACTION_MODE=deterministic
export NAVRL_SPEED_GOVERNOR=off
export NAVRL_V2_RESULT_DIR="${REPO_ROOT}/results/navrl_s1_search_state_seed331/${SUFFIX}"
export NAVRL_EVAL_REFLECTION_MODE=original
export NAVRL_PHYSICAL_GEOMETRY_VERSION=v2
export NAVRL_TARGET_BOX_XY_M=0.283

CKPT_SHA256="$(sha256sum "${S1_CHECKPOINT}" | awk '{print $1}')"
echo "[s1-eval] arm=${ARM} masked=${MASKED} seed=${NAVRL_SEED} densities=${NAVRL_V2_DENSITIES} episodes=2049"
echo "[s1-eval] checkpoint=$(basename "${S1_CHECKPOINT}") sha256=${CKPT_SHA256} gen_ppo=forbidden"
echo "[s1-eval] result_root=${NAVRL_V2_RESULT_DIR} action=deterministic governor=off"
if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
    exit 0
fi
exec "${SCRIPT_DIR}/eval_navrl_v2_density_sweep.sh" "${S1_CHECKPOINT}" 2049
