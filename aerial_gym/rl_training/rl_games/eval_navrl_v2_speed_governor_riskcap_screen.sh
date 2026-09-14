#!/usr/bin/env bash
# Preregistered R2b screen: frozen ep24000, unseen seed44, off versus one riskcap candidate.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if (( $# > 1 )); then
    echo "usage: $0 [ep24000-checkpoint]" >&2
    exit 2
fi

PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PYTHON
SOURCE="${1:-runs/ppo_260802_0020_navrl_v2-recover-curriculum-continue-s1/nn/last_gen_ppo_ep_24000_rew_44.73549.pth}"
SOURCE_SHA="82f7978b42d9d9e95adcc638a40ae85fb3736fd897ae39cb8aa8333be39cf23f"
RESULT_ROOT="../../../results/navrl_v2_ep24000_riskcap_seed44_screen"

if [[ ! -f "${SOURCE}" ]]; then
    echo "[riskcap-screen] source checkpoint missing: ${SOURCE}" >&2
    exit 2
fi
if [[ -e "${RESULT_ROOT}" ]]; then
    echo "[riskcap-screen] refusing to overwrite ${RESULT_ROOT}" >&2
    exit 2
fi
ACTUAL_SHA="$(${PYTHON} - "${SOURCE}" <<'PY'
import hashlib
from pathlib import Path
import sys
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
if [[ "${ACTUAL_SHA}" != "${SOURCE_SHA}" ]]; then
    echo "[riskcap-screen] source SHA mismatch: ${ACTUAL_SHA}" >&2
    exit 2
fi

export NAVRL_V2_DENSITIES=205
export NAVRL_SEED=44
export NAVRL_V2_ACTION_MODE=deterministic
export NAVRL_EVAL_REFLECTION_MODE=original
export NAVRL_SPEED_GOVERNOR_FIXED_MPS=2.0
export NAVRL_SPEED_GOVERNOR_FREE_MPS=3.53553390593
export NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M=0.45
export NAVRL_SPEED_GOVERNOR_MARGIN_M=0.45
export NAVRL_SPEED_GOVERNOR_SLOW_M=3.0
export NAVRL_SPEED_GOVERNOR_RELEASE_M=5.0
export NAVRL_SPEED_GOVERNOR_TTC_S=1.2
export NAVRL_SPEED_GOVERNOR_BRAKE_MPS2=2.9608856678
export NAVRL_SPEED_GOVERNOR_REACTION_S=0.1

run_cell() {
    local tag="$1"
    local mode="$2"
    echo "[riskcap-screen] cell=${tag} mode=${mode} seed=44"
    NAVRL_SPEED_GOVERNOR="${mode}" \
    NAVRL_V2_RESULT_DIR="${RESULT_ROOT}/${tag}" \
        ./eval_navrl_v2_density_sweep.sh "${SOURCE}" 2049
}

run_cell off off
run_cell riskcap riskcap
if [[ "${NAVRL_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "[riskcap-screen] PREFLIGHT PASS (two cells; no results written)"
    exit 0
fi
"${PYTHON}" ../../../tools/summarize_navrl_v2_riskcap.py "${RESULT_ROOT}"
