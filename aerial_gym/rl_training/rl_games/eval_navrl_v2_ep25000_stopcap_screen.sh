#!/usr/bin/env bash
# Preregistered stopcap screen: frozen ep25000, unseen seed49, five governor arms.
# Prereg: docs/prereg_2026-09-02_speed_governor_stopcap_screen.md (gates frozen before results).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if (( $# > 1 )); then
    echo "usage: $0 [ep25000-checkpoint]" >&2
    exit 2
fi

PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PYTHON
SOURCE="${1:-runs/ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1/nn/last_gen_ppo_ep_25000_rew_39.742134.pth}"
SOURCE_SHA="f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40"
RESULT_ROOT="../../../results/navrl_v2_ep25000_stopcap_seed49_screen"

if [[ ! -f "${SOURCE}" ]]; then
    echo "[stopcap-screen] source checkpoint missing: ${SOURCE}" >&2
    exit 2
fi
if [[ -e "${RESULT_ROOT}" ]]; then
    echo "[stopcap-screen] refusing to overwrite ${RESULT_ROOT}" >&2
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
    echo "[stopcap-screen] source SHA mismatch: ${ACTUAL_SHA}" >&2
    exit 2
fi

export NAVRL_V2_DENSITIES=205
export NAVRL_SEED=49
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
    echo "[stopcap-screen] cell=${tag} mode=${mode} seed=49"
    NAVRL_SPEED_GOVERNOR="${mode}" \
    NAVRL_V2_RESULT_DIR="${RESULT_ROOT}/${tag}" \
        ./eval_navrl_v2_density_sweep.sh "${SOURCE}" 2049
}

run_cell off off
run_cell fixed2p0 fixed
run_cell riskcap riskcap
run_cell stopcap stopcap
run_cell ttc ttc

if [[ "${NAVRL_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "[stopcap-screen] PREFLIGHT PASS (five cells; no results written)"
    exit 0
fi

"${PYTHON}" ../../../tools/summarize_navrl_v2_stopcap_screen.py "${RESULT_ROOT}"
