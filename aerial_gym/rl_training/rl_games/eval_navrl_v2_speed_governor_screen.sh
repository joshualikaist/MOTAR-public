#!/usr/bin/env bash
# Preregistered R2 inference-only screen: one frozen policy, five horizontal speed layers.
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
CALIBRATION="../../../results/navrl_v2_speed_governor_braking.json"
RESULT_ROOT="../../../results/navrl_v2_ep24000_speed_governor_screen"

if [[ ! -f "${SOURCE}" || ! -f "${CALIBRATION}" ]]; then
    echo "[speedgov-screen] missing source checkpoint or braking calibration" >&2
    exit 2
fi
if [[ -e "${RESULT_ROOT}" ]]; then
    echo "[speedgov-screen] refusing to overwrite ${RESULT_ROOT}" >&2
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
    echo "[speedgov-screen] source SHA mismatch: ${ACTUAL_SHA}" >&2
    exit 2
fi

read -r TTC_S BRAKE_MPS2 REACTION_S <<< "$(${PYTHON} - "${CALIBRATION}" <<'PY'
import json
from pathlib import Path
import sys
d=json.loads(Path(sys.argv[1]).read_text())
r=d["recommended"]
print(r["ttc_s"], r["brake_mps2"], r["reaction_s"])
PY
)"

export NAVRL_V2_DENSITIES=205
export NAVRL_SEED=42
export NAVRL_V2_ACTION_MODE=deterministic
export NAVRL_EVAL_REFLECTION_MODE=original
export NAVRL_SPEED_GOVERNOR_FREE_MPS=3.53553390593
export NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M=0.45
export NAVRL_SPEED_GOVERNOR_MARGIN_M=0.45
export NAVRL_SPEED_GOVERNOR_SLOW_M=3.0
export NAVRL_SPEED_GOVERNOR_TTC_S="${TTC_S}"
export NAVRL_SPEED_GOVERNOR_BRAKE_MPS2="${BRAKE_MPS2}"
export NAVRL_SPEED_GOVERNOR_REACTION_S="${REACTION_S}"

run_cell() {
    local tag="$1"
    local mode="$2"
    local fixed="$3"
    echo "[speedgov-screen] cell=${tag} mode=${mode} fixed=${fixed}"
    NAVRL_SPEED_GOVERNOR="${mode}" \
    NAVRL_SPEED_GOVERNOR_FIXED_MPS="${fixed}" \
    NAVRL_V2_RESULT_DIR="${RESULT_ROOT}/${tag}" \
        ./eval_navrl_v2_density_sweep.sh "${SOURCE}" 2049
}

run_cell off off 2.0
run_cell fixed2p0 fixed 2.0
run_cell fixed1p5 fixed 1.5
run_cell clearance clearance 2.0
run_cell ttc ttc 2.0

if [[ "${NAVRL_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "[speedgov-screen] PREFLIGHT PASS (five cells; no results written)"
    exit 0
fi

"${PYTHON}" ../../../tools/summarize_navrl_v2_speed_governor.py "${RESULT_ROOT}"
