#!/usr/bin/env bash
# v2 held-out density curve for the CURRENT frozen candidate (ep25000 + riskcap).
#
# The dashboard's v2 density curve is rebuilt from results/navrl_v2_ep24000_heldout, i.e. the
# PREVIOUS checkpoint with the governor off. That mismatch is currently disclosed in a label only.
# This measures the same densities under the frozen candidate so the curve describes the policy
# the rest of the page describes.
#
# Densities span the v2 curriculum: 130/160/190 trained, 205 the density it reached, 220
# generalisation -- the same grid the ep24000 sweep used, so the two are directly comparable.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PYTHON
POLICY="runs/ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1/nn/last_gen_ppo_ep_25000_rew_39.742134.pth"
POLICY_SHA="f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40"
RESULT_ROOT="../../../results/navrl_v2_density_curve_riskcap"
PREFLIGHT="${PREFLIGHT:-0}"

[[ -f "${POLICY}" ]] || { echo "[dens] policy missing" >&2; exit 2; }
if [[ -e "${RESULT_ROOT}" && "${PREFLIGHT}" != "1" ]]; then
    echo "[dens] refusing to overwrite ${RESULT_ROOT}" >&2; exit 2
fi
ACTUAL_SHA="$("${PYTHON}" - "${POLICY}" "${POLICY_SHA}" <<'PY'
import hashlib, sys
from pathlib import Path
p, expected = Path(sys.argv[1]), sys.argv[2]
actual = hashlib.sha256(p.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f"[dens] policy SHA mismatch: {actual}")
print(actual)
PY
)"

export NAVRL_V2_DENSITIES="130 160 190 205 220"
export NAVRL_V2_ACTION_MODE=deterministic
export NAVRL_EVAL_REFLECTION_MODE=original
export NAVRL_SPEED_GOVERNOR=riskcap
export NAVRL_SEED=47
export NAVRL_SPEED_GOVERNOR_FIXED_MPS=2.0
export NAVRL_SPEED_GOVERNOR_FREE_MPS=3.53553390593
export NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M=0.45
export NAVRL_SPEED_GOVERNOR_MARGIN_M=0.45
export NAVRL_SPEED_GOVERNOR_SLOW_M=3.0
export NAVRL_SPEED_GOVERNOR_RELEASE_M=5.0
export NAVRL_SPEED_GOVERNOR_TTC_S=1.2
export NAVRL_SPEED_GOVERNOR_BRAKE_MPS2=2.9608856678
export NAVRL_SPEED_GOVERNOR_REACTION_S=0.1
unset NAVRL_DETECTOR_CHECKPOINT

echo "[dens] densities=${NAVRL_V2_DENSITIES} policy=ep25000+riskcap sha=${ACTUAL_SHA:0:12}"
if [[ "${PREFLIGHT}" == "1" ]]; then echo "[dens] PREFLIGHT PASS"; exit 0; fi
NAVRL_V2_RESULT_DIR="${RESULT_ROOT}" ./eval_navrl_v2_density_sweep.sh "${POLICY}" 2049
echo "[dens] done -> ${RESULT_ROOT}"
