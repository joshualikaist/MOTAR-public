#!/usr/bin/env bash
# A7 P3: frozen ep25000, seed 49, 205 bars, the registered law x geometry factorial.
# Prereg: docs/prereg_2026-09-05_a7_arc_attribution.md.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if (( $# > 1 )); then
    echo "usage: $0 [ep25000-checkpoint]" >&2
    exit 2
fi

PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
PREFLIGHT_ONLY="${NAVRL_PREFLIGHT_ONLY:-0}"
REPO="$(readlink -f -- ../../..)"
SOURCE="${1:-runs/ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1/nn/last_gen_ppo_ep_25000_rew_39.742134.pth}"
SOURCE_SHA="f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40"
RESULT_ROOT="${REPO}/results/navrl_arc_attribution_205bars_seed49"
SHARED_BUNDLE="${RESULT_ROOT}/source_bundle"

if [[ ! -f "${SOURCE}" ]]; then
    echo "[arc-attribution] source checkpoint missing: ${SOURCE}" >&2
    exit 2
fi
if [[ -e "${RESULT_ROOT}" || -L "${RESULT_ROOT}" ]]; then
    echo "[arc-attribution] refusing to overwrite ${RESULT_ROOT}; partial resume is forbidden" >&2
    exit 2
fi
ACTUAL_SHA="$("${PYTHON}" - "${SOURCE}" <<'PY'
import hashlib
from pathlib import Path
import sys
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
if [[ "${ACTUAL_SHA}" != "${SOURCE_SHA}" ]]; then
    echo "[arc-attribution] source SHA mismatch: ${ACTUAL_SHA}" >&2
    exit 2
fi

# A closed evaluation: only the preregistered NAVRL settings reach the evaluator. In particular,
# inherited force/goal-band/detector/target-motion/distractor settings cannot change this lineage.
while IFS= read -r variable; do
    unset "${variable}"
done < <(compgen -v NAVRL_)
unset PYTHONPATH PYTHONHOME
export PYTHON PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO}"
export NAVRL_REQUIRE_SOURCE_ROOT="${REPO}"
export NAVRL_PREFLIGHT_ONLY="${PREFLIGHT_ONLY}"
export GPU4GB=0
export NUM_ENVS=128
export AERIAL_GYM_SIM_NAME=base_sim
export NAVRL_V2_FORCE=0
export NAVRL_V2_ALLOW_DETECTOR_THRESHOLD_MISMATCH=0
export NAVRL_V2_DENSITIES=205
export NAVRL_SEED=49
export NAVRL_V2_ACTION_MODE=deterministic
export NAVRL_EVAL_REFLECTION_MODE=original
export NAVRL_V2_GOAL_DIST_MIN=6
export NAVRL_V2_GOAL_DIST_MAX=28
export NAVRL_V2_TARGET_PATTERN=mixed
export NAVRL_DISTRACTOR_COUNT=0
# A7 includes clearance occupancy and intervention costs. The 09-02 screen lacked this
# read-only telemetry; M5 must establish that the added instrumentation preserves its baseline.
export NAVRL_CONTACT_GEOMETRY=1
export NAVRL_STAR_CONVEX_SHADOW=0
export NAVRL_SPEED_GOVERNOR_DIAG=1
export NAVRL_SPEED_GOVERNOR_FIXED_MPS=2.0
export NAVRL_SPEED_GOVERNOR_FREE_MPS=3.53553390593
export NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M=0.45
export NAVRL_SPEED_GOVERNOR_MARGIN_M=0.45
export NAVRL_SPEED_GOVERNOR_SLOW_M=3.0
export NAVRL_SPEED_GOVERNOR_RELEASE_M=5.0
export NAVRL_SPEED_GOVERNOR_TTC_S=1.0
export NAVRL_SPEED_GOVERNOR_BRAKE_MPS2=2.0
export NAVRL_SPEED_GOVERNOR_REACTION_S=0.1
export NAVRL_V2_SHARED_SOURCE_BUNDLE="${SHARED_BUNDLE}"

RUN_COMMIT="$(git -C "${REPO}" rev-parse HEAD)"
require_frozen_source() {
    # A CPU-only preflight may run before the implementation commit; actual rollouts may not.
    if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
        return
    fi
    local status current_commit
    status="$(git -C "${REPO}" status --porcelain=v1 --untracked-files=all \
        -- aerial_gym tools resources/robots)"
    if [[ -n "${status}" ]]; then
        echo "[arc-attribution] runtime/launcher sources must be committed: ${status}" >&2
        exit 3
    fi
    current_commit="$(git -C "${REPO}" rev-parse HEAD)"
    if [[ "${current_commit}" != "${RUN_COMMIT}" ]]; then
        echo "[arc-attribution] git HEAD changed during evaluation; this root is VOID" >&2
        exit 3
    fi
}

run_cell() {
    local mode="$1"
    require_frozen_source
    echo "[arc-attribution] cell=${mode} seed=49 bars=205 brake=2.0 commit=${RUN_COMMIT}"
    NAVRL_SPEED_GOVERNOR="${mode}" \
    NAVRL_V2_RESULT_DIR="${RESULT_ROOT}/${mode}" \
        bash ./eval_navrl_v2_density_sweep.sh "${SOURCE}" 2049
    require_frozen_source
}

run_cell riskcap
run_cell stopcap
run_cell dwa_arc
run_cell riskcap_arc

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
    echo "[arc-attribution] PREFLIGHT PASS (four cells; no results written)"
    exit 0
fi
echo "[arc-attribution] COMPLETE: ${RESULT_ROOT}"
echo "[arc-attribution] After P1/P2/P3 finish, run: python tools/build_a7_arc_attribution_table.py"
