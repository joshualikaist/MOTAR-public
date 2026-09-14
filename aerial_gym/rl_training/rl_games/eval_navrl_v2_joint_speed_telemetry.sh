#!/usr/bin/env bash
# Frozen-policy, one-cell descriptive diagnosis of speed allocation before bar contact.
# This is evaluation only.  It does not compare or tune riskcap parameters.
set -euo pipefail
CALLER_PWD="${PWD}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PYTHON PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Editable installs can otherwise resolve aerial_gym from the primary dirty worktree while this
# diagnostic runs from an isolated worktree.  That silently drops the opt-in telemetry code.
IMPORT_ORIGIN="$(${PYTHON} - "${REPO_ROOT}" <<'PY'
import importlib.util
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
spec = importlib.util.find_spec("aerial_gym")
if spec is None or spec.origin is None:
    raise SystemExit("missing aerial_gym import")
print(pathlib.Path(spec.origin).resolve())
PY
)"
EXPECTED_ORIGIN="$(readlink -f -- "${REPO_ROOT}/aerial_gym/__init__.py")"
[[ "${IMPORT_ORIGIN}" == "${EXPECTED_ORIGIN}" ]] || {
    echo "[joint-speed] aerial_gym import escaped worktree: ${IMPORT_ORIGIN}" >&2; exit 3;
}
POLICY_REL="aerial_gym/rl_training/rl_games/runs/ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1/nn/last_gen_ppo_ep_25000_rew_39.742134.pth"
if [[ -n "${POLICY:-}" ]]; then
    if [[ "${POLICY}" != /* ]]; then
        POLICY="${CALLER_PWD}/${POLICY}"
    fi
else
    POLICY="${REPO_ROOT}/${POLICY_REL}"
    if [[ ! -f "${POLICY}" ]]; then
        # Ignored runs/ are not duplicated into Git worktrees.  The shared common Git directory
        # belongs to the primary worktree; checkpoint identity remains enforced by POLICY_SHA.
        GIT_COMMON_DIR="$(git -C "${REPO_ROOT}" rev-parse --git-common-dir)"
        if [[ "${GIT_COMMON_DIR}" != /* ]]; then
            GIT_COMMON_DIR="${REPO_ROOT}/${GIT_COMMON_DIR}"
        fi
        PRIMARY_ROOT="$(dirname "$(readlink -f -- "${GIT_COMMON_DIR}")")"
        POLICY="${PRIMARY_ROOT}/${POLICY_REL}"
    fi
fi
POLICY_SHA="f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40"
if [[ -n "${RESULT_ROOT:-}" ]]; then
    if [[ "${RESULT_ROOT}" != /* ]]; then
        RESULT_ROOT="${CALLER_PWD}/${RESULT_ROOT}"
    fi
else
    RESULT_ROOT="${REPO_ROOT}/results/navrl_v2_joint_speed_allocation_seed379"
fi
EPISODES="${EPISODES:-4097}"
PREFLIGHT="${PREFLIGHT:-0}"

[[ -f "${POLICY}" ]] || { echo "[joint-speed] missing policy: ${POLICY}" >&2; exit 2; }
[[ "${EPISODES}" =~ ^[1-9][0-9]*$ ]] || { echo "[joint-speed] invalid EPISODES" >&2; exit 2; }
if (( EPISODES < 2049 )); then
    echo "[joint-speed] preregistration requires EPISODES>=2049" >&2; exit 2
fi
if [[ -e "${RESULT_ROOT}" && "${PREFLIGHT}" != "1" ]]; then
    echo "[joint-speed] refusing to overwrite ${RESULT_ROOT}" >&2; exit 2
fi

ACTUAL_SHA="$(${PYTHON} - "${POLICY}" <<'PY'
import hashlib, sys
from pathlib import Path
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
[[ "${ACTUAL_SHA}" == "${POLICY_SHA}" ]] || {
    echo "[joint-speed] policy SHA mismatch: ${ACTUAL_SHA}" >&2; exit 3;
}

# Exact frozen navigation/control candidate.  These constants come from the preregistered braking
# probe and existing riskcap campaign; changing any one creates a different experiment.
export NAVRL_V2_DENSITIES=205
export NAVRL_V2_ACTION_MODE=deterministic
export NAVRL_EVAL_REFLECTION_MODE=original
export NAVRL_SPEED_GOVERNOR=riskcap
export NAVRL_SEED=379
export NAVRL_SPEED_GOVERNOR_FIXED_MPS=2.0
export NAVRL_SPEED_GOVERNOR_FREE_MPS=3.53553390593
export NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M=0.45
export NAVRL_SPEED_GOVERNOR_MARGIN_M=0.45
export NAVRL_SPEED_GOVERNOR_SLOW_M=3.0
export NAVRL_SPEED_GOVERNOR_RELEASE_M=5.0
export NAVRL_SPEED_GOVERNOR_TTC_S=1.2
export NAVRL_SPEED_GOVERNOR_BRAKE_MPS2=2.9608856678
export NAVRL_SPEED_GOVERNOR_REACTION_S=0.1
export NAVRL_JOINT_SPEED_TELEMETRY=1
unset NAVRL_DETECTOR_CHECKPOINT

if [[ "${PREFLIGHT}" == "1" ]]; then
    echo "[joint-speed] PREFLIGHT PASS | ep25000+riskcap bars=205 seed=379 episodes=${EPISODES}"
    exit 0
fi

NAVRL_V2_RESULT_DIR="${RESULT_ROOT}" \
    ./eval_navrl_v2_density_sweep.sh "${POLICY}" "${EPISODES}"

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "${PYTHON}" \
    "${REPO_ROOT}/tools/analyze_navrl_joint_speed.py" \
    "${RESULT_ROOT}/205bars.json" "${RESULT_ROOT}/205bars.receipt.json" \
    --output "${RESULT_ROOT}/assessment.json"
echo "[joint-speed] done -> ${RESULT_ROOT}/assessment.json"
