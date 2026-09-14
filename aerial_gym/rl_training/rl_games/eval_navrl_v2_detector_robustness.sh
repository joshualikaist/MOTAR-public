#!/usr/bin/env bash
# R3 held-out detector/perception robustness screen for the frozen ep25000 + riskcap policy.
# Usage: ./eval_navrl_v2_detector_robustness.sh [trained-detector-checkpoint]
#
# SUPERSEDED LATENCY CELLS: the archived latency_0p1s / latency_0p2s results in
# results/navrl_v2_detector_robustness (37.82% / 18.50%) were produced BEFORE the ego-motion
# fix, i.e. with the delayed measurement lifted to world using the current pose instead of the
# pose it was taken at. Under the corrected model 0.1 s costs 2.5 pp, not 42.7 pp, and latency
# is no longer the dominant perception axis (WORKLOG 2026-08-06,
# results/navrl_v2_latency_ego_motion). Re-running this script now picks up the corrected model
# by default, so its latency numbers will NOT reproduce the archived ones -- that is intended.
# The non-latency cells are unaffected: the fix is arithmetically a no-op at zero latency.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PYTHON

POLICY="runs/ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1/nn/last_gen_ppo_ep_25000_rew_39.742134.pth"
POLICY_SHA="f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40"
RESULT_ROOT="../../../results/navrl_v2_detector_robustness"
LEARNED_DETECTOR="${1:-../../../artifacts/navrl_target_detector_v1.pth}"
PREFLIGHT="${PREFLIGHT:-0}"

if [[ ! -f "${POLICY}" ]]; then
    echo "[r3] policy checkpoint missing: ${POLICY}" >&2
    exit 2
fi
if [[ -e "${RESULT_ROOT}" && "${PREFLIGHT}" != "1" ]]; then
    echo "[r3] refusing to overwrite ${RESULT_ROOT}" >&2
    exit 2
fi
if [[ ! -f "${LEARNED_DETECTOR}" && "${PREFLIGHT}" != "1" ]]; then
    echo "[r3] learned detector checkpoint missing: ${LEARNED_DETECTOR}" >&2
    echo "[r3] run tools/train_navrl_target_detector.py first" >&2
    exit 2
fi

ACTUAL_POLICY_SHA="$(
    "${PYTHON}" - "${POLICY}" "${POLICY_SHA}" <<'PY'
import hashlib
import math
from pathlib import Path
import re
import sys
import torch

trained, expected = Path(sys.argv[1]), sys.argv[2]
actual = hashlib.sha256(trained.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f"[r3] policy SHA mismatch: {actual}")
payload = torch.load(trained, map_location="cpu", weights_only=False)
state = payload.get("env_state") or {}
checks = {
    "filename": re.fullmatch(r"last_gen_ppo_ep_25000_rew_-?[0-9]+(?:\.[0-9]+)?\.pth", trained.name) is not None,
    "epoch": int(payload.get("epoch", -1)) == 25000,
    "bars": int(state.get("n_bars_active", -1)) == 205,
    "selector": state.get("cfg_obstacle_selector") == "cluster_sector",
    "governor": state.get("cfg_speed_governor_mode") == "riskcap",
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("[r3] invalid policy checkpoint: " + ", ".join(failed))
print(actual)
PY
)"

LEARNED_SHA=""
if [[ -f "${LEARNED_DETECTOR}" ]]; then
    LEARNED_SHA="$(
        "${PYTHON}" - "${LEARNED_DETECTOR}" <<'PY'
import hashlib
from pathlib import Path
import sys

path = Path(sys.argv[1])
digest = hashlib.sha256()
with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
    )"
elif [[ "${PREFLIGHT}" == "1" ]]; then
    LEARNED_SHA="preflight"
else
    echo "[r3] learned detector checkpoint missing: ${LEARNED_DETECTOR}" >&2
    echo "[r3] run tools/train_navrl_target_detector.py first" >&2
    exit 2
fi

export NAVRL_V2_DENSITIES=205
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

run_cell() {
    local tag="$1"
    local perturb="$2"
    local dropout="$3"
    local latency="$4"
    local range_error="$5"
    local detector="${6:-}"
    echo "[r3] cell=${tag} perturb=${perturb} dropout=${dropout} latency=${latency}s range_error=${range_error}m detector=${detector:-bootstrap}"
    if [[ "${PREFLIGHT}" == "1" ]]; then
        return 0
    fi
    if [[ -n "${detector}" ]]; then
        NAVRL_PERCEPTION_PERTURB="${perturb}" \
        NAVRL_DETECTION_DROPOUT="${dropout}" \
        NAVRL_DETECTION_LATENCY_S="${latency}" \
        NAVRL_RANGE_ERROR_M="${range_error}" \
        NAVRL_DETECTOR_CHECKPOINT="${detector}" \
        NAVRL_V2_RESULT_DIR="${RESULT_ROOT}/${tag}" \
            ./eval_navrl_v2_density_sweep.sh "${POLICY}" 2049
    else
        unset NAVRL_DETECTOR_CHECKPOINT
        NAVRL_PERCEPTION_PERTURB="${perturb}" \
        NAVRL_DETECTION_DROPOUT="${dropout}" \
        NAVRL_DETECTION_LATENCY_S="${latency}" \
        NAVRL_RANGE_ERROR_M="${range_error}" \
        NAVRL_V2_RESULT_DIR="${RESULT_ROOT}/${tag}" \
            ./eval_navrl_v2_density_sweep.sh "${POLICY}" 2049
    fi
}

# One-axis-at-a-time screen on a fresh held-out seed. Baseline uses the bootstrap segmenter.
run_cell analytic_clean 0 0 0 0 ""
run_cell dropout_0p3 1 0.3 0 0 ""
run_cell latency_0p1s 1 0 0.1 0 ""
run_cell latency_0p2s 1 0 0.2 0 ""
run_cell range_error_0p15m 1 0 0 0.15 ""
run_cell range_error_0p30m 1 0 0 0.30 ""
run_cell learned_clean 0 0 0 0 "${LEARNED_DETECTOR}"

if [[ "${PREFLIGHT}" == "1" ]]; then
    echo "[r3] PREFLIGHT PASS (seven cells; no results written)"
    exit 0
fi

"${PYTHON}" ../../../tools/summarize_navrl_v2_detector_robustness.py \
    "${RESULT_ROOT}" \
    --policy-sha "${ACTUAL_POLICY_SHA}" \
    --learned-detector-sha "${LEARNED_SHA}"
