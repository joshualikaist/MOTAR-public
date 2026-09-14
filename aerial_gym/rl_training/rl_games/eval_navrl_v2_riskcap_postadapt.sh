#!/usr/bin/env bash
# Final held-out validation for the R2b riskcap adaptation.
# Usage: ./eval_navrl_v2_riskcap_postadapt.sh runs/.../last_gen_ppo_ep_25000_....pth
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if (( $# != 1 )); then
    echo "usage: $0 trained-ep25000-checkpoint" >&2
    exit 2
fi

PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PYTHON
SOURCE="runs/ppo_260802_0020_navrl_v2-recover-curriculum-continue-s1/nn/last_gen_ppo_ep_24000_rew_44.73549.pth"
SOURCE_SHA="82f7978b42d9d9e95adcc638a40ae85fb3736fd897ae39cb8aa8333be39cf23f"
TRAINED="$1"
RESULT_ROOT="../../../results/navrl_v2_riskcap_postadapt"

if [[ ! -f "${SOURCE}" || ! -f "${TRAINED}" ]]; then
    echo "[riskcap-post] source or trained checkpoint missing" >&2
    exit 2
fi
if [[ -e "${RESULT_ROOT}" ]]; then
    echo "[riskcap-post] refusing to overwrite ${RESULT_ROOT}" >&2
    exit 2
fi

TRAINED_SHA="$(${PYTHON} - "${SOURCE}" "${SOURCE_SHA}" "${TRAINED}" <<'PY'
import hashlib
import math
from pathlib import Path
import re
import sys
import torch

source, expected_source_sha, trained = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
if hashlib.sha256(source.read_bytes()).hexdigest() != expected_source_sha:
    raise SystemExit("[riskcap-post] frozen source SHA mismatch")
payload = torch.load(trained, map_location="cpu", weights_only=False)
state = payload.get("env_state") or {}
run_root = trained.parent.parent
checks = {
    "canonical_filename": re.fullmatch(r"last_gen_ppo_ep_25000_rew_-?[0-9]+(?:\.[0-9]+)?\.pth", trained.name) is not None,
    "normal_completion": (run_root / ".aerial_training_finished").is_file(),
    "epoch": int(payload.get("epoch", -1)) == 25000,
    "frame": int(payload.get("frame", -1)) == 102400000,
    "bars": int(state.get("n_bars_active", -1)) == 205,
    "selector": state.get("cfg_obstacle_selector") == "cluster_sector",
    "governor": state.get("cfg_speed_governor_mode") == "riskcap",
    "fixed": math.isclose(float(state.get("cfg_speed_governor_fixed_mps", -1)), 2.0),
    "free": math.isclose(float(state.get("cfg_speed_governor_free_mps", -1)), 3.53553390593),
    "half_width": math.isclose(float(state.get("cfg_speed_governor_half_width_m", -1)), 0.45),
    "margin": math.isclose(float(state.get("cfg_speed_governor_margin_m", -1)), 0.45),
    "slow": math.isclose(float(state.get("cfg_speed_governor_slow_m", -1)), 3.0),
    "release": math.isclose(float(state.get("cfg_speed_governor_release_m", -1)), 5.0),
    "ttc": math.isclose(float(state.get("cfg_speed_governor_ttc_s", -1)), 1.2),
    "brake": math.isclose(float(state.get("cfg_speed_governor_brake_mps2", -1)), 2.9608856678),
    "reaction": math.isclose(float(state.get("cfg_speed_governor_reaction_s", -1)), 0.1),
    "target_exclusion": state.get("cfg_speed_governor_target_exclusion") == "camera_lidar_association",
    "training_contract": int(state.get("cfg_training_seed", -1)) == 1
    and int(state.get("cfg_training_num_envs", -1)) == 128
    and state.get("cfg_training_profile") == "main"
    and state.get("cfg_training_sim") == "base_sim"
    and math.isclose(float(state.get("cfg_action_learning_rate", -1)), 5e-6),
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("[riskcap-post] invalid trained checkpoint: " + ", ".join(failed))
print(hashlib.sha256(trained.read_bytes()).hexdigest())
PY
)"

export NAVRL_V2_DENSITIES=205
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
    local checkpoint="$1"
    local tag="$2"
    local mode="$3"
    local seed="$4"
    local speed="$5"
    echo "[riskcap-post] cell=${tag} mode=${mode} seed=${seed} speed=${speed:-uniform}"
    if [[ -n "${speed}" ]]; then
        NAVRL_V2_FIXED_TARGET_SPEED="${speed}" \
        NAVRL_SPEED_GOVERNOR="${mode}" NAVRL_SEED="${seed}" \
        NAVRL_V2_RESULT_DIR="${RESULT_ROOT}/${tag}" \
            ./eval_navrl_v2_density_sweep.sh "${checkpoint}" 2049
    else
        NAVRL_SPEED_GOVERNOR="${mode}" NAVRL_SEED="${seed}" \
        NAVRL_V2_RESULT_DIR="${RESULT_ROOT}/${tag}" \
            ./eval_navrl_v2_density_sweep.sh "${checkpoint}" 2049
    fi
}

# Completely unseen uniform seed: replicate the mechanism and isolate the value of adaptation.
run_cell "${SOURCE}" uniform_off off 45 ""
run_cell "${SOURCE}" uniform_source_riskcap riskcap 45 ""
run_cell "${TRAINED}" uniform_trained_riskcap riskcap 45 ""

"${PYTHON}" ../../../tools/summarize_navrl_v2_riskcap_postadapt.py \
    "${RESULT_ROOT}" --select-only --trained-sha "${TRAINED_SHA}"
WINNER_KIND="$(${PYTHON} - "${RESULT_ROOT}/selection.json" <<'PY'
import json
from pathlib import Path
import sys
print(json.loads(Path(sys.argv[1]).read_text())["winner_kind"])
PY
)"
case "${WINNER_KIND}" in
    source) WINNER_CHECKPOINT="${SOURCE}" ;;
    trained) WINNER_CHECKPOINT="${TRAINED}" ;;
    *) echo "[riskcap-post] invalid winner kind: ${WINNER_KIND}" >&2; exit 2 ;;
esac

# A second unseen seed on the three target-speed extremes/interior point.
for speed in 0.3 0.9 1.5; do
    tag="${speed//./p}"
    run_cell "${SOURCE}" "fixed_${tag}_off" off 46 "${speed}"
    run_cell "${WINNER_CHECKPOINT}" "fixed_${tag}_winner" riskcap 46 "${speed}"
done

"${PYTHON}" ../../../tools/summarize_navrl_v2_riskcap_postadapt.py \
    "${RESULT_ROOT}" --trained-sha "${TRAINED_SHA}"
