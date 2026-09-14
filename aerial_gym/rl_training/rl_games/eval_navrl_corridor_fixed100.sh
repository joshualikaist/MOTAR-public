#!/usr/bin/env bash
# Held-out four-speed evaluation for a corridor-token checkpoint at fixed 100 bars.
#
# Usage:
#   ./eval_navrl_corridor_fixed100.sh CHECKPOINT [games_per_cell] [result_dir]
#
# This wrapper pins every representation field that changes the policy-input contract.  Calling
# the generic evaluator without NAVRL_CORRIDOR_TOKENS would build the historical 898-D actor and
# fail to load a 946-D corridor checkpoint (or, worse, make an incomparable evaluation if future
# compatibility code became permissive).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CKPT="${1:?usage: $0 CHECKPOINT [games_per_cell] [result_dir]}"
GAMES="${2:-1000}"
RESULT_ARG="${3:-}"

if [[ ! "${GAMES}" =~ ^[0-9]+$ ]] || (( GAMES < 100 )); then
    echo "[corridor-eval] games_per_cell must be an integer >= 100" >&2
    exit 2
fi
if [[ ! -f "${CKPT}" ]]; then
    echo "[corridor-eval] checkpoint not found: ${CKPT}" >&2
    exit 2
fi
CKPT="$(readlink -f "${CKPT}")"
PY="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"

export NAVRL_VISION=1
export NAVRL_PERCEPTION=1
export NAVRL_GENERAL_TRAIN=1
export NAVRL_OBSTACLE_SELECTOR=cluster_sector
export NAVRL_OBSTACLE_CLUSTER_GAP_M="${NAVRL_OBSTACLE_CLUSTER_GAP_M:-0.45}"
export NAVRL_OBSTACLE_SECTORS="${NAVRL_OBSTACLE_SECTORS:-8}"
export NAVRL_OBSTACLE_SUPPRESS_DEG="${NAVRL_OBSTACLE_SUPPRESS_DEG:-10}"
export NAVRL_OBSTACLE_FOV_DEG="${NAVRL_OBSTACLE_FOV_DEG:-240}"
export NAVRL_MAX_OBSTACLES="${NAVRL_MAX_OBSTACLES:-8}"
export NAVRL_LIDAR_HBEAMS="${NAVRL_LIDAR_HBEAMS:-72}"
export NAVRL_LIDAR_VBEAMS="${NAVRL_LIDAR_VBEAMS:-4}"
export NAVRL_LIDAR_RANGE="${NAVRL_LIDAR_RANGE:-12}"
export NAVRL_GENERAL_GOAL_DIST_MIN="${NAVRL_GENERAL_GOAL_DIST_MIN:-4}"
export NAVRL_GENERAL_GOAL_DIST_MAX="${NAVRL_GENERAL_GOAL_DIST_MAX:-16}"
export NAVRL_K_MIN_FINAL="${NAVRL_K_MIN_FINAL:-10}"
export NAVRL_K_FINAL="${NAVRL_K_FINAL:-16}"

export NAVRL_CORRIDOR_TOKENS="${NAVRL_CORRIDOR_TOKENS:-6}"
export NAVRL_CORRIDOR_HORIZON_M="${NAVRL_CORRIDOR_HORIZON_M:-6.0}"
export NAVRL_CORRIDOR_MIN_WIDTH_M="${NAVRL_CORRIDOR_MIN_WIDTH_M:-0.55}"

export DENSITIES=100
export TARGET_SPEEDS="${TARGET_SPEEDS:-0.0 0.5 1.0 1.5}"
export PURSUER_SPEEDS=2.5
export PLAY_GAMES_NUM="${GAMES}"
export NUM_ENVS="${NUM_ENVS:-128}"
export HEADLESS=True
export NAVRL_SEED="${NAVRL_SEED:-1}"

"${PY}" - "${CKPT}" <<'PY'
import os
import sys
import torch

path = sys.argv[1]
state = torch.load(path, map_location="cpu", weights_only=False).get("env_state", {})
expected = {
    "cfg_corridor_tokens": int(os.environ["NAVRL_CORRIDOR_TOKENS"]),
    "cfg_corridor_horizon_m": float(os.environ["NAVRL_CORRIDOR_HORIZON_M"]),
    "cfg_corridor_min_width_m": float(os.environ["NAVRL_CORRIDOR_MIN_WIDTH_M"]),
    "cfg_obstacle_selector": os.environ["NAVRL_OBSTACLE_SELECTOR"],
    "cfg_obstacle_cluster_gap_m": float(os.environ["NAVRL_OBSTACLE_CLUSTER_GAP_M"]),
    "cfg_obstacle_sectors": int(os.environ["NAVRL_OBSTACLE_SECTORS"]),
    "cfg_obstacle_suppress_deg": float(os.environ["NAVRL_OBSTACLE_SUPPRESS_DEG"]),
    "cfg_token_fov_deg": float(os.environ["NAVRL_OBSTACLE_FOV_DEG"]),
    "cfg_max_obstacles": int(os.environ["NAVRL_MAX_OBSTACLES"]),
    "cfg_lidar_hbeams": int(os.environ["NAVRL_LIDAR_HBEAMS"]),
    "cfg_lidar_vbeams": int(os.environ["NAVRL_LIDAR_VBEAMS"]),
    "cfg_lidar_max_range": float(os.environ["NAVRL_LIDAR_RANGE"]),
    "cfg_general_goal_dist_min": float(os.environ["NAVRL_GENERAL_GOAL_DIST_MIN"]),
    "cfg_general_goal_dist_max": float(os.environ["NAVRL_GENERAL_GOAL_DIST_MAX"]),
}
bad = []
legacy = {
    "cfg_corridor_horizon_m": 6.0,
    "cfg_corridor_min_width_m": 0.55,
}
for key, wanted in expected.items():
    saved = state.get(key, legacy.get(key))
    same = (
        str(saved).lower() == str(wanted).lower()
        if isinstance(wanted, str)
        else saved is not None and abs(float(saved) - float(wanted)) <= 1e-6
    )
    if not same:
        bad.append(f"{key}: checkpoint={saved!r} requested={wanted!r}")
if bad:
    raise SystemExit("[corridor-eval] checkpoint/eval contract mismatch | " + "; ".join(bad))
print("[corridor-eval] checkpoint contract: PASS")
PY

if [[ -n "${RESULT_ARG}" ]]; then
    export RESULT_DIR="${RESULT_ARG}"
else
    epoch="$(
        basename "${CKPT}" |
        sed -n 's/.*_ep_\([0-9][0-9]*\)_.*/\1/p'
    )"
    stamp="$(date +%y%m%d_%H%M%S)"
    export RESULT_DIR="train_session_logs/eval_results/corridor_fixed100_ep${epoch:-unknown}_${stamp}"
fi

echo "[corridor-eval] checkpoint=${CKPT}"
echo "[corridor-eval] schema=cluster-sector+corridor${NAVRL_CORRIDOR_TOKENS} \
obs=$((898 + NAVRL_CORRIDOR_TOKENS * 8)) bars=100 target={${TARGET_SPEEDS// /,}} \
goal=${NAVRL_GENERAL_GOAL_DIST_MIN}..${NAVRL_GENERAL_GOAL_DIST_MAX}m \
games/cell=${GAMES} seed=${NAVRL_SEED}"

./eval_navrl_speed_density_grid.sh "${CKPT}"

"${PY}" ../../../tools/evaluate_corridor_gate.py \
    "${RESULT_DIR}/results.csv" \
    --output "${RESULT_DIR}/gate.json"

echo "[corridor-eval] complete -> ${RESULT_DIR}"
