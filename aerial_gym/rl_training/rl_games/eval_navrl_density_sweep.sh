#!/usr/bin/env bash
# Held-out density sweep for a vision checkpoint (논문 밀도–성능 곡선).
# ALWAYS pass last_gen_ppo_ep_* — NOT gen_ppo.pth (best-reward = low-density policy).
#
# Usage:
#   ./eval_navrl_density_sweep.sh runs/ppo_XXXX_navrl/nn/last_gen_ppo_ep_9000.pth
#   ./eval_navrl_density_sweep.sh runs/ppo_XXXX_navrl/nn/last_gen_ppo_ep_9000.pth 2500
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONNOUSERSITE=1
# This is the V1-ARENA sweep (24x24 m, 2 m bars, relax placement, 300-step episodes). Pin the
# v1 arena explicitly instead of relying on them being unset: a shell that previously ran the
# v2 launcher still exports NAVRL_ARENA_XY=40 etc., and because the observation width is
# identical the checkpoint would load fine and silently be scored on the wrong task.
# Use eval_navrl_v2_density_sweep.sh for v2 checkpoints.
export NAVRL_ARENA_XY=24
export NAVRL_ARENA_Z=3
export NAVRL_BAR_POOL=bars
export NAVRL_PLACEMENT_MODE=random
export NAVRL_EPISODE_LEN_STEPS=300
export NAVRL_VISION=1
# Perception Transformer is the current policy family. Without this, play_navrl.sh picks
# ppo_navrl_vision.yaml (1265-dim CNN) and the checkpoint cannot load at all.
export NAVRL_PERCEPTION="${NAVRL_PERCEPTION:-1}"
export NAVRL_K_FINAL=16
export NAVRL_K_MIN_FINAL=10

# SCALING-CRITICAL: these must match the values the checkpoint was TRAINED with. lidar range is both
# the sensor horizon and the observation divisor (scan/range); max_velocity and yaw_rate_max scale
# both observations and action limits. Leaving them unset silently reverts to the config defaults
# (4.0 / 2.0 / 2.5) and produces a wrong density curve that looks like a bad policy, not a bad eval.
# Defaults below match the validated 2026-07-27 FOV-240 representation policy. Override every
# representation field together when evaluating an older checkpoint.
export NAVRL_GENERAL_TRAIN="${NAVRL_GENERAL_TRAIN:-1}"
export NAVRL_LIDAR_RANGE="${NAVRL_LIDAR_RANGE:-12}"
export NAVRL_MAX_VELOCITY="${NAVRL_MAX_VELOCITY:-2.5}"
export NAVRL_YAW_RATE_MAX="${NAVRL_YAW_RATE_MAX:-3.0}"
export NAVRL_TILT_COMP="${NAVRL_TILT_COMP:-1}"
export NAVRL_MAX_BARS="${NAVRL_MAX_BARS:-150}"
export NAVRL_MAX_OBSTACLES="${NAVRL_MAX_OBSTACLES:-8}"
export NAVRL_LIDAR_HBEAMS="${NAVRL_LIDAR_HBEAMS:-72}"
export NAVRL_LIDAR_VBEAMS="${NAVRL_LIDAR_VBEAMS:-4}"
export NAVRL_OBSTACLE_SELECTOR="${NAVRL_OBSTACLE_SELECTOR:-greedy_suppress}"
export NAVRL_OBSTACLE_CLUSTER_GAP_M="${NAVRL_OBSTACLE_CLUSTER_GAP_M:-0.45}"
export NAVRL_OBSTACLE_SECTORS="${NAVRL_OBSTACLE_SECTORS:-${NAVRL_MAX_OBSTACLES}}"
export NAVRL_OBSTACLE_SUPPRESS_DEG="${NAVRL_OBSTACLE_SUPPRESS_DEG:-10}"
export NAVRL_OBSTACLE_FOV_DEG="${NAVRL_OBSTACLE_FOV_DEG:-240}"
echo "[eval_sweep] lidar=${NAVRL_LIDAR_RANGE}m scan=${NAVRL_LIDAR_VBEAMS}x${NAVRL_LIDAR_HBEAMS} \
tokens=${NAVRL_MAX_OBSTACLES} fov=${NAVRL_OBSTACLE_FOV_DEG} selector=${NAVRL_OBSTACLE_SELECTOR} \
cluster_gap=${NAVRL_OBSTACLE_CLUSTER_GAP_M}m sectors=${NAVRL_OBSTACLE_SECTORS} \
suppress=+-${NAVRL_OBSTACLE_SUPPRESS_DEG} \
vmax=${NAVRL_MAX_VELOCITY} yaw=${NAVRL_YAW_RATE_MAX} general=${NAVRL_GENERAL_TRAIN}"

CKPT="${1:?usage: $0 <last_gen_ppo_ep_XXXX.pth> [games_per_cell]}"
GAMES="${2:-2500}"
DENSITIES=(25 50 75 110 130)

echo "[eval_sweep] ckpt=${CKPT} games/cell=${GAMES}"
for N in "${DENSITIES[@]}"; do
  echo "======== density ${N} bars ========"
  NAVRL_NUM_BARS="${N}" NUM_ENVS="${NUM_ENVS:-128}" HEADLESS=True PLAY_GAMES_NUM="${GAMES}" \
    ./play_navrl.sh "${CKPT}" 2>&1 \
    | tee "train_session_logs/eval_${N}bars_$(date +%y%m%d_%H%M).log"
done
echo "[eval_sweep] done. logs in train_session_logs/eval_*bars_*.log"
