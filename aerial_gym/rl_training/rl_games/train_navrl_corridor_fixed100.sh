#!/usr/bin/env bash
# Corridor-token A/B arm: fixed-100-bars adaptation from the FIX3 ep13000 checkpoint.
#
# B arm of the P0-P3 corridor plan (WORKLOG 2026-07-31). The A arm is the frozen baseline:
# ep13000 held-out 65.63% / ep12500 64.53% at 100 bars (four-speed weighted, 1000 eps/cell).
# This arm appends 6 corridor (free-gap affordance) tokens -- geometry validated by
# tools/probe_corridor_geometry.py (center-ray 100%, bound-on-bar 97.8%, width 98.8%) --
# and warm-starts the 17-token checkpoint through runner._expand_corridor_checkpoint
# (position-embedding row 18 + fresh corridor projection + zero critic columns; Adam reset).
#
# Pilot gate (site/plan contract): held-out capture >= 68%, >= +3pp vs ep12500, bar-contact down.
# EVAL NOTE: any eval of the resulting checkpoints MUST export NAVRL_CORRIDOR_TOKENS=6
# (the obs contract), or the checkpoint will not load.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export CKPT="${CKPT:-runs/ppo_260731_0226_navrl_fix3-low-lr-replay-b5-100bars-s1/nn/last_gen_ppo_ep_13000_rew_-0.8341096.pth}"
export MAX_EPOCHS="${MAX_EPOCHS:-13800}"
export SEED="${SEED:-1}"
export AERIAL_RUN_TAG="${AERIAL_RUN_TAG:-corridor6-fixed100-s${SEED}}"
export TRAIN_SESSION_LOG="${TRAIN_SESSION_LOG:-train_session_logs/corridor6_fixed100_$(date +%y%m%d_%H%M%S).log}"

# Obstacle-token representation: unchanged cluster-sector contract (matches the checkpoint).
export NAVRL_OBSTACLE_SELECTOR=cluster_sector
export NAVRL_OBSTACLE_CLUSTER_GAP_M="${NAVRL_OBSTACLE_CLUSTER_GAP_M:-0.45}"
export NAVRL_OBSTACLE_SECTORS="${NAVRL_OBSTACLE_SECTORS:-8}"
export NAVRL_OBSTACLE_SUPPRESS_DEG=10

# Corridor tokens: THE experimental variable of this arm.
export NAVRL_CORRIDOR_TOKENS="${NAVRL_CORRIDOR_TOKENS:-6}"
export NAVRL_CORRIDOR_HORIZON_M="${NAVRL_CORRIDOR_HORIZON_M:-6.0}"
export NAVRL_CORRIDOR_MIN_WIDTH_M="${NAVRL_CORRIDOR_MIN_WIDTH_M:-0.55}"
export NAVRL_CORRIDOR_WARMSTART=1
export NAVRL_ALLOW_CORRIDOR_WARMSTART=1

# Fixed-density adaptation, same contract as the FIX3 low-LR replay blocks: the loaded backbone
# must not be washed out while the fresh corridor projection learns.
export NAVRL_CONTROLLED_ABLATION=1
export NAVRL_FIXED_BARS="${NAVRL_FIXED_BARS:-100}"
export NAVRL_LEARNING_RATE="${NAVRL_LEARNING_RATE:-3e-5}"
export NAVRL_RESET_DENSITY_WINDOW=1
export NAVRL_DENSITY_RESUME_WARMUP="${NAVRL_DENSITY_RESUME_WARMUP:-0}"

echo "[corridor-ab] B arm | bars=${NAVRL_FIXED_BARS} corridor_tokens=${NAVRL_CORRIDOR_TOKENS} horizon=${NAVRL_CORRIDOR_HORIZON_M}m min_width=${NAVRL_CORRIDOR_MIN_WIDTH_M}m lr=${NAVRL_LEARNING_RATE}"
echo "[corridor-ab] warm-start ${CKPT} -> obs 898+$((NAVRL_CORRIDOR_TOKENS * 8))"
exec ./train_navrl_corrected_squashed_density.sh "$@"
