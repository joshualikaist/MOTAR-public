#!/usr/bin/env bash
# Controlled fixed-85 warm-start: replace greedy angular suppression with cluster-sector selection.
#
# The observation remains 898-D. Adjacent LiDAR surface returns are clustered, every non-empty
# forward sector reserves its nearest cluster, empty sector capacity is filled by remaining nearest
# clusters, and the final eight proposals remain nearest-first for checkpoint adaptation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export CKPT="${CKPT:-runs/ppo_260730_1104_navrl_corrected-squashed-density25to110-s1/nn/last_gen_ppo_ep_8350_rew_22.905773.pth}"
export MAX_EPOCHS="${MAX_EPOCHS:-12000}"
export SEED="${SEED:-1}"
export AERIAL_RUN_TAG="${AERIAL_RUN_TAG:-corrected-squashed-density-cluster-sector-s${SEED}}"
export TRAIN_SESSION_LOG="${TRAIN_SESSION_LOG:-train_session_logs/corrected_squashed_density_cluster_sector_$(date +%y%m%d_%H%M%S).log}"

export NAVRL_OBSTACLE_SELECTOR=cluster_sector
export NAVRL_OBSTACLE_CLUSTER_GAP_M="${NAVRL_OBSTACLE_CLUSTER_GAP_M:-0.45}"
export NAVRL_OBSTACLE_SECTORS="${NAVRL_OBSTACLE_SECTORS:-8}"
export NAVRL_OBSTACLE_SUPPRESS_DEG=10
export NAVRL_ALLOW_SELECTOR_WARMSTART=1

# Default = representation ablation (density frozen at NAVRL_FIXED_BARS). Pass
# NAVRL_CONTROLLED_ABLATION=0 to run the density CURRICULUM instead (85 -> NAVRL_DENSITY_FINAL):
# the base script then unsets FIXED_BARS/NUM_BARS so the checkpoint's n_bars_active is restored
# and promotion past 85 is possible. The old unconditional `export NAVRL_CONTROLLED_ABLATION=1`
# silently discarded that override and re-froze every resume at 85 bars (found 2026-07-30).
export NAVRL_CONTROLLED_ABLATION="${NAVRL_CONTROLLED_ABLATION:-1}"
if [[ "${NAVRL_CONTROLLED_ABLATION}" == "1" ]]; then
    export NAVRL_FIXED_BARS="${NAVRL_FIXED_BARS:-85}"
fi
export NAVRL_RESET_DENSITY_WINDOW=1
export NAVRL_DENSITY_RESUME_WARMUP="${NAVRL_DENSITY_RESUME_WARMUP:-250}"

echo "[cluster-sector] warm-start | mode=$([[ "${NAVRL_CONTROLLED_ABLATION}" == "1" ]] && echo "fixed-${NAVRL_FIXED_BARS:-85}bars-ablation" || echo "density-curriculum->${NAVRL_DENSITY_FINAL:-110}") selector=${NAVRL_OBSTACLE_SELECTOR} gap=${NAVRL_OBSTACLE_CLUSTER_GAP_M}m sectors=${NAVRL_OBSTACLE_SECTORS} checkpoint=${CKPT}"
echo "[cluster-sector] output is visible here and saved; another terminal can run:"
echo "[cluster-sector]   ./watch_navrl_training.sh"
exec ./train_navrl_corrected_squashed_density.sh "$@"
