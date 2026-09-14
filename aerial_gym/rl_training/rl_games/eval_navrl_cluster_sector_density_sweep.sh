#!/usr/bin/env bash
# Evaluate a cluster-sector checkpoint under the exact representation used for training.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export NAVRL_OBSTACLE_SELECTOR=cluster_sector
export NAVRL_OBSTACLE_CLUSTER_GAP_M="${NAVRL_OBSTACLE_CLUSTER_GAP_M:-0.45}"
export NAVRL_OBSTACLE_SECTORS="${NAVRL_OBSTACLE_SECTORS:-8}"

exec ./eval_navrl_density_sweep.sh "$@"
