#!/usr/bin/env bash
# Controlled 85-bar warm-start experiment: widen obstacle-token suppression from ±10° to ±15°.
#
# Run this script in the foreground to see the training dashboard directly. The same output is
# saved under train_session_logs/ and can also be followed with ./watch_navrl_training.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export CKPT="${CKPT:-runs/ppo_260730_1104_navrl_corrected-squashed-density25to110-s1/nn/last_gen_ppo_ep_8350_rew_22.905773.pth}"
export MAX_EPOCHS="${MAX_EPOCHS:-12000}"
export SEED="${SEED:-1}"
export AERIAL_RUN_TAG="${AERIAL_RUN_TAG:-corrected-squashed-density-suppress15-s${SEED}}"
export TRAIN_SESSION_LOG="${TRAIN_SESSION_LOG:-train_session_logs/corrected_squashed_density_suppress15_$(date +%y%m%d_%H%M%S).log}"

export NAVRL_OBSTACLE_SUPPRESS_DEG=15
# The representation shape remains 898-D, but its selection semantics intentionally change.
# Keep the preflight strict for every other policy-input field.
export NAVRL_ALLOW_SUPPRESS_WARMSTART=1
# This is a representation ablation, not a density continuation. Holding 85 bars prevents a lucky
# promotion from changing the task under only one side of the comparison. Explicitly discard the
# checkpoint's in-progress ±10-degree competence window as well.
export NAVRL_CONTROLLED_ABLATION=1
export NAVRL_FIXED_BARS="${NAVRL_FIXED_BARS:-85}"
export NAVRL_RESET_DENSITY_WINDOW=1
export NAVRL_DENSITY_RESUME_WARMUP="${NAVRL_DENSITY_RESUME_WARMUP:-250}"

echo "[suppress15] controlled warm-start | bars=${NAVRL_FIXED_BARS} suppress=+-15deg checkpoint=${CKPT}"
echo "[suppress15] output is visible here and saved; another terminal can run:"
echo "[suppress15]   ./watch_navrl_training.sh"
exec ./train_navrl_corrected_squashed_density.sh "$@"
