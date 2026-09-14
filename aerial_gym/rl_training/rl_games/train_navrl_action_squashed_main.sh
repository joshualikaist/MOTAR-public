#!/usr/bin/env bash
# Main-GPU bounded-action experiment.
#
# Warm-starts the best early 75-bar checkpoint, keeps density fixed for a controlled 3000-epoch
# comparison, and replaces the legacy clipped Gaussian with a likelihood-correct tanh-squashed
# Gaussian. Both stochastic and deterministic actions are strictly bounded.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export NAVRL_ACTION_POLICY="squashed_gaussian"
export NAVRL_ACTION_STD="${NAVRL_ACTION_STD:-0.35,0.35,0.05,0.08}"
# The legacy clipped policy encoded lateral commands with very large raw means because every
# |mu_y|>1 looked identical to the environment. Recalibrate only that inherited axis; this is not a
# speed limit, and PPO remains free to grow the mean again when boundary control is truly useful.
export NAVRL_ACTION_MU_SCALE="${NAVRL_ACTION_MU_SCALE:-1.0,0.4,1.0,1.0}"
# This is a competent warm-start and the exploration scale is explicit/fixed. The legacy entropy
# bonus is therefore unnecessary and must not inflate a hidden log-std again.
export NAVRL_ENTROPY_COEF="${NAVRL_ENTROPY_COEF:-0.0}"
export NAVRL_RESET_ACTOR_OPTIMIZER="${NAVRL_RESET_ACTOR_OPTIMIZER:-1}"
export NAVRL_ACTION_DIAG="${NAVRL_ACTION_DIAG:-1}"

export NAVRL_FIXED_BARS="${NAVRL_FIXED_BARS:-75}"
export NUM_ENVS="${NUM_ENVS:-128}"
export SEED="${SEED:-1}"
export MAX_EPOCHS="${MAX_EPOCHS:-22050}"  # ep19050 + 3000 controlled epochs
export AERIAL_RUN_TAG="${AERIAL_RUN_TAG:-action-squashed-main-s${SEED}}"
export TRAIN_SESSION_LOG="${TRAIN_SESSION_LOG:-train_session_logs/action_squashed_main_$(date +%y%m%d_%H%M%S).log}"

export CKPT="${CKPT:-runs/ppo_260727_2324_navrl/nn/last_gen_ppo_ep_19050_rew_31.79068.pth}"
if [[ ! -f "${CKPT}" ]]; then
    echo "[action_squashed_main] checkpoint not found: ${CKPT}" >&2
    exit 2
fi

echo "[action_squashed_main] tanh-squashed Gaussian | fixed 75 bars | seed=${SEED}"
echo "[action_squashed_main] checkpoint=${CKPT} std=${NAVRL_ACTION_STD} mu_scale=${NAVRL_ACTION_MU_SCALE}"
exec ./train_navrl_general_repr_density.sh "$@"
