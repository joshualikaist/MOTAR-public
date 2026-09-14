#!/usr/bin/env bash
# Main-GPU squashed-Gaussian v2 stability pilot.
#
# The first squashed run eliminated out-of-support actions but its lateral latent mean regrew toward
# the tanh boundary while PPO KL rose far above the documented target. It stopped at epoch 19551
# with a non-finite actor loss. This branch keeps the same ep19050 checkpoint, task, seed and action
# noise, and changes only the update safety controls needed to test that measured failure mode.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export NAVRL_ACTION_POLICY="squashed_gaussian"
export NAVRL_ACTION_STD="${NAVRL_ACTION_STD:-0.35,0.35,0.05,0.08}"
export NAVRL_ACTION_MU_SCALE="${NAVRL_ACTION_MU_SCALE:-1.0,0.4,1.0,1.0}"
export NAVRL_ENTROPY_COEF="${NAVRL_ENTROPY_COEF:-0.0}"
export NAVRL_RESET_ACTOR_OPTIMIZER="${NAVRL_RESET_ACTOR_OPTIMIZER:-1}"
export NAVRL_ACTION_DIAG="${NAVRL_ACTION_DIAG:-1}"

# Wide numerical clamp: prevents exp(log_ratio) overflow without replacing PPO's ordinary 0.2 clip.
export NAVRL_PPO_LOG_RATIO_CLAMP="${NAVRL_PPO_LOG_RATIO_CLAMP:-10.0}"
# Stop updating the actor for the rest of an over-shifted minibatch sequence. The YAML target is
# 0.016; 0.04 allows normal variation but blocks the 0.08-0.15 regime measured before the failure.
export NAVRL_PPO_KL_STOP="${NAVRL_PPO_KL_STOP:-0.04}"
# Softly discourage deterministic lateral actions from living at the tanh boundary. tanh(1.25)
# is 0.848, or 2.12 m/s at the 2.5 m/s task scale; faster noisy actions remain available.
export NAVRL_LATENT_MARGIN_Y="${NAVRL_LATENT_MARGIN_Y:-1.25}"
export NAVRL_LATENT_MARGIN_COEF="${NAVRL_LATENT_MARGIN_COEF:-0.01}"
# The failed run used 1e-4 and reached 7-9x its KL target. A real one-epoch restore smoke measured
# KL=0.187 even at 3e-5, while 5e-6 measured KL=0.00521. This pilot therefore uses the measured safe
# value. It changes actor LR only; the separately owned critic retains its validated 1e-4 config.
export NAVRL_LEARNING_RATE="${NAVRL_LEARNING_RATE:-5e-6}"

export NAVRL_FIXED_BARS="${NAVRL_FIXED_BARS:-75}"
export NUM_ENVS="${NUM_ENVS:-128}"
export SEED="${SEED:-1}"
# 500 controlled pilot epochs. Continue only if capture remains competent, KL stays bounded and
# edge98_y no longer trends toward the previous 30%.
export MAX_EPOCHS="${MAX_EPOCHS:-19550}"
export AERIAL_RUN_TAG="${AERIAL_RUN_TAG:-action-squashed-v2-main-s${SEED}}"
export TRAIN_SESSION_LOG="${TRAIN_SESSION_LOG:-train_session_logs/action_squashed_v2_main_$(date +%y%m%d_%H%M%S).log}"

export CKPT="${CKPT:-runs/ppo_260727_2324_navrl/nn/last_gen_ppo_ep_19050_rew_31.79068.pth}"
if [[ ! -f "${CKPT}" ]]; then
    echo "[action_squashed_v2_main] checkpoint not found: ${CKPT}" >&2
    exit 2
fi

echo "[action_squashed_v2_main] tanh-squashed v2 | fixed 75 bars | seed=${SEED}"
echo "[action_squashed_v2_main] checkpoint=${CKPT} lr=${NAVRL_LEARNING_RATE} std=${NAVRL_ACTION_STD}"
echo "[action_squashed_v2_main] safety | log_ratio=+-${NAVRL_PPO_LOG_RATIO_CLAMP} kl_stop=${NAVRL_PPO_KL_STOP} lateral_margin=${NAVRL_LATENT_MARGIN_Y}@${NAVRL_LATENT_MARGIN_COEF}"
exec ./train_navrl_general_repr_density.sh "$@"
