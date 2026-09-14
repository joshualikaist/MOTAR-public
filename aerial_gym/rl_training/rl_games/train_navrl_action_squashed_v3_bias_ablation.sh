#!/usr/bin/env bash
# REJECTED 2026-07-28: signed batch-mean penalty shrank magnitude without creating left/right
# state dependence. Kept only to reproduce the ablation; this is not a main-training launcher.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ "${ALLOW_REJECTED_ABLATION:-0}" != "1" ]]; then
  echo "[action_squashed_v3_bias_ablation] rejected ablation; set ALLOW_REJECTED_ABLATION=1 only to reproduce it." >&2
  exit 3
fi

export NAVRL_ACTION_POLICY="squashed_gaussian"
export NAVRL_ACTION_STD="${NAVRL_ACTION_STD:-0.35,0.35,0.05,0.08}"
export NAVRL_ACTION_MU_SCALE="${NAVRL_ACTION_MU_SCALE:-1.0,0.4,1.0,1.0}"
export NAVRL_ENTROPY_COEF="${NAVRL_ENTROPY_COEF:-0.0}"
export NAVRL_RESET_ACTOR_OPTIMIZER="${NAVRL_RESET_ACTOR_OPTIMIZER:-0}"
export NAVRL_ACTION_DIAG="${NAVRL_ACTION_DIAG:-1}"

export NAVRL_PPO_LOG_RATIO_CLAMP="${NAVRL_PPO_LOG_RATIO_CLAMP:-10.0}"
export NAVRL_PPO_KL_STOP="${NAVRL_PPO_KL_STOP:-0.04}"
export NAVRL_LATENT_MARGIN_Y="${NAVRL_LATENT_MARGIN_Y:-1.25}"
export NAVRL_LATENT_MARGIN_COEF="${NAVRL_LATENT_MARGIN_COEF:-0.01}"
# The measured v2 action was +y in 100% of samples at every tested density. Squared
# minibatch-mean(mu_y) is zero for balanced [-m,+m] avoidance regardless of its magnitude.
export NAVRL_LATERAL_BIAS_COEF="${NAVRL_LATERAL_BIAS_COEF:-0.002}"
export NAVRL_LEARNING_RATE="${NAVRL_LEARNING_RATE:-5e-6}"

export NAVRL_FIXED_BARS="${NAVRL_FIXED_BARS:-75}"
export NUM_ENVS="${NUM_ENVS:-128}"
export SEED="${SEED:-1}"
export MAX_EPOCHS="${MAX_EPOCHS:-19850}"
export AERIAL_RUN_TAG="${AERIAL_RUN_TAG:-action-squashed-v3-bias-ablation-s${SEED}}"
export TRAIN_SESSION_LOG="${TRAIN_SESSION_LOG:-train_session_logs/action_squashed_v3_bias_ablation_$(date +%y%m%d_%H%M%S).log}"

export CKPT="${CKPT:-runs/ppo_260728_1958_navrl_action-squashed-v2-main-s1/nn/last_gen_ppo_ep_19550_rew_33.75813.pth}"
if [[ ! -f "${CKPT}" ]]; then
    echo "[action_squashed_v3_bias_ablation] checkpoint not found: ${CKPT}" >&2
    exit 2
fi

echo "[action_squashed_v3_bias_ablation] fixed 75 bars | seed=${SEED} | 300-epoch pilot"
echo "[action_squashed_v3_bias_ablation] checkpoint=${CKPT} lr=${NAVRL_LEARNING_RATE} std=${NAVRL_ACTION_STD}"
echo "[action_squashed_v3_bias_ablation] bias_y=${NAVRL_LATERAL_BIAS_COEF} margin=${NAVRL_LATENT_MARGIN_Y}@${NAVRL_LATENT_MARGIN_COEF}"
exec ./train_navrl_general_repr_density.sh "$@"
