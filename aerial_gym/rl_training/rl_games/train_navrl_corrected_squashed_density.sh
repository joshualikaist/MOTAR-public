#!/usr/bin/env bash
# P6B: continue the validated corrected/squashed policy through 25 -> 110 bars.
#
# The 500-epoch pilot passed a 5-speed held-out gate before this launcher was introduced.
# Keep the bounded action contract explicit: silently falling back to the legacy Gaussian would
# recreate the lateral boundary-collapse failure.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export CKPT="${CKPT:-runs/ppo_260729_2225_navrl_corrected-squashed-fresh-pilot-s1/nn/last_gen_ppo_ep_500_rew_102.53619.pth}"
export MAX_EPOCHS="${MAX_EPOCHS:-8000}"
export NUM_ENVS="${NUM_ENVS:-128}"
export SEED="${SEED:-1}"
export AERIAL_RUN_TAG="${AERIAL_RUN_TAG:-corrected-squashed-density25to110-s${SEED}}"
export TRAIN_SESSION_LOG="${TRAIN_SESSION_LOG:-train_session_logs/corrected_squashed_density_$(date +%y%m%d_%H%M%S).log}"

export NAVRL_DENSITY_FINAL="${NAVRL_DENSITY_FINAL:-110}"
export NAVRL_DENSITY_STEP="${NAVRL_DENSITY_STEP:-5}"
export NAVRL_DENSITY_THRESHOLD="${NAVRL_DENSITY_THRESHOLD:-0.70}"
export NAVRL_DENSITY_WARMUP="${NAVRL_DENSITY_WARMUP:-1000}"
export NAVRL_DENSITY_CHECK_EPS="${NAVRL_DENSITY_CHECK_EPS:-16384}"
if [[ "${NAVRL_CONTROLLED_ABLATION:-0}" != "1" ]]; then
    unset NAVRL_FIXED_BARS NAVRL_NUM_BARS
fi

export NAVRL_ACTION_POLICY=squashed_gaussian
export NAVRL_ACTION_STD=0.35,0.35,0.05,0.08
export NAVRL_ACTION_MU_SCALE=1.0,0.4,1.0,1.0
export NAVRL_ENTROPY_COEF=0.0
export NAVRL_RESET_ACTOR_OPTIMIZER=0
export NAVRL_LEARNING_RATE="${NAVRL_LEARNING_RATE:-1e-4}"
export NAVRL_PPO_LOG_RATIO_CLAMP=10.0
export NAVRL_PPO_KL_STOP=0.04
export NAVRL_LATENT_MARGIN_Y=1.25
export NAVRL_LATENT_MARGIN_COEF=0.01
export NAVRL_ACTION_DIAG=1
unset NAVRL_LATERAL_BIAS_COEF NAVRL_REFLECTION_COEF NAVRL_TRUNCATED_DMIN

if [[ "${NAVRL_CONTROLLED_ABLATION:-0}" == "1" ]]; then
    echo "[corrected_squashed_density] controlled bounded-policy ablation | fixed bars=${NAVRL_FIXED_BARS}"
else
    echo "[corrected_squashed_density] validated bounded checkpoint -> P6B 25..${NAVRL_DENSITY_FINAL}"
fi
echo "[corrected_squashed_density] actual-vehicle motiondiag enabled; watch commanded_stall"
exec ./train_navrl_general_repr_density.sh "$@"
