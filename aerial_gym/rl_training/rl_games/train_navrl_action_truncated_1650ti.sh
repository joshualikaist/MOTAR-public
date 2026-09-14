#!/usr/bin/env bash
# 4-GB/GTX-1650-Ti bounded-action experiment.
#
# Uses the same checkpoint, seed, density, PPO batch and fixed base std as the main experiment, but
# samples from the scale-adjusted truncated Gaussian proposed for PPO at AAAI 2025. This is a real
# alternative hypothesis, not merely a second std setting.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v python >/dev/null 2>&1 && [[ -z "${PYTHON:-}" ]]; then
    echo "[action_truncated_1650ti] activate the aerialgym conda environment first." >&2
    exit 2
fi
export PYTHON="${PYTHON:-$(command -v python)}"
export GPU4GB=1
export AERIAL_GYM_SIM_NAME="${AERIAL_GYM_SIM_NAME:-base_sim_4gb}"

export NAVRL_ACTION_POLICY="truncated_gaussian"
export NAVRL_ACTION_STD="${NAVRL_ACTION_STD:-0.35,0.35,0.05,0.08}"
export NAVRL_ACTION_MU_SCALE="${NAVRL_ACTION_MU_SCALE:-1.0,0.4,1.0,1.0}"
export NAVRL_TRUNCATED_DMIN="${NAVRL_TRUNCATED_DMIN:-0.01}"
export NAVRL_ENTROPY_COEF="${NAVRL_ENTROPY_COEF:-0.0}"
export NAVRL_RESET_ACTOR_OPTIMIZER="${NAVRL_RESET_ACTOR_OPTIMIZER:-1}"
export NAVRL_ACTION_DIAG="${NAVRL_ACTION_DIAG:-1}"

export NAVRL_FIXED_BARS="${NAVRL_FIXED_BARS:-75}"
export NUM_ENVS="${NUM_ENVS:-128}"
export SEED="${SEED:-1}"
export MAX_EPOCHS="${MAX_EPOCHS:-22050}"  # same 3000 epochs / interactions as main
export AERIAL_RUN_TAG="${AERIAL_RUN_TAG:-action-truncated-1650ti-s${SEED}}"
export TRAIN_SESSION_LOG="${TRAIN_SESSION_LOG:-train_session_logs/action_truncated_1650ti_$(date +%y%m%d_%H%M%S).log}"

export CKPT="${CKPT:-runs/ppo_260727_2324_navrl/nn/last_gen_ppo_ep_19050_rew_31.79068.pth}"
if [[ ! -f "${CKPT}" ]]; then
    echo "[action_truncated_1650ti] checkpoint not found: ${CKPT}" >&2
    echo "Copy the ep19050 checkpoint from the main machine and pass CKPT=/path/to/file." >&2
    exit 2
fi

echo "[action_truncated_1650ti] scale-adjusted truncated Gaussian | fixed 75 bars | seed=${SEED}"
echo "[action_truncated_1650ti] checkpoint=${CKPT} std=${NAVRL_ACTION_STD} \
mu_scale=${NAVRL_ACTION_MU_SCALE} d_min=${NAVRL_TRUNCATED_DMIN}"
exec ./train_navrl_general_repr_density.sh "$@"
