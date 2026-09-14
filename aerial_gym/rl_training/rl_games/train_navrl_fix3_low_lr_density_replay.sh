#!/usr/bin/env bash
# FIX3 adaptation pilot: low-LR blockwise density rehearsal from the best pre-FIX3 checkpoint.
#
# The asset manager exposes one global active-obstacle count, so a physically honest replay cannot
# mix bar counts per environment in one PPO batch. Instead, keep optimizer/model state continuous
# across short fixed-density blocks: 85 -> 90 -> 95 -> 100 bars. Keep the full 200-epoch
# 100-bar exposure in ONE process so the same-density capture guard retains its rolling peak.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONNOUSERSITE=1

START_CKPT="${CKPT:-runs/ppo_260730_2111_navrl_cluster-sector-density85to110-v2-s1/nn/last_gen_ppo_ep_12500_rew_-0.342975.pth}"
if [[ ! -f "${START_CKPT}" ]]; then
    echo "[fix3-replay] checkpoint not found: ${START_CKPT}" >&2
    exit 2
fi

START_EPOCH="$(
    "${PYTHON}" - "${START_CKPT}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(checkpoint["epoch"]))
PY
)"
if [[ ! "${START_EPOCH}" =~ ^[0-9]+$ ]]; then
    echo "[fix3-replay] invalid checkpoint epoch: ${START_EPOCH}" >&2
    exit 2
fi

export NUM_ENVS="${NUM_ENVS:-128}"
export SEED="${SEED:-1}"
export NAVRL_LEARNING_RATE="${NAVRL_LEARNING_RATE:-3e-5}"
export NAVRL_RESET_ACTOR_OPTIMIZER="${NAVRL_RESET_ACTOR_OPTIMIZER:-0}"
export NAVRL_RESET_DENSITY_WINDOW=1
export NAVRL_DENSITY_RESUME_WARMUP=0
export NAVRL_DENSITY_GUARD_WINDOW_EPOCHS="${NAVRL_DENSITY_GUARD_WINDOW_EPOCHS:-20}"
export NAVRL_DENSITY_GUARD_MIN_EPOCHS="${NAVRL_DENSITY_GUARD_MIN_EPOCHS:-40}"
export NAVRL_DENSITY_GUARD_MIN_PEAK="${NAVRL_DENSITY_GUARD_MIN_PEAK:-0.55}"
export NAVRL_DENSITY_GUARD_DROP="${NAVRL_DENSITY_GUARD_DROP:-0.10}"
export NAVRL_DENSITY_GUARD_PATIENCE="${NAVRL_DENSITY_GUARD_PATIENCE:-10}"

# Override for a smoke test, e.g. REPLAY_BARS="85" REPLAY_EPOCHS="50".
read -r -a REPLAY_BAR_LIST <<< "${REPLAY_BARS:-85 90 95 100}"
read -r -a REPLAY_EPOCH_LIST <<< "${REPLAY_EPOCHS:-100 100 100 200}"
if (( ${#REPLAY_BAR_LIST[@]} == 0 || ${#REPLAY_BAR_LIST[@]} != ${#REPLAY_EPOCH_LIST[@]} )); then
    echo "[fix3-replay] REPLAY_BARS and REPLAY_EPOCHS must contain equally many entries." >&2
    exit 2
fi

SESSION_LOG="${TRAIN_SESSION_LOG:-train_session_logs/fix3_low_lr_density_replay_$(date +%y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "${SESSION_LOG}")"
exec > >(tee -a "${SESSION_LOG}") 2>&1

echo "[fix3-replay] source=${START_CKPT} epoch=${START_EPOCH}"
echo "[fix3-replay] schedule bars=${REPLAY_BAR_LIST[*]} epochs=${REPLAY_EPOCH_LIST[*]}"
echo "[fix3-replay] lr=${NAVRL_LEARNING_RATE} optimizer_reset=${NAVRL_RESET_ACTOR_OPTIMIZER}"
echo "[fix3-replay] guard drop=${NAVRL_DENSITY_GUARD_DROP} window=${NAVRL_DENSITY_GUARD_WINDOW_EPOCHS} patience=${NAVRL_DENSITY_GUARD_PATIENCE}"

current_ckpt="${START_CKPT}"
current_epoch="${START_EPOCH}"
for index in "${!REPLAY_BAR_LIST[@]}"; do
    bars="${REPLAY_BAR_LIST[$index]}"
    epochs="${REPLAY_EPOCH_LIST[$index]}"
    if [[ ! "${bars}" =~ ^[0-9]+$ ]] || (( bars < 1 || bars > 150 )); then
        echo "[fix3-replay] invalid bars at block $((index + 1)): ${bars}" >&2
        exit 2
    fi
    if [[ ! "${epochs}" =~ ^[0-9]+$ ]] || (( epochs < 50 || epochs % 50 != 0 )); then
        echo "[fix3-replay] block epochs must be a positive multiple of 50: ${epochs}" >&2
        exit 2
    fi
    end_epoch=$((current_epoch + epochs))
    tag="fix3-low-lr-replay-b$((index + 1))-${bars}bars-s${SEED}"
    block_log="train_session_logs/${tag}_$(date +%y%m%d_%H%M%S).log"

    echo "[fix3-replay] block $((index + 1))/${#REPLAY_BAR_LIST[@]} | bars=${bars} epoch=${current_epoch}->${end_epoch}"
    env \
        CKPT="${current_ckpt}" \
        MAX_EPOCHS="${end_epoch}" \
        NAVRL_CONTROLLED_ABLATION=1 \
        NAVRL_FIXED_BARS="${bars}" \
        AERIAL_RUN_TAG="${tag}" \
        TRAIN_SESSION_LOG="${block_log}" \
        ./train_navrl_corrected_squashed_density_cluster_sector.sh

    next_ckpt="$(
        find runs -path "*/nn/last_gen_ppo_ep_${end_epoch}_rew_*.pth" -type f \
            ! -name '*_rlnorm.pth' -printf '%T@ %p\n' |
            sort -nr |
            sed -n '1s/^[^ ]* //p'
    )"
    if [[ -z "${next_ckpt}" || ! -f "${next_ckpt}" ]]; then
        echo "[fix3-replay] block ${index} did not produce epoch ${end_epoch}; stopping instead of chaining an ambiguous checkpoint." >&2
        exit 4
    fi
    current_ckpt="${next_ckpt}"
    current_epoch="${end_epoch}"
    echo "[fix3-replay] block complete | checkpoint=${current_ckpt}"
done

printf '%s\n' "${current_ckpt}" > train_session_logs/fix3_low_lr_density_replay_latest_checkpoint.txt
echo "[fix3-replay] COMPLETE | final_checkpoint=${current_ckpt}"
echo "[fix3-replay] next: evaluate this checkpoint with eval_navrl_cluster_sector_density_sweep.sh"
