#!/usr/bin/env bash
# Stage G0: teach the altitude-PI perception+Transformer checkpoint the required randomized
# drone/target spawn distribution WITHOUT changing LiDAR range yet.
#
# Usage:
#   ./train_navrl_general_8m_finetune.sh
#   CKPT=runs/.../nn/gen_ppo.pth MAX_EPOCHS=5000 ./train_navrl_general_8m_finetune.sh
#
# The source checkpoint is near epoch 3800, so max_epochs=5000 adds about 1200 epochs. --branch_run
# guarantees that this experiment gets a new run folder instead of contaminating the PI source run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONNOUSERSITE=1

export NAVRL_VISION=1
export NAVRL_PERCEPTION=1
export NAVRL_GENERAL_TRAIN=1
export NAVRL_PERCEPTION_PERTURB=0

# One learning change only: randomized spawn generalization. Preserve the verified 8 m PI task.
export NAVRL_LIDAR_RANGE=8
export NAVRL_MAX_BARS=150
export NAVRL_NUM_BARS=25
export NAVRL_DENSITY_CURRICULUM=0
export NAVRL_K_COMPETENCE=1
export NAVRL_K_FINAL=16
export NAVRL_K_MIN_FINAL=10
export NAVRL_FOV_CURRICULUM_EPOCHS=1000000
export NAVRL_MAX_VELOCITY=2.5
# Pinned explicitly (not inherited from the caller's shell) so this recipe is reproducible and
# so it stays a true one-variable comparison against train_navrl_general_12m_lookahead.sh.
export NAVRL_TILT_COMP=1
export NAVRL_ALT_HOLD_VMAX=2.5
export NAVRL_YAW_RATE_MAX=3.0
export NAVRL_TARGET_SPEED_MIN=0.0
export NAVRL_TARGET_SPEED_FINAL=1.5
export NAVRL_TARGET_PATTERN=mixed
unset NAVRL_TARGET_SPEED

# A 1 m tolerance is justified by the paired checkpoint-only sweep: it recovers transient
# generalized-spawn exits, while the probe remains available to measure residual OOB.
export NAVRL_OOB_MARGIN=1.0
export NAVRL_CRASH_DIAG=1
export NAVRL_OOB_PROBE=0

CKPT="${CKPT:-runs/ppo_260724_0110_navrl/nn/gen_ppo.pth}"
MAX_EPOCHS="${MAX_EPOCHS:-5000}"
NUM_ENVS="${NUM_ENVS:-128}"
SEED="${SEED:-1}"

if [[ ! -f "${CKPT}" ]]; then
    echo "checkpoint not found: ${CKPT}" >&2
    exit 1
fi

echo "[general_8m] randomized drone/target spawn, 25 bars, LiDAR=8m, OOB margin=1m"
echo "[general_8m] checkpoint=${CKPT} max_epochs=${MAX_EPOCHS} envs=${NUM_ENVS} seed=${SEED}"

exec env NUM_ENVS="${NUM_ENVS}" ./train_navrl.sh \
    --checkpoint "${CKPT}" --branch_run --max_epochs "${MAX_EPOCHS}" --seed "${SEED}"
