#!/usr/bin/env bash
# Candidate #2: extend the obstacle look-ahead (LiDAR 8 -> 12 m) to attack bar_contact (~13% of
# episodes = the dominant crash cause; below/oob are now both < 2.5% and untouched by altitude work).
# Warm-starts from the tilt-comp general-spawn policy (1052). Everything else is IDENTICAL to
# train_navrl_general_8m_finetune.sh so LiDAR range is the ONLY changed variable.
#
# Usage:
#   ./train_navrl_general_12m_lookahead.sh
#   CKPT=runs/.../nn/gen_ppo.pth MAX_EPOCHS=7000 ./train_navrl_general_12m_lookahead.sh
#
# NOTE: changing NAVRL_LIDAR_RANGE changes the static-scan normalization (scan/range), so the
# warm-start needs a re-adaptation window -- expect a transient capture dip that recovers over a few
# hundred epochs. Watch crashdiag `bar_contact` share: if it does NOT fall, look-ahead is not the
# limiter (8 m = 3.2 s at 2.5 m/s is already generous for a single dodge) and the next lever is
# obstacle-token capacity (MAX_OBSTACLES 5->8, candidate 4), not more range.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONNOUSERSITE=1

export NAVRL_VISION=1
export NAVRL_PERCEPTION=1
export NAVRL_GENERAL_TRAIN=1
export NAVRL_PERCEPTION_PERTURB=0
export NAVRL_TILT_COMP=1

# The ONE changed variable vs the 8 m recipe: obstacle look-ahead 8 -> 12 m.
export NAVRL_LIDAR_RANGE=12
export NAVRL_MAX_BARS=150
export NAVRL_NUM_BARS=25
export NAVRL_DENSITY_CURRICULUM=0
export NAVRL_K_COMPETENCE=1
export NAVRL_K_FINAL=16
export NAVRL_K_MIN_FINAL=10
export NAVRL_FOV_CURRICULUM_EPOCHS=1000000
export NAVRL_MAX_VELOCITY=2.5
export NAVRL_ALT_HOLD_VMAX=2.5
export NAVRL_YAW_RATE_MAX=3.0
export NAVRL_TARGET_SPEED_MIN=0.0
export NAVRL_TARGET_SPEED_FINAL=1.5
export NAVRL_TARGET_PATTERN=mixed
unset NAVRL_TARGET_SPEED

export NAVRL_OOB_MARGIN=1.0
export NAVRL_CRASH_DIAG=1
export NAVRL_OOB_PROBE=0

CKPT="${CKPT:-runs/ppo_260724_1052_navrl/nn/gen_ppo.pth}"
MAX_EPOCHS="${MAX_EPOCHS:-7000}"
NUM_ENVS="${NUM_ENVS:-128}"
SEED="${SEED:-1}"

if [[ ! -f "${CKPT}" ]]; then
    echo "checkpoint not found: ${CKPT}" >&2
    exit 1
fi

echo "[general_12m] LiDAR look-ahead 12m (only change vs 8m), warm-start from tilt-comp policy"
echo "[general_12m] checkpoint=${CKPT} max_epochs=${MAX_EPOCHS} envs=${NUM_ENVS} seed=${SEED}"

exec env NUM_ENVS="${NUM_ENVS}" ./train_navrl.sh \
    --checkpoint "${CKPT}" --branch_run --max_epochs "${MAX_EPOCHS}" --seed "${SEED}"
