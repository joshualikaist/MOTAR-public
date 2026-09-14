#!/usr/bin/env bash
# Candidate #4 historical fresh-run recipe: 8 tokens + 72-beam obstacle representation.
#
# The combined change improved bar_contact, but the original capacity/quantization explanation was
# not supported by the first-generation probe. Probe v2 now uses range+bearing GT association and
# FOV-conditioned denominators; do not infer which sub-knob caused the gain from the old token_err.
#
# All three change the observation layout (574 -> 898), so this is a FRESH run: no warm-start is
# possible from any existing checkpoint. Everything else matches train_navrl_general_12m_lookahead.sh
# (the current best recipe, capture 0.856), so the representation is the only changed variable.
#
# Usage:
#   ./train_navrl_general_repr.sh
#   MAX_EPOCHS=8000 NUM_ENVS=128 ./train_navrl_general_repr.sh
#
# Watch (crashdiag prints every ~2048 episodes):
#   NavRL barprobe v2 hit_token_given_fov -- range+bearing match, conditioned on selectable hits
#   NavRL barprobe v2 unique/duplicate    -- whether suppression wastes slots on the same bar
#   NavRL crashdiag bar_contact           -- the actual safety outcome
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

# --- the three representation changes (the ONLY difference from the 12 m recipe) ---
export NAVRL_MAX_OBSTACLES="${NAVRL_MAX_OBSTACLES:-8}"
export NAVRL_OBSTACLE_SUPPRESS_DEG="${NAVRL_OBSTACLE_SUPPRESS_DEG:-10}"
export NAVRL_LIDAR_HBEAMS="${NAVRL_LIDAR_HBEAMS:-72}"

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
export NAVRL_BAR_PROBE=1   # so the token/crowding forensics are logged throughout training
export NAVRL_OOB_PROBE=0

MAX_EPOCHS="${MAX_EPOCHS:-8000}"
NUM_ENVS="${NUM_ENVS:-128}"
SEED="${SEED:-1}"

echo "[general_repr] FRESH run (obs 574 -> 898; no warm-start possible)"
echo "[general_repr] tokens=${NAVRL_MAX_OBSTACLES} fov=${NAVRL_OBSTACLE_FOV_DEG:-360}deg \
suppress=+-${NAVRL_OBSTACLE_SUPPRESS_DEG}deg hbeams=${NAVRL_LIDAR_HBEAMS} \
lidar=${NAVRL_LIDAR_RANGE}m"
echo "[general_repr] max_epochs=${MAX_EPOCHS} envs=${NUM_ENVS} seed=${SEED}"

exec env NUM_ENVS="${NUM_ENVS}" ./train_navrl.sh --max_epochs "${MAX_EPOCHS}" --seed "${SEED}"
