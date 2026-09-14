#!/usr/bin/env bash
# NavRL++-Target generalized moving-target curriculum.
#
# Stage A (epoch 0..4000): 25 bars, distance 7 -> 16 m.
# Stage B (epoch 4000..): competence-gated density 25 -> 110 bars.  The final 10..16 m
# distance window remains active, but 25% of resets sample 5..10 m to retain close-range
# avoidance/reacquisition. Density is promoted only after >=65% capture over 4096 episodes.
#
# Sensor corruption remains off until detector/tracker validation. Target motion is enabled from
# the beginning and widens from a 0.25 m/s floor to a 0.75 m/s curriculum ceiling.
#
# Usage:
#   ./train_navrl_perception_staged.sh
#   ./train_navrl_perception_staged.sh --seed 2
#   ./train_navrl_perception_staged.sh --checkpoint runs/.../nn/last_gen_ppo_....pth
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTHONNOUSERSITE=1
export NAVRL_PERCEPTION=1
export NAVRL_VISION=1
export NAVRL_GENERAL_TRAIN=1
export NAVRL_MAX_BARS=150
export NAVRL_TARGET_SPEED_MIN=0.25
export NAVRL_TARGET_SPEED_FINAL=0.75

# Distance first. Goals remain inside the 20 m camera range.
export NAVRL_K_FINAL=16
export NAVRL_K_MIN_FINAL=10
export NAVRL_K_WARMUP=4000
export NAVRL_K_MIN_RAMP_START=2500
export NAVRL_K_MIN_RAMP_EPOCHS=1500

# Density starts only when the distance ramp has completed.
export NAVRL_DENSITY_CURRICULUM=1
export NAVRL_DENSITY_START=25
export NAVRL_DENSITY_FINAL=110
export NAVRL_DENSITY_STEP=5
export NAVRL_DENSITY_THRESHOLD=0.65
export NAVRL_DENSITY_WARMUP=4000
export NAVRL_DENSITY_CHECK_EPS=4096
export NAVRL_DENSITY_EASY_GOAL_MIX=0.25
export NAVRL_DENSITY_EASY_GOAL_MIN=5
export NAVRL_DENSITY_EASY_GOAL_MAX=10

MAX_EPOCHS="${MAX_EPOCHS:-10000}"
NUM_ENVS="${NUM_ENVS:-128}"

echo "[perception_staged] randomized drone/target spawn + distance 0..4000 -> density 25..110"
echo "[perception_staged] easy-distance replay=25% target-speed=0.25..0.75 epochs=${MAX_EPOCHS} envs=${NUM_ENVS}"

exec env NUM_ENVS="${NUM_ENVS}" ./train_navrl.sh --max_epochs "${MAX_EPOCHS}" "$@"
