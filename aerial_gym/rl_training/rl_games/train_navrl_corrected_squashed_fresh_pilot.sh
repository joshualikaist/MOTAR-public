#!/usr/bin/env bash
# Fresh bounded-action pilot on the corrected sensor/target-motion environment.
#
# Do not pass a checkpoint.  The legacy pilot showed that the corrected perception is accurate,
# but its unbounded Gaussian actor saturated at the task clamp.  This wrapper keeps the exact same
# 25-bar environment and switches only the action distribution plus its measured PPO safeguards.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export NAVRL_FRESH_BOUNDED_ACTOR=1
export MAX_EPOCHS="${MAX_EPOCHS:-500}"
export NUM_ENVS="${NUM_ENVS:-128}"
export SEED="${SEED:-1}"
export AERIAL_RUN_TAG="${AERIAL_RUN_TAG:-corrected-squashed-fresh-pilot-s${SEED}}"
export TRAIN_SESSION_LOG="${TRAIN_SESSION_LOG:-train_session_logs/corrected_squashed_fresh_pilot_$(date +%y%m%d_%H%M%S).log}"

echo "[corrected_squashed_fresh_pilot] fresh tanh-squashed pilot; no legacy checkpoint"
exec ./train_navrl_sensorfix_fresh_pilot.sh "$@"
