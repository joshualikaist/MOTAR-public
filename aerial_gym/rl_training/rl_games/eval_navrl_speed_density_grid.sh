#!/usr/bin/env bash
# Evaluate whether low target/pursuer speed removes high-density crashes.
# Usage: NAVRL_VISION=1 NAVRL_PERCEPTION=1 ./eval_navrl_speed_density_grid.sh CKPT
set -euo pipefail
CALLER_PWD="${PWD}"
cd "$(dirname "${BASH_SOURCE[0]}")"

CKPT="${1:?usage: $0 CHECKPOINT}"
if [[ "${CKPT}" != /* ]]; then
  CKPT="${CALLER_PWD}/${CKPT}"
fi
GAMES="${PLAY_GAMES_NUM:-1000}"
DENSITIES="${DENSITIES:-110 130 150}"
TARGET_SPEEDS="${TARGET_SPEEDS:-0.0 0.5}"
PURSUER_SPEEDS="${PURSUER_SPEEDS:-0.75 1.0 1.5 2.0}"
EPISODE_LEN_STEPS="${EPISODE_LEN_STEPS:-300}"
HEADLESS="${HEADLESS:-True}"
NUM_ENVS="${NUM_ENVS:-128}"

if [[ -z "${PYTHON:-}" && -x /home/fair/miniconda3/envs/aerialgym/bin/python ]]; then
  export PYTHON=/home/fair/miniconda3/envs/aerialgym/bin/python
fi
PY="${PYTHON:-python}"
PY_BIN="$(command -v "${PY}")"
export PATH="$(dirname "${PY_BIN}"):${PATH}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[speed-density] checkpoint not found: ${CKPT}" >&2
  exit 2
fi
CKPT="$(readlink -f "${CKPT}")"
RUN_TAG="$(basename "$(dirname "$(dirname "${CKPT}")")")"
STAMP="$(date +%y%m%d_%H%M%S)"
RESULT_DIR="${RESULT_DIR:-train_session_logs/eval_results/speed_density_${RUN_TAG}_${STAMP}}"
mkdir -p "${RESULT_DIR}"
RESULT_CSV="${RESULT_DIR}/results.csv"
printf 'bars,target_speed,pursuer_limit,episodes,capture_rate,crash_rate,timeout_rate,bar_contact_rate,below_rate,json\n' \
  > "${RESULT_CSV}"

export NAVRL_VISION=1
export NAVRL_PERCEPTION=1
export NAVRL_TARGET_PATTERN="${NAVRL_TARGET_PATTERN:-mixed}"
export NAVRL_MAX_BARS="${NAVRL_MAX_BARS:-150}"
export NAVRL_GENERAL_TRAIN="${NAVRL_GENERAL_TRAIN:-1}"
export NAVRL_BULK_EVAL=1
export NAVRL_EVAL_CHECKPOINT="${CKPT}"
export NAVRL_ACTION_DIAG=1
export NAVRL_CRASH_DIAG=1

# SCALING-CRITICAL: must match the checkpoint's training values (lidar range is also the scan
# normalizer). Unset -> config defaults (4.0 / 2.5) -> wrong curve that looks like a bad policy.
export NAVRL_LIDAR_RANGE="${NAVRL_LIDAR_RANGE:-12}"
export NAVRL_YAW_RATE_MAX="${NAVRL_YAW_RATE_MAX:-3.0}"
export NAVRL_TILT_COMP="${NAVRL_TILT_COMP:-1}"
export NAVRL_MAX_OBSTACLES="${NAVRL_MAX_OBSTACLES:-8}"
export NAVRL_LIDAR_HBEAMS="${NAVRL_LIDAR_HBEAMS:-72}"
export NAVRL_LIDAR_VBEAMS="${NAVRL_LIDAR_VBEAMS:-4}"
export NAVRL_OBSTACLE_SUPPRESS_DEG="${NAVRL_OBSTACLE_SUPPRESS_DEG:-10}"
export NAVRL_OBSTACLE_FOV_DEG="${NAVRL_OBSTACLE_FOV_DEG:-240}"
# The pursuer-speed axis below sweeps NAVRL_MAX_VELOCITY, which ALSO rescales the ego-velocity
# observation. Pin the altitude-hold authority so it does NOT shrink with the swept speed, otherwise
# "slow pursuer crashes less" is confounded with "slow pursuer cannot hold altitude".
export NAVRL_ALT_HOLD_VMAX="${NAVRL_ALT_HOLD_VMAX:-2.5}"

echo "[speed-density] representation scan=${NAVRL_LIDAR_VBEAMS}x${NAVRL_LIDAR_HBEAMS} \
tokens=${NAVRL_MAX_OBSTACLES} fov=${NAVRL_OBSTACLE_FOV_DEG} \
suppress=+-${NAVRL_OBSTACLE_SUPPRESS_DEG} lidar=${NAVRL_LIDAR_RANGE}m"
echo "[speed-density] games/cell=${GAMES} envs=${NUM_ENVS} headless=${HEADLESS} results=${RESULT_DIR}"

for density in ${DENSITIES}; do
  for target_speed in ${TARGET_SPEEDS}; do
    for pursuer_speed in ${PURSUER_SPEEDS}; do
      echo "[speed-density] bars=${density} target=${target_speed} pursuer_limit=${pursuer_speed}"
      CELL_TAG="bars${density}_target${target_speed}_pursuer${pursuer_speed}"
      RESULT_JSON="${RESULT_DIR}/${CELL_TAG}.json"
      CELL_LOG="${RESULT_DIR}/${CELL_TAG}.log"
      NAVRL_NUM_BARS="${density}" NAVRL_TARGET_SPEED="${target_speed}" \
      NAVRL_MAX_VELOCITY="${pursuer_speed}" NAVRL_EPISODE_LEN_STEPS="${EPISODE_LEN_STEPS}" \
      PLAY_GAMES_NUM="${GAMES}" NAVRL_BULK_EVAL_JSON="${RESULT_JSON}" \
      NUM_ENVS="${NUM_ENVS}" HEADLESS="${HEADLESS}" \
        ./play_navrl.sh "${CKPT}" 2>&1 | tee "${CELL_LOG}"
      if [[ ! -s "${RESULT_JSON}" ]]; then
        echo "[speed-density] ERROR: cell finished without result JSON: ${RESULT_JSON}" >&2
        exit 3
      fi
      "${PY}" - "${RESULT_JSON}" "${RESULT_CSV}" <<'PY'
import csv
import json
import sys

source, destination = sys.argv[1:3]
with open(source, encoding="utf-8") as stream:
    result = json.load(stream)
condition = result["condition"]
outcome = result["outcome"]
causes = result["crash_causes"]
episodes = int(result["actual_episodes"])
row = [
    condition["bars"],
    condition["target_speed_mps"],
    condition["pursuer_max_speed_mps"],
    episodes,
    outcome["capture_rate"],
    outcome["crash_rate"],
    outcome["timeout_rate"],
    causes["bar_contact"] / max(1, episodes),
    causes["below"] / max(1, episodes),
    source,
]
with open(destination, "a", newline="", encoding="utf-8") as stream:
    csv.writer(stream).writerow(row)
print(
    "[speed-density] RESULT "
    f"n={episodes} capture={100 * outcome['capture_rate']:.2f}% "
    f"crash={100 * outcome['crash_rate']:.2f}% "
    f"timeout={100 * outcome['timeout_rate']:.2f}%"
)
PY
    done
  done
done
echo "[speed-density] done -> ${RESULT_CSV}"
