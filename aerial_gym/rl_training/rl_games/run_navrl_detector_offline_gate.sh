#!/usr/bin/env bash
# Gate 3 stage A: offline detector dataset -> candidate training -> validation calibration -> test.
# This does not load or train PPO and does not launch navigation evaluation unless the offline gate
# later passes and the user explicitly runs the separate A/B launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PYTHON PYTHONNOUSERSITE=1
PYTHON_BIN="$(cd "$(dirname "${PYTHON}")" && pwd)"
export PATH="${PYTHON_BIN}:${PATH}"

OUTPUT="artifacts/navrl_target_detector_v2.pth"
RESULT_ROOT="results/navrl_detector_offline_gate_v2"
RUN_LOG="results/navrl_detector_offline_gate_v2.run.log"

ARGS=(
    --output "${OUTPUT}"
    --result-root "${RESULT_ROOT}"
    --train-frames 8192
    --validation-frames 2048
    --test-frames 4096
    --num-envs 64
    --epochs 10
    --batch-size 128
)

if [[ "${PREFLIGHT:-0}" == "1" ]]; then
    "${PYTHON}" tools/train_navrl_target_detector_v2.py "${ARGS[@]}" --preflight
    exit 0
fi

mkdir -p results
echo "[detector-gate] train/validation/test=8192/2048/4096 seeds=71/73/79"
echo "[detector-gate] PPO=frozen/not-loaded candidates=balanced_bce,focal_dice"
echo "[detector-gate] log=${RUN_LOG} summary=${RESULT_ROOT}/summary.md"
"${PYTHON}" tools/train_navrl_target_detector_v2.py "${ARGS[@]}" 2>&1 | tee "${RUN_LOG}"
