#!/usr/bin/env bash
# Inference-only fixed-target-speed causal evaluation for the frozen ep24000 policy.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
CKPT="${NAVRL_FIXED_SPEED_CKPT:-runs/ppo_260802_0020_navrl_v2-recover-curriculum-continue-s1/nn/last_gen_ppo_ep_24000_rew_44.73549.pth}"
EXPECTED_SHA256="82f7978b42d9d9e95adcc638a40ae85fb3736fd897ae39cb8aa8333be39cf23f"
RESULT_ROOT="${NAVRL_FIXED_SPEED_RESULT_ROOT:-${SCRIPT_DIR}/../../../results/navrl_v2_ep24000_fixed_speed}"

if [[ ! -f "${CKPT}" ]]; then
    echo "[fixed-speed] checkpoint not found: ${CKPT}" >&2
    exit 2
fi
ACTUAL_SHA256="$(sha256sum "${CKPT}" | awk '{print $1}')"
if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
    echo "[fixed-speed] checkpoint SHA-256 mismatch: ${ACTUAL_SHA256}" >&2
    exit 2
fi
if [[ -e "${RESULT_ROOT}" ]]; then
    echo "[fixed-speed] refusing to overwrite existing result root: ${RESULT_ROOT}" >&2
    exit 2
fi
mkdir -p "${RESULT_ROOT}"

echo "[fixed-speed] evaluation only; frozen weights; no training"
echo "[fixed-speed] checkpoint=${CKPT} sha256=${EXPECTED_SHA256}"
echo "[fixed-speed] condition=205 bars seed=42 deterministic original speeds=0.3,0.9,1.5"

for SPEED in 0.3 0.9 1.5; do
    SPEED_TAG="${SPEED/./p}"
    echo "======== fixed target speed ${SPEED} m/s ========"
    NAVRL_V2_FIXED_TARGET_SPEED="${SPEED}" \
    NAVRL_V2_DENSITIES=205 \
    NAVRL_SEED=42 \
    NAVRL_V2_ACTION_MODE=deterministic \
    NAVRL_EVAL_REFLECTION_MODE=original \
    NAVRL_V2_RESULT_DIR="${RESULT_ROOT}/speed_${SPEED_TAG}" \
        ./eval_navrl_v2_density_sweep.sh "${CKPT}" 2049
done

"${PYTHON}" "${SCRIPT_DIR}/../../../tools/summarize_navrl_v2_fixed_speed.py" "${RESULT_ROOT}"
echo "[fixed-speed] done | ${RESULT_ROOT}/summary.md"
