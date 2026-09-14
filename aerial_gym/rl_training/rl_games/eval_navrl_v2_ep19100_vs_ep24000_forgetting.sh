#!/usr/bin/env bash
# Inference-only 2x2 forgetting audit: ep19100 vs ep24000 at 205 bars.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
OLD_CKPT="runs/ppo_260801_1235_navrl_v2-recover-curriculum-s1/nn/last_gen_ppo_ep_19100_rew_43.892498.pth"
NEW_CKPT="runs/ppo_260802_0020_navrl_v2-recover-curriculum-continue-s1/nn/last_gen_ppo_ep_24000_rew_44.73549.pth"
OLD_SHA="82d1eeac1798b4b465274551ec2363fb377be781c0a58e930577ddb822f55044"
NEW_SHA="82f7978b42d9d9e95adcc638a40ae85fb3736fd897ae39cb8aa8333be39cf23f"
RESULT_ROOT="${NAVRL_FORGETTING_RESULT_ROOT:-${SCRIPT_DIR}/../../../results/navrl_v2_ep19100_vs_ep24000_forgetting}"

check_checkpoint() {
    local checkpoint="$1"
    local expected_sha="$2"
    if [[ ! -f "${checkpoint}" ]]; then
        echo "[forgetting] checkpoint not found: ${checkpoint}" >&2
        exit 2
    fi
    local actual_sha
    actual_sha="$(sha256sum "${checkpoint}" | awk '{print $1}')"
    if [[ "${actual_sha}" != "${expected_sha}" ]]; then
        echo "[forgetting] checkpoint SHA-256 mismatch: ${checkpoint}" >&2
        exit 2
    fi
}

check_checkpoint "${OLD_CKPT}" "${OLD_SHA}"
check_checkpoint "${NEW_CKPT}" "${NEW_SHA}"
if [[ -e "${RESULT_ROOT}" ]]; then
    echo "[forgetting] refusing to overwrite existing result root: ${RESULT_ROOT}" >&2
    exit 2
fi
mkdir -p "${RESULT_ROOT}"

run_cell() {
    local epoch="$1"
    local checkpoint="$2"
    local condition="$3"
    local output="${RESULT_ROOT}/ep${epoch}_${condition}"
    echo "======== epoch ${epoch} · ${condition} · 205 bars ========"
    if [[ "${condition}" == "fast1p5" ]]; then
        NAVRL_V2_FIXED_TARGET_SPEED=1.5 \
        NAVRL_V2_DENSITIES=205 NAVRL_SEED=42 \
        NAVRL_V2_ACTION_MODE=deterministic NAVRL_EVAL_REFLECTION_MODE=original \
        NAVRL_V2_RESULT_DIR="${output}" \
            ./eval_navrl_v2_density_sweep.sh "${checkpoint}" 2049
    else
        env -u NAVRL_V2_FIXED_TARGET_SPEED -u NAVRL_TARGET_SPEED \
        NAVRL_V2_DENSITIES=205 NAVRL_SEED=42 \
        NAVRL_V2_ACTION_MODE=deterministic NAVRL_EVAL_REFLECTION_MODE=original \
        NAVRL_V2_RESULT_DIR="${output}" \
            ./eval_navrl_v2_density_sweep.sh "${checkpoint}" 2049
    fi
}

echo "[forgetting] evaluation only; frozen weights; no training"
echo "[forgetting] ep19100(pre-205) vs ep24000(after 4,900 epochs at 205)"
for CONDITION in uniform fast1p5; do
    run_cell 19100 "${OLD_CKPT}" "${CONDITION}"
    run_cell 24000 "${NEW_CKPT}" "${CONDITION}"
done

"${PYTHON}" "${SCRIPT_DIR}/../../../tools/summarize_navrl_v2_forgetting.py" "${RESULT_ROOT}"
echo "[forgetting] done | ${RESULT_ROOT}/summary.md"
