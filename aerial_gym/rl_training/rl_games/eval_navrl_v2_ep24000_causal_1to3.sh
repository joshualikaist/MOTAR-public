#!/usr/bin/env bash
# Frozen-weight causal checks 1--3 preregistered in RESEARCH_PLAN.md §8.8.
# This launches evaluation only. It never invokes a training launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
CKPT="${1:-${SCRIPT_DIR}/runs/ppo_260802_0020_navrl_v2-recover-curriculum-continue-s1/nn/last_gen_ppo_ep_24000_rew_44.73549.pth}"
EXPECTED_SHA="82f7978b42d9d9e95adcc638a40ae85fb3736fd897ae39cb8aa8333be39cf23f"
OUT_ROOT="${NAVRL_CAUSAL_RESULT_ROOT:-${REPO_ROOT}/results/navrl_v2_ep24000_causal_1to3}"

if [[ ! -f "${CKPT}" ]]; then
    echo "[causal-1to3] checkpoint not found: ${CKPT}" >&2
    exit 2
fi
ACTUAL_SHA="$(${PYTHON} - "${CKPT}" <<'PY'
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], "rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(chunk)
print(h.hexdigest())
PY
)"
if [[ "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]]; then
    echo "[causal-1to3] frozen checkpoint SHA mismatch: ${ACTUAL_SHA}" >&2
    exit 2
fi
if [[ -e "${OUT_ROOT}" ]]; then
    echo "[causal-1to3] refusing to overwrite existing result root: ${OUT_ROOT}" >&2
    exit 2
fi

for MODE in original conjugate; do
    RESULT_DIR="${OUT_ROOT}/mirror_${MODE}"
    NAVRL_V2_DENSITIES=205 \
    NAVRL_SEED=42 \
    NAVRL_V2_ACTION_MODE=deterministic \
    NAVRL_EVAL_REFLECTION_MODE="${MODE}" \
    NAVRL_REFLECTION_DIAG_JSON="${RESULT_DIR}/reflection_actions.json" \
    NAVRL_V2_RESULT_DIR="${RESULT_DIR}" \
        "${SCRIPT_DIR}/eval_navrl_v2_density_sweep.sh" "${CKPT}" 4096
done

SEED43_DIR="${OUT_ROOT}/seed43"
NAVRL_V2_DENSITIES=205 \
NAVRL_SEED=43 \
NAVRL_V2_ACTION_MODE=deterministic \
NAVRL_EVAL_REFLECTION_MODE=original \
NAVRL_V2_RESULT_DIR="${SEED43_DIR}" \
    "${SCRIPT_DIR}/eval_navrl_v2_density_sweep.sh" "${CKPT}" 2049

# Outcome-side complement to the exact action pairs. This instrumentation-only replay records the
# initial actor-frame target bearing and must reproduce the original arm's aggregate counts exactly.
BEARING_DIR="${OUT_ROOT}/bearing_original"
NAVRL_V2_DENSITIES=205 \
NAVRL_SEED=42 \
NAVRL_V2_ACTION_MODE=deterministic \
NAVRL_EVAL_REFLECTION_MODE=original \
NAVRL_V2_RESULT_DIR="${BEARING_DIR}" \
    "${SCRIPT_DIR}/eval_navrl_v2_density_sweep.sh" "${CKPT}" 4096

"${PYTHON}" "${REPO_ROOT}/tools/summarize_navrl_v2_causal_1to3.py" \
    --root "${OUT_ROOT}" \
    --seed42 "${REPO_ROOT}/results/navrl_v2_ep24000_action_deterministic/205bars.json"

echo "[causal-1to3] complete: ${OUT_ROOT}/summary.md"
