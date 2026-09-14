#!/usr/bin/env bash
# Preregistered 3-arm gate-passing rerun; see
# docs/prereg_2026-08-14_detector_coupling_binbias.md.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
POLICY="runs/ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1/nn/last_gen_ppo_ep_25000_rew_39.742134.pth"
POLICY_SHA="f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40"
DETECTOR="../../../artifacts/navrl_target_detector_v7_confirmatory.pth"
DETECTOR_SHA="85c7974bcd85c627170c5bd63030144d1c5dc2a11e5d64829cad38f615c5d5d7"
PROFILE_ROOT="../../../results/navrl_detector_v7_error_profile_seed419"
RESULT_ROOT="../../../results/navrl_detector_coupling_binbias_seed431"
PARAMS="${PROFILE_ROOT}/noise_params_binbias.env"
ARM_SEED=431
NOISE_SEED=9431
EPISODES="${EPISODES:-2049}"
PREFLIGHT="${PREFLIGHT:-0}"

for f in "${POLICY}" "${DETECTOR}" "${PROFILE_ROOT}/paired_errors.npz" \
         "${PROFILE_ROOT}/profile.json" "${PARAMS}"; do
    [[ -f "${f}" ]] || { echo "[binbias] missing ${f}" >&2; exit 2; }
done
"${PYTHON}" - "${POLICY}" "${POLICY_SHA}" "${DETECTOR}" "${DETECTOR_SHA}" <<'PY'
import hashlib, pathlib, sys
for path, expected in ((sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])):
    actual = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"[binbias] SHA mismatch {path}: {actual}")
PY

# Gate is evaluated before any arm and its threshold is fixed at ±10%.
mkdir -p "${RESULT_ROOT}"
"${PYTHON}" ../../../tools/verify_navrl_detector_noise_gate.py \
    --npz "${PROFILE_ROOT}/paired_errors.npz" \
    --profile-json "${PROFILE_ROOT}/profile.json" \
    --seed "${NOISE_SEED}" --tolerance 0.10 \
    --out "${RESULT_ROOT}/quality_gate.json"

export NAVRL_V2_DENSITIES=205
export NAVRL_V2_ACTION_MODE=deterministic
export NAVRL_EVAL_REFLECTION_MODE=original
export NAVRL_SPEED_GOVERNOR=riskcap
export NAVRL_SPEED_GOVERNOR_FIXED_MPS=2.0
export NAVRL_SPEED_GOVERNOR_FREE_MPS=3.53553390593
export NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M=0.45
export NAVRL_SPEED_GOVERNOR_MARGIN_M=0.45
export NAVRL_SPEED_GOVERNOR_SLOW_M=3.0
export NAVRL_SPEED_GOVERNOR_RELEASE_M=5.0
export NAVRL_SPEED_GOVERNOR_TTC_S=1.2
export NAVRL_SPEED_GOVERNOR_BRAKE_MPS2=2.9608856678
export NAVRL_SPEED_GOVERNOR_REACTION_S=0.1
export NAVRL_DETECTOR_THRESHOLD=0.55
export NAVRL_DETNOISE_SEED="${NOISE_SEED}"

run_arm() {
    local arm="$1" mode="$2" out="${RESULT_ROOT}/$1"
    echo "[binbias] arm=${arm} seed=${ARM_SEED}"
    [[ "${PREFLIGHT}" == "1" ]] && return 0
    if [[ -f "${out}/205bars.json" && -f "${out}/205bars.receipt.json" ]]; then
        echo "[binbias] SKIP complete ${out}"
        return 0
    fi
    [[ ! -e "${out}" ]] || { echo "[binbias] refusing partial ${out}" >&2; exit 2; }
    if [[ "${mode}" == "v7" ]]; then
        env -u NAVRL_V2_ALLOW_DETECTOR_THRESHOLD_MISMATCH \
            NAVRL_DETECTOR_CHECKPOINT="${DETECTOR}" \
            NAVRL_SEED="${ARM_SEED}" NAVRL_V2_RESULT_DIR="${out}" \
            ./eval_navrl_v2_density_sweep.sh "${POLICY}" "${EPISODES}"
    elif [[ "${mode}" == "clean" ]]; then
        env -u NAVRL_DETECTOR_CHECKPOINT -u NAVRL_V2_ALLOW_DETECTOR_THRESHOLD_MISMATCH \
            NAVRL_SEED="${ARM_SEED}" NAVRL_V2_RESULT_DIR="${out}" \
            ./eval_navrl_v2_density_sweep.sh "${POLICY}" "${EPISODES}"
    else
        # shellcheck disable=SC1090
        source "${PARAMS}"
        env -u NAVRL_DETECTOR_CHECKPOINT -u NAVRL_V2_ALLOW_DETECTOR_THRESHOLD_MISMATCH \
            NAVRL_SEED="${ARM_SEED}" NAVRL_V2_RESULT_DIR="${out}" \
            NAVRL_DETNOISE_BEARING_STD_RAD="${NAVRL_DETNOISE_BEARING_STD_RAD}" \
            NAVRL_DETNOISE_RANGE_STD_M="${NAVRL_DETNOISE_RANGE_STD_M}" \
            NAVRL_DETNOISE_RANGE_RHO="${NAVRL_DETNOISE_RANGE_RHO}" \
            NAVRL_DETNOISE_RANGE_BIAS_M="${NAVRL_DETNOISE_RANGE_BIAS_M}" \
            NAVRL_DETNOISE_RANGE_SIGMA_PROFILE="${NAVRL_DETNOISE_RANGE_SIGMA_PROFILE}" \
            NAVRL_DETNOISE_RANGE_BIAS_PROFILE="${NAVRL_DETNOISE_RANGE_BIAS_PROFILE}" \
            NAVRL_DETNOISE_DROPOUT_P01="${NAVRL_DETNOISE_DROPOUT_P01}" \
            NAVRL_DETNOISE_DROPOUT_P10="${NAVRL_DETNOISE_DROPOUT_P10}" \
            NAVRL_DETNOISE_SCALE=1.0 \
            ./eval_navrl_v2_density_sweep.sh "${POLICY}" "${EPISODES}"
    fi
}

run_arm analytic_clean clean
run_arm analytic_noise_1p0_binbias noise
run_arm learned_v7 v7

if [[ "${PREFLIGHT}" != "1" ]]; then
    "${PYTHON}" ../../../tools/analyse_navrl_detector_coupling_binbias.py "${RESULT_ROOT}"
fi
