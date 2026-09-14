#!/usr/bin/env bash
# 4-1 bottleneck: is the v7 detector's -5.19 pp navigation cost the POLICY being coupled to the
# analytic detector's output statistics, or a property of v7's outputs?
#
# Preregistered in docs/prereg_2026-08-13_detector_coupling.md BEFORE any measurement. Two stages:
#
#   profile  seed 419 -- run v7 alongside the analytic detector on identical frames and record its
#            error. The analytic head drives; v7 only observes, so the trajectory is the one the
#            noise arms will be evaluated on.
#   arms     seed 409 -- five cells with the frozen policy: analytic clean, analytic + v7-shaped
#            noise at 0.5x / 1.0x / 1.5x, and real v7.
#
# The dose ladder exists because a single point match is weak evidence: what we want to see is the
# curve passing through -5.19 pp at 1.0x, with monotone rungs either side. Two extra cells, ~10 min.
#
# The whole contract is exported ONCE here. A previous profiling attempt was launched by hand and
# silently ran with the speed governor off, which is a different trajectory distribution; that run
# is preserved as results/navrl_detector_v7_error_profile_seed401_VOID_governor_off. Do not invoke
# eval_navrl_v2_density_sweep.sh directly for this experiment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PYTHON
POLICY="runs/ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1/nn/last_gen_ppo_ep_25000_rew_39.742134.pth"
POLICY_SHA="f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40"
DETECTOR="../../../artifacts/navrl_target_detector_v7_confirmatory.pth"
DETECTOR_SHA="85c7974bcd85c627170c5bd63030144d1c5dc2a11e5d64829cad38f615c5d5d7"
PROFILE_ROOT="../../../results/navrl_detector_v7_error_profile_seed419"
ARMS_ROOT="../../../results/navrl_detector_coupling_probe_seed409"
PROFILE_SEED=419
ARM_SEED=409
EPISODES="${EPISODES:-2049}"
PREFLIGHT="${PREFLIGHT:-0}"

STAGE="${1:-all}"
if [[ ! "${STAGE}" =~ ^(profile|arms|all)$ ]]; then
    echo "usage: $0 [profile|arms|all]" >&2
    exit 2
fi

if [[ ! -f "${POLICY}" || ! -f "${DETECTOR}" ]]; then
    echo "[coupling] pinned policy or detector is missing" >&2
    exit 2
fi
"${PYTHON}" - "${POLICY}" "${POLICY_SHA}" "${DETECTOR}" "${DETECTOR_SHA}" <<'PY'
import hashlib
from pathlib import Path
import sys
for raw, expected in ((sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])):
    actual = hashlib.sha256(Path(raw).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"[coupling] SHA mismatch for {raw}: {actual}")
PY

# ---- the contract, exported once (prereg section 3) ------------------------------------------
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
# Threshold is pinned at the checkpoint's own 0.55 for EVERY arm. Verification D2 (seeds 191/193)
# already showed the threshold is not the cause -- v7@0.70 minus v7@0.55 was +1.44 pp with a CI
# spanning zero -- so re-tuning it here would reintroduce the confound the prereg forbids.
export NAVRL_DETECTOR_THRESHOLD=0.55
export NAVRL_DETNOISE_SEED=9409

# ---- profiling stage --------------------------------------------------------------------------
run_profile() {
    local out="${PROFILE_ROOT}"
    echo "[coupling] profile | seed=${PROFILE_SEED} v7-vs-analytic on identical frames"
    if [[ "${PREFLIGHT}" == "1" ]]; then return 0; fi
    if [[ -f "${out}/paired_errors.npz" ]]; then
        echo "[coupling] SKIP complete ${out}"
        return 0
    fi
    if [[ -e "${out}" ]]; then
        echo "[coupling] refusing partial ${out}; inspect/move it manually" >&2
        exit 2
    fi
    mkdir -p "${out}"
    # No NAVRL_DETECTOR_CHECKPOINT: the ANALYTIC head drives. The v7 head is loaded separately as
    # the profile-only head, which never touches the tracker, the map or the observation.
    env -u NAVRL_DETECTOR_CHECKPOINT -u NAVRL_V2_ALLOW_DETECTOR_THRESHOLD_MISMATCH \
        NAVRL_SEED="${PROFILE_SEED}" \
        NAVRL_V2_RESULT_DIR="${out}/cell" \
        NAVRL_DETPROFILE_CHECKPOINT="${DETECTOR}" \
        NAVRL_DETPROFILE_OUT="${out}/paired_errors.npz" \
        NAVRL_DETPROFILE_MAX_STEPS=4000 \
        ./eval_navrl_v2_density_sweep.sh "${POLICY}" "${EPISODES}"
}

# ---- arm stage --------------------------------------------------------------------------------
# Noise parameters come from the profile; analyse_navrl_detector_coupling.py writes them here.
NOISE_ENV="${PROFILE_ROOT}/noise_params.env"

run_arm() {
    local arm="$1" scale="$2" use_v7="$3"
    local out="${ARMS_ROOT}/${arm}"
    echo "[coupling] arm=${arm} scale=${scale} v7=${use_v7}"
    if [[ "${PREFLIGHT}" == "1" ]]; then return 0; fi
    if [[ -f "${out}/205bars.json" && -f "${out}/205bars.receipt.json" ]]; then
        echo "[coupling] SKIP complete ${out}"
        return 0
    fi
    if [[ -e "${out}" ]]; then
        echo "[coupling] refusing partial cell ${out}; inspect/move it manually" >&2
        exit 2
    fi

    if [[ "${use_v7}" == "1" ]]; then
        env -u NAVRL_V2_ALLOW_DETECTOR_THRESHOLD_MISMATCH \
            NAVRL_DETECTOR_CHECKPOINT="${DETECTOR}" \
            NAVRL_SEED="${ARM_SEED}" NAVRL_V2_RESULT_DIR="${out}" \
            ./eval_navrl_v2_density_sweep.sh "${POLICY}" "${EPISODES}"
    elif [[ "${scale}" == "0" ]]; then
        env -u NAVRL_DETECTOR_CHECKPOINT -u NAVRL_V2_ALLOW_DETECTOR_THRESHOLD_MISMATCH \
            NAVRL_SEED="${ARM_SEED}" NAVRL_V2_RESULT_DIR="${out}" \
            ./eval_navrl_v2_density_sweep.sh "${POLICY}" "${EPISODES}"
    else
        # shellcheck disable=SC1090
        source "${NOISE_ENV}"
        env -u NAVRL_DETECTOR_CHECKPOINT -u NAVRL_V2_ALLOW_DETECTOR_THRESHOLD_MISMATCH \
            NAVRL_SEED="${ARM_SEED}" NAVRL_V2_RESULT_DIR="${out}" \
            NAVRL_DETNOISE_BEARING_STD_RAD="${NAVRL_DETNOISE_BEARING_STD_RAD}" \
            NAVRL_DETNOISE_RANGE_STD_M="${NAVRL_DETNOISE_RANGE_STD_M}" \
            NAVRL_DETNOISE_RANGE_RHO="${NAVRL_DETNOISE_RANGE_RHO}" \
            NAVRL_DETNOISE_RANGE_BIAS_M="${NAVRL_DETNOISE_RANGE_BIAS_M}" \
            NAVRL_DETNOISE_RANGE_SIGMA_PROFILE="${NAVRL_DETNOISE_RANGE_SIGMA_PROFILE}" \
            NAVRL_DETNOISE_DROPOUT_P01="${NAVRL_DETNOISE_DROPOUT_P01}" \
            NAVRL_DETNOISE_DROPOUT_P10="${NAVRL_DETNOISE_DROPOUT_P10}" \
            NAVRL_DETNOISE_SCALE="${scale}" \
            ./eval_navrl_v2_density_sweep.sh "${POLICY}" "${EPISODES}"
    fi
}

if [[ "${STAGE}" == "profile" || "${STAGE}" == "all" ]]; then
    run_profile
fi

if [[ "${STAGE}" == "arms" || "${STAGE}" == "all" ]]; then
    if [[ ! -f "${NOISE_ENV}" && "${PREFLIGHT}" != "1" ]]; then
        echo "[coupling] ${NOISE_ENV} missing -- run tools/analyse_navrl_detector_coupling.py profile first" >&2
        exit 2
    fi
    mkdir -p "${ARMS_ROOT}"
    run_arm analytic_clean      0    0
    run_arm analytic_noise_0p5  0.5  0
    run_arm analytic_noise_1p0  1.0  0
    run_arm analytic_noise_1p5  1.5  0
    run_arm learned_v7          0    1
fi

echo "[coupling] stage ${STAGE} complete"
