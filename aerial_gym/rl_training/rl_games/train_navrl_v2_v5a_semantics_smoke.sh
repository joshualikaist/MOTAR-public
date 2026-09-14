#!/usr/bin/env bash
# Verification 5A: fresh 1,000-epoch engineering smoke for corrected episode semantics.
#
# This is deliberately NOT a performance experiment.  It checks that the exact 600-action
# horizon and rl_games time-limit bootstrap signal survive a real fresh PPO run without NaN/Inf,
# pathological KL, or checkpoint-provenance drift.  Representation, detector, action policy,
# environment, and curriculum remain the canonical v2-search baseline.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if (( $# != 0 )); then
    echo "[v5a] no CLI arguments are accepted; this is a closed fresh-run contract." >&2
    exit 2
fi
if [[ -n "${CKPT:-}" ]]; then
    echo "[v5a] refusing inherited CKPT: ${CKPT}" >&2
    exit 2
fi

# Preserve only the Python selection and the explicit no-GPU preflight switch.  Old interactive
# exports have repeatedly contaminated NavRL experiments, so clear every NavRL/Aerial experiment
# variable before pinning this arm.  The child launcher sets the complete canonical v2 contract.
V5A_PYTHON="${PYTHON:-}"
V5A_PREFLIGHT="${V5A_PREFLIGHT_ONLY:-0}"
while IFS= read -r name; do
    case "${name}" in
        NAVRL_*|AERIAL_RUN_TAG|AERIAL_GYM_SIM_NAME|TRAIN_SESSION_LOG|TRAIN_LIVE_LOG|MAX_EPOCHS|SEED|NUM_ENVS|FILE|TASK|GPU4GB|CKPT|ALLOW_CONCURRENT)
            unset "${name}"
            ;;
    esac
done < <(compgen -v)
if [[ -n "${V5A_PYTHON}" ]]; then
    export PYTHON="${V5A_PYTHON}"
fi

export MAX_EPOCHS=1000
export SEED=197
export NAVRL_V2_PROFILE=main
export AERIAL_RUN_TAG=v2-v5a-semantics-smoke-s197
export TRAIN_SESSION_LOG="train_session_logs/v5a_semantics_smoke_$(date +%y%m%d_%H%M%S).log"
export TRAIN_LIVE_LOG=train_session_logs/current_v5a_semantics_smoke.log
export NAVRL_V2_CONTRACT_PREFLIGHT_ONLY="${V5A_PREFLIGHT}"

echo "[v5a] scope=engineering-smoke (NO performance claim)"
echo "[v5a] fresh=1 seed=${SEED} epochs=${MAX_EPOCHS} checkpoint=none"
echo "[v5a] intervention=exact-600-actions + rl_games time_outs bootstrap"
echo "[v5a] controls=canonical-v2 analytic-detector cluster_sector squashed_gaussian governor-off pose-noise-off"

exec ./train_navrl_v2_search.sh
