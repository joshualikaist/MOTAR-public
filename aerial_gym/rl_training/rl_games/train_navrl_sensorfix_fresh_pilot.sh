#!/usr/bin/env bash
# Fresh 25-bar pilot for the corrected LiDAR-bearing, camera far-plane and target-motion contract.
#
# This run is intentionally short and causal: it checks the corrected observation/environment
# before density curriculum or an alternative action distribution is introduced.  Old checkpoints
# were trained with mirrored obstacle bearings (and, at 12 m, a camera far-plane phantom wall), so
# warm-starting one would invalidate the pilot.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONNOUSERSITE=1

# Refuse accidental resume flags and inherited checkpoint settings.
if [[ -n "${CKPT:-}" ]]; then
    echo "[sensorfix_fresh_pilot] CKPT must be unset: this is a fresh run." >&2
    exit 2
fi
for arg in "$@"; do
    case "${arg}" in
        --checkpoint|--checkpoint=*|--resume_in_place|--branch_run)
            echo "[sensorfix_fresh_pilot] resume/checkpoint flags are forbidden: ${arg}" >&2
            exit 2
            ;;
    esac
done

export NAVRL_VISION=1
export NAVRL_PERCEPTION=1
export NAVRL_GENERAL_TRAIN=1
export NAVRL_PERCEPTION_PERTURB=0
export NAVRL_TILT_COMP=1

# Corrected policy-input contract.  Keep these values explicit rather than inheriting a shell.
export NAVRL_MAX_OBSTACLES=8
export NAVRL_OBSTACLE_FOV_DEG=240
export NAVRL_OBSTACLE_SUPPRESS_DEG=10
export NAVRL_LIDAR_HBEAMS=72
export NAVRL_LIDAR_VBEAMS=4
export NAVRL_LIDAR_RANGE=12

# Fixed sparse arena: this pilot diagnoses the sensor/environment fixes, not density promotion.
export NAVRL_MAX_BARS=150
export NAVRL_NUM_BARS=25
export NAVRL_DENSITY_CURRICULUM=0
export NAVRL_K_COMPETENCE=1
export NAVRL_K_FINAL=16
export NAVRL_K_MIN_FINAL=10
export NAVRL_FOV_CURRICULUM_EPOCHS=1000000

export NAVRL_MAX_VELOCITY=2.5
export NAVRL_ALT_HOLD_VMAX=2.5
export NAVRL_YAW_RATE_MAX=3.0
export NAVRL_TARGET_SPEED_MIN=0.0
export NAVRL_TARGET_SPEED_FINAL=1.5
export NAVRL_TARGET_PATTERN=mixed
unset NAVRL_TARGET_SPEED

export NAVRL_OOB_MARGIN=1.0
export NAVRL_CRASH_DIAG=1
export NAVRL_BAR_PROBE=1
export NAVRL_OOB_PROBE=0
export NAVRL_ACTION_DIAG=1

# Default to the causal legacy baseline.  The dedicated bounded wrapper opts into the only other
# supported mode; arbitrary inherited action settings are never trusted.
if [[ "${NAVRL_FRESH_BOUNDED_ACTOR:-0}" == "1" ]]; then
    export NAVRL_ACTION_POLICY=squashed_gaussian
    export NAVRL_ACTION_STD=0.35,0.35,0.05,0.08
    export NAVRL_ACTION_MU_SCALE=1.0,0.4,1.0,1.0
    export NAVRL_ENTROPY_COEF=0.0
    export NAVRL_RESET_ACTOR_OPTIMIZER=0
    export NAVRL_LEARNING_RATE=1e-4
    export NAVRL_PPO_LOG_RATIO_CLAMP=10.0
    export NAVRL_PPO_KL_STOP=0.04
    export NAVRL_LATENT_MARGIN_Y=1.25
    export NAVRL_LATENT_MARGIN_COEF=0.01
    unset NAVRL_LATERAL_BIAS_COEF NAVRL_REFLECTION_COEF NAVRL_TRUNCATED_DMIN
else
    export NAVRL_ACTION_POLICY=legacy
    unset NAVRL_ACTION_STD NAVRL_ACTION_MU_SCALE NAVRL_ENTROPY_COEF
    unset NAVRL_RESET_ACTOR_OPTIMIZER NAVRL_LEARNING_RATE
    unset NAVRL_PPO_LOG_RATIO_CLAMP NAVRL_PPO_KL_STOP
    unset NAVRL_LATENT_MARGIN_Y NAVRL_LATENT_MARGIN_COEF
    unset NAVRL_LATERAL_BIAS_COEF NAVRL_REFLECTION_COEF NAVRL_TRUNCATED_DMIN
fi

MAX_EPOCHS="${MAX_EPOCHS:-500}"
NUM_ENVS="${NUM_ENVS:-128}"
SEED="${SEED:-1}"
export AERIAL_RUN_TAG="${AERIAL_RUN_TAG:-sensorfix-fresh-pilot-s${SEED}}"
export TRAIN_SESSION_LOG="${TRAIN_SESSION_LOG:-train_session_logs/sensorfix_fresh_pilot_$(date +%y%m%d_%H%M%S).log}"

mkdir -p train_session_logs
if [[ "${ALLOW_CONCURRENT:-0}" != "1" ]]; then
    ACTIVE_PIDS="$(pgrep -f '[r]unner.py .*--task navrl_task .*--train' | tr '\n' ' ' || true)"
    if [[ -n "${ACTIVE_PIDS// }" ]]; then
        echo "[sensorfix_fresh_pilot] refusing duplicate NavRL training; active PID(s): ${ACTIVE_PIDS}" >&2
        exit 3
    fi
    exec 9>train_session_logs/.sensorfix_fresh_pilot.lock
    if ! flock -n 9; then
        echo "[sensorfix_fresh_pilot] another fresh-pilot launcher holds the lock." >&2
        exit 3
    fi
fi

echo "[sensorfix_fresh_pilot] FRESH 25-bar causal pilot; checkpoint restore is disabled"
echo "[sensorfix_fresh_pilot] representation | tokens=${NAVRL_MAX_OBSTACLES} fov=${NAVRL_OBSTACLE_FOV_DEG}deg suppress=+-${NAVRL_OBSTACLE_SUPPRESS_DEG}deg scan=${NAVRL_LIDAR_VBEAMS}x${NAVRL_LIDAR_HBEAMS} lidar=${NAVRL_LIDAR_RANGE}m"
echo "[sensorfix_fresh_pilot] target | pattern=${NAVRL_TARGET_PATTERN} speed ceiling 0->${NAVRL_TARGET_SPEED_FINAL}m/s over 3000 epochs"
echo "[sensorfix_fresh_pilot] action | policy=${NAVRL_ACTION_POLICY} diagnostic=on std=${NAVRL_ACTION_STD:-learned} mu_scale=${NAVRL_ACTION_MU_SCALE:-1}"
echo "[sensorfix_fresh_pilot] safety | reward-collapse=off NaN/Inf=fail-fast log_ratio=${NAVRL_PPO_LOG_RATIO_CLAMP:-off} kl_stop=${NAVRL_PPO_KL_STOP:-off} margin_y=${NAVRL_LATENT_MARGIN_Y:-off}@${NAVRL_LATENT_MARGIN_COEF:-off}"
echo "[sensorfix_fresh_pilot] max_epochs=${MAX_EPOCHS} envs=${NUM_ENVS} seed=${SEED}"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "[sensorfix_fresh_pilot] preflight-only check complete; training was not started."
    exit 0
fi

exec env NUM_ENVS="${NUM_ENVS}" ./train_navrl.sh \
    --disable_collapse_early_stop --max_epochs "${MAX_EPOCHS}" --seed "${SEED}" "$@"
