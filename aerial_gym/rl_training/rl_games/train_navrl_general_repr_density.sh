#!/usr/bin/env bash
# Stage C: density curriculum from the validated general-spawn FOV-240 representation policy.
#
# This is deliberately separate from the older train_navrl_vision_seq_density.sh.  The old launcher
# selects a different observation schema and did not pin the representation settings, so using it
# with the 898-D checkpoint either fails shape loading or silently evaluates the wrong policy input.
#
# Usage:
#   ./train_navrl_general_repr_density.sh
#   CKPT=runs/.../nn/last_gen_ppo_ep_XXXX.pth MAX_EPOCHS=45000 ./train_navrl_general_repr_density.sh
# Foreground (training output is visible here and also saved to a log):
#   CKPT=runs/.../nn/last_gen_ppo_ep_XXXX.pth MAX_EPOCHS=12000 \
#     NAVRL_OBSTACLE_SUPPRESS_DEG=15 ./train_navrl_corrected_squashed_density.sh
# Background (the stable current-training log remains directly watchable):
#   nohup env CKPT=runs/.../nn/last_gen_ppo_ep_XXXX.pth MAX_EPOCHS=45000 \
#     ./train_navrl_general_repr_density.sh > train_session_logs/night_density.out 2>&1 &
#   ./watch_navrl_training.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONNOUSERSITE=1

export NAVRL_VISION=1
export NAVRL_PERCEPTION=1
export NAVRL_GENERAL_TRAIN=1
export NAVRL_PERCEPTION_PERTURB=0
export NAVRL_TILT_COMP=1

# Policy-input contract of ppo_260727_0930. Keep all of these pinned together.
export NAVRL_MAX_OBSTACLES=8
export NAVRL_OBSTACLE_FOV_DEG=240
export NAVRL_OBSTACLE_SELECTOR="${NAVRL_OBSTACLE_SELECTOR:-greedy_suppress}"
# Keep 10 degrees as the validated default, but allow controlled same-shape ablations such as
# 15 degrees to pass through the launcher instead of being silently overwritten.
export NAVRL_OBSTACLE_SUPPRESS_DEG="${NAVRL_OBSTACLE_SUPPRESS_DEG:-10}"
export NAVRL_LIDAR_HBEAMS=72
export NAVRL_LIDAR_VBEAMS=4
export NAVRL_LIDAR_RANGE=12

export NAVRL_MAX_BARS=150
# A controlled A/B must hold the environment at the selected checkpoint density; otherwise the first
# policy to promote starts training on a different task and distribution effects become ambiguous.
# Normal Stage-C use leaves NAVRL_FIXED_BARS empty and retains the original curriculum.
NAVRL_FIXED_BARS="${NAVRL_FIXED_BARS:-}"
if [[ -n "${NAVRL_FIXED_BARS}" ]]; then
    if [[ ! "${NAVRL_FIXED_BARS}" =~ ^[0-9]+$ ]] || (( NAVRL_FIXED_BARS < 1 || NAVRL_FIXED_BARS > NAVRL_MAX_BARS )); then
        echo "[general_repr_density] NAVRL_FIXED_BARS must be an integer in 1..${NAVRL_MAX_BARS}" >&2
        exit 2
    fi
    export NAVRL_NUM_BARS="${NAVRL_FIXED_BARS}"
    # Keep the competence monitor active while capping final==current. This produces clean 16k
    # aggregate + stratified windows for an ablation without allowing a density promotion.
    export NAVRL_DENSITY_CURRICULUM=1
    export NAVRL_DENSITY_START="${NAVRL_FIXED_BARS}"
    export NAVRL_DENSITY_FINAL="${NAVRL_FIXED_BARS}"
else
    unset NAVRL_NUM_BARS  # an explicit value disables promotion even when the curriculum flag is on
    export NAVRL_DENSITY_CURRICULUM=1
    export NAVRL_DENSITY_START=25
    export NAVRL_DENSITY_FINAL="${NAVRL_DENSITY_FINAL:-110}"
fi
export NAVRL_DENSITY_STEP="${NAVRL_DENSITY_STEP:-5}"
export NAVRL_DENSITY_THRESHOLD="${NAVRL_DENSITY_THRESHOLD:-0.70}"
export NAVRL_DENSITY_WARMUP="${NAVRL_DENSITY_WARMUP:-1000}"
export NAVRL_DENSITY_CHECK_EPS="${NAVRL_DENSITY_CHECK_EPS:-16384}"
export NAVRL_DENSITY_STRATIFIED_GATE="${NAVRL_DENSITY_STRATIFIED_GATE:-0}"
export NAVRL_DENSITY_STRATIFIED_FLOOR="${NAVRL_DENSITY_STRATIFIED_FLOOR:-0.55}"
export NAVRL_DENSITY_STRATIFIED_MIN_EPS="${NAVRL_DENSITY_STRATIFIED_MIN_EPS:-512}"

export NAVRL_K_COMPETENCE=1
export NAVRL_K_FINAL=16
export NAVRL_K_MIN_FINAL=10
# General-spawn training samples a radial distance independently of the legacy left-to-right
# k-window. Pin that real contract explicitly so logs, checkpoints and gate strata agree.
export NAVRL_GENERAL_GOAL_DIST_MIN="${NAVRL_GENERAL_GOAL_DIST_MIN:-4}"
export NAVRL_GENERAL_GOAL_DIST_MAX="${NAVRL_GENERAL_GOAL_DIST_MAX:-16}"
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

# `latest` selects the newest non-derived periodic/final NavRL checkpoint after the duplicate-run
# guard succeeds. An explicit CKPT remains available for a deliberate branch from an older policy.
CKPT="${CKPT:-latest}"
MAX_EPOCHS="${MAX_EPOCHS:-45000}"
NUM_ENVS="${NUM_ENVS:-128}"
SEED="${SEED:-1}"

mkdir -p train_session_logs
if [[ "${ALLOW_CONCURRENT:-0}" != "1" ]]; then
    ACTIVE_PIDS="$(pgrep -f '[r]unner.py .*--task navrl_task .*--train' | tr '\n' ' ' || true)"
    if [[ -n "${ACTIVE_PIDS// }" ]]; then
        echo "[general_repr_density] refusing duplicate NavRL training; active PID(s): ${ACTIVE_PIDS}" >&2
        echo "[general_repr_density] monitor the active run instead, or set ALLOW_CONCURRENT=1 intentionally." >&2
        exit 3
    fi
    exec 9>train_session_logs/.general_repr_density.lock
    if ! flock -n 9; then
        echo "[general_repr_density] another density launcher holds the training lock." >&2
        exit 3
    fi
fi

if [[ "${CKPT}" == "latest" ]]; then
    LATEST_ENTRY="$(
        find runs -path '*/nn/last_gen_ppo_ep_*.pth' -type f ! -name '*_rlnorm.pth' \
            -printf '%T@ %p\n' | sort -nr | sed -n '1p'
    )"
    CKPT="${LATEST_ENTRY#* }"
    if [[ -z "${LATEST_ENTRY}" || -z "${CKPT}" ]]; then
        echo "[general_repr_density] no last_gen_ppo_ep_*.pth checkpoint found under runs/." >&2
        exit 2
    fi
    echo "[general_repr_density] auto-selected latest checkpoint: ${CKPT}"
fi

PREFLIGHT_ARGS=(
    "${CKPT}"
    --max-epochs "${MAX_EPOCHS}"
    # A fixed-density rehearsal may intentionally move down from a denser checkpoint. Validate
    # against the built physical capacity here; NAVRL_NUM_BARS remains the exact runtime contract.
    --density-final "$([[ -n "${NAVRL_FIXED_BARS}" ]] && echo "${NAVRL_MAX_BARS}" || echo "${NAVRL_DENSITY_FINAL}")"
    --min-k-max "${NAVRL_K_FINAL}"
    --min-task-steps "$((3000 * 32))"
)
case "${NAVRL_ALLOW_SUPPRESS_WARMSTART:-0}" in
    0) ;;
    1) PREFLIGHT_ARGS+=(--allow-contract-override cfg_obstacle_suppress_deg) ;;
    *)
        echo "[general_repr_density] NAVRL_ALLOW_SUPPRESS_WARMSTART must be 0 or 1." >&2
        exit 2
        ;;
esac
case "${NAVRL_ALLOW_SELECTOR_WARMSTART:-0}" in
    0) ;;
    1)
        PREFLIGHT_ARGS+=(
            --allow-contract-override cfg_obstacle_selector
            --allow-contract-override cfg_obstacle_cluster_gap_m
            --allow-contract-override cfg_obstacle_sectors
        )
        ;;
    *)
        echo "[general_repr_density] NAVRL_ALLOW_SELECTOR_WARMSTART must be 0 or 1." >&2
        exit 2
        ;;
esac
case "${NAVRL_ALLOW_CORRIDOR_WARMSTART:-0}" in
    0) ;;
    1)
        PREFLIGHT_ARGS+=(
            --allow-contract-override cfg_corridor_tokens
            --allow-contract-override cfg_corridor_horizon_m
            --allow-contract-override cfg_corridor_min_width_m
        )
        ;;
    *)
        echo "[general_repr_density] NAVRL_ALLOW_CORRIDOR_WARMSTART must be 0 or 1." >&2
        exit 2
        ;;
esac
"${PYTHON}" navrl_checkpoint_preflight.py "${PREFLIGHT_ARGS[@]}"

if [[ -n "${NAVRL_FIXED_BARS}" ]]; then
    echo "[general_repr_density] controlled A/B | fixed bars=${NAVRL_FIXED_BARS} (promotion capped; gate monitor on)"
else
    echo "[general_repr_density] Stage C | 25 -> ${NAVRL_DENSITY_FINAL} bars, step=${NAVRL_DENSITY_STEP}"
    echo "[general_repr_density] promotion | threshold=${NAVRL_DENSITY_THRESHOLD} \
check_eps=${NAVRL_DENSITY_CHECK_EPS} warmup=${NAVRL_DENSITY_WARMUP}"
fi
echo "[general_repr_density] general goals | radial=${NAVRL_GENERAL_GOAL_DIST_MIN}..${NAVRL_GENERAL_GOAL_DIST_MAX}m \
stratified_gate=${NAVRL_DENSITY_STRATIFIED_GATE} floor=${NAVRL_DENSITY_STRATIFIED_FLOOR}"
echo "[general_repr_density] representation | tokens=${NAVRL_MAX_OBSTACLES} \
fov=${NAVRL_OBSTACLE_FOV_DEG}deg selector=${NAVRL_OBSTACLE_SELECTOR} \
suppress=+-${NAVRL_OBSTACLE_SUPPRESS_DEG}deg \
cluster_gap=${NAVRL_OBSTACLE_CLUSTER_GAP_M:-0.45}m sectors=${NAVRL_OBSTACLE_SECTORS:-${NAVRL_MAX_OBSTACLES}} \
scan=${NAVRL_LIDAR_VBEAMS}x${NAVRL_LIDAR_HBEAMS} lidar=${NAVRL_LIDAR_RANGE}m"
echo "[general_repr_density] action | policy=${NAVRL_ACTION_POLICY:-legacy} \
std=${NAVRL_ACTION_STD:-learned} mu_scale=${NAVRL_ACTION_MU_SCALE:-1} \
entropy=${NAVRL_ENTROPY_COEF:-yaml} lr=${NAVRL_LEARNING_RATE:-yaml} \
diag=${NAVRL_ACTION_DIAG:-0}"
echo "[general_repr_density] safety | reward-collapse=off NaN/Inf=fail-fast \
log_ratio=${NAVRL_PPO_LOG_RATIO_CLAMP:-off} kl_stop=${NAVRL_PPO_KL_STOP:-off} \
margin_y=${NAVRL_LATENT_MARGIN_Y:-off}@${NAVRL_LATENT_MARGIN_COEF:-off} \
bias_y=${NAVRL_LATERAL_BIAS_COEF:-off} reflect=${NAVRL_REFLECTION_COEF:-off}"
echo "[general_repr_density] checkpoint=${CKPT} max_epochs=${MAX_EPOCHS} envs=${NUM_ENVS} seed=${SEED}"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "[general_repr_density] preflight-only check complete; training was not started."
    exit 0
fi

exec env NUM_ENVS="${NUM_ENVS}" ./train_navrl.sh \
    --checkpoint "${CKPT}" --branch_run --disable_collapse_early_stop \
    --max_epochs "${MAX_EPOCHS}" --seed "${SEED}" "$@"
