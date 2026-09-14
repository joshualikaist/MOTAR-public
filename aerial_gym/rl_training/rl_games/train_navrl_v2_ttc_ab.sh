#!/usr/bin/env bash
# TTC selector A/B -- both arms, one script, on the 4 GB card.
#
#   ARM=baseline ./train_navrl_v2_ttc_ab.sh     # cluster_sector (bearing-ranked tokens)
#   ARM=ttc      ./train_navrl_v2_ttc_ab.sh     # ttc_sector     (threat-ranked tokens)
#
# WHY ONE SCRIPT FOR BOTH ARMS
# ----------------------------
# Everything except NAVRL_OBSTACLE_SELECTOR is set below the ARM switch, so the two arms cannot
# drift apart. This project has already lost hours twice to a launcher that silently changed a
# second variable (a hardcoded NAVRL_CONTROLLED_ABLATION that blocked density promotion; an
# unconditional `unset NAVRL_NUM_BARS` that reverted a fixed-density probe), and an A/B is exactly
# where that class of bug is invisible: both arms still run, they just answer different questions.
#
# WHAT IS BEING TESTED
# --------------------
# cluster_sector allocates the 8 token slots by BEARING (one cluster per forward sector). The crash
# probe on run ppo_260731_1722 (2300 epochs, 70 bars) measured where that proxy breaks:
#   * 23.6% of the bars actually STRUCK were outside the 240-degree token window entirely
#     (hit_fov = 0.764) -- while searching, the drone yaws hard, so a bar leaves the window while
#     the velocity vector still points at it;
#   * of the hits inside the window, 11.7% still had no token (hit_token_given_fov = 0.883),
#     because a sector reserves only its single nearest cluster.
# ttc_sector ranks by closing time instead (range / closing speed), so a bar is tokenized because
# the drone is moving into it, not because of where it sits. Clustering is shared verbatim, so this
# isolates the RANKING, not the grouping. Observation stays 8x12 -> both arms warm-start from the
# same checkpoint with no surgery.
#
# HONEST PRIOR: representation changes have a poor track record here. Token capacity 5->8 was
# REJECTED (coverage fell 0.647 -> 0.40-0.53), beams 36->72 was REJECTED (token error rose
# 0.57 -> 0.72-1.13 m), and corridor tokens missed their gate (+1.57pp vs +3pp required). The one
# intervention that did work was narrowing the selection FOV 360->240, which is also a "what do we
# pick" change -- the same family as this one. Expect a small effect and measure it coldly.
#
# PREREGISTERED GATE (decide BEFORE looking at results)
# -----------------------------------------------------
# Compare the two arms by HELD-OUT evaluation at the same fixed density, not by training curves.
# ttc_sector advances only if BOTH hold:
#     capture(ttc) - capture(baseline) >= +2.0 pp
#     crash(ttc)   - crash(baseline)   <= -2.0 pp
# Ties or a capture gain paid for with more crashes = rejected, recorded, and the arm is frozen.
# Judge at the FINAL checkpoint. The corridor A/B dipped mid-run (0.6420 -> 0.5993 -> 0.6610)
# before recovering, and ttc_sector additionally has to re-adapt to a changed input distribution
# while the baseline arm does not -- an asymmetry that penalises it early by construction.
#
# EVAL (per arm, after both finish)
#   GPU4GB=1 NAVRL_V2_DENSITIES=70 \
#     ./eval_navrl_v2_density_sweep.sh runs/<arm run>/nn/last_gen_ppo_ep_XXXX.pth
#   The evaluator derives and verifies the selector and 4 GB runtime contract from the checkpoint.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if (( $# != 0 )); then
    echo "[ttc-ab] extra runner arguments are forbidden because they can invalidate the A/B contract: $*" >&2
    exit 2
fi

ARM="${ARM:-}"
case "${ARM}" in
    baseline) SELECTOR=cluster_sector ;;
    ttc)      SELECTOR=ttc_sector ;;
    *)
        echo "usage: ARM=baseline|ttc $0 [extra runner args]" >&2
        echo "  baseline = cluster_sector (bearing-ranked)   ttc = ttc_sector (threat-ranked)" >&2
        exit 2
        ;;
esac

# ---- THE experimental variable, and the only difference between the arms ----
export NAVRL_OBSTACLE_SELECTOR="${SELECTOR}"

# ---- shared: fixed density, so no promotion can occur mid-comparison ----
# 70 bars because the warm-start checkpoint is a converged 70-bar policy and the crash floor was
# characterised there (13.1% extrapolated). A promotion during either arm would change the task
# under one arm and not the other.
export NAVRL_DENSITY_CURRICULUM=0
export NAVRL_NUM_BARS=70
export NAVRL_MAX_BARS=300          # keep the PhysX actor count at the validated 4 GB profile
export NAVRL_DENSITY_MIN_EPOCHS=0  # inert with the curriculum off; pinned so a stray export cannot revive it

# ---- shared: identical start point and budget ----
export CKPT="${CKPT:-runs/ppo_260731_1940_navrl_v2-search-thr80-s1/nn/last_gen_ppo_ep_3250_rew_128.5965.pth}"
export NAVRL_V2_ALLOW_RESUME=1
export SEED="${SEED:-1}"
# Budget matched in SAMPLES, not in epoch count. The warm-start checkpoint comes off the 3070 at
# 128 envs (32x128 = 4096 samples/epoch); this card runs 64 envs (32x64 = 2048), so an epoch here
# is worth half of one there. "+1000 epochs" would have delivered 2.05M steps -- half the intended
# adaptation. +2000 epochs = 4.1M steps = the 1000 3070-equivalent epochs actually meant.
# This is not a neutral shortfall: the baseline arm starts already converged for its selector while
# the ttc arm starts off-distribution, so a short budget penalises only the arm that needs to adapt.
export MAX_EPOCHS=5250   # 3250 + 2000 @ 2048 samples/ep = 4.1M steps of adaptation
export AERIAL_RUN_TAG="v2-ttc-${ARM}-s${SEED}"
export TRAIN_SESSION_LOG="train_session_logs/v2_ttc_${ARM}_$(date +%y%m%d_%H%M%S)_$$.log"
export TRAIN_LIVE_LOG=train_session_logs/current_training.log
export AERIAL_GYM_SIM_NAME=base_sim_4gb
export NUM_ENVS=64
export HEADLESS=True
export NAVRL_SEED="${SEED}"
export FILE=ppo_navrl_perception_transformer.yaml
export TASK=navrl_task
unset ALLOW_CONCURRENT NAVRL_NETWORK_OVERRIDE
unset NAVRL_GENERAL_EVAL NAVRL_INTERACTIVE NAVRL_BULK_EVAL NAVRL_BULK_EVAL_JSON
unset NAVRL_GENERAL_RESULTS_JSON NAVRL_EVAL_CHECKPOINT NAVRL_EVAL_TARGET_SPEED_FINAL
unset NAVRL_EVAL_FULL_DISTRIBUTION
unset NAVRL_EVAL_RUN_NONCE NAVRL_EVAL_PROFILE NAVRL_SIM_PHYSICS_CONTRACT
unset NAVRL_LEGACY_VISION NAVRL_OOB_PROBE
# This is a selector A/B, not a recovery descendant. Clear stale recovery provenance before the
# common continuation launcher serializes the environment into its output checkpoint.
unset NAVRL_RECOVERY_STAGE NAVRL_RECOVERY_SOURCE_EPOCH
unset NAVRL_RECOVERY_SOURCE_SHA256 NAVRL_RECOVERY_SMOKE_REQUIRED_EPOCHS
unset NAVRL_RECOVERY_SMOKE_BARS NAVRL_RECOVERY_EVAL_ATTESTATION_SHA256
unset NAVRL_RECOVERY_EVAL_ATTESTATION_B64
export NAVRL_TARGET_SPEED_RAMP_EPOCHS=300
export NAVRL_OBSTACLE_TTC_IDLE_S=30.0
export NAVRL_OBSTACLE_TTC_MIN_SPEED=0.15
export NAVRL_CORRIDOR_HORIZON_M=6.0
export NAVRL_CORRIDOR_MIN_WIDTH_M=0.55
export NAVRL_FOV_CURRICULUM_EPOCHS=3000
export NAVRL_DETECTOR_MIN_PIXELS=2
export NAVRL_DETECTOR_THRESHOLD=0.55
unset NAVRL_DETECTOR_CHECKPOINT
export NAVRL_DETECTION_DROPOUT=0.3
export NAVRL_RGB_NOISE_STD=0.015
export NAVRL_DEPTH_NOISE_STD=0.02
export NAVRL_MAX_TILT_DEG=45.0

# Freeze optimizer/safety settings as part of the A/B. Inherited shell values must not make one
# machine repeat the failed 1e-4 update geometry or silently disable rollback/margin protection.
export NAVRL_LEARNING_RATE=3e-5
export NAVRL_RESET_ACTOR_OPTIMIZER=0
export NAVRL_ACTION_DIAG=1
export NAVRL_PPO_LOG_RATIO_CLAMP=10.0
export NAVRL_PPO_KL_STOP=0.04
export NAVRL_PPO_EPOCH_ROLLBACK=1
export NAVRL_PPO_ROLLBACK_LR_FACTOR=0.5
export NAVRL_PPO_ROLLBACK_MIN_LR=1e-6
export NAVRL_PPO_ROLLBACK_PATIENCE=5
export NAVRL_DENSITY_GUARD_WINDOW_EPOCHS=50
export NAVRL_DENSITY_GUARD_MIN_EPOCHS=100
export NAVRL_DENSITY_GUARD_MIN_PEAK=0.50
export NAVRL_DENSITY_GUARD_DROP=0.25
export NAVRL_DENSITY_GUARD_PATIENCE=25
export NAVRL_LATENT_MARGIN=2.0,1.25,2.0,2.0
export NAVRL_LATENT_MARGIN_COEF=0.01

if [[ ! -f "${CKPT}" ]]; then
    echo "[ttc-ab] checkpoint not found: ${CKPT}" >&2
    echo "[ttc-ab] pass CKPT=<path to a 70-bar v2 checkpoint>" >&2
    exit 2
fi

# Prefer an explicit PYTHON, then PATH (conda activate), then common install locations.
if [[ -z "${PYTHON:-}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON="$(command -v python)"
    elif [[ -x "${HOME}/miniconda3/envs/aerialgym/bin/python" ]]; then
        PYTHON="${HOME}/miniconda3/envs/aerialgym/bin/python"
    elif [[ -x /home/fair/miniconda3/envs/aerialgym/bin/python ]]; then
        PYTHON=/home/fair/miniconda3/envs/aerialgym/bin/python
    else
        echo "[ttc-ab] no aerialgym python found; activate conda or set PYTHON=..." >&2
        exit 2
    fi
fi
"${PYTHON}" - "${CKPT}" <<'PY'
import hashlib
from pathlib import Path
import sys

import torch

path = Path(sys.argv[1])
want_sha256 = "99da09e7d37a417f6628a6fc1f2180d70a41de1598a0552491df5a438c61ab37"
digest = hashlib.sha256(path.read_bytes()).hexdigest()
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
state = checkpoint.get("env_state") or {}
problems = []
if digest != want_sha256:
    problems.append(f"SHA-256={digest}, expected the audited ep3250 source")
if int(checkpoint.get("epoch", -1)) != 3250:
    problems.append(f"epoch={checkpoint.get('epoch')!r}, expected 3250")
if int(state.get("n_bars_active", -1)) != 70:
    problems.append(f"bars={state.get('n_bars_active')!r}, expected 70")
if state.get("cfg_action_policy") != "squashed_gaussian":
    problems.append(f"action policy={state.get('cfg_action_policy')!r}, expected squashed_gaussian")
if state.get("cfg_obstacle_selector") != "cluster_sector":
    problems.append(f"source selector={state.get('cfg_obstacle_selector')!r}, expected cluster_sector")
if problems:
    print("[ttc-ab] refusing unverified warm-start checkpoint:", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    raise SystemExit(2)
print("[ttc-ab] checkpoint provenance/SHA-256 OK")
PY

echo "[ttc-ab] ARM=${ARM}  selector=${NAVRL_OBSTACLE_SELECTOR}  bars=${NAVRL_NUM_BARS} (fixed)"
echo "[ttc-ab] warm-start ${CKPT}"
echo "[ttc-ab] gate: capture >= +2.0pp AND crash <= -2.0pp vs the baseline arm, held-out, at the FINAL ckpt"

if [[ "${NAVRL_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "[ttc-ab] PREFLIGHT PASS (training not started)"
    exit 0
fi

# 4 GB preset -> v2 contract. Both arms inherit arena, sensors, reward and target motion from
# train_navrl_v2_search.sh rather than restating them here.
exec ./train_navrl_v2_search_4gb.sh --checkpoint "${CKPT}" --branch_run
