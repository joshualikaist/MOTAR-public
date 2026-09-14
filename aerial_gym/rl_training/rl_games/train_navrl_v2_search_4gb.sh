#!/usr/bin/env bash
# Task v2 "search arena" on a 4 GB card (GTX 1650 Ti). Same task contract as
# train_navrl_v2_search.sh; only the VRAM-bound knobs differ.
#
# Sizing rationale (PhysX rigid-actor count is the binding constraint, not mesh memory --
# a 300-bar env is only ~360 KB of warp mesh/BVH):
#   documented 4 GB limits (train_navrl.sh:30-36)
#     256 envs x 150 bars = 38,400 actors  -> safe max WITHOUT vision
#     512 envs x 150 bars = 76,800 actors  -> dies (PhysX pair buffer overflow)
#     128 envs x 150 bars = 19,200 actors  -> the validated 4 GB *vision* preset
#   v2 doubles the bars, so matching the validated vision actor count means halving envs:
#     64 envs x 300 bars  = 19,200 actors  -> this launcher
# 64 is also the minimum the PPO batch allows: minibatch 2048, horizon 32 -> 32*64 = 2048.
#
# MEASURED 2026-07-31 on the 3070 with this exact preset (base_sim_4gb, 64 envs, 40x40,
# 300-bar build): peak 3425 MiB of 4096 while TRAINING, 2561 MiB for the env alone with all
# 300 bars active. ~670 MiB of headroom, so a 1650 Ti fits. Note VRAM does NOT grow as the
# density curriculum promotes: all NAVRL_MAX_BARS bars are created as rigid actors at build
# time and inactive ones are parked at -1000, so the 70-bar and 300-bar peaks matched to
# within 3 MiB.
#
# Cost of the smaller fleet: half the batch per epoch versus the 8 GB machine, so learning is
# slower and noisier. Per the project rule, do NOT pool results from this machine with 3070
# runs -- use it for evaluation (eval_navrl_v2_density_sweep.sh with NUM_ENVS=64) or for an
# independent seed that is reported separately.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# base_sim_4gb shrinks the PhysX GPU buffers (max_gpu_contact_pairs 2^24 -> 2^22,
# default_buffer_size_multiplier 10 -> 2). Physics behaviour is unchanged; these are capacity
# ceilings that alone exceed a 4 GB card at the 8000-env defaults.
export GPU4GB=1
export NAVRL_V2_PROFILE=4gb
export AERIAL_GYM_SIM_NAME=base_sim_4gb
export NUM_ENVS=64
export AERIAL_RUN_TAG="${AERIAL_RUN_TAG:-v2-search-4gb-s${SEED:-1}}"
export TRAIN_SESSION_LOG="${TRAIN_SESSION_LOG:-train_session_logs/v2_search_4gb_$(date +%y%m%d_%H%M%S).log}"

echo "[v2-4gb] 4 GB preset | sim=${AERIAL_GYM_SIM_NAME} envs=${NUM_ENVS} (batch 32x${NUM_ENVS})"
echo "[v2-4gb] actor budget: ${NUM_ENVS} envs x 300 bars = $((NUM_ENVS * 300)) rigid actors"
exec ./train_navrl_v2_search.sh "$@"
