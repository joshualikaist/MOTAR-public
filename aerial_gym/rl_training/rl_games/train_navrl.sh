#!/usr/bin/env bash
# === NavRL Phase-1 학습 (짧은 실행 래퍼) ===
# 기존의 긴 명령
#   PYTHONNOUSERSITE=1 python runner.py --file ppo_navrl_cnn.yaml --task navrl_task \
#       --num_envs 256 --headless True --use_warp True --train 2>&1 | tee train_....log
# 을 이 한 줄로:
#
#   처음부터:     ./train_navrl.sh
#   env 수 변경:  NUM_ENVS=512 ./train_navrl.sh          # RTX 3070 8GB 기본은 256
#   중단 지점 재개: ./train_navrl.sh --checkpoint runs/ppo_XXXX_navrl/nn/gen_ppo.pth
#   완료 run 분기:  위와 동일 (완료 표식이 있으면 새 run 폴더를 자동 생성)
#   같은 폴더 강제: ./train_navrl.sh --checkpoint ... --resume_in_place
#   MLP 베이스라인: FILE=ppo_navrl.yaml ./train_navrl.sh
#   다른 파이썬:  PYTHON=/path/to/conda/envs/aerialgym/bin/python ./train_navrl.sh
#
# 추가 인자(예: --checkpoint, --seed)는 그대로 runner.py 로 전달된다("$@").
set -euo pipefail

# runs/·nn/ 출력이 이 폴더에 생기도록 cwd 를 rl_games 로 고정 (1904 등 기존 run 위치와 동일).
cd "$(dirname "${BASH_SOURCE[0]}")"

# AirSim user-site numpy 가 conda numpy 를 가리는 문제 차단 (activate.d 가 이미 설정하지만 이중 안전).
export PYTHONNOUSERSITE=1

# Perception is a strict sub-mode of vision; make the dependency explicit and foolproof.
if [ "${NAVRL_PERCEPTION:-0}" = "1" ]; then
    export NAVRL_VISION=1
fi

# 4 GB VRAM 보조 머신 프리셋(GPU4GB=1): 세 조각을 한 번에 켠다 —
#   (1) base_sim_4gb 로 PhysX GPU 버퍼 축소(navrl_task_config 가 AERIAL_GYM_SIM_NAME 을 읽음),
#   (2) 4gb yaml(PPO 하이퍼파라미터는 3070 과 동일, minibatch=4096), (3) NUM_ENVS 기본 256.
# NUM_ENVS 기본 256 인 이유: minibatch=4096 은 batch=32*NUM_ENVS>=4096(=NUM_ENVS>=128)를 요구하고,
# 맨몸(기본 max_bars=150) 으로 4GB 에서 안전한 최대가 256 임(512x150 은 PhysX pair 버퍼 오버플로우로 죽음).
# density sweep 은 max_bars 25/50 라 가벼워, 3070 과 맞추려 NUM_ENVS=512 를 직접 붙여 돌린다(검증됨 ~2.5GB):
#   NAVRL_MAX_BARS=50 NAVRL_NUM_BARS=50 NUM_ENVS=512 GPU4GB=1 ./train_navrl.sh
# GPU4GB 를 세팅하지 않으면 8 GB 메인 머신 동작은 기존 그대로. (개별 변수로 오버라이드 가능.)
# 4GB(1650 Ti 등): PhysX GPU 버퍼 축소 프리셋. yaml/NUM_ENVS 선택은 아래 모드 블록에서 함께 결정.
if [ "${GPU4GB:-0}" = "1" ]; then
    export AERIAL_GYM_SIM_NAME="${AERIAL_GYM_SIM_NAME:-base_sim_4gb}"
fi

# yaml + 기본 env 수 선택 (VISION 이 GPU4GB 보다 우선 — 조합이면 vision_4gb 를 고른다).
#   NAVRL_VISION=1 NAVRL_PERCEPTION=1 → RGB-D/LiDAR perception + Transformer (권장)
#   NAVRL_VISION=1                 → 기존 semantic prototype baseline
#   NAVRL_VISION=1 NAVRL_LSTM=1    → 위 + 부분관측 LSTM
#   NAVRL_VISION=1 GPU4GB=1        → 4GB 비전 (N=128, minibatch 2048, base_sim_4gb)
#   GPU4GB=1 (비전 아님)           → 4GB LiDAR CNN
if [ "${NAVRL_VISION:-0}" = "1" ]; then
    if [ "${NAVRL_PERCEPTION:-0}" = "1" ]; then
        FILE="${FILE:-ppo_navrl_perception_transformer.yaml}"; DEF_ENVS=128
    elif [ "${GPU4GB:-0}" = "1" ]; then
        FILE="${FILE:-ppo_navrl_vision_4gb.yaml}"; DEF_ENVS=128
    elif [ "${NAVRL_LSTM:-0}" = "1" ]; then
        FILE="${FILE:-ppo_navrl_vision_lstm.yaml}"; DEF_ENVS=256
    else
        FILE="${FILE:-ppo_navrl_vision.yaml}"; DEF_ENVS=256
    fi
elif [ "${GPU4GB:-0}" = "1" ]; then
    FILE="${FILE:-ppo_navrl_cnn_4gb.yaml}"; DEF_ENVS=256
fi

PY="${PYTHON:-python}"                 # 활성화된 conda env 의 python (2번째 머신 이식용, 경로 하드코딩 X)
FILE="${FILE:-ppo_navrl_cnn.yaml}"     # NavRL LiDAR CNN (권장). MLP 는 ppo_navrl.yaml
TASK="${TASK:-navrl_task}"
NUM_ENVS="${NUM_ENVS:-${DEF_ENVS:-256}}"  # 8GB 기본 256. vision+4GB 는 128. (직접 오버라이드 가능)

mkdir -p runs train_session_logs
# Seconds + launcher PID prevent a quick retry from truncating the previous minute-resolution log.
LOG="${TRAIN_SESSION_LOG:-train_session_logs/train_$(date +%y%m%d_%H%M%S)_$$.log}"
mkdir -p "$(dirname "${LOG}")"
LIVE_LOG="${TRAIN_LIVE_LOG:-train_session_logs/current_training.log}"
mkdir -p "$(dirname "${LIVE_LOG}")"
LOG_ABS="$(realpath -m "${LOG}")"
# Resolve the parent directory, not the live symlink itself. Resolving LIVE_LOG before replacing it
# printed the previous session's target even though ln correctly updated the link.
LIVE_LOG_ABS="$(realpath -m "$(dirname "${LIVE_LOG}")")/$(basename "${LIVE_LOG}")"
ln -sfn "${LOG_ABS}" "${LIVE_LOG}"
echo "[train_navrl] py=${PY} file=${FILE} task=${TASK} envs=${NUM_ENVS}"
echo "[train_navrl] log → ${LOG_ABS}"
echo "[train_navrl] live log → ${LIVE_LOG_ABS}"
echo "[train_navrl] watch anytime: ./watch_navrl_training.sh"
echo "[train_navrl] TensorBoard: tensorboard --logdir $(pwd)/runs"

"${PY}" runner.py --file "${FILE}" --task "${TASK}" \
    --num_envs "${NUM_ENVS}" --headless True --use_warp True --train "$@" \
    2>&1 | tee "${LOG}"
