#!/usr/bin/env bash
# === NavRL Phase 3 — 센서 전용 + 완만한 순차 밀도 커리큘럼 (권장 다음 run) ===
#
# 설계 의도 (WORKLOG 2026-07-22):
#   - 밀도: +5막대씩, capture≥55%일 때만 승급, 110막대에서 정지 (기하 절벽 ~115 직전)
#   - 거리: 목표를 시야 안(≤16 m)으로 캡 → 거리·밀도 커리큘럼 끝-충돌 방지
#   - 평가: 항상 last_gen_ppo_ep_* 사용 (gen_ppo.pth = 저밀도 best 함정)
#
# 사용:
#   cd aerial_gym/rl_training/rl_games
#   ./train_navrl_vision_seq_density.sh              # 처음부터
#   ./train_navrl_vision_seq_density.sh --seed 2     # 재현성 seed
#   CKPT=runs/.../last_gen_ppo_ep_XXXX.pth ./train_navrl_vision_seq_density.sh --checkpoint "$CKPT"
#
# 모니터링 (다른 터미널, repo 루트 또는 rl_games 에서):
#   tensorboard --logdir src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/runs --port 6006
#   TB: navrl/n_bars_active (계단 +5), navrl/captured_rate, navrl/crash_rate
#   콘솔: "density curriculum promoted | bars 25 -> 30 ..."
#
# 승급마다 체크포인트 스냅샷 (선택):
#   ./snapshot_density_ckpts.sh runs/ppo_YYMMDD_HHMM_navrl
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTHONNOUSERSITE=1
export NAVRL_VISION=1
export NAVRL_MAX_BARS=150

# --- 밀도 커리큘럼 (완만) ---
export NAVRL_DENSITY_CURRICULUM=1
export NAVRL_DENSITY_START=25
export NAVRL_DENSITY_FINAL=110
export NAVRL_DENSITY_STEP=5
export NAVRL_DENSITY_THRESHOLD=0.55
export NAVRL_DENSITY_WARMUP=1000
export NAVRL_DENSITY_CHECK_EPS=4096

# --- 거리 커리큘럼 (얕게, 검출기 20 m 시야 안) ---
export NAVRL_K_FINAL=16
export NAVRL_K_MIN_FINAL=10
export NAVRL_K_WARMUP=8000

SEED="${SEED:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-9000}"
NUM_ENVS="${NUM_ENVS:-256}"

echo "[seq_density] vision + gentle density (25→110 step +5) + shallow goals (k≤16m)"
echo "[seq_density] seed=${SEED} max_epochs=${MAX_EPOCHS} num_envs=${NUM_ENVS}"
echo "[seq_density] extra args: $*"

exec ./train_navrl.sh --seed "${SEED}" --max_epochs "${MAX_EPOCHS}" "$@"
