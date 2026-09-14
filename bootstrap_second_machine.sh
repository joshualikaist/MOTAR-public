#!/usr/bin/env bash
# =============================================================================
# One-shot second-machine setup for MOTAR.
#
# 자동화 안 되는 수동 전제 2가지 (스크립트로 불가):
#   1. Miniconda/Anaconda 설치 (conda 가 PATH 에 있어야 함).            → SETUP §1
#   2. Isaac Gym Preview 4 다운로드(NVIDIA 로그인) 후 ~/isaacgym 에 압축해제
#      (즉 ~/isaacgym/python 이 존재).  경로 다르면 ISAACGYM_PATH= 로 지정. → SETUP §2
#
# 사용법 (클론한 MOTAR 폴더 안에서):
#   git clone https://github.com/joshualikaist/MOTAR.git aerial_gym_simulator
#   cd aerial_gym_simulator
#   ./bootstrap_second_machine.sh
#
# 하는 일: conda env 생성 → Isaac Gym / MOTAR / rl_games / warp / urdfpy 설치 →
#          PYTHONNOUSERSITE 설정 → import 스모크 테스트. (urdfpy 는 원본 mmatl/urdfpy 를 clone)
#
# 오버라이드(환경변수):
#   ENV_NAME(기본 aerialgym)  PY_VER(3.8)  ISAACGYM_PATH(~/isaacgym)
#   URDFPY_URL(mmatl/urdfpy)  CLONE_NAVRL=1(참고 repo도 clone, 기본 off)
# =============================================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-aerialgym}"
PY_VER="${PY_VER:-3.8}"
ISAACGYM_PATH="${ISAACGYM_PATH:-$HOME/isaacgym}"
URDFPY_URL="${URDFPY_URL:-https://github.com/mmatl/urdfpy.git}"
CLONE_NAVRL="${CLONE_NAVRL:-0}"
NAVRL_URL="${NAVRL_URL:-https://github.com/Zhefan-Xu/NavRL.git}"

MOTAR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "$MOTAR_DIR")"          # 상위 폴더 → urdfpy 를 형제로 clone
URDFPY_DIR="$SRC_DIR/urdfpy"

say() { printf '\n\033[1;36m[bootstrap] %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m[bootstrap] ERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# --- 전제 확인 ---
command -v conda >/dev/null 2>&1 || die "conda 가 PATH 에 없음 — Miniconda 먼저 설치 (SETUP §1)."
[ -d "$ISAACGYM_PATH/python" ] || die "Isaac Gym 이 $ISAACGYM_PATH/python 에 없음 — Preview 4 다운로드(NVIDIA 로그인)·압축해제 하거나 ISAACGYM_PATH= 지정 (SETUP §2)."

say "conda env '$ENV_NAME' (python $PY_VER)"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  say "  '$ENV_NAME' 이미 존재 — 재사용"
else
  conda create -n "$ENV_NAME" "python=$PY_VER" -y
fi
conda activate "$ENV_NAME"

say "(1/4) Isaac Gym  (pip install -e $ISAACGYM_PATH/python)"
pip install -e "$ISAACGYM_PATH/python"

say "(2/4) MOTAR  (pip install -e .)"
cd "$MOTAR_DIR"
pip install -e .

say "(3/4) rl_games 1.6.5 + warp-lang 1.0.0"
pip install rl-games==1.6.5 warp-lang==1.0.0

say "(4/4) urdfpy  (원본 clone + install → $URDFPY_DIR)"
[ -d "$URDFPY_DIR/.git" ] || git clone "$URDFPY_URL" "$URDFPY_DIR"
pip install -e "$URDFPY_DIR"

if [ "$CLONE_NAVRL" = "1" ]; then
  NAVREF="$(dirname "$SRC_DIR")/reference/NavRL"
  say "(opt) NavRL 참고 repo → $NAVREF"
  [ -d "$NAVREF/.git" ] || git clone "$NAVRL_URL" "$NAVREF"
fi

say "PYTHONNOUSERSITE=1  (~/.local numpy 가 conda numpy 가리는 것 차단 → warp LiDAR)"
conda env config vars set PYTHONNOUSERSITE=1 -n "$ENV_NAME" >/dev/null
export PYTHONNOUSERSITE=1

say "스모크 테스트 (import)"
python -c "import aerial_gym; from aerial_gym.registry.task_registry import task_registry; import torch; print('  ok — torch', torch.__version__, '| cuda', torch.cuda.is_available())" \
  || die "스모크 테스트 실패 — 위 트레이스 확인."

say "완료 ✅  다음:"
cat <<EOF
  conda deactivate && conda activate $ENV_NAME       # PYTHONNOUSERSITE 반영
  cd "$MOTAR_DIR"
  python -m unittest discover -s tests -p 'test_navrl*.py'

  학습은 일반 train_navrl.sh 대신 README/RESEARCH_PLAN의 현재 고정 launcher를 사용하세요.
  GTX 1650 Ti는 학습 계보를 만들기보다 held-out 평가용 GPU4GB=1 profile을 권장합니다.
EOF
