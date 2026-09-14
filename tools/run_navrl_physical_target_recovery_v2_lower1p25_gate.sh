#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export NAVRL_TARGET_BRAKING_CONTRACT_VARIANT=baseline_1p25
export PYTHONNOUSERSITE=1
export PYTHONPATH="${root}:${root}/tools${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$(dirname "${python}"):${PATH}"
exec "${python}" "${root}/tools/verify_navrl_physical_target_recovery_v2_gate.py" "$@"
