#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NAVRL_TARGET_BRAKING_CONTRACT_VARIANT=baseline_1p25
exec bash "${script_dir}/run_navrl_physical_target_braking_v2_fresh.sh" "$@"
