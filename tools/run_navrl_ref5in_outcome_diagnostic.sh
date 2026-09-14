#!/usr/bin/env bash
# Closed launcher for the post-P2 descriptive outcome-strata diagnostic.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (( $# != 0 )); then
    echo "[ref5in-outcome-diagnostic] no arguments are accepted" >&2
    exit 2
fi
export PYTHONNOUSERSITE=1
exec /home/fair/miniconda3/envs/aerialgym/bin/python \
    tools/run_navrl_ref5in_outcome_diagnostic.py run
