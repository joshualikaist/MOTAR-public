#!/usr/bin/env bash
# Closed wrapper for the preregistered ref5in P2 held-out decision cell.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (( $# != 0 )); then
    echo "[ref5in-p2] no arguments are accepted" >&2
    exit 2
fi
PYTHON=/home/fair/miniconda3/envs/aerialgym/bin/python
export PYTHONNOUSERSITE=1
exec "${PYTHON}" tools/attest_navrl_ref5in_p2.py run
