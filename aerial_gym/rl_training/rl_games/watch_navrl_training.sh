#!/usr/bin/env bash
# Follow the active NavRL training log through a stable path.
#
# Usage:
#   ./watch_navrl_training.sh
#   ./watch_navrl_training.sh train_session_logs/specific_run.log
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

LOG_PATH="${1:-train_session_logs/current_training.log}"
if [[ ! -e "${LOG_PATH}" ]]; then
    echo "[watch_navrl] no active/current log found at ${LOG_PATH}" >&2
    RECENT_LOG="$(
        find train_session_logs -maxdepth 1 -type f -name '*.log' \
            -printf '%T@ %p\n' 2>/dev/null | sort -nr | sed -n '1s/^[^ ]* //p'
    )"
    if [[ -n "${RECENT_LOG}" ]]; then
        echo "[watch_navrl] most recent log: ${RECENT_LOG}" >&2
        echo "[watch_navrl] run: ./watch_navrl_training.sh '${RECENT_LOG}'" >&2
    fi
    exit 2
fi

echo "[watch_navrl] following $(realpath "${LOG_PATH}")"
echo "[watch_navrl] press Ctrl-C to stop watching; training keeps running."
exec tail -n 120 -F "${LOG_PATH}"
