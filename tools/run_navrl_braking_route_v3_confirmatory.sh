#!/usr/bin/env bash
set -euo pipefail
: "${NAVRL_V3_OUTPUT_ROOT:?set a new, unique confirmatory output root}"
: "${NAVRL_V3_CELL_RUNNER:?set the tracked executable v3 cell adapter}"
: "${NAVRL_V3_BRAKING_RECEIPT:?set the raw braking receipt.json}"
: "${NAVRL_V3_BRAKING_RECEIPT_SHA256:?set its exact SHA256}"
: "${NAVRL_V3_PILOT_SUMMARY:?set the verified PASS pilot summary.json}"
: "${NAVRL_TARGET_BRAKING_CONTRACT_VARIANT:=canonical_1p5}"
export NAVRL_TARGET_BRAKING_CONTRACT_VARIANT
exec python3 tools/run_navrl_braking_route_v3_gate.py confirmatory \
  --output-root "${NAVRL_V3_OUTPUT_ROOT}" \
  --cell-runner "${NAVRL_V3_CELL_RUNNER}" \
  --braking-receipt "${NAVRL_V3_BRAKING_RECEIPT}" \
  --braking-receipt-sha256 "${NAVRL_V3_BRAKING_RECEIPT_SHA256}" \
  --pilot-summary "${NAVRL_V3_PILOT_SUMMARY}"
