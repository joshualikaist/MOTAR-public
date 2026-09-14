#!/usr/bin/env bash
# Matched 1,000-epoch adaptation arm authorized only by the preregistered R2 screen.
#
# The screen summary is not a suggestion: this launcher reruns its validator and refuses to train
# unless the separately held-out minimum-intervention riskcap passed its frozen GO/NO-GO gate. That keeps a noisy
# smoke result or a hand-edited environment variable from silently changing the experiment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if (( $# != 0 )); then
    echo "[speedgov-train] runner arguments are forbidden: $*" >&2
    exit 2
fi

export PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
    echo "[speedgov-train] Python not executable: ${PYTHON}" >&2
    exit 2
fi
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONNOUSERSITE=1

SOURCE_REL="runs/ppo_260802_0020_navrl_v2-recover-curriculum-continue-s1/nn/last_gen_ppo_ep_24000_rew_44.73549.pth"
SOURCE_SHA="82f7978b42d9d9e95adcc638a40ae85fb3736fd897ae39cb8aa8333be39cf23f"
SCREEN_ROOT="../../../results/navrl_v2_ep24000_riskcap_seed44_screen"
SUMMARY="${SCREEN_ROOT}/summary.json"
export CKPT="${CKPT:-${SOURCE_REL}}"

if [[ ! -f "${CKPT}" || ! -f "${SUMMARY}" ]]; then
    echo "[speedgov-train] missing frozen source or completed screen summary" >&2
    exit 2
fi

# Revalidate both held-out cells and rebuild the derived summary before reading the decision.
"${PYTHON}" ../../../tools/summarize_navrl_v2_riskcap.py "${SCREEN_ROOT}"

read -r SELECTED_MODE FIXED_MPS FREE_MPS HALF_WIDTH_M MARGIN_M SLOW_M RELEASE_M TTC_S BRAKE_MPS2 REACTION_S <<< "$(${PYTHON} - "${SUMMARY}" "${CKPT}" "${SOURCE_SHA}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

summary_path, checkpoint_path, expected_sha = map(Path, sys.argv[1:])
expected_sha = str(expected_sha)
payload = json.loads(summary_path.read_text(encoding="utf-8"))
actual_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
problems = []
if actual_sha != expected_sha:
    problems.append(f"checkpoint SHA={actual_sha}, expected {expected_sha}")
if payload.get("source_checkpoint_sha256") != expected_sha:
    problems.append("summary source SHA mismatch")
if payload.get("adaptive_go") is not True:
    problems.append("adaptive GO/NO-GO gate did not pass")
selected = payload.get("selected_tag")
if selected != "riskcap":
    problems.append(f"selected arm is not the preregistered riskcap: {selected!r}")
rows = {row.get("tag"): row for row in payload.get("rows", [])}
row = rows.get(selected) or {}
if row.get("screen_pass") is not True:
    problems.append("selected arm lacks a passing screen result")
params = row.get("parameters") or {}
mode = row.get("mode")
if mode != selected:
    problems.append(f"selected mode mismatch: tag={selected!r}, mode={mode!r}")
required = (
    "speed_governor_fixed_mps",
    "speed_governor_free_mps",
    "speed_governor_half_width_m",
    "speed_governor_margin_m",
    "speed_governor_slow_m",
    "speed_governor_release_m",
    "speed_governor_ttc_s",
    "speed_governor_brake_mps2",
    "speed_governor_reaction_s",
)
missing = [key for key in required if key not in params]
if missing:
    problems.append("missing selected parameters: " + ", ".join(missing))
if problems:
    print("[speedgov-train] refusing unauthorized adaptation:", file=sys.stderr)
    for problem in problems:
        print("  - " + problem, file=sys.stderr)
    raise SystemExit(2)
print(mode, *(params[key] for key in required))
PY
)"

export NAVRL_SPEED_GOVERNOR="${SELECTED_MODE}"
export NAVRL_SPEED_GOVERNOR_FIXED_MPS="${FIXED_MPS}"
export NAVRL_SPEED_GOVERNOR_FREE_MPS="${FREE_MPS}"
export NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M="${HALF_WIDTH_M}"
export NAVRL_SPEED_GOVERNOR_MARGIN_M="${MARGIN_M}"
export NAVRL_SPEED_GOVERNOR_SLOW_M="${SLOW_M}"
export NAVRL_SPEED_GOVERNOR_RELEASE_M="${RELEASE_M}"
export NAVRL_SPEED_GOVERNOR_TTC_S="${TTC_S}"
export NAVRL_SPEED_GOVERNOR_BRAKE_MPS2="${BRAKE_MPS2}"
export NAVRL_SPEED_GOVERNOR_REACTION_S="${REACTION_S}"
export NAVRL_SPEED_GOVERNOR_DIAG=1

# Fixed-density, matched-budget continuation from the frozen source policy.
export NAVRL_V2_ALLOW_RESUME=1
export NAVRL_V2_PROFILE=main
export NAVRL_OBSTACLE_SELECTOR=cluster_sector
export SEED=1
export NAVRL_DENSITY_CURRICULUM=0
export NAVRL_NUM_BARS=205
export NAVRL_DENSITY_START=205
export NAVRL_DENSITY_FINAL=205
export NAVRL_DENSITY_WARMUP=0
export NAVRL_DENSITY_MIN_EPOCHS=0
export NAVRL_RESET_DENSITY_WINDOW=1
export NAVRL_LEARNING_RATE=5e-6
export NAVRL_RESET_ACTOR_OPTIMIZER=0
export MAX_EPOCHS=25000
export AERIAL_RUN_TAG="v2-speedgov-ep24000-205bars-main-${SELECTED_MODE}-s1"
export TRAIN_SESSION_LOG="train_session_logs/v2_ep24000_speedgov_${SELECTED_MODE}_$(date +%y%m%d_%H%M%S)_$$.log"
export TRAIN_LIVE_LOG="train_session_logs/current_training.log"

unset NAVRL_RECOVERY_STAGE NAVRL_RECOVERY_SOURCE_EPOCH
unset NAVRL_RECOVERY_SOURCE_SHA256 NAVRL_RECOVERY_SMOKE_REQUIRED_EPOCHS
unset NAVRL_RECOVERY_SMOKE_BARS NAVRL_RECOVERY_EVAL_ATTESTATION_SHA256
unset NAVRL_RECOVERY_EVAL_ATTESTATION_B64
unset NAVRL_GENERAL_EVAL NAVRL_INTERACTIVE NAVRL_BULK_EVAL NAVRL_BULK_EVAL_JSON
unset NAVRL_GENERAL_RESULTS_JSON NAVRL_EVAL_CHECKPOINT NAVRL_EVAL_TARGET_SPEED_FINAL
unset NAVRL_EVAL_FULL_DISTRIBUTION NAVRL_EVAL_ACTION_MODE NAVRL_V2_ACTION_MODE
unset NAVRL_EVAL_RUN_NONCE NAVRL_EVAL_PROFILE NAVRL_SIM_PHYSICS_CONTRACT
unset ALLOW_CONCURRENT NAVRL_V2_FORCE

echo "[speedgov-train] authorized by ${SUMMARY}"
echo "[speedgov-train] mode=${SELECTED_MODE} fixed=${FIXED_MPS} free=${FREE_MPS} margin=${MARGIN_M} slow=${SLOW_M} release=${RELEASE_M} ttc=${TTC_S}"
echo "[speedgov-train] source=${CKPT} SHA-256=${SOURCE_SHA}"
echo "[speedgov-train] fixed 205 bars | epoch 24001..25000 | 4,096,000 samples | lr=5e-6"

if [[ "${NAVRL_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    export NAVRL_V2_CONTRACT_PREFLIGHT_ONLY=1
fi
exec ./train_navrl_v2_search.sh --checkpoint "${CKPT}" --branch_run
