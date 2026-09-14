#!/usr/bin/env bash
# Fixed-205-bar representation A/B from the frozen ep24000 ceiling checkpoint.
#
#   ARM=baseline PROFILE=main ./train_navrl_v2_ep24000_ttc_ab.sh
#   ARM=ttc      PROFILE=main ./train_navrl_v2_ep24000_ttc_ab.sh
#   ARM=baseline PROFILE=4gb  ./train_navrl_v2_ep24000_ttc_ab.sh  # GTX 1650 Ti
#   ARM=ttc      PROFILE=4gb  ./train_navrl_v2_ep24000_ttc_ab.sh
#
# Preregistered comparison: evaluate each FINAL checkpoint at 205 bars with the same held-out seed
# and deterministic action selection. TTC advances only if capture improves by >=2.0 percentage
# points AND crash falls by >=2.0 points versus the same-profile baseline. This launcher trains no
# curriculum: density, sample budget and every PPO/task setting are identical within a profile.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if (( $# != 0 )); then
    echo "[ep24000-ttc-ab] runner arguments are forbidden: $*" >&2
    exit 2
fi

case "${ARM:-}" in
    baseline) SELECTOR=cluster_sector ;;
    ttc)      SELECTOR=ttc_sector ;;
    *)
        echo "usage: ARM=baseline|ttc PROFILE=main|4gb $0" >&2
        exit 2
        ;;
esac

PROFILE="${PROFILE:-main}"
case "${PROFILE}" in
    main)
        ADAPT_EPOCHS=1000
        NUM_ENVS_EXPECTED=128
        ;;
    4gb)
        ADAPT_EPOCHS=2000
        NUM_ENVS_EXPECTED=64
        ;;
    *)
        echo "[ep24000-ttc-ab] PROFILE must be main or 4gb; got: ${PROFILE}" >&2
        exit 2
        ;;
esac

export PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
if [[ "${PYTHON}" == */* ]]; then
    export PATH="$(dirname "${PYTHON}"):${PATH}"
fi
export PYTHONNOUSERSITE=1

SOURCE_REL="runs/ppo_260802_0020_navrl_v2-recover-curriculum-continue-s1/nn/last_gen_ppo_ep_24000_rew_44.73549.pth"
SOURCE_SHA256="82f7978b42d9d9e95adcc638a40ae85fb3736fd897ae39cb8aa8333be39cf23f"
export CKPT="${CKPT:-${SOURCE_REL}}"
if [[ ! -f "${CKPT}" ]]; then
    echo "[ep24000-ttc-ab] checkpoint not found: ${CKPT}" >&2
    echo "[ep24000-ttc-ab] copy the frozen ep24000 file or set CKPT=/path/to/it" >&2
    exit 2
fi

"${PYTHON}" - "${CKPT}" "${SOURCE_SHA256}" <<'PY'
import hashlib
import math
from pathlib import Path
import sys

import torch

path = Path(sys.argv[1])
expected_sha = sys.argv[2]
actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
state = checkpoint.get("env_state") or {}
expected = {
    "cfg_arena_xy": 40.0,
    "cfg_arena_z": 3.0,
    "cfg_bar_pool": "bars_h3",
    "cfg_placement_mode": "navrl_band",
    "cfg_episode_len_steps": 600.0,
    "cfg_general_goal_dist_min": 6.0,
    "cfg_general_goal_dist_max": 28.0,
    "cfg_lidar_max_range": 12.0,
    "cfg_lidar_hbeams": 72,
    "cfg_lidar_vbeams": 4,
    "cfg_max_obstacles": 8,
    "cfg_token_fov_deg": 240.0,
    "cfg_obstacle_selector": "cluster_sector",
    "cfg_action_policy": "squashed_gaussian",
    "cfg_action_std": "0.35,0.35,0.05,0.08",
    "cfg_action_mu_scale": "1.0,0.4,1.0,1.0",
    "n_bars_active": 205,
}
problems = []
if actual_sha != expected_sha:
    problems.append(f"SHA-256={actual_sha}, expected {expected_sha}")
if int(checkpoint.get("epoch", -1)) != 24000:
    problems.append(f"epoch={checkpoint.get('epoch')!r}, expected 24000")
for key, want in expected.items():
    got = state.get(key)
    if isinstance(want, float):
        try:
            ok = math.isclose(float(got), want, rel_tol=0.0, abs_tol=1e-6)
        except (TypeError, ValueError):
            ok = False
    else:
        ok = got == want
    if not ok:
        problems.append(f"{key}={got!r}, expected {want!r}")
if problems:
    print("[ep24000-ttc-ab] refusing unverified source:", file=sys.stderr)
    for problem in problems:
        print("  - " + problem, file=sys.stderr)
    raise SystemExit(2)
print("[ep24000-ttc-ab] checkpoint provenance/SHA-256 PASS")
PY

# The main TTC arm is only interpretable after the matched cluster-sector baseline has been
# evaluated and frozen. Validate the immutable result/receipt/snapshot here instead of relying on
# a human remembering that ordering constraint.
if [[ "${PROFILE}" == "main" && "${ARM}" == "ttc" ]]; then
    BASELINE_RESULT="../../../results/navrl_v2_ep24000_ttc_main_baseline/205bars.json"
    "${PYTHON}" - "${BASELINE_RESULT}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

result_path = Path(sys.argv[1]).resolve()
expected_checkpoint_sha = "169ddcddb83c9d74df5c79252274660bc9c52e32d7d5144d325698e32b1d9b08"

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if not result_path.is_file():
    raise SystemExit(
        "[ep24000-ttc-ab] main TTC blocked: canonical baseline held-out result is missing: "
        + str(result_path)
    )
payload = json.loads(result_path.read_text(encoding="utf-8"))
receipt_path = result_path.with_suffix(".receipt.json")
snapshot_path = result_path.parent / "checkpoint_snapshot.pth"
log_path = result_path.with_suffix(".log")
problems = []
for path, label in ((receipt_path, "receipt"), (snapshot_path, "checkpoint snapshot"), (log_path, "log")):
    if not path.is_file():
        problems.append(f"missing {label}: {path}")

condition = payload.get("condition", {})
contract = payload.get("v2_evaluation_contract", {})
outcome = payload.get("outcome", {})
actual = int(payload.get("actual_episodes", -1))
checks = {
    "schema_version": payload.get("schema_version") == 1,
    "requested_episodes": int(payload.get("requested_episodes", -1)) == 2049,
    "actual_episodes": actual >= 2049,
    "checkpoint_sha256": payload.get("checkpoint_sha256") == expected_checkpoint_sha,
    "snapshot_sha256_field": payload.get("evaluated_checkpoint_snapshot_sha256") == expected_checkpoint_sha,
    "bars": int(condition.get("bars", -1)) == 205,
    "seed": int(condition.get("seed", -1)) == 42,
    "action_selection": condition.get("action_selection") == "deterministic",
    "reflection_mode": condition.get("reflection_mode") == "original",
    "target_speed_mode": condition.get("target_speed_mode") == "uniform",
    "selector": contract.get("obstacle_selector") == "cluster_sector",
    "runtime_profile": contract.get("runtime_profile") == "main",
    "physics": contract.get("sim_physics_contract") == "base_sim_dt0.01",
    "outcome_accounting": sum(int(outcome.get(key, -actual - 1)) for key in ("captured", "crash", "timeout")) == actual,
}
problems.extend(name for name, passed in checks.items() if not passed)

if receipt_path.is_file() and snapshot_path.is_file() and log_path.is_file():
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    artifact_checks = {
        "result_sha256": receipt.get("result_sha256") == sha256(result_path),
        "receipt_source_sha256": receipt.get("source_checkpoint_sha256") == expected_checkpoint_sha,
        "receipt_snapshot_sha256": receipt.get("evaluated_checkpoint_snapshot_sha256") == expected_checkpoint_sha,
        "snapshot_bytes": sha256(snapshot_path) == expected_checkpoint_sha,
        "log_sha256": receipt.get("log_sha256") == sha256(log_path),
        "receipt_nonce": receipt.get("evaluation_nonce") == condition.get("evaluation_nonce"),
        "receipt_episodes": int(receipt.get("actual_episodes", -1)) == actual,
        "receipt_condition": int(receipt.get("bars", -1)) == 205 and int(receipt.get("seed", -1)) == 42,
    }
    problems.extend(name for name, passed in artifact_checks.items() if not passed)

if problems:
    print("[ep24000-ttc-ab] main TTC blocked: invalid baseline held-out artifact", file=sys.stderr)
    for problem in problems:
        print("  - " + problem, file=sys.stderr)
    raise SystemExit(2)

capture = float(outcome["capture_rate"])
crash = float(outcome["crash_rate"])
print(
    "[ep24000-ttc-ab] baseline held-out PASS | "
    f"n={actual} capture={capture:.6f} crash={crash:.6f} | "
    f"TTC primary thresholds capture>={capture + 0.020:.6f} crash<={crash - 0.020:.6f}"
)
print(
    "[ep24000-ttc-ab] canonical deployment reference | "
    "ep24000 capture=0.724390 crash=0.250732; primary A/B PASS alone does not replace it"
)
PY
fi

# The only arm-specific setting.
export NAVRL_OBSTACLE_SELECTOR="${SELECTOR}"

# Shared task and optimizer continuation contract.
export NAVRL_V2_ALLOW_RESUME=1
export NAVRL_V2_PROFILE="${PROFILE}"
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
export MAX_EPOCHS=$((24000 + ADAPT_EPOCHS))
export AERIAL_RUN_TAG="v2-ep24000-205bars-${PROFILE}-${ARM}-s1"
export TRAIN_SESSION_LOG="train_session_logs/v2_ep24000_205bars_${PROFILE}_${ARM}_$(date +%y%m%d_%H%M%S)_$$.log"
export TRAIN_LIVE_LOG="train_session_logs/current_training.log"

# This is a new representation A/B branch, not a continuation of the recovery certification.
unset NAVRL_RECOVERY_STAGE NAVRL_RECOVERY_SOURCE_EPOCH
unset NAVRL_RECOVERY_SOURCE_SHA256 NAVRL_RECOVERY_SMOKE_REQUIRED_EPOCHS
unset NAVRL_RECOVERY_SMOKE_BARS NAVRL_RECOVERY_EVAL_ATTESTATION_SHA256
unset NAVRL_RECOVERY_EVAL_ATTESTATION_B64
unset NAVRL_GENERAL_EVAL NAVRL_INTERACTIVE NAVRL_BULK_EVAL NAVRL_BULK_EVAL_JSON
unset NAVRL_GENERAL_RESULTS_JSON NAVRL_EVAL_CHECKPOINT NAVRL_EVAL_TARGET_SPEED_FINAL
unset NAVRL_EVAL_FULL_DISTRIBUTION NAVRL_EVAL_ACTION_MODE NAVRL_V2_ACTION_MODE
unset NAVRL_EVAL_RUN_NONCE NAVRL_EVAL_PROFILE NAVRL_SIM_PHYSICS_CONTRACT
unset ALLOW_CONCURRENT NAVRL_V2_FORCE

SAMPLES=$((ADAPT_EPOCHS * 32 * NUM_ENVS_EXPECTED))
echo "[ep24000-ttc-ab] arm=${ARM} selector=${SELECTOR} profile=${PROFILE} bars=205 fixed"
echo "[ep24000-ttc-ab] budget=${ADAPT_EPOCHS} epochs x 32 x ${NUM_ENVS_EXPECTED} = ${SAMPLES} samples; final_epoch=${MAX_EPOCHS}"
echo "[ep24000-ttc-ab] source=${CKPT}"
echo "[ep24000-ttc-ab] gate=TTC capture >= baseline +0.020 AND crash <= baseline -0.020 at final checkpoint"

if [[ "${NAVRL_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    export NAVRL_V2_CONTRACT_PREFLIGHT_ONLY=1
fi
exec ./train_navrl_v2_search.sh --checkpoint "${CKPT}" --branch_run
