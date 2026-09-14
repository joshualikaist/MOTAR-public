#!/usr/bin/env bash
# Publication re-baseline: 3 policy/controller arms x 5 densities = 15 held-out cells.
#
# A = ep24000 / governor off
# B = ep24000 / frozen riskcap
# C = ep25000 / frozen riskcap
#
# Every cell uses seed53, deterministic actions, the corrected exact-600 evaluator contract, and
# one immutable shared runtime-source bundle. Cells run sequentially on one GPU and completed cells
# are resume-safe: rerunning this launcher skips a complete cell but refuses a partial directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PYTHON PYTHONNOUSERSITE=1

SOURCE="runs/ppo_260802_0020_navrl_v2-recover-curriculum-continue-s1/nn/last_gen_ppo_ep_24000_rew_44.73549.pth"
SOURCE_SHA="82f7978b42d9d9e95adcc638a40ae85fb3736fd897ae39cb8aa8333be39cf23f"
TRAINED="runs/ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1/nn/last_gen_ppo_ep_25000_rew_39.742134.pth"
TRAINED_SHA="f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40"

SEED=53
GAMES=2049
DENSITIES=(130 160 190 205 220)
RESULT_ROOT="${NAVRL_ABC_RESULT_ROOT:-${REPO_ROOT}/results/navrl_v2_governor_adaptation_abc_seed${SEED}_schema2}"
SHARED_BUNDLE="${RESULT_ROOT}/source_bundle"
CAMPAIGN_CONTRACT="${RESULT_ROOT}/campaign_contract.json"
PREFLIGHT="${PREFLIGHT:-0}"

if [[ ! "${SEED}" =~ ^[1-9][0-9]*$ || ! "${GAMES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[abc15] seed and games must be positive integers; seed=${SEED} games=${GAMES}" >&2
    exit 2
fi
if [[ ! -f "${SOURCE}" || ! -f "${TRAINED}" ]]; then
    echo "[abc15] required ep24000/ep25000 checkpoint is missing" >&2
    exit 2
fi

"${PYTHON}" - "${SOURCE}" "${SOURCE_SHA}" 24000 "${TRAINED}" "${TRAINED_SHA}" 25000 <<'PY'
import hashlib
from pathlib import Path
import sys
import torch

for index in (1, 4):
    path = Path(sys.argv[index])
    expected_sha = sys.argv[index + 1]
    expected_epoch = int(sys.argv[index + 2])
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        raise SystemExit(f"[abc15] checkpoint SHA mismatch: {path} -> {actual_sha}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("env_state") or {}
    if int(payload.get("epoch", -1)) != expected_epoch:
        raise SystemExit(f"[abc15] checkpoint epoch mismatch: {path}")
    if int(state.get("n_bars_active", -1)) != 205:
        raise SystemExit(f"[abc15] checkpoint is not the frozen 205-bar policy: {path}")
    if state.get("cfg_obstacle_selector") != "cluster_sector":
        raise SystemExit(f"[abc15] checkpoint selector is not cluster_sector: {path}")
PY

export NAVRL_V2_DENSITIES="130 160 190 205 220"
export NAVRL_V2_ACTION_MODE=deterministic
export NAVRL_EVAL_REFLECTION_MODE=original
export NAVRL_SEED="${SEED}"
export NAVRL_SPEED_GOVERNOR_FIXED_MPS=2.0
export NAVRL_SPEED_GOVERNOR_FREE_MPS=3.53553390593
export NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M=0.45
export NAVRL_SPEED_GOVERNOR_MARGIN_M=0.45
export NAVRL_SPEED_GOVERNOR_SLOW_M=3.0
export NAVRL_SPEED_GOVERNOR_RELEASE_M=5.0
export NAVRL_SPEED_GOVERNOR_TTC_S=1.2
export NAVRL_SPEED_GOVERNOR_BRAKE_MPS2=2.9608856678
export NAVRL_SPEED_GOVERNOR_REACTION_S=0.1
unset NAVRL_DETECTOR_CHECKPOINT NAVRL_V2_FIXED_TARGET_SPEED

preflight_arm() {
    local checkpoint="$1"
    local mode="$2"
    NAVRL_SPEED_GOVERNOR="${mode}" NAVRL_PREFLIGHT_ONLY=1 \
        ./eval_navrl_v2_density_sweep.sh "${checkpoint}" "${GAMES}"
}

if [[ "${PREFLIGHT}" == "1" ]]; then
    preflight_arm "${SOURCE}" off
    preflight_arm "${SOURCE}" riskcap
    preflight_arm "${TRAINED}" riskcap
    echo "[abc15] PREFLIGHT PASS | 15 cells, seed=${SEED}, games/cell=${GAMES}"
    exit 0
fi

if [[ -e "${RESULT_ROOT}" && ! -f "${CAMPAIGN_CONTRACT}" ]]; then
    echo "[abc15] refusing unrelated/partial result root without campaign contract: ${RESULT_ROOT}" >&2
    exit 2
fi
mkdir -p "${RESULT_ROOT}"

"${PYTHON}" - "${CAMPAIGN_CONTRACT}" "${SOURCE_SHA}" "${TRAINED_SHA}" \
    "${SEED}" "${GAMES}" "${BASH_SOURCE[0]}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
launcher = Path(sys.argv[6]).resolve()
expected = {
    "schema_version": 2,
    "campaign": "governor_adaptation_abc",
    "source_checkpoint_sha256": sys.argv[2],
    "trained_checkpoint_sha256": sys.argv[3],
    "seed": int(sys.argv[4]),
    "games_per_cell": int(sys.argv[5]),
    "densities": [130, 160, 190, 205, 220],
    "action_selection": "deterministic",
    "reflection_mode": "original",
    "arms": {
        "A_off": "ep24000/governor-off",
        "B_source_riskcap": "ep24000/riskcap",
        "C_trained_riskcap": "ep25000/riskcap",
    },
    "launcher": str(launcher),
    "launcher_sha256": hashlib.sha256(launcher.read_bytes()).hexdigest(),
}
if path.exists():
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != expected:
        raise SystemExit("[abc15] existing campaign contract does not match this invocation")
else:
    path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

# Preserve a human-readable combined log while still showing progress in the calling terminal.
exec > >(tee -a "${RESULT_ROOT}/campaign.log") 2>&1

validate_shared_bundle() {
    [[ ! -e "${SHARED_BUNDLE}" ]] && return 0
    "${PYTHON}" - "${SHARED_BUNDLE}/source_manifest.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1]).resolve()
if not manifest_path.is_file():
    raise SystemExit("[abc15] shared source manifest is missing")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("schema_version") != 2:
    raise SystemExit("[abc15] shared source manifest schema is not 2")
repo = Path(manifest["repository_root"]).resolve()

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

files = manifest.get("runtime_files") or []
if len(files) != int(manifest.get("runtime_file_count", -1)) or not files:
    raise SystemExit("[abc15] shared source file accounting is invalid")
for entry in files:
    original = (repo / entry["path"]).resolve()
    snapshot = (manifest_path.parent / entry["snapshot"]).resolve()
    if not original.is_file() or digest(original) != entry["sha256"]:
        raise SystemExit(f"[abc15] runtime source changed since campaign start: {entry['path']}")
    if not snapshot.is_file() or digest(snapshot) != entry["sha256"]:
        raise SystemExit(f"[abc15] shared source snapshot is invalid: {entry['snapshot']}")
PY
}

run_cell() {
    local arm="$1"
    local checkpoint="$2"
    local mode="$3"
    local bars="$4"
    local cell_dir="${RESULT_ROOT}/${arm}/${bars}bars"
    local result="${cell_dir}/${bars}bars.json"
    local receipt="${cell_dir}/${bars}bars.receipt.json"

    if [[ -e "${cell_dir}" ]]; then
        if [[ -f "${result}" && -f "${receipt}" ]]; then
            echo "[abc15] SKIP complete | arm=${arm} bars=${bars}"
            return 0
        fi
        echo "[abc15] partial cell directory requires manual inspection/move: ${cell_dir}" >&2
        exit 3
    fi

    validate_shared_bundle
    mkdir -p "$(dirname "${cell_dir}")"
    echo "[abc15] RUN | arm=${arm} bars=${bars} seed=${SEED} games=${GAMES}"
    NAVRL_V2_DENSITIES="${bars}" \
    NAVRL_SPEED_GOVERNOR="${mode}" \
    NAVRL_V2_SHARED_SOURCE_BUNDLE="${SHARED_BUNDLE}" \
    NAVRL_V2_RESULT_DIR="${cell_dir}" \
        ./eval_navrl_v2_density_sweep.sh "${checkpoint}" "${GAMES}"
}

echo "[abc15] campaign start/resume | seed=${SEED} games/cell=${GAMES} root=${RESULT_ROOT}"
for bars in "${DENSITIES[@]}"; do run_cell A_off "${SOURCE}" off "${bars}"; done
for bars in "${DENSITIES[@]}"; do run_cell B_source_riskcap "${SOURCE}" riskcap "${bars}"; done
for bars in "${DENSITIES[@]}"; do run_cell C_trained_riskcap "${TRAINED}" riskcap "${bars}"; done

validate_shared_bundle
"${PYTHON}" - "${RESULT_ROOT}" "${SOURCE_SHA}" "${TRAINED_SHA}" "${SEED}" "${GAMES}" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
source_sha, trained_sha = sys.argv[2], sys.argv[3]
seed, games = int(sys.argv[4]), int(sys.argv[5])
manifest_path = (root / "source_bundle/source_manifest.json").resolve()
manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
densities = [130, 160, 190, 205, 220]
arms = {
    "A_off": (source_sha, "off"),
    "B_source_riskcap": (source_sha, "riskcap"),
    "C_trained_riskcap": (trained_sha, "riskcap"),
}
cells = []

for arm, (checkpoint_sha, governor) in arms.items():
    for bars in densities:
        path = root / arm / f"{bars}bars" / f"{bars}bars.json"
        receipt_path = path.with_name(f"{bars}bars.receipt.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        condition = payload.get("condition") or {}
        contract = payload.get("v2_evaluation_contract") or {}
        outcome = payload.get("outcome") or {}
        failures = []
        if payload.get("checkpoint_sha256") != checkpoint_sha:
            failures.append("checkpoint SHA")
        if condition.get("seed") != seed or condition.get("bars") != bars:
            failures.append("seed/bars")
        if condition.get("speed_governor_mode") != governor:
            failures.append("governor")
        if condition.get("action_selection") != "deterministic":
            failures.append("action selection")
        if condition.get("reflection_mode") != "original":
            failures.append("reflection")
        if payload.get("runtime_source_manifest_sha256") != manifest_sha:
            failures.append("shared source manifest")
        if receipt.get("runtime_source_manifest_sha256") != manifest_sha:
            failures.append("receipt source manifest")
        if receipt.get("schema_version") != 2 or contract.get("schema_version") != 2:
            failures.append("schema-v2")
        if contract.get("episode_limit_steps") != 600 or contract.get("episode_limit_comparator") != "gte":
            failures.append("exact-600 horizon")
        actual = int(payload.get("actual_episodes", -1))
        if actual < games or actual >= games + 128:
            failures.append("episode count")
        counts = [int(outcome.get(name, -1)) for name in ("captured", "crash", "timeout")]
        if min(counts) < 0 or sum(counts) != actual:
            failures.append("outcome accounting")
        if failures:
            raise SystemExit(f"[abc15] invalid {arm}/{bars}: {', '.join(failures)}")
        cells.append({
            "arm": arm,
            "bars": bars,
            "episodes": actual,
            "captured": counts[0],
            "crashes": counts[1],
            "timeouts": counts[2],
            "capture_rate": counts[0] / actual,
            "crash_rate": counts[1] / actual,
            "timeout_rate": counts[2] / actual,
            "result": str(path),
        })

at = {(row["arm"], row["bars"]): row for row in cells}

def contrast(treatment, control, bars):
    hi, lo = at[(treatment, bars)], at[(control, bars)]
    p_hi, p_lo = hi["capture_rate"], lo["capture_rate"]
    diff = p_hi - p_lo
    se = math.sqrt(p_hi * (1 - p_hi) / hi["episodes"] + p_lo * (1 - p_lo) / lo["episodes"])
    return {
        "bars": bars,
        "capture_delta_pp": 100 * diff,
        "unadjusted_95ci_pp": [100 * (diff - 1.95996398454 * se), 100 * (diff + 1.95996398454 * se)],
    }

summary = {
    "schema_version": 2,
    "campaign": "governor_adaptation_abc",
    "seed": seed,
    "games_per_cell_requested": games,
    "densities": densities,
    "trained_support_max_bars": 205,
    "runtime_source_manifest": str(manifest_path),
    "runtime_source_manifest_sha256": manifest_sha,
    "source_checkpoint_sha256": source_sha,
    "trained_checkpoint_sha256": trained_sha,
    "cells": cells,
    "sequential_contrasts": {
        "B_minus_A_governor": [contrast("B_source_riskcap", "A_off", bars) for bars in densities],
        "C_minus_B_adaptation": [contrast("C_trained_riskcap", "B_source_riskcap", bars) for bars in densities],
    },
    "interpretation": [
        "B-A is the sequential governor contribution at a fixed ep24000 checkpoint.",
        "C-B is the sequential adaptation contribution under the same frozen riskcap.",
        "The displayed confidence intervals are unadjusted descriptive intervals across five densities.",
        "220 bars is OOD and is not part of the trained-support primary claim.",
    ],
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = [
    "# Governor/adaptation A/B/C — schema-v2 re-baseline",
    "",
    f"seed {seed}; deterministic; exact 600 actions; shared source `{manifest_sha[:12]}…`; ~{games} episodes/cell.",
    "",
    "| bars | A ep24000/off | B ep24000/riskcap | C ep25000/riskcap | B−A | C−B |",
    "|---:|---:|---:|---:|---:|---:|",
]
for bars in densities:
    a, b, c = (at[(arm, bars)]["capture_rate"] for arm in arms)
    lines.append(
        f"| {bars}{' (OOD)' if bars > 205 else ''} | {100*a:.2f}% | {100*b:.2f}% | "
        f"{100*c:.2f}% | {100*(b-a):+.2f} pp | {100*(c-b):+.2f} pp |"
    )
lines += [
    "",
    "B−A is the governor sequential contribution; C−B is the adaptation sequential contribution.",
    "Do not reinterpret either as an interaction. 220 bars is OOD.",
]
(root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[abc15] SUMMARY PASS -> {root / 'summary.md'}")
PY

echo "[abc15] COMPLETE | 15/15 cells | summary=${RESULT_ROOT}/summary.md"
