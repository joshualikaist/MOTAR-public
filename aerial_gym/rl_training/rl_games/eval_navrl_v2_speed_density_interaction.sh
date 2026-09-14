#!/usr/bin/env bash
# Preregistered Gate 2: target-speed x density interaction on the frozen v2 candidate.
#
# Grid: 2 unused seeds x 2 in-support endpoint speeds x 4 in-distribution densities = 16 cells.
# Policy/controller: ep25000 + frozen riskcap in every cell.
# Primary test: aggregate-binomial logistic density x speed interaction, adjusted for seed.
#
# Cells run sequentially on one GPU. A completed cell is skipped on restart; a partial cell
# directory is never overwritten. All cells share one immutable runtime-source bundle.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PYTHON PYTHONNOUSERSITE=1

POLICY="runs/ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1/nn/last_gen_ppo_ep_25000_rew_39.742134.pth"
POLICY_SHA="f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40"
SEEDS=(59 61)
SPEEDS=(0.3 1.5)
DENSITIES=(130 160 190 205)
GAMES=2049
RESULT_ROOT="${NAVRL_SPEED_DENSITY_RESULT_ROOT:-${REPO_ROOT}/results/navrl_v2_speed_density_interaction_seed59_61_schema2}"
SHARED_BUNDLE="${RESULT_ROOT}/source_bundle"
CAMPAIGN_CONTRACT="${RESULT_ROOT}/campaign_contract.json"
PREFLIGHT="${PREFLIGHT:-0}"

[[ -f "${POLICY}" ]] || { echo "[sd16] frozen policy missing: ${POLICY}" >&2; exit 2; }

"${PYTHON}" - "${POLICY}" "${POLICY_SHA}" <<'PY'
import hashlib
from pathlib import Path
import sys
import torch

path, expected_sha = Path(sys.argv[1]), sys.argv[2]
actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
if actual_sha != expected_sha:
    raise SystemExit(f"[sd16] policy SHA mismatch: {actual_sha}")
payload = torch.load(path, map_location="cpu", weights_only=False)
state = payload.get("env_state") or {}
if int(payload.get("epoch", -1)) != 25000:
    raise SystemExit("[sd16] frozen policy is not epoch 25000")
if int(state.get("n_bars_active", -1)) != 205:
    raise SystemExit("[sd16] frozen policy is not the 205-bar candidate")
if state.get("cfg_obstacle_selector") != "cluster_sector":
    raise SystemExit("[sd16] frozen policy selector is not cluster_sector")
PY

export NAVRL_V2_ACTION_MODE=deterministic
export NAVRL_EVAL_REFLECTION_MODE=original
export NAVRL_SPEED_GOVERNOR=riskcap
export NAVRL_SPEED_GOVERNOR_FIXED_MPS=2.0
export NAVRL_SPEED_GOVERNOR_FREE_MPS=3.53553390593
export NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M=0.45
export NAVRL_SPEED_GOVERNOR_MARGIN_M=0.45
export NAVRL_SPEED_GOVERNOR_SLOW_M=3.0
export NAVRL_SPEED_GOVERNOR_RELEASE_M=5.0
export NAVRL_SPEED_GOVERNOR_TTC_S=1.2
export NAVRL_SPEED_GOVERNOR_BRAKE_MPS2=2.9608856678
export NAVRL_SPEED_GOVERNOR_REACTION_S=0.1
unset NAVRL_DETECTOR_CHECKPOINT

if [[ "${PREFLIGHT}" == "1" ]]; then
    for seed in "${SEEDS[@]}"; do
        for speed in "${SPEEDS[@]}"; do
            NAVRL_SEED="${seed}" NAVRL_V2_FIXED_TARGET_SPEED="${speed}" \
            NAVRL_V2_DENSITIES="130 160 190 205" NAVRL_PREFLIGHT_ONLY=1 \
                ./eval_navrl_v2_density_sweep.sh "${POLICY}" "${GAMES}"
        done
    done
    echo "[sd16] PREFLIGHT PASS | 16 cells, seeds=59,61 speeds=0.3,1.5 bars=130,160,190,205"
    exit 0
fi

if [[ -e "${RESULT_ROOT}" && ! -f "${CAMPAIGN_CONTRACT}" ]]; then
    echo "[sd16] refusing unrelated/partial result root without campaign contract: ${RESULT_ROOT}" >&2
    exit 2
fi
mkdir -p "${RESULT_ROOT}"

"${PYTHON}" - "${CAMPAIGN_CONTRACT}" "${POLICY_SHA}" "${GAMES}" "${BASH_SOURCE[0]}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
launcher = Path(sys.argv[4]).resolve()
expected = {
    "schema_version": 2,
    "campaign": "speed_density_interaction",
    "policy_checkpoint_sha256": sys.argv[2],
    "games_per_cell": int(sys.argv[3]),
    "seeds": [59, 61],
    "target_speeds_mps": [0.3, 1.5],
    "densities": [130, 160, 190, 205],
    "trained_support_max_bars": 205,
    "action_selection": "deterministic",
    "reflection_mode": "original",
    "speed_governor_mode": "riskcap",
    "primary_model": "binomial_logit(capture) ~ seed + density + fast + density:fast",
    "primary_test": "likelihood-ratio test of density:fast, 1 df",
    "launcher": str(launcher),
    "launcher_sha256": hashlib.sha256(launcher.read_bytes()).hexdigest(),
}
if path.exists():
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != expected:
        raise SystemExit("[sd16] existing campaign contract does not match this launcher")
else:
    path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

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
    raise SystemExit("[sd16] shared source manifest is missing")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("schema_version") != 2:
    raise SystemExit("[sd16] shared source manifest schema is not 2")
repo = Path(manifest["repository_root"]).resolve()
files = manifest.get("runtime_files") or []
if len(files) != int(manifest.get("runtime_file_count", -1)) or not files:
    raise SystemExit("[sd16] shared source file accounting is invalid")
for entry in files:
    original = (repo / entry["path"]).resolve()
    snapshot = (manifest_path.parent / entry["snapshot"]).resolve()
    expected = entry["sha256"]
    for candidate, label in ((original, "runtime source"), (snapshot, "source snapshot")):
        if not candidate.is_file() or hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"[sd16] {label} changed: {entry['path']}")
PY
}

run_cell() {
    local seed="$1"
    local speed="$2"
    local bars="$3"
    local speed_tag="${speed/./p}"
    local cell_dir="${RESULT_ROOT}/seed${seed}/speed_${speed_tag}/${bars}bars"
    local result="${cell_dir}/${bars}bars.json"
    local receipt="${cell_dir}/${bars}bars.receipt.json"

    if [[ -e "${cell_dir}" ]]; then
        if [[ -f "${result}" && -f "${receipt}" ]]; then
            echo "[sd16] SKIP complete | seed=${seed} speed=${speed} bars=${bars}"
            return 0
        fi
        echo "[sd16] partial cell directory requires manual inspection/move: ${cell_dir}" >&2
        exit 3
    fi

    validate_shared_bundle
    mkdir -p "$(dirname "${cell_dir}")"
    echo "[sd16] RUN | seed=${seed} speed=${speed} bars=${bars} games=${GAMES}"
    NAVRL_SEED="${seed}" \
    NAVRL_V2_FIXED_TARGET_SPEED="${speed}" \
    NAVRL_V2_DENSITIES="${bars}" \
    NAVRL_V2_SHARED_SOURCE_BUNDLE="${SHARED_BUNDLE}" \
    NAVRL_V2_RESULT_DIR="${cell_dir}" \
        ./eval_navrl_v2_density_sweep.sh "${POLICY}" "${GAMES}"
}

echo "[sd16] campaign start/resume | 16 cells root=${RESULT_ROOT}"
for seed in "${SEEDS[@]}"; do
    for speed in "${SPEEDS[@]}"; do
        for bars in "${DENSITIES[@]}"; do
            run_cell "${seed}" "${speed}" "${bars}"
        done
    done
done

validate_shared_bundle
"${PYTHON}" - "${RESULT_ROOT}" "${POLICY_SHA}" "${GAMES}" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

root = Path(sys.argv[1]).resolve()
policy_sha = sys.argv[2]
games = int(sys.argv[3])
seeds, speeds, densities = [59, 61], [0.3, 1.5], [130, 160, 190, 205]
manifest_path = (root / "source_bundle/source_manifest.json").resolve()
manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
cells = []

for seed in seeds:
    for speed in speeds:
        speed_tag = str(speed).replace(".", "p")
        for bars in densities:
            path = root / f"seed{seed}" / f"speed_{speed_tag}" / f"{bars}bars" / f"{bars}bars.json"
            receipt_path = path.with_name(f"{bars}bars.receipt.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            condition = payload.get("condition") or {}
            contract = payload.get("v2_evaluation_contract") or {}
            outcome = payload.get("outcome") or {}
            failures = []
            if payload.get("checkpoint_sha256") != policy_sha:
                failures.append("checkpoint SHA")
            if condition.get("seed") != seed or condition.get("bars") != bars:
                failures.append("seed/bars")
            if condition.get("speed_governor_mode") != "riskcap":
                failures.append("governor")
            if condition.get("action_selection") != "deterministic":
                failures.append("action selection")
            if condition.get("reflection_mode") != "original":
                failures.append("reflection")
            if not math.isclose(float(condition.get("target_speed_mps", -1)), speed, abs_tol=1e-12):
                failures.append("target speed")
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
                raise SystemExit(f"[sd16] invalid seed{seed}/speed{speed}/{bars}: {', '.join(failures)}")
            cells.append({
                "seed": seed,
                "target_speed_mps": speed,
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

def design(include_interaction):
    rows, captured, episodes = [], [], []
    density_center = sum(densities) / len(densities)
    for cell in cells:
        density_scaled = (cell["bars"] - density_center) / 30.0
        fast = float(cell["target_speed_mps"] == 1.5)
        row = [1.0, float(cell["seed"] == 61), density_scaled, fast]
        if include_interaction:
            row.append(density_scaled * fast)
        rows.append(row)
        captured.append(cell["captured"])
        episodes.append(cell["episodes"])
    return np.asarray(rows), np.asarray(captured, dtype=float), np.asarray(episodes, dtype=float)

def fit(include_interaction):
    x, y, n = design(include_interaction)
    beta = np.zeros(x.shape[1], dtype=float)
    for _ in range(100):
        eta = x @ beta
        probability = 1.0 / (1.0 + np.exp(-np.clip(eta, -40.0, 40.0)))
        gradient = x.T @ (y - n * probability)
        weight = n * probability * (1.0 - probability)
        information = x.T @ (weight[:, None] * x)
        step = np.linalg.solve(information, gradient)
        beta += step
        if float(np.max(np.abs(step))) < 1e-12:
            break
    eta = x @ beta
    log_likelihood = float(np.sum(y * eta - n * np.logaddexp(0.0, eta)))
    covariance = np.linalg.inv(information)
    return beta, covariance, log_likelihood

reduced_beta, _, reduced_ll = fit(False)
full_beta, full_covariance, full_ll = fit(True)
lr_statistic = max(0.0, 2.0 * (full_ll - reduced_ll))
primary_p = math.erfc(math.sqrt(lr_statistic / 2.0))
interaction_beta = float(full_beta[-1])
interaction_se = math.sqrt(float(full_covariance[-1, -1]))

summary = {
    "schema_version": 2,
    "campaign": "speed_density_interaction",
    "policy_checkpoint_sha256": policy_sha,
    "runtime_source_manifest": str(manifest_path),
    "runtime_source_manifest_sha256": manifest_sha,
    "games_per_cell_requested": games,
    "seeds": seeds,
    "target_speeds_mps": speeds,
    "densities": densities,
    "trained_support_max_bars": 205,
    "cells": cells,
    "primary_test": {
        "model": "binomial_logit(capture) ~ seed + density + fast + density:fast",
        "density_scale_bars": 30.0,
        "interaction_log_odds_per_30_bars": interaction_beta,
        "interaction_standard_error": interaction_se,
        "likelihood_ratio_chi2": lr_statistic,
        "degrees_of_freedom": 1,
        "p_value": primary_p,
        "alpha": 0.05,
        "interaction_detected": primary_p < 0.05,
    },
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

at = {(c["seed"], c["target_speed_mps"], c["bars"]): c for c in cells}
lines = [
    "# Speed × density interaction — schema-v2 primary test",
    "",
    f"Seeds 59/61; deterministic ep25000+riskcap; exact 600 actions; shared source `{manifest_sha[:12]}…`.",
    "",
]
for seed in seeds:
    lines += [f"## Seed {seed}", "", "| bars | 0.3 m/s | 1.5 m/s | fast−slow |", "|---:|---:|---:|---:|"]
    for bars in densities:
        slow = at[(seed, 0.3, bars)]["capture_rate"]
        fast = at[(seed, 1.5, bars)]["capture_rate"]
        lines.append(f"| {bars} | {100*slow:.2f}% | {100*fast:.2f}% | {100*(fast-slow):+.2f} pp |")
    lines.append("")
lines += [
    "## Preregistered primary test",
    "",
    "Aggregate-binomial logistic model: `capture ~ seed + density + fast + density:fast`.",
    f"Interaction LR chi-square(1) = **{lr_statistic:.4f}**, p = **{primary_p:.6g}**.",
    f"Decision at alpha=0.05: **{'interaction detected' if primary_p < 0.05 else 'interaction not detected'}**.",
    "",
    "All four densities are within the trained support. No OOD 220-bar cell enters the primary test.",
]
(root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[sd16] SUMMARY PASS -> {root / 'summary.md'}")
PY

echo "[sd16] COMPLETE | 16/16 cells | summary=${RESULT_ROOT}/summary.md"
