#!/usr/bin/env bash
# Gate 3 stage B: frozen-policy analytic-bootstrap vs learned-detector navigation A/B.
#
# 2 unused seeds x 2 detector arms x 205 bars = 4 cells. The primary endpoint is pooled capture
# non-inferiority of learned vs analytic with a preregistered -2 percentage-point margin. PPO,
# riskcap, threshold, source, action selection, and exact-600 horizon are identical across arms.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PYTHON PYTHONNOUSERSITE=1
PYTHON_BIN="$(cd "$(dirname "${PYTHON}")" && pwd)"
export PATH="${PYTHON_BIN}:${PATH}"

POLICY="runs/ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1/nn/last_gen_ppo_ep_25000_rew_39.742134.pth"
POLICY_SHA="f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40"
DETECTOR="../../../artifacts/navrl_target_detector_v2.pth"
DETECTOR_RECEIPT="../../../artifacts/navrl_target_detector_v2.receipt.json"
DETECTOR_SHA="8da32d6f21bfbd3bdd5ec5de9ef9cb09e8deb4bd5ce511630e19afee33f26f10"
THRESHOLD=0.55
SEEDS=(83 89)
GAMES=2049
NONINFERIORITY_MARGIN_PP=2.0
RESULT_ROOT="${NAVRL_DETECTOR_AB_RESULT_ROOT:-${REPO_ROOT}/results/navrl_v2_detector_navigation_ab_seed83_89_schema2}"
SHARED_BUNDLE="${RESULT_ROOT}/source_bundle"
CAMPAIGN_CONTRACT="${RESULT_ROOT}/campaign_contract.json"
PREFLIGHT="${PREFLIGHT:-0}"

"${PYTHON}" - "${POLICY}" "${POLICY_SHA}" "${DETECTOR}" "${DETECTOR_SHA}" \
    "${DETECTOR_RECEIPT}" "${THRESHOLD}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import torch

policy, policy_sha = Path(sys.argv[1]), sys.argv[2]
detector, detector_sha = Path(sys.argv[3]), sys.argv[4]
receipt_path, threshold = Path(sys.argv[5]), float(sys.argv[6])
for path, expected, label in (
    (policy, policy_sha, "policy"),
    (detector, detector_sha, "detector"),
):
    if not path.is_file():
        raise SystemExit(f"[det-ab] missing {label}: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"[det-ab] {label} SHA mismatch: {actual}")
payload = torch.load(policy, map_location="cpu", weights_only=False)
state = payload.get("env_state") or {}
if int(payload.get("epoch", -1)) != 25000 or int(state.get("n_bars_active", -1)) != 205:
    raise SystemExit("[det-ab] policy is not the frozen ep25000/205-bar candidate")
if state.get("cfg_obstacle_selector") != "cluster_sector":
    raise SystemExit("[det-ab] policy selector is not cluster_sector")
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
if not receipt.get("gate_passed"):
    raise SystemExit("[det-ab] offline detector gate did not pass")
if receipt.get("artifact_sha256") != detector_sha:
    raise SystemExit("[det-ab] detector receipt SHA mismatch")
if abs(float(receipt.get("selected_threshold", -1)) - threshold) > 1e-12:
    raise SystemExit("[det-ab] detector threshold differs from validation-selected threshold")
PY

export NAVRL_V2_DENSITIES=205
export NAVRL_V2_ACTION_MODE=deterministic
export NAVRL_EVAL_REFLECTION_MODE=original
export NAVRL_SPEED_GOVERNOR=riskcap
export NAVRL_DETECTOR_THRESHOLD="${THRESHOLD}"
export NAVRL_PERCEPTION_PERTURB=0
export NAVRL_SPEED_GOVERNOR_FIXED_MPS=2.0
export NAVRL_SPEED_GOVERNOR_FREE_MPS=3.53553390593
export NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M=0.45
export NAVRL_SPEED_GOVERNOR_MARGIN_M=0.45
export NAVRL_SPEED_GOVERNOR_SLOW_M=3.0
export NAVRL_SPEED_GOVERNOR_RELEASE_M=5.0
export NAVRL_SPEED_GOVERNOR_TTC_S=1.2
export NAVRL_SPEED_GOVERNOR_BRAKE_MPS2=2.9608856678
export NAVRL_SPEED_GOVERNOR_REACTION_S=0.1

preflight_arm() {
    local seed="$1"
    local detector="${2:-}"
    if [[ -n "${detector}" ]]; then
        NAVRL_SEED="${seed}" NAVRL_DETECTOR_CHECKPOINT="${detector}" NAVRL_PREFLIGHT_ONLY=1 \
            ./eval_navrl_v2_density_sweep.sh "${POLICY}" "${GAMES}"
    else
        unset NAVRL_DETECTOR_CHECKPOINT
        NAVRL_SEED="${seed}" NAVRL_PREFLIGHT_ONLY=1 \
            ./eval_navrl_v2_density_sweep.sh "${POLICY}" "${GAMES}"
    fi
}

if [[ "${PREFLIGHT}" == "1" ]]; then
    for seed in "${SEEDS[@]}"; do
        preflight_arm "${seed}" ""
        preflight_arm "${seed}" "${DETECTOR}"
    done
    echo "[det-ab] PREFLIGHT PASS | 4 cells seeds=83,89 bars=205 margin=-2pp"
    exit 0
fi

if [[ -e "${RESULT_ROOT}" && ! -f "${CAMPAIGN_CONTRACT}" ]]; then
    echo "[det-ab] refusing result root without matching campaign contract: ${RESULT_ROOT}" >&2
    exit 2
fi
mkdir -p "${RESULT_ROOT}"

"${PYTHON}" - "${CAMPAIGN_CONTRACT}" "${POLICY_SHA}" "${DETECTOR_SHA}" \
    "${THRESHOLD}" "${GAMES}" "${BASH_SOURCE[0]}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
launcher = Path(sys.argv[6]).resolve()
expected = {
    "schema_version": 2,
    "campaign": "detector_navigation_ab",
    "policy_checkpoint_sha256": sys.argv[2],
    "learned_detector_sha256": sys.argv[3],
    "detector_threshold": float(sys.argv[4]),
    "games_per_cell": int(sys.argv[5]),
    "seeds": [83, 89],
    "bars": 205,
    "arms": ["analytic_bootstrap", "learned_v2"],
    "action_selection": "deterministic",
    "reflection_mode": "original",
    "speed_governor_mode": "riskcap",
    "primary_endpoint": "pooled capture learned-minus-analytic",
    "noninferiority_margin_pp": -2.0,
    "decision": "PASS iff two-sided 95% CI lower bound is greater than -2.0 pp",
    "launcher": str(launcher),
    "launcher_sha256": hashlib.sha256(launcher.read_bytes()).hexdigest(),
}
if path.exists():
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != expected:
        raise SystemExit("[det-ab] existing campaign contract differs from launcher")
else:
    path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

exec > >(tee -a "${RESULT_ROOT}/campaign.log") 2>&1

run_cell() {
    local seed="$1"
    local arm="$2"
    local detector="${3:-}"
    local cell_dir="${RESULT_ROOT}/seed${seed}/${arm}/205bars"
    local result="${cell_dir}/205bars.json"
    local receipt="${cell_dir}/205bars.receipt.json"
    if [[ -e "${cell_dir}" ]]; then
        if [[ -f "${result}" && -f "${receipt}" ]]; then
            echo "[det-ab] SKIP complete | seed=${seed} arm=${arm}"
            return 0
        fi
        echo "[det-ab] partial cell requires manual inspection/move: ${cell_dir}" >&2
        exit 3
    fi
    echo "[det-ab] RUN | seed=${seed} arm=${arm} games=${GAMES}"
    if [[ -n "${detector}" ]]; then
        NAVRL_SEED="${seed}" NAVRL_DETECTOR_CHECKPOINT="${detector}" \
        NAVRL_V2_SHARED_SOURCE_BUNDLE="${SHARED_BUNDLE}" NAVRL_V2_RESULT_DIR="${cell_dir}" \
            ./eval_navrl_v2_density_sweep.sh "${POLICY}" "${GAMES}"
    else
        unset NAVRL_DETECTOR_CHECKPOINT
        NAVRL_SEED="${seed}" NAVRL_V2_SHARED_SOURCE_BUNDLE="${SHARED_BUNDLE}" \
        NAVRL_V2_RESULT_DIR="${cell_dir}" \
            ./eval_navrl_v2_density_sweep.sh "${POLICY}" "${GAMES}"
    fi
}

echo "[det-ab] campaign start/resume | root=${RESULT_ROOT}"
for seed in "${SEEDS[@]}"; do
    run_cell "${seed}" analytic_bootstrap ""
    run_cell "${seed}" learned_v2 "${DETECTOR}"
done

"${PYTHON}" - "${RESULT_ROOT}" "${POLICY_SHA}" "${DETECTOR_SHA}" "${GAMES}" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
policy_sha, detector_sha, games = sys.argv[2], sys.argv[3], int(sys.argv[4])
seeds = [83, 89]
arms = {"analytic_bootstrap": "", "learned_v2": detector_sha}
manifest = root / "source_bundle/source_manifest.json"
manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
cells = []
for seed in seeds:
    for arm, expected_detector_sha in arms.items():
        path = root / f"seed{seed}" / arm / "205bars/205bars.json"
        receipt_path = path.with_name("205bars.receipt.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        condition = payload.get("condition") or {}
        contract = payload.get("v2_evaluation_contract") or {}
        outcome = payload.get("outcome") or {}
        failures = []
        if payload.get("checkpoint_sha256") != policy_sha:
            failures.append("policy SHA")
        if payload.get("runtime_source_manifest_sha256") != manifest_sha:
            failures.append("shared source")
        if receipt.get("runtime_source_manifest_sha256") != manifest_sha:
            failures.append("receipt source")
        if receipt.get("evaluated_detector_snapshot_sha256", "") != expected_detector_sha:
            failures.append("detector SHA")
        if condition.get("seed") != seed or condition.get("bars") != 205:
            failures.append("seed/bars")
        if condition.get("action_selection") != "deterministic":
            failures.append("action")
        if condition.get("speed_governor_mode") != "riskcap":
            failures.append("governor")
        if abs(float(contract.get("detector_threshold", -1)) - 0.55) > 1e-12:
            failures.append("threshold")
        if contract.get("episode_limit_steps") != 600 or contract.get("timeout_observed_at_step") != 600:
            failures.append("exact-600")
        actual = int(payload.get("actual_episodes", -1))
        counts = [int(outcome.get(name, -1)) for name in ("captured", "crash", "timeout")]
        if actual < games or actual >= games + 128 or min(counts) < 0 or sum(counts) != actual:
            failures.append("outcomes")
        if failures:
            raise SystemExit(f"[det-ab] invalid seed{seed}/{arm}: {', '.join(failures)}")
        cells.append({
            "seed": seed,
            "arm": arm,
            "episodes": actual,
            "captured": counts[0],
            "crashes": counts[1],
            "timeouts": counts[2],
            "capture_rate": counts[0] / actual,
            "crash_rate": counts[1] / actual,
            "timeout_rate": counts[2] / actual,
            "result": str(path),
        })

def pooled(arm, field):
    selected = [cell for cell in cells if cell["arm"] == arm]
    count_field = {"capture_rate": "captured", "crash_rate": "crashes", "timeout_rate": "timeouts"}[field]
    successes = sum(cell[count_field] for cell in selected)
    episodes = sum(cell["episodes"] for cell in selected)
    return successes, episodes, successes / episodes

_, n_a, p_a = pooled("analytic_bootstrap", "capture_rate")
_, n_l, p_l = pooled("learned_v2", "capture_rate")
diff = p_l - p_a
se = math.sqrt(p_a * (1 - p_a) / n_a + p_l * (1 - p_l) / n_l)
capture_ci = [diff - 1.95996398454 * se, diff + 1.95996398454 * se]
noninferiority_pass = capture_ci[0] > -0.02
_, _, crash_a = pooled("analytic_bootstrap", "crash_rate")
_, _, crash_l = pooled("learned_v2", "crash_rate")
_, _, timeout_a = pooled("analytic_bootstrap", "timeout_rate")
_, _, timeout_l = pooled("learned_v2", "timeout_rate")

summary = {
    "schema_version": 2,
    "campaign": "detector_navigation_ab",
    "policy_checkpoint_sha256": policy_sha,
    "learned_detector_sha256": detector_sha,
    "runtime_source_manifest_sha256": manifest_sha,
    "seeds": seeds,
    "bars": 205,
    "cells": cells,
    "primary_noninferiority": {
        "learned_minus_analytic_capture_pp": 100 * diff,
        "two_sided_95ci_pp": [100 * capture_ci[0], 100 * capture_ci[1]],
        "margin_pp": -2.0,
        "passed": noninferiority_pass,
    },
    "pooled": {
        "analytic": {"capture": p_a, "crash": crash_a, "timeout": timeout_a, "episodes": n_a},
        "learned": {"capture": p_l, "crash": crash_l, "timeout": timeout_l, "episodes": n_l},
    },
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
at = {(cell["seed"], cell["arm"]): cell for cell in cells}
lines = [
    "# Detector navigation A/B — schema-v2",
    "",
    "Frozen ep25000+riskcap, 205 bars, deterministic exact-600; learned threshold 0.55.",
    "",
    "| seed | analytic capture/crash/timeout | learned capture/crash/timeout | capture delta |",
    "|---:|---:|---:|---:|",
]
for seed in seeds:
    a, learned = at[(seed, "analytic_bootstrap")], at[(seed, "learned_v2")]
    lines.append(
        "| %d | %.2f/%.2f/%.2f%% | %.2f/%.2f/%.2f%% | %+.2f pp |"
        % (
            seed,
            100 * a["capture_rate"], 100 * a["crash_rate"], 100 * a["timeout_rate"],
            100 * learned["capture_rate"], 100 * learned["crash_rate"], 100 * learned["timeout_rate"],
            100 * (learned["capture_rate"] - a["capture_rate"]),
        )
    )
lines += [
    "",
    "## Primary non-inferiority result",
    "",
    "- pooled analytic capture: **%.2f%%** (n=%d)" % (100 * p_a, n_a),
    "- pooled learned capture: **%.2f%%** (n=%d)" % (100 * p_l, n_l),
    "- learned−analytic: **%+.2f pp**, 95%% CI **[%+.2f, %+.2f] pp**"
    % (100 * diff, 100 * capture_ci[0], 100 * capture_ci[1]),
    "- preregistered margin −2.0 pp: **%s**" % ("PASS" if noninferiority_pass else "FAIL"),
]
(root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("[det-ab] SUMMARY PASS -> %s" % (root / "summary.md"))
PY

echo "[det-ab] COMPLETE | 4/4 cells | summary=${RESULT_ROOT}/summary.md"
