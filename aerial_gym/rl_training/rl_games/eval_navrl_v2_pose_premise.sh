#!/usr/bin/env bash
# 검증 3: how sensitive is P3's -2.5 pp latency residual to its exact-clock/exact-pose premise?
#
# Preregistered in WORKLOG 2026-08-12 before any data. Frozen ep25000+riskcap, 205 bars,
# deterministic, exact-600, tau = 0.1 s with P3 (capture-time pose transform) at its default ON,
# unused seed 163, ~2049 episodes/cell. 12 cells:
#
#   exact                            (anchor: P3 with a perfect clock and perfect odometry)
#   clock offset  +0.02/+0.05/+0.10  (pose stamped LATE; +0.10 = +tau lands on the current pose
#                 -0.02/-0.05         and reproduces the naive pre-P3 transform -- built-in anchor)
#   pose noise    0.01/0.03/0.10 m   (gaussian odometry position error, per axis)
#   yaw noise     0.5/2/5 deg        (gaussian odometry yaw error about world z)
#
# Fractional offsets interpolate between odometry samples, so the offset arms jointly exercise
# clock skew AND pose interpolation. Read: capture vs the exact anchor per cell. The +0.10 cell
# is expected DOWN NEAR THE NAIVE-TRANSFORM level (~38% on other seeds); if it is not, the knob
# is not doing what it claims and the campaign is void. No retry.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PYTHON

POLICY="runs/ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1/nn/last_gen_ppo_ep_25000_rew_39.742134.pth"
POLICY_SHA="f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40"
RESULT_ROOT="../../../results/navrl_v2_pose_premise_seed163"
PREFLIGHT="${PREFLIGHT:-0}"
EPISODES="${EPISODES:-2049}"

if [[ ! -f "${POLICY}" ]]; then
    echo "[pose-premise] policy checkpoint missing: ${POLICY}" >&2
    exit 2
fi
if [[ -e "${RESULT_ROOT}" && "${PREFLIGHT}" != "1" ]]; then
    echo "[pose-premise] refusing to overwrite ${RESULT_ROOT}" >&2
    exit 2
fi

ACTUAL_POLICY_SHA="$(
    "${PYTHON}" - "${POLICY}" "${POLICY_SHA}" <<'PY'
import hashlib
from pathlib import Path
import sys

policy, expected = Path(sys.argv[1]), sys.argv[2]
actual = hashlib.sha256(policy.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f"[pose-premise] policy SHA mismatch: {actual}")
print(actual)
PY
)"

export NAVRL_V2_DENSITIES=205
export NAVRL_V2_ACTION_MODE=deterministic
export NAVRL_EVAL_REFLECTION_MODE=original
export NAVRL_SPEED_GOVERNOR=riskcap
export NAVRL_SEED=163
export NAVRL_SPEED_GOVERNOR_FIXED_MPS=2.0
export NAVRL_SPEED_GOVERNOR_FREE_MPS=3.53553390593
export NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M=0.45
export NAVRL_SPEED_GOVERNOR_MARGIN_M=0.45
export NAVRL_SPEED_GOVERNOR_SLOW_M=3.0
export NAVRL_SPEED_GOVERNOR_RELEASE_M=5.0
export NAVRL_SPEED_GOVERNOR_TTC_S=1.2
export NAVRL_SPEED_GOVERNOR_BRAKE_MPS2=2.9608856678
export NAVRL_SPEED_GOVERNOR_REACTION_S=0.1
unset NAVRL_DETECTOR_CHECKPOINT  # analytic bootstrap: isolates the pose premise from 검증 2

run_cell() {
    local tag="$1"
    local clock="$2"
    local pos_noise="$3"
    local yaw_noise="$4"
    echo "[pose-premise] cell=${tag} clock=${clock}s pos=${pos_noise}m yaw=${yaw_noise}deg"
    if [[ "${PREFLIGHT}" == "1" ]]; then
        return 0
    fi
    NAVRL_PERCEPTION_PERTURB=1 \
    NAVRL_DETECTION_DROPOUT=0 \
    NAVRL_RANGE_ERROR_M=0 \
    NAVRL_DETECTION_LATENCY_S=0.1 \
    NAVRL_POSE_CLOCK_OFFSET_S="${clock}" \
    NAVRL_POSE_NOISE_POS_M="${pos_noise}" \
    NAVRL_POSE_NOISE_YAW_DEG="${yaw_noise}" \
    NAVRL_V2_RESULT_DIR="${RESULT_ROOT}/${tag}" \
        ./eval_navrl_v2_density_sweep.sh "${POLICY}" "${EPISODES}"
}

run_cell exact          0      0    0
run_cell clk_p0p02      0.02   0    0
run_cell clk_p0p05      0.05   0    0
run_cell clk_p0p10      0.10   0    0
run_cell clk_m0p02     -0.02   0    0
run_cell clk_m0p05     -0.05   0    0
run_cell posn_0p01      0      0.01 0
run_cell posn_0p03      0      0.03 0
run_cell posn_0p10      0      0.10 0
run_cell yaw_0p5        0      0    0.5
run_cell yaw_2          0      0    2
run_cell yaw_5          0      0    5

if [[ "${PREFLIGHT}" == "1" ]]; then
    echo "[pose-premise] PREFLIGHT PASS | 12 cells seed=163 tau=0.1 P3=on"
    exit 0
fi

"${PYTHON}" - "${RESULT_ROOT}" "${ACTUAL_POLICY_SHA}" <<'PY'
import json
import math
from pathlib import Path
import sys

root, policy_sha = Path(sys.argv[1]), sys.argv[2]
ORDER = ["exact", "clk_m0p05", "clk_m0p02", "clk_p0p02", "clk_p0p05", "clk_p0p10",
         "posn_0p01", "posn_0p03", "posn_0p10", "yaw_0p5", "yaw_2", "yaw_5"]
cells = {}
for tag in ORDER:
    payload = json.loads((root / tag / "205bars.json").read_text(encoding="utf-8"))
    out = payload["outcome"]
    cells[tag] = {"episodes": payload["actual_episodes"], "captured": out["captured"],
                  "capture": out["capture_rate"], "crash": out["crash_rate"],
                  "timeout": out["timeout_rate"]}

anchor = cells["exact"]


def diff_ci(c):
    pa, pb = c["capture"], anchor["capture"]
    d = pa - pb
    se = math.sqrt(pa * (1 - pa) / c["episodes"] + pb * (1 - pb) / anchor["episodes"])
    return d * 100, (d - 1.96 * se) * 100, (d + 1.96 * se) * 100


summary = {"policy_sha256": policy_sha, "seed": 163, "tau_s": 0.1,
           "contract": "205bars/deterministic/riskcap/exact-600",
           "cells": cells,
           "delta_vs_exact_pp": {t: diff_ci(cells[t]) for t in ORDER if t != "exact"}}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")
lines = ["# P3 pose-premise sensitivity (ep25000+riskcap, tau=0.1, seed 163, 205 bars)", "",
         "| cell | capture | crash | timeout | Δ vs exact (95% CI) |", "|---|---:|---:|---:|---:|"]
for tag in ORDER:
    c = cells[tag]
    if tag == "exact":
        delta = "— (anchor)"
    else:
        d, lo, hi = diff_ci(c)
        delta = f"{d:+.2f} pp [{lo:+.2f}, {hi:+.2f}]"
    lines.append(f"| {tag} | {c['capture']*100:.2f}% | {c['crash']*100:.2f}% | "
                 f"{c['timeout']*100:.2f}% | {delta} |")
lines += ["", "clk_p0p10 (= +tau) must sit near the naive-transform level; it validates the knob."]
(root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
echo "[pose-premise] done -> ${RESULT_ROOT}/summary.{md,json}"
