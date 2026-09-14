#!/usr/bin/env bash
# Latency P3 arms: lift the delayed detection with the pose it was TAKEN at.
#
# Three fixes have now been measured and rejected (WORKLOG 2026-08-05/06):
#   P0 forward predict      capture -0.10 pp   (removed the v*tau target lag, changed nothing)
#   P0+P1 lidar backup      capture -8.85 pp   (more mis-association onto bars)
#   P2 map carve-out fix    capture +1.84 pp   (right sign, recovers only ~11% of the bar hits)
#
# What every one of them left untouched: observe() lifted a t-tau VEHICLE-frame measurement to
# world with the pose at t, so the drone's OWN motion over tau entered every KF correction.
# Measured on this policy's own trajectories (tests/test_navrl_latency_compensate.py):
#   translation at 2.33 m/s mean speed .... 0.233 m
#   yaw at 0.81 rad/s mean rate .......... 0.408 m of lateral error at target range
#   combined ............................. 0.255 m   (vs the 0.150 m target lag P0 removed)
# P3 buffers the pose beside the detection and lifts with it, which drops the static-target
# error to exactly 0. What remains is a measurement that is accurate but tau old -- precisely
# the residual P0 was written for, hence the p3_p0 arm.
#
#   arm                        latency  P3   P0   P2
#   latency_0p1s_raw             0.1    off  off  off    (37.82 / 58.22 재현, 앵커)
#   latency_0p1s_p3              0.1    ON   off  off
#   latency_0p1s_p3_p0           0.1    ON   ON   off    ego-motion + target lag 모두 제거
#   latency_0p1s_p3_p0_predict   0.1    ON   ON   predict  전체 스택
#
# GO gate (unchanged): capture >= 65% AND crash >= 10 pp below the raw arm. bar_contact 절대수도
# raw 대비 함께 보고한다.
#
# Usage:
#   ./eval_navrl_v2_latency_budget.sh
#   PREFLIGHT=1 ./eval_navrl_v2_latency_budget.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PYTHON

POLICY="runs/ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1/nn/last_gen_ppo_ep_25000_rew_39.742134.pth"
POLICY_SHA="f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40"
RESULT_ROOT="../../../results/navrl_v2_latency_budget"
BASELINE_ROOT="../../../results/navrl_v2_latency_compensate"
PREFLIGHT="${PREFLIGHT:-0}"
EPISODES="${EPISODES:-2049}"

if [[ ! -f "${POLICY}" ]]; then
    echo "[lat-budget] policy checkpoint missing: ${POLICY}" >&2
    exit 2
fi
if [[ -e "${RESULT_ROOT}" && "${PREFLIGHT}" != "1" ]]; then
    echo "[lat-budget] refusing to overwrite ${RESULT_ROOT}" >&2
    exit 2
fi

ACTUAL_POLICY_SHA="$(
    "${PYTHON}" - "${POLICY}" "${POLICY_SHA}" <<'PY'
import hashlib
from pathlib import Path
import re
import sys
import torch

trained, expected = Path(sys.argv[1]), sys.argv[2]
actual = hashlib.sha256(trained.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f"[lat-budget] policy SHA mismatch: {actual}")
payload = torch.load(trained, map_location="cpu", weights_only=False)
state = payload.get("env_state") or {}
checks = {
    "filename": re.fullmatch(r"last_gen_ppo_ep_25000_rew_-?[0-9]+(?:\.[0-9]+)?\.pth", trained.name) is not None,
    "epoch": int(payload.get("epoch", -1)) == 25000,
    "bars": int(state.get("n_bars_active", -1)) == 205,
    "selector": state.get("cfg_obstacle_selector") == "cluster_sector",
    "governor": state.get("cfg_speed_governor_mode") == "riskcap",
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("[lat-budget] invalid policy checkpoint: " + ", ".join(failed))
print(actual)
PY
)"

# Frozen evaluation contract -- byte-identical to the P0/P1/P2 and R3 harnesses.
export NAVRL_V2_DENSITIES=205
export NAVRL_V2_ACTION_MODE=deterministic
export NAVRL_EVAL_REFLECTION_MODE=original
export NAVRL_SPEED_GOVERNOR=riskcap
export NAVRL_SEED=47
export NAVRL_SPEED_GOVERNOR_FIXED_MPS=2.0
export NAVRL_SPEED_GOVERNOR_FREE_MPS=3.53553390593
export NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M=0.45
export NAVRL_SPEED_GOVERNOR_MARGIN_M=0.45
export NAVRL_SPEED_GOVERNOR_SLOW_M=3.0
export NAVRL_SPEED_GOVERNOR_RELEASE_M=5.0
export NAVRL_SPEED_GOVERNOR_TTC_S=1.2
export NAVRL_SPEED_GOVERNOR_BRAKE_MPS2=2.9608856678
export NAVRL_SPEED_GOVERNOR_REACTION_S=0.1
unset NAVRL_DETECTOR_CHECKPOINT  # bootstrap segmenter, same as the R3 analytic arms

run_cell() {
    local tag="$1"
    local latency="$2"
    local ego_fix="$3"
    local compensate="$4"
    local obstacle_fix="$5"
    echo "[lat-budget] cell=${tag} latency=${latency}s P3=${ego_fix} P0=${compensate} P2=${obstacle_fix}"
    if [[ "${PREFLIGHT}" == "1" ]]; then
        return 0
    fi
    NAVRL_PERCEPTION_PERTURB=1 \
    NAVRL_DETECTION_DROPOUT=0 \
    NAVRL_RANGE_ERROR_M=0 \
    NAVRL_DETECTION_LATENCY_S="${latency}" \
    NAVRL_LATENCY_EGO_MOTION_FIX="${ego_fix}" \
    NAVRL_LATENCY_COMPENSATE="${compensate}" \
    NAVRL_LATENCY_LIDAR_BACKUP=0 \
    NAVRL_LATENCY_OBSTACLE_FIX="${obstacle_fix}" \
    NAVRL_V2_RESULT_DIR="${RESULT_ROOT}/${tag}" \
        ./eval_navrl_v2_density_sweep.sh "${POLICY}" "${EPISODES}"
}

# Latency budget curve under the CORRECTED (timestamp-aware) model. 0.1 s is already measured
# at 78.04% in results/navrl_v2_latency_ego_motion; these extend the curve outward to find where
# a real pipeline's latency actually starts to cost capture.
run_cell latency_0p2s_p3  0.2 1 0 off
run_cell latency_0p3s_p3  0.3 1 0 off
run_cell latency_0p5s_p3  0.5 1 0 off

if [[ "${PREFLIGHT}" == "1" ]]; then
    echo "[lat-budget] PREFLIGHT PASS (no results written)"
    exit 0
fi

"${PYTHON}" - "${RESULT_ROOT}" "${BASELINE_ROOT}" "${ACTUAL_POLICY_SHA}" <<'PY'
import json
from pathlib import Path
import sys

root, baseline_root, policy_sha = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]


def load(cell_dir):
    payload = json.loads((cell_dir / "205bars.json").read_text(encoding="utf-8"))
    out, causes = payload["outcome"], payload["crash_causes"]
    return {
        "episodes": payload["actual_episodes"],
        "capture": out["capture_rate"],
        "crash": out["crash_rate"],
        "timeout": out["timeout_rate"],
        "bar_contact": causes["bar_contact"],
        "out_of_bounds": causes["out_of_bounds"],
        "closest_nocrash_mean_m": out["closest_nocrash_mean_m"],
    }


cells = {d.name: load(d) for d in sorted(p for p in root.iterdir() if p.is_dir())}
ego_root = Path("../../../results/navrl_v2_latency_ego_motion")
# Anchors measured in earlier runs under the identical contract: no latency, and 0.1 s with the
# corrected transform. Re-running them would only re-measure a known number.
cells["latency_0p1s_p3"] = load(ego_root / "latency_0p1s_p3")
cells["analytic_clean"] = load(baseline_root / "analytic_clean")
# For reference only: 0.1 s WITHOUT the ego-motion fix, i.e. the number R3 reported.
uncorrected_0p1 = load(ego_root / "latency_0p1s_raw")
clean = cells["analytic_clean"]

order = ["analytic_clean", "latency_0p1s_p3", "latency_0p2s_p3", "latency_0p3s_p3",
         "latency_0p5s_p3"]
order = [t for t in order if t in cells] + [t for t in cells if t not in order]
budget = {}
for tag in order:
    if tag == "analytic_clean":
        continue
    c = cells[tag]
    budget[tag] = {
        "capture_delta_vs_clean_pp": (c["capture"] - clean["capture"]) * 100.0,
        "crash_delta_vs_clean_pp": (c["crash"] - clean["crash"]) * 100.0,
        "bar_contact_delta_vs_clean": c["bar_contact"] - clean["bar_contact"],
        # The operating question: is this latency still cheap enough to fly with?
        "within_5pp_of_clean": (clean["capture"] - c["capture"]) <= 0.05,
    }

(root / "summary.json").write_text(
    json.dumps({"policy_sha256": policy_sha,
                "contract": "seed47/205bars/deterministic/riskcap",
                "ego_motion_fix": True,
                "uncorrected_0p1s_reference": uncorrected_0p1,
                "cells": cells, "budget": budget}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
lines = ["# latency budget under the corrected (timestamp-aware) model", "",
         "All latency cells run with NAVRL_LATENCY_EGO_MOTION_FIX=1, i.e. the delayed detection",
         "is lifted to world with the pose it was taken at. For scale, 0.1 s WITHOUT that fix --",
         f"the number R3 reported -- was capture {uncorrected_0p1['capture']*100:.2f}% / crash "
         f"{uncorrected_0p1['crash']*100:.2f}%.", "",
         "| cell | episodes | capture | crash | timeout | bar contacts | OOB | closest mean (m) |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
for tag in order:
    c = cells[tag]
    lines.append(f"| {tag} | {c['episodes']} | {c['capture']*100:.2f}% | {c['crash']*100:.2f}% | "
                 f"{c['timeout']*100:.2f}% | {c['bar_contact']} | {c['out_of_bounds']} | "
                 f"{c['closest_nocrash_mean_m']:.3f} |")
lines += ["", "## cost vs clean (within 5 pp = still flyable)", ""]
for tag, b in budget.items():
    lines.append(f"- {tag}: capture {b['capture_delta_vs_clean_pp']:+.2f} pp, "
                 f"crash {b['crash_delta_vs_clean_pp']:+.2f} pp, "
                 f"bar contacts {b['bar_contact_delta_vs_clean']:+d} -> "
                 f"{'OK' if b['within_5pp_of_clean'] else 'DEGRADED'}")
(root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
echo "[lat-budget] done -> ${RESULT_ROOT}/summary.{md,json}"
