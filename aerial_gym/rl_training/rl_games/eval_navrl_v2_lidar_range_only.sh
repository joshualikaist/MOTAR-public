#!/usr/bin/env bash
# Should the LiDAR target correction update axes it never measured?
#
# _associate_lidar_target measures ONE thing: a range along a bearing. The bearing it reports and
# the vertical component of the measurement it hands the filter are the tracker's own prediction
# read back. Applied as a full 3-D update with a diagonal R, that shrinks the lateral and
# vertical covariance on information nobody observed, cancelling the growth process noise should
# produce. Measured on a target the camera has lost, the filter reports 0.09 m of lateral sigma
# while the true error runs to 3.27 m over 20 steps -- and the policy reads that covariance in
# its target token, so it is told "certain" exactly when it should be told "lost".
#
# H3 shapes R about the measurement ray, sigma_r^2*uu^T + sigma_perp^2*(I - uu^T), keeping the
# range update and leaving the unobserved directions to process noise. Lateral sigma then grows
# 0.16 -> 1.47 m over those same 20 blind steps.
#
# This is the principled version of H2. Rather than discarding the fallback (+3.42/+3.09 pp under
# dropout across two seeds, but -0.19/-1.02 pp on clean), keep its range information and stop
# overstating its precision. If the covariance lie is the channel, H3 should match or beat H2
# under dropout WITHOUT H2's clean cost.
#
#   arm                       dropout  assoc  range-only
#   analytic_clean               0       on      off      (80.54 / 17.17 재현)
#   dropout_0p3_raw            0.3       on      off      (67.84 / 29.33 재현)
#   dropout_0p3_range_only     0.3       on       ON      ← 가설 검정
#   dropout_0p3_no_assoc       0.3      off      off      (H2 기준선, 71.25% 재현)
#   clean_range_only             0       on       ON      (무해성)
#
# Usage:
#   ./eval_navrl_v2_lidar_range_only.sh
#   NAVRL_RANGEONLY_SEED=51 ./eval_navrl_v2_lidar_range_only.sh
#   PREFLIGHT=1 ./eval_navrl_v2_lidar_range_only.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PYTHON

POLICY="runs/ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1/nn/last_gen_ppo_ep_25000_rew_39.742134.pth"
POLICY_SHA="f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40"
# Seed 47 is the frozen contract; a second seed decides adoption. The backfill arm was rejected
# because its clean regression replicated across seeds, so this one faces the same bar.
SEED="${NAVRL_RANGEONLY_SEED:-47}"
RESULT_ROOT="../../../results/navrl_v2_lidar_range_only"
if [[ "${SEED}" != "47" ]]; then
    RESULT_ROOT="${RESULT_ROOT}_seed${SEED}"
fi
PREFLIGHT="${PREFLIGHT:-0}"
EPISODES="${EPISODES:-2049}"

if [[ ! -f "${POLICY}" ]]; then
    echo "[range-only] policy checkpoint missing: ${POLICY}" >&2
    exit 2
fi
if [[ -e "${RESULT_ROOT}" && "${PREFLIGHT}" != "1" ]]; then
    echo "[range-only] refusing to overwrite ${RESULT_ROOT}" >&2
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
    raise SystemExit(f"[range-only] policy SHA mismatch: {actual}")
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
    raise SystemExit("[range-only] invalid policy checkpoint: " + ", ".join(failed))
print(actual)
PY
)"

# Frozen evaluation contract -- byte-identical to the R3 and latency harnesses.
export NAVRL_V2_DENSITIES=205
export NAVRL_V2_ACTION_MODE=deterministic
export NAVRL_EVAL_REFLECTION_MODE=original
export NAVRL_SPEED_GOVERNOR=riskcap
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
unset NAVRL_DETECTOR_CHECKPOINT  # bootstrap segmenter, same as the R3 analytic arms

run_cell() {
    local tag="$1"
    local perturb="$2"
    local dropout="$3"
    local assoc="$4"
    local range_only="$5"
    echo "[range-only] cell=${tag} perturb=${perturb} dropout=${dropout} assoc=${assoc} range_only=${range_only}"
    if [[ "${PREFLIGHT}" == "1" ]]; then
        return 0
    fi
    NAVRL_PERCEPTION_PERTURB="${perturb}" \
    NAVRL_DETECTION_DROPOUT="${dropout}" \
    NAVRL_DETECTION_LATENCY_S=0 \
    NAVRL_RANGE_ERROR_M=0 \
    NAVRL_TARGET_MASK_BACKFILL=0 \
    NAVRL_LIDAR_TARGET_ASSOC="${assoc}" \
    NAVRL_LIDAR_RANGE_ONLY_UPDATE="${range_only}" \
    NAVRL_V2_RESULT_DIR="${RESULT_ROOT}/${tag}" \
        ./eval_navrl_v2_density_sweep.sh "${POLICY}" "${EPISODES}"
}

run_cell analytic_clean          0 0   1 0
run_cell dropout_0p3_raw         1 0.3 1 0
run_cell dropout_0p3_range_only  1 0.3 1 1
run_cell dropout_0p3_no_assoc    1 0.3 0 0
run_cell clean_range_only        0 0   1 1

if [[ "${PREFLIGHT}" == "1" ]]; then
    echo "[range-only] PREFLIGHT PASS (no results written)"
    exit 0
fi

"${PYTHON}" - "${RESULT_ROOT}" "${ACTUAL_POLICY_SHA}" <<'PY'
import json
import os
from pathlib import Path
import sys

root, policy_sha = Path(sys.argv[1]), sys.argv[2]


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
        "target_visible": payload["action"]["context"]["target_visible"]["fraction"],
    }


cells = {d.name: load(d) for d in sorted(p for p in root.iterdir() if p.is_dir())}
clean, raw, fixed = cells["analytic_clean"], cells["dropout_0p3_raw"], cells["dropout_0p3_range_only"]
span = clean["capture"] - raw["capture"]
excess_bars = raw["bar_contact"] - clean["bar_contact"]
verdict = {
    "capture_delta_vs_raw_pp": (fixed["capture"] - raw["capture"]) * 100.0,
    "recovered_fraction": (fixed["capture"] - raw["capture"]) / span if span else None,
    "bar_contact_delta_vs_raw": fixed["bar_contact"] - raw["bar_contact"],
    "bar_contact_recovered_fraction": (
        (raw["bar_contact"] - fixed["bar_contact"]) / excess_bars if excess_bars else None
    ),
    # The fix must not cost anything when the camera never misses a frame.
    "clean_regression_pp": (cells["clean_range_only"]["capture"] - clean["capture"]) * 100.0,
}
# ~2050 episodes puts the inter-arm standard error near 1.3 pp, so anything smaller is noise.
verdict["hypothesis"] = (
    "SUPPORTED" if verdict["capture_delta_vs_raw_pp"] >= 4.0
    else ("REJECTED" if verdict["capture_delta_vs_raw_pp"] <= 1.3 else "INCONCLUSIVE")
)

(root / "summary.json").write_text(
    json.dumps({"policy_sha256": policy_sha,
                "contract": f"seed{os.environ['NAVRL_SEED']}/205bars/deterministic/riskcap",
                "cells": cells, "verdict": verdict}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
lines = ["# LiDAR range-only update A/B under detection dropout (ep25000+riskcap, seed %s, 205 bars)" % os.environ["NAVRL_SEED"],
         "",
         "| cell | episodes | capture | crash | bar contacts | OOB | fused visible | closest mean (m) |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
for tag, c in cells.items():
    lines.append(f"| {tag} | {c['episodes']} | {c['capture']*100:.2f}% | {c['crash']*100:.2f}% | "
                 f"{c['bar_contact']} | {c['out_of_bounds']} | {c['target_visible']*100:.2f}% | "
                 f"{c['closest_nocrash_mean_m']:.3f} |")
recovered = ("n/a" if verdict["recovered_fraction"] is None
             else f"{verdict['recovered_fraction']*100:.1f}%")
bars = ("n/a" if verdict["bar_contact_recovered_fraction"] is None
        else f"{verdict['bar_contact_recovered_fraction']*100:.1f}%")
lines += ["", f"## verdict: {verdict['hypothesis']}", "",
          f"- capture vs dropout_raw: {verdict['capture_delta_vs_raw_pp']:+.2f} pp "
          f"= {recovered} of the dropout loss",
          f"- bar contacts vs dropout_raw: {verdict['bar_contact_delta_vs_raw']:+d} "
          f"= {bars} of the excess removed",
          f"- clean regression from the fix: {verdict['clean_regression_pp']:+.2f} pp",
          "",
          "Inter-arm SE is about 1.3 pp at this episode count, so SUPPORTED needs >= 4 pp."]
(root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
echo "[range-only] done -> ${RESULT_ROOT}/summary.{md,json}"
